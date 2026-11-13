#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "q36_act_quant.cuh"
#include "strata/kernels/cpu/iq4xs_vnni.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/strict_math.hpp"
#include "strata/kernels/iq_kernels.hpp"

namespace C = strata::kernels::cpu;
namespace K = strata::kernels;

#define CK(x)                                                                                        \
    do {                                                                                             \
        const cudaError_t e_ = (x);                                                                  \
        if (e_ != cudaSuccess) {                                                                     \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(2);                                                                            \
        }                                                                                            \
    } while (0)

namespace {

constexpr int kIq3S = 21, kIq4Xs = 23, kQ8K = 15;
constexpr int kN = 2048, kFF = 512;
constexpr int kIq3sBlock = 110, kIq4xsBlock = 136, kBlockValues = 256;
constexpr size_t kActStride = C::kNativeActBytes;

std::mt19937 rng(20261113);

uint16_t float_to_half(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, 4);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int exponent = (int) ((bits >> 23) & 0xFF) - 127 + 15;
    const uint32_t mantissa = (bits >> 13) & 0x3FFu;
    return (uint16_t) (sign | ((uint32_t) exponent << 10) | mantissa);   // the test's scales are normal halves: truncation is enough
}

/// Random block bytes with a finite block scale d of a realistic size.
void fill_rows(uint8_t* rows, size_t count, int values, int block_bytes) {
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(0.002f, 0.05f);
    const size_t blocks = count * (size_t) (values / kBlockValues);
    for (size_t b = 0; b < blocks; ++b) {
        uint8_t* block = rows + b * (size_t) block_bytes;
        for (int i = 0; i < block_bytes; ++i) block[i] = (uint8_t) byte(rng);
        const uint16_t d = float_to_half((byte(rng) & 1) ? scale(rng) : -scale(rng));
        std::memcpy(block, &d, 2);
    }
}

struct Case {
    const char* name;
    std::vector<std::vector<int>> tokens_of_expert;   // per expert: the tokens routed to it
};

int run_case(const Case& c, int tokens, bool odd4, float magnitude) {
    C::NativeFmt fmt;
    std::string err;
    if (!C::native_fmt(kIq3S, kIq4Xs, kN, kFF, fmt, err)) {
        std::fprintf(stderr, "native_fmt: %s\n", err.c_str());
        return 1;
    }
    const K::NativeExpertLayout L = K::native_expert_layout(kIq3S, kIq4Xs, kN, kFF);
    if (!K::native_expert_strict_supported(L) || !C::native_strict_supported(fmt, odd4)) {
        std::fprintf(stderr, "the strict kernels do not support this build or CPU (STRATA_IQ_VBMI=1 and AVX-512 are needed)\n");
        return 1;
    }
    const int experts = (int) c.tokens_of_expert.size();
    std::vector<uint8_t> blobs((size_t) experts * L.bytes + 64);
    for (int e = 0; e < experts; ++e) {
        uint8_t* blob = blobs.data() + (size_t) e * L.bytes;
        fill_rows(blob, kFF, kN, kIq3sBlock);
        fill_rows(blob + L.up_off, kFF, kN, kIq3sBlock);
        fill_rows(blob + L.down_off, kN, kFF, kIq4xsBlock);
    }
    std::vector<float> x((size_t) tokens * kN);
    std::normal_distribution<float> normal(0.f, magnitude);
    for (float& v : x) v = normal(rng);

    // the CPU: the pool's order for one expert job (gate/up and SwiGLU, quantize, down)
    std::vector<uint8_t> acts((size_t) tokens * kActStride);
    for (int t = 0; t < tokens; ++t) C::native_quant_act(fmt, x.data() + (size_t) t * kN, acts.data() + (size_t) t * kActStride);
    int entries = 0;
    for (const auto& list : c.tokens_of_expert) entries += (int) list.size();
    std::vector<float> cpu_hidden((size_t) entries * kFF), cpu_out((size_t) entries * kN);
    std::vector<int32_t> grp_start, ent_tok, ent_dst;
    const size_t odd_row = (size_t) (kN / kBlockValues) * kIq4xsBlock;
    std::vector<uint8_t> repacked;
    int entry = 0;
    for (int e = 0; e < experts; ++e) {
        const std::vector<int>& list = c.tokens_of_expert[(size_t) e];
        grp_start.push_back(entry);
        const uint8_t* blob = blobs.data() + (size_t) e * L.bytes;
        C::NativeFmt job_fmt = fmt;
        if (odd4) {   // gate and up repacked to 4 bits, the down rows after them: what the engine gives the pool with --iq3s-4bit
            repacked.assign(2 * (size_t) kFF * odd_row + (size_t) kN * L.d_row + 64, 0);
            C::iq3s_repack_to_4bit(blob, repacked.data(), (size_t) kFF * (kN / kBlockValues));
            C::iq3s_repack_to_4bit(blob + L.up_off, repacked.data() + (size_t) kFF * odd_row, (size_t) kFF * (kN / kBlockValues));
            std::memcpy(repacked.data() + 2 * (size_t) kFF * odd_row, blob + L.down_off, (size_t) kN * L.d_row);
            job_fmt.gu_odd4 = true;
            job_fmt.gu_row = odd_row;
            job_fmt.up_off = (size_t) kFF * odd_row;
            job_fmt.down_off = 2 * (size_t) kFF * odd_row;
            blob = repacked.data();
        }
        for (size_t first = 0; first < list.size(); first += 8) {   // the pool gives a kernel at most 8 tokens
            const int nt = (int) std::min<size_t>(8, list.size() - first);
            const void* act_of[8];
            float* hidden_of[8];
            float* out_of[8];
            std::vector<uint8_t> hq((size_t) nt * C::kNativeHBytes);
            const void* hq_of[8];
            for (int t = 0; t < nt; ++t) {
                act_of[t] = acts.data() + (size_t) list[first + (size_t) t] * kActStride;
                hidden_of[t] = cpu_hidden.data() + (size_t) (entry + t) * kFF;
                out_of[t] = cpu_out.data() + (size_t) (entry + t) * kN;
                hq_of[t] = hq.data() + (size_t) t * C::kNativeHBytes;
            }
            C::native_gu_rows(job_fmt, blob, act_of, nt, hidden_of, 0, kFF);
            for (int t = 0; t < nt; ++t) C::native_quant_h(job_fmt, hidden_of[t], hq.data() + (size_t) t * C::kNativeHBytes);
            C::native_down_rows(job_fmt, blob, hq_of, nt, out_of, 0, kN);
            for (int t = 0; t < nt; ++t) {
                ent_tok.push_back(list[first + (size_t) t]);
                ent_dst.push_back(entry + t);
            }
            entry += nt;
        }
    }
    grp_start.push_back(entry);

    // the GPU: the strict kernels on the same blobs (always the IQ3_S rows) and the device quantizer
    uint8_t *d_blobs = nullptr, *d_acts = nullptr, *d_hq = nullptr;
    float *d_x = nullptr, *d_hidden = nullptr, *d_out = nullptr;
    unsigned long long* d_grp_ptr = nullptr;
    int32_t *d_grp_start = nullptr, *d_n_groups = nullptr, *d_ent_tok = nullptr, *d_ent_dst = nullptr;
    const size_t hq_stride = K::native_expert_strict_hq_stride(kFF);
    CK(cudaMalloc(&d_blobs, blobs.size()));
    CK(cudaMalloc(&d_acts, (size_t) tokens * kActStride));
    CK(cudaMalloc(&d_hq, (size_t) entries * hq_stride));
    CK(cudaMalloc(&d_x, x.size() * 4));
    CK(cudaMalloc(&d_hidden, cpu_hidden.size() * 4));
    CK(cudaMalloc(&d_out, cpu_out.size() * 4));
    CK(cudaMalloc(&d_grp_ptr, (size_t) (experts + 1) * 8));
    CK(cudaMalloc(&d_grp_start, (size_t) (experts + 1) * 4));
    CK(cudaMalloc(&d_n_groups, 4));
    CK(cudaMalloc(&d_ent_tok, (size_t) entries * 4));
    CK(cudaMalloc(&d_ent_dst, (size_t) entries * 4));
    std::vector<unsigned long long> grp_ptr((size_t) experts + 1, 0);
    for (int e = 0; e < experts; ++e) grp_ptr[(size_t) e] = (unsigned long long) (uintptr_t) (d_blobs + (size_t) e * L.bytes);
    const int32_t n_groups = experts;
    CK(cudaMemcpy(d_blobs, blobs.data(), blobs.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_grp_ptr, grp_ptr.data(), grp_ptr.size() * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_grp_start, grp_start.data(), grp_start.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_n_groups, &n_groups, 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ent_tok, ent_tok.data(), ent_tok.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ent_dst, ent_dst.data(), ent_dst.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(d_out, 0, cpu_out.size() * 4));
    q36act::launch(kQ8K, d_x, d_acts, kN, tokens, kActStride, nullptr);
    K::native_expert_strict_gu(L, d_grp_ptr, d_grp_start, d_n_groups, d_ent_tok, experts, d_acts, kActStride, odd4, d_hidden, nullptr);
    q36act::launch(kQ8K, d_hidden, d_hq, kFF, entries, hq_stride, nullptr);
    K::native_expert_strict_down(L, d_grp_ptr, d_grp_start, d_n_groups, d_ent_dst, experts, d_hq, hq_stride, d_out, nullptr);
    CK(cudaGetLastError());
    std::vector<float> gpu_hidden(cpu_hidden.size()), gpu_out(cpu_out.size());
    CK(cudaMemcpy(gpu_hidden.data(), d_hidden, gpu_hidden.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gpu_out.data(), d_out, gpu_out.size() * 4, cudaMemcpyDeviceToHost));
    for (void* p : {(void*) d_blobs, (void*) d_acts, (void*) d_hq, (void*) d_x, (void*) d_hidden, (void*) d_out, (void*) d_grp_ptr,
                    (void*) d_grp_start, (void*) d_n_groups, (void*) d_ent_tok, (void*) d_ent_dst})
        CK(cudaFree(p));

    size_t hidden_bad = 0, out_bad = 0;
    for (size_t i = 0; i < cpu_hidden.size(); ++i) hidden_bad += std::memcmp(&cpu_hidden[i], &gpu_hidden[i], 4) != 0;
    for (size_t i = 0; i < cpu_out.size(); ++i) out_bad += std::memcmp(&cpu_out[i], &gpu_out[i], 4) != 0;
    std::printf("%-22s %s magnitude %-6g: %d entries, hidden %zu of %zu differ, out %zu of %zu differ\n", c.name, odd4 ? "4-bit " : "iq3_s ",
                (double) magnitude, entries, hidden_bad, cpu_hidden.size(), out_bad, cpu_out.size());
    if (hidden_bad != 0) {
        for (size_t i = 0; i < cpu_hidden.size(); ++i)
            if (std::memcmp(&cpu_hidden[i], &gpu_hidden[i], 4) != 0) {
                std::printf("    first hidden difference: entry %zu row %zu, CPU %.9g GPU %.9g\n", i / kFF, i % kFF, (double) cpu_hidden[i],
                            (double) gpu_hidden[i]);
                break;
            }
    }
    return hidden_bad != 0 || out_bad != 0;
}

/// The strict exp against the C library's: how often the float results differ, and by how many units in the last place at most.
void report_exp() {
    size_t count = 0, differ = 0;
    int worst = 0;
    for (float v = -30.f; v <= 30.f; v += 0.000037f) {
        const float gate = v;
        const float strict = C::strict_swiglu(gate, 1.f);
        const float library = (gate / (1.f + std::exp(-gate))) * 1.f;
        int32_t a, b;
        std::memcpy(&a, &strict, 4);
        std::memcpy(&b, &library, 4);
        ++count;
        if (a != b) {
            ++differ;
            worst = std::max(worst, std::abs(a - b));
        }
    }
    std::printf("strict SwiGLU against the C library's exp: %zu of %zu values differ, at most %d ulp\n", differ, count, worst);
}

}  // namespace

int main() {
    setenv("STRATA_IQ_VBMI", "1", 1);
    C::set_strict_swiglu(true);
    report_exp();
    const Case cases[] = {
        {"one token", {{0}}},
        {"two experts", {{0, 1, 2}, {1}}},
        {"five and one", {{0, 1, 2, 3, 4}, {2}}},
        {"eight, three, one", {{0, 1, 2, 3, 4, 5, 6, 7}, {1, 4, 6}, {5}}},
        {"eleven in one expert", {{0, 1, 2, 3, 4, 5, 6, 7, 0, 3, 5}}},
    };
    int bad = 0;
    for (const Case& c : cases)
        for (const bool odd4 : {false, true})
            for (const float magnitude : {1.f, 0.02f, 40.f}) bad += run_case(c, 8, odd4, magnitude);
    std::printf(bad == 0 ? "PASS: the GPU strict kernels give the CPU kernels' bits\n" : "FAIL: %d cases differ\n", bad);
    return bad != 0;
}
