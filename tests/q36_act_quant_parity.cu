#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "ggml.h"
#include "q36_act_quant.cuh"
#include "q36_overlap.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

namespace C = strata::kernels::cpu;

static_assert(q36act::kTypeQ8_0 == (int) GGML_TYPE_Q8_0, "Q8_0 type id");
static_assert(q36act::kTypeQ8_K == (int) GGML_TYPE_Q8_K, "Q8_K type id");

#define CK(x)                                                                                        \
    do {                                                                                             \
        const cudaError_t e_ = (x);                                                                  \
        if (e_ != cudaSuccess) {                                                                     \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(2);                                                                            \
        }                                                                                            \
    } while (0)

namespace {

std::mt19937 rng(20260611);

enum Shape { kNormal, kSmall, kLarge, kSparse, kTies, kHalves, kTiny, kHuge, kShapes };
const char* kShapeName[kShapes] = {"normal", "small", "large", "sparse", "ties", "halves", "tiny", "huge"};

void fill(std::vector<float>& x, int shape, int64_t n) {
    std::normal_distribution<float> g(0.f, 1.f);
    std::uniform_real_distribution<float> u(0.f, 1.f);
    for (int64_t i = 0; i < n; ++i) {
        float v = g(rng);
        switch (shape) {
            case kSmall: v *= 1e-3f; break;
            case kLarge: v *= 1e3f; break;
            case kSparse: v = (i / 256) % 3 == 0 ? 0.f : (u(rng) < 0.7f ? 0.f : v); break;
            case kTies: v = (u(rng) < 0.5f ? 1.f : -1.f) * (u(rng) < 0.2f ? 2.5f : std::floor(std::fabs(v) * 2.f) * 0.5f); break;
            case kHalves: {
                // block max is 127, so the scale is exactly 1 and every other value is an exact tie k + 0.5
                const int64_t at = i % 256;
                v = at == 0 ? 127.f : (std::floor(u(rng) * 100.f) - 50.f) + 0.5f;
                if (at % 32 == 0) v = 127.f;
                break;
            }
            case kTiny: v *= 1e-38f; break;
            case kHuge: v *= 1e30f; break;
            default: break;
        }
        x[(size_t) i] = v;
    }
}

struct Case {
    const char* name;
    int type;
};

}  // namespace

int main() {
    const q36::Overlap all = q36::parse_overlap("all", nullptr, nullptr);
    const q36::Overlap some = q36::parse_overlap("glue, quant", "5", "1");
    const q36::Overlap none = q36::parse_overlap("0", nullptr, nullptr);
    if (!(all.quant && all.glue && all.commit && !all.stats && !all.verify) || !(some.quant && some.glue && !some.commit) ||
        !(some.stats && some.stats_every == 5 && some.verify) || none.any()) {
        std::printf("FAIL: the STRATA_Q36_OVERLAP list parser\n");
        return 1;
    }

    // weight types whose gate/up activation is a Q8_K or Q8_0 image
    const Case cases[] = {{"IQ2_XXS", 16}, {"IQ2_XS", 17}, {"IQ2_S", 22}, {"IQ3_XXS", 18}, {"IQ3_S", 21}, {"IQ4_XS", 23},
                          {"IQ4_NL", 20}, {"Q2_K", 10}, {"Q3_K", 11}, {"Q4_K", 12}, {"Q5_K", 13}, {"Q6_K", 14},
                          {"Q4_0", 2}, {"Q5_0", 6}, {"Q8_0", 8}, {"MXFP4", (int) GGML_TYPE_MXFP4}};
    const int64_t widths[] = {256, 2048, 2560};
    const int rows = 5, trials = 40;

    int64_t checked = 0, bad = 0, host_only = 0;
    uint8_t* d_dst = nullptr;
    float* d_x = nullptr;
    const size_t stride = C::kNativeActBytes;
    CK(cudaMalloc((void**) &d_dst, rows * stride));
    CK(cudaMalloc((void**) &d_x, (size_t) rows * 2560 * 4));
    std::vector<uint8_t> dev(rows * stride), host(stride);
    for (const Case& c : cases) {
        for (const int64_t n : widths) {
            C::NativeFmt f;
            std::string err;
            if (!C::native_fmt(c.type, 8, n, 512, f, err)) continue;   // a width the type has no whole blocks for
            if (!q36act::supported(f.gu_act, n)) {
                std::printf("%-8s n=%-5lld act type %d: host path only\n", c.name, (long long) n, f.gu_act);
                ++host_only;
                continue;
            }
            if (f.act_bytes != q36act::row_bytes(f.gu_act, n)) {
                std::printf("FAIL %s n=%lld: row bytes %zu vs %zu\n", c.name, (long long) n, f.act_bytes,
                            q36act::row_bytes(f.gu_act, n));
                return 1;
            }
            int64_t mism = 0;
            for (int shape = 0; shape < kShapes; ++shape) {
                for (int t = 0; t < trials; ++t) {
                    std::vector<float> x((size_t) rows * (size_t) n);
                    fill(x, shape, (int64_t) x.size());
                    CK(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
                    CK(cudaMemset(d_dst, 0x5a, rows * stride));
                    q36act::launch(f.gu_act, d_x, d_dst, n, rows, stride, nullptr);
                    CK(cudaGetLastError());
                    CK(cudaMemcpy(dev.data(), d_dst, rows * stride, cudaMemcpyDeviceToHost));
                    for (int r = 0; r < rows; ++r) {
                        std::memset(host.data(), 0x5a, host.size());
                        C::native_quant_act(f, x.data() + (size_t) r * (size_t) n, host.data());
                        ++checked;
                        if (!q36act::same_row(f.gu_act, n, host.data(), dev.data() + (size_t) r * stride)) {
                            if (mism++ < 3) std::printf("MISMATCH %s n=%lld shape %s trial %d row %d\n", c.name, (long long) n, kShapeName[shape], t, r);
                            ++bad;
                        }
                    }
                }
            }
            std::printf("%-8s n=%-5lld act type %2d: %s\n", c.name, (long long) n, f.gu_act, mism == 0 ? "identical" : "MISMATCH");
        }
    }
    CK(cudaFree(d_dst));
    CK(cudaFree(d_x));
    std::printf("q36_act_quant_parity: %lld rows compared, %lld mismatches, %lld (type, width) pairs left to the host\n",
                (long long) checked, (long long) bad, (long long) host_only);
    return bad == 0 && checked > 0 ? 0 : 1;
}
