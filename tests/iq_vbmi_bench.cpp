#include <pthread.h>
#include <time.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "ggml-cpu.h"
#include "ggml.h"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq3s_vbmi.hpp"
#include "strata/kernels/cpu/iq4xs_vnni.hpp"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

using namespace strata::kernels::cpu;

namespace {

constexpr int kEmbd = 2048, kFf = 512, kMaxTokens = 12;
constexpr int kDownRows = kEmbd;
std::mt19937 rng(4242);

double now() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

double mismatch(float want, float got) {
    if (!std::isfinite(want) || !std::isfinite(got)) return INFINITY;
    return std::fabs((double) want - (double) got);
}

struct Expert {
    size_t row = 0;
    std::vector<uint8_t> blob;
    Expert() : row(ggml_row_size(GGML_TYPE_IQ3_S, kEmbd)), blob(2 * kFf * row) {
        for (auto& byte : blob) byte = (uint8_t) rng();
        std::uniform_real_distribution<float> scale(0.0005f, 0.06f);
        for (size_t b = 0; b < blob.size() / sizeof(block_iq3_s); ++b) {
            const ggml_fp16_t d = ggml_fp32_to_fp16(scale(rng));
            std::memcpy(blob.data() + b * sizeof(block_iq3_s), &d, 2);
        }
    }
    size_t up_off() const { return row * kFf; }
};

std::vector<block_q8_K> make_activation() {
    std::normal_distribution<float> normal(0.f, 1.f);
    const float amplitude = std::exp(std::uniform_real_distribution<float>(-3.f, 2.f)(rng));
    std::vector<float> x(kEmbd);
    for (auto& v : x) v = normal(rng) * amplitude;
    std::vector<block_q8_K> y(kEmbd / QK_K);
    ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(x.data(), y.data(), kEmbd);
    return y;
}

void reference_rows(const Expert& e, const block_q8_K* y, float* ff) {
    const auto dot = ggml_get_type_traits_cpu(GGML_TYPE_IQ3_S)->vec_dot;
    for (int r = 0; r < kFf; ++r) {
        float gate = 0.f, up = 0.f;
        dot(kEmbd, &gate, 0, e.blob.data() + (size_t) r * e.row, 0, y, 0, 1);
        dot(kEmbd, &up, 0, e.blob.data() + e.up_off() + (size_t) r * e.row, 0, y, 0, 1);
        ff[r] = (gate / (1.f + std::exp(-gate))) * up;
    }
}

struct DownRows {
    size_t row = 0;
    std::vector<uint8_t> blob;
    DownRows() : row(ggml_row_size(GGML_TYPE_IQ4_XS, kFf)), blob((size_t) kDownRows * row) {
        for (auto& byte : blob) byte = (uint8_t) rng();
        std::uniform_real_distribution<float> scale(0.0005f, 0.06f);
        for (size_t b = 0; b < blob.size() / sizeof(block_iq4_xs); ++b) {
            const ggml_fp16_t d = ggml_fp32_to_fp16(scale(rng));
            std::memcpy(blob.data() + b * sizeof(block_iq4_xs), &d, 2);
        }
    }
};

std::vector<block_q8_K> make_hidden() {
    std::normal_distribution<float> normal(0.f, 1.f);
    const float amplitude = std::exp(std::uniform_real_distribution<float>(-3.f, 2.f)(rng));
    std::vector<float> x(kFf);
    for (auto& v : x) v = normal(rng) * amplitude;
    std::vector<block_q8_K> y(kFf / QK_K);
    ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(x.data(), y.data(), kFf);
    return y;
}

int check_down() {
    int failures = 0;
    const auto dot = ggml_get_type_traits_cpu(GGML_TYPE_IQ4_XS)->vec_dot;
    for (int nt = 1; nt <= kMaxTokens; ++nt) {
        double worst = 0, scale = 0;
        for (int round = 0; round < 4; ++round) {
            DownRows rows;
            std::vector<std::vector<block_q8_K>> act;
            std::vector<const void*> act_ptr;
            std::vector<std::vector<float>> want(nt, std::vector<float>(kDownRows)), got(nt, std::vector<float>(kDownRows));
            std::vector<float*> got_ptr;
            for (int t = 0; t < nt; ++t) {
                act.push_back(make_hidden());
                act_ptr.push_back(act.back().data());
                got_ptr.push_back(got[t].data());
                for (int r = 0; r < kDownRows; ++r) dot(kFf, &want[t][r], 0, rows.blob.data() + (size_t) r * rows.row, 0, act.back().data(), 0, 1);
            }
            const int r0 = round % 2 == 0 ? 0 : 3, r1 = round == 3 ? kDownRows - 9 : kDownRows;
            iq4xs_vnni_rows(rows.blob.data(), rows.row, kFf, act_ptr.data(), nt, got_ptr.data(), r0, r1);
            for (int t = 0; t < nt; ++t)
                for (int r = r0; r < r1; ++r) {
                    worst = std::max(worst, mismatch(want[t][r], got[t][r]));
                    scale = std::max(scale, (double) std::fabs(want[t][r]));
                }
        }
        const double relative = worst / (scale + 1e-30);
        const bool ok = relative < 1e-6;
        std::printf("down nt=%2d  largest difference %.3g of %.3g (relative %.2g)  %s\n", nt, worst, scale, relative, ok ? "ok" : "FAIL");
        failures += !ok;
    }
    return failures;
}

void bench_down(int threads, double seconds) {
    constexpr int kSets = 160;
    const auto dot = ggml_get_type_traits_cpu(GGML_TYPE_IQ4_XS)->vec_dot;
    for (int nt : {1, 3}) {
        for (int variant = 0; variant < 2; ++variant) {
            std::vector<DownRows> sets((size_t) threads * kSets);
            std::vector<long> done((size_t) threads, 0);
            std::vector<std::vector<std::vector<block_q8_K>>> thread_act((size_t) threads);
            for (auto& per_thread : thread_act)
                for (int t = 0; t < nt; ++t) per_thread.push_back(make_hidden());
            std::atomic<int> ready{0};
            std::atomic<bool> go{false}, stop{false};
            std::vector<std::thread> pool;
            for (int th = 0; th < threads; ++th)
                pool.emplace_back([&, th] {
                    const auto& act = thread_act[(size_t) th];
                    std::vector<const void*> act_ptr;
                    std::vector<std::vector<float>> out(nt, std::vector<float>(kDownRows));
                    std::vector<float*> out_ptr;
                    for (int t = 0; t < nt; ++t) {
                        act_ptr.push_back(act[t].data());
                        out_ptr.push_back(out[t].data());
                    }
                    ready++;
                    while (!go.load()) std::this_thread::yield();
                    size_t at = 0;
                    while (!stop.load()) {
                        const DownRows& e = sets[(size_t) th * kSets + (at++ % kSets)];
                        if (variant == 0)
                            for (int t = 0; t < nt; ++t)
                                for (int r = 0; r < kDownRows; ++r) dot(kFf, &out[t][r], 0, e.blob.data() + (size_t) r * e.row, 0, act[t].data(), 0, 1);
                        else
                            iq4xs_vnni_rows(e.blob.data(), e.row, kFf, act_ptr.data(), nt, out_ptr.data(), 0, kDownRows);
                        done[(size_t) th]++;
                    }
                });
            while (ready.load() < threads) std::this_thread::yield();
            go = true;
            const double t0 = now();
            std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
            stop = true;
            const double wall = now() - t0;
            for (auto& t : pool) t.join();
            long total = 0;
            for (long d : done) total += d;
            const double bytes = (double) sets[0].blob.size();
            std::printf("down nt=%d threads=%d %-22s %8.0f experts/s  %6.1f us per expert and thread  %5.1f GB/s\n", nt, threads,
                        variant == 0 ? "ggml dot per token" : "nibble lookup + VNNI", total / wall, wall * threads / (double) total * 1e6,
                        total / wall * bytes / 1e9);
        }
    }
}

int check() {
    int failures = 0;
    for (int nt = 1; nt <= kMaxTokens; ++nt) {
        double worst = 0, scale = 0;
        for (int round = 0; round < 12; ++round) {
            Expert expert;
            std::vector<std::vector<block_q8_K>> act;
            std::vector<const void*> act_ptr;
            std::vector<std::vector<float>> want(nt, std::vector<float>(kFf)), got(nt, std::vector<float>(kFf));
            std::vector<float*> got_ptr;
            for (int t = 0; t < nt; ++t) {
                act.push_back(make_activation());
                act_ptr.push_back(act.back().data());
                got_ptr.push_back(got[t].data());
                reference_rows(expert, act.back().data(), want[t].data());
            }
            const int r0 = round % 3 == 0 ? 0 : 7, r1 = round % 3 == 2 ? kFf - 5 : kFf;
            iq3s_vbmi_gu_rows(expert.blob.data(), expert.row, expert.up_off(), kEmbd, act_ptr.data(), nt, got_ptr.data(), r0, r1);
            for (int t = 0; t < nt; ++t)
                for (int r = r0; r < r1; ++r) {
                    worst = std::max(worst, mismatch(want[t][r], got[t][r]));
                    scale = std::max(scale, (double) std::fabs(want[t][r]));
                }
        }
        const double relative = worst / (scale + 1e-30);
        const bool ok = relative < 1e-6;
        std::printf("nt=%2d  largest difference %.3g of %.3g (relative %.2g)  %s\n", nt, worst, scale, relative, ok ? "ok" : "FAIL");
        failures += !ok;
    }
    return failures;
}

void bench(int threads, double seconds) {
    constexpr int kExperts = 48;
    for (int nt : {1, 2, 3}) {
        for (int variant = 0; variant < 2; ++variant) {
            std::vector<Expert> experts((size_t) threads * kExperts);
            std::vector<long> done((size_t) threads, 0);
            std::vector<std::vector<std::vector<block_q8_K>>> thread_act((size_t) threads);
            for (auto& per_thread : thread_act)
                for (int t = 0; t < nt; ++t) per_thread.push_back(make_activation());
            std::atomic<int> ready{0};
            std::atomic<bool> go{false}, stop{false};
            std::vector<std::thread> pool;
            for (int th = 0; th < threads; ++th)
                pool.emplace_back([&, th] {
                    const auto& act = thread_act[(size_t) th];
                    std::vector<const void*> act_ptr;
                    std::vector<std::vector<float>> out(nt, std::vector<float>(kFf));
                    std::vector<float*> out_ptr;
                    for (int t = 0; t < nt; ++t) {
                        act_ptr.push_back(act[t].data());
                        out_ptr.push_back(out[t].data());
                    }
                    ready++;
                    while (!go.load()) std::this_thread::yield();
                    size_t at = 0;
                    while (!stop.load()) {
                        const Expert& e = experts[(size_t) th * kExperts + (at++ % kExperts)];
                        if (variant == 0)
                            for (int t = 0; t < nt; ++t) reference_rows(e, act[t].data(), out[t].data());
                        else
                            iq3s_vbmi_gu_rows(e.blob.data(), e.row, e.up_off(), kEmbd, act_ptr.data(), nt, out_ptr.data(), 0, kFf);
                        done[(size_t) th]++;
                    }
                });
            while (ready.load() < threads) std::this_thread::yield();
            go = true;
            const double t0 = now();
            std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
            stop = true;
            const double wall = now() - t0;
            for (auto& t : pool) t.join();
            long total = 0;
            for (long d : done) total += d;
            const double bytes = 2.0 * kFf * (double) experts[0].row;
            std::printf("nt=%d threads=%d %-22s %8.0f experts/s  %6.1f us per expert and thread  %5.1f GB/s\n", nt, threads,
                        variant == 0 ? "ggml dot per token" : "byte-permute kernel", total / wall, wall * threads / (double) total * 1e6,
                        total / wall * bytes / 1e9);
        }
    }
}

// The IQ3_S gate/up matrices stored as 4-bit IQ4_XS rows of 2048 weights (the repack of backlog item L3): the same kernel
// as the down projection, 2 * kFf rows of 8 blocks each.
constexpr int kGuRows = 2 * kFf;

struct Gu4Rows {
    size_t row = 0;
    std::vector<uint8_t> blob;
    Gu4Rows() : row(ggml_row_size(GGML_TYPE_IQ4_XS, kEmbd)), blob((size_t) kGuRows * row) {
        for (auto& byte : blob) byte = (uint8_t) rng();
        std::uniform_real_distribution<float> scale(0.0005f, 0.06f);
        for (size_t b = 0; b < blob.size() / sizeof(block_iq4_xs); ++b) {
            const ggml_fp16_t d = ggml_fp32_to_fp16(scale(rng));
            std::memcpy(blob.data() + b * sizeof(block_iq4_xs), &d, 2);
        }
    }
};

int check_gu4() {
    int failures = 0;
    const auto dot = ggml_get_type_traits_cpu(GGML_TYPE_IQ4_XS)->vec_dot;
    for (int nt : {1, 2, 3, 8}) {
        double worst = 0, scale = 0;
        for (int round = 0; round < 3; ++round) {
            Gu4Rows rows;
            std::vector<std::vector<block_q8_K>> act;
            std::vector<const void*> act_ptr;
            std::vector<std::vector<float>> want(nt, std::vector<float>(kGuRows)), got(nt, std::vector<float>(kGuRows));
            std::vector<float*> got_ptr;
            for (int t = 0; t < nt; ++t) {
                act.push_back(make_activation());
                act_ptr.push_back(act.back().data());
                got_ptr.push_back(got[t].data());
                for (int r = 0; r < kGuRows; ++r) dot(kEmbd, &want[t][r], 0, rows.blob.data() + (size_t) r * rows.row, 0, act.back().data(), 0, 1);
            }
            iq4xs_vnni_rows(rows.blob.data(), rows.row, kEmbd, act_ptr.data(), nt, got_ptr.data(), 0, kGuRows);
            for (int t = 0; t < nt; ++t)
                for (int r = 0; r < kGuRows; ++r) {
                    worst = std::max(worst, mismatch(want[t][r], got[t][r]));
                    scale = std::max(scale, (double) std::fabs(want[t][r]));
                }
        }
        const double relative = worst / (scale + 1e-30);
        const bool ok = relative < 1e-6;
        std::printf("gate/up as 4-bit nt=%2d  largest difference %.3g of %.3g (relative %.2g)  %s\n", nt, worst, scale, relative, ok ? "ok" : "FAIL");
        failures += !ok;
    }
    return failures;
}

// Throughput of both gate/up forms. `sets` experts per thread: a large number streams from memory, 1 stays in the cache.
void bench_gu(int threads, double seconds, int sets) {
    for (int nt : {1, 2, 3}) {
        for (int variant = 0; variant < 2; ++variant) {
            std::vector<Expert> iq3((size_t) threads * (variant == 0 ? sets : 0));
            std::vector<Gu4Rows> q4((size_t) threads * (variant == 1 ? sets : 0));
            std::vector<long> done((size_t) threads, 0);
            std::vector<std::vector<std::vector<block_q8_K>>> thread_act((size_t) threads);
            for (auto& per_thread : thread_act)
                for (int t = 0; t < nt; ++t) per_thread.push_back(make_activation());
            std::atomic<int> ready{0};
            std::atomic<bool> go{false}, stop{false};
            std::vector<std::thread> pool;
            for (int th = 0; th < threads; ++th)
                pool.emplace_back([&, th] {
                    const auto& act = thread_act[(size_t) th];
                    std::vector<const void*> act_ptr;
                    std::vector<std::vector<float>> out(nt, std::vector<float>(kGuRows));
                    std::vector<float*> out_ptr;
                    for (int t = 0; t < nt; ++t) {
                        act_ptr.push_back(act[t].data());
                        out_ptr.push_back(out[t].data());
                    }
                    ready++;
                    while (!go.load()) std::this_thread::yield();
                    size_t at = 0;
                    while (!stop.load()) {
                        const size_t k = (size_t) th * sets + (at++ % sets);
                        if (variant == 0) iq3s_vbmi_gu_rows(iq3[k].blob.data(), iq3[k].row, iq3[k].up_off(), kEmbd, act_ptr.data(), nt, out_ptr.data(), 0, kFf);
                        else iq4xs_vnni_rows(q4[k].blob.data(), q4[k].row, kEmbd, act_ptr.data(), nt, out_ptr.data(), 0, kGuRows);
                        done[(size_t) th]++;
                    }
                });
            while (ready.load() < threads) std::this_thread::yield();
            go = true;
            const double t0 = now();
            std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
            stop = true;
            const double wall = now() - t0;
            for (auto& t : pool) t.join();
            long total = 0;
            for (long d : done) total += d;
            const double bytes = variant == 0 ? (double) iq3[0].blob.size() : (double) q4[0].blob.size();
            std::printf("gate/up nt=%d threads=%d sets=%-3d %-14s %8.0f experts/s  %6.1f us per expert and thread  %5.1f GB/s\n", nt, threads, sets,
                        variant == 0 ? "IQ3_S kernel" : "4-bit repack", total / wall, wall * threads / (double) total * 1e6, total / wall * bytes / 1e9);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    ggml_cpu_init();
    bool run_bench = false;
    int threads = 1;
    double seconds = 2.0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0) run_bench = true;
        else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--seconds") == 0 && i + 1 < argc) seconds = std::atof(argv[++i]);
        else {
            std::fprintf(stderr, "usage: %s [--bench] [--threads N] [--seconds S]\n", argv[0]);
            return 2;
        }
    }
    if (!cpu_avx512_bitalg_ok()) {
        std::fprintf(stderr, "this CPU has no AVX-512 VBMI and BITALG: nothing to test\n");
        return 77;
    }
    const int failures = check() + check_down() + check_gu4();
    if (run_bench) {
        bench(std::max(1, std::min(threads, 8)), seconds);
        bench_down(std::max(1, std::min(threads, 8)), seconds);
        bench_gu(std::max(1, std::min(threads, 8)), seconds, 48);
        bench_gu(std::max(1, std::min(threads, 8)), seconds, 1);
    }
    std::printf("%s\n", failures == 0 ? "iq vbmi parity: ok" : "iq vbmi parity: FAILED");
    return failures == 0 ? 0 : 1;
}
