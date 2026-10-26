#include <pthread.h>
#include <time.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "ggml-cpu.h"
#include "ggml.h"
#include "strata/kernels/cpu/kq_avx512.hpp"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

using namespace strata::kernels::cpu;

namespace {

double now() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

std::mt19937 rng(12345);

void fill_weights(uint8_t* p, size_t bytes, size_t block_bytes) {
    for (size_t i = 0; i < bytes; ++i) p[i] = (uint8_t) rng();
    std::uniform_real_distribution<float> scale(0.0005f, 0.06f);
    for (size_t b = 0; b < bytes / block_bytes; ++b) {
        ggml_fp16_t d = ggml_fp32_to_fp16(scale(rng)), dmin = ggml_fp32_to_fp16(scale(rng));
        std::memcpy(p + b * block_bytes, &d, 2);
        std::memcpy(p + b * block_bytes + 2, &dmin, 2);
    }
}

std::vector<block_q8_K> make_activation(int n) {
    std::normal_distribution<float> g(0.f, 1.f);
    std::vector<float> x((size_t) n);
    const float amp = std::exp(std::uniform_real_distribution<float>(-3.f, 2.f)(rng));
    for (auto& v : x) v = g(rng) * amp;
    std::vector<block_q8_K> y((size_t) n / 256);
    ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(x.data(), y.data(), n);
    return y;
}

bool same(float a, float b) { return std::memcmp(&a, &b, 4) == 0; }

// Gate/up: ggml's two dot products and the SwiGLU of the engine's generic path against the kernel
int check_gu(int rows, int n, int nt) {
    const size_t row = (size_t) n / 256 * sizeof(block_q4_K);
    std::vector<uint8_t> blob(2 * rows * row);
    fill_weights(blob.data(), blob.size(), sizeof(block_q4_K));
    std::vector<std::vector<block_q8_K>> act(nt);
    std::vector<const void*> ap(nt);
    for (int t = 0; t < nt; ++t) { act[t] = make_activation(n); ap[t] = act[t].data(); }
    std::vector<std::vector<float>> got(nt, std::vector<float>(rows)), want(nt, std::vector<float>(rows));
    std::vector<float*> gp(nt);
    for (int t = 0; t < nt; ++t) gp[t] = got[t].data();
    kq512_gu_rows(blob.data(), row, rows * row, n, ap.data(), nt, gp.data(), 0, rows);
    const auto dot = ggml_get_type_traits_cpu(GGML_TYPE_Q4_K)->vec_dot;
    for (int r = 0; r < rows; ++r)
        for (int t = 0; t < nt; ++t) {
            float g = 0.f, u = 0.f;
            dot(n, &g, 0, blob.data() + r * row, 0, ap[t], 0, 1);
            dot(n, &u, 0, blob.data() + rows * row + r * row, 0, ap[t], 0, 1);
            want[t][r] = (g / (1.f + std::exp(-g))) * u;
        }
    int bad = 0;
    for (int t = 0; t < nt; ++t)
        for (int r = 0; r < rows; ++r) bad += !same(got[t][r], want[t][r]);
    return bad;
}

int check_down(int rows, int n, int nt) {
    const size_t row = (size_t) n / 256 * sizeof(block_q5_K);
    std::vector<uint8_t> blob(rows * row);
    fill_weights(blob.data(), blob.size(), sizeof(block_q5_K));
    std::vector<std::vector<block_q8_K>> act(nt);
    std::vector<const void*> ap(nt);
    for (int t = 0; t < nt; ++t) { act[t] = make_activation(n); ap[t] = act[t].data(); }
    std::vector<std::vector<float>> got(nt, std::vector<float>(rows));
    std::vector<float*> gp(nt);
    for (int t = 0; t < nt; ++t) gp[t] = got[t].data();
    kq512_down_rows(blob.data(), row, n, ap.data(), nt, gp.data(), 0, rows);
    const auto dot = ggml_get_type_traits_cpu(GGML_TYPE_Q5_K)->vec_dot;
    int bad = 0;
    for (int r = 0; r < rows; ++r)
        for (int t = 0; t < nt; ++t) {
            float s = 0.f;
            dot(n, &s, 0, blob.data() + r * row, 0, ap[t], 0, 1);
            bad += !same(got[t][r], s);
        }
    return bad;
}

struct Job {
    const uint8_t* w;
    size_t rows, row;
    int n, nt, reps;
    bool ggml;
    bool down;
    const void* const* act;
    float* sink;
};

void* run(void* arg) {
    Job* j = (Job*) arg;
    float out[8][4096];
    float* op[8];
    for (int t = 0; t < 8; ++t) op[t] = out[t];
    const auto dot = ggml_get_type_traits_cpu(j->down ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K)->vec_dot;
    const size_t units = j->down ? j->rows : j->rows / 2;   // gate/up: a unit is a gate row and an up row
    const size_t up_off = units * j->row;
    for (int rep = 0; rep < j->reps; ++rep) {
        const size_t step = 512;
        for (size_t r0 = 0; r0 < units; r0 += step) {
            const size_t r1 = std::min(units, r0 + step);
            if (j->ggml) {
                for (size_t r = r0; r < r1; ++r)
                    for (int t = 0; t < j->nt; ++t) {
                        float s;
                        dot(j->n, &s, 0, j->w + r * j->row, 0, j->act[t], 0, 1);
                        if (!j->down) dot(j->n, &s, 0, j->w + up_off + r * j->row, 0, j->act[t], 0, 1);
                        out[t][r % 4096] = s;
                    }
            } else if (j->down) {
                kq512_down_rows(j->w, j->row, j->n, j->act, j->nt, op, (int) (r0 % 4096), (int) (r0 % 4096 + (r1 - r0)));
            } else {
                kq512_gu_rows(j->w, j->row, up_off, j->n, j->act, j->nt, op, (int) (r0 % 4096), (int) (r0 % 4096 + (r1 - r0)));
            }
        }
    }
    *j->sink = out[0][0];
    return nullptr;
}

double bench(bool down, bool use_ggml, int threads, int nt, bool hot) {
    const int n = down ? 512 : 2048;
    const size_t row = (size_t) n / 256 * (down ? sizeof(block_q5_K) : sizeof(block_q4_K));
    const size_t rows = hot ? 600 : (size_t) (192u << 20) / row;
    std::vector<std::vector<uint8_t>> w(threads, std::vector<uint8_t>(rows * row + 4096));
    std::vector<std::vector<block_q8_K>> act(nt);
    std::vector<const void*> ap(nt);
    for (int t = 0; t < nt; ++t) { act[t] = make_activation(n); ap[t] = act[t].data(); }
    std::vector<Job> jobs(threads);
    std::vector<float> sinks(threads);
    for (int i = 0; i < threads; ++i) {
        fill_weights(w[i].data(), rows * row, down ? sizeof(block_q5_K) : sizeof(block_q4_K));
        jobs[i] = {w[i].data(), rows, row, n, nt, hot ? 3000 : 1, use_ggml, down, ap.data(), &sinks[i]};
    }
    double best = 1e9;
    for (int rep = 0; rep < 3; ++rep) {
        std::vector<pthread_t> th(threads);
        const double a = now();
        for (int i = 0; i < threads; ++i) pthread_create(&th[i], nullptr, run, &jobs[i]);
        for (int i = 0; i < threads; ++i) pthread_join(th[i], nullptr);
        best = std::min(best, now() - a);
    }
    return (double) threads * rows * row * jobs[0].reps / best / 1e9;
}

}  // namespace

int main(int argc, char** argv) {
    ggml_cpu_init();
    int bad = 0, cases = 0;
    for (int trial = 0; trial < 40; ++trial)
        for (int nt : {1, 2, 3, 5, 8}) {
            bad += check_gu(48, 2048, nt) + check_down(96, 512, nt);
            ++cases;
        }
    std::printf("bit-exact comparison against ggml-cpu: %d cases, %d mismatching values\n", cases, bad);
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        for (int nt : {1, 2, 3})
            for (bool down : {false, true}) {
                std::printf("%s, %d token(s): ", down ? "Q5_K down " : "Q4_K gate/up", nt);
                std::printf("hot 1 thread ggml %.1f / avx512 %.1f GB/s;  cold 4 threads ggml %.1f / avx512 %.1f GB/s\n",
                            bench(down, true, 1, nt, true), bench(down, false, 1, nt, true), bench(down, true, 4, nt, false),
                            bench(down, false, 4, nt, false));
                std::fflush(stdout);
            }
    }
    return bad == 0 ? 0 : 1;
}
