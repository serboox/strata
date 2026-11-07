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

ggml_type ggml_type_of(int type) { return type == 10 ? GGML_TYPE_Q2_K : type == 13 ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K; }

size_t block_bytes_of(int type) { return type == 10 ? sizeof(block_q2_K) : type == 13 ? sizeof(block_q5_K) : sizeof(block_q4_K); }

// valid blocks of a K-quant row: random quants and scales, finite half-precision super-scales where the type keeps them
void fill_blocks(uint8_t* p, size_t bytes, int type) {
    if (type != 10) { fill_weights(p, bytes, block_bytes_of(type)); return; }
    for (size_t i = 0; i < bytes; ++i) p[i] = (uint8_t) rng();
    std::uniform_real_distribution<float> scale(0.0005f, 0.06f);
    for (size_t b = 0; b < bytes / sizeof(block_q2_K); ++b) {
        ggml_fp16_t d = ggml_fp32_to_fp16(scale(rng)), dmin = ggml_fp32_to_fp16(scale(rng));
        std::memcpy(p + b * sizeof(block_q2_K) + 80, &d, 2);
        std::memcpy(p + b * sizeof(block_q2_K) + 82, &dmin, 2);
    }
}

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

// Plain rows of either type, as a CPU path without a fused gate/up step calls them: every row length the kernel accepts, a row range
// that does not start at 0, a row stride larger than the row, a weight address that is not 64-byte aligned, and more tokens than
// one pass of the kernel takes
int check_rows(int type, int rows, int n, int nt, size_t pad, size_t offset) {
    const size_t block = block_bytes_of(type);
    const size_t row = (size_t) n / 256 * block;
    const size_t stride = row + pad;
    const int first = 3;
    std::vector<uint8_t> blob(rows * stride + 128);
    uint8_t* w = blob.data() + offset;
    for (int r = 0; r < rows; ++r) fill_blocks(w + (size_t) r * stride, row, type);
    std::vector<std::vector<block_q8_K>> act(nt);
    std::vector<const void*> ap(nt);
    for (int t = 0; t < nt; ++t) { act[t] = make_activation(n); ap[t] = act[t].data(); }
    std::vector<std::vector<float>> got(nt, std::vector<float>(rows, 12345.f));
    std::vector<float*> gp(nt);
    for (int t = 0; t < nt; ++t) gp[t] = got[t].data();
    if (!kq512_rows_supported(type, n)) return 1;
    kq512_rows(type, w, stride, n, ap.data(), nt, gp.data(), first, rows);
    const auto dot = ggml_get_type_traits_cpu(ggml_type_of(type))->vec_dot;
    int bad = 0;
    for (int t = 0; t < nt; ++t) {
        for (int r = 0; r < first; ++r) bad += !same(got[t][r], 12345.f);   // rows before the range stay untouched
        for (int r = first; r < rows; ++r) {
            float s = 0.f;
            dot(n, &s, 0, w + (size_t) r * stride, 0, ap[t], 0, 1);
            bad += !same(got[t][r], s);
        }
    }
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

// Plain rows of one type (the call pattern of a runtime that does not fuse gate and up), ggml's per-row dot against the kernel
struct RowsJob {
    const uint8_t* w;
    size_t rows, row;
    int type, n, reps;
    bool ggml;
    const void* act;
    float* sink;
};

void* run_rows(void* arg) {
    RowsJob* j = (RowsJob*) arg;
    float out[16];
    const auto dot = ggml_get_type_traits_cpu(ggml_type_of(j->type))->vec_dot;
    for (int rep = 0; rep < j->reps; ++rep)
        for (size_t r0 = 0; r0 < j->rows; r0 += 16) {
            const int count = (int) std::min<size_t>(16, j->rows - r0);
            if (j->ggml) {
                for (int i = 0; i < count; ++i) dot(j->n, &out[i], 0, j->w + (r0 + i) * j->row, 0, j->act, 0, 1);
            } else {
                float* op[1] = {out};
                const void* ap[1] = {j->act};
                kq512_rows(j->type, j->w + r0 * j->row, j->row, j->n, ap, 1, op, 0, count);
            }
        }
    *j->sink = out[0];
    return nullptr;
}

double bench_rows(int type, int n, bool use_ggml, int threads, bool hot) {
    const size_t block = block_bytes_of(type);
    const size_t row = (size_t) n / 256 * block;
    const size_t rows = hot ? 600 : (size_t) (192u << 20) / row;
    std::vector<std::vector<uint8_t>> w(threads, std::vector<uint8_t>(rows * row + 4096));
    std::vector<block_q8_K> act = make_activation(n);
    std::vector<RowsJob> jobs(threads);
    std::vector<float> sinks(threads);
    for (int i = 0; i < threads; ++i) {
        fill_blocks(w[i].data(), rows * row, type);
        jobs[i] = {w[i].data(), rows, row, type, n, hot ? 3000 : 1, use_ggml, act.data(), &sinks[i]};
    }
    double best = 1e9;
    for (int rep = 0; rep < 3; ++rep) {
        std::vector<pthread_t> th(threads);
        const double a = now();
        for (int i = 0; i < threads; ++i) pthread_create(&th[i], nullptr, run_rows, &jobs[i]);
        for (int i = 0; i < threads; ++i) pthread_join(th[i], nullptr);
        best = std::min(best, now() - a);
    }
    return (double) threads * rows * row * jobs[0].reps / best / 1e9;
}

// ggml's own dot product of any K-quant against a Q8_K activation, rows quantized from random floats so every block is valid
struct TypeJob {
    const uint8_t* w;
    size_t rows, row;
    int n, reps;
    ggml_type type;
    const void* act;
    float* sink;
};

void* run_type(void* arg) {
    TypeJob* j = (TypeJob*) arg;
    float out = 0.f;
    const auto dot = ggml_get_type_traits_cpu(j->type)->vec_dot;
    for (int rep = 0; rep < j->reps; ++rep)
        for (size_t r = 0; r < j->rows; ++r) dot(j->n, &out, 0, j->w + r * j->row, 0, j->act, 0, 1);
    *j->sink = out;
    return nullptr;
}

double bench_type(ggml_type type, int n, int threads, bool hot) {
    const auto* traits = ggml_get_type_traits(type);
    const size_t row = ggml_row_size(type, n);
    const size_t rows = hot ? 600 : (size_t) (192u << 20) / row;
    std::normal_distribution<float> g(0.f, 1.f);
    std::vector<float> x((size_t) n);
    std::vector<std::vector<uint8_t>> w(threads, std::vector<uint8_t>(rows * row + 4096));
    const size_t distinct = std::min<size_t>(rows, 512);
    for (int t = 0; t < threads; ++t) {
        for (size_t r = 0; r < distinct; ++r) {
            for (auto& v : x) v = g(rng);
            traits->from_float_ref(x.data(), w[t].data() + r * row, n);
        }
        for (size_t r = distinct; r < rows; ++r) std::memcpy(w[t].data() + r * row, w[t].data() + (r % distinct) * row, row);
    }
    std::vector<block_q8_K> act = make_activation(n);
    std::vector<TypeJob> jobs(threads);
    std::vector<float> sinks(threads);
    for (int i = 0; i < threads; ++i) jobs[i] = {w[i].data(), rows, row, n, hot ? 3000 : 1, type, act.data(), &sinks[i]};
    double best = 1e9;
    for (int rep = 0; rep < 3; ++rep) {
        std::vector<pthread_t> th(threads);
        const double a = now();
        for (int i = 0; i < threads; ++i) pthread_create(&th[i], nullptr, run_type, &jobs[i]);
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
    for (int type : {10, 12, 13})
        for (int blocks = 1; blocks <= 16; ++blocks)
            for (int nt : {1, 3, 9})
                for (int variant = 0; variant < 3; ++variant) {
                    bad += check_rows(type, 29, blocks * 256, nt, variant == 1 ? 16 : 0, variant == 2 ? 7 : 0);
                    ++cases;
                }
    bad += (int) kq512_rows_supported(12, 0) + (int) kq512_rows_supported(12, -256) + (int) kq512_rows_supported(12, 17 * 256) +
           (int) kq512_rows_supported(12, 300) + (int) kq512_rows_supported(11, 256) + (int) kq512_rows_supported(14, 256) + (int) kq512_gu_supported(12, -256) +
           (int) kq512_down_supported(13, 0) + !kq512_rows_supported(13, 4096);
    ++cases;
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
    if (argc > 1 && std::strcmp(argv[1], "--bench-rows") == 0) {
        for (int type : {10, 12, 13})
            for (int n : {2048, 1536}) {
                std::printf("%s rows of %d values: ", type == 13 ? "Q5_K" : type == 10 ? "Q2_K" : "Q4_K", n);
                std::printf("hot 1 thread ggml %.1f / avx512 %.1f GB/s;  cold 4 threads ggml %.1f / avx512 %.1f GB/s\n",
                            bench_rows(type, n, true, 1, true), bench_rows(type, n, false, 1, true), bench_rows(type, n, true, 4, false),
                            bench_rows(type, n, false, 4, false));
                std::fflush(stdout);
            }
    }
    if (argc > 1 && std::strcmp(argv[1], "--bench-types") == 0) {
        for (ggml_type type : {GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K})
            for (int n : {2048, 1536}) {
                std::printf("ggml %s rows of %d values: hot 1 thread %.1f GB/s;  cold 4 threads %.1f GB/s\n", ggml_type_name(type), n,
                            bench_type(type, n, 1, true), bench_type(type, n, 4, false));
                std::fflush(stdout);
            }
    }
    return bad == 0 ? 0 : 1;
}
