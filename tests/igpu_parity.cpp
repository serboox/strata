#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "ggml-cpu.h"
#include "ggml.h"
#include "strata/kernels/igpu_experts.hpp"

namespace ig = strata::kernels::igpu;

namespace {

int64_t gEmbd = 512, gFf = 256, gExperts = 16;
constexpr double kShare = 0.5;
constexpr int kResident = 2, kMaxEntries = 16, kRows = 6;
// ggml-vulkan runs up to 8 entries per call through its vector kernels and more through the matrix ones: both are checked
constexpr int kCallSizes[] = {1, 3, 8, 9, 40};
constexpr size_t kCallSizeCount = sizeof(kCallSizes) / sizeof(kCallSizes[0]);

// The iGPU may not be worse than a few times the CPU path's own distance to exact arithmetic (both quantise the
// activations differently); the CPU-to-iGPU distance has an absolute cap on top.
constexpr double kVsIdealFactor = 3.0, kVsIdealSlack = 1e-3, kVsCpuMax = 5e-2;

struct Case {
    const char* name;
    ggml_type gu, d;
};

const Case kCases[] = {
    {"IQ4_XS / IQ4_XS", GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_XS}, {"IQ3_S / IQ4_XS", GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS},
    {"IQ3_S / IQ3_S", GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_S},     {"Q4_K / Q5_K", GGML_TYPE_Q4_K, GGML_TYPE_Q5_K},
    {"Q4_K / Q6_K", GGML_TYPE_Q4_K, GGML_TYPE_Q6_K},         {"Q5_K / Q6_K", GGML_TYPE_Q5_K, GGML_TYPE_Q6_K},
    {"Q8_0 / Q8_0", GGML_TYPE_Q8_0, GGML_TYPE_Q8_0},         {"IQ4_XS / Q5_K", GGML_TYPE_IQ4_XS, GGML_TYPE_Q5_K},
};

std::mt19937 rng(20260311);

struct Tensor3 {
    ggml_type type;
    int64_t row_len, rows;   // per expert: `rows` rows of `row_len` elements
    size_t row_bytes, expert_bytes;
    std::vector<uint8_t> data;   // gExperts experts
};

Tensor3 make_tensor(ggml_type type, int64_t row_len, int64_t rows, float sigma) {
    Tensor3 t{type, row_len, rows, ggml_row_size(type, row_len), 0, {}};
    t.expert_bytes = t.row_bytes * (size_t) rows;
    const int64_t n_rows = rows * gExperts;
    std::vector<float> f((size_t) (n_rows * row_len));
    std::normal_distribution<float> g(0.f, sigma);
    for (float& v : f) v = g(rng);
    std::vector<float> imatrix((size_t) row_len, 1.0f);
    t.data.resize(t.row_bytes * (size_t) n_rows);
    ggml_quantize_chunk(type, f.data(), t.data.data(), 0, n_rows, row_len, ggml_quantize_requires_imatrix(type) ? imatrix.data() : nullptr);
    return t;
}

void dequant_row(const Tensor3& t, int expert, int64_t row, float* out) {
    const uint8_t* p = t.data.data() + (size_t) expert * t.expert_bytes + (size_t) row * t.row_bytes;
    ggml_get_type_traits(t.type)->to_float(p, out, t.row_len);
}

void cpu_expert(const Tensor3& g, const Tensor3& u, const Tensor3& d, int e, const float* x, float* out) {
    const ggml_type vg = ggml_get_type_traits_cpu(g.type)->vec_dot_type, vd = ggml_get_type_traits_cpu(d.type)->vec_dot_type;
    std::vector<uint8_t> xq(ggml_row_size(vg, gEmbd)), aq(ggml_row_size(vd, gFf));
    ggml_get_type_traits_cpu(vg)->from_float(x, xq.data(), gEmbd);
    std::vector<float> act((size_t) gFf);
    for (int64_t r = 0; r < gFf; ++r) {
        float a = 0.f, b = 0.f;
        ggml_get_type_traits_cpu(g.type)->vec_dot((int) gEmbd, &a, 0, g.data.data() + (size_t) e * g.expert_bytes + (size_t) r * g.row_bytes, 0, xq.data(), 0, 1);
        ggml_get_type_traits_cpu(u.type)->vec_dot((int) gEmbd, &b, 0, u.data.data() + (size_t) e * u.expert_bytes + (size_t) r * u.row_bytes, 0, xq.data(), 0, 1);
        act[(size_t) r] = a / (1.f + std::exp(-a)) * b;
    }
    ggml_get_type_traits_cpu(vd)->from_float(act.data(), aq.data(), gFf);
    for (int64_t r = 0; r < gEmbd; ++r)
        ggml_get_type_traits_cpu(d.type)->vec_dot((int) gFf, &out[r], 0, d.data.data() + (size_t) e * d.expert_bytes + (size_t) r * d.row_bytes, 0, aq.data(), 0, 1);
}

void ideal_expert(const Tensor3& g, const Tensor3& u, const Tensor3& d, int e, const float* x, float* out) {
    std::vector<float> wrow((size_t) std::max(gEmbd, gFf));
    std::vector<double> act((size_t) gFf);
    for (int64_t r = 0; r < gFf; ++r) {
        double a = 0, b = 0;
        dequant_row(g, e, r, wrow.data());
        for (int64_t i = 0; i < gEmbd; ++i) a += (double) wrow[(size_t) i] * x[i];
        dequant_row(u, e, r, wrow.data());
        for (int64_t i = 0; i < gEmbd; ++i) b += (double) wrow[(size_t) i] * x[i];
        act[(size_t) r] = a / (1.0 + std::exp(-a)) * b;
    }
    for (int64_t r = 0; r < gEmbd; ++r) {
        double s = 0;
        dequant_row(d, e, r, wrow.data());
        for (int64_t i = 0; i < gFf; ++i) s += (double) wrow[(size_t) i] * act[(size_t) i];
        out[r] = (float) s;
    }
}

double rel_error(const float* a, const float* b, int64_t n) {
    double num = 0, den = 0;
    for (int64_t i = 0; i < n; ++i) {
        num += ((double) a[i] - b[i]) * ((double) a[i] - b[i]);
        den += (double) b[i] * b[i];
    }
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

struct Spread {
    double max = 0, sum = 0;
    int n = 0;
    void add(double v) {
        max = std::max(max, v);
        sum += v;
        ++n;
    }
    double mean() const { return n ? sum / n : 0; }
};

struct LayerData {
    Tensor3 gate, up, down;
};

}  // namespace


int bench() {
    gEmbd = 2048;
    gFf = 512;
    gExperts = 16;
    constexpr int kLayers = 4, kRepeats = 40;
    const int sizes[] = {1, 2, 4, 8, 16};
    std::vector<float> rows((size_t) (32 * gEmbd));
    for (float& v : rows) v = std::normal_distribution<float>(0.f, 1.f)(rng);
    const Case formats[] = {{"IQ4_XS / IQ4_XS", GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_XS}, {"IQ3_S / IQ4_XS", GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS}};
    for (const Case& c : formats) {
        const LayerData data = {make_tensor(c.gu, gEmbd, gFf, 0.03f), make_tensor(c.gu, gEmbd, gFf, 0.03f), make_tensor(c.d, gFf, gEmbd, 0.03f)};
        std::vector<ig::ExpertSource> src((size_t) kLayers, ig::ExpertSource{(int) c.gu, (int) c.d, data.gate.data.data(), data.up.data.data(), data.down.data.data(),
                                                                          data.gate.expert_bytes, data.down.expert_bytes, 0});
        ig::Options opt;
        opt.n_embd = gEmbd;
        opt.n_ff = gFf;
        opt.n_expert = gExperts;
        opt.share = kShare;
        if (const char* dev = std::getenv("STRATA_IGPU_DEVICE")) opt.device = dev;
        std::string err;
        auto share = ig::ExpertShare::create(opt, src, err);
        if (!share) {
            std::printf("cannot start: %s\n", err.c_str());
            return 2;
        }
        std::printf("%s, 2048 x 512, %d experts per layer on %s\n%8s %12s %12s %14s\n", c.name, share->experts_per_layer(), share->device_name().c_str(),
                    "entries", "median ms", "p90 ms", "us per entry");
        for (int n : sizes) {
            std::vector<float> out((size_t) (n * gEmbd));
            std::vector<ig::Entry> entries((size_t) n);
            for (int i = 0; i < n; ++i) entries[(size_t) i] = {rows.data() + (size_t) i * gEmbd, i % share->experts_per_layer(), out.data() + (size_t) i * gEmbd};
            std::vector<double> ms;
            ig::Stats before;
            for (int r = -3; r < kRepeats; ++r) {
                if (r == 0) before = share->stats();
                const auto t0 = std::chrono::steady_clock::now();
                share->submit((size_t) (r + 3) % kLayers, entries.data(), n);
                share->wait();
                if (r >= 0) ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
            std::sort(ms.begin(), ms.end());
            const ig::Stats after = share->stats();
            std::printf("%8d %12.3f %12.3f %14.1f %10.3f %10.3f %10.3f\n", n, ms[ms.size() / 2], ms[ms.size() * 9 / 10], 1000.0 * ms[ms.size() / 2] / n,
                        (after.graph_ms - before.graph_ms) / kRepeats, (after.io_ms - before.io_ms) / kRepeats, (after.run_ms - before.run_ms) / kRepeats);
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    ggml_cpu_init();
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) return bench();
    std::vector<float> rows((size_t) (kRows * gEmbd));
    for (float& v : rows) v = std::normal_distribution<float>(0.f, 1.f)(rng);
    int failures = 0;
    Spread by_call_size[kCallSizeCount];
    std::printf("%-18s %10s %10s %10s %10s %10s %10s %9s\n", "formats (gate,up/down)", "gpu~cpu max", "mean", "gpu~exact max", "mean",
                "cpu~exact max", "mean", "ms/call");
    for (const Case& c : kCases) {
        LayerData layer[2] = {{make_tensor(c.gu, gEmbd, gFf, 0.03f), make_tensor(c.gu, gEmbd, gFf, 0.03f), make_tensor(c.d, gFf, gEmbd, 0.03f)},
                              {make_tensor(c.gu, gEmbd, gFf, 0.03f), make_tensor(c.gu, gEmbd, gFf, 0.03f), make_tensor(c.d, gFf, gEmbd, 0.03f)}};
        std::vector<ig::ExpertSource> src(3);   // layer 1 is left to the other devices
        const int which[3] = {0, -1, 1};
        for (int li = 0; li < 3; ++li) {
            if (which[li] < 0) continue;
            const LayerData& l = layer[which[li]];
            src[(size_t) li] = {(int) c.gu, (int) c.d, l.gate.data.data(), l.up.data.data(), l.down.data.data(),
                                l.gate.expert_bytes, l.down.expert_bytes, kResident};
        }
        ig::Options opt;
        opt.n_embd = gEmbd;
        opt.n_ff = gFf;
        opt.n_expert = gExperts;
        opt.share = kShare;
        opt.max_entries = kMaxEntries;
        if (const char* dev = std::getenv("STRATA_IGPU_DEVICE")) opt.device = dev;
        std::string err;
        auto share = ig::ExpertShare::create(opt, src, err);
        if (!share) {
            std::printf("%-18s cannot start: %s\n", c.name, err.c_str());
            return 2;
        }
        int bad_owns = 0;
        for (int li = 0; li < 3; ++li)
            for (int e = 0; e < gExperts; ++e) {
                const bool want = which[li] >= 0 && ((e - kResident + gExperts) % gExperts) < share->experts_per_layer();
                bad_owns += share->owns((size_t) li, e) != want;
            }
        std::vector<int> owned;
        for (int e = 0; e < gExperts; ++e)
            if (share->owns(0, e)) owned.push_back(e);

        Spread vs_cpu, vs_ideal, cpu_vs_ideal;
        double total_ms = 0;
        int calls = 0;
        for (int li : {0, 2}) {
            const LayerData& l = layer[which[li]];
            for (size_t k = 0; k < kCallSizeCount; ++k) {
                const int n = kCallSizes[k];
                std::vector<ig::Entry> entries((size_t) n);
                std::vector<float> got((size_t) (n * gEmbd)), cpu((size_t) gEmbd), ideal((size_t) gEmbd);
                for (int i = 0; i < n; ++i)
                    entries[(size_t) i] = {rows.data() + (size_t) (i % kRows) * gEmbd, owned[(size_t) i % owned.size()], got.data() + (size_t) i * gEmbd};
                const auto t0 = std::chrono::steady_clock::now();
                share->submit((size_t) li, entries.data(), n);
                share->wait();
                total_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                ++calls;
                for (int i = 0; i < n; ++i) {
                    const ig::Entry& en = entries[(size_t) i];
                    cpu_expert(l.gate, l.up, l.down, en.expert, en.x, cpu.data());
                    ideal_expert(l.gate, l.up, l.down, en.expert, en.x, ideal.data());
                    const double to_cpu = rel_error(en.out, cpu.data(), gEmbd);
                    vs_cpu.add(to_cpu);
                    vs_ideal.add(rel_error(en.out, ideal.data(), gEmbd));
                    cpu_vs_ideal.add(rel_error(cpu.data(), ideal.data(), gEmbd));
                    by_call_size[k].add(to_cpu);
                }
            }
        }
        const bool ok = bad_owns == 0 && vs_ideal.max <= kVsIdealFactor * cpu_vs_ideal.max + kVsIdealSlack && vs_cpu.max <= kVsCpuMax;
        std::printf("%-18s %10.2e %10.2e %10.2e %10.2e %10.2e %10.2e %9.2f  %s%s\n", c.name, vs_cpu.max, vs_cpu.mean(), vs_ideal.max,
                    vs_ideal.mean(), cpu_vs_ideal.max, cpu_vs_ideal.mean(), total_ms / calls, ok ? "ok" : "FAIL",
                    bad_owns ? " (owns() table wrong)" : "");
        failures += !ok;
    }
    std::printf("iGPU against ggml-cpu, worst row over all formats, by entries per call:");
    for (size_t k = 0; k < kCallSizeCount; ++k) std::printf("  %d: %.2e", kCallSizes[k], by_call_size[k].max);
    std::printf("\n");
    std::printf("%s\n", failures ? "igpu_parity: FAILED" : "igpu_parity: all formats within tolerance");
    return failures ? 1 : 0;
}
