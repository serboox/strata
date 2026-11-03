#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "strata/core/experts_used.hpp"
#include "strata/kernels/native_router.hpp"
#include "../src/qwen35/q36_kernels.cuh"

namespace {

constexpr int kExperts = 256;
constexpr int kMaxTokens = 8;
constexpr int kMaxK = 8;
constexpr int kLogitsStride = kExperts + 1;   // the engine's rows carry the shared expert's gate after the router logits
constexpr int kHidden = 2048;
constexpr unsigned long long kBlobBytes = 4096;
constexpr float kWeightTolerance = 1e-5f;
constexpr float kCombineTolerance = 1e-4f;

int failures = 0;
int checks = 0;

#define CK(x)                                                                                    \
    do {                                                                                         \
        const cudaError_t e_ = (x);                                                              \
        if (e_ != cudaSuccess) {                                                                 \
            std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));     \
            std::exit(2);                                                                        \
        }                                                                                        \
    } while (0)

void check(bool ok, const std::string& what) {
    ++checks;
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL %s\n", what.c_str());
    }
}

template <class T>
T* device_copy(const std::vector<T>& host) {
    T* d = nullptr;
    CK(cudaMalloc(&d, host.size() * sizeof(T)));
    CK(cudaMemcpy(d, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

template <class T>
T* device_zeros(size_t n) {
    T* d = nullptr;
    CK(cudaMalloc(&d, n * sizeof(T)));
    CK(cudaMemset(d, 0, n * sizeof(T)));
    return d;
}

template <class T>
std::vector<T> to_host(const T* d, size_t n) {
    std::vector<T> host(n);
    CK(cudaMemcpy(host.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return host;
}

enum class Shape { Distinct, Ties };

/// Rows of router logits. Distinct: every logit of a row differs from the others by a margin far above the fast-math error of
/// the probabilities. Ties: groups of equal logits, where the lower expert index must win.
std::vector<float> make_logits(int n_tok, Shape shape, std::mt19937& rng) {
    std::vector<float> logits((size_t) n_tok * kLogitsStride, 0.0f);
    for (int t = 0; t < n_tok; ++t) {
        std::vector<int> order(kExperts);
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);
        for (int x = 0; x < kExperts; ++x) {
            const int rank = shape == Shape::Distinct ? x : x / 4;
            logits[(size_t) t * kLogitsStride + order[x]] = 0.02f * (float) rank - 2.5f;
        }
        logits[(size_t) t * kLogitsStride + kExperts] = 7.0f;
    }
    return logits;
}

struct Route {
    std::vector<int32_t> ids;
    std::vector<double> weights;
};

/// Top-k by logit (equal logits: lower index first), the weights the softmax over the k picked.
Route reference_route(const std::vector<float>& logits, int n_tok, int k) {
    Route r;
    for (int t = 0; t < n_tok; ++t) {
        const float* row = logits.data() + (size_t) t * kLogitsStride;
        std::vector<int> order(kExperts);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return row[a] > row[b]; });
        double sum = 0;
        for (int i = 0; i < k; ++i) sum += std::exp((double) row[order[i]] - (double) row[order[0]]);
        for (int i = 0; i < k; ++i) {
            r.ids.push_back(order[i]);
            r.weights.push_back(std::exp((double) row[order[i]] - (double) row[order[0]]) / sum);
        }
    }
    return r;
}

struct Grouping {
    std::vector<unsigned long long> ptr;
    std::vector<int32_t> start, dst, tok;
};

/// The entries (token * k + pick) sorted by (expert, entry), one group per expert; with `slot_of`, only experts with a slot.
Grouping reference_grouping(const std::vector<int32_t>& ids, int k, const std::vector<int32_t>* slot_of,
                            const uint8_t* blobs) {
    std::vector<int> entries;
    for (int e = 0; e < (int) ids.size(); ++e)
        if (!slot_of || (*slot_of)[(size_t) ids[(size_t) e]] >= 0) entries.push_back(e);
    std::stable_sort(entries.begin(), entries.end(), [&](int a, int b) { return ids[(size_t) a] < ids[(size_t) b]; });
    Grouping g;
    for (size_t rank = 0; rank < entries.size(); ++rank) {
        const int id = ids[(size_t) entries[rank]];
        if (rank == 0 || id != ids[(size_t) entries[rank - 1]]) {
            const size_t slot = slot_of ? (size_t) (*slot_of)[(size_t) id] : (size_t) id;
            g.ptr.push_back((unsigned long long) (blobs + slot * kBlobBytes));
            g.start.push_back((int32_t) rank);
        }
        g.dst.push_back(entries[rank]);
        g.tok.push_back(entries[rank] / k);
    }
    g.start.push_back((int32_t) entries.size());
    return g;
}

void compare_route(const std::string& label, const std::vector<int32_t>& ids, const std::vector<float>& weights, const Route& want) {
    check(ids == want.ids, label + ": ids");
    bool weights_ok = weights.size() == want.weights.size();
    for (size_t i = 0; weights_ok && i < weights.size(); ++i)
        weights_ok = std::fabs((double) weights[i] - want.weights[i]) <= kWeightTolerance;
    check(weights_ok, label + ": weights");
}

void compare_grouping(const std::string& label, const Grouping& want, const std::vector<unsigned long long>& ptr,
                      const std::vector<int32_t>& start, const std::vector<int32_t>& dst, const std::vector<int32_t>& tok,
                      int32_t n_groups) {
    const size_t groups = want.ptr.size();
    check((size_t) n_groups == groups, label + ": group count");
    if ((size_t) n_groups != groups) return;
    check(std::equal(want.ptr.begin(), want.ptr.end(), ptr.begin()), label + ": group blobs");
    check(std::equal(want.start.begin(), want.start.end(), start.begin()), label + ": group starts");
    check(std::equal(want.dst.begin(), want.dst.end(), dst.begin()), label + ": entry order");
    check(std::equal(want.tok.begin(), want.tok.end(), tok.begin()), label + ": entry tokens");
}

void test_router(int k, int n_tok, Shape shape, std::mt19937& rng) {
    const std::vector<float> logits = make_logits(n_tok, shape, rng);
    float* d_logits = device_copy(logits);
    const size_t picks = (size_t) n_tok * k;
    int32_t* d_ids = device_zeros<int32_t>(picks);
    float* d_weights = device_zeros<float>(picks);
    cudaStream_t stream = nullptr;
    CK(cudaStreamCreate(&stream));
    // the router's own layout has no gate column; the fused kernel reads the same rows through its stride
    std::vector<float> packed((size_t) n_tok * kExperts);
    for (int t = 0; t < n_tok; ++t)
        std::copy_n(logits.begin() + (long) t * kLogitsStride, kExperts, packed.begin() + (long) t * kExperts);
    float* d_packed = device_copy(packed);
    strata::kernels::native_router_topk(d_packed, d_ids, d_weights, kExperts, k, n_tok, stream);
    CK(cudaStreamSynchronize(stream));
    const Route want = reference_route(logits, n_tok, k);
    const std::string tag = "k " + std::to_string(k) + " T " + std::to_string(n_tok) + (shape == Shape::Ties ? " ties" : "");
    compare_route("native_router_topk " + tag, to_host(d_ids, picks), to_host(d_weights, picks), want);
    CK(cudaStreamDestroy(stream));
    CK(cudaFree(d_logits));
    CK(cudaFree(d_packed));
    CK(cudaFree(d_ids));
    CK(cudaFree(d_weights));
}

void test_fused_route_group(int k, int n_tok, bool hybrid, Shape shape, std::mt19937& rng) {
    const std::vector<float> logits = make_logits(n_tok, shape, rng);
    float* d_logits = device_copy(logits);
    const size_t picks = (size_t) n_tok * k;
    int32_t* d_ids = device_zeros<int32_t>(picks);
    float* d_weights = device_zeros<float>(picks);
    std::vector<int32_t> slot_of(kExperts, -1);
    int slots = 0;
    for (int x = 0; x < kExperts; ++x)
        if (x % 3 != 0) slot_of[(size_t) x] = slots++;
    int32_t* d_slot_of = device_copy(slot_of);
    const uint8_t* blobs = reinterpret_cast<const uint8_t*>(0x100000);   // only ever added to, never read
    const size_t max_groups = std::min<size_t>(picks, kExperts) + 1;
    unsigned long long* d_ptr = device_zeros<unsigned long long>(max_groups);
    int32_t* d_start = device_zeros<int32_t>(max_groups);
    int32_t* d_n_groups = device_zeros<int32_t>(1);
    int32_t* d_dst = device_zeros<int32_t>(picks);
    int32_t* d_tok = device_zeros<int32_t>(picks);
    cudaStream_t stream = nullptr;
    CK(cudaStreamCreate(&stream));
    const bool launched = hybrid ? q36k::route_group_launch<true>(k, d_logits, d_ids, d_weights, n_tok, kLogitsStride, d_slot_of,
                                                                  blobs, kBlobBytes, d_ptr, d_start, d_n_groups, d_dst, d_tok, stream)
                                 : q36k::route_group_launch<false>(k, d_logits, d_ids, d_weights, n_tok, kLogitsStride, nullptr,
                                                                   blobs, kBlobBytes, d_ptr, d_start, d_n_groups, d_dst, d_tok,
                                                                   stream);
    CK(cudaGetLastError());
    CK(cudaStreamSynchronize(stream));
    const std::string tag = std::string(hybrid ? "hybrid " : "") + "k " + std::to_string(k) + " T " + std::to_string(n_tok) +
                            (shape == Shape::Ties ? " ties" : "");
    check(launched == q36k::route_group_supports(k), "k_route_group launch " + tag);
    const std::vector<int32_t> ids = to_host(d_ids, picks);
    const Route want = reference_route(logits, n_tok, k);
    compare_route("k_route_group " + tag, ids, to_host(d_weights, picks), want);
    const Grouping grouping = reference_grouping(ids, k, hybrid ? &slot_of : nullptr, blobs);
    const int32_t n_groups = to_host(d_n_groups, 1)[0];
    compare_grouping("k_route_group " + tag, grouping, to_host(d_ptr, max_groups), to_host(d_start, max_groups),
                     to_host(d_dst, picks), to_host(d_tok, picks), n_groups);
    CK(cudaStreamDestroy(stream));
    for (void* p : {(void*) d_logits, (void*) d_ids, (void*) d_weights, (void*) d_slot_of, (void*) d_ptr, (void*) d_start,
                    (void*) d_n_groups, (void*) d_dst, (void*) d_tok})
        CK(cudaFree(p));
}

void test_group_small(int k, int n_tok, bool hybrid, std::mt19937& rng) {
    const int n_ent = n_tok * k;
    std::vector<int32_t> ids((size_t) n_ent);
    std::uniform_int_distribution<int> pick(0, kExperts - 1);
    for (auto& id : ids) id = pick(rng);   // repeats on purpose: several tokens choosing one expert
    std::vector<int32_t> slot_of(kExperts, -1);
    int slots = 0;
    for (int x = 0; x < kExperts; ++x)
        if (x % 3 != 0) slot_of[(size_t) x] = slots++;
    int32_t* d_ids = device_copy(ids);
    int32_t* d_slot_of = device_copy(slot_of);
    const uint8_t* blobs = reinterpret_cast<const uint8_t*>(0x100000);
    const size_t max_groups = (size_t) n_ent + 1;
    unsigned long long* d_ptr = device_zeros<unsigned long long>(max_groups);
    int32_t* d_start = device_zeros<int32_t>(max_groups);
    int32_t* d_n_groups = device_zeros<int32_t>(1);
    int32_t* d_dst = device_zeros<int32_t>((size_t) n_ent);
    int32_t* d_tok = device_zeros<int32_t>((size_t) n_ent);
    if (hybrid)
        q36k::k_group_small<true><<<1, 64>>>(d_ids, n_ent, k, d_slot_of, blobs, kBlobBytes, d_ptr, d_start, d_n_groups, d_dst, d_tok);
    else
        q36k::k_group_small<false><<<1, 64>>>(d_ids, n_ent, k, nullptr, blobs, kBlobBytes, d_ptr, d_start, d_n_groups, d_dst, d_tok);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    const Grouping grouping = reference_grouping(ids, k, hybrid ? &slot_of : nullptr, blobs);
    compare_grouping(std::string("k_group_small ") + (hybrid ? "hybrid " : "") + "k " + std::to_string(k) + " T " + std::to_string(n_tok),
                     grouping, to_host(d_ptr, max_groups), to_host(d_start, max_groups), to_host(d_dst, (size_t) n_ent),
                     to_host(d_tok, (size_t) n_ent), to_host(d_n_groups, 1)[0]);
    for (void* p : {(void*) d_ids, (void*) d_slot_of, (void*) d_ptr, (void*) d_start, (void*) d_n_groups, (void*) d_dst, (void*) d_tok})
        CK(cudaFree(p));
}

void test_combine(int k, int n_tok, std::mt19937& rng) {
    std::uniform_real_distribution<float> value(-1.0f, 1.0f);
    std::vector<float> x((size_t) n_tok * kHidden), parts((size_t) n_tok * k * kHidden), weights((size_t) n_tok * k),
        shared((size_t) n_tok * kHidden), gate((size_t) n_tok);
    for (auto* v : {&x, &parts, &weights, &shared, &gate})
        for (auto& f : *v) f = value(rng);
    float* d_x = device_copy(x);
    float* d_parts = device_copy(parts);
    float* d_weights = device_copy(weights);
    float* d_shared = device_copy(shared);
    float* d_gate = device_copy(gate);
    q36k::k_moe_out_small<<<n_tok, 1024>>>(d_x, d_parts, nullptr, d_weights, k, d_shared, d_gate, 1, nullptr, nullptr, kHidden, 1e-6f);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    const std::vector<float> got = to_host(d_x, x.size());
    bool ok = true;
    for (int t = 0; t < n_tok && ok; ++t) {
        const double gate_value = 1.0 / (1.0 + std::exp(-(double) gate[(size_t) t]));
        for (int i = 0; i < kHidden && ok; ++i) {
            double acc = 0;
            for (int j = 0; j < k; ++j) acc += (double) weights[(size_t) t * k + j] * parts[((size_t) t * k + j) * kHidden + i];
            acc += gate_value * shared[(size_t) t * kHidden + i];
            ok = std::fabs((double) got[(size_t) t * kHidden + i] - ((double) x[(size_t) t * kHidden + i] + acc)) <= kCombineTolerance;
        }
    }
    check(ok, "k_moe_out_small k " + std::to_string(k) + " T " + std::to_string(n_tok));
    for (void* p : {(void*) d_x, (void*) d_parts, (void*) d_weights, (void*) d_shared, (void*) d_gate}) CK(cudaFree(p));
}

void test_resolve() {
    int64_t k = 0;
    std::string err;
    check(strata::core::resolve_experts_used(nullptr, 8, k, err) && k == 8, "resolve: unset keeps the file value");
    check(strata::core::resolve_experts_used("", 8, k, err) && k == 8, "resolve: empty keeps the file value");
    check(strata::core::resolve_experts_used("6", 8, k, err) && k == 6, "resolve: 6");
    check(strata::core::resolve_experts_used("1", 8, k, err) && k == 1, "resolve: 1");
    check(strata::core::resolve_experts_used("8", 8, k, err) && k == 8, "resolve: the file value itself");
    for (const char* bad : {"0", "-1", "9", "six", "6x", " ", "6.5", "99999999999999999999"}) {
        err.clear();
        check(!strata::core::resolve_experts_used(bad, 8, k, err) && !err.empty(), std::string("resolve: refuses '") + bad + "'");
    }
}

}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::fprintf(stderr, "experts_used_parity: no CUDA device\n");
        return 2;
    }
    std::mt19937 rng(20260611);
    test_resolve();
    for (int k = 1; k <= kMaxK; ++k) {
        for (int n_tok : {1, 3, kMaxTokens}) {
            for (Shape shape : {Shape::Distinct, Shape::Ties}) {
                test_router(k, n_tok, shape, rng);
                for (bool hybrid : {false, true}) test_fused_route_group(k, n_tok, hybrid, shape, rng);
            }
            for (bool hybrid : {false, true}) test_group_small(k, n_tok, hybrid, rng);
            test_combine(k, n_tok, rng);
        }
    }
    check(!q36k::route_group_supports(0) && !q36k::route_group_supports(kMaxK + 1), "route_group_supports bounds");
    check(!strata::kernels::native_router_topk_supported(kExperts, 0) && !strata::kernels::native_router_topk_supported(kExperts, kMaxK + 1) &&
              strata::kernels::native_router_topk_supported(kExperts, 1) && strata::kernels::native_router_topk_supported(kExperts, kMaxK),
          "native_router_topk_supported bounds");
    if (failures) {
        std::fprintf(stderr, "experts_used_parity: %d FAILED\n", failures);
        return 1;
    }
    std::printf("experts_used_parity: ok (%d checks, k 1..%d, T 1/3/%d)\n", checks, kMaxK, kMaxTokens);
    return 0;
}
