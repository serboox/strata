#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace strata::kernels::igpu {

/// One routed-expert row the iGPU computes: the layer's input row, the expert, and where its output row (n_embd floats,
/// not yet weighted by the router) goes.
struct Entry {
    const float* x = nullptr;
    int expert = 0;
    float* out = nullptr;
};

/// One layer's routed experts in the GGUF's own layout: [n_embd, n_ff, n_expert] for gate and up, [n_ff, n_embd,
/// n_expert] for down.  A layer without `gate` is left to the other devices.
struct ExpertSource {
    int gu_type = 0, d_type = 0;                  // ggml_type of gate/up and of down
    const uint8_t *gate = nullptr, *up = nullptr, *down = nullptr;
    size_t gu_bytes = 0, d_bytes = 0;             // one expert's bytes in a gate/up tensor and in the down tensor
    int resident = 0;                             // experts [0, resident) are in VRAM slots at start
};

struct Options {
    int64_t n_embd = 0, n_ff = 0, n_expert = 0;
    double share = 0.0;      // fraction of every layer's experts the iGPU holds, 0 < share < 1
    int max_entries = 256;   // most entries per graph run; a bigger call is run in pieces
    std::string device;      // a Vulkan device index or part of its name; empty: the first integrated GPU
};

struct Stats {
    long long entries = 0, calls = 0;
    double compute_ms = 0.0;   // worker, per call: from the start of the job to its last row written
    double graph_ms = 0.0;     // ... of which building and placing the graph
    double io_ms = 0.0;        // ... of which uploading the rows and downloading the results
    double run_ms = 0.0;       // ... of which running the graph on the device
    double wait_ms = 0.0;      // caller: time blocked in wait() after its own work was done
};

/// Share of the routed experts the iGPU takes, from STRATA_IGPU_EXPERTS (unset, 0 or invalid: 0).
inline double share_from_env() {
    const char* v = std::getenv("STRATA_IGPU_EXPERTS");
    if (v == nullptr || *v == '\0') return 0.0;
    char* end = nullptr;
    const double s = std::strtod(v, &end);
    return (end != v && *end == '\0' && s > 0.0 && s < 1.0) ? s : 0.0;
}

#if defined(STRATA_IGPU_VULKAN)

/// A fixed share of every layer's routed experts, held in iGPU buffers (compact copies, not the whole model), computed
/// by a worker thread while the caller runs the CPU pool.  Per entry the output is the expert FFN (down(silu(gate x) *
/// up x)) of the GGUF weights, as the CPU pool writes it.
class ExpertShare {
public:
    static std::unique_ptr<ExpertShare> create(const Options& opt, const std::vector<ExpertSource>& layers, std::string& err);
    ~ExpertShare();
    ExpertShare(const ExpertShare&) = delete;
    ExpertShare& operator=(const ExpertShare&) = delete;

    bool owns(size_t layer, int expert) const;
    /// Starts the entries of one layer on the worker; the rows `x` points to stay valid until wait() returns.
    void submit(size_t layer, const Entry* entries, int n);
    void wait();
    Stats stats() const;
    void reset_stats();
    int experts_per_layer() const;
    size_t resident_bytes() const;
    std::string device_name() const;

private:
    struct Impl;
    explicit ExpertShare(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

#else

class ExpertShare {
public:
    static std::unique_ptr<ExpertShare> create(const Options&, const std::vector<ExpertSource>&, std::string& err) {
        err = "this binary was built without STRATA_IGPU_VULKAN";
        return nullptr;
    }
    bool owns(size_t, int) const { return false; }
    void submit(size_t, const Entry*, int) {}
    void wait() {}
    Stats stats() const { return {}; }
    void reset_stats() {}
    int experts_per_layer() const { return 0; }
    size_t resident_bytes() const { return 0; }
    std::string device_name() const { return {}; }
};

#endif

}  // namespace strata::kernels::igpu
