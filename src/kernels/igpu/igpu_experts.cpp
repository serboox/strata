#include "strata/kernels/igpu_experts.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <set>
#include <thread>
#include <utility>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"
#include "ggml.h"

namespace strata::kernels::igpu {
namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

std::string lower(std::string s) {
    for (char& c : s) c = (char) std::tolower((unsigned char) c);
    return s;
}

bool all_digits(const std::string& s) {
    return !s.empty() && std::all_of(s.begin(), s.end(), [](char c) { return std::isdigit((unsigned char) c); });
}

struct LayerWeights {
    ggml_tensor *gate = nullptr, *up = nullptr, *down = nullptr;
    std::vector<int32_t> compact;   // expert -> its index in the compact tensors, -1: not here
};

struct Graph {
    ggml_context* ctx = nullptr;
    ggml_cgraph* gf = nullptr;
    ggml_tensor *x = nullptr, *ids = nullptr, *out = nullptr;
};

/// Every entry is its own "token" with one expert, so a call takes any set of (row, expert) pairs.
Graph build_graph(const LayerWeights& w, int64_t n_embd, int n) {
    Graph g;
    ggml_init_params p = {ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(32, false), nullptr, true};
    g.ctx = ggml_init(p);
    g.x = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, n_embd, 1, n);
    g.ids = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, 1, n);
    ggml_set_input(g.x);
    ggml_set_input(g.ids);
    ggml_tensor* gate = ggml_mul_mat_id(g.ctx, w.gate, g.x, g.ids);
    ggml_tensor* up = ggml_mul_mat_id(g.ctx, w.up, g.x, g.ids);
    ggml_tensor* act = ggml_swiglu_split(g.ctx, gate, up);
    g.out = ggml_mul_mat_id(g.ctx, w.down, act, g.ids);
    ggml_set_output(g.out);
    g.gf = ggml_new_graph_custom(g.ctx, 32, false);
    ggml_build_forward_expand(g.gf, g.out);
    return g;
}

ggml_backend_dev_t pick_device(const std::string& want, std::string& err) {
    ggml_backend_reg_t reg = ggml_backend_vk_reg();
    const size_t n = reg != nullptr ? ggml_backend_reg_dev_count(reg) : 0;
    std::string listing;
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, i);
        listing += (i ? "; " : "") + std::to_string(i) + " = " + ggml_backend_dev_description(dev);
    }
    if (n == 0) {
        err = "no Vulkan device";
        return nullptr;
    }
    if (want.empty()) {
        for (size_t i = 0; i < n; ++i) {
            ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_IGPU) return dev;
        }
        err = "no integrated GPU among the Vulkan devices (" + listing + "); STRATA_IGPU_DEVICE picks one";
        return nullptr;
    }
    if (all_digits(want)) {
        const size_t i = (size_t) std::atoll(want.c_str());
        if (i < n) return ggml_backend_reg_dev_get(reg, i);
    } else {
        for (size_t i = 0; i < n; ++i) {
            ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, i);
            if (lower(ggml_backend_dev_description(dev)).find(lower(want)) != std::string::npos) return dev;
        }
    }
    err = "no Vulkan device matches \"" + want + "\" (" + listing + ")";
    return nullptr;
}

}  // namespace

struct ExpertShare::Impl {
    Options opt;
    ggml_backend_dev_t dev = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t galloc = nullptr;
    ggml_context* wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    std::vector<LayerWeights> layers;
    int per_layer = 0;
    size_t bytes = 0;
    std::string device_name;

    std::thread worker;
    std::mutex mu;
    std::condition_variable cv_job, cv_done;
    bool has_job = false, stop = false;
    size_t job_layer = 0;
    std::vector<Entry> job;
    Stats st;
    std::vector<float> xbuf, obuf;
    std::vector<int32_t> idbuf;

    ~Impl() {
        {
            std::lock_guard<std::mutex> lock(mu);
            stop = true;
        }
        cv_job.notify_all();
        if (worker.joinable()) worker.join();
        if (galloc != nullptr) ggml_gallocr_free(galloc);
        if (wbuf != nullptr) ggml_backend_buffer_free(wbuf);
        if (wctx != nullptr) ggml_free(wctx);
        if (backend != nullptr) ggml_backend_free(backend);
    }

    bool probe_types(int gu_type, int d_type, std::string& err) {
        ggml_init_params p = {ggml_tensor_overhead() * 8, nullptr, true};
        ggml_context* ctx = ggml_init(p);
        LayerWeights w;
        w.gate = ggml_new_tensor_3d(ctx, (ggml_type) gu_type, opt.n_embd, opt.n_ff, 1);
        w.up = ggml_new_tensor_3d(ctx, (ggml_type) gu_type, opt.n_embd, opt.n_ff, 1);
        w.down = ggml_new_tensor_3d(ctx, (ggml_type) d_type, opt.n_ff, opt.n_embd, 1);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        bool ok = buf != nullptr;
        if (ok) {
            Graph g = build_graph(w, opt.n_embd, 1);
            for (int i = 0; i < ggml_graph_n_nodes(g.gf); ++i) ok = ok && ggml_backend_supports_op(backend, ggml_graph_node(g.gf, i));
            ggml_free(g.ctx);
        }
        if (!ok)
            err = std::string("the iGPU does not run the expert formats ") + ggml_type_name((ggml_type) gu_type) + " / " +
                  ggml_type_name((ggml_type) d_type);
        if (buf != nullptr) ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return ok;
    }

    bool init(const Options& o, const std::vector<ExpertSource>& src, std::string& err) {
        opt = o;
        dev = pick_device(opt.device, err);
        if (dev == nullptr) return false;
        device_name = ggml_backend_dev_description(dev);
        backend = ggml_backend_dev_init(dev, nullptr);
        if (backend == nullptr) {
            err = "cannot open the Vulkan device " + device_name;
            return false;
        }
        const int64_t ne = opt.n_expert;
        per_layer = (int) std::max<int64_t>(1, std::min<int64_t>(ne - 1, (int64_t) (opt.share * (double) ne + 0.5)));
        std::set<std::pair<int, int>> probed;
        int n_layers = 0;
        for (const ExpertSource& s : src) {
            if (s.gate == nullptr) continue;
            ++n_layers;
            if (probed.insert({s.gu_type, s.d_type}).second && !probe_types(s.gu_type, s.d_type, err)) return false;
        }
        if (n_layers == 0) {
            err = "no layer to take experts from";
            return false;
        }
        ggml_init_params p = {ggml_tensor_overhead() * (src.size() * 3 + 4), nullptr, true};
        wctx = ggml_init(p);
        layers.resize(src.size());
        for (size_t li = 0; li < src.size(); ++li) {
            const ExpertSource& s = src[li];
            if (s.gate == nullptr) continue;
            LayerWeights& w = layers[li];
            w.gate = ggml_new_tensor_3d(wctx, (ggml_type) s.gu_type, opt.n_embd, opt.n_ff, per_layer);
            w.up = ggml_new_tensor_3d(wctx, (ggml_type) s.gu_type, opt.n_embd, opt.n_ff, per_layer);
            w.down = ggml_new_tensor_3d(wctx, (ggml_type) s.d_type, opt.n_ff, opt.n_embd, per_layer);
            if (ggml_nbytes(w.gate) != (size_t) per_layer * s.gu_bytes || ggml_nbytes(w.down) != (size_t) per_layer * s.d_bytes) {
                err = "the GGUF's expert byte sizes differ from ggml's for " + std::string(ggml_type_name((ggml_type) s.gu_type));
                return false;
            }
            bytes += ggml_nbytes(w.gate) + ggml_nbytes(w.up) + ggml_nbytes(w.down);
        }
        size_t mem_free = 0, mem_total = 0;
        ggml_backend_dev_memory(dev, &mem_free, &mem_total);
        if (mem_total != 0 && bytes > mem_total) {
            err = "the share needs " + std::to_string(bytes >> 20) + " MiB, the device has " + std::to_string(mem_total >> 20);
            return false;
        }
        wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
        if (wbuf == nullptr) {
            err = "no iGPU memory for " + std::to_string(bytes >> 20) + " MiB of experts";
            return false;
        }
        ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::vector<uint8_t> stage;
        for (size_t li = 0; li < src.size(); ++li) {
            const ExpertSource& s = src[li];
            if (s.gate == nullptr) continue;
            LayerWeights& w = layers[li];
            // the experts right after the VRAM-resident block: the first decode steps already send misses here
            w.compact.assign((size_t) ne, -1);
            std::vector<int> chosen((size_t) per_layer);
            for (int i = 0; i < per_layer; ++i) {
                chosen[(size_t) i] = (int) ((s.resident + i) % ne);
                w.compact[(size_t) chosen[(size_t) i]] = i;
            }
            const struct {
                ggml_tensor* t;
                const uint8_t* base;
                size_t bytes;
            } parts[3] = {{w.gate, s.gate, s.gu_bytes}, {w.up, s.up, s.gu_bytes}, {w.down, s.down, s.d_bytes}};
            for (const auto& part : parts) {
                stage.resize((size_t) per_layer * part.bytes);
                for (int i = 0; i < per_layer; ++i)
                    std::memcpy(stage.data() + (size_t) i * part.bytes, part.base + (size_t) chosen[(size_t) i] * part.bytes, part.bytes);
                ggml_backend_tensor_set(part.t, stage.data(), 0, stage.size());
            }
        }
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        for (const LayerWeights& w : layers) {
            if (w.gate == nullptr) continue;
            Graph g = build_graph(w, opt.n_embd, opt.max_entries);
            const bool ok = ggml_gallocr_reserve(galloc, g.gf);
            ggml_free(g.ctx);
            if (!ok) {
                err = "cannot reserve the iGPU compute buffers";
                return false;
            }
            break;
        }
        worker = std::thread([this] { loop(); });
        return true;
    }

    void loop() {
        std::unique_lock<std::mutex> lock(mu);
        for (;;) {
            cv_job.wait(lock, [this] { return stop || has_job; });
            if (stop) return;
            lock.unlock();
            const auto t0 = Clock::now();
            Stats part;
            run(job_layer, job, part);
            const double ms = ms_since(t0);
            lock.lock();
            st.compute_ms += ms;
            st.graph_ms += part.graph_ms;
            st.io_ms += part.io_ms;
            st.run_ms += part.run_ms;
            st.entries += (long long) job.size();
            ++st.calls;
            has_job = false;
            cv_done.notify_all();
        }
    }

    [[noreturn]] static void fail(const char* what) {
        std::fprintf(stderr, "strata-q36: iGPU experts: %s; the result would be wrong, stopping\n", what);
        std::abort();
    }

    void run(size_t layer, const std::vector<Entry>& entries, Stats& part) {
        const LayerWeights& w = layers[layer];
        const size_t N = (size_t) opt.n_embd;
        for (size_t first = 0; first < entries.size(); first += (size_t) opt.max_entries) {
            const int m = (int) std::min<size_t>((size_t) opt.max_entries, entries.size() - first);
            auto t = Clock::now();
            Graph g = build_graph(w, opt.n_embd, m);
            if (!ggml_gallocr_alloc_graph(galloc, g.gf)) fail("cannot allocate the compute buffers");
            part.graph_ms += ms_since(t);
            t = Clock::now();
            xbuf.resize((size_t) m * N);
            obuf.resize((size_t) m * N);
            idbuf.resize((size_t) m);
            for (int i = 0; i < m; ++i) {
                const Entry& e = entries[first + (size_t) i];
                std::memcpy(xbuf.data() + (size_t) i * N, e.x, N * sizeof(float));
                idbuf[(size_t) i] = w.compact[(size_t) e.expert];
            }
            ggml_backend_tensor_set(g.x, xbuf.data(), 0, xbuf.size() * sizeof(float));
            ggml_backend_tensor_set(g.ids, idbuf.data(), 0, idbuf.size() * sizeof(int32_t));
            part.io_ms += ms_since(t);
            t = Clock::now();
            if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS) fail("the graph did not run");
            ggml_backend_synchronize(backend);
            part.run_ms += ms_since(t);
            t = Clock::now();
            ggml_backend_tensor_get(g.out, obuf.data(), 0, obuf.size() * sizeof(float));
            for (int i = 0; i < m; ++i) std::memcpy(entries[first + (size_t) i].out, obuf.data() + (size_t) i * N, N * sizeof(float));
            part.io_ms += ms_since(t);
            ggml_free(g.ctx);
        }
    }
};

ExpertShare::ExpertShare(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
ExpertShare::~ExpertShare() = default;

std::unique_ptr<ExpertShare> ExpertShare::create(const Options& opt, const std::vector<ExpertSource>& layers, std::string& err) {
    if (!(opt.share > 0.0 && opt.share < 1.0) || opt.n_embd <= 0 || opt.n_ff <= 0 || opt.n_expert < 2 || opt.max_entries < 1) {
        err = "invalid iGPU share options";
        return nullptr;
    }
    auto impl = std::make_unique<Impl>();
    if (!impl->init(opt, layers, err)) return nullptr;
    return std::unique_ptr<ExpertShare>(new ExpertShare(std::move(impl)));
}

bool ExpertShare::owns(size_t layer, int expert) const {
    if (layer >= impl_->layers.size()) return false;
    const std::vector<int32_t>& c = impl_->layers[layer].compact;
    return !c.empty() && expert >= 0 && (size_t) expert < c.size() && c[(size_t) expert] >= 0;
}

void ExpertShare::submit(size_t layer, const Entry* entries, int n) {
    if (n <= 0) return;
    {
        std::lock_guard<std::mutex> lock(impl_->mu);
        impl_->job.assign(entries, entries + n);
        impl_->job_layer = layer;
        impl_->has_job = true;
    }
    impl_->cv_job.notify_one();
}

void ExpertShare::wait() {
    const auto t0 = Clock::now();
    std::unique_lock<std::mutex> lock(impl_->mu);
    if (!impl_->has_job) return;
    impl_->cv_done.wait(lock, [this] { return !impl_->has_job; });
    impl_->st.wait_ms += ms_since(t0);
}

Stats ExpertShare::stats() const {
    std::lock_guard<std::mutex> lock(impl_->mu);
    return impl_->st;
}

void ExpertShare::reset_stats() {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->st = {};
}

int ExpertShare::experts_per_layer() const { return impl_->per_layer; }
size_t ExpertShare::resident_bytes() const { return impl_->bytes; }
std::string ExpertShare::device_name() const { return impl_->device_name; }

}  // namespace strata::kernels::igpu
