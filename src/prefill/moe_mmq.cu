// src/prefill/moe_mmq.cu - see include/strata/prefill/moe_mmq.hpp.  llama.cpp's MMQ (ggml-cuda, MIT) is compiled from
// the llama.cpp tree the build takes ggml from; src/prefill/ggml_cuda_host.cu supplies the few host symbols of
// ggml-cuda.cu it references.
#include "strata/prefill/moe_mmq.hpp"

#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"

#include "strata/prefill/mmq_kq_abi.hpp"

#include <climits>
#include <cstdio>
#include <cstdlib>

namespace strata::prefill::mmq {
namespace kq {
int abi_args_size() { return (int) sizeof(mmq_args); }
}  // namespace kq

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill mmq: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int64_t pad512(int64_t n) { return (n + 511) / 512 * 512; }

unsigned blocks(int64_t n) { return (unsigned) ((n + 255) / 256); }

__global__ void swiglu_kernel(const float* __restrict__ gu, float* __restrict__ h, int64_t rows, int64_t n_ff,
                              bool interleaved) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n_ff) return;
    const int64_t r = i / n_ff, k = i % n_ff;
    const float* row = gu + r * 2 * n_ff;
    const float g = interleaved ? row[2 * k] : row[k], u = interleaved ? row[2 * k + 1] : row[n_ff + k];
    h[i] = g / (1.0f + __expf(-g)) * u;
}

__global__ void iota_kernel(int32_t* dst, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (int32_t) i;
}

__global__ void f16_to_f32_kernel(const uint16_t* __restrict__ x, float* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __half2float(__ushort_as_half(x[i]));
}

__global__ void set_bounds_kernel(int32_t* d, int32_t rows) {
    d[0] = 0;
    d[1] = rows;
}

bool is_k_quant(int t) {
    return t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K || t == GGML_TYPE_Q6_K || t == GGML_TYPE_Q5_1;
}

// ggml_cuda_mul_mat_q's tile choice: the width J (8..128) that needs the fewest tiles for ncols_opt columns, among the
// tiles that exist for this card and fit its shared memory.  mmq_args carries the result as J_best.
int pick_J_best(ggml_type type, int64_t rows_x, int64_t ncols_opt) {
    int id = 0;
    CUDA_CHECK(cudaGetDevice(&id));
    const ggml_cuda_device_info& info = ggml_cuda_info();
    const int cc = info.devices[id].cc;
    const size_t smpbo = info.devices[id].smpbo;
    const bool fallback = ggml_cuda_mmq_needs_fallback(rows_x);
    int J_best = 0;
    int ntiles_J_best = INT_MAX;
    for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8) {
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, cc, GGML_PREC_Q8);
        if (config.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(config, cc) > smpbo) continue;
        const int ntiles_x = (int) ((ncols_opt + config.J - 1) / config.J);
        if (ntiles_x < ntiles_J_best) {
            J_best = J;
            ntiles_J_best = ntiles_x;
        }
    }
    return J_best;
}

}  // namespace

bool built() { return true; }

bool supported(int t) {
    if (t == GGML_TYPE_Q8_0) return true;   // the draft layer's dense matrices
    return is_k_quant(t) && kq::available();
}

bool fits(int t, int64_t w_rows) {
    if (!supported(t)) return false;
    const bool fallback = ggml_cuda_mmq_needs_fallback(w_rows);
    const ggml_cuda_device_info& info = ggml_cuda_info();
    for (int id = 0; id < info.device_count; ++id) {
        const int cc = info.devices[id].cc;
        const size_t smpbo = info.devices[id].smpbo;
        bool any = false;
        for (int J = 8; J <= 128 && !any; J += 8) {
            const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config((ggml_type) t, J, fallback, cc, GGML_PREC_Q8);
            any = c.type != GGML_TYPE_COUNT && mmq_get_nbytes_shared(c, cc) <= smpbo;
        }
        if (!any) {
            static bool said[GGML_TYPE_COUNT] = {};
            if (t >= 0 && t < GGML_TYPE_COUNT && !said[t]) {
                said[t] = true;
                std::fprintf(stderr, "strata: prompt kernels: llama.cpp's MMQ has no tile for %s (%lld rows) on GPU %d "
                                     "(cc %d, %zu bytes of shared memory per block): that product takes the non-MMQ path\n",
                             ggml_type_name((ggml_type) t), (long long) w_rows, id, cc, smpbo);
            }
            return false;
        }
    }
    return true;
}

size_t matrix_bytes(int t, int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) (cols / ggml_blck_size((ggml_type) t)) * ggml_type_size((ggml_type) t);
}

size_t q8_bytes(int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) pad512(cols) * sizeof(block_q8_1_mmq) / (4 * QK8_1) + 128 * sizeof(block_q8_1_mmq);
}

void quantize(const float* x, const int32_t* ids, void* xq, int t, int64_t cols, int64_t ld, int64_t rows, void* stream) {
    if (rows <= 0) return;
    quantize_mmq_q8_1_cuda(x, ids, xq, (ggml_type) t, cols, ld, rows * ld, rows * ld, pad512(cols), rows, 1, 1,
                           (cudaStream_t) stream);
    ck(cudaGetLastError(), "quantize");
}

Context::Context() {
    int dev = 0;
    cudaGetDevice(&dev);
    ctx_ = new ggml_backend_cuda_context(dev);
}
Context::~Context() { delete (ggml_backend_cuda_context*) ctx_; }

void Context::run(const Product& p, void* stream) {
    if (p.n <= 0 || p.max_rows <= 0) return;
    const ggml_type t = (ggml_type) p.type;
    const int64_t bpr = p.w_cols / ggml_blck_size(t);
    const mmq_args a = {(const char*) p.w, t, (const int*) p.xq, p.ids, p.bounds, p.dst, nullptr,
                        p.w_cols, p.w_rows, p.total_rows, bpr, p.total_rows, p.ld_dst,
                        p.n, p.n, (int64_t) (p.expert_bytes / ggml_type_size(t)), 0, 0,
                        1, 1, 0, 0, 0,
                        p.max_rows, pick_J_best(t, p.w_rows, p.max_rows)};
    auto& ctx = *(ggml_backend_cuda_context*) ctx_;
    const cudaStream_t s = (cudaStream_t) stream;
    if (t == GGML_TYPE_Q8_0) {
        mul_mat_q_case<GGML_TYPE_Q8_0, GGML_PREC_Q8>(ctx, a, s);
    } else if (is_k_quant(t)) {
        const int e = kq::run((int) t, &a, s);
        if (e != 0) {
            std::fprintf(stderr, "prefill mmq: the K-quant library failed on type %d (error %d)\n", (int) t, e);
            std::exit(1);
        }
    } else {
        std::fprintf(stderr, "prefill mmq: type %d is not covered\n", (int) t);
        std::exit(1);
    }
    ck(cudaGetLastError(), "mul_mat_q");
}

void swiglu(const float* gu, float* h, int64_t rows, int64_t n_ff, bool interleaved, void* stream) {
    if (rows <= 0) return;
    swiglu_kernel<<<blocks(rows * n_ff), 256, 0, (cudaStream_t) stream>>>(gu, h, rows, n_ff, interleaved);
    ck(cudaGetLastError(), "swiglu");
}

void iota(int32_t* dst, int64_t n, void* stream) {
    if (n <= 0) return;
    iota_kernel<<<blocks(n), 256, 0, (cudaStream_t) stream>>>(dst, n);
    ck(cudaGetLastError(), "iota");
}

void f16_to_f32(const uint16_t* x, float* y, int64_t n, void* stream) {
    if (n <= 0) return;
    f16_to_f32_kernel<<<blocks(n), 256, 0, (cudaStream_t) stream>>>(x, y, n);
    ck(cudaGetLastError(), "f16_to_f32");
}

void set_bounds(int32_t* dst, int32_t rows, void* stream) {
    set_bounds_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(dst, rows);
    ck(cudaGetLastError(), "set_bounds");
}

}  // namespace strata::prefill::mmq
