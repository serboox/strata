// include/strata/prefill/moe_mmq.hpp - the prompt path's matrix products through llama.cpp's MMQ kernels (ggml-cuda
// mmq.cuh, MIT): the weights stay quantized, the activations are rounded to q8_1, the products run on int8 tensor
// cores.  Q8_0 is compiled into the engine; the K-quants (Q4_K, Q5_K, Q6_K, Q5_1) live in libstrata_kq, opened on first use.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::prefill::mmq {

/// This build has the MMQ path.
bool built();
/// MMQ covers this ggml type: Q8_0, and the K-quants when libstrata_kq can be opened (STRATA_MMQ_KQ=1).
bool supported(int ggml_type);
/// `supported`, and on every visible GPU llama.cpp's MMQ has a tile for this type and a weight matrix of `w_rows` rows
/// that fits the card's shared memory - the same test its tile choice makes, which aborts the process ("J_best=0")
/// when nothing fits.  false (said once per type) keeps that product on the non-MMQ path.
bool fits(int ggml_type, int64_t w_rows);
/// Bytes of a [rows, cols] weight matrix in `ggml_type`.
size_t matrix_bytes(int ggml_type, int64_t rows, int64_t cols);
/// Bytes of `rows` activation rows of `cols` values quantized for MMQ (the row padded to 512 values).
size_t q8_bytes(int64_t rows, int64_t cols);

/// q8_1 activations for MMQ against weights of `ggml_type`: row i of the output is row ids[i] of x (or row i when
/// ids is null); `x` has `ld` floats per row.
void quantize(const float* x, const int32_t* ids, void* xq, int ggml_type, int64_t cols, int64_t ld, int64_t rows,
              void* stream);

/// One launch over n experts whose weights lie `expert_bytes` apart from `w`: for expert e, the activation rows
/// [bounds[e], bounds[e+1]) of `xq` (bounds on the device, n+1 entries) times its [w_rows, w_cols] matrix into
/// dst rows of the same indices (`ld_dst` floats apart, via `ids`: dst row = ids[row], an identity table works).
/// `total_rows`: the rows of xq; `max_rows`: the most rows one expert has (the launch grid).
struct Product {
    const void* w = nullptr;
    int type = -1;
    int64_t w_rows = 0, w_cols = 0;
    size_t expert_bytes = 0;
    int n = 0;
    const void* xq = nullptr;
    const int32_t* bounds = nullptr;
    const int32_t* ids = nullptr;
    int64_t total_rows = 0, max_rows = 0;
    float* dst = nullptr;
    int64_t ld_dst = 0;
};

/// The launch context (llama.cpp's MMQ keeps a small scratch pool for its stream-k fixup).  One per prompt path.
class Context {
public:
    Context();
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    void run(const Product& p, void* stream);

private:
    void* ctx_ = nullptr;
};

/// h[r, k] = silu(gate) * up of GU rows [2 n_ff wide]: interleaved (gate 2k, up 2k+1) or split (gate k, up n_ff + k:
/// GGUF).  FP32 out (the down product's quantizer reads floats).
void swiglu(const float* gu, float* h, int64_t rows, int64_t n_ff, bool interleaved, void* stream);

/// dst[i] = i for i < n (the identity row map MMQ's MoE mode writes through).
void iota(int32_t* dst, int64_t n, void* stream);

/// y[i] = float(x[i]) for FP16 bits x.
void f16_to_f32(const uint16_t* x, float* y, int64_t n, void* stream);
/// dst = {0, rows} (the bounds of one matrix; dst on the device).
void set_bounds(int32_t* dst, int32_t rows, void* stream);

}  // namespace strata::prefill::mmq
