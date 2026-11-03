#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace q36act {

// The activation formats the CPU expert dot products read (ggml block_q8_K and block_q8_0), written by the device with
// the arithmetic of the host quantizers (native_quant_act), so the bytes are identical.  Every product and sum that the
// host rounds in float is an explicit __fmul_rn / __fadd_rn / __fdiv_rn here: nvcc must not contract them into FMAs.
constexpr int kTypeQ8_0 = 8;    // GGML_TYPE_Q8_0
constexpr int kTypeQ8_K = 15;   // GGML_TYPE_Q8_K
constexpr int kQ8KValues = 256, kQ8KBytes = 292;   // float d; int8 qs[256]; int16 bsums[16]
constexpr int kQ8_0Values = 32, kQ8_0Bytes = 34;   // half d; int8 qs[32]
constexpr int kQ8KQsOffset = 4, kQ8KBsumsOffset = 260, kQ8KBsums = 16;
constexpr float kRoundMagic = 12582912.f;          // 1.5 * 2^23: ggml's nearest_int
constexpr int kMagicMantissa = 0x007fffff, kMagicOffset = 0x00400000;

inline bool supported(int act_type, int64_t n) {
    if (act_type == kTypeQ8_K) return n > 0 && n % kQ8KValues == 0;
    if (act_type == kTypeQ8_0) return n > 0 && n % kQ8_0Values == 0;
    return false;
}

inline size_t row_bytes(int act_type, int64_t n) {
    return act_type == kTypeQ8_K ? (size_t) (n / kQ8KValues) * kQ8KBytes : (size_t) (n / kQ8_0Values) * kQ8_0Bytes;
}

// One 256-thread block per 256 values: block (sb, row), thread j = value j of the super-block.
__global__ void k_quant_q8k(const float* x, uint8_t* dst, int64_t n, size_t dst_stride) {
    __shared__ float s_amax[8];
    __shared__ int s_first[8];
    __shared__ float s_pick;
    const int j = threadIdx.x, lane = j & 31, warp = j >> 5;
    const int64_t sb = blockIdx.x, row = blockIdx.y;
    const float v = x[(size_t) row * (size_t) n + (size_t) sb * kQ8KValues + j];
    uint8_t* blk = dst + (size_t) row * dst_stride + (size_t) sb * kQ8KBytes;

    float ax = fabsf(v);
    if (!(ax >= 0.f)) ax = 0.f;   // a NaN never wins the host's `ax > amax`
    float amax = ax;
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if (lane == 0) s_amax[warp] = amax;
    __syncthreads();
    amax = s_amax[0];
    for (int w = 1; w < 8; ++w) amax = fmaxf(amax, s_amax[w]);
    if (amax == 0.f) {   // the host leaves bsums untouched here; zeros are what the dot products need
        blk[kQ8KQsOffset + j] = 0;
        if (j < kQ8KBsums) reinterpret_cast<int16_t*>(blk + kQ8KBsumsOffset)[j] = 0;
        if (j == 0) *reinterpret_cast<float*>(blk) = 0.f;
        return;
    }
    // the sign-carrying value is the FIRST one of the largest magnitude
    int first = (ax == amax) ? j : kQ8KValues;
    for (int o = 16; o > 0; o >>= 1) first = min(first, __shfl_xor_sync(0xffffffffu, first, o));
    if (lane == 0) s_first[warp] = first;
    __syncthreads();
    first = s_first[0];
    for (int w = 1; w < 8; ++w) first = min(first, s_first[w]);
    if (j == first) s_pick = v;
    __syncthreads();
    const float iscale = __fdiv_rn(-127.f, s_pick);
    const float t = __fadd_rn(__fmul_rn(iscale, v), kRoundMagic);
    const int q = min((__float_as_int(t) & kMagicMantissa) - kMagicOffset, 127);
    const int8_t q8 = (int8_t) q;   // the host stores MIN(127, v) into an int8 and sums those bytes
    blk[kQ8KQsOffset + j] = (uint8_t) q8;
    int s = q8;
    for (int o = 8; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((j & 15) == 0) reinterpret_cast<int16_t*>(blk + kQ8KBsumsOffset)[j >> 4] = (int16_t) s;
    if (j == 0) *reinterpret_cast<float*>(blk) = __fdiv_rn(1.f, iscale);
}

// One warp per 32 values (ggml's AVX2 quantize_row_q8_0): thread block of 128 = four q8_0 blocks, block (g, row).
__global__ void k_quant_q8_0(const float* x, uint8_t* dst, int64_t n, size_t dst_stride) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int64_t nb = n / kQ8_0Values;
    const int64_t b = (int64_t) blockIdx.x * 4 + warp, row = blockIdx.y;
    if (b >= nb) return;
    const float v = x[(size_t) row * (size_t) n + (size_t) b * kQ8_0Values + lane];
    float ax = fabsf(v);
    if (!(ax >= 0.f)) ax = 0.f;
    float amax = ax;
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    uint8_t* blk = dst + (size_t) row * dst_stride + (size_t) b * kQ8_0Bytes;
    const float id = amax != 0.f ? __fdiv_rn(127.f, amax) : 0.f;
    // cvtps_epi32 turns an out-of-range or NaN product (a block of denormals: 127 / amax overflows) into INT_MIN
    const float scaled = __fmul_rn(v, id);
    const int rounded = fabsf(scaled) < 2147483648.f ? __float2int_rn(scaled) : INT_MIN;
    const int q = min(max(rounded, -128), 127);
    blk[2 + lane] = (uint8_t) (int8_t) q;
    if (lane == 0) *reinterpret_cast<__half*>(blk) = __float2half_rn(__fdiv_rn(amax, 127.f));
}

/// x [rows][n] floats -> rows of `row_bytes(act_type, n)` bytes, `dst_stride` apart.  supported(act_type, n) must hold.
inline void launch(int act_type, const float* x, uint8_t* dst, int64_t n, int rows, size_t dst_stride, cudaStream_t st) {
    if (act_type == kTypeQ8_K) {
        k_quant_q8k<<<dim3((unsigned) (n / kQ8KValues), (unsigned) rows), kQ8KValues, 0, st>>>(x, dst, n, dst_stride);
    } else {
        const unsigned groups = (unsigned) ((n / kQ8_0Values + 3) / 4);
        k_quant_q8_0<<<dim3(groups, (unsigned) rows), 128, 0, st>>>(x, dst, n, dst_stride);
    }
}

/// Whether two quantized rows are the same bytes.  A Q8_K block whose values are all zero leaves its bsums unwritten on
/// the host: those bytes are not compared (d and qs are, and with d = 0 the dot products never read them).
inline bool same_row(int act_type, int64_t n, const uint8_t* a, const uint8_t* b) {
    if (act_type != kTypeQ8_K) return std::memcmp(a, b, row_bytes(act_type, n)) == 0;
    for (int64_t sb = 0; sb < n / kQ8KValues; ++sb) {
        const uint8_t* pa = a + (size_t) sb * kQ8KBytes;
        const uint8_t* pb = b + (size_t) sb * kQ8KBytes;
        float da;
        std::memcpy(&da, pa, 4);
        const size_t cmp = da == 0.f ? (size_t) kQ8KBsumsOffset : (size_t) kQ8KBytes;
        if (std::memcmp(pa, pb, cmp) != 0) return false;
    }
    return true;
}

}  // namespace q36act
