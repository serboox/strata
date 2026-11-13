#pragma once

#include <cstdint>
#include <cstring>

// SwiGLU of the strict mode: one sequence of IEEE operations for the CPU and for the GPU, so both give the same bits.
// Include this file only from a translation unit that cannot contract a product and a sum into an FMA by itself:
// strict_math.cpp is built with -ffp-contract=off, and device code uses the _rn intrinsics, which nvcc never merges.

#if defined(__CUDACC__)
#define STRATA_STRICT_FN __device__ __forceinline__
#define STRATA_SM_DMUL(a, b) __dmul_rn((a), (b))
#define STRATA_SM_DADD(a, b) __dadd_rn((a), (b))
#define STRATA_SM_DFMA(a, b, c) __fma_rn((a), (b), (c))
#define STRATA_SM_FMUL(a, b) __fmul_rn((a), (b))
#define STRATA_SM_FADD(a, b) __fadd_rn((a), (b))
#define STRATA_SM_FDIV(a, b) __fdiv_rn((a), (b))
#define STRATA_SM_BITS_TO_DOUBLE(u) __longlong_as_double((long long) (u))
#else
#include <cmath>
#define STRATA_STRICT_FN inline
#define STRATA_SM_DMUL(a, b) ((a) * (b))
#define STRATA_SM_DADD(a, b) ((a) + (b))
#define STRATA_SM_DFMA(a, b, c) std::fma((a), (b), (c))
#define STRATA_SM_FMUL(a, b) ((a) * (b))
#define STRATA_SM_FADD(a, b) ((a) + (b))
#define STRATA_SM_FDIV(a, b) ((a) / (b))
namespace strata::kernels::strict_detail {
inline double bits_to_double(uint64_t u) {
    double d;
    std::memcpy(&d, &u, sizeof d);
    return d;
}
}  // namespace strata::kernels::strict_detail
#define STRATA_SM_BITS_TO_DOUBLE(u) ::strata::kernels::strict_detail::bits_to_double((uint64_t) (u))
#endif

namespace strata::kernels {

/// exp(x) rounded to float: k = round(x / ln 2), r = x - k ln 2, a degree-9 Taylor sum of exp(r) in double, times 2^k.
/// The sum is within 1e-11 of exp(r), far below half a float ulp; it is not glibc's expf bit for bit.
STRATA_STRICT_FN float strict_exp(float x) {
    constexpr double kInvLn2 = 1.4426950408889634074, kLn2 = 0.69314718055994530942;
    constexpr double kRound = 6755399441055744.0;   // 1.5 * 2^52: adding and subtracting it rounds to the nearest integer
    constexpr double kHigh = 89.0, kLow = -104.0;   // exp(89) is above FLT_MAX, exp(-104) is below half the smallest float
    constexpr double kC2 = 1.0 / 2.0, kC3 = 1.0 / 6.0, kC4 = 1.0 / 24.0, kC5 = 1.0 / 120.0, kC6 = 1.0 / 720.0;
    constexpr double kC7 = 1.0 / 5040.0, kC8 = 1.0 / 40320.0, kC9 = 1.0 / 362880.0;
    double xd = (double) x;
    if (!(xd == xd)) return x;
    if (xd > kHigh) xd = kHigh;
    if (xd < kLow) xd = kLow;
    const double z = STRATA_SM_DMUL(xd, kInvLn2);
    const double kd = STRATA_SM_DADD(STRATA_SM_DADD(z, kRound), -kRound);
    const double r = STRATA_SM_DFMA(-kd, kLn2, xd);
    double p = kC9;
    p = STRATA_SM_DFMA(p, r, kC8);
    p = STRATA_SM_DFMA(p, r, kC7);
    p = STRATA_SM_DFMA(p, r, kC6);
    p = STRATA_SM_DFMA(p, r, kC5);
    p = STRATA_SM_DFMA(p, r, kC4);
    p = STRATA_SM_DFMA(p, r, kC3);
    p = STRATA_SM_DFMA(p, r, kC2);
    p = STRATA_SM_DFMA(p, r, 1.0);
    p = STRATA_SM_DFMA(p, r, 1.0);
    const long long k = (long long) kd;
    const double scale = STRATA_SM_BITS_TO_DOUBLE((uint64_t) (k + 1023) << 52);
    return (float) STRATA_SM_DMUL(p, scale);
}

/// silu(gate) * up in the order of the CPU expert kernels: (gate / (1 + exp(-gate))) * up.
STRATA_STRICT_FN float strict_swiglu(float gate, float up) {
    const float denominator = STRATA_SM_FADD(1.f, strict_exp(-gate));
    return STRATA_SM_FMUL(STRATA_SM_FDIV(gate, denominator), up);
}

}  // namespace strata::kernels
