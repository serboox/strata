#include "strata/kernels/cpu/kq_avx512.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <cmath>
#include <cstring>

// Q4_K x Q8_K and Q5_K x Q8_K dot products for up to kMaxTokens activations per weight row.  The integer part of ggml's
// AVX2 kernel is exact, so only its sum over 8 int32 lanes has to survive: here 64 weights meet 64 activation bytes in
// one 512-bit multiply, and the two 256-bit halves are added afterwards.  Everything in floating point (the 8-lane
// accumulator, the minimum term, the final reduction) repeats ggml's operations in ggml's order.  The weights of a
// block are unpacked once for all the tokens.
namespace strata::kernels::cpu {
namespace {

constexpr int kMaxTokens = 8;
constexpr int kMaxBlocks = 16;   // rows of up to 4096 values
constexpr int kBlock = 256;

constexpr uint32_t kmask1 = 0x3f3f3f3f;
constexpr uint32_t kmask2 = 0x0f0f0f0f;
constexpr uint32_t kmask3 = 0x03030303;

inline float half_to_float(const void* block, int index) {
    uint16_t u;
    std::memcpy(&u, (const uint8_t*) block + 2 * index, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) u)));
}

inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

// the activation's 32-value groups in the order 0 2 1 3 4 6 5 7, so that 64 bytes of it meet the 64 weight bytes of
// two sub-blocks (one of the low and one of the high nibbles) in a single load
constexpr int kGroupOrder[8] = {0, 2, 1, 3, 4, 6, 5, 7};

struct Activations {
    alignas(64) int8_t qs[kMaxTokens][kMaxBlocks][kBlock];
    const block_q8_K* blk[kMaxTokens];
};

void reorder(Activations& a, const void* const* act, int nt, int nb) {
    for (int t = 0; t < nt; ++t) {
        a.blk[t] = (const block_q8_K*) act[t];
        for (int i = 0; i < nb; ++i)
            for (int g = 0; g < 8; ++g)
                std::memcpy(&a.qs[t][i][g * 32], a.blk[t][i].qs + kGroupOrder[g] * 32, 32);
    }
}

// Per 128-bit lane, the 16-bit scale `a` for the first two lanes and `b` for the last two
inline __m512i scale_mask(int a, int b) {
    const __m128i lo = _mm_set1_epi16((short) (2 * a | ((2 * a + 1) << 8)));
    const __m128i hi = _mm_set1_epi16((short) (2 * b | ((2 * b + 1) << 8)));
    return _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set_m128i(lo, lo)), _mm256_set_m128i(hi, hi), 1);
}

inline __m512i bit_pair_mask(int low_half, int high_half) {
    return _mm512_inserti64x4(_mm512_set1_epi8((char) (1 << low_half)), _mm256_set1_epi8((char) (1 << high_half)), 1);
}

struct Masks {
    __m512i m4 = _mm512_set1_epi8(0xF), sixteen = _mm512_set1_epi8(16);
    __m512i scale_lo0 = scale_mask(0, 2), scale_hi0 = scale_mask(1, 3), scale_lo1 = scale_mask(4, 6), scale_hi1 = scale_mask(5, 7);
    __m512i bit_lo0 = bit_pair_mask(0, 2), bit_hi0 = bit_pair_mask(1, 3), bit_lo1 = bit_pair_mask(4, 6), bit_hi1 = bit_pair_mask(5, 7);
};

template <bool Q5>
inline void row_dot(const void* row, int nb, int nt, const Masks& mk, const Activations& a, float* out) {
    __m256 acc[kMaxTokens];
    __m128 acc_m[kMaxTokens];
    float summs[kMaxTokens];
    for (int t = 0; t < nt; ++t) {
        acc[t] = _mm256_setzero_ps();
        acc_m[t] = _mm_setzero_ps();
        summs[t] = 0.f;
    }
    const __m128i mzero = _mm_setzero_si128();
    for (int i = 0; i < nb; ++i) {
        const uint8_t* block = (const uint8_t*) row + (size_t) i * (Q5 ? sizeof(block_q5_K) : sizeof(block_q4_K));
        const float xd = half_to_float(block, 0), xdmin = half_to_float(block, 1);
        uint32_t utmp[4];
        std::memcpy(utmp, block + 4, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32((int) utmp[3], (int) utmp[2], (int) utmp[1], (int) utmp[0]));
        const __m128i mins = _mm256_extracti128_si256(mins_and_scales, 1);
        const __m512i sc = _mm512_broadcast_i32x4(_mm256_extracti128_si256(mins_and_scales, 0));
        const __m512i scale_lo0 = _mm512_shuffle_epi8(sc, mk.scale_lo0), scale_hi0 = _mm512_shuffle_epi8(sc, mk.scale_hi0);
        const __m512i scale_lo1 = _mm512_shuffle_epi8(sc, mk.scale_lo1), scale_hi1 = _mm512_shuffle_epi8(sc, mk.scale_hi1);

        const uint8_t* qs = block + (Q5 ? 48 : 16);
        __m512i w_lo0, w_hi0, w_lo1, w_hi1;
        {
            const __m512i bits0 = _mm512_loadu_si512(qs), bits1 = _mm512_loadu_si512(qs + 64);
            w_lo0 = _mm512_and_si512(bits0, mk.m4);
            w_hi0 = _mm512_and_si512(_mm512_srli_epi16(bits0, 4), mk.m4);
            w_lo1 = _mm512_and_si512(bits1, mk.m4);
            w_hi1 = _mm512_and_si512(_mm512_srli_epi16(bits1, 4), mk.m4);
            if constexpr (Q5) {
                const __m512i high = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*) (block + 16)));
                w_lo0 = _mm512_mask_add_epi8(w_lo0, _mm512_test_epi8_mask(high, mk.bit_lo0), w_lo0, mk.sixteen);
                w_hi0 = _mm512_mask_add_epi8(w_hi0, _mm512_test_epi8_mask(high, mk.bit_hi0), w_hi0, mk.sixteen);
                w_lo1 = _mm512_mask_add_epi8(w_lo1, _mm512_test_epi8_mask(high, mk.bit_lo1), w_lo1, mk.sixteen);
                w_hi1 = _mm512_mask_add_epi8(w_hi1, _mm512_test_epi8_mask(high, mk.bit_hi1), w_hi1, mk.sixteen);
            }
        }
        for (int t = 0; t < nt; ++t) {
            const block_q8_K& y = a.blk[t][i];
            const float d = y.d * xd;
            const float dmin = -y.d * xdmin;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y.bsums);
            const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
            const __m128i prod = _mm_madd_epi16(mins, q8s);
            if constexpr (Q5) {
                const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
                summs[t] += dmin * _mm_extract_epi32(hsum, 0);
            } else {
                acc_m[t] = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m[t]);
            }
            const int8_t* q8 = a.qs[t][i];
            __m512i sumi = _mm512_madd_epi16(scale_lo0, _mm512_maddubs_epi16(w_lo0, _mm512_loadu_si512(q8)));
            sumi = _mm512_add_epi32(sumi, _mm512_madd_epi16(scale_hi0, _mm512_maddubs_epi16(w_hi0, _mm512_loadu_si512(q8 + 64))));
            sumi = _mm512_add_epi32(sumi, _mm512_madd_epi16(scale_lo1, _mm512_maddubs_epi16(w_lo1, _mm512_loadu_si512(q8 + 128))));
            sumi = _mm512_add_epi32(sumi, _mm512_madd_epi16(scale_hi1, _mm512_maddubs_epi16(w_hi1, _mm512_loadu_si512(q8 + 192))));
            const __m256i sum8 = _mm256_add_epi32(_mm512_castsi512_si256(sumi), _mm512_extracti64x4_epi64(sumi, 1));
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sum8), acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) {
        if constexpr (Q5) {
            out[t] = hsum_float_8(acc[t]) + summs[t];
        } else {
            __m128 m = _mm_add_ps(acc_m[t], _mm_movehl_ps(acc_m[t], acc_m[t]));
            m = _mm_add_ss(m, _mm_movehdup_ps(m));
            out[t] = hsum_float_8(acc[t]) + _mm_cvtss_f32(m);
        }
    }
}

constexpr int kPrefetchRows = 3;
inline void prefetch_row(const uint8_t* row, size_t bytes) {
    for (size_t o = 0; o < bytes; o += 64) __builtin_prefetch(row + o, 0, 3);
}

}  // namespace

// ggml type ids: Q4_K is 12, Q5_K is 13
bool kq512_gu_supported(int ggml_type, int n) noexcept { return ggml_type == 12 && n % kBlock == 0 && n / kBlock <= kMaxBlocks; }
bool kq512_down_supported(int ggml_type, int n) noexcept { return ggml_type == 13 && n % kBlock == 0 && n / kBlock <= kMaxBlocks; }

void kq512_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                   float* const* ff, int r0, int r1) {
    thread_local Activations a;
    const int nb = n / kBlock;
    reorder(a, act, nt, nb);
    const Masks mk;
    for (int r = r0; r < (r1 < r0 + kPrefetchRows ? r1 : r0 + kPrefetchRows); ++r) {
        prefetch_row(blob + (size_t) r * gu_row, gu_row);
        prefetch_row(blob + up_off + (size_t) r * gu_row, gu_row);
    }
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gate = blob + (size_t) r * gu_row;
        const uint8_t* up = blob + up_off + (size_t) r * gu_row;
        if (r + kPrefetchRows < r1) {
            prefetch_row(gate + (size_t) kPrefetchRows * gu_row, gu_row);
            prefetch_row(up + (size_t) kPrefetchRows * gu_row, gu_row);
        }
        float g[kMaxTokens], u[kMaxTokens];
        row_dot<false>(gate, nb, nt, mk, a, g);
        row_dot<false>(up, nb, nt, mk, a, u);
        for (int t = 0; t < nt; ++t) ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
    }
}

void kq512_down_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                     int r0, int r1) {
    thread_local Activations a;
    const int nb = n / kBlock;
    reorder(a, act, nt, nb);
    const Masks mk;
    for (int r = r0; r < (r1 < r0 + kPrefetchRows ? r1 : r0 + kPrefetchRows); ++r) prefetch_row(w + (size_t) r * row_bytes, row_bytes);
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t) r * row_bytes;
        if (r + kPrefetchRows < r1) prefetch_row(row + (size_t) kPrefetchRows * row_bytes, row_bytes);
        float s[kMaxTokens];
        row_dot<true>(row, nb, nt, mk, a, s);
        for (int t = 0; t < nt; ++t) out[t][r] = s[t];
    }
}

}  // namespace strata::kernels::cpu
