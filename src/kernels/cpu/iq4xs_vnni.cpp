#include "strata/kernels/cpu/iq4xs_vnni.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <cstring>
#include <vector>

namespace strata::kernels::cpu {
namespace {

constexpr int kBlockBytes = 136;
constexpr int kMaxTokens = 8;
constexpr int kPrefetchBytes = 2048;

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }
inline uint16_t u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline uint32_t u32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }

// One Q8_K block regrouped for the weight layout of a 128-weight half: the 16-byte groups at offsets 0 and 16 of each 32-byte
// sub-block (the low and the high nibbles of its 16 weight bytes), and minus 127 times their sums, which cancels the +127 offset
// that keeps the lookup values unsigned.
struct alignas(64) BlockActivation {
    uint8_t low[2][64];
    uint8_t high[2][64];
    int32_t correction[2][16];
    float d;
};

struct Constants {
    __m512i lookup;
    __m512i pick_low, pick_high;
    __m512i group_of_lane[2];
};

inline Constants make_constants() {
    Constants c;
    alignas(16) uint8_t offset_values[16];
    for (int i = 0; i < 16; ++i) offset_values[i] = (uint8_t) (kvalues_iq4nl[i] + 127);
    c.lookup = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) offset_values));
    c.pick_low = _mm512_set_epi64(13, 12, 9, 8, 5, 4, 1, 0);
    c.pick_high = _mm512_set_epi64(15, 14, 11, 10, 7, 6, 3, 2);
    c.group_of_lane[0] = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
    c.group_of_lane[1] = _mm512_set_epi32(7, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4);
    return c;
}

const Constants& constants() {
    static const Constants value = make_constants();
    return value;
}

inline void prepare(const block_q8_K& y, const Constants& c, BlockActivation& out) {
    const __m512i offset = _mm512_set1_epi8(127);
    for (int h = 0; h < 2; ++h) {
        const __m512i a0 = _mm512_loadu_si512(y.qs + 128 * h), a1 = _mm512_loadu_si512(y.qs + 128 * h + 64);
        const __m512i low = _mm512_permutex2var_epi64(a0, c.pick_low, a1), high = _mm512_permutex2var_epi64(a0, c.pick_high, a1);
        const __m512i sums = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), offset, low), offset, high);
        _mm512_store_si512(out.low[h], low);
        _mm512_store_si512(out.high[h], high);
        _mm512_store_si512(out.correction[h], _mm512_sub_epi32(_mm512_setzero_si512(), sums));
    }
    out.d = y.d;
}

template <int NT>
inline void row_dot(const uint8_t* row, int blocks, const Constants& c, const BlockActivation* act, float* result) {
    __m512 acc_f[NT];
    for (int t = 0; t < NT; ++t) acc_f[t] = _mm512_setzero_ps();
    const __m512i nibble = _mm512_set1_epi8(0x0F);
    for (int i = 0; i < blocks; ++i) {
        const uint8_t* block = row + (size_t) i * kBlockBytes;
        _mm_prefetch((const char*) block + kPrefetchBytes, _MM_HINT_T0);
        _mm_prefetch((const char*) block + kPrefetchBytes + 64, _MM_HINT_T0);
        const float d = h2f(u16(block));
        uint64_t scales = _pdep_u64(u32(block + 4), 0x0F0F0F0F0F0F0F0FULL) | (_pdep_u64(u16(block + 2), 0x0303030303030303ULL) << 4);
        const __m128i scale_bytes = _mm_sub_epi8(_mm_cvtsi64_si128((long long) scales), _mm_set1_epi8(32));
        const __m512i scale_lanes = _mm512_castsi256_si512(_mm256_cvtepi8_epi32(scale_bytes));
        const __m512i scale[2] = {_mm512_permutexvar_epi32(c.group_of_lane[0], scale_lanes), _mm512_permutexvar_epi32(c.group_of_lane[1], scale_lanes)};
        __m512i weight_low[2], weight_high[2];
        for (int h = 0; h < 2; ++h) {
            const __m512i packed = _mm512_loadu_si512(block + 8 + 64 * h);
            weight_low[h] = _mm512_shuffle_epi8(c.lookup, _mm512_and_si512(packed, nibble));
            weight_high[h] = _mm512_shuffle_epi8(c.lookup, _mm512_and_si512(_mm512_srli_epi16(packed, 4), nibble));
        }
        for (int t = 0; t < NT; ++t) {
            const BlockActivation& y = act[(size_t) t * blocks + i];
            __m512i total = _mm512_setzero_si512();
            for (int h = 0; h < 2; ++h) {
                __m512i sum = _mm512_load_si512(y.correction[h]);
                sum = _mm512_dpbusd_epi32(sum, weight_low[h], _mm512_load_si512(y.low[h]));
                sum = _mm512_dpbusd_epi32(sum, weight_high[h], _mm512_load_si512(y.high[h]));
                total = _mm512_add_epi32(total, _mm512_mullo_epi32(sum, scale[h]));
            }
            acc_f[t] = _mm512_fmadd_ps(_mm512_set1_ps(d * y.d), _mm512_cvtepi32_ps(total), acc_f[t]);
        }
    }
    for (int t = 0; t < NT; ++t) result[t] = _mm512_reduce_add_ps(acc_f[t]);
}

template <int NT>
void rows(const uint8_t* base, size_t row_bytes, int blocks, const BlockActivation* act, float* const* out, int r0, int r1) {
    float result[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<NT>(base + (size_t) r * row_bytes, blocks, constants(), act, result);
        for (int t = 0; t < NT; ++t) out[t][r] = result[t];
    }
}

}  // namespace

void iq4xs_vnni_rows(const uint8_t* base, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0, int r1) {
    const int blocks = n / QK_K;
    thread_local std::vector<BlockActivation> prepared;
    for (int t0 = 0; t0 < nt; t0 += kMaxTokens) {
        const int group = nt - t0 < kMaxTokens ? nt - t0 : kMaxTokens;
        prepared.resize((size_t) group * blocks);
        for (int t = 0; t < group; ++t) {
            const block_q8_K* y = (const block_q8_K*) act[t0 + t];
            for (int i = 0; i < blocks; ++i) prepare(y[i], constants(), prepared[(size_t) t * blocks + i]);
        }
        float* const* group_out = out + t0;
        switch (group) {
            case 1: rows<1>(base, row_bytes, blocks, prepared.data(), group_out, r0, r1); break;
            case 2: rows<2>(base, row_bytes, blocks, prepared.data(), group_out, r0, r1); break;
            case 3: rows<3>(base, row_bytes, blocks, prepared.data(), group_out, r0, r1); break;
            case 4: rows<4>(base, row_bytes, blocks, prepared.data(), group_out, r0, r1); break;
            case 5: rows<5>(base, row_bytes, blocks, prepared.data(), group_out, r0, r1); break;
            case 6: rows<6>(base, row_bytes, blocks, prepared.data(), group_out, r0, r1); break;
            case 7: rows<7>(base, row_bytes, blocks, prepared.data(), group_out, r0, r1); break;
            default: rows<8>(base, row_bytes, blocks, prepared.data(), group_out, r0, r1); break;
        }
    }
}

}  // namespace strata::kernels::cpu
