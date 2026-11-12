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
constexpr int kTileRows = 16;
// Rows of two blocks (n = 512, the down projection of the target model) get a fully unrolled kernel; other lengths use the generic loop.
constexpr int kPairBlocks = 2;

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
};

// The activations of one group of tokens: regrouped blocks ([token][block]) and the Q8_K scales ([block][token], zero for unused tokens).
struct PreparedActivations {
    const BlockActivation* blocks;
    const float* scale;
};

struct Constants {
    __m512i lookup;
    __m512i high_nibble_windows;
    __m512i pick_low, pick_high;
    __m512i group_of_lane[2];
    __m512i row_group_of_lane[kPairBlocks][2];
};

inline Constants make_constants(bool odd_table) {
    Constants c;
    alignas(16) uint8_t offset_values[16];
    // odd_table: the code c stands for 2c - 15 (the odd values -15 .. 15 of a repacked IQ3_S weight), not for ggml's non-linear table
    for (int i = 0; i < 16; ++i) offset_values[i] = (uint8_t) ((odd_table ? 2 * i - 15 : kvalues_iq4nl[i]) + 127);
    // Four copies of the table: vpermb reads six index bits, so the upper two bits of an index byte must not matter.
    c.lookup = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) offset_values));
    // vpmultishiftqb: result byte i is bits 8i+4..8i+11, i.e. the high nibble of byte i (its upper four bits are the next byte's low nibble).
    c.high_nibble_windows = _mm512_set1_epi64(0x3C342C241C140C04LL);
    c.pick_low = _mm512_set_epi64(13, 12, 9, 8, 5, 4, 1, 0);
    c.pick_high = _mm512_set_epi64(15, 14, 11, 10, 7, 6, 3, 2);
    c.group_of_lane[0] = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
    c.group_of_lane[1] = _mm512_set_epi32(7, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4);
    for (int block = 0; block < kPairBlocks; ++block)
        for (int h = 0; h < 2; ++h) {
            alignas(64) int32_t lanes[16];
            for (int lane = 0; lane < 16; ++lane) lanes[lane] = 8 * block + 4 * h + lane / 4;
            c.row_group_of_lane[block][h] = _mm512_load_si512(lanes);
        }
    return c;
}

thread_local bool use_odd_table = false;

const Constants& constants() {
    static const Constants plain = make_constants(false);
    static const Constants odd = make_constants(true);
    return use_odd_table ? odd : plain;
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
}

inline uint64_t scale_bytes_of(const uint8_t* block) {
    return _pdep_u64(u32(block + 4), 0x0F0F0F0F0F0F0F0FULL) | _pdep_u64(u16(block + 2), 0x3030303030303030ULL);
}

// Scales of the two blocks of one row: the sub-block scale of every lane per 128-weight half, and d times the activation scale
// (lane 8 * block + token).
struct PairScales {
    __m512 lane[kPairBlocks][2];
    __m512 block;
};

inline PairScales decode_pair(const uint8_t* row, const Constants& c, const PreparedActivations& act) {
    PairScales out;
    const __m128i bytes = _mm_sub_epi8(_mm_set_epi64x((long long) scale_bytes_of(row + kBlockBytes), (long long) scale_bytes_of(row)), _mm_set1_epi8(32));
    const __m512 lanes = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(bytes));
    for (int block = 0; block < kPairBlocks; ++block)
        for (int h = 0; h < 2; ++h) out.lane[block][h] = _mm512_permutexvar_ps(c.row_group_of_lane[block][h], lanes);
    const __m256i d_halves = _mm256_blend_epi32(_mm256_set1_epi16((short) u16(row)), _mm256_set1_epi16((short) u16(row + kBlockBytes)), 0xF0);
    out.block = _mm512_mul_ps(_mm512_cvtph_ps(d_halves), _mm512_loadu_ps(act.scale));
    return out;
}

template <int NT, int BLOCKS, bool PREFETCH>
inline void accumulate_row(const uint8_t* row, int runtime_blocks, const Constants& c, const PreparedActivations& act, __m512* acc_f) {
    const int blocks = BLOCKS ? BLOCKS : runtime_blocks;
    for (int t = 0; t < NT; ++t) acc_f[t] = _mm512_setzero_ps();
    PairScales pair;
    if constexpr (BLOCKS == kPairBlocks) pair = decode_pair(row, c, act);
    for (int i = 0; i < blocks; ++i) {
        const uint8_t* block = row + (size_t) i * kBlockBytes;
        if constexpr (PREFETCH) {
            _mm_prefetch((const char*) block + kPrefetchBytes, _MM_HINT_T0);
            _mm_prefetch((const char*) block + kPrefetchBytes + 64, _MM_HINT_T0);
        }
        __m512 scale[2], block_scale;
        int block_scale_first_lane = 0;
        if constexpr (BLOCKS == kPairBlocks) {
            scale[0] = pair.lane[i][0];
            scale[1] = pair.lane[i][1];
            block_scale = pair.block;
            block_scale_first_lane = 8 * i;
        } else {
            const __m128i scale_bytes = _mm_sub_epi8(_mm_cvtsi64_si128((long long) scale_bytes_of(block)), _mm_set1_epi8(32));
            const __m512 scale_lanes = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(scale_bytes));
            scale[0] = _mm512_permutexvar_ps(c.group_of_lane[0], scale_lanes);
            scale[1] = _mm512_permutexvar_ps(c.group_of_lane[1], scale_lanes);
            block_scale = _mm512_castps256_ps512(_mm256_mul_ps(_mm256_set1_ps(h2f(u16(block))), _mm256_loadu_ps(act.scale + (size_t) i * kMaxTokens)));
        }
        __m512i weight_low[2], weight_high[2];
        for (int h = 0; h < 2; ++h) {
            const __m512i packed = _mm512_loadu_si512(block + 8 + 64 * h);
            weight_low[h] = _mm512_permutexvar_epi8(packed, c.lookup);
            // The single-token kernel is bound by port 5 and the others by port 0; the shift runs on port 0, the multishift on port 5.
            const __m512i high_index = NT == 1 ? _mm512_srli_epi16(packed, 4) : _mm512_multishift_epi64_epi8(c.high_nibble_windows, packed);
            weight_high[h] = _mm512_permutexvar_epi8(high_index, c.lookup);
        }
        for (int t = 0; t < NT; ++t) {
            const BlockActivation& y = act.blocks[(size_t) t * blocks + i];
            __m512i sum[2];
            for (int h = 0; h < 2; ++h) {
                sum[h] = _mm512_load_si512(y.correction[h]);
                sum[h] = _mm512_dpbusd_epi32(sum[h], weight_low[h], _mm512_load_si512(y.low[h]));
                sum[h] = _mm512_dpbusd_epi32(sum[h], weight_high[h], _mm512_load_si512(y.high[h]));
            }
            // Every product and sum here is an integer below 2^24, so the float arithmetic is exact and equals the integer total.
            const __m512 total = _mm512_fmadd_ps(_mm512_cvtepi32_ps(sum[1]), scale[1], _mm512_mul_ps(_mm512_cvtepi32_ps(sum[0]), scale[0]));
            acc_f[t] = _mm512_fmadd_ps(_mm512_permutexvar_ps(_mm512_set1_epi32(block_scale_first_lane + t), block_scale), total, acc_f[t]);
        }
    }
}

// Sums the 16 lanes of each of 16 accumulators with the add tree of _mm512_reduce_add_ps (lane i + i+8, i + i+4, i + i+2, i + i+1),
// all rows at once. Result lane r is the sum of tile row r.
inline __m512 reduce_tile(const float* tile) {
    __m512 halves[8];
    for (int k = 0; k < 8; ++k) {
        const __m512 a = _mm512_load_ps(tile + 32 * k), b = _mm512_load_ps(tile + 32 * k + 16);
        halves[k] = _mm512_add_ps(_mm512_shuffle_f32x4(a, b, 0x44), _mm512_shuffle_f32x4(a, b, 0xEE));
    }
    __m512 quarters[4];
    for (int m = 0; m < 4; ++m)
        quarters[m] = _mm512_add_ps(_mm512_shuffle_f32x4(halves[2 * m], halves[2 * m + 1], 0x88), _mm512_shuffle_f32x4(halves[2 * m], halves[2 * m + 1], 0xDD));
    __m512 pairs[2];
    for (int p = 0; p < 2; ++p)
        pairs[p] = _mm512_add_ps(_mm512_shuffle_ps(quarters[2 * p], quarters[2 * p + 1], 0x44), _mm512_shuffle_ps(quarters[2 * p], quarters[2 * p + 1], 0xEE));
    const __m512 sums = _mm512_add_ps(_mm512_shuffle_ps(pairs[0], pairs[1], 0x88), _mm512_shuffle_ps(pairs[0], pairs[1], 0xDD));
    const __m512i row_order = _mm512_set_epi32(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
    return _mm512_permutexvar_ps(row_order, sums);
}

template <int NT, int BLOCKS, bool PREFETCH>
inline void store_tile(const uint8_t* base, size_t row_bytes, int blocks, const PreparedActivations& act, float* const* out, int r) {
    const Constants& c = constants();
    alignas(64) float tile[NT][kTileRows][16];
    for (int row = 0; row < kTileRows; ++row) {
        __m512 acc[NT];
        accumulate_row<NT, BLOCKS, PREFETCH>(base + (size_t) (r + row) * row_bytes, blocks, c, act, acc);
        for (int t = 0; t < NT; ++t) _mm512_store_ps(tile[t][row], acc[t]);
    }
    for (int t = 0; t < NT; ++t) _mm512_storeu_ps(out[t] + r, reduce_tile(tile[t][0]));
}

template <int NT, int BLOCKS>
void rows_of(const uint8_t* base, size_t row_bytes, int blocks, const PreparedActivations& act, float* const* out, int r0, int r1) {
    // A row prefetches the block kPrefetchBytes ahead (and one more line): only rows whose targets stay inside [r0, r1) do it.
    const int prefetch_rows = row_bytes == 0 ? r1 - r0 : (int) ((kPrefetchBytes + 2 * 64 + row_bytes - 1) / row_bytes) + 1;
    const int prefetch_end = r1 - prefetch_rows;
    int r = r0;
    for (; r + kTileRows <= r1 && r + kTileRows <= prefetch_end; r += kTileRows) store_tile<NT, BLOCKS, true>(base, row_bytes, blocks, act, out, r);
    for (; r + kTileRows <= r1; r += kTileRows) store_tile<NT, BLOCKS, false>(base, row_bytes, blocks, act, out, r);
    const Constants& c = constants();
    for (; r < r1; ++r) {
        __m512 acc[NT];
        accumulate_row<NT, BLOCKS, false>(base + (size_t) r * row_bytes, blocks, c, act, acc);
        for (int t = 0; t < NT; ++t) out[t][r] = _mm512_reduce_add_ps(acc[t]);
    }
}

template <int NT>
void rows(const uint8_t* base, size_t row_bytes, int blocks, const PreparedActivations& act, float* const* out, int r0, int r1) {
    if (blocks == kPairBlocks) rows_of<NT, kPairBlocks>(base, row_bytes, blocks, act, out, r0, r1);
    else rows_of<NT, 0>(base, row_bytes, blocks, act, out, r0, r1);
}

}  // namespace

namespace {
void rows_any(const uint8_t* base, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0, int r1);
}

void iq4xs_vnni_rows(const uint8_t* base, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0, int r1) {
    use_odd_table = false;
    rows_any(base, row_bytes, n, act, nt, out, r0, r1);
}

void iq4xs_vnni_rows_odd(const uint8_t* base, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0, int r1) {
    use_odd_table = true;
    rows_any(base, row_bytes, n, act, nt, out, r0, r1);
    use_odd_table = false;
}

namespace {
void rows_any(const uint8_t* base, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0, int r1) {
    const int blocks = n / QK_K;
    thread_local std::vector<BlockActivation> prepared;
    thread_local std::vector<float> act_scale;
    for (int t0 = 0; t0 < nt; t0 += kMaxTokens) {
        const int group = nt - t0 < kMaxTokens ? nt - t0 : kMaxTokens;
        prepared.resize((size_t) group * blocks);
        act_scale.assign((size_t) blocks * kMaxTokens, 0.f);
        for (int t = 0; t < group; ++t) {
            const block_q8_K* y = (const block_q8_K*) act[t0 + t];
            for (int i = 0; i < blocks; ++i) {
                prepare(y[i], constants(), prepared[(size_t) t * blocks + i]);
                act_scale[(size_t) i * kMaxTokens + t] = y[i].d;
            }
        }
        float* const* group_out = out + t0;
        const PreparedActivations activations{prepared.data(), act_scale.data()};
        switch (group) {
            case 1: rows<1>(base, row_bytes, blocks, activations, group_out, r0, r1); break;
            case 2: rows<2>(base, row_bytes, blocks, activations, group_out, r0, r1); break;
            case 3: rows<3>(base, row_bytes, blocks, activations, group_out, r0, r1); break;
            case 4: rows<4>(base, row_bytes, blocks, activations, group_out, r0, r1); break;
            case 5: rows<5>(base, row_bytes, blocks, activations, group_out, r0, r1); break;
            case 6: rows<6>(base, row_bytes, blocks, activations, group_out, r0, r1); break;
            case 7: rows<7>(base, row_bytes, blocks, activations, group_out, r0, r1); break;
            default: rows<8>(base, row_bytes, blocks, activations, group_out, r0, r1); break;
        }
    }
}

}  // namespace

void iq3s_repack_to_4bit(const uint8_t* src, uint8_t* dst, size_t blocks) {
    for (size_t b = 0; b < blocks; ++b) {
        const block_iq3_s* x = reinterpret_cast<const block_iq3_s*>(src) + b;
        uint8_t* out = dst + b * kBlockBytes;
        std::memcpy(out, &x->d, 2);
        uint16_t scales_h = 0;
        uint8_t* scales_l = out + 4;
        uint8_t* codes = out + 8;
        std::memset(scales_l, 0, 4);
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            const uint8_t* qs = x->qs + 8 * ib32;
            const uint8_t* signs = x->signs + 4 * ib32;
            const int qh = x->qh[ib32];
            int8_t w[32];
            for (int l = 0; l < 4; ++l) {
                const uint8_t* g1 = reinterpret_cast<const uint8_t*>(iq3s_grid + (qs[2 * l + 0] | ((qh << (8 - 2 * l)) & 256)));
                const uint8_t* g2 = reinterpret_cast<const uint8_t*>(iq3s_grid + (qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    w[8 * l + j] = (int8_t) ((signs[l] & kmask_iq2xs[j + 0]) ? -(int) g1[j] : (int) g1[j]);
                    w[8 * l + 4 + j] = (int8_t) ((signs[l] & kmask_iq2xs[j + 4]) ? -(int) g2[j] : (int) g2[j]);
                }
            }
            const int s = (x->scales[ib32 / 2] >> (4 * (ib32 & 1))) & 0xf;
            const int ls = 33 + 2 * s;   // (ls - 32) = 1 + 2s, the IQ3_S sub-block scale
            scales_l[ib32 / 2] |= (uint8_t) ((ls & 0xf) << (4 * (ib32 & 1)));
            scales_h |= (uint16_t) ((ls >> 4) << (2 * ib32));
            for (int j = 0; j < 16; ++j)
                codes[16 * ib32 + j] = (uint8_t) (((w[j] + 15) / 2) | (((w[j + 16] + 15) / 2) << 4));
        }
        std::memcpy(out + 2, &scales_h, 2);
    }
}

}  // namespace strata::kernels::cpu
