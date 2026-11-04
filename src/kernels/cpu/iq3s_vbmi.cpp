#include "strata/kernels/cpu/iq3s_vbmi.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <cmath>
#include <cstring>
#include <vector>

namespace strata::kernels::cpu {
namespace {

constexpr int kBlockBytes = 110;
constexpr int kMaxTokens = 8;
constexpr size_t kPrefetchBytes = 2048;
constexpr int kChunkBlocks = 8;      // blocks whose scale factors are converted together
constexpr int kOffsetMinTokens = 2;  // from this many tokens on, the weights are signed once per block and not every activation
constexpr int kRowBatch = 16;        // rows whose dots are computed before their SwiGLU

inline void prefetch_row(const uint8_t* row, size_t bytes) {
    for (size_t offset = 0; offset < bytes; offset += 64) _mm_prefetch((const char*) row + offset, _MM_HINT_T0);
}

inline int chunk_stride(int blocks) { return (blocks + kChunkBlocks - 1) / kChunkBlocks * kChunkBlocks; }

inline uint16_t u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline uint32_t u32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint64_t u64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

// The magnitudes of a grid word are odd numbers from 1 to 15, so two of its weights fit in one byte: pack 0 holds the weights at
// positions 0 and 1 of every group of four (low and high nibble), pack 1 the weights at positions 2 and 3.
struct GridPacks {
    alignas(64) uint8_t byte[2][512];
    GridPacks() {
        for (int e = 0; e < 512; ++e)
            for (int pack = 0; pack < 2; ++pack) {
                const uint32_t low = (iq3s_grid[e] >> (16 * pack)) & 0xFF, high = (iq3s_grid[e] >> (16 * pack + 8)) & 0xFF;
                byte[pack][e] = (uint8_t) (low | (high << 4));
            }
    }
};

const GridPacks& grid_packs() {
    static const GridPacks packs;
    return packs;
}

struct Tables {
    __m512i pack[2][8];
};

inline Tables load_tables(const GridPacks& grid) {
    Tables tables;
    for (int p = 0; p < 2; ++p)
        for (int j = 0; j < 8; ++j) tables.pack[p][j] = _mm512_load_si512(grid.byte[p] + 64 * j);
    return tables;
}

// 9-bit index = qs byte | (qh bit << 8). Index bits 6..0 pick inside a 128-entry table pair, bit 7 and the qh bit pick the pair.
inline __m512i lookup(const __m512i* table, __m512i idx, __mmask64 bit7, __mmask64 high) {
    const __m512i r0 = _mm512_permutex2var_epi8(table[0], idx, table[1]);
    const __m512i r1 = _mm512_permutex2var_epi8(table[2], idx, table[3]);
    const __m512i r2 = _mm512_permutex2var_epi8(table[4], idx, table[5]);
    const __m512i r3 = _mm512_permutex2var_epi8(table[6], idx, table[7]);
    return _mm512_mask_blend_epi8(high, _mm512_mask_blend_epi8(bit7, r0, r1), _mm512_mask_blend_epi8(bit7, r2, r3));
}

struct alignas(64) Plane {
    uint8_t bytes[64];
};

struct Constants {
    __m512i sign_bit[4];    // for plane q: byte p tests bit q (even p) or q+4 (odd p) of the sign byte p/2
    __m512i sign_bytes;     // byte p takes byte 32 + p/2 of the 64 bytes loaded at block + 42
    __m512i scale_bytes;    // byte 4i takes scale byte i/2, the other bytes take a zero byte: int32 lane i holds the scale of group i/2
    __m512i y_pick[4];      // for plane q: byte k takes activation byte 4k+q of a 128-byte window
};

inline Constants make_constants() {
    Constants c;
    for (int q = 0; q < 4; ++q) {
        alignas(64) uint8_t bit[64], ypick[64];
        for (int k = 0; k < 64; ++k) {
            bit[k] = (uint8_t) (1 << (q + 4 * (k % 2)));
            ypick[k] = (uint8_t) ((4 * (k % 32) + q) & 127);
        }
        c.sign_bit[q] = _mm512_load_si512(bit);
        c.y_pick[q] = _mm512_load_si512(ypick);
    }
    alignas(64) uint8_t pick[64];
    for (int k = 0; k < 64; ++k) pick[k] = (uint8_t) (32 + k / 2);
    c.sign_bytes = _mm512_load_si512(pick);
    alignas(64) uint8_t scale_pick[64];
    for (int b = 0; b < 64; ++b) scale_pick[b] = (uint8_t) (b % 4 == 0 ? b / 8 : 8);
    c.scale_bytes = _mm512_load_si512(scale_pick);
    return c;
}

const Constants& constants() {
    static const Constants value = make_constants();
    return value;
}

// one Q8_K block of 256 activations, regrouped so that plane q holds the bytes at positions 4e+q
inline void activation_planes(const int8_t* qs, const Constants& c, bool offset, Plane* out) {
    const __m512i a0 = _mm512_loadu_si512(qs), a1 = _mm512_loadu_si512(qs + 64);
    const __m512i a2 = _mm512_loadu_si512(qs + 128), a3 = _mm512_loadu_si512(qs + 192);
    for (int q = 0; q < 4; ++q) {
        const __m512i low = _mm512_permutex2var_epi8(a0, c.y_pick[q], a1);
        const __m512i high = _mm512_permutex2var_epi8(a2, c.y_pick[q], a3);
        const __m512i plane = _mm512_inserti64x4(low, _mm512_castsi512_si256(high), 1);
        _mm512_store_si512(out[q].bytes, offset ? _mm512_xor_si512(plane, _mm512_set1_epi8((char) 0x80)) : plane);
    }
}

// What the dot product needs from one weight block. With one token the grid magnitudes and sign masks stay apart and the signs are
// applied to the activation. With more tokens the signs are applied once to the weights, and `start` holds the correction for the
// activation offset (see accumulate).
template <int NT>
struct Decoded {
    __m512i grid[4];
    __mmask64 negate[4];
    __m512i scale;
    __m512i start;
};

template <int NT>
inline Decoded<NT> decode_block(const uint8_t* block, const Tables& tables, const Constants& c) {
    Decoded<NT> d;
    const __m512i idx = _mm512_loadu_si512(block + 2);
    const __mmask64 high = _cvtu64_mask64(u64(block + 66));
    const __mmask64 bit7 = _mm512_movepi8_mask(idx);
    const __m512i nibble = _mm512_set1_epi8(0x0F);
    const __m512i pack0 = lookup(tables.pack[0], idx, bit7, high), pack1 = lookup(tables.pack[1], idx, bit7, high);
    d.grid[0] = _mm512_and_si512(pack0, nibble);
    d.grid[1] = _mm512_and_si512(_mm512_srli_epi16(pack0, 4), nibble);
    d.grid[2] = _mm512_and_si512(pack1, nibble);
    d.grid[3] = _mm512_and_si512(_mm512_srli_epi16(pack1, 4), nibble);
    const __m512i signs = _mm512_permutexvar_epi8(c.sign_bytes, _mm512_loadu_si512(block + 42));
    for (int q = 0; q < 4; ++q) d.negate[q] = _mm512_test_epi8_mask(signs, c.sign_bit[q]);
    const uint64_t scale_values = _pdep_u64(u32(block + 106), 0x1E1E1E1E1E1E1E1EULL) | 0x0101010101010101ULL;
    d.scale = _mm512_permutexvar_epi8(c.scale_bytes, _mm512_zextsi128_si512(_mm_cvtsi64_si128((long long) scale_values)));
    d.start = _mm512_setzero_si512();
    if constexpr (NT >= kOffsetMinTokens) {
        const __m512i zero = _mm512_setzero_si512();
        for (int q = 0; q < 4; ++q) {
            d.grid[q] = _mm512_mask_sub_epi8(d.grid[q], d.negate[q], zero, d.grid[q]);
            d.start = _mm512_dpbusd_epi32(d.start, _mm512_set1_epi8((char) 0x80), d.grid[q]);
        }
        d.start = _mm512_sub_epi32(zero, d.start);
    }
    return d;
}

// acc_f[t] += factor[t][k] * (the integer dot product of the block with token t's activation block `y` + t * token_stride)
template <int NT>
inline void accumulate(const Decoded<NT>& d, const Plane* y, size_t token_stride, const float (*factor)[kChunkBlocks], int k, __m512* acc_f) {
    const __m512i zero = _mm512_setzero_si512();
    for (int t = 0; t < NT; ++t) {
        const Plane* planes_t = y + t * token_stride;
        __m512i sum = d.start;
        for (int q = 0; q < 4; ++q) {
            const __m512i activation = _mm512_load_si512(planes_t[q].bytes);
            if constexpr (NT >= kOffsetMinTokens) {
                // (activation + 128) * signed weight; `start` took 128 times the sum of the weights off
                sum = _mm512_dpbusd_epi32(sum, activation, d.grid[q]);
            } else {
                // Q8_K values are in -127..127, so negating a byte cannot overflow
                const __m512i signed_y = _mm512_mask_sub_epi8(activation, d.negate[q], zero, activation);
                sum = _mm512_dpbusd_epi32(sum, d.grid[q], signed_y);
            }
        }
        // the sum of 16 products is below 2^15, so the 16-bit multiply by the scale in the low half is exact
        const __m512i total = _mm512_madd_epi16(sum, d.scale);
        acc_f[t] = _mm512_fmadd_ps(_mm512_set1_ps(factor[t][k]), _mm512_cvtepi32_ps(total), acc_f[t]);
    }
}

// factor[t][k] = d of block chunk + k of the row times the scale of token t's activation block.
// Always inlined: out of line, the compiler builds the halves through extra loads and stores.
template <int NT>
[[gnu::always_inline]] inline void scale_factors(const uint8_t* row, int chunk, int count, const float* act_d, int act_stride,
                                                 float (*factor)[kChunkBlocks]) {
    alignas(16) uint16_t halves[kChunkBlocks] = {};
    // the constant trip count keeps the copy unrolled
    if (count == kChunkBlocks) {
        for (int k = 0; k < kChunkBlocks; ++k) halves[k] = u16(row + (size_t) (chunk + k) * kBlockBytes);
    } else {
        for (int k = 0; k < count; ++k) halves[k] = u16(row + (size_t) (chunk + k) * kBlockBytes);
    }
    // keeps the halves in memory: assembled in a register they would cost one vector insert each
    asm volatile("" : : "r"(halves) : "memory");
    const __m256 dx = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) halves));
    for (int t = 0; t < NT; ++t) _mm256_store_ps(factor[t], _mm256_mul_ps(dx, _mm256_loadu_ps(act_d + (size_t) t * act_stride + chunk)));
}

template <int NT>
inline void row_dot(const uint8_t* row, const uint8_t* ahead, int blocks, const Tables& tables, const Constants& c, const Plane* planes,
                    const float* act_d, float* result) {
    __m512 acc_f[NT];
    for (int t = 0; t < NT; ++t) acc_f[t] = _mm512_setzero_ps();
    const int act_stride = chunk_stride(blocks);
    const size_t token_stride = (size_t) blocks * 4;
    for (int chunk = 0; chunk < blocks; chunk += kChunkBlocks) {
        const int count = blocks - chunk < kChunkBlocks ? blocks - chunk : kChunkBlocks;
        alignas(32) float factor[NT][kChunkBlocks];
        scale_factors<NT>(row, chunk, count, act_d, act_stride, factor);
        const uint8_t* chunk_row = row + (size_t) chunk * kBlockBytes;
        const uint8_t* chunk_ahead = ahead + (size_t) chunk * kBlockBytes;
        auto prefetch_ahead = [&](int k) {
            _mm_prefetch((const char*) chunk_ahead + (size_t) k * kBlockBytes, _MM_HINT_T0);
            _mm_prefetch((const char*) chunk_ahead + (size_t) k * kBlockBytes + 64, _MM_HINT_T0);
        };
        if constexpr (NT >= kOffsetMinTokens) {
            // the next block is decoded before this one is multiplied, so the permutes of the decode (port 5) overlap the multiplies (port 0)
            Decoded<NT> current = decode_block<NT>(chunk_row, tables, c);
            for (int k = 0; k + 1 < count; ++k) {
                prefetch_ahead(k);
                const Decoded<NT> next = decode_block<NT>(chunk_row + (size_t) (k + 1) * kBlockBytes, tables, c);
                accumulate<NT>(current, planes + (size_t) (chunk + k) * 4, token_stride, factor, k, acc_f);
                current = next;
            }
            accumulate<NT>(current, planes + (size_t) (chunk + count - 1) * 4, token_stride, factor, count - 1, acc_f);
        } else {
            for (int k = 0; k < count; ++k) {
                prefetch_ahead(k);
                accumulate<NT>(decode_block<NT>(chunk_row + (size_t) k * kBlockBytes, tables, c), planes + (size_t) (chunk + k) * 4, token_stride,
                               factor, k, acc_f);
            }
        }
    }
    for (int t = 0; t < NT; ++t) result[t] = _mm512_reduce_add_ps(acc_f[t]);
}

template <int NT>
void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int blocks, const Plane* planes, const float* act_d, float* const* ff,
             int r0, int r1) {
    const Tables tables = load_tables(grid_packs());
    const Constants consts = constants();
    // the rows prefetch_rows ahead are read while the current row is computed; near the end of the range a row prefetches itself
    const int prefetch_rows = gu_row == 0 ? 0 : (int) ((kPrefetchBytes + gu_row - 1) / gu_row);
    for (int r = r0; r < r1 && r < r0 + prefetch_rows; ++r) {
        prefetch_row(blob + (size_t) r * gu_row, gu_row);
        prefetch_row(blob + up_off + (size_t) r * gu_row, gu_row);
    }
    for (int batch = r0; batch < r1; batch += kRowBatch) {
        const int rows = r1 - batch < kRowBatch ? r1 - batch : kRowBatch;
        float gate[kRowBatch][NT], up[kRowBatch][NT];
        for (int k = 0; k < rows; ++k) {
            const int r = batch + k;
            const uint8_t* gate_row = blob + (size_t) r * gu_row;
            const uint8_t* up_row = blob + up_off + (size_t) r * gu_row;
            const size_t ahead_bytes = r + prefetch_rows < r1 ? (size_t) prefetch_rows * gu_row : 0;
            row_dot<NT>(gate_row, gate_row + ahead_bytes, blocks, tables, consts, planes, act_d, gate[k]);
            row_dot<NT>(up_row, up_row + ahead_bytes, blocks, tables, consts, planes, act_d, up[k]);
        }
        // the exp call clobbers every vector register, so it stays out of the row loop above
        for (int k = 0; k < rows; ++k)
            for (int t = 0; t < NT; ++t) ff[t][batch + k] = (gate[k][t] / (1.f + std::exp(-gate[k][t]))) * up[k][t];
    }
}

}  // namespace

void iq3s_vbmi_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt, float* const* ff,
                       int r0, int r1) {
    const int blocks = n / QK_K;
    for (int t0 = 0; t0 < nt; t0 += kMaxTokens) {
        const int group = nt - t0 < kMaxTokens ? nt - t0 : kMaxTokens;
        thread_local std::vector<Plane> planes;
        thread_local std::vector<float> act_d;
        planes.resize((size_t) group * blocks * 4);
        const int act_stride = chunk_stride(blocks);
        act_d.assign((size_t) group * act_stride, 0.f);
        for (int t = 0; t < group; ++t) {
            const block_q8_K* y = (const block_q8_K*) act[t0 + t];
            for (int i = 0; i < blocks; ++i) {
                activation_planes(y[i].qs, constants(), group >= kOffsetMinTokens, &planes[((size_t) t * blocks + i) * 4]);
                act_d[(size_t) t * act_stride + i] = y[i].d;
            }
        }
        float* const* out = ff + t0;
        switch (group) {
            case 1: gu_rows<1>(blob, gu_row, up_off, blocks, planes.data(), act_d.data(), out, r0, r1); break;
            case 2: gu_rows<2>(blob, gu_row, up_off, blocks, planes.data(), act_d.data(), out, r0, r1); break;
            case 3: gu_rows<3>(blob, gu_row, up_off, blocks, planes.data(), act_d.data(), out, r0, r1); break;
            case 4: gu_rows<4>(blob, gu_row, up_off, blocks, planes.data(), act_d.data(), out, r0, r1); break;
            case 5: gu_rows<5>(blob, gu_row, up_off, blocks, planes.data(), act_d.data(), out, r0, r1); break;
            case 6: gu_rows<6>(blob, gu_row, up_off, blocks, planes.data(), act_d.data(), out, r0, r1); break;
            case 7: gu_rows<7>(blob, gu_row, up_off, blocks, planes.data(), act_d.data(), out, r0, r1); break;
            default: gu_rows<8>(blob, gu_row, up_off, blocks, planes.data(), act_d.data(), out, r0, r1); break;
        }
    }
}

}  // namespace strata::kernels::cpu
