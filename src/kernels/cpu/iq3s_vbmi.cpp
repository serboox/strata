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
constexpr int kPrefetchBytes = 2048;

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }
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

// 9-bit index = qs byte | (qh bit << 8). Index bits 6..0 pick inside a 128-entry table pair, bit 7 and the qh bit pick the pair.
inline __m512i lookup(const uint8_t* plane, __m512i idx, __mmask64 bit7, __mmask64 high) {
    const __m512i r0 = _mm512_permutex2var_epi8(_mm512_load_si512(plane), idx, _mm512_load_si512(plane + 64));
    const __m512i r1 = _mm512_permutex2var_epi8(_mm512_load_si512(plane + 128), idx, _mm512_load_si512(plane + 192));
    const __m512i r2 = _mm512_permutex2var_epi8(_mm512_load_si512(plane + 256), idx, _mm512_load_si512(plane + 320));
    const __m512i r3 = _mm512_permutex2var_epi8(_mm512_load_si512(plane + 384), idx, _mm512_load_si512(plane + 448));
    return _mm512_mask_blend_epi8(high, _mm512_mask_blend_epi8(bit7, r0, r1), _mm512_mask_blend_epi8(bit7, r2, r3));
}

struct alignas(64) Plane {
    uint8_t bytes[64];
};

struct Constants {
    __m512i sign_pick[4];   // for plane q: output bit i takes sign bit 4i+q of the 256 sign bits
    __m512i sign_qwords;    // sign qword k/2 into lane k
    __m512i scale_lanes;    // int16 lane l takes the scale of group l/4
    __m512i y_pick[4];      // for plane q: byte k takes activation byte 4k+q of a 128-byte window
};

inline Constants make_constants() {
    Constants c;
    for (int q = 0; q < 4; ++q) {
        alignas(64) uint8_t pick[64], ypick[64];
        for (int lane = 0; lane < 8; ++lane)
            for (int j = 0; j < 8; ++j) pick[lane * 8 + j] = (uint8_t) (32 * (lane % 2) + 4 * j + q);
        for (int k = 0; k < 64; ++k) ypick[k] = (uint8_t) ((4 * (k % 32) + q) & 127);
        c.sign_pick[q] = _mm512_load_si512(pick);
        c.y_pick[q] = _mm512_load_si512(ypick);
    }
    c.sign_qwords = _mm512_set_epi64(3, 3, 2, 2, 1, 1, 0, 0);
    c.scale_lanes = _mm512_set_epi16(7, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
    return c;
}

const Constants& constants() {
    static const Constants value = make_constants();
    return value;
}

// one Q8_K block of 256 activations, regrouped so that plane q holds the bytes at positions 4e+q
inline void activation_planes(const int8_t* qs, const Constants& c, Plane* out) {
    const __m512i a0 = _mm512_loadu_si512(qs), a1 = _mm512_loadu_si512(qs + 64);
    const __m512i a2 = _mm512_loadu_si512(qs + 128), a3 = _mm512_loadu_si512(qs + 192);
    for (int q = 0; q < 4; ++q) {
        const __m512i low = _mm512_permutex2var_epi8(a0, c.y_pick[q], a1);
        const __m512i high = _mm512_permutex2var_epi8(a2, c.y_pick[q], a3);
        _mm512_store_si512(out[q].bytes, _mm512_inserti64x4(low, _mm512_castsi512_si256(high), 1));
    }
}

template <int NT>
inline void row_dot(const uint8_t* row, int blocks, const GridPacks& grid, const Constants& c, const Plane* planes,
                    const float* act_d, float* result) {
    __m512 acc_f[NT];
    for (int t = 0; t < NT; ++t) acc_f[t] = _mm512_setzero_ps();
    const __m512i zero = _mm512_setzero_si512();
    for (int i = 0; i < blocks; ++i) {
        const uint8_t* block = row + (size_t) i * kBlockBytes;
        _mm_prefetch((const char*) block + kPrefetchBytes, _MM_HINT_T0);
        _mm_prefetch((const char*) block + kPrefetchBytes + 64, _MM_HINT_T0);
        const __m512i idx = _mm512_loadu_si512(block + 2);
        const __mmask64 high = _cvtu64_mask64(u64(block + 66));
        const __mmask64 bit7 = _mm512_test_epi8_mask(idx, _mm512_set1_epi8((char) 0x80));
        __m512i grid_q[4];
        __mmask64 negate[4];
        const __m512i nibble = _mm512_set1_epi8(0x0F);
        const __m512i pack0 = lookup(grid.byte[0], idx, bit7, high), pack1 = lookup(grid.byte[1], idx, bit7, high);
        grid_q[0] = _mm512_and_si512(pack0, nibble);
        grid_q[1] = _mm512_and_si512(_mm512_srli_epi16(pack0, 4), nibble);
        grid_q[2] = _mm512_and_si512(pack1, nibble);
        grid_q[3] = _mm512_and_si512(_mm512_srli_epi16(pack1, 4), nibble);
        const __m512i signs = _mm512_permutexvar_epi64(c.sign_qwords, _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*) (block + 74))));
        for (int q = 0; q < 4; ++q) negate[q] = _mm512_bitshuffle_epi64_mask(signs, c.sign_pick[q]);
        uint64_t nibbles = _pdep_u64(u32(block + 106), 0x0F0F0F0F0F0F0F0FULL);
        nibbles = (nibbles << 1) | 0x0101010101010101ULL;
        const __m512i scale = _mm512_permutexvar_epi16(c.scale_lanes, _mm512_castsi128_si512(_mm_cvtepu8_epi16(_mm_cvtsi64_si128((long long) nibbles))));
        const float dx = h2f(u16(block));
        for (int t = 0; t < NT; ++t) {
            const Plane* y = planes + ((size_t) t * blocks + i) * 4;
            __m512i sum = _mm512_setzero_si512();
            for (int q = 0; q < 4; ++q) {
                // Q8_K values are in -127..127, so negating a byte cannot overflow
                const __m512i activation = _mm512_load_si512(y[q].bytes);
                const __m512i signed_y = _mm512_mask_sub_epi8(activation, negate[q], zero, activation);
                sum = _mm512_add_epi16(sum, _mm512_maddubs_epi16(grid_q[q], signed_y));
            }
            const __m512i total = _mm512_madd_epi16(sum, scale);
            acc_f[t] = _mm512_fmadd_ps(_mm512_set1_ps(dx * act_d[(size_t) t * blocks + i]), _mm512_cvtepi32_ps(total), acc_f[t]);
        }
    }
    for (int t = 0; t < NT; ++t) result[t] = _mm512_reduce_add_ps(acc_f[t]);
}

template <int NT>
void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int blocks, const Plane* planes, const float* act_d, float* const* ff,
             int r0, int r1) {
    const GridPacks& grid = grid_packs();
    float gate[NT], up[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<NT>(blob + (size_t) r * gu_row, blocks, grid, constants(), planes, act_d, gate);
        row_dot<NT>(blob + up_off + (size_t) r * gu_row, blocks, grid, constants(), planes, act_d, up);
        for (int t = 0; t < NT; ++t) ff[t][r] = (gate[t] / (1.f + std::exp(-gate[t]))) * up[t];
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
        act_d.resize((size_t) group * blocks);
        for (int t = 0; t < group; ++t) {
            const block_q8_K* y = (const block_q8_K*) act[t0 + t];
            for (int i = 0; i < blocks; ++i) {
                activation_planes(y[i].qs, constants(), &planes[((size_t) t * blocks + i) * 4]);
                act_d[(size_t) t * blocks + i] = y[i].d;
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
