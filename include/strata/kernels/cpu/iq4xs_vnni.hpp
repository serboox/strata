#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// IQ4_XS rows: out[t][r] = row_r . act[t] for rows [r0, r1) and `nt` tokens (activations as Q8_K), with a nibble lookup and VNNI dot
/// products. Callers must check cpu_avx512_ok() first.
void iq4xs_vnni_rows(const uint8_t* rows, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0, int r1);

/// The same kernel for rows whose 4-bit codes stand for the odd values 2c - 15 (an IQ3_S matrix repacked by iq3s_repack_to_4bit).
void iq4xs_vnni_rows_odd(const uint8_t* rows, size_t row_bytes, int n, const void* const* act, int nt, float* const* out, int r0, int r1);

/// Repacks `blocks` IQ3_S blocks into IQ4_XS-layout blocks (136 bytes) with the odd-value code table: every weight keeps its value.
void iq3s_repack_to_4bit(const uint8_t* src, uint8_t* dst, size_t blocks);

}  // namespace strata::kernels::cpu
