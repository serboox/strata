#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// IQ3_S gate/up rows with SwiGLU, ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t]) for rows [r0, r1) and `nt` tokens, decoded
/// with byte permutes (AVX-512 VBMI) and no table gathers. Callers must check cpu_avx512_vbmi_ok() first.
void iq3s_vbmi_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt, float* const* ff,
                       int r0, int r1);

}  // namespace strata::kernels::cpu
