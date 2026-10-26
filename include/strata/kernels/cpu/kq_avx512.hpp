#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// Q4_K gate/up and Q5_K down rows for several tokens on AVX-512, each result equal bit for bit to ggml-cpu's AVX2 dot.
bool kq512_gu_supported(int ggml_type, int n) noexcept;
bool kq512_down_supported(int ggml_type, int n) noexcept;
void kq512_gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                   float* const* ff, int r0, int r1);
void kq512_down_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                     int r0, int r1);

}  // namespace strata::kernels::cpu
