#pragma once

namespace strata::kernels::cpu {

/// silu(gate) * up with the operation sequence that the GPU strict kernels use (strict_math_impl.hpp).
float strict_swiglu(float gate, float up) noexcept;

/// On: the IQ3_S and the repacked 4-bit gate/up kernels use strict_swiglu, not the C library's exp, so an expert gives the
/// same bits on the CPU and in a VRAM slot. Set once, before the pool runs its first job.
void set_strict_swiglu(bool on) noexcept;
bool strict_swiglu_on() noexcept;

}  // namespace strata::kernels::cpu
