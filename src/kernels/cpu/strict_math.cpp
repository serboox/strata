#include "strata/kernels/cpu/strict_math.hpp"

#include "strata/kernels/strict_math_impl.hpp"

#include <atomic>

namespace strata::kernels::cpu {
namespace {
std::atomic<bool> g_strict_swiglu{false};
}

float strict_swiglu(float gate, float up) noexcept { return ::strata::kernels::strict_swiglu(gate, up); }

void set_strict_swiglu(bool on) noexcept { g_strict_swiglu.store(on, std::memory_order_relaxed); }
bool strict_swiglu_on() noexcept { return g_strict_swiglu.load(std::memory_order_relaxed); }

}  // namespace strata::kernels::cpu
