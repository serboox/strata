#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace strata::core {

inline constexpr const char* kExpertsUsedEnv = "STRATA_Q36_EXPERTS_USED";

/// The routed experts per token a run uses: the GGUF's `expert_used_count` (`file_value`) when `text` is null or empty, else the
/// integer in `text`, which must lie in 1 .. file_value (a run never uses more experts than the file was trained with).
/// False with `err` for anything else, so a typo is never taken for the file's value.
inline bool resolve_experts_used(const char* text, int64_t file_value, int64_t& top_k, std::string& err) {
    top_k = file_value;
    if (text == nullptr || *text == '\0') return true;
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text, &end, 10);
    if (end == text || *end != '\0' || errno == ERANGE) {
        err = std::string(kExpertsUsedEnv) + "=" + text + " is not an integer";
        return false;
    }
    if (value < 1 || value > file_value) {
        err = std::string(kExpertsUsedEnv) + "=" + text + " is outside 1 .. " + std::to_string(file_value) +
              " (the model's expert_used_count)";
        return false;
    }
    top_k = value;
    return true;
}

}  // namespace strata::core
