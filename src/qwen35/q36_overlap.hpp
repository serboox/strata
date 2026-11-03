#pragma once
#include <cstdio>
#include <cstdlib>
#include <string>

namespace q36 {

/// STRATA_Q36_OVERLAP: which ways of overlapping the GPU and CPU work of a hybrid decode step are on.
struct Overlap {
    bool quant = false;    // the CPU path's activations are quantized on the GPU
    bool glue = false;     // the host's job building uses preallocated tables; one upload merges the CPU rows
    bool commit = false;   // the GDN state commit of a speculative step runs beside the next step's MTP forward
    bool stats = false;    // STRATA_Q36_OVERLAP_STATS: a timing line per forward on stderr
    bool verify = false;   // STRATA_Q36_VERIFY_OVERLAP: every switch's result is compared with the old way's
    int stats_every = 1;   // print every N-th forward

    bool any() const { return quant || glue || commit; }
};

/// "quant,glue", "all" or "1" (every switch there is), "0" / "" / "off" (none).  `rows` is known but not implemented: it
/// is named in the warning and ignored.  An unknown name is a warning too, never a silent no-op.
inline Overlap parse_overlap(const char* list, const char* stats, const char* verify) {
    Overlap o;
    if (list != nullptr) {
        const std::string s(list);
        size_t at = 0;
        while (at <= s.size()) {
            size_t end = s.find(',', at);
            if (end == std::string::npos) end = s.size();
            std::string tok = s.substr(at, end - at);
            at = end + 1;
            const size_t first = tok.find_first_not_of(" \t"), last = tok.find_last_not_of(" \t");
            tok = first == std::string::npos ? std::string() : tok.substr(first, last - first + 1);
            if (tok.empty() || tok == "0" || tok == "off") continue;
            if (tok == "1" || tok == "all") o.quant = o.glue = o.commit = true;
            else if (tok == "quant") o.quant = true;
            else if (tok == "glue") o.glue = true;
            else if (tok == "commit") o.commit = true;
            else if (tok == "rows")
                std::fprintf(stderr, "strata-q36: STRATA_Q36_OVERLAP: 'rows' (row wavefront) is not implemented, ignored\n");
            else std::fprintf(stderr, "strata-q36: STRATA_Q36_OVERLAP: unknown switch '%s' ignored\n", tok.c_str());
        }
    }
    if (stats != nullptr && stats[0] != 0 && std::string(stats) != "0") {
        o.stats = true;
        const int n = std::atoi(stats);
        o.stats_every = n > 1 ? n : 1;
    }
    o.verify = verify != nullptr && verify[0] != 0 && std::string(verify) != "0";
    return o;
}

inline Overlap overlap_from_env() {
    return parse_overlap(std::getenv("STRATA_Q36_OVERLAP"), std::getenv("STRATA_Q36_OVERLAP_STATS"),
                         std::getenv("STRATA_Q36_VERIFY_OVERLAP"));
}

}  // namespace q36
