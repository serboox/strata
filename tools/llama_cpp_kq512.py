import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KERNEL_HEADER = Path("include/strata/kernels/cpu/kq_avx512.hpp")
KERNEL_SOURCE = Path("src/kernels/cpu/kq_avx512.cpp")
CPU_DIR = Path("ggml/src/ggml-cpu")
TESTED_LLAMA_CPP_COMMIT = "18ab62a"

BRIDGE_SOURCE = """#include "strata/kernels/cpu/kq_avx512.hpp"

extern "C" bool strata_kq_rows(int type, const void * w, size_t row_bytes, int n, const void * act, float * out, int rows) {
    if (!strata::kernels::cpu::kq512_rows_supported(type, n)) {
        return false;
    }
    const void * acts[1] = { act };
    float * outs[1] = { out };
    strata::kernels::cpu::kq512_rows(type, (const uint8_t *) w, row_bytes, n, acts, 1, outs, 0, rows);
    return true;
}
"""

CMAKE_ANCHOR = "        ggml-cpu/quants.c\n"
CMAKE_SOURCES = ("strata_kq_avx512.cpp", "strata_kq_rows.cpp")

HOOK_FUNCTION_ANCHOR = "static void ggml_compute_forward_mul_mat_id_one_chunk("
HOOK_DECLARATION = """extern bool strata_kq_rows(int type, const void * w, size_t row_bytes, int n, const void * act, float * out, int rows);

static bool strata_kq_enabled(void) {
    static atomic_int state = -1;
    int value = atomic_load_explicit(&state, memory_order_relaxed);
    if (value < 0) {
        const char * env = getenv("STRATA_KQ512");
        value = (env == NULL || env[0] != '0') ? 1 : 0;
        atomic_store_explicit(&state, value, memory_order_relaxed);
    }
    return value == 1;
}

"""

DOT_LOOP = """                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                    vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
                }
"""
DOT_LOOP_WITH_KERNEL = """                const int64_t rows_here = MIN(iir0 + blck_0, ir0_end) - iir0;
                if ((type == GGML_TYPE_Q2_K || type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) && vec_dot_type == GGML_TYPE_Q8_K && strata_kq_enabled() &&
                    strata_kq_rows((int) type, src0_cur + iir0*nb01, nb01, (int) ne00, src1_col, tmp, (int) rows_here)) {
                    // the rows of this block were done by the kernel
                } else {
                    for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                        vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
                    }
                }
"""


HOOK_TYPES_Q4_Q5 = "(type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) && vec_dot_type"
HOOK_TYPES_Q2_Q4_Q5 = "(type == GGML_TYPE_Q2_K || type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) && vec_dot_type"


class PatchError(Exception):
    pass


def copy_if_different(source: Path, target: Path) -> bool:
    if target.is_file() and target.read_bytes() == source.read_bytes():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return True


def write_if_different(target: Path, text: str) -> bool:
    if target.is_file() and target.read_text(encoding="utf-8") == text:
        return False
    target.write_text(text, encoding="utf-8")
    return True


def plan_edits(path: Path, edits: list[tuple[str, str, str]]) -> str | None:
    """The text of `path` with every edit whose marker is missing applied; None when nothing is missing. Nothing is written."""
    original = path.read_text(encoding="utf-8")
    text = original
    for marker, old, new in edits:
        if marker in text:
            continue
        if text.count(old) != 1:
            raise PatchError(f"{path}: the expected place was not found exactly once: {old.strip().splitlines()[0]!r} (tested with llama.cpp {TESTED_LLAMA_CPP_COMMIT})")
        text = text.replace(old, new, 1)
    return None if text == original else text


def apply(llama: Path, repo: Path = REPO) -> list[str]:
    """Adds the Strata Q2_K, Q4_K and Q5_K expert kernels to ggml's mul_mat_id of a llama.cpp checkout; returns what changed.
    Every edit is planned before the first file is written, so a checkout that does not match is left as it was."""
    cpu = llama / CPU_DIR
    for required in (cpu / "ggml-cpu.c", cpu / "CMakeLists.txt", repo / KERNEL_HEADER, repo / KERNEL_SOURCE):
        if not required.is_file():
            raise PatchError(f"{required} does not exist")
    cmake_edits = [(name, CMAKE_ANCHOR, CMAKE_ANCHOR + f"        ggml-cpu/{name}\n") for name in reversed(CMAKE_SOURCES)]
    c_edits = [
        ("strata_kq_enabled(void)", HOOK_FUNCTION_ANCHOR, HOOK_DECLARATION + HOOK_FUNCTION_ANCHOR),
        ("strata_kq_rows((int) type", DOT_LOOP, DOT_LOOP_WITH_KERNEL),
        ("type == GGML_TYPE_Q2_K || type == GGML_TYPE_Q4_K", HOOK_TYPES_Q4_Q5, HOOK_TYPES_Q2_Q4_Q5),
    ]
    cmake_text = plan_edits(cpu / "CMakeLists.txt", cmake_edits)
    c_text = plan_edits(cpu / "ggml-cpu.c", c_edits)

    changed = []
    if copy_if_different(repo / KERNEL_HEADER, cpu / "strata" / "kernels" / "cpu" / "kq_avx512.hpp"):
        changed.append("strata/kernels/cpu/kq_avx512.hpp")
    if copy_if_different(repo / KERNEL_SOURCE, cpu / "strata_kq_avx512.cpp"):
        changed.append("strata_kq_avx512.cpp")
    if write_if_different(cpu / "strata_kq_rows.cpp", BRIDGE_SOURCE):
        changed.append("strata_kq_rows.cpp")
    if cmake_text is not None:
        (cpu / "CMakeLists.txt").write_text(cmake_text, encoding="utf-8")
        changed.append("CMakeLists.txt")
    if c_text is not None:
        (cpu / "ggml-cpu.c").write_text(c_text, encoding="utf-8")
        changed.append("ggml-cpu.c")
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="put the Strata Q2_K, Q4_K and Q5_K expert kernels into a llama.cpp checkout (STRATA_KQ512=0 switches them off at run time)")
    parser.add_argument("--llama", type=Path, required=True, help="the llama.cpp checkout")
    args = parser.parse_args(argv)
    try:
        changed = apply(args.llama)
    except PatchError as error:
        print(f"llama-cpp-kq512: {error}", file=sys.stderr)
        return 2
    print("llama-cpp-kq512: " + (", ".join(changed) if changed else "nothing to do, already applied"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
