import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import llama_cpp_kq512 as patcher

CMAKE = "add_library(ggml-cpu\n        ggml-cpu/ggml-cpu.c\n        ggml-cpu/quants.c\n        ggml-cpu/ops.cpp\n)\n"
SOURCE = "#include <stdlib.h>\n\n" + patcher.HOOK_FUNCTION_ANCHOR + "\n    int x;\n) {\n" + patcher.DOT_LOOP + "}\n"


def make_tree(root: Path, cmake: str = CMAKE, source: str = SOURCE) -> Path:
    cpu = root / "ggml" / "src" / "ggml-cpu"
    cpu.mkdir(parents=True)
    (cpu / "CMakeLists.txt").write_text(cmake, encoding="utf-8")
    (cpu / "ggml-cpu.c").write_text(source, encoding="utf-8")
    return root


class ApplyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.llama = make_tree(Path(self.directory.name))
        self.cpu = self.llama / patcher.CPU_DIR

    def test_the_first_run_changes_everything_and_the_second_changes_nothing(self):
        first = patcher.apply(self.llama)
        self.assertEqual(sorted(first), sorted(["strata/kernels/cpu/kq_avx512.hpp", "strata_kq_avx512.cpp", "strata_kq_rows.cpp", "CMakeLists.txt", "ggml-cpu.c"]))
        self.assertEqual(patcher.apply(self.llama), [])

    def test_the_kernel_sources_are_copied_byte_for_byte(self):
        patcher.apply(self.llama)
        self.assertEqual((self.cpu / "strata_kq_avx512.cpp").read_bytes(), (patcher.REPO / patcher.KERNEL_SOURCE).read_bytes())
        self.assertEqual((self.cpu / "strata/kernels/cpu/kq_avx512.hpp").read_bytes(), (patcher.REPO / patcher.KERNEL_HEADER).read_bytes())

    def test_a_changed_kernel_source_is_copied_again(self):
        patcher.apply(self.llama)
        (self.cpu / "strata_kq_avx512.cpp").write_text("stale", encoding="utf-8")
        self.assertEqual(patcher.apply(self.llama), ["strata_kq_avx512.cpp"])

    def test_the_build_file_lists_both_sources_after_the_anchor(self):
        patcher.apply(self.llama)
        text = (self.cpu / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("        ggml-cpu/quants.c\n        ggml-cpu/strata_kq_avx512.cpp\n        ggml-cpu/strata_kq_rows.cpp\n", text)

    def test_the_hook_sits_before_the_scalar_loop_and_keeps_it_as_the_fallback(self):
        patcher.apply(self.llama)
        text = (self.cpu / "ggml-cpu.c").read_text(encoding="utf-8")
        self.assertLess(text.index("strata_kq_enabled(void)"), text.index(patcher.HOOK_FUNCTION_ANCHOR))
        self.assertLess(text.index("strata_kq_rows((int) type"), text.index("vec_dot(ne00"))
        self.assertIn("STRATA_KQ512", text)

    def test_a_missing_anchor_is_an_error_that_names_the_place_and_leaves_the_tree_as_it_was(self):
        other = make_tree(Path(self.directory.name) / "other", source="int main(void) { return 0; }\n")
        before = sorted(path.relative_to(other) for path in other.rglob("*") if path.is_file())
        cmake_before = (other / patcher.CPU_DIR / "CMakeLists.txt").read_text(encoding="utf-8")
        with self.assertRaisesRegex(patcher.PatchError, "mul_mat_id_one_chunk"):
            patcher.apply(other)
        self.assertEqual(sorted(path.relative_to(other) for path in other.rglob("*") if path.is_file()), before)
        self.assertEqual((other / patcher.CPU_DIR / "CMakeLists.txt").read_text(encoding="utf-8"), cmake_before)

    def test_a_half_applied_tree_is_completed_by_the_next_run(self):
        patcher.apply(self.llama)
        cmake = self.cpu / "CMakeLists.txt"
        cmake.write_text(cmake.read_text(encoding="utf-8").replace("        ggml-cpu/strata_kq_avx512.cpp\n", ""), encoding="utf-8")
        self.assertEqual(patcher.apply(self.llama), ["CMakeLists.txt"])
        self.assertIn("strata_kq_avx512.cpp", cmake.read_text(encoding="utf-8"))
        source = self.cpu / "ggml-cpu.c"
        source.write_text(source.read_text(encoding="utf-8").replace(patcher.DOT_LOOP_WITH_KERNEL, patcher.DOT_LOOP), encoding="utf-8")
        self.assertEqual(patcher.apply(self.llama), ["ggml-cpu.c"])
        self.assertIn("strata_kq_rows((int) type", source.read_text(encoding="utf-8"))

    def test_a_tree_without_the_cpu_backend_is_an_error(self):
        with self.assertRaisesRegex(patcher.PatchError, "does not exist"):
            patcher.apply(Path(self.directory.name) / "empty")

    def test_the_command_line_reports_the_error_code(self):
        self.assertEqual(patcher.main(["--llama", str(Path(self.directory.name) / "empty")]), 2)
        self.assertEqual(patcher.main(["--llama", str(self.llama)]), 0)


if __name__ == "__main__":
    unittest.main()
