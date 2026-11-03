import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import bench_table


def result(label, budget, median, **extra):
    summary = {"runs": 3, "left_out": 0, "tok_s": [median - 1, median, median + 1], "median": median, "accepted_share": 0.84, "peak_c": 71.0}
    summary.update(extra)
    return {"label": label, "budget": budget, "turbo": "off", "summary": {"code": summary}, "aborted": None, "cleanup_problems": []}


class TableTests(unittest.TestCase):
    def test_rows_are_ordered_by_budget_then_label(self):
        results = [result("b", "none", 40.0), result("a", "16G", 30.0), result("c", "12G", 20.0), result("a", "none", 41.0)]
        lines = bench_table.table(results).splitlines()
        self.assertEqual([line.split("|")[1].strip() + "/" + line.split("|")[2].strip() for line in lines[2:]], ["c/12G", "a/16G", "a/none", "b/none"])

    def test_the_numbers_are_formatted(self):
        cells = bench_table.row(result("a", "16G", 30.0))
        self.assertEqual(cells[:7], ["a", "16G", "30.0", "29.0-31.0", "84 %", "71", "off"])
        self.assertEqual(cells[7:], ["-", "-", "ok"])

    def test_the_step_breakdown_is_shown_when_the_run_has_it(self):
        counted = result("a", "16G", 30.0, step_ms={"total": 67.9}, tokens_per_step=2.7)
        self.assertEqual(bench_table.row(counted)[7:9], ["67.9", "2.7"])

    def test_an_aborted_run_and_left_out_requests_are_named(self):
        broken = result("a", "16G", 30.0, left_out=2)
        broken["aborted"] = "stalling"
        self.assertEqual(bench_table.row(broken)[9], "aborted, 2 left out")

    def test_a_run_without_numbers_gets_dashes(self):
        empty = {"label": "a", "budget": "none", "summary": {}, "aborted": "x"}
        self.assertEqual(bench_table.row(empty)[2:6], ["-", "-", "-", "-"])

    def test_files_that_are_not_results_are_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            good, bad = Path(directory) / "good.json", Path(directory) / "bad.json"
            good.write_text(json.dumps(result("a", "none", 40.0)))
            bad.write_text("not json")
            self.assertEqual([r["file"] for r in bench_table.load_results([good, bad, Path(directory) / "missing.json"])], ["good.json"])

    def test_the_command_prints_the_table(self):
        with tempfile.TemporaryDirectory() as directory:
            good = Path(directory) / "good.json"
            good.write_text(json.dumps(result("a", "none", 40.0)))
            done = subprocess.run([sys.executable, "-I", str(TOOLS / "bench_table.py"), str(good)], capture_output=True, text=True, check=False)
        self.assertEqual(done.returncode, 0)
        self.assertIn("40.0", done.stdout)


if __name__ == "__main__":
    unittest.main()
