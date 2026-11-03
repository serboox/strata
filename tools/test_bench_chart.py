import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from xml.etree import ElementTree

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import bench_chart


def result(label, budget, measured, warmup=()):
    rows = [{"warmup": True, "tok_s": value, "stat_lines": 1} for value in warmup] + [{"warmup": False, "tok_s": value, "stat_lines": 1} for value in measured]
    return {"label": label, "budget": budget, "rows": rows}


class SpeedTests(unittest.TestCase):
    def test_median_burst_and_measured_speeds(self):
        item = bench_chart.speeds(result("a", "16G", [20.0, 17.0, 25.0], warmup=[33.0, 35.0]))
        self.assertEqual((item["median"], item["burst"], item["measured"]), (20.0, 35.0, [20.0, 17.0, 25.0]))

    def test_an_even_number_of_requests_takes_the_middle_pair(self):
        self.assertEqual(bench_chart.speeds(result("a", "16G", [10.0, 20.0, 30.0, 40.0]))["median"], 25.0)

    def test_a_row_with_a_missing_or_doubled_log_line_is_not_measured(self):
        data = result("a", "16G", [20.0])
        data["rows"].append({"warmup": False, "tok_s": 99.0, "stat_lines": 2})
        self.assertEqual(bench_chart.speeds(data)["measured"], [20.0])

    def test_no_rows_means_no_median(self):
        self.assertIsNone(bench_chart.speeds({"rows": []})["median"])


class DrawingTests(unittest.TestCase):
    def test_the_speed_chart_is_valid_svg_with_the_numbers_and_labels(self):
        results = [result("q4kxl_control", "16G", [17.0, 18.0, 17.5], [22.0]), result("iq4xs_kernels_on", "16G", [20.0, 25.0, 17.0], [35.0]),
                   result("iq4xs_kernels_on", "none", [40.0, 41.0, 39.0], [44.0])]
        svg = bench_chart.speed_chart(results, "Title <b>", "note & more")
        root = ElementTree.fromstring(svg)
        self.assertTrue(root.tag.endswith("svg"))
        self.assertIn("budget 16G", svg)
        self.assertIn("budget none", svg)
        self.assertIn("40.0", svg)
        self.assertIn("Title &lt;b&gt;", svg)

    def test_any_labels_can_be_drawn_as_bars_in_the_given_order(self):
        results = [result("tune_a", "none", [20.0, 21.0, 19.0]), result("tune_b", "none", [25.0, 26.0, 24.0])]
        svg = bench_chart.speed_chart(results, "Tuning", "note", bench_chart.series_for(["tune_b", "tune_a", "q4kxl_control"]))
        ElementTree.fromstring(svg)
        self.assertLess(svg.index("tune_b"), svg.index("tune_a"))
        self.assertIn("Q4_K_XL (control)", svg)

    def test_many_series_still_make_valid_svg_with_every_legend_entry(self):
        labels = ["tune_default", "tune_workers2", "tune_spin50", "tune_spin0", "tune_mtp3", "tune_control", "tune_default_again"]
        results = [result(label, "none", [20.0, 21.0, 19.0]) for label in labels]
        svg = bench_chart.speed_chart(results, "Tuning", "note", bench_chart.series_for(labels))
        ElementTree.fromstring(svg)
        for name in ("default (3 workers)", "2 workers", "no spin", "3 drafts", "default again", "measured request", "first request, cool"):
            self.assertIn(name, svg)

    def test_the_thermal_chart_is_valid_svg_and_marks_the_limit(self):
        rows = [("21:00:00", 50.0, 3300.0, 45.0), ("21:00:02", 82.0, 3300.0, 60.0), ("21:00:04", 78.0, 3200.0, 61.0)]
        svg = bench_chart.thermal_chart(rows, "Temperatures")
        ElementTree.fromstring(svg)
        self.assertIn("limit 80 C", svg)

    def test_the_thermal_log_is_read_and_bad_lines_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.log"
            path.write_text("time cpu_c cpu_mhz gpu_c\n21:00:00 50 3300 45 6.1\nbroken\n21:00:02 82 3300 60 15.0\n")
            self.assertEqual(bench_chart.read_thermal(path), [("21:00:00", 50.0, 3300.0, 45.0), ("21:00:02", 82.0, 3300.0, 60.0)])

    def test_the_size_comes_from_the_view_box(self):
        self.assertEqual(bench_chart.svg_size('<svg viewBox="0 0 960 480" xmlns="x">'), (960, 480))
        self.assertEqual(bench_chart.svg_size("<svg>"), (960, 460))

    def test_a_missing_chrome_gives_false(self):
        with mock.patch.object(bench_chart.subprocess, "run", side_effect=OSError("no chrome")):
            self.assertFalse(bench_chart.to_png(Path("/tmp/a.svg"), Path("/tmp/a.png")))


if __name__ == "__main__":
    unittest.main()
