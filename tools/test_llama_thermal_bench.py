import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import llama_thermal_bench as bench

TIMINGS = {"prompt_n": 40, "prompt_ms": 400.0, "prompt_per_second": 100.0, "predicted_n": 900, "predicted_ms": 45000.0, "predicted_per_second": 20.0}


def arguments(*extra):
    return bench.parse_args(["--server", "/bin/llama-server", "--model", "/models/m.gguf", *extra])


class CommandTests(unittest.TestCase):
    def test_the_server_command_keeps_the_experts_on_the_cpu_by_default(self):
        command = bench.server_command(arguments())
        self.assertEqual(command[:3], ["nice", "-n", "5"])
        self.assertEqual(command[command.index("--n-cpu-moe") + 1], "99")
        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        self.assertIn("--parallel", command)

    def test_the_placement_threads_and_extra_arguments_are_passed_on(self):
        command = bench.server_command(arguments("--n-cpu-moe", "31", "--threads", "5", "--ctx", "4096", "--server-arg=--no-mmap", "--server-arg=--mlock"))
        self.assertEqual(command[command.index("--n-cpu-moe") + 1], "31")
        self.assertEqual(command[command.index("-t") + 1], "5")
        self.assertEqual(command[command.index("-c") + 1], "4096")
        self.assertEqual(command[-2:], ["--no-mmap", "--mlock"])

    def test_bad_counts_are_refused(self):
        for bad in (["--tokens", "0"], ["--requests", "0"], ["--warmup", "-1"]):
            with self.subTest(bad=bad), self.assertRaises(SystemExit), mock.patch("sys.stderr"):
                arguments(*bad)

    def test_the_request_asks_for_a_fixed_number_of_tokens_unless_told_otherwise(self):
        body = bench.request_body("hello", 900, True)
        self.assertEqual((body["max_tokens"], body["temperature"], body["ignore_eos"], body["stream"]), (900, 0, True, False))
        self.assertFalse(bench.request_body("hello", 900, False)["ignore_eos"])


class TimingsTests(unittest.TestCase):
    def test_the_figures_of_the_timings_block_are_taken_over(self):
        self.assertEqual(bench.parse_timings({"timings": TIMINGS})["predicted_per_second"], 20.0)

    def test_a_missing_or_incomplete_block_is_an_error(self):
        for payload in ({}, {"timings": {"prompt_n": 1}}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                bench.parse_timings(payload)

    def test_a_block_that_is_not_an_object_is_a_type_error(self):
        with self.assertRaises(TypeError):
            bench.parse_timings({"timings": "x"})


class SummaryTests(unittest.TestCase):
    def rows(self, speeds):
        return [{"predicted_per_second": speed, "prompt_per_second": 100.0} for speed in speeds]

    def test_median_range_and_drift(self):
        summary = bench.summarize(self.rows([20, 22, 21, 19, 18, 17, 16, 15]), [])
        self.assertEqual((summary["decode_median"], summary["decode_min"], summary["decode_max"]), (18.5, 15, 22))
        self.assertEqual(summary["decode_first"], 20.0)
        self.assertEqual(summary["decode_last"], 16.0)

    def test_the_peaks_and_the_mean_clock_come_from_the_timeline(self):
        timeline = [{"pkg": 70, "gpu": 60, "mhz": 2100}, {"pkg": 79, "gpu": 66, "mhz": 2000}, {"pkg": None, "gpu": None, "mhz": None}]
        summary = bench.summarize(self.rows([20]), timeline)
        self.assertEqual((summary["pkg_peak"], summary["gpu_peak"], summary["mhz_mean"]), (79, 66, 2050))

    def test_the_drift_windows_never_overlap_unless_there_is_one_request(self):
        self.assertEqual([bench.drift_windows(count) for count in (1, 2, 3, 4, 7, 8, 12)],
                         [(1, 1), (1, 1), (2, 1), (2, 2), (4, 3), (5, 3), (5, 3)])
        summary = bench.summarize(self.rows([10, 20, 30]), [])
        self.assertEqual((summary["decode_first"], summary["decode_first_n"], summary["decode_last"], summary["decode_last_n"]), (15.0, 2, 30.0, 1))

    def test_nothing_measured_gives_no_speed_and_no_peaks(self):
        summary = bench.summarize([], [])
        self.assertEqual(summary["requests"], 0)
        self.assertNotIn("decode_median", summary)
        self.assertIsNone(summary["pkg_peak"])

    def test_the_mean_clock_reads_the_cpufreq_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            for number, khz in enumerate(("2100000", "2300000")):
                folder = Path(tmp) / f"cpu{number}" / "cpufreq"
                folder.mkdir(parents=True)
                (folder / "scaling_cur_freq").write_text(khz)
            self.assertEqual(bench.mean_cpu_mhz(f"{tmp}/cpu[0-9]*/cpufreq/scaling_cur_freq"), 2200)
            self.assertIsNone(bench.mean_cpu_mhz(f"{tmp}/nothing/*"))


class ServerTests(unittest.TestCase):
    def test_a_request_goes_to_the_chat_endpoint_and_the_answer_comes_back(self):
        seen = {}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                seen["path"] = self.path
                seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                data = json.dumps({"timings": TIMINGS}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        payload = bench.post_chat(server.server_address[1], bench.request_body("hi", 5, True), timeout=10)
        self.assertEqual(bench.parse_timings(payload)["predicted_n"], 900)
        self.assertEqual(seen["path"], "/v1/chat/completions")
        self.assertEqual(seen["body"]["max_tokens"], 5)

    def test_readiness_stops_waiting_when_the_server_ended(self):
        process = mock.Mock()
        process.poll.return_value = 1
        self.assertFalse(bench.wait_ready(1, process, timeout=5, sleep=lambda seconds: None))

    def test_readiness_gives_up_at_the_timeout(self):
        process = mock.Mock()
        process.poll.return_value = None
        ticks = iter(range(0, 100, 10))
        with mock.patch("urllib.request.urlopen", side_effect=OSError("refused")):
            self.assertFalse(bench.wait_ready(1, process, timeout=25, sleep=lambda seconds: None, clock=lambda: next(ticks)))

    def test_a_healthy_server_is_ready_at_once(self):
        process = mock.Mock()
        process.poll.return_value = None
        response = mock.MagicMock()
        response.__enter__.return_value.status = 200
        with mock.patch("urllib.request.urlopen", return_value=response):
            self.assertTrue(bench.wait_ready(1, process, timeout=5, sleep=lambda seconds: None))


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.server = Path(self.tmp.name) / "llama-server"
        self.server.write_text("")
        self.model = Path(self.tmp.name) / "m.gguf"
        self.model.write_bytes(b"x" * 1000)

    def args(self, *extra):
        return bench.parse_args(["--server", str(self.server), "--model", str(self.model), "--port", "1", *extra])

    def test_a_good_machine_has_no_problems(self):
        with mock.patch.object(bench.quant_bench, "port_in_use", return_value=False):
            self.assertEqual(bench.problems(self.args(), 0.3, 32 << 30), [])

    def test_missing_files_are_named(self):
        self.model.unlink()
        self.assertEqual(len(bench.problems(self.args(), 0.3, 32 << 30)), 1)
        self.server.unlink()
        self.assertEqual(len(bench.problems(self.args(), 0.3, 32 << 30)), 2)

    def test_too_little_memory_a_busy_machine_and_a_taken_port_are_reported_unless_forced(self):
        with mock.patch.object(bench.quant_bench, "port_in_use", return_value=True):
            found = bench.problems(self.args(), 3.0, 0)
            self.assertEqual(len(found), 3)
            forced = bench.problems(self.args("--force"), 3.0, 0)
        self.assertEqual(forced, ["port 1 is already in use"])


class CleanupTests(unittest.TestCase):
    def test_every_step_runs_and_the_first_error_is_raised_at_the_end(self):
        ran = []

        def failing(name):
            def step():
                ran.append(name)
                raise RuntimeError(name)
            return step

        with self.assertRaisesRegex(RuntimeError, "one"):
            bench.run_cleanup([lambda: ran.append("start"), failing("one"), failing("two"), lambda: ran.append("end")])
        self.assertEqual(ran, ["start", "one", "two", "end"])

    def test_an_interrupt_in_a_step_does_not_skip_the_later_steps(self):
        ran = []

        def interrupted():
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            bench.run_cleanup([interrupted, lambda: ran.append("server stopped")])
        self.assertEqual(ran, ["server stopped"])

    def test_without_errors_nothing_is_raised(self):
        bench.run_cleanup([lambda: None, lambda: None])


class SamplerTests(unittest.TestCase):
    def test_the_sampler_records_temperatures_and_the_governor_state(self):
        governor = mock.Mock()
        governor.status.return_value = {"duty": 0.2, "throttled_s": 3.5}
        with mock.patch.object(bench.quant_bench, "package_temperature", return_value=75.0), \
                mock.patch.object(bench.quant_bench, "gpu_temperature", return_value=61.0), mock.patch.object(bench, "mean_cpu_mhz", return_value=2100):
            sampler = bench.Sampler(governor, period=0.05)
            sampler.start()
            threading.Event().wait(0.3)
            timeline = sampler.stop()
        self.assertGreaterEqual(len(timeline), 2)
        self.assertEqual({key: timeline[0][key] for key in ("pkg", "gpu", "mhz", "duty", "throttled_s")},
                         {"pkg": 75.0, "gpu": 61.0, "mhz": 2100, "duty": 0.2, "throttled_s": 3.5})

    def test_the_sampler_works_without_a_governor(self):
        with mock.patch.object(bench.quant_bench, "package_temperature", return_value=50.0), \
                mock.patch.object(bench.quant_bench, "gpu_temperature", return_value=None), mock.patch.object(bench, "mean_cpu_mhz", return_value=None):
            sampler = bench.Sampler(None, period=0.05)
            sampler.start()
            threading.Event().wait(0.15)
            timeline = sampler.stop()
        self.assertIsNone(timeline[0]["duty"])


if __name__ == "__main__":
    unittest.main()
