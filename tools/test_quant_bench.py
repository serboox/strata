import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import quant_bench

GIB = 1 << 30
REAL_LINE = ("strata serve: session 0: prompt 9220 tokens = 8704 reused + 516 read in 3445 ms (149.8 tok/s), 227 generated in 6258 ms "
             "(36.3 tok/s), drafts accepted 148 of 156 (stop)")
NO_DRAFTS = "strata serve: session 3: prompt 100 tokens = 0 reused + 100 read in 900 ms (111.1 tok/s), 50 generated in 2000 ms (25.0 tok/s) (length)"
LOG_LINE = "strata serve: session 0: prompt 10 tokens = 0 reused + 10 read in 100 ms (100 tok/s), 100 generated in {ms} ms (x tok/s), drafts accepted 8 of 10 (length)\n"


def scratch(case: unittest.TestCase) -> Path:
    path = Path(tempfile.mkdtemp())
    case.addCleanup(lambda: subprocess.run(["rm", "-rf", str(path)], check=False))
    return path


def make_args(model: Path, **overrides) -> argparse.Namespace:
    values = {"model": model, "engine_root": quant_bench.ENGINE_ROOT, "budget": "16G", "port": 18124, "mtp": None, "max_load": 100.0, "force": False,
              "prompts": ["code"], "warmup": 1, "repeats": 2, "cool": 55.0, "cool_timeout": 1.0, "ready_timeout": 1.0, "max_tokens": 10, "label": "toy",
              "max_stall": 1000.0, "request_timeout": 5.0, "log_timeout": 0.0, "max_temp": 0.0, "allow_turbo": False, "exe": None, "env": {}, "save_log": None, "evict": [], "workers": None, "cpu_max_mhz": 0.0, "prewarm": False, "engine_args": [], "out": Path("/nonexistent/result.json")}
    values.update(overrides)
    return argparse.Namespace(**values)


class ParseTests(unittest.TestCase):
    def test_a_real_server_line_is_read(self):
        row = quant_bench.parse_session_line(REAL_LINE)
        self.assertEqual((row["prompt"], row["reused"], row["read"], row["prompt_ms"]), (9220, 8704, 516, 3445))
        self.assertEqual((row["generated"], row["decode_ms"], row["tok_s"]), (227, 6258, 36.27))
        self.assertEqual((row["accepted"], row["offered"]), (148, 156))

    def test_a_line_without_drafts_has_zero_of_zero(self):
        row = quant_bench.parse_session_line(NO_DRAFTS)
        self.assertEqual((row["tok_s"], row["accepted"], row["offered"]), (25.0, 0, 0))

    def test_other_lines_are_ignored(self):
        self.assertIsNone(quant_bench.parse_session_line("strata-q36: hybrid experts: 0 of 8993080 batch entries on the GPU"))
        self.assertIsNone(quant_bench.parse_session_line(""))

    def test_a_zero_duration_does_not_divide_by_zero(self):
        row = quant_bench.parse_session_line("session 1: prompt 5 tokens = 0 reused + 5 read in 10 ms (1 tok/s), 0 generated in 0 ms")
        self.assertEqual(row["tok_s"], 0.0)


COUNTER_LINE = REAL_LINE + " [spec 84 steps: draft 600 verify 4800 commit 300 ms; plain 2 steps 70 ms]"


class StepCounterTests(unittest.TestCase):
    def test_the_engine_counters_are_read(self):
        row = quant_bench.parse_session_line(COUNTER_LINE)
        self.assertEqual((row["spec_steps"], row["draft_ms"], row["verify_ms"], row["commit_ms"], row["plain_steps"], row["plain_ms"]),
                         (84, 600, 4800, 300, 2, 70))

    def test_a_line_without_counters_has_none(self):
        self.assertNotIn("spec_steps", quant_bench.parse_session_line(REAL_LINE))

    def test_the_summary_gives_milliseconds_per_step_and_tokens_per_step(self):
        row = quant_bench.parse_session_line(COUNTER_LINE)
        row.update({"prompt_name": "code", "warmup": False, "stat_lines": 1, "started_cool": True})
        entry = quant_bench.summarize([row])["code"]
        self.assertEqual(entry["step_ms"], {"draft": 7.14, "verify": 57.14, "commit": 3.57, "total": 67.85})
        self.assertEqual(entry["tokens_per_step"], 2.7)


class StatsTests(unittest.TestCase):
    def test_the_line_is_waited_for_and_found(self):
        log = scratch(self) / "server.log"
        log.write_text("old line\n")
        offset = log.stat().st_size
        polls = []

        def sleep(seconds):
            polls.append(seconds)
            if len(polls) == 3:
                with open(log, "a") as handle:
                    handle.write(LOG_LINE.format(ms=2500))

        ticks = iter(range(1000))
        row, count = quant_bench.collect_stats(log, offset, 30, sleep=sleep, clock=lambda: next(ticks))
        self.assertEqual((count, row["tok_s"]), (1, 40.0))
        self.assertEqual(len(polls), 3)

    def test_a_line_that_never_comes_gives_none_after_the_timeout(self):
        log = scratch(self) / "server.log"
        ticks = iter(range(0, 1000, 10))
        self.assertEqual(quant_bench.collect_stats(log, 0, 30, sleep=lambda s: None, clock=lambda: next(ticks)), (None, 0))

    def test_two_lines_are_reported_as_two(self):
        log = scratch(self) / "server.log"
        log.write_text(LOG_LINE.format(ms=2500) + LOG_LINE.format(ms=2000))
        row, count = quant_bench.collect_stats(log, 0, 0, sleep=lambda s: None)
        self.assertEqual((count, row["tok_s"]), (2, 50.0))


class SummaryTests(unittest.TestCase):
    @staticmethod
    def row(prompt, index, warmup, tok_s=None, accepted=80, offered=100, lines=1, cool=True):
        row = {"prompt_name": prompt, "index": index, "warmup": warmup, "stat_lines": lines, "started_cool": cool, "accepted": accepted, "offered": offered}
        if tok_s is not None:
            row["tok_s"] = tok_s
        return row

    def test_warmups_are_left_out_and_the_median_is_taken(self):
        rows = [self.row("code", 0, True, 20.0), self.row("code", 1, True, 30.0), self.row("code", 2, False, 34.0),
                self.row("code", 3, False, 36.0, 90), self.row("code", 4, False, 35.0, 70)]
        entry = quant_bench.summarize(rows)["code"]
        self.assertEqual((entry["runs"], entry["left_out"]), (3, 0))
        self.assertEqual((entry["median"], entry["min"], entry["max"]), (35.0, 34.0, 36.0))
        self.assertEqual(entry["accepted_share"], 0.8)

    def test_a_missing_or_doubled_line_and_a_hot_start_are_left_out_and_counted(self):
        rows = [self.row("code", 2, False, None, lines=0), self.row("code", 3, False, 30.0, lines=2), self.row("code", 4, False, 31.0, cool=False),
                self.row("code", 5, False, 33.0), self.row("code", 6, False, 35.0, cool=None)]
        entry = quant_bench.summarize(rows)["code"]
        self.assertEqual((entry["runs"], entry["left_out"]), (2, 3))
        self.assertEqual(entry["tok_s"], [33.0, 35.0])

    def test_no_drafts_offered_gives_no_share(self):
        self.assertIsNone(quant_bench.summarize([self.row("text", 2, False, 40.0, 0, 0)])["text"]["accepted_share"])

    def test_a_prompt_with_nothing_usable_still_reports_what_was_left_out(self):
        entry = quant_bench.summarize([self.row("code", 2, False, None, lines=0)])["code"]
        self.assertEqual(entry, {"runs": 0, "left_out": 1})


class TurboTests(unittest.TestCase):
    def test_the_switch_is_read(self):
        path = scratch(self) / "no_turbo"
        path.write_text("1\n")
        self.assertEqual(quant_bench.turbo_state(path), "off")
        path.write_text("0\n")
        self.assertEqual(quant_bench.turbo_state(path), "on")
        self.assertEqual(quant_bench.turbo_state(path.parent / "missing"), "unknown")


class EnvTests(unittest.TestCase):
    def test_pairs_are_split_at_the_first_equals_sign(self):
        self.assertEqual(quant_bench.parse_env(["A=1", "B=x=y", "C="]), {"A": "1", "B": "x=y", "C": ""})

    def test_an_item_without_a_key_or_an_equals_sign_stops_the_run(self):
        for item in ("A", "=1"):
            with self.assertRaises(SystemExit):
                quant_bench.parse_env([item])


class AnswerTests(unittest.TestCase):
    def test_the_text_of_the_answer_is_read_from_the_reply(self):
        reply = mock.MagicMock()
        reply.__enter__.return_value.read.return_value = json.dumps({"choices": [{"message": {"content": "hello"}}]}).encode()
        with mock.patch.object(quant_bench.urllib.request, "urlopen", return_value=reply):
            seconds, text = quant_bench.ask(1, "m", "p", 1)
        self.assertEqual(text, "hello")
        self.assertGreaterEqual(seconds, 0.0)

    def test_the_reasoning_text_is_used_when_the_content_is_empty(self):
        reply = mock.MagicMock()
        reply.__enter__.return_value.read.return_value = json.dumps({"choices": [{"message": {"content": "", "reasoning_content": "thinking"}}]}).encode()
        with mock.patch.object(quant_bench.urllib.request, "urlopen", return_value=reply):
            self.assertEqual(quant_bench.ask(1, "m", "p", 1)[1], "thinking")

    def test_a_reply_that_is_not_a_chat_completion_gives_an_empty_text(self):
        reply = mock.MagicMock()
        reply.__enter__.return_value.read.return_value = b"not json"
        with mock.patch.object(quant_bench.urllib.request, "urlopen", return_value=reply):
            self.assertEqual(quant_bench.ask(1, "m", "p", 1)[1], "")


class StallTests(unittest.TestCase):
    TEXT = "some avg10=1.50 avg60=0.40 avg300=0.10 total=12345\nfull avg10=7.25 avg60=2.00 avg300=0.50 total=6789\n"

    def test_the_full_stall_share_is_read(self):
        self.assertEqual(quant_bench.memory_stall(self.TEXT), 7.25)

    def test_a_file_without_a_full_line_means_no_stall(self):
        self.assertEqual(quant_bench.memory_stall("some avg10=3.00 avg60=0 avg300=0 total=1\n"), 0.0)
        self.assertEqual(quant_bench.memory_stall(""), 0.0)

    def test_an_unreadable_pressure_file_means_no_stall(self):
        self.assertEqual(quant_bench.current_stall(scratch(self) / "missing"), 0.0)


class GateTests(unittest.TestCase):
    def test_a_quiet_machine_with_room_has_no_problems(self):
        self.assertEqual(quant_bench.machine_problems(17 * GIB, "none", 0.4, 30 * GIB, 1.5), [])

    def test_load_and_memory_problems_are_named(self):
        problems = quant_bench.machine_problems(17 * GIB, "none", 2.0, 8 * GIB, 1.5)
        self.assertEqual(len(problems), 2)
        self.assertIn("load average is 2.00", problems[0])
        self.assertIn("8.0 GiB of memory", problems[1])

    def test_a_small_budget_does_not_need_free_memory_for_the_whole_file(self):
        self.assertEqual(quant_bench.machine_problems(17 * GIB, "16G", 0.4, 4 * GIB, 1.5), [])

    def test_the_cool_down_returns_at_once_when_cool_and_says_unknown_without_a_sensor(self):
        cool = quant_bench.wait_until_cool(55, 10, read=lambda: 40.0, sleep=lambda s: self.fail("slept"))
        self.assertEqual((cool[0], cool[2]), (40.0, True))
        unknown = quant_bench.wait_until_cool(55, 10, read=lambda: None, sleep=lambda s: self.fail("slept"))
        self.assertEqual((unknown[0], unknown[2]), (None, None))

    def test_the_cool_down_waits_then_gives_up_at_the_timeout(self):
        clock = iter(range(0, 100, 5))
        temperatures = iter([70.0, 62.0, 54.0])
        result = quant_bench.wait_until_cool(55, 100, read=lambda: next(temperatures), sleep=lambda s: None, clock=lambda: next(clock))
        self.assertEqual((result[0], result[2]), (54.0, True))
        clock = iter(range(0, 100, 5))
        result = quant_bench.wait_until_cool(55, 10, read=lambda: 90.0, sleep=lambda s: None, clock=lambda: next(clock))
        self.assertFalse(result[2])

    def test_the_cool_down_runs_the_check_on_every_wait(self):
        calls = []
        clock = iter(range(0, 100, 5))
        quant_bench.wait_until_cool(55, 12, read=lambda: 90.0, sleep=lambda s: None, clock=lambda: next(clock), check=lambda: calls.append(1))
        self.assertGreaterEqual(len(calls), 2)

    def test_the_package_sensor_is_found_by_its_type(self):
        root = scratch(self)
        for name, kind, temp in (("thermal_zone0", "acpitz", "60000"), ("thermal_zone1", "x86_pkg_temp", "48500")):
            (root / name).mkdir()
            (root / name / "type").write_text(kind + "\n")
            (root / name / "temp").write_text(temp + "\n")
        self.assertEqual(quant_bench.package_temperature(root), 48.5)
        self.assertIsNone(quant_bench.package_temperature(root / "missing"))

    def test_a_busy_port_is_found(self):
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            port = server.getsockname()[1]
            self.assertTrue(quant_bench.port_in_use(port))
        self.assertFalse(quant_bench.port_in_use(port))


class StopTests(unittest.TestCase):
    @staticmethod
    def result(code=0, stderr=""):
        return subprocess.CompletedProcess([], code, stdout="", stderr=stderr)

    def test_a_clean_stop_reports_nothing(self):
        process = mock.Mock()
        with mock.patch.object(quant_bench.subprocess, "run", return_value=self.result()), mock.patch.object(quant_bench, "scope_state", return_value="inactive"):
            self.assertEqual(quant_bench.stop_server("u", process), [])
        process.wait.assert_called_once()

    def test_a_failed_systemctl_and_a_scope_that_stays_are_reported(self):
        with mock.patch.object(quant_bench.subprocess, "run", return_value=self.result(5, "Failed to stop")), \
                mock.patch.object(quant_bench, "scope_state", return_value="active"), mock.patch.object(quant_bench.time, "sleep"):
            problems = quant_bench.stop_server("u", None)
        self.assertTrue(any("systemctl stop" in problem for problem in problems))
        self.assertIn("the scope is still active", problems)

    def test_a_scope_that_is_not_loaded_is_not_a_problem(self):
        with mock.patch.object(quant_bench.subprocess, "run", return_value=self.result(5, "Unit u.scope not loaded.")), \
                mock.patch.object(quant_bench, "scope_state", return_value="inactive"):
            self.assertEqual(quant_bench.stop_server("u", None), [])

    def test_a_systemd_run_that_will_not_exit_is_killed_by_group(self):
        process = mock.Mock(pid=4242)
        process.wait.side_effect = [subprocess.TimeoutExpired("x", 60), None]
        with mock.patch.object(quant_bench.subprocess, "run", return_value=self.result()), mock.patch.object(quant_bench, "scope_state", return_value="inactive"), \
                mock.patch.object(quant_bench.os, "killpg") as killpg:
            self.assertEqual(quant_bench.stop_server("u", process), [])
        killpg.assert_called_once_with(4242, signal.SIGKILL)

    def test_a_systemctl_that_cannot_run_is_reported_not_raised(self):
        with mock.patch.object(quant_bench.subprocess, "run", side_effect=OSError("no systemctl")), mock.patch.object(quant_bench, "scope_state", return_value="unknown"):
            problems = quant_bench.stop_server("u", None)
        self.assertEqual(len(problems), 1)
        self.assertIn("no systemctl", problems[0])


class CpuCapGuardTests(unittest.TestCase):
    def test_the_guard_runs_with_the_python_of_the_bench(self):
        with mock.patch.object(quant_bench.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout="", stderr="")) as run:
            self.assertEqual(quant_bench.put_cpu_cap_back(), [])
        command = run.call_args.args[0]
        self.assertEqual(command[0], sys.executable)
        self.assertTrue(command[1].endswith("tools/cpucap"))
        self.assertEqual(command[2], "guard")

    def test_a_failing_guard_is_reported_as_a_cleanup_problem(self):
        failed = subprocess.CompletedProcess([], 1, stdout="", stderr="2 limits could not be put back")
        with mock.patch.object(quant_bench.subprocess, "run", return_value=failed):
            self.assertEqual(quant_bench.put_cpu_cap_back(), ["cpucap guard: 2 limits could not be put back"])

    def test_a_guard_that_cannot_start_is_reported_and_does_not_raise(self):
        with mock.patch.object(quant_bench.subprocess, "run", side_effect=OSError("no such file")):
            self.assertEqual(quant_bench.put_cpu_cap_back(), ["cpucap guard: no such file"])


class SignalTests(unittest.TestCase):
    def test_sigterm_becomes_an_exit_that_lets_cleanup_run_and_the_old_handlers_come_back(self):
        before = signal.getsignal(signal.SIGTERM)
        previous = quant_bench.install_stop_handlers()
        try:
            with self.assertRaises(SystemExit) as raised:
                os.kill(os.getpid(), signal.SIGTERM)
                for _ in range(100):
                    pass
            self.assertIn("signal", str(raised.exception))
        finally:
            quant_bench.restore_handlers(previous)
        self.assertEqual(signal.getsignal(signal.SIGTERM), before)


class OtherStallTests(unittest.TestCase):
    def tree(self, groups):
        root = scratch(self)
        for path, text in groups.items():
            directory = root / path
            directory.mkdir(parents=True)
            (directory / "memory.pressure").write_text(text)
        return root

    @staticmethod
    def pressure(some: float, full: float = 0.0) -> str:
        return f"some avg10={some:.2f} avg60=0.00 avg300=0.00 total=0\nfull avg10={full:.2f} avg60=0.00 avg300=0.00 total=0\n"

    def test_the_worst_other_program_decides_and_its_some_figure_is_used(self):
        root = self.tree({"app.slice/a.scope": self.pressure(3.0, 1.0), "app.slice/b.scope": self.pressure(7.5), "session.slice/c": self.pressure(2.0)})
        self.assertEqual(quant_bench.other_stall("quantbench-1", root, fallback=lambda: 99.0), 7.5)

    def test_a_service_directly_under_the_user_manager_counts_too(self):
        root = self.tree({"app.slice/a.scope": self.pressure(1.0), "dbus.service": self.pressure(9.0)})
        self.assertEqual(quant_bench.other_stall("quantbench-1", root, fallback=lambda: 99.0), 9.0)

    def test_the_servers_own_scope_is_left_out(self):
        root = self.tree({"app.slice/quantbench-1.scope": self.pressure(40.0), "app.slice/other.scope": self.pressure(1.0)})
        self.assertEqual(quant_bench.other_stall("quantbench-1", root, fallback=lambda: 99.0), 1.0)

    def test_without_a_readable_cgroup_tree_the_system_wide_figure_stands_in(self):
        self.assertEqual(quant_bench.other_stall("quantbench-1", Path("/nonexistent-root"), fallback=lambda: 12.5), 12.5)

    def test_plain_files_are_skipped(self):
        root = scratch(self)
        (root / "app.slice").mkdir()
        (root / "app.slice" / "memory.pressure").write_text(self.pressure(50.0))
        (root / "cgroup.procs").write_text("1\n")
        self.assertEqual(quant_bench.other_stall("quantbench-1", root, fallback=lambda: 3.0), 3.0)


class AskTests(unittest.TestCase):
    def test_an_answer_is_returned(self):
        self.assertEqual(quant_bench.ask_watched(1, "m", "p", 1, 5, lambda: None, poll=0.01, send=lambda *a: 3.5), 3.5)

    def test_an_error_of_the_request_is_raised(self):
        def send(*arguments):
            raise OSError("refused")
        with self.assertRaisesRegex(OSError, "refused"):
            quant_bench.ask_watched(1, "m", "p", 1, 5, lambda: None, poll=0.01, send=send)

    def test_the_check_can_abandon_a_request_that_is_still_running(self):
        release = threading.Event()
        self.addCleanup(release.set)

        def check():
            raise SystemExit("someone needs the memory")
        with self.assertRaisesRegex(SystemExit, "someone needs"):
            quant_bench.ask_watched(1, "m", "p", 1, 5, check, poll=0.01, send=lambda *a: release.wait(10))

    def test_a_request_past_its_deadline_is_abandoned(self):
        release = threading.Event()
        self.addCleanup(release.set)
        with self.assertRaisesRegex(SystemExit, "no answer within"):
            quant_bench.ask_watched(1, "m", "p", 1, 0.05, lambda: None, poll=0.01, send=lambda *a: release.wait(10))


class EvictTests(unittest.TestCase):
    def test_each_file_is_dropped_from_the_cache(self):
        directory = scratch(self)
        first, second = directory / "a.gguf", directory / "b.gguf"
        first.write_bytes(b"x")
        second.write_bytes(b"y")
        with mock.patch.object(quant_bench.os, "posix_fadvise") as advise:
            self.assertEqual(quant_bench.evict_page_cache([first, second]), [])
        self.assertEqual(advise.call_count, 2)
        self.assertEqual(advise.call_args.args[1:], (0, 0, quant_bench.os.POSIX_FADV_DONTNEED))

    def test_a_file_that_cannot_be_opened_or_advised_is_reported_not_raised(self):
        directory = scratch(self)
        present = directory / "a.gguf"
        present.write_bytes(b"x")
        with mock.patch.object(quant_bench.os, "posix_fadvise", side_effect=OSError("no")):
            self.assertEqual(quant_bench.evict_page_cache([directory / "missing.gguf", present]), [str(directory / "missing.gguf"), str(present)])


class PrewarmTests(unittest.TestCase):
    def test_the_whole_file_is_read(self):
        path = scratch(self) / "m.gguf"
        path.write_bytes(b"x" * 100_000)
        self.assertEqual(quant_bench.prewarm_page_cache(path, chunk=4096), 100_000)

    def test_an_empty_file_reads_zero_bytes(self):
        path = scratch(self) / "empty.gguf"
        path.write_bytes(b"")
        self.assertEqual(quant_bench.prewarm_page_cache(path), 0)


class StartTests(unittest.TestCase):
    def test_the_server_is_told_not_to_end_other_engines(self):
        work = scratch(self)
        with mock.patch.object(quant_bench.subprocess, "Popen") as popen:
            quant_bench.start_server(quant_bench.ENGINE_ROOT, work / "config.json", 18123, "8G", "quantbench-1", work)
        self.assertEqual(popen.call_args.kwargs["env"]["STRATA_KEEP_LEFTOVERS"], "1")
        self.assertEqual(popen.call_args.args[0][popen.call_args.args[0].index("--") + 1:][:3], quant_bench.PARENT_DEATH_PREFIX)
        self.assertIn("--unit=quantbench-1", popen.call_args.args[0])


class ParentDeathTests(unittest.TestCase):
    def test_a_child_started_with_the_prefix_gets_sigterm_when_its_parent_is_killed(self):
        code = ("import subprocess, sys, time\n"
                f"child = subprocess.Popen({quant_bench.PARENT_DEATH_PREFIX!r} + [sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                "print(child.pid, flush=True)\n"
                "time.sleep(60)\n")
        parent = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
        child_pid = int(parent.stdout.readline())
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and "time.sleep(60)" not in Path(f"/proc/{child_pid}/cmdline").read_text().replace("\0", " "):
            time.sleep(0.02)   # setpriv has set the signal and started the interpreter only after this
        parent.kill()
        parent.wait()

        def alive() -> bool:
            try:
                return "Z" not in Path(f"/proc/{child_pid}/stat").read_text().split()[2]
            except FileNotFoundError:
                return False
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and alive():
            time.sleep(0.05)
        still = alive()
        if still:
            os.kill(child_pid, signal.SIGKILL)
        self.assertFalse(still)

    def test_without_setpriv_the_server_command_is_unchanged(self):
        work = scratch(self)
        with mock.patch.object(quant_bench.shutil, "which", return_value=None), mock.patch.object(quant_bench.subprocess, "Popen") as popen:
            quant_bench.start_server(quant_bench.ENGINE_ROOT, work / "config.json", 18123, None, "quantbench-1", work)
        command = popen.call_args.args[0]
        self.assertEqual(command[command.index("--") + 1], str(quant_bench.ENGINE_ROOT / ".venv" / "bin" / "python"))


class ConfigTests(unittest.TestCase):
    def test_the_config_points_at_the_model_and_the_budget_picks_the_profile(self):
        work = scratch(self)
        model = work / "candidate.gguf"
        path, log, ram = quant_bench.write_config(quant_bench.ENGINE_ROOT, model, "16G", 18123, 1, work)
        config = json.loads(path.read_text())
        self.assertEqual(config["args"][config["args"].index("--native") + 1], str(model))
        self.assertEqual(config["args"][config["args"].index("--mtp") + 1], "1")
        self.assertIn("direct", config["args"])
        self.assertEqual((config["port"], ram, log), (18123, "16G", work / "server.log"))
        self.assertEqual(config["log"], str(work / "server.log"))

    def test_a_temperature_limit_is_handed_to_a_launcher_that_has_one(self):
        calls = []

        def build_config(profile, port, sampling, max_temp=0):
            calls.append(max_temp)
            return {"args": [], "model_name": "toy"}

        fake = types.SimpleNamespace(PROFILES={"max": {"ram": "22G"}, "auto": {"ram": None}}, build_config=build_config, MODEL_FILE=None, RUN_DIR=None)
        work = scratch(self)
        with mock.patch.object(quant_bench, "load_launcher", return_value=fake):
            quant_bench.write_config(work, work / "m.gguf", "22G", 18123, None, work, 80.0)
            quant_bench.write_config(work, work / "m.gguf", "22G", 18123, None, work)
        self.assertEqual(calls, [80.0, 0])

    def test_a_cpu_cap_joins_the_thermal_block_of_the_config(self):
        work = scratch(self)
        (work / "serve").mkdir()
        (work / "serve" / "cpucap.py").write_text("")
        fake = types.SimpleNamespace(PROFILES={"max": {"ram": "22G"}}, build_config=lambda profile, port, sampling, max_temp=0: {"args": [], "thermal": {"max_temp_c": max_temp}},
                                     MODEL_FILE=None, RUN_DIR=None)
        with mock.patch.object(quant_bench, "load_launcher", return_value=fake):
            path, _, _ = quant_bench.write_config(work, work / "m.gguf", "22G", 18123, None, work, 80.0, cpu_max_mhz=2200.0)
        self.assertEqual(json.loads(path.read_text())["thermal"], {"max_temp_c": 80.0, "cpu_max_mhz": 2200.0})

    def test_there_is_no_cap_in_the_config_unless_asked_for(self):
        work = scratch(self)
        fake = types.SimpleNamespace(PROFILES={"max": {"ram": "22G"}}, build_config=lambda profile, port, sampling, max_temp=0: {"args": []}, MODEL_FILE=None, RUN_DIR=None)
        with mock.patch.object(quant_bench, "load_launcher", return_value=fake):
            path, _, _ = quant_bench.write_config(work, work / "m.gguf", "22G", 18123, None, work, 80.0)
        self.assertNotIn("thermal", json.loads(path.read_text()))

    def test_an_engine_root_without_the_cap_is_refused_when_one_is_asked_for(self):
        work = scratch(self)
        fake = types.SimpleNamespace(PROFILES={"max": {"ram": "22G"}}, build_config=lambda profile, port, sampling, max_temp=0: {"args": []}, MODEL_FILE=None, RUN_DIR=None)
        with mock.patch.object(quant_bench, "load_launcher", return_value=fake), self.assertRaises(SystemExit) as raised:
            quant_bench.write_config(work, work / "m.gguf", "22G", 18123, None, work, 80.0, cpu_max_mhz=2200.0)
        self.assertIn("no CPU frequency cap", str(raised.exception))

    def test_a_launcher_without_a_temperature_limit_is_refused_when_one_is_asked_for(self):
        work = scratch(self)
        fake = types.SimpleNamespace(PROFILES={"max": {"ram": "22G"}}, build_config=lambda profile, port, sampling: {"args": []}, MODEL_FILE=None, RUN_DIR=None)
        with mock.patch.object(quant_bench, "load_launcher", return_value=fake), self.assertRaises(SystemExit) as raised:
            quant_bench.write_config(work, work / "m.gguf", "22G", 18123, None, work, 80.0)
        self.assertIn("no temperature limit", str(raised.exception))

    def test_an_engine_binary_and_environment_can_be_set(self):
        work = scratch(self)
        path, _, _ = quant_bench.write_config(quant_bench.ENGINE_ROOT, work / "m.gguf", "16G", 18123, None, work, 0.0, Path("/tmp/engine"), {"A": "1"})
        config = json.loads(path.read_text())
        self.assertEqual((config["exe"], config["env"]), ("/tmp/engine", {"A": "1"}))

    def test_the_config_is_unchanged_without_an_engine_binary_and_environment(self):
        work = scratch(self)
        path, _, _ = quant_bench.write_config(quant_bench.ENGINE_ROOT, work / "m.gguf", "16G", 18123, None, work)
        config = json.loads(path.read_text())
        self.assertEqual(config["exe"], "build/strata-q36")
        self.assertNotIn("env", config)

    def test_more_engine_arguments_are_appended_after_the_profile_ones(self):
        work = scratch(self)
        path, _, _ = quant_bench.write_config(quant_bench.ENGINE_ROOT, work / "m.gguf", "16G", 18123, None, work, 0.0, None, None, None, ["--expert-slots", "32"])
        args = json.loads(path.read_text())["args"]
        self.assertEqual(args[-2:], ["--expert-slots", "32"])
        self.assertIn("--native", args)

    def test_the_worker_count_can_be_set(self):
        work = scratch(self)
        path, _, _ = quant_bench.write_config(quant_bench.ENGINE_ROOT, work / "m.gguf", "16G", 18123, None, work, 0.0, None, None, 2)
        config = json.loads(path.read_text())
        self.assertEqual(config["args"][config["args"].index("--pool-workers") + 1], "2")

    def test_every_budget_has_a_profile(self):
        launcher = quant_bench.load_launcher(quant_bench.ENGINE_ROOT)
        for profile in quant_bench.PROFILE_OF_BUDGET.values():
            self.assertIn(profile, launcher.PROFILES)


class MeasureTests(unittest.TestCase):
    def setUp(self):
        self.work = scratch(self)
        self.model = self.work / "m.gguf"
        self.model.write_bytes(b"x" * 1024)
        self.log = self.work / "server.log"
        self.config = self.work / "config.json"
        self.config.write_text(json.dumps({"model_name": "toy"}))
        self.stop = mock.Mock(return_value=[])
        self.start = mock.Mock(return_value=mock.Mock(poll=mock.Mock(return_value=None)))
        self.turbo, self.cpu_c, self.gpu_c = "off", 60.0, 70.0

    def run_measure(self, args, fake_ask, ready=True, stats=None, stall=0.0):
        patches = [mock.patch.object(quant_bench, "write_config", return_value=(self.config, self.log, "16G")),
                   mock.patch.object(quant_bench, "start_server", self.start),
                   mock.patch.object(quant_bench, "wait_ready", ready if callable(ready) else mock.Mock(return_value=ready)),
                   mock.patch.object(quant_bench, "stop_server", self.stop),
                   mock.patch.object(quant_bench, "put_cpu_cap_back", return_value=[]),
                   mock.patch.object(quant_bench, "ask_watched", fake_ask),
                   mock.patch.object(quant_bench, "wait_until_cool", return_value=(50.0, 0.0, True)), mock.patch.object(quant_bench.time, "sleep"),
                   mock.patch.object(quant_bench, "other_stall", return_value=stall),
                   mock.patch.object(quant_bench, "turbo_state", return_value=self.turbo),
                   mock.patch.object(quant_bench, "package_temperature", return_value=self.cpu_c),
                   mock.patch.object(quant_bench, "gpu_temperature", return_value=self.gpu_c)]
        if stats is not None:
            patches.append(mock.patch.object(quant_bench, "collect_stats", return_value=stats))
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        return quant_bench.measure(args, self.work, say=lambda text: None)

    def test_the_flow_starts_waits_asks_and_always_stops(self):
        speeds = iter([20.0, 30.0, 32.0])

        def fake_ask(port, name, prompt, max_tokens, timeout, check):
            with open(self.log, "a") as handle:
                handle.write(LOG_LINE.format(ms=int(100000 / next(speeds))))
            return 1.0, ""

        result = self.run_measure(make_args(self.model), fake_ask)
        self.assertEqual(len(result["rows"]), 3)
        self.assertEqual(result["summary"]["code"]["median"], 31.0)
        self.assertEqual(result["summary"]["code"]["accepted_share"], 0.8)
        self.assertIsNone(result["aborted"])
        self.stop.assert_called_once()

    def test_a_server_that_never_gets_ready_is_reported_and_stopped(self):
        (self.work / "server.out").write_text("boom\n")
        result = self.run_measure(make_args(self.model), lambda *a: (1.0, ""), ready=False)
        self.assertIn("boom", result["aborted"])
        self.assertEqual(result["rows"], [])
        self.stop.assert_called_once()

    def test_a_missing_log_line_is_kept_in_the_rows_and_left_out_of_the_summary(self):
        result = self.run_measure(make_args(self.model, warmup=0, repeats=1), lambda *a: (1.0, ""), stats=(None, 0))
        self.assertEqual(result["rows"][0]["stat_lines"], 0)
        self.assertEqual(result["summary"]["code"], {"runs": 0, "left_out": 1})

    def test_a_memory_stall_aborts_the_run_and_still_stops_the_server(self):
        result = self.run_measure(make_args(self.model, max_stall=5.0), lambda *a: (1.0, ""), stall=20.0)
        self.assertIn("stalling on memory", result["aborted"])
        self.stop.assert_called_once()

    def test_the_stall_of_other_programs_is_also_watched_while_the_server_starts(self):
        def wait_ready(port, process, timeout, check):
            check()
            return True
        result = self.run_measure(make_args(self.model, max_stall=5.0), lambda *a: (1.0, ""), ready=wait_ready, stall=20.0)
        self.assertIn("stalling on memory", result["aborted"])
        self.assertEqual(result["rows"], [])
        self.stop.assert_called_once()

    def test_the_stop_signals_are_ignored_while_the_server_is_stopped(self):
        seen = []
        self.stop.side_effect = lambda unit, process: seen.append(signal.getsignal(signal.SIGTERM)) or []
        before = signal.getsignal(signal.SIGTERM)
        self.run_measure(make_args(self.model, warmup=0, repeats=1), lambda *a: (1.0, ""), stats=(None, 0))
        self.assertEqual(seen, [signal.SIG_IGN])
        self.assertEqual(signal.getsignal(signal.SIGTERM), before)

    def test_the_cpu_and_gpu_peaks_are_kept_apart(self):
        self.cpu_c, self.gpu_c = 82.0, 71.0
        row = self.run_measure(make_args(self.model, warmup=0, repeats=1), lambda *a: (1.0, ""), stats=(None, 0))["rows"][0]
        self.assertEqual((row["peak_c"], row["peak_cpu_c"], row["peak_gpu_c"]), (82.0, 82.0, 71.0))

    def test_the_log_is_kept_next_to_the_result_by_default(self):
        self.log.write_text("thermal line\n")
        out = self.work / "result.json"
        self.run_measure(make_args(self.model, warmup=0, repeats=1, out=out), lambda *a: (1.0, ""), stats=(None, 0))
        self.assertEqual((self.work / "result.json.log").read_text(), "thermal line\n")

    def test_the_server_log_can_be_kept(self):
        self.log.write_text("engine counters\n")
        target = self.work / "kept.log"
        self.run_measure(make_args(self.model, warmup=0, repeats=1, save_log=target), lambda *a: (1.0, ""), stats=(None, 0))
        self.assertEqual(target.read_text(), "engine counters\n")

    def test_the_console_output_with_the_governor_lines_is_kept(self):
        (self.work / "server.out").write_text("[strata] thermal: cpu 80 C reached the limit\n")
        out = self.work / "result.json"
        self.run_measure(make_args(self.model, warmup=0, repeats=1, out=out), lambda *a: (1.0, ""), stats=(None, 0))
        self.assertEqual((self.work / "result.json.log.console").read_text(), "[strata] thermal: cpu 80 C reached the limit\n")

    def test_a_missing_console_output_is_not_an_error(self):
        out = self.work / "result.json"
        self.run_measure(make_args(self.model, warmup=0, repeats=1, out=out), lambda *a: (1.0, ""), stats=(None, 0))
        self.assertFalse((self.work / "result.json.log.console").exists())

    def test_the_console_output_is_kept_next_to_the_result_even_when_the_log_goes_elsewhere(self):
        (self.work / "server.out").write_text("governor line\n")
        out = self.work / "result.json"
        elsewhere = self.work / "elsewhere" / "server.log"
        elsewhere.parent.mkdir()
        self.run_measure(make_args(self.model, warmup=0, repeats=1, out=out, save_log=elsewhere), lambda *a: (1.0, ""), stats=(None, 0))
        self.assertEqual((self.work / "result.json.log.console").read_text(), "governor line\n")

    def test_stop_signals_are_blocked_while_the_server_starts_and_free_afterwards(self):
        seen = []

        def start(*args):
            seen.append(signal.pthread_sigmask(signal.SIG_BLOCK, []))
            return mock.Mock(poll=mock.Mock(return_value=None))

        self.start.side_effect = start
        self.run_measure(make_args(self.model, warmup=0, repeats=1), lambda *a: (1.0, ""), stats=(None, 0))
        self.assertTrue(all(number in seen[0] for number in quant_bench.STOP_SIGNALS))
        self.assertFalse(any(number in signal.pthread_sigmask(signal.SIG_BLOCK, []) for number in quant_bench.STOP_SIGNALS))

    def test_the_whole_answer_is_kept(self):
        result = self.run_measure(make_args(self.model, warmup=0, repeats=1), lambda *a: (1.0, "x" * 500), stats=(None, 0))
        self.assertEqual(len(result["rows"][0]["answer"]), 500)

    def test_cleanup_problems_are_carried_into_the_result(self):
        self.stop.return_value = ["the scope is still active"]
        result = self.run_measure(make_args(self.model, warmup=0, repeats=1), lambda *a: (1.0, ""), stats=(None, 0))
        self.assertEqual(result["cleanup_problems"], ["the scope is still active"])

    def test_a_busy_machine_is_refused_before_anything_starts(self):
        args = make_args(self.model, max_load=-1.0)
        with self.assertRaises(SystemExit) as raised, mock.patch.object(quant_bench, "start_server", side_effect=AssertionError("must not start")), \
                mock.patch.object(quant_bench, "write_config", return_value=(self.config, self.log, "16G")):
            quant_bench.measure(args, self.work, say=lambda text: None)
        self.assertIn("refusing to measure", str(raised.exception))

    def test_the_model_and_the_named_files_are_dropped_from_the_cache_before_the_start(self):
        order = []
        other = self.work / "other.gguf"
        with mock.patch.object(quant_bench, "evict_page_cache", side_effect=lambda paths: order.append(("evict", [Path(p).name for p in paths])) or []):
            self.start.side_effect = lambda *a, **k: order.append(("start",)) or mock.Mock(poll=mock.Mock(return_value=None))
            self.run_measure(make_args(self.model, evict=[other], warmup=0, repeats=1), lambda *a: (1.0, ""), stats=(None, 0))
        self.assertEqual(order[0], ("evict", ["m.gguf", "other.gguf"]))
        self.assertEqual(order[1], ("start",))

    def test_the_model_is_read_into_the_cache_after_the_eviction_when_asked(self):
        order = []
        with mock.patch.object(quant_bench, "evict_page_cache", side_effect=lambda paths: order.append("evict") or []), \
                mock.patch.object(quant_bench, "prewarm_page_cache", side_effect=lambda path: order.append("prewarm") or 1024):
            self.run_measure(make_args(self.model, prewarm=True, warmup=0, repeats=1), lambda *a: (1.0, ""), stats=(None, 0))
        self.assertEqual(order, ["evict", "prewarm"])

    def test_the_hottest_reading_of_a_request_is_kept_and_the_limit_is_recorded(self):
        result = self.run_measure(make_args(self.model, warmup=0, repeats=1, max_temp=80.0), lambda *a: (1.0, ""), stats=(None, 0))
        self.assertEqual(result["rows"][0]["peak_c"], 70.0)
        self.assertEqual((result["max_temp_c"], result["summary"]["code"]["left_out"]), (80.0, 1))

    def test_a_missing_sensor_leaves_the_peak_unknown(self):
        self.cpu_c = self.gpu_c = None
        result = self.run_measure(make_args(self.model, warmup=0, repeats=1), lambda *a: (1.0, ""), stats=(None, 0))
        self.assertIsNone(result["rows"][0]["peak_c"])

    def test_the_turbo_boost_being_on_is_refused_before_anything_starts(self):
        self.turbo = "on"
        with self.assertRaises(SystemExit) as raised:
            self.run_measure(make_args(self.model), lambda *a: (1.0, ""))
        self.assertIn("turbo", str(raised.exception))
        self.start.assert_not_called()

    def test_the_turbo_boost_can_be_allowed_explicitly(self):
        self.turbo = "on"
        result = self.run_measure(make_args(self.model, allow_turbo=True, warmup=0, repeats=1), lambda *a: (1.0, ""), stats=(None, 0))
        self.assertEqual(result["turbo"], "on")

    def test_a_port_that_is_taken_is_refused_before_anything_starts(self):
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            args = make_args(self.model, port=server.getsockname()[1])
            with self.assertRaises(SystemExit) as raised, mock.patch.object(quant_bench, "start_server", side_effect=AssertionError("must not start")), \
                    mock.patch.object(quant_bench, "write_config", return_value=(self.config, self.log, "16G")):
                quant_bench.measure(args, self.work, say=lambda text: None)
        self.assertIn("already in use", str(raised.exception))


class MainTests(unittest.TestCase):
    def run_main(self, result):
        work = scratch(self)
        model = work / "m.gguf"
        model.write_bytes(b"x")
        out = work / "result.json"
        with mock.patch.object(sys, "argv", ["quant_bench", "--model", str(model), "--label", "toy", "--out", str(out)]), \
                mock.patch.object(quant_bench, "measure", return_value=result):
            code = quant_bench.main()
        return code, out

    def test_a_good_run_writes_the_result_and_returns_zero(self):
        code, out = self.run_main({"aborted": None, "cleanup_problems": [], "summary": {}})
        self.assertEqual(code, 0)
        self.assertTrue(out.exists())

    def test_an_aborted_run_is_kept_apart_so_a_queue_does_not_take_it_for_done(self):
        code, out = self.run_main({"aborted": "stalling", "cleanup_problems": [], "summary": {}})
        self.assertEqual(code, 3)
        self.assertFalse(out.exists())
        self.assertTrue(out.with_name("result.json.aborted").exists())


if __name__ == "__main__":
    unittest.main()
