import contextlib
import io
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import strata_launch

SERVER = ["python", "serve/server.py", "--port", "8080"]


class ScopeTests(unittest.TestCase):
    def test_properties_combine_the_memory_limit_and_the_background_weights(self):
        self.assertEqual(
            strata_launch.scope_properties("16G", True),
            ["MemoryMax=16G", "MemorySwapMax=0", "CPUWeight=idle", "IOWeight=1"],
        )
        self.assertEqual(strata_launch.scope_properties(None, True), ["CPUWeight=idle", "IOWeight=1"])
        self.assertEqual(strata_launch.scope_properties("8G", False), ["MemoryMax=8G", "MemorySwapMax=0"])
        self.assertEqual(strata_launch.scope_properties(None, False), [])

    def test_no_properties_means_no_scope(self):
        self.assertEqual(strata_launch.wrap_in_scope(SERVER, []), SERVER)

    def test_dollar_signs_in_the_server_command_survive_systemd_run(self):
        command = strata_launch.wrap_in_scope(["python", "/srv/$dir/server.py", "plain"], ["CPUWeight=idle"])
        self.assertEqual(command[-3:], ["python", "/srv/$$dir/server.py", "plain"])

    def test_the_server_runs_inside_a_user_scope_that_carries_the_properties(self):
        command = strata_launch.wrap_in_scope(SERVER, ["CPUWeight=idle", "IOWeight=1"])
        self.assertEqual(
            command,
            ["systemd-run", "--user", "--scope", "--quiet", "-p", "CPUWeight=idle", "-p", "IOWeight=1", "--"] + SERVER,
        )


class ApplyPriorityTests(unittest.TestCase):
    def run_priority(self, mode, dry_run=False, scheduler_error=None, ionice_code=0):
        calls = {"scheduler": [], "ionice": []}

        def set_scheduler(pid, policy, param):
            calls["scheduler"].append((pid, policy, param.sched_priority))
            if scheduler_error is not None:
                raise scheduler_error

        def run(command, **kwargs):
            calls["ionice"].append(command)
            return subprocess.CompletedProcess(command, ionice_code, stdout="", stderr="ionice failed" if ionice_code else "")

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            strata_launch.apply_priority(mode, dry_run, set_scheduler=set_scheduler, run=run)
        return calls, stderr.getvalue()

    def test_background_sets_the_idle_scheduling_class_and_the_idle_io_class(self):
        calls, stderr = self.run_priority("background")
        self.assertEqual(calls["scheduler"], [(0, os.SCHED_IDLE, 0)])
        self.assertEqual(calls["ionice"], [["ionice", "-c3", "-p", str(os.getpid())]])
        self.assertEqual(stderr, "")

    def test_normal_changes_nothing(self):
        calls, stderr = self.run_priority("normal")
        self.assertEqual(calls, {"scheduler": [], "ionice": []})
        self.assertEqual(stderr, "")

    def test_a_dry_run_changes_nothing_and_says_so(self):
        calls, stderr = self.run_priority("background", dry_run=True)
        self.assertEqual(calls, {"scheduler": [], "ionice": []})
        self.assertIn("not applied in a dry run", stderr)

    def test_a_missing_ionice_is_a_warning_not_a_crash(self):
        def run(command, **kwargs):
            raise FileNotFoundError(2, "No such file or directory: 'ionice'")

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            strata_launch.apply_priority("background", False, set_scheduler=lambda *args: None, run=run)
        self.assertIn("could not set the idle I/O class", stderr.getvalue())

    def test_failures_are_warnings_and_the_other_step_still_runs(self):
        calls, stderr = self.run_priority("background", scheduler_error=PermissionError("denied"), ionice_code=1)
        self.assertEqual(len(calls["ionice"]), 1)
        self.assertIn("could not set the idle scheduling class: denied", stderr)
        self.assertIn("could not set the idle I/O class: ionice failed", stderr)


class InheritanceTests(unittest.TestCase):
    def test_what_the_launcher_sets_reaches_the_processes_it_starts(self):
        if shutil.which("ionice") is None:
            self.skipTest("no ionice")
        child = (
            "import os, subprocess, sys\n"
            f"sys.path.insert(0, {str(ROOT / 'tools')!r})\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "import strata_launch\n"
            "strata_launch.apply_priority('background', False)\n"
            "probe = 'import os, subprocess; print(os.sched_getscheduler(0) == os.SCHED_IDLE, subprocess.check_output([\"ionice\", \"-p\", str(os.getpid())], text=True).strip())'\n"
            "print(subprocess.check_output([sys.executable, '-c', probe], text=True).strip())\n"
        )
        result = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=60, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "True idle")


class ProbeTests(unittest.TestCase):
    def probe(self, **result):
        with mock.patch.object(strata_launch.subprocess, "run", **result) as run:
            return strata_launch.background_scope_works(), run

    def test_the_probe_accepts_only_a_scope_whose_weight_really_is_the_lowest(self):
        accepted, run = self.probe(return_value=subprocess.CompletedProcess([], 0, stdout="1\n"))
        self.assertTrue(accepted)
        command = run.call_args.args[0]
        self.assertEqual(command[:4], ["systemd-run", "--user", "--scope", "--quiet"])
        self.assertIn("CPUWeight=idle", command)
        self.assertEqual(command[-3:], [sys.executable, "-c", strata_launch.WEIGHT_PROBE])
        self.assertEqual(run.call_args.kwargs["timeout"], strata_launch.PROBE_TIMEOUT_SECONDS)

    def test_a_scope_that_started_with_its_weight_ignored_is_refused(self):
        for stdout, code in (("100\n", 0), ("", 0), ("1\n", 1)):
            with self.subTest(stdout=stdout, code=code):
                self.assertFalse(self.probe(return_value=subprocess.CompletedProcess([], code, stdout=stdout))[0])

    def test_a_missing_or_hung_systemd_run_is_refused(self):
        self.assertFalse(self.probe(side_effect=FileNotFoundError("systemd-run"))[0])
        self.assertFalse(self.probe(side_effect=subprocess.TimeoutExpired("systemd-run", 10))[0])

    def test_the_probe_code_reads_the_weight_of_its_own_group(self):
        code = strata_launch.WEIGHT_PROBE
        self.assertNotIn("$", code)
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30, check=False)
        self.assertIn(result.returncode, (0, 1))
        if result.returncode == 0:
            self.assertRegex(result.stdout.strip(), r"^\d+$")


if __name__ == "__main__":
    unittest.main()
