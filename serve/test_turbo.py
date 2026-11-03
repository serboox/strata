"""tools/turbo against a fake sysfs tree.

    python -m unittest serve.test_turbo -v
"""
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TURBO = Path(__file__).resolve().parents[1] / "tools" / "turbo"


class TurboTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def knob(self, name, value, freq="3300000"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value + "\n", encoding="ascii")
        cpufreq = self.root / "cpu0" / "cpufreq"
        cpufreq.mkdir(parents=True, exist_ok=True)
        (cpufreq / "scaling_max_freq").write_text(freq + "\n", encoding="ascii")
        return path

    def run_turbo(self, *args):
        env = dict(os.environ, TURBO_SYSFS_ROOT=str(self.root))
        return subprocess.run([sys.executable, "-I", str(TURBO), *args], capture_output=True, text=True, env=env, timeout=30)

    def test_intel_status_reads_no_turbo_the_right_way_round(self):
        self.knob("intel_pstate/no_turbo", "0")
        out = self.run_turbo("status")
        self.assertEqual(out.returncode, 0)
        self.assertIn("turbo: on (", out.stdout)
        self.knob("intel_pstate/no_turbo", "1")
        self.assertIn("turbo: off (", self.run_turbo("status").stdout)

    def test_intel_off_and_on(self):
        path = self.knob("intel_pstate/no_turbo", "0")
        out = self.run_turbo("off")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(path.read_text().strip(), "1")
        self.assertIn("turbo: off (", out.stdout)
        out = self.run_turbo("on")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(path.read_text().strip(), "0")

    def test_cpufreq_boost_is_the_other_way_round(self):
        path = self.knob("cpufreq/boost", "1")
        self.assertEqual(self.run_turbo("off").returncode, 0)
        self.assertEqual(path.read_text().strip(), "0")
        self.assertIn("turbo: off (", self.run_turbo("status").stdout)
        self.assertEqual(self.run_turbo("on").returncode, 0)
        self.assertEqual(path.read_text().strip(), "1")

    def test_already_in_the_state_writes_nothing(self):
        path = self.knob("intel_pstate/no_turbo", "1")
        path.chmod(0o444)
        out = self.run_turbo("off", "--no-sudo")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("already off", out.stdout)

    def test_a_knob_that_is_not_writable_fails_without_sudo(self):
        if os.geteuid() == 0:
            self.skipTest("root can write anything")
        path = self.knob("intel_pstate/no_turbo", "0")
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        out = self.run_turbo("off", "--no-sudo")
        self.assertEqual(out.returncode, 1)
        self.assertIn("not writable", out.stderr)
        self.assertEqual(path.read_text().strip(), "0")

    def test_a_cpu_without_either_knob_is_reported(self):
        out = self.run_turbo("status")
        self.assertEqual(out.returncode, 2)
        self.assertIn("neither", out.stderr)

    def test_quiet_prints_nothing_on_success(self):
        self.knob("intel_pstate/no_turbo", "0")
        out = self.run_turbo("off", "--quiet")
        self.assertEqual(out.returncode, 0)
        self.assertEqual(out.stdout, "")

    def test_the_cpu_limit_is_reported(self):
        self.knob("intel_pstate/no_turbo", "1", freq="3300000")
        self.assertIn("CPU limit 3300 MHz", self.run_turbo("status").stdout)


if __name__ == "__main__":
    unittest.main()
