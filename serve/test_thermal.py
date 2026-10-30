"""The temperature limit: the sensor reading, the control step, the governor thread, a real process being slowed.

    python -m unittest serve.test_thermal -v
"""
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from types import SimpleNamespace

from serve import thermal
from serve.server import Service
from serve.thermal import Governor, cpu_package_temp_c, next_duty


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="ascii")


def wait_for(cond, t=5.0):
    end = time.time() + t
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


class NextDutyTests(unittest.TestCase):
    def test_below_the_limit_nothing_is_paused(self):
        self.assertEqual(next_duty(70.0, 0.0, 80.0), 0.0)

    def test_at_the_limit_the_pause_grows_and_faster_the_hotter(self):
        at = next_duty(80.0, 0.0, 80.0)
        hotter = next_duty(85.0, 0.0, 80.0)
        self.assertAlmostEqual(at, 0.10)
        self.assertGreater(hotter, at)

    def test_the_pause_is_capped(self):
        duty = 0.0
        for _ in range(50):
            duty = next_duty(95.0, duty, 80.0)
        self.assertEqual(duty, thermal.MAX_DUTY)

    def test_inside_the_margin_it_holds(self):
        self.assertEqual(next_duty(78.0, 0.4, 80.0), 0.4)

    def test_well_below_the_limit_it_relaxes_to_zero(self):
        duty = 0.12
        for _ in range(10):
            duty = next_duty(60.0, duty, 80.0)
        self.assertEqual(duty, 0.0)


class SensorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def read(self):
        return cpu_package_temp_c(str(self.root / "hwmon"), str(self.root / "thermal"))

    def test_intel_package_reading(self):
        hw = self.root / "hwmon" / "hwmon7"
        write(hw / "name", "coretemp\n")
        write(hw / "temp1_label", "Package id 0\n")
        write(hw / "temp1_input", "58000\n")
        write(hw / "temp2_label", "Core 0\n")
        write(hw / "temp2_input", "91000\n")
        self.assertEqual(self.read(), 58.0)

    def test_other_sensors_are_ignored(self):
        write(self.root / "hwmon" / "hwmon1" / "name", "acpitz\n")
        write(self.root / "hwmon" / "hwmon1" / "temp1_input", "99000\n")
        self.assertIsNone(self.read())

    def test_amd_without_labels_uses_the_first_input(self):
        hw = self.root / "hwmon" / "hwmon3"
        write(hw / "name", "k10temp\n")
        write(hw / "temp1_input", "64500\n")
        self.assertEqual(self.read(), 64.5)

    def test_amd_tctl_label(self):
        hw = self.root / "hwmon" / "hwmon3"
        write(hw / "name", "k10temp\n")
        write(hw / "temp1_label", "Tctl\n")
        write(hw / "temp1_input", "71000\n")
        write(hw / "temp3_label", "Tccd1\n")
        write(hw / "temp3_input", "99000\n")
        self.assertEqual(self.read(), 71.0)

    def test_thermal_zone_fallback(self):
        write(self.root / "thermal" / "thermal_zone2" / "type", "x86_pkg_temp\n")
        write(self.root / "thermal" / "thermal_zone2" / "temp", "62000\n")
        self.assertEqual(self.read(), 62.0)

    def test_the_hottest_package_wins(self):
        for n, value in ((0, "55000"), (1, "73000")):
            hw = self.root / "hwmon" / f"hwmon{n}"
            write(hw / "name", "coretemp\n")
            write(hw / "temp1_label", "Package id %d\n" % n)
            write(hw / "temp1_input", value + "\n")
        self.assertEqual(self.read(), 73.0)

    def test_an_unreadable_value_is_skipped(self):
        hw = self.root / "hwmon" / "hwmon7"
        write(hw / "name", "coretemp\n")
        write(hw / "temp1_label", "Package id 0\n")
        write(hw / "temp1_input", "not a number\n")
        self.assertIsNone(self.read())


class Recorder:
    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, pid, sig):
        with self.lock:
            self.calls.append((pid, sig))

    def signals(self):
        with self.lock:
            return [sig for _, sig in self.calls]


class GovernorTests(unittest.TestCase):
    def make(self, temps, pid=4242, limit=80.0):
        self.messages = []
        self.rec = Recorder()
        self.temp = temps
        gov = Governor(limit, lambda: {"cpu": self.temp["cpu"], "gpu": self.temp.get("gpu")}, lambda: pid,
                       log=self.messages.append, period_s=0.02, sample_s=0.02, kill=self.rec)
        self.addCleanup(gov.stop)
        return gov

    def test_a_cool_machine_is_left_alone(self):
        gov = self.make({"cpu": 60.0})
        gov.start()
        time.sleep(0.3)
        gov.stop()
        self.assertNotIn(signal.SIGSTOP, self.rec.signals())
        self.assertEqual(gov.duty, 0.0)

    def test_a_hot_machine_pauses_the_engine_and_always_resumes_it(self):
        gov = self.make({"cpu": 86.0})
        gov.start()
        self.assertTrue(wait_for(lambda: self.rec.signals().count(signal.SIGSTOP) >= 3))
        gov.stop()
        signals = self.rec.signals()
        self.assertEqual(signals[-1], signal.SIGCONT)
        for first, second in zip(signals, signals[1:]):
            self.assertFalse(first == signal.SIGSTOP and second == signal.SIGSTOP)
        self.assertTrue(any("reached the 80 C limit" in m for m in self.messages))

    def test_the_resume_goes_to_the_process_that_was_stopped(self):
        pids = iter([100] + [200] * 1000)
        self.messages = []
        rec = self.rec = Recorder()
        gov = Governor(80.0, lambda: {"cpu": 90.0}, lambda: next(pids), log=self.messages.append, period_s=0.02, sample_s=0.02, kill=rec)
        self.addCleanup(gov.stop)
        gov._sample = lambda: setattr(gov, "duty", 0.5) if gov.duty < 0.5 else None
        gov.start()
        self.assertTrue(wait_for(lambda: (100, signal.SIGCONT) in rec.calls))
        gov.stop()
        first_stop = rec.calls.index((100, signal.SIGSTOP))
        self.assertEqual(rec.calls[first_stop + 1], (100, signal.SIGCONT))

    def test_the_hottest_source_decides_and_is_named(self):
        gov = self.make({"cpu": 60.0, "gpu": 84.0})
        gov.start()
        self.assertTrue(wait_for(lambda: gov.duty > 0))
        self.assertEqual(gov.hottest, "gpu")
        gov.stop()

    def test_cooling_down_ends_the_pausing(self):
        temps = {"cpu": 90.0}
        gov = self.make(temps)
        gov.start()
        self.assertTrue(wait_for(lambda: gov.duty > 0.3))
        temps["cpu"] = 60.0
        self.assertTrue(wait_for(lambda: gov.duty == 0.0))
        self.assertTrue(any("runs freely again" in m for m in self.messages))
        gov.stop()

    def test_no_readings_relaxes_instead_of_pausing_blindly(self):
        temps = {"cpu": 90.0}
        gov = self.make(temps)
        gov.start()
        self.assertTrue(wait_for(lambda: gov.duty > 0.3))
        temps["cpu"] = None
        self.assertTrue(wait_for(lambda: gov.duty == 0.0))
        gov.stop()

    def test_no_engine_means_no_signals(self):
        gov = self.make({"cpu": 90.0}, pid=None)
        gov.start()
        time.sleep(0.3)
        gov.stop()
        self.assertEqual(self.rec.calls, [])

    def test_a_vanished_engine_does_not_break_the_loop(self):
        def gone(pid, sig):
            raise ProcessLookupError

        gov = self.make({"cpu": 90.0})
        gov.kill = gone
        gov.start()
        time.sleep(0.2)
        gov.stop()

    def test_a_failing_sensor_does_not_break_the_loop(self):
        def boom():
            raise RuntimeError("sensor")

        gov = Governor(80.0, boom, lambda: 1, log=lambda m: None, period_s=0.02, sample_s=0.02, kill=Recorder())
        self.addCleanup(gov.stop)
        gov.start()
        time.sleep(0.2)
        gov.stop()
        self.assertEqual(gov.duty, 0.0)

    def test_status_reports_the_state(self):
        gov = self.make({"cpu": 90.0})
        gov.start()
        self.assertTrue(wait_for(lambda: gov.status()["duty"] > 0))
        status = gov.status()
        gov.stop()
        self.assertEqual(status["limit_c"], 80.0)
        self.assertEqual(status["hottest"], "cpu")


class CreateTests(unittest.TestCase):
    def make(self, cfg):
        return thermal.create(cfg, lambda: {}, lambda: None, log=lambda m: None)

    def test_off_when_absent_or_zero(self):
        self.assertIsNone(self.make(None))
        self.assertIsNone(self.make({}))
        self.assertIsNone(self.make({"max_temp_c": 0}))

    def test_a_limit_makes_a_governor(self):
        self.assertEqual(self.make({"max_temp_c": 80}).limit_c, 80.0)

    def test_nonsense_limits_are_refused(self):
        for bad in (True, "80", 20, 200, -5):
            with self.assertRaises(ValueError):
                self.make({"max_temp_c": bad})


class ServiceWiringTests(unittest.TestCase):
    def service(self):
        return Service.__new__(Service)

    def test_the_engine_pid_comes_from_the_engine_process(self):
        svc = self.service()
        svc.engine = SimpleNamespace(proc=SimpleNamespace(pid=4321))
        self.assertEqual(svc.engine_pid(), 4321)

    def test_an_exited_engine_has_no_pid(self):
        svc = self.service()
        svc.engine = SimpleNamespace(proc=SimpleNamespace(pid=4321, poll=lambda: 0))
        self.assertIsNone(svc.engine_pid())
        svc.engine = SimpleNamespace(proc=SimpleNamespace(pid=4321, poll=lambda: None))
        self.assertEqual(svc.engine_pid(), 4321)

    def test_an_engine_without_a_process_has_no_pid(self):
        svc = self.service()
        svc.engine = SimpleNamespace()
        self.assertIsNone(svc.engine_pid())
        svc.engine = SimpleNamespace(proc=None)
        self.assertIsNone(svc.engine_pid())

    def test_the_readings_carry_the_cpu_and_the_telemetry_gpu(self):
        svc = self.service()
        svc.telemetry = SimpleNamespace(snapshot=lambda: {"now": {"gpu_temp": 77}})
        readings = svc.thermal_readings()
        self.assertEqual(readings["gpu"], 77)
        self.assertIn("cpu", readings)

    def test_the_series_is_empty_without_a_governor_and_named_with_one(self):
        svc = self.service()
        svc.thermal = None
        self.assertEqual(svc._thermal_series(), {})
        svc.thermal = SimpleNamespace(duty=0.4, temp_c=83.0)
        self.assertEqual(svc._thermal_series(), {"thermal_duty": 0.4, "thermal_temp": 83.0})
        svc.thermal = SimpleNamespace(duty=0.0, temp_c=None)
        self.assertEqual(svc._thermal_series(), {"thermal_duty": 0.0})


@unittest.skipUnless(sys.platform.startswith("linux"), "needs /proc")
class RealProcessTests(unittest.TestCase):
    def cpu_seconds(self, pid):
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")

    def test_a_busy_process_is_slowed_and_then_released(self):
        child = subprocess.Popen([sys.executable, "-I", "-c", "while True: pass"])
        self.addCleanup(child.kill)
        gov = Governor(80.0, lambda: {"cpu": 95.0}, lambda: child.pid, log=lambda m: None, period_s=0.1, sample_s=0.1)
        self.addCleanup(gov.stop)
        gov.duty = thermal.MAX_DUTY
        gov._sample = lambda: None
        time.sleep(0.3)
        before = self.cpu_seconds(child.pid)
        gov.start()
        time.sleep(2.0)
        slowed = self.cpu_seconds(child.pid) - before
        gov.stop()
        state = Path(f"/proc/{child.pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        self.assertNotEqual(state, "T")
        time.sleep(0.3)
        mark = self.cpu_seconds(child.pid)
        time.sleep(1.0)
        free = self.cpu_seconds(child.pid) - mark
        self.assertLess(slowed, 1.0)
        self.assertGreater(free, 0.5)


if __name__ == "__main__":
    unittest.main()
