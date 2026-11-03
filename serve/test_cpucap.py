import importlib.machinery
import importlib.util
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from serve import cpucap
from serve.cpucap import CpuCap


def make_cpus(root, count=2, current="3300000", ceiling="3300000"):
    files = []
    for number in range(count):
        directory = Path(root) / f"cpu{number}" / "cpufreq"
        directory.mkdir(parents=True)
        (directory / "scaling_max_freq").write_text(current, encoding="ascii")
        (directory / "cpuinfo_max_freq").write_text(ceiling, encoding="ascii")
        files.append(directory / "scaling_max_freq")
    return files


class CpuCapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.messages = []

    def cap(self, mhz=2200):
        return CpuCap(mhz, log=self.messages.append, root=self.tmp.name)

    def test_the_cap_is_written_in_khz_to_every_cpu(self):
        files = make_cpus(self.tmp.name, count=3)
        self.cap().start()
        self.assertEqual([path.read_text() for path in files], ["2200000"] * 3)

    def test_stop_puts_the_old_values_back(self):
        files = make_cpus(self.tmp.name)
        files[1].write_text("2800000", encoding="ascii")
        cap = self.cap()
        cap.start()
        cap.stop()
        self.assertEqual([path.read_text() for path in files], ["3300000", "2800000"])

    def test_a_second_stop_changes_nothing(self):
        files = make_cpus(self.tmp.name)
        cap = self.cap()
        cap.start()
        cap.stop()
        files[0].write_text("2000000", encoding="ascii")
        cap.stop()
        self.assertEqual(files[0].read_text(), "2000000")

    def test_the_cap_never_exceeds_the_hardware_maximum(self):
        files = make_cpus(self.tmp.name, ceiling="3300000")
        self.cap(5000).start()
        self.assertEqual(files[0].read_text(), "3300000")

    def test_the_start_is_announced(self):
        make_cpus(self.tmp.name, count=2)
        self.cap().start()
        self.assertEqual(len(self.messages), 1)
        self.assertIn("2200 MHz on 2 policies", self.messages[0])

    def test_a_machine_without_cpufreq_is_left_alone_and_says_so(self):
        self.cap().start()
        self.assertEqual(len(self.messages), 1)
        self.assertIn("no cpufreq limit", self.messages[0])

    @unittest.skipIf(os.geteuid() == 0, "root can write to a read-only file")
    def test_an_unwritable_limit_is_reported_and_is_not_fatal(self):
        files = make_cpus(self.tmp.name, count=2)
        files[0].chmod(0o444)
        self.cap().start()
        self.assertEqual(files[0].read_text(), "3300000")
        self.assertEqual(files[1].read_text(), "2200000")
        self.assertTrue(any("1 of 2 limits could not be written" in message for message in self.messages))


class DescribeAndResetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_the_description_names_the_limit_and_the_hardware_maximum(self):
        make_cpus(self.tmp.name, count=2, current="2200000")
        self.assertEqual(cpucap.describe(self.tmp.name), "highest CPU frequency 2200 MHz on 2 CPUs (hardware maximum 3300 MHz)")

    def test_a_machine_without_cpufreq_is_described_as_such(self):
        self.assertEqual(cpucap.describe(self.tmp.name), "no cpufreq limit on this machine")

    def test_reset_puts_every_limit_to_the_hardware_maximum(self):
        files = make_cpus(self.tmp.name, count=2, current="2200000")
        self.assertEqual(cpucap.reset(self.tmp.name), [])
        self.assertEqual([path.read_text() for path in files], ["3300000"] * 2)

    def test_apply_does_not_register_the_restore_at_exit(self):
        files = make_cpus(self.tmp.name)
        cap = CpuCap(2200, log=lambda message: None, root=self.tmp.name)
        with mock.patch.object(cpucap.atexit, "register") as register:
            cap.apply()
        register.assert_not_called()
        self.assertEqual(files[0].read_text(), "2200000")

    def test_start_registers_the_restore_at_exit(self):
        make_cpus(self.tmp.name)
        cap = CpuCap(2200, log=lambda message: None, root=self.tmp.name)
        with mock.patch.object(cpucap.atexit, "register") as register:
            cap.start()
        register.assert_called_once_with(cap.stop)


class HoldTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.messages = []

    def cap(self):
        return CpuCap(2200, log=self.messages.append, root=self.tmp.name)

    def test_cpus_that_share_a_policy_are_handled_once(self):
        shared = Path(self.tmp.name) / "policy0"
        shared.mkdir()
        (shared / "scaling_max_freq").write_text("3300000", encoding="ascii")
        (shared / "cpuinfo_max_freq").write_text("3300000", encoding="ascii")
        for number in range(2):
            cpu = Path(self.tmp.name) / f"cpu{number}"
            cpu.mkdir()
            (cpu / "cpufreq").symlink_to(shared)
        cap = self.cap()
        cap.start()
        self.assertEqual((shared / "scaling_max_freq").read_text(), "2200000")
        cap.stop()
        self.assertEqual((shared / "scaling_max_freq").read_text(), "3300000")

    def test_a_second_start_keeps_the_first_original_values(self):
        files = make_cpus(self.tmp.name)
        cap = self.cap()
        cap.start()
        cap.start()
        cap.stop()
        self.assertEqual(files[0].read_text(), "3300000")

    def test_a_limit_changed_by_someone_else_is_left_alone_at_stop(self):
        files = make_cpus(self.tmp.name, count=2)
        cap = self.cap()
        cap.start()
        files[0].write_text("1500000", encoding="ascii")
        cap.stop()
        self.assertEqual([path.read_text() for path in files], ["1500000", "3300000"])

    def test_an_interrupt_in_the_middle_of_the_writes_still_restores_the_first_ones(self):
        files = make_cpus(self.tmp.name, count=3)
        real_write = cpucap._write_text
        calls = []

        def interrupted(path, value):
            calls.append(path)
            if len(calls) == 2:
                raise KeyboardInterrupt
            return real_write(path, value)

        cap = self.cap()
        with mock.patch.object(cpucap, "_write_text", side_effect=interrupted), mock.patch.object(cpucap.atexit, "register") as register:
            with self.assertRaises(KeyboardInterrupt):
                cap.start()
        register.assert_called_once_with(cap.stop)
        cap.stop()
        self.assertEqual([path.read_text() for path in files], ["3300000"] * 3)

    def test_a_restore_that_fails_is_reported_and_kept_for_another_try(self):
        files = make_cpus(self.tmp.name)
        cap = self.cap()
        cap.start()
        real_write = cpucap._write_text
        with mock.patch.object(cpucap, "_write_text", return_value="read-only"):
            cap.stop()
        self.assertTrue(any("could not be put back" in message for message in self.messages))
        cap.stop()
        self.assertEqual(files[0].read_text(), "3300000")
        self.assertIsNotNone(real_write)

    def test_apply_returns_the_failures(self):
        make_cpus(self.tmp.name, count=1)
        with mock.patch.object(cpucap, "_write_text", return_value="read-only"):
            self.assertEqual(self.cap().apply(), ["read-only"])


def load_cli():
    path = str(Path(__file__).resolve().parent.parent / "tools" / "cpucap")
    loader = importlib.machinery.SourceFileLoader("cpucap_cli", path)
    spec = importlib.util.spec_from_loader("cpucap_cli", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cli = load_cli()
        for patcher in (mock.patch.object(cpucap, "CPU_ROOT", self.tmp.name), mock.patch("sys.stdout", io.StringIO()),
                        mock.patch("sys.stderr", io.StringIO())):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.files = make_cpus(self.tmp.name, count=2)

    def test_set_writes_the_cap_and_succeeds(self):
        self.assertEqual(self.cli.main(["set", "2200"]), 0)
        self.assertEqual([path.read_text() for path in self.files], ["2200000"] * 2)

    def test_set_fails_with_a_nonzero_code_when_nothing_can_be_written(self):
        with mock.patch.object(cpucap, "_write_text", return_value="read-only"):
            self.assertEqual(self.cli.main(["set", "2200"]), 1)

    def test_reset_puts_the_hardware_maximum_back(self):
        self.cli.main(["set", "2200"])
        self.assertEqual(self.cli.main(["reset"]), 0)
        self.assertEqual([path.read_text() for path in self.files], ["3300000"] * 2)

    def test_reset_fails_with_a_nonzero_code_when_a_limit_cannot_be_written(self):
        with mock.patch.object(cpucap, "_write_text", return_value="read-only"):
            self.assertEqual(self.cli.main(["reset"]), 1)

    def test_set_refuses_a_value_outside_the_range(self):
        with self.assertRaises(SystemExit):
            self.cli.main(["set", "100"])


class CreateTests(unittest.TestCase):
    def make(self, cfg):
        return cpucap.create(cfg, log=lambda message: None)

    def test_off_when_absent_or_zero(self):
        self.assertIsNone(self.make(None))
        self.assertIsNone(self.make({}))
        self.assertIsNone(self.make({"max_temp_c": 80}))
        self.assertIsNone(self.make({"cpu_max_mhz": 0}))

    def test_a_value_makes_a_cap(self):
        self.assertEqual(self.make({"cpu_max_mhz": 2200}).max_mhz, 2200.0)

    def test_nonsense_values_are_refused(self):
        for bad in (True, "2200", 100, 20000, -5):
            with self.assertRaises(ValueError):
                self.make({"cpu_max_mhz": bad})


if __name__ == "__main__":
    unittest.main()
