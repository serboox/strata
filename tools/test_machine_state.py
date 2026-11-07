import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import machine_state

MEMINFO = "MemTotal:       32000000 kB\nMemAvailable:   20971520 kB\nSwapTotal:       8388608 kB\nSwapFree:        8000000 kB\nBuffers: 1 kB\n"


def scratch(case):
    directory = tempfile.TemporaryDirectory()
    case.addCleanup(directory.cleanup)
    return Path(directory.name)


class ReadTests(unittest.TestCase):
    def test_the_usage_windows_are_read_and_empty_ones_dropped(self):
        path = scratch(self) / "usage.json"
        path.write_text(json.dumps({"five_hour": {"utilization": 12.0, "resets_at": "t"}, "spend": {"utilization": None}, "other": 3}))
        self.assertEqual(machine_state.read_usage(path), {"five_hour": {"utilization": 12.0, "resets_at": "t"}})

    def test_a_missing_or_broken_usage_file_gives_nothing(self):
        directory = scratch(self)
        (directory / "bad.json").write_text("{")
        self.assertEqual(machine_state.read_usage(directory / "missing.json"), {})
        self.assertEqual(machine_state.read_usage(directory / "bad.json"), {})

    def test_memory_is_in_gib(self):
        self.assertEqual(machine_state.meminfo_gib(MEMINFO), {"total_gib": 30.5, "available_gib": 20.0, "swap_used_gib": 0.37})

    def test_processes_are_found_by_their_command_line(self):
        proc = scratch(self)
        for pid, command in (("101", b"/x/strata-q36 --serve\0--native\0m.gguf\0"), ("102", b"/usr/bin/firefox\0"), ("103", b"python3\0serve/server.py\0--port\x008080\0")):
            (proc / pid).mkdir()
            (proc / pid / "cmdline").write_bytes(command)
        (proc / "self").mkdir()
        found = machine_state.running_processes(proc)
        self.assertEqual(sorted(entry.split(":")[0] for entry in found), ["101", "103"])


    def test_a_watchdog_warning_that_names_a_server_is_not_a_server(self):
        proc = scratch(self)
        for pid, command in (("201", b"bash\0/x/cpu-hog-watchdog\0--prompt\0614066\0llama-server\0"), ("202", b"notify-send\0-a\0cpu-hog-watchdog\0llama-server is stuck\0"),
                             ("203", b"/x/llama-server\0-m\0m.gguf\0"), ("204", b"python3\0-I\0tools/machine_state.py\0llama-server\0")):
            (proc / pid).mkdir()
            (proc / pid / "cmdline").write_bytes(command)
        found = machine_state.running_processes(proc)
        self.assertEqual([entry.split(":")[0] for entry in found], ["203"])

    def test_a_server_whose_path_or_arguments_contain_a_helper_name_is_still_a_server(self):
        proc = scratch(self)
        commands = (b"/opt/cpu-hog-watchdog/llama-server\0-m\0m.gguf\0", b"llama-server\0-m\0/models/cpu-hog-watchdog.gguf\0", b"llama-server\0-m\0/models/machine_state.gguf\0")
        for index, command in enumerate(commands):
            (proc / str(300 + index)).mkdir()
            (proc / str(300 + index) / "cmdline").write_bytes(command)
        found = machine_state.running_processes(proc)
        self.assertEqual(sorted(entry.split(":")[0] for entry in found), ["300", "301", "302"])


class CpuCapTests(unittest.TestCase):
    def fake_cpus(self, limits):
        root = scratch(self)
        for index, value in enumerate(limits):
            (root / f"cpu{index}" / "cpufreq").mkdir(parents=True)
            (root / f"cpu{index}" / "cpufreq" / "scaling_max_freq").write_text(f"{value}\n")
        (root / "cpufreq").mkdir()
        return root

    def test_the_snapshot_holds_every_limit_and_the_restore_puts_changed_ones_back(self):
        root = self.fake_cpus([3300000, 3300000])
        saved = machine_state.cpu_cap_snapshot(root)
        self.assertEqual(sorted(saved.values()), ["3300000", "3300000"])
        (root / "cpu1" / "cpufreq" / "scaling_max_freq").write_text("2200000")
        self.assertEqual(machine_state.cpu_cap_restore(saved), [])
        self.assertEqual((root / "cpu1" / "cpufreq" / "scaling_max_freq").read_text(), "3300000")

    def test_a_file_that_cannot_be_written_is_reported(self):
        root = self.fake_cpus([3300000])
        saved = machine_state.cpu_cap_snapshot(root)
        (root / "cpu0" / "cpufreq" / "scaling_max_freq").write_text("2200000")
        with mock.patch.object(Path, "write_text", side_effect=PermissionError("denied")):
            failures = machine_state.cpu_cap_restore(saved)
        self.assertEqual(len(failures), 1)
        self.assertIn("denied", failures[0])

    def test_a_machine_without_cpufreq_has_an_empty_snapshot(self):
        self.assertEqual(machine_state.cpu_cap_snapshot(scratch(self)), {})


class CollectTests(unittest.TestCase):
    def collect(self, load=0.5, available_kib=20971520, turbo="off", processes=(), usage=None, **arguments):
        meminfo = f"MemTotal: 32000000 kB\nMemAvailable: {available_kib} kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n"
        with mock.patch.object(machine_state.os, "getloadavg", return_value=(load, load, load)), \
                mock.patch.object(Path, "read_text", return_value=meminfo), \
                mock.patch.object(machine_state.quant_bench, "package_temperature", return_value=52.0), \
                mock.patch.object(machine_state.quant_bench, "gpu_temperature", return_value=48.0), \
                mock.patch.object(machine_state.quant_bench, "turbo_state", return_value=turbo), \
                mock.patch.object(machine_state, "running_processes", return_value=list(processes)), \
                mock.patch.object(machine_state, "read_usage", return_value=usage or {}):
            return machine_state.collect(**arguments)

    def test_a_quiet_machine_has_no_problems(self):
        state = self.collect()
        self.assertTrue(state["quiet"])
        self.assertEqual((state["cpu_temp_c"], state["gpu_temp_c"], state["turbo"]), (52.0, 48.0, "off"))

    def test_every_problem_is_named(self):
        state = self.collect(load=3.0, available_kib=1048576, turbo="on", processes=["1: strata-q36"], usage={"five_hour": {"utilization": 99.5, "resets_at": "t"}},
                             min_available_gib=24)
        self.assertFalse(state["quiet"])
        self.assertEqual(len(state["problems"]), 5)

    def test_the_command_exits_with_one_when_the_machine_is_not_quiet(self):
        with mock.patch.object(machine_state, "collect", return_value={"quiet": False, "problems": ["x"]}), mock.patch.object(sys, "argv", ["machine_state"]):
            self.assertEqual(machine_state.main(), 1)

    def test_an_existing_output_file_is_refused(self):
        out = scratch(self) / "state.json"
        out.write_text("x")
        with mock.patch.object(sys, "argv", ["machine_state", "--out", str(out)]), self.assertRaises(SystemExit):
            machine_state.main()


if __name__ == "__main__":
    unittest.main()
