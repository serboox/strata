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
