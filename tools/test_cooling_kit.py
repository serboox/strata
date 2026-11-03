import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock
from xml.etree import ElementTree

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import cooling_kit


def scratch(case):
    directory = tempfile.TemporaryDirectory()
    case.addCleanup(directory.cleanup)
    return Path(directory.name)


def put(path: Path, text) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text))


def fake_machine(case, fans=(0, 0, 0), energy=1_000_000, readable=True):
    root = scratch(case)
    put(root / "hwmon" / "hwmon7" / "name", "coretemp")
    for index, (label, millidegrees) in enumerate((("Package id 0", 70000), ("Core 0", 64000), ("Core 1", 68000)), start=1):
        put(root / "hwmon" / "hwmon7" / f"temp{index}_label", label)
        put(root / "hwmon" / "hwmon7" / f"temp{index}_input", millidegrees)
    put(root / "hwmon" / "hwmon6" / "name", "bitland_mifs")
    for index, (label, rpm) in enumerate(zip(("CPU", "GPU", "SYS"), fans), start=1):
        put(root / "hwmon" / "hwmon6" / f"fan{index}_label", label)
        put(root / "hwmon" / "hwmon6" / f"fan{index}_input", rpm)
    put(root / "thermal" / "thermal_zone0" / "type", "x86_pkg_temp")
    put(root / "thermal" / "thermal_zone0" / "temp", 72000)
    for cpu, khz in ((0, 2200000), (1, 2100000), (10, 2000000), (2, 2200000)):
        put(root / "cpu" / f"cpu{cpu}" / "cpufreq" / "scaling_cur_freq", khz)
    put(root / "power" / "intel-rapl:0" / "name", "package-0")
    put(root / "power" / "intel-rapl:0" / "energy_uj", energy)
    put(root / "power" / "intel-rapl:0" / "max_energy_range_uj", 10_000_000)
    if not readable:
        (root / "power" / "intel-rapl:0" / "energy_uj").chmod(0)
        case.addCleanup((root / "power" / "intel-rapl:0" / "energy_uj").chmod, 0o644)
    put(root / "profile", "balanced\n")
    clock = iter([0.0, 2.0, 4.0, 6.0, 8.0])
    return root, cooling_kit.Sensors(root / "hwmon", root / "cpu", root / "thermal", root / "power", root / "profile", gpu=lambda: 55.0, clock=lambda: next(clock))


def sample(t, pkg, fan=None, power=None, mhz=2200, core=None):
    return {"t_s": t, "pkg_c": pkg, "core_max_c": core if core is not None else pkg, "gpu_c": 50.0, "freq_mean_mhz": mhz, "freq_min_mhz": mhz,
            "freq_max_mhz": mhz, "power_w": power, "profile": "balanced", "cpu_mhz": [mhz, mhz], "fans": {"CPU": fan, "GPU": fan, "SYS": fan}}


def row(tok_s, warmup=False, lines=1, t=None, wall=20.0, cool=True):
    result = {"prompt_name": "code", "warmup": warmup, "tok_s": tok_s, "stat_lines": lines, "started_cool": cool, "wall_s": wall}
    if t is not None:
        result["kit_t_s"] = t
    return result


def args(**changes):
    base = dict(model=Path("/m.gguf"), duration=300.0, max_tokens=900, workers=6, limit=80.0, start_temp=55.0, cool_timeout=1.0, period=2.0, env={"STRATA_POOL_SPIN_US": "0"},
                engine_args=[], prewarm=False, force=False, setup="on a stand", sample_only=None, png=False)
    base.update(changes)
    return Namespace(**base)


class SensorTests(unittest.TestCase):
    def test_every_sensor_is_read_from_its_files(self):
        _, sensors = fake_machine(self, fans=(3900, 0, 2500))
        reading = sensors.read(1.04)
        self.assertEqual((reading["t_s"], reading["pkg_c"], reading["core_max_c"], reading["gpu_c"]), (1.0, 72.0, 68.0, 55.0))
        self.assertEqual(reading["cpu_mhz"], [2200, 2100, 2200, 2000])
        self.assertEqual((reading["freq_min_mhz"], reading["freq_max_mhz"]), (2000, 2200))
        self.assertEqual(reading["fans"], {"CPU": 3900.0, "GPU": 0.0, "SYS": 2500.0})
        self.assertEqual(reading["profile"], "balanced")

    def test_power_is_the_energy_difference_over_time_and_survives_a_counter_wrap(self):
        root, sensors = fake_machine(self, energy=1_000_000)
        self.assertIsNone(sensors.power_w())
        put(root / "power" / "intel-rapl:0" / "energy_uj", 21_000_000 % 10_000_000 + 0)
        put(root / "power" / "intel-rapl:0" / "energy_uj", 2_000_000)
        self.assertEqual(sensors.power_w(), 0.5)
        put(root / "power" / "intel-rapl:0" / "energy_uj", 1_000_000)
        self.assertEqual(sensors.power_w(), 4.5)

    def test_an_unreadable_energy_counter_gives_no_power_and_is_reported(self):
        if os.geteuid() == 0:
            self.skipTest("root reads everything")
        _, sensors = fake_machine(self, readable=False)
        self.assertIsNone(sensors.power_w())
        self.assertFalse(sensors.rapl_readable())
        notes = cooling_kit.sensor_notes([sample(0, 60.0, fan=100)], sensors)
        self.assertTrue(any("not readable (root only)" in note and "sudo -A chmod a+r" in note and "SUDO_PROMPT" in note for note in notes))

    def test_a_machine_without_sensors_reads_nothing_and_does_not_fail(self):
        empty = scratch(self)
        sensors = cooling_kit.Sensors(empty, empty, empty, empty, empty / "profile", gpu=lambda: None)
        reading = sensors.read(0)
        self.assertEqual((reading["pkg_c"], reading["core_max_c"], reading["freq_mean_mhz"], reading["fans"], reading["profile"]), (None, None, None, {}, None))
        notes = cooling_kit.sensor_notes([reading], sensors)
        self.assertTrue(any("NOT READABLE" in note for note in notes))
        self.assertTrue(any("no fan sensor" in note for note in notes))


class FanTests(unittest.TestCase):
    def test_a_fan_that_reads_zero_all_run_is_an_unreliable_sensor(self):
        samples = [sample(t, 60.0, fan=0) for t in range(0, 10, 2)]
        report = cooling_kit.fan_report(samples, "CPU", 60.0)
        self.assertFalse(report["reliable"])
        self.assertIn("idle fan or a dead sensor", report["note"])

    def test_a_hot_package_with_a_zero_fan_says_the_sensor_is_not_wired(self):
        samples = [sample(t, 78.0, fan=0) for t in range(0, 10, 2)]
        report = cooling_kit.fan_report(samples, "CPU", 78.0)
        self.assertFalse(report["reliable"])
        self.assertIn("most likely not wired", report["note"])

    def test_a_fan_that_spins_is_reliable_with_mean_and_max(self):
        samples = [sample(t, 60.0, fan=rpm) for t, rpm in ((0, 0), (2, 3000), (4, 5000))]
        report = cooling_kit.fan_report(samples, "CPU", 60.0)
        self.assertEqual((report["reliable"], report["mean_rpm"], report["max_rpm"]), (True, 2666.7, 5000))

    def test_too_few_readings_are_not_believed(self):
        self.assertFalse(cooling_kit.fan_report([sample(0, 60.0, fan=3000)], "CPU", 60.0)["reliable"])


class AnalysisTests(unittest.TestCase):
    def test_time_to_a_threshold_is_the_first_sample_at_or_above_it(self):
        samples = [sample(0, 50.0), sample(2, 74.9), sample(4, 75.0), sample(6, 80.5)]
        self.assertEqual(cooling_kit.first_at_or_above(samples, 75.0), 4)
        self.assertEqual(cooling_kit.first_at_or_above(samples, 80.0), 6)
        self.assertIsNone(cooling_kit.first_at_or_above(samples, 90.0))

    def test_sustained_speed_is_the_median_of_the_second_half_of_usable_requests(self):
        rows = [row(35.0, warmup=True), row(34.0), row(32.0), row(28.0), row(27.0), row(99.0, lines=2), row(26.0, cool=False)]
        speeds = cooling_kit.split_speeds(rows)
        self.assertEqual((speeds["tok_s_first"], speeds["tok_s_sustained"], speeds["requests"]), (34.0, 27.5, 4))

    def test_no_usable_request_gives_no_speed(self):
        self.assertEqual(cooling_kit.split_speeds([row(10.0, warmup=True)]), {"tok_s_first": None, "tok_s_sustained": None, "requests": 0})

    def test_the_step_summary_has_times_means_fans_and_speeds(self):
        samples = [sample(0, 55.0, fan=0, power=10.0), sample(60, 70.0, fan=3000, power=20.0), sample(120, 76.0, fan=4000, power=24.0), sample(180, 81.0, fan=5000)]
        rows = [row(30.0, t=80.0, wall=20.0), row(28.0, t=100.0), row(26.0, t=120.0)]
        step = {"index": 1, "label": "cap2200_balanced", "cap_mhz": 2200, "profile": "balanced"}
        summary = cooling_kit.summarize_step(step, samples, rows, "limit 80 C", True)
        self.assertEqual((summary["t75_s"], summary["t80_s"]), (120, 180))
        self.assertEqual(summary["load_started_s"], 60.0)
        self.assertEqual((summary["t75_s_after_load"], summary["t80_s_after_load"]), (60.0, 120.0))
        self.assertEqual((summary["pkg_mean_c"], summary["pkg_max_c"]), (70.5, 81.0))
        self.assertEqual((summary["power_mean_w"], summary["power_max_w"]), (18.0, 24.0))
        self.assertEqual(summary["tok_s_sustained"], 27.0)
        cpu = next(fan for fan in summary["fans"] if fan["label"] == "CPU")
        self.assertEqual((cpu["mean_rpm"], cpu["max_rpm"], cpu["reliable"]), (3000.0, 5000, True))

    def test_a_step_that_never_gets_hot_has_no_threshold_times(self):
        summary = cooling_kit.summarize_step({"index": 1, "label": "a", "cap_mhz": 2200, "profile": "p"}, [sample(0, 50.0), sample(2, 52.0)], [], "duration", True)
        self.assertEqual((summary["t75_s"], summary["t80_s"], summary["tok_s_sustained"]), (None, None, None))


class PlanTests(unittest.TestCase):
    def cpucap(self):
        return mock.Mock(MIN_MHZ=800.0, MAX_MHZ=10000.0)

    def test_caps_are_read_in_ghz_and_checked(self):
        self.assertEqual(cooling_kit.parse_caps("2.2,2.4,2.6,2.8", self.cpucap()), [2200, 2400, 2600, 2800])
        for bad in ("2.2,x", "0.1", "2.2,2.2", "50"):
            with self.assertRaises(SystemExit):
                cooling_kit.parse_caps(bad, self.cpucap())

    def test_steps_keep_one_profile_together_and_are_numbered(self):
        steps = cooling_kit.build_steps([2200, 2400], ["balanced", "performance"])
        self.assertEqual([(step["index"], step["profile"], step["cap_mhz"]) for step in steps],
                         [(1, "balanced", 2200), (2, "balanced", 2400), (3, "performance", 2200), (4, "performance", 2400)])
        self.assertEqual(steps[0]["label"], "cap2200_balanced")

    def test_another_profile_without_the_flag_is_refused_with_the_command_to_run(self):
        with self.assertRaises(SystemExit) as stop:
            cooling_kit.check_profiles(["performance"], "balanced", ["balanced", "performance"], apply=False)
        message = str(stop.exception)
        self.assertIn("sudo -A tee /sys/firmware/acpi/platform_profile", message)
        self.assertIn("SUDO_PROMPT=", message)
        self.assertIn("--apply-profile", message)
        cooling_kit.check_profiles(["performance"], "balanced", ["balanced", "performance"], apply=True)
        cooling_kit.check_profiles(["balanced"], "balanced", ["balanced"], apply=False)

    def test_an_unknown_profile_is_refused(self):
        with self.assertRaises(SystemExit) as stop:
            cooling_kit.check_profiles(["turbo-mode"], "balanced", ["balanced", "performance"], apply=True)
        self.assertIn("no profile", str(stop.exception))

    def test_the_benchmark_command_is_the_standard_sustained_run(self):
        command = cooling_kit.bench_command(args(env={"STRATA_POOL_SPIN_US": "0", "X": "1"}, prewarm=True), {"label": "cap2200_balanced", "cap_mhz": 2200}, Path("/o/b.json"))
        text = " ".join(command)
        for piece in ("quant_bench.py", "--cpu-max-mhz 2200", "--workers 6", "--env STRATA_POOL_SPIN_US=0", "--env X=1", "--prewarm", "--warmup 0",
                      "--max-temp 80", "--cool 200", "--out /o/b.json", "--model /m.gguf"):
            self.assertIn(piece, text)
        self.assertNotIn("--force", command)
        self.assertNotIn("--prewarm", cooling_kit.bench_command(args(), {"label": "a", "cap_mhz": 2200}, Path("/o/b.json")))

    def test_enough_requests_are_asked_for_to_fill_the_duration(self):
        self.assertGreaterEqual(cooling_kit.requests_for(300) * cooling_kit.SECONDS_PER_REQUEST, 300)


class ProfileSwitchTests(unittest.TestCase):
    def test_the_switch_uses_sudo_with_the_askpass_flag_and_a_russian_reason(self):
        profile = scratch(self) / "profile"
        put(profile, "balanced")
        seen = {}

        def fake_run(command, **options):
            seen.update(command=command, options=options)
            profile.write_text(options["input"])
            return subprocess.CompletedProcess(command, 0, "", "")

        self.assertIsNone(cooling_kit.apply_profile("performance", profile, run=fake_run))
        self.assertEqual(seen["command"][:3], ["sudo", "-A", "tee"])
        self.assertEqual(seen["options"]["env"]["SUDO_PROMPT"], cooling_kit.SUDO_REASON_PROFILE)
        self.assertIn("timeout", seen["options"])

    def test_a_failed_or_ignored_switch_is_reported(self):
        profile = scratch(self) / "profile"
        put(profile, "balanced")
        failing = lambda command, **options: subprocess.CompletedProcess(command, 1, "", "denied")
        self.assertIn("denied", cooling_kit.apply_profile("performance", profile, run=failing))
        ignored = lambda command, **options: subprocess.CompletedProcess(command, 0, "", "")
        self.assertIn("after the write", cooling_kit.apply_profile("performance", profile, run=ignored))
        broken = mock.Mock(side_effect=OSError("no sudo"))
        self.assertIn("no sudo", cooling_kit.apply_profile("performance", profile, run=broken))


class FakeProcess:
    def __init__(self, lines, polls_until_exit=3, returncode=0):
        self.stderr = iter(lines)
        self.polls = 0
        self.polls_until_exit = polls_until_exit
        self.returncode = None
        self.final = returncode
        self.signals = []

    def poll(self):
        self.polls += 1
        if self.polls > self.polls_until_exit:
            self.returncode = self.final
        return self.returncode

    def send_signal(self, number):
        self.signals.append(number)
        self.returncode = 3

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.signals.append(signal.SIGKILL)


class StepTests(unittest.TestCase):
    def sensors(self, temperatures):
        readings = iter(temperatures)
        sensors = mock.Mock()
        sensors.read.side_effect = lambda elapsed: sample(elapsed, next(readings), fan=3000)
        return sensors

    def test_the_step_stops_the_benchmark_the_moment_the_package_hits_the_limit(self):
        process = FakeProcess([], polls_until_exit=99)
        samples = []
        ticks = iter(range(0, 1000, 2))
        reason = cooling_kit.sample_step(process, self.sensors([60.0, 70.0, 79.9, 80.0, 90.0]), args(), 0.0, samples, lambda: next(ticks), lambda seconds: None)
        self.assertEqual(reason, "limit 80 C")
        self.assertEqual(len(samples), 4)
        cooling_kit.stop_process(process)
        self.assertEqual(process.signals, [signal.SIGTERM])

    def test_the_step_ends_at_the_duration_or_when_the_benchmark_ends(self):
        ticks = iter([0, 100, 200, 300, 600])
        reason = cooling_kit.sample_step(FakeProcess([], 99), self.sensors([60.0] * 6), args(duration=300.0), 0.0, [], lambda: next(ticks), lambda seconds: None)
        self.assertEqual(reason, "duration")
        reason = cooling_kit.sample_step(FakeProcess([], 1), self.sensors([60.0] * 6), args(), 0.0, [], iter(range(0, 100, 2)).__next__, lambda seconds: None)
        self.assertEqual(reason, "benchmark ended")

    def test_a_whole_step_collects_rows_and_samples_and_stops_the_child(self):
        lines = ["quant-bench: read 16.0 GiB\n", json.dumps(row(30.0)) + "\n", json.dumps(row(28.0)) + "\n", "{not json\n"]
        process = FakeProcess(lines, polls_until_exit=2)
        popen = mock.Mock(return_value=process)
        ticks = iter(range(0, 1000, 5))
        out = scratch(self)
        summary, samples, rows = cooling_kit.run_step({"index": 1, "label": "cap2200_balanced", "cap_mhz": 2200, "profile": "balanced"}, args(), self.sensors([60.0, 62.0, 64.0, 66.0]),
                                                      out, lambda text: None, popen=popen, sleep=lambda seconds: None, clock=lambda: next(ticks),
                                                      wait_cool=lambda limit, timeout: (54.0, 12.0, True))
        self.assertEqual(len(rows), 2)
        self.assertEqual(summary["stop_reason"], "benchmark ended")
        self.assertEqual(summary["tok_s_sustained"], 28.0)
        self.assertEqual(summary["started_cool"], True)
        self.assertEqual(popen.call_args.args[0][popen.call_args.args[0].index("--label") + 1], "cap2200_balanced")

    def test_a_benchmark_that_dies_is_reported_with_its_last_words(self):
        process = FakeProcess(["quant-bench: the server did not become ready: boom\n"], polls_until_exit=1, returncode=3)
        ticks = iter(range(0, 1000, 5))
        summary, _, _ = cooling_kit.run_step({"index": 1, "label": "a", "cap_mhz": 2200, "profile": "p"}, args(), self.sensors([60.0] * 5), scratch(self), lambda text: None,
                                             popen=mock.Mock(return_value=process), sleep=lambda seconds: None, clock=lambda: next(ticks),
                                             wait_cool=lambda limit, timeout: (50.0, 0.0, True))
        self.assertIn("benchmark failed (3)", summary["stop_reason"])
        self.assertIn("did not become ready", summary["stop_reason"])

    def test_a_benchmark_that_does_not_stop_is_killed(self):
        process = FakeProcess([], 99)
        process.send_signal = lambda number: process.signals.append(number)
        process.wait = mock.Mock(side_effect=[subprocess.TimeoutExpired("q", 1), 0])
        cooling_kit.stop_process(process, wait_s=0.01)
        self.assertEqual(process.signals, [signal.SIGTERM, signal.SIGKILL])


class OutputTests(unittest.TestCase):
    def summaries(self):
        steps = []
        for index, (cap, speed, pkg, fan) in enumerate(((2200, 29.5, 76.0, 0), (2400, 25.0, 80.5, 0)), start=1):
            samples = [sample(t, 50.0 + t / 10, fan=fan) for t in range(0, 300, 20)]
            samples.append(sample(300, pkg, fan=fan))
            summary = cooling_kit.summarize_step({"index": index, "label": f"cap{cap}_balanced", "cap_mhz": cap, "profile": "balanced"}, samples,
                                                 [row(speed + 1, t=100.0), row(speed, t=200.0)], "duration", True)
            steps.append((summary, samples))
        return steps

    def test_the_table_fits_one_screen_and_marks_unreliable_fans(self):
        table = cooling_kit.format_table([summary for summary, _ in self.summaries()])
        self.assertTrue(all(len(line) <= 120 for line in table.splitlines()))
        self.assertIn("n/a", table)
        self.assertIn("cap2200_balanced", table)
        self.assertIn("29.5", table)

    def test_the_chart_is_valid_svg_with_the_steps_and_the_limit(self):
        pairs = self.summaries()
        svg = cooling_kit.cooling_chart([summary for summary, _ in pairs], {summary["label"]: samples for summary, samples in pairs}, 80.0, "Title <b>", "note & more")
        root = ElementTree.fromstring(svg)
        self.assertTrue(root.tag.endswith("svg"))
        for text in ("cap2200_balanced", "cap2400_balanced", "80 C", "75 C", "29.5"):
            self.assertIn(text, svg)

    def test_the_samples_csv_has_one_row_per_sample_with_fans_and_cores(self):
        path = scratch(self) / "samples.csv"
        cooling_kit.write_samples_csv(path, "a", [sample(0, 50.0, fan=100), sample(2, 51.0, fan=200)], header=True)
        cooling_kit.write_samples_csv(path, "b", [sample(0, 52.0, fan=300)], header=False)
        lines = path.read_text().splitlines()
        self.assertEqual(len(lines), 4)
        self.assertTrue(lines[0].startswith("step,t_s,pkg_c,core_max_c,gpu_c,freq_mean_mhz"))
        for column in ("fan_CPU_rpm", "fan_GPU_rpm", "fan_SYS_rpm", "cpu0_mhz", "cpu1_mhz", "power_w", "profile"):
            self.assertIn(column, lines[0])
        self.assertTrue(lines[3].startswith("b,0,52.0"))


class MainTests(unittest.TestCase):
    def run_main(self, *arguments):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = cooling_kit.main(list(arguments))
        return code, output.getvalue()

    def test_dry_run_prints_the_plan_and_runs_and_writes_nothing(self):
        model = scratch(self) / "m.gguf"
        model.write_bytes(b"x")
        out = scratch(self) / "out"
        fake = mock.Mock()
        fake.read.return_value = sample(0, 50.0, fan=0)
        fake.profile.return_value = "balanced"
        fake.rapl_dir = None
        with mock.patch.object(cooling_kit, "Sensors", return_value=fake), mock.patch.object(subprocess, "Popen", side_effect=AssertionError("must not start")), \
                mock.patch.object(subprocess, "run", side_effect=AssertionError("must not run sudo")):
            code, text = self.run_main("--model", str(model), "--setup", "on a stand", "--dry-run", "--out-dir", str(out))
        self.assertEqual(code, 0)
        self.assertFalse(out.exists())
        self.assertIn("4 steps", text)
        for cap in ("2200", "2400", "2600", "2800"):
            self.assertIn(f"--cpu-max-mhz {cap}", text)
        self.assertIn("'on a stand'", text)
        self.assertIn("a fan that reads 0 here may be idle", text)

    def test_sample_only_reports_the_sensors_and_starts_no_benchmark(self):
        fake = mock.Mock()
        fake.read.side_effect = lambda elapsed: sample(elapsed, 55.0, fan=0)
        fake.rapl_dir = None
        ticks = iter([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        lines = []
        with mock.patch.object(subprocess, "Popen", side_effect=AssertionError("must not start")):
            samples = cooling_kit.sample_only(args(sample_only=3.0, period=1.0), fake, say=lines.append, sleep=lambda seconds: None, clock=lambda: next(ticks))
        text = "\n".join(lines)
        self.assertEqual(len(samples), 3)
        self.assertIn("sampled 3 times", text)
        self.assertIn("fan CPU", text)
        self.assertIn("UNRELIABLE", text)

    def test_the_sample_only_option_runs_no_benchmark_and_needs_no_model(self):
        with mock.patch.object(cooling_kit, "sample_only", return_value=[]) as sampler, mock.patch.object(subprocess, "Popen", side_effect=AssertionError("must not start")):
            code, _ = self.run_main("--sample-only", "3", "--model", "/nonexistent/model.gguf")
        self.assertEqual(code, 0)
        sampler.assert_called_once()
        with self.assertRaises(SystemExit):
            cooling_kit.main(["--sample-only", "0"])

    def test_a_profile_listed_twice_is_refused(self):
        model = scratch(self) / "m.gguf"
        model.write_bytes(b"x")
        fake = mock.Mock()
        fake.profile.return_value = "balanced"
        with mock.patch.object(cooling_kit, "Sensors", return_value=fake), self.assertRaises(SystemExit) as stop:
            cooling_kit.main(["--model", str(model), "--profiles", "current,balanced", "--dry-run"])
        self.assertIn("twice", str(stop.exception))

    def test_a_missing_model_is_refused(self):
        with self.assertRaises(SystemExit):
            cooling_kit.main(["--model", "/nonexistent/model.gguf"])


class RunAllTests(unittest.TestCase):
    def test_everything_the_kit_changed_is_put_back_and_the_results_are_written(self):
        limits = scratch(self)
        cap_file = limits / "cpu0" / "cpufreq" / "scaling_max_freq"
        put(cap_file, "3300000")
        saved = {str(cap_file): "3300000"}
        out = scratch(self) / "result"
        steps = cooling_kit.build_steps([2200, 2400], ["balanced", "performance"])
        calls = []

        def fake_step(step, run_args, sensors, out_dir, say, **kwargs):
            cap_file.write_text(str(step["cap_mhz"] * 1000))
            calls.append((step["label"], active["profile"]))
            samples = [sample(0, 55.0, fan=1000), sample(2, 60.0, fan=1500), sample(4, 65.0, fan=2000)]
            return cooling_kit.summarize_step(step, samples, [row(30.0, t=10.0), row(29.0, t=30.0)], "duration", True), samples, []

        active = {"profile": "balanced"}

        def fake_apply(name, *rest, **options):
            active["profile"] = name
            return None

        with mock.patch.object(cooling_kit, "run_step", side_effect=fake_step), mock.patch.object(cooling_kit, "apply_profile", side_effect=fake_apply), \
                mock.patch.object(cooling_kit.machine_state, "cpu_cap_snapshot", return_value=saved), mock.patch.object(cooling_kit.machine_state, "running_processes", return_value=[]), \
                contextlib.redirect_stdout(io.StringIO()):
            code = cooling_kit.run_all(args(setup="stand"), steps, mock.Mock(), out, "balanced", say=lambda text: None)
        self.assertEqual(code, 0)
        self.assertEqual(cap_file.read_text(), "3300000")
        self.assertEqual(active["profile"], "balanced")
        self.assertEqual([label for label, _ in calls], [step["label"] for step in steps])
        for name in ("samples.csv", "result.json", "table.txt", "chart.svg"):
            self.assertTrue((out / name).exists(), name)
        result = json.loads((out / "result.json").read_text())
        self.assertEqual(result["setup"], "stand")
        self.assertEqual(len(result["steps"]), 4)
        self.assertEqual(result["restored"]["profile"], "back to balanced")
        self.assertEqual(result["restored"]["cpu_cap"], "unchanged")

    def test_an_interrupted_run_still_restores_and_keeps_the_finished_steps(self):
        limits = scratch(self)
        cap_file = limits / "cpu0" / "cpufreq" / "scaling_max_freq"
        put(cap_file, "3300000")
        out = scratch(self) / "result"
        steps = cooling_kit.build_steps([2200, 2400], ["balanced"])
        state = {"count": 0}

        def fake_step(step, run_args, sensors, out_dir, say, **kwargs):
            state["count"] += 1
            cap_file.write_text("2200000")
            if state["count"] == 2:
                raise SystemExit("stopped by signal 15")
            samples = [sample(0, 55.0, fan=1000), sample(2, 60.0, fan=1500), sample(4, 65.0, fan=2000)]
            return cooling_kit.summarize_step(step, samples, [row(30.0)], "duration", True), samples, []

        with mock.patch.object(cooling_kit, "run_step", side_effect=fake_step), mock.patch.object(cooling_kit.machine_state, "cpu_cap_snapshot", return_value={str(cap_file): "3300000"}), \
                mock.patch.object(cooling_kit.machine_state, "running_processes", return_value=[]), contextlib.redirect_stdout(io.StringIO()):
            code = cooling_kit.run_all(args(), steps, mock.Mock(), out, "balanced", say=lambda text: None)
        self.assertEqual(code, 0)
        self.assertEqual(cap_file.read_text(), "3300000")
        self.assertEqual(len(json.loads((out / "result.json").read_text())["steps"]), 1)


if __name__ == "__main__":
    unittest.main()
