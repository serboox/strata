import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock
from xml.etree import ElementTree

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import lever_matrix

CACHE_TEXT = "// a comment\nSTRATA_IGPU_VULKAN:BOOL=ON\nOTHER:STRING=x\n"
SAME_ANSWER = "def cache(): pass"


def scratch(case: unittest.TestCase) -> Path:
    directory = tempfile.TemporaryDirectory()
    case.addCleanup(directory.cleanup)
    return Path(directory.name)


def definition(name: str, number: int, variants: list | None = None, **extra) -> dict:
    data = {"name": name, "lever": number, "title": f"{name} title", "variants": variants if variants is not None else [{"name": "on", "env": {}}],
            "needs": {"build_option": "", "model_file": "", "root": False}, "quality_check": "", "parity_check": "", "notes": ""}
    data.update(extra)
    return data


FIXTURES = [
    definition("cooling", 1, kind="system", variants=[{"name": "cap", "launcher_args": ["--cpu-max-mhz", "2200"]}]),
    definition("overlap", 2, variants=[{"name": "on", "env": {"STRATA_OVERLAP": "1"}},
                                       {"name": "deep", "env": {"STRATA_OVERLAP": "2"}, "engine_args": ["--overlap-depth", "2"]}],
               parity_check="greedy answers are identical"),
    definition("igpu_experts", 3, variants=[{"name": "on", "env": {"STRATA_IGPU": "1"}, "launcher_args": ["--igpu"]}],
               needs={"build_option": "-DSTRATA_IGPU_VULKAN=ON", "model_file": "", "root": False}),
    definition("experts_used", 4, variants=[{"name": "k6", "env": {"STRATA_Q36_K": "6"}, "engine_args": ["--no-igpu"]}]),
    definition("dense_q6k", 5, kind="model-file", variants=[{"name": "q6k", "model_file": "models/q6k.gguf"}], quality_check="tools/ppl"),
]


class FakeRunner(lever_matrix.Runner):
    def __init__(self, speeds=None, answers=None, failing=None, quiet=True):
        self.speeds, self.answers, self.failing, self.quiet = speeds or {}, answers or {}, failing or {}, quiet
        self.stop_on = None
        self.leftover = []
        self.restores = 0
        self.keep_cap = False
        self.temperature_now = 50.0
        self.now = 0.0
        self.cap = "3300000"
        self.commands = []

    def execute(self, command, timeout):
        self.commands.append(command)
        label = command[command.index("--label") + 1]
        self.cap = "2200000"
        if label == self.stop_on:
            raise lever_matrix.SessionStopped("stopped by signal SIGTERM")
        if label in self.failing:
            return self.failing[label]
        rows = [{"warmup": True, "tok_s": 40.0, "stat_lines": 1, "answer": "warm", "peak_c": 60.0}]
        rows += [{"warmup": False, "tok_s": speed, "stat_lines": 1, "answer": self.answers.get(label, SAME_ANSWER), "peak_c": 70.0 + index}
                 for index, speed in enumerate(self.speeds.get(label, [30.0, 31.0, 29.0]))]
        Path(command[command.index("--out") + 1]).write_text(json.dumps({"label": label, "budget": "none", "rows": rows, "summary": {}, "aborted": None,
                                                                          "cleanup_problems": []}))
        return 0

    def sleep(self, seconds):
        self.now += seconds

    def clock(self):
        return self.now

    def temperature(self):
        return self.temperature_now

    def load(self):
        return 0.2

    def machine(self, max_load):
        return {"quiet": self.quiet, "problems": [] if self.quiet else ["load average 3.00 is above 1.5"]}

    def processes(self):
        return self.leftover if self.commands else ["1: /usr/bin/llama-server already running"]

    def cap_snapshot(self):
        return {"cpu0": self.cap}

    def cap_restore(self, snapshot):
        self.restores += 1
        if not self.keep_cap:
            self.cap = snapshot["cpu0"]
        return []

    def timestamp(self):
        return "20260101-000000"


class WorldCase(unittest.TestCase):
    def setUp(self):
        self.root = scratch(self)
        self.levers_dir = self.root / "levers.d"
        self.levers_dir.mkdir()
        (self.root / "build").mkdir()
        (self.root / "build" / "CMakeCache.txt").write_text(CACHE_TEXT)
        (self.root / "models").mkdir()
        for name in ("q6k.gguf", "default.gguf", "other.gguf"):
            (self.root / "models" / name).write_bytes(b"x")
        for data in FIXTURES:
            self.write(data)

    def write(self, data: dict, filename: str | None = None) -> None:
        (self.levers_dir / f"{filename or data['name']}.json").write_text(json.dumps(data))

    def argv(self, command: str, *options: str) -> list:
        common = [command, "--levers-dir", str(self.levers_dir), "--engine-root", str(self.root)]
        if command == "list":
            return common
        return [*common, "--model", str(self.root / "models" / "default.gguf"), "--out-dir", str(self.root / "out"), *options]

    def plan(self, *options: str) -> lever_matrix.Plan:
        args = lever_matrix.parse_arguments(self.argv("plan", *options), FakeRunner())
        return lever_matrix.build_plan(lever_matrix.load_levers(self.levers_dir), args)

    def labels(self, *options: str) -> list:
        return [run.label for run in self.plan(*options).runs]

    def invoke(self, runner: FakeRunner, command: str, *options: str) -> tuple:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            status = lever_matrix.main(self.argv(command, *options), runner)
        return status, out.getvalue(), err.getvalue()


class LoadTests(WorldCase):
    def test_levers_are_sorted_by_number_and_the_kind_defaults_to_engine(self):
        levers = lever_matrix.load_levers(self.levers_dir)
        self.assertEqual([lever.name for lever in levers], ["cooling", "overlap", "igpu_experts", "experts_used", "dense_q6k"])
        self.assertEqual([lever.kind for lever in levers], ["system", "engine", "engine", "engine", "model-file"])

    def test_the_definitions_of_this_checkout_are_valid(self):
        if not lever_matrix.LEVERS_DIR.is_dir():
            self.skipTest("no levers.d directory")
        for lever in lever_matrix.load_levers(lever_matrix.LEVERS_DIR):
            self.assertTrue(lever.variants)

    def test_a_missing_directory_and_broken_json_are_named(self):
        with self.assertRaisesRegex(lever_matrix.PlanError, "does not exist"):
            lever_matrix.load_levers(self.root / "nowhere")
        (self.levers_dir / "broken.json").write_text("{")
        with self.assertRaisesRegex(lever_matrix.PlanError, "broken.json"):
            lever_matrix.load_levers(self.levers_dir)

    def test_a_duplicate_name_names_both_files(self):
        self.write(definition("overlap", 2), "second")
        with self.assertRaisesRegex(lever_matrix.PlanError, r"second.json.*overlap.*already used by overlap.json"):
            lever_matrix.load_levers(self.levers_dir)

    def test_bad_definitions_get_a_message_with_the_file(self):
        cases = [({"lever": 0}, "'lever' must be an integer"), ({"lever": 7}, "'lever' must be an integer"), ({"lever": "2"}, "'lever' must be an integer"),
                 ({"lever": True}, "'lever' must be an integer"), ({"lever": 2.0}, "'lever' must be an integer"),
                 ({"name": "Bad-Name"}, "lowercase letters"), ({"kind": "weird"}, "unknown kind 'weird'"), ({"variants": []}, "non-empty list"),
                 ({"variants": [{"name": "a", "env": {"K": 1}}]}, "strings to strings"), ({"variants": [{"name": "a:b"}]}, "the name may hold only"),
                 ({"variants": [{"name": "a"}, {"name": "a"}]}, "must be unique"), ({"variants": [{"name": "a", "engine_args": "x"}]}, "list of strings"),
                 ({"kind": "model-file"}, "needs a model_file"), ({"variants": [{"name": "a", "model_file": "x"}]}, "only allowed for kind model-file"),
                 ({"needs": {"root": "yes"}}, "needs.root"), ({"title": ""}, "'title' must be a non-empty string")]
        for change, message in cases:
            with self.subTest(change=change):
                data = {**definition("bad", 2), **change}
                (self.levers_dir / "bad.json").write_text(json.dumps(data))
                with self.assertRaisesRegex(lever_matrix.PlanError, f"bad.json.*{message}|bad.json: {message}"):
                    lever_matrix.load_levers(self.levers_dir)

    def test_a_top_level_list_is_refused(self):
        (self.levers_dir / "list.json").write_text("[]")
        with self.assertRaisesRegex(lever_matrix.PlanError, "list.json: the top level"):
            lever_matrix.load_levers(self.levers_dir)


class AvailabilityTests(WorldCase):
    def cache(self, text: str | None) -> None:
        path = self.root / "build" / "CMakeCache.txt"
        path.unlink()
        if text is not None:
            path.write_text(text)

    def test_a_build_option_must_be_on_in_the_cmake_cache(self):
        self.assertIsNone(lever_matrix.build_option_problem("-DSTRATA_IGPU_VULKAN=ON", self.root))
        self.assertIsNone(lever_matrix.build_option_problem("STRATA_IGPU_VULKAN=ON", self.root))
        self.assertIsNone(lever_matrix.build_option_problem("", self.root))
        self.cache("STRATA_IGPU_VULKAN:BOOL=OFF\n")
        self.assertRegex(lever_matrix.build_option_problem("-DSTRATA_IGPU_VULKAN=ON", self.root), "is OFF.*ON is needed")

    def test_a_missing_cache_or_entry_says_why(self):
        self.cache("OTHER:STRING=x\n")
        self.assertRegex(lever_matrix.build_option_problem("-DSTRATA_IGPU_VULKAN=ON", self.root), "is not in .*CMakeCache.txt")
        self.cache(None)
        self.assertRegex(lever_matrix.build_option_problem("-DSTRATA_IGPU_VULKAN=ON", self.root), "CMakeCache.txt is missing")

    def test_a_model_file_is_found_relative_absolute_or_by_pattern(self):
        self.assertEqual(lever_matrix.resolve_model_file("models/q6k.gguf", self.root), self.root / "models" / "q6k.gguf")
        self.assertEqual(lever_matrix.resolve_model_file(str(self.root / "models" / "q6k.gguf"), Path("/elsewhere")), self.root / "models" / "q6k.gguf")
        self.assertEqual(lever_matrix.resolve_model_file("models/q6*.gguf", self.root), self.root / "models" / "q6k.gguf")
        self.assertIsNone(lever_matrix.resolve_model_file("models/missing.gguf", self.root))
        self.assertIsNone(lever_matrix.resolve_model_file("nodir/*.gguf", self.root))

    def test_levers_report_their_own_reasons(self):
        levers = {lever.name: lever for lever in lever_matrix.load_levers(self.levers_dir)}
        self.assertRegex(lever_matrix.lever_problem(levers["cooling"], self.root), "cooling_kit.py")
        self.assertIsNone(lever_matrix.variant_problem(levers["dense_q6k"], levers["dense_q6k"].variants[0], self.root))
        (self.root / "models" / "q6k.gguf").unlink()
        self.assertRegex(lever_matrix.variant_problem(levers["dense_q6k"], levers["dense_q6k"].variants[0], self.root), "models/q6k.gguf does not exist")

    def test_the_list_shows_every_lever_and_why_one_is_unavailable(self):
        self.cache("STRATA_IGPU_VULKAN:BOOL=OFF\n")
        status, out, _ = self.invoke(FakeRunner(), "list")
        self.assertEqual(status, 0)
        for text in ("lever 1: cooling [system, run tools/cooling_kit.py]", "lever 3: igpu_experts [engine]", "lever 5: dense_q6k [model-file]",
                     "UNAVAILABLE: build option STRATA_IGPU_VULKAN is OFF", "launcher args (not passed) --igpu", 'variant deep', '(first: used for "on")',
                     "status: available"):
            self.assertIn(text, out)


class PlanTests(WorldCase):
    def test_each_runs_every_available_lever_alone_between_two_baselines(self):
        self.assertEqual(self.labels("--each"), ["baseline", "overlap", "igpu_experts", "experts_used", "dense_q6k", "baseline_again"])
        plan = self.plan("--each")
        self.assertEqual([run.index for run in plan.runs], list(range(6)))
        self.assertEqual([run.role for run in plan.runs], ["baseline", "lever", "lever", "lever", "lever", "baseline"])

    def test_each_with_variants_adds_every_variant(self):
        self.assertEqual(self.labels("--each", "--variants"),
                         ["baseline", "overlap", "overlap.deep", "igpu_experts", "experts_used", "dense_q6k", "baseline_again"])

    def test_the_system_lever_is_never_part_of_a_sweep_and_is_reported(self):
        for option in ("--each", "--all", "--pairs"):
            self.assertFalse(any("cooling" in label for label in self.labels(option)))
        self.assertEqual(self.plan("--each").skipped[0][0], "cooling")

    def test_all_puts_every_available_lever_into_one_run(self):
        plan = self.plan("--all")
        self.assertEqual([run.label for run in plan.runs], ["baseline", "overlap+igpu_experts+experts_used+dense_q6k", "baseline_again"])
        combined = plan.runs[1]
        self.assertEqual(combined.env, {"STRATA_POOL_SPIN_US": "0", "STRATA_IQ_VBMI": "1", "STRATA_OVERLAP": "1", "STRATA_IGPU": "1", "STRATA_Q36_K": "6"})
        self.assertEqual(combined.engine_args, ("--no-igpu",))

    def test_pairs_runs_every_pair_of_available_levers(self):
        plan = self.plan("--pairs")
        labels = [run.label for run in plan.runs]
        self.assertEqual(len(labels), 4 * 3 // 2 + 2)
        self.assertEqual(labels[1], "overlap+igpu_experts")
        self.assertEqual(labels[-2], "experts_used+dense_q6k")
        self.assertEqual(labels[0], "baseline")
        self.assertEqual(labels[-1], "baseline_again")

    def test_pairs_need_two_available_levers(self):
        self.write(definition("overlap", 2, kind="system"))
        for name in ("igpu_experts", "experts_used"):
            (self.levers_dir / f"{name}.json").unlink()
        with self.assertRaisesRegex(lever_matrix.PlanError, "no lever or pair of levers"):
            self.plan("--pairs")

    def test_a_named_lever_uses_its_first_variant_and_a_named_variant_another(self):
        self.assertEqual(self.plan("--levers", "overlap").runs[1].env["STRATA_OVERLAP"], "1")
        run = self.plan("--levers", "overlap:deep,experts_used").runs[1]
        self.assertEqual((run.label, run.env["STRATA_OVERLAP"], run.engine_args), ("overlap.deep+experts_used", "2", ("--overlap-depth", "2", "--no-igpu")))

    def test_a_model_file_lever_replaces_the_model_of_its_run_only(self):
        plan = self.plan("--levers", "dense_q6k,overlap")
        self.assertEqual([run.model.name for run in plan.runs], ["default.gguf", "q6k.gguf", "default.gguf"])

    def test_two_model_file_levers_with_different_files_are_refused_and_the_same_file_is_fine(self):
        self.write(definition("dense_other", 5, kind="model-file", variants=[{"name": "o", "model_file": "models/other.gguf"}]))
        with self.assertRaisesRegex(lever_matrix.PlanError, "dense_q6k and dense_other.*cannot be combined"):
            self.plan("--levers", "dense_q6k,dense_other")
        with self.assertRaisesRegex(lever_matrix.PlanError, "cannot be combined"):
            self.plan("--all")
        self.write(definition("dense_other", 5, kind="model-file", variants=[{"name": "o", "model_file": "models/q6k.gguf"}]))
        self.assertEqual(self.plan("--levers", "dense_q6k,dense_other").runs[1].model.name, "q6k.gguf")

    def test_a_lever_that_cannot_run_here_is_refused_with_the_reason(self):
        (self.root / "build" / "CMakeCache.txt").write_text("STRATA_IGPU_VULKAN:BOOL=OFF\n")
        with self.assertRaisesRegex(lever_matrix.PlanError, "lever igpu_experts is not available here: build option STRATA_IGPU_VULKAN is OFF"):
            self.plan("--levers", "overlap,igpu_experts")
        with self.assertRaisesRegex(lever_matrix.PlanError, "cooling is not available here: a system lever.*cooling_kit.py"):
            self.plan("--levers", "cooling")
        self.assertNotIn("igpu_experts", self.labels("--each"))
        self.assertIn("igpu_experts", dict(self.plan("--each").skipped))

    def test_the_command_prints_the_reason_and_not_a_traceback(self):
        (self.root / "models" / "q6k.gguf").unlink()
        status, out, err = self.invoke(FakeRunner(), "plan", "--levers", "dense_q6k")
        self.assertEqual((status, out), (2, ""))
        self.assertTrue(err.startswith("lever-matrix: lever dense_q6k is not available here: the model file models/q6k.gguf does not exist"))
        self.assertNotIn("Traceback", err)

    def test_selection_mistakes_get_clear_errors(self):
        for options, message in ((("--levers", "nope"), "unknown lever 'nope'; known levers: cooling, overlap"),
                                 (("--levers", "overlap:zzz"), "lever overlap has no variant 'zzz'; variants: on, deep"),
                                 (("--levers", "overlap,overlap:deep"), "lever overlap is selected twice"),
                                 (("--all", "--variants"), "--variants works only with --each")):
            with self.subTest(options=options), self.assertRaisesRegex(lever_matrix.PlanError, message):
                self.plan(*options)

    def test_the_selection_flags_exclude_each_other_and_one_is_needed(self):
        for options in ((), ("--each", "--all")):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                lever_matrix.parse_arguments(self.argv("plan", *options), FakeRunner())

    def test_the_default_output_directory_has_the_start_time(self):
        argv = ["run", "--levers-dir", str(self.levers_dir), "--engine-root", str(self.root), "--each"]
        args = lever_matrix.parse_arguments(argv, FakeRunner())
        self.assertEqual(args.out_dir, self.root / "run" / "experiments" / "levers" / "20260101-000000")
        self.assertEqual(args.model, self.root / lever_matrix.DEFAULT_MODEL)

    def test_the_help_holds_the_schema_and_the_merge_rule(self):
        text = lever_matrix.build_parser().format_help()
        for part in ('"launcher_args"', '"kind": "engine" | "system" | "model-file"', "needs.model_file", "first variant of a lever", "baseline env",
                     "different values from two sources is an error naming both sources"):
            self.assertIn(part, text)


class MergeTests(WorldCase):
    def test_a_conflict_between_two_levers_names_both_sources(self):
        self.write(definition("clash", 4, variants=[{"name": "x", "env": {"STRATA_OVERLAP": "3"}}]))
        with self.assertRaisesRegex(lever_matrix.PlanError, "STRATA_OVERLAP is '1' from lever overlap:on and '3' from lever clash:x"):
            self.plan("--all")
        with self.assertRaisesRegex(lever_matrix.PlanError, "STRATA_OVERLAP"):
            self.plan("--levers", "overlap,clash")

    def test_the_same_value_from_two_sources_is_fine(self):
        self.write(definition("twin", 4, variants=[{"name": "x", "env": {"STRATA_OVERLAP": "1"}}]))
        self.assertEqual(self.plan("--levers", "overlap,twin").runs[1].env["STRATA_OVERLAP"], "1")

    def test_the_user_env_conflicts_with_a_lever_and_with_the_baseline(self):
        with self.assertRaisesRegex(lever_matrix.PlanError, "STRATA_OVERLAP is '1' from lever overlap:on and '9' from --env"):
            self.plan("--levers", "overlap", "--env", "STRATA_OVERLAP=9")
        with self.assertRaisesRegex(lever_matrix.PlanError, "STRATA_POOL_SPIN_US is '0' from the baseline env and '50' from --env"):
            self.plan("--each", "--env", "STRATA_POOL_SPIN_US=50")

    def test_the_baseline_env_can_be_replaced_and_the_user_env_reaches_every_run(self):
        plan = self.plan("--levers", "overlap", "--baseline-env", "STRATA_POOL_SPIN_US=50", "--env", "STRATA_POOL_SPIN_US=50", "--env", "EXTRA=1")
        self.assertTrue(all(run.env["STRATA_POOL_SPIN_US"] == "50" and run.env["EXTRA"] == "1" for run in plan.runs))

    def test_identical_engine_arguments_are_passed_once_in_order(self):
        merged = lever_matrix.merge_engine_args([("--a", "1"), ("--b", "1", "--a", "1"), ("--c", "-1")])
        self.assertEqual(merged, ("--a", "1", "--b", "1", "--c", "-1"))
        self.assertEqual(lever_matrix.merge_engine_args([("--no-igpu",), ("--no-igpu",)]), ("--no-igpu",))
        self.assertEqual(self.plan("--levers", "experts_used", "--engine-args", "--no-igpu --x 2").runs[1].engine_args, ("--no-igpu", "--x", "2"))


class CommandTests(WorldCase):
    def command_of(self, run: lever_matrix.Run, *options: str) -> list:
        args = lever_matrix.parse_arguments(self.argv("run", "--each", *options), FakeRunner())
        return lever_matrix.quant_bench_command(run, args, Path("/out/raw/x.json"))

    def test_the_command_for_quant_bench_carries_the_settings_of_the_run(self):
        run = self.plan("--levers", "overlap:deep,dense_q6k", "--engine-args=--x").runs[1]
        command = self.command_of(run, "--prewarm")
        self.assertEqual(command[:2], [sys.executable, str(lever_matrix.QUANT_BENCH)])
        pairs = {command[index]: command[index + 1] for index in range(2, len(command) - 1) if command[index].startswith("--")}
        self.assertEqual((pairs["--model"], pairs["--label"], pairs["--cpu-max-mhz"], pairs["--workers"], pairs["--max-temp"]),
                         (str(self.root / "models" / "q6k.gguf"), "overlap.deep+dense_q6k", "2200", "6", "80"))
        self.assertEqual((pairs["--cool"], pairs["--cool-timeout"], pairs["--repeats"], pairs["--warmup"], pairs["--out"]), ("55", "900", "3", "2", "/out/raw/x.json"))
        self.assertIn("--prewarm", command)
        self.assertNotIn("--force", command)
        self.assertIn("STRATA_OVERLAP=2", command)
        self.assertIn("STRATA_POOL_SPIN_US=0", command)
        self.assertIn("--engine-args=--overlap-depth 2 --x", command)

    def test_an_engine_flag_alone_stays_one_argument_and_launcher_args_are_not_passed(self):
        run = self.plan("--levers", "igpu_experts,experts_used").runs[1]
        command = self.command_of(run)
        self.assertIn("--engine-args=--no-igpu", command)
        self.assertNotIn("--igpu", command)
        _, out, _ = self.invoke(FakeRunner(), "plan", "--levers", "igpu_experts")
        self.assertIn("launcher args of igpu_experts:on, not passed: --igpu", out)

    def test_the_plan_prints_every_command_the_hints_and_the_output_directory(self):
        status, out, _ = self.invoke(FakeRunner(), "plan", "--each")
        self.assertEqual(status, 0)
        self.assertEqual(out.count("command: "), 6)
        self.assertIn(f"output {self.root / 'out'}", out)
        self.assertIn("hint overlap: parity check: greedy answers are identical; quality check (not run): none", out)
        self.assertIn("hint dense_q6k: parity check: none; quality check (not run): tools/ppl", out)
        self.assertIn("left out: cooling:", out)


def record(index: int, label: str, speeds: list, role: str = "lever", answer: str = SAME_ANSWER, peaks: list | None = None, **plan) -> dict:
    peaks = peaks or [70.0] * len(speeds)
    rows = [{"warmup": True, "tok_s": 50.0, "stat_lines": 1, "answer": "warm", "peak_c": 99.0}]
    rows += [{"warmup": False, "tok_s": speed, "stat_lines": 1, "answer": answer, "peak_c": peak} for speed, peak in zip(speeds, peaks)]
    info = {"index": index, "role": role, "levers": [] if role == "baseline" else [{"lever": label, "variant": "on", "first": True, "kind": "engine"}],
            "returncode": 0, "started_cool": True, **plan}
    return {"label": label, "result": {"label": label, "budget": "none", "rows": rows, "aborted": None, "cleanup_problems": []}, "plan": info}


class SummaryTests(unittest.TestCase):
    def summary(self, records: list) -> dict:
        args = type("Args", (), {"out_dir": Path("/out")})()
        plan = lever_matrix.Plan(tuple(types.SimpleNamespace(picks=()) for _ in records), ())
        return lever_matrix.summarize(records, plan, {"stopped": None, "cap_problems": [], "leftover_processes": []}, args)

    def test_the_median_and_the_hottest_measured_reading(self):
        item = record(0, "baseline", [30.0, 20.0, 40.0], "baseline", peaks=[71.0, 79.5, 75.0])
        self.assertEqual(lever_matrix.median_speed(item["result"]), 30.0)
        self.assertEqual(lever_matrix.peak_temperature(item["result"]), 79.5)
        self.assertIsNone(lever_matrix.median_speed({"rows": []}))

    def test_a_request_with_a_doubled_log_line_or_a_hot_start_is_not_counted(self):
        item = record(0, "baseline", [30.0, 31.0], "baseline")
        item["result"]["rows"].append({"warmup": False, "tok_s": 99.0, "stat_lines": 2})
        item["result"]["rows"].append({"warmup": False, "tok_s": 98.0, "stat_lines": 1, "started_cool": False})
        self.assertEqual(lever_matrix.median_speed(item["result"]), 30.5)

    def test_delta_is_against_the_first_baseline_until_the_second_exists_then_against_their_mean(self):
        first = record(0, "baseline", [30.0] * 3, "baseline")
        lever = record(1, "overlap", [33.0] * 3)
        rows = self.summary([first, lever])["runs"]
        self.assertEqual((rows[0]["delta_pct"], rows[1]["delta_pct"]), (0.0, 10.0))
        again = record(2, "baseline_again", [28.0] * 3, "baseline")
        summary = self.summary([first, lever, again])
        self.assertEqual(summary["baseline_mean_tok_s"], 29.0)
        self.assertEqual([row["delta_pct"] for row in summary["runs"]], [3.4, 13.8, -3.4])

    def test_drift_is_flagged_unstable_above_five_percent_either_way(self):
        def status(second: float) -> tuple:
            summary = self.summary([record(0, "baseline", [30.0] * 3, "baseline"), record(1, "baseline_again", [second] * 3, "baseline")])
            return summary["baseline_drift_pct"], summary["drift_status"]
        self.assertEqual(status(31.0), (3.3, "stable"))
        self.assertEqual(status(31.5), (5.0, "stable"))
        self.assertEqual(status(28.0), (-6.7, "UNSTABLE"))
        self.assertEqual(status(32.0), (6.7, "UNSTABLE"))
        self.assertEqual(self.summary([record(0, "baseline", [30.0] * 3, "baseline")])["drift_status"], "n/a")

    def test_parity_compares_the_first_measured_answer_with_the_first_baseline(self):
        records = [record(0, "baseline", [30.0] * 3, "baseline"), record(1, "same", [30.0] * 3), record(2, "other", [30.0] * 3, answer="different"),
                   record(3, "empty", [30.0] * 3, answer=""), record(4, "baseline_again", [30.0] * 3, "baseline", answer="drifted")]
        self.assertEqual([row["parity"] for row in self.summary(records)["runs"]], ["n/a", "identical", "DIFFERS", "n/a", "DIFFERS"])
        self.assertEqual(lever_matrix.parity_status(None, "x"), "n/a")

    def test_quality_is_never_run_and_problems_are_listed(self):
        failed = record(1, "overlap", [], returncode=3)
        failed["result"]["aborted"] = "the server died"
        hot = record(2, "igpu", [30.0] * 3, started_cool=False)
        rows = self.summary([record(0, "baseline", [30.0] * 3, "baseline"), failed, hot])["runs"]
        self.assertEqual({row["quality"] for row in rows}, {"not run"})
        self.assertEqual(rows[0]["problems"], [])
        self.assertEqual(rows[1]["problems"], ["aborted: the server died"])
        self.assertEqual(rows[2]["problems"], ["started hot"])

    def test_the_text_table_has_the_columns_and_flags(self):
        records = [record(0, "baseline", [30.0] * 3, "baseline", peaks=[70.0, 78.0, 72.0]), record(1, "baseline_again", [20.0] * 3, "baseline")]
        text = lever_matrix.format_summary(self.summary(records))
        for part in ("run  label", "tok/s", "delta", "max C", "parity", "quality", "problems", "30.0", "78", "not run", "(UNSTABLE)", "-33.3 %"):
            self.assertIn(part, text)


class SessionTests(WorldCase):
    def session(self, runner: FakeRunner, *options: str) -> tuple:
        return self.invoke(runner, "run", "--each", "--cool-timeout", "30", *options)

    def test_a_session_runs_in_plan_order_and_writes_every_result(self):
        runner = FakeRunner(speeds={"baseline": [30.0] * 3, "overlap": [33.0] * 3, "baseline_again": [30.3] * 3})
        status, out, _ = self.session(runner)
        self.assertEqual(status, 0)
        labels = [command[command.index("--label") + 1] for command in runner.commands]
        self.assertEqual(labels, ["baseline", "overlap", "igpu_experts", "experts_used", "dense_q6k", "baseline_again"])
        out_dir = self.root / "out"
        self.assertEqual(sorted(path.name for path in out_dir.iterdir()), ["00_baseline.json", "01_overlap.json", "02_igpu_experts.json", "03_experts_used.json",
                                                                       "04_dense_q6k.json", "05_baseline_again.json", "chart.svg", "raw", "summary.json",
                                                                       "summary.txt"])
        item = json.loads((out_dir / "01_overlap.json").read_text())
        self.assertEqual(item["plan"]["env"], {"STRATA_POOL_SPIN_US": "0", "STRATA_IQ_VBMI": "1", "STRATA_OVERLAP": "1"})
        self.assertEqual(item["plan"]["levers"][0]["lever"], "overlap")
        self.assertEqual(item["plan"]["command"], runner.commands[1])
        self.assertIn("rows", item)
        self.assertEqual(item["plan"]["parity_check"], {"overlap": "greedy answers are identical"})
        summary = json.loads((out_dir / "summary.json").read_text())
        self.assertEqual((summary["runs_done"], summary["runs"][1]["delta_pct"], summary["runs"][1]["parity"], summary["drift_status"]), (6, 9.5, "identical", "stable"))
        ElementTree.fromstring((out_dir / "chart.svg").read_text())
        self.assertIn("overlap", (out_dir / "chart.svg").read_text())
        self.assertIn("overlap", (out_dir / "summary.txt").read_text())
        self.assertIn("baseline mean", out)

    def test_the_summary_is_rewritten_after_every_run(self):
        seen = []
        runner = FakeRunner()
        original = runner.execute

        def watch(command, timeout):
            summary = self.root / "out" / "summary.json"
            seen.append(json.loads(summary.read_text())["runs_done"] if summary.exists() else None)
            return original(command, timeout)
        runner.execute = watch
        self.session(runner)
        self.assertEqual(seen, [None, 1, 2, 3, 4, 5])

    def test_a_failed_lever_run_does_not_stop_the_session_but_the_exit_status_says_so(self):
        runner = FakeRunner(failing={"overlap": 3})
        status, _, _ = self.session(runner)
        self.assertEqual((status, len(runner.commands)), (3, 6))
        summary = json.loads((self.root / "out" / "summary.json").read_text())
        self.assertEqual(summary["runs"][1]["problems"], ["aborted: quant_bench left no result (exit status 3)"])

    def test_a_failed_baseline_stops_the_session(self):
        runner = FakeRunner(failing={"baseline": 3})
        status, _, _ = self.session(runner)
        self.assertEqual((status, len(runner.commands)), (3, 1))
        summary = json.loads((self.root / "out" / "summary.json").read_text())
        self.assertIn("the baseline run 0 failed", summary["session"]["stopped"])
        self.assertEqual(runner.restores, 1)

    def test_an_aborted_result_file_is_read_and_kept(self):
        runner = FakeRunner()
        original = runner.execute

        def abort(command, timeout):
            original(command, timeout)
            out = Path(command[command.index("--out") + 1])
            if "--label" in command and command[command.index("--label") + 1] == "overlap":
                data = json.loads(out.read_text())
                data["aborted"] = "the server died during the run"
                out.with_name(out.name + ".aborted").write_text(json.dumps(data))
                out.unlink()
                return 3
            return 0
        runner.execute = abort
        self.session(runner)
        summary = json.loads((self.root / "out" / "summary.json").read_text())
        self.assertEqual(summary["runs"][1]["problems"], ["aborted: the server died during the run"])

    def test_a_package_that_does_not_cool_stops_the_session_unless_continue_hot(self):
        runner = FakeRunner()
        runner.temperature_now = 70.0
        status, _, _ = self.session(runner)
        self.assertEqual((status, runner.commands), (3, []))
        self.assertIn("stayed above 55 C", json.loads((self.root / "out" / "summary.json").read_text())["session"]["stopped"])
        (self.root / "out" / "summary.json").unlink()
        for path in list((self.root / "out").iterdir()):
            if path.is_file():
                path.unlink()
        (self.root / "out" / "raw").rmdir()
        (self.root / "out").rmdir()
        runner = FakeRunner()
        runner.temperature_now = 70.0
        status, _, _ = self.session(runner, "--continue-hot")
        self.assertEqual((status, len(runner.commands)), (3, 6))
        item = json.loads((self.root / "out" / "01_overlap.json").read_text())
        self.assertIs(item["plan"]["started_cool"], False)

    def test_the_machine_must_be_quiet_unless_forced(self):
        runner = FakeRunner(quiet=False)
        status, _, err = self.session(runner)
        self.assertEqual((status, runner.commands, (self.root / "out").exists()), (2, [], False))
        self.assertIn("the machine is not quiet: load average 3.00", err)
        status, _, _ = self.session(runner, "--force")
        self.assertEqual((status, len(runner.commands)), (0, 6))
        self.assertIn("--force", runner.commands[0])

    def test_a_missing_model_or_an_existing_output_directory_stops_before_anything_runs(self):
        (self.root / "out").mkdir()
        runner = FakeRunner()
        self.assertEqual(self.session(runner)[0], 2)
        (self.root / "out").rmdir()
        (self.root / "models" / "default.gguf").unlink()
        status, _, err = self.session(runner)
        self.assertEqual((status, runner.commands), (2, []))
        self.assertIn("default.gguf does not exist", err)

    def test_the_cpu_cap_is_checked_and_put_back_and_leftovers_are_reported(self):
        runner = FakeRunner()
        self.session(runner)
        self.assertEqual((runner.cap, runner.restores), ("3300000", 1))
        (self.root / "out").rename(self.root / "first")
        runner = FakeRunner()
        runner.keep_cap = True
        runner.leftover = ["1: /usr/bin/llama-server already running", "4242: /x/strata-q36 --serve"]
        status, out, _ = self.session(runner)
        self.assertEqual(status, 3)
        self.assertIn("CPU CAP PROBLEM: cpu0 is 2200000, it was 3300000", out)
        self.assertIn("LEFTOVER PROCESS: 4242: /x/strata-q36 --serve", out)
        self.assertNotIn("LEFTOVER PROCESS: 1:", out)
        summary = json.loads((self.root / "out" / "summary.json").read_text())
        self.assertEqual(summary["session"]["leftover_processes"], ["4242: /x/strata-q36 --serve"])

    def test_a_stop_signal_ends_the_session_cleanly_with_what_exists(self):
        runner = FakeRunner()
        runner.stop_on = "igpu_experts"
        status, _, _ = self.session(runner)
        self.assertEqual((status, len(runner.commands)), (3, 3))
        summary = json.loads((self.root / "out" / "summary.json").read_text())
        self.assertEqual((summary["runs_done"], summary["session"]["stopped"]), (2, "stopped by signal SIGTERM"))
        self.assertTrue((self.root / "out" / "chart.svg").exists())
        self.assertEqual((runner.cap, runner.restores), ("3300000", 1))
        self.assertEqual(signal.getsignal(signal.SIGTERM), signal.SIG_DFL)

    def test_a_chart_that_cannot_be_rendered_as_png_is_reported_and_does_not_fail_the_session(self):
        runner = FakeRunner()
        with mock.patch.object(lever_matrix.bench_chart, "to_png", return_value=False) as render:
            status, _, err = self.session(runner, "--png")
        self.assertEqual(status, 0)
        self.assertEqual(render.call_count, 1)
        self.assertIn("the PNG could not be made", err)

    def test_a_dry_run_prints_the_plan_and_changes_nothing(self):
        runner = FakeRunner()
        status, out, _ = self.invoke(runner, "run", "--each", "--dry-run")
        self.assertEqual((status, runner.commands, runner.restores, (self.root / "out").exists()), (0, [], 0, False))
        self.assertIn(f"output {self.root / 'out'}", out)
        self.assertEqual(out.count("command: "), 6)


class ProcessTests(unittest.TestCase):
    def test_a_stop_signal_becomes_an_exception_and_the_handlers_come_back(self):
        previous = lever_matrix.install_stop_handlers()
        try:
            with self.assertRaisesRegex(lever_matrix.SessionStopped, "SIGTERM"):
                os.kill(os.getpid(), signal.SIGTERM)
        finally:
            lever_matrix.quant_bench.restore_handlers(previous)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous[signal.SIGTERM])

    def test_a_child_that_runs_too_long_is_stopped_gently_and_a_finished_one_reports_its_status(self):
        runner = lever_matrix.Runner()
        self.assertEqual(runner.execute([sys.executable, "-c", "import sys; sys.exit(7)"], 30), 7)
        started = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        runner.terminate(started, grace=10)
        self.assertEqual(started.returncode, -signal.SIGTERM)
        self.assertEqual(runner.execute([sys.executable, "-c", "import time; time.sleep(30)"], 0.3), lever_matrix.TIMEOUT_STATUS)

    def test_a_stop_signal_during_a_run_stops_the_child_and_reaches_the_session(self):
        runner = lever_matrix.Runner()
        children = []
        original = subprocess.Popen

        def spawn(*arguments, **options):
            children.append(original(*arguments, **options))
            return children[-1]
        timer = threading.Timer(0.5, os.kill, [os.getpid(), signal.SIGTERM])
        previous = lever_matrix.install_stop_handlers()
        timer.start()
        try:
            with mock.patch.object(subprocess, "Popen", spawn), self.assertRaises(lever_matrix.SessionStopped):
                runner.execute([sys.executable, "-c", "import time; time.sleep(30)"], 30)
        finally:
            timer.cancel()
            lever_matrix.quant_bench.restore_handlers(previous)
        self.assertEqual(children[0].returncode, -signal.SIGTERM)

    def test_a_child_that_ignores_sigterm_is_killed_after_the_grace_period(self):
        child = subprocess.Popen([sys.executable, "-c", "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); time.sleep(30)"],
                                 stdout=subprocess.PIPE, text=True)
        self.assertEqual(child.stdout.readline().strip(), "ready")
        lever_matrix.Runner().terminate(child, grace=0.3)
        child.stdout.close()
        self.assertEqual(child.returncode, -signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
