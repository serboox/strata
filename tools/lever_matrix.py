import argparse
import itertools
import json
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_chart
import machine_state
import quant_bench

TOOLS = Path(__file__).resolve().parent
ENGINE_ROOT = TOOLS.parent
LEVERS_DIR = TOOLS / "levers.d"
QUANT_BENCH = TOOLS / "quant_bench.py"
DEFAULT_MODEL = Path("models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-IQ4_XS-mtp.gguf")
OUT_ROOT = Path("run/experiments/levers")
RAW_DIR = "raw"
CACHE_FILE = Path("build/CMakeCache.txt")
SYSTEM_TOOL = "tools/cooling_kit.py"
BASELINE_ENV = {"STRATA_POOL_SPIN_US": "0", "STRATA_IQ_VBMI": "1"}   # what the launcher sets for the IQ models; without the kernels the baseline is about half as fast
BASELINE_LABEL, BASELINE_AGAIN_LABEL = "baseline", "baseline_again"
ROLE_BASELINE, ROLE_LEVER = "baseline", "lever"
KIND_ENGINE, KIND_SYSTEM, KIND_MODEL_FILE = "engine", "system", "model-file"
KINDS = (KIND_ENGINE, KIND_SYSTEM, KIND_MODEL_FILE)
LEVER_NUMBERS = range(1, 6)
LEVER_NAME = re.compile(r"[a-z][a-z0-9_]*")
VARIANT_NAME = re.compile(r"[A-Za-z0-9_.-]+")
FILE_UNSAFE = re.compile(r"[^A-Za-z0-9_.+-]")
ARGUMENT_NUMBER = re.compile(r"-?\d+(\.\d+)?")
UNSTABLE_DRIFT_PCT = 5.0
GRACE_SECONDS = 120.0
KILL_WAIT_SECONDS = 30.0
LOAD_POLL_SECONDS = 10.0
TIMEOUT_STATUS = -1
EXIT_PLAN_ERROR, EXIT_SESSION_PROBLEM = 2, 3
PARITY_SAME, PARITY_DIFFERS, PARITY_NA = "identical", "DIFFERS", "n/a"
QUALITY_STATUS = "not run"
TEXT_LIMIT = 70
TABLE_LEGEND = ("tok/s: median of the usable measured requests; delta: against the mean of the baseline runs so far; "
                "max C: hottest reading of the measured requests.")

SCHEMA_HELP = """\
lever definitions: one JSON file per lever in the levers directory (tools/levers.d/*.json):
  {"name": "<unique lowercase_underscores>", "lever": <1..5>, "title": "<one line>",
   "kind": "engine" | "system" | "model-file"   (optional, default "engine"),
   "variants": [{"name": "<short>", "env": {"STRATA_...": "..."}, "engine_args": ["..."], "launcher_args": ["..."],
                 "model_file": "<optional path, relative to the engine root or absolute>"}],
   "needs": {"build_option": "<cmake -D flag such as -DSTRATA_IGPU_VULKAN=ON, or empty>", "model_file": "<hint or path pattern, or empty>",
             "root": false},
   "quality_check": "<tool path or empty>", "parity_check": "<how to prove identical output or empty>", "notes": "<what to watch>"}
kinds:
  engine      measured here through quant_bench with the variant's env and engine_args (launcher_args are only printed).
  system      not run here: use tools/cooling_kit.py; never part of --each, --all or --pairs.
  model-file  measured with the variant's model_file (else needs.model_file) in place of --model; the file must exist.
              It combines with engine levers; two model-file levers with different files cannot be combined.
The first variant of a lever is used when the lever is asked "on": --levers overlap; another one: --levers overlap:other_variant.
"""
MERGE_HELP = """\
environment of a run = baseline env (default STRATA_POOL_SPIN_US=0 and STRATA_IQ_VBMI=1, replaced by --baseline-env) + the env of each selected variant + --env.
The same key with different values from two sources is an error naming both sources; the same value twice is fine.
Identical engine arguments from several sources are passed once, in order.
"""


class PlanError(Exception):
    """A problem with the lever definitions, the selection or the start conditions, shown without a traceback."""


class SessionStopped(Exception):
    """A stop signal arrived while a session was running."""


@dataclass(frozen=True)
class Variant:
    name: str
    env: dict
    engine_args: tuple
    launcher_args: tuple
    model_file: str


@dataclass(frozen=True)
class Lever:
    name: str
    number: int
    title: str
    kind: str
    variants: tuple
    build_option: str
    model_hint: str
    needs_root: bool
    quality_check: str
    parity_check: str
    notes: str
    source: str


@dataclass(frozen=True)
class Pick:
    lever: Lever
    variant: Variant

    @property
    def spec(self) -> str:
        return f"{self.lever.name}:{self.variant.name}"

    @property
    def label(self) -> str:
        return self.lever.name if self.variant == self.lever.variants[0] else f"{self.lever.name}.{self.variant.name}"


@dataclass(frozen=True)
class Run:
    index: int
    label: str
    picks: tuple
    env: dict
    engine_args: tuple
    model: Path

    @property
    def role(self) -> str:
        return ROLE_LEVER if self.picks else ROLE_BASELINE

    @property
    def stem(self) -> str:
        return f"{self.index:02d}_{FILE_UNSAFE.sub('_', self.label)}"


@dataclass(frozen=True)
class Plan:
    runs: tuple
    skipped: tuple


def text_field(data: dict, key: str, where: str, required: bool = False) -> str:
    value = data.get(key, "")
    if not isinstance(value, str) or (required and not value.strip()):
        raise PlanError(f"{where}: {key!r} must be {'a non-empty string' if required else 'a string'}")
    return value


def string_list(data: dict, key: str, where: str) -> tuple:
    value = data.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise PlanError(f"{where}: {key!r} must be a list of strings")
    return tuple(value)


def parse_variant(data: object, source: str) -> Variant:
    if not isinstance(data, dict):
        raise PlanError(f"{source}: every variant must be an object")
    name = text_field(data, "name", source, required=True)
    where = f"{source}, variant {name!r}"
    if not VARIANT_NAME.fullmatch(name):
        raise PlanError(f"{where}: the name may hold only letters, digits, '_', '.' and '-'")
    env = data.get("env", {})
    if not isinstance(env, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in env.items()):
        raise PlanError(f"{where}: 'env' must map strings to strings (quote numbers)")
    return Variant(name, dict(env), string_list(data, "engine_args", where), string_list(data, "launcher_args", where),
                   text_field(data, "model_file", where))


def parse_lever(data: object, source: str) -> Lever:
    if not isinstance(data, dict):
        raise PlanError(f"{source}: the top level must be an object")
    name = text_field(data, "name", source, required=True)
    if not LEVER_NAME.fullmatch(name):
        raise PlanError(f"{source}: the name {name!r} must be lowercase letters, digits and underscores, starting with a letter")
    number = data.get("lever")
    if isinstance(number, bool) or not isinstance(number, int) or number not in LEVER_NUMBERS:
        raise PlanError(f"{source}: 'lever' must be an integer from {LEVER_NUMBERS[0]} to {LEVER_NUMBERS[-1]}, not {number!r}")
    kind = data.get("kind", KIND_ENGINE)
    if kind not in KINDS:
        raise PlanError(f"{source}: unknown kind {kind!r}; known kinds: {', '.join(KINDS)}")
    raw_variants = data.get("variants")
    if not isinstance(raw_variants, list) or not raw_variants:
        raise PlanError(f"{source}: 'variants' must be a non-empty list")
    variants = tuple(parse_variant(item, source) for item in raw_variants)
    names = [variant.name for variant in variants]
    if len(set(names)) != len(names):
        raise PlanError(f"{source}: variant names must be unique, got {', '.join(names)}")
    needs = data.get("needs", {})
    if not isinstance(needs, dict) or not isinstance(needs.get("root", False), bool):
        raise PlanError(f"{source}: 'needs' must be an object and needs.root a boolean")
    lever = Lever(name, number, text_field(data, "title", source, required=True), kind, variants, text_field(needs, "build_option", f"{source}, needs"),
                  text_field(needs, "model_file", f"{source}, needs"), needs.get("root", False), text_field(data, "quality_check", source),
                  text_field(data, "parity_check", source), text_field(data, "notes", source), source)
    if kind == KIND_MODEL_FILE and not all(variant.model_file or lever.model_hint for variant in variants):
        raise PlanError(f"{source}: kind {KIND_MODEL_FILE} needs a model_file in every variant or in needs.model_file")
    if kind != KIND_MODEL_FILE and any(variant.model_file for variant in variants):
        raise PlanError(f"{source}: a variant model_file is only allowed for kind {KIND_MODEL_FILE}")
    return lever


def load_levers(directory: Path) -> list:
    if not directory.is_dir():
        raise PlanError(f"the lever directory {directory} does not exist")
    levers, sources = [], {}
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise PlanError(f"{path.name}: {error}") from error
        lever = parse_lever(data, path.name)
        if lever.name in sources:
            raise PlanError(f"{path.name}: the lever name {lever.name!r} is already used by {sources[lever.name]}")
        sources[lever.name] = path.name
        levers.append(lever)
    return sorted(levers, key=lambda lever: (lever.number, lever.name))


def build_option_problem(option: str, root: Path) -> str | None:
    """Why the CMake option is not set in the build of this engine root; None when the cache holds it with the wanted value."""
    if not option:
        return None
    name, _, wanted = option.removeprefix("-D").partition("=")
    wanted = wanted or "ON"
    cache = root / CACHE_FILE
    try:
        lines = cache.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return f"build option {name}={wanted} is not confirmed: {cache} is missing (configure with cmake -D{name}={wanted})"
    for line in lines:
        key, _, value = line.partition("=")
        if key.split(":")[0] == name:
            return None if value.strip().upper() == wanted.upper() else f"build option {name} is {value.strip()} in {cache}, {wanted} is needed"
    return f"build option {name} is not in {cache} (configure with cmake -D{name}={wanted})"


def resolve_model_file(spec: str, root: Path) -> Path | None:
    path = Path(spec).expanduser()
    path = path if path.is_absolute() else root / path
    if any(mark in spec for mark in "*?["):
        matches = sorted(path.parent.glob(path.name)) if path.parent.is_dir() else []
        return matches[0] if matches else None
    return path if path.is_file() else None


def lever_problem(lever: Lever, root: Path) -> str | None:
    if lever.kind == KIND_SYSTEM:
        return f"a system lever, not runnable here: run {SYSTEM_TOOL}"
    return build_option_problem(lever.build_option, root)


def variant_problem(lever: Lever, variant: Variant, root: Path) -> str | None:
    problem = lever_problem(lever, root)
    if problem is not None or lever.kind != KIND_MODEL_FILE:
        return problem
    spec = variant.model_file or lever.model_hint
    return None if resolve_model_file(spec, root) else f"the model file {spec} does not exist (a relative path is under {root})"


def explicit_picks(text: str, levers: list, root: Path) -> list:
    by_name = {lever.name: lever for lever in levers}
    picks, chosen = [], set()
    for spec in (part.strip() for part in text.split(",") if part.strip()):
        name, _, variant_name = spec.partition(":")
        lever = by_name.get(name)
        if lever is None:
            raise PlanError(f"unknown lever {name!r}; known levers: {', '.join(by_name) or 'none'}")
        if name in chosen:
            raise PlanError(f"lever {name} is selected twice")
        chosen.add(name)
        variant = next((item for item in lever.variants if item.name == variant_name), None) if variant_name else lever.variants[0]
        if variant is None:
            raise PlanError(f"lever {name} has no variant {variant_name!r}; variants: {', '.join(item.name for item in lever.variants)}")
        problem = variant_problem(lever, variant, root)
        if problem is not None:
            raise PlanError(f"lever {name} is not available here: {problem}")
        picks.append(Pick(lever, variant))
    if not picks:
        raise PlanError("--levers names no lever")
    return picks


def candidate_picks(levers: list, root: Path, every_variant: bool) -> tuple:
    """The available (lever, variant) pairs that --each, --all and --pairs work on, and why the others are left out."""
    picks, skipped = [], []
    for lever in levers:
        for variant in lever.variants if every_variant else lever.variants[:1]:
            problem = variant_problem(lever, variant, root)
            if problem is None:
                picks.append(Pick(lever, variant))
            else:
                skipped.append((f"{lever.name}:{variant.name}" if every_variant else lever.name, problem))
    return picks, skipped


def merge_env(sources: list) -> dict:
    merged, origin = {}, {}
    for source, env in sources:
        for key, value in env.items():
            if key in merged and merged[key] != value:
                raise PlanError(f"environment variable {key} is {merged[key]!r} from {origin[key]} and {value!r} from {source}")
            merged[key], origin[key] = value, source
    return merged


def argument_units(arguments: tuple) -> list:
    """Split engine arguments into a flag with its values, so a repeated value is not mistaken for a repeated flag."""
    units = []
    for argument in arguments:
        if not units or (argument.startswith("-") and not ARGUMENT_NUMBER.fullmatch(argument)):
            units.append([argument])
        else:
            units[-1].append(argument)
    return [tuple(unit) for unit in units]


def merge_engine_args(sources: list) -> tuple:
    units = dict.fromkeys(unit for arguments in sources for unit in argument_units(tuple(arguments)))
    return tuple(argument for unit in units for argument in unit)


def run_model(picks: list, args) -> Path:
    files = {pick.lever.name: resolve_model_file(pick.variant.model_file or pick.lever.model_hint, args.engine_root)
             for pick in picks if pick.lever.kind == KIND_MODEL_FILE}
    if len(set(files.values())) > 1:
        raise PlanError(f"the levers {' and '.join(files)} need different model files and a run has one model: they cannot be combined")
    return next(iter(files.values()), args.model)


def make_run(index: int, label: str, picks: list, args) -> Run:
    env = merge_env([("the baseline env", args.baseline_env), *((f"lever {pick.spec}", pick.variant.env) for pick in picks), ("--env", args.env)])
    engine_args = merge_engine_args([*(pick.variant.engine_args for pick in picks), args.engine_args])
    return Run(index, label, tuple(picks), env, engine_args, run_model(picks, args))


def build_plan(levers: list, args) -> Plan:
    if args.variants and not args.each:
        raise PlanError("--variants works only with --each")
    skipped = []
    if args.levers is not None:
        combos = [explicit_picks(args.levers, levers, args.engine_root)]
    else:
        picks, skipped = candidate_picks(levers, args.engine_root, args.variants)
        if args.each:
            combos = [[pick] for pick in picks]
        elif args.all:
            combos = [picks]
        else:
            combos = [list(pair) for pair in itertools.combinations(picks, 2)]
    if not any(combos):
        reasons = "".join(f"\n  {name}: {reason}" for name, reason in skipped)
        raise PlanError("no lever or pair of levers is available to measure" + (f"; left out:{reasons}" if reasons else ""))
    runs = [make_run(0, BASELINE_LABEL, [], args)]
    for combo in combos:
        runs.append(make_run(len(runs), "+".join(pick.label for pick in combo), combo, args))
    runs.append(make_run(len(runs), BASELINE_AGAIN_LABEL, [], args))
    return Plan(tuple(runs), tuple(skipped))


def quant_bench_command(run: Run, args, out: Path) -> list:
    command = [sys.executable, str(QUANT_BENCH), "--model", str(run.model), "--label", run.label, "--budget", args.budget, "--prompts", *args.prompts,
               "--repeats", str(args.repeats), "--warmup", str(args.warmup), "--max-tokens", str(args.max_tokens), "--cpu-max-mhz", f"{args.cpu_max_mhz:g}",
               "--workers", str(args.workers), "--max-temp", f"{args.max_temp:g}", "--cool", f"{args.cool:g}", "--cool-timeout", f"{args.cool_timeout:g}",
               "--max-load", f"{args.max_load:g}", "--engine-root", str(args.engine_root)]
    command += [flag for flag, enabled in (("--prewarm", args.prewarm), ("--force", args.force)) if enabled]
    for key, value in run.env.items():
        command += ["--env", f"{key}={value}"]
    if run.engine_args:
        command.append(f"--engine-args={shlex.join(run.engine_args)}")
    return [*command, "--out", str(out)]


def variant_text(variant: Variant) -> str:
    parts = [f"env {' '.join(f'{key}={value}' for key, value in variant.env.items())}" if variant.env else "",
             f"engine args {shlex.join(variant.engine_args)}" if variant.engine_args else "",
             f"launcher args (not passed) {shlex.join(variant.launcher_args)}" if variant.launcher_args else "",
             f"model file {variant.model_file}" if variant.model_file else ""]
    return "; ".join(part for part in parts if part) or "no change"


def describe_lever(lever: Lever, root: Path) -> list:
    kind = f"{lever.kind}, run {SYSTEM_TOOL}" if lever.kind == KIND_SYSTEM else lever.kind
    problem = variant_problem(lever, lever.variants[0], root)
    lines = [f"lever {lever.number}: {lever.name} [{kind}] ({lever.source})", f"  {lever.title}",
             "  status: " + ("available" if problem is None else f"UNAVAILABLE: {problem}")]
    for index, variant in enumerate(lever.variants):
        issue = variant_problem(lever, variant, root) if lever_problem(lever, root) is None else None
        marker = ' (first: used for "on")' if index == 0 else ""
        flag = f" [UNAVAILABLE: {issue}]" if issue else ""
        lines.append(f"  variant {variant.name}{marker}{flag}: {variant_text(variant)}")
    needs = [f"build option {lever.build_option}" if lever.build_option else "", f"model file {lever.model_hint}" if lever.model_hint else "",
             "root" if lever.needs_root else ""]
    lines += [f"  needs: {', '.join(item for item in needs if item)}" if any(needs) else "",
              f"  parity check: {lever.parity_check}" if lever.parity_check else "", f"  quality check: {lever.quality_check}" if lever.quality_check else "",
              f"  notes: {lever.notes}" if lever.notes else ""]
    return [line for line in lines if line]


def check_hints(plan: Plan) -> list:
    """Per lever in the plan: what proves identical output and which quality tool to run (this runner runs neither)."""
    levers = {pick.lever.name: pick.lever for run in plan.runs for pick in run.picks}
    return [{"lever": lever.name, "parity_check": lever.parity_check, "quality_check": lever.quality_check}
            for lever in levers.values() if lever.parity_check or lever.quality_check]


def describe_plan(plan: Plan, args) -> list:
    lines = [f"plan: {len(plan.runs)} runs, one at a time, output {args.out_dir}"]
    for run in plan.runs:
        lines.append(f"{run.index:>3}  {run.label}  ({'no lever' if not run.picks else ' + '.join(pick.spec for pick in run.picks)})")
        lines.append(f"       model: {run.model}{'' if run.model.exists() else '  (MISSING)'}")
        lines.append(f"       env: {' '.join(f'{key}={value}' for key, value in run.env.items()) or 'none'}")
        if run.engine_args:
            lines.append(f"       engine args: {shlex.join(run.engine_args)}")
        for pick in run.picks:
            if pick.variant.launcher_args:
                lines.append(f"       launcher args of {pick.spec}, not passed: {shlex.join(pick.variant.launcher_args)}")
        lines.append(f"       command: {shlex.join(quant_bench_command(run, args, args.out_dir / RAW_DIR / f'{run.stem}.json'))}")
    lines += [f"left out: {name}: {reason}" for name, reason in plan.skipped]
    lines += [f"hint {item['lever']}: parity check: {item['parity_check'] or 'none'}; quality check (not run): {item['quality_check'] or 'none'}"
              for item in check_hints(plan)]
    return lines


def median_speed(result: dict) -> float | None:
    speeds = [row["tok_s"] for row in result.get("rows", []) if quant_bench.usable(row)]
    return round(statistics.median(speeds), 2) if speeds else None


def peak_temperature(result: dict) -> float | None:
    return quant_bench.hottest(*(row.get("peak_c") for row in result.get("rows", []) if not row.get("warmup")))


def first_answer(result: dict) -> str | None:
    for row in result.get("rows", []):
        if not row.get("warmup"):
            return row.get("answer") or None
    return None


def parity_status(answer: str | None, reference: str | None) -> str:
    if not answer or not reference:
        return PARITY_NA
    return PARITY_SAME if answer == reference else PARITY_DIFFERS


def baseline_medians(records: list) -> list:
    return [median_speed(record["result"]) for record in records if record["plan"]["role"] == ROLE_BASELINE]


def drift_pct(records: list) -> float | None:
    medians = baseline_medians(records)[:2]
    if len(medians) < 2 or None in medians or not medians[0]:
        return None
    return round((medians[1] - medians[0]) / medians[0] * 100, 1)


def run_problems(record: dict, median: float | None) -> list:
    result, plan = record["result"], record["plan"]
    problems = []
    if result.get("aborted"):
        problems.append(f"aborted: {result['aborted']}")
    if result.get("cleanup_problems"):
        problems.append("cleanup: " + "; ".join(result["cleanup_problems"]))
    if plan["returncode"] == TIMEOUT_STATUS:
        problems.append("timed out")
    elif plan["returncode"] and not problems:
        problems.append(f"exit status {plan['returncode']}")
    if plan["started_cool"] is False:
        problems.append("started hot")
    if median is None and not problems:
        problems.append("no usable request")
    return problems


def run_failed(record: dict) -> bool:
    """No result to compare with: the run aborted, exited with an error or has no usable request."""
    return bool(record["result"].get("aborted") or record["plan"]["returncode"] or median_speed(record["result"]) is None)


def summarize(records: list, plan: Plan, session: dict, args) -> dict:
    known = [median for median in baseline_medians(records) if median is not None]
    reference = statistics.mean(known) if known else None
    first = next((record for record in records if record["plan"]["role"] == ROLE_BASELINE), None)
    reference_answer = first_answer(first["result"]) if first else None
    rows = []
    for record in records:
        median = median_speed(record["result"])
        is_reference = record is first
        rows.append({"index": record["plan"]["index"], "label": record["label"],
                     "levers": [pick["lever"] + ("" if pick["first"] else f".{pick['variant']}") for pick in record["plan"]["levers"]],
                     "median_tok_s": median, "delta_pct": round((median / reference - 1) * 100, 1) if median is not None and reference else None,
                     "max_package_c": peak_temperature(record["result"]),
                     "parity": PARITY_NA if is_reference else parity_status(first_answer(record["result"]), reference_answer),
                     "quality": QUALITY_STATUS, "problems": run_problems(record, median)})
    drift = drift_pct(records)
    return {"out_dir": str(args.out_dir), "runs_planned": len(plan.runs), "runs_done": len(records),
            "baseline_mean_tok_s": round(reference, 2) if reference is not None else None, "baseline_drift_pct": drift,
            "drift_status": "n/a" if drift is None else ("UNSTABLE" if abs(drift) > UNSTABLE_DRIFT_PCT else "stable"),
            "runs": rows, "hints": check_hints(plan), "session": session}


def short(text: str) -> str:
    return text if len(text) <= TEXT_LIMIT else text[:TEXT_LIMIT - 3] + "..."


def figure(value: float | None, template: str, empty: str = "-") -> str:
    return empty if value is None else template.format(value)


def format_table(headers: tuple, rows: list) -> list:
    widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)]
    return ["  ".join(str(cell).ljust(width) for cell, width in zip(line, widths)).rstrip() for line in (headers, *rows)]


def format_summary(summary: dict) -> str:
    rows = [(row["index"], row["label"], "+".join(row["levers"]) or "-", figure(row["median_tok_s"], "{:.1f}"), figure(row["delta_pct"], "{:+.1f} %"),
             figure(row["max_package_c"], "{:.0f}"), row["parity"], row["quality"], short("; ".join(row["problems"])) or "-") for row in summary["runs"]]
    mean, drift = figure(summary["baseline_mean_tok_s"], "{:.1f} tok/s", "n/a"), figure(summary["baseline_drift_pct"], "{:+.1f} %", "n/a")
    lines = [f"output: {summary['out_dir']}", f"runs: {summary['runs_done']} of {summary['runs_planned']} done",
             f"baseline mean: {mean}; drift between the first and the second baseline: {drift} ({summary['drift_status']})", ""]
    lines += format_table(("run", "label", "levers", "tok/s", "delta", "max C", "parity", "quality", "problems"), rows)
    lines += ["", TABLE_LEGEND]
    for item in summary["hints"]:
        lines.append(f"{item['lever']}: parity check: {item['parity_check'] or 'none'}; quality check (not run): {item['quality_check'] or 'none'}")
    session = summary["session"]
    if session.get("stopped"):
        lines.append(f"SESSION STOPPED: {session['stopped']}")
    lines += [f"CPU CAP PROBLEM: {problem}" for problem in session.get("cap_problems", [])]
    lines += [f"LEFTOVER PROCESS: {process}" for process in session.get("leftover_processes", [])]
    return "\n".join(lines) + "\n"


def write_summary(out_dir: Path, summary: dict) -> None:
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    (out_dir / "summary.txt").write_text(format_summary(summary), encoding="utf-8")


def write_chart(out_dir: Path, records: list, args, say) -> None:
    labels = [record["label"] for record in records]
    note = f"cap {args.cpu_max_mhz:g} MHz, {args.workers} workers, limit {args.max_temp:g} C, {args.max_tokens} tokens, {args.repeats} requests"
    svg = bench_chart.speed_chart([record["result"] for record in records], f"Lever matrix, {args.model.name}", note, bench_chart.series_for(labels))
    (out_dir / "chart.svg").write_text(svg, encoding="utf-8")
    if args.png and not bench_chart.to_png((out_dir / "chart.svg").resolve(), (out_dir / "chart.png").resolve(), bench_chart.svg_size(svg)):
        say("lever-matrix: the PNG could not be made")


class Runner:
    """Everything a session does to the machine; tests replace it with a fake."""

    def execute(self, command: list, timeout: float) -> int:
        child = subprocess.Popen(command, start_new_session=True, stdout=subprocess.DEVNULL)
        try:
            return child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.terminate(child)
            return TIMEOUT_STATUS
        except BaseException:
            self.terminate(child)
            raise

    def terminate(self, child: subprocess.Popen, grace: float = GRACE_SECONDS) -> None:
        """SIGTERM so the child's own cleanup runs, SIGKILL when it is still there after `grace` seconds; stop signals wait meanwhile."""
        if child.poll() is not None:
            return
        signal.pthread_sigmask(signal.SIG_BLOCK, quant_bench.STOP_SIGNALS)
        try:
            child.send_signal(signal.SIGTERM)
            try:
                child.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=KILL_WAIT_SECONDS)
        finally:
            signal.pthread_sigmask(signal.SIG_UNBLOCK, quant_bench.STOP_SIGNALS)

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def clock(self) -> float:
        return time.monotonic()

    def temperature(self) -> float | None:
        return quant_bench.package_temperature()

    def load(self) -> float:
        return os.getloadavg()[0]

    def machine(self, max_load: float) -> dict:
        return machine_state.collect(max_load=max_load)

    def processes(self) -> list:
        return machine_state.running_processes()

    def cap_snapshot(self) -> dict:
        return machine_state.cpu_cap_snapshot()

    def cap_restore(self, snapshot: dict) -> list:
        return machine_state.cpu_cap_restore(snapshot)

    def timestamp(self) -> str:
        return time.strftime("%Y%m%d-%H%M%S")


def install_stop_handlers() -> dict:
    def stop(signum, frame):
        raise SessionStopped(f"stopped by signal {signal.Signals(signum).name}")
    return {number: signal.signal(number, stop) for number in quant_bench.STOP_SIGNALS}


def restore_caps(runner: Runner, snapshot: dict) -> list:
    """Put the CPU frequency limits back and read them again; returns everything that is still not as it was."""
    problems = runner.cap_restore(snapshot)
    after = runner.cap_snapshot()
    return problems + [f"{path} is {after.get(path)}, it was {saved}" for path, saved in snapshot.items() if after.get(path) != saved]


def settle(args, runner: Runner) -> tuple:
    """Wait until the package is cool and the load average is low; returns (cool state, seconds waited)."""
    _, waited, cool = quant_bench.wait_until_cool(args.cool, args.cool_timeout, runner.temperature, runner.sleep, runner.clock)
    for _ in range(max(1, int(args.cool_timeout // LOAD_POLL_SECONDS))):
        if runner.load() <= args.max_load:
            break
        runner.sleep(LOAD_POLL_SECONDS)
        waited += LOAD_POLL_SECONDS
    return cool, waited


def read_result(raw: Path, run: Run, args, returncode: int) -> dict:
    for path in (raw, raw.with_name(raw.name + ".aborted")):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return {"label": run.label, "budget": args.budget, "rows": [], "summary": {}, "cleanup_problems": [],
            "aborted": "timed out" if returncode == TIMEOUT_STATUS else f"quant_bench left no result (exit status {returncode})"}


def measure_run(run: Run, args, runner: Runner, cool: bool | None, waited: float) -> dict:
    raw = args.out_dir / RAW_DIR / f"{run.stem}.json"
    command = quant_bench_command(run, args, raw)
    returncode = runner.execute(command, args.run_timeout)
    result = read_result(raw, run, args, returncode)
    levers = [{"lever": pick.lever.name, "variant": pick.variant.name, "first": pick.variant == pick.lever.variants[0], "kind": pick.lever.kind}
              for pick in run.picks]
    plan = {"index": run.index, "role": run.role, "levers": levers, "env": run.env, "engine_args": list(run.engine_args), "model": str(run.model),
            "launcher_args_not_passed": {pick.spec: list(pick.variant.launcher_args) for pick in run.picks if pick.variant.launcher_args},
            "parity_check": {pick.lever.name: pick.lever.parity_check for pick in run.picks if pick.lever.parity_check},
            "quality_check": {pick.lever.name: pick.lever.quality_check for pick in run.picks if pick.lever.quality_check},
            "command": command, "started_cool": cool, "cool_wait_s": round(waited), "returncode": returncode}
    return {"label": run.label, "result": result, "plan": plan}


def preflight(plan: Plan, args, runner: Runner) -> None:
    for model in dict.fromkeys(run.model for run in plan.runs):
        if not model.is_file():
            raise PlanError(f"the model file {model} does not exist")
    if args.out_dir.exists():
        raise PlanError(f"{args.out_dir} exists")
    if not args.force:
        state = runner.machine(args.max_load)
        if not state["quiet"]:
            raise PlanError("the machine is not quiet: " + "; ".join(state["problems"]) + " (--force overrides)")


def run_session(plan: Plan, args, runner: Runner, say) -> int:
    preflight(plan, args, runner)
    (args.out_dir / RAW_DIR).mkdir(parents=True)
    records, session = [], {"stopped": None, "cap_problems": [], "leftover_processes": []}
    snapshot, running_before = runner.cap_snapshot(), runner.processes()
    previous = install_stop_handlers()
    try:
        for run in plan.runs:
            say(f"lever-matrix: run {run.index + 1} of {len(plan.runs)} ({run.label}): waiting for the package to be at or below {args.cool:g} C")
            cool, waited = settle(args, runner)
            if cool is False and not args.continue_hot:
                session["stopped"] = (f"the package stayed above {args.cool:g} C for {args.cool_timeout:g} s before run {run.index} ({run.label}); "
                                      f"--continue-hot measures anyway")
                break
            say(f"lever-matrix: measuring {run.label}")
            record = measure_run(run, args, runner, cool, waited)
            records.append(record)
            (args.out_dir / f"{run.stem}.json").write_text(json.dumps({**record["result"], "plan": record["plan"]}, indent=1) + "\n", encoding="utf-8")
            write_summary(args.out_dir, summarize(records, plan, session, args))
            if run.role == ROLE_BASELINE and run_failed(record):
                session["stopped"] = f"the baseline run {run.index} failed: {'; '.join(run_problems(record, median_speed(record['result'])))}"
                break
    except SessionStopped as stop:
        session["stopped"] = str(stop)
    finally:
        for number in quant_bench.STOP_SIGNALS:
            signal.signal(number, signal.SIG_IGN)
        session["cap_problems"] = restore_caps(runner, snapshot)
        session["leftover_processes"] = [process for process in runner.processes() if process not in running_before]
        quant_bench.restore_handlers(previous)
    summary = summarize(records, plan, session, args)
    write_summary(args.out_dir, summary)
    if records:
        write_chart(args.out_dir, records, args, say)
    print(format_summary(summary), end="")
    problem = session["stopped"] or session["cap_problems"] or session["leftover_processes"] or any(row["problems"] for row in summary["runs"])
    return EXIT_SESSION_PROBLEM if problem else 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    common.add_argument("--levers-dir", type=Path, default=LEVERS_DIR, help="directory with the lever definitions (default: tools/levers.d)")
    common.add_argument("--engine-root", type=Path, default=ENGINE_ROOT, help="checkout whose build/ and models/ are used")
    selection = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    group = selection.add_mutually_exclusive_group(required=True)
    group.add_argument("--levers", metavar="a,b:variant", help="one run with these levers together; the first variant unless NAME:VARIANT is given")
    group.add_argument("--each", action="store_true", help="every available engine or model-file lever alone")
    group.add_argument("--all", action="store_true", help="every available lever in one run")
    group.add_argument("--pairs", action="store_true", help="every pair of available levers")
    selection.add_argument("--variants", action="store_true", help="with --each: every variant of every lever, not only the first")
    selection.add_argument("--model", type=Path, help="model file (default: <engine-root>/models/qwen3.6-35b-a3b/...IQ4_XS-mtp.gguf)")
    selection.add_argument("--budget", default="none", choices=list(quant_bench.PROFILE_OF_BUDGET))
    selection.add_argument("--prompts", nargs="+", choices=list(quant_bench.PROMPTS), default=["code"])
    selection.add_argument("--repeats", type=int, default=3)
    selection.add_argument("--warmup", type=int, default=2)
    selection.add_argument("--max-tokens", type=int, default=900)
    selection.add_argument("--cpu-max-mhz", type=float, default=2200.0)
    selection.add_argument("--workers", type=int, default=6)
    selection.add_argument("--max-temp", type=float, default=80.0)
    selection.add_argument("--cool", type=float, default=55.0, help="start every run and every request with the package at or below this many degrees")
    selection.add_argument("--cool-timeout", type=float, default=900.0)
    selection.add_argument("--max-load", type=float, default=1.5, help="load average below which the next run may start")
    selection.add_argument("--continue-hot", action="store_true", help="go on when the package did not cool down in time (default: stop the session)")
    selection.add_argument("--run-timeout", type=float, default=3600.0, help="seconds one quant_bench run may take")
    selection.add_argument("--prewarm", action="store_true")
    selection.add_argument("--force", action="store_true", help="start although the machine is not quiet (also passed to quant_bench)")
    selection.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="environment variable for every run, repeatable")
    selection.add_argument("--baseline-env", action="append", metavar="KEY=VALUE", help="replaces the default baseline env, repeatable")
    selection.add_argument("--engine-args", type=shlex.split, default=[], metavar='"ARGS"', help="more engine arguments for every run, one quoted string")
    selection.add_argument("--out-dir", type=Path, help="default: <engine-root>/run/experiments/levers/<YYYYmmdd-HHMMSS>")
    selection.add_argument("--png", action="store_true", help="also render chart.png")
    selection.add_argument("--dry-run", action="store_true", help="print the plan and the command lines, run nothing, write nothing")
    parser = argparse.ArgumentParser(description="Run the performance levers of the engine through quant_bench, one run at a time, between two baselines.",
                                     epilog=SCHEMA_HELP + "\n" + MERGE_HELP, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", parents=[common], help="the levers, their variants and whether they can run here", allow_abbrev=False)
    for name, text in (("plan", "the runs and the exact quant_bench command lines"), ("run", "execute the plan")):
        commands.add_parser(name, parents=[common, selection], help=text, description=text, epilog=MERGE_HELP, allow_abbrev=False,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
    return parser


def parse_arguments(argv: list | None, runner: Runner) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.command != "list":
        args.env = quant_bench.parse_env(args.env)
        args.baseline_env = quant_bench.parse_env(args.baseline_env) if args.baseline_env else dict(BASELINE_ENV)
        args.model = args.model or args.engine_root / DEFAULT_MODEL
        args.out_dir = args.out_dir or args.engine_root / OUT_ROOT / runner.timestamp()
    return args


def progress(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def main(argv: list | None = None, runner: Runner | None = None) -> int:
    runner = runner or Runner()
    args = parse_arguments(argv, runner)
    try:
        levers = load_levers(args.levers_dir)
        if args.command == "list":
            print("\n".join(line for lever in levers for line in [*describe_lever(lever, args.engine_root), ""]) or f"no lever definitions in {args.levers_dir}")
            return 0
        plan = build_plan(levers, args)
        if args.command == "plan" or args.dry_run:
            print("\n".join(describe_plan(plan, args)))
            return 0
        return run_session(plan, args, runner, progress)
    except PlanError as error:
        print(f"lever-matrix: {error}", file=sys.stderr)
        return EXIT_PLAN_ERROR


if __name__ == "__main__":
    sys.exit(main())
