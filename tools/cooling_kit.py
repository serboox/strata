import argparse
import csv
import importlib.util
import json
import math
import os
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_chart
import machine_state
import quant_bench

ENGINE_ROOT = quant_bench.ENGINE_ROOT
QUANT_BENCH = Path(quant_bench.__file__).resolve()
HWMON_ROOT = Path("/sys/class/hwmon")
POWERCAP_ROOT = Path("/sys/class/powercap")
PLATFORM_PROFILE = Path("/sys/firmware/acpi/platform_profile")
PLATFORM_PROFILE_CHOICES = Path("/sys/firmware/acpi/platform_profile_choices")
CORE_SENSOR_NAME = "coretemp"
CORE_LABEL_PREFIX = "Core"
RAPL_PACKAGE_PREFIX = "package-"
CURRENT = "current"
DEFAULT_CAPS_GHZ = "2.2,2.4,2.6,2.8"
DEFAULT_LIMIT_C = 80.0
DEFAULT_START_TEMP_C = 55.0
THRESHOLDS_C = (75.0, 80.0)
SAMPLE_PERIOD_S = 2.0
DEFAULT_DURATION_S = 300.0
STARTUP_ALLOWANCE_S = 240.0
SECONDS_PER_REQUEST = 20.0
NO_COOL_WAIT_C = 200.0
STOP_WAIT_S = 120.0
SUDO_TIMEOUT_S = 120.0
MIN_FAN_SAMPLES = 3
FAN_EXPECTED_ABOVE_C = 65.0
SUDO_REASON_PROFILE = "сменить профиль охлаждения платформы"
SUDO_REASON_RAPL = "разрешить чтение счётчика энергии процессора до перезагрузки"
SAMPLE_COLUMNS = ("t_s", "pkg_c", "core_max_c", "gpu_c", "freq_mean_mhz", "freq_min_mhz", "freq_max_mhz", "power_w", "profile")
STEP_COLORS = bench_chart.PALETTE
AMBER = "#e08a2b"


def load_cpucap(root: Path = ENGINE_ROOT):
    spec = importlib.util.spec_from_file_location("strata_cpucap_for_cooling_kit", root / "serve" / "cpucap.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def read_number(path: Path) -> float | None:
    text = read_text(path)
    try:
        return float(text) if text is not None else None
    except ValueError:
        return None


class Sensors:
    """Reads the machine's temperatures, frequencies, fans, power and platform profile; every root can be pointed elsewhere for a test."""

    def __init__(self, hwmon_root: Path = HWMON_ROOT, cpu_root: Path = machine_state.CPU_ROOT, thermal_root: Path = quant_bench.THERMAL_ROOT,
                 powercap_root: Path = POWERCAP_ROOT, profile_file: Path = PLATFORM_PROFILE, gpu=quant_bench.gpu_temperature,
                 clock=time.monotonic):
        self.hwmon_root = hwmon_root
        self.cpu_root = cpu_root
        self.thermal_root = thermal_root
        self.powercap_root = powercap_root
        self.profile_file = profile_file
        self.gpu = gpu
        self.clock = clock
        self.rapl_dir = self.find_rapl_package()
        self.last_energy: tuple | None = None

    def hwmon_dirs(self) -> list:
        return sorted(self.hwmon_root.glob("hwmon*"))

    def core_max_c(self) -> float | None:
        readings = []
        for hwmon in self.hwmon_dirs():
            if read_text(hwmon / "name") != CORE_SENSOR_NAME:
                continue
            for label_file in hwmon.glob("temp*_label"):
                label = read_text(label_file) or ""
                value = read_number(label_file.with_name(label_file.name.replace("_label", "_input")))
                if label.startswith(CORE_LABEL_PREFIX) and value is not None:
                    readings.append(value / 1000)
        return max(readings) if readings else None

    def frequencies_mhz(self) -> list:
        values = []
        for path in sorted(self.cpu_root.glob("cpu[0-9]*/cpufreq/scaling_cur_freq"), key=lambda item: int(item.parent.parent.name[3:])):
            khz = read_number(path)
            if khz is not None:
                values.append(khz / 1000)
        return values

    def fans(self) -> dict:
        """RPM by fan label (the driver's own label, else `<hwmon name> fanN`); None for a fan that cannot be read."""
        fans = {}
        for hwmon in self.hwmon_dirs():
            for input_file in sorted(hwmon.glob("fan*_input")):
                label = read_text(input_file.with_name(input_file.name.replace("_input", "_label"))) or f"{read_text(hwmon / 'name')} {input_file.name[:-6]}"
                fans[label] = read_number(input_file)
        return fans

    def find_rapl_package(self) -> Path | None:
        for domain in sorted(self.powercap_root.glob("intel-rapl:*")):
            if (read_text(domain / "name") or "").startswith(RAPL_PACKAGE_PREFIX):
                return domain
        return None

    def rapl_readable(self) -> bool:
        return self.rapl_dir is not None and read_number(self.rapl_dir / "energy_uj") is not None

    def power_w(self) -> float | None:
        if self.rapl_dir is None:
            return None
        energy = read_number(self.rapl_dir / "energy_uj")
        now = self.clock()
        if energy is None:
            return None
        previous, self.last_energy = self.last_energy, (now, energy)
        if previous is None or now <= previous[0]:
            return None
        delta = energy - previous[1]
        if delta < 0:
            delta += read_number(self.rapl_dir / "max_energy_range_uj") or 0
        return round(delta / 1e6 / (now - previous[0]), 2) if delta >= 0 else None

    def profile(self) -> str | None:
        return read_text(self.profile_file)

    def read(self, elapsed_s: float) -> dict:
        frequencies = self.frequencies_mhz()
        sample = {"t_s": round(elapsed_s, 1), "pkg_c": quant_bench.package_temperature(self.thermal_root), "core_max_c": self.core_max_c(),
                  "gpu_c": self.gpu(), "freq_mean_mhz": round(statistics.fmean(frequencies)) if frequencies else None,
                  "freq_min_mhz": round(min(frequencies)) if frequencies else None, "freq_max_mhz": round(max(frequencies)) if frequencies else None,
                  "power_w": self.power_w(), "profile": self.profile(), "cpu_mhz": [round(value) for value in frequencies], "fans": self.fans()}
        return sample


def parse_caps(text: str, cpucap) -> list:
    """Frequency caps in MHz from a comma-separated list of GHz values."""
    caps = []
    for item in text.split(","):
        try:
            mhz = round(float(item) * 1000)
        except ValueError:
            raise SystemExit(f"cooling-kit: --caps wants GHz values such as 2.2,2.4, not {item!r}") from None
        if not cpucap.MIN_MHZ <= mhz <= cpucap.MAX_MHZ:
            raise SystemExit(f"cooling-kit: a cap of {mhz} MHz is outside {cpucap.MIN_MHZ:g} to {cpucap.MAX_MHZ:g} MHz")
        caps.append(mhz)
    if len(set(caps)) != len(caps):
        raise SystemExit("cooling-kit: --caps lists a value twice")
    return caps


def profile_command(name: str, profile_file: Path = PLATFORM_PROFILE) -> str:
    """The command the owner can paste to switch the platform profile; the kit itself runs it only with --apply-profile."""
    return f'echo {shlex.quote(name)} | SUDO_PROMPT="{SUDO_REASON_PROFILE}" sudo -A tee {profile_file} > /dev/null'


def rapl_command(sensors: Sensors) -> str:
    target = sensors.rapl_dir / "energy_uj" if sensors.rapl_dir else Path("/sys/class/powercap/intel-rapl:0/energy_uj")
    return f'SUDO_PROMPT="{SUDO_REASON_RAPL}" sudo -A chmod a+r {target}'


def apply_profile(name: str, profile_file: Path = PLATFORM_PROFILE, run=subprocess.run) -> str | None:
    """None when the platform profile now reads `name`, else why not. Only called after an explicit --apply-profile."""
    try:
        done = run(["sudo", "-A", "tee", str(profile_file)], input=name + "\n", text=True, capture_output=True, timeout=SUDO_TIMEOUT_S, check=False,
                   env={**os.environ, "SUDO_PROMPT": SUDO_REASON_PROFILE})
    except (OSError, subprocess.SubprocessError) as error:
        return f"sudo failed: {error}"
    if done.returncode != 0:
        return f"sudo tee exited with {done.returncode}: {done.stderr.strip()[:120]}"
    now = read_text(profile_file)
    return None if now == name else f"the profile reads {now!r} after the write, not {name!r}"


def build_steps(caps_mhz: list, profiles: list) -> list:
    """One step for every profile and cap, the profile as the outer loop so it changes once per profile."""
    steps = []
    for profile in profiles:
        for cap in caps_mhz:
            steps.append({"index": len(steps) + 1, "cap_mhz": cap, "profile": profile, "label": f"cap{cap}_{profile}"})
    return steps


def check_profiles(profiles: list, current: str | None, choices: list, apply: bool) -> None:
    for profile in profiles:
        if choices and profile not in choices:
            raise SystemExit(f"cooling-kit: the platform has no profile {profile!r} (choices: {' '.join(choices)})")
        if profile != current and not apply:
            raise SystemExit(f"cooling-kit: the machine is on the {current!r} profile and {profile!r} was asked for. Switch it yourself with\n"
                             f"  {profile_command(profile)}\nor pass --apply-profile to let the kit run that command (it restores {current!r} at the end)")


def requests_for(duration_s: float) -> int:
    return max(2, math.ceil(duration_s / SECONDS_PER_REQUEST) + 1)


def bench_command(args, step: dict, out: Path) -> list:
    command = [*(quant_bench.PARENT_DEATH_PREFIX if shutil.which("setpriv") else []), sys.executable, str(QUANT_BENCH), "--model", str(args.model),
               "--label", step["label"], "--budget", "none", "--prompts", "code", "--repeats", str(requests_for(args.duration)), "--warmup", "0",
               "--max-tokens", str(args.max_tokens), "--cpu-max-mhz", str(step["cap_mhz"]), "--workers", str(args.workers),
               "--cool", str(NO_COOL_WAIT_C), "--cool-timeout", "1", "--max-temp", f"{args.limit:g}", "--out", str(out)]
    for key, value in args.env.items():
        command += ["--env", f"{key}={value}"]
    if args.engine_args:
        command += ["--engine-args", shlex.join(args.engine_args)]
    if args.prewarm:
        command.append("--prewarm")
    if args.force:
        command.append("--force")
    return command


def first_at_or_above(samples: list, threshold: float) -> float | None:
    for sample in samples:
        if sample["pkg_c"] is not None and sample["pkg_c"] >= threshold:
            return sample["t_s"]
    return None


def mean_of(values: list) -> float | None:
    known = [value for value in values if value is not None]
    return round(statistics.fmean(known), 1) if known else None


def fan_report(samples: list, label: str, peak_c: float | None) -> dict:
    """Mean and maximum RPM of one fan, and whether the reading can be believed: a fan that reads 0 (or nothing) for the whole run is
    reported as an unreliable sensor, never as a stopped fan."""
    values = [sample["fans"].get(label) for sample in samples if label in sample["fans"]]
    known = [value for value in values if value is not None]
    report = {"label": label, "mean_rpm": mean_of(known), "max_rpm": max(known) if known else None, "reliable": True, "note": ""}
    if len(known) < MIN_FAN_SAMPLES:
        report.update(reliable=False, note="too few readings")
    elif max(known) == 0:
        hot = peak_c is not None and peak_c >= FAN_EXPECTED_ABOVE_C
        report.update(reliable=False, note="reads 0 for the whole run" + (f" although the package reached {peak_c:.0f} C: the sensor is most likely not wired"
                                                                          if hot else " (idle fan or a dead sensor)"))
    return report


def split_speeds(rows: list) -> dict:
    """The first usable request's speed and the median of the second half of the usable ones (the heat-soaked state)."""
    speeds = [row["tok_s"] for row in rows if quant_bench.usable(row)]
    if not speeds:
        return {"tok_s_first": None, "tok_s_sustained": None, "requests": 0}
    tail = speeds[len(speeds) // 2:] if len(speeds) > 1 else speeds
    return {"tok_s_first": speeds[0], "tok_s_sustained": round(statistics.median(tail), 2), "requests": len(speeds)}


def load_started_s(rows: list) -> float | None:
    """When the first request began, from the time its line was read and its own duration: the model is loaded by then."""
    for row in rows:
        if "kit_t_s" in row and "wall_s" in row:
            return round(row["kit_t_s"] - row["wall_s"], 1)
    return None


def summarize_step(step: dict, samples: list, rows: list, stop_reason: str, started_cool: bool | None) -> dict:
    peak = max([sample["pkg_c"] for sample in samples if sample["pkg_c"] is not None], default=None)
    labels = sorted({label for sample in samples for label in sample["fans"]})
    loaded = load_started_s(rows)
    summary = {**{key: step[key] for key in ("index", "label", "cap_mhz", "profile")}, "stop_reason": stop_reason, "started_cool": started_cool,
               "samples": len(samples), "pkg_mean_c": mean_of([sample["pkg_c"] for sample in samples]), "pkg_max_c": peak,
               "core_max_c": max([sample["core_max_c"] for sample in samples if sample["core_max_c"] is not None], default=None),
               "freq_mean_mhz": mean_of([sample["freq_mean_mhz"] for sample in samples]), "power_mean_w": mean_of([sample["power_w"] for sample in samples]),
               "power_max_w": max([sample["power_w"] for sample in samples if sample["power_w"] is not None], default=None),
               "load_started_s": loaded, "fans": [fan_report(samples, label, peak) for label in labels],
               **{f"t{int(threshold)}_s": first_at_or_above(samples, threshold) for threshold in THRESHOLDS_C}, **split_speeds(rows)}
    for threshold in THRESHOLDS_C:
        key = f"t{int(threshold)}_s"
        summary[f"{key}_after_load"] = round(summary[key] - loaded, 1) if summary[key] is not None and loaded is not None and summary[key] >= loaded else None
    return summary


def sensor_notes(samples: list, sensors: Sensors, with_fans: bool = True) -> list:
    """What the sensors did and did not give, for the owner to judge the data by."""
    peak = max([sample["pkg_c"] for sample in samples if sample["pkg_c"] is not None], default=None)
    notes = []
    if peak is None:
        notes.append("package temperature: NOT READABLE (no x86_pkg_temp thermal zone); the run cannot be judged")
    if all(sample["core_max_c"] is None for sample in samples):
        notes.append("core temperatures: not readable")
    if all(sample["gpu_c"] is None for sample in samples):
        notes.append("GPU temperature: not readable (nvidia-smi)")
    if all(sample["freq_mean_mhz"] is None for sample in samples):
        notes.append("CPU frequencies: not readable")
    if sensors.rapl_dir is None:
        notes.append("power: no RAPL package domain on this machine")
    elif not sensors.rapl_readable():
        notes.append(f"power: the RAPL counter is not readable (root only); to read it until the next reboot run\n  {rapl_command(sensors)}")
    labels = sorted({label for sample in samples for label in sample["fans"]})
    if not labels:
        notes.append("fans: no fan sensor found")
    for label in labels if with_fans else []:
        report = fan_report(samples, label, peak)
        if not report["reliable"]:
            notes.append(f"fan {label}: UNRELIABLE, {report['note']}")
    if not any(sample["profile"] for sample in samples):
        notes.append("platform profile: not readable")
    return notes


def format_table(steps: list) -> str:
    def cell(value, spec="{}"):
        return "-" if value is None else spec.format(value)

    def cpu_fan(step):
        fan = next((item for item in step["fans"] if "CPU" in item["label"].upper()), step["fans"][0] if step["fans"] else None)
        if fan is None or not fan["reliable"]:
            return "n/a", "n/a"
        return cell(fan["mean_rpm"], "{:.0f}"), cell(fan["max_rpm"], "{:.0f}")

    header = f"{'step':<24}{'sust t/s':>9}{'first':>7}{'75C s':>7}{'80C s':>7}{'mean C':>8}{'max C':>7}{'fan avg':>8}{'fan max':>8}  stop"
    lines = [header, "-" * len(header)]
    for step in steps:
        fan_mean, fan_max = cpu_fan(step)
        lines.append(f"{step['label']:<24}{cell(step['tok_s_sustained'], '{:.1f}'):>9}{cell(step['tok_s_first'], '{:.1f}'):>7}"
                     f"{cell(step['t75_s'], '{:.0f}'):>7}{cell(step['t80_s'], '{:.0f}'):>7}{cell(step['pkg_mean_c'], '{:.1f}'):>8}"
                     f"{cell(step['pkg_max_c'], '{:.0f}'):>7}{fan_mean:>8}{fan_max:>8}  {step['stop_reason']}")
    lines.append("times are seconds from the start of the step; fan columns are the CPU fan, n/a where the sensor is unreliable")
    return "\n".join(lines)


def write_samples_csv(path: Path, label: str, samples: list, header: bool) -> None:
    labels = sorted({fan for sample in samples for fan in sample["fans"]})
    cpus = max((len(sample["cpu_mhz"]) for sample in samples), default=0)
    columns = ["step", *SAMPLE_COLUMNS, *(f"fan_{fan}_rpm" for fan in labels), *(f"cpu{index}_mhz" for index in range(cpus))]
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if header:
            writer.writerow(columns)
        for sample in samples:
            writer.writerow([label, *(sample[name] for name in SAMPLE_COLUMNS), *(sample["fans"].get(fan) for fan in labels),
                             *(sample["cpu_mhz"][index] if index < len(sample["cpu_mhz"]) else None for index in range(cpus))])


def cooling_chart(steps: list, series: dict, limit_c: float, title: str, note: str) -> str:
    """Package temperature against time for every step (top) and the sustained speed of every step (bottom)."""
    width, left, right = 960, 64, 24
    top_h, bottom_h = 300, 190
    height = 70 + top_h + 60 + bottom_h + 60
    plot_w = width - left - right
    longest = max([sample["t_s"] for samples in series.values() for sample in samples] + [1.0])
    floor_c, ceiling_c = 40.0, max([sample["pkg_c"] for samples in series.values() for sample in samples if sample["pkg_c"] is not None] + [limit_c + 5])
    ceiling_c = (int(ceiling_c / 5) + 1) * 5
    upper = 70

    def x_of(t):
        return left + plot_w * t / longest

    def y_of(c):
        return upper + top_h * (1 - (c - floor_c) / (ceiling_c - floor_c))

    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img"><rect width="{width}" height="{height}" fill="#ffffff"/>',
             bench_chart.text(left, 26, title, 17, weight="bold"), bench_chart.text(left, 46, note, 12, fill=bench_chart.MUTED)]
    for tick in range(int(floor_c), int(ceiling_c) + 1, 10):
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y_of(tick):.1f}" y2="{y_of(tick):.1f}" stroke="{bench_chart.GRID}"/>')
        parts.append(bench_chart.text(left - 8, y_of(tick) + 4, f"{tick} C", 11, "end", bench_chart.MUTED))
    for level, color in ((limit_c, "#d62b2b"), (THRESHOLDS_C[0], AMBER)):
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y_of(level):.1f}" y2="{y_of(level):.1f}" stroke="{color}" stroke-dasharray="6 4"/>')
        parts.append(bench_chart.text(width - right, y_of(level) - 5, f"{level:.0f} C", 11, "end", color))
    for index, step in enumerate(steps):
        samples = [sample for sample in series.get(step["label"], []) if sample["pkg_c"] is not None]
        if not samples:
            continue
        points = " ".join(f"{x_of(sample['t_s']):.1f},{y_of(sample['pkg_c']):.1f}" for sample in samples)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{STEP_COLORS[index % len(STEP_COLORS)]}" stroke-width="1.8"/>')
    parts.append(bench_chart.text(left, upper + top_h + 20, "seconds from the start of the step", 12, fill=bench_chart.MUTED))
    parts.append(bench_chart.text(width - right, upper + top_h + 20, f"{longest:.0f} s", 11, "end", bench_chart.MUTED))
    legend_y = upper + top_h + 40
    for index, step in enumerate(steps):
        x = left + (index % 4) * 220
        y = legend_y + (index // 4) * 18
        parts.append(f'<rect x="{x}" y="{y - 11}" width="14" height="14" fill="{STEP_COLORS[index % len(STEP_COLORS)]}"/>')
        parts.append(bench_chart.text(x + 20, y, step["label"], 12))
    bars_top = legend_y + 18 * (math.ceil(len(steps) / 4)) + 30
    speeds = [step["tok_s_sustained"] for step in steps if step["tok_s_sustained"] is not None]
    peak_speed = (int(max(speeds + [10.0]) / 10) + 1) * 10
    parts.append(bench_chart.text(left, bars_top - 8, "sustained tokens per second", 12, fill=bench_chart.MUTED))
    slot = plot_w / max(1, len(steps))
    for index, step in enumerate(steps):
        value = step["tok_s_sustained"]
        if value is None:
            continue
        bar_h = bottom_h * value / peak_speed
        x = left + index * slot + slot * 0.15
        parts.append(f'<rect x="{x:.1f}" y="{bars_top + bottom_h - bar_h:.1f}" width="{slot * 0.7:.1f}" height="{bar_h:.1f}" '
                     f'fill="{STEP_COLORS[index % len(STEP_COLORS)]}" opacity="0.88"/>')
        parts.append(bench_chart.text(x + slot * 0.35, bars_top + bottom_h - bar_h - 6, f"{value:.1f}", 12, "middle", bench_chart.INK, "bold"))
    parts.append("</svg>")
    return "".join(parts)


def stderr_reader(stream, rows: list, lines: list, clock) -> None:
    """Collect the benchmark's stderr: each JSON line is one finished request (kept with the time it was read)."""
    for line in stream:
        text = line.strip()
        if text.startswith("{"):
            try:
                row = json.loads(text)
            except ValueError:
                row = None
            if isinstance(row, dict) and "prompt_name" in row:
                row["kit_t_s"] = round(clock(), 1)
                rows.append(row)
                continue
        lines.append(text)


def stop_process(process, wait_s: float = STOP_WAIT_S) -> None:
    """Ask the benchmark to stop (it then stops its server and puts the cap back); kill it only when it does not answer in time."""
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=wait_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def sample_step(process, sensors: Sensors, args, started: float, samples: list, clock, sleep) -> str:
    """Sample until the benchmark ends, the package reaches the limit or the duration is over; returns why it stopped."""
    deadline = args.duration + STARTUP_ALLOWANCE_S
    while True:
        elapsed = clock() - started
        sample = sensors.read(elapsed)
        samples.append(sample)
        if process.poll() is not None:
            return "benchmark ended"
        if sample["pkg_c"] is not None and sample["pkg_c"] >= args.limit:
            return f"limit {args.limit:g} C"
        if elapsed >= deadline:
            return "duration"
        sleep(args.period)


def run_step(step: dict, args, sensors: Sensors, out_dir: Path, say, popen=subprocess.Popen, sleep=time.sleep, clock=time.monotonic,
             wait_cool=quant_bench.wait_until_cool) -> tuple:
    """One benchmark under one cap: cool down, start the harness, sample every period, stop it. Returns (summary, samples, rows)."""
    temperature, waited, cool = wait_cool(args.start_temp, args.cool_timeout)
    say(f"cooling-kit: step {step['index']} {step['label']}: start at {temperature} C after {waited:.0f} s (cool: {cool})")
    out = out_dir / f"bench_{step['label']}.json"
    rows, lines, samples = [], [], []
    started = clock()
    process = popen(bench_command(args, step, out), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, start_new_session=False)
    reader = threading.Thread(target=stderr_reader, args=(process.stderr, rows, lines, lambda: clock() - started), daemon=True)
    reader.start()
    try:
        stop_reason = sample_step(process, sensors, args, started, samples, clock, sleep)
    finally:
        stop_process(process)
        reader.join(timeout=10)
    if stop_reason == "benchmark ended" and process.returncode not in (0, None):
        tail = " | ".join(lines[-3:])
        stop_reason = f"benchmark failed ({process.returncode}): {tail[:160]}"
    say(f"cooling-kit: step {step['index']} stopped: {stop_reason}")
    return summarize_step(step, samples, rows, stop_reason, cool), samples, rows


def write_result(out_dir: Path, args, steps: list, notes: list, started_at: str, restored: dict) -> None:
    result = {"setup": args.setup, "model": str(args.model), "limit_c": args.limit, "duration_s": args.duration, "workers": args.workers,
              "env": args.env, "start_temp_c": args.start_temp, "turbo": quant_bench.turbo_state(), "started": started_at,
              "sensor_notes": notes, "restored": restored, "steps": steps}
    target = out_dir / "result.json"
    temporary = out_dir / "result.json.tmp"
    temporary.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    (out_dir / "table.txt").write_text(format_table(steps) + "\n", encoding="utf-8")


def print_plan(args, steps: list, sensors: Sensors, out_dir: Path, current: str | None, say=print) -> None:
    say(f"cooling-kit plan: {len(steps)} steps of {args.duration:.0f} s each (+ up to {STARTUP_ALLOWANCE_S:.0f} s of start), "
        f"cool to {args.start_temp:g} C before each, stop a step at {args.limit:g} C, sample every {args.period:g} s")
    say(f"setup label: {args.setup!r}; model: {args.model}; workers: {args.workers}; env: {args.env}")
    say(f"platform profile now: {current!r}; output directory: {out_dir}")
    for step in steps:
        say(f"  step {step['index']}: cap {step['cap_mhz']} MHz, profile {step['profile']}")
        say("    " + shlex.join(bench_command(args, step, out_dir / f"bench_{step['label']}.json")))
    snapshot = sensors.read(0.0)
    say("sensors now: " + json.dumps({key: snapshot[key] for key in ("pkg_c", "core_max_c", "gpu_c", "freq_mean_mhz", "profile", "fans")}))
    for note in sensor_notes([snapshot], sensors, with_fans=False):
        say("  " + note)
    say("  a fan that reads 0 here may be idle; the run reports a fan that reads 0 throughout as an unreliable sensor")


def sample_only(args, sensors: Sensors, say=print, sleep=time.sleep, clock=time.monotonic) -> list:
    """Sample the sensors for args.sample_only seconds with no benchmark running, and report what they give."""
    started = clock()
    samples = []
    while True:
        elapsed = clock() - started
        samples.append(sensors.read(elapsed))
        if elapsed >= args.sample_only:
            break
        sleep(args.period)
    last = samples[-1]
    say(f"sampled {len(samples)} times in {last['t_s']:.0f} s: package {mean_of([s['pkg_c'] for s in samples])} C mean, "
        f"{max([s['pkg_c'] for s in samples if s['pkg_c'] is not None], default=None)} C max, "
        f"CPU {mean_of([s['freq_mean_mhz'] for s in samples])} MHz mean, power {mean_of([s['power_w'] for s in samples])} W mean, profile {last['profile']!r}")
    peak = max([s["pkg_c"] for s in samples if s["pkg_c"] is not None], default=None)
    for label in sorted({label for sample in samples for label in sample["fans"]}):
        report = fan_report(samples, label, peak)
        say(f"fan {label}: mean {report['mean_rpm']} max {report['max_rpm']} rpm" + ("" if report["reliable"] else f" - UNRELIABLE: {report['note']}"))
    for note in sensor_notes(samples, sensors, with_fans=False):
        say(note)
    return samples


def parse_args(argv: list | None = None):
    parser = argparse.ArgumentParser(
        description="Measure how the cooling of this laptop limits the sustained speed: for each CPU frequency cap (and optionally each platform "
                    "cooling profile) run the standard benchmark through tools/quant_bench.py for a fixed time while sampling the package and core "
                    "temperature, the CPU frequencies, the package power (when readable), the fan RPM and the platform profile every few seconds. "
                    "A step stops when the package reaches the limit. Writes samples.csv, result.json, table.txt and chart.svg.",
        epilog="The only things software can do about cooling are to measure it and to switch the platform profile. The kit never runs sudo "
               "unless --apply-profile is given, and never installs anything. A fan that reads 0 for the whole run is reported as an unreliable "
               "sensor, not as a stopped fan. Put the physical conditions (stand, cleaned fans, room) into --setup so runs can be compared.")
    parser.add_argument("--model", type=Path, default=ENGINE_ROOT / "models" / "qwen3.6-35b-a3b" / "Qwen3.6-35B-A3B-UD-IQ4_XS-mtp.gguf")
    parser.add_argument("--setup", default="", help="free text naming the physical conditions, kept in every result")
    parser.add_argument("--caps", default=DEFAULT_CAPS_GHZ, metavar="GHZ,GHZ", help=f"CPU frequency caps in GHz (default {DEFAULT_CAPS_GHZ})")
    parser.add_argument("--profiles", default=CURRENT, metavar="NAME,NAME",
                        help="platform profiles to test, from /sys/firmware/acpi/platform_profile_choices (default: the current one)")
    parser.add_argument("--apply-profile", action="store_true", help="let the kit switch the platform profile with `sudo -A tee` (asks for the password in a "
                        "dialog) and put the first one back at the end")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S, help=f"seconds per step; {STARTUP_ALLOWANCE_S:.0f} s for the model to load are added")
    parser.add_argument("--limit", type=float, default=DEFAULT_LIMIT_C, help="stop a step when the package reaches this many degrees; also the server's own limit")
    parser.add_argument("--start-temp", type=float, default=DEFAULT_START_TEMP_C, help="cool down to this package temperature before each step")
    parser.add_argument("--cool-timeout", type=float, default=900.0)
    parser.add_argument("--period", type=float, default=SAMPLE_PERIOD_S, help="seconds between samples")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="environment variable for the engine, repeatable")
    parser.add_argument("--engine-args", type=shlex.split, default=[], metavar='"ARGS"', help="more engine arguments, one quoted string")
    parser.add_argument("--prewarm", action="store_true", help="read the model into the page cache before each server start")
    parser.add_argument("--force", action="store_true", help="measure although the machine is busy")
    parser.add_argument("--out-dir", type=Path, help="result directory (must not exist; default run/experiments/cooling/<time>)")
    parser.add_argument("--png", action="store_true", help="also render chart.png with headless Chrome")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and the commands and run nothing")
    parser.add_argument("--sample-only", type=float, metavar="SECONDS", help="only sample the sensors for this long (no benchmark): checks the sensors")
    args = parser.parse_args(argv)
    args.env = {"STRATA_POOL_SPIN_US": "0", **quant_bench.parse_env(args.env)}
    return args


def main(argv: list | None = None) -> int:
    args = parse_args(argv)
    sensors = Sensors()
    if args.sample_only is not None:
        if args.sample_only <= 0:
            raise SystemExit("cooling-kit: --sample-only wants a positive number of seconds")
        samples = sample_only(args, sensors)
        if args.out_dir is not None:
            args.out_dir.mkdir(parents=True)
            write_samples_csv(args.out_dir / "samples.csv", "sample_only", samples, True)
        return 0
    if not args.model.exists():
        raise SystemExit(f"cooling-kit: {args.model} does not exist")
    cpucap = load_cpucap()
    caps = parse_caps(args.caps, cpucap)
    current = sensors.profile()
    choices = (read_text(PLATFORM_PROFILE_CHOICES) or "").split()
    profiles = [current if name == CURRENT else name for name in args.profiles.split(",")]
    if None in profiles:
        raise SystemExit("cooling-kit: the platform profile cannot be read, so 'current' is unknown; name the profiles or fix the sensor")
    if len(set(profiles)) != len(profiles):
        raise SystemExit("cooling-kit: --profiles lists a profile twice")
    check_profiles(profiles, current, choices, args.apply_profile)
    steps = build_steps(caps, profiles)
    out_dir = args.out_dir or ENGINE_ROOT / "run" / "experiments" / "cooling" / time.strftime("%Y%m%d-%H%M%S")
    if args.dry_run:
        print_plan(args, steps, sensors, out_dir, current)
        return 0
    return run_all(args, steps, sensors, out_dir, current)


def run_all(args, steps: list, sensors: Sensors, out_dir: Path, current: str | None, say=lambda text: print(text, file=sys.stderr, flush=True)) -> int:
    out_dir.mkdir(parents=True)
    saved_cap = machine_state.cpu_cap_snapshot()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    results, series, notes = [], {}, []
    previous = quant_bench.install_stop_handlers()
    active_profile = current
    restored = {}
    try:
        for step in steps:
            if step["profile"] != active_profile:
                problem = apply_profile(step["profile"])
                if problem is not None:
                    say(f"cooling-kit: cannot switch to {step['profile']}: {problem}")
                    break
                active_profile = step["profile"]
            summary, samples, _ = run_step(step, args, sensors, out_dir, say)
            results.append(summary)
            series[step["label"]] = samples
            write_samples_csv(out_dir / "samples.csv", step["label"], samples, header=len(results) == 1)
            notes = sensor_notes([sample for samples_of in series.values() for sample in samples_of], sensors)
            write_result(out_dir, args, results, notes, started_at, restored)
    except SystemExit as stop:
        say(f"cooling-kit: {stop}")
    finally:
        for number in quant_bench.STOP_SIGNALS:
            signal.signal(number, signal.SIG_IGN)
        if active_profile != current and current is not None:
            problem = apply_profile(current)
            restored["profile"] = f"back to {current}" if problem is None else f"NOT RESTORED: {problem}"
        failures = machine_state.cpu_cap_restore(saved_cap)
        restored["cpu_cap"] = "unchanged" if not failures else "NOT RESTORED: " + "; ".join(failures)
        quant_bench.restore_handlers(previous)
    leftovers = machine_state.running_processes()
    restored["leftover_processes"] = leftovers
    write_result(out_dir, args, results, notes, started_at, restored)
    if results:
        svg = cooling_chart(results, series, args.limit, "Sustained speed and package temperature by frequency cap", f"setup: {args.setup or 'not stated'}; limit {args.limit:g} C")
        (out_dir / "chart.svg").write_text(svg, encoding="utf-8")
        if args.png and not bench_chart.to_png((out_dir / "chart.svg").resolve(), (out_dir / "chart.png").resolve(), bench_chart.svg_size(svg)):
            say("cooling-kit: the PNG could not be made")
        print(format_table(results))
    for note in notes:
        print(note)
    for key, value in restored.items():
        print(f"restored {key}: {value}")
    print(f"results in {out_dir}")
    return 0 if results and not leftovers and "NOT RESTORED" not in json.dumps(restored) else 3


if __name__ == "__main__":
    sys.exit(main())
