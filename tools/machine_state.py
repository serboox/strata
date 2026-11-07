import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import quant_bench

GIB = 1 << 30
CPU_ROOT = Path("/sys/devices/system/cpu")
USAGE_FILE = Path.home() / ".claude" / "usage.json"
PROCESS_MARKERS = ("strata-q36", "llama-server", "llama-bench", "llama-perplexity", "serve/server.py")
HELPER_SCRIPTS = ("machine_state", "cpu-hog-watchdog")
HELPER_INTERPRETERS = ("bash", "sh", "python", "python3")
HELPER_PROGRAMS = ("notify-send",)


def read_usage(path: Path = USAGE_FILE) -> dict:
    """The utilization and reset time of each usage window, or an empty dict when the file is missing or not JSON."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {name: {"utilization": window.get("utilization"), "resets_at": window.get("resets_at")}
            for name, window in data.items() if isinstance(window, dict) and window.get("utilization") is not None}


def meminfo_gib(text: str) -> dict:
    values = {}
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        if name in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            values[name] = int(rest.split()[0]) * 1024
    swap_used = values.get("SwapTotal", 0) - values.get("SwapFree", 0)
    return {"total_gib": round(values.get("MemTotal", 0) / GIB, 1), "available_gib": round(values.get("MemAvailable", 0) / GIB, 1),
            "swap_used_gib": round(swap_used / GIB, 2)}


def is_helper(command: str) -> bool:
    """True for this tool and for the CPU watchdog, whose script and open warning dialog carry the name of the stuck server in their arguments."""
    words = command.split(" ")
    program = Path(words[0]).name
    if program in HELPER_PROGRAMS:
        return True
    return program in HELPER_INTERPRETERS and any(Path(word).stem in HELPER_SCRIPTS for word in words[1:3])


def running_processes(proc: Path = Path("/proc"), markers: tuple = PROCESS_MARKERS) -> list:
    """Command lines (shortened) of the processes that would disturb a measurement; this process and its parents are left out."""
    me = os.getpid()
    found = []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()
        except OSError:
            continue
        if command and any(marker in command.split(" ")[0] or f" {marker}" in command for marker in markers) and not is_helper(command):
            found.append(f"{entry.name}: {command[:100]}")
    return found


def cpu_cap_snapshot(root: Path = CPU_ROOT) -> dict:
    """The highest-frequency limit (kHz text) of every CPU by file path, for `cpu_cap_restore`."""
    snapshot = {}
    for path in sorted(root.glob("cpu[0-9]*/cpufreq/scaling_max_freq")):
        try:
            snapshot[str(path)] = path.read_text().strip()
        except OSError:
            continue
    return snapshot


def cpu_cap_restore(snapshot: dict) -> list:
    """Write the saved limits back where they differ; returns a message for each file that could not be put back."""
    failures = []
    for name, saved in snapshot.items():
        path = Path(name)
        try:
            if path.read_text().strip() == saved:
                continue
            path.write_text(saved)
        except OSError as error:
            failures.append(f"{name}: {error}")
    return failures


def collect(disk_path: str = "/home", max_load: float = 1.0, min_available_gib: float = 0.0) -> dict:
    load = os.getloadavg()
    memory = meminfo_gib(Path("/proc/meminfo").read_text())
    state = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "usage": read_usage(), "load": [round(value, 2) for value in load], "memory": memory,
             "disk_free_gib": round(shutil.disk_usage(disk_path).free / GIB, 1), "cpu_temp_c": quant_bench.package_temperature(),
             "gpu_temp_c": quant_bench.gpu_temperature(), "turbo": quant_bench.turbo_state(), "processes": running_processes()}
    problems = []
    if load[0] > max_load:
        problems.append(f"load average {load[0]:.2f} is above {max_load:g}")
    if memory["available_gib"] < min_available_gib:
        problems.append(f"{memory['available_gib']} GiB available, {min_available_gib:g} wanted")
    if state["turbo"] != "off":
        problems.append(f"turbo is {state['turbo']}")
    if state["processes"]:
        problems.append("engine or server processes are running")
    for name, window in state["usage"].items():
        if window["utilization"] is not None and window["utilization"] >= 99:
            problems.append(f"usage window {name} is at {window['utilization']} %")
    state["problems"] = problems
    state["quiet"] = not problems
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description="Print the state of the machine before a measurement; exit status 1 when it is not quiet.")
    parser.add_argument("--disk", default="/home", help="path whose free space is reported")
    parser.add_argument("--max-load", type=float, default=1.0)
    parser.add_argument("--min-available-gib", type=float, default=0.0)
    parser.add_argument("--out", type=Path, help="also write the JSON here (must not exist)")
    args = parser.parse_args()
    if args.out is not None and args.out.exists():
        parser.error(f"{args.out} exists")
    state = collect(args.disk, args.max_load, args.min_available_gib)
    text = json.dumps(state, indent=1)
    print(text)
    if args.out is not None:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0 if state["quiet"] else 1


if __name__ == "__main__":
    sys.exit(main())
