import argparse
import importlib.util
import inspect
import json
import os
import queue
import re
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

GIB = 1 << 30
ENGINE_ROOT = Path(__file__).resolve().parents[1]
THERMAL_ROOT = Path("/sys/class/thermal")
PRESSURE_FILE = Path("/proc/pressure/memory")
CGROUP_ROOT = Path("/sys/fs/cgroup")
USER_SERVICE = CGROUP_ROOT / "user.slice" / f"user-{os.getuid()}.slice" / f"user@{os.getuid()}.service"
PROFILE_OF_BUDGET = {"none": "auto", "22G": "max", "16G": "fast", "12G": "balanced", "8G": "lean"}
LIVE_STATES = ("active", "activating", "deactivating")
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
PROMPTS = {
    "code": "Write a complete, well documented Python module that implements a thread-safe LRU cache with a time-to-live per entry, "
            "type hints and unit tests that use unittest. Explain the design briefly first.",
    "text": "Explain step by step how a mixture-of-experts language model routes a token to its experts, why only a few experts are "
            "active per token, and what that means for memory bandwidth on a machine with a small GPU and a lot of system RAM.",
}
SESSION_LINE = re.compile(r"session (\d+): prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms .*?, (\d+) generated in (\d+) ms")
DRAFTS = re.compile(r"drafts accepted (\d+) of (\d+)")
STEPS = re.compile(r"\[spec (\d+) steps: draft (\d+) verify (\d+) commit (\d+) ms; plain (\d+) steps (\d+) ms\]")


def parse_session_line(line: str) -> dict | None:
    """The numbers of the server's per-request log line, or None for any other line."""
    match = SESSION_LINE.search(line)
    if match is None:
        return None
    _, prompt, reused, read, prompt_ms, generated, decode_ms = (int(value) for value in match.groups())
    row = {"prompt": prompt, "reused": reused, "read": read, "prompt_ms": prompt_ms, "generated": generated, "decode_ms": decode_ms,
           "tok_s": round(generated * 1000 / decode_ms, 2) if decode_ms else 0.0, "accepted": 0, "offered": 0}
    drafts = DRAFTS.search(line)
    if drafts is not None:
        row["accepted"], row["offered"] = int(drafts.group(1)), int(drafts.group(2))
    steps = STEPS.search(line)
    if steps is not None:
        row.update(zip(("spec_steps", "draft_ms", "verify_ms", "commit_ms", "plain_steps", "plain_ms"), (int(value) for value in steps.groups())))
    return row


def package_temperature(root: Path = THERMAL_ROOT) -> float | None:
    for zone in sorted(root.glob("thermal_zone*")):
        try:
            if (zone / "type").read_text().strip() == "x86_pkg_temp":
                return int((zone / "temp").read_text()) / 1000
        except (OSError, ValueError):
            continue
    return None


def gpu_temperature() -> float | None:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                timeout=10, check=False)
        return float(result.stdout.split()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def hottest(*readings) -> float | None:
    known = [reading for reading in readings if reading is not None]
    return max(known) if known else None


def wait_until_cool(limit: float, timeout: float, read=package_temperature, sleep=time.sleep, clock=time.monotonic, check=lambda: None) -> tuple:
    """(temperature, seconds waited, state): the state is True at or below the limit, False when the timeout ran out first and
    None when the sensor cannot be read. `check` runs on every wait and may raise to stop."""
    started = clock()
    while True:
        temperature = read()
        if temperature is None:
            return None, clock() - started, None
        if temperature <= limit:
            return temperature, clock() - started, True
        if clock() - started >= timeout:
            return temperature, clock() - started, False
        check()
        sleep(5)


def memory_stall(text: str, kind: str = "full") -> float:
    """The share of the last ten seconds in which every task (`full`) or at least one task (`some`) waited for memory, from the text of a
    memory.pressure file."""
    for line in text.splitlines():
        if line.startswith(kind):
            fields = dict(part.split("=", 1) for part in line.split()[1:])
            return float(fields.get("avg10", 0.0))
    return 0.0


def current_stall(path: Path = PRESSURE_FILE) -> float:
    try:
        return memory_stall(path.read_text())
    except OSError:
        return 0.0


def other_stall(unit: str, root: Path = USER_SERVICE, fallback=current_stall) -> float:
    """The worst `some avg10` memory pressure among the user's other programs: every cgroup of the user manager (a slice is entered, a
    service or scope is read as it is) except the server's own scope. The system-wide `full` figure stands in when no cgroup can be read,
    so it then counts the server too."""
    own = f"{unit}.scope"
    worst, seen = 0.0, False
    try:
        children = list(root.iterdir())
    except OSError:
        children = []
    groups = []
    for child in children:
        if child.name.endswith(".slice"):
            try:
                groups += [grandchild for grandchild in child.iterdir() if grandchild.is_dir()]
            except OSError:
                continue
        elif child.is_dir():
            groups.append(child)
    for directory in groups:
        if directory.name == own:
            continue
        try:
            worst = max(worst, memory_stall((directory / "memory.pressure").read_text(), "some"))
        except (OSError, ValueError):
            continue
        seen = True
    return worst if seen else fallback()


def evict_page_cache(paths: list) -> list:
    """Ask the kernel to drop the cached pages of these files (pages another run left behind would make a smaller budget look better
    than it is); returns the paths that could not be asked. Pages a process still has mapped stay."""
    failed = []
    for path in paths:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError:
            failed.append(str(path))
            continue
        try:
            os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            failed.append(str(path))
        finally:
            os.close(descriptor)
    return failed


def prewarm_page_cache(path: Path, chunk: int = 16 << 20) -> int:
    """Read the whole file once so every page is in the page cache; returns the bytes read."""
    total = 0
    with open(path, "rb", buffering=0) as handle:
        while True:
            data = handle.read(chunk)
            if not data:
                return total
            total += len(data)


def machine_problems(file_size: int, budget: str, load: float, available: int, max_load: float) -> list:
    """Why a measurement now would not be trustworthy; an empty list means the machine is quiet enough."""
    problems = []
    if load > max_load:
        problems.append(f"the load average is {load:.2f} (limit {max_load:g}): other programs are working")
    need = file_size + 2 * GIB
    if budget in ("none", "22G") and available < need:
        problems.append(f"{available / GIB:.1f} GiB of memory is available and the model needs {need / GIB:.1f} GiB in the page cache")
    return problems


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(2)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def usable(row: dict) -> bool:
    """A measured request counts only with exactly one log line and a start at or below the temperature limit (or no sensor)."""
    return not row["warmup"] and "tok_s" in row and row.get("stat_lines") == 1 and row.get("started_cool") is not False


def summarize(rows: list) -> dict:
    """Per prompt, the usable measured requests: the speeds, their median, and the share of drafts accepted; also how many
    measured requests were left out because of a missing or ambiguous log line or a start that was too hot."""
    summary = {}
    for prompt in sorted({row["prompt_name"] for row in rows}):
        measured = [row for row in rows if row["prompt_name"] == prompt and not row["warmup"]]
        good = [row for row in measured if usable(row)]
        entry = {"runs": len(good), "left_out": len(measured) - len(good)}
        if good:
            speeds = [row["tok_s"] for row in good]
            offered = sum(row["offered"] for row in good)
            counted = [row for row in good if row.get("spec_steps")]
            if counted:
                steps = sum(row["spec_steps"] for row in counted)
                entry["step_ms"] = {name: round(sum(row[f"{name}_ms"] for row in counted) / steps, 2) for name in ("draft", "verify", "commit")}
                entry["step_ms"]["total"] = round(sum(entry["step_ms"].values()), 2)
                entry["tokens_per_step"] = round(sum(row["generated"] for row in counted) / steps, 2)
            entry.update({"tok_s": speeds, "median": round(statistics.median(speeds), 2), "min": min(speeds), "max": max(speeds),
                          "peak_c": hottest(*(row.get("peak_c") for row in measured)),
                          "accepted_share": round(sum(row["accepted"] for row in good) / offered, 3) if offered else None})
        summary[prompt] = entry
    return summary


def turbo_state(path: Path = Path("/sys/devices/system/cpu/intel_pstate/no_turbo")) -> str:
    try:
        return "off" if path.read_text().strip() == "1" else "on"
    except OSError:
        return "unknown"


def load_launcher(root: Path):
    spec = importlib.util.spec_from_file_location("strata_launch_for_bench", root / "tools" / "strata_launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_env(items: list) -> dict:
    """KEY=VALUE strings from the command line as a dict; a string without `=` stops the run."""
    pairs = {}
    for item in items:
        key, equals, value = item.partition("=")
        if not equals or not key:
            raise SystemExit(f"quant-bench: --env wants KEY=VALUE, not {item!r}")
        pairs[key] = value
    return pairs


def write_config(root: Path, model: Path, budget: str, port: int, mtp: int | None, work: Path, max_temp: float = 0.0,
                 exe: Path | None = None, env: dict | None = None, workers: int | None = None, engine_args: list | None = None,
                 cpu_max_mhz: float = 0.0) -> tuple:
    """The server config the production launcher would write for this budget, pointed at `model` and, when `max_temp` is above zero,
    with the launcher's temperature limit in degrees; returns the config path, the log path and the memory budget of the profile."""
    launcher = load_launcher(root)
    launcher.MODEL_FILE = model
    launcher.RUN_DIR = work
    profile = dict(launcher.PROFILES[PROFILE_OF_BUDGET[budget]])
    if mtp is not None:
        profile["mtp"] = mtp
    if workers is not None:
        profile["workers"] = workers
    if max_temp > 0:
        if "max_temp" not in inspect.signature(launcher.build_config).parameters:
            raise SystemExit("quant-bench: the launcher of this engine root has no temperature limit (--max-temp 0 measures without one)")
        config = launcher.build_config(profile, port, "greedy", max_temp)
    else:
        config = launcher.build_config(profile, port, "greedy")
    if cpu_max_mhz:
        if not (root / "serve" / "cpucap.py").exists():
            raise SystemExit("quant-bench: the server of this engine root has no CPU frequency cap (--cpu-max-mhz 0 measures without one)")
        config.setdefault("thermal", {})["cpu_max_mhz"] = cpu_max_mhz
    config["log"] = str(work / "server.log")
    if engine_args:
        config["args"] = [*config["args"], *engine_args]
    if exe is not None:
        config["exe"] = str(exe)
    if env:
        config["env"] = {**config.get("env", {}), **env}
    path = work / "config.json"
    path.write_text(json.dumps(config, indent=1) + "\n", encoding="utf-8")
    return path, Path(config["log"]), profile["ram"]


# `setpriv --pdeathsig TERM` makes the kernel send SIGTERM to the server when the bench is gone, so a bench that is killed outright
# cannot leave a server behind.  Without setpriv the server just runs as before.
PARENT_DEATH_PREFIX = ["setpriv", "--pdeathsig", "TERM"]


def start_server(root: Path, config: Path, port: int, ram: str | None, unit: str, work: Path) -> subprocess.Popen:
    command = ["systemd-run", "--user", "--scope", "--quiet", f"--unit={unit}"]
    if ram:
        command += ["-p", f"MemoryMax={ram}", "-p", "MemorySwapMax=0"]
    command += ["--", *(PARENT_DEATH_PREFIX if shutil.which("setpriv") else []), str(root / ".venv" / "bin" / "python"), str(root / "serve" / "server.py"), "--engine", "strata", "--config", str(config),
                "--port", str(port)]
    environment = {**os.environ, "STRATA_KEEP_LEFTOVERS": "1"}   # this run must end only the engine it started, never another one's
    with open(work / "server.out", "wb") as output:
        return subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True, env=environment)


def wait_ready(port: int, process: subprocess.Popen, timeout: float, check=lambda: None) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        check()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3):
                return True
        except OSError:
            time.sleep(3)
    return False


def scope_state(unit: str) -> str:
    try:
        result = subprocess.run(["systemctl", "--user", "show", "-p", "ActiveState", "--value", f"{unit}.scope"], capture_output=True, text=True,
                                timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def stop_server(unit: str, process: subprocess.Popen | None) -> list:
    """Stop the server scope for certain and return what could not be done (an empty list means it is gone)."""
    problems = []

    def systemctl(*arguments):
        try:
            return subprocess.run(["systemctl", "--user", *arguments], capture_output=True, text=True, timeout=120, check=False)
        except (OSError, subprocess.SubprocessError) as error:
            problems.append(f"systemctl {arguments[0]}: {error}")
            return None

    result = systemctl("stop", f"{unit}.scope")
    if result is not None and result.returncode != 0 and "not loaded" not in result.stderr:
        problems.append(f"systemctl stop: {result.stderr.strip()[:120]}")
    if process is not None:
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                process.kill()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                problems.append("systemd-run did not exit")
    if scope_state(unit) in LIVE_STATES:
        systemctl("kill", "--signal=SIGKILL", "--kill-whom=all", f"{unit}.scope")
        systemctl("stop", f"{unit}.scope")
        time.sleep(2)
        if scope_state(unit) in LIVE_STATES:
            problems.append("the scope is still active")
    return problems


def install_stop_handlers() -> dict:
    """Turn SIGINT, SIGTERM and SIGHUP into SystemExit so the cleanup in `finally` runs; returns the handlers to restore."""
    def stop(signum, frame):
        raise SystemExit(f"quant-bench: stopped by signal {signum}")
    return {number: signal.signal(number, stop) for number in STOP_SIGNALS}


def restore_handlers(previous: dict) -> None:
    for number, handler in previous.items():
        signal.signal(number, handler)


def ask(port: int, model_name: str, prompt: str, max_tokens: int, timeout: float = 900.0) -> tuple:
    body = {"model": model_name, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0,
            "presence_penalty": 0, "frequency_penalty": 0, "stream": False}
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    try:
        message = json.loads(payload)["choices"][0]["message"]
        text = message.get("content") or message.get("reasoning_content") or ""
    except (ValueError, KeyError, IndexError, TypeError):
        text = ""
    return time.monotonic() - started, text


def ask_watched(port: int, model_name: str, prompt: str, max_tokens: int, timeout: float, check, poll: float = 5.0, send=ask):
    """`ask` with a total deadline; `check` runs every few seconds and may raise to abandon the request (the caller then stops
    the server, which ends the blocked request)."""
    answers = queue.Queue()

    def work():
        try:
            answers.put(("ok", send(port, model_name, prompt, max_tokens, timeout)))
        except Exception as error:  # noqa: BLE001 - handed to the waiting caller
            answers.put(("error", error))

    threading.Thread(target=work, daemon=True).start()
    deadline = time.monotonic() + timeout
    while True:
        try:
            kind, value = answers.get(timeout=poll)
        except queue.Empty:
            check()
            if time.monotonic() > deadline:
                raise SystemExit(f"quant-bench: no answer within {timeout:.0f} s") from None
            continue
        if kind == "error":
            raise value
        return value


def log_new_lines(log: Path, offset: int) -> list:
    if not log.exists():
        return []
    with open(log, "rb") as handle:
        handle.seek(offset)
        return handle.read().decode("utf-8", "replace").splitlines()


def collect_stats(log: Path, offset: int, timeout: float, sleep=time.sleep, clock=time.monotonic) -> tuple:
    """(the numbers of the request that just finished or None, how many per-request lines appeared): the engine writes its line
    after it has sent the answer, so wait for it for a while."""
    deadline = clock() + timeout
    while True:
        rows = [row for row in map(parse_session_line, log_new_lines(log, offset)) if row]
        if rows or clock() >= deadline:
            return (rows[-1] if rows else None), len(rows)
        sleep(0.5)


def measure(args, work: Path, say=print) -> dict:
    model = args.model.resolve()
    if turbo_state() != "off" and not args.allow_turbo:
        raise SystemExit("quant-bench: refusing to measure with the CPU turbo boost on or unknown (--allow-turbo overrides)")
    config, log, ram = write_config(args.engine_root, model, args.budget, args.port, args.mtp, work, args.max_temp, args.exe, args.env, args.workers, args.engine_args, args.cpu_max_mhz)
    load = os.getloadavg()[0]
    available = next((int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:")), 0)
    problems = machine_problems(model.stat().st_size, args.budget, load, available, args.max_load)
    if problems and not args.force:
        raise SystemExit("quant-bench: refusing to measure: " + "; ".join(problems) + " (--force overrides)")
    if port_in_use(args.port):
        raise SystemExit(f"quant-bench: port {args.port} is already in use, another server would answer in place of this one")
    for failed in evict_page_cache([model, *args.evict]):
        say(f"quant-bench: could not drop the cached pages of {failed}")
    if args.prewarm:
        say(f"quant-bench: read {prewarm_page_cache(model) / GIB:.1f} GiB of the model into the page cache")
    unit = f"quantbench-{os.getpid()}"
    process = None
    rows = []
    aborted = None
    previous = install_stop_handlers()
    peak, peak_cpu, peak_gpu = [], [], []

    def check():
        cpu_now, gpu_now = package_temperature(), gpu_temperature()
        peak_cpu.append(cpu_now)
        peak_gpu.append(gpu_now)
        peak.append(hottest(cpu_now, gpu_now))
        if process is not None and process.poll() is not None:
            raise SystemExit("quant-bench: the server died during the run")
        stall = other_stall(unit)
        if stall > args.max_stall:
            raise SystemExit(f"quant-bench: other programs are stalling on memory ({stall:.1f} % of the last 10 s): someone else needs it, stopping")

    try:
        signal.pthread_sigmask(signal.SIG_BLOCK, STOP_SIGNALS)   # a stop request must not arrive before the server is known to the cleanup
        try:
            process = start_server(args.engine_root, config, args.port, ram, unit, work)
        finally:
            signal.pthread_sigmask(signal.SIG_UNBLOCK, STOP_SIGNALS)
        if not wait_ready(args.port, process, args.ready_timeout, check):
            tail = (work / "server.out").read_text(errors="replace").splitlines()[-8:] if (work / "server.out").exists() else []
            raise SystemExit("quant-bench: the server did not become ready: " + " | ".join(tail))
        model_name = json.loads(config.read_text())["model_name"]
        for prompt_name in args.prompts:
            for index in range(args.warmup + args.repeats):
                check()
                temperature, waited, cool = wait_until_cool(args.cool, args.cool_timeout, check=check)
                offset = log.stat().st_size if log.exists() else 0
                peak.clear()
                peak_cpu.clear()
                peak_gpu.clear()
                check()
                wall, answer = ask_watched(args.port, model_name, PROMPTS[prompt_name], args.max_tokens, args.request_timeout, check)
                check()
                stats, lines = collect_stats(log, offset, args.log_timeout)
                row = {"prompt_name": prompt_name, "index": index, "warmup": index < args.warmup, "wall_s": round(wall, 2),
                       "temp_c": temperature, "cool_wait_s": round(waited), "started_cool": cool, "peak_c": hottest(*peak), "peak_cpu_c": hottest(*peak_cpu), "peak_gpu_c": hottest(*peak_gpu), "stat_lines": lines,
                       "answer": answer}
                if stats:
                    row.update(stats)
                rows.append(row)
                say(json.dumps(row))
    except SystemExit as stop:
        aborted = str(stop.code)
    finally:
        for number in STOP_SIGNALS:
            signal.signal(number, signal.SIG_IGN)   # a second signal must not cut the cleanup short
        cleanup_problems = stop_server(unit, process)
        restore_handlers(previous)
        for problem in cleanup_problems:
            say(f"quant-bench: CLEANUP PROBLEM {problem}")
    kept_log = args.save_log if args.save_log is not None else args.out.with_name(args.out.name + ".log")
    if log.exists():
        try:
            shutil.copyfile(log, kept_log)
        except OSError as error:
            say(f"quant-bench: could not keep the server log: {error}")
    console = log.with_name("server.out")
    if console.exists():
        try:
            shutil.copyfile(console, args.out.with_name(args.out.name + ".log.console"))
        except OSError as error:
            say(f"quant-bench: could not keep the server console output: {error}")
    commit = subprocess.run(["git", "-C", str(args.engine_root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    return {"label": args.label, "engine_commit": commit, "model": str(model), "model_gib": round(model.stat().st_size / GIB, 2), "budget": args.budget,
            "mtp": args.mtp, "turbo": turbo_state(), "prewarm": args.prewarm, "engine_args": args.engine_args, "max_temp_c": args.max_temp, "cpu_max_mhz": args.cpu_max_mhz, "exe": str(args.exe) if args.exe else None, "env": args.env, "load_before": round(load, 2), "available_gib_before": round(available / GIB, 1), "rows": rows,
            "summary": summarize(rows), "aborted": aborted, "cleanup_problems": cleanup_problems, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure the decode speed of the Strata server on one model file at one memory budget.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--budget", choices=list(PROFILE_OF_BUDGET), default="none")
    parser.add_argument("--prompts", nargs="+", choices=list(PROMPTS), default=["code"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--mtp", type=int, help="drafts per step (default: the profile's)")
    parser.add_argument("--workers", type=int, help="CPU expert worker threads (default: the profile's)")
    parser.add_argument("--cpu-max-mhz", type=float, default=0.0, metavar="MHZ",
                        help="the server holds the highest CPU frequency at this many MHz for the run and puts it back (default 0: no cap; "
                             "the limit files must be writable, see the frequency cap in docs/TUNING.md)")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--cool", type=float, default=55.0, help="start each request with the CPU package at or below this many degrees")
    parser.add_argument("--cool-timeout", type=float, default=600.0)
    parser.add_argument("--max-temp", type=float, default=80.0, help="the server's own temperature limit in degrees Celsius for the CPU and the GPU "
                        "(0 = none); the launcher's default")
    parser.add_argument("--allow-turbo", action="store_true", help="measure although the CPU turbo boost is on")
    parser.add_argument("--save-log", type=Path, help="copy the server log (with the engine's own counters) here when the run ends (default: <out>.log)")
    parser.add_argument("--evict", type=Path, action="append", default=[], help="another model file whose cached pages are dropped before the run, repeatable "
                        "(the model under test is always dropped)")
    parser.add_argument("--engine-args", type=shlex.split, default=[], metavar='"ARGS"', help="more arguments for the engine, one quoted string, e.g. \"--expert-slots 32\"")
    parser.add_argument("--prewarm", action="store_true", help="read the whole model file into the page cache before the server starts")
    parser.add_argument("--exe", type=Path, help="engine binary to run instead of the launcher's build/strata-q36")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="environment variable for the engine, repeatable")
    parser.add_argument("--max-load", type=float, default=1.5)
    parser.add_argument("--max-stall", type=float, default=5.0, help="stop when another program of the user spent this percent of the last 10 s waiting for memory")
    parser.add_argument("--ready-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=900.0, help="total seconds one request may take")
    parser.add_argument("--log-timeout", type=float, default=30.0, help="seconds to wait for the server's line about a finished request")
    parser.add_argument("--engine-root", type=Path, default=ENGINE_ROOT, help="checkout holding build/strata-q36 and the launcher")
    parser.add_argument("--out", type=Path, required=True, help="JSON file for the result (must not exist); an aborted run writes <out>.aborted")
    parser.add_argument("--force", action="store_true", help="measure although the machine is busy")
    args = parser.parse_args()
    args.env = parse_env(args.env)
    if args.exe is not None:
        args.exe = args.exe.expanduser().resolve()
    if args.exe is not None and not args.exe.exists():
        parser.error(f"{args.exe} does not exist")
    if args.out.exists():
        parser.error(f"{args.out} exists")
    if not args.model.exists():
        parser.error(f"{args.model} does not exist")
    with tempfile.TemporaryDirectory(prefix="quant-bench-") as work:
        result = measure(args, Path(work), lambda text: print(text, file=sys.stderr, flush=True))
    target = args.out.with_name(args.out.name + ".aborted") if result["aborted"] else args.out
    target.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=1))
    if result["aborted"] or result["cleanup_problems"]:
        print(f"quant-bench: {'ABORTED: ' + result['aborted'] if result['aborted'] else 'cleanup problems'}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
