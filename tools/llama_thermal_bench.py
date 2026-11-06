#!/usr/bin/env python3
import argparse
import glob
import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import quant_bench

from serve import cpucap, thermal

DEFAULT_PORT = 8091
DEFAULT_MAX_TEMP_C = 80.0
DEFAULT_CPU_MAX_MHZ = 2100.0
COOL_START_C = 55.0
COOL_WAIT_SECONDS = 900.0
READY_TIMEOUT_SECONDS = 600.0
READY_POLL_SECONDS = 2.0
STOP_TIMEOUT_SECONDS = 60.0
REQUEST_TIMEOUT_SECONDS = 1800.0
SAMPLE_SECONDS = 2.0
GIB = 1 << 30
CPUFREQ_FILES = "/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq"


def server_command(args: argparse.Namespace) -> list:
    command = ["nice", "-n", "5", str(args.server), "-m", str(args.model), "-ngl", "99", "-t", str(args.threads), "-fa", "on",
               "-c", str(args.ctx), "-b", str(args.batch), "-ub", str(args.ubatch), "--host", "127.0.0.1", "--port", str(args.port),
               "--no-webui", "--parallel", "1"]
    if args.n_cpu_moe is not None:
        command += ["--n-cpu-moe", str(args.n_cpu_moe)]
    return command + list(args.server_arg)


def request_body(prompt: str, max_tokens: int, ignore_eos: bool) -> dict:
    return {"model": "local", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0,
            "stream": False, "ignore_eos": ignore_eos}


def parse_timings(payload: dict) -> dict:
    """The per-request figures of the server's own `timings` block; the server measures them on its wall clock, so the pauses of the
    thermal governor are inside them."""
    timings = payload.get("timings")
    if timings is None:
        raise ValueError("the answer has no timings block (is this llama-server?)")
    if not isinstance(timings, dict):
        raise TypeError(f"the timings block is a {type(timings).__name__}, not an object")
    fields = ("prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms", "predicted_per_second")
    missing = [name for name in fields if not isinstance(timings.get(name), (int, float))]
    if missing:
        raise ValueError(f"the timings block lacks {', '.join(missing)}")
    return {name: float(timings[name]) for name in fields}


def post_chat(port: int, body: dict, timeout: float = REQUEST_TIMEOUT_SECONDS) -> dict:
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def drift_windows(count: int, first: int = 5, last: int = 3) -> tuple:
    """How many of the first and of the last requests are averaged to show the drift; the two windows never overlap unless there is
    a single request."""
    if count < 2:
        return count, count
    last_n = min(last, count // 2)
    return min(first, count - last_n), last_n


def summarize(rows: list, timeline: list, first: int = 5, last: int = 3) -> dict:
    """Median and range of the decode speed of the measured requests, the drift between the first and the last ones (the window
    sizes are in the result), and the peaks of the timeline."""
    speeds = [row["predicted_per_second"] for row in rows]
    summary = {"requests": len(rows)}
    if speeds:
        first_n, last_n = drift_windows(len(speeds), first, last)
        summary.update({"decode_median": round(statistics.median(speeds), 2), "decode_min": round(min(speeds), 2),
                        "decode_max": round(max(speeds), 2), "decode_first": round(statistics.fmean(speeds[:first_n]), 2),
                        "decode_first_n": first_n, "decode_last": round(statistics.fmean(speeds[-last_n:]), 2), "decode_last_n": last_n,
                        "prompt_median": round(statistics.median(row["prompt_per_second"] for row in rows), 1)})
    for key in ("pkg", "gpu"):
        values = [sample[key] for sample in timeline if isinstance(sample.get(key), (int, float))]
        summary[f"{key}_peak"] = max(values) if values else None
    freqs = [sample["mhz"] for sample in timeline if isinstance(sample.get("mhz"), (int, float))]
    summary["mhz_mean"] = round(statistics.fmean(freqs)) if freqs else None
    return summary


def mean_cpu_mhz(pattern: str = CPUFREQ_FILES) -> float | None:
    values = []
    for path in glob.glob(pattern):
        try:
            values.append(int(Path(path).read_text()) / 1000)
        except (OSError, ValueError):
            continue
    return round(statistics.fmean(values)) if values else None


def temperatures() -> dict:
    return {"cpu": quant_bench.package_temperature(), "gpu": quant_bench.gpu_temperature()}


class Sampler:
    """Writes the package and GPU temperature, the mean CPU clock and the governor's duty every few seconds."""

    def __init__(self, governor, period: float = SAMPLE_SECONDS):
        self.governor = governor
        self.period = period
        self.timeline: list = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._started = time.monotonic()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> list:
        self._stop.set()
        self._thread.join(timeout=5)
        return self.timeline

    def _loop(self) -> None:
        while not self._stop.is_set():
            status = self.governor.status() if self.governor else {}
            self.timeline.append({"t": round(time.monotonic() - self._started, 1), "pkg": quant_bench.package_temperature(),
                                  "gpu": quant_bench.gpu_temperature(), "mhz": mean_cpu_mhz(), "duty": status.get("duty"),
                                  "throttled_s": status.get("throttled_s")})
            self._stop.wait(self.period)


def wait_ready(port: int, process: subprocess.Popen, timeout: float = READY_TIMEOUT_SECONDS, sleep=time.sleep, clock=time.monotonic) -> bool:
    started = clock()
    while clock() - started < timeout:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
                if response.status == 200:
                    return True
        except OSError:
            pass
        sleep(READY_POLL_SECONDS)
    return False


def stop_server(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        pass


def run_cleanup(steps: list) -> None:
    """Runs every step even when one of them raises (an interrupt included); the first error is raised after the last step."""
    first = None
    for step in steps:
        try:
            step()
        except BaseException as error:  # noqa: BLE001 - the remaining steps must still run
            first = first or error
    if first is not None:
        raise first


def problems(args: argparse.Namespace, load: float, available: int) -> list:
    found = []
    if not Path(args.server).is_file():
        found.append(f"no llama-server at {args.server}")
    if not Path(args.model).is_file():
        found.append(f"no model file at {args.model}")
        return found
    needed = Path(args.model).stat().st_size + GIB
    if not args.force and available < needed:
        found.append(f"{available / GIB:.1f} GiB of memory is available, the model needs about {needed / GIB:.1f} GiB in the page cache (--force overrides)")
    if not args.force and load > args.max_load:
        found.append(f"the load average is {load:.1f}, above {args.max_load:.1f}: other work would distort the run (--force overrides)")
    if quant_bench.port_in_use(args.port):
        found.append(f"port {args.port} is already in use")
    return found


def available_memory() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def run(args: argparse.Namespace, say=print) -> dict:
    prompt = quant_bench.PROMPTS[args.prompt]
    result = {"model": str(args.model), "server": str(args.server), "n_cpu_moe": args.n_cpu_moe, "threads": args.threads,
              "max_temp_c": args.max_temp, "cpu_max_mhz": args.cpu_max_mhz, "tokens": args.tokens, "warmup": args.warmup,
              "turbo": quant_bench.turbo_state(), "started": time.strftime("%F %T")}
    cap = cpucap.CpuCap(args.cpu_max_mhz, log=say) if args.cpu_max_mhz else None
    process = governor = sampler = log = None
    pid_box = {"pid": None}
    previous_handlers = quant_bench.install_stop_handlers()
    try:
        if cap:
            cap.start()
        temperature, waited, state = quant_bench.wait_until_cool(COOL_START_C, COOL_WAIT_SECONDS)
        result["start_temp_c"], result["cool_wait_s"] = temperature, round(waited)
        if state is False:
            say(f"llama-thermal-bench: the package stays at {temperature:.0f} C after {waited:.0f} s, starting anyway")
        command = server_command(args)
        say("llama-thermal-bench: " + " ".join(command))
        log = open(args.work / "llama-server.log", "wb")  # noqa: SIM115 - closed in the finally block
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        pid_box["pid"] = process.pid
        if not wait_ready(args.port, process):
            raise SystemExit(f"llama-thermal-bench: the server did not become ready (see {args.work / 'llama-server.log'})")
        if args.max_temp:
            governor = thermal.Governor(args.max_temp, temperatures, lambda: pid_box["pid"], log=say)
            governor.start()
        sampler = Sampler(governor)
        sampler.start()
        body = request_body(prompt, args.tokens, not args.stop_at_eos)
        rows = []
        for number in range(args.warmup + args.requests):
            timings = parse_timings(post_chat(args.port, body))
            measured = number >= args.warmup
            say(f"llama-thermal-bench: request {number + 1}{'' if measured else ' (warm-up)'}: decode {timings['predicted_per_second']:.1f} tok/s, "
                f"prompt {timings['prompt_per_second']:.0f} tok/s, {timings['predicted_n']:.0f} tokens, package {quant_bench.package_temperature()} C")
            if measured:
                rows.append(timings)
        result["rows"] = rows
    finally:
        timeline = []

        def collect_timeline():
            if sampler:
                timeline.extend(sampler.stop())

        def stop_governor():
            if governor:
                result["governor"] = governor.status()
                governor.stop()
            pid_box["pid"] = None

        run_cleanup([collect_timeline, stop_governor, lambda: stop_server(process), lambda: log and log.close(), lambda: cap and cap.stop(),
                     lambda: quant_bench.restore_handlers(previous_handlers)])
    result["timeline"] = timeline
    result["summary"] = summarize(result.get("rows", []), timeline)
    return result


def parse_args(argv: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode speed of a model under llama-server with the CPU frequency cap and a temperature limit.")
    parser.add_argument("--server", required=True, type=Path, help="the llama-server binary")
    parser.add_argument("--model", required=True, type=Path, help="the GGUF file")
    parser.add_argument("--n-cpu-moe", type=int, default=99, help="layers whose experts stay on the CPU (99 = all of them; fewer puts the experts of the last layers in VRAM)")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--ubatch", type=int, default=512)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--max-temp", type=float, default=DEFAULT_MAX_TEMP_C, help="the temperature limit in C, 0 = none")
    parser.add_argument("--cpu-max-mhz", type=float, default=DEFAULT_CPU_MAX_MHZ, help="the highest CPU frequency in MHz, 0 = leave it")
    parser.add_argument("--prompt", choices=sorted(quant_bench.PROMPTS), default="code")
    parser.add_argument("--tokens", type=int, default=900, help="tokens generated per request")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--requests", type=int, default=8, help="measured requests, back to back")
    parser.add_argument("--stop-at-eos", action="store_true", help="let the model stop by itself (default: always generate --tokens)")
    parser.add_argument("--server-arg", action="append", default=[], help="an extra argument for llama-server, written with an equals sign: --server-arg=--no-mmap (repeat for more)")
    parser.add_argument("--max-load", type=float, default=1.5)
    parser.add_argument("--force", action="store_true", help="run although memory or load look wrong")
    parser.add_argument("--work", type=Path, default=Path("."), help="the folder for the server log")
    parser.add_argument("--out", type=Path, help="write the full result as JSON here")
    args = parser.parse_args(argv)
    if args.tokens < 1 or args.requests < 1 or args.warmup < 0:
        parser.error("--tokens and --requests must be positive, --warmup not negative")
    return args


def main() -> int:
    args = parse_args(sys.argv[1:])
    found = problems(args, os.getloadavg()[0], available_memory())
    if found:
        print("llama-thermal-bench: " + "; ".join(found), file=sys.stderr)
        return 2
    result = run(args)
    print(json.dumps(result["summary"], indent=2))
    if args.out:
        args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
