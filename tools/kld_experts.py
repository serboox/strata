import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(Path(__file__).resolve().parent))
import quant_bench  # noqa: E402
from serve.cpucap import MAX_MHZ, MIN_MHZ, CpuCap  # noqa: E402

EXPERTS_KEY = "qwen35moe.expert_used_count"
DEFAULT_KS = (8, 7, 6, 5, 4)
DEFAULT_CTX = 512
DEFAULT_CHUNKS = 24
DEFAULT_THREADS = 6
DEFAULT_MAX_TEMP = 80.0
DEFAULT_RESUME_MARGIN = 10.0
DEFAULT_COOL = 55.0
DEFAULT_BUDGET_MINUTES = 120.0
DEFAULT_LLAMA_ARGS = ["-ngl", "99", "--cpu-moe"]
MODEL_INFO_VERBOSITY = ["-lv", "4"]   # llama.cpp prints n_expert_used only at this level; the check of the override reads that line
DEFAULT_TEXT_FILES = ("README.md", "docs/TUNING.md", "docs/PI.md")
OUT_DIR = ROOT / "run" / "kld_experts"
CHARS_PER_TOKEN = 3.5
LOGITS_VOCAB_ESTIMATE = 250_000
LOGITS_BYTES_PER_TOKEN_ENTRY = 2
DISK_HEADROOM = 1.2
MIN_SCORED_TOKENS = 100
POLL_SECONDS = 3.0
COOL_POLL_SECONDS = 5.0
MAX_PAUSE_SECONDS = 1200.0
KILL_WAIT_SECONDS = 20.0
LLAMA_CPP_DIR_ENV = "LLAMA_CPP_DIR"
LLAMA_PERPLEXITY_ENV = "LLAMA_PERPLEXITY"
CONVENTIONAL_LLAMA_CPP = ("clang", "github.com", "ggml-org", "llama.cpp")
NUMBER = r"(-?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|nan|inf))"
STATISTICS = {
    "mean_kld": (r"^Mean\s+KLD:\s*" + NUMBER + r"\s*±\s*" + NUMBER, ("mean_kld", "mean_kld_err")),
    "p99_kld": (r"^99\.0%\s+KLD:\s*" + NUMBER, ("p99_kld",)),
    "median_kld": (r"^Median\s+KLD:\s*" + NUMBER, ("median_kld",)),
    "max_kld": (r"^Maximum KLD:\s*" + NUMBER, ("max_kld",)),
    "same_top_p": (r"^Same top p:\s*" + NUMBER + r"\s*±\s*" + NUMBER, ("same_top_pct", "same_top_pct_err")),
    "rms_dp": (r"^RMS Δp\s*:\s*" + NUMBER + r"\s*±\s*" + NUMBER, ("rms_dp_pct", "rms_dp_pct_err")),
    "ppl_q": (r"^Mean PPL\(Q\)\s*:\s*" + NUMBER + r"\s*±\s*" + NUMBER, ("ppl", "ppl_err")),
    "ppl_base": (r"^Mean PPL\(base\)\s*:\s*" + NUMBER + r"\s*±\s*" + NUMBER, ("ppl_base", "ppl_base_err")),
    "ppl_ratio": (r"^Mean PPL\(Q\)/PPL\(base\)\s*:\s*" + NUMBER + r"\s*±\s*" + NUMBER, ("ppl_ratio", "ppl_ratio_err")),
}
EXPERTS_LOGGED = re.compile(r"n_expert_used\s*=\s*(\d+)")
CSV_FIELDS = ("k", "mean_kld", "mean_kld_err", "p99_kld", "median_kld", "max_kld", "same_top_pct", "same_top_pct_err", "ppl", "ppl_base",
              "ppl_ratio", "ppl_ratio_err", "rms_dp_pct", "seconds", "pauses", "peak_c", "status")


def parse_ks(text: str) -> list:
    """The expert counts to measure, from a comma list such as `8,7,6`; each is a positive integer, none repeats."""
    try:
        values = [int(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a comma-separated list of integers")
    if not values or any(value < 1 for value in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"{text!r}: give each count once, every one at least 1")
    return values


def parse_statistics(text: str) -> dict:
    """The numbers of llama-perplexity's `--kl-divergence` summary; keys it did not print are missing."""
    found = {}
    for line in text.splitlines():
        for pattern, names in STATISTICS.values():
            match = re.match(pattern, line, re.IGNORECASE)
            if match is not None:
                found.update({name: float(value) for name, value in zip(names, match.groups())})
    return found


def parse_experts_logged(text: str) -> int | None:
    """The experts per token llama.cpp says it loaded the model with, or None when the log has no such line."""
    match = EXPERTS_LOGGED.search(text)
    return int(match.group(1)) if match else None


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def base_logits_bytes(ctx: int, chunks: int) -> int:
    """About how large the reference logits file gets: llama.cpp keeps every vocabulary entry of the second half of each chunk."""
    return chunks * (ctx - 1 - ctx // 2) * LOGITS_VOCAB_ESTIMATE * LOGITS_BYTES_PER_TOKEN_ENTRY


def default_text(root: Path, names=DEFAULT_TEXT_FILES) -> str:
    """The evaluation text when none is given: the project's own prose, in a fixed order, so a checkout always gives the same text."""
    return "\n\n".join((root / name).read_text(encoding="utf-8") for name in names)


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def perplexity_command(binary: Path, model: Path, text: Path, base_logits: Path, ctx: int, chunks: int, threads: int, extra: list,
                       save_base: bool, experts_used: int | None = None) -> list:
    """One llama-perplexity run: it saves the reference logits (`save_base`) or compares the model with them."""
    command = [str(binary), "-m", str(model), "-f", str(text), "-c", str(ctx), "--chunks", str(chunks), "-t", str(threads), *extra,
               *MODEL_INFO_VERBOSITY, "--kl-divergence-base", str(base_logits)]
    if not save_base:
        command.append("--kl-divergence")
    if experts_used is not None:
        command += ["--override-kv", f"{EXPERTS_KEY}=int:{experts_used}"]
    return command


def find_llama_perplexity(explicit: Path | None = None, environ=os.environ, which=shutil.which, home: Path | None = None) -> Path | None:
    """The llama-perplexity binary: the given path, $LLAMA_PERPLEXITY, $PATH, then a build directory of a llama.cpp checkout."""
    if explicit is not None:
        return explicit
    if environ.get(LLAMA_PERPLEXITY_ENV):
        return Path(environ[LLAMA_PERPLEXITY_ENV])
    found = which("llama-perplexity")
    if found:
        return Path(found)
    for checkout in llama_cpp_checkouts(environ, home):
        for candidate in sorted(checkout.glob("build*/bin/llama-perplexity")):
            if candidate.is_file():
                return candidate
    return None


def llama_cpp_checkouts(environ=os.environ, home: Path | None = None) -> list:
    checkouts = [Path(environ[LLAMA_CPP_DIR_ENV])] if environ.get(LLAMA_CPP_DIR_ENV) else []
    return checkouts + [(home or Path.home()).joinpath(*CONVENTIONAL_LLAMA_CPP)]


def build_hint(environ=os.environ, home: Path | None = None) -> str:
    """The command that builds llama-perplexity; this tool never runs it."""
    existing = [path for path in llama_cpp_checkouts(environ, home) if path.is_dir()]
    source = existing[0] if existing else Path("<llama.cpp checkout>")
    return (f"cmake -S {source} -B {source}/build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON && "
            f"nice -n 19 cmake --build {source}/build --target llama-perplexity -j3")


def base_signature(ref_model: Path, text_sha256: str, ctx: int, chunks: int, llama_args: list, binary: Path) -> dict:
    status = ref_model.stat()
    program = binary.stat()
    return {"ref_model": str(ref_model.resolve()), "ref_size": status.st_size, "ref_mtime_ns": status.st_mtime_ns,
            "text_sha256": text_sha256, "ctx": ctx, "chunks": chunks, "llama_args": llama_args,
            "binary": str(binary.resolve()), "binary_size": program.st_size, "binary_mtime_ns": program.st_mtime_ns}


def sidecar_path(base_logits: Path) -> Path:
    return base_logits.with_name(base_logits.name + ".json")


def reusable_base(base_logits: Path, signature: dict) -> tuple:
    """(reusable, reason): an existing logits file is used again only when it was saved from this reference model, text, context and chunks."""
    if not base_logits.exists():
        return False, "no file yet"
    try:
        saved = json.loads(sidecar_path(base_logits).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, "it has no readable description next to it"
    differing = sorted(key for key in signature if saved.get(key) != signature[key])
    return (False, "it was saved with a different " + ", ".join(differing)) if differing else (True, "")


def pause_needed(temperature: float | None, max_temp: float) -> bool:
    return temperature is not None and temperature >= max_temp


class WatchedRun:
    def __init__(self, returncode: int | None, seconds: float, pauses: int, paused_seconds: float, peak_c: float | None, stopped: str | None):
        self.returncode, self.seconds, self.pauses, self.paused_seconds, self.peak_c, self.stopped = returncode, seconds, pauses, paused_seconds, peak_c, stopped


def run_watched(command: list, log_file: Path, max_temp: float, resume_temp: float, deadline: float | None, read_temperature,
                popen=subprocess.Popen, send_signal=os.killpg, sleep=time.sleep, clock=time.monotonic, env: dict | None = None,
                max_pause: float = MAX_PAUSE_SECONDS) -> WatchedRun:
    """Runs the command with its output in `log_file`. At `max_temp` degrees it is stopped (SIGSTOP) until the machine is at or below
    `resume_temp`, then continued; it is killed at the deadline or when the cooling takes more than `max_pause` seconds."""
    started = clock()
    with open(log_file, "wb") as log:
        process = popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, env=env)
    peak, pauses, paused_seconds, stopped = None, 0, 0.0, None

    def signal_group(number):
        try:
            send_signal(process.pid, number)
        except ProcessLookupError:
            pass

    try:
        while process.poll() is None and stopped is None:
            if deadline is not None and clock() >= deadline:
                stopped = "time budget"
                break
            temperature = read_temperature()
            if temperature is not None:
                peak = temperature if peak is None else max(peak, temperature)
            if not pause_needed(temperature, max_temp):
                sleep(POLL_SECONDS)
                continue
            signal_group(signal.SIGSTOP)
            pauses += 1
            paused_at = clock()
            while stopped is None:
                sleep(COOL_POLL_SECONDS)
                temperature = read_temperature()
                if temperature is None or temperature <= resume_temp:
                    break
                if clock() - paused_at >= max_pause:
                    stopped = "too hot: the machine did not cool down"
                elif deadline is not None and clock() >= deadline:
                    stopped = "time budget"
            paused_seconds += clock() - paused_at
            if stopped is None:
                signal_group(signal.SIGCONT)
    finally:
        if process.poll() is None:
            signal_group(signal.SIGCONT)
            signal_group(signal.SIGKILL)
            try:
                process.wait(timeout=KILL_WAIT_SECONDS)
            except subprocess.TimeoutExpired:
                pass
    return WatchedRun(process.poll(), clock() - started, pauses, paused_seconds, peak, stopped)


def measurement_row(k: int, output: str, run: WatchedRun) -> dict:
    """One table row from the text of a `--kl-divergence` run at `k` experts; `status` is "ok" or says what is wrong."""
    row = {"k": k, "seconds": round(run.seconds, 1), "pauses": run.pauses, "peak_c": run.peak_c, **parse_statistics(output)}
    logged = parse_experts_logged(output)
    row["experts_logged"] = logged
    if run.stopped:
        row["status"] = f"stopped: {run.stopped}"
    elif run.returncode != 0:
        row["status"] = f"llama-perplexity exited with {run.returncode}"
    elif "mean_kld" not in row:
        row["status"] = f"no KL divergence summary (it needs at least {MIN_SCORED_TOKENS} scored tokens: more text or chunks)"
    elif logged is not None and logged != k:
        row["status"] = f"llama.cpp loaded the model with {logged} experts per token, not {k}: the override was not applied"
    elif logged is None:
        row["status"] = "unconfirmed: llama.cpp did not print the experts per token, so the override is not proven"
    else:
        row["status"] = "ok"
    return row


def format_value(row: dict, name: str, error: str | None = None, digits: int = 5) -> str:
    if name not in row:
        return "-"
    text = f"{row[name]:.{digits}f}"
    return f"{text} ± {row[error]:.{digits}f}" if error and error in row else text


def format_table(rows: list) -> str:
    header = ("K", "mean KLD", "99% KLD", "same token %", "PPL ratio", "status")
    body = [(str(row["k"]), format_value(row, "mean_kld", "mean_kld_err"), format_value(row, "p99_kld"),
             format_value(row, "same_top_pct", "same_top_pct_err", 3), format_value(row, "ppl_ratio", "ppl_ratio_err"), row["status"])
            for row in rows]
    widths = [max(len(line[column]) for line in [header, *body]) for column in range(len(header))]
    lines = ["  ".join(cell.ljust(width) for cell, width in zip(line, widths)).rstrip() for line in [header, *body]]
    return "\n".join([lines[0], "  ".join("-" * width for width in widths), *lines[1:]])


def format_csv(rows: list) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def parse_args(argv: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure what fewer routed experts per token cost in quality: llama.cpp's KL divergence of the model at each expert count "
                    "against a reference model, on the same text.")
    parser.add_argument("--model", type=Path, required=True, help="the GGUF file the engine serves (the candidate)")
    parser.add_argument("--ref-model", type=Path, required=True, help="the reference GGUF file, e.g. the Q8_0 one: the best that fits this machine")
    parser.add_argument("--ks", type=parse_ks, default=list(DEFAULT_KS), metavar="K,K,...",
                        help="expert counts to measure, in this order (default 8,7,6,5,4; 8 is the model's own count and shows the cost of the file itself)")
    parser.add_argument("--text", type=Path, help="the evaluation text; default: README.md, docs/TUNING.md and docs/PI.md of this checkout")
    parser.add_argument("--ctx", type=int, default=DEFAULT_CTX, help=f"tokens per chunk (default {DEFAULT_CTX})")
    parser.add_argument("--chunks", type=int, default=DEFAULT_CHUNKS, help=f"chunks to score (default {DEFAULT_CHUNKS}); the second half of each is scored")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS, help=f"CPU threads of llama-perplexity (default {DEFAULT_THREADS})")
    parser.add_argument("--llama-args", type=shlex.split, default=list(DEFAULT_LLAMA_ARGS), metavar='"ARGS"',
                        help='more arguments for llama-perplexity, one quoted string (default "-ngl 99 --cpu-moe": the dense part on the GPU, the experts in RAM)')
    parser.add_argument("--llama-perplexity", type=Path, help="the llama-perplexity binary (default: $LLAMA_PERPLEXITY, $PATH, a build of a llama.cpp checkout)")
    parser.add_argument("--base-logits", type=Path, help="the reference logits file (default: next to --out); an existing file is used again when it "
                                                          "was saved from the same reference model, text, context and chunks")
    parser.add_argument("--cpu-max-mhz", type=float, default=0.0, metavar="MHZ",
                        help="hold the highest CPU frequency at this many MHz while llama-perplexity runs and put it back at exit (0: leave it alone)")
    parser.add_argument("--max-temp", type=float, default=DEFAULT_MAX_TEMP, help=f"stop llama-perplexity while the CPU package or the GPU is at or above this many degrees (default {DEFAULT_MAX_TEMP:g})")
    parser.add_argument("--resume-margin", type=float, default=DEFAULT_RESUME_MARGIN, help="continue when the temperature fell this many degrees below --max-temp")
    parser.add_argument("--cool", type=float, default=DEFAULT_COOL, help="start each run with the CPU package at or below this many degrees")
    parser.add_argument("--cool-timeout", type=float, default=600.0)
    parser.add_argument("--budget-minutes", type=float, default=DEFAULT_BUDGET_MINUTES, help="stop for good after this many minutes of run time")
    parser.add_argument("--max-load", type=float, default=1.5, help="refuse to start above this one-minute load average")
    parser.add_argument("--allow-turbo", action="store_true", help="measure although the CPU turbo boost is on")
    parser.add_argument("--force", action="store_true", help="start although the machine is busy, the temperature cannot be read or the disk is small")
    parser.add_argument("--out", type=Path, help="prefix of the result files: <out>.json and <out>.csv (default: run/kld_experts/<time>)")
    parser.add_argument("--dry-run", action="store_true", help="print the exact commands and start nothing")
    return parser.parse_args(argv)


def check_arguments(args: argparse.Namespace) -> list:
    problems = []
    if args.ctx < 8 or args.chunks < 1 or args.threads < 1:
        problems.append("--ctx must be at least 8, --chunks and --threads at least 1")
    if not (args.cpu_max_mhz == 0 or (math.isfinite(args.cpu_max_mhz) and MIN_MHZ <= args.cpu_max_mhz <= MAX_MHZ)):
        problems.append(f"--cpu-max-mhz must be 0 (off) or from {MIN_MHZ:g} to {MAX_MHZ:g} MHz")
    if not (math.isfinite(args.max_temp) and 40 <= args.max_temp <= 100) or args.resume_margin <= 0 or args.cool >= args.max_temp:
        problems.append("--max-temp must be from 40 to 100, --resume-margin above 0 and --cool below --max-temp")
    if args.budget_minutes <= 0:
        problems.append("--budget-minutes must be above 0")
    if args.text is not None and not args.dry_run and not args.text.is_file():
        problems.append(f"{args.text} does not exist")
    for label, path in (("--model", args.model), ("--ref-model", args.ref_model), ("--llama-perplexity", args.llama_perplexity)):
        if not args.dry_run and path is not None and not path.is_file():
            problems.append(f"{label} {path} does not exist")
    return problems


def temperature_now():
    return quant_bench.hottest(quant_bench.package_temperature(), quant_bench.gpu_temperature())


def result_document(args, binary, text_info, signature, base_logits, rows, aborted, started_at) -> dict:
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    return {"tool": "kld_experts", "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started_at)), "engine_commit": commit,
            "llama_perplexity": str(binary), "model": str(args.model), "ref_model": str(args.ref_model), "text": text_info, "ctx": args.ctx,
            "chunks": args.chunks, "threads": args.threads, "llama_args": args.llama_args, "cpu_max_mhz": args.cpu_max_mhz,
            "max_temp_c": args.max_temp, "ks": args.ks, "base_logits": str(base_logits), "base_signature": signature, "aborted": aborted,
            "rows": rows}


def main(argv: list | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    problems = check_arguments(args)
    if problems:
        print("kld-experts: " + "; ".join(problems), file=sys.stderr)
        return 2
    binary = find_llama_perplexity(args.llama_perplexity)
    if binary is None and not args.dry_run:
        print("kld-experts: llama-perplexity was not found. Build it (this tool does not) with:\n  " + build_hint(), file=sys.stderr)
        return 2
    binary = binary or Path("llama-perplexity")
    started_at = time.time()
    out = (args.out or OUT_DIR / time.strftime("%Y%m%d-%H%M%S", time.localtime(started_at))).expanduser()
    out_json, out_csv = out.with_name(out.name + ".json"), out.with_name(out.name + ".csv")
    work = out.parent
    if args.text:
        text_path = args.text.expanduser()
        text = text_path.read_text(encoding="utf-8") if text_path.is_file() else ""
    else:
        text_path = out.with_name(out.name + ".text.txt")
        text = default_text(ROOT)
    text_info = {"path": str(text_path), "sha256": text_digest(text) if text else None, "chars": len(text), "tokens_estimated": estimate_tokens(text),
                 "source": "--text" if args.text else "default: " + ", ".join(DEFAULT_TEXT_FILES)}
    base_logits = (args.base_logits or out.with_name(out.name + ".base.kld")).expanduser()
    save_command = perplexity_command(binary, args.ref_model, text_path, base_logits, args.ctx, args.chunks, args.threads, args.llama_args, True)
    measure_commands = {k: perplexity_command(binary, args.model, text_path, base_logits, args.ctx, args.chunks, args.threads, args.llama_args, False, k)
                        for k in args.ks}
    wanted_tokens = args.ctx * args.chunks
    print(f"kld-experts: text {text_info['source']}: {text_info['chars']} characters, about {text_info['tokens_estimated']} tokens "
          f"(an estimate at {CHARS_PER_TOKEN} characters per token; llama-perplexity prints the exact count); {args.chunks} chunks of "
          f"{args.ctx} need {wanted_tokens}", file=sys.stderr)
    if text_info["tokens_estimated"] < wanted_tokens:
        print("kld-experts: WARNING the text may be too short for that many chunks; llama-perplexity then scores fewer", file=sys.stderr)
    print(f"kld-experts: the reference logits file will be about {base_logits_bytes(args.ctx, args.chunks) / quant_bench.GIB:.1f} GiB at {base_logits}", file=sys.stderr)
    if args.dry_run:
        print("# reference logits (skipped when an existing file matches)\n" + shlex.join(save_command))
        for k, command in measure_commands.items():
            print(f"# {k} experts per token\n" + shlex.join(command))
        return 0

    work.mkdir(parents=True, exist_ok=True)
    if out_json.exists() or out_csv.exists():
        print(f"kld-experts: {out_json} or {out_csv} exists", file=sys.stderr)
        return 2
    if args.text is None:
        text_path.write_text(text, encoding="utf-8")
    problems = []
    if quant_bench.turbo_state() != "off" and not args.allow_turbo:
        problems.append("the CPU turbo boost is on or unknown (--allow-turbo overrides)")
    if not args.force:
        if os.getloadavg()[0] > args.max_load:
            problems.append(f"the load average is {os.getloadavg()[0]:.2f}")
        if quant_bench.package_temperature() is None:
            problems.append("the CPU package temperature cannot be read, so the temperature limit could not work")
        free = shutil.disk_usage(base_logits.parent if base_logits.parent.exists() else work).free
        if not base_logits.exists() and free < base_logits_bytes(args.ctx, args.chunks) * DISK_HEADROOM:
            problems.append(f"{free / quant_bench.GIB:.1f} GiB is free for the logits file")
    if problems:
        print("kld-experts: refusing to start: " + "; ".join(problems) + " (--force overrides the machine checks)", file=sys.stderr)
        return 2

    signature = base_signature(args.ref_model, text_info["sha256"], args.ctx, args.chunks, args.llama_args, binary)
    reuse, reason = reusable_base(base_logits, signature)
    if base_logits.exists() and not reuse:
        print(f"kld-experts: {base_logits} exists but cannot be used: {reason}; give another --base-logits or remove it", file=sys.stderr)
        return 2
    deadline = time.monotonic() + args.budget_minutes * 60
    env = {**os.environ, "LD_LIBRARY_PATH": os.pathsep.join(filter(None, [str(binary.resolve().parent), os.environ.get("LD_LIBRARY_PATH", "")]))}
    cap = CpuCap(args.cpu_max_mhz, log=lambda line: print(line, file=sys.stderr)) if args.cpu_max_mhz else None
    previous = quant_bench.install_stop_handlers()
    rows, aborted = [], None
    try:
        if cap is not None:
            cap.start()
        if reuse:
            print(f"kld-experts: using the reference logits in {base_logits}", file=sys.stderr)
        else:
            run = run_checked("reference logits", save_command, work / (out.name + ".reference.log"), args, deadline, env)
            if run.stopped or run.returncode != 0 or not base_logits.exists():
                base_logits.unlink(missing_ok=True)
                raise SystemExit(f"kld-experts: the reference run failed ({run.stopped or 'exit ' + str(run.returncode)}); see {work / (out.name + '.reference.log')}")
            sidecar_path(base_logits).write_text(json.dumps(signature, indent=1) + "\n", encoding="utf-8")
        for k in args.ks:
            if time.monotonic() >= deadline:
                aborted = "time budget"
                rows.append({"k": k, "status": "skipped: time budget"})
                continue
            log_file = work / f"{out.name}.k{k}.log"
            run = run_checked(f"{k} experts per token", measure_commands[k], log_file, args, deadline, env)
            row = measurement_row(k, log_file.read_text(encoding="utf-8", errors="replace"), run)
            rows.append(row)
            print(f"kld-experts: K={k}: {row['status']}" + (f", mean KLD {row['mean_kld']:.5f}" if "mean_kld" in row else ""), file=sys.stderr)
            if run.stopped:
                aborted = run.stopped
    except SystemExit as stop:
        aborted = str(stop.code)
    finally:
        for number in quant_bench.STOP_SIGNALS:
            signal.signal(number, signal.SIG_IGN)
        if cap is not None:
            cap.stop()
        quant_bench.restore_handlers(previous)
    document = result_document(args, binary, text_info, signature, base_logits, rows, aborted, started_at)
    out_json.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    out_csv.write_text(format_csv(rows), encoding="utf-8")
    print(format_table(rows))
    print(f"kld-experts: wrote {out_json} and {out_csv}; the reference logits stay in {base_logits} (delete it when done)", file=sys.stderr)
    failed = aborted is not None or any(row["status"] != "ok" for row in rows)
    if failed:
        print(f"kld-experts: INCOMPLETE{': ' + aborted if aborted else ''}", file=sys.stderr)
    return 3 if failed else 0


def run_checked(label: str, command: list, log_file: Path, args, deadline: float, env: dict) -> WatchedRun:
    """Waits for the machine to cool, then runs the command under the temperature watch."""
    temperature, waited, cool = quant_bench.wait_until_cool(args.cool, args.cool_timeout)
    print(f"kld-experts: {label}: start at {temperature if temperature is not None else 'unknown'} C after {waited:.0f} s of cooling"
          + ("" if cool is not False else " (still above the --cool limit)"), file=sys.stderr)
    return run_watched(command, log_file, args.max_temp, args.max_temp - args.resume_margin, deadline, temperature_now, env=env)


if __name__ == "__main__":
    sys.exit(main())
