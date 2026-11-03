import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(Path(__file__).resolve().parent))
from serve.cpucap import MAX_MHZ, MIN_MHZ  # noqa: E402
from serve.thermal import MAX_LIMIT_C, MIN_LIMIT_C  # noqa: E402
MODEL_DIR = ROOT / "models" / "qwen3.6-35b-a3b"
MODEL_FILE = MODEL_DIR / "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
MODEL_NAME = "qwen3.6-35b-a3b"
RUN_DIR = ROOT / "run"

# What each RAM budget buys: the budget caps the server's cgroup (page cache included), the rest are engine settings
# measured on an RTX 3050 Ti Laptop (4 GB) with 4 physical cores; see docs/TUNING.md.
# auto: no memory limit. The page cache grows into free memory and the kernel takes it back when another program needs it,
# so the model is as fast as the free memory allows (about 32 tok/s with 23 GB free) and slows gradually, not suddenly.
PROFILES = {
    "auto": {"ram": None, "context": 65536, "chunk": 512, "group": 16, "att_scores": 4, "workers": 3, "mtp": 2, "kv": "q8", "io": "buffered", "cpu_experts": 96},
    "lean": {"ram": "8G", "context": 65536, "chunk": 512, "group": 16, "att_scores": 4, "workers": 3, "mtp": 2, "kv": "q8", "io": "direct", "cpu_experts": 0},
    "balanced": {"ram": "12G", "context": 65536, "chunk": 512, "group": 16, "att_scores": 4, "workers": 3, "mtp": 2, "kv": "q8", "io": "direct", "cpu_experts": 0},
    "fast": {"ram": "16G", "context": 65536, "chunk": 512, "group": 16, "att_scores": 4, "workers": 3, "mtp": 2, "kv": "q8", "io": "direct", "cpu_experts": 0},
    "max": {"ram": "22G", "context": 65536, "chunk": 512, "group": 16, "att_scores": 4, "workers": 3, "mtp": 2, "kv": "q8", "io": "buffered", "cpu_experts": 96},
}


CHECKPOINTS = 6   # prompt-prefix states kept for new conversations that start like an earlier one


def turn_start_token() -> int:
    """The chat template's turn-start token, whose second to last occurrence in a prompt ends the shared prefix."""
    try:
        return int(json.loads((MODEL_DIR / "pack" / "tokenizer" / "vocab.json").read_text(encoding="utf-8"))["<|im_start|>"])
    except (OSError, KeyError, ValueError):
        return -1


def engine_args(profile: dict) -> list:
    args = ["--native", str(MODEL_FILE), "--max-context", str(profile["context"]), "--chunk", str(profile["chunk"]),
            "--mmq-group", str(profile["group"]), "--att-scores", str(profile["att_scores"]), "--kv", profile["kv"],
            "--experts", "mapped", "--stream-io", profile["io"], "--mtp", str(profile["mtp"]), "--pool-workers",
            str(profile["workers"])]
    if profile["cpu_experts"]:
        args += ["--cpu-experts", str(profile["cpu_experts"])]
    token = turn_start_token()
    if token >= 0:
        args += ["--ckpt", str(CHECKPOINTS), "--ckpt-token", str(token), "--ckpt-dir", str(RUN_DIR / "ckpt")]
    return args


# Sampling the server applies to requests that set none of their own.  The engine drafts ahead with the model's MTP block
# only for greedy requests (temperature 0, no penalties); the model card's sampling (temperature, presence penalty) leaves
# the drafts unused, and a request that sets its own temperature is not greedy either.
GREEDY_SAMPLING = {"temperature": 0, "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty": 1}


IQ_EXPERT_TYPES = ("IQ3_S", "IQ4_XS")
IQ_KERNELS_ENV = {"STRATA_IQ_VBMI": "1"}
IQ_KERNELS_OFF_ENV = {"STRATA_IQ_VBMI": "0"}

# The setting measured on the author's laptop (i7-11370H, turbo off, 80 C limit): the highest CPU frequency, the pool workers
# and the pool spin that held 30 tokens per second for minutes instead of 11 to 15; see "The frequency cap" in docs/TUNING.md.
DEFAULT_CPU_MAX_MHZ = 2200.0
CAPPED_WORKERS = 6
CAPPED_ENV = {"STRATA_POOL_SPIN_US": "0"}


def resolve_cpu_cap(profile_name: str, requested: float | None) -> float:
    """The CPU frequency cap in MHz: what was asked for, else the measured default for the profile without a memory limit."""
    if requested is not None:
        return requested
    return DEFAULT_CPU_MAX_MHZ if profile_name == "auto" else 0.0


EXPERTS_USED_ENV = "STRATA_Q36_EXPERTS_USED"
MODEL_EXPERTS_USED = 8   # what the Qwen3.6-35B-A3B file routes per token; the engine refuses more than a file's own count


def experts_used_env(experts_used: int | None) -> dict:
    """The engine environment that makes it route a token to fewer experts than the file says; empty keeps the file's count."""
    return {} if experts_used is None else {EXPERTS_USED_ENV: str(experts_used)}


def engine_environment(kernel_env: dict | None, cpu_max_mhz: float, extra: dict | None = None) -> dict | None:
    env = {**(kernel_env or {}), **(CAPPED_ENV if cpu_max_mhz else {}), **(extra or {})}
    return env or None


OVERLAP_SWITCHES = ("quant", "glue", "commit")


def overlap_list(text: str) -> str:
    """A --overlap value: all, or a comma list of the engine's overlap switches."""
    names = [name.strip() for name in text.split(",") if name.strip()]
    unknown = [name for name in names if name != "all" and name not in OVERLAP_SWITCHES]
    if not names or unknown:
        raise argparse.ArgumentTypeError(f"{text!r}: use all or a comma list of {', '.join(OVERLAP_SWITCHES)}")
    return ",".join(names)


def overlap_environment(switches: str | None) -> dict | None:
    """The engine environment of --overlap; None leaves STRATA_Q36_OVERLAP alone."""
    return {"STRATA_Q36_OVERLAP": switches} if switches else None


def uses_iq_experts(model: Path) -> bool:
    """Whether the expert tensors of this GGUF file are stored as IQ3_S or IQ4_XS, the formats the AVX-512 kernels serve."""
    import gguf_reader
    return any(tensor.type_name in IQ_EXPERT_TYPES for tensor in gguf_reader.GGUFFile(model).tensors if "_exps" in tensor.name)


def iq_kernel_env(mode: str, model: Path) -> dict | None:
    """The engine environment that switches the IQ kernels on or off; off is set explicitly so that a value inherited from the caller's
    environment cannot turn them on, and None leaves the environment alone (auto with a model that has no IQ experts)."""
    if mode == "off":
        return IQ_KERNELS_OFF_ENV
    if mode == "on" or uses_iq_experts(model):
        return IQ_KERNELS_ENV
    return None


def iq_kernel_state(env: dict | None) -> str:
    if env == IQ_KERNELS_ENV:
        return "on"
    return "off" if env == IQ_KERNELS_OFF_ENV else "not needed"


def build_config(profile: dict, port: int, sampling: str, max_temp: float = 0, env: dict | None = None) -> dict:
    config = {"exe": "build/strata-q36", "args": engine_args(profile), "cwd": str(ROOT),
              "tokenizer": str(MODEL_DIR / "pack" / "tokenizer"), "model_name": MODEL_NAME, "arch": "qwen35moe",
              "log": str(ROOT / "logs" / "strata-qwen36.log"), "host": "127.0.0.1", "port": port}
    if sampling == "greedy":
        config["sampling"] = GREEDY_SAMPLING
    if max_temp > 0:
        config["thermal"] = {"max_temp_c": max_temp}
    if profile.get("cpu_max_mhz"):
        config.setdefault("thermal", {})["cpu_max_mhz"] = profile["cpu_max_mhz"]
    if env:
        config["env"] = dict(env)
    return config


def apply_turbo(mode: str, dry_run: bool) -> None:
    """Switch the CPU turbo boost the way the launcher was asked to; "keep" leaves it alone. A failure is a warning only."""
    if mode == "keep":
        return
    if dry_run:
        print(f"strata-launch: turbo {mode} (not applied in a dry run)", file=sys.stderr)
        return
    result = subprocess.run([str(ROOT / "tools" / "turbo"), mode, "--no-sudo", "--quiet"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"strata-launch: could not switch turbo {mode}: {result.stderr.strip() or 'see tools/turbo status'}", file=sys.stderr)


# The lowest weight and the idle I/O weight: when another program wants the CPU or the disk, this one gets about a percent
BACKGROUND_PROPERTIES = ["CPUWeight=idle", "IOWeight=1"]


def apply_priority(mode: str, dry_run: bool, set_scheduler=os.sched_setscheduler, run=subprocess.run) -> None:
    """Make this process, and everything it starts, yield the CPU and the disk; "normal" leaves them alone.

    The scheduling class (SCHED_IDLE) and the idle I/O class are inherited by the server and the engine. They only order the
    tasks inside one control group, so the cgroup weight of `wrap_in_scope` does the work between this server and the
    browser. A failure is a warning only.
    """
    if mode == "normal":
        return
    if dry_run:
        print("strata-launch: background priority (not applied in a dry run)", file=sys.stderr)
        return
    try:
        set_scheduler(0, os.SCHED_IDLE, os.sched_param(0))
    except OSError as error:
        print(f"strata-launch: could not set the idle scheduling class: {error}", file=sys.stderr)
    try:
        result = run(["ionice", "-c3", "-p", str(os.getpid())], capture_output=True, text=True, check=False)
    except OSError as error:
        print(f"strata-launch: could not set the idle I/O class: {error}", file=sys.stderr)
        return
    if result.returncode != 0:
        print(f"strata-launch: could not set the idle I/O class: {result.stderr.strip() or 'ionice failed'}", file=sys.stderr)


# prints the CPU weight of the control group it runs in
WEIGHT_PROBE = "g = open('/proc/self/cgroup').read().split(':', 2)[2].strip(); print(open('/sys/fs/cgroup' + g + '/cpu.weight').read().strip())"
PROBE_TIMEOUT_SECONDS = 10


def background_scope_works() -> bool:
    """Whether the background weights really apply here, not only whether systemd-run accepts them: a scope can start with its
    weights ignored when the controllers are not delegated to the user manager."""
    probe = ["systemd-run", "--user", "--scope", "--quiet"]
    for prop in BACKGROUND_PROPERTIES:
        probe += ["-p", prop]
    try:
        result = subprocess.run(probe + [sys.executable, "-c", WEIGHT_PROBE], capture_output=True, text=True, timeout=PROBE_TIMEOUT_SECONDS, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and result.stdout.strip() == "1"


def scope_properties(memory_budget: str | None, background: bool, memory_low: int | None = None) -> list:
    properties = [f"MemoryMax={memory_budget}", "MemorySwapMax=0"] if memory_budget else []
    if memory_low:
        properties.append(f"MemoryLow={memory_low}")
    return properties + (BACKGROUND_PROPERTIES if background else [])


GIB = 1 << 30
SIZE_UNITS = {"": 1, "K": 1 << 10, "M": 1 << 20, "G": GIB, "T": 1 << 40}
PROTECT_MARGIN = 2 * GIB    # what the server needs next to the cached model: KV cache, buffers, the Python server
CGROUP_ROOT = Path("/sys/fs/cgroup")


def parse_size(text: str) -> int:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGT]?)", text.strip().upper())
    if match is None:
        raise ValueError(f"cannot read the size {text!r}")
    return int(float(match.group(1)) * SIZE_UNITS[match.group(2)])


def protect_bytes(io: str, budget: str | None, model_size: int) -> int | None:
    """What to ask the kernel to keep in memory for the server: the cached model and the engine around it, within the budget.
    None when the profile reads the model with O_DIRECT, which never fills the page cache."""
    if io != "buffered":
        return None
    wanted = model_size + PROTECT_MARGIN
    return min(wanted, parse_size(budget)) if budget else wanted


def memory_available() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def prewarm_plan(io: str, budget: str | None, model_size: int, available: int) -> tuple[int | None, str]:
    """How many bytes of the model to read into the page cache before the server starts, or None and the reason not to."""
    if io != "buffered":
        return None, "this profile reads the model with O_DIRECT, so the page cache would not be used"
    if budget and model_size > parse_size(budget):
        return None, f"the model does not fit the {budget} budget"
    needed = model_size + PROTECT_MARGIN
    if available < needed:
        return None, f"{available / GIB:.1f} GiB of memory is available and the model needs {needed / GIB:.1f} GiB, so reading it would push other programs out"
    return model_size, ""


def scope_ancestors(uid: int) -> list:
    """The control groups above a new user scope: app.slice of the user manager and everything over it."""
    manager = f"user.slice/user-{uid}.slice/user@{uid}.service"
    return ["user.slice", f"user.slice/user-{uid}.slice", manager, f"{manager}/app.slice"]


def protection_gaps(needed: int, uid: int | None = None, root: Path = CGROUP_ROOT) -> list:
    """The levels above the server scope whose memory.low is below `needed`, as (path, value or None when unreadable).
    The kernel honours the MemoryLow of the scope only up to the lowest value above it, and shares that value in proportion
    among every scope that asks, so this is a necessary condition, not a guarantee."""
    gaps = []
    for path in scope_ancestors(os.getuid() if uid is None else uid):
        try:
            text = (root / path / "memory.low").read_text(encoding="ascii").strip()
        except OSError:
            gaps.append((path, None))
            continue
        if text != "max" and int(text) < needed:
            gaps.append((path, int(text)))
    return gaps


def wrap_in_scope(server: list, properties: list, always: bool = False) -> list:
    """The server command inside its own systemd scope, which carries the memory limit and the background weights.
    Without properties the command is left alone unless `always` asks for a scope anyway."""
    if not properties and not always:
        return server
    command = ["systemd-run", "--user", "--scope", "--quiet"]
    for prop in properties:
        command += ["-p", prop]
    # systemd-run reads a $ in its arguments as the start of a variable and $$ as one literal dollar
    return command + ["--"] + [argument.replace("$", "$$") for argument in server]


def user_scope_works() -> bool:
    """Whether systemd-run can start a user scope here: the program can exist while no user manager is running."""
    try:
        result = subprocess.run(["systemd-run", "--user", "--scope", "--quiet", "true"], capture_output=True, timeout=PROBE_TIMEOUT_SECONDS, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def scope_available(have_systemd: bool, base_properties: list, dry_run: bool, probe=user_scope_works) -> tuple[bool, str]:
    """Whether memory protection and the prewarm can count on a scope of their own, and a warning when they cannot.
    The properties that exist anyway already depend on a scope, so only a start that would run without one is probed."""
    if not have_systemd:
        return False, "systemd-run is not available, so memory protection is off and prewarm runs in the current control group"
    if base_properties or dry_run or probe():
        return True, ""
    return False, "no user systemd manager is reachable, so memory protection is off and prewarm runs in the current control group"


def plan_memory(protect: bool | None, prewarm: bool | None, io: str, budget: str | None, model_size: int, have_systemd: bool,
                base_properties: list, dry_run: bool, available=memory_available, gaps=protection_gaps, probe=user_scope_works) -> tuple:
    """What to do about keeping the model in memory: (MemoryLow bytes or None, bytes to prewarm or None, whether the prewarm gets a
    scope of its own, messages). None means on unless switched off: it stays silent when it does not apply, and True says why."""
    messages = []
    memory_low = None
    if protect is not False:
        try:
            memory_low = protect_bytes(io, budget, model_size)
        except ValueError as error:
            messages.append(f"memory protection skipped: {error}")
        if memory_low is None and io != "buffered":
            if protect:
                messages.append("memory protection skipped: this profile reads the model with O_DIRECT, so the page cache is not used")
        elif memory_low is not None and not have_systemd:
            if protect:
                messages.append("memory protection skipped: it needs systemd-run")
            memory_low = None
        elif memory_low is not None:
            for path, value in gaps(memory_low):
                allowed = "no memory.low" if value is None else f"{value / GIB:.1f} GiB"
                messages.append(f"memory protection is only partial: /{path} allows {allowed}, the server asks for {memory_low / GIB:.1f} GiB "
                                f"(see docs/TUNING.md, Keeping the model in memory)")
    prewarm_bytes = None
    if prewarm is not False:
        try:
            prewarm_bytes, reason = prewarm_plan(io, budget, model_size, available())
        except (ValueError, OSError) as error:
            prewarm_bytes, reason = None, str(error)
        if prewarm_bytes is None and (prewarm or io == "buffered"):
            messages.append(f"prewarm skipped: {reason}")
    scope_for_prewarm = False
    if memory_low or prewarm_bytes:
        usable, warning = scope_available(have_systemd, base_properties, dry_run, probe)
        if warning:
            messages.append(warning)
        if not usable:
            memory_low = None
        scope_for_prewarm = usable and bool(prewarm_bytes)
    return memory_low, prewarm_bytes, scope_for_prewarm, messages


# Abbreviations that stay accepted although --protect-ram, --prewarm and their --no- forms share these prefixes.
OLD_ABBREVIATIONS = {"--pro": "--profile", "--n": "--no-limit", "--no": "--no-limit", "--no-": "--no-limit"}


def restore_old_abbreviations(arguments: list) -> list:
    restored = []
    for argument in arguments:
        name, equals, value = argument.partition("=")
        restored.append(OLD_ABBREVIATIONS.get(name, name) + equals + value)
    return restored


def ensure_tokenizer() -> None:
    if (MODEL_DIR / "pack" / "tokenizer" / "vocab.json").exists():
        return
    subprocess.run([sys.executable, str(ROOT / "tools" / "strata_tokenizer.py"), "--gguf", str(MODEL_FILE), "--out",
                    str(MODEL_DIR / "pack")], check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start the Qwen3.6-35B-A3B server inside a memory budget.")
    parser.add_argument("--profile", choices=list(PROFILES), default="auto")
    parser.add_argument("--model", type=Path, help="the GGUF file to serve (default: the Q4_K_XL file); its MTP block must be in it")
    parser.add_argument("--iq-kernels", choices=("auto", "on", "off"), default="auto",
                        help="the AVX-512 kernels for IQ3_S and IQ4_XS experts: auto (default) turns them on when the model has such "
                             "experts, on and off force it; the Q4_K_XL file is never affected")
    parser.add_argument("--overlap", nargs="?", const="all", type=overlap_list, metavar="LIST",
                        help="overlap GPU and CPU work inside a decode step (off by default): all, or a comma list of "
                             f"{', '.join(OVERLAP_SWITCHES)}; the output is the same as without it (docs/TUNING.md)")
    parser.add_argument("--ram", help="memory budget of the server, e.g. 9G (page cache included); default: the profile's")
    parser.add_argument("--context", type=int, help="context window in tokens")
    parser.add_argument("--chunk", type=int, help="prompt chunk in tokens (more is faster, and takes VRAM)")
    parser.add_argument("--sampling", choices=("greedy", "card"), default="greedy",
                        help="greedy: MTP drafts speed decoding up (default); card: the model card's sampling, no drafts")
    parser.add_argument("--mtp", type=int, help="MTP drafts per speculative step (0 turns drafting off)")
    parser.add_argument("--workers", type=int, help="CPU expert worker threads (the main thread computes too)")
    parser.add_argument("--max-temp", type=float, default=80.0, metavar="C",
                        help="pause the engine for part of the time whenever the CPU package or the GPU reaches this many "
                             "degrees Celsius (default 80, 0 turns the limit off)")
    parser.add_argument("--cpu-max-mhz", type=float, default=None, metavar="MHZ",
                        help="hold the highest CPU frequency at this many MHz while the server runs and put it back at exit "
                             f"(default {DEFAULT_CPU_MAX_MHZ:g} with the auto profile, with {CAPPED_WORKERS} workers and no pool spin; 0 turns the "
                             "cap off, and so do the other profiles); a lower cap lets the model run without pauses, see the frequency cap in docs/TUNING.md")
    parser.add_argument("--experts-used", type=int, default=None, metavar="K",
                        help=f"route every token to K experts instead of the model's {MODEL_EXPERTS_USED} (1 to {MODEL_EXPERTS_USED}): fewer expert products per "
                             "token, so a faster and cooler decode at some cost in quality, which tools/kld_experts.py measures; "
                             "default: the model's own count")
    parser.add_argument("--turbo", choices=("off", "on", "keep"), default="off",
                        help="CPU turbo boost at start: off (default), on, or keep what it is (tools/turbo does the same on its own)")
    parser.add_argument("--priority", choices=("background", "normal"), default="background",
                        help="background (default): the server yields the CPU and the disk to everything else, so the machine "
                             "stays responsive while it works; normal: no change")
    parser.add_argument("--protect-ram", action=argparse.BooleanOptionalAction, default=None,
                        help="ask the kernel to keep the cached model in memory (MemoryLow on the server scope); soft, and only "
                             "fully honoured when the levels above allow it, see docs/TUNING.md; on by default, "
                             "--no-protect-ram turns it off")
    parser.add_argument("--prewarm", action=argparse.BooleanOptionalAction, default=None,
                        help="read the model file into the page cache before the server starts; skipped when the memory is not "
                             "free for it; on by default, --no-prewarm turns it off")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-limit", action="store_true", help="do not run the server in a memory-limited cgroup")
    parser.add_argument("--dry-run", action="store_true", help="write the config and print the command, start nothing")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args(restore_old_abbreviations(sys.argv[1:]))
    if not (args.max_temp == 0 or (math.isfinite(args.max_temp) and MIN_LIMIT_C <= args.max_temp <= MAX_LIMIT_C)):
        parser.error(f"--max-temp must be 0 (off) or from {MIN_LIMIT_C:g} to {MAX_LIMIT_C:g} degrees Celsius, not {args.max_temp:g}")

    cpu_max_mhz = resolve_cpu_cap(args.profile, args.cpu_max_mhz)
    if not (cpu_max_mhz == 0 or (math.isfinite(cpu_max_mhz) and MIN_MHZ <= cpu_max_mhz <= MAX_MHZ)):
        parser.error(f"--cpu-max-mhz must be 0 (off) or from {MIN_MHZ:g} to {MAX_MHZ:g} MHz, not {cpu_max_mhz:g}")

    if args.experts_used is not None and not 1 <= args.experts_used <= MODEL_EXPERTS_USED:
        parser.error(f"--experts-used must be from 1 to {MODEL_EXPERTS_USED}, not {args.experts_used}")

    global MODEL_FILE
    if args.model:
        MODEL_FILE = args.model.expanduser().resolve()
        if MODEL_DIR.resolve() not in MODEL_FILE.parents:
            parser.error(f"--model must be a file inside {MODEL_DIR}: the tokenizer there belongs to this model")
    profile = dict(PROFILES[args.profile])
    if args.ram:
        profile["ram"] = args.ram
    if args.context:
        profile["context"] = args.context
    if args.chunk:
        profile["chunk"] = args.chunk
    if args.mtp is not None:
        profile["mtp"] = args.mtp
    if cpu_max_mhz:
        profile["cpu_max_mhz"] = cpu_max_mhz
        profile["workers"] = CAPPED_WORKERS
    if args.workers:
        profile["workers"] = args.workers
    for path in (ROOT / "build" / "strata-q36", MODEL_FILE):
        if not path.exists():
            sys.exit(f"strata-launch: {path} is missing (see README.md)")
    ensure_tokenizer()
    RUN_DIR.mkdir(exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    config_path = RUN_DIR / "qwen36.json"
    kernel_env = iq_kernel_env(args.iq_kernels, MODEL_FILE)
    lever_env = {**experts_used_env(args.experts_used), **(overlap_environment(args.overlap) or {})}
    config = build_config(profile, args.port, args.sampling, args.max_temp, engine_environment(kernel_env, cpu_max_mhz, lever_env))
    config_path.write_text(json.dumps(config, indent=1) + "\n", encoding="utf-8")

    server = [str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config",
              str(config_path), "--port", str(args.port)]
    memory_budget = None if args.no_limit else profile["ram"]
    background = args.priority == "background"
    have_systemd = shutil.which("systemd-run") is not None
    if memory_budget and not have_systemd:
        print("strata-launch: systemd-run is not available, the memory budget is not enforced", file=sys.stderr)
        memory_budget = None
    if background and not (have_systemd and (args.dry_run or background_scope_works())):
        print("strata-launch: this machine does not give the server its own CPU weight, so only the idle scheduling class is set", file=sys.stderr)
        background = False
    model_size = MODEL_FILE.stat().st_size
    memory_low, prewarm_bytes, scope_for_prewarm, messages = plan_memory(
        args.protect_ram, args.prewarm, profile["io"], memory_budget, model_size, have_systemd,
        scope_properties(memory_budget, background), args.dry_run)
    for message in messages:
        print(f"strata-launch: {message}", file=sys.stderr)
    if prewarm_bytes:
        server = [server[0], str(ROOT / "tools" / "strata_prewarm.py"), "--limit", str(prewarm_bytes), str(MODEL_FILE), "--"] + server
    command = wrap_in_scope(server, scope_properties(memory_budget, background, memory_low), always=scope_for_prewarm)
    print(f"strata-launch: profile {args.profile}, memory budget {profile['ram'] or 'none (elastic page cache)'}, context {profile['context']}, "
          f"port {args.port}, temperature limit {f'{args.max_temp:g} C' if args.max_temp > 0 else 'none'}, turbo {args.turbo}, "
          f"priority {args.priority}, model {MODEL_FILE.name}, iq kernels {iq_kernel_state(kernel_env)}"
          + (f", overlap {args.overlap}" if args.overlap else "")
          + (f", experts per token {args.experts_used}" if args.experts_used is not None else "")
          + (f", cpu cap {cpu_max_mhz:g} MHz" if cpu_max_mhz else "") + (f", memory protection {memory_low / GIB:.1f} GiB" if memory_low else "")
          + (f", prewarm {prewarm_bytes / GIB:.1f} GiB" if prewarm_bytes else ""), file=sys.stderr)
    apply_turbo(args.turbo, args.dry_run)
    apply_priority(args.priority, args.dry_run)
    if args.dry_run:
        print(" ".join(command))
        return 0
    os.execvp(command[0], command)


if __name__ == "__main__":
    sys.exit(main())
