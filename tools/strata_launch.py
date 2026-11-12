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
DENSE_Q5K_FILE = MODEL_DIR / "Qwen3.6-35B-A3B-UD-IQ4_XS-mtp-d5.gguf"
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


MAX_EXPERT_PREFETCH = 8   # the engine takes at most this many guessed experts per layer
PINNED_EXPERTS_IO = "pinned"   # the experts are copied once into page-locked memory, so the page cache is not used after the start
DEFAULT_EXPERT_PREFETCH = 3   # measured: P=4 does not pay, and P=3 gives about 10 percent per step (docs/TPS_REPORT.md)
PREFETCH_DEFAULT_PROFILES = ("auto",)   # the other profiles cap the memory below the 15 GiB of page-locked experts


def engine_args(profile: dict) -> list:
    args = ["--native", str(MODEL_FILE), "--max-context", str(profile["context"]), "--chunk", str(profile["chunk"]),
            "--mmq-group", str(profile["group"]), "--att-scores", str(profile["att_scores"]), "--kv", profile["kv"],
            "--experts", "pinned" if profile.get("expert_prefetch") else "mapped", "--stream-io", profile["io"], "--mtp",
            str(profile["mtp"]), "--pool-workers", str(profile["workers"])]
    if profile.get("expert_prefetch") is not None:
        args += ["--expert-prefetch", str(profile["expert_prefetch"])]
    if profile.get("vram_reserve") is not None:
        args += ["--vram-reserve", str(profile["vram_reserve"])]
    if profile.get("parallel", 0) > 1:
        args += ["--batch", str(profile["parallel"]), "--batch-mtp"]
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
DEFAULT_CPU_MAX_MHZ = 2100.0
RESIDENT_EXPERTS_CONTEXT = 16384   # tokens; the K/V cache takes 0.18 GiB instead of 0.73 GiB
RESIDENT_EXPERTS_RESERVE_MIB = 300
CAPPED_WORKERS = 6
CAPPED_ENV = {"STRATA_POOL_SPIN_US": "0"}


def resolve_cpu_cap(profile_name: str, requested: float | None) -> float:
    """The CPU frequency cap in MHz: what was asked for, else the measured default for the profile without a memory limit."""
    if requested is not None:
        return requested
    return DEFAULT_CPU_MAX_MHZ if profile_name == "auto" else 0.0


EXPERTS_USED_ENV = "STRATA_Q36_EXPERTS_USED"
MODEL_EXPERTS_USED = 8   # what the Qwen3.6-35B-A3B file routes per token; the engine refuses more than a file's own count
DEFAULT_EXPERTS_USED = 7   # mean KLD about 0.025 against 8 experts, a step about 10 percent shorter (docs/TPS_REPORT.md)


def default_model_file() -> Path:
    """The file served when --model is not given: the dense Q5_K variant of the IQ4_XS file (made with tools/requant_dense.py) when it
    exists, else the Q4_K_XL file."""
    return DENSE_Q5K_FILE if DENSE_Q5K_FILE.exists() else MODEL_FILE


def resolve_stack(experts_used: int | None, expert_prefetch: int | None, profile_name: str, model_size: int, available: int | None,
                  budget: int | None, overlap: bool, igpu_experts: bool) -> tuple:
    """The experts per token and the prefetch the server starts with, as (experts, prefetch or 0 for off, notes). Nothing asked means
    the measured stack: 7 experts, and a prefetch of 3 where it can run, which needs the memory for the page-locked experts both free
    (`available`) and inside the server's cgroup limit (`budget`, None for no limit). An explicit value wins."""
    experts = DEFAULT_EXPERTS_USED if experts_used is None else experts_used
    if expert_prefetch is not None:
        return experts, expert_prefetch, []
    if profile_name not in PREFETCH_DEFAULT_PROFILES:
        return experts, 0, []
    if overlap or igpu_experts:
        return experts, 0, ["expert prefetch is off: it does not combine with --overlap or --igpu-experts"]
    needed = model_size + PROTECT_MARGIN
    if available is None:
        return experts, 0, ["expert prefetch is off: the free memory could not be read"]
    if available < needed:
        free = f"{available / GIB:.1f} GiB of memory is available and the page-locked experts need about {needed / GIB:.1f} GiB"
        return experts, 0, [f"expert prefetch is off: {free}"]
    if budget is not None and budget < needed:
        limit = f"the memory budget of {budget / GIB:.1f} GiB is below the {needed / GIB:.1f} GiB the page-locked experts need"
        return experts, 0, [f"expert prefetch is off: {limit}"]
    return experts, DEFAULT_EXPERT_PREFETCH, []


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


IGPU_SHARE_ENV = "STRATA_IGPU_EXPERTS"


def igpu_share(text: str) -> float:
    """A share of the routed experts for the iGPU: 0 (off) or a fraction above 0 and below 1."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if value == 0 or (math.isfinite(value) and 0 < value < 1):
        return value
    raise argparse.ArgumentTypeError(f"the share must be 0 (off) or above 0 and below 1, not {text}")


def with_igpu(env: dict | None, share: float) -> dict | None:
    """The engine environment with the iGPU's share of the experts added; unchanged when the share is 0."""
    return {**(env or {}), IGPU_SHARE_ENV: f"{share:g}"} if share else env


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
    if profile.get("cpu_adaptive") and profile.get("cpu_max_mhz"):
        config["thermal"]["cpu_adaptive"] = True
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


PLATFORM_PROFILE_FILE = "/sys/firmware/acpi/platform_profile"
PLATFORM_PROFILES_FILE = "/sys/firmware/acpi/platform_profile_choices"


def apply_gpu_clocks(spec: str | None, dry_run: bool, run=subprocess.run) -> None:
    """Lock the GPU's graphics clocks with nvidia-smi (a range 'MIN,MAX' in MHz), or release the lock with 'reset'. Needs root; failures are warnings."""
    if not spec:
        return
    if spec != "reset" and not re.fullmatch(r"\d{3,4},\d{3,4}", spec):
        print(f"strata-launch: --gpu-clocks wants MIN,MAX in MHz or 'reset', not {spec!r}", file=sys.stderr)
        return
    command = ["sudo", "-A", "nvidia-smi", "-rgc"] if spec == "reset" else ["sudo", "-A", "nvidia-smi", "-lgc", spec]
    if dry_run:
        print(f"strata-launch: {' '.join(command[2:])} (not applied in a dry run)", file=sys.stderr)
        return
    env = {**os.environ, "SUDO_PROMPT": "lock or release the GPU clocks for the model server"}
    result = run(command, capture_output=True, text=True, check=False, env=env)
    if result.returncode != 0:
        print(f"strata-launch: could not set the GPU clocks: {(result.stderr or result.stdout).strip() or 'refused'}", file=sys.stderr)


def apply_platform_profile(name: str | None, dry_run: bool, run=subprocess.run) -> None:
    """Switch the firmware's cooling profile (balanced-performance and the like) once at start; the file belongs to root, so a
    failed direct write falls back to sudo -A. A refusal (the firmware rejects `performance` here) is a warning only. The old
    profile is not put back at exit: set it again with --platform-profile."""
    if not name:
        return
    try:
        choices = Path(PLATFORM_PROFILES_FILE).read_text(encoding="ascii").split()
        current = Path(PLATFORM_PROFILE_FILE).read_text(encoding="ascii").strip()
    except OSError as error:
        print(f"strata-launch: no platform profile on this machine ({error})", file=sys.stderr)
        return
    if name not in choices:
        print(f"strata-launch: platform profile {name!r} is not offered by the firmware (choices: {', '.join(choices)})", file=sys.stderr)
        return
    if name == current:
        return
    if dry_run:
        print(f"strata-launch: platform profile {current} -> {name} (not applied in a dry run)", file=sys.stderr)
        return
    try:
        Path(PLATFORM_PROFILE_FILE).write_text(name, encoding="ascii")
        return
    except OSError:
        pass
    env = {**os.environ, "SUDO_PROMPT": "switch the cooling profile of the laptop for the model server"}
    result = run(["sudo", "-A", "tee", PLATFORM_PROFILE_FILE], input=name, text=True, capture_output=True, check=False, env=env)
    if result.returncode != 0:
        print(f"strata-launch: could not switch the platform profile to {name}: {result.stderr.strip() or 'refused'}", file=sys.stderr)


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
        return None, "this setting does not use the page cache (O_DIRECT reads or page-locked experts)"
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
                messages.append("memory protection skipped: this setting does not use the page cache (O_DIRECT reads or page-locked experts)")
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
    parser.add_argument("--model", type=Path, help="the GGUF file to serve (default: the dense Q5_K variant of the IQ4_XS file when it exists, "
                                                   "else the Q4_K_XL file); its MTP block must be in it")
    parser.add_argument("--iq-kernels", choices=("auto", "on", "off"), default="auto",
                        help="the AVX-512 kernels for IQ3_S and IQ4_XS experts: auto (default) turns them on when the model has such "
                             "experts, on and off force it; the Q4_K_XL file is never affected")
    parser.add_argument("--overlap", nargs="?", const="all", type=overlap_list, metavar="LIST",
                        help="overlap GPU and CPU work inside a decode step (off by default): all, or a comma list of "
                             f"{', '.join(OVERLAP_SWITCHES)}; the output is the same as without it (docs/TUNING.md)")
    parser.add_argument("--igpu-experts", type=igpu_share, default=0.0, metavar="SHARE",
                        help="let the Vulkan iGPU compute this fraction of every layer's routed experts next to the CPU pool "
                             "(0, the default, is off); needs a build with -DSTRATA_IGPU_VULKAN=ON, see docs/TUNING.md")
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
    parser.add_argument("--resident-experts", action="store_true",
                        help=f"a {RESIDENT_EXPERTS_CONTEXT}-token window and a {RESIDENT_EXPERTS_RESERVE_MIB} MiB VRAM reserve, so that about 9 experts per layer stay in VRAM "
                             "(44.8 tok/s measured with --expert-prefetch 3 and 8 experts); an explicit --context or --vram-reserve wins")
    parser.add_argument("--warm-experts", type=int, default=0, metavar="KIB",
                        help="with --expert-prefetch, pull the first KIB KiB of every matrix of the guessed experts into the CPU cache while the GPU runs the mixer (0 = off)")
    parser.add_argument("--platform-profile", default=None, metavar="NAME",
                        help="switch the firmware's cooling profile at start (for example balanced-performance; the file belongs to root, so a password dialog may appear); not put back at exit")
    parser.add_argument("--draft-list", type=Path, default=None, metavar="FILE",
                        help="the MTP drafts pick among the token ids in this int32 file only (tools/draft_list.py makes it); the verify pass keeps the whole vocabulary")
    parser.add_argument("--chain-device", action="store_true",
                        help="chain the MTP drafts on the device without a trip to the host between them (off by default)")
    parser.add_argument("--ngram", type=str, default="", metavar="MATCH[,MAX]",
                        help="draft from an earlier copy of the last MATCH tokens of the context (for example 8,4) when it gives more drafts than the MTP block (off by default)")
    parser.add_argument("--spec-sampling", action="store_true",
                        help="requests with a temperature or penalties use the MTP drafts too, with the exact rejection rule (off by default)")
    parser.add_argument("--iq3s-4bit", action="store_true",
                        help="repack the IQ3_S gate/up matrices into 4-bit codes at start for the CPU kernel (same values, +2 GiB of RAM; mapped experts only, "
                             "so not with --expert-prefetch; off by default)")
    parser.add_argument("--strict", action="store_true",
                        help="bit-exact output: every expert on the CPU (no --expert-prefetch, no resident experts in VRAM), so the numbers are the CPU kernels' only")
    parser.add_argument("--attn-ahead", action="store_true",
                        help="the decode attention kernel loads 16 V cells together instead of 8 (same bits; off by default)")
    parser.add_argument("--prefetch-lead2", action="store_true",
                        help="with --expert-prefetch, guess the experts two layers ahead instead of one (three rings; off by default)")
    parser.add_argument("--head2", action="store_true",
                        help="with --draft-list: the verify pass takes the best of the listed tokens when the row norms prove no other token can beat it, else the whole matrix runs (off by default)")
    parser.add_argument("--kv-lazy", type=int, default=0, metavar="TIER_CELLS",
                        help="give the K/V cache physical memory as the context grows (CUDA virtual memory) and let the experts' slots use the rest, leaving room for TIER_CELLS cells; "
                             "slots (and a layer's guess slots with them) are given up layer by layer when the context outgrows the room; one session (off by default)")
    parser.add_argument("--lend-buffers", action="store_true",
                        help="the expert staging buffers (about 70 MiB) hold video memory only while a prompt is read, the rest of the time it is free for the K/V cache (off by default)")
    parser.add_argument("--parallel", type=int, default=1, metavar="N",
                        help="N sessions decode together: their rows share one window, so one pass over the experts serves all of them and the GPU's mixers "
                             "of one session fill the gaps of another (about 1.3 times the total tokens a second for two requests, nothing for one; more heat)")
    parser.add_argument("--gpu-clocks", default=None, metavar="MIN,MAX",
                        help="lock the GPU clocks (MHz) while the server runs so that the dense weights are read at full speed from the first step; needs root (sudo -A), "
                             "put back by --gpu-clocks reset; the card heats the shared cooler more")
    parser.add_argument("--mtp-adapt", action="store_true",
                        help="after a step whose first draft was rejected, verify one draft only in the next step (the text is unchanged; off by default)")
    parser.add_argument("--cpu-adaptive", action="store_true",
                        help="let a controller move the CPU frequency cap between 1800 and 2600 MHz so that the hottest sensor stays near 77 C "
                             "(the cap above is its starting value; off by default, the fixed cap stays), see the frequency cap in docs/TUNING.md")
    parser.add_argument("--experts-used", type=int, default=None, metavar="K",
                        help=f"route every token to K experts instead of the model's {MODEL_EXPERTS_USED} (1 to {MODEL_EXPERTS_USED}): fewer expert products per "
                             "token, so a faster and cooler decode at some cost in quality, which tools/kld_experts.py measures; "
                             f"default: {DEFAULT_EXPERTS_USED}, the measured stack (mean KLD about 0.025); {MODEL_EXPERTS_USED} gives the model's own count")
    parser.add_argument("--expert-prefetch", type=int, default=None, metavar="P",
                        help=f"copy P (1 to {MAX_EXPERT_PREFETCH}) guessed experts of the next layer to the GPU while the CPU computes the current layer, so the GPU "
                             "takes part of the expert work; keeps all the experts page-locked in RAM (--experts pinned, about 15 GiB), "
                             "see the expert prefetch section of docs/TUNING.md; 0 turns it off; "
                             f"default: {DEFAULT_EXPERT_PREFETCH} on the {'/'.join(PREFETCH_DEFAULT_PROFILES)} profile when the memory is free for the experts "
                             "and neither --overlap nor --igpu-experts is used, else off")
    parser.add_argument("--vram-reserve", type=int, default=None, metavar="MIB",
                        help="the VRAM the expert placement leaves free, in MiB (the engine's default is 768); a smaller value leaves room for "
                             "resident experts or guess slots at a long context, with less margin for other programs")
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

    if args.strict:   # every expert on the CPU with the CPU's bits: no prefetch and no experts resident in VRAM
        args.expert_prefetch = 0
        args.resident_experts = False
        args.warm_experts = 0
        if args.vram_reserve is not None:
            parser.error("--strict leaves the VRAM reserve alone (resident experts would round on the GPU)")
    if args.expert_prefetch is not None and not 0 <= args.expert_prefetch <= MAX_EXPERT_PREFETCH:
        parser.error(f"--expert-prefetch must be from 0 (off) to {MAX_EXPERT_PREFETCH}, not {args.expert_prefetch}")
    if args.vram_reserve is not None and args.vram_reserve < 0:
        parser.error(f"--vram-reserve must not be negative, not {args.vram_reserve}")
    if args.experts_used is not None and not 1 <= args.experts_used <= MODEL_EXPERTS_USED:
        parser.error(f"--experts-used must be from 1 to {MODEL_EXPERTS_USED}, not {args.experts_used}")

    global MODEL_FILE
    MODEL_FILE = default_model_file()
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
    if args.cpu_adaptive and cpu_max_mhz:
        profile["cpu_adaptive"] = True
        profile["workers"] = CAPPED_WORKERS
    if args.workers:
        profile["workers"] = args.workers
    if args.parallel > 1:
        profile["parallel"] = args.parallel
    if args.resident_experts:
        profile["context"] = args.context or RESIDENT_EXPERTS_CONTEXT
        profile["vram_reserve"] = RESIDENT_EXPERTS_RESERVE_MIB
    if args.vram_reserve is not None:
        profile["vram_reserve"] = args.vram_reserve
    for path in (ROOT / "build" / "strata-q36", MODEL_FILE):
        if not path.exists():
            sys.exit(f"strata-launch: {path} is missing (see README.md)")
    model_size = MODEL_FILE.stat().st_size
    try:
        available = memory_available()
    except OSError:
        available = None
    try:
        budget_bytes = None if args.no_limit or not profile["ram"] else parse_size(profile["ram"])
    except ValueError:
        budget_bytes = None
    experts_used, expert_prefetch, stack_notes = resolve_stack(args.experts_used, args.expert_prefetch, args.profile, model_size, available,
                                                               budget_bytes, bool(args.overlap), bool(args.igpu_experts))
    for note in stack_notes:
        print(f"strata-launch: {note}", file=sys.stderr)
    profile["expert_prefetch"] = expert_prefetch
    ensure_tokenizer()
    RUN_DIR.mkdir(exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    config_path = RUN_DIR / "qwen36.json"
    kernel_env = iq_kernel_env(args.iq_kernels, MODEL_FILE)
    lever_env = {**experts_used_env(experts_used), **(overlap_environment(args.overlap) or {}), **({"STRATA_Q36_MTP_ADAPT": "1"} if args.mtp_adapt else {}), **({"STRATA_Q36_CHAIN_DEV": "1"} if args.chain_device else {}), **({"STRATA_Q36_LEND": "1"} if args.lend_buffers else {}), **({"STRATA_Q36_KV_LAZY": "1", "STRATA_Q36_KV_TIER": str(args.kv_lazy)} if args.kv_lazy else {}), **({"STRATA_Q36_HEAD2": "1"} if args.head2 else {}), **({"STRATA_Q36_STRICT": "1"} if args.strict else {}), **({"STRATA_Q36_PF_LEAD": "2"} if args.prefetch_lead2 else {}), **({"STRATA_Q36_ATTN_AHEAD": "16"} if args.attn_ahead else {}), **({"STRATA_IQ3S_4BIT": "1"} if args.iq3s_4bit else {}), **({"STRATA_Q36_SPEC_SAMPLING": "1"} if args.spec_sampling else {}), **({"STRATA_Q36_NGRAM": str(args.ngram)} if args.ngram else {}), **({"STRATA_Q36_WARM": str(args.warm_experts)} if args.warm_experts else {}), **({"STRATA_Q36_DRAFT_LIST": str(args.draft_list)} if args.draft_list else {})}
    config = build_config(profile, args.port, args.sampling, args.max_temp,
                          with_igpu(engine_environment(kernel_env, cpu_max_mhz, lever_env), args.igpu_experts))
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
    memory_io = PINNED_EXPERTS_IO if expert_prefetch else profile["io"]
    memory_low, prewarm_bytes, scope_for_prewarm, messages = plan_memory(
        args.protect_ram, args.prewarm, memory_io, memory_budget, model_size, have_systemd,
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
          + (f", iGPU experts {args.igpu_experts:g}" if args.igpu_experts else "")
          + f", experts per token {experts_used}" + (f", expert prefetch {expert_prefetch}" if expert_prefetch else "")
          + (f", cpu cap {cpu_max_mhz:g} MHz" if cpu_max_mhz else "") + (f", memory protection {memory_low / GIB:.1f} GiB" if memory_low else "")
          + (f", prewarm {prewarm_bytes / GIB:.1f} GiB" if prewarm_bytes else ""), file=sys.stderr)
    apply_turbo(args.turbo, args.dry_run)
    apply_platform_profile(args.platform_profile, args.dry_run)
    apply_gpu_clocks(args.gpu_clocks, args.dry_run)
    apply_priority(args.priority, args.dry_run)
    if args.dry_run:
        print(" ".join(command))
        return 0
    os.execvp(command[0], command)


if __name__ == "__main__":
    sys.exit(main())
