import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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


def build_config(profile: dict, port: int, sampling: str) -> dict:
    config = {"exe": "build/strata-q36", "args": engine_args(profile), "cwd": str(ROOT),
              "tokenizer": str(MODEL_DIR / "pack" / "tokenizer"), "model_name": MODEL_NAME, "arch": "qwen35moe",
              "log": str(ROOT / "logs" / "strata-qwen36.log"), "host": "127.0.0.1", "port": port}
    if sampling == "greedy":
        config["sampling"] = GREEDY_SAMPLING
    return config


def ensure_tokenizer() -> None:
    if (MODEL_DIR / "pack" / "tokenizer" / "vocab.json").exists():
        return
    subprocess.run([sys.executable, str(ROOT / "tools" / "strata_tokenizer.py"), "--gguf", str(MODEL_FILE), "--out",
                    str(MODEL_DIR / "pack")], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the Qwen3.6-35B-A3B server inside a memory budget.")
    parser.add_argument("--profile", choices=list(PROFILES), default="auto")
    parser.add_argument("--ram", help="memory budget of the server, e.g. 9G (page cache included); default: the profile's")
    parser.add_argument("--context", type=int, help="context window in tokens")
    parser.add_argument("--chunk", type=int, help="prompt chunk in tokens (more is faster, and takes VRAM)")
    parser.add_argument("--sampling", choices=("greedy", "card"), default="greedy",
                        help="greedy: MTP drafts speed decoding up (default); card: the model card's sampling, no drafts")
    parser.add_argument("--mtp", type=int, help="MTP drafts per speculative step (0 turns drafting off)")
    parser.add_argument("--workers", type=int, help="CPU expert worker threads (the main thread computes too)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-limit", action="store_true", help="do not run the server in a memory-limited cgroup")
    parser.add_argument("--dry-run", action="store_true", help="write the config and print the command, start nothing")
    args = parser.parse_args()

    profile = dict(PROFILES[args.profile])
    if args.ram:
        profile["ram"] = args.ram
    if args.context:
        profile["context"] = args.context
    if args.chunk:
        profile["chunk"] = args.chunk
    if args.mtp is not None:
        profile["mtp"] = args.mtp
    if args.workers:
        profile["workers"] = args.workers
    for path in (ROOT / "build" / "strata-q36", MODEL_FILE):
        if not path.exists():
            sys.exit(f"strata-launch: {path} is missing (see README.md)")
    ensure_tokenizer()
    RUN_DIR.mkdir(exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    config_path = RUN_DIR / "qwen36.json"
    config_path.write_text(json.dumps(build_config(profile, args.port, args.sampling), indent=1) + "\n", encoding="utf-8")

    server = [str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config",
              str(config_path), "--port", str(args.port)]
    command = server
    if profile["ram"] is None:
        args.no_limit = True
    if not args.no_limit and shutil.which("systemd-run"):
        command = ["systemd-run", "--user", "--scope", "--quiet", "-p", f"MemoryMax={profile['ram']}", "-p",
                   "MemorySwapMax=0", "--"] + server
    elif not args.no_limit:
        print("strata-launch: systemd-run is not available, the memory budget is not enforced", file=sys.stderr)
    print(f"strata-launch: profile {args.profile}, memory budget {profile['ram'] or 'none (elastic page cache)'}, context {profile['context']}, "
          f"port {args.port}", file=sys.stderr)
    if args.dry_run:
        print(" ".join(command))
        return 0
    os.execvp(command[0], command)


if __name__ == "__main__":
    sys.exit(main())
