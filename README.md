# Strata, Qwen3.6-35B-A3B edition

A trimmed copy of [Strata](https://github.com/Niko1221/Strata) that does one thing: serve **Qwen3.6-35B-A3B** (UD-Q4_K_XL,
22.9 GB) on a PC with a small GPU and a modest RAM budget, through an OpenAI- and Anthropic-compatible HTTP API, so that a
console agent such as [pi](https://www.npmjs.com/package/@earendil-works/pi-coding-agent) can work with it locally.

The upstream repository ships many models, a desktop control panel, Windows and macOS builds and a lot of tooling. Here
only the Qwen3.6 path is kept: the C++/CUDA engine (`strata-q36`), the HTTP server with its metrics collectors, a launcher
and a pi profile.

## What it needs

- Linux x86-64 with AVX2 (AVX-512 is used when present), an NVIDIA GPU with 4 GB of VRAM or more
- CUDA toolkit 12.x or 13.x, CMake 3.24+, Ninja, a C++20 compiler (clang or gcc)
- Python 3.12+ for the server, in a virtual environment (tested with 3.12 and 3.14)
- 23 GB of disk for the model and 8 GB of free RAM or more, about 23 GB for full speed (see [docs/TUNING.md](docs/TUNING.md))

## Quick start

```sh
tools/setup_venv.sh        # .venv from requirements.lock (hash-pinned)
tools/build.sh             # build/strata-q36 and build/libstrata_kq.so, for the installed GPU
tools/fetch_model.sh       # the pinned GGUF into models/qwen3.6-35b-a3b, size and SHA-256 checked
bin/strata-serve           # the server on http://127.0.0.1:8080, no memory limit: the page cache grows as far as free memory allows
tools/spi             # pi on that server inside a sandbox: only the current folder is writable (setup: docs/PI.md)
```

The default profile `auto` sets no memory limit: the kernel keeps the model's experts in the page cache while memory is
free and takes them back when another program asks, so the speed (about 37 tokens per second with 23 GB free) falls
gradually instead of the program failing. `bin/strata-serve --profile lean|balanced|fast|max` caps the server's memory
(8, 12, 16 or 22 GB) when a hard budget is needed; `--ram 14G` sets one by hand. A hard cap costs speed, because the
experts that are not in the cache come from the SSD: roughly 10, 17, 25 and 33 tokens per second on the author's laptop,
measured before the AVX-512 kernels (see [docs/TUNING.md](docs/TUNING.md)).
`--context` and `--chunk` change the context window and the prompt chunk, the two things that compete for the 4 GB of
VRAM. Everything the launcher does is visible with `--dry-run`.

## How it fits in a small machine

The model is a mixture of experts: 256 experts per layer, 8 used per token. The attention, the recurrent layers and the
shared expert (2 GB) sit in VRAM. The routed experts (18.7 GB) are **not** copied anywhere: the CPU computes them straight
from the memory-mapped GGUF, so the host holds only the page cache, which the kernel trims under the memory budget. Long
prompts are processed in chunks, and each chunk streams the experts once through O_DIRECT reads into the GPU, where the
prompt kernels run (int8 matrix multiply). Decoding uses the model's own MTP block for speculative drafting when the
request is greedy.

The K/V cache is stored as 8-bit blocks (ggml's q8_0 layout, 53% of FP16), which is what makes a 64K-token window fit in
the VRAM that is left.
The default window is 65,536 tokens, with about 0.15 GiB of VRAM left (0.49 GiB at 32K): 69,391 is the most this card holds, and
128K needs the K/V cache out of VRAM or at lower precision (see "Long context and video memory" in
[docs/TUNING.md](docs/TUNING.md)). If another program holds VRAM, start with `--context 32768`.

The long-form explanation with charts, in Russian and English, is `docs/index.html`; `docs/TUNING.md` has the tables and `docs/PI.md` the agent setup.

## Layout

| Path | What it is |
| --- | --- |
| `src/qwen35/` | the engine: `q36_engine.cu`, `q36_kernels.cuh`, `q36_main.cpp` |
| `src/kernels/`, `src/prefill/`, `include/` | CPU expert kernels, CUDA kernels, the int8 prompt kernels |
| `third_party/llama.cpp/` | the part of ggml the engine links (CPU backend, quantization, MMQ kernels); see `VERSION.txt` |
| `serve/` | the HTTP server, the web chat page, telemetry and Prometheus collectors |
| `strata_control/`, `strata_cli/` | server helpers (config, tokenizer packs, benchmark client) |
| `tools/` | build, setup, model download, launcher, GGUF and tokenizer tools |

`tools/vendor_ggml.sh <llama.cpp checkout>` refreshes `third_party/llama.cpp`. A newer ggml may need small changes in
`src/prefill/` (the MMQ entry points change between releases).

## Waiting in pi

The engine keeps checkpoints of its state at the start of a conversation (system prompt and tools), in memory and in
`run/ckpt`, so a new pi session reads about 20 tokens instead of 1,500: a question that waited 21 s now takes 1-6 s. The
first question in a new directory reads the shared start once (9 s with the model in the page cache; the CPU computes
part of the experts next to the card). Details in [docs/PI.md](docs/PI.md) and
[docs/TUNING.md](docs/TUNING.md).

## Metrics

The server keeps its request accounting, the telemetry module and the two endpoints from upstream: `GET /metrics` (JSON,
the slots and the last requests) and `GET /metrics/prometheus` (Prometheus text format). They are kept on purpose for
later use.

## Tests

```sh
tools/run_tests.sh         # every serve/test_*.py module, one process each
```

## License

See `LICENSE`. The vendored ggml sources keep their own MIT license (`third_party/llama.cpp/LICENSE`).
