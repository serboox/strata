# LFM2.5-24B-A2B on this laptop: measured speed and the plan for Strata

Measured on one machine: i7-11370H (4 cores, 8 threads), RTX 3050 Ti Laptop (4 GB), 31 GB RAM, turbo off, file
`LFM2.5-24B-A2B-UD-Q4_K_XL.gguf` (13.85 GiB, the Unsloth "Preview" build). The plan with all backlog items is in
[lfm25-backlog.html](lfm25-backlog.html); this page holds what has been measured and checked so far.

## What the file is (read from its header with `tools/lfm_model_map.py`)

| | |
| --- | --- |
| architecture | `lfm2moe`, 24 billion parameters, 2 billion active per token |
| layers | 40: 30 short-convolution layers and 10 attention layers (`ccAcccAccc...`, attention at 2, 6, 10, ..., 38) |
| attention | 32 heads, 8 key/value heads, head size 64, a norm on q and k, RoPE base 1e7, 20 KiB of cache per token in f16 |
| short convolution | input projection to three parts (b, c, x), depthwise convolution of width 3 on b times x, times c, output projection |
| feed-forward | 2 dense layers first (11776 wide), then 38 layers with 64 experts of size 1536, 4 used per token, no shared expert |
| router | sigmoid of the router logits, a bias used only to pick the top 4, the weights of the picked experts normalized |
| vocabulary | 128000, the embedding is used again as the output matrix |
| context | 131072 |
| languages | this file names 10 in its metadata: en, ar, zh, fr, de, ja, ko, es, pt, it; the card of the final release on Hugging Face (LiquidAI/LFM2.5-24B-A2B) lists 16, Russian and Polish among them, so this file (its name says "Preview") may be an older build |
| quantization in this file | experts Q4_K or Q5_K (gate, up) and Q5_K or Q6_K (down); attention, dense layers and the embedding Q8_0; the convolution projections Q4_K |

The 428 tensors match this layout exactly (`python3 tools/lfm_model_map.py MODEL.gguf` prints the check and the sizes). The
experts are 13.06 GiB of the 13.85 GiB (94.3 %); one decoded token reads 876 MB of them, a layer of experts is 352 MiB. The CPU
worker of Strata has kernels for the formats of this file (Q4_K, Q5_K, Q6_K and Q8_0; the Q4_K and Q5_K ones are the kernels sped
up on 4 November).

The math follows the reference of llama.cpp (`src/models/lfm-base.cpp`, `lfm2moe.cpp`, `build_moe_ffn` in `llama-graph.cpp`):
a block is `x + op(rms(x, attn_norm))` with `op` the convolution or the attention, then `x + ffn(rms(x, ffn_norm))`; the last norm
is `token_embd_norm`.

## Speed with llama.cpp under the 80 C rule

llama.cpp 18ab62a built with CUDA for sm_86; `tools/llama_thermal_bench.py` runs `llama-server` with the CPU cap and the same
temperature governor as the Strata server (the process is paused for part of every half second above 80 C), then sends 2 warm-up
and 8 measured requests of 900 tokens back to back, every run started with the package at 55 C or below. Decode speed is taken from
the server's own timings, so the pauses are inside it.

Where the experts live (llama-bench, 2.1 GHz cap, 128 generated tokens, 6 threads, `-ub 256`, no temperature limit):

| `--n-cpu-moe` (layers below it keep their experts on the CPU) | tokens per second |
| --- | --- |
| 99 (all 38 MoE layers on the CPU) | 21.3 |
| 33 (31 MoE layers on the CPU, 7 in VRAM) | 25.4 |
| 32 | does not fit: the compute buffers no longer allocate in 4 GiB |

Sustained decode with `--n-cpu-moe 33`, context 8192, MTP does not exist for this model:

| CPU cap | median tok/s | range | first 5 / last 3 requests | package peak, C | engine paused |
| --- | --- | --- | --- | --- | --- |
| 2.1 GHz | 25.1 | 24.85 to 25.22 | 25.0 / 25.1 | 70 | 0 s |
| 2.4 GHz | 27.0 | 25.2 to 27.6 | 27.2 / 26.3 | 74 | 0 s |
| 2.7 GHz | 27.95 | 26.2 to 29.2 | 28.1 / 27.5 | 79 | 9 s |
| 3.0 GHz | 20.7 | 15.6 to 24.5 | 21.4 / 17.7 | 79 | 123 s |
| none (3.3 GHz) | 10.8 | 9.9 to 14.5 | 11.4 / 11.5 | 79 | 463 s of about 770 |

With the Strata kernels in the expert matmuls (the section "Fewer cycles per row" below) and 4 threads, the same test:

| CPU cap | threads | median tok/s | range | package peak, C | engine paused |
| --- | --- | --- | --- | --- | --- |
| 2.4 GHz | 4 | 31.1 | 30.6 to 31.3 | 73 | 0 s |
| 2.7 GHz | 4 | **32.4** | 30.0 to 33.2 | 78 | 5 s |
| 3.0 GHz | 4 | 23.7 | 20.5 to 32.1 | 82 | 87 s |
| 2.7 GHz | 6 | 29.9 | 26.2 to 32.3 | 80 | 17 s |

The limit moves from 28.0 to **32.4 tokens per second** (+16 %) at the same 2.7 GHz cap, and 4 threads beat 6 under the temperature
limit (32.4 against 29.9): the two extra threads only spin in the barriers and warm the package.

Without a cap the package jumps to the limit within seconds and the governor pauses the engine 35 to 85 % of the time, so the
sustained speed falls below half; at 3.0 GHz the engine is already paused for 123 s and the speed drifts down from 21.4 to 17.7.
The highest speed that still keeps the temperature under 80 C without real pauses is at a cap of about 2.7 GHz: **about 28
tokens per second**; 2.4 GHz gives 27 with 5 C more room, and at 2.1 GHz the limit is never reached (70 C). The GPU peaks at 62 to 66 C in every run. One machine, one prompt (a short code request, so the prompt time of these runs
says nothing); the speeds before the 80 C rule are not comparable with these.

Prompt reading (llama-bench, 2.1 GHz cap, no temperature limit, 2 repeats, tokens per second); the micro-batch decides how much
of the card is left for experts:

| micro-batch | experts on the CPU | pp512 | pp2048 | pp4096 |
| --- | --- | --- | --- | --- |
| 256 | 33 | 156 | 156 | 155 |
| 512 | 34 | 268 | 270 | 267 |

For comparison, Strata reads a prompt of Qwen3.6 at 86 tokens per second with the Q4_K_XL file and 132 to 147 with IQ4_XS (README,
"Which file to run"); these are different engines on different models, so it is a pointer and not a verdict (item B07 measures
Qwen on this llama.cpp build).

## Where the time of one decoded token goes

llama.cpp, 2.1 GHz cap, 6 threads, `--n-cpu-moe 33`, 26.5 tokens per second, 37.7 ms per token. A `perf record` of the whole process
(user-space samples of all threads):

| where | share of the samples |
| --- | --- |
| `ggml_vec_dot_q4_K_q8_K` (experts, gate and up) | 38 % |
| `ggml_vec_dot_q5_K_q8_K` (experts, down and some gate/up) | 24 % |
| `ggml_vec_dot_q6_K_q8_K` | 1.4 % |
| OpenMP barrier wait (the threads idle while the GPU part and the transfers of a layer run) | 27 % |
| quantizing the activations, swiglu, copies, the rest | about 8 % |

What follows from it, with the numbers of `tools/lfm_model_map.py`:

- One token reads 876 MB of experts, about 715 MB of them on the CPU (31 of 38 layers). The CPU part takes about 24 ms, so the experts
  stream at only about 30 GB/s, while the same memory delivered 43 GB/s to the AVX-512 Q5_K kernel of Strata in a test with 4 threads
  at the same cap. The kernel, not the memory, sets the pace.
- A quarter of the time (about 10 ms) is not CPU work at all: the GPU part (conv, attention, dense layers, seven layers of experts,
  the 278 MB output matrix) and the round trip to the CPU twice per layer. During it the CPU threads wait and spin.
- A CUDA timeline of the decode (nsys with the graphs traced per node, 88 tokens, 2.1 GHz cap) shows the GPU part: 8.5 ms of kernels
  per token, 81 % of it quantized matrix-vector products (119 per token), 0.4 ms of copies (104 per token, 1.8 MB), and 38 router calls
  per token. The weights it reads are the 0.85 GB outside the experts plus 160 MB of the seven resident layers, which is about
  145 GB/s over the matrix-vector kernels, 85 % of the 172 GB/s ceiling of the card used in TUNING.md. So the GPU part is probably
  bound by the memory of the card and not by launches (the graph launches are 4 % of the CUDA API time); this is read from the
  timeline, not confirmed with a memory counter.
- The ggml loop calls the dot product once per row, and a row of this model is 8 blocks (gate, up) or 6 blocks (down): the call and
  the horizontal sum are a large part of the work per row.
- The experts of the CPU layers sit in the CUDA_Host (pinned) buffer type, so the runtime weight repack of llama.cpp never applies to
  them; with and without `--no-host`/`--repack` the speed is the same (26.0 to 26.7, noise).
- Threads: 3 give 21.4, 4 give 25.9, 5 give 25.6, 6 give 26.0 to 26.5, 8 give 21.1 tokens per second. The four physical cores
  already saturate what the kernel can read; more threads only spin in the barriers (and warm the package).

## Fewer cycles per row: the Strata kernels inside llama.cpp

The AVX-512 Q4_K and Q5_K kernels of Strata (bit-identical to ggml's dot, checked on 392 cases of `tests/kq512_parity.cpp`, now also
for plain rows of 1 to 16 blocks through `kq512_rows`) were called from `ggml_compute_forward_mul_mat_id_one_chunk` instead of the
per-row dot (a patch of about 70 lines, recipe in the experiment folder; `STRATA_KQ512=0` switches it off). Same model, same
placement, 2.1 GHz cap, tokens per second of llama-bench (128 tokens, 3 repeats, order A B B A):

| threads | ggml dot | Strata kernels | gain |
| --- | --- | --- | --- |
| 6 | 26.7 | 30.5 | +14 % |
| 4 | 25.8 | 30.4 | +18 % |

On plain rows of the shapes of this model (`kq512_parity --bench-rows`, 2.1 GHz, 4 threads, cold memory, one token):

| rows | ggml dot | Strata kernel |
| --- | --- | --- |
| Q4_K, 2048 values (gate, up) | 27.4 GB/s | 38.4 GB/s |
| Q4_K, 1536 values | 18.5 GB/s | 38.3 GB/s |
| Q5_K, 2048 values | 28.8 GB/s | 38.2 GB/s |
| Q5_K, 1536 values (down) | 28.6 GB/s | 30.2 GB/s |

One thread on hot data reads 11.7 GB/s with the kernel against 8.5 with ggml (25 cycles per block of 256 weights at 2.1 GHz); four
threads on cold data reach 38 GB/s for the rows of 2048 values. That is close to the 43 GB/s this machine gave in the best test, so
for the gate and up matrices the kernel is near the memory. The down rows of 1536 values read only 30 GB/s, so that matmul may still
have some headroom (it needs profiling); on the whole a faster dot product is not expected to add much, and what remains is the
part outside the experts.

## Expert deferral, measured

The expert with the lowest rank of every MoE layer on the CPU (layers 2 to 32) is computed on the CPU while the GPU already works on the
next layer, and its result is added one layer later. It was built in the llama.cpp graph of LFM2 and in its scheduler (the deferred CPU
split starts without waiting for the GPU), and the overlapped run gives the same text as the same graph run one step after another.

Speed (llama-bench, 2.1 GHz cap, 4 threads, `--n-cpu-moe 33`, 128 generated tokens, two runs per row in an A/B/B/A order):

| deferred experts of 4 | tokens per second | change |
| --- | --- | --- |
| 0 (stock graph) | 30.3 (30.5 and 30.05) | |
| 1 | 31.95 | +5.5 % |
| 2 | 31.5 | +4.1 % |
| 3 | 31.85 | +5.2 % |

Quality (llama-perplexity on 6 chunks of 512 tokens of English text and code, CPU only, KL divergence against the stock graph):

| deferred experts of 4 | mean KLD | same top token | perplexity |
| --- | --- | --- | --- |
| none, one layer computed in two groups (the noise floor) | 0.016 | 95.0 % | +0.7 % |
| 1 | 0.074 | 87.6 % | +1.6 % |
| 2 | 0.165 | 82.4 % | +5.0 % |
| 3 | 0.355 | 75.3 % | +14.6 % |

The gain is about a seventh of the estimate. The likely reason (not measured separately): the whole GPU part is 9 ms per token, about 0.2 ms per
layer, and the copies and the wait for the router stay on the critical path, so there is little to hide. The cost is real: LFM has no
shared expert, so a missing expert is not covered by anything. The patch stays in the private archive, off by default.

## Experts in a smaller format, measured

The Unsloth file `UD-Q3_K_XL` (10.2 GiB against 13.9 GiB) has IQ3_XXS gate and up experts and IQ4_XS down experts, which is a quarter
fewer bytes per token. Run with the same llama.cpp (stock dot products, the Strata kernels cover only Q4_K and Q5_K there):

| file | `-ncmoe` | tokens per second |
| --- | --- | --- |
| UD-Q4_K_XL | 33 | 29.0 and 30.0 (two series) |
| UD-Q3_K_XL | 33 | 11.9 (the first run of the file, spread of 4.2 over its 3 repeats) |
| UD-Q3_K_XL | 31 | 16.4 |
| UD-Q3_K_XL | 30 | 16.5 |
| UD-Q3_K_XL | 29 or 28 | does not fit in 4 GiB |

(llama-bench, 2.1 GHz cap, 4 threads, 128 generated tokens. 33 is the best placement of Q4_K_XL that fits, 30 the best of Q3_K_XL.)

Difference of the output from UD-Q4_K_XL (llama-perplexity on 30 chunks of 512 tokens, CPU only): mean KLD 0.136, same top token 83.9 %,
perplexity 12.07 against 12.05 (ratio 1.002 with a standard error of 0.007). The perplexity is the same within the noise, so this does
not show that the file is worse, only that its output is clearly different.

The profile of the decode with this file: `ggml_vec_dot_iq3_xxs_q8_K` 59 % of the CPU samples, `ggml_vec_dot_iq4_xs_q8_K` 14 %, the wait at the
thread barrier 14 %. The likely reason for the slowness (cycles per byte were not measured): the IQ3_XXS dot product is a table lookup
per group of values, so the file has fewer bytes but the loop needs more CPU time per byte than the Q4_K loop. To beat the 30 tokens
per second of Q4_K_XL the token rate would have to rise 1.8 times, so the IQ3_XXS dot product would have to be more than twice as fast
as ggml's, for an output that is clearly different from Q4_K_XL, so the route is closed for now. The file `UD-IQ3_XXS` (8.3 GiB, IQ2_S
gate and up experts) was not tried: its experts are two-bit, so its output would differ more.

## What is left, and what each step would give (estimates from the measurements above)

Per token at 30.5 tokens per second (33 ms): the CPU experts about 20 ms, the GPU part about 9 ms, synchronization and the rest about 4 ms.

| idea | what it does | estimate | cost or risk |
| --- | --- | --- | --- |
| overlap the GPU part with the CPU experts (the "expert deferral" of KTransformers) | measured, see above | +4 to +5 % | not worth it: the output changes (mean KLD 0.07 for one deferred expert) |
| experts in a smaller format (IQ3 or IQ4 class) | about 25 % fewer bytes on the CPU | measured for UD-Q3_K_XL, see above: 16.5 against 29 to 30 tokens per second, each at its best placement (about 45 % slower) | no Strata kernel for IQ3_XXS; the output differs from Q4_K_XL (KLD 0.136) |
| smaller non-expert weights on the GPU (output matrix Q8_0 to Q6_K or Q4_K, attention and dense Q8_0 to Q6_K) | less GPU time | about +3 to +6 % | quality |
| the most used experts of all layers in VRAM instead of whole layers (a static cache, chosen on other prompts) | on held-out prompts the same number of experts (448, the count of 7 whole layers; the experts differ in size between layers) covers 22 to 30 % of the expert picks against 18.4 % for 7 whole layers | about +3 to +9 % (estimate: CPU expert work 4 to 14 % lower at 20 ms of 33 ms) | needs per-expert slots and a split of every expert operation between GPU and CPU |
| an eighth layer of experts in VRAM | one layer less on the CPU | about +2 % | does not fit at context 8192 today |
| speculative decoding with the vendor draft model DSpark (327.7M parameters, block 9, SGLang only) | more tokens per step | none expected | a verify step of 3 tokens reads about 2.8 times the expert bytes (4 experts of 64 per token, picked independently) and, at 80 % acceptance, yields about 2.4 tokens (the Qwen precedent: +8 %); a step of 9 tokens reads about 44 % of all experts |
| a faster dot product | fewer cycles per block | at most a few percent | the kernel is near the memory |

## What is not measured yet

- The time to the first token with a conversation history, the speeds at 8K and more of context (the rest of item B01), the
  prompt reading under the 80 C rule.
- The speed of Qwen3.6 on the same llama.cpp build (item B07), to say which model is faster here; Strata's own Qwen numbers
  (33 tokens per second sustained with speculation, see TUNING.md) are not a comparison, because LFM has no draft block.
- Quality: only the KL divergence, the top-token agreement and the perplexity of the deferral and of the Q3_K_XL file against Q4_K_XL are measured,
  on English text and code (items C01 to C11 and Russian, named on the card of the release but not in the metadata of this file, are not).

## Strata port: what can be reused and what has to be written

Reused as they are: the CPU expert pool and its Q4_K, Q5_K and Q6_K kernels, the Q8_0 row kernel, the thermal governor, the
frequency cap and its guard, the launcher and the server frame. To write for `lfm2moe`:

1. the loader and the tensor map (the check above is the specification; the C++ side reads the same names and shapes);
2. the short-convolution block with its state (two samples per channel and layer), decode and prompt variants;
3. attention with head size 64, 8 key/value heads, the norms on q and k, and the cache of 10 layers;
4. the router: sigmoid, the pick by biased score, normalized weights, 4 of 64 experts, two dense layers first;
5. the tokenizer with the `lfm2` pre-tokenizer, the chat template and the tool-call dialect;
6. parity tests against llama.cpp (logits within tolerance, equal greedy text), then tuning;
7. a model-adapter interface so that the Qwen path stays untouched and the file decides the engine (`general.architecture`).

The first expected result of the port is the placement already measured here (attention, convolution, dense layers and the
embedding in VRAM, seven layers of experts resident, the rest on the CPU) with the Strata expert kernels and 4 threads, where
llama.cpp with the kernels patched in reaches 32.4 tokens per second at the 80 C limit (28.0 with its own dot product and 6 threads);
the port has to match that before it can claim anything on top, and it owns the synchronization between the two halves of a layer,
which is where the overlap above would live.
