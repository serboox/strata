# Backlog: lossless speed-up of Qwen3.6-35B-A3B toward 50-60 tok/s

Source: [qwen36-lossless-backlog.html](qwen36-lossless-backlog.html) (the reasoning, charts and evidence). This file is the work list.

Limits: the same model, all 8 experts, the weights as in the file (no coarser quantisation), software only, package and GPU below 80 C.
Baseline: 65 ms a step, 34.9 tok/s (measured); best measured inside the limits 44.8 tok/s (16K window, resident slots, prefetch 3).
Every gain below is an estimate (a share of the 65 ms step when the item is applied alone), not a measurement. Nothing is implemented.

Classes: **bit-exact** (same numbers as today), **drafts only** (only proposed tokens change, text is the same), **rounding** (GPU-computed experts round activations in blocks of 32 instead of 256; KLD about 0.005).

## Items, most likely first

| ID | Item | Probability | Gain alone | Class | Effort | Depends on | First cheap check | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L1 | Hot experts in the VRAM the context does not use: tiered K/V cache (8K/16K/32K/64K) and a watched 300 MiB reserve instead of 768 MiB; the rest goes to resident expert slots | 85% | -9..-13 ms (-4..-7 ms on top of prefetch) | rounding | medium | - | replay `STRATA_ROUTE_TRACE` offline: hit rate at 1, 5, 9, 12, 16 slots per layer | not started |
| L2 | Closed-loop frequency governor (aim 77 C, steps of 100 MHz, 1.8-3.3 GHz) in front of the 80 C pause | 80% | -2..-4 ms (+3..+6% on short and medium requests) | bit-exact | small | - | none; compare with the fixed 2.1 GHz cap on 8 back-to-back requests | not started |
| L3 | Exact 4-bit repack of the IQ3_S gate/up matrices (same decoded values, 136-byte block, one byte shuffle); +2.0 GiB RAM | 75% | -8..-14 ms (-6..-10 ms after L1 and prefetch) | bit-exact | medium-large | GPU gate/up 4-bit path for resident and guessed experts | microbenchmark next to `iq_vbmi_bench`; go if a block costs under 25 cycles at 2 rows | microbenchmark done: GO (see Results); engine work not started |
| L4 | Cheap draft head: context tokens plus a frequency list (16-32K rows) instead of a full pass over the 248,320-row output matrix, twice a step | 65% | -4..-6 ms | drafts only | medium | new Q6_K row-gather kernel | offline: share of accepted drafts inside "context + top N" for N = 8, 16, 32K; go at 99% | not started |
| L5 | Draft depth chosen for every step from the MTP block's own probability and the last step's acceptance | 60% | +3..+12% tok/s | drafts only | small-medium | - | log (draft probability, accepted) for a few thousand steps; pick the threshold offline | not started |
| L6 | Audit of the 12 ms outside the layers: chain the two drafts on the device, keep the MTP block's hot experts in VRAM | 50% | -1..-3 ms | bit-exact | small-medium | - | `STRATA_Q36_OVERLAP_STATS=1` and one `nsys` run of the benchmark mode with MTP | not started |
| L7 | PCIe prefetch, second version: guess two layers ahead, a share for the MTP block | 45% | -1..-4 ms | rounding | medium | existing prefetch | recall of a guess two layers ahead from the route trace | not started |
| L8 | n-gram drafts as a second proposer when the last 6-12 tokens repeat earlier context | 45% | +5..+14% tok/s on copying turns | drafts only | medium | cheaper rows (L1, L3) | rerun the n-gram statistics on the owner's own sessions | not started |
| L9 | Dense GPU kernels near the card's read speed; fewer launches (about 670 a token) | 40% | -1..-4 ms | bit-exact | large | - | stall reasons in Nsight Compute (needs root once) | not started |
| L10 | Decode attention at a long context: K/V read closer to 172 GB/s | 40% | -4 ms at 16K fill, -8..-10 ms at 28K | bit-exact | medium | - | per-kernel times of a long-context step in `nsys` | not started |
| L11 | One arena for prompt buffers and expert slots (lend about 330 MiB to slots while decoding) | 40% | -2..-4 ms, any context | rounding | large | L1 | same trace replay as L1 with 3-5 slots | not started |
| L12 | Platform profile `balanced-performance` | 30% | 0..-5 ms (unknown) | bit-exact | one test | root once; the owner declined this test earlier | one `tools/cooling_kit.py` run with the profile | not started |
| L13 | Warm the guessed experts into the CPU caches while the GPU runs the mixer | 25% | -1..-2 ms | bit-exact | small | - | hardware counters of the pool: cycles stalled on memory | not started |
| L14 | Exact two-stage verify head (top 4 bits first, exact rows of the few candidates) | 25% | about -1 ms | bit-exact | medium | - | candidate-set size on dumped logits | not started |

## Conditional items

| ID | Item | When it matters | Status |
| --- | --- | --- | --- |
| C1 | Strict mode: GPU experts reproduce the CPU bits (quantise activations in blocks of 256, keep the CPU's float order) | no speed; turns L1, L7, L11 and today's prefetch into bit-exact | not started |
| C2 | Speculative sampling for requests with a temperature (standard rejection rule, exact distribution) | clients that sample; a 30 October prototype gave +39..+126% at temperature 0.7 (not committed) | not started |
| C3 | Two interleaved streams (GPU mixers of one request while the CPU computes the experts of the other) | about 1.3 times the total tok/s for two parallel requests; nothing for one stream; more heat | not started |

## Checks that need no code

| # | Check | Feeds | Status |
| --- | --- | --- | --- |
| K1 | Record `STRATA_ROUTE_TRACE=<file>` during normal work | L1, L7, L11 | not started |
| K2 | Step split with `STRATA_Q36_OVERLAP_STATS=1` | L6 | not started |
| K3 | Draft statistics: acceptance against draft probability, accepted tokens inside "context + top N" | L4, L5, L8 | not started |
| K4 | The best current start inside the limits: `bin/strata-serve --model models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-IQ4_XS-mtp.gguf --experts-used 8 --expert-prefetch 3 --context 16384 --vram-reserve 300` (44.8 tok/s, 16K window) | baseline | measured (E11) |
| K5 | `--vram-reserve 300` on the default stack (Q5_K dense frees about 0.68 GiB): resident slots at 64K? | L1 | not measured |

## Suggested order

1. K1-K3 (a day, no risk). 2. L3 microbenchmark. 3. L2. 4. L1, then L4 and L5, a standard run after each (8 requests of 900 tokens back to back, package peak recorded). 5. L3 in full if its microbenchmark said go. Then the rest by probability.

Acceptance of every item: the sustained figure (median of the second half of a back-to-back series, package peak next to it), not the first request on a cool machine; bit-exact items keep the greedy checksums; rounding items get a KLD run.

## Projection

Prefetch + L1 + L3 + L2 + L4: 44.7 ms (50.8 tok/s) at the low ends, 42.5 ms (53.4) weighted by probability, 37.5 ms (60.5) at the middle estimates; L5 comes on top as more tokens a step. At a 16-32K fill every figure is about 15% lower unless L10 lands.

## Results

### L3 microbenchmark (12 November 2026)

- **Expected.** A gate/up row stored as 4-bit IQ4_XS-format codes (same decoded values as IQ3_S) should cost fewer cycles than the IQ3_S kernel; go if under 25 cycles a block at 2 rows, pool time about x0.69.
- **Done.** `tests/iq_vbmi_bench.cpp` got `check_gu4` (the existing `iq4xs_vnni_rows` kernel on rows of 2048 weights against ggml's IQ4_XS dot, 1, 2, 3 and 8 tokens) and `bench_gu` (both gate/up forms, streaming from 48 experts per thread and cache-resident with 1). Run: `ninja iq_vbmi_bench && ./iq_vbmi_bench --bench --threads N --seconds 2`. Synthetic experts, no model. Turbo off, no frequency cap, machine not idle (load 0.7).
- **Got.** Correctness: relative error 1e-7 to 1.7e-7 against ggml for every token count, so the kernel already handles 8-block rows. Speed, one expert (gate and up) per call, us per expert and thread:

  | | IQ3_S kernel | 4-bit repack | ratio |
  | --- | --- | --- | --- |
  | 1 thread, streaming, 1 / 2 / 3 rows | 98.5 / 123.7 / 138.1 | 71.8 / 72.2 / 93.8 | 0.73 / 0.58 / 0.68 |
  | 1 thread, in cache, 1 / 2 / 3 rows | 122.8 / 135.4 / 160.9 | 41.3 / 63.8 / 85.5 | 0.34 / 0.47 / 0.53 |
  | 4 threads, streaming, 2 / 3 rows | 124.6 / 136.8 | 85.8 / 94.0 | 0.69 / 0.69 |

  At 4 threads the repack reads 52.0 / 47.4 GB/s: the RAM ceiling (52-54 GB/s) is reached, so there the gain is memory-bound, and it matches the x0.69 estimate.
- **Concluded.** GO. The IQ3_S gate/up cost drops by about 31% at the engine's 4 threads and 2-3 rows. Nothing else is needed on the CPU side: the existing 4-bit kernel does the work. What remains is the engine part: a one-time repack of IQ3_S gate/up into the 136-byte layout at load, +2.0 GiB RAM, and the GPU gate/up path for resident and guessed experts (they read the same arena).
- **Incident.** The 4-thread run without a frequency cap took the package to 86 C for a few seconds, above the 80 C limit. Later runs of this kind use `tools/cpucap` at 2200 MHz.

### Implementation status (12 November 2026; code written, nothing built or run since the first benchmark)

Every switch is off by default. The code of the items below has not been compiled; the first build will show syntax and type errors.

| Item | Code | Switch |
| --- | --- | --- |
| L2 closed-loop governor | `serve/capctl.py`, `CpuCap.set_mhz`, unit tests (these passed) | `--cpu-adaptive` |
| L3 4-bit repack of IQ3_S gate/up | `iq3s_repack_to_4bit`, odd-table variant of the IQ4_XS kernel, CPU dispatch (`NativeFmt::gu_odd4`), load-time repack, check in `iq_vbmi_bench`; CPU path of the mapped experts only (the GPU paths and the pinned arena keep IQ3_S) | `--iq3s-4bit` |
| L4 draft head over a token list | `STRATA_Q36_DRAFT_LIST`, a copy of the listed output rows in VRAM, `tools/draft_list.py`; the list is by frequency only (no context tokens yet) | `--draft-list FILE` |
| L5 draft depth per step | one draft after a step whose first draft failed (no draft probabilities) | `--mtp-adapt` |
| L6 chain the drafts on the device | draft tokens go device to device, picks return once | `--chain-device` |
| L8 n-gram drafts | `ngram_continuation`, an external draft list in `spec_step` | `--ngram MATCH[,MAX]` |
| L10 attention | V-load depth 16 instead of 8 (same order of sums) | `--attn-ahead` |
| L12 platform profile | `apply_platform_profile` (sudo -A fallback) | `--platform-profile NAME` |
| L7 prefetch two layers ahead | guess for layer li+2 after layer li's router, three rings and three guess buffers, per-layer ready flags | `--prefetch-lead2` |
| L1 hot experts, elastic | `--resident-experts` (16K window, 300 MiB reserve) and `--kv-lazy TIER`: K/V on CUDA virtual memory, pages mapped as the position grows (`VmmBuf`, `kv_ensure`), the experts' slots sized for the tier, whole layers of slots (with their guess slots, so also with prefetch) given up when the context outgrows it; one session | `--resident-experts`, `--kv-lazy` |
| L9 dense GPU | the software side only: GPU clock lock (`--gpu-clocks MIN,MAX`, release with `reset`); no kernel was rewritten (they need stall data from Nsight Compute first) | `--gpu-clocks` |
| L11 shared arena | the expert staging pair, the prompt attention score buffers and the tails of the row-proportional mixer scratch buffers (the shared region sits on virtual memory, decode keeps only the pages of its 8 rows) are mapped only while a prompt is read (`lent_alloc`, `arena_map`) | `--lend-buffers` |
| L13 warm guessed experts | prefetcht2 of the guessed experts' first bytes | `--warm-experts KIB` |
| C1 strict mode | every expert on the CPU: no prefetch, no resident experts | `--strict` |
| C2 speculative sampling | `target_distribution`, `judge_sampled`, a judge hook in `spec_step` | `--spec-sampling` |
| L14 two-stage verify head | `head_rows_pruned`: the listed tokens first, row norms (`k_prune_count`) prove that no other token can beat their best, else the whole matrix; needs `--draft-list` and a Q6_K head; counts hits and misses | `--head2` |
| C3 two streams | the engine's batch window (`--batch N --batch-mtp`): the rows of N sessions share one pass over the experts; a layer-by-layer interleave of two windows was not written | `--parallel N` |

C1 is the CPU-bit form only (`--strict`, also an engine mode): the GPU expert kernels were not rewritten to the CPU's rounding.

### First model runs of the new switches (13 November 2026)

- **Expected.** L3 about -31% of the IQ3_S gate/up time (microbenchmark); the draft items a few percent on this prompt; the others small.
- **Done.** `tools/levers_lossless.d/*.json` (one lever each) run through `tools/lever_matrix.py run --each --budget none --prompts code --repeats 2 --warmup 1 --max-tokens 500 --cpu-max-mhz 2200`, between two baselines; IQ4_XS file with the MTP block, the code prompt, package cooled to 55 C before each request. Two runs: the first (L3 only) and the second (the rest).
- **Got.**

  | Switch | tok/s | against the baselines | Text | Peak C |
  | --- | --- | --- | --- | --- |
  | baseline (first run) | 34.1 and 36.5 | drift +6.9% (flagged unstable) | - | 62-64 |
  | baseline (second run) | 36.5 and 36.6 | drift +0.4% (stable) | - | 62-66 |
  | `--iq3s-4bit` (L3) | 42.7 and 42.8 | +17% to +25% | differs | 62-63 |
  | `--mtp-adapt` (L5) | 36.8 | +0.6% | identical | 64 |
  | `--chain-device` (L6) | 36.6 | +0.2% | identical | 63 |
  | `--attn-ahead` (L10) | 36.6 | +0.3% | identical | 63 |
  | `--ngram 8,4` (L8) | 36.7 | +0.5% | identical | 64 |
  | prefetch 3 + `--warm-experts 64` (L13) | 40.0 | +9.3% (prefetch 3 alone was +10% to +13% in November, not run in this session) | differs (prefetch rounds on the GPU) | 64 |

- **Concluded.** L3 is the one that pays: +17% with a stable baseline, and it runs hot no more than the baseline. Its text differs from the baseline's because the 4-bit kernel adds floats in another order than the IQ3_S kernel (the same reason the IQ3_S kernel itself is not bit-identical to ggml); the quality cost has not been measured (`tools/kld_experts.py` is the check). L5, L6, L8 and L10 are correct (identical text) and inside the noise on this prompt and at a short context: L10 matters at a long context, L8 on text that repeats itself, so they were not measured where they should pay. L13 gave no visible gain over the known prefetch figure; it needs a run next to prefetch 3 alone. Not run yet: L1 (`--kv-lazy`, `--resident-experts`), L4/L14 (need a token list), L7, L11, C2, C3 (`--parallel 2`) and the clock lock.

### Prefetch, lazy K/V and the combinations (13 November 2026)

- **Expected.** The singles add: L1 gives resident experts where the 64K K/V cache sat, the draft items add a little, L3 cuts the CPU expert time. Prefetch (needs the experts pinned) and the 4-bit repack (mapped experts only) exclude each other, so two families.
- **Done.** The lever definitions now sit together in `tools/levers_lossless.d` (9 files); each session is `lever_matrix.py run --levers A,B,C --budget none --prompts code --repeats 2 --warmup 1 --max-tokens 500 --cpu-max-mhz 2200 --max-load 3.5` between two baselines. In the engine, `--kv-lazy` takes a 300 MiB margin instead of 768 MiB (`--vram-reserve` still wins): the first run left 0 resident slots because the old margin ate the room the lazy cache freed.
- **Got.** Baselines 36.2 to 36.7 tok/s (drift under 1.1%), package 59-66 C throughout.

  | Setting | tok/s | against the baseline | Text |
  | --- | --- | --- | --- |
  | prefetch 3 (reference) | 40.3 | +10.1% | differs (GPU rounding) |
  | prefetch 3 + guess two layers ahead (L7) | 38.3 | +4.6% | differs |
  | prefetch 3 + lazy K/V, 768 MiB margin (first run) | 39.9 | +9.0%, 0 resident slots | differs |
  | prefetch 3 + lazy K/V, 300 MiB margin (L1) | 43.8 | +20.4% | differs |
  | prefetch 3 + lazy K/V + warm experts (64 KiB) + `--mtp-adapt` + `--chain-device` + `--ngram 8,4` | 44.9 | +22.9% | differs |
  | `--iq3s-4bit` alone (first session) | 42.8 | +17% | differs |
  | `--iq3s-4bit` + `--mtp-adapt` + `--chain-device` + `--attn-ahead` + `--ngram 8,4` | not measured: the run was aborted by the runner ("other programs are stalling on memory", 15.7%) | - | - |

- **Concluded.**
  - **Best measured set (family with prefetch):** `--expert-prefetch 3 --kv-lazy 8192 --warm-experts 64 --mtp-adapt --chain-device --ngram 8,4`, 44.9 tok/s, +22.9% against the unchanged engine (36.6) and +11% against prefetch 3 alone. About three quarters of that is L1 (the lazy K/V cache with resident slots), the rest is the draft switches and the warm-up, which are each inside the noise alone and together add about 1.1 tok/s.
  - **L7 does not pay** (38.3 against 40.3 for prefetch 3 alone): the guess made a layer earlier is no better (22% of the entries from the ring, as before) and the third ring costs a layer of copies. Leave `--prefetch-lead2` off.
  - **The 4-bit repack costs memory, not 2 GiB:** the engine keeps the GGUF's gate/up bytes (the GPU and prompt paths read them) and adds an anonymous copy of the repacked gate/up, about 10.6 GiB. With the 17 GiB model in the page cache and the desktop, the runner saw memory stalls in the combined session. The first session (+17%) passed. The family needs either that memory free or the original gate/up pages dropped after the repack (not written), and its combination was not measured.
  - Not compared yet: the best set with the 4-bit family's gain, the quality (KLD) of the 4-bit repack and of the prefetch family, a long-context run for `--attn-ahead` and `--ngram`, the final set on both prompts.

### The 4-bit family with the original pages dropped (13 November 2026)

- **Expected.** Dropping the original gate/up pages from the page cache after the repack (`--iq3s-4bit-drop`) removes the memory stalls that aborted the combined session, and the draft switches add a little on top of the repack.
- **Done.** `STRATA_IQ3S_4BIT_DROP` (`posix_fadvise(DONTNEED)` on the gate/up ranges of every layer after its repack), launcher flag `--iq3s-4bit-drop`, lever variant `iq3s_4bit:drop`; the same matrix settings as before.
- **Got.** Baselines 36.5 to 36.7 tok/s, package 63-64 C, no memory stalls.

  | Setting | tok/s | against the baseline | Text |
  | --- | --- | --- | --- |
  | `--iq3s-4bit --iq3s-4bit-drop` | 42.6 | +16.7% | differs |
  | the same + `--mtp-adapt` + `--chain-device` + `--attn-ahead` + `--ngram 8,4` | 42.8 | +16.8% | differs |

- **Concluded.** The drop works: the session that stalled before ran clean. The draft switches add nothing on top of the repack (42.8 against 42.6), so for this family `--iq3s-4bit --iq3s-4bit-drop` is the whole setting. The memory price: about 10.6 GiB of anonymous memory for the repacked copy, with the 10 GiB of original gate/up pages dropped from the cache (a prompt chunk then reads them from the SSD again; prompt speed was not measured).
- **Overall.** The prefetch family is the best measured: `--expert-prefetch 3 --kv-lazy 8192 --warm-experts 64 --mtp-adapt --chain-device --ngram 8,4` at 44.9 tok/s (+22.9%), against 42.8 (+16.8%) for the 4-bit family. They exclude each other (the repack is written for the mapped experts only; the pinned arena keeps IQ3_S). A repack for the pinned arena (CPU jobs reading repacked copies next to the GPU's IQ3_S slots) is the open step that could add the two gains, at a memory cost the machine cannot pay without dropping something else.
