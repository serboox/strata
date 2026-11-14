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
| L9 dense GPU | GPU clock lock (`--gpu-clocks MIN,MAX`, release with `reset`); the kernels were chosen by an Nsight Systems profile (see the L9 block below): the prefetch guess kernel `k_pf_logits` was rewritten (96 blocks instead of 3, the same bits), the output matrix is read less often through the draft token list, and the strict mode added two expert kernels | `--gpu-clocks`, `--draft-list` |
| L11 shared arena | the expert staging pair, the prompt attention score buffers and the tails of the row-proportional mixer scratch buffers (the shared region sits on virtual memory, decode keeps only the pages of its 8 rows) are mapped only while a prompt is read (`lent_alloc`, `arena_map`) | `--lend-buffers` |
| L13 warm guessed experts | prefetcht2 of the guessed experts' first bytes | `--warm-experts KIB` |
| C1 strict mode | the experts in VRAM slots and the guessed experts run GPU kernels with the CPU kernels' arithmetic (`strict_gu_kernel`, `strict_down_kernel`, a shared SwiGLU in `strict_math_impl.hpp`), so an expert gives the same bits wherever it is computed; `--strict-cpu` keeps the first form (every expert on the CPU) | `--strict`, `--strict-cpu` |
| C2 speculative sampling | `target_distribution`, `judge_sampled`, a judge hook in `spec_step` | `--spec-sampling` |
| L14 two-stage verify head | `head_rows_pruned`: the listed tokens first, row norms (`k_prune_count`) prove that no other token can beat their best, else the whole matrix; needs `--draft-list` and a Q6_K head; counts hits and misses | `--head2` |
| C3 two streams | the engine's batch window (`--batch N --batch-mtp`): the rows of N sessions share one pass over the experts; with `--interleave` the window runs as two windows that alternate layer by layer (`forward_pair`): the GPU computes one window's mixer, router and resident experts while the CPU computes the other window's experts | `--parallel N`, `--interleave` |

C1 and C3 were first written in a reduced form (every expert on the CPU; the batch window only). Their full forms are described in the block "C1 and C3 in full form" below.

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

### Quality of the new switches (teacher-forced KLD, 13 November 2026)

- **Expected.** The 4-bit repack keeps every weight value, so its output should differ from the baseline only by the order of float additions; the prefetch family should stay near the 0.005 measured for the prefetch in November (GPU rounding of the activations).
- **Done.** The method of `docs/TPS_REPORT.md` (E6): `strata-q36 --tokens ... --force ... --dump-full-logits`, 40 prompt tokens and 220 forced tokens of this project's text, 261 positions, logits of each setting against the unchanged engine (`--experts mapped`, `STRATA_IQ_VBMI=1`), 2200 MHz cap. A control: the unchanged engine with ggml's own IQ3_S dot (`STRATA_NO_IQ_VBMI=1`) against the same reference, to see how much a pure change of summation order moves the number.
- **Got.**

  | Setting against the reference | mean KLD | median | same best token | largest KLD |
  | --- | --- | --- | --- | --- |
  | the same engine again | 0 | 0 | 100% | 0 |
  | ggml's IQ3_S dot instead of the vbmi kernel (control: order of sums only) | 0.0042 | 0.0013 | 96.2% | 0.111 |
  | `--iq3s-4bit` | 0.0044 | 0.0015 | 97.7% | 0.081 |
  | `--iq3s-4bit --iq3s-4bit-drop` | 0.0044 | 0.0015 | 97.7% | 0.081 |
  | prefetch 3 + lazy K/V + warm experts | 0.0042 | 0.0016 | 96.9% | 0.069 |

- **Concluded.** The 4-bit repack sits at the noise floor of this test: a change of nothing but the order of float sums (the control) already gives 0.0042, and the repack gives 0.0044. So the repack is as faithful as the IQ3_S kernel is to ggml; no weight changes. The prefetch family is at 0.0042, below the 0.0057 of the November prefetch test and a sixth of what 7 experts cost (0.025). Dropping the original pages changes nothing in the numbers (identical rows). Limits: 261 positions, one text, no task-level check (HumanEval or tool calls) of the final set. The draft switches (`--mtp-adapt`, `--chain-device`, `--ngram`) and `--attn-ahead` do not change the logits of a window (their text was identical in the speed runs).

### Long context (13 November 2026)

- **Expected.** The lazy K/V cache gives resident slots only while the context is short, and gives them up as it grows (the eviction code); `--attn-ahead` should help the decode attention a little. So the gain should shrink with the prompt length, as the prefetch gain did in November (+25% at 16K and +12% at 32K for the old stack).
- **Done.** `lc_bench.py` (a prompt of about 16K and of 32K tokens, 300 tokens decoded, MTP 2): the unchanged engine against the best set (`--expert-prefetch 3 --kv-lazy 8192 --warm-experts 64 --mtp-adapt --chain-device --attn-ahead`), one run each, 2200 MHz cap, package 70-73 C at the prompt peak.
- **Got.**

  | Prompt | Setting | decode tok/s | drafts accepted | prefill tok/s |
  | --- | --- | --- | --- | --- |
  | 16K | unchanged | 31.8 | 0.645 | 128.8 |
  | 16K | best set | 36.9 (+16%) | 0.678 | 126.2 |
  | 32K | unchanged | 27.6 | 0.568 | 106.4 |
  | 32K | best set | 27.9 (+1%) | 0.612 | 108.2 |

- **Concluded.** The eviction code ran without an error at both lengths (no out-of-memory message): the slots were given up as the context grew. The gain falls from +22.9% at a short prompt to +16% at 16K and to nothing at 32K, where every resident slot is gone and the attention takes a larger share of the step; the part of the 32K difference that is not noise is the higher draft acceptance (0.612 against 0.568). Prefill is unchanged at both lengths, so the lent buffers cost nothing there. One run per cell, so differences of a few percent are inside the noise. The set is for contexts up to about 16K; beyond that it is neutral.

### Target 50 tok/s with a quality loss accepted (13 November 2026)

- **Expected.** The dense Q4_K file (E4: -14% a step, KLD 0.04) plus fewer experts plus the best switches should pass 50 tok/s; the November maximum stack (Q4_K, 5 experts, prefetch 3) gave 52.1 and 55.4 with resident slots. The open question was whether 50 holds when the package heats.
- **Done.** `tools/levers_lossless.d/experts_k.json` (K = 7, 6, 5); lever-matrix sessions on `Qwen3.6-35B-A3B-UD-IQ4_XS-mtp-d4.gguf` (dense Q4_K), code prompt, 2200 MHz cap, between two baselines; then a sustained test: 8 requests of 900 tokens back to back with no cool-down (`--cool 90`), at the caps 2200 and 2100 MHz.
- **Got.** Short requests, cool package (baseline 42.4 tok/s, the dense Q4_K file with 8 experts and no switch): the set `--expert-prefetch 3 --kv-lazy 8192 --warm-experts 64 --mtp-adapt --chain-device --ngram 8,4` with 6 experts 54.8 tok/s (+29.2%, 63 C), with 5 experts 54.7 (+28.7%). Sustained, 8 requests in a row:

  | Cap | Setting | tok/s per request | mean | lowest | peak C |
  | --- | --- | --- | --- | --- | --- |
  | 2200 MHz | the unchanged engine on the same file | 42.5, 42.3, 43.1, 42.7, 38.3, 40.7, 34.3, 37.3 | 40.4 | 34.3 | 79 |
  | 2200 MHz | set + 6 experts | 52.3, 50.9, 51.2, 52.8, 49.9, 51.2, 51.7, 51.5 | 51.4 | 49.9 | 76 |
  | 2100 MHz | set + 6 experts | 53.2, 52.1, 51.6, 55.8, 49.9, 52.7, 52.1, 51.2 | 52.3 | 49.9 | 76 |

- **Concluded.** 50 tok/s holds under sustained heat: the set stays between 49.9 and 55.8 over eight requests in a row while the unchanged engine on the same file sags from 42 to 34 once the package passes 78 C. 2100 MHz (the launcher's default cap) is slightly faster and cooler than 2200, so the cap stays at its default. Five experts gain nothing over six (54.7 against 54.8), so the quality cost stops at six. The final setting:

  ```
  bin/strata-serve --model models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-IQ4_XS-mtp-d4.gguf \
    --experts-used 6 --expert-prefetch 3 --kv-lazy 8192 --warm-experts 64 --mtp-adapt --chain-device --ngram 8,4
  ```

  Quality cost (from the November runs of the same ingredients, not of this exact set): dense Q4_K with 6 experts and prefetch 3 gave a mean KLD of 0.066 and the same best token in 90.0% of the positions (E18); the Q5_K variant of 6 experts gave HumanEval 152 of 164 and tool calls 89 of 89 and 36 of 36 (E17). The exact final set has no HumanEval, tool-call or KLD run of its own yet, and its minimum of 49.9 tok/s in one request of each sustained series is a hair under 50: "stable 50" means a mean of 51 to 52 with single requests down to 49.9. Limits: one run per cap, the code prompt only, a short context (the gain was neutral at 32K, see above).

**HumanEval of the final set (13 November 2026).** The engine's own server, 164 problems, greedy, thinking off, the set of the 50 tok/s target (dense Q4_K file, 6 experts, prefetch 3, lazy K/V, warm experts, `--mtp-adapt`, `--chain-device`, `--ngram 8,4`): **146 of 164 (89.0%)**, 1028 s. The unchanged engine gave 153 (`docs/TPS_REPORT.md` E8), the default stack (Q5_K, 7 experts, prefetch 3) 150, Q4_K with 7 experts 152, Q5_K with 6 experts 152. So this set loses 7 problems against the baseline, more than the 2 to 5 of the earlier stacks, which the sign test of E8 called noise; the loss is probably real and comes from the dense Q4_K file together with 6 experts (each alone cost 0 to 3 problems). Problems failed: 10, 32, 37, 38, 50, 62, 83, 93, 99, 115, 124, 126, 129, 130, 132, 134, 145, 163. This is the price of 50 tok/s that the owner accepted; where a smaller loss is wanted, the Q5_K dense file with 6 experts (HumanEval 152) is the next step down in speed (about 47 tok/s in November, not measured with the new switches). The tool-call tests were not run for this exact set.

**The launcher preset and two launcher faults found on the way (13 November 2026).** `bin/strata-serve --fast50` starts the measured 50 tok/s set (dense Q4_K file, 6 experts, prefetch 3, lazy K/V of 8192 cells, 64 KiB warm-up, adaptive draft depth, chained drafts, n-gram drafts 8,4); every flag given explicitly wins. Two faults of the earlier launcher edits were found while checking its dry-run output and fixed: (1) the `--cpu-adaptive` edit (L2) had moved the line that sets the 6 pool workers of the capped start into the adaptive branch, so a plain start used 3 workers from that commit until now; (2) the new long flags made the old short forms `--ch`, `--d`, `--dr`, `--mt`, `--r`, `--s` and `--w` ambiguous; they map back to the old options in `OLD_ABBREVIATIONS`. The speed runs went through `lever_matrix.py`, which passes its own `--workers 6`, so the measured numbers are not affected. `serve/test_launch_fast50.py` and the existing launcher tests pass (136 tests).

**Live check of `--fast50` (13 November 2026).** `bin/strata-serve --fast50 --port 18090`, one chat request (a Python Fibonacci function, 117 tokens) and one request with a tool list: the chat answer is correct and runs at **58.4 tok/s**; the tool request returns `get_weather` with `{"city":"Paris"}` at 57.2 tok/s. The first attempt found a real fault of the lazy K/V cache: the tool prompt (272 tokens) makes the server save a checkpoint at the turn boundary, and `ckpt_copy` read the K/V cells with one 2D copy whose range spans the pages that the lazy cache has not mapped, so the engine stopped with `cudaMemcpy2DAsync ... invalid argument` and the server ended it after 90 seconds of silence. Fixed by copying head by head when the cache is lazy (`ckpt_copy`, the restore path also maps its pages first); the rerun passed. The other 2D copies of the K/V cache are in the session copy of the batch mode, which the lazy cache does not run with (one session only). The tests of the earlier speed runs never saved a checkpoint, which is why the fault was not seen before.

### L9: where the GPU time goes, the draft list and the faster guess kernel (13 November 2026)

- **Expected.** A profile should show which GPU kernels matter at a decode step before any kernel is rewritten; the output matrix (Q6_K, read three times a step) looked like the first target, the guess kernel of the prefetch (`k_pf_logits`) the second.
- **Done.** Nsight Systems (`nsys profile --trace=cuda`, benchmark mode, the final set on the dense Q4_K file, 6 experts): kernel time by kernel. Then three changes: a list of 22,768 frequent tokens for the draft head (`tools/draft_lists/code_ru.i32`, made by `tools/draft_list.py` from the tokens of this project's code and docs, English and Russian: 151 files, 1.57 million tokens), the pruned verify head (`--head2`, norm bound) and a rewritten `k_pf_logits` (grid of 32 by T blocks, a warp per expert, the same order of sums as before, so the same bits).
- **Got.** The profile (benchmark mode, so prompt and decode together):

  | Kernel | Share of GPU time | Calls | Per call |
  | --- | --- | --- | --- |
  | dense Q4_K matvec (`native_q4_k_mmvq`) | 27.1% | 63,224 | 25.5 us |
  | prefetch guess (`k_pf_logits`, before the rewrite) | 14.5% | 11,778 | 73.6 us |
  | output matrix, Q6_K (`native_q6_k_mmvq`) | 12.7% | 308 | 2,464 us |
  | IQ4_XS down on resident experts | 6.6% | 11,359 | 34.5 us |
  | IQ3_S gate/up on resident experts | 6.3% | 12,280 | 30.5 us |
  | GDN recurrence (`k_gdn_seq`) | 3.9% | 9,090 | 25.5 us |

  The output matrix reads at 170 GB/s (the card's 172): at its roofline, so only fewer passes help. Speed of the final set on the dense Q4_K file, code prompt, baseline 42.1 to 42.4 tok/s:

  | Setting (6 experts, prefetch 3, lazy K/V, warm, draft switches) | tok/s | against the baseline |
  | --- | --- | --- |
  | without a token list | 54.8 | +29.2% |
  | with the token list | 58.1 | +36.8% |
  | with the token list, faster guess kernel | 58.2 | +37.3% |
  | with the token list, faster guess kernel, `--head2` | 58.9 | +40.0% |

- **Concluded.** The token list pays: +6% (the draft head reads 5 to 10% of the output matrix twice a step; the verify pass keeps the whole vocabulary, so the text is the one the engine would write without the list; only the drafts change). It is now a default of `--fast50` when the file exists. Two limits: the list comes from English code, English and Russian docs, so an answer in another language loses drafts (it costs speed, not correctness); and `--draft-list FILE` takes any other list. The faster guess kernel gave +0.2% only: the guess runs while the CPU computes, so it was never on the critical path, the 14.5% of GPU time is idle time that the kernel filled; the rewrite stays (same bits, less GPU busy time, so a little less heat). `--head2` adds 0.7 tok/s (+1.2%): small, and its hit rate is not printed, so it stays opt-in. The next GPU target by this profile is the dense Q4_K matvec (27%), which at 25 us a call is already near the bandwidth of its 1.9 GiB read per step: not rewritten.

**Tool calls of the final set (13 November 2026).** The generated set of `docs/TPS_REPORT.md` E15, against the engine's chat endpoint with nine tools, temperature 0, thinking off, on the dense Q4_K file with 6 experts, prefetch 3, lazy K/V, 64 KiB warm-up, adaptive draft depth, chained drafts, n-gram drafts and the token list: **main set 87 of 89** (the unchanged engine 88, dense Q4_K with 7 experts 88; the two misses are `select` cases, 30 of 32); **hard set 35 of 36** with the scorer that applies the edit and runs the test (the unchanged engine 36, Q4_K with 7 experts 36). A first reading of the hard set gave 26 of 36 and looked like a break of the multi-step chains (`fix_chain` 1 of 10); it was the strict scorer of `tc_hard.py`, which demands that `old_text` be exactly one buggy line and rejects correct edits that quote the whole function. The unchanged engine scores 30 of 36 with it (the same as in November), the final set 26; rescored with the applied-edit scorer (`tc_rescore.py`) the unchanged engine, dense Q4_K with 6 experts and prefetch alone, the lazy K/V cache alone, and the draft switches alone give 36, 35, 36 and 36 of 36, and the whole final set 35 of 36. So the new switches do not hurt the agent tests; the one miss in the hard set is a parallel call that returned one of two calls (as at 6 and 5 experts in November). The HumanEval loss (146 of 164 against 153) therefore stays the cost of dense Q4_K with 6 experts, not of the switches. Lesson for the test scripts: report the applied-edit score next to the strict one, as E15 does.

### C1 and C3 in full form (13 November 2026; code written and compiled, not run with the model yet)

**C1, strict mode on the GPU.** The goal: an expert gives the same bits on the CPU and in a VRAM slot, so the output does not depend on which experts are resident or guessed by the prefetch. What the CPU kernels compute, and what the new GPU kernels repeat:

```
activation x (n_embd floats) -> Q8_K blocks of 256: int8 values and one float scale per block   (the device quantiser already writes the host's bytes)
one row of gate, up or down:
    16 float lanes, all zero
    for every weight block of 256:
        lane += fl(d * activation scale) * (exact integer: sum of weight * activation, times the sub-block scales)     one FMA per lane
    row = the lanes added as _mm512_reduce_add_ps adds them: (i, i+8), then (i, i+4), (i, i+2), (i, i+1)
h = (gate / (1 + exp(-gate))) * up -> Q8_K -> the down rows the same way
```

- The integer part is exact on both sides, so only the float steps need the same order: one multiply for the factor, one FMA per block and lane, the fixed add tree. In `strict_gu_kernel` and `strict_down_kernel` (`src/kernels/cuda/iq_kernels.cu`) a warp is one row, lane `column` (0 to 15) is the CPU lane, and the two threads of a column each take half of its products. The float steps use the `_rn` intrinsics, which the compiler never merges.
- Which weights a lane holds differs by CPU kernel, and the GPU kernel follows the kernel the CPU really runs: an IQ3_S lane (`iq3s_vbmi.cpp`) is the weights 16 L to 16 L + 15 of the block; a 4-bit lane (`iq4xs_vnni.cpp`, also used for gate/up with `--iq3s-4bit`) is, in each half of the block, the weights 4 m to 4 m + 3 and 16 + 4 m to 16 + 4 m + 3 of one sub-block. With `--iq3s-4bit` the slot keeps the IQ3_S rows (the same integers) and the GPU kernel takes the 4-bit lanes.
- SwiGLU was the open point: the CPU kernels call the C library's `exp`, which a GPU cannot be made to repeat bit for bit without copying one library version. So the strict mode has its own `exp` (`include/strata/kernels/strict_math_impl.hpp`: range reduction and a degree-9 polynomial in double, every step an explicit operation), built once for the CPU (`strict_math.cpp`, compiled with `-ffp-contract=off`) and once for the GPU. In strict mode the IQ3_S and the 4-bit gate/up kernels of the CPU call it. Consequence: the text of `--strict` can differ in rare last bits from the text of `--strict-cpu`, which keeps the C library's `exp`; each of the two is the same on every run and for every placement of the experts.
- The hidden row h is quantised on the device by the quantiser that already exists for the activations (`q36act`), and a probe at start compares its bytes with the host quantiser's for both sizes.
- `setup_strict` decides at start. Every hybrid layer must have IQ3_S gate/up and IQ4_XS down, the CPU must run the AVX-512 kernels (`--iq-kernels on`), and both probes must pass. Otherwise the engine says why, empties the slots, turns the prefetch off and computes every expert on the CPU, as `--strict-cpu` does. So the dense Q4_K and Q5_K files keep their dense matrices on the GPU as before (those are not routed experts), and their experts are IQ3_S and IQ4_XS: they qualify.
- Not covered: prompt chunks larger than the hybrid row limit (they take the streamed MMQ path, as before; the launcher's chunk of 512 rows stays under the limit), and a layer whose experts are all in VRAM (no CPU form to agree with).

**C3, two windows layer by layer.** A layer is now two functions: `layer_front` (mixer, router, the experts the GPU computes, the shared expert: all queued on the stream) and `layer_back` (the CPU pool's experts, their upload and merge, the end of the MoE block). `forward_pair` runs two windows with this queue order:

```
front(a, 0), front(b, 0)
for every layer l:
    back(a, l)        host: a's experts on the CPU pool      GPU meanwhile: front(b, l)
    front(a, l + 1)
    back(b, l)        host: b's experts on the CPU pool      GPU meanwhile: a's merge and front(a, l + 1)
    front(b, l + 1)
one pass of the output matrix for the rows of both windows
```

- `run_rows` splits a batch window of two or more sessions into two halves of its groups. The second window uses the same buffers from its first row on (`WinView`: the residual, the experts' input and output rows, the router results, the host rows of the CPU experts, the verify buffers), so the callers and `commit_rows` see one window; only its row tables, their pinned staging and its routing event are separate.
- The order keeps each window's `layer_back` next to its `layer_front` of the next layer, because `moe_end` leaves the quantised input of the next layer in a buffer both windows share.
- The prefetch guesses serve the first window only: the guess slots of a layer hold one window's guess. The routing counts of the second window follow the first one's, so the slot rebalance sees every row.
- It needs hybrid main layers, no graphs, no overlap switches and no integrated-GPU experts; otherwise the window runs as before. The engine prints one line at the first interleaved window.
- Expected: the batch window already shares the GPU pass, so the interleave adds the GPU time of a layer that the CPU waited for, about gpu / (gpu + 2 cpu) of a window; with 0.4 ms of GPU and 1.2 ms of CPU a layer that is about +14% on two sessions. Not measured.

**What the first runs must check** (in this order): (1) `tests/iq_vbmi_bench` style check of `strict_swiglu` against the C library on a sweep of inputs (the largest difference in ulps); (2) `--strict` must write the same text for the same prompt with no resident expert (a VRAM reserve that leaves no slot) and with 21 of them, and with the prefetch on and off; (3) `--parallel 2 --interleave` must write the same text per session as `--parallel 2`, then the total tokens a second of both.

### The middle variant and the search for 50 tok/s with a small loss (13 November 2026)

- **Expected.** Dense Q4_K with 7 experts scored 88 of 89 and 36 of 36 on the tool-call sets and 152 of 164 on HumanEval before the new switches. With them it should come close to the 6-expert set in speed and lose less code quality.
- **Done.** Code prompt, 2200 MHz cap, the switches of the final set (prefetch 3, lazy K/V, warm-up, adaptive and chained drafts, n-gram drafts, the token list). Short runs are 3 requests of 700 tokens with a cool-down; the sustained run is 8 requests of 900 tokens back to back. HumanEval: 164 problems, greedy, thinking off, the engine's own server.
- **Got.**

  | Setting | Short, tok/s | Sustained, tok/s | HumanEval | Mean KLD of the setting's parts (`TPS_REPORT.md`) |
  | --- | --- | --- | --- | --- |
  | unchanged engine, IQ4_XS file, 8 experts | 36.6 | not measured | 153 | 0 (noise floor 0.006) |
  | dense Q5_K, 8 experts, switches | 51.4 | not measured | not measured | not measured for Q5_K |
  | dense Q5_K, 7 experts, switches | 54.9 | not measured | 150 without the switches | 0.025 for 7 experts |
  | dense Q4_K, 8 experts, switches | 55.2 | mean 50.6, median 52.5, lowest 43.1, peak 77 C | not measured (the run was stopped) | 0.0125 for dense Q4_K |
  | dense Q4_K, 7 experts, switches (the middle variant) | 57.3 | mean 54.3, median 55.4, lowest 48.2, peak 77 C | 149 (152 without the switches) | 0.0125 and 0.025 |
  | dense Q4_K, 6 experts, switches (`--fast50` until this record) | 58.2 | mean 51.4 without the token list | 146 | 0.0125 and 0.048 |

- **Concluded.** The middle variant holds 50 tok/s on average in a sustained run (one request of eight fell to 48.2) and loses 4 HumanEval problems against the unchanged engine, 3 fewer than the 6-expert set; the HumanEval noise between runs of one setting is about 3 problems. The token list is what gives the reserve: the 6-expert set without it held 51.4. Zero loss at 50 tok/s is not reached by any setting: every one above the first row uses the dense Q4_K or Q5_K file. The least-loss candidate, dense Q4_K with all 8 experts (only the dense matrices change, KLD 0.0125, twice the noise floor), averages 50.6 tok/s in the sustained run, but three of its eight requests fell below 50 (49.2, 47.9, 43.1): it reaches 50 on average and does not hold it. Its HumanEval was not measured: the background run was stopped with the session. **Decision: `--fast50` now uses 7 experts** (54.3 on average, HumanEval 149); `--fast50 --experts-used 8` is the gentler setting at about 50 on average, `--fast50 --experts-used 6` the fastest one.

### First checks of C1 (13 November 2026)

- **Expected.** The strict GPU kernels give the bits of the CPU kernels.
- **Done.** `tests/strict_expert_parity.cu` (`ninja strict_expert_parity && ./strict_expert_parity`, no model needed): random IQ3_S and IQ4_XS experts of the model's size, 8 tokens, groups of 1 to 11 entries, three activation scales, both lane layouts (IQ3_S and repacked 4-bit). The CPU side runs the pool's order (`native_gu_rows`, `native_quant_h`, `native_down_rows`), the GPU side the device quantiser and the two strict kernels. It also compares the strict SwiGLU with the C library's.
- **Got.** 30 of 30 cases: no hidden value and no output value differs. The strict SwiGLU differs from the C library's in 466 of 1,634,311 values, by at most 3 units in the last place.
- **Concluded.** The kernels are right at the unit level on the first run.

### Engine checks of C1 and C3 (13 November 2026)

- **Expected.** C1: `--strict` writes the same text wherever the experts are computed. C3: `--parallel 2 --interleave` writes the same text per session as `--parallel 2` (in strict mode, where the placement of experts cannot change it) and gives more tokens a second in total.
- **Done.** The engine's own server on the default file (dense Q5_K, 7 experts), one request of 400 tokens at temperature 0. C1: three starts with `--strict`: slots and prefetch 3, slots only, no slots (a VRAM reserve of 3400 MiB). C3: two different requests at once, context 16384, prefetch 3, without and with `--interleave`, once plain and once with `--strict`.
- **Got.** C1: the first start fell back to the CPU for every layer, because the file keeps the down matrices of 4 layers in Q6_K (found with a new log line that names the layer and its formats). The decision is now made per layer: 37 layers use the strict GPU kernels, the 4 others keep every expert on the CPU and get no prefetch guess. After that the three starts wrote the same text (400 tokens) with 18.8%, 7.8% and 0% of the expert entries computed on the GPU. The same text also came from session a of the two-session start. C3: the interleaved windows ran (the engine's line "interleaved batch windows" appeared, no failure), but:

  | Two requests of 400 tokens | Wall time | Total tok/s | Text against the plain window (strict mode) |
  | --- | --- | --- | --- |
  | `--parallel 2` | 14.5 s | 55.0 | the reference |
  | `--parallel 2 --interleave` | 17.9 s | 44.7 | one session the same, the other different |
  | `--parallel 2 --strict` | 14.9 s | 53.6 | the reference |
  | `--parallel 2 --strict --interleave` | 18.3 s | 43.7 | one session the same, the other different |

- **Concluded.** C1 works in the engine: the strict mode keeps the slots and the prefetch and its text does not depend on the placement; it costs about 3% of the two-session speed here. C3 does not pay and is not correct yet. It is 19% slower than the plain batch window: two windows of 2 rows double the kernel launches of the GPU side and the jobs of the CPU pool, the second window has no prefetch, and the plain window already shares one pass. And in strict mode one of the two sessions gets a different text, so a fault remains in one window's path (not found: the review of the shared buffers and of the prefetch slots did not show it). `--interleave` stays off by default and should not be used; the next step, if this is taken up again, is to find which window differs with two sessions of the same prompt.

### Eight experts at a steady 50 tok/s: an analysis (14 November 2026, no runs)

`docs/qwen36-eight-experts-50.html` (English and Russian) asks whether all 8 experts can run at a steady 50 tok/s inside 80 C by software alone. It is an analysis of the figures above, not a measurement.

- **Where 8 experts stand.** Dense Q4_K, default switches, 8 requests in a row: mean 50.6, worst 43.1 tok/s (7 experts: 54.3 and 48.2). The dips are pause episodes of the thermal governor.
- **Heat budget.** The two heat-limited series give about 540 ms of CPU expert time per second (54.3 tok/s x 9.6 ms and 50.6 x 11.0 ms per token, model estimates). Today 8 experts cost about 11.0 ms per token, so about 49 tok/s is steady.
- **The combination that was never run.** The 4-bit CPU kernel (L3, +17% alone, 9.3 ms less a step) was measured only without resident slots, without the token list and on the Q8_0 dense file. Prefetch excludes it, but prefetch adds only about 5 points of expert entries once 21 slots exist (logs: 3 slots 7.8%, prefetch alone 18.8%, slots and prefetch 28.7%). With the 4-bit kernel and slots, 8 experts cost about 9.3 ms per token, less than today's 7-expert default.
- **Projection (estimates).** Dense Q4_K with 8 experts: about 60 tok/s short (57 to 63), 55 to 57 in a row; dense Q5_K about 56 short; the original file about 52 short.
- **Next run.** `bin/strata-serve --experts-used 8 --expert-prefetch 0 --iq3s-4bit --iq3s-4bit-drop` (a launcher dry run shows that the flags combine with the default set); 8 requests of 900 tokens in a row; accept when every request is at or above 50 tok/s with the peak under 80 C. Then HumanEval and the tool-call sets, a pinned form of the repacked copy, and `--cpu-adaptive` against the pauses.
