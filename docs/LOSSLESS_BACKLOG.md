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
| L13 warm guessed experts | prefetcht2 of the guessed experts' first bytes | `--warm-experts KIB` |
| L1 hot experts | launcher part only | `--resident-experts` |
| C1 strict mode | every expert on the CPU: no prefetch, no resident experts | `--strict` |
| C2 speculative sampling | `target_distribution`, `judge_sampled`, a judge hook in `spec_step` | `--spec-sampling` |

Not written: L1's elastic K/V tiers and slot dropping, L7 (guess two layers ahead needs a third ring), L9 (dense GPU kernels), L11 (shared arena for prompt buffers and slots), L14 (two-stage verify head: the cheap bound found for it was too loose to prune rows), C1's GPU-side bit-exactness (the strict flag is the CPU-only form), C3 (two interleaved streams).
