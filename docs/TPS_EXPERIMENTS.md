# Raising the decode speed of Qwen3.6-35B-A3B on a 4 GiB laptop: what was tried and what not to repeat

A ledger of the speed experiments of November 2026 (software and weight changes only; no cooling, overclocking or core pinning).
Every row says what was measured, the number, and whether the idea is worth another run. Details of the engine switches are in
`docs/TUNING.md`; the open work items are in `docs/BACKLOG.md`.

## Setup of the measurements

- Machine: RTX 3050 Ti Laptop (4 GiB), 31 GiB RAM, 4 physical cores, DDR5. Model: `UD-IQ4_XS` with the MTP block grafted from the
  Q4_K_XL file (17.0 GiB); the routed experts (14.66 GiB) are read in place from the page cache and computed on the CPU pool.
- Engine speed: `tools/lever_matrix.py` or `tools/quant_bench.py`, 2.2 GHz CPU cap, 6 pool workers, `STRATA_POOL_SPIN_US=0`,
  `STRATA_IQ_VBMI=1`, MTP 2, the code prompt, 900 tokens, three repeats, the package cooled to 55 C before each request.
  Baseline 35.1 to 35.3 tok/s, drafts accepted 0.635, one step about 65 ms for 2.27 tokens.
- Quality: `llama-perplexity --kl-divergence` on the CPU (`-dev none`) against the served IQ4_XS file, 24 chunks of 512 tokens of
  this project's text (noise floor of the mean KLD about 0.006); HumanEval pass@1 (164 problems, greedy, thinking off) through
  `llama-server` with the same file.

## Verdict

Decode speed can be raised in software. The large levers change the output: fewer routed experts per token, and a lighter dense
part (Q5_K or Q4_K instead of Q8_0). Letting the GPU compute a few guessed experts (`--expert-prefetch`) gives about 10 percent
per step for a small change in the output (a mean KLD of about 0.005). A good compromise among the measured combinations is the
dense part in Q5_K with 7 experts and a prefetch of 3: a step in 52.1 ms against 64.5 (46.2 against 35.2 tok/s) for a mean KLD of
about 0.025; the fastest measured, with the dense part in Q4_K, takes 50.3 ms for about 0.04 (see "Settings that stack"). The decode step is a strict chain GPU, then CPU experts, then GPU, so the GPU is idle for much of it; the
options to change that are listed at the end.

## Fewer experts per token (K, run-time setting `--experts-used`)

| K | engine tok/s | gain | HumanEval | mean KLD | same best token | perplexity |
| --- | --- | --- | --- | --- | --- | --- |
| 8 (file) | 35.1 | - | 151 (92.1 %) | - | - | - |
| 7 | 40.2 | +14.4 % | 152 (92.7 %) | 0.025 | 93.1 % | +1.7 % |
| 6 | 41.9 | +18.7 % | 151 (92.1 %) | 0.060 | 89.3 % | +5.4 % |
| 5 | 45.9 | +30.6 % | 150 (91.5 %) | 0.122 | 85.3 % | +10.8 % |
| 4 | not run | - | 142 (86.6 %) | - | - | - |
| 3 | not run | - | 114 (69.5 %) | - | - | - |

The pass rate holds down to K=5 (one run differs by 1 to 2 problems) and breaks at K=4. Not measured: tool calls and agent runs
at a lower K. Overlap on top of K: 6 experts +23.1 % against +20.9 % alone (4 November).

Variants of the same idea that do not beat the plain setting:

| Idea | Result | Do not repeat unless |
| --- | --- | --- |
| K=4 in one group of 8 layers only (llama.cpp patch) | mean KLD 0.049 / 0.055 / 0.037 / 0.029 / 0.057 for layers 0-7 / 8-15 / 16-23 / 24-31 / 32-39; uniform K=6 in the same run 0.0485 while saving 25 % of the expert work, against 10 % saved by one group | the per-layer sensitivity is wanted for another reason; the later layers are the cheapest to cut, but not enough to beat a uniform K |
| Dynamic top-p over the router probabilities (llama.cpp patch) | p=0.95: 7.92 experts on average, KLD 0.0078; p=0.9: 6.99, 0.0228; p=0.8: 5.86, 0.0604; p=0.7: 4.85, 0.122: about the same KLD as a static K of the same average (K=7 0.025, K=6 0.060), and the number of rows per step would vary | the engine gets a cost model that profits from a variable K |

## Dense weights and other weight changes

| Idea | Result | Status |
| --- | --- | --- |
| Dense part Q8_0 to Q6_K, attention + GDN + shared expert | +0.9 % per step (+1.1 % tok/s), KLD 0.0124 | measured, small |
| Dense part Q8_0 to Q6_K, all groups (with output and embeddings) | +3.2 % per step (+9.7 % tok/s on the code prompt, +2.9 % on a text prompt: part of it is higher draft acceptance), KLD 0.0125 | measured; per unit of KLD K=7 is the better trade (+10.8 % per step at KLD 0.025 against +3.2 % at 0.0125) |
| Dense part Q8_0 to Q5_K / Q4_K (all groups; made with `tools/requant_dense.py --to q5_K` / `--to q4_K`, branch `requant_dense_kquants`) | 12 November, code prompt, MTP, baseline 35.2 tok/s and 64.5 ms a step: Q5_K 42.1 tok/s and 58.8 ms (-9 % per step), Q4_K 42.6 tok/s and 55.5 ms (-14 %); llama.cpp KLD against the IQ4_XS file: Q5_K 0.0216 (perplexity +1.1 %, same best token 93.8 %), Q4_K 0.0428 (+1.7 %, 92.1 %) | measured; the step gain is larger than the Q6_K slope suggested (-3 %) and worth its KLD, see "Settings that stack" |
| Lower-bit expert files, `UD-IQ3_XXS` (13.2 GB) and `UD-Q3_K_S` (15.4 GB) | KLD 0.071 and 0.051, perplexity +2.3 % and +0.9 %, same best token in 89.0 % and 91.3 %. At first the engine refused both files: the down projections are IQ3_XXS and IQ3_S, which were not in `STRATA_D_FMTS` in `src/kernels/cuda/iq_kernels.cu` (the gate/up lists did contain 18, 21 and 22); the branch `iq3_down_formats` adds them. | Speed of the `UD-Q3_K_S` file with the MTP block grafted (`tools/gguf_graft.py --prefix blk.40. --set-u32 qwen35moe.block_count=41 --add-u32 qwen35moe.nextn_predict_layers=1 BASE DONOR OUT`, donor the Q4_K_XL file, as in the README), 11 November: 13.6 / 12.8 tok/s on the code / text prompt against 35.1 / 34.5 for the IQ4_XS file in the same session, drafts accepted 0.62 / 0.57, so 2.6 to 2.7 times slower (the CPU kernels for IQ3_XXS gate/up and IQ3_S down are not the fast ones of IQ3_S / IQ4_XS). The `UD-IQ3_XXS` file (gate/up IQ2_S, down IQ3_XXS) ran at 17.1 / 16.4 tok/s against 35.1 / 34.5 (IQ4_XS again 35.2 / 34.6), drafts accepted 0.69 / 0.62, 2.1 times slower. The byte saving is 26 % and 15 % of the expert bytes, but the CPU kernel speed decides, so these files lose to K=6 (+18.7 % at KLD 0.060) and K=7 (+14.4 % at 0.025) unless IQ3_XXS and IQ3_S get kernels as fast as the IQ4_XS ones |
| Rank-based mixed precision of the experts (Q4 for the top ranks, Q2 for the rest) | judged unattractive on this base file: IQ4_XS is already about 3.4 to 4 bits; see `docs/MIXED_PRECISION_EXPERTS.md` | not implemented for Qwen3.6 |

## Settings that stack (12 November)

Code prompt, MTP 2, 2.2 GHz cap, baseline 34.9 to 35.3 tok/s and 64.4 to 65.1 ms a step (the baseline runs lie between 34.9 and 35.3 tok/s, about 1 percent apart). The KLD is
the mean over 261 teacher-forced positions of the engine's own logits against the CPU-only run of the unchanged file (40 prompt tokens
and 260 forced tokens of this project's text, `strata-q36 --tokens ... --force ... --dump-full-logits`); it is noisy at that size,
the maximum single-position values are 0.2 to 1.1, and the KLDs do not add up exactly.

| Setting | tok/s | ms a step | step against the baseline | mean KLD | same best token |
| --- | --- | --- | --- | --- | --- |
| baseline | 35.2 | 64.5 | - | 0 | - |
| prefetch 3 | 42.1 (drafts accepted 0.72) | 58.0 | -10 % | 0.0052 | 96.9 % |
| 7 experts | about 40 (earlier run) | about 58 | about -10 % | 0.0185 | 93.1 % |
| dense Q5_K | 42.1 (accepted 0.69) | 58.8 | -9 % | 0.0241 | 92.7 % |
| dense Q4_K | 42.6 (accepted 0.67) | 55.5 | -14 % | 0.0369 | 92.0 % |
| 6 experts | 41.8 | 55.3 | -14 % | 0.0376 | 91.6 % |
| 6 experts + prefetch 3 | 44.9 | 51.3 | -20 % | 0.0384 | 90.8 % |
| dense Q4_K + prefetch 3 | 44.6 (accepted 0.59) | 53.4 | -17 % | 0.0355 | 91.6 % |
| dense Q5_K + 7 experts + prefetch 3 | 46.2 (accepted 0.71) | 52.1 | -19 % | 0.0253 | 91.6 % |
| dense Q4_K + 7 experts + prefetch 3 | 48.3 (accepted 0.75) | 50.3 | -22 % | 0.0407 | 91.6 % |

The tok/s of the rows with a high draft acceptance flatter them; the step time is the fair figure. Dense Q5_K with 7 experts and a
prefetch of 3 is 0.8 ms a step slower than 6 experts with a prefetch of 3 and has a KLD about a third lower (0.0253 against
0.0384); dense Q4_K with 7 experts and a prefetch of 3 is the fastest measured. Quality on tasks (tool calls, agent runs) was not measured for any of these.

## Lossless and system-level ideas that gave nothing (or little)

| Idea | Result |
| --- | --- |
| Overlap of GPU and CPU work inside a step (`STRATA_Q36_OVERLAP=all`) | +2.6 % per step, the only lossless gain found |
| Huge pages for the expert reads (microbenchmark, 6 threads, random 1.9 MB blocks) | 52.4 GB/s from mapped file pages, 52.4 from 4 KB anonymous pages, 53.2 from 2 MB pages: nothing |
| Software prefetch distance of the IQ kernels (`STRATA_IQ_PREFETCH`) | 0 / 2048 (default) / 4096 / 8192 bytes: 35.28 / 35.24 / 35.15 / 35.23 tok/s |
| MTP depth | depth 3: 33.0 tok/s (-6.4 %, acceptance 0.523); at K=6 depth 1 / 2 / 3 give 41.2 / 41.9 / 39.3; depth 2 stays the best |
| Spin wait of idle pool workers under sustained heat | default 32.9 / 32.9 against 28.0 / 30.4 with a 100 us threshold |
| Integrated GPU for a share of the experts | -48 to -66 % in every setting (shares 0.15 to 0.35) |
| Next-layer expert prediction for prefetch from disk | 77.5 / 88.3 / 94.5 % of the real picks among 8 / 12 / 20 predicted, no speed gain at a 12 GB budget |
| Cache policy (LFU, pinning, oracle) | LRU is within 2 points of the oracle; pinning is worse |
| Experts in the free VRAM | 75 experts cover 0.8 % of the accesses; 8,458 of 10,240 experts are used within 600 tokens: the routing is almost flat |
| n-gram lookup drafting on generated code | 1.69 tokens per step against 2.68 for MTP |
| n-gram drafting on a rewrite-the-file prompt (llama.cpp, no MTP, 13.1 tok/s without drafts) | `ngram-mod` 34.6 tok/s on the first request (647 of 704 drafts accepted), 20.2 on the second, 15.6 for `ngram-simple`, 14.7 for `ngram-map-k`; in the engine the cost of an extra row probably caps the gain (see B5 in `docs/BACKLOG.md`) |

## Where the time goes

Per token, decode without MTP (`strata-q36 --bench-pp 64 --bench-tg 100` under `nsys profile --trace=cuda`; 218,177 kernels in
17.8 s, about 670 per token): the GPU kernels are busy about 45 % of the time, about 16.7 of 37 ms, which is the 2.1 GiB of dense
weights read at about two thirds of the card's bandwidth. The other 20 ms is the CPU reading the 8 experts of each of the 40 layers
(15.7 MB per layer, about 0.63 GB per token). A layer runs GPU, then CPU experts, then GPU again, so the two never overlap beyond
what the overlap switch hides. With MTP the step carries 3 rows and takes about 65 ms: the dense weights are read once (an estimate of 14 to 17 ms), and
every extra row costs the CPU about 16 to 21 ms (older measurements: one row about 33 ms, three rows about 75 ms; the cycle model in
`docs/TUNING.md` puts an extra token at 62 % of the first one), so the GPU duty is lower still, about a quarter. That figure is an
estimate from these step times, not an `nsys` measurement with MTP.

Host to device copy with pinned memory: 6.1 GB/s, 258 us per 1.5 MB piece (an expert is about 1.5 to 1.9 MB); the CPU computes an expert in
about 52 us (0.42 ms per layer for 8 experts). DRAM read by 6 threads: 52 GB/s at best; the expert path reaches about 24 to 30 GB/s.

Measuring tips: a `serve` run under `nsys` loses its report when the launcher stops the engine, so profile the benchmark mode,
which exits by itself. The engine counters (`CPU path ms`) are printed only at a normal exit.

## Ways to put more load on the GPU (proposals; the PCIe prefetch is implemented and measured)

Reaching 80 to 100 % GPU duty in one decode stream is not possible while the experts are computed on the CPU after the router:
the chain GPU, CPU, GPU is a data dependency. What can change the balance:

| Option | Idea | Estimate |
| --- | --- | --- |
| Two or more streams interleaved layer by layer | while the CPU computes the experts of stream A, the GPU runs the dense part of stream B | without MTP about 1.8 times the aggregate tokens per second and about 84 % GPU duty for two streams; with MTP the CPU share is larger and the gain smaller (about 1.3 times); no gain for one stream |
| Stream part of the experts to the GPU over PCIe, with next-layer prediction | the GPU takes the experts it can receive in time | PCIe adds about 6 GB/s to the CPU's 24 to 30: at best +10 %, with a prediction hit rate of 77 to 88 % less; large change. Written as the opt-in `--expert-prefetch P` (needs `--experts pinned`, see `docs/TUNING.md`) and measured on 11 November: +12.8 percent (P=3) and +7.4 percent (P=2) in tok/s, about +9 to +10 percent per step, 16 to 22 percent of the expert entries computed on the GPU; the first version was -32 percent because its copies kept the copy engine busy when the CPU rows had to go to the card. With 6 experts it adds about 7 percent per step (44.9 against 41.8 tok/s); P=4 does not pay (the engine settles on 3); the changed rounding costs a mean KLD of about 0.006 (97 percent same best token), a quarter of 7 experts' 0.025 |
| Make the CPU phase shorter than the GPU phase | lower K, cheaper CPU kernels, a CPU-friendly repacked layout | K=5 puts the CPU phase near 12 ms against 17 ms on the GPU; then the dense weights dominate and a lower-bit dense part pays |
| Faster dense kernels | the GPU part runs at about 2/3 of the card's bandwidth; 85 to 90 % would save about 4 ms of 37 | +10 %, lossless, lowers the GPU duty but shortens the step |
| Skew the rows of a verify window by one layer | row 2's attention on the GPU while the CPU computes row 1's experts; legal, because rows depend on each other only through the K/V and the recurrent state | the overlap can hide at most the GPU phase (an estimate of 14 to 17 ms of 65), and un-batching the rows costs the CPU about 24 % more for a 3-row window (576 against 463 cycles per block in the cycle model of `docs/TUNING.md`, about 10 ms): a few percent at best |
| Apply the MoE output one layer late (deferral) | removes the dependency | measured earlier on another model at +5 % and lossy; not recommended |

## What was not covered

- Tool-call accuracy and multi-step agent quality at K below 8, with a Q5_K or Q4_K dense part and with the prefetch (HumanEval only
  covers code, and only K was run on it).
- HumanEval or another task benchmark of the stacked settings.
- n-gram drafting inside the engine and `nsys` of a step with MTP.
