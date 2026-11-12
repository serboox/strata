# Decode speed of Qwen3.6-35B-A3B on a 4 GiB laptop: what we expected, what we did, what we got, what we concluded

This is the full record of the speed work of November 2026 on `Qwen3.6-35B-A3B` (the `UD-IQ4_XS` file with the MTP block). It is
written so that a reader who was not in the conversation can see, for every experiment, four things: **Expected** (why it was
tried and what the estimate was), **Done** (the code, the files and the commands), **Got** (the numbers) and **Concluded** (what
to do with it). Two shorter files sit next to it:

- `docs/TPS_EXPERIMENTS.md` is the ledger of numbers (one table per topic, no narrative);
- `docs/TPS_BACKLOG.md` is the list of work items T1 to T9 with a status and a pointer into this file.

The engine switches are documented in `docs/TUNING.md`. The code was written on four branches that were merged
into `main` on 12 November (section 5); every engine switch is off in the engine itself, and the launcher defaults of section 6 are the only
change of default behaviour.

## 1. Goal and limits

The question: can the decode speed of this model on this laptop (RTX 3050 Ti with 4 GiB, 31 GiB RAM, 4 physical cores) be
raised by software and weight changes alone, without better cooling, overclocking or pinning cores? The tested speed is the
engine's server mode with MTP drafts (2 drafts, about 2.27 tokens per step), the code prompt, with the CPU capped at 2.2 GHz
and the package kept below 80 C, so that the runs are comparable and the machine does not throttle.

Answer: **yes**. The unchanged engine does 35.2 tok/s (64.5 ms a step). The best measured safe stack (dense Q5_K, 7 experts,
prefetch 3) does 44 to 46 tok/s (a step of 52 to 53 ms, -19 percent per step; the first run gave 46.2 tok/s and 52.1 ms, two repeats 44.1 and 43.7 tok/s at 53.1 and 53.4 ms, because the drafts were accepted 0.71 times in the first run and 0.67 in the repeats) and the fastest (dense Q4_K) 48.3 tok/s (50.3 ms, -22
percent), at a small and measured loss of output fidelity. Tool-call accuracy and HumanEval stay inside the noise of the tests.

How to read the numbers:

- **Step time (ms)** is the fair speed figure. tok/s also depends on how many drafts happened to be accepted, which differs a
  little between settings, so it flatters some of them.
- **KLD** is the mean KL divergence of the changed setting's next-token distribution from the unchanged one's, on the same
  text. 0 means identical. Rough scale used here: 0.006 is the noise floor of the llama.cpp test, 0.025 is "7 experts", 0.06 is
  "6 experts", 0.12 is "5 experts". "Same best token" is the share of positions where both pick the same top token.
- A difference of 1 to 3 problems or cases in a pass count is noise (section 4, E8).

## 2. Summary table

| # | Experiment | Expected | Got | Verdict |
| --- | --- | --- | --- | --- |
| E1 | Fewer experts per token (K) | the CPU expert work is about 52 of 75 ms, so every expert removed shortens the step | K=7 +14 %, K=6 +19 %, K=5 +31 % tok/s; HumanEval flat down to K=5, breaks at K=4 | the biggest lever; costs fidelity |
| E2 | Per-layer K, dynamic top-p, mixed-precision experts | smarter than a uniform K | no better than a uniform K at the same average | closed |
| E3 | Dense part Q8_0 to Q6_K | 23 % fewer dense bytes per step | +1 % to +3 % per step | not worth it |
| E4 | Dense part to Q5_K and Q4_K | more savings than the Q6_K slope | -9 % and -14 % per step | worth it, stacks |
| E5 | Lower-bit expert files (IQ3_XXS, Q3_K_S) | fewer bytes for the CPU to read | 2.1 to 2.7 times slower: no fast CPU kernels | closed until kernels exist |
| E6 | Prefetch of guessed experts to the GPU over PCIe | the GPU idles during the CPU phase; PCIe adds 6 GB/s to the CPU's 24 to 30 | about -10 % per step, KLD 0.005 | worth it, opt-in |
| E7 | Settings that stack | the gains should add | -19 % to -22 % per step | best measured |
| E8 | HumanEval of the stacks (T9) | no visible loss | 150 to 153 of 164, differences inside noise | passes |
| E9 | Long generation (T2) | no drift in speed, heat or memory | the stack holds +33 % under sustained heat; no leak | passes |
| E10 | Long context, 16K and 32K (T3) | the gain might shrink | +25 % at 16K and +12 % at 32K for the stack | holds |
| E11 | Resident VRAM slots with prefetch (T5) | resident experts help at a shorter context | 44.8 tok/s at 16K context with 300 MiB reserve | works, costs VRAM margin |
| E12 | MTP depth on the faster step (T7) | a shorter step might favour depth 3 | depth 1 to 3: 44.6 / 43.7 / 42.4 tok/s, spread about 2 | depth 2 stays |
| E13 | n-gram drafting statistics (T8) | maybe a cheap second drafter | at most +2.4 % at today's row cost | not worth building |
| E14 | Lossless and system-level ideas | small free gains | only overlap, +2.6 % | nothing to do |
| E15 | Tool-call accuracy of the stacks (T1) | stacks could hurt exact arguments | all stacks 88 to 89 of 89 and 36 of 36; only plain K=5 and K=4 fall | passes |
| E16 | Which dense groups to compress (T4) | groups may differ in cost per millisecond saved | the gain is not additive: three groups together -10.5 % per step, all groups -14.5 %, single groups at most -3.7 % | convert all groups |
| E17 | More aggressive stacks (T6) | HumanEval should hold to K=5, KLD should reach 0.06 to 0.12 | 6 experts 47.4 tok/s (-23 % per step), 5 experts 50.1 (-28 %); KLD 0.048 and 0.091; HumanEval 152 and 150, tool calls 89 and 88 of 89 | works; the KLD is the real price |
| E18 | The maximum stack | step arithmetic: 52 to 54 tok/s for dense Q4_K + 5 experts + prefetch 3 | 52.1 tok/s (-32 % per step), 55.4 with resident slots at a 16K context; KLD 0.083 | maximum profile, not the default |

## 3. How the numbers were measured

| What | Tool | Fixed conditions |
| --- | --- | --- |
| Engine speed | `tools/quant_bench.py` (a server run through the launcher) and `tools/lever_matrix.py` | 2.2 GHz CPU cap, 6 pool workers, `STRATA_POOL_SPIN_US=0`, `STRATA_IQ_VBMI=1`, MTP 2, code prompt (and a text prompt where noted), 900 tokens, 3 repeats after 2 warm-ups, package cooled to 55 C before each request, abort on 80 C or when other programs stall on memory; the baseline is run before and after each group of settings and sits within 1 percent |
| Fidelity of the llama.cpp path | `llama-perplexity --kl-divergence` on the CPU against the served IQ4_XS file | 24 chunks of 512 tokens of this project's text; noise floor of the mean KLD about 0.006 |
| Fidelity of the engine path | `strata-q36 --tokens ... --force ... --dump-full-logits` (teacher forcing: 40 prompt tokens and 260 forced tokens of this project's text), then the KLD of the logits against a CPU-only run of the unchanged file (mapped experts, no prefetch) | one decode row at a time; two CPU-only runs are identical (KLD 0); the value is noisy at 261 positions and KLDs do not add up exactly |
| Code quality | HumanEval pass@1, 164 problems, greedy, thinking off, the generated programs run in a `bwrap` sandbox without network | through `llama-server` for the K ladder (E1) and through the engine's own server for the stacks (E8) |
| Tool-call accuracy | a generated set of cases against the engine's chat endpoint with a `tools` list (section 4, E15) | `temperature 0`, thinking off, one request per case, scores computed from the returned `tool_calls` |
| Long runs | six requests of 2250 tokens in a row, the engine's RSS, the VRAM and the package temperature sampled | no pause between requests |
| Where the time goes | `nsys profile --trace=cuda` on the engine's benchmark mode (a server run under `nsys` loses its report) | 218,177 kernels in 17.8 s |

Process hygiene during all runs: one engine at a time under a systemd scope, `tools/cpucap` set to 2200 MHz and reset after
each run, runs waited for a load average below 1.3, and a run that was disturbed by another job was repeated rather than used.
The measurement scripts (chains that run the settings one after another) and the raw results are kept in
`docs_private/experiments/tps_goal/scripts` and `.../results`; they contain absolute paths of this machine, so they are not
part of the public tree.

## 4. The experiments

### E1. Fewer experts per token (K)

- **Expected.** The routed experts are about 52 of the 75 ms of a decode step (the rest is the GPU part), so a smaller number of
  experts per token should shorten the step about in proportion, at the price of quality, because the model was trained with 8.
- **Done.** The run-time setting `--experts-used K` / `STRATA_Q36_EXPERTS_USED=K` (engine, already on `main`) was measured with
  `tools/lever_matrix.py` (speed, 4 and 7 November), with llama.cpp's KLD test and with HumanEval through `llama-server` with
  `--override-kv qwen35moe.expert_used_count=int:K` (same selection rule as the engine: top K by probability, weights renormalised).
- **Got.**

  | K | engine tok/s | gain | HumanEval | mean KLD | same best token | perplexity |
  | --- | --- | --- | --- | --- | --- | --- |
  | 8 (file) | 35.1 | - | 151 (92.1 %) | - | - | - |
  | 7 | 40.2 | +14.4 % | 152 (92.7 %) | 0.025 | 93.1 % | +1.7 % |
  | 6 | 41.9 | +18.7 % | 151 (92.1 %) | 0.060 | 89.3 % | +5.4 % |
  | 5 | 45.9 | +30.6 % | 150 (91.5 %) | 0.122 | 85.3 % | +10.8 % |
  | 4 | not run | - | 142 (86.6 %) | - | - | - |
  | 3 | not run | - | 114 (69.5 %) | - | - | - |

- **Concluded.** The largest single lever. The HumanEval pass rate holds down to K=5 and breaks at K=4; the tool-call test (E15)
  puts the floor of a plain lower K at 6 or 7 for agent use. K=7 is the gentle setting (KLD 0.025); K=6 and K=5 are for a user who
  accepts visible changes.

### E2. Variants of K that did not beat the uniform K (closed)

- **Expected.** Some layers might tolerate a lower K than others, and a router-probability threshold (top-p) might spend experts
  only where they matter.
- **Done.** Patches in a local llama.cpp clone (not in this repository): K=4 in one group of 8 layers at a time; a dynamic top-p
  over the router probabilities.
- **Got.** K=4 in one group: mean KLD 0.049 / 0.055 / 0.037 / 0.029 / 0.057 for layers 0-7 / 8-15 / 16-23 / 24-31 / 32-39, while
  a uniform K=6 in the same run cost 0.0485 and saves 25 % of the expert work against 10 % for one group. Top-p 0.95: 7.92
  experts on average, KLD 0.0078; 0.9: 6.99, 0.0228; 0.8: 5.86, 0.0604; 0.7: 4.85, 0.122, about the KLD of a static K with the same
  average, and the work per step would vary. The rank-based mixed precision of the experts (Q4 for the top ranks, Q2 for the rest)
  was judged unattractive on this base file (IQ4_XS is already about 3.4 to 4 bits, see `docs/MIXED_PRECISION_EXPERTS.md`) and not
  implemented.
- **Concluded.** Closed. The later layers are the cheapest to cut, but not enough to beat a uniform K.

### E3. Dense part Q8_0 to Q6_K

- **Expected.** The dense matrices (attention, recurrent blocks, shared expert, MTP projection, embeddings) are Q8_0 in video
  memory and are read on every step. Q6_K has 23 percent fewer bytes, so maybe -5 percent of the step.
- **Done.** `tools/requant_dense.py` (the tool existed on `main`) made a Q6_K variant of the file. Q6_K already worked on all
  GPU paths; what it loses are the Q8_0-only fast paths.
- **Got.** Attention, recurrent blocks and shared expert: +0.9 % per step, KLD 0.0124. All groups: +3.2 % per step (+9.7 % tok/s on
  the code prompt, mostly a higher draft acceptance, +2.9 % on a text prompt), KLD 0.0125.
- **Concluded.** Not worth it: per unit of KLD, K=7 is the better trade (+10.8 % per step at KLD 0.025 against +3.2 % at 0.0125).

### E4. Dense part Q8_0 to Q5_K and Q4_K

- **Expected.** The Q6_K step gain was disappointing, but the K-quant kernels are cheaper per byte than Q8_0's, so Q5_K and Q4_K
  were expected to give more than the byte count alone suggests.
- **Done.** `tools/requant_dense.py --to q5_K` and `--to q4_K` (branch `requant_dense_kquants`): a new `Target` table with the
  type id, block size and a tolerance against ggml's reference quantiser (0.05 / 0.08 / 0.16 for Q6_K / Q5_K / Q4_K in the
  self-test), and `quantize_to` / `dequantize_from` that call ggml's `quantize_row_*_ref`. The default Q6_K path is unchanged.
  The engine needed no change: Q5_K and Q4_K dense matrices already ran through the generic GPU paths. Files made: dense Q5_K
  16.3 GiB and dense Q4_K 16.1 GiB (with the MTP block, all groups; `output.weight` is Q6_K already and is kept).
- **Got.** Code prompt, MTP, baseline 35.2 tok/s and 64.5 ms a step: Q5_K 42.1 tok/s and 58.8 ms (-9 % per step), Q4_K 42.6 tok/s
  and 55.5 ms (-14 %). KLD against the IQ4_XS file in llama.cpp: Q5_K 0.0216 (perplexity +1.1 %, same best token 93.8 %), Q4_K 0.0428
  (+1.7 %, 92.1 %); in the engine's teacher-forced test 0.0241 and 0.0369.
- **Concluded.** The step gain is three to four times the Q6_K slope and worth its KLD. They stack with K and the prefetch (E7).

### E5. Lower-bit expert files

- **Expected.** The CPU phase reads the experts from RAM, so smaller expert files (`UD-IQ3_XXS` 13.2 GB, `UD-Q3_K_S` 15.4 GB
  against 17.7 GB for IQ4_XS) should shorten it. Quality, measured first: KLD 0.071 and 0.051, perplexity +2.3 % and +0.9 %, same
  best token in 89.0 % and 91.3 %.
- **Done.** At first the engine refused both files: their down projections are IQ3_XXS and IQ3_S, which were not in `STRATA_D_FMTS`
  in `src/kernels/cuda/iq_kernels.cu` (the gate/up lists already had 18, 21 and 22). Branch `iq3_down_formats` adds X(18) and X(21)
  to the list (Q6_K and non-Q6_K variants). The MTP block was grafted into each file with `tools/gguf_graft.py` (donor: the Q4_K_XL
  file) so the speed test could use drafts.
- **Got.** `UD-Q3_K_S`: 13.6 / 12.8 tok/s on the code / text prompt against 35.1 / 34.5 for IQ4_XS in the same session (2.6 to 2.7
  times slower, drafts accepted 0.62 / 0.57). `UD-IQ3_XXS` (gate/up IQ2_S, down IQ3_XXS): 17.1 / 16.4 against 35.1 / 34.5 (2.1 times
  slower, 0.69 / 0.62).
- **Concluded.** Closed. The byte saving (26 % and 15 % of the expert bytes) is lost because the CPU kernels for IQ3_XXS, IQ2_S and
  IQ3_S are not the fast ones of IQ3_S and IQ4_XS; these files lose to K=6 and K=7 unless those formats get equally fast kernels.
  The grafted files were deleted.

### E6. Prefetch of guessed experts to the GPU over PCIe (`--expert-prefetch P`)

- **Expected.** A layer is a chain GPU, then CPU experts, then GPU, so the GPU is idle for about three quarters of an MTP step
  and the PCIe link is unused. A copy of 6.1 GB/s adds roughly a fifth of the CPU's 24 to 30 GB/s of expert bandwidth: an upper
  bound of about +10 percent, less the share of wrong guesses (77 to 88 percent of the real picks were among 8 to 20 predicted).
  The design was chosen by the owner as "staged GPU share": first the VRAM accounting, then prediction and copy.
- **Done.** Branch `gpu_expert_share_stage1`, mainly `src/qwen35/q36_engine.cu` (about +300 lines): after layer i's router result is
  queued, the residual stream is normalised with layer i+1's MoE norm and multiplied with its router (`k_pf_logits`, `k_pf_select`);
  the best P experts are the guess; the guessed expert blobs are copied from the page-locked arena into a ring (or into guess slots
  after the resident slots) on a separate copy stream, queued after the CPU rows' upload and waiting for an event behind it; the GPU
  computes the entries whose expert is in the ring and the CPU pool gets only the rest. The number of guesses follows the copy
  speed: a layer is "late" when the copy engine is still busy, and when more than a sixteenth of 64 layers were late the engine
  uses one guess less (three calm stretches raise it again). Needs `--experts pinned` (a copy from mapped pages blocks the host for
  about 266 us per 1.5 MB and `cudaHostRegister` on them is refused), no overlap switch, no iGPU share. Also added: the VRAM
  accounting line in the log, `--vram-reserve MiB`, the launcher flags `--expert-prefetch` and `--vram-reserve` (`tools/strata_launch.py`),
  lever 6 in `tools/lever_matrix.py` with `tools/levers.d/expert_prefetch.json` and the updated test.
  Tests: the lever matrix runs (code and text prompts, baseline before and after), the benchmark mode, the teacher-forced KLD, and
  a review pass by an independent reviewer before each commit.
- **Got.** Server mode, MTP, real text, baseline 65.0 ms a step and 34.7 tok/s: P=3 gave 39.1 tok/s (+12.8 %) and 58.6 / 59.4 ms a
  step on the code / text prompt (-9.8 % and -8.3 %), P=2 +7.4 %; 16 to 22 percent of the expert entries came from the ring; 0.4
  percent late layers. P=4 does not pay (the engine settles on 3 of 4). With 6 experts it adds 7 % per step (44.9 against 41.8 tok/s).
  The changed rounding (the GPU quantises activations in blocks of 32 as q8_1, the CPU in blocks of 256 as Q8_K) costs a mean KLD of
  0.0057 for P=3 (same best token 97.3 %) and 0.0062 for P=2; the text departs from the baseline's after 360 to 1050 characters when a
  close token flips, and stays coherent. In the benchmark mode with synthetic tokens P=2 gave +7.5 % and P=3 +4 %, with 97 percent
  right guesses, which real text does not give.
  The first version was **-32 percent**: the guess kernels were queued before the router, so its result came late, and the copies
  kept the copy engine busy while the CPU rows had to go to the card. Moving the guess after the router, queueing the copies after
  the rows behind an event, and checking lateness on the copy engine fixed it. Independent review also found and fixed: an explicit
  `--expert-prefetch 0` overridden by the environment, counters not reset after warm-up, resident slots lost when the arena was
  not pinned, an explicit `--vram-reserve` shrunk by the nearly-fit branch, and lateness not observed when no entry missed.
- **Concluded.** About -10 percent per step for a KLD of 0.005, which is the cheapest gain in KLD terms (7 experts cost 0.025 for
  about the same step gain). Opt-in, because it needs 14.66 GiB of page-locked RAM and about a minute at start. Not done: a guess
  two layers ahead, a share for the MTP block, a guess from more than the current residual stream.

### E7. Settings that stack

- **Expected.** The three levers act on different parts of the step (CPU work, GPU dense reads, idle GPU), so the gains should add.
- **Done.** Engine runs of each combination on the code prompt, MTP 2, plus the teacher-forced KLD of each.
- **Got.** Baseline 34.9 to 35.3 tok/s and 64.4 to 65.1 ms a step.

  | Setting | tok/s | ms a step | step against the baseline | mean KLD | same best token |
  | --- | --- | --- | --- | --- | --- |
  | baseline | 35.2 | 64.5 | - | 0 | - |
  | prefetch 3 | 42.1 (accepted 0.72) | 58.0 | -10 % | 0.0052 | 96.9 % |
  | 7 experts | about 40 | about 58 | about -10 % | 0.0185 | 93.1 % |
  | dense Q5_K | 42.1 (accepted 0.69) | 58.8 | -9 % | 0.0241 | 92.7 % |
  | dense Q4_K | 42.6 (accepted 0.67) | 55.5 | -14 % | 0.0369 | 92.0 % |
  | 6 experts | 41.8 | 55.3 | -14 % | 0.0376 | 91.6 % |
  | 6 experts + prefetch 3 | 44.9 | 51.3 | -20 % | 0.0384 | 90.8 % |
  | dense Q4_K + prefetch 3 | 44.6 (accepted 0.59) | 53.4 | -17 % | 0.0355 | 91.6 % |
  | dense Q5_K + 7 experts + prefetch 3 | 46.2 (accepted 0.71) | 52.1 | -19 % | 0.0253 | 91.6 % |
  | dense Q4_K + 7 experts + prefetch 3 | 48.3 (accepted 0.75) | 50.3 | -22 % | 0.0407 | 91.6 % |

  Both prompts, final runs (baseline 34.4 and 34.6 tok/s on the code prompt, 34.2 and 34.3 on the text prompt): dense Q5_K with
  prefetch 3 ran 42.5 and 44.0 tok/s on the code prompt and 36.6 and 40.1 on the text prompt; dense Q5_K with 7 experts and prefetch
  3 ran 43.7 on the code prompt and 41.8 on the text prompt. 40 tok/s was passed on both prompts with the 7-expert stacks.
- **Concluded.** The gains do add (about -19 to -22 percent per step). Dense Q5_K + 7 experts + prefetch 3 is the recommended
  compromise (KLD 0.025, a third lower than 6 experts with prefetch at 0.8 ms a step more); dense Q4_K + 7 experts + prefetch 3 is the
  fastest of the stacks of this table (the 5- and 6-expert stacks of E17 and E18 are faster still, at a higher KLD).

### E8. HumanEval of the stacks and the noise of the comparison (T9)

- **Expected.** HumanEval would show a drop if the stacks hurt code writing; a difference of one to three problems cannot be
  told from noise by the counts alone, and a run of one setting is deterministic, so repeating it adds nothing.
- **Done.** The engine's own server, 164 problems, greedy, thinking off, for four settings; then a problem-by-problem comparison
  with an exact two-sided sign test.
- **Got.** Baseline 153, prefetch 3 alone 153, dense Q5_K + 7 experts + prefetch 3 150, dense Q4_K + 7 experts + prefetch 3 152.
  Against the baseline the Q5_K stack lost 5 problems and gained 2 (p 0.45), the Q4_K stack lost 4 and gained 3 (p 1.0), prefetch
  alone lost 2 and gained 2 (p 1.0). Problems 83 and 115 fail in both stacks. (The llama.cpp ladder in E1 gives 151 for the same file
  because it is another runner.)
- **Concluded.** No measurable loss on code writing for any stack.

### E9. Stability under a long generation (T2)

- **Expected.** The prefetch and the lighter dense part must not drift in speed, heat or memory over thousands of tokens; the
  copy engine and page-locked memory are new moving parts.
- **Done.** Six requests of 2250 tokens in a row without a pause, code prompt, 2.2 GHz cap, 80 C limit, RSS, VRAM and package
  temperature sampled; baseline against the stack (dense Q5_K, 7 experts, prefetch 3).
- **Got.** The baseline sags from 34.7 to 28.1-31.8 tok/s once the package reaches 79 C (average 31.6). The stack goes 47.1 / 47.8 /
  41.0 / 32.2 / 37.9 / 45.6 (average 41.9, +33 percent; worst request 32.2 against 28.1); its package peaked at 82 C against 79 C.
  RSS grows only while loading (baseline 6.7 to 17.7 GiB, stack 13 to 23.6 GiB with the page-locked experts) and is flat afterwards;
  VRAM flat.
- **Concluded.** No leak and no drift beyond the thermal one. The stack keeps its lead under sustained heat but runs hotter (82 C
  against 79 C at the same 2.2 GHz cap), probably because the CPU and the GPU are busy a larger share of the time.

### E10. Prefetch at 16K and 32K context (T3)

- **Expected.** Attention gets dearer and the K/V cache takes VRAM at a long context, so the gain might shrink.
- **Done.** A prompt of 16,384 and of 32,768 tokens through the server, 300 tokens decoded, thinking off, decode speed read from
  the session line; baseline against prefetch 3 against the stack.
- **Got.** Decode tok/s at 16K / 32K: baseline 30.0 / 29.8, prefetch 3 32.2 / 31.6 (a first 32K run gave 21.9 and was not
  repeated), the stack 37.4 / 33.5 (+25 and +12 percent). Prefill tok/s: baseline 118 / 103, prefetch 124 / 108, stack 145 / 117.
  21 to 22 percent of the expert entries came from the ring at both lengths.
- **Concluded.** The prefetch gain holds at long context (+6 to +7 % alone); the whole stack's gain shrinks to +12 % at 32K as attention
  takes a larger share.

### E11. Prefetch with resident VRAM slots (T5)

- **Expected.** At a shorter context some experts can live in VRAM; the code path for resident slots together with guess slots
  existed and had never been run.
- **Done.** `--max-context 16384` with `--vram-reserve`, baseline against prefetch 3, with the log's slot lines read.
- **Got.** The default reserve leaves 1 resident expert per layer: 36.2 tok/s, with prefetch 3 40.4. `--vram-reserve 300` leaves 9
  resident experts per layer: 43.0 tok/s (step 55.7 ms against 66.9); reserve 300 with prefetch 3 (6 resident, 3 guess slots)
  44.8 tok/s and 53.3 ms.
- **Concluded.** It works and gains a little more, but the price is a context of a quarter the size and a VRAM margin of 300 MiB.
  Useful only for short-context use.

### E12. MTP depth on the faster step (T7)

- **Expected.** Depth 3 did not pay at 65 ms a step; a shorter step might change that.
- **Done.** `--mtp 1, 2, 3` on the dense Q5_K stack, code prompt.
- **Got.** Depth 1 44.6 tok/s (0.815 accepted per draft), depth 2 43.7 (0.666), depth 3 42.4 (0.599). The spread inside one run is
  about 2 tok/s.
- **Concluded.** Depth 2 stays the default; the differences are at the noise.

### E13. n-gram drafting statistics (T8)

- **Expected.** An n-gram draft (copy the continuation of an earlier identical stretch) could add tokens per step for edits and
  tool arguments, which repeat text; whether it pays depends on the cost of the extra verify row.
- **Done.** `b5_stats.py` over 600,000 generated tokens of three recorded agent sessions (a proxy: the texts are another model's
  edits and tool arguments), with the row-cost model of `docs/TUNING.md`.
- **Got.** A draft from the context where the last 4 to 12 tokens match, with an extra row costing 21 ms as measured, gains at
  best 2.4 percent (match 12, up to 4 tokens, 28.7 percent of the tokens by n-gram, 3.45 accepted per n-gram step); with an extra
  row at 12 ms it would gain 14 percent (match 6, up to 4 tokens).
- **Concluded.** Not worth building at today's row cost; look again if rows get cheaper (item B5 in `docs/BACKLOG.md`).

### E14. Lossless and system-level ideas

| Idea | Expected | Got |
| --- | --- | --- |
| Overlap of GPU and CPU work inside a step (`STRATA_Q36_OVERLAP=all`) | hide part of the GPU phase | +2.6 % per step, the only lossless gain found |
| Huge pages for the expert reads | fewer TLB misses | 52.4 GB/s from mapped file pages, 52.4 from 4 KB anonymous pages, 53.2 from 2 MB pages: nothing |
| Software prefetch distance of the IQ kernels (`STRATA_IQ_PREFETCH`) | better latency hiding | 0 / 2048 (default) / 4096 / 8192 bytes: 35.28 / 35.24 / 35.15 / 35.23 tok/s |
| MTP depth 3 (before the faster step) | more drafts | 33.0 tok/s (-6.4 %, acceptance 0.523); at K=6 depth 1 / 2 / 3 give 41.2 / 41.9 / 39.3 |
| Spin wait of idle pool workers under sustained heat | less wake-up latency | default 32.9 / 32.9 against 28.0 / 30.4 with a 100 us threshold: default stays |
| Integrated GPU for a share of the experts | extra compute | -48 to -66 % in every setting (shares 0.15 to 0.35) |
| Next-layer expert prediction for prefetch from disk | fewer cold reads | 77.5 / 88.3 / 94.5 % of the real picks among 8 / 12 / 20 predicted, no speed gain at a 12 GB budget |
| Cache policy (LFU, pinning, oracle) | fewer misses | LRU within 2 points of the oracle; pinning is worse |
| Experts kept in free VRAM | cover the hot experts | 75 experts cover 0.8 % of the accesses; 8,458 of 10,240 experts are used within 600 tokens: the routing is almost flat |
| n-gram lookup drafting on generated code | cheap extra drafts | 1.69 tokens per step against 2.68 for MTP |
| n-gram drafting on a rewrite-the-file prompt (llama.cpp, no MTP, 13.1 tok/s without drafts) | big gain on copied text | `ngram-mod` 34.6 tok/s on the first request, 20.2 on the second, 15.6 for `ngram-simple`; in the engine the cost of an extra row caps the gain (E13) |

Conclusion for the group: with the experts on the CPU there is no lossless gain left except the overlap switch. Where the time goes
(measured with `nsys` in the benchmark mode, no MTP): the GPU kernels are busy about 45 percent of a 37 ms token, about 16.7 ms,
which is the 2.1 GiB of dense weights read at about two thirds of the card's bandwidth; the other 20 ms is the CPU reading 8
experts in each of 40 layers (about 0.63 GB per token). With MTP a step carries 3 rows and takes about 65 ms, and every extra row
costs the CPU 16 to 21 ms (an estimate from the step times, not an `nsys` run with MTP). A host-to-device copy with pinned memory
runs at 6.1 GB/s (258 us per 1.5 MB expert), the CPU computes an expert in about 52 us, DRAM read by 6 threads peaks at 52 GB/s and the
expert path reaches 24 to 30 GB/s.

### E15. Tool-call accuracy of the stacks (T1)

- **Expected.** HumanEval sees only code writing, but the engine is used for agents. Fewer experts and a coarser dense part might
  show first in exact tool names and arguments, in long chains and in the decision not to call a tool at all. K=4 and K=5 were
  expected to degrade; the stacks (K=6 or 7) were the open question.
- **Done.** A generated, seeded (20261112) set against the engine's `/v1/chat/completions` with nine tools (`read_file`,
  `write_file`, `edit_file`, `list_dir`, `search_code`, `run_command`, `get_weather`, `create_event`, `send_email`), `tool_choice auto`,
  `temperature 0`, thinking off. A call counts only if the tool name and every required argument match exactly (strings compared
  case-insensitively, `body_contains` for email text). The runner starts the engine for each setting through the same launcher
  as the speed tests, and one setting takes about three minutes.
  - **Main set, 89 cases:** 32 `select` (the right tool and path or pattern from a plain request), 10 `edit` (replace text, create a
    file), 12 `precision` (calendar date and time in two spellings, e-mail address and subject), 10 `no_tool` (a question that must be
    answered without a call), 20 `chain` (a tool result is given and the next call must be right, e.g. list a folder then open the
    named file; run tests then open the failing test), 5 `chain_answer` (use a weather result in a sentence, no call).
  - **Hard set, 36 cases** (added because the main set saturates): 10 `code` (write a whole Python function into a file with
    `write_file`; the returned file is run in the sandbox against assertions), 8 `parallel` (two reads or two weather lookups in one
    answer), 10 `fix_chain` (run tests, read the file, then edit it: the edit is applied to the file text and the result is run
    against a test), 8 `extract` (take a value out of a 14-field JSON blob and write it into a config). The first scorer demanded that
    `old_text` be exactly the one buggy line, which rejected correct edits that quoted the whole function; it was replaced by applying the
    edit and running the test, and a `read_file` of the target path before the edit counts as correct. Both scorers are kept (strict and
    applied) in the results.
- **Got.**

  | Setting | main set (of 89) | hard set (of 36) |
  | --- | --- | --- |
  | baseline | 88 | 36 |
  | prefetch 3 | 88 | 36 |
  | dense Q5_K + 7 experts + prefetch 3 | 89 | 36 |
  | dense Q4_K + 7 experts + prefetch 3 | 88 | 36 |
  | dense Q5_K + 6 experts + prefetch 3 | 89 | 36 |
  | dense Q5_K + 5 experts + prefetch 3 | 88 | 36 |
  | 7 experts | 88 | 36 |
  | 6 experts | 89 | 35 |
  | 5 experts | 85 | 35 |
  | 4 experts | 78 | 23 |

  The one miss shared by seven of the ten settings (case 14) is a `search_code` call with path `.` where the request named
  the folder `web`: an ambiguous request, not a degradation. At 5 experts three cases are added to the shared one: one more path given as `.`, one edit
  that began with a read and one with no call; at 4 experts 8 of the 11 misses are no call at all (select, edit and precision), which is
  the failure that matters for an agent. On the hard set the misses at 6 and 5 experts are one parallel call that came back with
  one of two calls and one function that fails its assertions; at 4 experts the hard set falls to 23 of 36 (3 `write_file` cases with
  no call or no path, 3 parallel cases, 6 fix chains that read the file again instead of editing it, one extract with no value).
- **Concluded.** None of the stacks (prefetch, dense Q5_K or Q4_K, 5 to 7 experts together with them) lowers tool-call accuracy
  beyond a single case, which is inside the noise of 89 cases. Only a plain K of 5 starts to show it and K=4 clearly breaks it, in
  line with HumanEval. The test is generated and small: a difference of 1 to 3 cases is not a finding, and it cannot replace a
  real agent session.

### E16. Which dense groups to compress (T4)

- **Expected.** The dense part has three groups (attention in the 10 full-attention layers and the MTP block, the recurrent blocks of
  the 30 other layers, the shared expert) that differ in size (Q8_0 bytes: recurrent 1.07 GB, attention 0.32 GB, shared expert
  0.14 GB) and perhaps in how much quality a Q4_K copy costs per millisecond saved. If one group carries most of the KLD, leaving
  it at Q8_0 would give most of the speed for less loss.
- **Done.** One Q4_K file per group (`tools/requant_dense.py --groups ssm|attn|shexp --to q4_K`, made from the MTP file), each with a
  speed run (code prompt, 3 repeats) and the engine's teacher-forced KLD; each file was deleted after its runs.
- **Got.** The work was done in two steps. First one file per group (ssm, attn, shexp). Their step gains added up to under 5 percent
  against the 14 percent of the whole-dense Q4_K file of E4, so a second step made three more files: the three groups together, the
  embeddings alone and the MTP projection alone (`--groups embd`, `--groups mtp`).

  | Q4_K for | Q8_0 bytes | tok/s | drafts accepted | step, ms | step saved | mean KLD | same best token |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | nothing (baseline) | - | 34.92 | 0.635 | 65.0 | - | - | - |
  | recurrent blocks (ssm) | 1.07 GB | 38.18 | 0.695 | 62.6 | 2.4 ms (-3.7 %) | 0.0273 | 92.3 % |
  | attention (attn) | 0.32 GB | 38.19 | 0.738 | 64.8 | 0.2 ms | 0.0161 | 92.3 % |
  | shared expert (shexp) | 0.14 GB | 35.48 | 0.647 | 64.7 | 0.3 ms | 0.0087 | 95.8 % |
  | embeddings (embd) | 0.54 GB | 37.60 | 0.710 | 64.4 | 0.6 ms (-1 %) | 0.0086 | 93.5 % |
  | MTP projection (mtp) | 0.009 GB | 35.13 | 0.629 | 64.3 | 0.7 ms (inside the noise) | 0 (it acts on the drafts only) | 100 % |
  | ssm + attn + shexp | 1.53 GB | 42.82 | 0.747 | 58.2 | 6.8 ms (-10.5 %) | 0.0362 | 91.6 % |
  | all groups (the d4 file of E4) | 2.07 GB | 42.58 | 0.684 | 55.6 | 9.4 ms (-14.5 %) | 0.0369 | 92.0 % |

  The step is derived as (1 + 2 x accepted share) / tok/s. The second baseline of this chain was aborted by the bench's guard (other
  programs stalling on memory), so the two baselines of the T6 chain serve: 34.92 and 34.80 tok/s, 65.0 and 65.2 ms. The KLD of the
  last row is from the stack chain of E7; the others are from the teacher-forced engine test of E7's method.
- **Concluded.** The effects do not add. Converted one by one the three big groups save 2.4 + 0.2 + 0.3 ms, together 6.8 ms; the
  KLD does not add either (0.027 + 0.016 + 0.009 singly, 0.036 together). The embeddings and the MTP projection add about 2.6 ms more
  when the rest is converted (9.4 against 6.8 ms) for almost no KLD, since the d4 file's 0.0369 is hardly above the 0.0362 of the
  three groups. Why a mixed file gains less than the sum is not established; the likely cause, not tested, is that a file with one
  group in Q4_K and the rest in Q8_0 keeps the slower general path for some of the matrices. Practical rule: convert every group (`--groups all`,
  as the d4 and d5 files are made); leaving one out was only measured for the embeddings and the MTP projection, and it costs speed
  without saving KLD. The recurrent blocks carry most of the gain and most of the KLD among the single groups, the shared expert and
  the embeddings the least KLD. Limits: one run per file (a spread of about 1 ms in step time), the draft acceptance moves between 0.63
  and 0.75 and enters the derived step, and only Q4_K was tested per group (the Q5_K default file was not).

### E17. More aggressive stacks (T6)

- **Expected.** Where do quality and speed go when the stack takes 6 or 5 experts together with dense Q5_K and prefetch 3? From
  the K ladder: HumanEval should hold to K=5, KLD should reach 0.06 to 0.12.
- **Done.** Speed (code prompt, 3 repeats, baseline and the 7-expert stack in the same chain), teacher-forced engine KLD, and
  HumanEval on the engine for K=6 and K=5 with dense Q5_K and prefetch 3.
- **Got.** Code prompt, MTP 2, same session as the baseline (34.92 tok/s, 0.635 drafts accepted, step 65.0 ms). The step time is
  derived as (1 + 2 x accepted share) / tok/s, because the bench summary of these runs has no step-time field.

  | Stack (dense Q5_K, prefetch 3) | tok/s | drafts accepted | step, ms | step against the baseline | mean KLD | same best token | HumanEval |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | 7 experts (E7, first run) | 46.2 | 0.71 | 52.1 | -19 % | 0.0253 | 91.6 % | 150 |
  | 7 experts (repeat of this chain) | 44.1 | 0.671 | 53.1 | -18 % | - | - | - |
  | 6 experts | 47.4 | 0.687 | 50.0 | -23 % | 0.0482 | 90.4 % | 152 |
  | 5 experts | 50.1 | 0.671 | 46.8 | -28 % | 0.0907 | 86.6 % | 150 |

  HumanEval against the baseline's 153, problem by problem (the method of E8): the 6-expert stack lost 5 problems (83, 93, 99, 109,
  116) and gained 4 (38, 62, 124, 147), exact two-sided p 1.0; the 5-expert stack lost 6 (32, 64, 83, 113, 116, 156) and gained 3 (38,
  124, 147), p 0.51. Problem 83 and 116 fail in both. Tool-call accuracy of the same stacks (E15): 6 experts 89 of 89 and 36 of
  36, 5 experts 88 of 89 and 36 of 36. Two runs of the chain (the 7-expert stack and a second baseline) were aborted because other
  programs took memory and CPU; they were repeated at the end (section "Repeats" of the ledger).
- **Concluded.** Quality does not fall where the K ladder (E1) suggested it would, probably because the quality tests are coarse:
  HumanEval and the tool-call set do not see a mean KLD of 0.09 (5 experts) in a stack with a Q5_K dense part. Speed keeps rising: 50 tok/s
  and a step 28 percent shorter at 5 experts. The KLD is the honest warning: at 0.09 about one position in seven gets another best
  token (86.6 percent same), which a short test may not catch and a long generation may show as a changed wording (not tested). The 7-expert
  stack stays the recommended default; 6 experts (KLD 0.048) is the fast profile, 5 experts the maximum profile for users who
  accept visibly different output.

### E18. The maximum stack: dense Q4_K, 5 or 6 experts, prefetch 3 (and resident slots at a short context)

- **Expected.** The gains of E7 add up, so the step of dense Q4_K + 5 experts + prefetch 3 should be about 46.8 ms (the Q5_K + 5 experts
  figure of E17) minus the 1.8 ms that Q4_K saves against Q5_K at 7 experts: about 45 ms, that is 52 to 54 tok/s, at a KLD near 0.1. Resident
  slots at a short context added about 10 percent on the plain file (E11), so a few percent more were expected on top.
- **Done.** The chain `tmax_chain.sh`: the dense Q4_K file (E4) with prefetch 3 and 6 or 5 experts, code prompt, MTP 2, 3 repeats; the
  same with 5 experts at `--max-context 16384 --vram-reserve 300` (resident slots next to the guess slots, as in E11); the
  teacher-forced engine KLD of the 6- and 5-expert settings. HumanEval and the tool-call tests were not run for these exact files
  (they were for the Q5_K versions, E15 and E17).
- **Got.** Baselines 34.92 and 34.80 tok/s, 65.0 and 65.2 ms.

  | Setting (dense Q4_K, prefetch 3) | tok/s | drafts accepted | step, ms | step against the baseline | mean KLD | same best token |
  | --- | --- | --- | --- | --- | --- | --- |
  | 7 experts (E7) | 48.3 | 0.75 | 50.3 | -22 % | 0.0407 | 91.6 % |
  | 6 experts | 50.7 | 0.697 | 47.2 | -27 % | 0.0659 | 90.0 % |
  | 5 experts | 52.1 | 0.653 | 44.3 | -32 % | 0.0834 | 86.2 % |
  | 5 experts, context 16,384, VRAM reserve 300 MiB | 55.4 | 0.659 | 41.8 | -36 % | not measured | - |

- **Concluded.** The estimate held: 52.1 tok/s and 44.3 ms for 5 experts (estimated 52 to 54 and 45), and the resident slots add 3.4 tok/s
  (+6 percent) at a context a quarter of the default and a VRAM margin of 300 MiB. 55 tok/s is the highest figure measured on this
  machine. The price is a mean KLD of 0.083 (about one best token in seven differs) and, for the last row, a short context; its KLD was
  not measured, and resident experts are computed on the GPU, whose rounding differs from the CPU's as in E6. This is a maximum profile
  for users who accept visibly different output, not a default; the default is the 7-expert Q5_K stack (section 6).

## 5. Code and tool changes (merged into `main` on 12 November together with this record)

| Branch | Commits | Files | What |
| --- | --- | --- | --- |
| `gpu_expert_share_stage1` | `26ae6a45`, `1ea2cdee`, `b106132f`, `a782efb6`, `12b2c374`, `66395a3e`, `5743903b` | `src/qwen35/q36_engine.cu`, `tools/strata_launch.py`, `tools/lever_matrix.py`, `tools/levers.d/expert_prefetch.json`, `tools/test_lever_matrix.py`, `docs/TUNING.md`, `docs/TPS_EXPERIMENTS.md` | `--expert-prefetch P` (`STRATA_Q36_EXPERT_PREFETCH`), the VRAM accounting line, `--vram-reserve MiB`, adaptive guess count, resident plus guess slots, launcher flags, lever 6 of the lever matrix (`LEVER_NUMBERS = range(1, 7)`, the test rejects 7); the fix of the first version's timing |
| `iq3_down_formats` (stacked on the one above) | `8ffffe05`, `b922cb3a`, `5e57d6dc`, `e91aba5a`, `f9ff8405`, `78f82229` and the commits of this record | `src/kernels/cuda/iq_kernels.cu`, the three docs | IQ3_XXS and IQ3_S down projections accepted by the engine (`STRATA_D_FMTS`), the ledger, the backlog and this file |
| `requant_dense_kquants` | `19894117` | `tools/requant_dense.py` | `--to q6_K|q5_K|q4_K`, the `Target` table, `quantize_to` / `dequantize_from` |
| `qwen36_default_stack` | `ffd0c62f` | `tools/strata_launch.py`, `serve/test_launch_stack.py`, `README.md`, `docs/PI.md`, `docs/TUNING.md` | the launcher defaults of section 6: the dense Q5_K file, 7 experts, a prefetch of 3 where it can run, and the opt-outs |

Every new engine switch is off in the engine itself and a missing condition makes the engine say which and run as before; the defaults of
section 6 live in the launcher, not in the engine. The measurement
chains and the tool-call evaluator are in `docs_private/experiments/tps_goal/scripts` (they carry local paths); the evaluator's case
design is described in E15 so it can be rebuilt.

## 6. Conclusions and recommendations

1. **The decode speed can be raised by about 18 to 22 percent per step in software**, from 35 to 44 to 48 tok/s on the code prompt, and
   by about 33 percent under sustained heat (the stack keeps its speed better while running 3 C hotter, E9; the cause is not established). The
   maximum profile (dense Q4_K, 5 experts, prefetch 3) reaches 52 tok/s (-32 % per step) at a KLD of 0.083, and 55 tok/s with resident
   slots at a 16K context (its KLD was not measured, E18).
2. **Recommended stack:** dense Q5_K file, 7 experts, `--experts pinned --expert-prefetch 3`. Engine KLD about 0.025, HumanEval and
   the tool-call tests inside the noise; 44 to 46 tok/s (the repeat of the speed run gave 44.1 and 43.7 against 46.2 in the first run). If
   more speed is wanted at KLD about 0.04, use the dense Q4_K file instead.
3. **Defaults.** Since 12 November `bin/strata-serve`, and so `spi`, start this stack unless told otherwise (branch `qwen36_default_stack`, commit
   `ffd0c62f`): the dense Q5_K file when it exists (else the Q4_K_XL file), 7 experts, and a prefetch of 3 on the `auto` profile when the memory
   is free for the page-locked experts and no `--overlap` or `--igpu-experts` is given. `--experts-used 8 --expert-prefetch 0` gives the
   model's own routing back; `spi` keeps the IQ4_XS and Q4_K_XL entries with those two flags for output that must not differ from the
   original model's. The file is made with `tools/requant_dense.py --groups all --to q5_K` (README, "The default stack").
4. **Cheapest in fidelity** is the prefetch alone (KLD 0.005, -10 % per step); it needs 14.66 GiB of page-locked RAM.
5. **Do not repeat:** lower-bit expert files without fast kernels, per-layer K, top-p, mixed-precision experts, n-gram drafting
   in the engine at today's row cost, MTP depth other than 2, huge pages, software prefetch distance, the iGPU share, the
   Q6_K dense part.
6. **Limits of the evidence:** the quality tests are small (HumanEval 164 problems, 89 plus 36 generated tool-call cases, 261
   teacher-forced positions); the n-gram statistics come from another model's texts; the prefetch was measured on one machine; no
   multi-turn agent session was run.

Choosing a setting by the loss that is accepted (baseline 34.9 tok/s, step 65.0 ms; the step saved per 0.01 of mean KLD falls, on the
whole, with every step towards more speed, so the knee of the curve is dense Q5_K with 7 experts and prefetch 3):

| Setting | tok/s | step, ms | step saved, ms | mean KLD | ms saved per 0.01 KLD |
| --- | --- | --- | --- | --- | --- |
| prefetch 3 alone | 42.1 | 58.0 | 7.0 | 0.005 | 13.5 |
| dense Q5_K + 7 experts + prefetch 3 | 46.2 | 52.1 | 12.9 | 0.025 | 5.1 |
| dense Q4_K + 7 experts + prefetch 3 | 48.3 | 50.3 | 14.7 | 0.041 | 3.6 |
| dense Q5_K + 6 experts + prefetch 3 | 47.4 | 50.0 | 15.0 | 0.048 | 3.1 |
| dense Q5_K + 5 experts + prefetch 3 | 50.1 | 46.8 | 18.2 | 0.091 | 2.0 |
| dense Q4_K + 6 experts + prefetch 3 | 50.7 | 47.2 | 17.8 | 0.066 | 2.7 |
| dense Q4_K + 5 experts + prefetch 3 | 52.1 | 44.3 | 20.7 | 0.083 | 2.5 |
| the same with resident slots (16K context) | 55.4 | 41.8 | 23.2 | not measured | - |

The tok/s column depends on how many drafts were accepted in that run (0.64 to 0.75 per draft), so the step time is the fair
comparison. Only the dense Q5_K + 7 experts row was run again (53.1 and 53.4 ms, 44.1 and 43.7 tok/s, drafts accepted 0.67), so the
other rows are single runs with a spread of about 1 ms in step time and up to 2 tok/s. The KLD of the engine's test has 261 positions and is noisy, so neighbouring rows are only approximately ordered.

## 7. Open decisions and leftovers

- **Files on disk** (about 33 GB): `models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-IQ4_XS-mtp-d5.gguf` (dense Q5_K) is the default model of the
  launcher and of `spi`: keep it. `...-d4.gguf` (dense Q4_K) is needed only for the maximum profile and to repeat the Q4_K runs; delete it
  when they are not wanted (it can be made again with `tools/requant_dense.py --groups all --to q4_K`).
- **Not covered:** a long multi-turn agent session on a stack, `nsys` of a step with MTP, two interleaved streams (estimated at 1.3
  times the aggregate with MTP, no gain for one stream), a share of the MTP block for the GPU, faster dense kernels (+10 %
  estimated, lossless).
