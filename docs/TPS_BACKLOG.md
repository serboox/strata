# Backlog of the November 2026 speed experiments

The full record of each item (what was expected, what was done, the numbers, the conclusion) is in `docs/TPS_REPORT.md`; the
numbers are also kept in the ledger `docs/TPS_EXPERIMENTS.md`. This file is the list of the items T1 to T9, with the reason each was
added, how it was run and where its result is. An item is closed when its result is in the report and the ledger.

What could raise the speed further with all 8 experts and the weights unchanged (proposals only, ordered by how likely each is to
pay; nothing in it is implemented) is in [qwen36-lossless-backlog.html](qwen36-lossless-backlog.html).

Reference points (code prompt, MTP 2, 2.2 GHz cap, 80 C limit): baseline 35.2 tok/s and 64.5 ms a step; the stack "dense Q5_K + 7
experts + prefetch 3" runs a step in 52.1 ms (46.2 tok/s, engine KLD about 0.025; two repeats gave 53.1 and 53.4 ms, 44.1 and 43.7 tok/s); with the dense part in Q4_K 50.3 ms (48.3 tok/s,
KLD about 0.04). HumanEval on the engine: baseline 153 of 164, the two stacks 150 and 152.

| # | Item | Why | How | Status | Result in one line | Report |
| --- | --- | --- | --- | --- | --- | --- |
| T1 | Quality on tool calls and multi-step agent tasks for the baseline and the stacks | HumanEval sees only code writing; the agent use is what the engine is for | 89 generated function-calling cases (one step, chains, no-tool questions, precise arguments) and 36 harder ones (write whole functions, parallel calls, fix chains, value extraction), run against the engine's chat endpoint for 10 settings | done 12 November | every stack scores 88 to 89 of 89 and 36 of 36; only plain 5 experts (85, 35) and 4 experts (78, 23) fall | E15 |
| T2 | Stability under a long generation | the prefetch and the lighter dense part must not drift in speed, heat or memory | six requests of 2250 tokens without a pause, RSS, VRAM and package temperature sampled | done 12 November | the stack keeps +33 percent under heat (average 41.9 against 31.6 tok/s), no leak; package peak 82 C against 79 C | E9 |
| T3 | Prefetch at 16K and 32K context | attention gets dearer and the K/V cache takes VRAM; the gain might shrink | a long prompt through the server, decode speed from the session line, baseline against prefetch against the stack | done 12 November | decode 30.0 / 29.8 (base), 32.2 / 31.6 (prefetch), 37.4 / 33.5 (stack): +25 and +12 percent | E10 |
| T4 | Which dense groups to compress | the attention, the recurrent blocks and the shared expert may differ in what they cost in quality per millisecond saved | one Q4_K file per group (`tools/requant_dense.py --groups`), engine speed and teacher-forced KLD each | done 12 November | Q4_K per group: recurrent blocks -3.7 % per step, attention, shared expert, embeddings and MTP projection each about -1 % or less; three groups together -10.5 %, all groups -14.5 % (the effects do not add): convert all groups | E16 |
| T5 | Prefetch with resident VRAM slots | at a shorter context some experts can live in VRAM; the code path existed and was never run | `--max-context 16384` with `--vram-reserve`, baseline against prefetch 3, the log's slot lines | done 12 November | reserve 300 MiB with prefetch 3: 44.8 tok/s, 53.3 ms a step, against 36.2 for the default | E11 |
| T6 | More aggressive stacks | where does HumanEval start to fall? | 6 and 5 experts with the dense Q5_K and prefetch 3: speed, KLD, HumanEval | done 12 November | dense Q5_K + prefetch 3 with 6 experts: 47.4 tok/s, step -23 %, KLD 0.048, HumanEval 152; with 5 experts: 50.1 tok/s, -28 %, KLD 0.091, HumanEval 150: quality tests do not see the loss, the KLD does | E17 |
| T7 | MTP depth on the new step | depth 3 did not pay at 65 ms a step; a shorter step may change that | `--mtp 1, 2, 3` on the Q5_K stack | done 12 November | 44.6 / 43.7 / 42.4 tok/s, spread about 2: depth 2 stays | E12 |
| T8 | n-gram statistics on real agent text | decides whether n-gram drafting in the engine (B5 in `docs/BACKLOG.md`) is worth building | `b5_stats.py` over 600,000 generated tokens of three recorded agent sessions, with the row-cost model | done 12 November | at most +2.4 percent at today's 21 ms an extra row; +14 percent at 12 ms: not worth building now | E13 |
| T9 | Noise of the HumanEval comparison | a difference of one to three problems cannot be told from noise by the counts alone | compare which problems flip between the baseline and a stack, per problem | done 12 November | Q5_K stack lost 5 and gained 2 (p 0.45), Q4_K stack lost 4 and gained 3 (p 1.0): inside noise | E8 |

The branches were merged into `main` on 12 November. What remains with the owner is not backlog work: the dense Q4_K file (`...-d4.gguf`)
may be deleted if the maximum profile is not wanted, and the dense Q5_K file (`...-d5.gguf`) is the default model and stays (section 7 of
the report).
