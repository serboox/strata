# Backlog: what is left to check after the November 2026 speed experiments

The ledger is `docs/TPS_EXPERIMENTS.md`; this file lists what is still to be measured, why, how, and what came of it. An item is
closed by writing its result into the ledger and marking it here.

Reference points (code prompt, MTP 2, 2.2 GHz cap, 80 C limit): baseline 35.2 tok/s and 64.5 ms a step; the stack "dense Q5_K + 7
experts + prefetch 3" runs a step in 52.1 ms (46.2 tok/s, engine KLD about 0.025); with the dense part in Q4_K 50.3 ms (48.3 tok/s,
KLD about 0.04). HumanEval on the engine: baseline 153 of 164, the two stacks 150 and 152.

| # | Item | Why | How | Status |
| --- | --- | --- | --- | --- |
| T1 | Quality on tool calls and multi-step agent tasks for the baseline and the stacks | HumanEval sees only code writing; the agent use is what the engine is for | generated function-calling cases with exact expected calls (one step and a short chain), run against the engine's server for each setting | open |
| T2 | Stability under a long generation | the prefetch and the lighter dense part must not drift in speed, heat or memory over thousands of tokens | four requests of 3000 tokens per setting, sampling the engine's RSS, the VRAM and the package temperature | open |
| T3 | Prefetch at 16K and 32K context | attention gets dearer and the K/V cache takes VRAM; the gain might shrink | a long prompt through the server, decode speed read from the session line, baseline against prefetch against the stack | open |
| T4 | Which dense groups to compress | the attention, the recurrent blocks and the shared expert may differ in what they cost in quality per millisecond saved | one Q4_K file per group (`tools/requant_dense.py --groups`), engine speed and the engine's teacher-forced KLD each | open |
| T5 | Prefetch with resident VRAM slots | at a shorter context some experts can live in VRAM; the code path exists and was never run | `--max-context 16384` with `--vram-reserve`, baseline against prefetch 3, the log's slot lines | open |
| T6 | More aggressive stacks | where does HumanEval start to fall? | 6 and 5 experts with the dense Q5_K and prefetch 3: speed, KLD, HumanEval | open |
| T7 | MTP depth on the new step | depth 3 did not pay at 65 ms a step; a shorter step may change that | `--mtp 1, 2, 3` on the Q5_K stack | open |
| T8 | n-gram statistics on real agent text | decides whether n-gram drafting in the engine (B5 in `docs/BACKLOG.md`) is worth building | `b5_stats.py` over recorded transcripts: the share of tokens an n-gram draft would have predicted, with the row-cost model | done 12 November on 600,000 generated tokens of three recorded agent sessions (a proxy: the texts are another model's edits and tool arguments): a draft taken from the context where the last 4 to 12 tokens match, with an extra row costing 21 ms as measured, gains at best 2.4 percent (match 12, up to 4 tokens, 28.7 percent of the tokens by n-gram, 3.45 accepted per n-gram step); with an extra row at 12 ms it gains 14 percent (match 6, up to 4 tokens). Not worth building at today's row cost; look again if rows get cheaper |
| T9 | Noise of the HumanEval comparison | a difference of one to three problems cannot be told from noise by the counts alone; a run of one setting is deterministic, so repeating it adds nothing | compare which problems flip between the baseline and a stack, per problem | done 12 November: against the baseline the Q5_K stack lost 5 problems and gained 2 (exact two-sided p 0.45), the Q4_K stack lost 4 and gained 3 (p 1.0): both inside noise; problems 83 and 115 fail in both stacks |
