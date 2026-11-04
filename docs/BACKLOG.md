# Backlog: faster decoding

All gains below are estimates, not measurements, unless a line says otherwise. Baselines come from
[TUNING.md](TUNING.md): about 37 tok/s with greedy + MTP 2 (75 ms per step, 2.7 tokens per step), about 30 tok/s without
drafting.

| ID | Item | Estimated gain | Depends on |
| --- | --- | --- | --- |
| B1 | Measure real agent traffic | none by itself; decides B2, B5, B6 | - |
| B2 | Speculative sampling for non-greedy requests | up to +20-25% on those requests only | B1 |
| B3 | Start the next draft on the GPU while the CPU verifies | up to +15% | - |
| B4 | VNNI kernels that read a weight once for several rows | +10-20% | - |
| B5 | n-gram drafting on long matches | unknown (hypothesis) | B1 |
| B6 | Top-2 branch at the first MTP position | about +0.1 token per step, likely below its cost | B1 |

```
B1 ──► B2   if the non-greedy share of generated tokens is notable
 ├───► B5   if long n-gram matches are frequent in the agent's text
 └───► B6   close without building if first-position acceptance is high
B3, B4      independent, can start at any time; B4 is the first choice if B1 shows few non-greedy requests
```

## Results of the 2026-10-30 experiments

Full report: [decoding-speedup-report.html](decoding-speedup-report.html); patch, scripts and raw data: `experiments/`.
Measured on a laptop with about 12 GB of page cache for the model (not 23 GB), with noise of 10-50% between repeats, so
only ratios inside one series are used.

| ID | Status | Result |
| --- | --- | --- |
| B1 | done | pi sends no temperature or penalties (0 of 30 requests), so its traffic is all greedy. Decode is about 9% of model time in the long recorded session and 44% in short tasks. D=2 is still the best draft count. Acceptance by position about 94% / 83% / 76%. An extra verify row costs about 20 ms. |
| B2 | implemented, verified | `STRATA_Q36_SPEC_SAMPLING=1` (off by default): greedy text unchanged, distributions not distinguishable (300 samples per mode), +39% .. +126% tok/s on requests with temperature 0.7. No effect on pi. Patch in `experiments/`, not committed. |
| B3 | closed | The next draft needs the last layer's hidden state, so no overlap is possible; ceiling with a free draft is 12%. `--draft-vocab` (draft head over the first N token ids) cuts draft time 9.7 -> 3.7 ms but acceptance falls 88% -> 72%, net loss. |
| B4 | closed | The AVX-512 kernels already unpack a weight once for several tokens. VNNI does not reduce the operation count for K-quants (a scale per 32 values). Ceiling at most 6% of a step. |
| B5 | closed | Calibrated simulation: best +9% only in the long copy-heavy session, +1..4% on short tasks, negative for typical settings. |
| B6 | closed | Ceiling about +2% against about +25% step cost. |

New candidates (not measured):

- B7. Draft-head token list ranked by real frequency instead of the first N ids; ceiling about 7% of a step.
- B8. Prompt reading in long sessions: in the recorded long session about 91% of model time was prompt reading. Cause
  found and addressed in pi itself: a 67-call loop of identical results (51% of tool-output tokens), nine minutes per
  compaction (cold summary request, long summary, cold read of a 22K-token tail) and no line numbers in `read`. Done as
  the pi extension `tools/pi-strata` (branch `pi_strata_extension`, not committed); results in
  [pi-extension-report.html](pi-extension-report.html): -16% model time on a nine-module review, 26 of 27 exact
  `file:LINE` references (1 of 27 before), the loop guard stops the recorded session at turn 68 of 128.

## B1. Measure real agent traffic

Goal: replace the guesses behind B2, B5 and B6 with numbers from real agent sessions.

- Record agent (pi) sessions against the server; the repository holds no recorded sessions (`logs/` has only the server
  log), so recording comes first.
- Share of requests and of generated tokens by sampling mode: greedy (the launcher default), request-set temperature or
  penalties.
- Accepted drafts per position (draft 1 and draft 2) and tokens per step, split by request kind.
- Match-length distribution of an n-gram lookup over the context at each generated position, and how often the
  continuation after the match is correct (offline, no engine change; input for B5).
- Per-request `drafts_offered` / `drafts_accepted` are already in `/metrics` and the Prometheus output. Per-request
  sampling parameters and per-position acceptance were not found in a quick look at `serve/`; add them to the request
  accounting if they are missing.

Done when: one table with the four measurements above is added to TUNING.md.

## B2. Speculative sampling for non-greedy requests

Today a request that sets its own temperature or penalties is decoded without MTP (about 30 instead of 37 tok/s).

- Standard rejection sampling keeps the output distribution equal to the target model's. A draft picked by argmax is a
  point mass, so a draft is accepted with the target probability of that token, and on rejection the next token is
  sampled from the residual distribution.
- Acceptance will be lower than the 84-90% of greedy decoding, so the real gain is below the upper bound.
- Greedy requests must stay bit-identical: the 96-token checksums in TUNING.md must not change.
- Check distribution equivalence statistically (token frequencies over many samples, with and without drafting), not by
  comparing single outputs.

Start only if B1 shows a notable non-greedy share. Done when: the equivalence check passes, the greedy checksums are
unchanged and tok/s is measured on non-greedy requests.

## B3. Start the next draft on the GPU while the CPU verifies

Listed as untried in TUNING.md (up to +15%, a calculation). The draft only proposes, so greedy text must stay identical.

Done when: the greedy checksums are unchanged and the speed gain is measured with the ABBAAB order and several runs per
variant, because a hot CPU moves the result by tens of percent (see "Benchmarking pitfalls" in TUNING.md).

## B4. VNNI kernels that read a weight once for several rows

Listed as untried in TUNING.md (+10-20%, a calculation). It lowers the compute cost of multi-row verification; it does
not lower the bytes read from RAM, which is the other ceiling.

The CPU expert kernels now give the same bits as ggml. Keep that, or state the deviation and re-run the bit checks
described in TUNING.md.

Done when: the bit checks pass and the measured step time at 1.5K fill is lower than the 64.5-83.0 ms of the current
build.

## B5. n-gram drafting on long matches

Hypothesis only. The earlier test gave 1.69 tokens per step against 2.68 for MTP, on generated code. An agent copies
existing code (file contents, edits, tool arguments), which that test did not cover.

- Step 1 is offline: use the match statistics from B1; no engine change.
- Step 2, only if long matches are frequent: draft from the n-gram only when the match is at least N tokens long,
  otherwise use MTP. Each extra row adds expert reads (a 3-row step costs about 75 ms against about 33 ms for one row),
  so an extra row must pay for itself; set N and the row cap from those costs.

Done when: the offline statistics say go or no-go, and, if go, a prototype is measured on the recorded sessions.

## B6. Top-2 branch at the first MTP position

Expected gain is about 0.1 token per step against about +10 ms per step for the extra row (estimate). Close without
implementing if B1 shows high acceptance at the first position; build only if B1 contradicts the estimate.

## Not planned

UBTree and DFlash-style tree verification. Reasons, as of the check on 2026-10-30:

- No UBTree drafter for Qwen exists (only Ling-3.0 tiny and flash); the reference engine is an SGLang fork.
- A verification pass over a 32-node tree needs about 160 of 256 experts per layer (independent-routing estimate), roughly
  12 GB read per step against 0.6 GB for one token. At the measured 54 GB/s that is at least 220 ms for the CPU part, so a
  step would need about 8-9 accepted tokens to match today's 2.7 per 75 ms.
- Qwen3.6 has recurrent layers, so a tree needs per-branch state rollback; the engine verifies only linear windows.
- The DFlash drafter for Qwen3.6-35B-A3B (0.4B, BF16) does not fit in the VRAM left at a 64K window.
