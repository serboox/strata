# Rank-based mixed precision of MoE experts: study and backlog

Status: study and plan only. Nothing is built, downloaded or measured for this idea yet.

## The idea

A MoE decode step reads the selected experts from system RAM on every token. The router gives each of the 8 selected experts of Qwen3-Coder-30B-A3B a weight. The experts with the largest weights matter most, so read the top few at higher precision and the rest at lower precision, and fewer bytes cross the memory bus.

Two full copies do not fit the machine (31 GiB RAM: Q4_K experts 16.3 GiB plus Q2_K experts about 8.9 GiB, 25 GiB with the dense part). So the plan uses **layered storage**:

- layer 1: the expert matrix quantized to Q2_K (2.625 bits per weight), read for every selected expert;
- layer 2: Q2_K of the residual (weights minus dequantized layer 1), read only for the top `n_hi` experts;
- an expert of rank <= `n_hi` computes `dot(L1, x) + dot(L2, x)` (about 5.25 bits per weight), the others compute `dot(L1, x)` only (2.625 bits).

Average read per token with `n_hi` = 2 of 8: (2 x 5.25 + 6 x 2.625) / 8 = 3.28 bits per weight against 4.85 for Q4_K_M (-32 %). With `n_hi` = 1: 2.95 bits (-39 %); with 3: 3.61 (-26 %); with 4: 3.94 (-19 %). The file grows from 17.3 to about 18.7 GiB (+8 %).

## Expected speed (estimate, not measured)

Cost model calibrated on LFM2.5-24B (33 tokens per second measured under the 80 C rule): GPU part, CPU part (bytes / 36 MiB per ms), 0.13 ms of synchronization per CPU layer, times 0.95. For Qwen3-Coder-30B-A3B at Q4_K_M with 14 % of the experts on the GPU the model gives about 27 tokens per second.

| Variant | CPU bytes | Estimate |
| --- | --- | --- |
| baseline Q4_K_M | 100 % | 27 |
| mixed, `n_hi` = 3 | 74 % | 32 |
| mixed, `n_hi` = 2 | 68 % | 34 |
| mixed, `n_hi` = 1 | 61 % | 36 |
| static k = 6 / 5 / 4 (for comparison) | 75 / 63 / 50 % | 32 / 35 / 39 |

The mixed rows assume that the Q2_K dot product reads bytes as fast as the Q4_K kernel does. That is not known: mainline llama.cpp has only an AVX2 Q2_K dot (`ggml/src/ggml-cpu/arch/x86/quants.c:1638`), no AVX-512 version. Error of the model: +-25 %.

**Honest reading:** at equal bytes the mixed scheme gives about the speed of a static k of 5 to 6, but all 8 experts still contribute. Whether that is better than a smaller k, or than a plain lower-bit file (Q3_K_M, Q2_K), is the open question that decides the project (gate G2 below).

## What the study found

Source marks: **[opened]** the page was read by a research pass, **[checked]** also checked by hand, **[snippet]** only seen in search results, **[unverified]** do not rely on it.

### Prior art for the precision split

- HOBBIT, arXiv 2411.01433 [opened]: experts ranked by gating weight; the score of an expert is the sum of the gating weights of the experts ranked above it. Thresholds for Mixtral-8x7B: T1 = 0.6, T2 = 0.9 (score <= T1 high precision, between low precision, above T2 skipped; the top expert is always high precision); about 67 % high, 30 % low, 3 % skipped. Built on llama.cpp (about 8,000 lines). Reported 1.31x (Mixtral) and 1.42x (Phi-MoE) over llama.cpp on an RTX 4090 with CPU, accuracy drop of at most 1 %. No code link was found.
- MC-MoE, arXiv 2410.06270 [opened]: static per-expert bit-width (1, 2 or 3 bits) from activation frequency times routing weight, training-free; Mixtral-8x7B at 2.54 bits on average loses 3.8 % (uniform 2-bit GPTQ loses 28.6 %); its dynamic part skips the second expert when its weight ratio is low (under 1 % loss).
- DynaExq, arXiv 2511.15015 [opened]: hot experts in high precision, cold experts in INT4 or INT2, chosen by selection counts. Qwen3-30B-A3B-Instruct average score: FP16 65.29, static INT4 63.98, DynaExq 64.38.
- MxMoE (arXiv 2505.05799) and BitsMoE (arXiv 2606.00079): [snippet].

### Prior art for layered storage

- RRQ, "Recurrent Residual Quantization", arXiv 2608.04048 [checked abstract]: a 2-bit base plus 2-bit residual stages made by round-to-nearest, calibration-free, one checkpoint for several precisions. WikiText-2 perplexity [opened]: Llama-3.1-8B 2-bit base 28.00, 4-bit (base plus one residual) 7.09 against GPTQ 4-bit 6.75; Qwen3-8B 10.12 against 10.48; Qwen3-14B 9.23 against 8.88. The package is 4-5 % larger than a MatGPTQ multi-bit package. No kernels, no speed. This is exactly our construction, with plain round-to-nearest instead of K-quants.
- Any-Precision LLM, arXiv 2402.10517 [opened]: bit-plane upscaling from a 3-bit seed; GPU kernels only. MatQuant, arXiv 2502.06786 [opened]: needs training; plain truncation to 2 bits is bad (Gemma-2 2B: 37.68 / perplexity 17.99 against 51.33 / 3.84 for a normal 2-bit). BitStack, arXiv 2410.23918 [snippet].
- Nothing was found about nested or layered tensors in llama.cpp or ik_llama.cpp.

### Router and expert statistics

- Nobody publishes the weight mass of the top-1, top-2, top-4 experts of Qwen3-30B-A3B. One paper (arXiv 2602.02443, model not confirmed) says the rank 1 to rank 5 gap exceeds 3.9 % and ranks 5 to 32 add less than 1.5 % [opened]. We must measure it (M1).
- Layers 1 to 3 hold "super experts" (L1 E68, L2 E92, L3 E82); pruning them raises WikiText-2 perplexity of Qwen3-30B-A3B from 8.70 to 59.86; experts of the last layer are not critical (arXiv 2507.23279) [opened]. The first layers must stay at high precision for all ranks.
- Popularity: a preprint on cache behaviour (arXiv 2608.18261) reports that 95 % of the traffic uses 52.5 % of the experts, and an LRU cache of 13.4 % of the experts serves 66 % of the picks; the model of the traces is not confirmed and the paper is unreviewed [unverified].
- Reducing the active experts: "Don't Skip the Baseline", arXiv 2610.28369 [checked abstract]: a static smaller k is hard to beat by dynamic skipping on Qwen3-30B-A3B; the numbers are in figures only. A llama.cpp discussion reports 21-26 % faster decode at top-4 on Qwen3-Coder-30B-A3B on an EPYC CPU, quality "nearly identical" at half the experts, visible loss at a quarter (author's rating, one machine) [snippet].

### Quantization facts

- Qwen3-30B-A3B base, KL divergence against BF16 and WikiText-2 perplexity (BF16 9.0703), table of a community quantizer: Q4_K_M 0.0101 / 9.2092; UD-IQ2_M 0.0666 / 10.3726; IQ2_M 0.0691 / 9.9788. No Q2_K or Q3_K_M row [opened].
- Importance matrix: with few calibration tokens many experts get no statistics; the author of ik_llama.cpp says about 500 tokens of one topic activate nearly all experts and that mainline refused to save a partial matrix (a 2025 comment, not rechecked) [opened].
- CPU speed of 2- to 4-bit dot products: only dense-model prompt-processing tables exist (Zen4: Q2_K_S 115 against 259 tokens per second in ik_llama.cpp), no decode numbers and no GB/s [opened]. We have to measure it.
- On this machine (LFM2.5-24B, measured): the IQ3_XXS dot product made decode 45 % slower and the profile was 59 % in that one function, so a "denser" format is not automatically faster; Q2_K is a plain K-quant and should behave better, but that is not measured.

## Gates

- **G1 (preliminary after M1, final after M2):** top-2 of 8 must carry enough of the router weight (guide: at least 60 % on average, measured in M1) and the accuracy loss at k = 6 must be small (measured in M2), otherwise stop.
- **G2 (after R3):** at equal average bytes the layered scheme must beat both a static k and a uniform low-bit file (Q3_K_M from a public quantizer, no engineering) by a clear margin in KL divergence and in a code benchmark. If not, stop and use the simpler option.
- **G3 (after R4):** end-to-end speed under the 80 C rule at least 30 % above the Q4_K_M baseline of the same model on the same build.

## Backlog

Sizes: S under half a day, M one to two days, L several days. Kinds: decision, measure, build.

### D: decisions

| ID | Kind | Item | Size | Done when |
| --- | --- | --- | --- | --- |
| D1 | decision | Target and gates: model (Qwen3-Coder-30B-A3B), speed target, quality limits for G1-G3, code benchmark subset (HumanEval+ and a LiveCodeBench slice) | S | numbers written here |
| D2 | decision | Memory plan: about 19 GiB file on a 31 GiB machine (pinned host buffers against mmap), what else must stay resident | S | a start-up check that refuses to run when the budget is exceeded |
| D3 | decision | Licence of the model and of the derived file (check the model card before publishing anything) | S | one line in this file |

### M: measurements that decide go or no-go (need the 17.3 GiB Q4_K_M file)

| ID | Item | Size | Deps | Done when |
| --- | --- | --- | --- | --- |
| M1 | Router statistics on real code prompts (preliminary verdict on G1): mass of the top-1, top-2, top-4 experts per layer and per token, popularity skew (share of picks covered by the top 5, 10, 14 % of experts), reuse between adjacent tokens. Use the existing logging patch of the llama.cpp clone (note: the top-k tensor is not contiguous) | M | D1 | tables per layer, preliminary verdict on G1 and on a hot set |
| M2 | Static k sweep (8, 6, 5, 4, final verdict on G1) with `--override-kv qwen3moe.expert_used_count=int:K` (key name for this architecture to confirm): KL divergence, code benchmark, speed | M | M1 | table k against quality and speed |
| M3 | Uniform baselines from public files at equal bytes: Q3_K_M and Q2_K (KL divergence and code benchmark, speed) | M | D1 | table, the reference for G2 |
| M4 | Q2_K dot throughput on this CPU (extend `kq512_parity --bench-rows`): GB/s against Q4_K at 1536- and 2048-value rows | S | none | numbers decide whether R5 (an AVX-512 kernel) is needed first |
| M5 | Requantization error: Q4_K_M to Q2_K, then residual to Q2_K; relative RMS error of layer 1 and of layer 1 plus 2 against the Q4_K weights, on a sample of expert tensors | S | T1 | error table |
| M6 | Importance matrix coverage: how many tokens of code activate all 128 experts in all 48 layers; build and check an imatrix from the Q4_K_M model | M | M1 | no expert without statistics |

### T: tooling

| ID | Item | Size | Deps | Done when |
| --- | --- | --- | --- | --- |
| T1 | Layered quantizer (C++ on the ggml API): reads a GGUF, dequantizes each expert tensor, writes layer 1 and layer 2 as two Q2_K tensors `blk.N.ffn_{gate,up,down}_exps.weight` and `..._exps_r.weight`, keeps all other tensors; per-layer exceptions (first 3 layers stay plain Q4_K with both ranks); optional imatrix. API: `ggml_quantize_chunk`, `ggml_get_type_traits(...)->to_float`, `gguf_init_empty`, `gguf_add_tensor`, `gguf_write_to_file` | L | D1 | file written, round-trip error as in M5 |
| T2 | Verifier: tensor names, shapes, types, byte sizes, residual correctness on a sample | S | T1 | script with tests, run in CI-less local checks |
| T3 | Same pipeline for LFM2.5-24B-A2B (local file, top-4, `n_hi` = 2), to test the code on a model that is already here | M | T1 | file and quality numbers |

### R: runtime (llama.cpp clone, branch of the patched checkout)

Code map of the study (paths relative to the llama.cpp checkout, commit 18ab62a):

- tensor enums `src/llama-arch.h:566-568`, names `src/llama-arch.cpp:539-542`, tensor-info table `src/llama-arch.cpp:1119-1121`; layer fields `src/llama-model.h:351-353`;
- loading: Qwen3-MoE `src/models/qwen3moe.cpp:52-54`, LFM2-MoE `src/models/lfm2moe.cpp:44-47`, LFM2 `src/models/lfm2.cpp:103-106`; optional tensors use `TENSOR_NOT_REQUIRED` (`src/llama-model-loader.h:65`, example `src/models/ernie4-5.cpp:54`);
- graph: `build_moe_ffn` `src/llama-graph.cpp:2193` and `:2241`, call in `src/models/qwen3moe.cpp:129-141`, LFM2 in `src/models/lfm-base.cpp:107-125`; the rank split of the ids already exists in the deferral patch (`docs_private/experiments/lfm25/lfm_defer.patch`);
- CPU `mul_mat_id` `ggml/src/ggml-cpu/ggml-cpu.c:1505` and `:1622`, Q2_K type traits `:308-311`, x86 Q2_K dot `arch/x86/quants.c:1638`; CUDA Q2_K `ggml/src/ggml-cuda/mmvq.cu:51`;
- trap: `tensor_is_expert()` in `src/llama-quant.cpp:167` matches only `_exps.` and `_chexps.` and would not treat `_exps_r.` as an expert tensor.

| ID | Item | Size | Deps | Done when |
| --- | --- | --- | --- | --- |
| R1 | Tensor enums, names, layer fields and optional loading of the `_exps_r` tensors for qwen3moe and lfm2moe; fix `tensor_is_expert()` | M | T1 | a layered file loads, a plain file still loads |
| R2 | Layered MoE graph: `gate = L1(all ranks) + L2(first n_hi)`, `up` likewise, SwiGLU on the joined result, `down` likewise, weights, sum; `n_hi` from the command line per layer (default all-high for layers 0 to 2); same CPU/GPU split structure and synchronization count as before | L | R1 | numerically equal to the plain model when `n_hi` = k and layer 2 is exact |
| R3 | Quality harness: KL divergence of `n_hi` in {8, 4, 3, 2, 1, 0} against Q4_K_M on CPU (`-dev none`), plus the code benchmark | M | R2, M2, M3 | verdict on G2 |
| R4 | Speed: llama-bench and the thermal bench under the 80 C rule (quiet machine) | S | R3 | verdict on G3 |
| R5 | Strata AVX-512 kernel for Q2_K rows (mainline has AVX2 only), with the parity test in the style of `kq512_parity`; then one pass that computes the L1 and L2 dots of an expert together on the shared activation | L | M4 | bit-identical to ggml, faster than it |
| R6 | CUDA check for GPU-resident layered experts (Q2_K is supported in `mmvq.cu`) | S | R2 | same result on GPU and CPU |

### V: variants (after the first result)

| ID | Item | Size | Deps |
| --- | --- | --- | --- |
| V1 | Dynamic number of high-precision experts by the cumulative gating weight (HOBBIT thresholds) instead of a fixed rank; needs masked ids and a CPU kernel that skips them | L | R4 |
| V2 | Hot set of experts resident in VRAM from the popularity of M1, in the denser format | L | M1, R2 |
| V3 | Per-layer schedule of `n_hi` from sensitivity (protect the super experts of the first layers) | M | R3 |
| V4 | Pruned base (REAP, 20 % fewer experts) to cut RAM and bytes; the pruned model exists only as safetensors, so it needs a conversion | L | R3 |

### E: evaluation and shipping

| ID | Item | Size | Deps | Done when |
| --- | --- | --- | --- | --- |
| E1 | End-to-end speed and quality against Qwen3.6-35B-A3B on the Strata engine for the user's coding tasks | M | R4 | table in docs |
| E2 | Ship: a tool in `tools/` that applies the llama.cpp patch (style of `tools/llama_cpp_kq512.py`) and the quantizer, with tests | M | E1 | one-command reproduction |
| E3 | Cleanup: temporary files, the 17 GiB download, the layered file when no longer needed | S | E2 | disk and processes checked |

## Risks and open questions

- Plain Q2_K of the lower ranks may be too coarse (the RRQ base alone has perplexity 28 on Llama-3.1-8B); the rank split depends on how much weight the top experts carry (M1).
- Double quantization: layer 1 is made from already quantized Q4_K weights, so the top experts can at best equal Q4_K_M, not the original; the imatrix of M6 may help layer 1.
- The two dot products of a high-precision expert double the arithmetic for those experts; if the CPU part is compute-bound rather than memory-bound the gain shrinks (M4, R5).
- Synchronization stays: every layer still has CPU work, so about 13 % of the token time (5 ms) is not touched by this idea.
- RAM: about 19 GiB for the file plus the desktop; D2.
- The statistics and speed claims marked [snippet] or [unverified] above must not be used for decisions.
