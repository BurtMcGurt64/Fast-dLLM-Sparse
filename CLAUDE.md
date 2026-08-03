# Fast-dLLM v2 Lab Notes

## Project Goal
Characterize sparsity improvements for Fast-dLLM v2, a block-diffusion LLM on Qwen2.5-7B-Instruct.
It decodes 32-token blocks via three forward types (prefill, small step, commit step). The primary
direction was **Top-K Block Loading**: use commit-step attention scores to load only the K most-
attended prefix blocks during denoising small steps, reducing KV bytes read.

---

## ⭐ CURRENT STATUS (2026-07-22) — READ THIS FIRST, supersedes older framing below

**The investigation is COMPLETE. Now in WRITE-UP mode** (honest characterization paper; draft at
`v2/sparsity/writeup/main.tex`). Do NOT relitigate closed directions below — they are closed with
both measurement AND literature.

**Central findings this project established (all measured on Fast-dLLM v2 7B, RTX PRO 6000 Blackwell):**
1. **Content selection ≈ position-only (StreamingLLM).** commit-score top-K does NOT beat sink+recency
   pinning on retrieval F1, at ANY budget (frac 2%→20%, hotpotqa) or needle depth. Verified via
   `baseline_compare.py` (dense/streaming/commit/mage modes) + `needle_test.py`. Gap +0.005–0.014 = noise.
2. **First-block-dense mechanism = the real reason accuracy is "preserved."** Retrieval completes in
   DENSE forwards (block 1 for commit/streaming; MAGE's dense first-step), so KV sparsity can't affect
   short-answer accuracy — commit/streaming leave block 1 dense. When you DO sparsify the retrieval
   block (MAGE arms block 1), accuracy drops 6–10%. So "preservation" is structural, not selection.
3. **Attention structure explains it:** ~28% content mass at 8k but DIFFUSE (frac90 ~28% = ~70 blocks,
   ~0.3% each) and REDUNDANT (plain≈streaming≈dense); sink mass (~50%) is largely NULL softmax
   release-valve, not info; content info propagates FORWARD (early layers gather middle/needle, late
   layers integrate → recency captures it). See `attn_mass_analysis.py` (per-step, true-softmax mass).
4. **Coherence contribution VALIDATED (the one real positive):** plain top-K degenerates long
   generation (loops); sink+recency pinning fixes it. n=30 gov_report @8k matched 20% frac:
   pinned rep 0.12 ≈ dense 0.08 ≪ plain 0.31; degen 4/30 ≈ dense 3/30 ≪ plain 13/30. `coherence_eval.py`.
5. **Inference-only is CLOSED on every lever** (each: measurement + published prior art):
   KV sparsity (~1.3× e2e, ≈StreamingLLM), NFE (needs training, published: dParallel/T3D), verification
   (self-confirming — bidirectional in-block attn), learned unmask policy (p at info ceiling),
   layer/Q sparsity (v2 has no layer redundancy; published: ES-dLLM/ShortGPT/Layer-Collapse).
   Field is SATURATED (see [[field-saturation]] memory).

**The paper (honest, hard-to-scoop):** "KV-selection strategy is irrelevant to accuracy in
block-diffusion LLMs — content, position, and oracle methods are equivalent — because retrieval
completes in dense forwards; content attention is diffuse/redundant; and we present a
coherence-preserving pinning method." + the mechanism (first-block-dense, forward propagation).
Prior work (MAGE etc.) never runs the position-only baseline that reveals the equivalence.

**Speedup numbers to report (field-standard = decode-throughput vs Fast-dLLM v2, like ES-dLLM's
1.2–1.85× vs DualCache):** ~2× per small-step, ~1.6× decode-phase, ~1.3× e2e at realistic gen length.
Be EXPLICIT which scope. Report at long generation / throughput (the convention) but flag quality
holds only to ~500 tok on this checkpoint. Baseline = optimized Fast-dLLM v2 (a HARD baseline — vs
vanilla it'd be huge; that's a strength, not weakness).

**MAGE caveat:** our `mage` mode is a REIMPLEMENTATION, not published MAGE. Do NOT claim published
MAGE underperforms — frame as "arming the retrieval block costs accuracy," a mechanism, not a ranking.

**Personal/strategic context:** undergrad, summer project, PI wants an ICLR-2027-cycle paper (~Sept
2026 deadline). Considering a pivot to VLA efficiency ([[vla-pivot]]) as a NEXT project (toolkit
transfers, less-saturated). The write-up + pivot coexist. See memory: [[selection-ablation]],
[[coherence-contribution]], [[layer-sparsity-closed]], [[field-saturation]], [[speedup-ceiling]].

> ⚠️ **READ THIS FIRST — the project has been fully characterized and the earlier optimistic
> framing was wrong.** Inference-only top-K on this fixed checkpoint gives **~1.3× end-to-end in
> practice** on realistic tasks and is capped at **~2× as an absolute ceiling** (only reachable in an
> unrealistic regime). The "3–5×" that motivated the project was a *component measured in isolation*
> (KV-read kernel), never end-to-end. Every number below marked "measured" is real (RTX PRO 6000
> Blackwell); the old *theoretical projections* (1.65×/2.17×/2.95×) were **disproven** — they counted
> KV bytes only and omitted the weight floor, prefill, commit steps, throughput saturation, and the
> first-block-dense penalty. The only lever that breaks past ~2× is reducing the NUMBER of forward
> passes (NFE), which requires **training** — out of scope for a systems lab, and a naive attempt
> mode-collapses (see NFE section). **[RESOLVED 2026-07-22: decision made — honest characterization
> paper; see CURRENT STATUS at top. The selection mechanism was further shown to ≈ StreamingLLM, and
> "preservation" traced to first-block-dense decoding. Don't reopen.]**

---

## THE COMPLETE CHARACTERIZATION (measured, this is the ground truth)

### The ceiling is a waterfall — each scope adds unprunable work (B=32/8k)
| Scope | Speedup | What it adds that top-K can't touch |
|---|---|---|
| Per small step | **2.13×** | weight floor (~half the bytes) |
| Decode phase | **1.59×** | commit steps re-read full KV |
| End-to-end | **1.36×** | prefill (compute-bound) |

Top-K only speeds up **one slice** — KV reads in the small steps. Everything else (weights, prefill,
commit steps, generation-loop overhead) is unprunable and dominates end-to-end.

### Measured byte split (`measure_kv_bytes.py`, actual allocated tensors, B=32/8k)
- Weights: **13.05 GB (46.5%)**, KV cache: **15.03 GB (53.5%)**. Even pruning KV to **zero** → ~2×.
  The "5×" (= 1/frac) needs weights = 0, which never happens.
- KV re-read multiplier ~**1.7–1.85×** (`torch.cat` + Flash both read the prefix), backed out from timing.

### Per-eval ceiling from the runtime breakdown (`runtime_breakdown.py`, B=1)
The small-step share of e2e = the fraction top-K can touch; `1/(1−share)` is the absolute ceiling.
| eval | small-step % | prefill % | absolute ceiling | realistic (at 2.4× small-step) |
|---|---|---|---|---|
| gsm8k | 90.6% | 2.6% | 10.66× (needs FREE small steps — impossible) | **~2.1×** |
| math | 91.4% | 0.9% | 11.64× | **~2.1×** |
| narrativeqa 8k | 53.3% | 44.0% | 2.14× | **~1.45×** |
The absolute ceiling is a red herring (it assumes free small steps). The **realistic** column caps at
~2× because the small step itself has a within-step weight floor (top-K caps it at ~2.4×, see below).

### Measured batched speedup on the best long-gen task (`derisk_genlen.py`)
gov_report, recency-pinned top-K, generation length controlled: **1.32× (B=16), 1.34× (B=32)** at
256 tokens/8k. Decode-only asymptote (prefill removed) ~1.6×. **Practical realistic number: ~1.3×.**

---

## Memory vs Compute — VERIFIED (`memory_vs_compute.py`)
Empirical hardware peaks measured on-device (no hardcoded values): **1.46 TB/s, 379 TFLOP/s → roofline
ridge ≈ 260 FLOP/byte** (RTX PRO 6000 Blackwell).

- **Small step is memory-bound at every batch up to 64** — compute never exceeds 21% of peak. It does
  NOT become compute-bound as batch grows (kills the "push batch → compute-bound" objection).
- **Per-step top-K speedup GROWS with batch: 1.9× (B=16) → 2.15× (B=32) → 2.4× (B=64).** It is *not*
  capped at 2× — it climbs toward 1/frac=5× as KV overtakes weights. (This corrects the old "1.89×
  per step" framing that implied a fixed cap.)
- **But it is NOT cleanly bandwidth-saturated at high batch**: achieved bandwidth falls 64% → 20% of
  peak from B=1 → B=64. So "runtime ∝ bytes" is first-order only — real kernel/launch overhead grows.
  **Anchor arguments on the direct `topk_x` measurement, not on an idealized roofline.**
- Two-part model: **attention/KV** (top-K prunes it; always memory-bound, AI=56; <1% of FLOPs) vs
  **FFN/projections** (top-K can't touch; memory-bound at low batch, compute-bound at B×q > 260, i.e.
  ~B=32 at q=8). So at high batch the *weights* become a compute floor — unprunable by top-K OR INT8.

---

## Why batch / context / generation-length don't break the ceiling
- **Batch** amortizes the fixed weights → per-step speedup climbs — BUT (1) prefill & commit scale
  *with* batch so they never shrink their e2e share, (2) throughput saturates ~B=8–16 (tok/s flat past
  there), (3) latency rises 9× by B=64, (4) B=32/16k OOMs. Net e2e ~1.4×.
- **Context** grows the prunable KV *and* the unprunable prefill at the same rate → they **cancel**.
  Measured e2e: 8k=1.23×, 16k=1.24× (flat). Distinct mechanism from batch (cancellation, not saturation).
- **Generation length** is the only lever that moves e2e (dilutes fixed prefill) — but asymptotes to the
  decode ceiling (~1.6×), and realistic tasks generate too little to get there.
- **First-block-dense penalty**: block 1 runs dense (no prior commit score) → 1/N of blocks unprunable →
  hits SHORT generation hardest (narrativeqa ~1.4 blocks → ~1.1×). Measured via gen-length sweep:
  64tok→1.09×, 256→1.31×, 512→1.29×.
- **B=1 timing is UNRELIABLE**: at B=1, KV is only ~7% of the byte budget → real per-token speedup
  ~1.06×; wall-per-example is dominated by generation-length + Python-loop noise. **Always measure
  speedup at batch (≥16).**

Full write-up: `v2/sparsity/WHY_BATCH_CONTEXT_DONT_HELP.md`.

---

## THE PAPERS' ACTUAL SERVING REGIME (important — the premise was mismatched)
Both Fast-dLLM papers were checked (arxiv 2505.22618, 2509.26328):
- **v1**: GSM8K/MATH/HumanEval/MBPP, gen 256/512/1024, **batch size 1 only**. Headline "27.6×" is
  single-request latency.
- **v2**: GSM8K/MATH/HumanEval/MBPP/MMLU/GPQA/IFEval, **context length ~2048**, batch **1–64** (Fig 5).
  Throughput vs AR Qwen: **2.54× at B=1 → ~1.5× at B=64 (A100), 1.8× (H100)** — the diffusion advantage
  *shrinks* with batch (the compute-bound dynamic).

**Implication**: the papers serve at **short context (~2k) on reasoning benchmarks**, NOT long-context.
The old "target B=32, context 8k–32k" premise was chosen to make KV sparsity matter, but it's off-
distribution for how the model is used. At 2k context, KV is <25% of the byte budget → top-K helps
~1.1×. The model's *value* is NFE (parallel decode per block); KV sparsity is orthogonal to it.

---

## Current Implementation: TopKBlockLoader (`v2/sparsity/topk.py`)
Monkey-patches `model.forward`. At each commit step: disarms selection (scores see full prefix),
captures per-layer query-weighted block scores, arms subsequent small steps with selected token indices
(per-sequence gather, `[B, K_tokens]` — no union explosion). Block 1 runs dense.

**Now supports (added this project):**
- `frac=` — **dynamic budget** `round(frac × current_n_blocks)` that scales with the growing context
  (prompt + generated). Use this for long generation; fixed `k=` collapses to too-few blocks as the
  context grows. (Pass `k` OR `frac`.)
- `n_sink=2, n_recency=4` — **sink + recency pinning**: blocks 0–1 and the last N blocks are always kept
  (scores forced to +∞ before top-K). **This fixes long-generation coherence** (see below).

### THE RECENCY-PINNING FIX (a real, block-diffusion-specific mechanism)
Plain top-K **degrades long-generation coherence** — as generation proceeds it prunes the model's own
recent output, so it loses the thread and **loops/repeats** (`long_generation_eval.py`, repetition score).
Pinning sink+recency fixes it: on gov_report, `n_recency=8` → **14/15 generations coherent** (rep ≤0.19),
vs severe degeneration without it. This is the most contribution-worthy piece — the commit-score oracle
+ guaranteed recency is unique to block diffusion.

---

## NFE / Training direction (the only >2× lever — and why it's stuck)
Fewer forward passes per block is the one thing that beats the ceiling (multiplicative). Probed it:
- **Base-model quality-vs-NFE cliff** (`nfe_probe_stage1.py`): GSM8K holds to threshold ~0.6 (~1.5×
  fewer forwards at ~7% accuracy cost), then **cliffs** (0.5→52%, 0.4→38%, 0.3→32% vs 73.5% ref).
- **Naive high-mask LoRA fine-tune** (`nfe_probe_stage2b.py`): training loss drops cleanly (6.2→1.5) but
  **generation MODE-COLLAPSES** into repeated digits — quality holds ~step 25, dead by step 100, and the
  cliff **never moves**. Classic teacher-forcing gap (the `[noisy|clean]` doubling lets the model cheat
  on the training loss). Confirmed at multiple LRs/mask ratios.
- **Conclusion**: reaching lower NFE needs a *generation-aware* objective (self-distillation on the
  model's own decoded outputs / consistency) — a **novel-loss ML-algorithms project**, out of scope for
  a systems lab, and non-trivial (the naive version fails). The flex-attention training path *does* run
  on this box (`nfe_probe_stage2a_smoke.py` passes).

---

## Attention Structure Findings (still valid, measured, B=1)
Scripts: `block_attention_dist.py`, `block_attention_heatmap.py`. frac90 = fraction of prefix blocks
capturing 90% of attention at the last commit step (n=40).

- **frac90 ≈ 18–20% across QA/retrieval tasks and context lengths** (narrativeqa 18–20%, qasper ~20%,
  2wikimqa 15–16%, multifieldqa ~20%). Context-length independent. gov_report rises to 28–34% at full
  generation (summarization attends to more sections).
- **Structural pattern: Sink (blocks 0–1, ~40–50% mass) + Recency (last 2 blocks, ~30–35%) = 78–82% of
  attention (deterministic); content routing is the sparse 18–22% middle.** ← this motivates the
  recency-pinning fix above.
- **Diffusion is ~2× more concentrated than AR** (frac90 18% vs 38% on narrativeqa, controlled single-
  step comparison). Attributed to the commit-step objective (predict 32 tokens at once → sharper routing).
- **Score reuse across commits is NOT viable** (`commit_score_stability.py`): Pearson r≈0.84 but
  Jaccard@20%≈0.55 → ~29% of blocks change per commit. Full rescoring required each commit (it's free —
  scoring happens during the commit forward).

---

## Bugs Fixed
| Bug | Impact | Fix |
|---|---|---|
| `logits_to_keep=0` in generation_functions | lm_head [B,L,vocab]=80GB at B=32 → OOM | `logits_to_keep=1` in prefill calls |
| HF `sdpa` `repeat_kv` when mask≠None | KV H_kv=4→28, +30GB → OOM | `F.scaled_dot_product_attention(enable_gqa=True)` in modeling.py |
| Structural prefill arming (block 1) | −0.033–0.044 F1 (block 1 wrong context) | Reverted; block 1 runs dense |
| Batched decoding corrupts ragged prompts | GSM8K accuracy 62%→25% at bs=8 | Use bs=1 for gsm8k/math evals (ragged); longbench pads uniformly so batches OK |
| Fixed `k` collapses on long-gen short prompts | k=1 for a growing 30-block context → degeneration | `frac=` dynamic budget |

**B=32/16k OOM root cause**: NOT the KV cache (28 GiB, fits) — it's the **prefill FFN activation spike**
(~37 GiB, gate+up over the full prompt). Fixable with **chunked prefill**. Weights ~15.2 GB, matmul
params 7.07B (`measure_kv_bytes.py`).

---

## What Has Been Ruled Out
- **Big inference-only speedup (>2×)**: impossible — waterfall dilution + within-step weight floor +
  throughput saturation. ~1.3× practical, ~2× absolute ceiling (stacked with INT8, best regime).
- **"Push batch higher"**: per-step grows but throughput saturates ~B=8–16, latency rises, OOM at 16k.
- **"Longer context helps"**: cancels (prefill grows with prunable KV).
- **"Find a long-generation task to reach the ceiling"**: gov_report (long-context) is prefill-dominated
  → ~1.33×; the short-context+long-gen regime that would give ~2× is where the model DEGENERATES.
- **FFN sparsity as an escape**: activation sparsity hits the same batch-union problem + weak on SwiGLU;
  pruning/MoE needs retraining. Quantization (the working form) is bounded (~1.35× INT8) and mirrors top-K.
- **Sparse prefill at 8–16k**: prefill is FFN-dominated below ~28k crossover → ~1.1×. Only helps >28k.
- **Naive NFE fine-tuning**: mode-collapses (see NFE section). Needs generation-aware objective (training).
- **B=1 as a speedup benchmark**: KV ~7% of bytes → real ~1.06×; wall-clock is generation-length noise.
- **Sparse Q / FP8 / FLOPs-crossover motivation / union top-K / score reuse / tps-when-lengths-differ**:
  ruled out earlier (memory-bound so FLOPs don't matter; per-seq gather solves union; full rescoring needed).

---

## Experiment Scripts
| Script | Purpose |
|---|---|
| `v2/sparsity/topk.py` | TopKBlockLoader — now has `selection=` ("commit"/"streaming"/"mage") + `frac`, `n_sink`, `n_recency` |
| `v2/sparsity/baseline_compare.py` | **[2026-07] dense vs streaming(position) vs commit(content) vs mage — the selection≈StreamingLLM result** |
| `v2/sparsity/needle_test.py` | **[2026-07] needle depth-sweep; all 100% → retrieval is dense (first-block-dense)** |
| `v2/sparsity/full_eval.py` | **[2026-07] accuracy + tok/s across evals/batch/gen; `--batch_sizes` sweep, `--force_gen_len`** |
| `v2/sparsity/attn_mass_analysis.py` | **[2026-07] per-step TRUE-softmax attention mass (heatmap + sink/recency/content%/frac90/entropy); needs modeling.py `_capture_block_mass`** |
| `v2/sparsity/coherence_eval.py` | **[2026-07] dense vs plain vs pinned coherence (rep/distinct4); `--long_context`, `--force_len`** |
| `v2/sparsity/verify_probe.py` | [2026-07] commit-verify probe — verification is self-confirming (dead) |
| `v2/sparsity/unmask_policy_probe.py` | [2026-07] learned-unmask-policy probe — p at info ceiling (dead) |
| `v2/sparsity/memory_vs_compute.py` | **memory-vs-compute regime, empirical peaks, topk_x vs batch** |
| `v2/sparsity/runtime_breakdown.py` | **prefill/commit/small time split per eval → ceiling** |
| `v2/sparsity/long_generation_eval.py` | **dense vs sparse on long-gen tasks + coherence/repetition + text** |
| `v2/sparsity/measure_kv_bytes.py` | resident weight vs KV bytes (+ ncu path); `ncu_kv_byte_sweep.sh` |
| `v2/sparsity/derisk_genlen.py` | batched dense-vs-sparse e2e speedup (forces equal gen length) |
| `v2/sparsity/profile_step.py` | single dense-vs-sparse small step (fake KV) |
| `v2/train_scripts/nfe_probe_stage1.py` | quality-vs-NFE cliff (gsm8k/math), threshold sweep |
| `v2/train_scripts/nfe_probe_stage2a_smoke.py` | training-forward smoke test (flex attention runs) |
| `v2/train_scripts/nfe_probe_stage2b.py` | high-mask LoRA fine-tune + before/after cliff (mode-collapses) |
| `v2/sparsity/plot_scaling.py` | 4-panel scaling figures |
| `v2/sparsity/eval_longbench_sweep.py` | LongBench F1 + speedup at varying K, ctx (has `load_longbench_subset`) |
| `v2/sparsity/baseline_quality_eval.py` | baseline F1 across subsets × context (model works to 32k) |
| `v2/experiments/*` | frac90 / attention structure / AR comparison / commit-score stability |

### Key docs (read these)
- **`v2/sparsity/writeup/main.tex`** — the paper draft (current deliverable).
- **`v2/sparsity/MEETING_BRIEF_2026-07-20.md`, `FUTURE_DIRECTIONS_2026-07-20.md`** — the four closures + VLA pivot.
- `SPEEDUP_CEILING.md`, `SCALING_ANALYSIS.md` — ceiling evidence & cost model.
- `WHY_BATCH_CONTEXT_DONT_HELP.md` — the definitive batch/context explanation.
- `SLIDES_WHY_CAPPED.md` — full presentation deck (why capped, measured-vs-projected).
- `MEETING_SUMMARY.md` — one-page summary (pre-selection-ablation; see CURRENT STATUS for the newer findings).

---

## Model Architecture Reference

**This is NOT a standard HuggingFace model.** Do not assume standard transformers conventions.

### Files
- `v2/model/modeling.py` — `Fast_dLLM_QwenForCausalLM`, `Fast_dLLM_QwenModel`, `Fast_dLLM_QwenAttention`,
  `Fast_dLLM_QwenMLP`, `Fast_dLLM_QwenRMSNorm`, `Fast_dLLM_QwenRotaryEmbedding`
- `v2/generation_functions.py` — `batch_sample` (bound via `types.MethodType`, not a model method)
- `v2/quantize.py` — `Int8ComputeLinear`, `apply_int8_compute`, `load_model`
- `v2/kernel.py` — `FusedFFN` (INT8), `FusedFFN_INT4`, `FusedFFN_FP8`
- `v2/train_scripts/finetune.py` — lmflow wrapper; mask token is `"|<MASK>|"` (id **151665**), `bd_size` config

### Backbone
Qwen2.5-7B: 28 layers, hidden=3584, FFN=18944, 28 Q heads, 4 KV heads (GQA ratio=7), head_dim=128, bf16.
Load via `quantize.load_model()` (snapshot_download + manual load_state_dict). Do NOT use
`AutoModelForCausalLM.from_pretrained` or `Fast_dLLM_QwenForCausalLM.from_pretrained` — both silently skip
weight loading. q/k/v proj `bias=True`, o_proj `bias=False`. `lm_head.weight` tied to `embed_tokens`.
`max_position_embeddings=32768`.

### Generation Loop (3 forward pass types)
1. **Prefill** (`update_past_key_values=True`, `use_block_cache=False`, q=prompt): dense pass over prompt,
   writes prefix cache, uses `eval_mask`.
2. **Small step** (`update_past_key_values=False, use_block_cache=True`, q=8): diffusion denoising over the
   current 32-token block via 8-token sub-blocks. Attention mask `None` when block cache populated. ~94%
   of decode forwards. **This is the only forward top-K speeds up.**
3. **Commit step** (`update_past_key_values=True, use_block_cache=False`, q=32): commits block KV to prefix
   cache, predicts first token of next block. ~6% of forwards. **Attention scores captured here for top-K.**
Ratio: ~16 small steps per commit (~17 forwards per 32-token block).

### KV Cache API — Non-Standard (critical)
Attention reads the prefix cache as `torch.cat((past_key_value.layers[idx].keys, key_states), dim=-2)`
— NOT `key_cache[idx]`. Always check `hasattr(DynamicCache(), 'layers')` is True first.

### Training vs Inference
Training: input `[noisy_x_t | x_0]` doubled along seq dim, `fused_flex_attention` + `block_diff_mask`,
masked-diffusion cross-entropy on masked tokens, per-block mask ratio `t ~ Uniform(0,1)` (modeling.py
~657–718). Inference: single block, SDPA (`is_causal=False`), boolean `eval_mask`. The `if self.training`
branches are load-bearing. Logit shift: `logits = cat([logits[:,:1], logits[:,:-1]], dim=1)` (predicts
token[i], not next) — preserve it.

### Sampler
`sampler = types.MethodType(Fast_dLLM_QwenForCausalLM.batch_sample, model)`. Calls `self.sample_with_top_p`
(defined on the model class). At temperature=0 returns argmax; top-p skipped. Threshold controls
unmasking: `unmask_idx = (x1_p > threshold)`, ≥1 unmask/step guaranteed. **Lower threshold → fewer steps
→ lower NFE.**
