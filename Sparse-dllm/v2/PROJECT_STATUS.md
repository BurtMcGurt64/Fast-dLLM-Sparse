# Fast-dLLM v2 Sparsity Project — Status & Handoff (2026-08-07)

## Who / context
Undergrad in Yuke Wang's ML-systems lab (Rice), summer project → co-authoring with postdoc **Xinze**.
Efficiency for **Fast-dLLM v2** (block-diffusion LLM, Qwen2.5-7B, 32-token blocks; decodes a block over
several denoising "small steps"). Private fork `BurtMcGurt64/Fast-dLLM-Sparse`; work in `Sparse-dllm/v2/`,
pushed from local mac → pulled on server `yuke-team` (`/data/datasets/eddie/Fast-dLLM`), run on an idle GPU
(`CUDA_VISIBLE_DEVICES=7`).

**Ground rules:** verify every number against a re-run (never trust recall); lit-check novelty before
building; honest framing only (label speedups by scope; baseline = *optimized* Fast-dLLM v2).

---

## The paper thesis
A **composable family of sparsity axes for block-diffusion serving**, organized by the roofline (each axis
attacks a different hardware bottleneck):

| axis | method | status |
|---|---|---|
| **Compute** | token policy (Xinze's), extended to **b>1** | ✅ built, measured, mine — free ~1.3× |
| **Memory** | KV streaming sparsity | ✅ built, measured — composes to ~1.8× @ high batch |
| **Depth** | layer/module router | signal question closed; composition/training OPEN |

**Measured results:**
- **Token policy (b>1):** accuracy-neutral on 4 LongBench tasks; ~1.3× decode throughput, grows with batch.
- **KV streaming:** composes with token → ~1.8× at bs=128; speed-for-accuracy, **task-dependent**
  (near-free on extractive QA ~−0.01 F1; ~−0.06 on summarization).
- **KV *selection* is irrelevant:** commit-score top-K = streaming = 0.222 F1 (content = position), verified
  fresh via `baseline_compare.py`. So the KV cost is from *having a budget*, not *how you pick*.

---

## What's CLOSED (do not reopen)
**A novel convergence *signal* for a module-skip router.** Tested exhaustively (see probe methodology below):
staleness (l2/cosine), KL, **key-drift ‖ΔK‖, value-drift ‖ΔV‖**, cross-token influence, scalar vs
full-vector, single-state vs temporal, matched-capacity. **No signal beats a fair per-layer hidden probe**
once you control for sample size and capacity.
- The exciting n=16 "key-drift beats SkipGPT" (0.744 vs 0.709) was a **small-sample + PCA-truncation
  artifact**: at n=100 the scalar loses (0.702 vs 0.735); the matched-capacity temporal `kv_DELTA` beat the
  hidden at PCA-64 (0.778 vs 0.734) but the gap **collapsed to noise** as the hidden got fair capacity
  (pca64 +0.044 → pca256 +0.014 ≈ 1.3σ).
- **Why, on principle:** `K, V = W·LN(hidden)` → the hidden *contains* K and V; the key is only a
  better-*conditioned* feature (packs attn-relevant signal into a compact subspace surviving PCA), not more
  informative. **Use the hidden as the router input.**
- **Prior art that owns the mechanism:** **SureLock** (ICLR 2026 — converged-token locking in masked
  diffusion, KL signal, caches KV, skips Q-proj+FFN) and **SkipGPT** (ICML 2025 — module-decoupled learned
  router on the hidden, AR).

## What's OPEN (the reopened direction)
The probe tested *frozen-model predictability*, NOT these — so they are **not** closed:
1. **A *trained* router + LoRA (co-adapted model)** — Xinze's actual proposal. SkipGPT works on the hidden
   (the input our probe says is best), so a trained router for block diffusion plausibly works; untested.
2. **Composition of all three axes** (token + KV + layer) in block-diffusion serving — unowned by anyone.
3. **Actual wall-clock speedup / co-design / kernels** — never measured (only prediction AUC).
4. **Per-*step* depth schedule** (do early denoising steps need full depth, late steps tolerate shallow?) —
   different granularity than per-token; untested; possibly block-diffusion-specific.

**Reframe:** the contribution is a **systems/co-design composition paper** (fits Yuke's agenda), not a novel
signal — signal = hidden (SkipGPT-style) is fine. Risk: individual pieces are prior art; value is
integration + co-design + characterization + realized speedup; needs training.

## Next steps (cheap → expensive)
1. **Per-step depth schedule** probe (cheap, could be block-diffusion-novel).
2. **Composition-speedup test:** add fixed/oracle layer-skip to the composition harness, measure wall-clock
   — does depth-skip stack with token+KV? (no training needed).
3. **Train SkipGPT-style router + LoRA for block diffusion** — big build; only if 1–2 promising.
4. **Bankable regardless:** token-policy ablations — KL vs l2 vs cosine (was mid-adding KL as a freeze
   metric), attn-vs-FFN, threshold Pareto.

---

## `interventional_probe.py` — full methodology

**Goal.** Decide whether a *convergence signal* can predict which (token, layer, module) is **skippable**,
and whether any such signal beats what SkipGPT's router already sees (the hidden state). "Module" ∈
{attention, MLP}.

**L1 (fast proxy).** The true convergence signal is how little a token changes between *consecutive real
denoising steps*. Running full trajectories per intervention is expensive, so L1 uses a **two-state proxy**:
build the next 32-token block at two masking levels —
- **early** = 70% masked (unconverged), **late** = 25% masked (mostly settled, the reference).
The early→late change stands in for "change across denoising." This is the main limitation (a proxy for a
proxy); the real-trajectory version is "L2," unbuilt.

**Per example (one prompt):**
1. Prefill a real prompt (from GSM8K stream); form the next 32-token block.
2. Forward the **late** block once = the **reference**; record each token's decoded argmax.
3. **Intervention (the "skippability" label):** for each (token *t*, layer *l*, module *m*),
   **residual-skip** module *m* for token *t* — replace its output with 0 so only the residual passes
   (`x_{l+1}=x_l`, exactly SkipGPT's skip semantics) — re-forward, and check whether *t*'s decoded argmax
   **flipped**. `skippable = 1` if unchanged. Also record `cross_flip` = how many *other* block tokens
   flipped (KV-coupling leakage). This is a **counterfactual**, not a correlation.

**Signals compared (the "baseline ladder"), each scored by how well it separates skippable-vs-not:**
- `hnorm` = ‖hidden‖; `settled` = is-unmasked flag — trivial baselines.
- `stale_l2` = ‖Δhidden‖/‖hidden‖; `stale_kl` = KL(logits_late‖logits_early) — hidden/output staleness
  (stale_kl = SureLock's signal).
- `kv_stale` = ‖ΔK_t‖/‖K_t‖ (**key/routing** channel); `v_stale` = ‖ΔV_t‖/‖V_t‖ (**value/content**
  channel); `kv_infl` = (block-local attention *into* t from unsettled tokens) × ‖ΔK‖ (cross-token).
- **Trained probes (the fair SkipGPT baselines):**
  - `hidden_probe (per-layer)` = logistic on the per-layer **pre-module hidden** at the late state
    (SkipGPT's *actual* router input, 1 state).
  - `hidden_DELTA (per-layer)` = logistic on the per-layer hidden at **both** states (SkipGPT + the *same*
    temporal info the KV signals get — the fair temporal baseline).
  - `kv_DELTA` = logistic on the full **key vectors** at both states (the matched-capacity temporal-KV
    analog of hidden_DELTA).

**Scoring & honesty controls:**
- **AUC**, computed with **example-held-out cross-validation** (folds split by example, so tokens from the
  same block never leak across train/test). Trained probes use StandardScaler + PCA(→64 by default) +
  logistic.
- **Bootstrap CIs over examples** (not rows — rows within a block are correlated → the honest *n* is the
  number of examples).
- **Confound check:** skip% for converged (settled) vs unconverged (masked) tokens by depth — to separate a
  real convergence effect from "late-layer skips just have less room to propagate."
- **Comparable label:** skippable = decoded-argmax-unchanged (absolute), so skip% is comparable across
  modules/depths (an earlier per-group-percentile version trivially forced ~50% and was fixed).
- **Noise-aware verdict:** requires a *single scalar* (or the matched-capacity `kv_DELTA`) to beat the fair
  baseline — multi-feature *combos* overfit small-n noise and don't count (they produced a false-positive
  ffn "win").
- **Capacity-fairness sweep:** re-fits `hidden_DELTA` and `kv_DELTA` at PCA ∈ {128, 256, full} — PCA-64
  truncates the 3584-d hidden (~1% kept) far more than the 512-d key (~6% kept), so this checks whether a
  KV "win" is just the hidden being truncation-starved (it was).

**How to run:**
```bash
CUDA_VISIBLE_DEVICES=7 python interventional_probe.py --n 100          # real run (n=examples)
CUDA_VISIBLE_DEVICES=7 python interventional_probe.py --n 2 --smoke    # ~1 min sanity
```
Needs `scikit-learn`. Capture hooks live in `model/modeling.py` (`_capture_kv` → per-layer Q/K/V snapshots;
`_capture_hidden` → per-layer pre-attn/pre-mlp hidden; `_skip_attn_pos`/`_skip_ffn_pos` → residual-skip a
module for given token positions; `_capture_contrib` → per-module ‖out‖/‖in‖).

**Verdict reached:** no signal (incl. matched-capacity temporal key) beats the fair hidden baseline once
capacity/sample are controlled → **use the hidden; the signal question is closed** (see "What's CLOSED").

---

## Key files
- `interventional_probe.py` — skippability probe (above).
- `run_policy_eval.py` — token/KV/compose eval (b>1, `--kv_mode none|streaming|compose|fourway`,
  `--metric l2|cosine`, `--task gsm8k|math|longbench`, `--lb_tasks`, `--repeat`, warmup).
- `SPARSITY_FINDINGS.md` — token+KV composition writeup (meeting-ready).
- `KV_KEYDRIFT_FINDING.md` — key-drift writeup, marked CLOSED.
- `baseline_compare.py` (main repo `v2/sparsity/`) — dense/streaming/commit/mage selection ablation.

## Meta
Skepticism repeatedly caught inflated/premature results (n=16 key-drift, the "lossless" claim, the
capacity confound) before they reached Xinze — the most valuable instinct in the project. Open question for
the next session: **does the depth axis compose for real speedup, and is a trained router worth building?** —
decided by cheap tests (per-step depth schedule, composition-speedup) *before* committing to training.
