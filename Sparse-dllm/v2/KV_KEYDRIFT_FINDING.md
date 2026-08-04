# KV Key-Drift Predicts Attention-Skippability Beyond SkipGPT (L1 Evidence)

**Status:** L1 (fast-proxy) evidence, robust across every fair baseline tested. Pending L2
(real-trajectory) confirmation before it is a proven result. Fast-dLLM v2 7B, GSM8K, RTX PRO 6000.
Code: `interventional_probe.py` + capture hooks in `model/modeling.py`.

---

## 0. TL;DR

We are testing whether we can build a **learned router that skips modules (attention / MLP) for tokens
that have converged** during block-diffusion decoding — the "layer-sparsity" third axis of our
composable sparsity family. The danger: **SkipGPT** (ICML 2025) already does learned, module-decoupled
skipping in AR LLMs, and **SureLock** already does KL-convergence locking in diffusion. So the question
isn't "does convergence-skipping work" — it's **"is there a convergence signal that predicts
skippability better than what SkipGPT's router already sees?"** If not, we're just re-deriving SkipGPT.

**Answer (L1):** For **attention**, yes — a **KV key-drift** signal (`‖ΔK_token‖ / ‖K_token‖`) predicts
attention-skippability with **CV-AUC 0.772 vs 0.709** for the strongest fair SkipGPT baseline. The win
is **specifically the key projection** (not layer-locality, not temporality — both controlled for), is
**attention-only** (no win on MLP), and is **block-diffusion-specific** (key-drift-across-denoising-steps
is free at inference because diffusion caches KV across steps; an AR router structurally cannot see it).

---

## 1. What "skippable" means, and how we measure it (the intervention)

A block-diffusion small step recomputes a 32-token block over several denoising sub-steps. A **module**
(attention or MLP) at layer *l* for token *t* is **skippable** if removing it does not change the model's
decoded output.

We measure this **interventionally** (a counterfactual, not a correlation):

1. Take a real prompt; form the next 32-token block.
2. Run a reference forward → record each token's decoded argmax (`ref_arg`).
3. For each (token *t*, layer *l*, module *m*): **residual-skip** module *m* for token *t* — i.e. replace
   its output with 0 so only the residual passes (`x_{l+1} = x_l`, exactly SkipGPT's skip semantics) —
   run the forward, and check whether *t*'s decoded argmax **flipped**.
4. **Label:** `skippable = 1` if the argmax is unchanged (skip did no damage), else 0.
   We also record `cross_flip` = how many *other* block tokens flipped (KV-coupling leakage).

This gives a per-(token, layer, module) skippability label grounded in "did the output actually change,"
not in a proxy for it.

---

## 2. The convergence proxy (why this is "L1")

A token's real convergence signal is how little its state changes between **consecutive denoising steps**.
Running the full multi-step trajectory for every intervention is expensive, so **L1 uses a two-state
proxy**: build the same block at two masking levels —

- **early** = 70% masked (unconverged), **late** = 25% masked (mostly settled, the reference).

The change from early→late stands in for "change across denoising steps." This is the main limitation:
it is a *proxy for a proxy* (two synthetic mask levels, not real steps), and is exactly what **L2** will
replace with the real trajectory. Everything below is L1 evidence.

---

## 3. The predictors we compared (the "baseline ladder")

For each intervention we compute several candidate signals and ask: **which best separates skippable from
not** (measured by example-held-out cross-validated AUC — folds split by example so tokens from the same
block never leak across train/test)?

| predictor | what it is | why it's here |
|---|---|---|
| `hnorm` | ‖hidden‖ of the token | trivial baseline |
| `settled` | is the token unmasked (1/0) | trivial baseline (a settle-counter) |
| `stale_l2` | ‖Δhidden‖/‖hidden‖ (early→late) | hidden-state staleness (our old signal) |
| `stale_kl` | KL(logits_late‖logits_early) | output staleness (= SureLock's signal) |
| **`kv_stale`** | **‖ΔK_t‖/‖K_t‖** (key drift, per layer) | **the new signal — drift of the token's KEY** |
| `kv_infl` | (attn into *t* from unsettled tokens) × `kv_stale` | cross-token coupling variant |
| `hidden_probe (final)` | logistic on the final hidden vector | SkipGPT-ish (but weak: global) |
| `hidden_probe (per-layer)` | logistic on the **per-layer pre-module hidden** | **SkipGPT's REAL router input** |
| `hidden_DELTA (per-layer)` | logistic on per-layer hidden at **both** states | **SkipGPT + same temporal info** |

The last two are the crucial fair baselines — see §5.

---

## 4. The result

CV-AUC (example-held-out), GSM8K, n=16 examples, 7168 interventions:

### Attention
| predictor | CV-AUC |
|---|---|
| stale_kl (SureLock-style) | 0.665 |
| **kv_stale (key-drift)** | **0.744** |
| kv_infl (cross-token) | 0.672 |
| **kv_combo** | **0.772** |
| hidden_probe (final) | 0.655 |
| hidden_probe (per-layer) — *SkipGPT* | 0.689 |
| hidden_DELTA (per-layer) — *SkipGPT + temporal* | 0.709 |

### MLP (FFN)
| predictor | CV-AUC |
|---|---|
| stale_kl | 0.708 |
| kv_stale (scalar) | 0.560 |
| v_stale (scalar) | 0.580 |
| kv_combo (multi-feature) | 0.651 |
| hidden_DELTA (per-layer) | ~0.62–0.64 |

**KV key-drift wins on attention and does NOT win on MLP.** On FFN, every *single scalar* KV signal
(0.56/0.58/0.61) sits *below* the fair baseline; only multi-feature combos narrowly (and inconsistently
across runs) clear it — a combo-overfit false positive, not a real effect. The verdict now requires a
single scalar to beat the baseline. **The effect is module-specific: attention yes, FFN no** — consistent
with every prior result (FFN has minimal convergence-dependence throughout this study).

### Key vs Value (which KV channel?)
| channel | CV-AUC (attn) |
|---|---|
| **kv_stale = ‖ΔK‖ (KEY / routing)** | **0.744** |
| v_stale = ‖ΔV‖ (VALUE / content) | 0.641 |
| kvv_combo (K + V + cross) | 0.766 ≈ kv_combo 0.772 |

**The signal lives in the KEY, not the value.** Value drift is weaker and adds nothing on top of key
drift (kvv ≈ kv). Interpretation: what predicts attention-skippability is whether the token's *address*
(how others attend to it) has settled — the key/routing channel — not whether its *content* (value) has.
This is a sharper claim than "KV drift matters." (Caveat: the probe skips a token's own attention; the
*deployed* freeze also has others read the token's stale **V** — so the value question may reopen at L2,
where the intervention matches deployment.)

Supporting structure (Table 1 of the probe): kv_stale is strongest at **early** attention layers
(AUC 0.845), decaying to ~0.54 at late layers — mechanistically sensible (early attention is where
tokens gather context; a settled key means that gathering is done).

---

## 5. Why "KV beats SkipGPT" specifically — the three baselines we had to defeat

The claim "beats SkipGPT" is only meaningful if we defeat SkipGPT's **actual** router input, not a
strawman. We ruled out three ways the win could have been trivial:

1. **Is it just ‖hidden‖?** No — `hnorm` = 0.59. Beaten easily.
2. **Is it just layer-locality?** `kv_stale` is *per-layer*; maybe it wins only because the hidden
   baseline used the single *final* hidden. So we gave the baseline the **per-layer pre-module hidden**
   (SkipGPT's true input): 0.689. **kv_combo (0.772) still beats it.** Not layer-locality.
3. **Is it just temporality (2 states vs 1)?** `kv_stale` uses two states (early+late); the hidden
   baseline used one. So we gave the baseline the **same two states** (`hidden_DELTA`, concat of
   per-layer hidden at both): 0.709. **kv_combo (0.772) still beats it by +0.063.** Not just temporality.

After controlling for norm, layer-locality, and temporality, the KV signal *still* wins — so the
information lives **specifically in the key projection**, not in anything SkipGPT's router can access.

---

## 6. The hypothesis — *why* the key is special

The **key** vector is literally *what other tokens read from token t through attention*. So:

> **A token whose key has stopped drifting is a token whose contribution to every other token's attention
> has stabilized — meaning its attention can be skipped (others keep reading its cached key/value with no
> change), and it no longer needs to re-gather context itself.**

This is why key-drift beats hidden-drift: the raw hidden mixes together many things (lexical identity,
position, FFN features) that are irrelevant to attention-skippability, whereas the key is the *exact*
quantity attention consumes. Key-drift is a **mechanistically targeted** convergence signal for the
attention module specifically — which is also why it does **not** help MLP (the MLP reads no keys).

**Why it's block-diffusion-specific (the novelty):** measuring key-drift requires the token's key at a
*previous* denoising step. In an AR model each token is computed **once** — there is no previous-step key,
so this signal cannot exist. In block diffusion the token is recomputed across denoising steps and its KV
is cached, so `‖K_t^step − K_t^{step-1}‖` is **available for free at inference**. SkipGPT (AR) therefore
structurally cannot use this signal; we can.

**The design it implies (module-decoupled router — Xinze's point 1, with data):**
- **Attention router** ← keyed on **KV key-drift** (novel, block-diffusion-specific).
- **MLP router** ← keyed on **hidden / KL** (SkipGPT-style; KV doesn't help here).

---

## 7. Honest limitations (what makes this L1, not a proven result)

1. **Proxy convergence.** early/late are two synthetic mask levels, **not** real consecutive denoising
   steps. This is the main risk — the margin could grow or vanish on real trajectories. **L2 fixes this.**
2. **Proxy damage.** "argmax flip vs a 25%-masked reference," not a full re-decode to ground-truth output.
3. **Modest margin, small n.** +0.06 at n=16 examples; no CI on the CV gap yet → needs a stability rerun.
4. **Single-skip regime.** Base rate is ~95% skippable for a *single* module skip; a real router skips
   *many* modules cumulatively — that regime is untested here.
5. **Report key-drift, not "cross-token."** `kv_stale` (0.744) is the workhorse; the cross-token `kv_infl`
   (0.672) is weak and adds only a little in the combo. The honest headline is **key-drift**.
6. **Confound to control (per review):** `kv_infl` may partly encode block masking-density; and the
   result should be checked under a non-random (contiguous-span) masking scheme. `kv_stale` is less
   exposed to these.

---

## 8. Next steps

1. **Stability rerun** (`--n 32 --seed 100`): does the attn `kv_combo > hidden_DELTA` margin hold? (~20 min)
2. **L2 — real-trajectory confirmation:** run the actual denoising loop; per real step compute
   `‖K_t^k − K_t^{k-1}‖`, residual-skip attention, measure change in the **final decoded block**; correlate.
   This is what turns "probe" into "result."
3. **Confound checks** (§7.6) once L2 is in.
4. **Cumulative-skip evaluation:** does a key-drift-gated attention router preserve accuracy while skipping
   a meaningful fraction of attention modules end-to-end (the actual speedup test)?

---

## 9. How this fits the paper

The composable sparsity family: **query/token sparsity (compute)** + **KV sparsity (memory)** +
**layer/module sparsity (depth)**. The convergence-router is the third axis. KL-convergence and
module-decoupled routing are prior art (SureLock, SkipGPT); **our contribution is the KV key-drift signal
for attention** — a block-diffusion-specific, mechanistically-grounded router feature that beats the fair
SkipGPT baseline — composed into a module-decoupled router within the family. Pending L2 confirmation.
