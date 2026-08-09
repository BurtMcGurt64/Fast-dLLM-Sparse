# A1 — Generation-Aware Loss for Trained Dynamic Depth in Block-Diffusion LLMs

*Fast-dLLM v2 (Qwen2.5-7B block diffusion). Prototype: `v2/train_scripts/depth_router_train_a1.py`.*

## 1. Goal

Train a **learned, per-token, dynamic-depth** policy (skip transformer layers for some tokens
during the cheap denoising "small steps") that **preserves generation quality** while cutting
per-step compute. The architecture has the capacity for this (measured, teacher-forced:
~22% layer-work cut at 96% agreement, learned routing beats static +53% and random +49%). The
open problem is *training* it without destroying generation.

## 2. Why the naive objective fails (the teacher-forcing gap)

The straightforward objective reuses the model's native masked-diffusion loss: take a block, mask
some tokens at ratio `t`, and predict them — using the **doubled `[noisy | clean]`** input on the
**training (`gen_mask`) path**. Two fatal properties:

1. **The clean answer is in the input.** The model minimizes loss by *copying* the clean half, not
   by learning to generate.
2. **The states are ground-truth-forced snapshots**, a distribution that does not match what the
   model sees when it decodes on its own.

**Measured consequence (this project):** gsm8k accuracy collapses monotonically with fine-tuning —
62.5% (no FT) → 7.5% (100 steps) → 0% (400 steps) — into numeric mode-collapse (coherent prose,
runaway/garbled digits). The collapse happens **even with the depth router doing no skipping at
all** (`keep=1.0`), proving the culprit is the *objective*, not the routing. This is the same wall
the NFE fine-tune hit (`nfe_probe_stage2b`).

## 3. The A1 loss

### 3.1 Setup and notation

A block of `B` tokens is decoded over denoising steps. At step `k`, the **state** `x⁽ᵏ⁾` is the
partially-decoded block (some tokens committed, the rest `[MASK]`). Let

- `p^student_θ(· | x⁽ᵏ⁾)` — the **student**: model with LoRA + depth routers active on small steps
  (reduced depth), evaluated on the **generation path** (`eval_mask`, `use_cache`).
- `p^orig(· | x⁽ᵏ⁾)` — the **teacher**: the **frozen original** model (LoRA off, full depth), same
  path. It is fixed (no gradient).
- `A_k` — the set of still-masked positions at `x⁽ᵏ⁾` (the positions whose predictions drive the
  next unmasking, i.e. what actually matters for the decode).
- `keep(θ)` — expected fraction of layer-executions retained (compute proxy); `ρ` — target keep.

### 3.2 On-policy rollout (no gradient)

Starting from an all-masked block, iterate: run the **student** forward → unmask the positions whose
confidence exceeds the threshold (≥1 per step) → repeat until the block is filled. This yields the
trajectory of states `x⁽⁰⁾, x⁽¹⁾, …, x⁽ᴷ⁾` **the student itself visits**. Prefill and commit are
**dense** (routers off).

### 3.3 Objective

$$
\mathcal{L}(\theta)=\underbrace{\frac{1}{K}\sum_{k=1}^{K}\frac{1}{|A_k|}\sum_{i\in A_k}
\mathrm{KL}\!\Big(p^{\text{orig}}(\cdot\mid x^{(k)})_i \;\big\|\; p^{\text{student}}_\theta(\cdot\mid x^{(k)})_i\Big)}_{\text{on-policy self-distillation to the frozen original}}
\;+\;\lambda\underbrace{\big(\text{keep}(\theta)-\rho\big)^2}_{\text{compute budget}}
$$

The KL is the standard distillation direction (`KL(teacher ‖ student)`, teacher as reference), taken
only over masked positions. The gradient flows through the student forward via a straight-through
Gumbel gate on the routers; the discrete unmasking during rollout is **not** differentiated (rollout
is `no_grad`; the loss is re-computed with gradient on the recorded states).

### 3.4 Why this closes the teacher-forcing gap

| property | naive | A1 |
|---|---|---|
| input contains the clean answer? | **yes → copy cheat** | **no** |
| states from | GT-forced snapshots | **student's own rollout (on-policy)** |
| target | ground-truth tokens | **frozen original's distribution** |
| forward path | training (`gen_mask`, doubled) | **generation (`eval_mask`)** |

Because the target is a fixed good generator and the states are the student's own, the student can at
best *reproduce the original's generation* — it cannot drift onto an off-distribution copy solution.

## 4. Where the diffusion-specific advantages compose in

These are **terms/structure of this loss**, not bolt-ons — and they are what an autoregressive
layer-skip method cannot express:

- **Recoverable-KV / dense-commit.** Prefill and the commit step run dense, so the KV written to the
  prefix cache (seen by all later tokens) is **exact**. Only the transient small steps are
  depth-reduced, and their errors are corrected by later steps / the dense commit. The loss can
  therefore *tolerate* larger small-step divergence as long as the **committed block** matches — add
  extra weight on the final state's KL, or a hard dense-commit constraint. In AR, a skipped layer's
  KV is a permanent hole; here it is recoverable.
- **Convergence axis (Hook A).** Because A1 trains on the *real trajectory*, the per-token convergence
  signal (unmask confidence / step index) is a genuine observable. Feeding it as a router input lets
  the router allocate depth by how converged each token is — the block-diffusion-specific routing that
  was invisible to the single-snapshot de-risk. *(Not yet wired in the prototype — the next step.)*

## 5. Relationship to prior art

- **CD4LM** (consistency distillation for dLLMs): distills for **fewer steps** (NFE). A1 distills for
  **less depth per step** — an orthogonal, composable axis, and A1 is **on-policy** (trajectory
  rollout) rather than snapshot-masked.
- **Mixture-of-Depths / FlexiDepth / Dr.LLM**: trained dynamic depth, but **autoregressive**; none
  have a refinement axis or recoverable-KV.
- **dLLM-Cache / ES-dLLM**: training-free feature/token reuse on unoptimized models; A1 is trained and
  targets a distilled model that lacks the free slack those methods exploit.

## 6. Open design decisions (for discussion)

1. **Teacher:** frozen original (safest, anchors to the good generator) vs current model at full depth.
2. **What to match:** full-distribution KL on masked positions vs extra weight on the committed block
   vs top-k agreement.
3. **Budget schedule:** fixed `ρ` vs a curriculum that anneals keep from 1.0 downward (start dense,
   sparsify gradually — never leave the manifold), possibly a Lagrangian instead of a quadratic penalty.
4. **Convergence conditioning:** which signal (confidence, entropy, step index) feeds the router.
5. **Cost:** on-policy rollout + a per-state grad forward is expensive; batching, state subsampling,
   and multi-block rollout are the scaling levers.

## 7. How success is measured

**Not** by the training loss (the naive method also had great training metrics — that is the whole
lesson). Success = the trained checkpoint **still generates**:

```
depth_router_gen_eval.py --load <ckpt>   →   routed_acc ≈ orig_acc   (vs naive FT's 0%)
```

at `keep < 1`. That, plus a real compute-skip kernel to convert the layer-cut into throughput, is the
paper.

## 8. One-line summary

> *Train the depth router by on-policy self-distillation: roll out the model's own block-diffusion
> decoding with depth-reduced small steps and a dense commit, and make the depth-reduced forward match
> the frozen original model's predictions along that trajectory. It is generation-aware by
> construction (its own states, no copyable answer, the generation path), which is precisely what the
> teacher-forcing gap requires — and it is where recoverable-KV and the convergence axis compose in.*
