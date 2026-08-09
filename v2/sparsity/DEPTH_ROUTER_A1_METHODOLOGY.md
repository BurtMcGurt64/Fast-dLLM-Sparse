# Trained Dynamic Depth for Block-Diffusion LLMs — Methodology, Loss, Results, Next Steps

*Fast-dLLM v2 (Qwen2.5-7B block diffusion, RTX PRO 6000). Companion to `A1_GENERATION_AWARE_LOSS.md`
(loss detail) and `RUNLOG.md` (exact commands + all result numbers).
Scripts: `v2/train_scripts/depth_router_derisk.py`, `depth_router_gen_eval.py`, `depth_router_train_a1.py`.*

---

## ★ STATUS UPDATE (2026-08-08) — READ FIRST (supersedes framing below)

**The loss is NOT the novelty — reframe.** On-policy self-distillation for dLLMs is a populated 2026
line: **dOPSD (2607.04428)**, "Learning from the Self-future" (2606.18195), D-OPSD, TAD, GDSD. But they
use a **temporal-privilege** teacher (future decode steps) for **reasoning quality**, with **no
architectural change**. A1's teacher privilege is **depth** (full→reduced) for **compute reduction** via
dynamic depth — orthogonal. So: cite the OPSD family as machinery; **the contribution is the *method*
(first trained dynamic-depth for block-diffusion) + the *diffusion-specific mechanism cluster* below.**

**The contribution = a mutually-reinforcing cluster (all measured, see RUNLOG §3-4):**
1. **A1 makes trained dynamic depth work** — naive fine-tuning collapses generation (teacher-forcing gap,
   62.5%→0%); on-policy self-distillation to the frozen original fixes it.
2. **Recoverable-KV validated** — forcing ~20% skip onto the COMMIT craters gsm8k 70%→15%, while the same
   ~20% on small steps is free. The commit is load-bearing; small-step skips are recoverable (AR can't do this).
3. **Emergent convergence allocation** — under just a loose average budget the router SPONTANEOUSLY keeps
   more depth near the commit (early 0.69 / late 0.85), mirroring finding 2. Forcing uniform depth hurts
   (53%); an explicit hand-designed schedule is NOT needed (emergent ≥ engineered). → step-conditioned
   schedule direction (#2) is dropped.

*One-line:* the learned router discovers and exploits the reversible-vs-irreversible structure of
block-diffusion decoding — something AR cannot do and the OPSD papers don't address.

**Frontier (gsm8k, n=100, orig=68%):** accuracy PRESERVED within noise down to keep_ss ~0.79 (~20%
small-step layer-compute). "Preserved," not "improved" (the +2/+3 is n=100 noise, ±9%).

**Corrections to the sections below:**
- §0/§5 call A1 "an on-policy self-distillation *loss*" as the novelty — WRONG post-dOPSD; it's cited
  machinery (see reframe above).
- Recoverable-KV (§2.2) is now VALIDATED by ablation (was asserted); emergent allocation is a NEW finding.
- MATH eval needed a parser fix (`_math_pred`: take FIRST boxed, cut hallucinated "Problem:") — earlier
  MATH numbers were a broken-eval artifact.

**Correct evaluation protocol (not yet run):** train A1 ONCE on BROAD/diverse prompts, eval the SAME
checkpoint on all tasks. Do NOT fine-tune per eval (defeats the point — like quantization, compress once,
deploy everywhere). Current results are gsm8k-trained (prototype convenience); MATH was transfer.

**Next:** (a) postdoc scoping meeting (strong hand: method + validated cluster + honest scoop awareness);
(b) rigor — broad-train-once, eval multi-task, n=300-500, ≥2 seeds; (c) throughput kernel (speedup is NOT
yet measured — gating simulates skipping; predicted ~1.1x standalone / ~2x stacked IF skips are whole-layer).

---

## 0. TL;DR

We add a **learned, per-token, dynamic-depth** policy to Fast-dLLM v2: during the cheap denoising
"small steps," a lightweight router skips transformer layers for some tokens; prefill and the block
commit run dense. Naïvely fine-tuning this collapses generation (the teacher-forcing gap: gsm8k
62.5% → 0%). We fix it with **A1**, an **on-policy self-distillation loss** that trains on the model's
own decoding trajectory against the frozen original model. A1 restores generation (dense accuracy
preserved) and yields a working quality/compute knob: **~15–20% of small-step layer compute is free on
gsm8k within measurement noise.** This is the project's first genuine positive efficiency result. It
attacks the weight floor that KV/token sparsity cannot, and stacks with the existing ~1.8× token/KV work.

---

## 1. Goal and why it was thought closed

**Goal.** Port Skip-GPT-style layer/module sparsity to block diffusion, exploiting diffusion-specific
signals (token convergence across denoising steps; KV that can be refreshed before a token commits).

**Why inference-only was closed.** v2 is distilled/SOTA-optimized → no *static* layer redundancy
(cosine-sim profiling: no layer exceeds 0.95 in/out similarity), and training-free reuse
(ES-dLLM, dLLM-Cache) rides on redundancy v2 lacks. A learned router on a *frozen* model can't beat a
hidden-state baseline (the earlier key-drift result). So the only open door is **co-training** the
backbone + router so v2 *becomes* depth-adaptive (MoD/FlexiDepth/Dr.LLM are the AR analogs; none exist
for block diffusion). That requires training — which is where the difficulty lives.

---

## 2. Methodology — the investigation arc

Each script answers one question and de-risks the next, cheapest-first.

### 2.1 Capacity probe (teacher-forced) — `depth_router_derisk.py`
**Question:** does the architecture even have room for per-token dynamic depth?
**Method:** LoRA (base frozen) + a per-layer router (tiny MLP + gate). Gating *simulates* a skip via
`out = input + g·(layer_out − input)` (g=0 ≡ skip; measures quality, not speed). Loss = KL to the dense
model + budget. Runs on the teacher-forced masked-snapshot path.
**Controls (the actual instrument):**
- **static per-layer** (skip the N lowest-value whole layers at matched budget) — rules out "just pick
  which layers to drop."
- **random per-token** (skip at matched per-layer rates, random tokens) — rules out "just trained
  stochastic-depth tolerance."
**Result:** ~22% layer-work cut at **96% argmax-agreement** to dense; **dynamic ≫ static (+53–72 pts)**
and **dynamic ≫ random (+49–57 pts)**. So the capacity is real and the routing is genuinely learned.
*Caveat found:* the router **ignores the block-level convergence signal `t`** (t-spread ≈ 0) — because a
single snapshot can't expose the denoising trajectory. Teacher-forced numbers are a **proxy**, not a
generation result.

### 2.2 Generation gate — `depth_router_gen_eval.py`
**Question:** does any of this survive *real* decoding?
**Method:** install the trained LoRA+router into the actual generation loop (`batch_sample`). **Prefill
and commit run dense (routers bypassed); only small steps are gated** — this is the **recoverable-KV /
dense-commit** mechanism: the KV written to the prefix cache is always exact, so small-step skips are
correctable. Reports three baselines — `orig` (LoRA off), `lora_dense` (LoRA on, no routing), `routed` —
on gsm8k (bs=1, ragged).
**Result (naïve fine-tune):** **collapse.** `orig` 62.5% → `lora_dense` **0%**. Coherent prose, garbled
numbers (numeric mode-collapse). Dose-dependent: 62.5% (0 steps) → 7.5% (100) → 0% (400). Critically,
this happened **even with the router doing no skipping** (`keep=1.0`), proving the culprit is the
**training objective**, not the routing.

### 2.3 Diagnosis — the teacher-forcing gap
The naïve objective trains on masked *snapshots* with the **doubled `[noisy | clean]`** input on the
training path. The model minimizes loss by **copying the clean half**, and the snapshot distribution ≠
the generation distribution. So training metrics look great while generation dies. This is the same wall
the NFE fine-tune (`nfe_probe_stage2b`) hit — now observed a second, independent time.

### 2.4 The fix — A1 generation-aware training — `depth_router_train_a1.py`
On-policy self-distillation on the generation path (see §3). This is where the diffusion-specific hooks
compose in, and it turns the collapse into a tunable knob (see §4).

---

## 3. The A1 loss

### 3.1 Setup
At denoising step `k`, the block state `x⁽ᵏ⁾` is partially decoded (some tokens committed, rest
`[MASK]`). Define:
- **student** `p^student_θ(· | x⁽ᵏ⁾)` — LoRA + routers active on small steps (reduced depth), on the
  **generation path** (`eval_mask`, `model.eval()`).
- **teacher** `p^orig(· | x⁽ᵏ⁾)` — the **frozen original** model (LoRA off, full depth). Fixed.
- `A_k` — still-masked positions at `x⁽ᵏ⁾`; `keep(θ)` — fraction of layer-executions retained; `ρ` — target.

### 3.2 Objective

    L(θ) = (1/K) Σ_k  (1/|A_k|) Σ_{i∈A_k} KL( p^orig(·|x⁽ᵏ⁾)_i ‖ p^student_θ(·|x⁽ᵏ⁾)_i )
           + λ · (keep(θ) − ρ)²

**Rollout (no grad):** decode the block with the *student itself*; record the states `x⁽ᵏ⁾` it visits.
**Loss (grad):** re-run each recorded state through teacher (no grad) and student (grad); KL on masked
positions; add the budget term. Gradient flows through the router via a **straight-through gate**; the
discrete unmasking is not differentiated.

### 3.3 Why it closes the teacher-forcing gap (three ingredients)
1. **On-policy** — trains on the student's *own* rolled-out states → no clean answer to copy; train
   distribution = generation distribution.
2. **Self-distillation to the frozen original** — the target is a fixed good generator → the student can
   at best reproduce it, so it cannot drift into a copy/collapse solution.
3. **Generation forward path** (`eval_mask`), not the doubled-batch training path.

|  | naïve | A1 |
|---|---|---|
| clean answer in input? | yes → copy cheat | **no** |
| states from | GT-forced snapshots | **student's own rollout** |
| target | ground-truth tokens | **frozen original's distribution** |
| forward path | training (doubled) | **generation (`eval_mask`)** |

### 3.4 Where the diffusion advantages compose in
- **Recoverable-KV / dense-commit:** prefill + commit are dense → committed KV exact; the loss tolerates
  small-step divergence as long as the committed block matches. (AR can't: a skipped layer's KV is a
  permanent hole.)
- **Convergence axis (Hook A):** because A1 trains on the real trajectory, per-token convergence
  (confidence/step) is a genuine observable the router can key on. *Not yet wired* — the next lever.

### 3.5 Training-stability fixes that made the router actually gate at inference
- **Deterministic straight-through gate** (`ste` mode): forward is `logit>0` (same as inference), so
  train-keep == inference-keep. (An earlier Gumbel gate satisfied the budget with *noise*, so gating
  vanished at deterministic inference — `keep_ss` stuck at 1.0.)
- **Budget curriculum:** anneal `ρ` from 1.0 → target (start dense, sparsify gradually).
- **Lower init gate bias (1.0)** + stronger `λ` so logits can actually cross zero (real skips).

---

## 4. Results

### 4.1 A1 fixes the collapse
`lora_dense` (A1-trained, run dense) **matches `orig`** — vs **0%** for the naïve fine-tune. Training
stays coherent throughout (no numeric collapse). The teacher-forcing gap is closed.

### 4.2 The quality/compute frontier
gsm8k, n=100, threshold 0.9, `orig = 68%`. `keep_ss` = small-step keep-rate (inference).

| target_keep | keep_ss | routed_acc | vs orig | small-step compute cut ≈ 0.94·(1−keep_ss) |
|---|---|---|---|---|
| 0.90 | 0.96 | 70% | +2 | ~4% |
| 0.85 | 0.93 | 72% | +4 | ~7% |
| 0.80 | 0.89 | 71% | +3 | ~10% |
| 0.75 | 0.86 | 69% | +1 | ~13% |
| 0.70 | 0.83 | 61% | −7 (outlier) | ~16% |
| 0.65 | 0.79 | 70% | +2 | ~20% |

**Read:** accuracy is preserved **within noise** down to `keep_ss ≈ 0.79` (~20% small-step layer-compute
cut); **no clean cliff** observed. The `0.70 → 61%` point is a **non-monotonic outlier** (the deeper 0.65
scored higher at identical seed/data) — the spread is **noise-dominated**: n=100 gives ±~9% CI, plus
run-to-run training variance (KL bounced 0.25–0.44 at deep targets). Honest claim: *no resolvable
degradation to ~20% small-step cut, at a ±~7% noise floor.*

### 4.3 What this is worth (stated carefully)
- **First genuine positive efficiency result** for the project — not collapse, not a proxy.
- It cuts **whole layers**, so it reduces **weight reads** — the memory-bound floor that KV/token top-K
  could not touch — and **stacks** with the existing ~1.8× token/KV sparsity.
- It is **compute-cut, not throughput** (gating simulates skipping); gsm8k only; number is within-noise.

---

## 5. Contribution summary (for write-up / advisor)

> Naïve fine-tuning of a distilled block-diffusion LLM for dynamic depth collapses generation via the
> teacher-forcing gap (same wall as low-NFE training). We introduce **A1**, an on-policy
> self-distillation loss — roll out the model's own block decoding with depth-reduced small steps and a
> dense commit, and match the frozen original model's predictions along that trajectory. A1 makes trained
> dynamic depth **viable** on block diffusion, exposing a quality/compute knob (~15–20% small-step layer
> compute free on gsm8k, within noise) and composing the block-diffusion-specific recoverable-KV and
> convergence signals that AR layer-skipping cannot express.

---

## 6. Next steps

1. **Rigor before headline:** nail one operating point (target ≈ 0.78, `keep_ss` ≈ 0.86) at **n=300–500
   + a second seed** to beat the ±~7% noise; add a **second task** (math / a LongBench subset).
2. **Hook A (convergence routing):** feed per-token confidence (or step index) to the router so it
   allocates depth by convergence — the block-diffusion-specific lever expected to extend the frontier
   and stabilize the deep points.
3. **Throughput kernel:** convert the ~20% layer-cut into wall-clock (real per-token layer skipping /
   token gather-scatter), then measure decode-phase and e2e speedup stacked with token/KV sparsity.
4. **Scale the training** (batching, multi-block rollout, state subsampling) — the prototype is bs=1,
   1 block/example, on-policy rollout + per-state grad forward.

---

## 7. Honest limitations

- Numbers are **teacher-forced-proxy → generation-validated for accuracy only**; throughput is not yet
  measured.
- **Noise floor ±~7%** at n=100 + training variance; the frontier is "flat within noise," not precisely
  characterized.
- **gsm8k only** so far; **prototype-scale** training (bs=1, 1 block/example).
- The router currently ignores convergence (`t`); the win so far is **per-token/content** routing
  ("MoD-for-diffusion"). The differentiated diffusion-specific story (convergence + recoverable-KV as a
  measured advantage) requires Hook A and the throughput kernel.

---

## 8. Script reference

| script | role |
|---|---|
| `depth_router_derisk.py` | teacher-forced capacity probe + controls (static, random); defines `DepthRouter`, `RoutedLayer`, `wrap_layers`, `set_router_state` (gate modes: `soft`/`ste`/`hard`) |
| `depth_router_gen_eval.py` | installs router into real decode loop (dense prefill+commit, gated small steps); `orig`/`lora_dense`/`routed` gsm8k accuracy + `keep_ss` |
| `depth_router_train_a1.py` | A1 generation-aware training: on-policy rollout + self-distillation to frozen original + budget curriculum |
| `A1_GENERATION_AWARE_LOSS.md` | the loss, in depth (derivation, prior art, open design decisions) |
