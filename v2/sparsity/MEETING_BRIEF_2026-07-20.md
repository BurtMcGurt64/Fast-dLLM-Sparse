# Sparsity on Fast-dLLM v2 — Meeting Brief (2026-07-20)

## Bottom line
Inference-only sparsity on this checkpoint is **structurally capped**: ~**1.30× shippable**
today, ~**1.68× decode-only ceiling** (top-K alone, measured), ~**2× absolute** only when
stacked with INT8 in the best regime. The one lever that would approach the ceiling — long
generation — is **triply blocked** on this checkpoint. Getting **past 2×** requires reducing
the number of forward passes, which is a **training** contribution and is now largely
**published** (dParallel ICLR'26, T3D). Decision needed on contribution bar + direction.

## The long-generation question from last meeting — answered with data
Held frac=20% constant, forced generation length, B=16 / 8k / narrativeqa (clean:
climb = prefill amortization only):

| gen len | 64 | 256 | 512 | 1024 | 2048 | decode-only |
|---|---|---|---|---|---|---|
| speedup | 1.09× | 1.29× | 1.30× | 1.37× | 1.55× | **1.68×** |

- **Longer generation helps** — the intuition was correct; the curve climbs as predicted.
- But **shippable (coherent, ≤512 tok) ≈ 1.30×**; 1.55× @ 2048 is forced/incoherent output.
- Asymptote **1.68× (top-K alone)**, consistent with the ~1.6× decode ceiling. INT8 stacking
  needed to approach 2× (the weight half top-K can't touch).

## Why inference-only is capped — four independent measurements
1. **Byte split** (B=32/8k): weights 46.5% (unprunable by KV top-K), KV 53.5%. Pruning KV to
   zero → ~2×. That's the wall.
2. **Verification is architecturally self-confirming**: bidirectional in-block attention →
   commit logits rubber-stamp draft errors (detection recall ~7–16%, ≈ false-alarm rate).
   Held-out complementary-mask verifier also blind. → cheap speculative/verify is dead.
3. **Schedules are load-bearing**: sub-block serialization protects quality; removing it
   (sbs=32, th≤0.6) costs −19% acc. Threshold's whole prize = 1.57× @ −6% acc.
4. **Confidence is at the information ceiling**: learned probe on the hidden state (AUC 0.735)
   is *worse* than raw p (0.825); p already passes through the trained lm_head. No decode-time
   signal beats it. → learned unmasking policy on frozen backbone is dead.

## The long-gen regime is triple-blocked (why the ceiling isn't reachable in practice)
1. **Memory**: prefill FFN activation spike OOMs at high batch/context (needs chunked prefill).
2. **Length**: Qwen2.5-7B-Instruct base doesn't generate long CoT — it's an instruct, not a
   reasoning, model. Off-distribution: the papers serve at ~2k context on reasoning benchmarks.
3. **Quality**: coherence collapses past ~500 tokens → the forced-gen speedup is not shippable.

## Options (need your read on the bar + venue + timeline)
- **A. Rigorous ~2× systems/measurement paper on the current model.** No model switch.
  Achievable; integration-flavored; much of the neighborhood published in the last 12 months
  (MAGE, LoSA, HERALD, dLLM-Serve, Sparse-dLLM, SimSD).
- **B. Switch to a long-generation-capable block-diffusion model.** Makes the long-gen speedup
  *shippable* (removes blocks 2–3); **still caps at ~2×** (block 1 / structural). Cost: weeks
  to re-port infra (non-standard cache API, commit-score capture, top-K patch) to a new arch.
- **C. Training axis (consistency / certainty-forcing distillation).** The only **>2×** lever
  (reduces forward passes, multiplies with sparsity). But dParallel/T3D already published the
  core idea on general/semi-AR dLLMs; ML-algorithms scope.

## Ask
Is a rigorous ~2× (measured, honest) the target for our venue, or does the bar require >2×?
If >2×, that's axis C (training) — do we adopt it, change checkpoint (B), or adjust scope?
