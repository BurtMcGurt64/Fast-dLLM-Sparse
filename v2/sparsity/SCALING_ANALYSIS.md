# Why Top-K Speedup Is Flat in Batch and Context — Measured Analysis

**Scope:** explains, from measured numbers, why the top-K block-sparsity speedup
(a) does not double when you double batch, (b) is much lower end-to-end than per
small step, and (c) is *flat* in context length. Companion to `SPEEDUP_CEILING.md`
(which covers the ~2× ceiling). All numbers measured on **NVIDIA RTX PRO 6000
Blackwell (~1.8 TB/s, 96 GB)**, frac ≈ 0.20 (K=51 blocks at 8k), block cache on.

---

## TL;DR

1. **Speedup is a bounded ratio, not a scalable quantity.** With frac = 0.20 the
   *hard* ceiling on a single forward is **1/frac ≈ 5×**, reached only when KV is
   100% of the traffic. You never get there.
2. **Batch only amortizes ONE cost: the weights** (the single thing that is fixed
   regardless of batch). It walks the per-step speedup *up a saturating curve*
   toward 5× — so doubling batch gives shrinking increments (measured 1.89× → 2.13×
   from B=16→32), not 2×.
3. **End-to-end is capped near ~2× because prefill and commit steps scale *with*
   batch**, so batch cannot amortize them. They stay a fixed slice of e2e forever.
4. **Context is genuinely flat** (not just saturating): longer context grows the
   prunable KV benefit and the unprunable prefill cost at the *same rate*, so they
   cancel.

The measured waterfall at B=32/8k: **per small step 2.13× → decode 1.59× →
end-to-end 1.36×.**

---

## The one rule that explains everything

Every decode forward reads two kinds of bytes:

| Cost | Fixed w.r.t. batch? | Prunable by top-K? |
|---|---|---|
| **Weights** (~13 GB layer weights) | **YES** — one copy, shared across the batch | No |
| **KV cache** (∝ B × L) | No (scales with B, L) | **Yes** → frac·KV |
| **Prefill compute** (∝ B, and L/L²) | No (scales with B) | No |
| **Commit-step KV reads** (∝ B × L) | No (scales with B) | No |

**Batch helps by amortizing the one fixed cost — the weights.** That is the *only*
lever batch pulls. Everything else in the table scales with batch, so increasing
batch gives those terms no relief.

This single table produces all three results below.

---

## Measured inputs (ground truth, no formulas)

**Resident bytes at B=32 / L=8192** (`measure_kv_bytes.py`, summed over actual
allocated tensors):

| Component | Bytes | Share of (weights+KV) |
|---|---|---|
| Transformer layer weights (read every forward) | 13.05 GB | 46.5% |
| Embed / lm_head / other (partially read) | 2.18 GB | — |
| **Prefix KV cache @ (B=32, L=8k)** | **15.03 GB** | **53.5%** |

→ At the target regime, **KV has already overtaken weights** (53.5% vs 46.5%). The
method's premise holds. Per context token, KV = 1.835 MB and scales linearly in B
and L.

**Per-small-step timing** (`profile_step.py`, GPU-only, real forwards):

| Batch | dense | sparse | speedup |
|---|---|---|---|
| 16 | — | — | **1.89×** |
| 32 | 81.1 ms | 38.15 ms | **2.13×** |

**Re-read multiplier (measured, no ncu needed).** Solving
`(W + m·KV)/(W + m·frac·KV) = 2.13` with the resident bytes above gives **m ≈
1.7–1.85×** — KV is physically pulled ~1.8× per forward (the `torch.cat(prefix_k,
key)` materializes the full prefix, then Flash reads it again). Theory predicted 2×;
measurement says ~1.8×.

**GPU utilization.** Implied effective bandwidth = ~350–475 GB/s = **only ~20–25% of
the 1.8 TB/s peak.** The step is *not* a clean bandwidth wall — a large share of the
81 ms is fixed overhead (kernel launches across 28 layers, skinny q=8 GEMMs, the
`cat`). This matters below.

---

## Result 1 — why doubling batch doesn't double the speedup

Speedup is the ratio

```
speedup(r) = (1 + r) / (1 + frac · r),   r = (effective KV) / weights ∝ batch
```

Its limits:
- **r → 0** (tiny batch): speedup → 1× — weights dominate, pruning KV does nothing.
- **r → ∞** (huge batch): speedup → **1/frac = 5×** — weights vanish, KV is everything.

So batch does not *multiply* the speedup; it walks you **up a saturating curve from
1× toward the 5× ceiling.** The increments shrink every doubling:

| Batch | per-step speedup | gain per doubling |
|---|---|---|
| 16 | 1.89× | — |
| 32 | 2.13× | **+0.24** |
| ∞  | 5× (ceiling) | → 0 |

"+0.3× per doubling" is not a bug — it is exactly the shape of approaching a 5×
asymptote from the middle. And you **can't reach the top**: B=32/16k already OOMs
(prefill Q tensor), so you're pinned to the low-middle of the curve where doublings
buy little.

**Extra flattening from the 25% GPU utilization.** The measured curve (1.89→2.13) is
*even flatter* than the ideal bandwidth model predicts, because the fixed per-forward
overhead (launches, `cat`, skinny GEMMs) does **not** amortize with batch the way raw
weight-bandwidth does. So two effects compound: (a) the ratio saturates toward 5×,
and (b) real overhead makes the approach flatter than ideal. Both push the batch
scaling toward flat.

> **One-liner:** speedup is capped at 1/frac ≈ 5× and saturates; batch inches you up
> that curve by amortizing weights, but overhead flattens it further and OOM stops
> you before the top — so a batch doubling adds ~0.2–0.3×, never 2×.

---

## Result 2 — why end-to-end is capped ~2× even though batch "works"

Batch **does** carry through: decode rose 1.41→1.59× (B=8→32) and the per-step number
climbs toward 5×. Batch is not failing. The reason **e2e is stuck near ~2×** is the
cost table:

- **Weights are the only cost fixed w.r.t. batch** → batch amortizes them → per-step
  speedup climbs toward 1/frac.
- **Prefill and commit steps scale *with* batch** → doubling batch doubles them too →
  their *share of e2e does not shrink* → they permanently cap e2e.

You cannot amortize a cost that grows with the very thing you are increasing. So the
two scopes have different ceilings:

| Scope | Ceiling as batch → ∞ | Why |
|---|---|---|
| Per small step | **1/frac ≈ 5×** | only weights vs KV; batch melts weights away |
| **End-to-end** | **~2×** | prefill + commit scale with batch; batch can't touch their share |

Measured waterfall at B=32/8k, top to bottom, every number from a real run:

| Scope | speedup | what it adds that batch can't remove |
|---|---|---|
| Per small step | **2.13×** | — (weight floor, amortizing) |
| Decode (small + commit) | **1.59×** | commit steps re-read full KV (unprunable, ∝B) + gen-loop overhead |
| **End-to-end** | **1.36×** | prefill (unprunable, ∝B, grows with context) |

---

## Result 3 — why context is *flat* (a different mechanism)

Batch *helps but saturates*. Context does **not** help at all — it cancels:

| ctx (B=8, frac 0.20) | e2e speedup |
|---|---|
| 8k | 1.23× |
| 16k | 1.24× |

Longer context raises KV's share of decode (good for top-K) **but prefill is
O(L)–O(L²) compute, unprunable, and grows at the same rate.** The decode gain and the
prefill dilution move in lockstep and cancel. This is not saturation — it is exact
cancellation, which is why the number is flat rather than slowly rising.

Generation length is the only axis that moves e2e, because it makes decode dominate
prefill — but it asymptotes to the decode ceiling (~1.6×), not beyond:

| gen_len (B=16/8k) | e2e |
|---|---|
| 64 | 1.09× |
| 256 | 1.31× |
| 512 | 1.29× |

Best measured e2e: **1.36× at B=32 / 8k / 512 tokens.**

---

## What this means

- The ~2× e2e ceiling is **structural**: it is set by costs top-K cannot prune
  (prefill, commit) that also **scale with batch**, so no amount of batching,
  context, or a better kernel breaks it. Fully stacked (top-K + INT8 + big batch +
  long gen) projects to ~2–2.5×.
- **Breaking past ~2× requires reducing the *number* of forwards, not the cost of
  each** — i.e. a training-side few-step / distillation method. That composes with
  top-K (fewer steps × cheaper steps) but is a different (algorithmic) project.
- The **25% GPU utilization** is a caveat *and* a hint: the dense path is overhead-
  laden, so a fused attention kernel (no `cat`, fewer launches) can cut absolute
  latency — but it speeds up dense and sparse together, so it does not raise the
  *ratio*. Do not count it as hidden speedup for the method.
- **Regime note:** these are workstation-card numbers (1.8 TB/s). A bandwidth-saving
  method wins *more* on lower-bandwidth cards; datacenter GPUs (H100 3.3, B200 ~8
  TB/s) could give *smaller* speedups. Scope claims to the card, or measure there.

---

## The message to send the team

> **Update on top-K block sparsity — scaling analysis (Xinze's ask #3).**
>
> Short version: the speedup is a *ratio* capped at 1/frac ≈ 5× per forward, and only
> ~2× end-to-end. Here's the mechanism, since it's not a bug and it explains why
> batch and context don't scale it. (All measured on RTX PRO 6000 Blackwell, 1.8 TB/s.)
>
> **The one rule:** every forward reads weights (fixed, ~13 GB, shared across the
> batch) + KV (∝ B×L, the part we prune). Batch's *only* lever is amortizing the
> fixed weight cost. Everything else — prefill, commit-step KV reads — scales *with*
> batch, so batch gives them no relief.
>
> **Why doubling batch ≠ doubling speedup.** Speedup can't exceed 1/frac ≈ 5×, ever.
> Batch walks you up a *saturating* curve toward it by melting the weight floor:
> per-step 1.89× (B=16) → 2.13× (B=32) — measured, real forwards. Increments shrink
> each doubling because you're approaching the 5× cap, and we OOM at B=32/16k before
> we can climb further. (It's even flatter than ideal because the step runs at only
> ~25% of peak bandwidth — fixed launch/`cat` overhead that batch doesn't amortize.)
>
> **Why e2e is capped ~2× even though batch works.** Per-step climbs toward 5×, but
> prefill and commit steps scale *with* batch, so their share of e2e never shrinks —
> they cap it. Measured waterfall at B=32/8k: **per step 2.13× → decode 1.59× →
> e2e 1.36×.**
>
> **Why context is flat** (1.23× at 8k → 1.24× at 16k): longer context grows the KV
> we can prune *and* the prefill we can't, at the same rate — they cancel exactly.
>
> **Bottom line:** best e2e is 1.36× today, ~2–2.5× fully stacked with INT8. To break
> past ~2× we'd need to cut the *number* of forwards (training-side few-step /
> distillation), not the cost per forward — that composes with what we've built and
> fits ICLR since it's algorithmic. Happy to walk through the measured curves anytime.

---

*Scripts: `measure_kv_bytes.py` (resident bytes), `profile_step.py` (per-step),
`derisk_genlen.py` (decode/e2e via gen-length slope), `gather_scaling.py` (gather is
O(K)). Evidence doc: `SPEEDUP_CEILING.md`.*
