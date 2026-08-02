# tokenskip — a factorized study of token-skipping in block diffusion

**Not a SureLock/ES-dLLM clone.** We *decompose* token-skipping into independent, swappable
components and measure each one's contribution to quality and **wall-clock** — so we can attribute
where the win (and the quality risk) actually comes from on Fast-dLLM v2 (block diffusion).

## The two things we factorize (Eddie's design)

**1. What we freeze for a locked token** (separable axes):
- `ffn`   — reuse the token's cached FFN output; still recompute its attention.
- `attn`  — token doesn't re-attend (reuse cached attention output); still recompute FFN.
- `both`  — full lock (= SureLock/ES-dLLM behavior).

**2. Staleness metric** (how we detect a token has "settled"):
- `cosine` — angle only, scale-invariant, convention (dLLM-Cache/Learning-to-Cache). Xinze's choice.
- `l2`     — relative L2 `‖ΔK‖/‖K‖`, captures magnitude changes cosine misses (attention is
             magnitude-sensitive). Ablation.
- (later) `kl`, `confidence`, `learned` — for comparison / the predictive-locking follow-up.

A locked token always keeps its **cached K/V** so others can still attend to it; the axes above
only change *what compute we avoid* for that token.

## The ablation grid

`freeze ∈ {ffn, attn, both}` × `metric ∈ {cosine, l2}` × threshold sweep → measure per cell:
FLOP reduction, **wall-clock TPS**, quality (GSM8K / a LongBench task).

## The one measurement rule that makes-or-breaks it

Token-skipping cuts **compute, not weight-loading** → it only helps in the **compute-bound regime**.
SureLock gets 1.3× at N≥256/B≥4 but **0× at N=64/B=1** (memory-bound). **Always measure at B≥4, N≥256**,
or you'll see nothing and wrongly conclude it fails.

## The money plot

FLOP-reduction (x) vs wall-clock-speedup (y), per cell. If a gap appears (50% FLOP → ~1.3× wall-clock),
that's the irregular-access / memory wall, measured by us. If not, we've beaten the family's ceiling.
Honest prior: `ffn`-freeze gives ~all the win; `attn`-freeze likely marginal (test the long-context crossover).

## Files
- `staleness.py`  — pluggable staleness metrics (cosine, l2), per-KV-head, conservative aggregation.
- `skip.py`       — [TODO] lock tracker + freeze-target logic, wired into the v2 small-step forward.
- `instrument.py` — [TODO] FLOP counter + timed TPS sweep + quality harness.

See `NOTES.md` for the load-bearing base-model changes.
