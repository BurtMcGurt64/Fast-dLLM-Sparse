# Slides: Why Top-K Block Loading Caps Well Below 5× — Full Presentation Script
*Complete slide-by-slide deck for the Monday meeting. Each slide has on-slide text, the graphic to use,
speaker notes, and what to emphasize. Every number is measured on RTX PRO 6000 Blackwell (empirical
peaks: 1.46 TB/s, 379 TFLOP/s), frac ≈ 0.20, unless marked "projected."*

---

## PRESENTATION STRATEGY (read before building)

- **The misconception to kill:** "any model is improvable, so we can get these gains." People slide from
  your *specific* claim (top-K gives 5×) to a *vague prior* (models aren't perfect → room exists).
- **The reframe that stops it:** make the talk about **byte accounting**, not model quality. A forward
  pass must move a fixed set of bytes; top-K removes only *some*. You can't argue "the model's
  improvable" against "the weights are 13 GB and get read every step."
- **Recurring move:** every claim = one *measured* number. Say "every number here is measured" once, up
  front, and let it carry the talk. The original 5× was *projected from a component*; this is *weighed*.
- **The load-bearing sentence the whole deck proves:** *Top-K speeds up one slice — KV reads in the
  small steps. Everything else (weights, prefill, commit) is untouched and dominates end-to-end. So the
  small step gets 2.4× but end-to-end is 1.36×.*
- **Two honesty rules that make you bulletproof:** (1) distinguish **measured 1.36×** from **projected
  ~2×**; (2) anchor "memory-bound" on the direct top-K measurement, not an idealized roofline.

---

## Slide 1 — The claim we tested (a method, not "make the model better")

**On-slide text:**
- Top-K block loading: at each commit step, score prefix blocks by attention; during decode, load only
  the ~20% most-attended KV blocks.
- Hypothesis: ~5× speedup at serving scale (large batch, long context).

**Graphic:** simple diagram — a strip of KV blocks, 20% highlighted "loaded," 80% greyed "skipped."

**Say:** "This is a specific, falsifiable method: prune the KV cache to 20% during decode. Not 'optimize
the model' — a concrete byte-reduction technique. Let me show what it actually delivers and why."

**Emphasize:** anchor the whole talk on the *specific method* so no one drifts to "but models are
improvable." You're testing a claim, not judging the model.

---

## Slide 2 — The result: a 3.6× gap

**On-slide text:**
- Kernel works: 19–25% of blocks loaded, **no accuracy drop** (LongBench F1 −0.002 to −0.009).
- Hoped ~5×. **Measured best end-to-end: 1.36×** (B=32, 8k context, 512 tokens).

**Graphic:** two bars — "Hoped 5×" vs "Measured 1.36×" — with a big gap arrow labeled "where did 3.6× go?"

**Say:** "The method is correct and lossless. But the end-to-end speedup is 1.36×, not 5×. This isn't a
bug — the method works. The *number* is the surprise. The rest of this talk accounts for that gap,
line by line."

**Emphasize:** it's not "the method failed." It works. The gap is the story, and it's fully explained.

---

## Slide 3 — The framework: a decode step's byte budget *(load-bearing reframe)*

**On-slide text:**
- Every decode forward reads two things from memory:
  - **Weights** (~13 GB, fixed — read every step, shared across the batch)
  - **KV cache** (∝ batch × context — the part top-K prunes)
- Runtime is driven by bytes moved, not by "how good the model is."

**Graphic:** a horizontal "byte budget" bar split into two segments: WEIGHTS (fixed) | KV (grows with batch×ctx).

**Say:** "Here's the frame for the whole talk. A forward pass physically moves bytes from memory. Two
kinds: the weights, which you re-read every single step no matter what, and the KV cache. Top-K prunes
KV. That's the entire lever."

**Emphasize:** THIS is the slide that moves the room from "is the model good?" to "what bytes must
move?" Slow down here. You cannot skip reading the weights — that's physics, not a quality question.

---

## Slide 4 — What top-K can and cannot touch

**On-slide text:**
- Top-K prunes **KV bytes only.**
- It physically cannot reduce: **weights**, **prefill**, or **commit steps** (which re-read full KV).

**Graphic:** the byte-budget bar again, with a scissors icon over only the KV segment; weights/prefill/
commit marked "off-limits."

**Say:** "Whatever speedup exists lives entirely inside the KV slice. The weights, the prefill pass, and
the commit steps are off-limits to this method — by construction, not by effort. No amount of tuning
lets top-K touch them."

**Emphasize:** bound the tool. This sets up every "why can't you just…" answer later — because the
answer is always "that's not KV, so top-K can't reach it."

---

## Slide 5 — The measured byte split *(the money slide)*

**On-slide text:**
- Measured (actual allocated tensors), B=32/8k: **weights 13.05 GB (46.5%), KV 15.03 GB (53.5%)**.
- Even pruning KV to **zero** → ~2×. The "5×" (= 1/frac) needs KV = **100%** of bytes (weights = 0) — never happens.

**Graphic:** a stacked bar, weights (grey, ~half) + KV (blue, ~half), with a dashed line showing "prune KV
→ this much left" and the resulting ~2×. Use `D_bytes.png` / `measure_kv_bytes` output.

**Say:** "I measured it — not a formula, the actual tensors. At B=32/8k, weights are 46% of the bytes and
KV is 54%. So even if top-K deleted the *entire* KV cache, you'd get about 2×, because you still read the
weight half. The 5× requires the weights to be zero. They're never zero."

**Emphasize:** the ceiling is *arithmetic*, not a quality judgment. Half your byte budget is untouchable.
This is where "we can get 5×" dies. The killer line: *"which of these 13 GB of weights should top-K
not read?"*

---

## Slide 6 — What I was wrong about *(own it, one clean sentence)*

**On-slide text:**
- The "5×" = the KV read measured **in isolation** (= 1/frac).
- I extrapolated one component to the whole pipeline, assuming weights/prefill/commit were negligible.
  They're half-plus of the work.
- Lesson: measure at the scope you claim (end-to-end), not a component.

**Graphic:** a small funnel — "5× (KV kernel, isolated)" at the top narrowing to "1.36× (end-to-end)" at
the bottom.

**Say:** "Here's my mistake, plainly: the 5× was the KV read timed by itself, and I projected it to
end-to-end as if everything else was free. It isn't. The fix is to measure at the scope you're
claiming."

**Emphasize:** own the *precise* error in one sentence, then move on — don't apologize twice. This builds
credibility for every measured number that follows.

---

## Slide 7 — The waterfall: each scope adds unprunable work

**On-slide text (a table):**
| Scope | Speedup (B=32/8k) | What it adds that top-K can't touch |
|---|---|---|
| Per small step | **2.13×** | weight floor |
| Decode phase | **1.59×** | commit steps re-read full KV |
| End-to-end | **1.36×** | prefill (compute) |

**Graphic:** a descending bar chart / waterfall: 2.13 → 1.59 → 1.36, each drop annotated with the
unprunable cost added. Use `B_waterfall.png`.

**Say:** "The 3.6× didn't vanish — it's accounted for. One small step in isolation: 2.13×. Add the commit
steps that re-read the full KV: down to 1.59×. Add prefill: 1.36×. Every step down is unprunable work
you pick up as you zoom from the kernel to the whole system."

**Emphasize:** this is the concrete accounting of the gap. The top of the waterfall (2.13×, even 2.4×
later) is *not* the deliverable — the bottom (1.36×) is.

---

## Slide 8 — Is it memory-bound? I measured the regime *(pre-empts "push batch → compute-bound")*

**On-slide text:**
- Empirical peaks: **1.46 TB/s, 379 TFLOP/s** → ridge = **260 FLOP/byte**.
- Small step is memory-bound at every batch to 64 (compute never > 21% of peak).
- Direct proof: reducing KV to 20% gives **1.9× (B=16) → 2.15× (B=32) → 2.4× (B=64)** — *grows* with batch.
  Compute-bound would give ~1×.

**Graphic:** two lines vs batch — `topk_x` (rising 1.9→2.4) and `ach_FL%` (flat, low ~10–21%). From
`memory_vs_compute`.

**Say:** "A fair objection: 'at high batch you're not memory-bound.' I measured it. I benchmarked the
GPU's own peak bandwidth and FLOPS — no vendor numbers — and the small step never exceeds 21% of peak
compute, up to batch 64. And the direct test: cutting KV bytes speeds it up *more* as batch grows, 1.9×
to 2.4×. If it were compute-bound, that would be ~1×. So it's memory-bound, and *more* so at high
batch, not less."

**Emphasize:** turn the objection around. Anchor on `topk_x` (direct) — *not* on hitting peak bandwidth
(you don't; there's overhead). If pushed: "the *weights* drift toward compute-bound at high batch — but
that only makes them a *harder* floor to prune, not an easier one."

---

## Slide 9 — "Just use a bigger batch" — throughput saturates

**On-slide text:**
- Per-step gain *grows* with batch — but **throughput saturates ~B=8–16** and **latency rises 9× by B=64**.
- Throughput (tok/s): B=8 → 2291, B=16 → 2455, B=64 → 3352 (only 1.5× for 8× more batch).

**Graphic:** throughput vs batch curve — steep to B≈8, then flat. From `memory_vs_compute` tok/s column.

**Say:** "So push batch to 256? Two problems. Throughput — the thing that matters for serving cost —
flattens by batch 16; you get 1.5× more throughput for 8× more batch. And per-forward latency rises 9×.
So you can't productively reach the huge-batch regime where per-step would be large. There's no free
lunch in batch."

**Emphasize:** this *replaces* the old (wrong) "per-step saturates" claim. Per-step grows; *throughput*
saturates. That's the honest, measured reason batch doesn't rescue you.

---

## Slide 10 — "Just use longer context" — it cancels

**On-slide text:**
- Longer context grows the prunable KV **and** the unprunable prefill at the same rate → they cancel.
- Measured end-to-end (B=8): **8k = 1.23×, 16k = 1.24×** (flat).

**Graphic:** flat line, 8k→16k, ~1.23×. From `C_context_flat.png`.

**Say:** "Longer context gives top-K more KV to prune — but it inflates prefill, which top-K can't touch,
by the same amount. They cancel. This isn't diminishing returns; it's exact cancellation. Context is
not a free lever either."

**Emphasize:** distinct mechanism from batch (cancellation, not saturation). Two different objections,
two different measured refutations.

---

## Slide 11 — The general principle *(the punchline — ends the misconception)*

**On-slide text:**
- Inference-only methods reduce the **cost per forward pass** — bounded by the unprunable floor.
- They **cannot** reduce the **number of forward passes.**
- More requires *fewer passes* = changing decoding = **training** — a different axis.
- "The model is improvable" is **true** — via *training*, not inference tricks.

**Graphic:** two axes drawn as perpendicular arrows: horizontal = "cost per pass (inference — bounded)";
vertical = "number of passes (training — the real lever)". Top-K lives on the bounded axis.

**Say:** "Here's the general law. Any inference-only method can only make each forward cheaper, and
that's floored. It can't reduce how *many* forwards you do — that's baked into the model's decoding, and
changing it means training. So yes, the model is improvable — by training. That's a different tool.
Inference sparsity is a fixed tool with a fixed reach, and the model being improvable doesn't extend
that reach."

**Emphasize:** THIS kills "any model is improvable." Separate the two axes explicitly. You're not saying
the model is perfect — you're saying this *class of method* is bounded, and the unbounded lever is a
different (training) axis. *(Optional analogy: tire pressure helps a bounded amount; doubling the
mileage needs a new engine. "Cars are improvable" is true and irrelevant to what tire pressure does.)*

---

## Slide 12 — Why it generalizes: every inference lever hits the same walls

**On-slide text:**
- INT8/FP8 (weights): ~1.35× / ~1.13×, fades where KV dominates (mirror of top-K).
- FFN activation sparsity: dies on the **batch-union** problem (like KV), weak on SwiGLU.
- FFN pruning / MoE: needs **retraining** → training axis.
- Speculative decoding: redundant — block diffusion already decodes in parallel.

**Graphic:** a small table, each method → its wall (floor / batch-union / retraining / redundant).

**Say:** "It's not that we picked a weak method. *Every* inference-only lever is bounded by the same
three walls: an unprunable floor, the batch-union problem, or the retraining wall. Quantization is the
one that works, and it's ~1.35×. Compose the bounded ones and you get ~2×. No single one blows past it."

**Emphasize:** closes the "but what about method X" whack-a-mole *in advance*. It's a physical law of
the regime, not a one-method weakness.

---

## Slide 13 — Measured vs projected: exactly what we know *(honesty = credibility)*

**On-slide text:**
- **Measured:** top-K alone = **1.36×** e2e. Every scope dilutes. 5× measurably impossible.
- **Projected:** fully stacked (top-K + INT8 + best regime) ≈ **~2×** (range 1.5–2.5×) — *not yet measured e2e.*
- Next: measure the stack directly to replace the projection.

**Graphic:** a number line from 1× to 5×: solid dot at 1.36× "measured," shaded band 1.5–2.5× "projected
ceiling," and 5× crossed out "measurably impossible."

**Say:** "To be precise about what we actually know: the measured number is 1.36×. The ~2× is a
projection — 1.36× plus INT8 plus the best regime — and I haven't measured that stack end-to-end yet.
What *is* airtight is that 5× is impossible: I have the diluting number at every scope. So don't hear
'2× is guaranteed' — hear '1.36× measured, ~2× projected, 5× ruled out.'"

**Emphasize:** claiming exactly as much as the evidence supports is what makes the *whole* deck
trustworthy. This is the opposite of the original 5× (which over-claimed). It's your strongest slide,
not your weakest.

---

## Slide 14 — Next directions *(brief)*

**On-slide text:**
- **In-scope (systems, ~2× realistic):** commit-score **KV eviction** → throughput; **chunked prefill**
  → unblock long-context (OOM is prefill activations, not KV); measure the top-K + INT8 stack.
- **The insight, not a stack of methods:** the commit step is a *free* KV-importance oracle unique to
  block diffusion; diffusion attention is 2× more concentrated than AR.
- **Out-of-scope (the only >2× lever):** reduce the *number* of passes via training — naive attempt
  mode-collapses.
- **Decision for the group:** target a rigorous ~2× built on one insight, or rethink scope.

**Graphic:** none needed, or a simple two-column "in-scope / out-of-scope" split.

**Say:** "So where next. In-scope: eviction for throughput, chunked prefill to unblock long context, and
actually measuring the stack. The *contribution* isn't a pile of methods — it's the one block-diffusion-
specific insight: the commit step is a free importance oracle. Out-of-scope is the only >2× lever —
fewer passes via training, which mode-collapsed on a naive try. The decision I need from the room: is a
rigorous ~2× built on that insight the target, or do we rethink scope?"

**Emphasize:** end on the *decision*, and on *one insight* — not a leaderboard number. Hand the scope
call to the room.

---

### Delivery checklist
- Slides 3→5 are the spine. If you land those three, the rest is confirmation. Rehearse them cold.
- Repeat the frame: *"what bytes must move, and which can top-K remove?"*
- Keep Slide 6 to one sentence. Own it, move on.
- Slide 8 is your armor against "push batch up" — lead with `topk_x`, not the roofline.
- Slide 13 (measured vs projected) is what makes you un-attackable. Do not skip it to look more confident.
- Know these cold: **1.36× / 2.13× / 1.59×**, **weights 13 GB vs KV 15 GB**, **peaks 1.46 TB/s & 379 TFLOP/s
  → ridge 260**, **topk_x 1.9→2.4×**, **context 1.23→1.24×**.
