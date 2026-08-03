# Why Higher Batch and Longer Context Don't Break the Ceiling
*The definitive, corrected explanation. Supersedes the batch/context framing in SCALING_ANALYSIS.md
(which predates the memory_vs_compute measurements). All numbers measured, RTX PRO 6000, frac ≈ 0.20.*

---

## The trap that causes all the confusion: "help" means THREE different things

"Does batch/context help?" is ambiguous. There are three speedup metrics, and batch/context hit each
differently. Conflating them is the whole confusion:

| Metric | What it is | Effect of batch | Effect of context |
|---|---|---|---|
| **Per-step speedup** | one small-step forward, dense vs sparse | **grows** (1.9→2.4×) | neutral |
| **End-to-end latency speedup** | full generation wall-clock | helps *modestly*, diminishing | **flat** (cancels) |
| **Throughput** (tok/s) | serving cost metric | **saturates ~B=8–16** | — |

The exciting fact "per-step grows to 2.4×" is TRUE. It just doesn't become a big *end-to-end* win, for
the reasons below.

---

## PART 1 — BATCH

### What batch physically does (two opposite effects)
- **Weights (~13 GB) are read once per forward and *shared* across the whole batch.** So per-sequence
  weight cost *drops* as batch grows — this is amortization.
- **KV cache scales with batch** (each sequence has its own). So KV bytes *grow* with batch.

Net: as batch grows, the byte mix shifts **weight-dominated → KV-dominated.**

### Why per-step top-K speedup GROWS with batch (measured 1.9→2.15→2.4×)
Top-K prunes KV. As batch grows, KV becomes a bigger share of the bytes, so pruning it removes a bigger
share → per-step speedup climbs toward the `1/frac = 5×` asymptote (reached only if weights become
negligible, i.e. infinite batch). **So at the per-step level, batch genuinely helps — a lot.** If this
were the whole story, you'd push batch up and approach 5×.

### Three walls stop that from becoming a big end-to-end win

**Wall 1 — e2e contains unprunable prefill + commit, which *also* scale with batch.**
Top-K only speeds the small steps. Prefill and commit steps are unprunable, and they scale with batch
*just like* the small steps. So they stay a **fixed fraction** of end-to-end no matter the batch — they
permanently dilute the per-step gain. This is the waterfall: per-step **2.15× → e2e 1.36×** at B=32.
Batch can't shrink a fraction that grows *with* batch.

**Wall 2 — throughput saturates at B≈8–16 (measured).**
Throughput (tok/s) rises steeply to B≈8 (476 → 2291), then flattens (2291 → 3352 at B=64) — only **1.5×
for 8× more batch.** Why? The decode kernel becomes bandwidth-*inefficient* at high batch: achieved
bandwidth falls from **64% → 20%** of peak. So the metric that actually sets serving cost stops
improving. You can't *productively* run huge batch.

**Wall 3 — latency rises and you OOM.**
Per-forward time rises **9×** from B=1 to B=64. And B=32/16k OOMs (on prefill activations). So you
physically cannot reach the huge-batch regime where per-step would approach 5×.

### Net for batch
Batch helps end-to-end — but *modestly, with diminishing returns*, and the productive regime is B≈16–32
where e2e is ~1.4×. You can neither reach the batch where per-step nears 5×, nor escape the prefill/
commit dilution once you're there.

---

## PART 2 — CONTEXT

### What context physically does (two effects, opposite signs for the speedup)
- **Decode:** each step reads KV ∝ L. Longer context → *more* KV to prune → top-K saves **more**.
  (Helps the speedup.)
- **Prefill:** processes L tokens, unprunable. Longer context → *more* prefill time. (Hurts the speedup
  — a bigger unprunable chunk.)

### Why they CANCEL (measured e2e: 8k = 1.23×, 16k = 1.24×, flat)
Both the **prunable** part (decode KV) and the **unprunable** part (prefill) grow ~proportionally with L.
So the *fraction* of total time that top-K can prune stays roughly constant → the speedup stays flat.

Concretely: double the context → you double the KV you prune **and** double the prefill you can't. The
gain and the dilution move in lockstep and cancel exactly.

### Why this is DIFFERENT from batch
- **Batch** amortizes the *fixed* weights while growing KV → shifts the mix *toward* the prunable part →
  per-step speedup grows.
- **Context** grows the prunable (decode KV) and the unprunable (prefill) *together* → no shift in the
  mix → speedup flat.

That's why batch helps (a bit) and context does nothing: batch changes the *ratio* of prunable to
unprunable; context scales both sides equally.

### The one context regime that could matter (still bounded)
At **very long context (>~28k)**, prefill attention (O(L²)) starts to dominate prefill and could be cut
by *sparse prefill attention* — but (a) that's a different method (compute, not KV pruning), and (b)
below 28k the prefill **FFN** dominates, so it does nothing at 8k–16k.

---

## PART 3 — THE UNIFIED PICTURE

|  | Per-step | End-to-end | Throughput |
|---|---|---|---|
| **Batch** | grows toward 5× | helps modestly, diminishing, can't push far | saturates ~B=8–16 |
| **Context** | neutral | flat (cancels) | — |

Neither reaches 5×, because:
1. The **unprunable floors** (weights, prefill, commit) never disappear.
2. **Batch can't be pushed** to the regime where those floors are negligible (throughput saturates,
   latency rises 9×, OOM).
3. **Context grows the floors** (prefill) as fast as it grows the prunable part (decode KV).

---

## THE ONE-SENTENCE VERSION
Batch shifts the byte mix toward the prunable KV — so the *per-step* speedup grows toward 5× — but you
can't productively run the huge batch where that would pay off (throughput saturates, latency and memory
blow up), and prefill/commit dilute it to end-to-end anyway; context grows the prunable and unprunable
parts *together*, so it cancels. Either way the unprunable floors cap end-to-end at ~2×.

## The mental model to keep
Picture a forward's time as a bar: **[ weights | prefill | commit | prunable KV ]**.
- Top-K shrinks only the last segment.
- **Batch** shrinks the *weights* segment's per-token share (amortization) → the prunable segment
  becomes relatively bigger → top-K helps more per step. But the bar's *total* keeps growing and the
  other segments grow with it, so end-to-end barely moves.
- **Context** stretches the prefill segment and the prunable-KV segment by the same factor → their ratio
  is unchanged → the speedup doesn't move at all.
