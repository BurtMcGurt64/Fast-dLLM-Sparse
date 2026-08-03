# Group Meeting — 4-Slide Outline

*Tone: calm, numbers-forward, honest update — not a confession. Open line:*
> "I've got an honest update. Since last time I dug into validating the method, and the picture
> changed in an important way — I'd rather we know it now than later. Let me walk through what I
> found, then talk about where we go next."

---

## Slide 1 — Recap + the turn
- **Goal:** commit-score Top-K KV selection for Fast-dLLM v2 (load only the most-attended prefix blocks during denoising).
- **What I did since last time:** ran the baselines the dLLM-KV papers *skip* — position-only (StreamingLLM), oracle, matched budgets, across multiple evals.
- **Result: the picture changed.** Next slides = what I found.

*Say:* "Three weeks ago I was bullish on the selection method. I stress-tested it, and the honest read is different."

---

## Slide 2 — What I found (measured)
1. **Selection is ~irrelevant to accuracy.** content(commit) ≈ position(StreamingLLM) — within noise (Δ ∈ [−0.011, +0.017]) across 4 evals × 3 budgets (v2); a MAGE-style variant (score at the first denoising step) does *worse*. → **the method reduces to StreamingLLM.**
2. **Why (mechanism):** retrieval completes in **dense forwards** ("first-block-dense"); content attention is **diffuse + redundant** (~80% of mass is positional sink+recency). So *which* blocks you keep barely matters.
3. **The one real positive:** coherence-preserving pinning fixes long-gen degeneration — degen rate **dense 3/30 ≈ pinned 4/30 ≪ plain 13/30**.
4. **Two-model pruning contrast:** 2:4 weight pruning **craters v2 (0.71→0.24)** but **LLaDA survives (0.76→0.60)** → compressibility tracks distillation.

---

## Slide 3 — Honest read
- **Not a novel selection method** — it's StreamingLLM (sink+recency) ported to block diffusion.
- **Potential contribution = analysis, not speedup:** the equivalence + first-block-dense mechanism + coherence result + compressibility contrast.
- **Speedup, honest, by scope (vs optimized Fast-dLLM v2):** ~2× per small-step / ~1.6× decode / **~1.3× e2e** — modest, not a headline.
- **Novelty risk (want the group's read):** recent work may already imply the equivalence — HERALD (near-lossless @5–10% budget), "Attention Sinks in DLMs" (robust to masking sinks), Sangam / DiLaServe / dLLM-Serve. Needs a close read to confirm what survives.
- **Caveats:** `mage` = my reimplementation (not published MAGE); equivalence tested on **v2 only** (not LLaDA/Dream); pruning contrast is n=2 + confounded.

---

## Slide 4 — Where to go next (proposal + your input)
- **Option A — Harvest a modest, honest paper.** The two possibly-uncontributed pieces: (1) first-block-dense as an *eval-methodology* point, (2) the coherence-degeneration result. Target workshop/TMLR — *after* a close read of HERALD/Sangam/DiLaServe to lock novelty.
- **Option B — Wind down, pivot toward the lab's strength.** Kernel / hardware co-design (SpInfer / TC-GNN-style, FP4 & tensor cores) — a real moat, less crowded than dLLM serving.
- **My ask:** which fork makes sense? And — since you know this space — **is the equivalence novel, or already known** given the recent serving/KV work?

*Say:* "I have a view — I lean toward [A/B] — but I want your read, especially on the novelty question."
