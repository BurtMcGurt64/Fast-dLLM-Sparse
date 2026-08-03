# Paper Pitch — KV/Weight Sparsity in Block-Diffusion LLMs

*Working doc, 2026-07-23. Audience: PI + PhD collaborator. Purpose: agree on framing before writeup.*

Working titles (pick a lane):
- **"Selection-Agnostic: Which KV You Keep Doesn't Matter in Block-Diffusion LLMs — and Why"**
- **"The Slack Is Already Gone: Inference-Only Acceleration Limits of Distilled Block-Diffusion LLMs"**

---

## 1. One-paragraph pitch

Diffusion-LLM efficiency papers assume that a *smarter* choice of which KV/blocks to keep buys accuracy at a given budget. We show that in a SOTA block-diffusion model (Fast-dLLM v2, Qwen2.5-7B), **the selection strategy is essentially irrelevant to accuracy** — content-based (commit-score oracle), position-only (StreamingLLM), and a MAGE-style variant are equivalent within noise across 4 retrieval evals × 3 budgets — because retrieval completes in the model's *dense* forwards and the residual content attention is diffuse and redundant. We trace this to a single property: a distilled block-diffusion model has **no exploitable slack**, which also explains why layer-skipping, NFE reduction, and weight pruning all fail on it. We make this quantitative with a **two-model compressibility contrast** (2:4 pruning costs the distilled model 3× more accuracy than a less-distilled one) and contribute the one thing that *does* help — a **coherence-preserving pinning method** that prevents long-generation degeneration under KV sparsity.

## 2. What makes it unique (and why prior work missed it)

- **Nobody runs the position-only baseline.** MAGE / LoSA / Sparse-dLLM propose content-based KV selection for dLLMs but never compare against StreamingLLM-style sink+recency pinning at matched budget. That comparison is the whole result: it reveals the "smart selection" gains are attributable to *keeping any reasonable blocks*, not to the content signal.
- **The mechanism is new:** *first-block-dense* — short-answer retrieval finishes in dense forwards, so KV sparsity structurally can't hurt it; when you *do* sparsify the retrieval block (our MAGE reimpl), accuracy drops 6–10%. "Preservation" is structural, not a property of the clever method.
- **One explanation unifies five closures.** Content≈position, no layer redundancy, few-step NFE, and 2:4-craters all follow from "distillation squeezed out the slack." We turn that from an anecdote into a measured, cross-model claim.

## 3. Main contributions (in order of strength)

1. **The equivalence result.** content ≈ position ≈ oracle KV selection in block diffusion, multi-eval, multi-budget. *[measured]*
2. **The mechanism.** first-block-dense + diffuse/redundant content attention explains *why*. *[measured]*
3. **Compressibility tracks distillation.** two-model 2:4 contrast. *[measured]*
4. **A coherence-preserving method.** plain top-K degenerates long generation; sink+recency pinning fixes it — block-diffusion-specific. *[measured, reconfirm]*

## 4. The evidence we have (measured)

**Selection ablation (v2, LongBench F1, n=60, ctx 8k):** `commit − streaming` ∈ **[−0.011, +0.017]** across {hotpotqa, 2wikimqa, musique, narrativeqa} × frac {5,10,20}% — all within noise, mean ≈ 0. commit retains 93–102% of dense; streaming 86–102%. MAGE-reimpl fires and consistently *underperforms* (arming the retrieval block costs accuracy). Only signal: a faint musique edge at 20% (+0.017) — a modest multi-hop niche, not a headline.

**Two-model 2:4 pruning contrast (Wanda 2:4, GSM8K, n=100):**

| Model | Distillation | dense → 2:4 | drop | retains |
|---|---|---|---|---|
| Fast-dLLM v2 | SOTA-distilled | 0.71 → 0.24 | **−0.47** | 34% |
| LLaDA-8B | less-distilled | 0.76 → 0.60 | **−0.16** | 79% |

Same pruning, ~3× more damage to the distilled model. Corroborated by Layer-Collapse (LLaDA redundancy from overtraining).

**Attention structure (per-step true-softmax):** sink ~40–50% + recency ~30–35% = ~80% positional/deterministic; residual ~20% content is diffuse (frac90 ~18–28% ≈ ~70 blocks) and redundant (plain ≈ streaming ≈ dense). *This is why a smart selector has nothing to catch.*

**Coherence (gov_report 8k, n=30, matched 20% frac):** degeneration rate dense 3/30 ≈ pinned 4/30 ≪ plain 13/30 (mean rep 0.08 / 0.12 / 0.31). *[reconfirm on current checkpoint]*

## 5. The speedup story (honest, labeled by scope)

All vs the **optimized Fast-dLLM v2** baseline (a *hard* baseline — that's a strength):

| Scope | Speedup | Capped by |
|---|---|---|
| per small-step | ~2× (1.9–2.4× w/ batch) | weight floor (~46% of bytes) |
| decode-phase | ~1.6× | commit steps re-read full KV |
| end-to-end (realistic gen) | ~1.3× | prefill (compute-bound) |

**We do NOT claim a large v2 speedup — the data caps it at ~1.3–1.6×.** The honest path to a *bigger* number is composition on a slack-ier model: 2:4 (survives on LLaDA) × KV top-K could clear **>2× on LLaDA**, framed as *"the method's benefit scales with a model's slack, which we characterize"* — not cherry-picking. *[pending, Tier C]*

## 6. Honest caveats (state these before a reviewer does)

- The pinning that works **reduces to StreamingLLM** ported to diffusion — the contribution is **not a novel selection method**; it's the equivalence + mechanism + compressibility finding + the coherence result.
- The two-model contrast is **not fully controlled** (v2 few-shot vs LLaDA zero-shot chat, different arch/gen). Frame as *consistent with* distillation removing slack, not proven causation. LLaDA still drops 16 pts — it's more robust, not immune.
- `mage` is **our reimplementation**, not published MAGE. Frame differences as a mechanism ("arming the retrieval block costs accuracy"), never "published MAGE underperforms."
- Diffusion-specific pruning *mechanism* is already taken (OBS-Diff, Sink-Aware Pruning) — our pruning use is characterization/composition, not a new pruning method.

## 7. Why this is the *right* paper for a saturated field

The dLLM-efficiency space is crowded (dParallel, MAGE, HERALD, ES-dLLM, Sparse-dLLM). A bare "1.6× KV method" competes badly and gets scooped in a cycle. **The characterization is the differentiator** — nobody ran the position baseline, nobody has the compressibility-tracks-distillation contrast. Leaning into the finding is the *more acceptable and more durable* choice, not a retreat. Non-negotiable: a "big-speedup-on-v2 method" paper doesn't honestly exist; everything else (title, emphasis) is negotiable.

## 8. What's left to do

**Core (needed for submission):**
- [ ] Re-run `coherence_eval.py` n=30 on current checkpoint — reconfirm pinned ≈ dense ≪ plain.
- [ ] `full_eval.py` throughput pass — nail the ~2× / ~1.6× / ~1.3× numbers by scope.
- [ ] Writeup: fold results into `main.tex`; cite + differentiate StreamingLLM, MAGE/LoSA/Sparse-dLLM, OBS-Diff, Sink-Aware Pruning, Layer-Collapse, dParallel/T3D, ES-dLLM.

**Strengthening (raises the bar):**
- [ ] 3rd model (Dream) for the compressibility contrast — hardens n=2 → n=3.
- [ ] Controlled pruning comparison (matched prompting/gen) to tighten the two-model claim.
- [ ] Tier C: LLaDA 2:4 × KV top-K composed e2e speedup — the honest >2× number (lit-check LLaDA selection-ablation first, per novelty rule).

**Decision (with PI, before writing):**
- [ ] Agree framing: unified method + characterization, characterization leads. Settle title/emphasis.

## 9. Reproducibility (scripts)

- `v2/sparsity/baseline_compare.py` — selection ablation (dense/streaming/commit/mage).
- `v2/sparsity/prune_probe.py` — Wanda 2:4 on v2.
- `v1/llada/prune_probe_llada.py` — Wanda 2:4 on LLaDA (two-model contrast).
- `v2/sparsity/coherence_eval.py`, `full_eval.py`, `attn_mass_analysis.py` — coherence, throughput, attention structure.
