# Fast-dLLM v2 Sparsity — Work Summary & Direction Decision
*Eddie — for the Monday meeting. All numbers measured on RTX PRO 6000 Blackwell (1.8 TB/s), frac ≈ 0.20.*

## Bottom line
Top-K block loading works and is correctly implemented, but **inference-only speedup on this
checkpoint is structurally capped at ~2× end-to-end** (best measured 1.36×; ~2–2.5× fully
stacked). The only lever that breaks past 2× is reducing the *number* of forward passes (NFE) —
which is a **training-time / algorithmic** contribution, not a systems one, and a naive attempt
mode-collapses the model. **Decision needed:** target a rigorous ~2× systems serving system, or
rethink scope (checkpoint / timeline / added people).

---

## 1. What was built and validated
- **PagedAttention top-K kernel**: loads only **19–25% of KV blocks** per forward. **No accuracy
  drop** (LongBench F1 −0.002 to −0.009; GSM8K flat). Correctness verified vs dense.
- **Method**: commit-step attention scores select the top-K prefix blocks; small steps load only
  those. Per-sequence gather (no union explosion at B=32). Gather is O(K), not O(L).

## 2. Core finding: the ~2× ceiling (the logic, not a bug)
Every decode forward reads two things: **model weights** (~13 GB, fixed, shared across the batch,
**not** prunable) and **KV cache** (∝ batch×context, the part top-K prunes). Measured resident
bytes at B=32/8k: **weights 13.1 GB (46.5%), KV 15.0 GB (53.5%)**.

The speedup is a *ratio capped at 1/frac ≈ 5× per forward*, and it dilutes as you zoom out because
each larger scope adds unprunable work:

| Scope | Measured speedup (B=32/8k) | What it adds that can't be pruned |
|---|---|---|
| Per small step | **2.13×** | weight floor (~half the bytes) |
| Decode phase | **1.59×** | commit steps re-read full KV |
| End-to-end | **1.36×** | prefill (compute-bound) |

**Why batch doesn't rescue it:** batch only amortizes the *fixed weight* cost, walking the per-step
number up a saturating curve toward 5× (measured 1.89× → 2.13× from B=16→32 — shrinking gains).
Prefill and commit steps scale *with* batch, so batch can't shrink their share → e2e stays ~2×.
**Why context doesn't help:** longer context grows the prunable KV *and* the unprunable prefill at
the same rate — they cancel (measured e2e 8k=1.23× vs 16k=1.24×, flat).

*(The "5×" that motivated the project was the KV-read kernel measured in isolation — never e2e.)*

## 3. Methodology (how these were measured, not assumed)
- **Byte split**: summed actual allocated tensors (weights vs KV cache), not a formula.
- **Per-step speedup**: real dense-vs-sparse forwards, CUDA-event timing; KV re-read multiplier
  (~1.8×) back-derived from timing + resident bytes — hardware-consistent.
- **Decode isolation**: differenced out prefill via a generation-length slope.
- **Bugs caught by smoke-checks before trusting results**: batched decoding corrupts ragged prompts
  (caught by bs=1 vs bs=8 mismatch); MATH eval was a bad fixture (non-numeric answers).
- **OOM diagnosis**: B=32/16k OOM is the *prefill FFN activation spike* (~37 GB), **not** KV — so
  it's fixable with chunked prefill, and it does not change the ceiling.

## 4. The only >2× lever: NFE — investigated and (in-scope) closed
Fewer forward passes is the one thing that beats the ceiling (multiplicative, composes with
sparsity). I probed it cheaply:
- **Base model quality-vs-NFE curve (GSM8K)**: accuracy holds to threshold ~0.6 (~1.5× fewer
  forwards at ~7% cost), then **cliffs** (0.5→52%, 0.4→38%, 0.3→32% vs 73.5% ref). So there's a
  gap; the question is whether training can recover it.
- **Naive high-mask LoRA fine-tune**: training loss dropped cleanly (6.2→1.5), but **generation
  collapsed** — quality held at step 25, dead by step 100, degenerating into repeated digits
  (mode collapse). The cliff **never moved** even at the best checkpoint. Classic teacher-forcing
  gap: low training loss, zero generation gain.
- **Conclusion**: getting NFE reduction to work needs a *generation-aware training objective*
  (self-distillation / consistency) — a **novel-loss ML-algorithms project**, out of scope for a
  systems lab, and non-trivial (the naive version fails).

## 5. Where this leaves us
```
systems-side, in-scope        → ~2× ceiling (proven)
>2×                           → NFE reduction → training/algorithm → out of scope
```
This is a genuine bind between what's physically achievable in-scope and a bigger number.

**Achievable systems paper (no training, reuses everything built):** a co-designed serving system
for block-diffusion LLMs — top-K sparsity + **KV eviction/paging** (commit-scores as a free
eviction oracle) + **chunked prefill** (unblocks the OOM long-context regime) + INT8 — delivering a
measured ~2× e2e / higher throughput, plus the first efficiency characterization of this model class.

## 6. Decision I need from the group
1. Is a **rigorous ~2× systems serving system** the target for ICLR? (Achievable in the timeline.)
2. If a bigger number is required, that needs the NFE/training axis — do we **change checkpoint**
   (one where few-step isn't baked in), **add scope/people** (co-own the algorithm work), or
   **adjust timeline/venue**?
3. What's the priority order among eviction / chunked prefill / INT8 for the next ~9 weeks?
```
```
