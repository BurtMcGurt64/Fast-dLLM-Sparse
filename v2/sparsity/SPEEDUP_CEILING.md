# Speedup Ceiling Analysis — Top-K Sparsity on Fast-dLLM v2

**Bottom line:** Inference-only sparsity on the *fixed* Fast-dLLM v2 checkpoint is
bounded at **~2× end-to-end latency speedup** (measured **1.36×** at B=32/8k today;
projected ~2–2.5× fully stacked). The cap is **structural, not an implementation
bug** — the attention code and gather are correct and efficient. Breaking past it
requires **training-time methods** (few-step / distillation), not more inference
kernels. The "4–5×" that motivated the project was the KV-read kernel measured in
isolation — the narrowest scope, never end-to-end.

---

## The cost model

Per-forward decode cost = **W + KV**:
- **W** ≈ 14.4 GB weight bandwidth, read *every* forward, fixed → the **weight floor**
- **KV** ∝ B×L, read ~2× (cat + flash)

End-to-end = **prefill + Σ(forwards)**:
- **prefill** is compute-bound, grows with context, and is **not reducible by a KV method**
- decode = **~17 forwards per 32-token block** (16 small steps + 1 commit)

Top-K prunes only KV → `frac×KV`. It cannot touch **W**, **prefill**, or the **commit steps**.

---

## Speedup waterfall (each scope adds unprunable work)

| Scope | Speedup | Unprunable at this level |
|---|---|---|
| KV-read kernels only | ~5× | nothing (= 1/frac) |
| One small step (q=8) | 1.89× | weights (W floor) |
| Decode phase (small + commit) | 1.4–1.6× | commit steps read full KV |
| **End-to-end** | **1.1–1.36×** | **prefill (compute-bound, grows w/ ctx)** |

---

## Measured evidence

narrativeqa, `block_cache=True`, frac≈0.20, this repo. Ratios are stable across
runs even though absolute wall-times are warmup/clock-sensitive.

**1. Decode-only speedup rises weakly with batch, then saturates (8k):**

| Batch | decode speedup |
|---|---|
| 8  | 1.41× |
| 16 | 1.41× |
| 32 | 1.59× |

**2. Weight floor dominates at low batch (B=8/8k, context-slope decomposition):**
- W floor = **1.505 ms/tok**, KV read = **1.107 ms/tok** → weights are **58%** of decode
- ⇒ decode ceiling = **1.74×** even with *perfect* KV elimination

**3. Longer context does NOT help e2e (B=8, frac 0.20):**

| ctx | e2e speedup |
|---|---|
| 8k  | 1.23× |
| 16k | 1.24× |

Longer context raises decode KV% but inflates (unprunable) prefill by the same
amount → the two cancel.

**4. Gen-length confirms prefill dilution (B=16/8k):** 64 tok → 1.09×, 128 → 1.20×,
256 → 1.31×, 512 → 1.29×. e2e only converges to decode speedup as generation → ∞.

**5. Best measured e2e:** B=32/8k, 512 tok → **1.36×**.

**6. The implementation is not the bottleneck:**
- Gather cost ∝ **K, not L** (microbench: 0.021 ms flat across 8k/16k/32k at fixed K)
- Commit-step scoring is **free** (same-path baseline = 1.00×)
- Attention forward review: `gather → cat → SDPA` all over the *pruned* K_tok
  length; no full-L operation; correct

**7. Quality is not the limiter:** F1 drop −0.002 to −0.009 at frac 0.20; model holds
to 32k. The ceiling is physical, not accuracy-driven.

---

## The three unprunable components (why e2e is capped)

1. **Weight floor** — read every forward; only partially amortized by batch, saturates by B=32.
2. **Commit steps** — must read the full prefix KV to rescore; ~6% of forwards but cost ∝ L.
3. **Prefill** — compute-bound, grows with context; a KV-sparsity method cannot touch it.

---

## Why every inference axis caps

| Axis | Result | Why capped |
|---|---|---|
| KV pruning (top-K) | ~1.6× decode | weight floor (low B) + frac≈0.20 quality limit |
| Weight quant (INT8/FP8 FFN) | ~1.1–1.35× | fixed win; fades where KV dominates (mirror of top-K) |
| Batch size | 1.41→1.59× | amortizes weights but saturates; memory pressure at B=32 |
| Context length | e2e flat | raises decode KV% but inflates prefill equally |
| NFE (denoising steps) | quality collapses | checkpoint trained on a fixed confidence-threshold schedule |

---

## Projected ceiling, fully stacked

top-K + INT8 + B=32 + 16k + long generation → **~2×, maybe 2.2–2.5× e2e.** Not 3–5×.

---

## What would break past it

Only the **NFE axis via training** (few-step / distillation) has uncapped,
*multiplicative* headroom — it reduces the *number* of times you pay `W + KV`
rather than the cost of each. That is a training project, not an inference kernel.
The KV-sparsity work composes with it (fewer steps × cheaper steps) but cannot
substitute for it.
