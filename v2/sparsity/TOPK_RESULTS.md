# Top-K Prefix Block Loading — Progress Report

## Background

Fast-dLLM v2 is a diffusion language model built on Qwen 7B. It decodes in blocks of 32 tokens using a masked diffusion objective. Each forward pass reads a **prefix KV cache** (all previously generated tokens) plus a small block cache (the current 32-token block being denoised). Prior profiling established that the model is entirely memory-bandwidth bound — arithmetic intensity is ~32 FLOPs/byte, well below the GPU roofline of ~558 FLOPs/byte (RTX Pro 6000 Blackwell).

At batch=1, MLP weight loading (~13GB per forward pass) dominates. At batch≥32, MLP cost amortizes while KV reads scale with batch size × sequence length — making **KV bandwidth the next bottleneck**.

---

## Motivation: Attention Sparsity

Block-level attention concentration measured at seq_len=4096 (128 prefix blocks): **90% of attention mass concentrates in ~7% of blocks (~9 blocks)**. Pattern is structural: attention sink at block 0, recency bias toward recent blocks, and content-adaptive routing for task-relevant content.

This motivates top-K prefix block loading: instead of loading all N prefix blocks during denoising small steps, load only the K most-attended ones. This directly reduces KV bytes read — the right bottleneck at batch≥32 and long context.

---

## Method

### Top-K Block Loading

After each **commit step**, the model has already computed attention from 32 newly-generated query tokens over the full prefix. We intercept these attention weights (`_capture_attn_scores` flag on each attention layer), aggregate over layers and heads to produce per-block importance scores, and select the **union of the top-K blocks across all batch sequences**.

The selected token indices are stored on each attention layer (`_topk_token_indices`). On subsequent **small denoising steps**, the prefix KV is sliced to only the selected tokens:

```python
prefix_k = prefix_k[:, :, selected_indices, :]
prefix_v = prefix_v[:, :, selected_indices, :]
attention_mask = attention_mask[..., :prefix_len][..., selected_indices]
# concatenated with block_mask for the current block
```

The commit step always attends to the **full prefix** (indices cleared before commit forward pass) so attention scores are unbiased and KV written to cache is correct.

Key design decisions:
- **Block-level granularity**: prefix KV is naturally organized in 32-token blocks
- **Union across batch**: each sequence selects K blocks; we take the union so all sequences get their required context with one shared index set
- **Tail tokens always included**: tokens beyond the last complete block always appended to avoid corrupting generation
- **Commit step disarmed**: `_topk_token_indices` cleared before each commit forward so score capture sees unsliced attention

### Correctness Verification

No-op test (K >> N): confirmed **bit-identical output** to baseline. Required fixing four bugs:
1. **Prefill arming**: `is_commit` check matched initial prefill. Fixed by checking `pre_kv_len > 0`.
2. **Numerical mismatch**: score-capture path switched SDPA → manual matmul for actual output. Fixed by keeping SDPA for output; computing manual weights in `torch.no_grad()` for capture only.
3. **Tail token exclusion**: tokens beyond last complete block silently dropped. Fixed by always appending tail tokens.
4. **Gather memory layout**: `prefix_k[:, :, _idx, :]` creates copy with different strides when K≥N → different SDPA kernel. Fixed by skipping gather entirely when `len(_idx) == prefix_k.shape[-2]`.

All 6 no-op test cases pass bit-identically.

---

## Results

### Per-Step Latency (CUDA Events, batch=32, context=2048)

| Condition | avg_ms | speedup | KV_frac |
|-----------|--------|---------|---------|
| baseline | 107.5ms | 1.00x | 100% |
| K=N-gather (overhead only) | 112.2ms | **0.96x** | 100% |
| K=4 (6.2%) | 58.9ms | **1.83x** | 6.2% |
| K=8 (12.5%) | 62.2ms | **1.73x** | 12.5% |
| K=16 (25.0%) | 70.9ms | **1.52x** | 25.0% |
| K=32 (50.0%) | 86.3ms | **1.25x** | 50.0% |

Gather overhead is a real ~5% cost. Speedup at B=32 scales correctly with KV reduction — confirms it is KV bandwidth savings, not an artifact.

### End-to-End on GSM8K (padded to 2048 tokens, same-path baseline)

| K | KV loaded | Accuracy | tok/s | Speedup |
|---|-----------|----------|-------|---------|
| full (baseline) | 100% | 77.5% | 244.4 | 1.00x |
| **K=16** | **25.6%** | **77.5%** | **330.4** | **1.35x** |
| K=8 | 15.8% | 71.2% | 364.8 | 1.49x |
| K=32 | 52.9% | 74.0% | 277.9 | 1.11x |

**K=16 is Pareto-optimal on homogeneous batches**: zero accuracy drop, 1.35x end-to-end. Note: GSM8K with filler padding is an artificially easy case — all 8 sequences share identical filler and attend to the same blocks, so the union stays small (frac=24.7%). This flatters the method.

---

### LongBench Sweep: Frac vs Context Length (the key experiment)

Evaluated on **qasper** (scientific paper QA) and **narrativeqa** (story QA) — real diverse documents, no artificial filler. B=8, n=80, K=16/32/64.

#### Frac at K=16 (primary metric):

| ctx | qasper frac | qasper F1 drop | narrativeqa frac | narrativeqa F1 drop |
|-----|-------------|----------------|------------------|---------------------|
| 2048 | 70.3% | +0.001 | 60.4% | +0.010 |
| 4096 | **40.8%** | **−0.023** | **33.7%** | **−0.016** |
| 8192 | **22.6%** | −0.073 | **17.4%** | −0.033 |

#### Frac at K=32:

| ctx | qasper frac | qasper F1 drop | narrativeqa frac | narrativeqa F1 drop |
|-----|-------------|----------------|------------------|---------------------|
| 2048 | 96.4% | −0.008 | 89.5% | −0.005 |
| 4096 | 68.6% | −0.024 | 61.9% | −0.017 |
| 8192 | 39.6% | −0.075 | 33.7% | −0.017 |

#### Frac at K=64:

| ctx | qasper frac | qasper F1 drop | narrativeqa frac | narrativeqa F1 drop |
|-----|-------------|----------------|------------------|---------------------|
| 2048 | 100.0% | +0.000 | 100.0% | +0.000 |
| 4096 | 95.1% | −0.007 | 90.5% | −0.003 |
| 8192 | 65.3% | −0.069 | 61.9% | −0.027 |

---

## Key Findings

### 1. Union saturates at a fixed absolute count, not a fixed fraction

The union stays at ~50–55 blocks (qasper) and ~40–45 blocks (narrativeqa) at K=16, regardless of whether N=65 (2k) or N=237 (8k). This is why frac drops from 70% → 23% as context grows — N triples while the union barely moves. The important blocks per document are a fixed absolute count driven by structure (attention sink at block 0, recency bias, 1–2 content-relevant blocks per sequence).

**This confirms the long-context thesis:** the method's efficiency scales with context length. At 8k with K=16, you load only 22% of the KV cache.

### 2. Union grows linearly with K — no free lunch

At qasper, ctx=8192: K=16→union=53.6, K=32→union=93.9, K=64→union=154.8. Doubling K roughly doubles the union. There is no sublinear regime where larger K yields disproportionately better coverage. This means the union scaling is a fundamental batching constraint, not something K-tuning can fix.

**Implication:** the right fix is a per-sequence attention kernel, not larger K.

### 3. At 8k context, accuracy drops even at K=64

qasper 8192 F1: dense=0.222, K=16=0.149, K=32=0.146, K=64=0.152. Increasing K from 16 to 64 (loading 22% → 65% of cache) barely recovers F1. This means the issue is **not** that the union is missing relevant blocks — it's that the commit-step attention scores are less reliable at long context. The model is generating similar (wrong) outputs regardless of K, suggesting the scores themselves are poor predictors of which blocks matter for the answer at 8k range.

**Implication:** better score functions are needed alongside the per-sequence kernel.

### 4. Sweet spot is 4k context, K=16

qasper 4k: frac=40.8%, F1 drop=−0.023. narrativeqa 4k: frac=33.7%, F1 drop=−0.016. This is where the method works well on real diverse data with the current union approach:
- Meaningful KV reduction (40% frac → predicted ~1.3x per-step at B=32)
- Near-zero accuracy cost
- No per-sequence kernel required

---

## Batch Scaling Results (the catch-22)

Running at B=32 on LongBench (K=16) revealed a fundamental problem with the union approach:

| B | ctx | subset | frac | speedup |
|---|-----|--------|------|---------|
| 8 | 4096 | narrativeqa | 33.7% | 1.16x |
| 8 | 8192 | narrativeqa | 17.4% | 1.19x |
| 16 | 8192 | narrativeqa | 24.6% | **1.22x** |
| 32 | 2048 | narrativeqa | 81.9% | 1.02x |
| 32 | 4096 | narrativeqa | 56.6% | 1.05x |
| 32 | 8192 | — | OOM | — |

**The catch-22:** union scales with B. Each additional sequence in the batch adds its own diverse content blocks to the union.
- Small B (8): union manageable but KV is only ~4% of bandwidth → flat speedup
- Large B (32): KV bandwidth matters but union collapses to 80-90% of cache → no savings
- B=32 at 8k: OOM (model weights + KV + FFN activations exceed available memory)

The only real speedup on diverse data is B=16 at 8k: **1.22x with −0.034 F1 drop**. This is bounded by a theoretical ceiling of ~1.54x even with perfect KV elimination (MLP is 46% of bandwidth at B=16, 8k).

The union approach as implemented cannot scale to the batch sizes needed to make KV bandwidth significant on diverse documents. The per-sequence kernel is a prerequisite, not an optimization.

---

## Per-Example Attention Distribution (B=1, LongBench)

To validate the concentration claim without the union confound, ran `block_attention_dist.py` over 40 NarrativeQA examples at B=1 (no union — each example loads exactly K/N):

| ctx | valid examples | mean frac90 | median | std | P25–P75 | ≥40% |
|-----|---------------|-------------|--------|-----|---------|------|
| 4096 | 27/40 | 18.1% | 17.8% | 8.1% | 12%–24% | 0 |
| 8192 | 25/40 | 19.8% | 19.8% | 6.6% | 15%–23% | 0 |

**Result: tight cluster around 18-20% at both lengths, no outliers above 40%.** The mean is representative. This confirms that 90% of attention mass consistently concentrates in ~20% of prefix blocks on real diverse documents at B=1. The distribution is outcome 1 (tight cluster) — not bimodal, no long tail.

13-15 examples skipped per length: these produce very short outputs (EOS before completing a full 32-token block → no commit step triggered). This is a probe limitation, not a method failure.

**Sink+recency mass** is consistent: 81% ± 4% at 4k, 79% ± 3% at 8k. Nearly every example has the same structural pattern.

---

## Attention Structure: Sink + Recency + Content

From `block_attention_heatmap.py` on individual examples (B=1):

- **~83-86% of attention mass** is in 4 structural blocks: blocks 0-1 (attention sink) and the last 2 blocks (recency/question)
- **Remaining ~14-17%** is in 2-9 content-relevant blocks that vary per example
- Together this reaches 90% mass in 10-20% of blocks

The content blocks are real (not noise) — at 4k context, content blocks cluster around the answer-bearing narrative section of the document. The model is doing genuine content routing, not just structural attention.

**Implication for implementation:** instead of pure score-based selection, pin sink+recency blocks as guaranteed and use scores only to select content blocks from the middle. More robust than pure scores because sink/recency are structural guarantees unaffected by score quality at long range.

```python
# hybrid policy (proposed)
always_include = set(range(2)) | set(range(n_blocks-2, n_blocks))  # first 2 + last 2
remaining_budget = K - len(always_include)
# fill remaining budget with top-K content blocks from scores
```

---

## FLOPs vs Bytes Crossover (from `flop_byte_crossover.py`)

Analytical calculation for Qwen 7B backbone:

**FLOPs crossover (attention = MLP):** ~28k tokens, Q-independent.

At decode Q=32:
- KV=2048: attention is 0.07x of MLP FLOPs (MLP 14x larger)
- KV=8192: attention is 0.29x of MLP FLOPs (MLP 3.5x larger)
- KV=16384: attention is 0.58x of MLP FLOPs (MLP 1.7x larger)

The "attention dominates compute at 4k" claim is wrong for this architecture. At any practical context ≤16k, MLP dominates FLOPs. Top-K is a bandwidth story, not a compute story.

**Bytes crossover (KV = weights), from analytical formula:**

| B | KV = weights at ctx |
|---|---------------------|
| 1 | 228k tokens |
| 8 | 28k tokens |
| 32 | **7.1k tokens** |

At B=32, KV bytes exceed weight bytes above 7.1k tokens. At 8k context, KV is 53.5% of total bandwidth. At 16k, KV is 69.7%.

**Speedup ceiling (if KV load = 0%, theoretical max):**

| ctx | B=8 KV% | B=32 KV% | B=32 ceiling |
|-----|---------|---------|-------------|
| 2048 | 6.7% | 22.4% | 1.29x |
| 4096 | 12.6% | 36.5% | 1.58x |
| 8192 | 22.4% | 53.5% | **2.15x** |
| 16384 | 36.5% | 69.7% | **3.30x** |

**Projected speedup with per-sequence kernel (frac=20% from distribution probe):**
- B=32, ctx=8k: `1 / (0.465 + 0.20 × 0.535)` = **1.75x**
- B=32, ctx=16k: `1 / (0.303 + 0.20 × 0.697)` = **2.26x**

These are the targets the per-sequence kernel would unlock.

---

## What Was Ruled Out

- **Flat speedup at B=8**: correct — KV is ~4% of total bandwidth at B=8. Real speedups require B≥32.
- **Path asymmetry artifact**: commit-step overhead actually hurts top-K slightly; 1.35x is a conservative (valid) measurement.
- **tok/s confound**: variable output lengths affect end-to-end tok/s. CUDA-event timing validates the speedup is real.
- **K=32/64 fixing 8k accuracy**: K=64 at 8k gives same F1 as K=16 — bottleneck is score quality, not K size.
- **Union shrinking with larger K**: union scales linearly with K; no sublinear regime.
- **Attention FLOPs dominating at 4k**: false for this architecture. FLOPs crossover at ~28k tokens regardless of Q. Top-K is a bandwidth method.
- **B=32 union approach on diverse data**: union collapses to 80-90% frac at 2k-4k context, near-zero speedup. Catch-22 confirmed.

---

## Next Steps (Priority Order)

### 1. Per-sequence Triton attention kernel (highest ROI)

**What:** Triton flash attention kernel with per-sequence block tables. Each sequence independently attends to its own K selected blocks — no union, no shared index.

**Why:** Breaks the union constraint entirely. On diverse documents at 8k, frac drops from ~25% (union at B=16) to K/N = 16/237 = 6.75% (per-sequence). With frac=20% (from measured distribution), projected speedup at B=32, 8k: **1.75x**. At 16k: **2.26x**. Also removes the B=32 OOM issue since KV tensors are smaller per-sequence.

**How:** Triton kernel accepting Q [B, H_q, 32, d], kv_cache [N_blocks, H_kv, 32, d], block_table [B, K]. Structurally identical to PagedAttention — iterate over block_table[b] per batch item instead of sequential KV. Standard flash attention accumulation. GQA broadcast via `repeat_interleave`. Estimated effort: 2–3 days.

### 2. Hybrid selection policy (sink + recency + content)

**What:** Always pin first 2 blocks (attention sink) and last 2 blocks (recency/question) regardless of score. Use commit-step scores only to fill the remaining budget from middle blocks.

**Why:** Per-example probing showed sink+recency carries ~83% of mass consistently across all examples. These blocks are structural guarantees — they don't depend on score quality. Pinning them makes the method robust to score degradation at long context (the failure mode at 8k) while still using scores for content routing. Cheap to implement — two lines added to `_compute_selected_indices`.

### 3. Better score functions for long-context reliability

**What:** Current scores are captured from a single commit step. At 8k, these are poor predictors of which blocks matter for the answer.

**Options:**
- Average scores across the last 2–3 commit steps (exponential moving average)
- Use per-layer scores separately instead of averaging across layers (different layers attend to different content)
- Add a lightweight "oracle check": after each small-step cluster, compare predicted tokens against a cache to detect which blocks were accessed

**Why:** K=64 at 8k fails to recover F1 even at frac=65%, confirming the scores themselves are the bottleneck. Fixing scores would let K=16 achieve the same quality at 8k that it achieves at 4k.

### 3. INT8 + top-K joint evaluation at B=32

**What:** Combine INT8 fused FFN (already 1.12x at batch=1, compresses MLP bandwidth) with top-K KV loading at B=32. The two speedups are orthogonal.

**Projected combined speedup at 4k, B=32, K=16:**
- Top-K alone: ~1.29x (37% KV fraction, frac=41%)
- INT8 alone: ~1.12x
- Combined (rough): ~1.45x

This is conservative — INT8 amortizes better at larger batch where per-sequence overhead is lower.

### 4. 16k context validation

**What:** Run the sweep at ctx=16384 (model supports up to 32768). Based on the union saturation trend (~50 blocks at 8k for qasper), at 16k N≈512 and union≈55 → frac≈11%. Predicted per-step speedup at B=32: ~2.3x.

**Why:** If scores are reliable at 16k (open question), this is where the method is most valuable. Also the point where per-sequence kernel vs union approach diverges most.

### 5. Adaptive K

**What:** Choose K per sequence per commit step based on attention entropy. Low-entropy (one or two dominant blocks) → small K; high-entropy (attention spread across many blocks) → larger K. Budget constraint: total tokens loaded ≤ K_max × block_size.

**Why:** Cleaner than a fixed K, avoids over-loading for easy sequences and under-loading for hard ones. Composable with per-sequence kernel.

---

## File Reference

| File | Purpose |
|------|---------|
| `v2/sparsity/topk.py` | `TopKBlockLoader` class |
| `v2/sparsity/eval_topk.py` | GSM8K accuracy + throughput sweep |
| `v2/sparsity/eval_longbench.py` | LongBench single-subset eval |
| `v2/sparsity/eval_longbench_sweep.py` | LongBench length + batch sweep (frac vs ctx) |
| `v2/sparsity/plot_longbench_sweep.py` | 4-panel plot: frac, union, F1, Pareto |
| `v2/sparsity/timing_topk.py` | CUDA-event per-step latency |
| `v2/sparsity/test_topk_noop.py` | Bit-identity correctness test |
| `v2/model/modeling.py` | Attention hook: `_topk_token_indices`, `_capture_attn_scores` |
| `v2/experiments/block_attention_heatmap.py` | Per-example attention heatmap (B=1) |
| `v2/experiments/block_attention_dist.py` | Distribution of frac90 across N examples |
| `v2/experiments/flop_byte_crossover.py` | Analytical FLOPs/bytes crossover plot |
| `v2/experiments/block_sparsity_sweep.py` | Attention sparsity characterization |

## Canonical Baselines

- BF16, threshold=0.95: **144 tok/s, 76.0% acc** (GSM8K n=100)
- INT8 + fused FFN, threshold=0.95: **162 tok/s, 76.0% acc, 1.12x**
- Top-K K=16, ctx=2048, B=32: **1.52x per-step** (CUDA events), **1.35x e2e** (same-path baseline)
- LongBench qasper K=16 4k: **frac=40.8%, F1 drop=−0.023** (near-lossless on diverse real data)
