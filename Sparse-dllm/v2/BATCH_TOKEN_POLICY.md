# Batched Token-Policy (b>1) — Method

Extension of Xinze's token-policy decoding to `batch_size > 1` on Fast-dLLM v2 (block diffusion).
Implemented in `batch_sample_token_policy_flashattention` (`generation_functions.py`); measured by
`run_policy_eval.py`.

## 1. Base method (bs=1) — recap

During block denoising, each of the `block_size` (=32) positions is denoised over multiple small
steps. A position **freezes** once its *final hidden state stops changing across consecutive steps*
(staleness below a threshold) and stays frozen for `token_policy_freeze_step` (=3) steps. Frozen
positions are **not recomputed** — their hidden/logits are reused from cache and their K/V stay in
the block cache (so other tokens still attend to them). Only **active** (non-frozen) positions go
through `forward_block_positions`, which computes them and writes their K/V into the full block cache
via `index_copy_`. This is "sparse queries, full KV" — you compute fewer query positions per step.

Signal = staleness of the token's hidden state across steps. Freeze when settled.

## 2. The b>1 extension — union of active positions

Different sequences freeze different positions, so the active set is **ragged** across the batch. To
compute in one batched forward we take the **union** of active positions:

- **Per-sequence freeze state:** `freeze_steps_left` is `[B, block_size]`; `prev_hidden` is
  `[B, block_size, D]`; `prev_valid` is `[B, block_size]` (has this (seq,pos) been seen yet).
- **Active positions = union across the batch:**
  `active = (freeze_steps_left == 0).any(dim=0).nonzero()` — positions active in **any** sequence.
- **One batched sparse forward:** `block_tokens[:, active]` → `[B, n_union]` through
  `forward_block_positions`; K/V written to the block cache at `active` for all B; `cached_hidden`
  and `cached_logits` updated at `active` for all B.
- **Vectorized staleness** over `[B, n_union]`: for each (sequence, active position), compare current
  vs previous hidden (metric below); freeze where stable **and** previously seen →
  `newly_frozen_mask [B, block]`.
- **Freeze update:** decrement `freeze_steps_left`, then `freeze_steps_left[newly_frozen_mask] = N`.

**Compute skipped per step = positions frozen in *every* sequence** (`block − union`). This is the
**batch-union** property: a position is only skippable if all sequences agree it's settled.

### Correctness
A union position that is active in sequence B but "frozen" in sequence A is still recomputed for A
(it's in the union). That's safe — freezing is an *optimization*, not a correctness requirement, so a
fresh recompute is always valid. Hence **no per-sequence masked writes are needed**; we update all B
at the union positions.

## 3. Staleness metric (swappable): cosine vs L2

Set via `STALENESS_METRIC=cosine|l2` (env) or `set_staleness_metric()`. Per position:
- **cosine:** freeze when `cos(h_t, h_{t-1}) > thr` (angle only; magnitude-blind). thr ~0.99.
- **l2:** freeze when `‖h_t − h_{t-1}‖ / ‖h_{t-1}‖ < thr` (relative change; magnitude-aware). thr ~0.02.

**Finding:** l2 ≫ cosine — l2 is near-lossless at high skip while cosine over-locks (it freezes
tokens whose *direction* is stable but *magnitude* still shifts, which changes attention).

## 4. Skip-fraction accounting
`self._tp_active` / `self._tp_total` accumulate `decoder_tokens_this_step` vs `block_size` per step;
`skip% = 1 − _tp_active/_tp_total` = fraction of positions frozen in all sequences (the batch-union
curve).

## 5. Results (GSM8K, l2 thr=0.02, freeze=3; vs optimized Fast-dLLM v2 baseline)

| bs | speedup | skip% | pol−ORG acc gap |
|---|---|---|---|
| 1 | 1.01× | 47% | 0.016 |
| 8 | 1.13× | 53% | ~0.09 |
| 16 | 1.15× | 49% | ~0.08 |
| 32 | 1.25× | 48% | ~0.06 |
| 64 | **1.37×** | 43% | ~0.03 |

- **Speedup grows with batch** (memory-bound at bs=1 → compute-bound at high batch, where skipping
  compute translates to wall-clock).
- **skip% holds ~50%** — batch-union is mild (sequences freeze correlated positions with a shared
  few-shot prefix).

## 6. Caveats
- **bs=1 gives no win** — the small step is memory-bound (weight loads dominate); skipping compute
  doesn't reduce weight bytes. The win requires the compute-bound regime (high batch).
- **Union ≠ per-sequence saving.** All sequences compute the union, so per-sequence sparsity isn't
  captured. A **ragged/varlen kernel** (each sequence computes only its own active tokens) would give
  more — but efficient ragged batched sparse compute for diffusion is an open problem.
- **Accuracy at bs>1 is ragged-prompt confounded** (ORG itself drops). The pol−ORG *gap* isolates the
  policy's cost; use `--uniform` (uniform-length prompts) + larger n for trustworthy absolute numbers.

## 7. Files
- `generation_functions.py` — `batch_sample_token_policy_flashattention` (batched), metric switch
  (`_STALENESS_METRIC`, `set_staleness_metric`).
- `model/modeling.py` — `forward_block_positions` (sparse active-position forward), `layer_policy_l2`.
- `run_policy_eval.py` — batch sweep driver (accuracy + tok/s + skip%).
