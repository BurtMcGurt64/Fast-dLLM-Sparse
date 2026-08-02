"""
Per-step latency timing for top-K prefix block loading.

Uses CUDA events to measure wall time for individual small-step forward passes,
bypassing the tok/s confound (variable output length, batch padding effects).

Conditions measured:
  baseline     — no loader, SDPA on full prefix (reference)
  K=N-gather   — loader armed with all tokens, but gather FORCED (same code path as K<N,
                 but loads everything) — isolates gather overhead from KV reduction
  K=4/8/16/.. — loader armed with K blocks, actual pruning

If K=4 and K=16 give the same speedup over baseline, the speedup is from gather/mask
overhead, not KV bandwidth. If speedup scales with 1 - (K/N), KV reduction is real.

Run:
    CUDA_VISIBLE_DEVICES=2 uv run v2/sparsity/timing_topk.py --context_len 2048 --batch_size 8
    CUDA_VISIBLE_DEVICES=2 uv run v2/sparsity/timing_topk.py --context_len 2048 --batch_size 32
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../experiments"))

import argparse
import torch
from transformers import AutoTokenizer

from helper import load_model
from topk import TopKBlockLoader

BLOCK_SIZE = 32
_FILLER    = "The quick brown fox jumps over the lazy dog. " * 400


# ---------------------------------------------------------------------------
# Prefix cache
# ---------------------------------------------------------------------------

def build_prefix_cache(model, tokenizer, device, context_len, batch_size):
    """Prefill with filler text to get a realistic prefix KV cache."""
    ids = tokenizer(_FILLER, return_tensors="pt").input_ids[0]
    if len(ids) < context_len:
        ids = ids.repeat((context_len // len(ids)) + 2)
    ids = ids[:context_len].unsqueeze(0).expand(batch_size, -1).to(device)

    with torch.no_grad():
        out = model(
            input_ids            = ids,
            use_cache            = True,
            update_past_key_values = True,
            block_size           = BLOCK_SIZE,
        )
    prompt_len = ids.shape[1]
    n_blocks   = prompt_len // BLOCK_SIZE
    print(f"  prefix: {prompt_len} tokens, {n_blocks} complete blocks, "
          f"tail={prompt_len - n_blocks * BLOCK_SIZE} tokens")
    return out.past_key_values, prompt_len, n_blocks


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def time_forward(model, dummy_ids, prefix_kv, n_warmup, n_measure):
    """
    Time n_measure individual small-step forwards after n_warmup warm-up passes.
    Returns (mean_ms, min_ms, all_ms).
    dummy_ids: [B, BLOCK_SIZE] input for the small step.
    """
    for _ in range(n_warmup):
        with torch.no_grad():
            model(
                input_ids              = dummy_ids,
                use_cache              = True,
                past_key_values        = prefix_kv,
                update_past_key_values = False,
                block_size             = BLOCK_SIZE,
            )
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_measure)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(n_measure)]

    for s, e in zip(starts, ends):
        s.record()
        with torch.no_grad():
            model(
                input_ids              = dummy_ids,
                use_cache              = True,
                past_key_values        = prefix_kv,
                update_past_key_values = False,
                block_size             = BLOCK_SIZE,
            )
        e.record()

    torch.cuda.synchronize()
    ms_list = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return sum(ms_list) / len(ms_list), min(ms_list)


def arm_and_time(model, dummy_ids, prefix_kv, indices, n_warmup, n_measure):
    """Arm _topk_token_indices directly on attention modules, time, then clear."""
    for layer in model.model.layers:
        layer.self_attn._topk_token_indices = indices
    try:
        avg, mn = time_forward(model, dummy_ids, prefix_kv, n_warmup, n_measure)
    finally:
        for layer in model.model.layers:
            layer.self_attn._topk_token_indices = None
    return avg, mn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str, default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    parser.add_argument("--device",      type=str, default="cuda:0")
    parser.add_argument("--context_len", type=int, default=2048)
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--k_values",    type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--warmup",      type=int, default=8)
    parser.add_argument("--measure",     type=int, default=40)
    args = parser.parse_args()

    print("Loading model...")
    model     = load_model(args.model, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    device    = args.device

    print(f"\nBuilding prefix cache (context_len={args.context_len}, batch={args.batch_size})...")
    prefix_kv, prompt_len, n_blocks = build_prefix_cache(
        model, tokenizer, device, args.context_len, args.batch_size
    )

    # dummy small-step input: one block of 32 masked tokens
    dummy_ids = torch.zeros(args.batch_size, BLOCK_SIZE, dtype=torch.long, device=device)

    results = {}

    # ------------------------------------------------------------------
    # 1. True baseline — no loader, SDPA on full prefix
    # ------------------------------------------------------------------
    print("\n[1] Timing baseline (no loader)...", flush=True)
    avg, mn = time_forward(model, dummy_ids, prefix_kv, args.warmup, args.measure)
    results["baseline"] = dict(avg=avg, mn=mn, loaded=n_blocks, total=n_blocks)
    print(f"    avg={avg:.2f}ms  min={mn:.2f}ms")

    # ------------------------------------------------------------------
    # 2. K=N-gather — loader code path, but ALL tokens in the index
    #    (forces the gather + mask-slice path with no actual KV reduction)
    #    Use prompt_len-1 so len(_idx) < prefix_k.shape[-2] is satisfied
    #    and the gather branch executes. Measures pure overhead of gather/mask.
    # ------------------------------------------------------------------
    print("\n[2] Timing K=N-gather (all tokens, gather forced)...", flush=True)
    full_idx = torch.arange(prompt_len - 1, dtype=torch.long, device=device)
    avg, mn = arm_and_time(model, dummy_ids, prefix_kv, full_idx, args.warmup, args.measure)
    results["K=N-gather"] = dict(avg=avg, mn=mn, loaded=n_blocks, total=n_blocks)
    print(f"    avg={avg:.2f}ms  min={mn:.2f}ms  (loaded {n_blocks}/{n_blocks} blocks)")

    # ------------------------------------------------------------------
    # 3. K values — select first K complete blocks as representative indices
    #    (avoids running full generation; selection pattern approximates real)
    # ------------------------------------------------------------------
    for k in args.k_values:
        k_actual = min(k, n_blocks)
        # select first k_actual complete blocks (approximates attention-sink + recency pattern)
        token_indices = list(range(k_actual * BLOCK_SIZE))
        # always include tail tokens (partial block beyond last complete block)
        tail_start = n_blocks * BLOCK_SIZE
        if tail_start < prompt_len:
            token_indices.extend(range(tail_start, prompt_len))
        idx = torch.tensor(sorted(set(token_indices)), dtype=torch.long, device=device)
        frac = k_actual / n_blocks

        print(f"\n[K={k}] Timing ({k_actual}/{n_blocks} blocks = {frac:.1%})...", flush=True)
        avg, mn = arm_and_time(model, dummy_ids, prefix_kv, idx, args.warmup, args.measure)
        results[f"K={k}"] = dict(avg=avg, mn=mn, loaded=k_actual, total=n_blocks)
        print(f"    avg={avg:.2f}ms  min={mn:.2f}ms")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    base_avg = results["baseline"]["avg"]
    base_mn  = results["baseline"]["mn"]

    print(f"\n{'Condition':>14}  {'avg_ms':>8}  {'min_ms':>8}  "
          f"{'speedup(avg)':>13}  {'speedup(min)':>13}  {'KV_frac':>8}")
    print("-" * 76)
    for label, r in results.items():
        frac_str = f"{r['loaded']}/{r['total']} ({r['loaded']/r['total']:.1%})"
        print(f"  {label:>12}  {r['avg']:>8.2f}  {r['mn']:>8.2f}  "
              f"  {base_avg/r['avg']:>10.3f}x  {base_mn/r['mn']:>10.3f}x  "
              f"  {frac_str}")

    print(f"\nbatch={args.batch_size}  context={args.context_len}  "
          f"warmup={args.warmup}  measure={args.measure}")
    print("speedup(min) is more reliable — min removes queueing noise.")
    print("If speedup scales with 1-KV_frac → KV bandwidth is the bottleneck.")
    print("If speedup is flat → overhead (gather/mask) dominates, not KV savings.")


if __name__ == "__main__":
    main()
