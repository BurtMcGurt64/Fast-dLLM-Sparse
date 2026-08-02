"""
Block-level attention sparsity sweep across seq_len x batch_size.

Runs a prefill pass to build a real prefix KV cache, then runs a commit-step
forward (Q=32) with attn capture enabled, and reports how many of the N prefix
blocks are needed to capture X% of attention mass.

This directly answers: at what context lengths does top-K block loading become
useful, and how sparse is attention at those lengths?
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from helper import (
    load_model, enable_attn_capture, disable_attn_capture, collect_attn_weights
)

BLOCK_SIZE  = 32
Q_COMMIT    = 32   # commit step: 32 query tokens attend to full prefix
SEQ_LENS    = [256, 512, 1024, 2048, 4096]
BATCH_SIZES = [1, 4, 8, 16, 32]
THRESHOLDS  = [0.50, 0.80, 0.90, 0.95]

_TEXT_CACHE = None

def _get_long_text(tokenizer, target_len):
    """
    Build a token sequence of exactly target_len tokens from real GSM8K text.
    Repeats the corpus if needed so the prefix is always real text, not zeros.
    """
    global _TEXT_CACHE
    if _TEXT_CACHE is None:
        ds = load_dataset("gsm8k", "main", split="test")
        # concatenate all questions into one long string
        _TEXT_CACHE = " ".join(ds[:200]["question"])

    tokens = tokenizer(_TEXT_CACHE, return_tensors="pt").input_ids[0]

    if len(tokens) < target_len:
        repeats = (target_len // len(tokens)) + 2
        tokens = tokens.repeat(repeats)

    return tokens[:target_len]


def block_mass_by_position(weights, block_size=32):
    """Returns mean attention mass per block position, averaged over layers/heads/Q/batch."""
    totals = None
    count = 0
    for w in weights.values():
        B, H, Q, KV = w.shape
        n_blocks = KV // block_size
        if n_blocks == 0:
            continue
        w_trimmed = w[..., :n_blocks * block_size]
        block_mass = w_trimmed.reshape(B, H, Q, n_blocks, block_size).sum(dim=-1)
        # mean over B, H, Q → [n_blocks]
        mean_mass = block_mass.mean(dim=(0, 1, 2))
        if totals is None:
            totals = mean_mass
        else:
            totals = totals + mean_mass
        count += 1
    return (totals / count).cpu().tolist() if totals is not None else []


def build_prefix_cache(model, tokenizer, seq_len, batch_size, device):
    """Prefill with real GSM8K text to populate the prefix KV cache."""
    tokens = _get_long_text(tokenizer, seq_len)
    # expand to batch
    input_ids = tokens.unsqueeze(0).expand(batch_size, -1).contiguous().to(device)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            use_cache=True,
            update_past_key_values=True,
            block_size=BLOCK_SIZE,
        )
    return out.past_key_values


def measure_sparsity(weights, thresholds, block_size):
    """
    Given weights {layer_idx: tensor [B, H, Q, KV]}, compute mean K per threshold.
    Averages across layers, heads, query positions, and batch items.
    Returns ({threshold: mean_K}, num_blocks).
    """
    all_k = {t: [] for t in thresholds}
    num_blocks = None

    for w in weights.values():
        B, H, Q, KV = w.shape
        n_blocks = KV // block_size
        if n_blocks == 0:
            continue
        num_blocks = n_blocks

        # sum attention mass within each block
        w_trimmed = w[..., :n_blocks * block_size]
        block_mass = w_trimmed.reshape(B, H, Q, n_blocks, block_size).sum(dim=-1)
        # [B, H, Q, n_blocks]

        sorted_mass = block_mass.sort(dim=-1, descending=True).values
        cumsum = sorted_mass.cumsum(dim=-1)

        for t in thresholds:
            # minimum K such that top-K blocks cover >= t of attention mass
            k = (cumsum < t).sum(dim=-1) + 1  # [B, H, Q]
            all_k[t].append(k.float().mean().item())

    mean_k = {t: sum(v) / len(v) for t, v in all_k.items() if v}
    return mean_k, num_blocks


def run_sweep(model, tokenizer, device, batch_sizes, seq_lens):
    threshold_headers = "  ".join(f"K@{int(t*100)}%" for t in THRESHOLDS)
    header = f"{'batch':>6}  {'seq_len':>8}  {'N_blk':>6}  {threshold_headers}  {'K@90%/N':>8}"
    print(header)
    print("-" * len(header))

    for seq_len in seq_lens:
        print(f"  --- seq_len={seq_len} ---")
        for batch_size in batch_sizes:
            torch.cuda.empty_cache()
            try:
                past_kv = build_prefix_cache(model, tokenizer, seq_len, batch_size, device)

                # commit step: Q=32 zeros (content doesn't matter for this structural measurement)
                q_ids = torch.zeros(batch_size, Q_COMMIT, dtype=torch.long, device=device)

                enable_attn_capture(model)
                with torch.no_grad():
                    model(
                        input_ids=q_ids,
                        use_cache=True,
                        past_key_values=past_kv,
                        update_past_key_values=True,
                        block_size=BLOCK_SIZE,
                    )
                weights = collect_attn_weights(model)
                disable_attn_capture(model)

                results, num_blocks = measure_sparsity(weights, THRESHOLDS, BLOCK_SIZE)
                if num_blocks is None:
                    print(f"  {batch_size:>6}  {seq_len:>8}  (prefix too short)")
                    continue

                k_strs = "  ".join(f"{results[t]:>6.1f}" for t in THRESHOLDS)
                k90_frac = results.get(0.90, float("nan")) / num_blocks
                print(f"  {batch_size:>6}  {seq_len:>8}  {num_blocks:>6}  {k_strs}  {k90_frac:>7.1%}")

                # only print position distribution for batch=1 to keep output readable
                if batch_size == 1:
                    mass = block_mass_by_position(weights, block_size=BLOCK_SIZE)
                    top3_idx = sorted(range(len(mass)), key=lambda i: mass[i], reverse=True)[:3]
                    top3_str = ", ".join(f"blk{i}({mass[i]:.2f})" for i in top3_idx)
                    last_mass = sum(mass[-3:]) if len(mass) >= 3 else sum(mass)
                    print(f"  {'':>6}  {'':>8}  top-3 blocks: {top3_str}  last-3 mass: {last_mass:.2f}")

            except torch.cuda.OutOfMemoryError:
                disable_attn_capture(model)
                torch.cuda.empty_cache()
                print(f"  {batch_size:>6}  {seq_len:>8}  OOM")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str, default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    parser.add_argument("--device",      type=str, default="cuda:0")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=BATCH_SIZES)
    parser.add_argument("--seq_lens",    type=int, nargs="+", default=SEQ_LENS)
    args = parser.parse_args()

    print("Loading model...")
    model = load_model(args.model, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    run_sweep(model, tokenizer, args.device, args.batch_sizes, args.seq_lens)


if __name__ == "__main__":
    main()
