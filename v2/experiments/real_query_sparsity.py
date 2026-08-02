"""
Compares block-level attention distribution between:
  - zero query tokens (what block_sparsity_sweep.py used)
  - real query tokens (first 32 tokens of a GSM8K answer)

The commit step attends from Q=32 newly-decoded tokens to the prefix KV cache.
Using actual answer tokens as Q gives a more faithful picture of which prefix
blocks matter during real generation.
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

BLOCK_SIZE = 32
Q_LEN      = 32    # commit step query length
SEQ_LENS   = [256, 512, 1024, 2048, 4096]
THRESHOLDS = [0.50, 0.80, 0.90, 0.95]

_DS_CACHE = None

def _get_gsm8k():
    global _DS_CACHE
    if _DS_CACHE is None:
        _DS_CACHE = load_dataset("gsm8k", "main", split="test")
    return _DS_CACHE


def build_prefix_ids(tokenizer, seq_len, device):
    """Fill seq_len tokens with real concatenated GSM8K question text."""
    ds = _get_gsm8k()
    text = " ".join(ds[:200]["question"])
    tokens = tokenizer(text, return_tensors="pt").input_ids[0]
    if len(tokens) < seq_len:
        tokens = tokens.repeat((seq_len // len(tokens)) + 2)
    return tokens[:seq_len].unsqueeze(0).to(device)  # [1, seq_len]


def build_real_query_ids(tokenizer, device, n_examples=8):
    """
    Build Q_LEN query token sequences from real GSM8K answers.
    Takes first Q_LEN tokens of each answer; pads with zeros if answer is short.
    Returns tensor [n_examples, Q_LEN].
    """
    ds = _get_gsm8k()
    rows = []
    for i in range(n_examples):
        answer_text = ds[i]["answer"]
        tokens = tokenizer(answer_text, return_tensors="pt").input_ids[0]
        if len(tokens) < Q_LEN:
            pad = torch.zeros(Q_LEN - len(tokens), dtype=torch.long)
            tokens = torch.cat([tokens, pad])
        rows.append(tokens[:Q_LEN])
    return torch.stack(rows).to(device)  # [n_examples, Q_LEN]


def block_mass_by_position(weights, block_size=32):
    """Mean attention mass per block position, averaged over layers/heads/Q/batch."""
    totals = None
    count = 0
    for w in weights.values():
        B, H, Q, KV = w.shape
        n_blocks = KV // block_size
        if n_blocks == 0:
            continue
        w_trimmed = w[..., :n_blocks * block_size]
        block_mass = w_trimmed.reshape(B, H, Q, n_blocks, block_size).sum(dim=-1)
        mean_mass = block_mass.mean(dim=(0, 1, 2))
        totals = mean_mass if totals is None else totals + mean_mass
        count += 1
    return (totals / count).cpu().tolist() if totals is not None else []


def measure_sparsity(weights, thresholds, block_size):
    all_k = {t: [] for t in thresholds}
    num_blocks = None
    for w in weights.values():
        B, H, Q, KV = w.shape
        n_blocks = KV // block_size
        if n_blocks == 0:
            continue
        num_blocks = n_blocks
        w_trimmed = w[..., :n_blocks * block_size]
        block_mass = w_trimmed.reshape(B, H, Q, n_blocks, block_size).sum(dim=-1)
        sorted_mass = block_mass.sort(dim=-1, descending=True).values
        cumsum = sorted_mass.cumsum(dim=-1)
        for t in thresholds:
            k = (cumsum < t).sum(dim=-1) + 1
            all_k[t].append(k.float().mean().item())
    mean_k = {t: sum(v) / len(v) for t, v in all_k.items() if v}
    return mean_k, num_blocks


def run_one(model, q_ids, past_kv, label):
    """Run a commit-step forward with given Q tokens, print sparsity + top-3 blocks."""
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
    mass = block_mass_by_position(weights, BLOCK_SIZE)

    if num_blocks is None:
        print(f"    [{label}] too short")
        return

    k_strs = "  ".join(f"K@{int(t*100)}%={results[t]:.1f}" for t in THRESHOLDS)
    top3_idx = sorted(range(len(mass)), key=lambda i: mass[i], reverse=True)[:5]
    top3_str = ", ".join(f"blk{i}({mass[i]:.3f})" for i in top3_idx)
    last3 = sum(mass[-3:]) if len(mass) >= 3 else sum(mass)
    first3 = sum(mass[:3])
    print(f"    [{label}]  {k_strs}")
    print(f"      top-5 blocks: {top3_str}")
    print(f"      first-3 mass: {first3:.3f}  last-3 mass: {last3:.3f}  N={num_blocks}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    type=str, default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    parser.add_argument("--device",   type=str, default="cuda:0")
    parser.add_argument("--seq_lens", type=int, nargs="+", default=SEQ_LENS)
    args = parser.parse_args()

    print("Loading model...")
    model = load_model(args.model, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    device = args.device

    # build real query tokens once (reuse across seq_lens)
    real_q = build_real_query_ids(tokenizer, device, n_examples=8)
    zero_q = torch.zeros(1, Q_LEN, dtype=torch.long, device=device)

    for seq_len in args.seq_lens:
        print(f"\n=== seq_len={seq_len} ===")
        torch.cuda.empty_cache()

        # build prefix from real GSM8K text
        prefix_ids = build_prefix_ids(tokenizer, seq_len, device)

        # prefill — batch=1 prefix, then expand for the real-query batch
        with torch.no_grad():
            out = model(
                input_ids=prefix_ids,
                use_cache=True,
                update_past_key_values=True,
                block_size=BLOCK_SIZE,
            )
        past_kv = out.past_key_values

        # zero query (baseline)
        run_one(model, zero_q, past_kv, "zero query (token_id=0)")

        # real answer tokens — run each example separately, average reported separately
        # use first example only for direct comparison, then average
        run_one(model, real_q[:1], past_kv, "real query (1 answer, first 32 tok)")

        # batch of 8 real answers averaged
        try:
            # need to expand past_kv to batch=8
            # easier: just report per-example and print average
            all_mass = []
            all_k = {t: [] for t in THRESHOLDS}
            for i in range(len(real_q)):
                q = real_q[i:i+1]  # [1, Q_LEN]
                enable_attn_capture(model)
                with torch.no_grad():
                    model(
                        input_ids=q,
                        use_cache=True,
                        past_key_values=past_kv,
                        update_past_key_values=True,
                        block_size=BLOCK_SIZE,
                    )
                weights = collect_attn_weights(model)
                disable_attn_capture(model)
                m = block_mass_by_position(weights, BLOCK_SIZE)
                if m:
                    all_mass.append(m)
                res, num_blocks = measure_sparsity(weights, THRESHOLDS, BLOCK_SIZE)
                for t in THRESHOLDS:
                    if t in res:
                        all_k[t].append(res[t])

            if all_mass and num_blocks:
                avg_mass = [sum(all_mass[j][i] for j in range(len(all_mass))) / len(all_mass)
                            for i in range(len(all_mass[0]))]
                top5_idx = sorted(range(len(avg_mass)), key=lambda i: avg_mass[i], reverse=True)[:5]
                top5_str = ", ".join(f"blk{i}({avg_mass[i]:.3f})" for i in top5_idx)
                avg_k_str = "  ".join(f"K@{int(t*100)}%={sum(all_k[t])/len(all_k[t]):.1f}" for t in THRESHOLDS if all_k[t])
                first3 = sum(avg_mass[:3])
                last3 = sum(avg_mass[-3:])
                print(f"    [real query avg over {len(all_mass)} examples]  {avg_k_str}")
                print(f"      top-5 blocks: {top5_str}")
                print(f"      first-3 mass: {first3:.3f}  last-3 mass: {last3:.3f}  N={num_blocks}")
        except torch.cuda.OutOfMemoryError:
            disable_attn_capture(model)
            torch.cuda.empty_cache()
            print("    [real query avg] OOM")


if __name__ == "__main__":
    main()
