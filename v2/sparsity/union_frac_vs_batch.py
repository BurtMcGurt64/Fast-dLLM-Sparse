"""
Union fraction vs. batch size experiment.

At batch size B, TopKBlockLoader takes the *union* of top-K selected blocks
across all B sequences. On diverse documents this union grows toward N,
eliminating the speedup. This experiment measures how fast it grows.

Plots union_frac vs B at fixed frac_target (e.g. top-20% per sequence).
Expected curve: B=1 → ~20%, B=32 → ~90%+. This figure directly motivates
the need for a per-sequence kernel.

Run:
    CUDA_VISIBLE_DEVICES=0 uv run v2/sparsity/union_frac_vs_batch.py \
        --subset narrativeqa --context_len 8192 \
        --batch_sizes 1 2 4 8 16 32 --frac 0.20 --n_examples 64
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../experiments"))

import argparse
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from helper import load_model
from eval_longbench import load_longbench, run_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        type=str, default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    parser.add_argument("--device",       type=str, default="cuda:0")
    parser.add_argument("--subset",       type=str, default="narrativeqa")
    parser.add_argument("--context_len",  type=int, default=8192)
    parser.add_argument("--batch_sizes",  type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--frac",         type=float, default=0.20,
                        help="Per-sequence top-K fraction (K = frac * N_blocks)")
    parser.add_argument("--n_examples",   type=int, default=64,
                        help="Total examples to draw (must be >= max batch size)")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    print("Loading model...")
    model     = load_model(args.model, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # load enough examples for the largest batch
    max_bs = max(args.batch_sizes)
    n_load = max(args.n_examples, max_bs)

    print(f"\nLoading {n_load} examples at ctx={args.context_len}...")
    torch.cuda.empty_cache()
    all_batches_bs1 = load_longbench(
        args.subset, tokenizer, args.device,
        context_len=args.context_len,
        batch_size=1,
        n_examples=n_load,
        seed=args.seed,
    )
    # all_batches_bs1: list of (encoded, answers) each with batch_size=1

    results = {}  # batch_size -> (union_frac, f1)

    for bs in args.batch_sizes:
        print(f"\n{'='*50}")
        print(f"B={bs}  ctx={args.context_len}  frac={args.frac:.0%}")
        print(f"{'='*50}")
        torch.cuda.empty_cache()

        # re-batch the single-example batches into groups of bs
        from transformers import AutoTokenizer as _T
        examples_enc = [enc for enc, _ in all_batches_bs1[:n_load]]
        examples_ans = [ans for _, ans in all_batches_bs1[:n_load]]

        # left-pad and collate into batches of size bs
        orig_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

        batches = []
        for i in range(0, len(examples_enc) - bs + 1, bs):
            chunk_enc = examples_enc[i:i+bs]
            # examples_ans[j] is [answers_for_example_j] from batch_size=1 loads;
            # flatten the outer list so chunk_ans[k] = ['gold1', 'gold2', ...]
            chunk_ans = [ans[0] for ans in examples_ans[i:i+bs]]
            input_ids_list = [e.input_ids[0] for e in chunk_enc]
            max_len = max(t.shape[0] for t in input_ids_list)
            padded = torch.stack([
                torch.cat([
                    torch.full((max_len - t.shape[0],), tokenizer.pad_token_id,
                               dtype=torch.long, device=args.device),
                    t
                ])
                for t in input_ids_list
            ])
            import types as _types
            from transformers.tokenization_utils_base import BatchEncoding
            encoded = BatchEncoding({"input_ids": padded,
                                     "attention_mask": (padded != tokenizer.pad_token_id).long()})
            flat_ans = [a for sublist in chunk_ans for a in sublist]
            # run_eval expects list of (encoded, answers_list_per_batch_item)
            batches.append((encoded, chunk_ans))

        tokenizer.padding_side = orig_side

        try:
            # K = frac * N_blocks per sequence; TopKBlockLoader takes K as integer
            # N_blocks at this context = context_len // 32
            n_blocks = args.context_len // 32
            k = max(1, int(round(args.frac * n_blocks)))
            print(f"  K={k} blocks per sequence (out of ~{n_blocks})")

            f1, tps, n, stats = run_eval(
                model, tokenizer, batches, k=k, device=args.device,
                max_new_tokens=args.max_new_tokens,
            )
            union_frac = stats.get("avg_frac", float("nan"))
            results[bs] = (union_frac, f1, tps)
            print(f"  union_frac={union_frac:.1%}  F1={f1:.3f}  tok/s={tps:.1f}  n={n}")

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  B={bs}: OOM — skip")
            results[bs] = (float("nan"), float("nan"), float("nan"))

    # summary
    print(f"\n\n{'='*50}")
    print(f"Union fraction vs batch size — {args.subset} ctx={args.context_len} frac={args.frac:.0%}")
    print(f"{'='*50}")
    print(f"  {'B':>4}  {'union_frac':>12}  {'F1':>7}  {'tok/s':>8}")
    print("  " + "-" * 38)
    for bs in args.batch_sizes:
        if bs not in results:
            continue
        uf, f1, tps = results[bs]
        print(f"  {bs:>4}  {uf:>11.1%}  {f1:>6.3f}  {tps:>8.1f}")

    # plot
    valid_bs    = [bs for bs in args.batch_sizes if bs in results and not math.isnan(results[bs][0])]
    union_fracs = [results[bs][0] for bs in valid_bs]

    # theoretical union fraction for naive union over independent uniform selections:
    # E[union/N] = 1 - (1 - frac)^B
    all_bs_for_theory = args.batch_sizes  # include OOM'd batch sizes in the curve
    theory_fracs = [1.0 - (1.0 - args.frac) ** b for b in all_bs_for_theory]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(all_bs_for_theory, [f * 100 for f in theory_fracs], 's--', color="#FF5722",
            linewidth=1.5, markersize=5, label="naive union (theoretical)", alpha=0.8)
    ax.plot(valid_bs, [f * 100 for f in union_fracs], 'o-', color="#2196F3",
            linewidth=2, markersize=7, label="per-sequence (measured)")
    ax.axhline(args.frac * 100, color="green", linestyle="--", linewidth=1.5,
               label=f"target ({args.frac:.0%})")
    ax.axhline(100, color="gray", linestyle=":", linewidth=1, label="full prefix (100%)")

    ax.set_xlabel("Batch size (B)")
    ax.set_ylabel("Fraction of prefix blocks loaded")
    ax.set_title(f"Per-sequence vs naive-union KV loading\n"
                 f"{args.subset}  ctx={args.context_len}  K=top-{args.frac:.0%} per sequence")
    ax.set_xscale("log", base=2)
    ax.set_xticks(args.batch_sizes)
    ax.set_xticklabels([str(b) for b in args.batch_sizes])
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(os.path.dirname(__file__),
                            f"union_frac_vs_batch_{args.subset}_ctx{args.context_len}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
