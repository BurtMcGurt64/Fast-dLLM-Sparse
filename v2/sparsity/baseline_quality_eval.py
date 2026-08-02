"""
Baseline quality validation across subsets and context lengths (no sparsity).

Runs the model with full prefix attention (K=None) on LongBench tasks at
multiple context lengths to confirm generation quality doesn't collapse at 8k+.
This is the prerequisite before claiming speedup at long context: if baseline
F1 degrades from 4k→8k, the serving regime is capped at 4k.

Outputs a subset × context_len F1 table.

Run:
    CUDA_VISIBLE_DEVICES=0 uv run v2/sparsity/baseline_quality_eval.py \
        --context_lens 4096 8192 --n_examples 60

    # quick check
    CUDA_VISIBLE_DEVICES=0 uv run v2/sparsity/baseline_quality_eval.py \
        --context_lens 4096 8192 --n_examples 20 --subsets narrativeqa qasper
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../experiments"))

import argparse
import torch
from transformers import AutoTokenizer

from helper import load_model
from eval_longbench import load_longbench, run_eval

DEFAULT_SUBSETS = ["narrativeqa", "qasper", "2wikimqa", "multifieldqa_en"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          type=str, default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    parser.add_argument("--device",         type=str, default="cuda:0")
    parser.add_argument("--subsets",        type=str, nargs="+", default=DEFAULT_SUBSETS)
    parser.add_argument("--context_lens",   type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--batch_size",     type=int, default=1)
    parser.add_argument("--n_examples",     type=int, default=60)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    print("Loading model...")
    model     = load_model(args.model, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # results[subset][ctx_len] = f1
    results = {s: {} for s in args.subsets}

    for subset in args.subsets:
        for ctx_len in args.context_lens:
            print(f"\n{'='*60}")
            print(f"subset={subset}  ctx_len={ctx_len}  B={args.batch_size}")
            print(f"{'='*60}")
            torch.cuda.empty_cache()

            try:
                batches = load_longbench(
                    subset, tokenizer, args.device,
                    context_len=ctx_len,
                    batch_size=args.batch_size,
                    n_examples=args.n_examples,
                    seed=args.seed,
                )
                f1, tps, n, _ = run_eval(
                    model, tokenizer, batches,
                    k=None,
                    device=args.device,
                    max_new_tokens=args.max_new_tokens,
                )
                results[subset][ctx_len] = f1
                print(f"  F1={f1:.3f}  tok/s={tps:.1f}  n={n}")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                results[subset][ctx_len] = None
                print("  OOM")

    # summary table
    print(f"\n\n{'='*60}")
    print("Baseline F1 — no sparsity, full prefix attention")
    print(f"{'='*60}")

    ctx_lens = args.context_lens

    def fmt_ctx(c):
        return f"{c//1024}k" if c % 1024 == 0 else str(c)

    header = f"{'subset':<20}" + "".join(f"  {fmt_ctx(c):>8}" for c in ctx_lens)
    if len(ctx_lens) > 1:
        delta_label = f"{fmt_ctx(ctx_lens[0])}→{fmt_ctx(ctx_lens[-1])}"
        header += f"  {delta_label:>10}"
    print(header)
    print("-" * len(header))

    for subset in args.subsets:
        row = f"{subset:<20}"
        vals = [results[subset].get(c) for c in ctx_lens]
        for v in vals:
            row += f"  {v:>7.3f}" if v is not None else f"  {'OOM':>7}"
        if len(ctx_lens) > 1 and vals[0] is not None and vals[1] is not None:
            delta = vals[1] - vals[0]
            row += f"  {delta:>+7.3f}"
        print(row)

    print(f"\nn_examples={args.n_examples}  batch_size={args.batch_size}  "
          f"max_new_tokens={args.max_new_tokens}")


if __name__ == "__main__":
    main()
