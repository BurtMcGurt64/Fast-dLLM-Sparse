"""
SELECTION-MECHANISM ABLATION — the make-or-break test for the block-diffusion oracle.

Coherence is fixed by sink+recency PINS (StreamingLLM does that too — see coherence_eval),
so the content-selection mechanism must earn its keep on ACCURACY: retrieving the right
MIDDLE blocks that position-only pinning throws away. This script compares, at a MATCHED
KV budget (only the SELECTION differs):

  dense       — full KV (reference)
  streaming   — sink + recency-fill, NO content selection      (StreamingLLM baseline)
  commit      — OUR method: commit-step scoring + sink/recency pins
  mage        — first-denoising-step scoring + pins            (MAGE-style baseline)

Verdict:
  commit >> streaming  -> content selection beats position-only; the oracle is defensible.
  commit ~= streaming  -> method collapses to StreamingLLM; contribution is thin.
  commit vs mage       -> does the commit-step signal (clean 32-tok queries) beat first-step
                          (all-mask queries)? defends the *commit* oracle specifically.

Run (retrieval subsets where the answer is in the middle):
  CUDA_VISIBLE_DEVICES=0 uv run v2/sparsity/baseline_compare.py \
      --subset hotpotqa --context_len 8192 --frac 0.2 --n_examples 60

  # smoke-test MAGE first (it moves the capture point — verify it's not garbage):
  CUDA_VISIBLE_DEVICES=0 uv run v2/sparsity/baseline_compare.py \
      --subset hotpotqa --n_examples 5 --methods dense mage
"""
import sys
import os
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "../experiments"))

import time
import types
import argparse
import torch

from transformers import AutoTokenizer
from helper import load_model
from generation_functions import Fast_dLLM_QwenForCausalLM as GenClass
from topk import TopKBlockLoader
from eval_longbench_sweep import load_longbench_subset, score_answer

BLOCK_SIZE = 32


def run_method(model, tokenizer, batches, method, frac, n_sink, n_recency,
               device, max_new_tokens):
    """method='dense' => no loader; else install TopKBlockLoader with that selection
    at the matched frac budget. Returns (avg_f1, avg_frac, n)."""
    loader = None
    if method != "dense":
        loader = TopKBlockLoader(model, frac=frac, n_sink=n_sink, n_recency=n_recency,
                                 selection=method).install()
    sampler = types.MethodType(GenClass.batch_sample, model)
    total_f1, total = 0.0, 0
    try:
        for encoded, answers_list in batches:
            input_ids = encoded.input_ids
            B = input_ids.shape[0]
            plen = input_ids.shape[1]
            seq_len = torch.full((B,), plen, dtype=torch.long, device=device)
            out = sampler(input_ids=input_ids, tokenizer=tokenizer, block_size=BLOCK_SIZE,
                          max_new_tokens=max_new_tokens, small_block_size=8,
                          min_len=plen, seq_len=seq_len, threshold=0.95, temperature=0.0)
            for b, answers in enumerate(answers_list):
                gen = tokenizer.decode(out[b][plen:], skip_special_tokens=True).strip()
                total_f1 += score_answer(gen, answers)
                total += 1
        frac_actual = 0.0
        if loader is not None:
            _, _, frac_actual = loader.avg_stats
    finally:
        if loader is not None:
            loader.uninstall()
    return total_f1 / max(1, total), frac_actual, total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--subset", default="hotpotqa",
                   help="retrieval subset where the answer is in the middle (hotpotqa, "
                        "2wikimqa, musique, narrativeqa, qasper)")
    p.add_argument("--context_len", type=int, default=8192)
    p.add_argument("--frac", type=float, default=0.2)
    p.add_argument("--n_sink", type=int, default=2)
    p.add_argument("--n_recency", type=int, default=8)
    p.add_argument("--n_examples", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=4,
                   help="LongBench pads uniformly so batches are OK; drop to 1 if anything looks off")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--methods", nargs="+", default=["dense", "streaming", "commit", "mage"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"Loading model + {args.subset} @ {args.context_len}...", flush=True)
    model = load_model(args.model, device=args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    batches = load_longbench_subset(args.subset, tokenizer, args.device,
                                    args.context_len, args.batch_size, args.n_examples, args.seed)

    print(f"\n{args.subset}  frac={args.frac:.0%}  sink={args.n_sink} recency={args.n_recency}  "
          f"(only the SELECTION differs across rows)\n")
    print(f"  {'method':>10} {'F1':>7} {'frac':>7} {'n':>5}")
    print("  " + "-" * 34)
    rows = {}
    for m in args.methods:
        t0 = time.perf_counter()
        f1, frac_a, n = run_method(model, tokenizer, batches, m, args.frac,
                                   args.n_sink, args.n_recency, args.device, args.max_new_tokens)
        rows[m] = f1
        print(f"  {m:>10} {f1:>7.3f} {frac_a:>6.1%} {n:>5}  ({time.perf_counter() - t0:.0f}s)",
              flush=True)

    print()
    if "dense" in rows and rows["dense"] > 0:
        for m in args.methods:
            if m != "dense":
                print(f"  {m:>10} retains {rows[m] / rows['dense']:.0%} of dense F1")
    if "commit" in rows and "streaming" in rows:
        gap = rows["commit"] - rows["streaming"]
        verdict = ("content selection EARNS its keep (oracle defensible)" if gap > 0.02
                   else "selection ≈ position-only — method COLLAPSES to StreamingLLM")
        print(f"\n  commit − streaming = {gap:+.3f}  → {verdict}")
    if "commit" in rows and "mage" in rows:
        print(f"  commit − mage      = {rows['commit'] - rows['mage']:+.3f}  "
              f"(commit-step vs first-step scoring signal)")


if __name__ == "__main__":
    main()
