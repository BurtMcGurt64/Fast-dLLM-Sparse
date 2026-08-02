"""
SPARSITY -> NFE PROBE — does attention sparsity cut the NUMBER of forward passes
(not just the bytes per forward)?

The hypothesis (the one thing that would turn the capped byte-lever into an
uncapped NFE-lever):
  Fast-dLLM v2 unmasks tokens whose confidence clears `threshold` each denoising
  step (>=1 per step). FEWER tokens per step => MORE steps => higher NFE. If
  pruning distracting far-context with top-K makes the model SHARPER on the
  current block, more tokens clear the threshold per step => FEWER steps =>
  LOWER NFE, at the SAME quality. That is a multiplicative speedup (fewer W+KV
  reads), composes with the byte saving, and breaks the ~1.3x byte ceiling.

  It could also go the other way (pruning removes needed context => LESS
  confident => MORE steps). That is exactly why this is a cheap experiment and
  not an assumption.

What it measures, dense vs top-K-sparse, at MATCHED threshold, per eval:
  - accuracy            (must be ~equal for an apples-to-apples "same quality")
  - NFE / example       (forward-pass count via NFECounter)
  - TPF = tokens/forward (the field's metric; higher = fewer steps per token)

Read:
  At a threshold where accuracy is preserved (sparse_acc ~= dense_acc), is
  sparse TPF > dense TPF?  If yes -> sparsity reduces NFE at equal quality ->
  the lever is real. Sweep thresholds to see if the sparse quality-vs-NFE curve
  DOMINATES the dense one (same accuracy at lower NFE, or the cliff shifts).

IMPORTANT — make the pruning ACTUALLY BITE. On short reasoning prompts the
sink+recency pins keep most blocks (we measured frac ~57% at ctx=512), so the
hypothesis is untestable there. Use small pins (n_sink=1, n_recency=2) and/or a
low frac so top-K prunes a real fraction; watch the reported frac (want it well
below 1.0). B=1 (ragged reasoning; batched decode corrupts + amortizes NFE).

Run:
  CUDA_VISIBLE_DEVICES=0 uv run v2/sparsity/sparsity_nfe_probe.py \
      --dataset gsm8k --n_examples 200 --max_new_tokens 256 \
      --thresholds 0.95 0.9 0.8 0.7 0.6 --frac 0.25 --n_sink 1 --n_recency 2
  CUDA_VISIBLE_DEVICES=0 uv run v2/sparsity/sparsity_nfe_probe.py \
      --dataset math  --n_examples 200 --max_new_tokens 512 \
      --thresholds 0.95 0.9 0.8 0.7 0.6 --frac 0.25 --n_sink 1 --n_recency 2
"""
import sys
import os
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))               # v2/
sys.path.insert(0, os.path.join(HERE, "../experiments"))   # helper
sys.path.insert(0, os.path.join(HERE, "../train_scripts"))  # nfe_probe_stage1

import time
import types
import argparse
import torch

from transformers import AutoTokenizer
from helper import load_model
from generation_functions import Fast_dLLM_QwenForCausalLM as GenClass
from topk import TopKBlockLoader
from nfe_probe_stage1 import EVALS, build_batches, NFECounter, BLOCK_SIZE, STOP_TOKEN


def run_condition(model, tokenizer, sampler, batches, threshold, max_new_tokens,
                  parse_pred, loader, label=""):
    """One (threshold, dense|sparse) condition at B=1. Returns dict of metrics.
    NFE is counted with the loader INSTALLED so the (tiny) capture overhead is
    included exactly as it would be in a real sparse run."""
    if loader is not None:
        loader.install()
    correct = total = total_nfe = total_gen = 0
    t_start = time.perf_counter()
    print(f"    [{label} th={threshold:.2f}] running {len(batches)} examples...", flush=True)
    try:
        for i, ((ids, seq_len, min_len, true_lens), golds) in enumerate(batches):
            with NFECounter(model) as ctr:
                out = sampler(
                    input_ids=ids, tokenizer=tokenizer, block_size=BLOCK_SIZE,
                    max_new_tokens=max_new_tokens, small_block_size=8,
                    min_len=min_len, seq_len=seq_len, threshold=threshold,
                    temperature=0.0, use_block_cache=True,
                )
            total_nfe += ctr.count
            elapsed = time.perf_counter() - t_start
            print(f"      ex {i+1:>3}/{len(batches)}  nfe={ctr.count:>4}  "
                  f"{elapsed/(i+1):.1f}s/ex  ({elapsed:.0f}s total)", flush=True)
            for b in range(len(true_lens)):
                gen_ids = out[b][true_lens[b]:].tolist()
                if STOP_TOKEN in gen_ids:
                    gen_ids = gen_ids[:gen_ids.index(STOP_TOKEN)]
                total_gen += len(gen_ids)
                pred = parse_pred(tokenizer.decode(gen_ids, skip_special_tokens=True))
                correct += int(pred is not None and pred == golds[b])
                total += 1
    finally:
        frac = 1.0
        if loader is not None:
            try:
                _, _, frac = loader.avg_stats
            except Exception:
                pass
            loader.uninstall()
    n = max(1, total)
    return dict(acc=correct / n, nfe_per_ex=total_nfe / n,
                gen_per_ex=total_gen / n,
                tpf=(total_gen / total_nfe if total_nfe else 0.0),
                frac=frac)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dataset", choices=list(EVALS), default="gsm8k")
    p.add_argument("--n_examples", type=int, default=200)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--thresholds", type=float, nargs="+",
                   default=[0.95, 0.9, 0.8, 0.7, 0.6])
    p.add_argument("--frac", type=float, default=0.25, help="top-K budget = round(frac*n_blocks)")
    p.add_argument("--n_sink", type=int, default=1, help="keep small so pruning actually bites")
    p.add_argument("--n_recency", type=int, default=2, help="keep small so pruning actually bites")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"Loading model + {args.dataset}...", flush=True)
    model = load_model(args.model, device=args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    sampler = types.MethodType(GenClass.batch_sample, model)

    ev = EVALS[args.dataset]
    data = ev["load"](args.n_examples, args.seed)
    prompts = [ev["prompt"](q) for q, _ in data]
    golds = [g for _, g in data]
    batches = build_batches(prompts, golds, tokenizer, args.device, 1)  # B=1
    parse_pred = ev["parse_pred"]

    print(f"\n{args.dataset}  n={len(data)}  B=1  max_new={args.max_new_tokens}  "
          f"frac={args.frac}  sink={args.n_sink}  recency={args.n_recency}")
    print("  (TPF = generated tokens / forward pass; higher = fewer denoising steps)\n")
    hdr = (f"  {'thresh':>6} | {'D_acc':>6} {'D_NFE':>7} {'D_TPF':>6} | "
           f"{'S_acc':>6} {'S_NFE':>7} {'S_TPF':>6} {'frac':>5} | "
           f"{'dAcc':>6} {'TPF x':>6}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows = []
    for th in sorted(args.thresholds, reverse=True):
        t0 = time.perf_counter()
        d = run_condition(model, tokenizer, sampler, batches, th,
                          args.max_new_tokens, parse_pred, loader=None, label="dense")
        loader = TopKBlockLoader(model, frac=args.frac, block_size=BLOCK_SIZE,
                                 n_sink=args.n_sink, n_recency=args.n_recency)
        s = run_condition(model, tokenizer, sampler, batches, th,
                          args.max_new_tokens, parse_pred, loader=loader, label="sparse")
        dt = time.perf_counter() - t0
        tpf_ratio = s["tpf"] / d["tpf"] if d["tpf"] else 0.0
        rows.append((th, d, s, tpf_ratio))
        print(f"  {th:>6.2f} | {d['acc']:>6.1%} {d['nfe_per_ex']:>7.1f} {d['tpf']:>6.2f} | "
              f"{s['acc']:>6.1%} {s['nfe_per_ex']:>7.1f} {s['tpf']:>6.2f} {s['frac']:>5.0%} | "
              f"{s['acc'] - d['acc']:>+6.1%} {tpf_ratio:>5.2f}x   ({dt:.0f}s)", flush=True)

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    # find the lowest-threshold row where sparse accuracy still matches dense (<=1pt drop)
    ok = [r for r in rows if r[2]["acc"] - r[1]["acc"] >= -0.01]
    if ok:
        th, d, s, ratio = ok[-1]
        print(f"  At threshold={th:.2f} (accuracy matched: dense {d['acc']:.1%} vs sparse {s['acc']:.1%}):")
        print(f"    dense  NFE/ex={d['nfe_per_ex']:.1f}  TPF={d['tpf']:.2f}")
        print(f"    sparse NFE/ex={s['nfe_per_ex']:.1f}  TPF={s['tpf']:.2f}  (frac kept={s['frac']:.0%})")
        if ratio > 1.03:
            print(f"  -> SUPPORTED: sparsity gives {ratio:.2f}x more tokens/forward at equal quality.")
            print(f"     This is an NFE lever (composes with the byte saving); pursue + prior-art check.")
        elif ratio < 0.97:
            print(f"  -> REFUTED (opposite sign): sparsity COSTS {1/ratio:.2f}x NFE. Pruning hurts confidence.")
        else:
            print(f"  -> NULL: TPF unchanged ({ratio:.2f}x). Sparsity does not move NFE; stays a byte-only lever.")
    else:
        print("  Sparse accuracy never matched dense at these settings -> pruning too aggressive")
        print("  (raise --frac / pins) OR the effect is quality-negative. Inspect the table.")
    print("\n  Also check: does the sparse cliff (accuracy fall-off vs threshold) sit at a")
    print("  LOWER threshold than dense? If so, sparsity lets you decode more aggressively.")


if __name__ == "__main__":
    main()
