"""
FULL EVAL — accuracy AND throughput for dense / plain-topK / pinned-topK, across
evals that each exercise a different (context, batch, gen) regime. One natural-
generation run per (eval, method) yields BOTH numbers.

Methods (all vs dense Fast-dLLM v2):
  dense   — full KV, no loader (baseline)
  plain   — top-K, NO pins (n_sink=0, n_recency=0)          [pure content selection]
  pinned  — top-K + sink/recency pins (n_sink=2, n_recency=8) [OUR method]

Speedup metric = tokens/sec (throughput), which is LENGTH-NORMALIZED: if a method
decodes each token faster, its tok/s is higher regardless of how many tokens it
emits. So natural generation gives a fair speedup AND valid accuracy from one run —
no stop-token forcing needed. (--force_gen_len isolates per-token speed at matched
length, but accuracy under it is unreliable; default off.)

Also reported per run: NFE/token (forwards per generated token — catches the
"degrade quality -> lower confidence -> more steps" backfire) and avg gen length.

Regime map (each eval = a regime; override with --context_len/--batch_size/--gen):
  gsm8k, math        short ctx, batch=1 (ragged), moderate gen   [KV barely applies]
  narrativeqa/hotpotqa/gov_report   8k ctx, batched, F1          [KV matters]
  RULER / HumanEval / MBPP          stubbed (need data-gen / exec sandbox)

Run:
  CUDA_VISIBLE_DEVICES=0 uv run v2/sparsity/full_eval.py \
      --evals gsm8k math narrativeqa hotpotqa gov_report \
      --methods dense plain pinned --frac 0.2 --n_examples 60
"""
import sys
import os
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "../experiments"))
sys.path.insert(0, os.path.join(HERE, "../train_scripts"))

import time
import types
import argparse
import contextlib
import torch

from transformers import AutoTokenizer
from helper import load_model
from generation_functions import Fast_dLLM_QwenForCausalLM as GenClass
from topk import TopKBlockLoader
from nfe_probe_stage1 import EVALS as SHORT_EVALS, build_batches, NFECounter
from eval_longbench_sweep import load_longbench_subset, score_answer

BLOCK_SIZE = 32
STOP_TOKEN = 151645

# method name -> loader kwargs (None = dense, no loader)
METHODS = {
    "dense":  None,
    "plain":  dict(n_sink=0, n_recency=0),   # pure content top-K, no pins
    "pinned": dict(n_sink=2, n_recency=8),   # OUR method
}

# eval -> default regime. kind "short" = gsm8k/math (batch=1); "longbench" = F1 subsets.
EVAL_REGISTRY = {
    "gsm8k":       dict(kind="short",     batch=1, gen=256, ctx=None),
    "math":        dict(kind="short",     batch=1, gen=512, ctx=None),
    "narrativeqa": dict(kind="longbench", batch=8, gen=64,  ctx=8192, subset="narrativeqa"),
    "hotpotqa":    dict(kind="longbench", batch=8, gen=64,  ctx=8192, subset="hotpotqa"),
    "gov_report":  dict(kind="longbench", batch=8, gen=256, ctx=8192, subset="gov_report"),
    # --- stubs (need extra infra) ---
    # "ruler_niah":  dict(kind="ruler", ...)   # TODO: RULER synthetic data-gen, sweep ctx 4-32k
    # "humaneval":   dict(kind="code", ...)    # TODO: needs execution sandbox for pass@1
}


@contextlib.contextmanager
def suppress_stop(model):
    orig = model.sample_with_top_p

    def wrapped(logits, *a, **k):
        logits[..., STOP_TOKEN] = float("-inf")
        return orig(logits, *a, **k)

    model.sample_with_top_p = wrapped
    try:
        yield
    finally:
        model.sample_with_top_p = orig


@torch.no_grad()
def timed_generate(model, sampler, tokenizer, ids, min_len, seq_len, max_new,
                   force_gen_len, device):
    """One generation, timed + NFE-counted. Returns (out, wall, nfe, prompt_len)."""
    ctx = suppress_stop(model) if force_gen_len else contextlib.nullcontext()
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with NFECounter(model) as ctr, ctx:
        out = sampler(input_ids=ids, tokenizer=tokenizer, block_size=BLOCK_SIZE,
                      max_new_tokens=max_new, small_block_size=8, min_len=min_len,
                      seq_len=seq_len, threshold=0.95, temperature=0.0,
                      use_block_cache=True)
    torch.cuda.synchronize(device)
    return out, time.perf_counter() - t0, ctr.count


def run_method(model, tokenizer, sampler, eval_name, cfg, method, frac,
               force_gen_len, device, n_examples, seed):
    """Run one (eval, method). Returns dict(acc, tps, nfe_per_tok, avg_gen, frac, n)."""
    loader = None
    if METHODS[method] is not None:
        loader = TopKBlockLoader(model, frac=frac, selection="commit",
                                 **METHODS[method]).install()

    total_correct = total = 0
    total_new = total_time = total_nfe = 0.0
    max_new = force_gen_len or cfg["gen"]

    try:
        if cfg["kind"] == "short":
            ev = SHORT_EVALS[eval_name]
            data = ev["load"](n_examples, seed)
            prompts = [ev["prompt"](q) for q, _ in data]
            golds = [g for _, g in data]
            batches = build_batches(prompts, golds, tokenizer, device, cfg["batch"])
            for (ids, seq_len, min_len, true_lens), gchunk in batches:
                out, wall, nfe = timed_generate(model, sampler, tokenizer, ids,
                                                min_len, seq_len, max_new,
                                                force_gen_len, device)
                total_time += wall
                total_nfe += nfe
                for b in range(len(true_lens)):
                    gen = tokenizer.decode(out[b][true_lens[b]:], skip_special_tokens=True)
                    total_new += max(0, out[b].shape[0] - true_lens[b])
                    if not force_gen_len:
                        pred = ev["parse_pred"](gen)
                        total_correct += int(pred is not None and pred == gchunk[b])
                    total += 1

        elif cfg["kind"] == "longbench":
            batches = load_longbench_subset(cfg["subset"], tokenizer, device,
                                            cfg["ctx"], cfg["batch"], n_examples, seed)
            for encoded, answers_list in batches:
                ids = encoded.input_ids
                B, plen = ids.shape[0], ids.shape[1]
                seq_len = torch.full((B,), plen, dtype=torch.long, device=device)
                out, wall, nfe = timed_generate(model, sampler, tokenizer, ids,
                                                plen, seq_len, max_new,
                                                force_gen_len, device)
                total_time += wall
                total_nfe += nfe
                for b, answers in enumerate(answers_list):
                    gen = tokenizer.decode(out[b][plen:], skip_special_tokens=True).strip()
                    total_new += max(0, out[b].shape[0] - plen)
                    if not force_gen_len:
                        total_correct += score_answer(gen, answers)
                    total += 1
        else:
            raise ValueError(f"eval kind {cfg['kind']!r} not implemented (stub)")

        frac_actual = loader.avg_stats[2] if loader is not None else 0.0
    finally:
        if loader is not None:
            loader.uninstall()

    return dict(
        score=(total_correct / total if total else 0.0),
        tps=(total_new / total_time if total_time else 0.0),
        nfe_per_tok=(total_nfe / total_new if total_new else 0.0),
        avg_gen=(total_new / total if total else 0.0),
        frac=frac_actual, n=total,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--evals", nargs="+", default=["gsm8k", "narrativeqa", "gov_report"],
                   help=f"any of: {list(EVAL_REGISTRY)}")
    p.add_argument("--methods", nargs="+", default=["dense", "plain", "pinned"])
    p.add_argument("--frac", type=float, default=0.2)
    p.add_argument("--n_examples", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--context_len", type=int, default=None, help="override eval ctx")
    p.add_argument("--batch_size", type=int, default=None, help="override eval batch")
    p.add_argument("--batch_sizes", type=int, nargs="+", default=None,
                   help="sweep multiple batch sizes (batch ablation); overrides --batch_size")
    p.add_argument("--gen", type=int, default=None, help="override eval max_new_tokens")
    p.add_argument("--force_gen_len", type=int, default=None,
                   help="suppress stop, generate exactly N (pure per-token speed; accuracy invalid)")
    args = p.parse_args()

    print(f"Loading model...", flush=True)
    model = load_model(args.model, device=args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    sampler = types.MethodType(GenClass.batch_sample, model)

    all_rows = []
    for eval_name in args.evals:
        if eval_name not in EVAL_REGISTRY:
            print(f"  [skip] {eval_name}: not in registry / stubbed", flush=True)
            continue
        cfg = dict(EVAL_REGISTRY[eval_name])
        if args.context_len is not None: cfg["ctx"] = args.context_len
        if args.gen is not None:         cfg["gen"] = args.gen
        score_name = "F1" if cfg["kind"] == "longbench" else "acc"

        batch_list = args.batch_sizes or [args.batch_size or cfg["batch"]]
        for batch in batch_list:
            cfg["batch"] = batch
            print(f"\n{'='*74}\n{eval_name}  kind={cfg['kind']}  ctx={cfg['ctx']}  "
                  f"batch={batch}  gen={args.force_gen_len or cfg['gen']}\n{'='*74}", flush=True)
            print(f"  {'method':>8} {score_name:>7} {'tok/s':>8} {'speedup':>8} "
                  f"{'nfe/tok':>8} {'gen':>6} {'frac':>6}", flush=True)
            print("  " + "-" * 60)

            dense_tps = None
            for method in args.methods:
                try:
                    r = run_method(model, tokenizer, sampler, eval_name, cfg, method,
                                   args.frac, args.force_gen_len, args.device,
                                   args.n_examples, args.seed)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"  {method:>8}  OOM", flush=True)
                    continue
                if method == "dense":
                    dense_tps = r["tps"]
                spd = (r["tps"] / dense_tps) if dense_tps else 0.0
                all_rows.append((eval_name, batch, method, r, spd))
                print(f"  {method:>8} {r['score']:>7.3f} {r['tps']:>8.1f} {spd:>7.2f}x "
                      f"{r['nfe_per_tok']:>8.2f} {r['avg_gen']:>6.0f} {r['frac']:>6.1%}",
                      flush=True)

    print(f"\n{'='*74}\nSUMMARY  (speedup = tok/s vs dense; score = acc or F1)\n{'='*74}")
    print(f"  {'eval':>12} {'batch':>5} {'method':>8} {'score':>7} {'speedup':>8} {'nfe/tok':>8}")
    for eval_name, batch, method, r, spd in all_rows:
        print(f"  {eval_name:>12} {batch:>5} {method:>8} {r['score']:>7.3f} {spd:>7.2f}x {r['nfe_per_tok']:>8.2f}")
    print("\n  Read: pinned score should ≈ dense; speedup should be >1 where KV matters")
    print("  (long ctx) and ≈1 where it doesn't (gsm8k/math short ctx). nfe/tok rising on")
    print("  plain/pinned vs dense = the quality->more-steps backfire.")


if __name__ == "__main__":
    main()
