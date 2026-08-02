"""
Where does end-to-end generation time actually go? Measures the runtime split across
the three forward types (prefill / commit / small step) + loop overhead, per eval.

This measures the ceiling DIRECTLY: top-K only speeds up the SMALL steps. So the
fraction of e2e that is small-step time is an upper bound on what top-K can touch, and
1/(1 - that fraction) is the absolute e2e speedup ceiling — even if small steps were
infinitely fast. It varies by eval (long-context short-answer = prefill-dominated =
tiny ceiling; long-generation = small-step-dominated = larger ceiling), which is why
"the speedup depends on the workload."

How it works: wraps model.forward, categorizes each call by its kwargs
(update_past_key_values + query length), and times it with CUDA events. Overhead =
wall-clock minus summed forward time (sampling / unmask / dispatch — also unprunable).

Run:
  CUDA_VISIBLE_DEVICES=0 uv run v2/sparsity/runtime_breakdown.py \
      --evals gsm8k math longbench:narrativeqa \
      --n 40 --max_new_tokens 256 --context_len 8192
"""
import sys
import os
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "../experiments"))
sys.path.insert(0, os.path.join(HERE, "../train_scripts"))  # nfe_probe_stage1 (gsm8k/math loaders)

import time
import types
import argparse
from collections import defaultdict
import torch

from transformers import AutoTokenizer
from helper import load_model
from generation_functions import Fast_dLLM_QwenForCausalLM as GenClass


class ForwardTimer:
    """Categorize + time every model.forward call during generation."""
    def __init__(self, model, block_size):
        self.model = model
        self.block_size = block_size
        self._orig = model.forward
        self.events = []  # (category, start_event, end_event)

    def _category(self, kwargs):
        upkv = kwargs.get("update_past_key_values", False)
        ids = kwargs.get("input_ids", None)
        q = ids.shape[1] if ids is not None else 0
        if not upkv:
            return "small"          # denoising small step (block cache)
        if q > self.block_size:
            return "prefill"        # dense pass over the prompt
        return "commit"             # commit block KV (q == block_size)

    def __enter__(self):
        def wrapped(*args, **kwargs):
            cat = self._category(kwargs)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            out = self._orig(*args, **kwargs)
            e.record()
            self.events.append((cat, s, e))
            return out
        self.model.forward = wrapped
        return self

    def __exit__(self, *x):
        self.model.forward = self._orig

    def summary(self):
        torch.cuda.synchronize()
        agg = defaultdict(lambda: [0, 0.0])  # cat -> [count, ms]
        for cat, s, e in self.events:
            agg[cat][0] += 1
            agg[cat][1] += s.elapsed_time(e)
        return agg


def load_batches(eval_name, tokenizer, device, n, batch_size, context_len, seed):
    """Return list of (input_ids, seq_len_tensor, min_len) for batch_sample."""
    if eval_name.startswith("longbench:"):
        from eval_longbench_sweep import load_longbench_subset
        subset = eval_name.split(":", 1)[1]
        raw = load_longbench_subset(subset, tokenizer, device, context_len, batch_size, n, seed)
        out = []
        for encoded, _ans in raw:
            ids = encoded.input_ids
            plen = ids.shape[1]
            out.append((ids, torch.full((ids.shape[0],), plen, device=device, dtype=torch.long), plen))
        return out
    else:
        from nfe_probe_stage1 import EVALS, build_batches
        ev = EVALS[eval_name]
        data = ev["load"](n, seed)
        prompts = [ev["prompt"](q) for q, _ in data]
        golds = [g for _, g in data]
        raw = build_batches(prompts, golds, tokenizer, device, batch_size)
        return [(ids, seq_len, min_len) for (ids, seq_len, min_len, _t), _g in raw]


def run_eval(model, tokenizer, sampler, batches, block_size, max_new_tokens, threshold, device):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with ForwardTimer(model, block_size) as timer:
        with torch.no_grad():
            for ids, seq_len, min_len in batches:
                sampler(input_ids=ids, tokenizer=tokenizer, block_size=block_size,
                        max_new_tokens=max_new_tokens, small_block_size=8,
                        min_len=min_len, seq_len=seq_len, threshold=threshold,
                        temperature=0.0, use_block_cache=True)
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1e3
    return timer.summary(), wall_ms


def report(name, agg, wall_ms):
    fwd_ms = sum(v[1] for v in agg.values())
    overhead = max(0.0, wall_ms - fwd_ms)
    rows = [("prefill", *agg.get("prefill", [0, 0.0])),
            ("commit",  *agg.get("commit",  [0, 0.0])),
            ("small",   *agg.get("small",   [0, 0.0])),
            ("overhead", 0, overhead)]

    print(f"\n=== {name} ===")
    print(f"  {'category':>9} {'count':>7} {'total_ms':>10} {'%e2e':>7} {'ms/fwd':>8}")
    for cat, cnt, ms in rows:
        pct = ms / wall_ms if wall_ms else 0
        per = ms / cnt if cnt else 0
        print(f"  {cat:>9} {cnt:>7} {ms:>10.1f} {pct:>6.1%} {per:>8.2f}")
    print(f"  {'TOTAL':>9} {'':>7} {wall_ms:>10.1f} {'100.0%':>7}")

    small_frac = agg.get("small", [0, 0.0])[1] / wall_ms if wall_ms else 0
    unprunable = 1 - small_frac
    ceil_inf = 1 / unprunable if unprunable > 0 else float("inf")
    def e2e_at(s):
        return 1 / (unprunable + small_frac / s)
    print(f"  -> top-K can touch (small-step share of e2e): {small_frac:.1%}")
    print(f"  -> unprunable (prefill+commit+overhead):       {unprunable:.1%}")
    print(f"  -> ABSOLUTE e2e ceiling (small-step -> 0):      {ceil_inf:.2f}x")
    print(f"  -> e2e IF small-step hits 2.4x (needs B~64):    {e2e_at(2.4):.2f}x")
    print(f"  -> e2e IF small-step hits 5.0x (unreachable):   {e2e_at(5.0):.2f}x")
    print(f"     NOTE: pair the small-step speedup with the SAME batch as this run.")
    print(f"     At B=1 the small step barely speeds up (~1.04x) -> B=1 e2e ~= {e2e_at(1.04):.2f}x.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--evals", nargs="+", default=["gsm8k", "longbench:narrativeqa"],
                   help="e.g. gsm8k math longbench:narrativeqa longbench:qasper")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--context_len", type=int, default=8192, help="prompt length for longbench evals")
    p.add_argument("--block_size", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("Loading model...", flush=True)
    model = load_model(args.model, device=args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    sampler = types.MethodType(GenClass.batch_sample, model)

    results = []
    for ev in args.evals:
        print(f"\nLoading {ev}...", flush=True)
        try:
            batches = load_batches(ev, tokenizer, args.device, args.n, args.batch_size,
                                   args.context_len, args.seed)
        except Exception as e:
            print(f"  skip {ev}: {e}", flush=True)
            continue
        agg, wall = run_eval(model, tokenizer, sampler, batches, args.block_size,
                             args.max_new_tokens, args.threshold, args.device)
        report(ev, agg, wall)
        results.append((ev, agg, wall))

    print("\n" + "=" * 60)
    print("Takeaway: the small-step share IS the prunable fraction. The absolute")
    print("ceiling (1 / unprunable) is measured, per eval — and it's why long-context")
    print("short-answer evals (prefill-dominated) cap lowest.")


if __name__ == "__main__":
    main()
