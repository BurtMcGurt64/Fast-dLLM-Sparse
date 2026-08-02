"""
THROUGHPUT PROBE — the go/no-go for the throughput-framed systems project.

Latency (per-request wall-clock) is weight-floor-bound at ~1.3x. THROUGHPUT
(tokens/sec across a saturated batch) is a different, friendlier number: sparsity's
per-step win grows with batch (1.9x@B16 -> 2.4x@B64 in memory_vs_compute), and
freeing KV lets you fit more concurrent requests. This script measures what the
CURRENT implementation actually delivers, so you know the honest baseline and where
the systems headroom is BEFORE building anything.

Measures, dense vs recency-pinned-sparse, at FIXED gen length (stop suppressed so
every sequence does equal work), swept over batch size:
  - throughput  = total generated tokens / wall-clock  (tok/s)  <- the headline metric
  - peak memory (GB)                                            <- capacity story
  - max batch that fits before OOM                              <- capacity ceiling

How to read it (the three things that decide the project):
  1. Does sparse tok/s pull AHEAD of dense as batch grows?  -> the throughput win is real.
  2. Are dense & sparse peak-mem EQUAL at each batch?        -> current top-K does NOT
     evict (full KV resident), so it buys NO extra batch capacity yet. That gap is the
     motivation for the eviction part of the system (memory -> bigger batch -> more tok/s).
  3. Where does each OOM?                                    -> the batch ceiling chunked
     prefill would lift.

Run:
  CUDA_VISIBLE_DEVICES=0 uv run v2/sparsity/throughput_probe.py \
      --subset narrativeqa --context_len 512 --gen_len 512 \
      --batch_sizes 8 16 32 48 64 --frac 0.2 --n_recency 8 --n_batches 2
"""
import sys
import os
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "../experiments"))

import time
import types
import argparse
import contextlib
import torch
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

from transformers import AutoTokenizer
from helper import load_model
from generation_functions import Fast_dLLM_QwenForCausalLM as GenClass
from topk import TopKBlockLoader
from eval_longbench_sweep import load_longbench_subset

BLOCK_SIZE = 32
STOP_TOKEN = 151645


@contextlib.contextmanager
def suppress_stop_token(model, stop_token=STOP_TOKEN):
    """Force equal gen length for a clean throughput comparison (applied to both)."""
    orig = model.sample_with_top_p

    def wrapped(logits, *a, **kw):
        logits[..., stop_token] = float("-inf")
        return orig(logits, *a, **kw)

    model.sample_with_top_p = wrapped
    try:
        yield
    finally:
        model.sample_with_top_p = orig


def measure(model, tokenizer, sampler, batches, gen_len, device, loader=None):
    """Return (tok_per_s, peak_gb, frac). Times only the timed region after warmup."""
    if loader is not None:
        loader.install()
    torch.cuda.reset_peak_memory_stats(device)
    total_tokens = 0
    try:
        with suppress_stop_token(model), torch.no_grad():
            # warmup on the first batch (kernel autotuning etc.)
            enc0 = batches[0][0]
            B0, p0 = enc0.input_ids.shape
            _ = sampler(input_ids=enc0.input_ids, tokenizer=tokenizer, block_size=BLOCK_SIZE,
                        max_new_tokens=gen_len, small_block_size=8, min_len=p0,
                        seq_len=torch.full((B0,), p0, dtype=torch.long, device=device),
                        threshold=0.95, temperature=0.0, use_block_cache=True)
            torch.cuda.synchronize(device)

            t0 = time.perf_counter()
            for encoded, _ans in batches:
                ids = encoded.input_ids
                B, plen = ids.shape
                out = sampler(input_ids=ids, tokenizer=tokenizer, block_size=BLOCK_SIZE,
                              max_new_tokens=gen_len, small_block_size=8, min_len=plen,
                              seq_len=torch.full((B,), plen, dtype=torch.long, device=device),
                              threshold=0.95, temperature=0.0, use_block_cache=True)
                for b in range(B):
                    total_tokens += out[b].shape[0] - plen
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - t0
    finally:
        frac = 0.0
        if loader is not None:
            try:
                _, _, frac = loader.avg_stats
            except Exception:
                pass
            loader.uninstall()
    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    return (total_tokens / elapsed if elapsed else 0.0), peak_gb, frac


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--subset", default="narrativeqa")
    p.add_argument("--context_len", type=int, default=512)
    p.add_argument("--gen_len", type=int, default=512)
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[8, 16, 32, 48, 64])
    p.add_argument("--n_batches", type=int, default=2, help="timed batches per condition")
    p.add_argument("--frac", type=float, default=0.2)
    p.add_argument("--n_sink", type=int, default=2)
    p.add_argument("--n_recency", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("Loading model...", flush=True)
    model = load_model(args.model, device=args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    sampler = types.MethodType(GenClass.batch_sample, model)
    device = args.device

    print(f"\nsubset={args.subset} ctx={args.context_len} gen_len={args.gen_len} "
          f"frac={args.frac} sink={args.n_sink} recency={args.n_recency}\n", flush=True)
    hdr = (f"  {'batch':>5} | {'D_tok/s':>9} {'D_peakGB':>9} | "
           f"{'S_tok/s':>9} {'S_peakGB':>9} {'frac':>5} | {'tput x':>6}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows = []
    d_max = s_max = 0
    for B in args.batch_sizes:
        # need enough examples to fill n_batches of size B
        try:
            batches = load_longbench_subset(args.subset, tokenizer, device, args.context_len,
                                            B, B * args.n_batches, args.seed)
        except Exception as e:
            print(f"  {B:>5} | data load failed: {e}", flush=True)
            continue

        torch.cuda.empty_cache()
        try:
            d_tps, d_mem, _ = measure(model, tokenizer, sampler, batches, args.gen_len, device)
            d_max = B
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  {B:>5} | dense OOM", flush=True)
            continue

        torch.cuda.empty_cache()
        loader = TopKBlockLoader(model, frac=args.frac, block_size=BLOCK_SIZE,
                                 n_sink=args.n_sink, n_recency=args.n_recency)
        try:
            s_tps, s_mem, frac = measure(model, tokenizer, sampler, batches, args.gen_len,
                                         device, loader=loader)
            s_max = B
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  {B:>5} | {d_tps:>9.0f} {d_mem:>9.1f} | sparse OOM", flush=True)
            continue

        ratio = s_tps / d_tps if d_tps else 0.0
        rows.append((B, d_tps, d_mem, s_tps, s_mem, frac, ratio))
        print(f"  {B:>5} | {d_tps:>9.0f} {d_mem:>9.1f} | {s_tps:>9.0f} {s_mem:>9.1f} "
              f"{frac:>5.0%} | {ratio:>5.2f}x", flush=True)

    print("\n" + "=" * 70)
    print("THROUGHPUT VERDICT")
    print("=" * 70)
    if rows:
        best = max(rows, key=lambda r: r[6])
        print(f"  best throughput ratio: {best[6]:.2f}x at batch {best[0]} "
              f"({best[3]:.0f} vs {best[1]:.0f} tok/s)")
        print(f"  ratio vs batch: " + "  ".join(f"B{r[0]}={r[6]:.2f}x" for r in rows))
        mem_equal = all(abs(r[2] - r[4]) / max(r[2], 1) < 0.05 for r in rows)
        print(f"  dense/sparse peak mem ~equal: {mem_equal}  "
              f"-> {'no capacity gain yet (needs EVICTION)' if mem_equal else 'sparse frees memory'}")
        print(f"  max batch before OOM: dense={d_max}  sparse={s_max}  "
              f"-> the ceiling chunked-prefill would lift")
    print("\n  Read:")
    print("   - ratio RISING with batch  -> throughput win is real; systems work unlocks more.")
    print("   - ratio flat ~1.3x + mem equal -> current impl is capped; the PROJECT is the")
    print("     systems work (eviction -> capacity, kernel -> bandwidth, chunked prefill -> batch).")


if __name__ == "__main__":
    main()
