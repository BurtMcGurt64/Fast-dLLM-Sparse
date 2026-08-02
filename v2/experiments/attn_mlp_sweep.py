"""
2D sweep: attention vs MLP runtime across batch_size x seq_len.

Measures how the attn/MLP balance shifts as batch grows — at batch=1 MLP
dominates (memory-bound weights), but weights amortize with batch while KV
reads scale with batch, so attention becomes the bottleneck at larger batches.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import argparse
import torch
import numpy as np
from collections import defaultdict
from transformers.cache_utils import DynamicCache

from helper import load_model
from model.modeling import Fast_dLLM_QwenAttention, Fast_dLLM_QwenMLP

DynamicCache.__getitem__ = lambda self, layer_idx: (self.layers[layer_idx].keys, self.layers[layer_idx].values)

BATCH_SIZES = [1, 4, 8, 16, 32]
SEQ_LENS    = [128, 256, 512, 1024, 2048]
Q_SMALL     = 8


# ---------------------------------------------------------------------------
# Timing hooks (same pattern as attn_ffn_sweep.py in Fast-dLLM-Lab)
# ---------------------------------------------------------------------------

def attach_timing_hooks(model):
    timings = defaultdict(list)
    handles = []

    def make_hooks(key):
        start_event = [None]

        def pre(module, args, kwargs=None):
            torch.cuda.synchronize()
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            start_event[0] = e

        def post(module, args, kwargs=None, output=None):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize()
            timings[key].append(start_event[0].elapsed_time(end))

        return pre, post

    for _, module in model.named_modules():
        if isinstance(module, Fast_dLLM_QwenAttention):
            pre, post = make_hooks("attn")
            handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
            handles.append(module.register_forward_hook(post, with_kwargs=True))
        elif isinstance(module, Fast_dLLM_QwenMLP):
            pre, post = make_hooks("mlp")
            handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
            handles.append(module.register_forward_hook(post, with_kwargs=True))

    return handles, timings


def remove_hooks(handles):
    for h in handles:
        h.remove()


# ---------------------------------------------------------------------------
# Build prefix KV cache for a given (batch, seq_len)
# ---------------------------------------------------------------------------

def build_prefix_cache(model, batch_size, seq_len, device):
    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model.forward(
            input_ids=input_ids,
            use_cache=True,
            update_past_key_values=True,
            block_size=32,
        )
    return out.past_key_values


# ---------------------------------------------------------------------------
# Single measurement: n_warmup + n_iters small-step forwards
# ---------------------------------------------------------------------------

def measure(model, batch_size, past_kv, device, n_iters=10, n_warmup=3):
    input_ids = torch.zeros(batch_size, Q_SMALL, dtype=torch.long, device=device)
    handles, timings = attach_timing_hooks(model)

    for _ in range(n_warmup):
        with torch.inference_mode():
            model.forward(input_ids=input_ids, use_cache=True,
                          past_key_values=past_kv, update_past_key_values=False,
                          use_block_cache=True)
    torch.cuda.synchronize()
    timings["attn"].clear()
    timings["mlp"].clear()

    wall_times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            model.forward(input_ids=input_ids, use_cache=True,
                          past_key_values=past_kv, update_past_key_values=False,
                          use_block_cache=True)
        torch.cuda.synchronize()
        wall_times.append((time.perf_counter() - t0) * 1000)

    remove_hooks(handles)

    n_layers = model.config.num_hidden_layers
    attn_per_pass = np.array(timings["attn"]).reshape(n_iters, n_layers).sum(axis=1)
    mlp_per_pass  = np.array(timings["mlp"]).reshape(n_iters, n_layers).sum(axis=1)

    return {
        "attn_ms": float(np.mean(attn_per_pass)),
        "mlp_ms":  float(np.mean(mlp_per_pass)),
        "wall_ms": float(np.mean(wall_times)),
    }


# ---------------------------------------------------------------------------
# 2D sweep
# ---------------------------------------------------------------------------

def run_sweep(model, device, batch_sizes, seq_lens, n_iters):
    header = f"{'batch':>6}  {'seq_len':>8}  {'attn_ms':>9}  {'mlp_ms':>9}  {'wall_ms':>9}  {'attn%':>7}  {'mlp%':>7}  {'mlp/attn':>9}"
    print(header)
    print("-" * len(header))

    rows = []
    for seq_len in seq_lens:
        for batch_size in batch_sizes:
            print(f"  batch={batch_size}, seq_len={seq_len} — building cache...", end=" ", flush=True)
            past_kv = build_prefix_cache(model, batch_size, seq_len, device)
            print("measuring...", end=" ", flush=True)

            r = measure(model, batch_size, past_kv, device, n_iters=n_iters)

            attn_pct = 100 * r["attn_ms"] / r["wall_ms"]
            mlp_pct  = 100 * r["mlp_ms"]  / r["wall_ms"]
            ratio    = r["mlp_ms"] / r["attn_ms"] if r["attn_ms"] > 0 else float("inf")

            print(f"\r  {batch_size:>6}  {seq_len:>8}  {r['attn_ms']:>9.2f}  {r['mlp_ms']:>9.2f}  "
                  f"{r['wall_ms']:>9.2f}  {attn_pct:>6.1f}%  {mlp_pct:>6.1f}%  {ratio:>9.2f}x")

            rows.append(dict(batch=batch_size, seq_len=seq_len,
                             attn_pct=attn_pct, mlp_pct=mlp_pct,
                             mlp_attn_ratio=ratio, **r))

    return rows


def plot_heatmap(rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    batch_sizes = sorted(set(r["batch"]   for r in rows))
    seq_lens    = sorted(set(r["seq_len"] for r in rows))

    attn_grid = np.zeros((len(seq_lens), len(batch_sizes)))
    for r in rows:
        i = seq_lens.index(r["seq_len"])
        j = batch_sizes.index(r["batch"])
        attn_grid[i, j] = r["attn_pct"]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(attn_grid, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=100)
    ax.set_xticks(range(len(batch_sizes)))
    ax.set_xticklabels(batch_sizes)
    ax.set_yticks(range(len(seq_lens)))
    ax.set_yticklabels(seq_lens)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Prefix seq_len")
    ax.set_title("Attention % of wall time\n(higher = more benefit from top-K block loading)")
    plt.colorbar(im, ax=ax, label="attn %")

    for i in range(len(seq_lens)):
        for j in range(len(batch_sizes)):
            ax.text(j, i, f"{attn_grid[i,j]:.0f}%", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig("attn_mlp_sweep.png", dpi=150, bbox_inches="tight")
    print("Saved: attn_mlp_sweep.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str, default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    parser.add_argument("--device",      type=str, default="cuda:0")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=BATCH_SIZES)
    parser.add_argument("--seq_lens",    type=int, nargs="+", default=SEQ_LENS)
    parser.add_argument("--n_iters",     type=int, default=10)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    print("Loading model...")
    model = load_model(args.model, device=args.device)

    rows = run_sweep(model, args.device, args.batch_sizes, args.seq_lens, args.n_iters)
    plot_heatmap(rows)


if __name__ == "__main__":
    main()
