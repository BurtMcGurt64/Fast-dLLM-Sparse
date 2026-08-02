"""
Is the decode small step MEMORY-bound or COMPUTE-bound — and does it change with batch?

The whole "top-K caps at ~2×" argument assumes runtime tracks bytes loaded (memory-
bound). If the step becomes compute-bound as batch grows, that assumption breaks. This
script MEASURES the regime instead of assuming it. Nothing hardware-specific is
hardcoded: peak bandwidth and peak FLOPS are measured empirically on the actual GPU.

Three independent tests, strongest first:

(1) TIME-vs-BATCH SCALING (assumption-free — needs no byte/flop model):
    Run a real small-step forward at increasing batch; measure per-forward time.
      - time ~FLAT as batch grows  -> memory-bound (weights re-read regardless of batch;
        extra sequences are ~free). Throughput rises ~linearly.
      - time ~LINEAR in batch       -> compute-bound (each sequence costs its own FLOPs).
    Repeat at two context lengths: if the rise is steeper at longer context, the rise is
    KV-memory-driven (memory-bound); if context doesn't matter, it's compute-driven.

(2) ROOFLINE (empirical peaks): achieved bandwidth vs achieved FLOPS as % of measured
    peak. Whichever saturates first is the bound. Bytes/FLOPs are derived from the actual
    model + runtime shapes, not hardcoded.

(3) THE METHOD TEST (most decision-relevant): dense vs top-K small step across batch. If
    the top-K speedup HOLDS/GROWS with batch, KV reads are on the critical path -> memory-
    bound and the method works. If it VANISHES toward 1×, it's compute-bound and top-K is
    pointless. This directly tests the thing the ceiling argument depends on.

Run:
  CUDA_VISIBLE_DEVICES=0 uv run v2/sparsity/memory_vs_compute.py \
      --batches 1 2 4 8 16 32 64 --context_lens 4096 8192 --frac 0.2
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../experiments"))

import argparse
import torch
import torch.nn as nn
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

from helper import load_model
from topk import TopKBlockLoader

BLOCK_SIZE = 32
Q = 8  # small-step query length


# ---------------------------------------------------------------- empirical peaks
def measure_peak_bw(device, gb=4.0, iters=50):
    """Peak achievable HBM bandwidth via a large copy (reads + writes = 2x size)."""
    n = int(gb * 1e9 / 2)
    x = torch.randn(n, device=device, dtype=torch.bfloat16)
    y = torch.empty_like(x)
    for _ in range(5):
        y.copy_(x)
    torch.cuda.synchronize()
    s = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        s[i].record(); y.copy_(x); e[i].record()
    torch.cuda.synchronize()
    ms = sum(a.elapsed_time(b) for a, b in zip(s, e)) / iters
    bytes_moved = 2 * n * 2  # read x + write y, bf16
    return bytes_moved / (ms / 1e3)  # bytes/sec


def measure_peak_flops(device, n=8192, iters=50):
    """Peak achievable bf16 matmul throughput via a large square GEMM."""
    a = torch.randn(n, n, device=device, dtype=torch.bfloat16)
    b = torch.randn(n, n, device=device, dtype=torch.bfloat16)
    for _ in range(5):
        _ = a @ b
    torch.cuda.synchronize()
    s = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        s[i].record(); _ = a @ b; e[i].record()
    torch.cuda.synchronize()
    ms = sum(x.elapsed_time(y) for x, y in zip(s, e)) / iters
    return (2 * n ** 3) / (ms / 1e3)  # flops/sec


# ---------------------------------------------------------------- model byte/flop counts (derived)
def param_bytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters())


def linear_matmul_params(model):
    """Params that participate in per-token matmuls (proj + mlp + lm_head)."""
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            total += m.weight.numel()
    return total


def kv_bytes(cache):
    total = 0
    layers = getattr(cache, "layers", None)
    if layers is not None:
        for lyr in layers:
            for nm in ("keys", "values"):
                t = getattr(lyr, nm, None)
                if t is not None:
                    total += t.numel() * t.element_size()
    return total


# ---------------------------------------------------------------- forward setup
def make_fake_prefix_kv(model, B, L, device):
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    sin = cos = torch.ones(1, L, 64, device=device, dtype=torch.bfloat16)
    cache_position = torch.arange(L, device=device)
    for layer in model.model.layers:
        H_kv = layer.self_attn.k_proj.out_features // layer.self_attn.head_dim
        d = layer.self_attn.head_dim
        k = torch.randn(B, H_kv, L, d, device=device, dtype=torch.bfloat16)
        v = torch.randn(B, H_kv, L, d, device=device, dtype=torch.bfloat16)
        cache.update(k, v, layer.self_attn.layer_idx,
                     {"sin": sin, "cos": cos, "cache_position": cache_position})
    return cache


def time_forward(model, prefix_kv, block_kv, x_t, iters=20, warmup=5):
    def fwd():
        model.forward(input_ids=x_t, past_key_values=prefix_kv, use_cache=True,
                      update_past_key_values=False, use_block_cache=True,
                      block_past_key_values=block_kv, replace_position=0, block_size=BLOCK_SIZE)
    with torch.no_grad():
        for _ in range(warmup):
            fwd()
        torch.cuda.synchronize()
        s = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        e = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            s[i].record(); fwd(); e[i].record()
        torch.cuda.synchronize()
    return sum(a.elapsed_time(b) for a, b in zip(s, e)) / iters  # ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--context_lens", type=int, nargs="+", default=[4096, 8192])
    p.add_argument("--frac", type=float, default=0.2)
    args = p.parse_args()
    dev = args.device

    print("Loading model...", flush=True)
    model = load_model(args.model, device=dev)
    model.eval()

    print("Measuring empirical hardware peaks (no hardcoded values)...", flush=True)
    peak_bw = measure_peak_bw(dev)
    peak_fl = measure_peak_flops(dev)
    print(f"  peak bandwidth : {peak_bw/1e12:.2f} TB/s   (measured via copy)")
    print(f"  peak bf16 FLOPS: {peak_fl/1e12:.1f} TFLOP/s (measured via GEMM)")

    W = param_bytes(model)
    P_mm = linear_matmul_params(model)
    print(f"  weight bytes   : {W/1e9:.1f} GB   matmul params: {P_mm/1e9:.2f}B")

    print(f"\n{'='*94}")
    print("SMALL STEP (q=8). time-flat => memory-bound; time-linear-in-batch => compute-bound")
    print(f"{'='*94}")
    hdr = (f"  {'ctx':>6} {'B':>4} {'time_ms':>8} {'t/t(B=1)':>9} {'tok/s':>10} "
           f"{'ach_BW%':>8} {'ach_FL%':>8} {'topk_x':>7} {'verdict':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr)))

    for L in args.context_lens:
        prefix_kv = make_fake_prefix_kv(model, max(args.batches), L, dev)
        base_t = None
        for B in args.batches:
            torch.cuda.empty_cache()
            try:
                pk = make_fake_prefix_kv(model, B, L, dev)
                x_t = torch.randint(0, 1000, (B, Q), device=dev)
                # init block cache
                with torch.no_grad():
                    init = model.forward(input_ids=x_t, past_key_values=pk, use_cache=True,
                                         update_past_key_values=False, use_block_cache=True,
                                         block_size=BLOCK_SIZE)
                bk = init.block_past_key_values

                # dense timing
                loader = TopKBlockLoader(model, k=10 ** 9, block_size=BLOCK_SIZE)
                loader._clear_layers()
                t_dense = time_forward(model, pk, bk, x_t)

                # sparse timing (top-K frac)
                n_blocks = L // BLOCK_SIZE
                kk = max(1, int(args.frac * n_blocks))
                block_ids = torch.stack([torch.randperm(n_blocks, device=dev)[:kk].sort().values
                                         for _ in range(B)])
                off = torch.arange(BLOCK_SIZE, device=dev)
                tok_idx = (block_ids.unsqueeze(-1) * BLOCK_SIZE + off).reshape(B, kk * BLOCK_SIZE)
                loader._arm_layers(tok_idx, block_ids)
                t_sparse = time_forward(model, pk, bk, x_t)
                loader._clear_layers()

                # derived byte/flop counts for THIS (B,L)
                bytes_moved = W + kv_bytes(pk)              # weights + KV (resident; KV re-read ~2x extra)
                flops = 2 * P_mm * (B * Q)                  # dominant matmul flops
                ach_bw = bytes_moved / (t_dense / 1e3) / peak_bw
                ach_fl = flops / (t_dense / 1e3) / peak_fl
                if base_t is None:
                    base_t = t_dense
                verdict = "MEM" if ach_bw > ach_fl else "COMPUTE"
                topk_x = t_dense / t_sparse
                print(f"  {L:>6} {B:>4} {t_dense:>8.2f} {t_dense/base_t:>9.2f} "
                      f"{B*Q/(t_dense/1e3):>10.0f} {ach_bw:>7.0%} {ach_fl:>7.0%} "
                      f"{topk_x:>6.2f}x {verdict:>9}", flush=True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  {L:>6} {B:>4}  OOM", flush=True)
                break

    print("\nHow to read:")
    print("  t/t(B=1) ~ 1 across batch  => time is flat => MEMORY-bound (batch nearly free).")
    print("  t/t(B=1) ~ B               => time linear  => COMPUTE-bound.")
    print("  ach_BW% >> ach_FL%         => memory-bound (bandwidth saturates first).")
    print("  topk_x stays high with B   => KV reads are on the critical path => method works.")
    print("  topk_x -> 1.0 at high B    => compute-bound => reducing KV bytes stops helping.")
    print("  Steeper time rise at 8k than 4k => the rise is KV-driven (memory), not compute.")


if __name__ == "__main__":
    main()
