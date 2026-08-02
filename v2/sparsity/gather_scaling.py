"""
Micro-benchmark: does the top-K KV gather cost scale with K (tokens selected)
or with L (full prefix length)? This decides whether the Triton kernel is the
right fix.

The sparse decode path does torch.gather over prefix_k/prefix_v of shape
[B, H_kv, L, d] to pull out K_tok selected tokens. If gather cost ∝ K, longer
context is free; if gather cost ∝ L, the gather tax grows with context and
cancels the KV savings (which is what the e2e slopes showed: decode speedup
1.40x@8k -> 1.27x@16k).

Two sweeps at fixed d/heads (Qwen2-7B KV geometry: H_kv=4, d=128, 28 layers):
  - FIXED K:   K_tok held constant while L grows. If time rises -> scales with L.
  - FIXED frac: K_tok = 0.20 * L (realistic). Shows the true per-context tax.

Also times a contiguous narrow() of the same K_tok tokens as the "ideal kernel"
lower bound (what a block-sparse kernel that reads only selected blocks costs).

Run:
    CUDA_VISIBLE_DEVICES=0 uv run v2/sparsity/gather_scaling.py
"""
import argparse
import torch

BLOCK = 32
H_KV = 4
D = 128
LAYERS = 28  # gather happens once per layer per K and per V


def time_op(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / iters


def build(B, L, k_tok, device):
    kv = torch.randn(B, H_KV, L, D, device=device, dtype=torch.bfloat16)
    # per-sequence selected token indices, sorted (mirrors TopKBlockLoader)
    idx = torch.stack([
        torch.randperm(L, device=device)[:k_tok].sort().values for _ in range(B)
    ])                                             # [B, k_tok]
    gather_index = idx.view(B, 1, k_tok, 1).expand(B, H_KV, k_tok, D)
    return kv, gather_index


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lens", type=int, nargs="+", default=[8192, 16384, 32768])
    p.add_argument("--fixed_k", type=int, default=51, help="fixed K (blocks) for the K-held-constant sweep")
    p.add_argument("--frac", type=float, default=0.20)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()
    B, dev = args.batch, args.device

    def run_row(L, k_tok):
        kv, gidx = build(B, L, k_tok, dev)
        # one gather (K) — real path does this for K and V, so ×2 per layer ×LAYERS
        g = time_op(lambda: torch.gather(kv, 2, gidx))
        # ideal: contiguous read of the same number of tokens (block-sparse kernel lower bound)
        narrow = time_op(lambda: kv[:, :, :k_tok, :].contiguous())
        per_step = g * 2 * LAYERS  # K+V, all layers, per forward
        return g, narrow, per_step

    print(f"B={B}  H_kv={H_KV}  d={D}  (per-op ms; per_step = gather ×2(K,V) ×{LAYERS} layers)\n")

    print("FIXED K (K_tok constant; rising time => cost scales with L, not K):")
    print(f"  {'L':>7}  {'K_tok':>6}  {'gather ms':>10}  {'narrow ms':>10}  {'gather/step ms':>15}")
    kfix = args.fixed_k * BLOCK
    for L in args.lens:
        g, n, ps = run_row(L, min(kfix, L))
        print(f"  {L:>7}  {min(kfix, L):>6}  {g:>10.4f}  {n:>10.4f}  {ps:>15.2f}")

    print("\nFIXED frac (K_tok = frac*L; the realistic per-context tax):")
    print(f"  {'L':>7}  {'K_tok':>6}  {'gather ms':>10}  {'narrow ms':>10}  {'gather/step ms':>15}")
    for L in args.lens:
        k_tok = max(BLOCK, int(args.frac * L))
        g, n, ps = run_row(L, k_tok)
        print(f"  {L:>7}  {k_tok:>6}  {g:>10.4f}  {n:>10.4f}  {ps:>15.2f}")

    print("\nRead:")
    print("  FIXED-K row rising with L  -> gather cost ∝ L (strided access over full tensor).")
    print("     That is the anti-scaling: the Triton block-sparse kernel removes it.")
    print("  narrow ms is the contiguous lower bound: a good kernel should approach it.")
    print("  gather/step is the per-forward tax; compare to KV read savings to see the leak.")


if __name__ == "__main__":
    main()
