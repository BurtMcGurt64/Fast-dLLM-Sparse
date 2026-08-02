"""
Profile a single dense vs sparse small step, with NVTX annotations for
attention-only isolation in Nsight Systems.

Run with nsys to get attention-vs-FFN breakdown:
    CUDA_VISIBLE_DEVICES=0 nsys profile --stats=true -o /tmp/nsys_out --force-overwrite true \\
        uv run v2/sparsity/profile_step.py

nvtx_sum will show four buckets:
  step_dense   — full dense forward (N_MEASURE calls)
  step_sparse  — full sparse forward (N_MEASURE calls)
  attn_dense   — attention module only, dense (N_MEASURE × 28 calls)
  attn_sparse  — attention module only, sparse (N_MEASURE × 28 calls)

Per-step attention time = bucket_total / N_MEASURE
Per-layer attention time = bucket_total / (N_MEASURE × 28)

Or for a quick wall-clock breakdown without Nsight:
    CUDA_VISIBLE_DEVICES=0 uv run v2/sparsity/profile_step.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../experiments"))

import torch
import torch.cuda.nvtx as nvtx
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
from helper import load_model   # also patches DynamicCache.__getitem__
from topk import TopKBlockLoader

DEVICE      = "cuda"
BLOCK_SIZE  = 32
CTX_LEN     = 8192
BATCH_SIZE  = 16
K           = 51
N_WARMUP    = 5
N_MEASURE   = 20

def make_fake_prefix_kv(model, B, L, device):
    """Build a fake DynamicCache with random KV of length L."""
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    sin = cos = torch.ones(1, L, 64, device=device, dtype=torch.bfloat16)
    cache_position = torch.arange(L, device=device)
    for layer in model.model.layers:
        H_kv = layer.self_attn.k_proj.out_features // layer.self_attn.head_dim
        d    = layer.self_attn.head_dim
        k    = torch.randn(B, H_kv, L, d, device=device, dtype=torch.bfloat16)
        v    = torch.randn(B, H_kv, L, d, device=device, dtype=torch.bfloat16)
        cache.update(k, v, layer.self_attn.layer_idx,
                     {"sin": sin, "cos": cos, "cache_position": cache_position})
    return cache

def time_fn_wall(fn, n_warmup, n_measure):
    """Wall-clock time including Python dispatch overhead."""
    for _ in range(n_warmup):
        fn()
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_measure):
        fn()
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_measure * 1000

def time_fn_cuda(fn, n_warmup, n_measure):
    """GPU-only time via CUDA Events — excludes Python dispatch overhead."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_measure)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(n_measure)]
    for i in range(n_measure):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / n_measure

def install_nvtx_hooks(model):
    """
    Register forward hooks on every attention module to emit NVTX ranges.
    The range name is read from step_label[0] at call time, so switching
    between 'attn_dense' and 'attn_sparse' only requires mutating that list.
    Returns (step_label, hook_handles) — call h.remove() on each handle to clean up.
    """
    step_label = ["dense"]

    def pre_hook(module, args):
        nvtx.range_push(f"attn_{step_label[0]}")

    def post_hook(module, args, output):
        nvtx.range_pop()

    handles = []
    for layer in model.model.layers:
        handles.append(layer.self_attn.register_forward_pre_hook(pre_hook))
        handles.append(layer.self_attn.register_forward_hook(post_hook))

    return step_label, handles

def main():
    print("Loading model...")
    model = load_model()
    model.eval()

    B   = BATCH_SIZE
    L   = CTX_LEN
    q   = 8  # small step query length

    prefix_kv = make_fake_prefix_kv(model, B, L, DEVICE)
    x_t = torch.randint(0, 1000, (B, q), device=DEVICE)

    # Initialize block_kv with one forward pass (no block_past_key_values).
    # Subsequent calls pass the populated cache + replace_position=0, which
    # sets attention_mask=None → Flash Attention path.
    print("Initializing block cache...")
    with torch.no_grad():
        init_out = model.forward(
            input_ids=x_t,
            past_key_values=prefix_kv,
            use_cache=True,
            update_past_key_values=False,
            use_block_cache=True,
            block_size=BLOCK_SIZE,
        )
    block_kv = init_out.block_past_key_values

    loader = TopKBlockLoader(model, k=K, block_size=BLOCK_SIZE)
    n_blocks  = L // BLOCK_SIZE
    k_actual  = min(K, n_blocks)
    block_ids = torch.stack([
        torch.randperm(n_blocks, device=DEVICE)[:k_actual].sort().values
        for _ in range(B)
    ])  # [B, k_actual]
    offsets   = torch.arange(BLOCK_SIZE, device=DEVICE)
    token_idx = (block_ids.unsqueeze(-1) * BLOCK_SIZE + offsets).reshape(B, k_actual * BLOCK_SIZE)

    # Install NVTX hooks — step_label controls the range name at call time
    step_label, nvtx_handles = install_nvtx_hooks(model)

    def _forward():
        model.forward(
            input_ids=x_t,
            past_key_values=prefix_kv,
            use_cache=True,
            update_past_key_values=False,
            use_block_cache=True,
            block_past_key_values=block_kv,
            replace_position=0,
            block_size=BLOCK_SIZE,
        )

    def dense_step():
        with torch.no_grad():
            nvtx.range_push("step_dense")
            _forward()
            nvtx.range_pop()

    def sparse_step():
        with torch.no_grad():
            nvtx.range_push("step_sparse")
            _forward()
            nvtx.range_pop()

    print(f"\nB={B}  L={L}  q={q}  K={K}  frac={K/n_blocks:.1%}")
    print(f"Expected KV bytes dense : {B * L * 4 * 128 * 4 * 28 / 1e9:.2f} GB")
    print(f"Expected KV bytes sparse: {B * k_actual * BLOCK_SIZE * 4 * 128 * 4 * 28 / 1e9:.2f} GB")

    # --- Dense ---
    print(f"\nMeasuring dense (layers disarmed)...")
    step_label[0] = "dense"
    loader._clear_layers()
    dense_wall = time_fn_wall(dense_step, N_WARMUP, N_MEASURE)
    dense_gpu  = time_fn_cuda(dense_step, N_WARMUP, N_MEASURE)

    # --- Sparse ---
    print(f"Measuring sparse (layers armed with K={K} blocks)...")
    step_label[0] = "sparse"
    loader._arm_layers(token_idx, block_ids)
    sparse_wall = time_fn_wall(sparse_step, N_WARMUP, N_MEASURE)
    sparse_gpu  = time_fn_cuda(sparse_step, N_WARMUP, N_MEASURE)
    loader._clear_layers()

    # Clean up hooks
    for h in nvtx_handles:
        h.remove()

    W_GB       = 14.4
    kv_dense   = B * L           * 4 * 128 * 2 * 2 * 28 / 1e9  # K+V, bf16, 28 layers
    kv_sparse  = B * k_actual * BLOCK_SIZE * 4 * 128 * 2 * 2 * 28 / 1e9

    print(f"\n=== Per small step (B={B}, L={L}, q={q}, K={K}) ===")
    print(f"  {'':20s}  {'dense':>10s}  {'sparse':>10s}  {'ratio':>8s}")
    print(f"  {'Wall-clock (ms)':20s}  {dense_wall:10.2f}  {sparse_wall:10.2f}  {dense_wall/sparse_wall:8.3f}x")
    print(f"  {'GPU only (ms)':20s}  {dense_gpu:10.2f}  {sparse_gpu:10.2f}  {dense_gpu/sparse_gpu:8.3f}x")
    print()
    total_dense  = W_GB + kv_dense
    total_sparse = W_GB + kv_sparse
    print(f"  Bytes (W+KV)   dense={total_dense:.2f} GB  sparse={total_sparse:.2f} GB")
    print(f"  Eff BW         dense={total_dense/dense_gpu*1e3:.0f} GB/s  sparse={total_sparse/sparse_gpu*1e3:.0f} GB/s")
    print(f"  BW theory      {total_dense/total_sparse:.3f}x  (W-only theory: {W_GB/(W_GB*(1-kv_dense/(kv_dense+W_GB))+kv_sparse):.3f}x)")
    print()
    print(f"  NVTX attention breakdown (nsys nvtx_sum):")
    print(f"    attn_dense  total / {N_MEASURE} calls = per-step attention time (dense)")
    print(f"    attn_sparse total / {N_MEASURE} calls = per-step attention time (sparse)")
    print(f"    Expected attn speedup if compute-bound: {L / (k_actual * BLOCK_SIZE):.1f}x")
    print(f"    Expected attn speedup if BW-bound:      {kv_dense / kv_sparse:.1f}x  (same as KV ratio)")

if __name__ == "__main__":
    main()
