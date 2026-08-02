"""
MEASURED KV vs weight bytes at (B, L) — no closed-form byte formulas.

Two independent, measured views (theory is only used to *check* them, never to
produce them):

(1) RESIDENT TENSOR BYTES  (always available, no special perms)
    Sum the actual allocated tensors:
      - transformer layer weights = sum over model.model.layers.parameters()
      - KV cache bytes            = sum over the prefix cache's real K/V tensors
    These are the objects that physically sit in HBM and get read every forward.
    This is a measurement of the model's memory image, not a derivation.

(2) HARDWARE DRAM TRAFFIC  (run this file under Nsight Compute)
    One small-step forward is wrapped in NVTX range "measured_step". ncu, scoped
    to that range, reports dram__bytes_read.sum summed over every kernel — the
    real bytes pulled from HBM. Sweep --context_len and fit bytes = a + b*L:
        a       = weight + fixed-activation floor (L-independent, read every fwd)
        b * L   = KV traffic at length L
    Comparing b*L against the resident KV bytes from (1) tells you how many times
    KV is physically re-read (cat + flash => ~2x). All measured.

--------------------------------------------------------------------------------
Standalone (measurement 1 only, no ncu, ~30s):
    CUDA_VISIBLE_DEVICES=0 uv run v2/sparsity/measure_kv_bytes.py \
        --context_len 8192 --batch_size 32
    # add --sparse to also arm top-K (confirms the sparse path is resident-smaller)

Under Nsight Compute (measurement 2), one L at a time:
    ncu --nvtx --nvtx-include "measured_step/" \
        --metrics dram__bytes_read.sum,dram__bytes_write.sum \
        --target-processes all --csv --log-file kvbytes_dense_L8192.csv \
        python v2/sparsity/measure_kv_bytes.py --context_len 8192 --batch_size 32
    # repeat for L in 1024 2048 4096 8192 (see ncu_kv_byte_sweep.sh)

Then fit the measured DRAM bytes into weight-floor vs KV-slope:
    uv run v2/sparsity/measure_kv_bytes.py --fit \
        1024 kvbytes_dense_L1024.csv 2048 kvbytes_dense_L2048.csv \
        4096 kvbytes_dense_L4096.csv 8192 kvbytes_dense_L8192.csv
--------------------------------------------------------------------------------
Notes:
  - ncu counters may be locked on shared boxes (ERR_NVGPUCTRPERM). Fix: cluster
    admin sets NVreg_RestrictProfilingToAdminUsers=0, or run ncu with sudo.
    Measurement (1) needs no perms and is the ground-truth fallback.
  - Metric name varies by ncu version; if dram__bytes_read.sum is unavailable try
    dram__bytes.sum (total) or dram__sectors_read.sum * 32 (bytes/sector).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../experiments"))

import io
import csv
import argparse
import torch
import torch.cuda.nvtx as nvtx

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

from helper import load_model
from topk import TopKBlockLoader

BLOCK_SIZE = 32


def make_fake_prefix_kv(model, B, L, device):
    """Real KV tensors of length L (random values — byte count is data-independent)."""
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


def bytes_of(t):
    return t.numel() * t.element_size()


def layer_weight_bytes(model):
    """Transformer block weights — read in full every forward."""
    return sum(bytes_of(p) for p in model.model.layers.parameters())


def other_weight_bytes(model):
    """Embeddings / lm_head / final norm — only partially read per forward."""
    layer_ids = {id(p) for p in model.model.layers.parameters()}
    return sum(bytes_of(p) for p in model.parameters() if id(p) not in layer_ids)


def kv_cache_bytes(cache):
    """Sum the actual resident K/V tensors in the prefix cache."""
    total = 0
    layers = getattr(cache, "layers", None)
    if layers is not None:
        for lyr in layers:
            for name in ("keys", "values"):
                t = getattr(lyr, name, None)
                if t is not None:
                    total += bytes_of(t)
        if total:
            return total
    for attr in ("key_cache", "value_cache"):
        for t in getattr(cache, attr, []) or []:
            if t is not None:
                total += bytes_of(t)
    return total


# ---------------------------------------------------------------------------
# Measurement 2 post-processing: parse ncu CSVs, fit weight floor vs KV slope
# ---------------------------------------------------------------------------
def _load_ncu_rows(path):
    with open(path) as f:
        lines = f.readlines()
    hdr = next((i for i, l in enumerate(lines) if "Metric Name" in l), None)
    if hdr is None:
        return []
    return list(csv.DictReader(io.StringIO("".join(lines[hdr:]))))


def _sum_metric(path, metric="dram__bytes_read.sum"):
    total = 0.0
    for row in _load_ncu_rows(path):
        if row.get("Metric Name", "").strip() == metric:
            val = row.get("Metric Value", "").replace(",", "").strip()
            try:
                total += float(val)
            except ValueError:
                pass
    return total


def do_fit(pairs, resident_kv_gb=None):
    Ls, byts = [], []
    print("Measured DRAM read bytes per forward (summed over all kernels in range):")
    for L, path in pairs:
        if not os.path.exists(path):
            print(f"  L={L:>6}  MISSING {path}")
            continue
        total = _sum_metric(path)
        Ls.append(L)
        byts.append(total)
        print(f"  L={L:>6}  {total/1e9:8.3f} GB")

    if len(Ls) < 2:
        print("\nNeed >=2 context lengths to separate weight floor from KV slope.")
        return

    import numpy as np
    b, a = np.polyfit(Ls, byts, 1)  # bytes = a + b*L
    print("\nMEASURED decomposition  (least-squares fit  bytes = a + b*L):")
    print(f"  weight + fixed floor  a = {a/1e9:7.3f} GB   (L-independent, read every forward)")
    print(f"  KV traffic slope      b = {b:8.1f} bytes / context token")
    for L in (8192, 16384, 32768):
        print(f"    -> KV traffic @ L={L:>6}: {b*L/1e9:7.3f} GB   "
              f"(KV share = {b*L/(a+b*L):5.1%})")
    if resident_kv_gb is not None:
        # measured re-read multiplier: DRAM KV traffic / resident KV bytes
        kv_traffic_8k = b * 8192 / 1e9
        print(f"\n  Resident KV @ 8k (measurement 1) = {resident_kv_gb:.3f} GB")
        print(f"  Measured KV re-read multiplier   = {kv_traffic_8k/resident_kv_gb:.2f}x "
              f"(expect ~2x from cat + flash)")


def run_workload(args):
    device = args.device
    print("Loading model...", flush=True)
    model = load_model(args.model, device=device)
    model.eval()

    B, L, q = args.batch_size, args.context_len, 8
    prefix_kv = make_fake_prefix_kv(model, B, L, device)
    x_t = torch.randint(0, 1000, (B, q), device=device)

    # populate the block cache so subsequent forwards take the Flash small-step path
    with torch.no_grad():
        init_out = model.forward(
            input_ids=x_t, past_key_values=prefix_kv, use_cache=True,
            update_past_key_values=False, use_block_cache=True, block_size=BLOCK_SIZE,
        )
    block_kv = init_out.block_past_key_values

    # ----- Measurement 1: resident tensor bytes (ground truth) -----
    w_layers = layer_weight_bytes(model)
    w_other = other_weight_bytes(model)
    kv = kv_cache_bytes(prefix_kv)
    denom = w_layers + kv
    print("\n=== MEASURED resident bytes (actual allocated tensors) ===")
    print(f"  B={B}  L={L}  dtype={next(model.parameters()).dtype}")
    print(f"  transformer layer weights (read every forward): {w_layers/1e9:7.3f} GB")
    print(f"  embed/lm_head/other (partially read)          : {w_other/1e9:7.3f} GB")
    print(f"  prefix KV cache @ (B,L)                        : {kv/1e9:7.3f} GB")
    print(f"  -> KV share of (layer_weights + KV)           : {kv/denom:6.1%}")
    print(f"  -> weight-floor share                         : {w_layers/denom:6.1%}")
    print(f"  (per-context-token KV = {kv/L:,.0f} bytes; scales linearly with B and L)")

    # optional: arm top-K so ncu sees the sparse path's real traffic
    loader = TopKBlockLoader(model, k=args.k, block_size=BLOCK_SIZE)
    if args.sparse:
        n_blocks = L // BLOCK_SIZE
        k_act = min(args.k, n_blocks)
        block_ids = torch.stack([
            torch.randperm(n_blocks, device=device)[:k_act].sort().values
            for _ in range(B)
        ])
        offsets = torch.arange(BLOCK_SIZE, device=device)
        token_idx = (block_ids.unsqueeze(-1) * BLOCK_SIZE + offsets).reshape(B, k_act * BLOCK_SIZE)
        loader._arm_layers(token_idx, block_ids)
        print(f"  (SPARSE armed: K={k_act} blocks, frac={k_act/n_blocks:.1%})")
    else:
        loader._clear_layers()

    def one_forward():
        with torch.no_grad():
            model.forward(
                input_ids=x_t, past_key_values=prefix_kv, use_cache=True,
                update_past_key_values=False, use_block_cache=True,
                block_past_key_values=block_kv, replace_position=0, block_size=BLOCK_SIZE,
            )

    # warmup OUTSIDE the range — ncu --nvtx-include filters metrics to "measured_step"
    for _ in range(args.warmup):
        one_forward()
    torch.cuda.synchronize()

    # ----- Measurement 2 target: exactly one forward, captured by ncu -----
    nvtx.range_push("measured_step")
    one_forward()
    torch.cuda.synchronize()
    nvtx.range_pop()

    print("\nProfiled ONE forward inside NVTX range 'measured_step'.")
    print("Re-run under `ncu --nvtx --nvtx-include \"measured_step/\" --metrics "
          "dram__bytes_read.sum ...` to capture real DRAM traffic for this forward.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--context_len", type=int, default=8192)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--k", type=int, default=51)
    p.add_argument("--sparse", action="store_true")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--fit", nargs="+", default=None,
                   help="L1 csv1 L2 csv2 ... : parse ncu CSVs and fit weight-floor vs KV-slope")
    p.add_argument("--resident_kv_gb", type=float, default=None,
                   help="resident KV @ 8k from measurement 1, to report the re-read multiplier")
    args = p.parse_args()

    if args.fit:
        toks = args.fit
        pairs = [(int(toks[i]), toks[i + 1]) for i in range(0, len(toks) - 1, 2)]
        do_fit(pairs, resident_kv_gb=args.resident_kv_gb)
        return

    run_workload(args)


if __name__ == "__main__":
    main()
