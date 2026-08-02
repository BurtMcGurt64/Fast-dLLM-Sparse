"""
MLP vs Attention: FLOPs and bytes crossover analysis for Fast-dLLM v2.

Answers: at what context length does attention become the dominant term
in the denoising steps (Q=8 small-step, Q=32 commit), for FLOPs and bytes?

Key finding: the FLOPs crossover is Q-independent (~28k tokens for Qwen 7B),
so it's the same for Q=8, Q=32, or full prefill. At practical context lengths
(≤16k), MLP dominates FLOPs by 4-7x regardless of Q.

The bytes crossover IS batch-dependent:
  B=8  → ~30k tokens   (KV never dominant at practical lengths)
  B=32 → ~7.4k tokens  (KV dominates at 8k+ context)

Speedup mechanism is memory bandwidth, not FLOPs.

Run:
    uv run v2/experiments/flop_byte_crossover.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

# ---------------------------------------------------------------------------
# Model config (Qwen 7B backbone)
# ---------------------------------------------------------------------------
N_LAYERS   = 28
HIDDEN     = 3584
FFN_DIM    = 18944
N_Q_HEADS  = 28
N_KV_HEADS = 4
HEAD_DIM   = 128
BYTES_PER_PARAM = 2  # BF16

# ---------------------------------------------------------------------------
# Analytical formulas (per-layer, one forward pass)
# ---------------------------------------------------------------------------

def mlp_flops(q_tokens):
    """FLOPs for gate + up + down projections (one layer)."""
    return 6 * q_tokens * HIDDEN * FFN_DIM


def attn_flops(q_tokens, kv_len):
    """FLOPs for Q@K^T and Attn@V (one layer, GQA with 28 Q heads)."""
    return 4 * q_tokens * kv_len * N_Q_HEADS * HEAD_DIM


def mlp_bytes():
    """Weight bytes for gate + up + down (one layer, BF16)."""
    return 3 * HIDDEN * FFN_DIM * BYTES_PER_PARAM


def kv_bytes_per_layer(batch, kv_len):
    """KV cache bytes read during attention (one layer, K and V)."""
    return batch * kv_len * N_KV_HEADS * HEAD_DIM * 2 * BYTES_PER_PARAM


def attn_proj_bytes():
    """Q/K/V/O projection weight bytes (one layer, BF16)."""
    q_bytes  = HIDDEN * (N_Q_HEADS * HEAD_DIM) * BYTES_PER_PARAM
    kv_bytes = 2 * HIDDEN * (N_KV_HEADS * HEAD_DIM) * BYTES_PER_PARAM
    o_bytes  = (N_Q_HEADS * HEAD_DIM) * HIDDEN * BYTES_PER_PARAM
    return q_bytes + kv_bytes + o_bytes


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

KV_LENS = np.logspace(np.log10(512), np.log10(32768), 200)

Q_CONFIGS = {
    "small step (Q=8)":  8,
    "commit step (Q=32)": 32,
}

BATCH_SIZES = [1, 8, 32]

# ---------------------------------------------------------------------------
# Plot 1: FLOPs ratio (attention / MLP) vs KV_len for different Q
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Fast-dLLM v2 (Qwen 7B): Attention vs MLP — FLOPs and Bytes\n"
             "Denoising steps: small-step Q=8, commit Q=32",
             fontsize=11, fontweight="bold")

ax_flop, ax_byte = axes

# FLOPs plot
colors_q = {"small step (Q=8)": "#2196F3", "commit step (Q=32)": "#FF5722"}
for label, q in Q_CONFIGS.items():
    ratios = [attn_flops(q, kv) / mlp_flops(q) for kv in KV_LENS]
    ax_flop.plot(KV_LENS, ratios, label=label, linewidth=2,
                 color=colors_q[label])

ax_flop.axhline(1.0, color="black", linewidth=1.2, linestyle="--", label="crossover (ratio=1)")

# annotate crossover
crossover_kv = 6 * FFN_DIM / 4  # derived analytically
ax_flop.axvline(crossover_kv, color="gray", linewidth=1, linestyle=":")
ax_flop.text(crossover_kv * 1.05, 0.15, f"crossover\n~{crossover_kv/1000:.0f}k tokens\n(Q-independent)",
             fontsize=8, color="gray")

# shade practical context range
ax_flop.axvspan(512, 16384, alpha=0.06, color="green", label="practical range (≤16k)")

ax_flop.set_xscale("log", base=2)
ax_flop.set_xlabel("KV length (prefix tokens)")
ax_flop.set_ylabel("Attention FLOPs / MLP FLOPs  (per layer)")
ax_flop.set_title("FLOPs ratio — crossover is Q-independent at ~28k tokens\n"
                  "At ≤16k context, MLP dominates FLOPs for any Q",
                  fontsize=9)
ax_flop.set_xticks([1024, 2048, 4096, 8192, 16384, 32768])
ax_flop.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax_flop.set_ylim(0, 1.5)
ax_flop.legend(fontsize=8)
ax_flop.grid(True, alpha=0.3)

# print key FLOPs ratios
print("FLOPs ratio (attention/MLP) at decode Q=32:")
for kv in [2048, 4096, 8192, 16384]:
    r = attn_flops(32, kv) / mlp_flops(32)
    print(f"  KV={kv:5d}: {r:.3f}x  (MLP is {1/r:.1f}x more FLOPs)")

# ---------------------------------------------------------------------------
# Bytes plot: KV bytes / weight bytes vs KV_len for different batch sizes
# ---------------------------------------------------------------------------

colors_b = {1: "#9C27B0", 8: "#2196F3", 32: "#FF5722"}

total_weight_bytes = (mlp_bytes() + attn_proj_bytes()) * N_LAYERS

for B in BATCH_SIZES:
    total_kv = [kv_bytes_per_layer(B, kv) * N_LAYERS for kv in KV_LENS]
    ratios_b = [kv / total_weight_bytes for kv in total_kv]
    ax_byte.plot(KV_LENS, ratios_b, label=f"B={B}", linewidth=2, color=colors_b[B])

    # crossover point
    cross_kv = total_weight_bytes / (kv_bytes_per_layer(B, 1) * N_LAYERS)
    if cross_kv <= 32768:
        ax_byte.axvline(cross_kv, color=colors_b[B], linewidth=0.8, linestyle=":")
        ax_byte.text(cross_kv * 1.03, 0.15 + 0.2 * BATCH_SIZES.index(B),
                     f"B={B}\ncross≈{cross_kv/1000:.0f}k",
                     fontsize=7, color=colors_b[B])

ax_byte.axhline(1.0, color="black", linewidth=1.2, linestyle="--", label="crossover (ratio=1)")
ax_byte.axvspan(512, 16384, alpha=0.06, color="green", label="practical range (≤16k)")

ax_byte.set_xscale("log", base=2)
ax_byte.set_xlabel("KV length (prefix tokens)")
ax_byte.set_ylabel("KV bytes / Weight bytes  (all 28 layers)")
ax_byte.set_title("Bytes ratio — crossover IS batch-dependent\n"
                  "B=32: KV > weights at >7.4k  |  B=8: not until ~30k",
                  fontsize=9)
ax_byte.set_xticks([1024, 2048, 4096, 8192, 16384, 32768])
ax_byte.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax_byte.set_ylim(0, 2.5)
ax_byte.legend(fontsize=8)
ax_byte.grid(True, alpha=0.3)

# print key byte ratios
print("\nBytes ratio (KV/weights) at B=32:")
for kv in [2048, 4096, 8192, 16384]:
    r = kv_bytes_per_layer(32, kv) * N_LAYERS / total_weight_bytes
    print(f"  KV={kv:5d}: {r:.3f}x  ({'KV dominant' if r > 1 else 'weight dominant'})")

print("\nBytes crossover (KV = weights):")
for B in BATCH_SIZES:
    cross = total_weight_bytes / (kv_bytes_per_layer(B, 1) * N_LAYERS)
    print(f"  B={B:2d}: {cross:.0f} tokens ({cross/1000:.1f}k)")

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "flop_byte_crossover.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")
plt.show()

# ---------------------------------------------------------------------------
# Summary table for slides
# ---------------------------------------------------------------------------
print("\n" + "="*65)
print("Summary: what fraction of total bandwidth does KV loading represent?")
print("(= how much speedup is theoretically possible from KV pruning)")
print("="*65)
print(f"{'ctx':>6}  {'B=8 KV%':>9}  {'B=32 KV%':>10}  {'B=32 speedup ceiling (K=0)':>28}")
for kv in [2048, 4096, 8192, 16384]:
    kv_b8  = kv_bytes_per_layer(8,  kv) * N_LAYERS
    kv_b32 = kv_bytes_per_layer(32, kv) * N_LAYERS
    w      = total_weight_bytes
    frac8  = kv_b8  / (w + kv_b8)
    frac32 = kv_b32 / (w + kv_b32)
    ceil32 = 1 / (1 - frac32)   # speedup if KV bytes = 0
    print(f"  {kv:>5}  {frac8:>8.1%}  {frac32:>9.1%}  {ceil32:>28.2f}x")
