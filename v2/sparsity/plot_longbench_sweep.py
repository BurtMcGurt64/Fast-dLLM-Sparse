"""
Plot LongBench length-sweep results.

Hardcoded data from eval_longbench_sweep.py runs.
Add new K rows to RESULTS dict as you collect them.

Run:
    uv run v2/sparsity/plot_longbench_sweep.py
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

# ---------------------------------------------------------------------------
# Data — update as new K runs come in
# Format: RESULTS[subset][K][ctx_len] = {dense_f1, topk_f1, union, frac, n_blocks}
# K=None means dense baseline only (no topk row)
# ---------------------------------------------------------------------------

CTX_LENS = [2048, 4096, 8192]

RESULTS = {
    "qasper": {
        16: {
            2048: dict(dense_f1=0.219, topk_f1=0.220, union=46.0, frac=0.703, n_blocks=65.5),
            4096: dict(dense_f1=0.227, topk_f1=0.204, union=52.9, frac=0.408, n_blocks=129.5),
            8192: dict(dense_f1=0.222, topk_f1=0.149, union=53.6, frac=0.226, n_blocks=236.9),
        },
        32: {
            2048: dict(dense_f1=0.219, topk_f1=0.211, union=63.1, frac=0.964, n_blocks=65.5),
            4096: dict(dense_f1=0.227, topk_f1=0.202, union=88.9, frac=0.686, n_blocks=129.5),
            8192: dict(dense_f1=0.222, topk_f1=0.146, union=93.9, frac=0.396, n_blocks=236.9),
        },
        64: {
            2048: dict(dense_f1=0.219, topk_f1=0.219, union=65.5, frac=1.000, n_blocks=65.5),
            4096: dict(dense_f1=0.227, topk_f1=0.220, union=123.2, frac=0.951, n_blocks=129.5),
            8192: dict(dense_f1=0.222, topk_f1=0.152, union=154.8, frac=0.653, n_blocks=236.9),
        },
    },
    "narrativeqa": {
        16: {
            2048: dict(dense_f1=0.154, topk_f1=0.164, union=39.5, frac=0.604, n_blocks=65.4),
            4096: dict(dense_f1=0.183, topk_f1=0.167, union=43.6, frac=0.337, n_blocks=129.4),
            8192: dict(dense_f1=0.196, topk_f1=0.163, union=44.9, frac=0.174, n_blocks=257.4),
        },
        32: {
            2048: dict(dense_f1=0.154, topk_f1=0.149, union=58.5, frac=0.895, n_blocks=65.4),
            4096: dict(dense_f1=0.183, topk_f1=0.166, union=80.1, frac=0.619, n_blocks=129.4),
            8192: dict(dense_f1=0.196, topk_f1=0.179, union=86.6, frac=0.337, n_blocks=257.4),
        },
        64: {
            2048: dict(dense_f1=0.154, topk_f1=0.154, union=65.4, frac=1.000, n_blocks=65.4),
            4096: dict(dense_f1=0.183, topk_f1=0.180, union=117.1, frac=0.905, n_blocks=129.4),
            8192: dict(dense_f1=0.196, topk_f1=0.169, union=159.4, frac=0.619, n_blocks=257.4),
        },
    },
}

# roofline-predicted per-step speedup at B=32 given frac
# speedup ≈ total_bw_baseline / total_bw_topk
# total_bw_baseline = mlp_bw + kv_bw
# total_bw_topk     = mlp_bw + frac * kv_bw
# kv_fraction (of total bw) at B=32:
#   ctx=2048: ~22%, ctx=4096: ~37%, ctx=8192: ~54%
KV_FRAC_AT_B32 = {2048: 0.22, 4096: 0.37, 8192: 0.54}

def predicted_speedup(frac, ctx_len):
    kv = KV_FRAC_AT_B32[ctx_len]
    mlp = 1.0 - kv
    return 1.0 / (mlp + frac * kv)

COLORS = {"qasper": "#2196F3", "narrativeqa": "#FF5722"}
K_STYLES  = {16: "-o", 32: "--s", 64: ":^"}
K_ALPHAS  = {16: 1.0,  32: 0.75, 64: 0.5}
SUBSET_LABELS = {"qasper": "Qasper", "narrativeqa": "NarrativeQA"}

# ---------------------------------------------------------------------------
# Figure: 4 panels
#   TL: frac vs context_len per K (key efficiency result)
#   TR: union vs K at ctx=8192 (shows linear union growth — the bottleneck)
#   BL: F1 vs context_len per K, with dense baseline
#   BR: F1 drop vs frac (accuracy-efficiency Pareto)
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Top-K Prefix Block Loading — LongBench Length Sweep (B=8, n=80)",
             fontsize=13, fontweight="bold")

ax_frac, ax_union_k = axes[0]
ax_f1,   ax_pareto  = axes[1]

K_VALUES = sorted(next(iter(RESULTS.values())).keys())

for subset, sub_data in RESULTS.items():
    color = COLORS[subset]
    label = SUBSET_LABELS[subset]

    for k, k_data in sub_data.items():
        style = K_STYLES.get(k, "-o")
        alpha = K_ALPHAS.get(k, 1.0)
        ctxs   = sorted(k_data.keys())
        fracs  = [k_data[c]["frac"] * 100 for c in ctxs]
        unions = [k_data[c]["union"] for c in ctxs]
        n_blks = [k_data[c]["n_blocks"] for c in ctxs]
        dense  = [k_data[c]["dense_f1"] for c in ctxs]
        topk   = [k_data[c]["topk_f1"] for c in ctxs]

        # --- TL: frac vs ctx ---
        ax_frac.plot(ctxs, fracs, style, color=color,
                     label=f"{label} K={k}", linewidth=2, markersize=7, alpha=alpha)

        # --- BL: F1 vs ctx ---
        if k == min(K_VALUES):  # draw dense baseline once per subset
            ax_f1.plot(ctxs, dense, "-D", color=color, linewidth=2, markersize=6,
                       label=f"{label} dense", alpha=0.4)
        ax_f1.plot(ctxs, topk, style, color=color,
                   label=f"{label} K={k}", linewidth=2, markersize=7, alpha=alpha)

        # --- BR: F1 drop vs frac (Pareto) ---
        for i, ctx in enumerate(ctxs):
            drop = topk[i] - dense[i]
            frac = k_data[ctx]["frac"] * 100
            marker = {"o": "o", "s": "s", "^": "^"}.get(style[-1], "o")
            ax_pareto.scatter(frac, drop, color=color, marker=marker,
                              s=60, alpha=alpha, zorder=3)
            ax_pareto.annotate(f"K={k}\n{ctx//1024}k",
                               (frac, drop), textcoords="offset points",
                               xytext=(4, 0), fontsize=6, color=color, alpha=alpha)

    # --- TR: union vs K at each ctx (subset = line style, ctx = color shade) ---
    ctx_shades = {2048: 0.4, 4096: 0.7, 8192: 1.0}
    for ctx, shade in ctx_shades.items():
        unions_at_ctx = [sub_data[k][ctx]["union"] for k in K_VALUES if ctx in sub_data[k]]
        ks_at_ctx     = [k for k in K_VALUES if ctx in sub_data[k]]
        n             = sub_data[K_VALUES[0]][ctx]["n_blocks"]
        ls = "-" if subset == "qasper" else "--"
        ax_union_k.plot(ks_at_ctx, unions_at_ctx, ls + "o", color=color,
                        alpha=shade, linewidth=2, markersize=6,
                        label=f"{label} {ctx//1024}k (N={n:.0f})")
        # draw N as horizontal reference (once per ctx)
        if subset == "qasper":
            ax_union_k.axhline(n, color="gray", linestyle=":", linewidth=0.8, alpha=shade * 0.6)

# --- TL formatting ---
ax_frac.set_title("Frac loaded vs context length\n(primary efficiency metric)", fontsize=11)
ax_frac.set_xlabel("Context length (tokens)")
ax_frac.set_ylabel("Frac = union / N  (%)")
ax_frac.set_xscale("log", base=2)
ax_frac.set_xticks(CTX_LENS)
ax_frac.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax_frac.yaxis.set_major_formatter(mticker.PercentFormatter())
ax_frac.set_ylim(0, 110)
ax_frac.legend(fontsize=7, ncol=2)
ax_frac.grid(True, alpha=0.3)

# --- TR formatting ---
ax_union_k.set_title("Union (blocks loaded) vs K\nat ctx=8192 — union grows linearly", fontsize=11)
ax_union_k.set_xlabel("K (blocks selected per sequence)")
ax_union_k.set_ylabel("Absolute union (blocks)")
ax_union_k.set_xticks(K_VALUES)
ax_union_k.legend(fontsize=7, ncol=2)
ax_union_k.grid(True, alpha=0.3)
ax_union_k.text(0.97, 0.05, "dotted = N (total)", transform=ax_union_k.transAxes,
                ha="right", fontsize=7, color="gray")

# --- BL formatting ---
ax_f1.set_title("F1 vs context length per K\n(gap to dense = accuracy cost)", fontsize=11)
ax_f1.set_xlabel("Context length (tokens)")
ax_f1.set_ylabel("F1 score")
ax_f1.set_xscale("log", base=2)
ax_f1.set_xticks(CTX_LENS)
ax_f1.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax_f1.set_ylim(0.10, 0.30)
ax_f1.legend(fontsize=7, ncol=2)
ax_f1.grid(True, alpha=0.3)

# --- BR formatting ---
ax_pareto.set_title("Accuracy drop vs frac loaded\n(Pareto: lower-left = better)", fontsize=11)
ax_pareto.set_xlabel("Frac loaded (%)")
ax_pareto.set_ylabel("F1 drop vs dense baseline")
ax_pareto.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax_pareto.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
ax_pareto.xaxis.set_major_formatter(mticker.PercentFormatter())
ax_pareto.grid(True, alpha=0.3)
# legend by subset color
for subset in RESULTS:
    ax_pareto.scatter([], [], color=COLORS[subset], label=SUBSET_LABELS[subset], s=40)
ax_pareto.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(os.path.dirname(__file__), "longbench_sweep.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()

# ---------------------------------------------------------------------------
# Predicted speedup table (roofline, B=32) from frac
# ---------------------------------------------------------------------------
print("\nPredicted per-step speedup at B=32 (roofline, from frac):")
print(f"{'subset':>14}  {'K':>4}  {'ctx':>6}  {'frac':>7}  {'speedup':>9}")
print("-" * 50)
for subset, sub_data in RESULTS.items():
    for k, k_data in sub_data.items():
        for ctx, row in sorted(k_data.items()):
            sp = predicted_speedup(row["frac"], ctx)
            print(f"  {subset:>12}  {k:>4}  {ctx:>6}  {row['frac']:>6.1%}  {sp:>8.2f}x")
