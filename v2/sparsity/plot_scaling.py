"""
Plot the measured top-K scaling story into Teams-ready PNGs.

All numbers are hardcoded from real runs (measure_kv_bytes.py, profile_step.py,
derisk_genlen.py) on RTX PRO 6000 Blackwell, frac~=0.20. No GPU/model needed to
plot — just: uv run v2/sparsity/plot_scaling.py  (or `python`, needs matplotlib).

Outputs (v2/sparsity/figures/):
  scaling_all.png       - 2x2 overview (drop this one in Teams)
  A_batch_saturates.png - speedup vs batch, saturating toward 1/frac
  B_waterfall.png       - per-step -> decode -> e2e dilution
  C_context_flat.png    - e2e vs context (cancels)
  D_bytes.png           - weights (fixed) vs KV (grows with batch)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------
# MEASURED DATA (source in comments)
# ----------------------------------------------------------------------------
FRAC = 0.199
CEIL = 1.0 / FRAC                       # per-step ceiling = 1/frac ~= 5.0x

# profile_step.py, GPU-only, per small step
PERSTEP = {16: 1.89, 32: 2.13}
# derisk_genlen.py decode-only speedup (gen-length slope), 8k, frac 0.20
DECODE = {8: 1.41, 16: 1.41, 32: 1.59}
# waterfall at B=32 / 8k
WATERFALL = [("Per small\nstep", 2.13), ("Decode\n(+commit)", 1.59), ("End-to-end\n(+prefill)", 1.36)]
# derisk e2e vs context, B=8, frac 0.20
CONTEXT = {8192: 1.23, 16384: 1.24}
# derisk e2e vs gen length, B=16 / 8k
GENLEN = {64: 1.09, 128: 1.20, 256: 1.31, 512: 1.29}
# measure_kv_bytes.py resident bytes at B=32/8k
W_LAYER_GB = 13.05                      # transformer weights, fixed every forward
KV_AT_B32_8K = 15.03                    # KV cache @ B=32, L=8192
GPU = "RTX PRO 6000 Blackwell (1.8 TB/s)"

BLUE, ORANGE, GREEN, GREY = "#1f77b4", "#ff7f0e", "#2ca02c", "#7f7f7f"


def _fit_c(points):
    """Fit speedup(B) = (1+cB)/(1+frac*cB) to measured points (least squares on c)."""
    Bs = np.array([b for b, _ in points]); ys = np.array([y for _, y in points])
    cs = np.linspace(0.005, 0.4, 4000)
    err = [np.sum(((1 + c*Bs)/(1 + FRAC*c*Bs) - ys)**2) for c in cs]
    return cs[int(np.argmin(err))]


def panel_batch(ax):
    c = _fit_c(list(PERSTEP.items()))
    B = np.linspace(1, 160, 400)
    ax.plot(B, (1 + c*B)/(1 + FRAC*c*B), "--", color=BLUE, alpha=0.6,
            label="bandwidth model (saturating)")
    ax.scatter(list(PERSTEP), list(PERSTEP.values()), s=90, color=BLUE, zorder=5,
               label="per small step (measured)")
    ax.scatter(list(DECODE), list(DECODE.values()), s=90, color=ORANGE, marker="s",
               zorder=5, label="decode (measured)")
    ax.axhline(CEIL, ls=":", color=GREY)
    ax.text(155, CEIL - 0.18, f"per-step ceiling = 1/frac = {CEIL:.1f}x",
            ha="right", va="top", color=GREY, fontsize=9)
    ax.axvline(32, ls=":", color=GREY, alpha=0.4)
    ax.text(35, 4.7, "OOM beyond B=32\nat 16k context", fontsize=8, color=GREY, va="top")
    ax.annotate("+0.24x per doubling\nsaturating, not 2x",
                xy=(32, 2.13), xytext=(74, 1.35), fontsize=9, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE))
    ax.set_xlabel("batch size"); ax.set_ylabel("speedup (x)")
    ax.set_title("Batch doesn't double the speedup — it saturates toward 1/frac")
    ax.set_ylim(1, 5.3); ax.legend(fontsize=8, loc="center right")


def panel_waterfall(ax):
    labels = [l for l, _ in WATERFALL]; vals = [v for _, v in WATERFALL]
    bars = ax.bar(labels, vals, color=[BLUE, ORANGE, GREEN], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.03, f"{v:.2f}x",
                ha="center", fontsize=11, fontweight="bold")
    ax.axhline(1.0, ls="-", color="k", lw=0.8)
    ax.axhline(CEIL, ls=":", color=GREY)
    ax.text(2.4, CEIL - 0.15, f"1/frac = {CEIL:.1f}x", ha="right", color=GREY, fontsize=9)
    ax.set_ylabel("speedup (x)"); ax.set_ylim(0, 5.3)
    ax.set_title("Each outer scope adds work batch can't remove (B=32, 8k)")


def panel_context(ax):
    xs = [c // 1024 for c in CONTEXT]; ys = list(CONTEXT.values())
    ax.plot(xs, ys, "-o", color=GREEN, markersize=9, lw=2)
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.015, f"{y:.2f}x", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels([f"{x}k" for x in xs])
    ax.set_xlabel("context length"); ax.set_ylabel("end-to-end speedup (x)")
    ax.set_ylim(1.0, 1.6)
    ax.text(0.5, 0.15, "prefill grows with context and cancels\nthe KV gain exactly -> flat",
            transform=ax.transAxes, ha="center", fontsize=9, color=GREY)
    ax.set_title("Context length: KV gain and prefill cost cancel (B=8)")


def panel_bytes(ax):
    Bs = [8, 16, 32, 64]
    kv = [KV_AT_B32_8K * b / 32 for b in Bs]
    w = [W_LAYER_GB] * len(Bs)
    x = np.arange(len(Bs))
    ax.bar(x, w, color=GREY, label="weights (fixed, shared across batch)")
    ax.bar(x, kv, bottom=w, color=BLUE, label="KV cache (scales with batch)")
    ax.axhline(W_LAYER_GB, ls=":", color="k", alpha=0.4)
    ax.set_xticks(x); ax.set_xticklabels([f"B={b}" for b in Bs])
    ax.set_ylabel("bytes read per forward (GB)")
    ax.set_title("Batch amortizes only the fixed weights (measured, 8k)")
    ax.text(0.03, 0.55, "prefill & commit also\nscale with batch ->\nbatch can't amortize them",
            transform=ax.transAxes, ha="left", va="top", fontsize=8, color=GREY)
    ax.legend(fontsize=8, loc="upper left")


def main():
    for name, fn, title in [
        ("A_batch_saturates", panel_batch, None),
        ("B_waterfall", panel_waterfall, None),
        ("C_context_flat", panel_context, None),
        ("D_bytes", panel_bytes, None),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5))
        fn(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    panel_batch(axes[0, 0]); panel_waterfall(axes[0, 1])
    panel_context(axes[1, 0]); panel_bytes(axes[1, 1])
    fig.suptitle(f"Top-K block sparsity: why speedup is flat in batch & context  "
                 f"[measured, {GPU}, frac~={FRAC:.0%}]", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUT, "scaling_all.png"), dpi=150)
    plt.close(fig)
    print(f"Wrote figures to {OUT}/")
    for f in sorted(os.listdir(OUT)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
