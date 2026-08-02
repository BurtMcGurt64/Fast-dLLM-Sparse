"""
Commit-step attention score stability experiment.

Measures how stable block-level attention scores are between consecutive
commit steps within a single generation. Reports two metrics:

  Pearson r: correlation of raw block scores between steps t and t+1
  Jaccard@K: overlap of the top-K selected block sets (K = top frac of blocks)

Pearson r tells you how similar the score distributions are. Jaccard@K tells
you directly what fraction of your selected blocks would be wrong if you reused
scores from the previous commit step — which is the metric that actually matters
for Top-K block loading.

For each example we run B=1 generation with BlockAttentionCapture, collect
scores at every commit step, then compute both metrics between steps t and t+1
over their shared prefix (prefix grows by 1 block per commit, so we compare
over the first min(len_t, len_{t+1}) blocks).

Run:
    CUDA_VISIBLE_DEVICES=0 uv run v2/experiments/commit_score_stability.py \
        --subset gov_report --context_lens 4096 8192 --n_examples 20 --max_new_tokens 512

    # quick sanity check
    CUDA_VISIBLE_DEVICES=0 uv run v2/experiments/commit_score_stability.py \
        --subset narrativeqa --context_lens 4096 --n_examples 5 --max_new_tokens 256
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import types
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer

DEFAULT_FRAC = 0.20  # top-K fraction for Jaccard overlap

from helper import load_model
from generation_functions import Fast_dLLM_QwenForCausalLM as GenClass
from block_attention_heatmap import BlockAttentionCapture, TEMPLATES, BLOCK_SIZE
from block_attention_dist import load_examples


# ---------------------------------------------------------------------------
# Per-example stability measurement
# ---------------------------------------------------------------------------

def jaccard(s1, s2, frac):
    """Jaccard overlap of the top-frac block sets of two score arrays."""
    k = max(1, int(round(len(s1) * frac)))
    set1 = set(np.argsort(s1)[::-1][:k])
    set2 = set(np.argsort(s2)[::-1][:k])
    return len(set1 & set2) / len(set1 | set2)


def measure_stability(model, tokenizer, encoded, device, max_new_tokens=64, frac=DEFAULT_FRAC):
    """
    Returns list of (pair_idx, r, jaccard) for each consecutive commit-step pair,
    or None if the example has fewer than 2 commit steps.
    """
    prompt_len = encoded.input_ids.shape[1]
    capture = BlockAttentionCapture(model)
    capture.install()
    sampler = types.MethodType(GenClass.batch_sample, model)

    try:
        seq_len_t = torch.tensor([prompt_len], dtype=torch.long, device=device)
        sampler(
            input_ids        = encoded.input_ids,
            tokenizer        = tokenizer,
            block_size       = BLOCK_SIZE,
            max_new_tokens   = max_new_tokens,
            small_block_size = 8,
            min_len          = prompt_len,
            seq_len          = seq_len_t,
            threshold        = 0.95,
            temperature      = 0.0,
        )
    finally:
        capture.uninstall()

    records = capture.records
    if len(records) < 2:
        return None

    pairs = []
    for i in range(len(records) - 1):
        s1 = records[i]["scores"]
        s2 = records[i + 1]["scores"]
        n_shared = min(len(s1), len(s2))
        if n_shared < 4:
            continue
        r = float(np.corrcoef(s1[:n_shared], s2[:n_shared])[0, 1])
        if np.isnan(r):
            continue
        j = jaccard(s1[:n_shared], s2[:n_shared], frac)
        pairs.append((i, r, j))

    return pairs if pairs else None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_stability(results_by_ctx, subset, frac=DEFAULT_FRAC, out_path=None):
    ctx_lens = sorted(results_by_ctx.keys())
    fig, axes = plt.subplots(2, len(ctx_lens), figsize=(6 * len(ctx_lens), 9), squeeze=False)

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    for col, (ctx, color) in enumerate(zip(ctx_lens, colors)):
        all_pairs = results_by_ctx[ctx]["pairs"]
        all_r = [r for _, r, _ in all_pairs]
        all_j = [j for _, _, j in all_pairs]

        by_pos_r = {}
        by_pos_j = {}
        for idx, r, j in all_pairs:
            by_pos_r.setdefault(idx, []).append(r)
            by_pos_j.setdefault(idx, []).append(j)

        positions = sorted(by_pos_r.keys())
        counts = [len(by_pos_r[p]) for p in positions]

        for row, (by_pos, all_vals, ylabel, threshold, thresh_label) in enumerate([
            (by_pos_r, all_r, "Pearson r (score correlation)", 0.90, "r=0.90"),
            (by_pos_j, all_j, f"Jaccard overlap (top-{frac:.0%} blocks)", 0.80, "J=0.80"),
        ]):
            ax = axes[row][col]
            means  = [np.mean(by_pos[p]) for p in positions]
            stds   = [np.std(by_pos[p])  for p in positions]

            ax.errorbar(positions, means, yerr=stds, fmt='o-', color=color,
                        capsize=4, linewidth=1.5, markersize=5, label="mean±std")
            ax.fill_between(positions,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            alpha=0.15, color=color)

            ax.axhline(threshold, color="gray", linestyle="--", linewidth=1,
                       label=thresh_label)
            ax.axhline(np.mean(all_vals), color="black", linestyle=":", linewidth=1,
                       label=f"mean={np.mean(all_vals):.3f}")

            ax.set_title(
                f"ctx≈{ctx}  {subset}\n"
                f"mean={np.mean(all_vals):.3f}  std={np.std(all_vals):.3f}  "
                f"min={np.min(all_vals):.3f}  (n={len(all_vals)})",
                fontsize=9
            )
            ax.set_xlabel("Commit-step pair index  (t → t+1)")
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            for p, c in zip(positions, counts):
                ax.text(p, 1.02, str(c), ha="center", fontsize=6, color="gray")

    fig.suptitle(
        f"Commit-step score stability — {subset}\n"
        f"Top row: Pearson r   |   Bottom row: Jaccard@{frac:.0%} block set overlap",
        fontsize=11
    )
    plt.tight_layout()

    if out_path is None:
        out_path = os.path.join(
            os.path.dirname(__file__),
            f"commit_score_stability_{subset}.png"
        )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          type=str, default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    parser.add_argument("--device",         type=str, default="cuda:0")
    parser.add_argument("--subset",         type=str, default="narrativeqa",
                        choices=list(TEMPLATES.keys()))
    parser.add_argument("--context_lens",   type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--n_examples",     type=int, default=30)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--frac",           type=float, default=DEFAULT_FRAC,
                        help="Top-K fraction for Jaccard overlap (default: 0.20)")
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    print("Loading model...")
    model     = load_model(args.model, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    results_by_ctx = {}

    for ctx_len in args.context_lens:
        print(f"\n{'='*60}")
        print(f"ctx_len={ctx_len}  subset={args.subset}  n={args.n_examples}")
        print(f"{'='*60}")

        torch.cuda.empty_cache()
        examples = load_examples(
            args.subset, tokenizer, args.device, ctx_len,
            args.n_examples, args.seed
        )

        all_pairs = []
        skipped = 0

        for i, (encoded, _) in enumerate(examples):
            prompt_len = encoded.input_ids.shape[1]
            print(f"  [{i+1:2d}/{len(examples)}] prompt_len={prompt_len}", end="", flush=True)

            try:
                pairs = measure_stability(model, tokenizer, encoded, args.device,
                                          args.max_new_tokens, args.frac)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(" OOM — skip")
                skipped += 1
                continue

            if pairs is None:
                print(" <2 commit steps — skip")
                skipped += 1
                continue

            rs = [r for _, r, _ in pairs]
            js = [j for _, _, j in pairs]
            print(f"  n_commits={len(pairs)+1}  r_mean={np.mean(rs):.3f}  "
                  f"r_min={np.min(rs):.3f}  J_mean={np.mean(js):.3f}  J_min={np.min(js):.3f}")
            all_pairs.extend(pairs)

        if not all_pairs:
            print("  No valid examples.")
            continue

        all_r = [r for _, r, _ in all_pairs]
        all_j = [j for _, _, j in all_pairs]
        n_valid = args.n_examples - skipped
        print(f"\n  Summary ({n_valid} examples, {len(all_pairs)} commit pairs, {skipped} skipped):")
        print(f"  Pearson r  — mean={np.mean(all_r):.3f}  std={np.std(all_r):.3f}  "
              f"min={np.min(all_r):.3f}  median={np.median(all_r):.3f}")
        print(f"  Jaccard@{args.frac:.0%} — mean={np.mean(all_j):.3f}  std={np.std(all_j):.3f}  "
              f"min={np.min(all_j):.3f}  median={np.median(all_j):.3f}")

        above_90r = sum(1 for r in all_r if r > 0.90)
        above_80j = sum(1 for j in all_j if j > 0.80)
        above_67j = sum(1 for j in all_j if j > 0.67)
        print(f"  r>0.90: {above_90r}/{len(all_r)} ({100*above_90r/len(all_r):.0f}%)")
        print(f"  J>0.80: {above_80j}/{len(all_j)} ({100*above_80j/len(all_j):.0f}%)  "
              f"J>0.67: {above_67j}/{len(all_j)} ({100*above_67j/len(all_j):.0f}%)")

        results_by_ctx[ctx_len] = {"pairs": all_pairs}

    if results_by_ctx:
        plot_stability(results_by_ctx, args.subset, args.frac)

    print("\nDone.")


if __name__ == "__main__":
    main()
