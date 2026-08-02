"""
EXPERIMENT 1 — Does longer context / more generation DE-sparsify attention?

This is the load-bearing measurement: if frac90 (the fraction of prefix blocks
that capture 90% of attention mass) RISES with context length or with generation
progress, the whole top-K direction weakens. If it stays flat, the fear is retired
with evidence.

Two axes, two plots:

  (A) frac90 vs CONTEXT LENGTH  — bigger PROMPT.
      For each context length, take the LAST commit step of each example and plot
      mean +/- std of frac90. Flat line => a bigger prompt does not de-concentrate.

  (B) frac90 vs N_BLOCKS across commits — bigger CACHE from GENERATING.
      Every commit step (the cache grows by one block each commit) is one point.
      Scatter + binned-mean line. Flat/decreasing => generating more does not
      de-concentrate; rising => it does.

Each plot shows TWO curves:
  - raw frac90            (all blocks, incl. sink + recency)
  - content-only frac90   (sink blocks 0-1 and recency last-2 EXCLUDED, then the
                           remaining "content" blocks renormalized). This isolates
                           whether the *content routing itself* stays sparse, or
                           whether the low raw frac90 is just the structural sink.

Run (single length to eyeball first):
    CUDA_VISIBLE_DEVICES=0 uv run v2/experiments/frac90_sweep.py \
        --subset narrativeqa --context_lens 8192 --n_examples 40 --max_new_tokens 256

Full sweep:
    CUDA_VISIBLE_DEVICES=0 uv run v2/experiments/frac90_sweep.py \
        --subset narrativeqa --context_lens 4096 8192 16384 32768 \
        --n_examples 40 --max_new_tokens 256

    # summarization de-concentrates more — good stress test:
    CUDA_VISIBLE_DEVICES=0 uv run v2/experiments/frac90_sweep.py \
        --subset gov_report --context_lens 4096 8192 16384 \
        --n_examples 30 --max_new_tokens 512
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import json
import types
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from helper import load_model
from generation_functions import Fast_dLLM_QwenForCausalLM as GenClass
from block_attention_heatmap import TEMPLATES, BLOCK_SIZE
from block_attention_dist import load_examples
from commit_score_capture import CommitScoreCapture


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def frac90(scores):
    """Fraction of blocks capturing 90% of mass. scores: np.array [n_blocks]."""
    total = scores.sum()
    if total <= 0 or len(scores) == 0:
        return None
    s = np.sort(scores)[::-1]
    cum = np.cumsum(s) / total
    k = int(np.searchsorted(cum, 0.90)) + 1
    return k / len(scores)


def content_split(scores):
    """
    Split a per-block score vector into structural (sink 0-1 + recency last-2)
    and content (everything else). Returns:
        content_frac90    — frac90 computed over content blocks only (renormalized)
        content_mass      — content mass / total mass
        sr_mass           — sink+recency mass / total mass
    content_frac90 is None if there are no content blocks.
    """
    n = len(scores)
    total = scores.sum()
    if total <= 0 or n == 0:
        return None, None, None
    struct_idx = {j for j in range(n) if j <= 1 or j >= n - 2}
    content_idx = [j for j in range(n) if j not in struct_idx]

    sr_mass = float(scores[list(struct_idx)].sum() / total)
    if not content_idx:
        return None, 0.0, sr_mass
    content_scores = scores[content_idx]
    content_mass = float(content_scores.sum() / total)
    cf90 = frac90(content_scores)
    return cf90, content_mass, sr_mass


# ---------------------------------------------------------------------------
# Probe one example — returns per-commit trajectory
# ---------------------------------------------------------------------------

def probe_trajectory(model, tokenizer, encoded, device, max_new_tokens):
    """
    Run one B=1 generation, return the list of per-commit records with metrics:
        [{n_blocks, frac90_raw, frac90_content, content_mass, sr_mass}, ...]
    ordered by commit step (i.e. by generation progress).
    """
    prompt_len = encoded.input_ids.shape[1]
    capture = CommitScoreCapture(model)
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

    traj = []
    for r in capture.records:
        scores = r["scores"]
        f_raw = frac90(scores)
        cf90, cmass, sr = content_split(scores)
        if f_raw is None:
            continue
        traj.append({
            "n_blocks":       r["n_blocks"],
            "frac90_raw":     f_raw,
            "frac90_content": cf90,
            "content_mass":   cmass,
            "sr_mass":        sr,
        })
    return traj


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_vs_context(by_ctx, subset, out_path):
    """Plot A: frac90 (raw + content) at the last commit, vs context length."""
    ctx_lens = sorted(by_ctx.keys())
    xs = np.array(ctx_lens, dtype=float)

    raw_mean = [np.mean([e["frac90_raw"] for e in by_ctx[c]]) for c in ctx_lens]
    raw_std  = [np.std([e["frac90_raw"]  for e in by_ctx[c]]) for c in ctx_lens]
    con      = {c: [e["frac90_content"] for e in by_ctx[c] if e["frac90_content"] is not None] for c in ctx_lens}
    con_mean = [np.mean(con[c]) if con[c] else np.nan for c in ctx_lens]
    con_std  = [np.std(con[c])  if con[c] else 0.0    for c in ctx_lens]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(xs, raw_mean, yerr=raw_std, marker="o", capsize=4,
                color="#2196F3", label="raw frac90 (all blocks)")
    ax.errorbar(xs, con_mean, yerr=con_std, marker="s", capsize=4,
                color="#FF5722", label="content-only frac90 (sink+recency excluded)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v/1024)}k"))
    ax.set_xlabel("Context length (prompt)")
    ax.set_ylabel("frac90  (blocks for 90% mass / N)")
    ax.set_ylim(0, max(0.6, np.nanmax(raw_mean) * 1.3))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title(f"(A) Does a bigger PROMPT de-sparsify attention? — {subset}, B=1\n"
                 f"flat line => longer context does NOT de-concentrate", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()


def plot_vs_nblocks(all_commits, subset, out_path):
    """Plot B: frac90 (raw + content) vs n_blocks, one point per commit step."""
    nb   = np.array([c["n_blocks"] for c in all_commits], dtype=float)
    raw  = np.array([c["frac90_raw"] for c in all_commits], dtype=float)
    con  = np.array([c["frac90_content"] if c["frac90_content"] is not None else np.nan
                     for c in all_commits], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(nb, raw, s=10, alpha=0.25, color="#2196F3")
    ax.scatter(nb, con, s=10, alpha=0.25, color="#FF5722")

    # binned means (quantile bins so bins are populated)
    def binned(x, y, color, label, marker):
        m = ~np.isnan(y)
        x, y = x[m], y[m]
        if len(x) < 8:
            return
        edges = np.quantile(x, np.linspace(0, 1, 9))
        edges = np.unique(edges)
        centers, means = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (x >= lo) & (x <= hi)
            if sel.sum() >= 3:
                centers.append(x[sel].mean())
                means.append(y[sel].mean())
        ax.plot(centers, means, marker=marker, color=color, linewidth=2, label=label)

    binned(nb, raw, "#0D47A1", "raw frac90 (binned mean)", "o")
    binned(nb, con, "#BF360C", "content-only frac90 (binned mean)", "s")

    ax.set_xlabel("N prefix blocks at commit step  (grows as generation proceeds)")
    ax.set_ylabel("frac90  (blocks for 90% mass / N)")
    ax.set_ylim(0, max(0.6, np.nanmax(raw) * 1.1))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title(f"(B) Does GENERATING more (bigger cache) de-sparsify? — {subset}, B=1\n"
                 f"flat/down => generating more does NOT de-concentrate", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         type=str, default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    parser.add_argument("--device",        type=str, default="cuda:0")
    parser.add_argument("--subset",        type=str, default="narrativeqa",
                        choices=list(TEMPLATES.keys()))
    parser.add_argument("--context_lens",  type=int, nargs="+", default=[4096, 8192, 16384, 32768])
    parser.add_argument("--n_examples",    type=int, default=40)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    print("Loading model...")
    model     = load_model(args.model, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    device    = args.device

    max_pos = getattr(model.config, "max_position_embeddings", 32768)
    ctx_lens = [c for c in args.context_lens if c <= max_pos]

    by_ctx      = {}   # ctx_len -> list of last-commit metric dicts (one per example)
    all_commits = []   # every commit step across everything (for plot B)

    for ctx_len in ctx_lens:
        print(f"\n{'='*60}\nctx_len={ctx_len}  subset={args.subset}  n={args.n_examples}\n{'='*60}")
        torch.cuda.empty_cache()
        examples = load_examples(args.subset, tokenizer, device, ctx_len, args.n_examples, args.seed)

        last_list = []
        for i, (encoded, answers) in enumerate(examples):
            prompt_len = encoded.input_ids.shape[1]
            print(f"  [{i+1:2d}/{len(examples)}] prompt_len={prompt_len}", end="", flush=True)
            try:
                traj = probe_trajectory(model, tokenizer, encoded, device, args.max_new_tokens)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(" OOM — skip")
                continue
            if not traj:
                print(" no commit steps — skip")
                continue
            all_commits.extend(traj)
            last = traj[-1]
            last_list.append(last)
            cf = last["frac90_content"]
            cf_str = f"{cf:.1%}" if cf is not None else "n/a"
            print(f"  frac90_raw={last['frac90_raw']:.1%}  frac90_content={cf_str}  "
                  f"content_mass={last['content_mass']:.1%}  commits={len(traj)}")

        if not last_list:
            print("  No valid examples.")
            continue
        by_ctx[ctx_len] = last_list

        raw = [e["frac90_raw"] for e in last_list]
        con = [e["frac90_content"] for e in last_list if e["frac90_content"] is not None]
        cm  = [e["content_mass"] for e in last_list]
        print(f"\n  Summary ({len(last_list)} examples):")
        print(f"    raw frac90     — mean={np.mean(raw):.1%}  median={np.median(raw):.1%}  std={np.std(raw):.1%}")
        if con:
            print(f"    content frac90 — mean={np.mean(con):.1%}  median={np.median(con):.1%}  std={np.std(con):.1%}")
        print(f"    content mass share — mean={np.mean(cm):.1%}  (rest is sink+recency)")

    # ------------------------------------------------------------------ plots
    if by_ctx:
        out_a = os.path.join(os.path.dirname(__file__), f"frac90_vs_context_{args.subset}.png")
        plot_vs_context(by_ctx, args.subset, out_a)
    if all_commits:
        out_b = os.path.join(os.path.dirname(__file__), f"frac90_vs_nblocks_{args.subset}.png")
        plot_vs_nblocks(all_commits, args.subset, out_b)

    # dump raw numbers for the meeting / further analysis
    dump = {
        "subset": args.subset,
        "by_context": {str(c): by_ctx[c] for c in by_ctx},
        "n_commits_total": len(all_commits),
    }
    out_json = os.path.join(os.path.dirname(__file__), f"frac90_sweep_{args.subset}.json")
    with open(out_json, "w") as f:
        json.dump(dump, f, indent=2, default=float)
    print(f"Saved: {out_json}")

    # ------------------------------------------------------------ verdict line
    if len(by_ctx) >= 2:
        cl = sorted(by_ctx.keys())
        first = np.mean([e["frac90_raw"] for e in by_ctx[cl[0]]])
        last  = np.mean([e["frac90_raw"] for e in by_ctx[cl[-1]]])
        delta = last - first
        print(f"\nVERDICT (context axis): raw frac90 {first:.1%} @ {cl[0]} -> {last:.1%} @ {cl[-1]}  "
              f"(Δ={delta:+.1%}).  "
              f"{'FLAT — fear retired.' if abs(delta) < 0.05 else 'RISING — attention de-sparsifies with length.'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
