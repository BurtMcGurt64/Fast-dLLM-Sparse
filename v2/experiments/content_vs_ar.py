"""
EXPERIMENT 2 — Is block-diffusion's "2x more concentrated" attention REAL content
routing, or just more aggressive attention-sink dumping?

Motivation: a reviewer who has read StreamingLLM will immediately ask whether the
low frac90 you report for the diffusion model is just the softmax sink (blocks 0-1)
plus recency — structural mass that carries no information — rather than genuinely
sharper routing to the relevant CONTENT blocks. If the diffusion advantage vanishes
once sink+recency are excluded, the motivation is a mirage. If it survives, you have
a defensible, block-diffusion-specific reason for KV sparsity.

Method: run the SAME prompts (same subset/seed/context) through
  - the diffusion model  (commit-step block scores, via CommitScoreCapture)
  - the AR backbone       (Qwen2.5-7B-Instruct, per-token attention accumulated)
and report, for each:
  - raw frac90            (all blocks)                         <- the existing headline
  - content-only frac90   (sink 0-1 + recency last-2 removed) <- the decisive number
  - sink+recency mass share
Models are loaded one at a time to fit memory.

Run:
    CUDA_VISIBLE_DEVICES=0 uv run v2/experiments/content_vs_ar.py \
        --subset narrativeqa --context_len 8192 --n_examples 40

    # control for query diversity (AR: first generated token only, matching a
    # single diffusion commit query):
    CUDA_VISIBLE_DEVICES=0 uv run v2/experiments/content_vs_ar.py \
        --subset narrativeqa --context_len 8192 --n_examples 40 --ar_single_step
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import gc
import json
import types
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

from helper import load_model
from generation_functions import Fast_dLLM_QwenForCausalLM as GenClass
from block_attention_heatmap import TEMPLATES, BLOCK_SIZE
from block_attention_dist import load_examples
from ar_attention_dist import ARAttentionAccumulator
from commit_score_capture import CommitScoreCapture

# The actual backbone the diffusion model was trained from (CLAUDE.md).
# Switch to Qwen/Qwen2-7B-Instruct to reproduce the earlier ar_attention_dist runs.
AR_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


# ---------------------------------------------------------------------------
# Shared metrics (identical rule applied to both models)
# ---------------------------------------------------------------------------

def frac90(scores):
    total = scores.sum()
    if total <= 0 or len(scores) == 0:
        return None
    s = np.sort(scores)[::-1]
    cum = np.cumsum(s) / total
    return (int(np.searchsorted(cum, 0.90)) + 1) / len(scores)


def metrics(scores):
    """raw frac90, content-only frac90, sink+recency mass share."""
    n = len(scores)
    total = scores.sum()
    if total <= 0 or n == 0:
        return None
    struct = {j for j in range(n) if j <= 1 or j >= n - 2}
    content_idx = [j for j in range(n) if j not in struct]
    sr_mass = float(scores[list(struct)].sum() / total)
    cf90 = frac90(scores[content_idx]) if content_idx else None
    return {"frac90_raw": frac90(scores), "frac90_content": cf90, "sr_mass": sr_mass}


# ---------------------------------------------------------------------------
# Diffusion probe -> last-commit block scores
# ---------------------------------------------------------------------------

def diff_scores(model, tokenizer, encoded, device, max_new_tokens):
    prompt_len = encoded.input_ids.shape[1]
    cap = CommitScoreCapture(model)
    cap.install()
    sampler = types.MethodType(GenClass.batch_sample, model)
    try:
        seq_len_t = torch.tensor([prompt_len], dtype=torch.long, device=device)
        sampler(
            input_ids=encoded.input_ids, tokenizer=tokenizer, block_size=BLOCK_SIZE,
            max_new_tokens=max_new_tokens, small_block_size=8, min_len=prompt_len,
            seq_len=seq_len_t, threshold=0.95, temperature=0.0,
        )
    finally:
        cap.uninstall()
    if not cap.records:
        return None
    return cap.records[-1]["scores"]


# ---------------------------------------------------------------------------
# AR probe -> accumulated prefix block scores
# ---------------------------------------------------------------------------

def ar_scores(model, tokenizer, encoded, device, max_new_tokens, single_step):
    prompt_len = encoded.input_ids.shape[1]
    accum = ARAttentionAccumulator(model)
    accum.install()
    input_ids = encoded.input_ids
    generated = []
    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=True, output_attentions=False)
            pkv = out.past_key_values
            nxt = out.logits[0, -1].argmax(-1, keepdim=True)
            generated.append(nxt.item())
            input_ids = nxt.unsqueeze(0)
            steps = 1 if single_step else max_new_tokens
            for _ in range(steps):
                if generated[-1] == tokenizer.eos_token_id:
                    break
                out = model(input_ids=input_ids, past_key_values=pkv,
                            use_cache=True, output_attentions=True)
                accum.flush_step()
                nxt = out.logits[0, -1].argmax(-1, keepdim=True)
                generated.append(nxt.item())
                pkv = out.past_key_values
                input_ids = nxt.unsqueeze(0)
    finally:
        accum.uninstall()
    return accum.block_scores(prompt_len)


# ---------------------------------------------------------------------------
# Run a whole model over all examples
# ---------------------------------------------------------------------------

def run_model(score_fn, examples, tag):
    out = []
    for i, (encoded, _) in enumerate(examples):
        print(f"  [{tag}] [{i+1:2d}/{len(examples)}] prompt_len={encoded.input_ids.shape[1]}",
              end="", flush=True)
        try:
            scores = score_fn(encoded)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(" OOM — skip")
            continue
        if scores is None:
            print(" no scores — skip")
            continue
        m = metrics(scores)
        if m is None:
            print(" empty — skip")
            continue
        out.append(m)
        cf = m["frac90_content"]
        print(f"  frac90_raw={m['frac90_raw']:.1%}  "
              + (f"frac90_content={cf:.1%}" if cf is not None else "frac90_content=n/a")
              + f"  sr_mass={m['sr_mass']:.1%}")
    return out


def summarize(results, name):
    raw = [r["frac90_raw"] for r in results]
    con = [r["frac90_content"] for r in results if r["frac90_content"] is not None]
    sr  = [r["sr_mass"] for r in results]
    s = {
        "raw_mean": float(np.mean(raw)), "raw_med": float(np.median(raw)),
        "content_mean": float(np.mean(con)) if con else None,
        "content_med": float(np.median(con)) if con else None,
        "sr_mean": float(np.mean(sr)), "n": len(results),
    }
    print(f"\n  {name}  (n={s['n']}):")
    print(f"    raw frac90     mean={s['raw_mean']:.1%}  median={s['raw_med']:.1%}")
    if con:
        print(f"    content frac90 mean={s['content_mean']:.1%}  median={s['content_med']:.1%}")
    print(f"    sink+recency mass mean={s['sr_mean']:.1%}")
    return s


# ---------------------------------------------------------------------------
# Plot: grouped bars, raw vs content, diffusion vs AR
# ---------------------------------------------------------------------------

def plot_compare(sd, sa, subset, ctx, out_path):
    groups = ["raw frac90\n(all blocks)", "content-only frac90\n(sink+recency excluded)"]
    diff_vals = [sd["raw_mean"], sd["content_mean"] or np.nan]
    ar_vals   = [sa["raw_mean"], sa["content_mean"] or np.nan]

    x = np.arange(len(groups)); w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 5))
    b1 = ax.bar(x - w/2, diff_vals, w, color="#2196F3", label="Fast-dLLM v2 (diffusion)")
    b2 = ax.bar(x + w/2, ar_vals,   w, color="#9E9E9E", label="Qwen2.5-7B (AR backbone)")
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            if not np.isnan(h):
                ax.text(b.get_x() + b.get_width()/2, h + 0.008, f"{h:.0%}",
                        ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("frac90  (lower = more concentrated)")
    ax.set_ylim(0, max(0.6, np.nanmax(diff_vals + ar_vals) * 1.3))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title(f"Is diffusion's concentration REAL content routing? — {subset}, ctx≈{ctx}, B=1\n"
                 f"If the gap holds in the RIGHT bars, the motivation survives sink-exclusion",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diff_model",    type=str, default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    parser.add_argument("--ar_model",      type=str, default=AR_MODEL_ID)
    parser.add_argument("--device",        type=str, default="cuda:0")
    parser.add_argument("--subset",        type=str, default="narrativeqa",
                        choices=list(TEMPLATES.keys()))
    parser.add_argument("--context_len",   type=int, default=8192)
    parser.add_argument("--n_examples",    type=int, default=40)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--ar_single_step", action="store_true",
                        help="AR: accumulate only the first generated token (match a single commit query)")
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    device = args.device
    ctx    = args.context_len

    # ---- Diffusion pass ----
    print("Loading diffusion model...")
    dmodel = load_model(args.diff_model, device=device)
    dtok   = AutoTokenizer.from_pretrained(args.diff_model, trust_remote_code=True)
    d_examples = load_examples(args.subset, dtok, device, ctx, args.n_examples, args.seed)
    print(f"\n{'='*60}\nDIFFUSION — {args.subset} ctx={ctx} n={args.n_examples}\n{'='*60}")
    d_results = run_model(
        lambda enc: diff_scores(dmodel, dtok, enc, device, args.max_new_tokens),
        d_examples, "diff")
    sd = summarize(d_results, "Fast-dLLM v2 (diffusion)")

    # free diffusion model before loading AR
    del dmodel
    gc.collect(); torch.cuda.empty_cache()

    # ---- AR pass (same prompts) ----
    print("\nLoading AR backbone...")
    atok  = AutoTokenizer.from_pretrained(args.ar_model, trust_remote_code=True)
    amodel = AutoModelForCausalLM.from_pretrained(
        args.ar_model, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager").to(device)
    amodel.eval()
    a_examples = load_examples(args.subset, atok, device, ctx, args.n_examples, args.seed)
    mode = "single-step" if args.ar_single_step else f"{args.max_new_tokens}-token avg"
    print(f"\n{'='*60}\nAR ({mode}) — {args.subset} ctx={ctx} n={args.n_examples}\n{'='*60}")
    a_results = run_model(
        lambda enc: ar_scores(amodel, atok, enc, device, args.max_new_tokens, args.ar_single_step),
        a_examples, "ar")
    sa = summarize(a_results, f"Qwen AR ({mode})")

    del amodel
    gc.collect(); torch.cuda.empty_cache()

    # ---- verdict ----
    print(f"\n{'='*60}\nVERDICT\n{'='*60}")
    print(f"  RAW frac90:     diffusion {sd['raw_mean']:.1%}  vs  AR {sa['raw_mean']:.1%}")
    if sd["content_mean"] and sa["content_mean"]:
        dc, ac = sd["content_mean"], sa["content_mean"]
        ratio = ac / dc if dc > 0 else float("inf")
        print(f"  CONTENT frac90: diffusion {dc:.1%}  vs  AR {ac:.1%}   (AR/diff = {ratio:.2f}x)")
        if ratio >= 1.3:
            print("  => Diffusion is genuinely sharper on CONTENT routing (not just sinks). "
                  "Motivation SURVIVES sink-exclusion.")
        elif ratio <= 1.1:
            print("  => The advantage is mostly SINK/recency, not content. Motivation is WEAK — "
                  "the '2x' was largely structural.")
        else:
            print("  => Partial: some real content sharpening, some structural. Report honestly.")

    out_png = os.path.join(os.path.dirname(__file__),
                           f"content_vs_ar_{args.subset}_ctx{ctx}{'_single' if args.ar_single_step else ''}.png")
    if sd["content_mean"] and sa["content_mean"]:
        plot_compare(sd, sa, args.subset, ctx, out_png)

    out_json = os.path.join(os.path.dirname(__file__),
                            f"content_vs_ar_{args.subset}_ctx{ctx}.json")
    with open(out_json, "w") as f:
        json.dump({"diffusion": sd, "ar": sa, "ar_single_step": args.ar_single_step,
                   "subset": args.subset, "context_len": ctx}, f, indent=2)
    print(f"Saved: {out_json}")
    print("\nDone.")


if __name__ == "__main__":
    main()
