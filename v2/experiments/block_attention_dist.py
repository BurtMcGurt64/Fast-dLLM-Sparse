"""
Per-example distribution of attention concentration at a fixed context length.

For each example, runs B=1 generation with BlockAttentionCapture, records:
  - K@90%: how many blocks capture 90% of attention mass (last commit step)
  - K@90%/N: fraction of cache needed (frac)
  - sink+recency mass: how much is structural vs content-routing

Plots a histogram of frac across examples. The shape tells you whether the
aggregate frac number is trustworthy:
  - Tight cluster → mean is representative
  - Bimodal or wide spread → mean hides important variation

Also flags the question-mix confound: runs the same examples at multiple
context lengths (same shuffle seed) so length is the only variable.

Run:
    CUDA_VISIBLE_DEVICES=2 uv run v2/experiments/block_attention_dist.py \
        --subset narrativeqa --context_lens 4096 8192 --n_examples 40

    # just 8k to check outlier vs cluster
    CUDA_VISIBLE_DEVICES=2 uv run v2/experiments/block_attention_dist.py \
        --subset narrativeqa --context_lens 8192 --n_examples 40
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import re
import types
import zipfile
import shutil
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer

from helper import load_model
from generation_functions import Fast_dLLM_QwenForCausalLM as GenClass
from block_attention_heatmap import BlockAttentionCapture, TEMPLATES, BLOCK_SIZE


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_examples(subset, tokenizer, device, context_len, n_examples, seed=42):
    cache_dir = os.path.join(os.path.dirname(__file__), "../sparsity/.longbench_cache")
    jsonl_path = os.path.join(cache_dir, f"{subset}.jsonl")

    if not os.path.exists(jsonl_path):
        os.makedirs(cache_dir, exist_ok=True)
        zip_path = os.path.join(cache_dir, "data.zip")
        if not os.path.exists(zip_path):
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id="THUDM/LongBench", filename="data.zip",
                            repo_type="dataset", local_dir=cache_dir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(cache_dir)
        extracted = os.path.join(cache_dir, "data", f"{subset}.jsonl")
        if os.path.exists(extracted):
            shutil.move(extracted, jsonl_path)

    ds = load_dataset("json", data_files=jsonl_path, split="train")
    ds = ds.shuffle(seed=seed)
    if n_examples < len(ds):
        ds = ds.select(range(n_examples))

    template = TEMPLATES.get(subset, TEMPLATES["narrativeqa"])
    examples = []

    for ex in ds:
        context  = ex.get("context", "") or ex.get("input", "")
        question = ex.get("input", "")
        answers  = ex.get("answers", [])
        if isinstance(answers, str):
            answers = [answers]

        full_prompt = template.format(context=context, input=question)
        full_ids    = tokenizer(full_prompt, return_tensors="pt").input_ids[0]

        if len(full_ids) > context_len:
            parts      = template.split("{context}")
            pfx_len    = len(tokenizer(parts[0], return_tensors="pt").input_ids[0])
            sfx_raw    = parts[1].format(input=question) if "{input}" in parts[1] else parts[1]
            sfx_len    = len(tokenizer(sfx_raw, return_tensors="pt").input_ids[0])
            budget     = max(64, context_len - pfx_len - sfx_len)
            ctx_ids    = tokenizer(context, return_tensors="pt").input_ids[0][:budget]
            context    = tokenizer.decode(ctx_ids, skip_special_tokens=True)
            full_prompt = template.format(context=context, input=question)

        msgs    = [{"role": "user", "content": full_prompt}]
        text    = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(text, return_tensors="pt").to(device)
        examples.append((encoded, answers))

    return examples


# ---------------------------------------------------------------------------
# Run probe on one example, return stats dict
# ---------------------------------------------------------------------------

def probe_example(model, tokenizer, encoded, answers, device, max_new_tokens=64):
    prompt_len = encoded.input_ids.shape[1]
    capture    = BlockAttentionCapture(model)
    capture.install()
    sampler = types.MethodType(GenClass.batch_sample, model)

    try:
        seq_len_t = torch.tensor([prompt_len], dtype=torch.long, device=device)
        output = sampler(
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

    if not capture.records:
        return None  # generation too short, no commit step

    last   = capture.records[-1]
    scores = last["scores"]
    n_blks = last["n_blocks"]
    total  = scores.sum()

    if total == 0 or n_blks == 0:
        return None

    sorted_idx = np.argsort(scores)[::-1]
    sorted_s   = scores[sorted_idx]
    cumsum     = np.cumsum(sorted_s) / total

    k90 = int(np.searchsorted(cumsum, 0.90)) + 1
    k50 = int(np.searchsorted(cumsum, 0.50)) + 1

    sink_recent_idx = {j for j in range(n_blks) if j <= 1 or j >= n_blks - 2}
    sr_mass = scores[list(sink_recent_idx)].sum() / total

    gen_ids  = output[0][prompt_len:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    return {
        "n_blocks":    n_blks,
        "k90":         k90,
        "frac90":      k90 / n_blks,
        "k50":         k50,
        "frac50":      k50 / n_blks,
        "sr_mass":     float(sr_mass),
        "n_commits":   len(capture.records),
        "gen_tokens":  len(gen_ids),
        "gen_text":    gen_text[:80],
        "gold":        answers[0][:60] if answers else "?",
    }


# ---------------------------------------------------------------------------
# Plot histogram
# ---------------------------------------------------------------------------

def plot_dist(results_by_ctx, subset, out_path=None):
    ctx_lens = sorted(results_by_ctx.keys())
    n_panels = len(ctx_lens)

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), sharey=False)
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(f"Per-example attention concentration — {subset}, B=1\n"
                 f"frac = (blocks for 90% mass) / N", fontsize=11)

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    for ax, ctx, color in zip(axes, ctx_lens, colors):
        stats = results_by_ctx[ctx]
        fracs = [s["frac90"] for s in stats]
        sr    = [s["sr_mass"] for s in stats]

        mean_frac = np.mean(fracs)
        med_frac  = np.median(fracs)
        std_frac  = np.std(fracs)

        bins = np.linspace(0, 1, 21)
        ax.hist(fracs, bins=bins, color=color, alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.axvline(mean_frac, color="black", linewidth=1.5, linestyle="--",
                   label=f"mean={mean_frac:.1%}")
        ax.axvline(med_frac, color="gray", linewidth=1.5, linestyle=":",
                   label=f"median={med_frac:.1%}")

        ax.set_title(f"ctx≈{ctx}  (n={len(stats)})\n"
                     f"mean={mean_frac:.1%}  median={med_frac:.1%}  std={std_frac:.1%}",
                     fontsize=10)
        ax.set_xlabel("frac = K@90% / N")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # also show sink+recency mass distribution as overlay
        ax2 = ax.twinx()
        ax2.hist(sr, bins=bins, color="orange", alpha=0.3, edgecolor="none")
        ax2.set_ylabel("Count (sink+recency mass)", color="orange", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="orange")
        ax2.set_ylim(0, ax2.get_ylim()[1] * 3)  # keep overlay subtle

    # add shared legend note
    fig.text(0.5, 0.01, "Orange bars = sink+recency mass distribution (structural attention)",
             ha="center", fontsize=8, color="orange")

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if out_path is None:
        out_path = os.path.join(
            os.path.dirname(__file__),
            f"block_attn_dist_{subset}.png"
        )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        type=str, default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    parser.add_argument("--device",       type=str, default="cuda:0")
    parser.add_argument("--subset",       type=str, default="narrativeqa",
                        choices=list(TEMPLATES.keys()))
    parser.add_argument("--context_lens", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--n_examples",   type=int, default=40)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    print("Loading model...")
    model     = load_model(args.model, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    device    = args.device

    results_by_ctx = {}

    for ctx_len in args.context_lens:
        print(f"\n{'='*60}")
        print(f"ctx_len={ctx_len}  subset={args.subset}  n={args.n_examples}")
        print(f"{'='*60}")

        torch.cuda.empty_cache()
        examples = load_examples(
            args.subset, tokenizer, device, ctx_len,
            args.n_examples, args.seed
        )

        stats_list = []
        skipped    = 0

        for i, (encoded, answers) in enumerate(examples):
            prompt_len = encoded.input_ids.shape[1]
            print(f"  [{i+1:2d}/{len(examples)}] prompt_len={prompt_len}", end="", flush=True)

            try:
                s = probe_example(model, tokenizer, encoded, answers, device,
                                  args.max_new_tokens)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(" OOM — skip")
                skipped += 1
                continue

            if s is None:
                print(" no commit steps — skip")
                skipped += 1
                continue

            stats_list.append(s)
            print(f"  frac90={s['frac90']:.1%}  sr={s['sr_mass']:.1%}  "
                  f"commits={s['n_commits']}  gen={s['gen_tokens']}tok")

        if not stats_list:
            print("  No valid examples.")
            continue

        results_by_ctx[ctx_len] = stats_list

        # summary
        fracs = [s["frac90"] for s in stats_list]
        srs   = [s["sr_mass"] for s in stats_list]
        print(f"\n  Summary ({len(stats_list)} valid, {skipped} skipped):")
        print(f"  frac90 — mean={np.mean(fracs):.1%}  median={np.median(fracs):.1%}  "
              f"std={np.std(fracs):.1%}  min={np.min(fracs):.1%}  max={np.max(fracs):.1%}")
        print(f"  sink+recency mass — mean={np.mean(srs):.1%}  "
              f"std={np.std(srs):.1%}")

        # percentile breakdown
        p25, p75 = np.percentile(fracs, [25, 75])
        tight    = sum(1 for f in fracs if f < 0.20)
        medium   = sum(1 for f in fracs if 0.20 <= f < 0.40)
        high     = sum(1 for f in fracs if f >= 0.40)
        print(f"  frac90 < 20%: {tight} examples  |  20-40%: {medium}  |  ≥40%: {high}")
        print(f"  P25={p25:.1%}  P75={p75:.1%}")

    if results_by_ctx:
        plot_dist(results_by_ctx, args.subset)

    print("\nDone.")


if __name__ == "__main__":
    main()
