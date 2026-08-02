"""
Visualize which prefix blocks the model attends to across commit steps (B=1).

For each commit step during generation, captures the per-block attention mass
(averaged across all layers and heads). Plots as a heatmap:
  rows    = commit steps (as generation progresses)
  columns = prefix block index (0 → N-1)
  color   = attention mass (brighter = more attended)

Shows three structural features:
  - Attention sink: block 0 always bright
  - Recency bias: last few blocks always bright
  - Content routing: specific middle blocks selectively attended

Run:
    CUDA_VISIBLE_DEVICES=2 uv run v2/experiments/block_attention_heatmap.py \
        --subset narrativeqa --context_lens 2048 4096 8192

    # single length, save to specific path
    CUDA_VISIBLE_DEVICES=2 uv run v2/experiments/block_attention_heatmap.py \
        --subset qasper --context_lens 4096 --example_idx 0
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
import matplotlib.colors as mcolors
from datasets import load_dataset
from transformers import AutoTokenizer

from helper import load_model
from generation_functions import Fast_dLLM_QwenForCausalLM as GenClass

BLOCK_SIZE = 32

TEMPLATES = {
    "qasper": (
        "You are given a scientific paper and a question. Answer the question "
        "in as few words as possible based on the paper.\n\n"
        "Paper:\n{context}\n\nQuestion: {input}\n\nAnswer:"
    ),
    "narrativeqa": (
        "You are given a story and a question. Answer the question as concisely "
        "as possible.\n\nStory:\n{context}\n\nQuestion: {input}\n\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. "
        "Only give me the answer and do not output any other words.\n\n"
        "Passages:\n{context}\n\nQuestion: {input}\n\nAnswer:"
    ),
    "gov_report": (
        "You are given a government report. Write a one-paragraph summary.\n\n"
        "Report:\n{context}\n\nSummary:"
    ),
    "multifieldqa_en": (
        "Read the following passage and answer the question in as few words as possible.\n\n"
        "Passage:\n{context}\n\nQuestion: {input}\n\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the provided passages. "
        "Only give me the answer and do not output any other words.\n\n"
        "Passages:\n{context}\n\nQuestion: {input}\n\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the provided passages. "
        "Only give me the answer and do not output any other words.\n\n"
        "Passages:\n{context}\n\nQuestion: {input}\n\nAnswer:"
    ),
}


# ---------------------------------------------------------------------------
# Attention capture
# ---------------------------------------------------------------------------

class BlockAttentionCapture:
    """
    Wraps model.forward to record per-block attention scores at every commit step.
    No pruning — purely observational.
    """

    def __init__(self, model, block_size=BLOCK_SIZE):
        self.model      = model
        self.block_size = block_size
        self.records    = []   # list of (prefix_kv_len, block_scores [n_blocks])
        self._original_forward = None

    def install(self):
        capture       = self
        original_fwd  = self.model.forward
        self._original_forward = original_fwd

        def _wrapped(*args, **kwargs):
            pre_kv_len = 0
            try:
                pkv = kwargs.get('past_key_values', None)
                if pkv is not None:
                    pre_kv_len = pkv.layers[0].keys.shape[-2]
            except Exception:
                pass

            is_commit = (kwargs.get('update_past_key_values', False)
                         and not kwargs.get('use_block_cache', False)
                         and pre_kv_len > 0)

            if is_commit:
                for layer in capture.model.model.layers:
                    layer.self_attn._capture_attn_scores = True

            output = original_fwd(*args, **kwargs)

            if is_commit:
                for layer in capture.model.model.layers:
                    layer.self_attn._capture_attn_scores = False

                # read attention weights from all layers
                all_weights = []
                for layer in capture.model.model.layers:
                    w = getattr(layer.self_attn, '_last_attn_weights', None)
                    if w is not None:
                        all_weights.append(w)  # [1, H, Q, KV]

                if all_weights:
                    try:
                        prefix_kv_len = output.past_key_values.layers[0].keys.shape[-2]
                    except Exception:
                        prefix_kv_len = pre_kv_len

                    weights  = torch.stack(all_weights, dim=0).mean(dim=0)  # [1, H, Q, KV]
                    kv_len   = weights.shape[-1]
                    score_len = min(prefix_kv_len, kv_len)
                    n_blocks  = score_len // capture.block_size

                    if n_blocks > 0:
                        per_tok = weights[0, :, :, :n_blocks * capture.block_size].mean(dim=(0, 1))
                        block_scores = per_tok.reshape(n_blocks, capture.block_size).sum(dim=-1)
                        capture.records.append({
                            "prefix_kv_len": prefix_kv_len,
                            "n_blocks":      n_blocks,
                            "scores":        block_scores.float().cpu().numpy(),
                        })

            return output

        self.model.forward = _wrapped
        return self

    def uninstall(self):
        if self._original_forward is not None:
            self.model.forward = self._original_forward
            self._original_forward = None
        for layer in self.model.model.layers:
            layer.self_attn._capture_attn_scores = False


# ---------------------------------------------------------------------------
# Data loading (single example)
# ---------------------------------------------------------------------------

def load_example(subset, tokenizer, device, context_len, example_idx=0, seed=42):
    cache_dir = os.path.join(os.path.dirname(__file__), "../sparsity/.longbench_cache")
    jsonl_path = os.path.join(cache_dir, f"{subset}.jsonl")

    if not os.path.exists(jsonl_path):
        os.makedirs(cache_dir, exist_ok=True)
        zip_path = os.path.join(cache_dir, "data.zip")
        if not os.path.exists(zip_path):
            print("  Downloading LongBench data.zip...", flush=True)
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id="THUDM/LongBench", filename="data.zip",
                repo_type="dataset", local_dir=cache_dir,
            )
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(cache_dir)
        extracted = os.path.join(cache_dir, "data", f"{subset}.jsonl")
        if os.path.exists(extracted):
            shutil.move(extracted, jsonl_path)

    ds = load_dataset("json", data_files=jsonl_path, split="train")
    ds = ds.shuffle(seed=seed)
    ex = ds[example_idx]

    template = TEMPLATES.get(subset, TEMPLATES["narrativeqa"])
    context  = ex.get("context", "") or ex.get("input", "")
    question = ex.get("input", "")
    answers  = ex.get("answers", [])
    if isinstance(answers, str):
        answers = [answers]

    full_prompt = template.format(context=context, input=question)
    full_ids    = tokenizer(full_prompt, return_tensors="pt").input_ids[0]

    if len(full_ids) > context_len:
        parts       = template.split("{context}")
        prefix_len  = len(tokenizer(parts[0], return_tensors="pt").input_ids[0])
        suffix_raw  = parts[1].format(input=question) if "{input}" in parts[1] else parts[1]
        suffix_len  = len(tokenizer(suffix_raw, return_tensors="pt").input_ids[0])
        budget      = max(64, context_len - prefix_len - suffix_len)
        ctx_ids     = tokenizer(context, return_tensors="pt").input_ids[0][:budget]
        context     = tokenizer.decode(ctx_ids, skip_special_tokens=True)
        full_prompt = template.format(context=context, input=question)

    msgs = [{"role": "user", "content": full_prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(text, return_tensors="pt").to(device)

    actual_len = encoded.input_ids.shape[1]
    n_blocks   = actual_len // BLOCK_SIZE
    print(f"  prompt_len={actual_len} ({n_blocks} complete blocks)  "
          f"answers={answers[:1]}")
    return encoded, answers, actual_len


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_heatmap(records, context_len, subset, out_path=None):
    if not records:
        print("  No commit steps recorded — nothing to plot.")
        return

    # build padded matrix [n_steps × max_n_blocks]
    max_blocks = max(r["n_blocks"] for r in records)
    n_steps    = len(records)
    mat        = np.zeros((n_steps, max_blocks))

    for i, r in enumerate(records):
        nb = r["n_blocks"]
        mat[i, :nb] = r["scores"]

    # normalize each row independently so every commit step is comparable
    row_max = mat.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1.0
    mat_norm = mat / row_max

    # average attention score per block across all commit steps
    avg_scores = mat.mean(axis=0)
    total = avg_scores.sum()
    avg_norm = avg_scores / (avg_scores.max() + 1e-9)

    # find how many blocks capture 50% and 90% of average attention mass
    sorted_idx  = np.argsort(avg_scores)[::-1]
    cumsum      = np.cumsum(avg_scores[sorted_idx]) / (total + 1e-9)
    n_50        = int(np.searchsorted(cumsum, 0.50)) + 1
    n_90        = int(np.searchsorted(cumsum, 0.90)) + 1
    top90_idx   = set(sorted_idx[:n_90].tolist())
    top50_idx   = set(sorted_idx[:n_50].tolist())

    fig, axes = plt.subplots(
        2, 1, figsize=(min(18, max(10, max_blocks // 3)), 7),
        gridspec_kw={"height_ratios": [5, 1.5]}, sharex=True
    )
    fig.suptitle(
        f"Prefix block attention — {subset}, ctx≈{context_len} tokens, B=1\n"
        f"{n_steps} commit steps  |  {max_blocks} prefix blocks  |  "
        f"top-{n_50} blocks → 50% mass,  top-{n_90} blocks → 90% mass  "
        f"({n_90/max_blocks*100:.1f}% of cache)",
        fontsize=10
    )

    # --- main heatmap ---
    ax = axes[0]
    im = ax.imshow(mat_norm, aspect="auto", cmap="hot", vmin=0, vmax=1,
                   origin="upper", interpolation="nearest")
    ax.set_ylabel("Commit step")
    ax.set_yticks(range(0, n_steps, max(1, n_steps // 8)))
    plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01,
                 label="normalized attention mass (per row)")

    # shade columns that carry 90% of average mass
    for idx in top90_idx:
        ax.axvspan(idx - 0.5, idx + 0.5, color="deepskyblue", alpha=0.18)
    # shade columns that carry 50% of average mass (brighter)
    for idx in top50_idx:
        ax.axvspan(idx - 0.5, idx + 0.5, color="deepskyblue", alpha=0.35)

    # legend patches
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="deepskyblue", alpha=0.55, label=f"top-{n_50} blocks (50% mass)"),
        Patch(facecolor="deepskyblue", alpha=0.25, label=f"top-{n_90} blocks (90% mass)"),
    ], fontsize=8, loc="upper left")

    # --- average score bar ---
    ax2 = axes[1]
    colors = ["deepskyblue" if i in top90_idx else "steelblue"
              for i in range(max_blocks)]
    ax2.bar(range(max_blocks), avg_norm, width=1.0, color=colors, alpha=0.85)
    ax2.set_xlabel("Prefix block index  (left = start of document, right = most recent)")
    ax2.set_ylabel("Avg attn\n(norm)")
    ax2.set_xlim(-0.5, max_blocks - 0.5)
    ax2.set_ylim(0, 1.15)
    ax2.axhline(avg_norm[sorted_idx[n_90 - 1]], color="deepskyblue",
                linestyle="--", linewidth=0.8, alpha=0.7, label="90% threshold")
    ax2.legend(fontsize=7)

    plt.tight_layout()

    if out_path is None:
        out_path = os.path.join(
            os.path.dirname(__file__),
            f"block_attn_heatmap_{subset}_ctx{context_len}.png"
        )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
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
    parser.add_argument("--context_lens", type=int, nargs="+", default=[2048, 4096, 8192])
    parser.add_argument("--example_idx", type=int, default=0,
                        help="Which shuffled example to use (same seed=42 shuffle as eval scripts)")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    print("Loading model...")
    model     = load_model(args.model, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    device    = args.device

    max_pos = getattr(model.config, 'max_position_embeddings', 32768)
    valid_lens = [c for c in args.context_lens if c <= max_pos]

    for ctx_len in valid_lens:
        print(f"\n{'='*60}")
        print(f"ctx_len={ctx_len}  subset={args.subset}  example={args.example_idx}")
        print(f"{'='*60}")

        torch.cuda.empty_cache()

        encoded, answers, prompt_len = load_example(
            args.subset, tokenizer, device, ctx_len,
            example_idx=args.example_idx, seed=args.seed
        )

        capture = BlockAttentionCapture(model)
        capture.install()
        sampler = types.MethodType(GenClass.batch_sample, model)

        try:
            seq_len_t = torch.tensor([prompt_len], dtype=torch.long, device=device)
            output = sampler(
                input_ids        = encoded.input_ids,
                tokenizer        = tokenizer,
                block_size       = BLOCK_SIZE,
                max_new_tokens   = args.max_new_tokens,
                small_block_size = 8,
                min_len          = prompt_len,
                seq_len          = seq_len_t,
                threshold        = 0.95,
                temperature      = 0.0,
            )
        finally:
            capture.uninstall()

        gen_ids  = output[0][prompt_len:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        gold_str = answers[0][:100] if answers else "?"
        print(f"  generated: {gen_text[:150]!r}")
        print(f"  gold:      {gold_str!r}")
        print(f"  commit steps recorded: {len(capture.records)}")

        if capture.records:
            last     = capture.records[-1]
            scores   = last["scores"]
            n_blks   = last["n_blocks"]
            total    = scores.sum()
            sorted_idx = np.argsort(scores)[::-1]
            sorted_s   = scores[sorted_idx]
            cumsum     = np.cumsum(sorted_s) / (total + 1e-9)
            blocks_90  = int(np.searchsorted(cumsum, 0.90)) + 1
            blocks_50  = int(np.searchsorted(cumsum, 0.50)) + 1

            print(f"  last commit: {n_blks} blocks | "
                  f"50% mass in top-{blocks_50} | "
                  f"90% mass in top-{blocks_90} ({blocks_90/n_blks*100:.1f}%)")

            # print which block indices carry the mass
            top90_blocks = sorted_idx[:blocks_90].tolist()
            top50_blocks = sorted_idx[:blocks_50].tolist()

            def classify(idx, n):
                if idx <= 1:
                    return "sink"
                elif idx >= n - 2:
                    return "recent"
                else:
                    return f"content(blk={idx})"

            print(f"  top-{blocks_50} blocks (50% mass): "
                  + ", ".join(f"{i}[{classify(i, n_blks)}]" for i in sorted(top50_blocks)))
            print(f"  top-{blocks_90} blocks (90% mass): "
                  + ", ".join(f"{i}[{classify(i, n_blks)}]" for i in sorted(top90_blocks)))

            # how much mass is pure sink+recency vs content
            sink_recent = {j for j in range(n_blks) if j <= 1 or j >= n_blks - 2}
            sr_mass  = scores[list(sink_recent)].sum() / (total + 1e-9)
            content_blocks_in_90 = [i for i in top90_blocks if i not in sink_recent]
            print(f"  sink+recency (blks 0-1 and last 2): {sr_mass*100:.1f}% of total mass")
            print(f"  content blocks in top-90%: {content_blocks_in_90 if content_blocks_in_90 else 'none — pure sink+recency'}")

            plot_heatmap(capture.records, ctx_len, args.subset)

    print("\nDone.")


if __name__ == "__main__":
    main()
