"""
Attention concentration analysis on Qwen2-7B-Instruct (autoregressive baseline).

Runs the same frac90 / sink+recency analysis as block_attention_dist.py but on
the standard AR model, so we can compare attention sparsity between diffusion
and autoregressive generation on identical prompts and context lengths.

AR generation: attention weights are [1, H, 1, KV] at each token step.
We accumulate block-level scores across all generated tokens, then compute
the same frac90 metric (how many prefix blocks capture 90% of total mass).

Run:
    CUDA_VISIBLE_DEVICES=2 uv run v2/experiments/ar_attention_dist.py \
        --subset narrativeqa --context_lens 4096 8192 --n_examples 40

    CUDA_VISIBLE_DEVICES=2 uv run v2/experiments/ar_attention_dist.py \
        --subset gov_report --context_lens 4096 8192 --n_examples 40 \
        --max_new_tokens 256
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import zipfile
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from block_attention_heatmap import TEMPLATES, BLOCK_SIZE

AR_MODEL_ID = "Qwen/Qwen2-7B-Instruct"


# ---------------------------------------------------------------------------
# Attention capture hook
# ---------------------------------------------------------------------------

class ARAttentionAccumulator:
    """
    Registers forward hooks on all attention layers to accumulate per-block
    attention scores across generated tokens.

    At each token step, each layer outputs attn_weights [1, H, 1, KV].
    We average across heads and accumulate the per-block sum across steps.
    """

    def __init__(self, model, block_size=BLOCK_SIZE):
        self.model      = model
        self.block_size = block_size
        self._hooks     = []
        self._step_weights = []  # per step: mean over layers of [KV] vector

    def install(self):
        def make_hook(layer_idx):
            def hook(module, input, output):
                # output is (attn_output, attn_weights, past_kv) or just attn_output
                # with output_attentions=True, attn_weights is the second element
                if isinstance(output, tuple) and len(output) > 1:
                    w = output[1]  # [B, H, q_len, KV]
                    if w is not None and w.shape[-2] == 1:  # only during generation
                        # mean over heads, squeeze q dim → [KV]
                        self._layer_buf[layer_idx] = w[0, :, 0, :].mean(0).float().cpu()
            return hook

        n_layers = len(self.model.model.layers)
        self._layer_buf = [None] * n_layers
        for i, layer in enumerate(self.model.model.layers):
            h = layer.self_attn.register_forward_hook(make_hook(i))
            self._hooks.append(h)
        return self

    def flush_step(self):
        """Call after each generated token to accumulate the step's attention."""
        valid = [b for b in self._layer_buf if b is not None]
        if valid:
            # average across layers → [KV]
            mean_kv = torch.stack(valid, 0).mean(0)
            self._step_weights.append(mean_kv)
        self._layer_buf = [None] * len(self._layer_buf)

    def uninstall(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def block_scores(self, prefix_len):
        """
        Average accumulated attention over all steps, bin into blocks,
        return numpy array of shape [n_blocks].
        """
        if not self._step_weights:
            return None

        # truncate each step to prefix_len before stacking (KV grows as tokens are generated)
        truncated = [w[:prefix_len] for w in self._step_weights]
        score_len = min(prefix_len, truncated[0].shape[0])
        all_w = torch.stack([w[:score_len] for w in truncated], 0).mean(0)  # [score_len]
        n_blocks  = score_len // self.block_size
        if n_blocks == 0:
            return None

        prefix_w = all_w[:n_blocks * self.block_size]
        scores   = prefix_w.reshape(n_blocks, self.block_size).sum(-1).numpy()
        return scores


# ---------------------------------------------------------------------------
# Data loading (reuses same logic / cache as block_attention_dist.py)
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
            parts   = template.split("{context}")
            pfx_len = len(tokenizer(parts[0], return_tensors="pt").input_ids[0])
            sfx_raw = parts[1].format(input=question) if "{input}" in parts[1] else parts[1]
            sfx_len = len(tokenizer(sfx_raw, return_tensors="pt").input_ids[0])
            budget  = max(64, context_len - pfx_len - sfx_len)
            ctx_ids = tokenizer(context, return_tensors="pt").input_ids[0][:budget]
            context = tokenizer.decode(ctx_ids, skip_special_tokens=True)
            full_prompt = template.format(context=context, input=question)

        msgs    = [{"role": "user", "content": full_prompt}]
        text    = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(text, return_tensors="pt").to(device)
        examples.append((encoded, answers))

    return examples


# ---------------------------------------------------------------------------
# Run one example
# ---------------------------------------------------------------------------

def probe_example(model, tokenizer, encoded, device, max_new_tokens=64, single_step=False):
    prompt_len = encoded.input_ids.shape[1]
    accum = ARAttentionAccumulator(model)
    accum.install()

    input_ids = encoded.input_ids
    past_key_values = None
    generated = []

    try:
        with torch.no_grad():
            # prefill — don't capture here, q_len=prompt_len would swamp block scores
            prefill_out = model(input_ids=input_ids, use_cache=True, output_attentions=False)
            past_key_values = prefill_out.past_key_values
            next_id = prefill_out.logits[0, -1].argmax(dim=-1, keepdim=True)
            generated.append(next_id.item())
            input_ids = next_id.unsqueeze(0)  # [1, 1]

            # generation steps — q_len=1, hooks capture properly
            steps = 1 if single_step else max_new_tokens
            for _ in range(steps):
                if generated[-1] == tokenizer.eos_token_id:
                    break
                out = model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=True,
                )
                accum.flush_step()

                next_id = out.logits[0, -1].argmax(dim=-1, keepdim=True)
                generated.append(next_id.item())
                past_key_values = out.past_key_values
                input_ids = next_id.unsqueeze(0)
    finally:
        accum.uninstall()

    scores = accum.block_scores(prompt_len)
    if scores is None:
        return None

    n_blks = len(scores)
    total  = scores.sum()
    if total == 0:
        return None

    sorted_idx = np.argsort(scores)[::-1]
    cumsum     = np.cumsum(scores[sorted_idx]) / total
    k90 = int(np.searchsorted(cumsum, 0.90)) + 1
    k50 = int(np.searchsorted(cumsum, 0.50)) + 1

    sink_recent_idx = {j for j in range(n_blks) if j <= 1 or j >= n_blks - 2}
    sr_mass = scores[list(sink_recent_idx)].sum() / total

    gen_text = tokenizer.decode(generated, skip_special_tokens=True).strip()

    return {
        "n_blocks":  n_blks,
        "k90":       k90,
        "frac90":    k90 / n_blks,
        "k50":       k50,
        "frac50":    k50 / n_blks,
        "sr_mass":   float(sr_mass),
        "gen_tokens": len(generated),
        "gen_text":  gen_text[:80],
    }


# ---------------------------------------------------------------------------
# Plot — same format as block_attention_dist.py for direct visual comparison
# ---------------------------------------------------------------------------

def plot_dist(results_by_ctx, subset, out_path=None):
    ctx_lens = sorted(k for k in results_by_ctx.keys() if isinstance(k, int))
    n_panels = len(ctx_lens)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), sharey=False)
    if n_panels == 1:
        axes = [axes]

    mode_str = "single-step (first token)" if results_by_ctx.get("_single_step") else "avg over all generated tokens"
    fig.suptitle(f"AR (Qwen2-7B-Instruct) attention concentration — {subset}, B=1\n"
                 f"frac = (blocks for 90% mass) / N  |  {mode_str}",
                 fontsize=11)

    colors = ["#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

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

        ax2 = ax.twinx()
        ax2.hist(sr, bins=bins, color="orange", alpha=0.3, edgecolor="none")
        ax2.set_ylabel("Count (sink+recency mass)", color="orange", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="orange")
        ax2.set_ylim(0, ax2.get_ylim()[1] * 3)

    fig.text(0.5, 0.01, "Orange = sink+recency mass distribution",
             ha="center", fontsize=8, color="orange")
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if out_path is None:
        out_path = os.path.join(
            os.path.dirname(__file__),
            f"ar_attn_dist_{subset}.png"
        )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          type=str, default=AR_MODEL_ID)
    parser.add_argument("--device",         type=str, default="cuda:0")
    parser.add_argument("--subset",         type=str, default="narrativeqa",
                        choices=list(TEMPLATES.keys()))
    parser.add_argument("--context_lens",   type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--n_examples",     type=int, default=40)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--single_step",    action="store_true",
                        help="Capture attention at first generated token only (controls for query diversity)")
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()
    print(f"  loaded: {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")

    results_by_ctx = {}

    for ctx_len in args.context_lens:
        print(f"\n{'='*60}")
        print(f"ctx_len={ctx_len}  subset={args.subset}  n={args.n_examples}  model=AR")
        print(f"{'='*60}")

        torch.cuda.empty_cache()
        examples = load_examples(
            args.subset, tokenizer, args.device, ctx_len,
            args.n_examples, args.seed,
        )

        stats_list = []
        skipped    = 0

        for i, (encoded, answers) in enumerate(examples):
            prompt_len = encoded.input_ids.shape[1]
            print(f"  [{i+1:2d}/{len(examples)}] prompt_len={prompt_len}", end="", flush=True)

            try:
                s = probe_example(model, tokenizer, encoded, args.device,
                                  args.max_new_tokens, args.single_step)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(" OOM — skip")
                skipped += 1
                continue

            if s is None:
                print(" no attention — skip")
                skipped += 1
                continue

            stats_list.append(s)
            print(f"  frac90={s['frac90']:.1%}  sr={s['sr_mass']:.1%}  gen={s['gen_tokens']}tok")

        if not stats_list:
            print("  No valid examples.")
            continue

        results_by_ctx[ctx_len] = stats_list

        fracs = [s["frac90"] for s in stats_list]
        srs   = [s["sr_mass"] for s in stats_list]
        print(f"\n  Summary ({len(stats_list)} valid, {skipped} skipped):")
        print(f"  frac90 — mean={np.mean(fracs):.1%}  median={np.median(fracs):.1%}  "
              f"std={np.std(fracs):.1%}  min={np.min(fracs):.1%}  max={np.max(fracs):.1%}")
        print(f"  sink+recency mass — mean={np.mean(srs):.1%}  std={np.std(srs):.1%}")

        p25, p75 = np.percentile(fracs, [25, 75])
        tight  = sum(1 for f in fracs if f < 0.20)
        medium = sum(1 for f in fracs if 0.20 <= f < 0.40)
        high   = sum(1 for f in fracs if f >= 0.40)
        print(f"  frac90 < 20%: {tight}  |  20-40%: {medium}  |  ≥40%: {high}")
        print(f"  P25={p25:.1%}  P75={p75:.1%}")

    if results_by_ctx:
        results_by_ctx["_single_step"] = args.single_step
        suffix = "_single_step" if args.single_step else ""
        out = os.path.join(os.path.dirname(__file__), f"ar_attn_dist_{args.subset}{suffix}.png")
        plot_dist(results_by_ctx, args.subset, out_path=out)

    print("\nDone.")


if __name__ == "__main__":
    main()
