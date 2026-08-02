"""
PER-STEP ATTENTION-MASS STRUCTURE — how prefix attention mass is distributed over blocks
and how it EVOLVES across denoising steps (NOT averaged over steps).

Uses the true-softmax `_block_mass` capture added to modeling.py (guarded, small steps only).
Runs DENSE generation (no top-K) so you see the real, unpruned distribution. For each
example it produces:
  - a HEATMAP: rows = denoising steps, cols = prefix blocks, color = attention mass
    (saved to v2/sparsity/figures/attn_mass_<eval>_ex<i>.png)
  - CONCRETE NUMBERS per step: sink% (blocks 0-1), recency% (last 2 blocks), content%
    (middle), frac90 (fraction of blocks holding 90% of mass), entropy of the block dist.

Start set: gsm8k (short ctx) + narrativeqa (8k). Extensible via EVALS below.

Scope notes (scale-ups, not bundled):
  - MBPP/HumanEval need a code data loader; AIME needs its data; RULER needs data-gen.
    Add an entry to EVALS with load()/prompt() and they work.
  - AR (Qwen2.5-Instruct) is a separate model/pipeline (no denoising steps) — see the
    existing v2/experiments/ar_attention_dist.py for the AR side; this script is the
    block-diffusion (per-step) side.

Run:
  CUDA_VISIBLE_DEVICES=0 uv run v2/sparsity/attn_mass_analysis.py \
      --evals gsm8k narrativeqa --n_examples 3 --max_new_tokens 128
"""
import sys
import os
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "../experiments"))
sys.path.insert(0, os.path.join(HERE, "../train_scripts"))

import json
import types
import argparse
import torch

from transformers import AutoTokenizer
from helper import load_model
from generation_functions import Fast_dLLM_QwenForCausalLM as GenClass
from nfe_probe_stage1 import EVALS as SHORT_EVALS
from eval_longbench_sweep import load_longbench_subset

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

BLOCK_SIZE = 32
FIG_DIR = os.path.join(HERE, "figures")


class MassCapture:
    """Wraps model.forward; on each small step, turns on the true-softmax mass capture,
    then aggregates per-layer [B,L_kv] into per-PREFIX-BLOCK mass, averaged over layers.
    Records one row per small step: block_mass[n_prefix_blocks] (normalized over prefix)."""
    def __init__(self, model, block_size=BLOCK_SIZE):
        self.model = model
        self.block_size = block_size
        self.records = []          # list of 1D np/torch arrays (per-block prefix mass)
        self._orig = None

    def _set(self, v):
        for layer in self.model.model.layers:
            layer.self_attn._capture_block_mass = v

    def install(self):
        self._orig = self.model.forward
        self.records = []

        def wrapped(*a, **k):
            is_small = (k.get('use_block_cache', False)
                        and not k.get('update_past_key_values', False))
            if is_small:
                self._set(True)
            out = self._orig(*a, **k)
            if is_small:
                self._set(False)
                masses = []
                for layer in self.model.model.layers:
                    bm = getattr(layer.self_attn, '_block_mass', None)
                    if bm is not None:
                        masses.append(bm[0])           # [L_kv] (B=1)
                        layer.self_attn._block_mass = None
                if masses:
                    M = torch.stack(masses).mean(0)     # [L_kv] avg over layers
                    prefix_len = M.shape[0] - self.block_size
                    if prefix_len >= self.block_size:
                        nb = prefix_len // self.block_size
                        blk = M[:nb * self.block_size].reshape(nb, self.block_size).sum(1)
                        s = blk.sum()
                        self.records.append((blk / s if s > 0 else blk).numpy())
            return out

        self.model.forward = wrapped
        return self

    def uninstall(self):
        if self._orig is not None:
            self.model.forward = self._orig
            self._orig = None
        self._set(False)


def step_numbers(blk, n_sink=2, n_recency=2):
    """Concrete numbers for one step's per-block mass (sums to ~1)."""
    import numpy as np
    nb = len(blk)
    sink = float(blk[:min(n_sink, nb)].sum())
    rec = float(blk[max(0, nb - n_recency):].sum())
    content = max(0.0, 1.0 - sink - rec)
    order = np.sort(blk)[::-1]
    cum = np.cumsum(order)
    k90 = int(np.searchsorted(cum, 0.9) + 1)
    frac90 = k90 / nb if nb else 0.0
    p = blk[blk > 0]
    ent = float(-(p * np.log(p)).sum()) if len(p) else 0.0
    return dict(sink=sink, recency=rec, content=content, frac90=frac90,
                entropy=ent, n_blocks=nb)


@torch.no_grad()
def run_example(model, sampler, tokenizer, ids, device, max_new_tokens):
    cap = MassCapture(model).install()
    plen = ids.shape[1]
    try:
        sampler(input_ids=ids, tokenizer=tokenizer, block_size=BLOCK_SIZE,
                max_new_tokens=max_new_tokens, small_block_size=8, min_len=plen,
                seq_len=torch.full((1,), plen, dtype=torch.long, device=device),
                threshold=0.95, temperature=0.0, use_block_cache=True)
    finally:
        cap.uninstall()
    return cap.records


def get_examples(eval_name, tokenizer, device, n, ctx, seed):
    if eval_name in SHORT_EVALS:
        ev = SHORT_EVALS[eval_name]
        data = ev["load"](n, seed)
        return [tokenizer(ev["prompt"](q), return_tensors="pt").input_ids.to(device)
                for q, _ in data]
    # else treat as a LongBench subset
    batches = load_longbench_subset(eval_name, tokenizer, device, ctx, 1, n, seed)
    return [enc.input_ids.to(device) for enc, _ in batches]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--evals", nargs="+", default=["gsm8k", "narrativeqa"])
    p.add_argument("--n_examples", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--ctx", type=int, default=8192, help="target context for LongBench evals")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_json", default=os.path.join(HERE, "attn_mass_numbers.json"))
    args = p.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    print("Loading model...", flush=True)
    model = load_model(args.model, device=args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    sampler = types.MethodType(GenClass.batch_sample, model)

    dump = {}
    for eval_name in args.evals:
        try:
            examples = get_examples(eval_name, tokenizer, args.device,
                                    args.n_examples, args.ctx, args.seed)
        except Exception as e:
            print(f"  [skip] {eval_name}: {e}", flush=True)
            continue

        print(f"\n{'='*74}\n{eval_name}\n{'='*74}", flush=True)
        dump[eval_name] = []
        for i, ids in enumerate(examples):
            recs = run_example(model, sampler, tokenizer, ids, args.device,
                               args.max_new_tokens)
            if not recs:
                print(f"  ex{i}: no steps captured (context too short?)", flush=True)
                continue

            import numpy as np
            nums = [step_numbers(r) for r in recs]
            n_blocks = nums[0]["n_blocks"]
            print(f"\n  ex{i}  (plen={ids.shape[1]}, {n_blocks} prefix blocks, "
                  f"{len(recs)} steps)")
            print(f"    {'step':>4} {'sink%':>6} {'recency%':>8} {'content%':>8} "
                  f"{'frac90':>7} {'entropy':>7}")
            # print a downsampled set of steps so it stays readable
            idxs = sorted(set([0, len(nums) // 4, len(nums) // 2,
                               3 * len(nums) // 4, len(nums) - 1]))
            for s in idxs:
                n = nums[s]
                print(f"    {s:>4} {n['sink']:>6.1%} {n['recency']:>8.1%} "
                      f"{n['content']:>8.1%} {n['frac90']:>7.1%} {n['entropy']:>7.2f}")
            # averages across steps (for the summary line)
            avg = {k: float(np.mean([n[k] for n in nums]))
                   for k in ["sink", "recency", "content", "frac90", "entropy"]}
            print(f"    {'mean':>4} {avg['sink']:>6.1%} {avg['recency']:>8.1%} "
                  f"{avg['content']:>8.1%} {avg['frac90']:>7.1%} {avg['entropy']:>7.2f}")
            dump[eval_name].append({"plen": int(ids.shape[1]), "n_blocks": n_blocks,
                                    "per_step": nums, "mean": avg})

            # heatmap: steps x blocks. Prefix GROWS across blocks (each commit adds a
            # block), so records have varying length — pad with NaN into a rectangle.
            if HAVE_MPL:
                maxb = max(len(r) for r in recs)
                H = np.full((len(recs), maxb), np.nan)
                for j, r in enumerate(recs):
                    H[j, :len(r)] = r
                cmap = plt.get_cmap("viridis").copy()
                cmap.set_bad("lightgray")  # not-yet-generated blocks
                plt.figure(figsize=(min(14, 0.06 * maxb + 4), 0.18 * len(recs) + 2))
                plt.imshow(H, aspect="auto", cmap=cmap,
                           vmax=np.nanpercentile(H, 99))
                plt.colorbar(label="prefix attention mass")
                plt.xlabel("prefix block (0=sink … N=recency)")
                plt.ylabel("denoising step")
                plt.title(f"{eval_name} ex{i}: per-step prefix attention mass")
                out = os.path.join(FIG_DIR, f"attn_mass_{eval_name}_ex{i}.png")
                plt.tight_layout(); plt.savefig(out, dpi=110); plt.close()
                print(f"    heatmap -> {out}")

    with open(args.save_json, "w") as f:
        json.dump(dump, f, indent=2)
    print(f"\nNumbers saved -> {args.save_json}")
    if not HAVE_MPL:
        print("(matplotlib not available — numbers only; pip install matplotlib for heatmaps)")
    print("\nRead: sink+recency high & flat across steps = position-dominated (StreamingLLM-like).")
    print("Content% rising mid-trajectory, or frac90 growing, = real content routing — the")
    print("regime where selection could beat position-only. Compare short (gsm8k) vs long ctx.")


if __name__ == "__main__":
    main()
