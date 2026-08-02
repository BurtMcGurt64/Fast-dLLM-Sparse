"""
PRUNE PROBE (Phase 1: quality gate) — one-shot Wanda 2:4 weight pruning on Fast-dLLM v2.

The question: does a SOTA-DISTILLED block-diffusion LLM survive 50% structured (2:4)
weight sparsity, or crater the way it resisted layer-skipping / NFE reduction? If it
survives, weight pruning composes with KV top-K to attack the ~46% weight floor (the
thing that caps KV sparsity at ~2x). If it craters, that's a publishable figure that
unifies with the KV / layer / NFE closures under one "no slack" story.

Wanda metric (Sun et al. 2306.11695): score_ij = |W_ij| * ||X_j||_2, keep top-N of
every M consecutive input channels per output row (2:4 -> keep 2 of every 4).

Diffusion-honest calibration: activation norms ||X_j|| are collected ONLY during the
denoising forwards we actually run and speed up (small-step + commit), NOT prefill's
clean prompt. Forward type is read exactly as topk.py does (update_past_key_values /
use_block_cache / prefix-kv-len).

Phase 1 needs NO sparse kernel: we zero weights in-place (bf16) and measure accuracy.
The real 2:4 tensor-core speedup (cuSPARSELt) is a separate Phase-2 integration and is
only worth wiring if this quality gate passes.

Run:
  CUDA_VISIBLE_DEVICES=0 uv run v2/sparsity/prune_probe.py --calib_n 16 --eval_n 100
  # compose with KV top-K (only meaningful if quality holds):
  CUDA_VISIBLE_DEVICES=0 uv run v2/sparsity/prune_probe.py --eval_n 100 --compose_topk --frac 0.2
"""
import sys
import os
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "../experiments"))
sys.path.insert(0, os.path.join(HERE, "../train_scripts"))

import argparse
import types
import torch

from transformers import AutoTokenizer
from helper import load_model
from generation_functions import Fast_dLLM_QwenForCausalLM as GenClass
from topk import TopKBlockLoader
from nfe_probe_stage1 import EVALS, build_batches

BLOCK_SIZE = 32
PROJ_NAMES = ["q_proj", "k_proj", "v_proj", "o_proj"]      # attention
MLP_NAMES = ["gate_proj", "up_proj", "down_proj"]           # mlp
# lm_head is EXCLUDED (tied to embed_tokens; pruning it corrupts the embedding table,
# and it never runs in the small step we care about).


def get_prunable_linears(model):
    """dict name -> nn.Linear for the 7*n_layers prunable projections (no lm_head)."""
    out = {}
    for i, layer in enumerate(model.model.layers):
        for n in PROJ_NAMES:
            out[f"layers.{i}.self_attn.{n}"] = getattr(layer.self_attn, n)
        for n in MLP_NAMES:
            out[f"layers.{i}.mlp.{n}"] = getattr(layer.mlp, n)
    return out


class Calibrator:
    """Accumulates per-input-channel activation energy (sum x^2, fp32) on each prunable
    linear, gated to small-step + commit forwards only. Wraps model.forward for the gate
    and registers a forward-pre-hook per linear."""

    def __init__(self, model):
        self.model = model
        self.linears = get_prunable_linears(model)
        self.sumsq = {}       # name -> [in_features] fp32
        self.ntok = {}        # name -> token count (sanity)
        self.active = False   # set per-forward by the type gate
        self._orig_forward = None
        self._handles = []

    def _make_hook(self, name):
        def hook(module, args):
            if not self.active:
                return
            x = args[0]
            x = x.reshape(-1, x.shape[-1]).float()
            s = (x * x).sum(dim=0)
            if name not in self.sumsq:
                self.sumsq[name] = s
                self.ntok[name] = x.shape[0]
            else:
                self.sumsq[name] += s
                self.ntok[name] += x.shape[0]
        return hook

    def __enter__(self):
        for name, lin in self.linears.items():
            self._handles.append(lin.register_forward_pre_hook(self._make_hook(name)))

        self._orig_forward = self.model.forward
        orig = self._orig_forward

        def gated_forward(*a, **k):
            # forward-type detection — identical logic to topk.py's _wrapped_forward
            pre_kv_len = 0
            pkv = k.get("past_key_values", None)
            if pkv is not None:
                try:
                    pre_kv_len = pkv.layers[0].keys.shape[-2]
                except Exception:
                    pre_kv_len = 0
            upd = k.get("update_past_key_values", False)
            is_commit = upd and not k.get("use_block_cache", False) and pre_kv_len > 0
            is_small = (pre_kv_len > 0) and not upd
            self.active = bool(is_small or is_commit)   # exclude prefill
            try:
                return orig(*a, **k)
            finally:
                self.active = False

        self.model.forward = gated_forward
        return self

    def __exit__(self, *x):
        for h in self._handles:
            h.remove()
        self._handles = []
        if self._orig_forward is not None:
            self.model.forward = self._orig_forward
            self._orig_forward = None

    def act_norm(self, name):
        """||X_j||_2 per input channel."""
        return self.sumsq[name].sqrt()


def nm_mask(score, n=2, m=4):
    """Boolean keep-mask: within every M consecutive input channels of each output row,
    keep the N highest-scoring. score: [out, in]."""
    out, inp = score.shape
    assert inp % m == 0, f"in_features {inp} not divisible by {m}"
    g = score.reshape(out, inp // m, m)
    keep_idx = g.argsort(dim=-1, descending=True)[..., :n]
    mask = torch.zeros_like(g, dtype=torch.bool)
    mask.scatter_(-1, keep_idx, True)
    return mask.reshape(out, inp)


@torch.no_grad()
def prune_wanda(model, calib, n=2, m=4, verbose=True):
    """Compute + apply Wanda N:M masks in-place. Returns dict name -> realized sparsity."""
    stats = {}
    for name, lin in calib.linears.items():
        if name not in calib.sumsq:
            if verbose:
                print(f"  [warn] {name}: no calibration activations captured — skipped", flush=True)
            continue
        W = lin.weight.data
        act = calib.act_norm(name).to(W.dtype)          # [in]
        score = W.abs().float() * act.float()[None, :]  # [out, in], Wanda metric
        mask = nm_mask(score, n=n, m=m).to(W.device)
        W.mul_(mask)                                     # zero pruned weights in place
        stats[name] = 1.0 - mask.float().mean().item()
    if verbose:
        avg = sum(stats.values()) / len(stats) if stats else 0.0
        print(f"  pruned {len(stats)} linears | mean sparsity {avg:.3f} "
              f"(target {1 - n / m:.3f})", flush=True)
    return stats


@torch.no_grad()
def run_calibration(model, sampler, tokenizer, prompts, device, calib_gen):
    """Generate on calibration prompts (DENSE, no topk) with the gated hooks active."""
    calib = Calibrator(model)
    with calib:
        for p in prompts:
            ids = tokenizer(p, return_tensors="pt").input_ids.to(device)
            plen = ids.shape[1]
            sampler(input_ids=ids, tokenizer=tokenizer, block_size=BLOCK_SIZE,
                    max_new_tokens=calib_gen, small_block_size=8, min_len=plen,
                    seq_len=torch.tensor([plen], device=device),
                    threshold=0.95, temperature=0.0, use_block_cache=True)
    return calib


@torch.no_grad()
def eval_gsm8k(model, sampler, tokenizer, batches, parse_pred, max_new, device):
    """Batched GSM8K accuracy (bs=1 recommended — ragged prompts)."""
    correct = total = 0
    for (ids, seq_len, min_len, true_lens), golds in batches:
        out = sampler(input_ids=ids, tokenizer=tokenizer, block_size=BLOCK_SIZE,
                      max_new_tokens=max_new, small_block_size=8, min_len=min_len,
                      seq_len=seq_len, threshold=0.95, temperature=0.0,
                      use_block_cache=True)
        for b in range(len(true_lens)):
            gen = tokenizer.decode(out[b][true_lens[b]:], skip_special_tokens=True)
            pred = parse_pred(gen)
            correct += int(pred is not None and pred == golds[b])
            total += 1
    return correct / total if total else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--sparsity", default="2:4", help="N:M pattern")
    p.add_argument("--calib_n", type=int, default=16, help="# calibration prompts")
    p.add_argument("--calib_gen", type=int, default=128, help="tokens generated per calib prompt")
    p.add_argument("--eval_n", type=int, default=100, help="# GSM8K eval examples")
    p.add_argument("--eval_gen", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_dense", action="store_true", help="skip the pre-prune baseline eval")
    p.add_argument("--compose_topk", action="store_true",
                   help="also eval pruned model WITH KV top-K installed (composition test)")
    p.add_argument("--frac", type=float, default=0.2, help="KV top-K budget for --compose_topk")
    args = p.parse_args()

    n, m = (int(x) for x in args.sparsity.split(":"))

    print("Loading model...", flush=True)
    model = load_model(args.model, device=args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    sampler = types.MethodType(GenClass.batch_sample, model)

    ev = EVALS["gsm8k"]
    # calibration prompts (disjoint offset from eval set to avoid overlap)
    calib_data = ev["load"](args.calib_n, args.seed + 1000)
    calib_prompts = [ev["prompt"](q) for q, _ in calib_data]
    # eval set (bs=1: gsm8k is ragged — see CLAUDE.md batched-decoding bug)
    eval_data = ev["load"](args.eval_n, args.seed)
    eval_prompts = [ev["prompt"](q) for q, _ in eval_data]
    eval_golds = [g for _, g in eval_data]
    eval_batches = build_batches(eval_prompts, eval_golds, tokenizer, args.device, 1)

    dense_acc = None
    if not args.skip_dense:
        print(f"\nEval DENSE (baseline) on {args.eval_n} GSM8K...", flush=True)
        dense_acc = eval_gsm8k(model, sampler, tokenizer, eval_batches,
                               ev["parse_pred"], args.eval_gen, args.device)
        print(f"  dense acc = {dense_acc:.3f}", flush=True)

    print(f"\nCalibrating on {args.calib_n} prompts (small-step + commit forwards only)...", flush=True)
    calib = run_calibration(model, sampler, tokenizer, calib_prompts, args.device, args.calib_gen)
    tok_range = (min(calib.ntok.values()), max(calib.ntok.values())) if calib.ntok else (0, 0)
    print(f"  captured {len(calib.sumsq)} linears | calib tokens/linear "
          f"{tok_range[0]}..{tok_range[1]}", flush=True)

    print(f"\nPruning Wanda {args.sparsity}...", flush=True)
    prune_wanda(model, calib, n=n, m=m)
    del calib
    torch.cuda.empty_cache()

    print(f"\nEval PRUNED ({args.sparsity}) on {args.eval_n} GSM8K...", flush=True)
    pruned_acc = eval_gsm8k(model, sampler, tokenizer, eval_batches,
                            ev["parse_pred"], args.eval_gen, args.device)
    print(f"  pruned acc = {pruned_acc:.3f}", flush=True)

    compose_acc = None
    if args.compose_topk:
        print(f"\nEval PRUNED + KV top-K (frac={args.frac}, pinned) on GSM8K...", flush=True)
        loader = TopKBlockLoader(model, frac=args.frac, selection="commit",
                                 n_sink=2, n_recency=8).install()
        try:
            compose_acc = eval_gsm8k(model, sampler, tokenizer, eval_batches,
                                     ev["parse_pred"], args.eval_gen, args.device)
        finally:
            loader.uninstall()
        print(f"  pruned+topk acc = {compose_acc:.3f}", flush=True)

    print(f"\n{'='*54}\nSUMMARY (GSM8K acc, n={args.eval_n})\n{'='*54}")
    if dense_acc is not None:
        print(f"  dense              {dense_acc:.3f}")
    print(f"  wanda {args.sparsity}         {pruned_acc:.3f}"
          + (f"   (drop {dense_acc - pruned_acc:+.3f})" if dense_acc is not None else ""))
    if compose_acc is not None:
        print(f"  wanda {args.sparsity} + topk  {compose_acc:.3f}")
    print("\n  Gate: if wanda-2:4 acc collapses -> v2 has no weight slack (publishable")
    print("  figure, unifies with KV/layer/NFE closures). If it holds -> proceed to")
    print("  Phase 2 (real cuSPARSELt 2:4 kernel + composed e2e speedup).")


if __name__ == "__main__":
    main()
