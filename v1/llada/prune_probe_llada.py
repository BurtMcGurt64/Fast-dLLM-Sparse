"""
PRUNE PROBE — LLaDA (Tier B: the two-model contrast).

Companion to v2/sparsity/prune_probe.py. Runs the SAME one-shot Wanda 2:4 weight
pruning on LLaDA-8B-Instruct (a LESS-distilled block/masked-diffusion LLM) to test the
"compressibility tracks distillation" thesis:

  v2 (SOTA-distilled) 2:4 -> GSM8K 0.71 -> 0.24 (CRATERS, no weight slack).
  If LLaDA (looser) SURVIVES 2:4, the two-model contrast shows distillation — not the
  diffusion architecture — is what removes the slack. That's the capstone figure.

Same Wanda metric (score_ij = |W_ij| * ||X_j||_2, keep 2 of every 4 input channels per
output row). Calibration activations captured on the denoising forwards of LLaDA's own
generation loop (LLaDA has no prefill/commit split — every model(x) call is a denoising
forward, so no forward-type gating is needed, unlike v2).

Prunes the 7 linears inside every transformer block (q/k/v_proj, attn_out, ff_proj,
up_proj, ff_out). The model-level LM head / embeddings are excluded by the `.blocks.`
name filter.

Phase 1 = quality only (zeroed bf16 weights, no sparse kernel).

Run FROM THIS DIRECTORY (v1/llada) so `generate` / `model.modeling_llada` import:
  cd v1/llada && CUDA_VISIBLE_DEVICES=<free_gpu> python prune_probe_llada.py \
      --calib_n 16 --eval_n 100
"""
import sys
import os
import re
import argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
from transformers import AutoTokenizer, AutoConfig
from model.modeling_llada import LLaDAModelLM
from generate import generate

MASK_ID = 126336


# ---- GSM8K eval (copied from v2 nfe_probe_stage1 for apples-to-apples parsing) ----
def _gsm8k_gold(answer):
    m = re.search(r"####\s*(-?[\d,]+)", answer)
    return m.group(1).replace(",", "") if m else None


def _gsm8k_pred(text):
    m = re.search(r"####\s*(-?[\d,]+)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?[\d,]+", text)
    return nums[-1].replace(",", "") if nums else None


def load_gsm8k(n, seed):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test").shuffle(seed=seed)
    ds = ds.select(range(min(n, len(ds))))
    return [(ex["question"], _gsm8k_gold(ex["answer"])) for ex in ds]


# ---- prunable linears: every nn.Linear INSIDE a transformer block ----
def get_prunable_linears(model):
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and ".blocks." in name:
            out[name] = mod
    return out


# ---- Wanda calibration + N:M masking (mirrors v2/sparsity/prune_probe.py) ----
class Calibrator:
    """Accumulate per-input-channel activation energy (sum x^2, fp32) on each prunable
    linear across LLaDA's denoising forwards. No forward-type gating needed."""

    def __init__(self, model):
        self.model = model
        self.linears = get_prunable_linears(model)
        self.sumsq = {}
        self.ntok = {}
        self._handles = []

    def _make_hook(self, name):
        def hook(module, args):
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
        return self

    def __exit__(self, *x):
        for h in self._handles:
            h.remove()
        self._handles = []

    def act_norm(self, name):
        return self.sumsq[name].sqrt()


def nm_mask(score, n=2, m=4):
    out, inp = score.shape
    assert inp % m == 0, f"in_features {inp} not divisible by {m}"
    g = score.reshape(out, inp // m, m)
    keep_idx = g.argsort(dim=-1, descending=True)[..., :n]
    mask = torch.zeros_like(g, dtype=torch.bool)
    mask.scatter_(-1, keep_idx, True)
    return mask.reshape(out, inp)


@torch.no_grad()
def prune_wanda(calib, n=2, m=4, verbose=True):
    stats = {}
    for name, lin in calib.linears.items():
        if name not in calib.sumsq:
            if verbose:
                print(f"  [warn] {name}: no calib activations — skipped", flush=True)
            continue
        W = lin.weight.data
        if W.shape[1] % m != 0:
            if verbose:
                print(f"  [skip] {name}: in_features {W.shape[1]} not divisible by {m}", flush=True)
            continue
        act = calib.act_norm(name)
        score = W.abs().float() * act.float()[None, :]
        mask = nm_mask(score, n=n, m=m).to(W.device)
        W.mul_(mask.to(W.dtype))
        stats[name] = 1.0 - mask.float().mean().item()
    if verbose:
        avg = sum(stats.values()) / len(stats) if stats else 0.0
        print(f"  pruned {len(stats)} linears | mean sparsity {avg:.3f} "
              f"(target {1 - n / m:.3f})", flush=True)
    return stats


# ---- generation / eval ----
@torch.no_grad()
def gen_one(model, tokenizer, question, device, steps, gen_len, block_len):
    m = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    ids = torch.tensor(tokenizer(text)["input_ids"], device=device).unsqueeze(0)
    out, _ = generate(model, ids, steps=steps, gen_length=gen_len, block_length=block_len,
                      temperature=0.0, remasking="low_confidence", mask_id=MASK_ID)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def eval_gsm8k(model, tokenizer, data, device, steps, gen_len, block_len):
    correct = 0
    for q, gold in data:
        gen = gen_one(model, tokenizer, q, device, steps, gen_len, block_len)
        pred = _gsm8k_pred(gen)
        correct += int(pred is not None and pred == gold)
    return correct / len(data) if data else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--sparsity", default="2:4")
    p.add_argument("--calib_n", type=int, default=16)
    p.add_argument("--eval_n", type=int, default=100)
    p.add_argument("--steps", type=int, default=256, help="denoising steps (== gen_len for 1 tok/step)")
    p.add_argument("--gen_len", type=int, default=256)
    p.add_argument("--block_len", type=int, default=32, help="semi-AR block (match v2 block_size=32)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_dense", action="store_true")
    args = p.parse_args()

    n, m = (int(x) for x in args.sparsity.split(":"))

    print("Loading LLaDA...", flush=True)
    # Compat shims: the shared v2 venv has a NEWER transformers than v1's custom
    # LLaDAModelLM was written against. Two finalize-step API changes need bridging;
    # both are safe because the checkpoint loads complete (no missing keys). This avoids
    # downgrading transformers in the shared venv (which would risk breaking v2).
    #  (1) new code reads `all_tied_weights_keys`; old model only has `_tied_weights_keys`.
    if not hasattr(LLaDAModelLM, "all_tied_weights_keys"):
        LLaDAModelLM.all_tied_weights_keys = {}
    #  (2) new code calls tie_weights(missing_keys=..., recompute_mapping=...); old
    #      override takes no kwargs. Wrap to swallow the new kwargs, keep real tying.
    if not getattr(LLaDAModelLM, "_tie_weights_kwargs_shim", False):
        _orig_tie = LLaDAModelLM.tie_weights
        def _tie_shim(self, *a, **k):
            return _orig_tie(self)
        LLaDAModelLM.tie_weights = _tie_shim
        LLaDAModelLM._tie_weights_kwargs_shim = True

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    config.flash_attention = True   # matches eval_llada.py's tested path; drop if flash-attn not installed
    config.use_cache = False        # custom forward reads config.use_cache; not defaulted in this config
    model = LLaDAModelLM.from_pretrained(args.model, trust_remote_code=True,
                                         torch_dtype=torch.bfloat16, config=config)
    model = model.to(args.device).eval()
    model.config.use_cache = False  # ensure it survives on the loaded model's config
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    calib_data = load_gsm8k(args.calib_n, args.seed + 1000)
    eval_data = load_gsm8k(args.eval_n, args.seed)

    dense_acc = None
    if not args.skip_dense:
        print(f"\nEval DENSE on {args.eval_n} GSM8K...", flush=True)
        dense_acc = eval_gsm8k(model, tokenizer, eval_data, args.device,
                               args.steps, args.gen_len, args.block_len)
        print(f"  dense acc = {dense_acc:.3f}", flush=True)

    print(f"\nCalibrating on {args.calib_n} prompts (all denoising forwards)...", flush=True)
    calib = Calibrator(model)
    with calib:
        for q, _ in calib_data:
            gen_one(model, tokenizer, q, args.device, args.steps, args.gen_len, args.block_len)
    tok_range = (min(calib.ntok.values()), max(calib.ntok.values())) if calib.ntok else (0, 0)
    print(f"  captured {len(calib.sumsq)} linears | calib tokens/linear "
          f"{tok_range[0]}..{tok_range[1]}", flush=True)

    print(f"\nPruning Wanda {args.sparsity}...", flush=True)
    prune_wanda(calib, n=n, m=m)
    del calib
    torch.cuda.empty_cache()

    print(f"\nEval PRUNED ({args.sparsity}) on {args.eval_n} GSM8K...", flush=True)
    pruned_acc = eval_gsm8k(model, tokenizer, eval_data, args.device,
                            args.steps, args.gen_len, args.block_len)
    print(f"  pruned acc = {pruned_acc:.3f}", flush=True)

    print(f"\n{'='*54}\nSUMMARY — LLaDA-8B (GSM8K acc, n={args.eval_n})\n{'='*54}")
    if dense_acc is not None:
        print(f"  dense              {dense_acc:.3f}")
    print(f"  wanda {args.sparsity}         {pruned_acc:.3f}"
          + (f"   (drop {dense_acc - pruned_acc:+.3f})" if dense_acc is not None else ""))
    print("\n  Contrast vs v2 (0.71 -> 0.24, drop 0.47): if LLaDA's drop is much smaller,")
    print("  distillation (not the diffusion architecture) removes the weight slack.")


if __name__ == "__main__":
    main()
