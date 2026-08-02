"""
STAGE 2b — the NFE fine-tune probe. Does high-mask-ratio LoRA fine-tuning move the
low-threshold accuracy cliff (i.e. teach the model to decode in fewer steps)?

Mechanism:
  - Reuse the model's NATIVE masked-diffusion objective (modeling.py:657) but bias
    the per-block mask ratio to Uniform(mask_low, 1.0) instead of Uniform(0,1). High
    mask ratio = many tokens masked per block = predicting many tokens at once =
    exactly the low-NFE decoding regime the base model isn't specialized for.
  - Manual LoRA on the proj layers (no peft; keeps model.model directly callable and
    memory tiny — base frozen). LoRA B init = 0, so it's identity at start.
  - Eval the SAME quality-vs-NFE curve (Stage 1) BEFORE and AFTER training, on the
    cliff-region thresholds. If the after-curve holds accuracy at lower thresholds
    than the before-curve, the cliff moved -> few-step training works -> Stage 2 is
    the direction. If it doesn't move, the cliff is (at least LoRA-)hard.

Confound to keep in mind: training uses a plain "Question:/Answer:" format (no
few-shot) for memory; eval uses the few-shot harness. A format mismatch can only
UNDERSTATE the effect, so a POSITIVE result is trustworthy; a flat result is
ambiguous (could be the mismatch). Good enough for a go/no-go probe.

Run (after Stage 2a passes):
  CUDA_VISIBLE_DEVICES=0 uv run v2/train_scripts/nfe_probe_stage2b.py \
      --train_examples 2000 --steps 300 --eval_n 100
"""
import sys
import os
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)                                   # sibling: nfe_probe_stage1
sys.path.insert(0, os.path.join(HERE, ".."))               # v2/
sys.path.insert(0, os.path.join(HERE, "../experiments"))   # v2/experiments (helper)

import types
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer
from helper import load_model
from generation_functions import Fast_dLLM_QwenForCausalLM as GenClass
from nfe_probe_stage1 import EVALS, build_batches, run_threshold

MASK_ID = 151665


# ---------------------------------------------------------------- LoRA ------
class LoRALinear(nn.Module):
    """base(x) + scale * B(A(x)); base frozen, A/B trainable. Identity at init (B=0)."""
    def __init__(self, base, r=16, alpha=32, dropout=0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        dev, dt = base.weight.device, base.weight.dtype
        self.A = nn.Linear(base.in_features, r, bias=False).to(dev, dt)
        self.B = nn.Linear(r, base.out_features, bias=False).to(dev, dt)
        nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.B.weight)
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.scale * self.B(self.A(self.drop(x)))


def inject_lora(model, r, alpha, targets=("q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj")):
    for p in model.parameters():
        p.requires_grad_(False)
    n = 0
    for module in model.modules():
        for cname, child in list(module.named_children()):
            if cname in targets and isinstance(child, nn.Linear):
                setattr(module, cname, LoRALinear(child, r, alpha))
                n += 1
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  injected LoRA into {n} linears, {trainable/1e6:.1f}M trainable params", flush=True)


# --------------------------------------------------- diffusion training batch
def make_diffusion_batch(input_ids, labels, mask_id, bd, mask_low):
    """Replicates modeling.py:657-675 but biases the block mask ratio high.
    Returns (doubled_input [B,2L], labels [B,L]) for the inner model's training forward."""
    B, L = input_ids.shape
    noisy = input_ids.clone()
    blk = input_ids.reshape(B * L // bd, bd)
    bb, ll = blk.shape
    t = torch.rand(bb, device=input_ids.device) * (1.0 - mask_low) + mask_low  # Uniform(mask_low,1)
    eps = 1e-3
    p_mask = ((1 - eps) * t + eps)[:, None].repeat(1, ll)
    mask_indices = torch.rand(bb, ll, device=input_ids.device) < p_mask
    x_t = torch.where(mask_indices, mask_id, blk).reshape(B, L)
    noisy[labels != -100] = x_t[labels != -100]
    lab = labels.clone()
    lab[noisy != mask_id] = -100                      # loss only on masked target tokens
    doubled = torch.cat([noisy, input_ids], dim=1)    # [noisy | clean] along seq
    return doubled, lab


def build_train_examples(tokenizer, n, seq_len, seed):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="train").shuffle(seed=seed).select(range(n))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    out = []
    for e in ds:
        p_ids = tokenizer(f"Question: {e['question']}\nAnswer:", add_special_tokens=False).input_ids
        t_ids = tokenizer(" " + e["answer"], add_special_tokens=False).input_ids
        ids = (p_ids + t_ids)[:seq_len]
        labels = ([-100] * len(p_ids) + t_ids)[:seq_len]
        if len(t_ids) == 0 or all(l == -100 for l in labels):
            continue
        k = seq_len - len(ids)
        ids += [pad] * k
        labels += [-100] * k
        out.append((ids, labels))
    return out


def evaluate(model, tokenizer, dataset, thresholds, eval_n, max_new_tokens, device, seed, debug=0):
    ev = EVALS[dataset]
    data = ev["load"](eval_n, seed)
    prompts = [ev["prompt"](q) for q, _ in data]
    golds = [g for _, g in data]
    batches = build_batches(prompts, golds, tokenizer, device, 1)
    sampler = types.MethodType(GenClass.batch_sample, model)
    model.eval()
    rows = []
    with torch.no_grad():
        for th in sorted(thresholds, reverse=True):
            acc, nfe = run_threshold(model, tokenizer, sampler, batches, th,
                                     max_new_tokens, ev["parse_pred"], debug=debug)
            rows.append((th, acc, nfe))
            debug = 0  # only dump for the first (highest) threshold
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dataset", default="gsm8k", choices=list(EVALS))
    p.add_argument("--train_examples", type=int, default=2000)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=256, help="multiple of bd_size (32)")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--eval_every", type=int, default=0,
                   help="run a quick generation check every N steps to catch WHERE quality breaks")
    p.add_argument("--mask_low", type=float, default=0.5,
                   help="block mask ratio ~ Uniform(mask_low, 1.0); higher = more few-step-specialized")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--eval_n", type=int, default=100)
    p.add_argument("--eval_thresholds", type=float, nargs="+", default=[0.7, 0.6, 0.5, 0.4, 0.3])
    p.add_argument("--eval_max_new_tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default=None, help="path to save LoRA adapter state_dict")
    args = p.parse_args()

    print("Loading model + injecting LoRA...", flush=True)
    model = load_model(args.model, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    bd = model.model.bd_size
    assert args.seq_len % bd == 0, f"seq_len must be a multiple of bd_size={bd}"
    inject_lora(model, args.lora_r, args.lora_alpha)
    V = model.config.vocab_size

    # ---- BEFORE (LoRA is identity: B=0) ----
    print("\n=== BEFORE fine-tune (cliff curve) ===", flush=True)
    before = evaluate(model, tokenizer, args.dataset, args.eval_thresholds,
                      args.eval_n, args.eval_max_new_tokens, args.device, args.seed)
    for th, acc, nfe in before:
        print(f"  th={th:.2f}  acc={acc:.1%}  NFE/ex={nfe:.1f}", flush=True)

    # ---- TRAIN ----
    print("\nBuilding train data...", flush=True)
    ex = build_train_examples(tokenizer, args.train_examples, args.seq_len, args.seed)
    print(f"  {len(ex)} training examples", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()
    g = torch.Generator().manual_seed(args.seed)
    print(f"\n=== TRAIN  steps={args.steps}  bs={args.batch_size}  mask_low={args.mask_low} ===", flush=True)
    for step in range(args.steps):
        idx = torch.randint(0, len(ex), (args.batch_size,), generator=g)
        input_ids = torch.tensor([ex[i][0] for i in idx], device=args.device)
        labels = torch.tensor([ex[i][1] for i in idx], device=args.device)
        doubled, lab = make_diffusion_batch(input_ids, labels, MASK_ID, bd, args.mask_low)
        out = model.model(input_ids=doubled, labels=lab, block_size=bd, use_cache=False)
        logits = model.lm_head(out.last_hidden_state[:, :args.seq_len, :])
        loss = F.cross_entropy(logits.reshape(-1, V).float(), lab.reshape(-1), ignore_index=-100)
        loss.backward()
        opt.step()
        opt.zero_grad()
        if step % 20 == 0 or step == args.steps - 1:
            print(f"  step {step:4d}  loss {loss.item():.4f}  peak {torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
        if args.eval_every and (step + 1) % args.eval_every == 0:
            traj = evaluate(model, tokenizer, args.dataset, [0.7, 0.4],
                            min(40, args.eval_n), args.eval_max_new_tokens, args.device, args.seed)
            print("      [gen check @step %d] " % (step + 1)
                  + "  ".join(f"th{th:.2f}={acc:.0%}" for th, acc, _ in traj), flush=True)
            model.train()

    if args.save:
        lora_sd = {k: v for k, v in model.state_dict().items() if ".A." in k or ".B." in k}
        torch.save(lora_sd, args.save)
        print(f"saved LoRA adapter -> {args.save}", flush=True)

    # ---- AFTER ----
    print("\n=== AFTER fine-tune (cliff curve) — dumping a few generations ===", flush=True)
    after = evaluate(model, tokenizer, args.dataset, args.eval_thresholds,
                     args.eval_n, args.eval_max_new_tokens, args.device, args.seed, debug=5)

    print("\n" + "=" * 52)
    print(f"CLIFF MOVEMENT  (dataset={args.dataset}, eval_n={args.eval_n})")
    print("=" * 52)
    print(f"  {'threshold':>9}  {'before':>8}  {'after':>8}  {'delta':>7}  {'NFE/ex':>7}")
    for (th, ba, _), (_, aa, an) in zip(before, after):
        print(f"  {th:>9.2f}  {ba:>8.1%}  {aa:>8.1%}  {aa-ba:>+7.1%}  {an:>7.1f}")
    print("\nRead: if AFTER holds accuracy at low thresholds where BEFORE collapsed,")
    print("the cliff moved -> few-step training works -> Stage 2 is the direction.")


if __name__ == "__main__":
    main()
