"""
A1 — GENERATION-AWARE training for the depth router (on-policy self-distillation).

This is the fix for the teacher-forcing gap that made the naive fine-tune collapse
(depth_router_derisk.py trained on masked SNAPSHOTS of the doubled [noisy|clean] batch on
the gen_mask TRAINING path -> the model copied the clean half -> 0% generation).

A1 changes THREE things, which together = "generation-aware":
  1. EVAL/GENERATION forward path (model.eval() -> eval_mask, use_cache), NOT the training
     doubled-batch path. So the training distribution = the generation distribution.
  2. ON-POLICY rollout: we decode the block with the STUDENT itself (routers on, small steps)
     and train on the states IT visits. No ground-truth clean answer is ever in the input ->
     nothing to copy.
  3. SELF-DISTILLATION target: match the FROZEN ORIGINAL model (LoRA off, full depth) on those
     states. The student can at best reproduce the original's generation -> can't collapse below it.

Loss (per block, over the student's rolled-out states x^(k), active = still-masked positions):
    L = mean_k KL( p_original(.|x^(k)) || p_student(.|x^(k)) )_active   +   lambda * (keep - target)^2
Prefill + commit run DENSE (recoverable-KV / dense-commit); only small steps are depth-reduced.

PROTOTYPE simplifications (flagged for the real version):
  - decodes 1 block/example (so all small steps share one prompt-prefill prefix — no per-step
    cache snapshotting); bs=1; prompts left-padded to a bd multiple.
  - router conditioned on t=neutral (0). The convergence signal (per-token confidence/step) is
    where Hook A would plug in next — TODO, needs feeding real confidence into the router input.

Validate success by (a) KL decreasing + keep -> target here, then (b) the REAL test:
    depth_router_gen_eval.py --load <ckpt>   -> routed_acc should now be ~= orig (not 0%).

Run:
  CUDA_VISIBLE_DEVICES=0 uv run v2/train_scripts/depth_router_train_a1.py \
      --steps 400 --target_keep 0.7 --save /tmp/depthr_a1.pt
"""
import sys
import os
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "../experiments"))

import argparse
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer
from helper import load_model
from nfe_probe_stage2b import inject_lora
from nfe_probe_stage1 import EVALS
from depth_router_derisk import wrap_layers, set_router_state
from depth_router_gen_eval import set_lora_enabled

MASK_ID = 151665


def build_prompts(tokenizer, n, seed, bd, device, dataset="gsm8k"):
    """few-shot prompts for the chosen task, left-padded to a multiple of bd so the generated
    block is bd-aligned (avoids the ragged-remainder path). Returns list of [1, L] id tensors."""
    data = EVALS[dataset]["load"](n, seed)
    pf = EVALS[dataset]["prompt"]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 151645
    out = []
    for q, _ in data:
        ids = tokenizer(pf(q), return_tensors="pt").input_ids[0]
        padn = (bd - ids.shape[0] % bd) % bd
        if padn:
            ids = torch.cat([torch.full((padn,), pad, dtype=torch.long), ids])
        out.append(ids.unsqueeze(0).to(device))
    return out


@torch.no_grad()
def prefill(model, wrappers, prompt_ids, bd):
    """Dense prefill with the ORIGINAL model (LoRA off) -> prefix KV shared by all block-1 small steps."""
    set_lora_enabled(model, False)
    set_router_state(wrappers, bypass=True)
    out = model.forward(input_ids=prompt_ids, use_cache=True, update_past_key_values=True,
                        block_size=bd, logits_to_keep=1)
    return out.past_key_values


@torch.no_grad()
def rollout_block(model, wrappers, prefix_kv, bd, sbs, threshold, top_p, temperature,
                  device, max_states):
    """On-policy: decode one block with the STUDENT (LoRA on, routers hard) against prefix_kv.
    Records the block state x^(k) fed to each small-step forward. Mirrors batch_sample's unmask."""
    set_lora_enabled(model, True)
    set_router_state(wrappers, bypass=False, hard=True)
    block = torch.full((1, bd), MASK_ID, device=device, dtype=torch.long)
    states = []
    for sb in range(bd // sbs):
        start, end = sb * sbs, sb * sbs + sbs
        guard = 0
        while (block[:, start:end] == MASK_ID).any() and guard < sbs + 2:
            guard += 1
            logits = model.forward(input_ids=block, use_cache=True, past_key_values=prefix_kv,
                                   update_past_key_values=False).logits
            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)   # predict token[i]
            states.append(block.clone())
            sub = logits[:, start:end, :]
            x1, p1 = model.sample_with_top_p(sub, top_p=top_p, temperature=temperature)
            x1p = torch.gather(p1, -1, x1.unsqueeze(-1)).squeeze(-1)
            mask_sub = (block[:, start:end] == MASK_ID)
            x1p = torch.where(mask_sub, x1p, torch.full_like(x1p, -float("inf")))
            unmask = x1p > threshold
            unmask[torch.arange(1, device=device), x1p.argmax(-1)] = True   # >=1 unmask/step
            unmask = unmask & mask_sub
            block[:, start:end][unmask] = x1[unmask]
    if len(states) > max_states:                      # bound the grad-pass memory
        keep = sorted(torch.randperm(len(states))[:max_states].tolist())
        states = [states[i] for i in keep]
    return states, block


def compute_loss(model, wrappers, states, prefix_kv, target_keep, lam, device, sched_slope=0.0):
    """Teacher = ORIGINAL (LoRA off, dense), no grad. Student = LoRA on, routers STE, grad.
    KL(teacher||student) on masked positions + STEP-CONDITIONED budget.
    sched_slope>0: per-state keep target rises with decode-progress -> skip MORE early (far from commit,
    most recoverable), LESS late (near commit). Matched average (centered on the batch's mean progress),
    so it's comparable to the flat (slope=0) budget."""
    # ---- teacher pass (no grad) ----
    set_lora_enabled(model, False)
    set_router_state(wrappers, bypass=True)
    t_logits = []
    with torch.no_grad():
        for s in states:
            lg = model.forward(input_ids=s, use_cache=True, past_key_values=prefix_kv,
                               update_past_key_values=False).logits
            t_logits.append(torch.cat([lg[:, :1, :], lg[:, :-1, :]], dim=1))
    # ---- student pass (grad) ----
    set_lora_enabled(model, True)
    set_router_state(wrappers, bypass=False, gate_mode="ste")  # deterministic gate + STE grad
    kl_terms, keep_terms, prog_terms = [], [], []
    for s, t in zip(states, t_logits):
        lg = model.forward(input_ids=s, use_cache=True, past_key_values=prefix_kv,
                           update_past_key_values=False).logits
        lg = torch.cat([lg[:, :1, :], lg[:, :-1, :]], dim=1)
        active = (s == MASK_ID)                              # [1, bd] positions being decided
        if active.any():
            sp = F.log_softmax(lg[active].float(), dim=-1)
            tp = F.softmax(t[active].float(), dim=-1)
            kl_terms.append(F.kl_div(sp, tp, reduction="batchmean"))
        gates = [w.last_gate_diff.mean() for w in wrappers if w.last_gate_diff is not None]
        if gates:
            keep_terms.append(torch.stack(gates).mean())
            prog_terms.append((s != MASK_ID).float().mean().detach())   # fraction of block decoded
    if not kl_terms or not keep_terms:
        return None, None, None, None, None
    kl = torch.stack(kl_terms).mean()
    keep_stack = torch.stack(keep_terms)                     # [n_states], has grad
    prog_stack = torch.stack(prog_terms)                     # [n_states], no grad
    target_k = (target_keep + sched_slope * (prog_stack - prog_stack.mean())).clamp(0.05, 1.0)
    budget = ((keep_stack - target_k) ** 2).mean()          # per-state -> allows within-state routing
    loss = kl + lam * budget
    # diagnostic: did the schedule bind? keep for early (less-decoded) vs late (more-decoded) states
    med = prog_stack.median()
    early = keep_stack[prog_stack <= med].mean().detach()    # early/volatile states (want lower keep)
    late = keep_stack[prog_stack > med].mean().detach()      # late/near-commit states (want higher keep)
    return loss, kl.detach(), keep_stack.mean().detach(), early, late


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--dataset", default="gsm8k", choices=list(EVALS),
                   help="task to train on (gsm8k/math). Eval with gen_eval --dataset <same or other>.")
    p.add_argument("--n_prompts", type=int, default=1000)
    p.add_argument("--small_block_size", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--target_keep", type=float, default=0.7, help="FINAL keep target (annealed to)")
    p.add_argument("--budget_warmup_frac", type=float, default=0.6,
                   help="curriculum: anneal target_keep 1.0 -> target over this frac of steps (start dense)")
    p.add_argument("--lambda_budget", type=float, default=15.0)
    p.add_argument("--sched_slope", type=float, default=0.0,
                   help="step-conditioned depth: >0 skips MORE early (far from commit), LESS late "
                        "(near commit); matched avg budget. 0 = flat (default).")
    p.add_argument("--init_keep_bias", type=float, default=1.0,
                   help="router bias init: lower = logits less saturated = router can move off keep=1")
    p.add_argument("--max_states", type=int, default=12, help="cap states/block for grad-pass memory")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sample_every", type=int, default=50, help="print a decoded block to watch coherence")
    p.add_argument("--save", default=None)
    args = p.parse_args()

    print("Loading model + LoRA + routers...", flush=True)
    model = load_model(args.model, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    bd = model.model.bd_size
    inject_lora(model, args.lora_r, args.lora_alpha)
    wrappers = wrap_layers(model, init_keep_bias=args.init_keep_bias)
    model.eval()   # CRITICAL: eval_mask/generation path (not the gen_mask training path)

    trainable = [q for q in model.parameters() if q.requires_grad]
    print(f"  {sum(x.numel() for x in trainable)/1e6:.1f}M trainable (LoRA + routers)", flush=True)
    prompts = build_prompts(tokenizer, args.n_prompts, args.seed, bd, args.device, dataset=args.dataset)
    print(f"  {len(prompts)} prompts", flush=True)
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    gen = torch.Generator().manual_seed(args.seed)

    print(f"\n=== A1 TRAIN  steps={args.steps}  target_keep={args.target_keep}  "
          f"sched_slope={args.sched_slope} ===", flush=True)
    run_kl, run_keep, run_e, run_l, seen = 0.0, 0.0, 0.0, 0.0, 0
    for step in range(args.steps):
        prompt = prompts[torch.randint(0, len(prompts), (1,), generator=gen).item()]
        prefix_kv = prefill(model, wrappers, prompt, bd)
        states, decoded = rollout_block(model, wrappers, prefix_kv, bd, args.small_block_size,
                                        args.threshold, 0.95, 0.0, args.device, args.max_states)
        if not states:
            continue
        # curriculum: start dense (target 1.0), anneal to args.target_keep over warmup -> hold
        warm = max(1, int(args.budget_warmup_frac * args.steps))
        cur_target = 1.0 - (1.0 - args.target_keep) * min(1.0, step / warm)
        loss, kl, keep, early, late = compute_loss(model, wrappers, states, prefix_kv,
                                                   cur_target, args.lambda_budget, args.device,
                                                   sched_slope=args.sched_slope)
        if loss is None:
            continue
        loss.backward()
        opt.step(); opt.zero_grad()
        run_kl += kl.item(); run_keep += keep.item(); seen += 1
        run_e += early.item(); run_l += late.item()

        if step % 20 == 0 or step == args.steps - 1:
            print(f"  step {step:4d}  loss {loss.item():.4f}  KL {run_kl/seen:.4f}  "
                  f"keep {run_keep/seen:.3f} (tgt {cur_target:.2f})  "
                  f"[early {run_e/seen:.2f} / late {run_l/seen:.2f}]  n_states {len(states)}  "
                  f"peak {torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
            run_kl, run_keep, run_e, run_l, seen = 0.0, 0.0, 0.0, 0.0, 0
        if args.sample_every and step % args.sample_every == 0:
            txt = tokenizer.decode(decoded[0], skip_special_tokens=True)
            print(f"      [student block @{step}] {txt!r}", flush=True)

    if args.save:
        sd = {k: v for k, v in model.state_dict().items()
              if ".A." in k or ".B." in k or ".router." in k}
        torch.save(sd, args.save)
        print(f"\nsaved LoRA+router -> {args.save}", flush=True)
    print("\nNEXT: depth_router_gen_eval.py --load <ckpt>  -> routed_acc ~= orig means A1 fixed the")
    print("collapse (naive FT gave 0%). Then wire per-token confidence into the router (Hook A) +")
    print("build the compute-skip kernel for throughput.")


if __name__ == "__main__":
    main()
