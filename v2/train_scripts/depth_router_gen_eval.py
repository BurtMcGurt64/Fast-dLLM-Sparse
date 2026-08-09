"""
DEPTH-ROUTER GENERATION EVAL — does the (teacher-forced-trained) depth router transfer
to REAL block-diffusion decoding, and does recoverable-KV/dense-commit preserve accuracy?

This is the paper-grade gate the teacher-forced probe (depth_router_derisk.py) could not
answer. It installs the trained LoRA+routers into the ACTUAL generation loop (batch_sample)
and measures gsm8k accuracy dense-vs-routed.

RECOVERABLE-KV / DENSE-COMMIT (Hook B, the AR-impossible bit) is implemented here:
  - prefill  -> routers BYPASSED (dense)
  - commit   -> routers BYPASSED (dense) => the KV written to the prefix cache is EXACT
  - small step -> routers ACTIVE (hard gates) => cheap approximate depth during denoising,
    which the next step / the dense commit can correct. In AR a skipped layer's KV is a
    permanent hole; here the block is revisited, so small-step skips are recoverable.

Detection of step type mirrors topk.py (update_past_key_values / use_block_cache / pre_kv_len).

WHAT THIS MEASURES: accuracy transfer + the recoverable-KV quality question. It does NOT
measure speedup — the gate simulates skipping by zeroing a layer's contribution (quality-
faithful, no compute saved). Real throughput needs a compute-skipping kernel (follow-on);
here we report the achieved small-step keep-rate as the compute-reduction the kernel would cash in.

Prereq: train + SAVE a checkpoint first:
  ... depth_router_derisk.py --target_keep 0.7 --steps 400 --save /tmp/depthr.pt
Then:
  CUDA_VISIBLE_DEVICES=0 uv run v2/train_scripts/depth_router_gen_eval.py \
      --load /tmp/depthr.pt --n_examples 200 --thresholds 0.9
"""
import sys
import os
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "../experiments"))

import types
import argparse
import torch

from transformers import AutoTokenizer
from helper import load_model
from generation_functions import Fast_dLLM_QwenForCausalLM as GenClass
from nfe_probe_stage2b import inject_lora, LoRALinear
from nfe_probe_stage1 import EVALS, build_batches, run_threshold
from depth_router_derisk import wrap_layers, set_router_state

MASK_ID = 151665


def set_lora_enabled(model, enabled):
    """Toggle LoRA on/off by zeroing its scale -> recovers the ORIGINAL base model when off.
    Lets us get the true (no-LoRA, no-routing) baseline without reloading the model."""
    for m in model.modules():
        if isinstance(m, LoRALinear):
            if not hasattr(m, "_scale0"):
                m._scale0 = m.scale
            m.scale = m._scale0 if enabled else 0.0


class RouterGenController:
    """Wraps model.forward during generation. mode='dense' bypasses routers everywhere
    (baseline); mode='routed' gates SMALL steps only and keeps prefill+commit dense
    (recoverable-KV). Tracks the achieved small-step keep-rate."""
    def __init__(self, model, wrappers, mode="routed", gate_commit=False, force_commit_keep=None):
        self.model = model
        self.wrappers = wrappers
        self.mode = mode
        self.gate_commit = gate_commit   # ablation: if True, ALSO gate the commit step (breaks recoverable-KV)
        self.force_commit_keep = force_commit_keep  # if set, FORCE random skip on commit to this keep-rate
                                                    # (router only skips ~8% on commit; this tests heavier commit skip)
        self._orig = None
        self._keep_sum = 0.0
        self._keep_n = 0
        self._keep_sum_c = 0.0   # commit-step keep-rate (to check the ablation actually gated the commit)
        self._keep_n_c = 0
        # small-step keep split by decode-progress: does the router naturally keep MORE near the commit?
        self._keep_early_sum = 0.0; self._keep_early_n = 0   # early states (block mostly masked)
        self._keep_late_sum = 0.0;  self._keep_late_n = 0    # late states (block mostly decoded)

    def reset(self):
        self._keep_sum, self._keep_n = 0.0, 0

    @property
    def avg_keep(self):
        return self._keep_sum / self._keep_n if self._keep_n else float("nan")

    @property
    def avg_keep_commit(self):
        return self._keep_sum_c / self._keep_n_c if self._keep_n_c else float("nan")

    @property
    def avg_keep_early(self):
        return self._keep_early_sum / self._keep_early_n if self._keep_early_n else float("nan")

    @property
    def avg_keep_late(self):
        return self._keep_late_sum / self._keep_late_n if self._keep_late_n else float("nan")

    def install(self):
        orig = self.model.forward
        self._orig = orig
        ctrl = self

        def wrapped(*a, **k):
            pre_kv = 0
            pkv = k.get("past_key_values", None)
            if pkv is not None:
                try:
                    pre_kv = pkv.layers[0].keys.shape[-2]
                except Exception:
                    try:
                        pre_kv = pkv.get_seq_length()
                    except Exception:
                        pre_kv = 0
            upkv = k.get("update_past_key_values", False)
            is_small = (pre_kv > 0) and (not upkv)      # denoising forward (not prefill/commit)
            is_commit = (pre_kv > 0) and upkv           # writes block KV to prefix cache

            for w in ctrl.wrappers:      # clear any forced random-skip from a previous forward
                w._rand_p = None
            # normally route only small steps (prefill+commit stay dense = recoverable-KV).
            # ablation: gate_commit=True also routes the commit -> committed KV becomes approximate.
            force_c = (is_commit and ctrl.gate_commit and ctrl.force_commit_keep is not None)
            route_now = (ctrl.mode == "routed") and (is_small or (ctrl.gate_commit and is_commit))
            if force_c:                  # FORCE heavy random skip on the commit (decisive recoverable-KV test)
                set_router_state(ctrl.wrappers, bypass=False)
                for w in ctrl.wrappers:
                    w._rand_p = ctrl.force_commit_keep
            elif route_now:
                set_router_state(ctrl.wrappers, bypass=False, hard=True)
            else:
                set_router_state(ctrl.wrappers, bypass=True)   # prefill / commit / dense mode

            out = orig(*a, **k)

            if route_now and (is_small or is_commit):
                ks = [w.last_gate.mean().item() for w in ctrl.wrappers if w.last_gate is not None]
                if ks:
                    kf = sum(ks) / len(ks)
                    if is_small:              # small-step keep (comparable across modes)
                        ctrl._keep_sum += kf; ctrl._keep_n += 1
                        # split by decode-progress of THIS forward's input (fraction not-masked)
                        ids = k.get("input_ids", a[0] if a else None)
                        if ids is not None:
                            prog = (ids != MASK_ID).float().mean().item()   # 0 = all masked (early), 1 = decoded (late)
                            if prog < 0.5:
                                ctrl._keep_early_sum += kf; ctrl._keep_early_n += 1
                            else:
                                ctrl._keep_late_sum += kf; ctrl._keep_late_n += 1
                    else:                     # commit keep (did the ablation actually gate the commit?)
                        ctrl._keep_sum_c += kf; ctrl._keep_n_c += 1
            return out

        self.model.forward = wrapped
        return self

    def uninstall(self):
        if self._orig is not None:
            self.model.forward = self._orig
            self._orig = None

    def __enter__(self):
        return self.install()

    def __exit__(self, *_):
        self.uninstall()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--load", default=None, help="LoRA+router checkpoint from depth_router_derisk --save")
    p.add_argument("--dataset", default="gsm8k", choices=list(EVALS))
    p.add_argument("--n_examples", type=int, default=200)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--small_block_size", type=int, default=8)
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.9])
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--debug", type=int, default=0,
                   help="dump this many (gold,pred,generation) per condition to see the FAILURE MODE")
    p.add_argument("--ablate_gate_commit", action="store_true",
                   help="also run routed WITH the commit step gated -> tests if dense-commit is load-bearing")
    p.add_argument("--force_commit_keep", type=float, default=None,
                   help="with --ablate_gate_commit: FORCE random skip on commit to this keep-rate "
                        "(e.g. 0.79 to match small steps). Decisive recoverable-KV test.")
    args = p.parse_args()

    print("Loading model + LoRA + routers...", flush=True)
    model = load_model(args.model, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # EXACT same construction order as depth_router_derisk so checkpoint keys line up
    inject_lora(model, args.lora_r, args.lora_alpha)
    wrappers = wrap_layers(model)
    if args.load:
        sd = torch.load(args.load, map_location=args.device)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  loaded {len(sd)} tensors  (unexpected keys: {len(unexpected)})", flush=True)
        if unexpected:
            print(f"  WARNING unexpected keys (naming mismatch?): {unexpected[:4]}", flush=True)
    else:
        print("  WARNING: no --load; routers are UNTRAINED (bias init -> ~all-keep). "
              "This is a plumbing smoke only; routed will ~= dense.", flush=True)
    model.eval()

    ev = EVALS[args.dataset]
    data = ev["load"](args.n_examples, args.seed)
    prompts = [ev["prompt"](q) for q, _ in data]
    golds = [g for _, g in data]
    batches = build_batches(prompts, golds, tokenizer, args.device, 1)   # bs=1 (ragged gsm8k)
    sampler = types.MethodType(GenClass.batch_sample, model)

    print(f"\n{args.dataset}  n={len(data)}  (orig vs LoRA-dense vs routed, recoverable-KV dense-commit)\n")
    print(f"  {'thresh':>6}  {'orig_acc':>8}  {'lora_dense':>10}  {'routed_acc':>10}  "
          f"{'r-orig':>7}  {'keep_ss':>7}  {'routed_NFE':>10}")
    print("  " + "-" * 78)
    for th in sorted(args.thresholds, reverse=True):
        # (a) ORIGINAL model: LoRA off, routers bypassed
        print(f"  [th={th:.2f}] (1/3) orig (LoRA off, n={len(batches)})...", flush=True)
        set_lora_enabled(model, False)
        with RouterGenController(model, wrappers, mode="dense"):
            o_acc, _ = run_threshold(model, tokenizer, sampler, batches, th,
                                     args.max_new_tokens, ev["parse_pred"],
                                     small_block_size=args.small_block_size)
        print(f"           orig_acc={o_acc:.1%}", flush=True)
        set_lora_enabled(model, True)
        # (b) LoRA-dense: LoRA on, routers bypassed (isolates whether LoRA itself shifted acc)
        print(f"  [th={th:.2f}] (2/3) lora_dense...", flush=True)
        with RouterGenController(model, wrappers, mode="dense"):
            d_acc, _ = run_threshold(model, tokenizer, sampler, batches, th,
                                     args.max_new_tokens, ev["parse_pred"],
                                     debug=args.debug, small_block_size=args.small_block_size)
        print(f"           lora_dense={d_acc:.1%}", flush=True)
        # (c) ROUTED: LoRA on, small steps gated, prefill+commit dense (recoverable-KV)
        print(f"  [th={th:.2f}] (3/3) routed...", flush=True)
        ctrl = RouterGenController(model, wrappers, mode="routed").install()
        r_acc, r_nfe = run_threshold(model, tokenizer, sampler, batches, th,
                                     args.max_new_tokens, ev["parse_pred"],
                                     small_block_size=args.small_block_size)
        keep = ctrl.avg_keep
        ke, kl = ctrl.avg_keep_early, ctrl.avg_keep_late
        ctrl.uninstall()
        print(f"           routed_acc={r_acc:.1%}  keep_ss={keep:.2f}  "
              f"[early {ke:.2f} / late {kl:.2f}]  "
              f"({'keeps MORE near commit' if kl > ke + 0.02 else 'flat/inverse'})", flush=True)
        print(f"  ROW  {th:>6.2f}  orig={o_acc:.1%}  lora_dense={d_acc:.1%}  routed={r_acc:.1%}  "
              f"(r-orig {r_acc-o_acc:+.1%})  keep_ss={keep:.2f}  NFE={r_nfe:.1f}", flush=True)

        # (ablation) routed but ALSO gate the commit step -> committed KV becomes approximate
        if args.ablate_gate_commit:
            print(f"  [th={th:.2f}] (ablation) routed + GATED COMMIT...", flush=True)
            ctrl2 = RouterGenController(model, wrappers, mode="routed", gate_commit=True,
                                        force_commit_keep=args.force_commit_keep).install()
            gc_acc, _ = run_threshold(model, tokenizer, sampler, batches, th,
                                      args.max_new_tokens, ev["parse_pred"],
                                      small_block_size=args.small_block_size)
            gc_keep, gc_keep_c = ctrl2.avg_keep, ctrl2.avg_keep_commit
            ctrl2.uninstall()
            print(f"  ABLATE gate-commit: acc={gc_acc:.1%} (vs routed {r_acc:.1%}, delta {gc_acc-r_acc:+.1%})  "
                  f"small-keep={gc_keep:.2f}  COMMIT-keep={gc_keep_c:.2f}", flush=True)
            print("    IF commit-keep ~1.0 -> commit was NOT actually gated -> inconclusive (no-op).", flush=True)
            print("    ELSE acc drop vs routed -> dense commit load-bearing; acc holds -> it is NOT.", flush=True)

    print("\nREAD:")
    print("  routed_acc ~= orig_acc at keep_ss<1  => full method (LoRA+routing) TRANSFERS to real")
    print("    decoding AND recoverable-KV/dense-commit holds => build the compute-skip kernel + throughput.")
    print("  lora_dense << orig  => LoRA training itself hurt gsm8k (retrain: raise lambda_ce / lower lr),")
    print("    separate from routing damage. routed << lora_dense => routing is the culprit -> need A1.")
    print("  keep_ss = small-step keep-rate; small steps ~94% of forwards, so a real kernel cashes in")
    print("    ~= 0.94 * (1 - keep_ss) layer-compute. (orig_acc should ~match your base gsm8k ref ~73.5%.)")


if __name__ == "__main__":
    main()
