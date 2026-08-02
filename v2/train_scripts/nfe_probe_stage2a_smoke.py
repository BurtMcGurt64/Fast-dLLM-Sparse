"""
STAGE 2a — TRAINING SMOKE TEST (run before building the full fine-tune).

Confirms the model's NATIVE masked-diffusion training forward runs end to end on
THIS box: loss computes and gradients flow. The training path uses flex attention
(create_block_mask / fused_flex_attention, modeling.py:511/281) — the one external
dependency likely to break on a fresh GPU/torch build. A training script can't be
"eyeballed" for correctness the way the eval was, so we validate the plumbing first.

PASS -> prints a finite loss + a gradient norm -> safe to build Stage 2b (LoRA
        fine-tune with high-mask-ratio bias for low-NFE specialization).
FAIL -> if create_block_mask / flex_attention errors, the training direction is
        blocked on this setup and that's the thing to fix (or a reason to weigh the
        set-diffusion pivot) BEFORE sinking days into a fine-tune.

Run:
  CUDA_VISIBLE_DEVICES=0 uv run v2/train_scripts/nfe_probe_stage2a_smoke.py
"""
import sys
import os
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "../experiments"))

import torch
from helper import load_model


def main():
    print("Loading model (train mode)...", flush=True)
    model = load_model("Efficient-Large-Model/Fast_dLLM_v2_7B", device="cuda")
    model.train()

    bd = model.model.bd_size
    print(f"bd_size = {bd}", flush=True)

    # tiny batch, length a multiple of bd so the block reshape in the training
    # forward (modeling.py:663) is exact
    B, L = 2, bd * 3
    torch.manual_seed(0)
    input_ids = torch.randint(0, 1000, (B, L), device="cuda")
    labels = input_ids.clone()
    labels[:, :bd] = -100  # first block = "prompt", no loss (mirrors real use)

    n_trainable = sum(p.requires_grad for p in model.parameters())
    print(f"trainable param tensors: {n_trainable}", flush=True)

    print("Forward (native masked-diffusion training path)...", flush=True)
    out = model(input_ids=input_ids, labels=labels)
    loss = out.loss
    finite = torch.isfinite(loss).item()
    print(f"  loss = {loss.item():.4f}   finite = {finite}", flush=True)
    assert finite, "loss is not finite — training objective is misconfigured"

    print("Backward...", flush=True)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients flowed — check requires_grad / the forward path"
    gnorm = torch.sqrt(sum((g.float() ** 2).sum() for g in grads)).item()
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  {len(grads)} grad tensors, global grad norm = {gnorm:.3f}", flush=True)
    print(f"  peak memory = {peak:.1f} GB", flush=True)

    print("\nPASS: native training forward + backward works on this box.")
    print("Flex-attention training path is fine -> safe to build Stage 2b (LoRA + high-mask fine-tune).")


if __name__ == "__main__":
    main()
