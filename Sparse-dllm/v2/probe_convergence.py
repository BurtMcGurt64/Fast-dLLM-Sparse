"""
CONVERGENCE-CONDITIONAL LAYER REDUNDANCY PROBE — the make-or-break test for the SkipGPT-style
convergence-router direction.

Hypothesis: in block diffusion, a *converged* token (already decided / unmasked) gets LESS from
later layers/modules than an *unconverged* (masked) token — i.e. layer redundancy is conditional on
convergence, even though it's absent on the AVERAGE token (prior finding: v2 has no static redundancy).
If true, a router conditioned on convergence can skip modules for settled tokens. If contribution is
flat regardless of convergence, the router has nothing to exploit -> kill the direction cheaply.

Method (controlled, single forward per example): prefill a real prompt, take the next 32-token block,
mask ~half its positions (unconverged) and keep the rest as the true tokens (converged). One block
forward with per-module contribution capture (modeling.py `_capture_contrib`). Compare, per layer
depth, the relative contribution ||module(x)||/||x|| of masked vs real positions, for attention vs MLP.

Read:  conv/unconv ratio < 1  => converged tokens use that module less => skippable (thesis holds).
       attn ratio vs ffn ratio => which module to route harder (Xinze's module-decoupling question).

Run (idle GPU):  CUDA_VISIBLE_DEVICES=7 python probe_convergence.py --n 40 --mask_frac 0.5
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from transformers.cache_utils import DynamicCache
DynamicCache.__getitem__ = lambda self, i: (self.layers[i].keys, self.layers[i].values)

from transformers import AutoTokenizer
from model.modeling import Fast_dLLM_QwenForCausalLM

MASK_ID = 151665
BLOCK = 32


def load_model(model_id, device):
    import json as _json
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from model.configuration import Fast_dLLM_QwenConfig
    snap = snapshot_download(model_id, local_files_only=True)
    config = Fast_dLLM_QwenConfig.from_pretrained(snap)
    model = Fast_dLLM_QwenForCausalLM(config).to(torch.bfloat16)
    with open(os.path.join(snap, "model.safetensors.index.json")) as f:
        index = _json.load(f)
    sd = {}
    for shard in sorted(set(index["weight_map"].values())):
        sd.update(load_file(os.path.join(snap, shard), device="cpu"))
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


@torch.no_grad()
def probe_one(model, ids, n_prompt_blocks, mask_frac, seed, device):
    """ids: 1D LongTensor long enough for n_prompt_blocks*32 prompt + one 32-tok block."""
    n_prompt = n_prompt_blocks * BLOCK
    if ids.shape[0] < n_prompt + BLOCK:
        return None
    prompt = ids[:n_prompt].unsqueeze(0).to(device)
    block_true = ids[n_prompt:n_prompt + BLOCK].clone().unsqueeze(0).to(device)   # [1,32] real tokens

    out = model.forward(input_ids=prompt, use_cache=True, update_past_key_values=True,
                        block_size=BLOCK, logits_to_keep=1)
    pkv = out.past_key_values

    g = torch.Generator().manual_seed(seed)
    mask_pos = (torch.rand(BLOCK, generator=g) < mask_frac)                        # True = masked (unconverged)
    if mask_pos.all() or (~mask_pos).all():                                        # need both groups
        mask_pos[0] = False; mask_pos[-1] = True
    block = block_true.clone()
    block[0, mask_pos] = MASK_ID

    for l in model.model.layers: l._capture_contrib = True
    model.forward(input_ids=block, use_cache=True, past_key_values=pkv,
                  update_past_key_values=False, use_block_cache=True, output_hidden_states=True)
    for l in model.model.layers: l._capture_contrib = False

    attn = torch.stack([l._attn_contrib[0] for l in model.model.layers]).cpu()     # [L, 32]
    ffn  = torch.stack([l._ffn_contrib[0]  for l in model.model.layers]).cpu()     # [L, 32]
    return attn, ffn, mask_pos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n", type=int, default=40, help="number of probe examples")
    p.add_argument("--prompt_blocks", type=int, default=6, help="prompt length in 32-tok blocks")
    p.add_argument("--mask_frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("Loading model...", flush=True)
    model = load_model(args.model, args.device)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Build one long token stream from GSM8K (cached) and slice non-overlapping probe windows.
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    text = "\n\n".join(f"Question: {ex['question']}\nAnswer: {ex['answer']}" for ex in ds.select(range(200)))
    stream = tok(text, return_tensors="pt").input_ids[0]
    win = args.prompt_blocks * BLOCK + BLOCK

    L = model.config.num_hidden_layers
    # accumulate mean contribution per layer, split by group (converged/real vs unconverged/masked)
    sums = {("attn", "conv"): torch.zeros(L), ("attn", "unconv"): torch.zeros(L),
            ("ffn", "conv"): torch.zeros(L), ("ffn", "unconv"): torch.zeros(L)}
    cnt = {"conv": 0, "unconv": 0}
    done = 0
    for k in range(args.n):
        start = k * win
        if start + win > stream.shape[0]:
            break
        r = probe_one(model, stream[start:start + win], args.prompt_blocks, args.mask_frac, args.seed + k, args.device)
        if r is None:
            continue
        attn, ffn, mask_pos = r                      # attn/ffn: [L,32]; mask_pos: [32] True=unconv
        conv, unconv = ~mask_pos, mask_pos
        sums[("attn", "conv")]   += attn[:, conv].sum(dim=1);   sums[("attn", "unconv")] += attn[:, unconv].sum(dim=1)
        sums[("ffn", "conv")]    += ffn[:, conv].sum(dim=1);    sums[("ffn", "unconv")]  += ffn[:, unconv].sum(dim=1)
        cnt["conv"] += int(conv.sum()); cnt["unconv"] += int(unconv.sum())
        done += 1

    def mean(mod, grp):  # per-layer mean contribution
        return sums[(mod, grp)] / max(1, cnt[grp])

    buckets = [("early", range(0, L // 3)), ("mid", range(L // 3, 2 * L // 3)), ("late", range(2 * L // 3, L))]
    print(f"\nConvergence-conditional redundancy probe — {done} examples, mask_frac={args.mask_frac}")
    print("  contribution = ||module(x)|| / ||x|| (small => module barely changes the token => skippable)")
    print("  conv = real/settled tokens; unconv = masked tokens. ratio = conv/unconv (< 1 => converged skippable)\n")
    print(f"  {'depth':>6} {'module':>5} {'unconv':>8} {'conv':>8} {'ratio':>7}")
    print("  " + "-" * 40)
    for name, rng in buckets:
        for mod in ("attn", "ffn"):
            u = mean(mod, "unconv")[list(rng)].mean().item()
            c = mean(mod, "conv")[list(rng)].mean().item()
            print(f"  {name:>6} {mod:>5} {u:>8.3f} {c:>8.3f} {c / max(1e-6, u):>7.2f}")
    print("\n  Read: if late-layer ratios (esp) are < ~0.9, converged tokens contribute less there ->")
    print("  convergence-conditional redundancy EXISTS -> a convergence router can skip modules. The")
    print("  attn-vs-ffn ratio gap says which module to route harder (Xinze's module-decoupling point).")
    print("  If ratios ~1.0 everywhere, redundancy is NOT convergence-conditional -> router has no signal.")


if __name__ == "__main__":
    main()
