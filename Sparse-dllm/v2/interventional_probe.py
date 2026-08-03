"""
L1 INTERVENTIONAL PROBE — does staleness predict module-skippability, beyond trivial baselines?

Tests the core hypothesis of the convergence-router direction (SkipGPT + block-diffusion staleness):
for a converged token, is skipping a module (attn/MLP) low-damage, and does STALENESS predict that
better than trivial baselines? This is the experiment that decides whether the router is novel or is
just SkipGPT, and which module/metric to route on.

METHOD (one-step counterfactual, self-contained — an L1 proxy for the full-trajectory oracle):
  1. Prefill a real prompt. Build the next 32-tok block in two states — "early" (heavily masked,
     unconverged) and "late" (lightly masked, mostly the true tokens, converged).
  2. Per-token STALENESS = change from early->late: l2 = ||h_late-h_early||/||h_early||;
     kl = KL(softmax(logits_late)||softmax(logits_early)). (Proxy for ||Δh|| across two denoise steps.)
  3. At the LATE state, for sampled (token t, layer l, module m): residual-skip module m for token t,
     one forward, and measure DAMAGE vs the no-skip reference:
       local = KL(ref_logits[t] || skip_logits[t])           # token t's own prediction shift
       cross = mean_{t'!=t} KL(ref_logits[t'] || skip_logits[t'])  # leakage into OTHER block tokens (KV coupling)
  4. GRADED analysis (not a brittle binary): Spearman(staleness, damage); plus AUC for a skippable
     label (local < eps) from each predictor, per module, per depth.

BASELINE LADDER (the key comparison — does the diffusion signal earn its keep?):
  raw ||hidden||  <  is-settled flag  <  staleness-l2  <  staleness-kl
  If staleness doesn't beat the settled flag / raw hidden, the router doesn't need hidden-state
  introspection and the direction collapses toward vanilla SkipGPT.

Run (idle GPU):   CUDA_VISIBLE_DEVICES=7 python interventional_probe.py --n 8
Smoke test first: CUDA_VISIBLE_DEVICES=7 python interventional_probe.py --n 2 --layers 12 --smoke
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np

from transformers.cache_utils import DynamicCache
DynamicCache.__getitem__ = lambda self, i: (self.layers[i].keys, self.layers[i].values)
from transformers import AutoTokenizer
from model.modeling import Fast_dLLM_QwenForCausalLM

MASK_ID = 151665
BLOCK = 32


def load_model(model_id, device):
    import json
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from model.configuration import Fast_dLLM_QwenConfig
    snap = snapshot_download(model_id, local_files_only=True)
    cfg = Fast_dLLM_QwenConfig.from_pretrained(snap)
    model = Fast_dLLM_QwenForCausalLM(cfg).to(torch.bfloat16)
    with open(os.path.join(snap, "model.safetensors.index.json")) as f:
        index = json.load(f)
    sd = {}
    for shard in sorted(set(index["weight_map"].values())):
        sd.update(load_file(os.path.join(snap, shard), device="cpu"))
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


def _kl(p_logits, q_logits):
    """KL(softmax(p) || softmax(q)) over last dim, per row. Inputs [...,V]."""
    p = torch.log_softmax(p_logits.float(), dim=-1)
    q = torch.log_softmax(q_logits.float(), dim=-1)
    return (p.exp() * (p - q)).sum(dim=-1)


@torch.no_grad()
def block_forward(model, block_ids, pkv, skip=None):
    """One block small-step forward. skip=(layer, module) residual-skips that module for the given
    token positions set on the layer beforehand. Returns logits[1,T,V], hidden[1,T,D]."""
    out = model.forward(input_ids=block_ids, use_cache=True, past_key_values=pkv,
                        update_past_key_values=False, use_block_cache=True, output_hidden_states=True)
    return out.logits, out.hidden_states[-1]


@torch.no_grad()
def probe_one(model, ids, n_prompt_blocks, device, layers, mask_early=0.7, mask_late=0.25, seed=0):
    n_prompt = n_prompt_blocks * BLOCK
    if ids.shape[0] < n_prompt + BLOCK:
        return None
    prompt = ids[:n_prompt].unsqueeze(0).to(device)
    true_block = ids[n_prompt:n_prompt + BLOCK].clone().unsqueeze(0).to(device)

    pkv = model.forward(input_ids=prompt, use_cache=True, update_past_key_values=True,
                        block_size=BLOCK, logits_to_keep=1).past_key_values

    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(BLOCK, generator=g)
    n_e, n_l = int(mask_early * BLOCK), int(mask_late * BLOCK)
    early = true_block.clone(); early[0, order[:n_e]] = MASK_ID          # heavily masked (unconverged)
    late = true_block.clone();  late[0, order[:n_l]] = MASK_ID           # lightly masked (converged)

    logits_e, h_e = block_forward(model, early, pkv)
    logits_l, h_l = block_forward(model, late, pkv)                      # reference state
    stale_l2 = ((h_l - h_e).float().norm(dim=-1) / h_e.float().norm(dim=-1).clamp_min(1e-6))[0].cpu().numpy()
    stale_kl = _kl(logits_l, logits_e)[0].cpu().numpy()
    settled = (late[0] != MASK_ID).cpu().numpy().astype(float)          # baseline: is-settled flag
    hnorm = h_l[0].float().norm(dim=-1).cpu().numpy()                   # baseline: raw ||hidden||
    ref = logits_l                                                      # [1,T,V]

    rows = []
    for l in layers:
        for mod in ("attn", "ffn"):
            for t in range(BLOCK):
                setattr(model.model.layers[l], f"_skip_{mod}_pos", torch.tensor([t], device=device))
                skip_logits, _ = block_forward(model, late, pkv)
                setattr(model.model.layers[l], f"_skip_{mod}_pos", None)
                kl_all = _kl(ref, skip_logits)[0]                       # [T]
                local = kl_all[t].item()
                cross = (kl_all.sum().item() - local) / (BLOCK - 1)
                rows.append(dict(layer=l, mod=mod, tok=t, local=local, cross=cross,
                                 stale_l2=float(stale_l2[t]), stale_kl=float(stale_kl[t]),
                                 settled=float(settled[t]), hnorm=float(hnorm[t])))
    return rows


def auc(score, label):
    """AUC that a LOW score predicts label=1 (skippable). Mann-Whitney on -score."""
    score = np.asarray(score, float); label = np.asarray(label, int)
    pos, neg = label == 1, label == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    ranks = np.empty(len(score))                      # rank by -score (low score => rank 1)
    ranks[np.argsort(-score)] = np.arange(1, len(score) + 1)
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n", type=int, default=8, help="probe examples")
    p.add_argument("--prompt_blocks", type=int, default=6)
    p.add_argument("--layers", default="", help="comma-sep layer indices to test (default: every 4th)")
    p.add_argument("--eps_pct", type=float, default=50.0, help="skippable = local damage below this percentile")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("Loading model...", flush=True)
    model = load_model(args.model, args.device)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    L = model.config.num_hidden_layers
    layers = [int(x) for x in args.layers.split(",") if x] or list(range(0, L, 4))
    if args.smoke:
        layers = layers[:1]

    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    text = "\n\n".join(f"Question: {e['question']}\nAnswer: {e['answer']}" for e in ds.select(range(300)))
    stream = tok(text, return_tensors="pt").input_ids[0]
    win = args.prompt_blocks * BLOCK + BLOCK

    rows = []
    for k in range(args.n):
        s = k * win
        if s + win > stream.shape[0]:
            break
        r = probe_one(model, stream[s:s + win], args.prompt_blocks, args.device, layers, seed=args.seed + k)
        if r:
            rows.extend(r)
        print(f"  example {k+1}/{args.n} done ({len(rows)} intervention points)", flush=True)

    if not rows:
        print("no data"); return
    depth_of = lambda l: "early" if l < L / 3 else ("mid" if l < 2 * L / 3 else "late")
    preds = ["hnorm", "settled", "stale_l2", "stale_kl"]

    print(f"\nInterventional skippability — {len(rows)} points ({args.n} examples), skippable = local damage "
          f"below {args.eps_pct:.0f}th pct")
    print("  Baseline ladder AUC (higher = predictor better separates skippable). settled/hnorm are the")
    print("  trivial baselines; staleness must BEAT them or the diffusion signal adds nothing.\n")
    print(f"  {'module':>5} {'depth':>5} {'n':>5} {'skip%':>6}  " + "  ".join(f"{p:>9}" for p in preds)
          + f"  {'ρ(l2,local)':>11} {'ρ(l2,cross)':>11}")
    print("  " + "-" * 100)
    for mod in ("attn", "ffn"):
        for d in ("early", "mid", "late"):
            grp = [r for r in rows if r["mod"] == mod and depth_of(r["layer"]) == d]
            if not grp:
                continue
            local = np.array([r["local"] for r in grp])
            cross = np.array([r["cross"] for r in grp])
            eps = np.percentile(local, args.eps_pct)
            label = (local <= eps).astype(int)                 # 1 = low damage = skippable
            aucs = [auc([r[p] for r in grp], label) for p in preds]  # low predictor -> skippable
            sp_l = spearman([r["stale_l2"] for r in grp], local)
            sp_c = spearman([r["stale_l2"] for r in grp], cross)
            print(f"  {mod:>5} {d:>5} {len(grp):>5} {label.mean():>5.0%}  "
                  + "  ".join(f"{a:>9.3f}" for a in aucs) + f"  {sp_l:>11.2f} {sp_c:>11.2f}")

    print("\n  READ:")
    print("  - stale_l2/stale_kl AUC > settled/hnorm AUC  => staleness carries signal beyond trivial => router justified.")
    print("  - stale AUC ~= settled AUC                    => a settle-counter suffices; no hidden-state router needed.")
    print("  - attn vs ffn skip% / AUC gap                => which module to route harder (Xinze's decoupling).")
    print("  - ρ(l2,cross) strongly negative               => converged tokens leak little into others => globally safe.")
    print("  NOTE: L1 proxy (2-step staleness, one-step damage). If signal is real, confirm with full-trajectory oracle (L2).")


if __name__ == "__main__":
    main()
