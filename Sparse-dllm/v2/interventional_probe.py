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


def compute_kv_signals(k_early, k_late, q_late, unsettled):
    """Per-layer, per-token KV-convergence signals (the untested dimension: layer-local + cross-token).
      kv_stale[l,t] = ||K_t^late - K_t^early|| / ||K_t^early||   (this token's KEY changed across states)
      kv_infl[l,t]  = (block-local attention INTO t from UNSETTLED tokens) * kv_stale   (cross-token damage)
    Both are things SkipGPT's per-token final-hidden router cannot see. Block-local softmax over the 32
    block tokens is a cheap proxy for within-block coupling (ignores prefix competition — noted limit)."""
    out = {}
    uns = unsettled.detach().cpu()
    for l in range(len(k_late)):
        ke = k_early[l][0].float(); kl = k_late[l][0].float()           # [H_kv, T, d]
        T = ke.shape[1]
        dnum = (kl - ke).transpose(0, 1).reshape(T, -1).norm(dim=1)     # [T]
        dden = ke.transpose(0, 1).reshape(T, -1).norm(dim=1).clamp_min(1e-6)
        kv_stale = (dnum / dden).cpu()                                  # [T]

        q = q_late[l][0].float(); H = q.shape[0]; d = q.shape[-1]        # q: [H, T, d]
        H_kv = kl.shape[0]
        k_exp = kl.repeat_interleave(H // H_kv, dim=0)                  # [H, T, d]
        A = torch.softmax((q @ k_exp.transpose(1, 2)) / (d ** 0.5), dim=-1)  # [H, T(query), T(key)]
        if uns.any():
            into_t = A[:, uns, :].mean(dim=(0, 1)).cpu()               # attn INTO each key t from unsettled queries
        else:
            into_t = A.mean(dim=(0, 1)).cpu()
        out[l] = (kv_stale.numpy(), (into_t * kv_stale).numpy())
    return out


@torch.no_grad()
def block_forward(model, block_ids, pkv):
    """One block small-step forward. Returns logits[1,T,V], hidden[1,T,D].
    Note: out.hidden_states is the FINAL hidden state tensor [1,T,D] (not a per-layer tuple)."""
    out = model.forward(input_ids=block_ids, use_cache=True, past_key_values=pkv,
                        update_past_key_values=False, use_block_cache=True, output_hidden_states=True)
    h = out.hidden_states
    if h.dim() == 2:
        h = h.unsqueeze(0)
    return out.logits, h


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

    def set_cap(v):
        for lyr in model.model.layers:
            lyr.self_attn._capture_kv = v
            lyr._capture_hidden = v

    def grab_hidden():
        return ([lyr._preattn_hidden[0].float().cpu().numpy().astype(np.float16) for lyr in model.model.layers],
                [lyr._premlp_hidden[0].float().cpu().numpy().astype(np.float16) for lyr in model.model.layers])

    set_cap(True)
    logits_e, h_e = block_forward(model, early, pkv)
    k_early = [lyr.self_attn._k_snap.clone() for lyr in model.model.layers]     # per layer [1,H_kv,T,d]
    preattn_e, premlp_e = grab_hidden()                                 # per-layer hidden at EARLY state
    logits_l, h_l = block_forward(model, late, pkv)                      # reference state
    k_late = [lyr.self_attn._k_snap.clone() for lyr in model.model.layers]
    q_late = [lyr.self_attn._q_snap.clone() for lyr in model.model.layers]
    preattn, premlp = grab_hidden()                                     # per-layer hidden at LATE state
    set_cap(False)

    stale_l2 = ((h_l - h_e).float().norm(dim=-1) / h_e.float().norm(dim=-1).clamp_min(1e-6))[0].cpu().numpy()
    stale_kl = _kl(logits_l, logits_e)[0].cpu().numpy()
    settled = (late[0] != MASK_ID).cpu().numpy().astype(float)          # baseline: is-settled flag
    unsettled = (late[0] == MASK_ID)                                    # bool [T]
    hnorm = h_l[0].float().norm(dim=-1).cpu().numpy()                   # baseline: raw ||hidden||
    kv_signals = compute_kv_signals(k_early, k_late, q_late, unsettled)  # {layer: (kv_stale[T], kv_infl[T])}
    ref = logits_l                                                      # [1,T,V]
    ref_arg = ref.argmax(-1)[0]                                         # [T] reference decoded ids

    rows = []
    for l in layers:
        for mod in ("attn", "ffn"):
            for t in range(BLOCK):
                setattr(model.model.layers[l], f"_skip_{mod}_pos", torch.tensor([t], device=device))
                skip_logits, _ = block_forward(model, late, pkv)
                setattr(model.model.layers[l], f"_skip_{mod}_pos", None)
                kl_all = _kl(ref, skip_logits)[0]                       # [T]
                flip = (skip_logits.argmax(-1)[0] != ref_arg)          # [T] bool: did the argmax flip
                local_flip = int(flip[t].item())                       # ABSOLUTE, comparable label
                cross_flip = (int(flip.sum().item()) - local_flip) / (BLOCK - 1)  # leakage into others
                kv_s, kv_i = kv_signals[l]
                rows.append(dict(layer=l, mod=mod, tok=t,
                                 local_flip=local_flip, cross_flip=cross_flip,
                                 local_kl=kl_all[t].item(),
                                 cross_kl=(kl_all.sum().item() - kl_all[t].item()) / (BLOCK - 1),
                                 stale_l2=float(stale_l2[t]), stale_kl=float(stale_kl[t]),
                                 settled=float(settled[t]), hnorm=float(hnorm[t]),
                                 kv_stale=float(kv_s[t]), kv_infl=float(kv_i[t])))
    # Full late-hidden per token (final) + per-layer pre-module hidden at LATE and EARLY states
    # (EARLY enables the hidden-DELTA baseline: same temporal info as KV, but on the hidden not the key).
    hidden_np = h_l[0].float().cpu().numpy().astype(np.float16)          # [T, D]
    return rows, hidden_np, preattn, premlp, preattn_e, premlp_e


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


def grouped_cv_auc(X, y, groups, n_pca=0, C=1.0):
    """Example-held-out CV AUC for a logistic probe. groups = example id per row (leakage-safe:
    tokens from the same block never span train/test). n_pca>0 => PCA-reduce first (for the
    high-dim hidden probe). Returns (mean test AUC over folds, list of per-fold AUCs)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score
    X = np.asarray(X, float); y = np.asarray(y, int); groups = np.asarray(groups)
    uniq = np.unique(groups)
    if len(uniq) < 3 or y.min() == y.max():
        return float("nan"), []
    folds = np.array_split(uniq, min(5, len(uniq)))
    aucs = []
    for te_g in folds:
        te = np.isin(groups, te_g); tr = ~te
        if y[tr].min() == y[tr].max() or y[te].min() == y[te].max():
            continue
        sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        if n_pca and n_pca < Xtr.shape[1]:
            pca = PCA(n_components=n_pca).fit(Xtr); Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
        clf = LogisticRegression(C=C, max_iter=2000).fit(Xtr, y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1]))
    return (float(np.mean(aucs)) if aucs else float("nan")), aucs


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
    hiddens = {}                                                         # (ex, tok) -> final hidden vector
    play = {}                                                            # (ex, layer, tok, mod) -> per-layer late hidden
    play_e = {}                                                          # ... early-state per-layer hidden (for delta)
    for k in range(args.n):
        s = k * win
        if s + win > stream.shape[0]:
            break
        out = probe_one(model, stream[s:s + win], args.prompt_blocks, args.device, layers, seed=args.seed + k)
        if out is None:
            continue
        r, hid, preattn, premlp, preattn_e, premlp_e = out
        for row in r:
            row["ex"] = k
            rows.append(row)
        for t in range(hid.shape[0]):
            hiddens[(k, t)] = hid[t]
        for l in layers:
            for t in range(preattn[l].shape[0]):
                play[(k, l, t, "attn")] = preattn[l][t]
                play[(k, l, t, "ffn")] = premlp[l][t]
                play_e[(k, l, t, "attn")] = preattn_e[l][t]
                play_e[(k, l, t, "ffn")] = premlp_e[l][t]
        print(f"  example {k+1}/{args.n} done ({len(rows)} intervention points)", flush=True)

    if not rows:
        print("no data"); return
    depth_of = lambda l: "early" if l < L / 3 else ("mid" if l < 2 * L / 3 else "late")
    preds = ["hnorm", "settled", "stale_l2", "stale_kl", "kv_stale", "kv_infl"]

    def disc_auc(grp, pred):
        """Discrimination AUC (direction-agnostic): how well `pred` separates skippable (argmax
        preserved) from not. max(auc, 1-auc) since predictors have different natural signs."""
        label = np.array([1 - r["local_flip"] for r in grp])          # 1 = argmax preserved = skippable
        a = auc([r[pred] for r in grp], label)
        return max(a, 1 - a) if a == a else float("nan")

    def boot_ci(grp, pred, nboot=300):
        """Bootstrap disc-AUC over EXAMPLES (not rows) — points within a block are correlated."""
        by_ex = {}
        for r in grp: by_ex.setdefault(r["ex"], []).append(r)
        exs = list(by_ex); rng = np.random.default_rng(0)
        vals = []
        for _ in range(nboot):
            samp = [x for e in rng.choice(exs, len(exs)) for x in by_ex[e]]
            v = disc_auc(samp, pred)
            if v == v: vals.append(v)
        if not vals: return float("nan"), float("nan")
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

    # --- Table 1: baseline-ladder discrimination AUC + comparable skip% (absolute argmax-flip label) ---
    print(f"\nInterventional skippability — {len(rows)} points, {args.n} examples.")
    print("  skip% = fraction where the token's decoded argmax is UNCHANGED after residual-skipping the module")
    print("  (absolute, comparable across rows). AUC = how well each predictor separates skippable; staleness")
    print("  must BEAT settled/hnorm (the trivial baselines) or the diffusion signal adds nothing over SkipGPT.\n")
    print(f"  {'module':>5} {'depth':>5} {'n':>5} {'skip%':>6} {'xflip%':>6}  " + "  ".join(f"{p:>8}" for p in preds)
          + f"  {'ρ(l2,dmg)':>9}")
    print("  " + "-" * 92)
    for mod in ("attn", "ffn"):
        for d in ("early", "mid", "late"):
            grp = [r for r in rows if r["mod"] == mod and depth_of(r["layer"]) == d]
            if not grp: continue
            skip_pct = np.mean([1 - r["local_flip"] for r in grp])
            xflip = np.mean([r["cross_flip"] for r in grp])            # leakage into other tokens
            aucs = [disc_auc(grp, p) for p in preds]
            sp = spearman([r["stale_l2"] for r in grp], [r["local_kl"] for r in grp])
            print(f"  {mod:>5} {d:>5} {len(grp):>5} {skip_pct:>5.0%} {xflip:>5.1%}  "
                  + "  ".join(f"{a:>8.3f}" for a in aucs) + f"  {sp:>9.2f}")

    # --- Table 2: the confound check — is the depth trend convergence-driven or "less room to propagate"? ---
    print("\n  Confound check — skip% for CONVERGED (settled) vs UNCONVERGED (masked) tokens, by depth.")
    print("  If converged >> unconverged at all depths -> convergence effect (good). If both rise together at")
    print("  late layers -> it's just 'late skips have less room to propagate', NOT your hypothesis.\n")
    print(f"  {'module':>5} {'depth':>5} {'conv skip%':>11} {'unconv skip%':>13} {'gap':>7}")
    print("  " + "-" * 50)
    for mod in ("attn", "ffn"):
        for d in ("early", "mid", "late"):
            grp = [r for r in rows if r["mod"] == mod and depth_of(r["layer"]) == d]
            c = [1 - r["local_flip"] for r in grp if r["settled"] > 0.5]
            u = [1 - r["local_flip"] for r in grp if r["settled"] <= 0.5]
            if not c or not u: continue
            print(f"  {mod:>5} {d:>5} {np.mean(c):>10.0%} {np.mean(u):>12.0%} {np.mean(c)-np.mean(u):>+7.0%}")

    # --- Table 3: headline — does staleness beat the trivial baselines? (bootstrap CI over examples) ---
    print("\n  HEADLINE — pooled discrimination AUC with 95% bootstrap CI over EXAMPLES (the honest n).")
    print("  stale_* CI clearing settled/hnorm CI = staleness carries real signal beyond trivial -> router justified.\n")
    for mod in ("attn", "ffn"):
        grp = [r for r in rows if r["mod"] == mod]
        print(f"    [{mod}]")
        for p in preds:
            lo, hi = boot_ci(grp, p)
            print(f"      {p:>9}: AUC {disc_auc(grp, p):.3f}  95%CI [{lo:.3f}, {hi:.3f}]")

    # --- Table 4: THE GATE — does staleness beat a TRAINED probe on the full hidden (SkipGPT's real
    #     router input)? And does a learned combo of ingredients beat hand-crafted KL? ---
    print("\n  ============================ THE ROUTER GATE ============================")
    print("  Example-held-out CV AUC. 'hidden_probe' = logistic on the FULL hidden vector (SkipGPT's")
    print("  real router input, not just its norm). 'learned_scalar' = logistic on {kl,l2,hnorm,settled}.")
    print("  DECISION: if stale_kl does NOT beat hidden_probe -> the convergence signal adds nothing over")
    print("  SkipGPT -> router not novel. If learned_scalar >> stale_kl -> a learned combo beats hand-KL.\n")
    try:
        for mod in ("attn", "ffn"):
            grp = [r for r in rows if r["mod"] == mod]
            y = np.array([1 - r["local_flip"] for r in grp])            # 1 = skippable (argmax preserved)
            groups = np.array([r["ex"] for r in grp])
            # single-predictor CV AUC (direction-agnostic via best-sign, using the raw scalar)
            def scalar_cv(key):
                X = np.array([[r[key]] for r in grp]); a, _ = grouped_cv_auc(X, y, groups)
                return max(a, 1 - a) if a == a else a
            kl_auc = scalar_cv("stale_kl")
            kvs_auc = scalar_cv("kv_stale")
            kvi_auc = scalar_cv("kv_infl")
            learned = np.array([[r["stale_kl"], r["stale_l2"], r["hnorm"], r["settled"]] for r in grp])
            ls_auc, _ = grouped_cv_auc(learned, y, groups)
            # KV combo = the untested dimension on its own (kv_stale + cross-token kv_infl)
            kvcombo = np.array([[r["kv_stale"], r["kv_infl"]] for r in grp])
            kvc_auc, _ = grouped_cv_auc(kvcombo, y, groups)
            H = np.array([hiddens[(r["ex"], r["tok"])] for r in grp], dtype=np.float32)
            hp_auc, _ = grouped_cv_auc(H, y, groups, n_pca=64)
            # FAIR SkipGPT baseline: per-layer pre-module hidden (its ACTUAL router input, layer-local, 1 state).
            Hpl = np.array([play[(r["ex"], r["layer"], r["tok"], mod)] for r in grp], dtype=np.float32)
            hpl_auc, _ = grouped_cv_auc(Hpl, y, groups, n_pca=64)
            # DECISIVE baseline: per-layer hidden at BOTH states (same temporal info as KV, on the hidden).
            # If KV still beats this, the signal is specifically the KEY, not just 'having a delta'.
            Hd = np.array([np.concatenate([play_e[(r["ex"], r["layer"], r["tok"], mod)],
                                           play[(r["ex"], r["layer"], r["tok"], mod)]]) for r in grp], dtype=np.float32)
            hd_auc, _ = grouped_cv_auc(Hd, y, groups, n_pca=64)
            print(f"    [{mod}]  skippable={y.mean():.0%}")
            print(f"      stale_kl (hand)          : CV-AUC {kl_auc:.3f}")
            print(f"      kv_stale (KV key-drift)  : CV-AUC {kvs_auc:.3f}   [layer-local key delta]")
            print(f"      kv_infl (KV cross-tok)   : CV-AUC {kvi_auc:.3f}   [into-t attn from unsettled x delta]")
            print(f"      kv_combo                 : CV-AUC {kvc_auc:.3f}   (kv_stale + kv_infl)")
            print(f"      hidden_probe (per-layer) : CV-AUC {hpl_auc:.3f}   <- SkipGPT baseline (1 state)")
            print(f"      hidden_DELTA (per-layer) : CV-AUC {hd_auc:.3f}   <- SkipGPT + SAME temporal info (2 states)")
            fair = hd_auc if hd_auc == hd_auc else (hpl_auc if hpl_auc == hpl_auc else hp_auc)
            best_kv = max(v for v in (kvs_auc, kvi_auc, kvc_auc) if v == v)
            if best_kv > fair + 0.02:
                print(f"      -> KV ({best_kv:.3f}) BEATS hidden-DELTA ({fair:.3f}) -> it's specifically the KEY,")
                print(f"         not just temporality -> genuinely beyond SkipGPT. Confirm w/ real-traj L2.")
            else:
                print(f"      -> KV ({best_kv:.3f}) does NOT beat hidden-DELTA ({fair:.3f}) -> the win was TEMPORALITY,")
                print(f"         which you could give SkipGPT too -> weaker claim (not key-specific).")
    except ImportError:
        print("  [skipped: scikit-learn not installed -> `uv pip install scikit-learn` and rerun]")

    print("\n  NOTE: L1 proxy — staleness is a 2-mask-level delta (not real consecutive-step ‖Δh‖), and damage")
    print("  is logit-argmax-flip vs a 25%-masked reference (not a full re-decode to ground truth). If the")
    print("  ladder shows staleness winning, confirm with real-trajectory staleness + full-decode oracle (L2).")


if __name__ == "__main__":
    main()
