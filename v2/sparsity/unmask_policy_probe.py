"""
LEARNED-UNMASKING-POLICY PROBE — is there a better unmask signal than raw confidence?

Context (see verify_probe.py results): at th=0.6 the sampler takes 1.57x fewer forwards
at -6% accuracy, 16% of tokens differ from dense but most are benign, and raw p barely
separates surviving tokens from dying ones. The open question: does ANY function of
decode-time state predict "this token will match the dense trajectory" better than p?
If yes, a tiny learned unmasking policy on the FROZEN backbone (no LM fine-tuning, no
mode-collapse risk) can unmask more per step at equal quality -> NFE cut that composes
with top-K byte savings.

Two subcommands:

  collect   Teacher-forced replay (same scheme as verify_probe): per block, decode the
            dense reference (th=0.95) and aggressive draft trajectories from the same
            prefix; commit the dense block. At EVERY small step, for EVERY still-masked
            position, record the candidate token (current argmax), decision features,
            and the (shifted) hidden state that produced it. Label = candidate matches
            the dense reference token at that position. B=1, ~5-8 s/example on one GPU.

  train     Split by prompt, then compare rankers of "will this candidate survive":
              - raw p (the shipped threshold rule's signal — the baseline to beat)
              - logistic probe on 10 cheap scalar features
              - linear probe on the 3584-dim hidden state
              - combined (optionally a small MLP with --mlp)
            Reports val AUC (overall + draft-states-only) and a per-state top-k
            simulation: if a policy unmasked the top-k candidates per step ranked by
            each scorer, what fraction would be wrong? Seconds to minutes of compute.

How to read the result:
  - hidden/combined AUC >> p AUC and top-k wrong-rate clearly below p's at k=2-8
        -> a learned scheduler is real: build the policy, recover the -6% at ~1.5x NFE.
  - all AUCs ~= p's -> the decode-time state contains no better signal; the
        inference-only axis is fully closed on this checkpoint (take the dossier to
        the training conversation).

Feature vector (order): [p1, margin(p1-p2), entropy(nats), pos_in_block/31,
subblock_idx/3, step_in_block/16, frac_masked_block, frac_masked_window,
left_neighbor_unmasked, right_neighbor_unmasked]

Run (GPU box):
  CUDA_VISIBLE_DEVICES=0 uv run v2/sparsity/unmask_policy_probe.py collect \
      --n_examples 60 --draft_thresholds 0.6 --out v2/sparsity/policy_probe_data.pt
  CUDA_VISIBLE_DEVICES=0 uv run v2/sparsity/unmask_policy_probe.py train \
      --data v2/sparsity/policy_probe_data.pt
"""
import sys
import os
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "../experiments"))
sys.path.insert(0, os.path.join(HERE, "../train_scripts"))

import time
import argparse
import torch

BLOCK_SIZE = 32
SMALL_BLOCK = 8
STOP_TOKEN = 151645
MASK_ID = 151665
N_FEATS = 10
FEAT_NAMES = ["p1", "margin", "entropy", "pos", "subblock", "step",
              "frac_masked_blk", "frac_masked_win", "left_unmasked", "right_unmasked"]


# ===========================================================================
# COLLECT
# ===========================================================================
def _shift(t):
    """Their logit convention: shifted[i] = raw[i-1] predicts token[i]."""
    return torch.cat([t[:, :1], t[:, :-1]], dim=1)


@torch.no_grad()
def decode_block_capture(model, x_block, past_key_values, threshold,
                         small_block_size=SMALL_BLOCK, block_size=BLOCK_SIZE):
    """decode_block from verify_probe.py (mirrors batch_sample's inner loop, B=1)
    plus per-step capture of every masked position's candidate token, features, and
    the shifted hidden state that produced its prediction. Labels are assigned by the
    caller once the dense reference block is known."""
    x_t = x_block.clone()
    block_past_key_values = None
    steps = 0
    stage = {"feat": [], "hidden": [], "pos": [], "cand": [], "step": []}

    for sb in range(block_size // small_block_size):
        s, e = sb * small_block_size, (sb + 1) * small_block_size
        while True:
            mask_idx = (x_t == MASK_ID)
            if mask_idx[:, s:e].sum() == 0:
                break
            if block_past_key_values is None or (x_t[:, s] == MASK_ID).any():
                out = model.forward(input_ids=x_t, use_cache=True, past_key_values=past_key_values,
                                    update_past_key_values=False, use_block_cache=True)
                block_past_key_values = out.block_past_key_values
                logits = _shift(out.logits)[:, s:e]
                hidden = _shift(out.hidden_states)[:, s:e]
            else:
                out = model.forward(input_ids=x_t[:, s:e], use_cache=True,
                                    past_key_values=past_key_values,
                                    update_past_key_values=False, use_block_cache=True,
                                    block_past_key_values=block_past_key_values,
                                    replace_position=s)
                logits = _shift(out.logits)
                hidden = _shift(out.hidden_states)
            steps += 1
            x_1, p_1t = model.sample_with_top_p(logits, temperature=0.0)

            # ---- capture: every still-masked position in the window is a candidate ----
            wmask = mask_idx[0, s:e]                       # [w]
            widx = wmask.nonzero(as_tuple=False).squeeze(-1)
            if len(widx) > 0:
                p = p_1t[0].float()                        # [w, V]
                top2 = p.topk(2, dim=-1).values            # [w, 2]
                ent = -(p * (p + 1e-9).log()).sum(-1)      # [w]
                P = widx + s                               # absolute block positions
                left = torch.where(P > 0, x_t[0].gather(0, (P - 1).clamp(min=0)) != MASK_ID,
                                   torch.ones_like(P, dtype=torch.bool))
                right = torch.where(P < block_size - 1,
                                    x_t[0].gather(0, (P + 1).clamp(max=block_size - 1)) != MASK_ID,
                                    torch.zeros_like(P, dtype=torch.bool))
                feat = torch.stack([
                    top2[widx, 0],
                    top2[widx, 0] - top2[widx, 1],
                    ent[widx],
                    P.float() / (block_size - 1),
                    (P // small_block_size).float() / max(1, block_size // small_block_size - 1),
                    torch.full_like(P, steps, dtype=torch.float) / 16.0,
                    mask_idx[0].sum().float().expand(len(widx)) / block_size,
                    wmask.sum().float().expand(len(widx)) / (e - s),
                    left.float(),
                    right.float(),
                ], dim=1)                                  # [m, N_FEATS]
                stage["feat"].append(feat.cpu())
                stage["hidden"].append(hidden[0, widx].to(torch.float16).cpu())
                stage["pos"].append(P.cpu())
                stage["cand"].append(x_1[0, widx].cpu())
                stage["step"].append(torch.full((len(widx),), steps, dtype=torch.long))

            # ---- normal unmask update (identical to batch_sample) ----
            x1_p = torch.gather(p_1t, -1, x_1.unsqueeze(-1)).squeeze(-1)
            x1_p = torch.where(mask_idx[:, s:e], x1_p, -torch.inf)
            unmask = x1_p > threshold
            unmask[torch.arange(x_1.shape[0]), x1_p.argmax(dim=-1)] = True
            unmask = unmask & mask_idx[:, s:e]
            x_t[:, s:e][unmask] = x_1[unmask]
    return x_t, steps, stage


def flush_stage(store, stage, ref_block, prompt_idx, block_idx, source):
    """Assign labels (candidate == dense reference token) and move to the global store."""
    if not stage["feat"]:
        return
    pos = torch.cat(stage["pos"])
    cand = torch.cat(stage["cand"])
    ref = ref_block[0].cpu()
    store["feat"].append(torch.cat(stage["feat"]))
    store["hidden"].append(torch.cat(stage["hidden"]))
    store["label"].append((cand == ref.gather(0, pos)).float())
    n = len(pos)
    store["pos"].append(pos)
    store["step"].append(torch.cat(stage["step"]))
    store["prompt"].append(torch.full((n,), prompt_idx, dtype=torch.long))
    store["block"].append(torch.full((n,), block_idx, dtype=torch.long))
    store["source"].append(torch.full((n,), source))


@torch.no_grad()
def cmd_collect(args):
    from transformers import AutoTokenizer
    from helper import load_model
    from nfe_probe_stage1 import EVALS

    print(f"Loading model + {args.dataset}...", flush=True)
    model = load_model(args.model, device=args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ev = EVALS[args.dataset]
    data = ev["load"](args.n_examples, args.seed)

    store = {k: [] for k in ["feat", "hidden", "label", "pos", "step",
                             "prompt", "block", "source"]}
    t0 = time.perf_counter()
    for i, (q, _gold) in enumerate(data):
        ids = tokenizer(ev["prompt"](q), return_tensors="pt").input_ids.to(args.device)
        L = ids.shape[1]
        assert L > BLOCK_SIZE
        prefill_len = L // BLOCK_SIZE * BLOCK_SIZE
        out = model.forward(input_ids=ids[:, :prefill_len], use_cache=True,
                            update_past_key_values=True, block_size=BLOCK_SIZE,
                            logits_to_keep=1)
        pkv = out.past_key_values
        masks = lambda n: torch.full((1, n), MASK_ID, dtype=torch.long, device=args.device)
        if L % BLOCK_SIZE == 0:
            cur = torch.cat([out.logits[:, -1:, :].argmax(-1), masks(BLOCK_SIZE - 1)], dim=1)
        else:
            cur = torch.cat([ids[:, prefill_len:], masks(BLOCK_SIZE - (L - prefill_len))], dim=1)

        for block_idx in range(args.max_new_tokens // BLOCK_SIZE):
            x_init = cur
            # dense reference trajectory (also a capture source: its early-step
            # candidates are the counterfactual "unmask now?" decisions)
            ref_block, _steps, ref_stage = decode_block_capture(
                model, x_init, pkv, args.ref_threshold)
            flush_stage(store, ref_stage, ref_block, i, block_idx, args.ref_threshold)
            # aggressive draft trajectories, same prefix, labeled against dense
            for th in args.draft_thresholds:
                _draft, _dsteps, stage = decode_block_capture(model, x_init, pkv, th)
                flush_stage(store, stage, ref_block, i, block_idx, th)

            out = model.forward(input_ids=ref_block, use_cache=True, past_key_values=pkv,
                                update_past_key_values=True, block_size=BLOCK_SIZE)
            pkv = out.past_key_values
            cur = torch.cat([out.logits[:, -1:, :].argmax(-1), masks(BLOCK_SIZE - 1)], dim=1)
            if (ref_block[0] == STOP_TOKEN).any():
                break
        n_rows = sum(t.shape[0] for t in store["label"])
        print(f"  [{i + 1}/{len(data)}] rows={n_rows} "
              f"({(time.perf_counter() - t0) / (i + 1):.1f}s/ex)", flush=True)

    ds = {k: torch.cat(v) for k, v in store.items()}
    ds["feat_names"] = FEAT_NAMES
    ds["args"] = vars(args)
    torch.save(ds, args.out)
    n = ds["label"].shape[0]
    print(f"\nSaved {n} decision candidates to {args.out} "
          f"({os.path.getsize(args.out) / 1e6:.0f} MB)")
    print(f"  base survival rate: {ds['label'].mean():.1%}   "
          f"sources: {sorted(set(ds['source'].tolist()))}")


# ===========================================================================
# TRAIN
# ===========================================================================
def auc(scores, labels):
    """Mann-Whitney AUC (ties broken by sort order — fine at this precision)."""
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.double)
    ranks[order] = torch.arange(len(scores), dtype=torch.double)
    pos = labels.bool()
    n1, n0 = pos.sum().item(), (~pos).sum().item()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return ((ranks[pos].sum().item() - n1 * (n1 - 1) / 2) / (n0 * n1))


def fit_probe(X_tr, y_tr, X_va, hidden_dim=0, epochs=300, lr=1e-2, device="cpu", seed=0):
    """Logistic regression (hidden_dim=0) or 1-hidden-layer MLP, full-batch Adam.
    Returns val scores."""
    torch.manual_seed(seed)
    X_tr, y_tr, X_va = X_tr.to(device), y_tr.to(device), X_va.to(device)
    d = X_tr.shape[1]
    if hidden_dim > 0:
        net = torch.nn.Sequential(torch.nn.Linear(d, hidden_dim), torch.nn.ReLU(),
                                  torch.nn.Linear(hidden_dim, 1)).to(device)
    else:
        net = torch.nn.Linear(d, 1).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    # class weighting: survivors dominate; upweight the rare negative class
    pos_frac = y_tr.mean().clamp(1e-4, 1 - 1e-4)
    w = torch.where(y_tr > 0.5, 1.0 / pos_frac, 1.0 / (1 - pos_frac))
    for _ in range(epochs):
        opt.zero_grad()
        out = net(X_tr).squeeze(-1)
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(
            out, y_tr, reduction="none") * w).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return net(X_va).squeeze(-1).cpu()


def topk_sim(state_ids, scores_dict, labels, ks=(1, 2, 4, 8)):
    """Per decision-state (one small step of one block): rank candidates by each
    scorer, take top-k, measure the wrong fraction among selected. This is the
    offline stand-in for 'unmask k per step with this policy'."""
    groups = {}
    for i, sid in enumerate(state_ids.tolist()):
        groups.setdefault(sid, []).append(i)
    out = {name: {k: [0, 0] for k in ks} for name in scores_dict}   # [wrong, total]
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        idx_t = torch.tensor(idxs)
        for name, sc in scores_dict.items():
            order = sc[idx_t].argsort(descending=True)
            for k in ks:
                if len(idxs) < k:
                    continue
                sel = idx_t[order[:k]]
                out[name][k][0] += int((labels[sel] == 0).sum())
                out[name][k][1] += k
    return out


def cmd_train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = torch.load(args.data, map_location="cpu", weights_only=False)
    feat, hidden, label = ds["feat"], ds["hidden"].float(), ds["label"]
    n = label.shape[0]
    print(f"{n} candidates from {ds['prompt'].max().item() + 1} prompts | "
          f"survival rate {label.mean():.1%} | sources {sorted(set(ds['source'].tolist()))}")

    # split by PROMPT to avoid leakage
    n_prompts = ds["prompt"].max().item() + 1
    torch.manual_seed(args.seed)
    perm = torch.randperm(n_prompts)
    val_prompts = set(perm[:max(1, int(n_prompts * args.val_frac))].tolist())
    va = torch.tensor([p.item() in val_prompts for p in ds["prompt"]])
    tr = ~va
    print(f"train {tr.sum()} rows / val {va.sum()} rows "
          f"({len(val_prompts)}/{n_prompts} prompts held out)")

    # standardize on train stats
    mu, sd = feat[tr].mean(0), feat[tr].std(0).clamp(min=1e-6)
    Xc = (feat - mu) / sd
    hmu, hsd = hidden[tr].mean(0), hidden[tr].std(0).clamp(min=1e-6)
    Xh = (hidden - hmu) / hsd
    y = label

    scores = {"raw_p": feat[va][:, 0].clone()}
    print("fitting probes (cheap / hidden / combined)...", flush=True)
    scores["cheap"] = fit_probe(Xc[tr], y[tr], Xc[va], epochs=args.epochs,
                                lr=1e-2, device=device, seed=args.seed)
    scores["hidden"] = fit_probe(Xh[tr], y[tr], Xh[va], epochs=args.epochs,
                                 lr=1e-3, device=device, seed=args.seed)
    Xb = torch.cat([Xc, Xh], dim=1)
    scores["combined"] = fit_probe(Xb[tr], y[tr], Xb[va],
                                   hidden_dim=args.mlp, epochs=args.epochs,
                                   lr=1e-3, device=device, seed=args.seed)

    y_va = y[va]
    draft = (ds["source"][va] < args.ref_threshold_cutoff)   # draft-trajectory states
    print(f"\nVal AUC — 'candidate token will match the dense reference'")
    print(f"  {'scorer':>9} {'overall':>8} {'draft-only':>10}")
    for name, sc in scores.items():
        print(f"  {name:>9} {auc(sc, y_va):>8.3f} {auc(sc[draft], y_va[draft]):>10.3f}")

    # per-state top-k policy simulation on val
    sid = (ds["prompt"][va] * 100000 + ds["block"][va] * 1000
           + ds["step"][va] * 10 + (ds["source"][va] * 7).long())
    sim = topk_sim(sid, scores, y_va)
    ks = (1, 2, 4, 8)
    print(f"\nPolicy simulation — wrong fraction among top-k unmasked per step "
          f"(lower is better; base wrong rate {1 - y_va.mean():.1%}):")
    print(f"  {'scorer':>9} " + " ".join(f"{'k=' + str(k):>8}" for k in ks))
    for name in scores:
        cells = []
        for k in ks:
            wrong, total = sim[name][k]
            cells.append(f"{wrong / total:>8.2%}" if total else f"{'—':>8}")
        print(f"  {name:>9} " + " ".join(cells))

    print("\nHow to read: if hidden/combined clearly beat raw_p (AUC and the k=2-8 rows),")
    print("a learned unmasking policy on the frozen backbone is real — next step is wiring")
    print("the probe into the sampler as the acceptance rule. If raw_p is on par with the")
    print("best probe, decode-time state has no better signal and the inference-only axis")
    print("is closed on this checkpoint.")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="run teacher-forced replay and capture decisions")
    c.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    c.add_argument("--device", default="cuda")
    c.add_argument("--dataset", default="gsm8k")
    c.add_argument("--n_examples", type=int, default=60)
    c.add_argument("--max_new_tokens", type=int, default=256)
    c.add_argument("--ref_threshold", type=float, default=0.95)
    c.add_argument("--draft_thresholds", type=float, nargs="+", default=[0.6])
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", default="v2/sparsity/policy_probe_data.pt")

    t = sub.add_parser("train", help="fit probes and compare against raw confidence")
    t.add_argument("--data", default="v2/sparsity/policy_probe_data.pt")
    t.add_argument("--val_frac", type=float, default=0.25)
    t.add_argument("--epochs", type=int, default=400)
    t.add_argument("--mlp", type=int, default=0,
                   help="hidden width for the combined probe (0 = logistic)")
    t.add_argument("--ref_threshold_cutoff", type=float, default=0.9,
                   help="source thresholds below this count as draft-trajectory states")
    t.add_argument("--seed", type=int, default=0)

    args = p.parse_args()
    if args.cmd == "collect":
        cmd_collect(args)
    else:
        cmd_train(args)


if __name__ == "__main__":
    main()
