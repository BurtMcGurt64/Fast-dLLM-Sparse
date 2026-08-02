"""
COMMIT-VERIFY DE-RISK PROBE — does the free dense commit forward detect draft errors?

The proposed mechanism ("commit-verified aggressive decoding") runs small steps cheap
(low threshold => fewer denoising steps; optionally top-K KV) and uses the commit step
— a dense forward that already exists in the loop — to verify the block: re-mask and
repair positions where the dense logits disagree. Whether this works hinges on numbers
we can measure OFFLINE, before building the repair loop:

  1. AGREEMENT   How often do aggressive drafts match what dense decoding produces?
  2. DETECTION   When they don't match, do verifier logits FLAG the bad positions
                 (recall) without flagging good ones (false alarms)? Two verifiers:
                 - IN-CONTEXT: the commit forward's own logits. Free, but bidirectional
                   in-block attention lets the hidden state at i-1 see token i, so it
                   self-confirms drafts (measured near-blind: R~=FPR).
                 - HELD-OUT: two complementary-mask forwards (the exact training
                   scheme) so every generated position is predicted WITHOUT seeing
                   itself. Costs +2 forwards/block, charged in the NFE estimate.
  3. NFE         Forwards saved by the draft, minus verify cost and a repair estimate.

Method (teacher-forced block replay, batch=1):
  prefill dense; then per 32-token block:
    - decode the block at the REFERENCE config (threshold 0.95, dense KV)
    - decode the SAME block from the SAME prefix at each aggressive draft config
      (small steps never write the prefix cache, so no snapshot needed)
    - dense commit-style scoring forward on each draft (no cache update) -> in-context
      verifier view; two complementary-mask forwards -> held-out verifier view
    - commit the REFERENCE block so the trajectory stays dense (ground truth)
  Metrics only count positions this block actually generated (masked in x_init),
  truncated at the stop token.

How to read the result:
  - agreement high + held-out recall high + false alarms near the dense held-out
    floor -> mechanism is alive: verified-NFE cut approximates the printed estimate.
  - agreement low OR the held-out verifier can't see the mismatches either -> the
    verification family is dead in all cheap forms; one day spent, not a month.

Run (GPU box):
  CUDA_VISIBLE_DEVICES=0 uv run v2/sparsity/verify_probe.py \
      --dataset gsm8k --n_examples 30 --draft_thresholds 0.9 0.75 0.6 0.5
  # add top-K KV sparsity to the drafts (long-context runs; gsm8k barely exercises it):
  CUDA_VISIBLE_DEVICES=0 uv run v2/sparsity/verify_probe.py --frac 0.25
  # drafts without sub-block serialization (min 1 step/block instead of 4):
  CUDA_VISIBLE_DEVICES=0 uv run v2/sparsity/verify_probe.py --draft_small_block_size 32

Companion sweep: quality-vs-small_block_size for the STANDARD sampler lives in the
stage-1 harness (`nfe_probe_stage1.py --small_block_sizes 8 16 32`).
"""
import sys
import os
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))                 # v2/
sys.path.insert(0, os.path.join(HERE, "../experiments"))     # helper (load_model + cache patches)
sys.path.insert(0, os.path.join(HERE, "../train_scripts"))   # nfe_probe_stage1 (EVALS)

import json
import time
import argparse
import torch

from transformers import AutoTokenizer
from helper import load_model
from nfe_probe_stage1 import EVALS
from topk import TopKBlockLoader

BLOCK_SIZE = 32
SMALL_BLOCK = 8
STOP_TOKEN = 151645
MASK_ID = 151665
PROB_TAUS = [0.3, 0.5, 0.7, 0.9]   # candidate acceptance thresholds p(tok) < tau => flag
VERIFY_FORWARDS = 2                 # held-out verification cost per block


# ---------------------------------------------------------------------------
# Top-K arm/disarm helpers. The loader arms layers at each commit; the reference
# decode and all verifier forwards must run DENSE, drafts run with the selection
# from the previous commit (exactly what the real mechanism would see).
# ---------------------------------------------------------------------------
def armed_state(model):
    a = model.model.layers[0].self_attn
    tok = getattr(a, '_topk_token_indices', None)
    blk = getattr(a, '_topk_block_ids', None)
    return (tok, blk) if tok is not None else None


def arm(model, state):
    for layer in model.model.layers:
        layer.self_attn._topk_token_indices = state[0]
        layer.self_attn._topk_block_ids = state[1]


def disarm(model):
    for layer in model.model.layers:
        layer.self_attn._topk_token_indices = None
        layer.self_attn._topk_block_ids = None


# ---------------------------------------------------------------------------
# Core: denoise one block from a fixed prefix. Mirrors batch_sample's inner loop
# (sub-block iteration, block cache, logit shift, threshold unmasking) at B=1.
# Never writes the prefix cache. temperature=0 so runs are deterministic and
# draft-vs-reference token comparison is meaningful.
# ---------------------------------------------------------------------------
@torch.no_grad()
def decode_block(model, x_block, past_key_values, threshold,
                 small_block_size=SMALL_BLOCK, block_size=BLOCK_SIZE, mask_id=MASK_ID):
    x_t = x_block.clone()
    block_past_key_values = None
    steps = 0
    for sb in range(block_size // small_block_size):
        s, e = sb * small_block_size, (sb + 1) * small_block_size
        while True:
            mask_idx = (x_t == mask_id)
            if mask_idx[:, s:e].sum() == 0:
                break
            if block_past_key_values is None or (x_t[:, s] == mask_id).any():
                out = model.forward(input_ids=x_t, use_cache=True, past_key_values=past_key_values,
                                    update_past_key_values=False, use_block_cache=True)
                logits, block_past_key_values = out.logits, out.block_past_key_values
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)[:, s:e]
            else:
                logits = model.forward(input_ids=x_t[:, s:e], use_cache=True,
                                       past_key_values=past_key_values,
                                       update_past_key_values=False, use_block_cache=True,
                                       block_past_key_values=block_past_key_values,
                                       replace_position=s).logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            steps += 1
            x_1, p_1t = model.sample_with_top_p(logits, temperature=0.0)
            x1_p = torch.gather(p_1t, -1, x_1.unsqueeze(-1)).squeeze(-1)
            x1_p = torch.where(mask_idx[:, s:e], x1_p, -torch.inf)
            unmask = x1_p > threshold
            unmask[torch.arange(x_1.shape[0]), x1_p.argmax(dim=-1)] = True
            unmask = unmask & mask_idx[:, s:e]
            x_t[:, s:e][unmask] = x_1[unmask]
    return x_t, steps


def shifted_probs(logits, tokens):
    """Their logit convention: shifted[i] = raw[i-1] predicts token[i]. Returns
    p(token[i]) and argmax at each position. Position 0 is never a generated
    position (always the previous commit's next-token or a prompt token)."""
    shifted = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    probs = torch.softmax(shifted.float(), dim=-1)
    p_tok = torch.gather(probs, -1, tokens.unsqueeze(-1)).squeeze(-1)[0]   # [32]
    argmax = probs.argmax(dim=-1)[0]                                      # [32]
    return p_tok, argmax


@torch.no_grad()
def dense_rescore(model, block_tokens, past_key_values):
    """Commit-style dense forward WITHOUT cache update — the in-context verifier's
    view of a draft block. Same logits the real commit forward would produce for
    free. Bidirectional in-block attention means each position can see itself."""
    logits = model.forward(input_ids=block_tokens, use_cache=True,
                           past_key_values=past_key_values,
                           update_past_key_values=False).logits
    return shifted_probs(logits, block_tokens)


@torch.no_grad()
def heldout_rescore(model, block_tokens, past_key_values, gen_mask):
    """Held-out verifier: two complementary-mask forwards (the training scheme) so
    each generated position is predicted WITHOUT seeing itself. Deterministic
    interleaved 50/50 split of the generated positions; non-generated positions
    (prompt remainder / carried next-token) stay visible in both forwards."""
    gen_idx = gen_mask.nonzero(as_tuple=False).squeeze(-1)
    p_tok = torch.zeros(BLOCK_SIZE, device=block_tokens.device)
    argmax = torch.full((BLOCK_SIZE,), -1, dtype=torch.long, device=block_tokens.device)
    for half in (gen_idx[0::2], gen_idx[1::2]):
        if len(half) == 0:
            continue
        x = block_tokens.clone()
        x[0, half] = MASK_ID
        logits = model.forward(input_ids=x, use_cache=True,
                               past_key_values=past_key_values,
                               update_past_key_values=False).logits
        p, a = shifted_probs(logits, block_tokens)
        p_tok[half] = p[half]
        argmax[half] = a[half]
    return p_tok, argmax


# ---------------------------------------------------------------------------
# Per-prompt probe loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_prompt(model, ids, cfgs, ref_threshold, max_new_tokens, use_topk,
               draft_small_block_size=SMALL_BLOCK):
    device = ids.device
    L = ids.shape[1]
    assert L > BLOCK_SIZE, "prompt shorter than one block not supported"
    prefill_len = L // BLOCK_SIZE * BLOCK_SIZE

    out = model.forward(input_ids=ids[:, :prefill_len], use_cache=True,
                        update_past_key_values=True, block_size=BLOCK_SIZE, logits_to_keep=1)
    pkv = out.past_key_values
    masks = lambda n: torch.full((1, n), MASK_ID, dtype=torch.long, device=device)
    if L % BLOCK_SIZE == 0:
        nxt = out.logits[:, -1:, :].argmax(-1)
        cur = torch.cat([nxt, masks(BLOCK_SIZE - 1)], dim=1)
    else:
        cur = torch.cat([ids[:, prefill_len:], masks(BLOCK_SIZE - (L - prefill_len))], dim=1)

    records = []
    ref_blocks = []
    for block_idx in range(max_new_tokens // BLOCK_SIZE):
        x_init = cur
        gen_mask = (x_init[0] == MASK_ID)   # positions this block actually generates

        saved = armed_state(model)
        disarm(model)
        ref_block, ref_steps = decode_block(model, x_init, pkv, ref_threshold)

        drafts = {}
        for cfg in cfgs:
            if use_topk and saved is not None:
                arm(model, saved)
            draft_block, draft_steps = decode_block(model, x_init, pkv, cfg,
                                                    small_block_size=draft_small_block_size)
            disarm(model)
            p_tok, argmax = dense_rescore(model, draft_block, pkv)
            ho_p, ho_argmax = heldout_rescore(model, draft_block, pkv, gen_mask)
            drafts[cfg] = (draft_block, draft_steps, p_tok, argmax, ho_p, ho_argmax)

        # held-out floor on the DENSE tokens: how often a masked forward disagrees
        # with what dense decoding itself produced — any acceptance rule must
        # tolerate at least this much flagging on good tokens
        disarm(model)
        ref_ho_p, ref_ho_argmax = heldout_rescore(model, ref_block, pkv, gen_mask)

        # commit the REFERENCE block (dense trajectory is the ground truth);
        # its logits give the dense self-consistency baseline for the acceptance rule
        out = model.forward(input_ids=ref_block, use_cache=True, past_key_values=pkv,
                            update_past_key_values=True, block_size=BLOCK_SIZE)
        pkv = out.past_key_values
        ref_p, ref_argmax = shifted_probs(out.logits, ref_block)
        nxt = out.logits[:, -1:, :].argmax(-1)

        # valid positions: generated by this block, up to and including the stop token
        valid = gen_mask.clone()
        stop_hit = False
        stops = ((ref_block[0] == STOP_TOKEN) & gen_mask).nonzero(as_tuple=False)
        if len(stops) > 0:
            stop_hit = True
            valid[stops[0].item() + 1:] = False

        vidx = valid.nonzero(as_tuple=False).squeeze(-1)
        block_rec = {
            "block_idx": block_idx,
            "n_valid": int(len(vidx)),
            "ref_steps": ref_steps,
            "ref_selfcons_mismatch": int((ref_argmax[vidx] != ref_block[0, vidx]).sum()),
            "ref_ho_mismatch": int((ref_ho_argmax[vidx] != ref_block[0, vidx]).sum()),
            "ref_p_tok": ref_p[vidx].tolist(),
            "ref_ho_p": ref_ho_p[vidx].tolist(),
            "cfgs": {},
        }
        for cfg, (draft_block, draft_steps, p_tok, argmax, ho_p, ho_argmax) in drafts.items():
            mm = (draft_block[0, vidx] != ref_block[0, vidx])
            block_rec["cfgs"][cfg] = {
                "draft_steps": draft_steps,
                "mismatch": mm.tolist(),
                "argmax_flag": (argmax[vidx] != draft_block[0, vidx]).tolist(),
                "p_tok": p_tok[vidx].tolist(),
                "ho_flag": (ho_argmax[vidx] != draft_block[0, vidx]).tolist(),
                "ho_p": ho_p[vidx].tolist(),
                "subblock": (vidx // SMALL_BLOCK).tolist(),
            }
        records.append(block_rec)
        ref_blocks.append(ref_block)

        cur = torch.cat([nxt, masks(BLOCK_SIZE - 1)], dim=1)
        if stop_hit:
            break

    full = torch.cat([ids[:, :prefill_len]] + ref_blocks, dim=1)
    return records, full


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def rate(num, den):
    return num / den if den else float("nan")


def summarize(all_records, cfgs, repair_small_steps, repair_commits):
    n_valid = sum(r["n_valid"] for r in all_records)
    ref_selfcons = sum(r["ref_selfcons_mismatch"] for r in all_records)
    ref_ho = sum(r["ref_ho_mismatch"] for r in all_records)
    ref_steps = sum(r["ref_steps"] for r in all_records)
    n_blocks = len(all_records)

    print(f"\nDense reference baseline over {n_blocks} blocks / {n_valid} generated tokens:")
    print(f"  in-context disagreement with dense tokens (self-confirming floor): "
          f"{rate(ref_selfcons, n_valid):.1%}")
    print(f"  HELD-OUT   disagreement with dense tokens (acceptance-rule floor):  "
          f"{rate(ref_ho, n_valid):.1%}")
    print(f"  small steps per block: {rate(ref_steps, n_blocks):.1f}   "
          f"forwards total (steps+commits): {ref_steps + n_blocks}")

    # single stats pass per config over both detectors
    summary = {}
    for cfg in cfgs:
        st = {
            "match": 0, "mm_total": 0,
            "ctx_mm_flagged": 0, "ctx_ok_flagged": 0,
            "ho_mm_flagged": 0, "ho_ok_flagged": 0,
            "draft_steps": 0, "clean_blocks": 0, "ho_flagged_blocks": 0,
            "tau": {t: [0, 0] for t in PROB_TAUS},   # held-out p: [mm_flagged, ok_flagged]
        }
        for r in all_records:
            c = r["cfgs"][cfg]
            st["draft_steps"] += c["draft_steps"]
            block_mm = block_flag = False
            for mm, ctx_fl, ho_fl, ho_p in zip(c["mismatch"], c["argmax_flag"],
                                               c["ho_flag"], c["ho_p"]):
                if mm:
                    st["mm_total"] += 1
                    st["ctx_mm_flagged"] += int(ctx_fl)
                    st["ho_mm_flagged"] += int(ho_fl)
                    block_mm = True
                else:
                    st["match"] += 1
                    st["ctx_ok_flagged"] += int(ctx_fl)
                    st["ho_ok_flagged"] += int(ho_fl)
                block_flag = block_flag or ho_fl
                for t in PROB_TAUS:
                    if ho_p < t:
                        st["tau"][t][0 if mm else 1] += 1
            st["clean_blocks"] += int(not block_mm)
            st["ho_flagged_blocks"] += int(block_flag)

        total = st["mm_total"] + st["match"]
        repair = st["ho_flagged_blocks"] * (repair_small_steps + repair_commits)
        draft_forwards = (st["draft_steps"] + n_blocks
                          + VERIFY_FORWARDS * n_blocks + repair)
        summary[cfg] = {
            "tok_agree": rate(st["match"], total),
            "blk_clean": rate(st["clean_blocks"], n_blocks),
            "steps_per_blk": rate(st["draft_steps"], n_blocks),
            "ctx_recall": rate(st["ctx_mm_flagged"], st["mm_total"]),
            "ctx_fpr": rate(st["ctx_ok_flagged"], st["match"]),
            "ho_recall": rate(st["ho_mm_flagged"], st["mm_total"]),
            "ho_fpr": rate(st["ho_ok_flagged"], st["match"]),
            "ho_missed": rate(st["mm_total"] - st["ho_mm_flagged"], total),
            "ho_flagged_blk": rate(st["ho_flagged_blocks"], n_blocks),
            "nfe_cut": rate(ref_steps + n_blocks, draft_forwards),
            "tau": {t: {"recall": rate(v[0], st["mm_total"]),
                        "fpr": rate(v[1], st["match"])}
                    for t, v in st["tau"].items()},
        }

    print(f"\nDrafting (agreement with the dense reference):")
    print(f"  {'draft':>8} {'tok-agree':>9} {'blk-clean':>9} {'steps/blk':>9}")
    for cfg in cfgs:
        s = summary[cfg]
        print(f"  {cfg:>8} {s['tok_agree']:>9.1%} {s['blk_clean']:>9.1%} "
              f"{s['steps_per_blk']:>9.1f}")

    print(f"\nVerifiers (flag = verifier argmax != draft token). NFE charges "
          f"+{VERIFY_FORWARDS} verify forwards/block and, per HELD-OUT-flagged block, "
          f"+{repair_small_steps} small steps +{repair_commits} commit:")
    print(f"  {'draft':>8} {'ctx-R':>6} {'ctx-FPR':>7} {'ho-R':>6} {'ho-FPR':>7} "
          f"{'miss%':>6} {'flag-blk':>8} {'NFE-cut':>7}")
    for cfg in cfgs:
        s = summary[cfg]
        print(f"  {cfg:>8} {s['ctx_recall']:>6.1%} {s['ctx_fpr']:>7.1%} "
              f"{s['ho_recall']:>6.1%} {s['ho_fpr']:>7.1%} {s['ho_missed']:>6.2%} "
              f"{s['ho_flagged_blk']:>8.1%} {s['nfe_cut']:>7.2f}x")

    print("\n  ctx = in-context commit logits (self-confirming; expected near-blind).")
    print("  ho  = held-out complementary-mask verifier. miss% = mismatches the held-out")
    print("  verifier does NOT flag, per generated token (residual quality risk).")
    print("  flag-blk = blocks the held-out verifier sends to repair (true + false alarms).")

    print("\n  Held-out probability detectors (flag if ho_p(draft tok) < tau):")
    for cfg in cfgs:
        parts = "  ".join(f"tau={t}: R={summary[cfg]['tau'][t]['recall']:.0%}/"
                          f"FPR={summary[cfg]['tau'][t]['fpr']:.0%}" for t in PROB_TAUS)
        print(f"    {cfg:>8}  {parts}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dataset", choices=list(EVALS), default="gsm8k")
    p.add_argument("--n_examples", type=int, default=30)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--ref_threshold", type=float, default=0.95)
    p.add_argument("--draft_thresholds", type=float, nargs="+", default=[0.9, 0.75, 0.6, 0.5])
    p.add_argument("--draft_small_block_size", type=int, default=SMALL_BLOCK,
                   help="sub-block size for DRAFT decoding (reference stays at 8). "
                        "32 removes the 4-sub-block serialization floor entirely.")
    p.add_argument("--frac", type=float, default=None,
                   help="if set, drafts run with TopKBlockLoader(frac=...) armed from the "
                        "previous commit (reference + verifier always dense)")
    p.add_argument("--n_sink", type=int, default=2)
    p.add_argument("--n_recency", type=int, default=2)
    p.add_argument("--repair_small_steps", type=int, default=4,
                   help="assumed extra small steps per FLAGGED block in the NFE estimate")
    p.add_argument("--repair_commits", type=int, default=1,
                   help="assumed extra commit forwards per flagged block")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_json", default=None, help="dump per-block records for later analysis")
    args = p.parse_args()

    print(f"Loading model + {args.dataset}...", flush=True)
    model = load_model(args.model, device=args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    ev = EVALS[args.dataset]
    data = ev["load"](args.n_examples, args.seed)
    cfgs = [f"th={t:g}" for t in args.draft_thresholds]
    cfg_thresholds = dict(zip(cfgs, args.draft_thresholds))

    loader = None
    if args.frac is not None:
        loader = TopKBlockLoader(model, frac=args.frac,
                                 n_sink=args.n_sink, n_recency=args.n_recency).install()
        print(f"Top-K armed for drafts: frac={args.frac} sink={args.n_sink} "
              f"recency={args.n_recency}", flush=True)
    if args.draft_small_block_size != SMALL_BLOCK:
        print(f"Draft sub-block size: {args.draft_small_block_size} "
              f"(reference stays at {SMALL_BLOCK})", flush=True)

    all_records = []
    correct = 0
    t0 = time.perf_counter()
    try:
        for i, (q, gold) in enumerate(data):
            ids = tokenizer(ev["prompt"](q), return_tensors="pt").input_ids.to(args.device)
            recs, full = run_prompt(
                model, ids,
                cfgs=[cfg_thresholds[c] for c in cfgs],
                ref_threshold=args.ref_threshold,
                max_new_tokens=args.max_new_tokens,
                use_topk=loader is not None,
                draft_small_block_size=args.draft_small_block_size,
            )
            # decode_block returns cfgs keyed by threshold value; rekey to names
            for r in recs:
                r["cfgs"] = {name: r["cfgs"][cfg_thresholds[name]] for name in cfgs}
                r["prompt_idx"] = i
            all_records.extend(recs)
            gen = tokenizer.decode(full[0, ids.shape[1]:], skip_special_tokens=True)
            pred = ev["parse_pred"](gen)
            ok = pred is not None and pred == gold
            correct += int(ok)
            print(f"  [{i + 1}/{len(data)}] blocks={len(recs)} "
                  f"ref={'OK' if ok else 'WRONG'} "
                  f"({(time.perf_counter() - t0) / (i + 1):.1f}s/ex)", flush=True)
    finally:
        if loader is not None:
            loader.uninstall()

    print(f"\nDense reference accuracy: {correct}/{len(data)} = {correct / len(data):.1%} "
          f"(sanity check vs the stage-1 sweep)")
    summary = summarize(all_records, cfgs, args.repair_small_steps, args.repair_commits)

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump({"args": vars(args), "summary": summary, "records": all_records}, f)
        print(f"\nPer-block records saved to {args.save_json}")

    print("\nDecision guide: the mechanism is ALIVE if some config shows NFE-cut >= ~1.5x with")
    print("ho-R high enough that miss% stays near the dense held-out floor. If mismatches are")
    print("frequent AND even the held-out verifier can't see them, the verification family is")
    print("dead on this checkpoint — that is decision-grade evidence for the training axis.")


if __name__ == "__main__":
    main()
