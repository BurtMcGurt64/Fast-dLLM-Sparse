"""
BATCHED LONG-GENERATION experiment — the measurement that decides whether the
honest ~2x regime is real on this checkpoint.

Motivation (see CLAUDE.md + WHY_BATCH_CONTEXT_DONT_HELP.md):
  Every prior e2e sweep was in the WRONG corner —
    * derisk_genlen / gen-length sweeps -> 8k context (long PROMPT) -> prefill-heavy
      -> e2e diluted to ~1.3x.
    * runtime_breakdown gsm8k/math       -> short prompt (prefill <3%) BUT B=1
      -> small step barely speeds up (~1.04x) -> e2e ~1.05x.
  The intersection [short prompt + long generation + real batch] was never
  measured. That corner is where prefill is negligible AND weights are amortized
  (per-step -> ~2.1x at B=32), so e2e should approach the per-step number.

This script measures that corner two ways, each with the batch size that is
actually VALID for it:

  --mode speedup   (batched, uniform prompts):
      Short uniform prompts via load_longbench_subset (the KNOWN-GOOD uniform
      batching path; ragged batching corrupts decode, so we do NOT use ragged
      prompts here), sweep generation length, dense vs recency-pinned-sparse
      wall-clock speedup + peak mem + sparse repetition score. This is the
      speedup/coherence number.

  --mode accuracy  (B=1, ragged reasoning tasks):
      GSM8K / MATH at natural CoT length, dense vs sparse, extract the final
      answer and report the ACCURACY DELTA (the "sparsity doesn't hurt" claim).
      Forced to B=1 because batched decoding corrupts ragged prompts. Accuracy
      is per-sequence invariant, so B=1 is the correct place to measure it.

Both modes use the REAL method: TopKBlockLoader with frac budget + sink/recency
pinning (n_sink, n_recency), not plain top-K.

Run:
  # 1) speedup + coherence in the short-prompt long-gen batched corner
  CUDA_VISIBLE_DEVICES=0 uv run v2/sparsity/batched_long_gen.py --mode speedup \
      --subset narrativeqa --context_len 512 --batch_size 32 \
      --gen_lens 256 512 1024 2048 --frac 0.2 --n_recency 8 --n_examples 32

  # 2) accuracy delta (dense vs sparse), B=1, natural CoT length
  CUDA_VISIBLE_DEVICES=0 uv run v2/sparsity/batched_long_gen.py --mode accuracy \
      --dataset gsm8k --n_examples 200 --max_new_tokens 256 --frac 0.2 --n_recency 8
  CUDA_VISIBLE_DEVICES=0 uv run v2/sparsity/batched_long_gen.py --mode accuracy \
      --dataset math  --n_examples 200 --max_new_tokens 512 --frac 0.2 --n_recency 8

Read:
  speedup  -> does e2e climb toward ~2x as gen_len grows (with prefill kept small
              by the short context)? and does sparse stay coherent (rep low)?
  accuracy -> is (sparse - dense) accuracy ~0? if yes, the speedup is free.
"""
import sys
import os
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))              # v2/
sys.path.insert(0, os.path.join(HERE, "../experiments"))  # helper
sys.path.insert(0, os.path.join(HERE, "../train_scripts"))  # nfe_probe_stage1 (gsm8k/math evals)

import time
import types
import argparse
import contextlib
import statistics as st
import torch
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

from transformers import AutoTokenizer
from helper import load_model
from generation_functions import Fast_dLLM_QwenForCausalLM as GenClass
from topk import TopKBlockLoader

BLOCK_SIZE = 32
STOP_TOKEN = 151645


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def repetition_score(ids):
    """Fraction of repeated 4-grams. >0.3 => degenerate/looping."""
    if len(ids) < 8:
        return 0.0
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    seen, rep = set(), 0
    for g in grams:
        if g in seen:
            rep += 1
        seen.add(g)
    return rep / len(grams)


@contextlib.contextmanager
def suppress_stop_token(model, stop_token=STOP_TOKEN):
    """Force full max_new_tokens by making the stop token unselectable. Applied
    identically to dense and sparse, so the ratio is unaffected. Use ONLY for
    controlled timing (--force_len); it is not a faithful serving length."""
    orig = model.sample_with_top_p

    def wrapped(logits, *a, **kw):
        logits[..., stop_token] = float("-inf")
        return orig(logits, *a, **kw)

    model.sample_with_top_p = wrapped
    try:
        yield
    finally:
        model.sample_with_top_p = orig


def make_loader(model, frac, k, n_sink, n_recency):
    if frac is not None:
        return TopKBlockLoader(model, frac=frac, block_size=BLOCK_SIZE,
                               n_sink=n_sink, n_recency=n_recency)
    return TopKBlockLoader(model, k=k, block_size=BLOCK_SIZE,
                           n_sink=n_sink, n_recency=n_recency)


def build_sampler(model):
    return types.MethodType(GenClass.batch_sample, model)


# --------------------------------------------------------------------------- #
# MODE 1 — batched speedup + coherence (short prompt, long gen, uniform batch)
# --------------------------------------------------------------------------- #
def run_speedup(model, tokenizer, args, device):
    from eval_longbench_sweep import load_longbench_subset

    sampler = build_sampler(model)
    print(f"Loading uniform batches: subset={args.subset} ctx={args.context_len} "
          f"B={args.batch_size} n={args.n_examples}", flush=True)
    batches = load_longbench_subset(
        args.subset, tokenizer, device, args.context_len,
        args.batch_size, args.n_examples, args.seed)

    def time_condition(loader, gen_len):
        """Returns (wall_per_example, avg_gen_tokens, frac, peak_gb, mean_rep)."""
        if loader is not None:
            loader.install()
        reps, total, total_gen = [], 0, 0
        torch.cuda.reset_peak_memory_stats(device)
        stop_ctx = suppress_stop_token(model) if args.force_len else contextlib.nullcontext()
        try:
            with stop_ctx, torch.no_grad():
                # warmup so autotuning doesn't taint the first timed batch
                enc0 = batches[0][0]
                _ = sampler(input_ids=enc0.input_ids, tokenizer=tokenizer,
                            block_size=BLOCK_SIZE, max_new_tokens=gen_len, small_block_size=8,
                            min_len=enc0.input_ids.shape[1],
                            seq_len=torch.full((enc0.input_ids.shape[0],), enc0.input_ids.shape[1],
                                               dtype=torch.long, device=device),
                            threshold=args.threshold, temperature=0.0, use_block_cache=True)
                torch.cuda.synchronize(device)

                t0 = time.perf_counter()
                for encoded, _ans in batches:
                    ids = encoded.input_ids
                    B, plen = ids.shape[0], ids.shape[1]
                    seq_len_t = torch.full((B,), plen, dtype=torch.long, device=device)
                    out = sampler(input_ids=ids, tokenizer=tokenizer, block_size=BLOCK_SIZE,
                                  max_new_tokens=gen_len, small_block_size=8, min_len=plen,
                                  seq_len=seq_len_t, threshold=args.threshold,
                                  temperature=0.0, use_block_cache=True)
                    for b in range(B):
                        gen = out[b][plen:].tolist()
                        if STOP_TOKEN in gen:
                            gen = gen[:gen.index(STOP_TOKEN)]
                        total_gen += len(gen)
                        reps.append(repetition_score(gen))
                        total += 1
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - t0
        finally:
            frac = 0.0
            if loader is not None:
                try:
                    _, _, frac = loader.avg_stats
                except Exception:
                    pass
                loader.uninstall()
        peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
        return (elapsed / total if total else 0.0,
                total_gen / total if total else 0.0,
                frac, peak_gb, st.mean(reps) if reps else 0.0)

    rows = []
    for gen_len in args.gen_lens:
        print(f"\n=== B={args.batch_size} ctx={args.context_len} gen_len={gen_len} "
              f"{'(forced)' if args.force_len else '(natural stop)'} ===", flush=True)
        torch.cuda.empty_cache()
        try:
            d_wall, d_gen, _, d_mem, d_rep = time_condition(None, gen_len)
            print(f"  dense : {d_wall:7.3f} s/ex  gen={d_gen:5.0f} tok  "
                  f"rep={d_rep:.2f}  peak={d_mem:.1f} GB", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); print("  dense : OOM", flush=True); continue

        loader = make_loader(model, args.frac, args.k, args.n_sink, args.n_recency)
        try:
            s_wall, s_gen, frac, s_mem, s_rep = time_condition(loader, gen_len)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); print("  sparse: OOM", flush=True); continue

        sp = d_wall / s_wall if s_wall else 0.0
        deg = "  <-- SPARSE DEGENERATE" if s_rep > 0.3 else ""
        print(f"  sparse: {s_wall:7.3f} s/ex  gen={s_gen:5.0f} tok  rep={s_rep:.2f}  "
              f"frac={frac:.1%}  peak={s_mem:.1f} GB  -> {sp:.2f}x{deg}", flush=True)
        rows.append((gen_len, d_wall, s_wall, sp, frac, d_gen, s_gen, d_rep, s_rep))

    print("\n" + "=" * 92)
    print(f"SPEEDUP [{args.subset}]  B={args.batch_size}  ctx={args.context_len}  "
          f"frac={args.frac}  sink={args.n_sink}  recency={args.n_recency}")
    print("=" * 92)
    print(f"  {'gen_len':>7}  {'dense s/ex':>11}  {'sparse s/ex':>12}  {'speedup':>8}  "
          f"{'frac':>6}  {'s_gen':>6}  {'d_rep':>6}  {'s_rep':>6}")
    print("  " + "-" * 84)
    for (gl, dw, sw, sp, fr, dg, sg, dr, sr) in rows:
        print(f"  {gl:>7}  {dw:>11.3f}  {sw:>12.3f}  {sp:>7.2f}x  {fr:>6.1%}  "
              f"{sg:>6.0f}  {dr:>6.2f}  {sr:>6.2f}")
    if len(rows) >= 2:
        # decode-only speedup: difference out fixed prefill via the gen_len slope
        (glo, dlo, slo, *_), (ghi, dhi, shi, *_) = rows[0], rows[-1]
        dec = (dhi - dlo) / (shi - slo) if (shi - slo) > 0 else 0.0
        print(f"\n  decode-only speedup (prefill removed via gen_len slope): {dec:.2f}x")
    print("\n  Read: e2e speedup should CLIMB with gen_len (prefill kept small by short ctx)")
    print("        and plateau at the decode-only number; s_rep low = recency fix holds.")


# --------------------------------------------------------------------------- #
# MODE 2 — accuracy delta (dense vs sparse), B=1, ragged reasoning tasks
# --------------------------------------------------------------------------- #
def run_accuracy(model, tokenizer, args, device):
    from nfe_probe_stage1 import EVALS, build_batches

    if args.batch_size != 1:
        print("WARNING: batched decoding corrupts ragged reasoning prompts -> accuracy\n"
              "         is INVALID at batch>1. Forcing batch_size=1 for a valid delta.\n", flush=True)
        args.batch_size = 1

    ev = EVALS[args.dataset]
    data = ev["load"](args.n_examples, args.seed)
    prompts = [ev["prompt"](q) for q, _ in data]
    golds = [g for _, g in data]
    batches = build_batches(prompts, golds, tokenizer, device, args.batch_size)
    parse_pred = ev["parse_pred"]
    sampler = build_sampler(model)

    def evaluate(loader):
        if loader is not None:
            loader.install()
        correct, total, gen_toks, reps = 0, 0, [], []
        try:
            with torch.no_grad():
                for (ids, seq_len, min_len, true_lens), gchunk in batches:
                    out = sampler(input_ids=ids, tokenizer=tokenizer, block_size=BLOCK_SIZE,
                                  max_new_tokens=args.max_new_tokens, small_block_size=8,
                                  min_len=min_len, seq_len=seq_len, threshold=args.threshold,
                                  temperature=0.0, use_block_cache=True)
                    for b in range(len(true_lens)):
                        gen_ids = out[b][true_lens[b]:].tolist()
                        if STOP_TOKEN in gen_ids:
                            gen_ids = gen_ids[:gen_ids.index(STOP_TOKEN)]
                        gen = tokenizer.decode(gen_ids, skip_special_tokens=True)
                        pred = parse_pred(gen)
                        correct += int(pred is not None and pred == gchunk[b])
                        total += 1
                        gen_toks.append(len(gen_ids))
                        reps.append(repetition_score(gen_ids))
        finally:
            if loader is not None:
                loader.uninstall()
        return (correct / total if total else 0.0,
                st.mean(gen_toks) if gen_toks else 0.0,
                st.mean(reps) if reps else 0.0)

    print(f"\n{args.dataset}  n={len(data)}  B=1  max_new={args.max_new_tokens}  "
          f"frac={args.frac}  sink={args.n_sink}  recency={args.n_recency}\n", flush=True)

    print("Running DENSE...", flush=True)
    d_acc, d_gen, d_rep = evaluate(None)
    print(f"  dense : acc={d_acc:.1%}  gen={d_gen:.0f} tok  rep={d_rep:.2f}", flush=True)

    print("Running SPARSE (recency-pinned top-K)...", flush=True)
    loader = make_loader(model, args.frac, args.k, args.n_sink, args.n_recency)
    s_acc, s_gen, s_rep = evaluate(loader)
    print(f"  sparse: acc={s_acc:.1%}  gen={s_gen:.0f} tok  rep={s_rep:.2f}", flush=True)

    print("\n" + "=" * 60)
    print(f"ACCURACY DELTA [{args.dataset}]  n={len(data)}")
    print("=" * 60)
    print(f"  dense  accuracy : {d_acc:.1%}")
    print(f"  sparse accuracy : {s_acc:.1%}")
    print(f"  delta (sparse - dense): {s_acc - d_acc:+.1%}")
    print(f"  sparse repetition: {s_rep:.2f}  (>0.3 => degeneration)")
    print("\n  Claim supported if delta ~ 0 (sparsity is free at this frac).")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["speedup", "accuracy"], required=True)
    p.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    p.add_argument("--device", default="cuda")
    # selection budget (pass frac OR k) + pinning
    p.add_argument("--frac", type=float, default=0.2, help="dynamic budget = round(frac*n_blocks)")
    p.add_argument("--k", type=int, default=None, help="fixed K (used only if --frac is unset)")
    p.add_argument("--n_sink", type=int, default=2)
    p.add_argument("--n_recency", type=int, default=8, help="always-keep most-recent blocks (coherence)")
    p.add_argument("--threshold", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    # speedup mode
    p.add_argument("--subset", default="narrativeqa", help="uniform-batchable longbench subset")
    p.add_argument("--context_len", type=int, default=512, help="SHORT prompt keeps prefill small")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--gen_lens", type=int, nargs="+", default=[256, 512, 1024, 2048])
    p.add_argument("--n_examples", type=int, default=32)
    p.add_argument("--force_len", action="store_true",
                   help="suppress stop token for controlled timing (NOT a faithful serving length)")
    # accuracy mode
    p.add_argument("--dataset", choices=["gsm8k", "math"], default="gsm8k")
    p.add_argument("--max_new_tokens", type=int, default=256)
    args = p.parse_args()
    if args.frac is not None and args.k is not None:
        args.frac = None if args.k else args.frac  # k wins if explicitly given

    print("Loading model...", flush=True)
    model = load_model(args.model, device=args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if args.mode == "speedup":
        run_speedup(model, tokenizer, args, args.device)
    else:
        run_accuracy(model, tokenizer, args, args.device)


if __name__ == "__main__":
    main()
