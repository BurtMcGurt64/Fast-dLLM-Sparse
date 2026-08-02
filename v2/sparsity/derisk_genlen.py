"""
STEP-0 DE-RISK: does e2e wall-clock speedup climb with generation length?

Thesis: the ~1.2x e2e number is a short-generation artifact, not a method
ceiling. Decode re-reads the 14.4GB weight stack ~17x per 32 generated tokens
while prefill reads it once, so once generation is a few hundred tokens decode
dominates e2e and the e2e speedup converges to the per-step speedup (~1.9x).

This script sweeps generation length and measures the dense-vs-sparse WALL-CLOCK
ratio at each length. It forces every sequence to generate exactly max_new_tokens
by suppressing the stop token (sets its logit to -inf so the sampler can never
pick it), so the batch never shrinks and gen length is controlled. The speedup
ratio is a hardware quantity (bytes read per forward) and does NOT depend on
whether the forced tokens are meaningful — the post-stop filler costs the GPU the
same as real output. Suppression is applied identically to dense and sparse, so it
cancels in the ratio. Quality is NOT measured here (validated elsewhere) — only
wall-clock, so dense vs sparse are compared at identical gen lengths.

This isolates the SYSTEMS question ("does the ratio climb as decode lengthens?")
from the separate task/checkpoint question ("do real workloads generate that
much?" — no, and this instruct checkpoint especially won't emit long CoT).

The e2e ratio rises as generation dilutes the fixed prefill, and asymptotes to the
decode-phase ratio (measured ~1.6x at 8k/B32 with top-K alone; higher as KV
overtakes the weight floor). It does NOT keep climbing without bound.

CONFOUND to control: fixed --k prunes a fixed block count, so as forced generation
grows the pruned *fraction* drifts down (K/N shrinks) and inflates the speedup for
the wrong reason. Use --dynamic_frac to hold the selection fraction constant
(loader budget = round(frac * current_blocks)), so a rising curve reflects prefill
amortization ONLY.

Run (single GPU):
    # clean prefill-amortization curve (fraction held constant):
    CUDA_VISIBLE_DEVICES=0 uv run v2/sparsity/derisk_genlen.py \
        --context_lens 8192 --batch_sizes 16 --dynamic_frac 0.2 \
        --gen_lens 64 256 512 1024 2048 --n_examples 32 --subset narrativeqa

Notes:
    - B=16/8k is the safe default; long gen_lens grow the KV cache and may OOM
      (handled: OOM is caught and that row is skipped) — drop batch or gen_len.
    - --dynamic_frac 0.2 matches the fixed-K=51@8k config but stays 20% as the
      context grows during generation (the correct control for this sweep).
    - --same_path_baseline additionally times a keep-all baseline that pays the same
      commit-step capture cost, isolating pruning from capture overhead.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../experiments"))

import time
import types
import argparse
import contextlib
import torch
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

from transformers import AutoTokenizer

from helper import load_model
from generation_functions import Fast_dLLM_QwenForCausalLM as GenClass
from topk import TopKBlockLoader
from eval_longbench_sweep import load_longbench_subset

BLOCK_SIZE = 32
STOP_TOKEN = 151645


@contextlib.contextmanager
def suppress_stop_token(model, stop_token=STOP_TOKEN):
    """Force generation to run the full max_new_tokens by making the stop token
    unselectable. Applied identically to dense and sparse, so the speedup ratio
    is unaffected; only absolute times shift slightly."""
    orig = model.sample_with_top_p

    def wrapped(logits, *a, **kw):
        logits[..., stop_token] = float("-inf")  # in-place: logits is discarded after
        return orig(logits, *a, **kw)

    model.sample_with_top_p = wrapped
    try:
        yield
    finally:
        model.sample_with_top_p = orig


def time_condition(model, tokenizer, batches, k, gen_len, use_block_cache,
                   device, warmup=True, frac=None, n_sink=2, n_recency=2):
    """Time one (k, gen_len) condition. k=None and frac=None => true dense (no
    loader). frac set => dynamic budget = round(frac * current_blocks), holding the
    selection fraction constant as generation grows (the clean control). n_sink /
    n_recency set the pinned blocks so speed is measured at the SAME pinning the
    coherence run validated (iso-quality speedup).
    Returns (wall_per_example, avg_gen_tokens, frac, peak_gb)."""
    loader = None
    if frac is not None:
        loader = TopKBlockLoader(model, frac=frac, n_sink=n_sink, n_recency=n_recency)
        loader.install()
    elif k is not None:
        loader = TopKBlockLoader(model, k=k)
        loader.install()

    sampler = types.MethodType(GenClass.batch_sample, model)

    def run_batch(encoded):
        input_ids = encoded.input_ids
        B = input_ids.shape[0]
        prompt_len = input_ids.shape[1]
        seq_len_t = torch.full((B,), prompt_len, dtype=torch.long, device=device)
        return sampler(
            input_ids=input_ids,
            tokenizer=tokenizer,
            block_size=BLOCK_SIZE,
            max_new_tokens=gen_len,
            small_block_size=8,
            min_len=prompt_len,
            seq_len=seq_len_t,
            threshold=0.95,
            temperature=0.0,
            use_block_cache=use_block_cache,
        )

    torch.cuda.reset_peak_memory_stats(device)
    total, total_gen = 0, 0

    try:
        with suppress_stop_token(model):
            # warmup once so kernel autotuning / caches don't taint the first timed batch
            if warmup and batches:
                _ = run_batch(batches[0][0])
                torch.cuda.synchronize(device)

            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            for encoded, _answers in batches:
                out = run_batch(encoded)
                B = encoded.input_ids.shape[0]
                prompt_len = encoded.input_ids.shape[1]
                for b in range(B):
                    total_gen += max(0, out[b].shape[0] - prompt_len)
                    total += 1
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - t0
    finally:
        frac = 0.0
        if loader is not None:
            _, _, frac = loader.avg_stats
            loader.uninstall()

    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    wall_per_ex = elapsed / total if total else 0.0
    avg_gen = total_gen / total if total else 0.0
    return wall_per_ex, avg_gen, frac, peak_gb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--subset", type=str, default="narrativeqa")
    p.add_argument("--context_lens", type=int, nargs="+", default=[8192])
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[16])
    p.add_argument("--k", type=int, default=51, help="fixed K (used when --frac is not set)")
    p.add_argument("--frac", type=float, default=None,
                   help="fixed initial K = round(frac*N) at the prompt context (drifts during gen)")
    p.add_argument("--dynamic_frac", type=float, default=None,
                   help="hold selection fraction constant DURING generation via the loader's "
                        "dynamic budget (round(frac*current_blocks)). Use this for the gen-length "
                        "sweep so a rising curve reflects prefill amortization, not frac drift.")
    p.add_argument("--n_sink", type=int, default=2, help="pinned sink blocks (dynamic_frac path)")
    p.add_argument("--n_recency", type=int, default=2,
                   help="pinned recency blocks — set to the value your coherence run validated "
                        "(e.g. 8) so speed is measured at the iso-quality config")
    p.add_argument("--gen_lens", type=int, nargs="+", default=[64, 128, 256, 512])
    p.add_argument("--n_examples", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_block_cache", action="store_true",
                   help="disable the q=8 block-cache fast path (matches old sweep; slower)")
    p.add_argument("--same_path_baseline", action="store_true",
                   help="also time a K>>N baseline that pays commit-capture cost (isolates pruning)")
    args = parser = p.parse_args()

    use_block_cache = not args.no_block_cache

    print("Loading model...", flush=True)
    model = load_model(args.model, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    device = args.device

    all_rows = []  # (B, ctx, gen_len, dense, sparse, speedup, frac, d_gen, k_gen, sp_same)
    for batch_size in args.batch_sizes:
        for context_len in args.context_lens:
            n_blocks = context_len // BLOCK_SIZE
            k = max(1, round(args.frac * n_blocks)) if args.frac else args.k
            if args.dynamic_frac is not None:
                budget_desc = f"dynamic_frac={args.dynamic_frac:.1%} (held constant during gen)"
            else:
                budget_desc = f"K={k}  (K/N = {k}/{n_blocks} = {k / n_blocks:.1%}, drifts during gen)"
            print(f"\n{'#'*76}\nconfig: subset={args.subset}  B={batch_size}  ctx={context_len}  "
                  f"{budget_desc}  block_cache={use_block_cache}\n{'#'*76}", flush=True)

            print("Loading data...", flush=True)
            try:
                batches = load_longbench_subset(
                    args.subset, tokenizer, device, context_len,
                    batch_size, args.n_examples, args.seed,
                )
            except Exception as e:
                print(f"  data load failed: {e}", flush=True)
                continue

            for gen_len in args.gen_lens:
                print(f"\n=== B={batch_size} ctx={context_len} gen_len={gen_len} ===", flush=True)
                torch.cuda.empty_cache()

                try:
                    d_wall, d_gen, _, d_mem = time_condition(
                        model, tokenizer, batches, None, gen_len, use_block_cache, device)
                    print(f"  dense : {d_wall:7.3f} s/ex  gen={d_gen:.0f} tok  peak={d_mem:.1f} GB", flush=True)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print("  dense : OOM", flush=True)
                    continue

                sp_same = None
                if args.same_path_baseline:
                    try:
                        s_wall, _, _, _ = time_condition(
                            model, tokenizer, batches,
                            None if args.dynamic_frac is not None else 100000,
                            gen_len, use_block_cache, device,
                            frac=1.0 if args.dynamic_frac is not None else None)
                        sp_same = d_wall / s_wall if s_wall else 0.0
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()

                try:
                    k_wall, k_gen, frac, k_mem = time_condition(
                        model, tokenizer, batches,
                        None if args.dynamic_frac is not None else k,
                        gen_len, use_block_cache, device,
                        frac=args.dynamic_frac,
                        n_sink=args.n_sink, n_recency=args.n_recency)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print("  sparse: OOM", flush=True)
                    continue

                speedup = d_wall / k_wall if k_wall else 0.0
                print(f"  sparse: {k_wall:7.3f} s/ex  gen={k_gen:.0f} tok  frac={frac:.1%}  "
                      f"peak={k_mem:.1f} GB  -> speedup={speedup:.2f}x", flush=True)
                all_rows.append((batch_size, context_len, gen_len, d_wall, k_wall,
                                 speedup, frac, d_gen, k_gen, sp_same))

    # -------------------------------------------------------------------
    print("\n\n" + "=" * 84)
    print(f"DE-RISK RESULT — e2e wall-clock speedup (subset={args.subset})")
    print("=" * 84)
    hdr = f"{'B':>4}  {'ctx':>7}  {'gen_len':>8}  {'dense s/ex':>11}  {'sparse s/ex':>12}  {'speedup':>8}  {'frac':>6}"
    if args.same_path_baseline:
        hdr += f"  {'spd(same)':>9}"
    print(hdr)
    print("-" * (len(hdr) + 2))
    for (bs, cx, gl, dw, kw, sp, fr, dg, kg, sps) in all_rows:
        line = f"{bs:>4}  {cx:>7}  {gl:>8}  {dw:>11.3f}  {kw:>12.3f}  {sp:>7.2f}x  {fr:>6.1%}"
        if args.same_path_baseline:
            line += f"  {(f'{sps:.2f}x' if sps else '—'):>9}"
        print(line)

    # decode-only speedup per (B, ctx): differences out prefill via the gen_len slope
    from collections import defaultdict
    groups = defaultdict(list)
    for (bs, cx, gl, dw, kw, *_rest) in all_rows:
        groups[(bs, cx)].append((gl, dw, kw))
    print("\nDECODE-ONLY speedup (prefill removed via gen_len slope; needs >=2 gen_lens):")
    print(f"  {'B':>4}  {'ctx':>7}  {'decode_speedup':>14}")
    for (bs, cx), pts in sorted(groups.items()):
        if len(pts) < 2:
            continue
        pts.sort()
        (_glo, dlo, klo), (_ghi, dhi, khi) = pts[0], pts[-1]
        dsp = (dhi - dlo) / (khi - klo) if (khi - klo) > 0 else 0.0
        print(f"  {bs:>4}  {cx:>7}  {dsp:>13.2f}x")

    print("\nInterpretation:")
    print("  vs BATCH  (fixed ctx):   rising decode_speedup = weight floor amortizing (the real lever).")
    print("  vs ctx (fixed B, frac):  rising decode_speedup = KV% of bandwidth growing.")


if __name__ == "__main__":
    main()
