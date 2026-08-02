"""
COHERENCE CONTRIBUTION — does plain top-K KV sparsity break long-generation
coherence, and does sink+recency pinning fix it (at the SAME prune budget)?

This is the one asset that survived every prior-art check: Focus-dLLM and the
other dLLM-sparsity papers explicitly do NOT address coherence/repetition, and
the fix is specific to block diffusion (commit-score selection + guaranteed
recency). This script isolates the mechanism with a clean 3-way comparison at a
MATCHED frac budget, so the only variable is the pinning.

Per long-form prompt, generates under three conditions and reports repetition +
the actual text (for eyeballing / LLM-judge):
  1. DENSE            — reference (full KV)
  2. PLAIN top-K      — frac budget, NO pins (n_sink=0, n_recency=0)  <- should degenerate
  3. PINNED top-K     — same frac, sink+recency pinned              <- should stay coherent

The claim you want to land: rep(PLAIN) >> rep(PINNED) ~= rep(DENSE) at equal
frac, i.e. the pinning — not the pruning — is what preserves coherence.

Run:
  CUDA_VISIBLE_DEVICES=0 uv run v2/sparsity/coherence_eval.py \
      --n 4 --max_new_tokens 1024 --frac 0.2 --n_sink 2 --n_recency 8 --show_text
  # push length to stress it:
  CUDA_VISIBLE_DEVICES=0 uv run v2/sparsity/coherence_eval.py \
      --n 4 --max_new_tokens 2048 --frac 0.2 --n_recency 8
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../experiments"))

import time
import types
import json
import argparse
import contextlib
import statistics as st
import torch

from transformers import AutoTokenizer
from helper import load_model
from generation_functions import Fast_dLLM_QwenForCausalLM as GenClass
from topk import TopKBlockLoader

BLOCK_SIZE = 32
STOP_TOKEN = 151645

LONGFORM_PROMPTS = [
    "Write a detailed, multi-paragraph technical explanation of how the transformer "
    "neural network architecture works — cover tokenization, self-attention, multi-head "
    "attention, feed-forward layers, positional encoding, and how it is trained. Be thorough "
    "and write at least several hundred words.",
    "Write a long, detailed short story about an astronaut who discovers an abandoned space "
    "station orbiting a distant planet. Include description, dialogue, and a full arc.",
    "Explain in depth the main causes, key events, and long-term consequences of the "
    "Industrial Revolution, across several detailed paragraphs.",
    "Write a comprehensive beginner's guide to training for and running a first marathon: "
    "training plans, nutrition, injury prevention, and race-day strategy. Be detailed.",
    "Write a detailed explanation of how vaccines work, covering the immune system, antigens, "
    "antibodies, memory cells, different vaccine types, and herd immunity. Several paragraphs.",
    "Write an in-depth travel guide to Japan for a first-time visitor: regions, culture, food, "
    "etiquette, transport, and a suggested two-week itinerary. Be thorough.",
]


@contextlib.contextmanager
def suppress_stop_token(model, stop_token=STOP_TOKEN):
    """Force generation to run the full max_new_tokens by making the stop token
    unselectable — applied identically to dense/plain/pinned so length is matched
    (removes the 'plain loops to the cap while pinned stops early' confound) and the
    long-gen regime is actually stressed."""
    orig = model.sample_with_top_p

    def wrapped(logits, *a, **kw):
        logits[..., stop_token] = float("-inf")
        return orig(logits, *a, **kw)

    model.sample_with_top_p = wrapped
    try:
        yield
    finally:
        model.sample_with_top_p = orig


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


def distinct4(ids):
    """Distinct-4gram ratio (1.0 = all unique; low = repetitive). Complements rep."""
    if len(ids) < 8:
        return 1.0
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return len(set(grams)) / len(grams)


def prompt_ids(tokenizer, prompt, device):
    try:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    except Exception:
        text = prompt + "\n\n"
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    return ids, ids.shape[1]


def generate(model, sampler, tokenizer, ids, plen, max_new_tokens, threshold, device,
             force_len=False):
    seq_len = torch.full((1,), plen, dtype=torch.long, device=device)
    ctx = suppress_stop_token(model) if force_len else contextlib.nullcontext()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), ctx:
        out = sampler(input_ids=ids, tokenizer=tokenizer, block_size=BLOCK_SIZE,
                      max_new_tokens=max_new_tokens, small_block_size=8,
                      min_len=plen, seq_len=seq_len, threshold=threshold,
                      temperature=0.0, use_block_cache=True)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    gen_ids = out[0][plen:].tolist()
    if STOP_TOKEN in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(STOP_TOKEN)]
    return wall, gen_ids, tokenizer.decode(gen_ids, skip_special_tokens=True)


def run_condition(model, sampler, tokenizer, ids, plen, args, device, loader_kwargs):
    loader = None
    if loader_kwargs is not None:
        loader = TopKBlockLoader(model, frac=args.frac, block_size=BLOCK_SIZE, **loader_kwargs)
        loader.install()
    try:
        wall, gen_ids, text = generate(model, sampler, tokenizer, ids, plen,
                                       args.max_new_tokens, args.threshold, device,
                                       force_len=args.force_len)
        frac = 1.0
        if loader is not None:
            try:
                _, _, frac = loader.avg_stats
            except Exception:
                pass
    finally:
        if loader is not None:
            loader.uninstall()
    return dict(wall=wall, n=len(gen_ids), rep=repetition_score(gen_ids),
                d4=distinct4(gen_ids), frac=frac, text=text)


CONDITIONS = [
    ("dense",  None),
    ("plain",  dict(n_sink=0, n_recency=0)),   # pure content top-K, no pins
    ("pinned", None),                          # filled from args below
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n", type=int, default=4, help="number of long-form prompts")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--frac", type=float, default=0.2)
    p.add_argument("--n_sink", type=int, default=2)
    p.add_argument("--n_recency", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.95)
    p.add_argument("--show_text", action="store_true")
    p.add_argument("--print_chars", type=int, default=400)
    p.add_argument("--dump_json", default="", help="write per-condition text here for LLM-judge")
    p.add_argument("--long_context", action="store_true",
                   help="use ~context_len LongBench prompts instead of the short generation prompts, "
                        "so PINNED runs at TRUE frac (matches the 8k speedup regime) and the "
                        "plain-vs-pinned comparison is frac-matched — isolates recency from 'keep more'")
    p.add_argument("--subset", default="gov_report",
                   help="LongBench subset for --long_context (gov_report = long summaries, good stress)")
    p.add_argument("--context_len", type=int, default=8192, help="target prompt length for --long_context")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force_len", action="store_true",
                   help="suppress the stop token so ALL conditions generate exactly max_new_tokens "
                        "— matches length across dense/plain/pinned and stresses the long-gen regime")
    args = p.parse_args()

    conds = [("dense", None),
             ("plain", dict(n_sink=0, n_recency=0)),
             ("pinned", dict(n_sink=args.n_sink, n_recency=args.n_recency))]

    print("Loading model...", flush=True)
    model = load_model(args.model, device=args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    sampler = types.MethodType(GenClass.batch_sample, model)

    if args.long_context:
        from eval_longbench_sweep import load_longbench_subset
        print(f"Loading {args.subset} @ ~{args.context_len} tok (pinned will run at true frac)...",
              flush=True)
        batches = load_longbench_subset(args.subset, tokenizer, args.device,
                                        args.context_len, 1, args.n, args.seed)
        examples = [(args.subset, enc.input_ids.to(args.device), enc.input_ids.shape[1])
                    for enc, _ans in batches]
    else:
        examples = [(p, *prompt_ids(tokenizer, p, args.device))
                    for p in LONGFORM_PROMPTS[:args.n]]

    per_cond = {name: [] for name, _ in conds}
    dump = []

    for i, (label, ids, plen) in enumerate(examples):
        print(f"\n{'='*80}\nEXAMPLE {i+1}/{len(examples)}  (plen={plen})\n{'='*80}", flush=True)
        row = {"prompt": label}
        for name, kw in conds:
            r = run_condition(model, sampler, tokenizer, ids, plen, args, args.device, kw)
            per_cond[name].append(r)
            flag = "  <-- DEGENERATE" if r["rep"] > 0.3 else ""
            print(f"  {name:>7}: {r['n']:>4}tok  rep={r['rep']:.2f}  distinct4={r['d4']:.2f}  "
                  f"frac={r['frac']:.0%}  {r['wall']:5.2f}s{flag}", flush=True)
            row[name] = {"rep": r["rep"], "d4": r["d4"], "frac": r["frac"],
                         "n": r["n"], "text": r["text"]}
            if args.show_text:
                print(f"           {r['text'][:args.print_chars].strip()!r}", flush=True)
        dump.append(row)

    print(f"\n{'='*80}\nSUMMARY  n={len(examples)}  frac~{args.frac:.0%}  "
          f"sink={args.n_sink} recency={args.n_recency}  max_new={args.max_new_tokens}"
          f"{'  [long_context ' + args.subset + ']' if args.long_context else ''}")
    print(f"{'='*80}")
    print(f"  {'cond':>7} {'mean_rep':>9} {'mean_d4':>8} {'#degen':>7} {'mean_frac':>10}")
    print("  " + "-" * 46)
    for name, _ in conds:
        rs = per_cond[name]
        mrep = st.mean(r["rep"] for r in rs)
        md4 = st.mean(r["d4"] for r in rs)
        ndeg = sum(1 for r in rs if r["rep"] > 0.3)
        mfrac = st.mean(r["frac"] for r in rs)
        print(f"  {name:>7} {mrep:>9.2f} {md4:>8.2f} {ndeg:>4}/{len(rs)} {mfrac:>10.0%}")

    print("\n  CLAIM lands if:  plain rep >> pinned rep ~= dense rep, at equal frac.")
    print("  (plain degenerates from pruning its own recent output; pinning fixes it.)")

    if args.dump_json:
        with open(args.dump_json, "w") as f:
            json.dump(dump, f, indent=2)
        print(f"\n  wrote text dump -> {args.dump_json}  (feed dense vs pinned to an LLM judge)")


if __name__ == "__main__":
    main()
