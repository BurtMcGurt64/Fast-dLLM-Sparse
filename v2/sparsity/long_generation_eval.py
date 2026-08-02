"""
Does the model generate MANY coherent blocks on a long-generation task, and does
top-K reach its ceiling there? The postdoc's fork: if the model produces long,
coherent output, top-K climbs toward ~2x with no training; if it degenerates
(repetition), you can't reach the ceiling without training.

For each example, runs DENSE and top-K SPARSE generation, and reports:
  - generated length (tokens + blocks) — is the model actually generating a lot?
  - repetition score (auto degeneration flag) — is it coherent?
  - wall-clock speedup (per example, since gen lengths may differ)
  - THE ACTUAL GENERATED TEXT, dense and sparse, so you can judge coherence yourself.

Tasks:
  gov_report  : LongBench summarization (long context ~8k, multi-hundred-token summaries)
  longform    : short-context prompts that ask for very long output (tests many-block coherence)

Run:
  CUDA_VISIBLE_DEVICES=0 uv run v2/sparsity/long_generation_eval.py \
      --task gov_report --n 3 --context_len 8192 --max_new_tokens 512 --frac 0.2
  CUDA_VISIBLE_DEVICES=0 uv run v2/sparsity/long_generation_eval.py \
      --task longform --n 4 --max_new_tokens 1024 --frac 0.2
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../experiments"))

import time
import types
import argparse
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
]


def repetition_score(ids):
    """Fraction of 4-grams that are repeats. High (>0.3) => degenerate/looping."""
    if len(ids) < 8:
        return 0.0
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    seen, rep = set(), 0
    for g in grams:
        if g in seen:
            rep += 1
        seen.add(g)
    return rep / len(grams)


def load_task(task, tokenizer, device, n, context_len, seed):
    """Return list of (input_ids, seq_len, min_len)."""
    if task == "gov_report":
        from eval_longbench_sweep import load_longbench_subset
        raw = load_longbench_subset("gov_report", tokenizer, device, context_len, 1, n, seed)
        out = []
        for encoded, _ans in raw:
            ids = encoded.input_ids
            plen = ids.shape[1]
            out.append((ids, torch.full((1,), plen, device=device, dtype=torch.long), plen))
        return out
    else:  # longform
        out = []
        for prompt in LONGFORM_PROMPTS[:n]:
            try:
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
            except Exception:
                text = prompt + "\n\n"
            ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            plen = ids.shape[1]
            out.append((ids, torch.full((1,), plen, device=device, dtype=torch.long), plen))
        return out


def generate(sampler, tokenizer, ids, seq_len, min_len, max_new_tokens, threshold):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = sampler(input_ids=ids, tokenizer=tokenizer, block_size=BLOCK_SIZE,
                      max_new_tokens=max_new_tokens, small_block_size=8,
                      min_len=min_len, seq_len=seq_len, threshold=threshold,
                      temperature=0.0, use_block_cache=True)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    gen_ids = out[0][min_len:].tolist()
    # trim at stop token for length/repetition
    if STOP_TOKEN in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(STOP_TOKEN)]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return wall, gen_ids, text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--task", choices=["gov_report", "longform", "math_cot"], default="gov_report")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--context_len", type=int, default=8192, help="prompt length for gov_report")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--frac", type=float, default=0.2)
    p.add_argument("--n_sink", type=int, default=2, help="always-keep this many blocks at the start")
    p.add_argument("--n_recency", type=int, default=4,
                   help="always-keep this many MOST-RECENT blocks (coherence — fixes long-gen looping)")
    p.add_argument("--threshold", type=float, default=0.95)
    p.add_argument("--show_text", action="store_true", help="print a short snippet of each generation")
    p.add_argument("--print_chars", type=int, default=300, help="chars of text to print when --show_text")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("Loading model...", flush=True)
    model = load_model(args.model, device=args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    sampler = types.MethodType(GenClass.batch_sample, model)
    examples = load_task(args.task, tokenizer, args.device, args.n, args.context_len, args.seed)

    rows = []
    for i, (ids, seq_len, min_len) in enumerate(examples):
        n_blocks = min_len // BLOCK_SIZE

        # DENSE
        d_wall, d_ids, d_text = generate(sampler, tokenizer, ids, seq_len, min_len,
                                         args.max_new_tokens, args.threshold)
        # SPARSE — frac budget scales with the growing context; sink+recency pinned
        loader = TopKBlockLoader(model, frac=args.frac, block_size=BLOCK_SIZE,
                                 n_sink=args.n_sink, n_recency=args.n_recency)
        loader.install()
        s_wall, s_ids, s_text = generate(sampler, tokenizer, ids, seq_len, min_len,
                                         args.max_new_tokens, args.threshold)
        try:
            _, _, frac = loader.avg_stats
        except Exception:
            frac = k / n_blocks if n_blocks else 0
        loader.uninstall()

        d_rep, s_rep = repetition_score(d_ids), repetition_score(s_ids)
        speedup = d_wall / s_wall if s_wall else 0
        rows.append((i + 1, min_len, len(d_ids), len(s_ids), d_wall, s_wall, speedup, d_rep, s_rep))

        flag = "  <-- SPARSE DEGENERATE" if s_rep > 0.3 else ""
        print(f"  ex{i+1:>2}: dense {len(d_ids):>4}tok/{d_wall:5.2f}s rep{d_rep:.2f}  |  "
              f"sparse {len(s_ids):>4}tok/{s_wall:5.2f}s rep{s_rep:.2f}  |  {speedup:5.2f}x{flag}", flush=True)
        if args.show_text:
            print(f"      DENSE : {d_text[:args.print_chars].strip()!r}")
            print(f"      SPARSE: {s_text[:args.print_chars].strip()!r}\n", flush=True)

    print(f"\n{'='*80}")
    print(f"SUMMARY [{args.task}]  n={len(rows)}  frac={args.frac:.0%} "
          f"sink={args.n_sink} recency={args.n_recency}")
    print(f"{'='*80}")
    print(f"  {'ex':>3} {'prompt':>7} {'d_tok':>6} {'s_tok':>6} {'d_rep':>6} {'s_rep':>6} {'speedup':>8}  flag")
    print("  " + "-" * 74)
    for (ix, plen, dtok, stok, dw, sw, sp, drep, srep) in rows:
        deg = "DEGEN" if srep > 0.3 else ""
        print(f"  {ix:>3} {plen:>7} {dtok:>6} {stok:>6} {drep:>6.2f} {srep:>6.2f} {sp:>7.2f}x  {deg}")
    if rows:
        import statistics as st
        col = lambda idx: [r[idx] for r in rows]
        n_deg = sum(1 for r in rows if r[8] > 0.3)
        print("  " + "-" * 74)
        print(f"  {'avg':>3} {'':>7} {st.mean(col(2)):>6.0f} {st.mean(col(3)):>6.0f} "
              f"{st.mean(col(7)):>6.2f} {st.mean(col(8)):>6.2f} {st.mean(col(6)):>7.2f}x")
        print(f"\n  speedup: mean {st.mean(col(6)):.2f}x  median {st.median(col(6)):.2f}x")
        print(f"  sparse degenerate (rep>0.3): {n_deg}/{len(rows)} examples")
    print("\n  Read: low s_rep across most rows -> coherent under sparsity (recency fix works).")
    print("        several DEGEN rows -> not robust; raise --n_recency, or it needs training.")
    print("        (use --show_text to eyeball actual generations.)")


if __name__ == "__main__":
    main()
