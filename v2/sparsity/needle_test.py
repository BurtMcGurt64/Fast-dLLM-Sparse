"""
NEEDLE-IN-HAYSTACK DEPTH SWEEP — the diagnostic that decides whether the commit-score
oracle has ANY winnable regime over position-only (StreamingLLM).

A fact ("secret access code for KEY is VALUE") is planted at a controlled DEPTH in an
~8k filler context; the model must retrieve it. StreamingLLM keeps sink (depth~0) +
recency (depth~1) but drops the MIDDLE — so at depth 0.25–0.75 it should FAIL while
content selection (commit) should succeed IF the oracle actually works.

  dense     — full KV (retrieves at all depths; upper bound)
  streaming — sink + recency (retrieves near 0 and 1, FAILS in the middle)
  commit    — commit-score content selection (retrieves the middle IFF the oracle works)

VERDICT:
  commit >> streaming at mid-depths  -> the oracle has a real regime -> build the method,
                                        then generalize (RULER / answer-depth-stratified real data).
  commit ~= streaming (both fail mid) -> selection cannot be saved -> lean on the coherence
                                        contribution instead.

This is a DEPTH CURVE (characterizes WHEN content beats position), a recognized long-context
figure — not a one-off eval. It gates the bigger investment before you make it.

Run:
  CUDA_VISIBLE_DEVICES=0 uv run v2/sparsity/needle_test.py \
      --context_len 8192 --frac 0.2 --n_needles 20 \
      --depths 0.05 0.25 0.5 0.75 0.95 --methods dense streaming commit
"""
import sys
import os
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "../experiments"))

import re
import time
import types
import random
import argparse
from collections import defaultdict
import torch

from transformers import AutoTokenizer
from helper import load_model
from generation_functions import Fast_dLLM_QwenForCausalLM as GenClass
from topk import TopKBlockLoader

BLOCK_SIZE = 32
KEYS = ["falcon", "cedar", "quartz", "harbor", "lantern", "meadow", "cobalt", "ember",
        "willow", "granite", "saffron", "cypress", "onyx", "marlin", "thistle", "beacon"]
FILLER = ("Entry {i}: the weather in district {i} stayed mild while the local market held "
          "steady and ordinary daily activity continued without any notable incident.")


def build_prompt(tokenizer, context_len, depth, key, value, device):
    """Build a ~context_len filler context with the needle planted at `depth`, then the
    question. Returns (input_ids[1,L], value). B=1 so no ragged-batch corruption."""
    needle = f" NOTE: the secret access code for {key} is {value}. "
    question = (f"\n\nWhat is the secret access code for {key}? "
                f"Respond with only the number.\n")

    sents, i = [], 0
    while True:
        sents.append(FILLER.format(i=i))
        i += 1
        if i % 40 == 0 and len(tokenizer(" ".join(sents)).input_ids) > context_len - 80:
            break
        if i > 8000:
            break

    pos = int(depth * len(sents))
    sents = sents[:pos] + [needle] + sents[pos:]
    user = " ".join(sents) + question
    try:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    except Exception:
        text = user
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    return ids, value


def parse_code(text):
    m = re.search(r"\d{5}", text)
    return int(m.group(0)) if m else None


@torch.no_grad()
def generate(model, sampler, tokenizer, ids, device, max_new=64):
    # max_new must be >= block_size (32) or batch_sample generates 0 blocks -> crashes
    plen = ids.shape[1]
    seq_len = torch.full((1,), plen, dtype=torch.long, device=device)
    out = sampler(input_ids=ids, tokenizer=tokenizer, block_size=BLOCK_SIZE,
                  max_new_tokens=max_new, small_block_size=8, min_len=plen,
                  seq_len=seq_len, threshold=0.95, temperature=0.0, use_block_cache=True)
    return tokenizer.decode(out[0][plen:], skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--context_len", type=int, default=8192)
    p.add_argument("--frac", type=float, default=0.2)
    p.add_argument("--n_sink", type=int, default=2)
    p.add_argument("--n_recency", type=int, default=8)
    p.add_argument("--n_needles", type=int, default=20)
    p.add_argument("--depths", type=float, nargs="+", default=[0.05, 0.25, 0.5, 0.75, 0.95])
    p.add_argument("--methods", nargs="+", default=["dense", "streaming", "commit"])
    p.add_argument("--max_new", type=int, default=64, help="must be >= block_size (32)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    assert args.max_new >= BLOCK_SIZE, f"--max_new must be >= {BLOCK_SIZE}"

    print("Loading model...", flush=True)
    model = load_model(args.model, device=args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    sampler = types.MethodType(GenClass.batch_sample, model)

    # pre-build prompts once (method-independent), same needles for every method
    random.seed(args.seed)
    print(f"Building {len(args.depths) * args.n_needles} needle prompts @ ~{args.context_len} tok...",
          flush=True)
    prompts = []  # (depth, ids, value)
    for depth in args.depths:
        for _ in range(args.n_needles):
            key = random.choice(KEYS)
            value = random.randint(10000, 99999)
            ids, val = build_prompt(tokenizer, args.context_len, depth, key, value, args.device)
            prompts.append((depth, ids, val))
    plen_example = prompts[0][1].shape[1]
    print(f"  example prompt length: {plen_example} tok\n", flush=True)

    results = {}   # method -> {depth: (correct, total)}
    for method in args.methods:
        loader = None
        if method != "dense":
            loader = TopKBlockLoader(model, frac=args.frac, n_sink=args.n_sink,
                                     n_recency=args.n_recency, selection=method).install()
        acc = defaultdict(lambda: [0, 0])
        t0 = time.perf_counter()
        try:
            for depth, ids, val in prompts:
                gen = generate(model, sampler, tokenizer, ids, args.device, args.max_new)
                acc[depth][0] += int(parse_code(gen) == val)
                acc[depth][1] += 1
        finally:
            if loader is not None:
                loader.uninstall()
        results[method] = {d: (c, n) for d, (c, n) in acc.items()}
        print(f"  {method:>10} done ({time.perf_counter() - t0:.0f}s)", flush=True)

    # ---- table: retrieval accuracy by depth ----
    print(f"\n{'='*72}\nNEEDLE RETRIEVAL ACCURACY vs DEPTH  (frac={args.frac:.0%}, "
          f"sink={args.n_sink} recency={args.n_recency}, n={args.n_needles}/depth)\n{'='*72}")
    hdr = f"  {'method':>10} " + " ".join(f"{'d=' + str(d):>7}" for d in args.depths)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for method in args.methods:
        cells = []
        for d in args.depths:
            c, n = results[method][d]
            cells.append(f"{c / n:>7.0%}" if n else f"{'—':>7}")
        print(f"  {method:>10} " + " ".join(cells))

    if "commit" in results and "streaming" in results:
        mids = [d for d in args.depths if 0.2 <= d <= 0.8]
        def midacc(m):
            cs = [results[m][d] for d in mids]
            tot = sum(n for _, n in cs)
            return sum(c for c, _ in cs) / tot if tot else 0.0
        gap = midacc("commit") - midacc("streaming")
        print(f"\n  MID-DEPTH (0.2–0.8) accuracy:  commit {midacc('commit'):.0%}  "
              f"streaming {midacc('streaming'):.0%}  gap {gap:+.0%}")
        verdict = ("ORACLE HAS A REAL REGIME -> build the method, then generalize (RULER/real data)"
                   if gap > 0.10 else
                   "commit ~= streaming in the middle -> selection can't be saved; lean on coherence")
        print(f"  VERDICT: {verdict}")
    print("\n  Read: dense should retrieve at all depths. streaming should be high at d~0.05")
    print("  (sink) and d~0.95 (recency) but DROP in the middle. commit beats streaming in the")
    print("  middle IFF the commit-score oracle actually finds low-recency content.")


if __name__ == "__main__":
    main()
