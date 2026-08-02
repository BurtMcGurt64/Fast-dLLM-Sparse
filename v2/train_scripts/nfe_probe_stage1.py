"""
STAGE 1 NFE PROBE — quality-vs-NFE curve for the BASE checkpoint (NO training).

Purpose: measure the prize before investing a week in a fine-tune. The confidence
threshold controls how many tokens unmask per denoising step:
    lower threshold -> more tokens per step -> FEWER forward passes = lower NFE
(there's always >=1 unmask per step, so it can't stall). We sweep the threshold
and, per setting, measure accuracy AND the real NFE (number of model.forward calls).

How to read the result:
  - Accuracy stays flat as NFE drops  -> free speedup, maybe no training needed.
  - Accuracy holds then CLIFFS at some low threshold -> that cliff is exactly what
    a few-step fine-tune (Stage 2) would need to recover.

Run the SAME sweep on >=2 datasets to check whether the cliff location is
eval-dependent (it usually is somewhat):
  CUDA_VISIBLE_DEVICES=0 uv run v2/train_scripts/nfe_probe_stage1.py \
      --dataset gsm8k --n_examples 200 --thresholds 0.95 0.9 0.8 0.7 0.6 0.5 0.4 0.3
  CUDA_VISIBLE_DEVICES=0 uv run v2/train_scripts/nfe_probe_stage1.py \
      --dataset math  --n_examples 200 --max_new_tokens 512 \
      --thresholds 0.95 0.9 0.8 0.7 0.6 0.5 0.4 0.3

KEEP --batch_size 1: batched decoding corrupts ragged prompts (accuracy invalid)
and makes NFE/ex an amortized, non-comparable number.
"""
import sys
import os
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))              # v2/
sys.path.insert(0, os.path.join(HERE, "../experiments"))  # v2/experiments (helper)

import re
import time
import types
import argparse
import torch

from transformers import AutoTokenizer
from helper import load_model
from generation_functions import Fast_dLLM_QwenForCausalLM as GenClass

BLOCK_SIZE = 32
STOP_TOKEN = 151645
MASK_ID = 151665

# ---------------------------------------------------------------------------
# Dataset registry. Each eval = how to load (question, gold) + how to prompt +
# how to parse a prediction. Add a new eval by adding one entry.
# ---------------------------------------------------------------------------
GSM8K_FEWSHOT = """Question: Natalia sold clips to 48 friends in April, and then she sold half as many clips in May. How many clips did she sell altogether in April and May?
Answer: In April she sold 48 clips. In May she sold 48 / 2 = 24 clips. Altogether she sold 48 + 24 = 72 clips.
#### 72

Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer: Per minute she earns 12 / 60 = $0.2. For 50 minutes she earned 50 * 0.2 = $10.
#### 10

Question: Betty is saving for a $100 wallet. She has half the money. Her parents give her $15 and her grandparents twice as much as her parents. How much more does she need?
Answer: Betty has 100 / 2 = $50. Grandparents give 2 * 15 = $30. Now she has 50 + 15 + 30 = $95. She needs 100 - 95 = $5.
#### 5

"""

MATH_FEWSHOT = """Problem: What is $2+2\\times 3$?
Solution: Multiplication first: $2\\times 3 = 6$. Then $2 + 6 = 8$. The answer is $\\boxed{8}$.

Problem: If $x/3 = 4$, what is $x$?
Solution: Multiply both sides by 3: $x = 12$. The answer is $\\boxed{12}$.

"""


def _gsm8k_gold(answer):
    m = re.search(r"####\s*(-?[\d,]+)", answer)
    return m.group(1).replace(",", "") if m else None


def _gsm8k_pred(text):
    m = re.search(r"####\s*(-?[\d,]+)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?[\d,]+", text)
    return nums[-1].replace(",", "") if nums else None


def _norm_math(s):
    """Light normalization so string-match on \\boxed content is usable. This
    undercounts absolute accuracy (equivalent forms like 1/2 vs 0.5 miss), but the
    CLIFF SHAPE — the relative drop as threshold falls — is what we care about."""
    if s is None:
        return None
    s = s.strip().strip("$").strip()
    for tok in [r"\boxed", r"\text", r"\left", r"\right", r"\!", r"\,", r"\ "]:
        s = s.replace(tok, "")
    s = s.replace("{", "").replace("}", "").replace(" ", "").rstrip(".")
    return s


def _math_pred(text):
    boxes = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if boxes:
        return _norm_math(boxes[-1])
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return _norm_math(nums[-1]) if nums else None


def _load_gsm8k(n, seed):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test").shuffle(seed=seed)
    ds = ds.select(range(min(n, len(ds))))
    return [(ex["question"], _gsm8k_gold(ex["answer"])) for ex in ds]


def _load_math(n, seed):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test").shuffle(seed=seed)
    ds = ds.select(range(min(n, len(ds))))
    return [(ex["problem"], _norm_math(ex["answer"])) for ex in ds]


EVALS = {
    "gsm8k": dict(load=_load_gsm8k, prompt=lambda q: GSM8K_FEWSHOT + f"Question: {q}\nAnswer:",
                  parse_pred=_gsm8k_pred),
    "math":  dict(load=_load_math, prompt=lambda q: MATH_FEWSHOT + f"Problem: {q}\nSolution:",
                  parse_pred=_math_pred),
}


class NFECounter:
    """Wrap model.forward to count forward passes during generation."""
    def __init__(self, model):
        self.model = model
        self._orig = model.forward
        self.count = 0

    def __enter__(self):
        def wrapped(*a, **k):
            self.count += 1
            return self._orig(*a, **k)
        self.model.forward = wrapped
        return self

    def __exit__(self, *x):
        self.model.forward = self._orig


def build_batches(prompts, golds, tokenizer, device, batch_size):
    """Tokenize prompts, right-pad per batch, return batch_sample-ready tuples."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else STOP_TOKEN
    batches = []
    for i in range(0, len(prompts), batch_size):
        pchunk, gchunk = prompts[i:i + batch_size], golds[i:i + batch_size]
        enc = [tokenizer(p, return_tensors="pt").input_ids[0] for p in pchunk]
        lens = [e.shape[0] for e in enc]
        ids = torch.full((len(enc), max(lens)), pad_id, dtype=torch.long)
        for j, e in enumerate(enc):
            ids[j, :e.shape[0]] = e
        seq_len = torch.tensor(lens, device=device)
        batches.append(((ids.to(device), seq_len, min(lens), lens), gchunk))
    return batches


def run_threshold(model, tokenizer, sampler, batches, threshold, max_new_tokens, parse_pred,
                  debug=0, small_block_size=8):
    correct, total, total_nfe, shown = 0, 0, 0, 0
    for (ids, seq_len, min_len, true_lens), golds in batches:
        with NFECounter(model) as ctr:
            out = sampler(
                input_ids=ids, tokenizer=tokenizer, block_size=BLOCK_SIZE,
                max_new_tokens=max_new_tokens, small_block_size=small_block_size,
                min_len=min_len, seq_len=seq_len, threshold=threshold,
                temperature=0.0, use_block_cache=True,
            )
        total_nfe += ctr.count
        for b in range(len(true_lens)):
            gen = tokenizer.decode(out[b][true_lens[b]:], skip_special_tokens=True)
            pred = parse_pred(gen)
            ok = pred is not None and pred == golds[b]
            correct += int(ok)
            total += 1
            if shown < debug:
                shown += 1
                print(f"\n--- example {shown} [{'OK' if ok else 'WRONG'}] ---")
                print(f"GOLD: {golds[b]!r}   PRED: {pred!r}")
                print(f"GEN TAIL: ...{gen[-300:]!r}")
    return correct / total, total_nfe / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dataset", choices=list(EVALS), default="gsm8k")
    p.add_argument("--n_examples", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=1,
                   help="KEEP AT 1. Batched decoding corrupts ragged prompts and amortizes NFE/ex.")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--thresholds", type=float, nargs="+",
                   default=[0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    p.add_argument("--small_block_sizes", type=int, nargs="+", default=[8],
                   help="also sweep the sub-block size (8 = shipped config; 32 removes the "
                        "4-sub-block serialization floor: min 1 small step per block instead "
                        "of 4). Full cross sweep: sizes x thresholds.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--debug", type=int, default=0,
                   help="print this many (gold, pred, generation tail) to check parsing")
    args = p.parse_args()

    if args.batch_size != 1:
        print("WARNING: batch_size != 1. Accuracy invalid + NFE/ex amortized. Use 1.\n", flush=True)

    print(f"Loading model + {args.dataset}...", flush=True)
    model = load_model(args.model, device=args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    ev = EVALS[args.dataset]
    data = ev["load"](args.n_examples, args.seed)
    prompts = [ev["prompt"](q) for q, _ in data]
    golds = [g for _, g in data]
    batches = build_batches(prompts, golds, tokenizer, args.device, args.batch_size)
    sampler = types.MethodType(GenClass.batch_sample, model)

    print(f"\n{args.dataset}  n={len(data)}  batch={args.batch_size}  max_new={args.max_new_tokens}\n")
    all_rows = {}
    for sbs in args.small_block_sizes:
        print(f"  small_block_size={sbs}")
        print(f"  {'threshold':>9}  {'accuracy':>8}  {'NFE/ex':>8}  {'sec/ex':>7}")
        print("  " + "-" * 40)
        rows = []
        for th in sorted(args.thresholds, reverse=True):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            acc, nfe = run_threshold(model, tokenizer, sampler, batches, th,
                                     args.max_new_tokens, ev["parse_pred"], debug=args.debug,
                                     small_block_size=sbs)
            torch.cuda.synchronize()
            sec = (time.perf_counter() - t0) / len(data)
            rows.append((th, acc, nfe, sec))
            print(f"  {th:>9.2f}  {acc:>8.1%}  {nfe:>8.1f}  {sec:>7.2f}", flush=True)
        all_rows[sbs] = rows
        print()

    base = all_rows[args.small_block_sizes[0]][0]
    print(f"Reference (sbs={args.small_block_sizes[0]}, threshold={base[0]:.2f}): "
          f"acc={base[1]:.1%}, NFE={base[2]:.1f}/ex")
    print("Prize per config (NFE reduction vs accuracy retained, vs the reference above):")
    for sbs in args.small_block_sizes:
        for th, acc, nfe, _ in all_rows[sbs]:
            cut = base[2] / nfe if nfe else 0
            print(f"  sbs={sbs} th={th:.2f}: {cut:.2f}x fewer forwards, "
                  f"accuracy {acc:.1%} ({acc - base[1]:+.1%} vs ref)")
    print(f"\nCompare the cliff location across --dataset gsm8k vs math. If the cliff is at a")
    print(f"similar threshold on both, the effect is structural and Stage 2 is worth it.")


if __name__ == "__main__":
    main()
