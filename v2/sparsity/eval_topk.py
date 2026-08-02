"""
Evaluates top-K prefix block loading on GSM8K.

Sweeps K values and reports accuracy + throughput vs. the full-prefix baseline.
Run at batch > 1 to see KV bandwidth savings (MLP weights amortize at batch > 1,
making KV reads the dominant cost and top-K the right lever).

Example
-------
    # baseline + K sweep at batch=8
    uv run v2/sparsity/eval_topk.py --batch_size 8 --n_questions 80

    # single K, larger run
    uv run v2/sparsity/eval_topk.py --batch_size 8 --k_values 8 --n_questions 200
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../experiments"))

import re
import time
import types
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from helper import load_model
from generation_functions import Fast_dLLM_QwenForCausalLM as GenClass
from topk import TopKBlockLoader

BLOCK_SIZE = 32
DEFAULT_K_VALUES = [4, 8, 16, 32, None]   # None = full prefix (baseline)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

_FILLER_TEXT = (
    "The quick brown fox jumps over the lazy dog. " * 200
)

def load_gsm8k_batched(n, tokenizer, device, batch_size, target_ctx_len=None):
    """
    Load n GSM8K questions.  If target_ctx_len is set, prepend filler text so
    the tokenized prompt reaches approximately that many tokens.  This puts the
    model in a long-context regime where top-K block loading can prune real blocks.
    """
    ds = load_dataset("gsm8k", "main", split="test")
    questions = ds[:n]["question"]
    answers   = [_extract_answer(ds[i]["answer"]) for i in range(n)]

    # compute filler prefix once (same for all questions)
    filler_prefix = ""
    if target_ctx_len is not None:
        # rough estimate: tokenize short question, see how many filler tokens needed
        sample_q = questions[0]
        sample_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": sample_q}], tokenize=False, add_generation_prompt=True
        )
        sample_len = len(tokenizer(sample_text, return_tensors="pt").input_ids[0])
        filler_tokens_needed = max(0, target_ctx_len - sample_len)
        if filler_tokens_needed > 0:
            filler_ids = tokenizer(_FILLER_TEXT, return_tensors="pt").input_ids[0]
            filler_prefix = tokenizer.decode(filler_ids[:filler_tokens_needed], skip_special_tokens=True)

    texts = []
    for q in questions:
        content = (filler_prefix + " " + q) if filler_prefix else q
        msgs = [{"role": "user", "content": content}]
        texts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    original_side        = tokenizer.padding_side
    tokenizer.padding_side = "left"
    tokenizer.pad_token    = tokenizer.eos_token

    batches = []
    for i in range(0, len(texts), batch_size):
        chunk   = texts[i:i + batch_size]
        encoded = tokenizer(chunk, return_tensors="pt", padding=True).to(device)
        batches.append((encoded, answers[i:i + batch_size]))

    tokenizer.padding_side = original_side

    if target_ctx_len is not None:
        actual_len = batches[0][0].input_ids.shape[1]
        print(f"  [ctx] target={target_ctx_len} actual prompt_len={actual_len} "
              f"n_blocks={actual_len // BLOCK_SIZE}")

    return batches


def _extract_answer(text):
    m = re.search(r'####\s*([\d,]+)', text)
    return m.group(1).replace(',', '') if m else None


def _check_answer(generated, ground_truth):
    if ground_truth is None:
        return None
    # try #### format first (standard GSM8K)
    m = re.search(r'####\s*([\d,]+)', generated)
    if m:
        return m.group(1).replace(',', '') == ground_truth
    # fallback: last number in text
    nums = re.findall(r'\b\d+(?:,\d+)*\b', generated)
    if not nums:
        return False
    return nums[-1].replace(',', '') == ground_truth


# ---------------------------------------------------------------------------
# Single evaluation run
# ---------------------------------------------------------------------------

def run_eval(model, tokenizer, batches, k, device,
             threshold=0.95, temperature=0.0, max_new_tokens=512,
             same_path_baseline=False):
    """
    Run generation on all batches with top-K=k (or full prefix if k is None).
    Returns (accuracy, tokens_per_second, n_evaluated, stats_dict).

    same_path_baseline: when k is None, install loader with K=10000 (no pruning)
      so baseline runs the same commit-step capture path as top-K rows.
      Cancels the commit-step overhead difference; speedup then measures
      pruning-only effect.
    """
    correct = 0
    total   = 0
    total_new_tokens = 0
    t_start = time.perf_counter()

    if k is not None:
        loader = TopKBlockLoader(model, k=k)
    elif same_path_baseline:
        loader = TopKBlockLoader(model, k=10000)  # K >> N, no pruning, same code path
    else:
        loader = None
    if loader is not None:
        loader.install()

    sampler = types.MethodType(GenClass.batch_sample, model)

    try:
        for encoded, ground_truths in batches:
            input_ids  = encoded.input_ids
            B          = input_ids.shape[0]
            prompt_len = input_ids.shape[1]

            # seq_len must be one entry per sequence, all equal to the padded prompt length.
            # batch_sample uses seq_len[b] as the position where generation starts for
            # sequence b — passing actual (unpadded) length would start generation mid-prompt.
            seq_len_t = torch.full((B,), prompt_len, dtype=torch.long, device=device)

            output = sampler(
                input_ids        = input_ids,
                tokenizer        = tokenizer,
                block_size       = BLOCK_SIZE,
                max_new_tokens   = max_new_tokens,
                small_block_size = 8,
                min_len          = prompt_len,
                seq_len          = seq_len_t,
                threshold        = threshold,
                temperature      = temperature,
            )

            for b, gt in enumerate(ground_truths):
                gen_ids  = output[b][prompt_len:]
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                if total == 0 and b == 0:
                    print(f"  [debug] prompt_len={prompt_len} gen_tokens={len(gen_ids)}")
                    print(f"  [debug] first answer: {gen_text[:200]!r}")
                ok = _check_answer(gen_text, gt)
                if ok is not None:
                    correct += int(ok)
                    total   += 1
                total_new_tokens += len(gen_ids)

    finally:
        stats = {}
        if loader is not None:
            avg_n, avg_u, avg_frac = loader.avg_stats
            stats = {"avg_n": avg_n, "avg_union": avg_u, "avg_frac": avg_frac}
            loader.uninstall()

    elapsed   = time.perf_counter() - t_start
    accuracy  = correct / total if total > 0 else 0.0
    tps       = total_new_tokens / elapsed

    return accuracy, tps, total, stats


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         type=str,   default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    parser.add_argument("--device",        type=str,   default="cuda:0")
    parser.add_argument("--n_questions",   type=int,   default=100)
    parser.add_argument("--batch_size",    type=int,   default=8)
    parser.add_argument("--threshold",     type=float, default=0.95)
    parser.add_argument("--max_new_tokens",type=int,   default=512)
    parser.add_argument("--k_values",      type=int,   nargs="+", default=None,
                        help="K values to sweep (default: 4 8 16 32 + baseline)")
    parser.add_argument("--context_len",   type=int,   default=None,
                        help="Pad each prompt with filler to ~this many tokens "
                             "(use 1024/2048/4096 to test long-context KV savings)")
    parser.add_argument("--debug_mask",      action="store_true",
                        help="Print attention_mask shape at layer 0 for first few small steps")
    parser.add_argument("--same_path_baseline", action="store_true",
                        help="Run baseline with loader installed (K=10000, no pruning) so "
                             "baseline and K-rows share the same commit-step code path. "
                             "Speedup then measures pruning only, not kernel difference.")
    args = parser.parse_args()

    k_values = args.k_values if args.k_values else DEFAULT_K_VALUES[:-1]
    # run baseline first so speedup is computable for all K rows
    k_values = [None] + list(k_values)

    print("Loading model...")
    model     = load_model(args.model, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    device    = args.device

    if args.debug_mask:
        for layer in model.model.layers:
            layer.self_attn._debug_mask = True

    print(f"\nLoading {args.n_questions} GSM8K questions, batch_size={args.batch_size}...")
    batches = load_gsm8k_batched(
        args.n_questions, tokenizer, device, args.batch_size,
        target_ctx_len=args.context_len
    )

    # quick sanity check: print first generated answer from baseline
    print("\n[sanity] running 1 question to check generation...")
    _sanity_batch = [(batches[0][0], batches[0][1][:1])]
    _sampler = types.MethodType(GenClass.batch_sample, model)
    _enc = _sanity_batch[0][0]
    _plen = _enc.input_ids.shape[1]
    _out = _sampler(
        input_ids=_enc.input_ids[:1], tokenizer=tokenizer,
        block_size=BLOCK_SIZE, max_new_tokens=200, small_block_size=8,
        min_len=_plen, seq_len=torch.tensor([_plen], device=device),
        threshold=args.threshold, temperature=0.0,
    )
    _gen = tokenizer.decode(_out[0][_plen:], skip_special_tokens=True).strip()
    print(f"[sanity] Q: {batches[0][1][0]}  →  A_gt: ???")
    print(f"[sanity] Generated: {_gen[:200]!r}\n")

    results = {}
    n_total = None

    for k in k_values:
        label = str(k) if k is not None else "full"
        print(f"  running K={label}...", flush=True)
        is_baseline = (k is None)
        acc, tps, n, stats = run_eval(
            model, tokenizer, batches, k, device,
            threshold=args.threshold, max_new_tokens=args.max_new_tokens,
            same_path_baseline=(is_baseline and args.same_path_baseline),
        )
        results[k] = (acc, tps, stats)
        n_total = n

    baseline_tps = results[None][1]

    print(f"\n{'K':>6}  {'acc':>7}  {'tok/s':>8}  {'speedup':>9}  {'avg_N':>6}  {'union':>6}  {'frac':>6}")
    print("-" * 62)
    for k in k_values:
        label = str(k) if k is not None else "full"
        acc, tps, stats = results[k]
        if k is None:
            speedup_str  = "1.00x (baseline)"
            stats_str    = f"{'—':>6}  {'—':>6}  {'—':>6}"
        else:
            speedup_str = f"{tps/baseline_tps:.2f}x"
            avg_n  = stats.get("avg_n", 0)
            avg_u  = stats.get("avg_union", 0)
            frac   = stats.get("avg_frac", 0)
            stats_str = f"{avg_n:>6.1f}  {avg_u:>6.1f}  {frac:>5.1%}"
        print(f"  {label:>4}  {acc:>6.1%}  {tps:>8.1f}  {speedup_str:<18}  {stats_str}")

    print(f"\n(n={n_total} questions evaluated per K)")
    print("avg_N = avg prefix blocks at commit time, union = avg blocks loaded across batch, frac = union/N")


if __name__ == "__main__":
    main()
