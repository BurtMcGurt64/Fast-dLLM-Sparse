"""
Top-K prefix block loading evaluation on LongBench.

Tests on naturally long documents (no artificial filler padding) to measure
whether top-K preserves accuracy on real long-context retrieval tasks.

Subsets used:
  qasper         — scientific paper QA, avg ~3.6k tokens
  multifieldqa_en — multi-field document QA, avg ~4.5k tokens
  2wikimqa       — multi-hop QA, avg ~5k tokens

Run:
    # baseline + K=16 at 2k context, batch=8
    CUDA_VISIBLE_DEVICES=2 uv run v2/sparsity/eval_longbench.py \
        --subset qasper --context_len 2048 --batch_size 8 --k_values 16

    # sweep context lengths and batch sizes
    CUDA_VISIBLE_DEVICES=2 uv run v2/sparsity/eval_longbench.py \
        --subset qasper --context_len 2048 4096 --batch_size 8 16 32 --k_values 8 16
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

# ---------------------------------------------------------------------------
# Prompt templates (task-specific, following LongBench conventions)
# ---------------------------------------------------------------------------

TEMPLATES = {
    "qasper": (
        "You are given a scientific paper and a question. Answer the question "
        "in as few words as possible based on the paper.\n\n"
        "Paper:\n{context}\n\nQuestion: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text carefully and answer the question with a short answer.\n\n"
        "Text:\n{context}\n\nQuestion: {input}\n\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. "
        "Only give me the answer and do not output any other words.\n\n"
        "Passages:\n{context}\n\nQuestion: {input}\n\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. "
        "Only give me the answer and do not output any other words.\n\n"
        "Passages:\n{context}\n\nQuestion: {input}\n\nAnswer:"
    ),
    "narrativeqa": (
        "You are given a story and a question. Answer the question as concisely "
        "as possible.\n\nStory:\n{context}\n\nQuestion: {input}\n\nAnswer:"
    ),
}

DEFAULT_TEMPLATE = (
    "Read the following text and answer the question in a few words.\n\n"
    "Text:\n{context}\n\nQuestion: {input}\n\nAnswer:"
)

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def normalize(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


def f1_score(pred, gold):
    pred_tokens  = normalize(pred).split()
    gold_tokens  = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return int(pred_tokens == gold_tokens)
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    p = len(common) / len(pred_tokens)
    r = len(common) / len(gold_tokens)
    return 2 * p * r / (p + r)


def score_answer(prediction, answers):
    """Max F1 across all acceptable answers."""
    if not answers:
        return 0.0
    pred_norm = normalize(prediction[:200])  # only first 200 chars of generated text
    return max(f1_score(pred_norm, normalize(a)) for a in answers)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_longbench(subset, tokenizer, device, context_len, batch_size, n_examples, seed=42):
    """
    Load LongBench subset, truncate context to context_len tokens, batch with left-padding.
    Returns list of (encoded_batch, answers_list) tuples.
    """
    print(f"  Loading LongBench/{subset}...", flush=True)
    # THUDM/LongBench data is in data.zip; download and extract on first use.
    cache_dir = os.path.join(os.path.dirname(__file__), ".longbench_cache")
    jsonl_path = os.path.join(cache_dir, f"{subset}.jsonl")

    if not os.path.exists(jsonl_path):
        os.makedirs(cache_dir, exist_ok=True)
        zip_path = os.path.join(cache_dir, "data.zip")
        if not os.path.exists(zip_path):
            print("  Downloading data.zip from HuggingFace...", flush=True)
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id="THUDM/LongBench",
                filename="data.zip",
                repo_type="dataset",
                local_dir=cache_dir,
            )
        import zipfile
        print("  Extracting data.zip...", flush=True)
        with zipfile.ZipFile(os.path.join(cache_dir, "data.zip"), "r") as zf:
            zf.extractall(cache_dir)
        # files may be extracted into a "data/" subdirectory
        extracted = os.path.join(cache_dir, "data", f"{subset}.jsonl")
        if os.path.exists(extracted):
            import shutil
            shutil.move(extracted, jsonl_path)

    ds = load_dataset("json", data_files=jsonl_path, split="train")

    # shuffle for diversity
    ds = ds.shuffle(seed=seed)
    if n_examples and n_examples < len(ds):
        ds = ds.select(range(n_examples))

    template = TEMPLATES.get(subset, DEFAULT_TEMPLATE)

    texts  = []
    all_answers = []
    actual_lengths = []

    for ex in ds:
        context  = ex.get("context", "") or ex.get("input", "")
        question = ex.get("input", "")
        answers  = ex.get("answers", [])
        if isinstance(answers, str):
            answers = [answers]

        # build full prompt, then truncate context if needed
        full_prompt = template.format(context=context, input=question)

        # tokenize full prompt to check length; if too long, truncate context
        full_ids = tokenizer(full_prompt, return_tensors="pt").input_ids[0]
        if len(full_ids) > context_len:
            # truncate context so total fits in context_len
            question_suffix = template.split("{context}")[1].format(input=question)
            suffix_ids = tokenizer(question_suffix, return_tensors="pt").input_ids[0]
            prefix_template = template.split("{context}")[0]
            prefix_ids = tokenizer(prefix_template, return_tensors="pt").input_ids[0]
            overhead = len(prefix_ids) + len(suffix_ids)
            budget = max(64, context_len - overhead)

            ctx_ids = tokenizer(context, return_tensors="pt").input_ids[0]
            ctx_ids = ctx_ids[:budget]
            truncated_context = tokenizer.decode(ctx_ids, skip_special_tokens=True)
            full_prompt = template.format(context=truncated_context, input=question)

        msgs = [{"role": "user", "content": full_prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        texts.append(text)
        all_answers.append(answers)
        actual_lengths.append(len(tokenizer(text).input_ids))

    avg_len = sum(actual_lengths) / len(actual_lengths)
    print(f"  {len(texts)} examples, avg prompt length: {avg_len:.0f} tokens "
          f"(target: {context_len})", flush=True)

    # tokenize with left-padding
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    tokenizer.pad_token    = tokenizer.eos_token

    batches = []
    for i in range(0, len(texts), batch_size):
        chunk   = texts[i:i + batch_size]
        ans     = all_answers[i:i + batch_size]
        encoded = tokenizer(chunk, return_tensors="pt", padding=True).to(device)
        batches.append((encoded, ans))

    tokenizer.padding_side = original_side
    return batches


# ---------------------------------------------------------------------------
# Single eval run
# ---------------------------------------------------------------------------

def run_eval(model, tokenizer, batches, k, device,
             temperature=0.0, max_new_tokens=64, same_path_baseline=False):
    """
    Generate answers for all batches with top-K=k (or full prefix if k is None).
    Returns (avg_f1, tok/s, n_evaluated, stats_dict).
    max_new_tokens=64: LongBench answers are short (1-5 words typically).
    """
    total_f1 = 0.0
    total    = 0
    total_new_tokens = 0
    t_start  = time.perf_counter()

    if k is not None:
        loader = TopKBlockLoader(model, k=k)
    elif same_path_baseline:
        loader = TopKBlockLoader(model, k=10000)
    else:
        loader = None

    if loader is not None:
        loader.install()

    sampler = types.MethodType(GenClass.batch_sample, model)

    try:
        for batch_idx, (encoded, ground_truths) in enumerate(batches):
            input_ids  = encoded.input_ids
            B          = input_ids.shape[0]
            prompt_len = input_ids.shape[1]
            seq_len_t  = torch.full((B,), prompt_len, dtype=torch.long, device=device)

            output = sampler(
                input_ids        = input_ids,
                tokenizer        = tokenizer,
                block_size       = BLOCK_SIZE,
                max_new_tokens   = max_new_tokens,
                small_block_size = 8,
                min_len          = prompt_len,
                seq_len          = seq_len_t,
                threshold        = 0.95,
                temperature      = temperature,
            )

            for b, answers in enumerate(ground_truths):
                gen_ids  = output[b][prompt_len:]
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                if total == 0 and batch_idx == 0 and b == 0:
                    print(f"  [debug] prompt_len={prompt_len} gen_tokens={len(gen_ids)}")
                    print(f"  [debug] generated: {gen_text[:150]!r}")
                    print(f"  [debug] gold: {answers!r}")
                f1 = score_answer(gen_text, answers)
                total_f1 += f1
                total    += 1
                total_new_tokens += len(gen_ids)

    finally:
        stats = {}
        if loader is not None:
            avg_n, avg_u, avg_frac = loader.avg_stats
            stats = {"avg_n": avg_n, "avg_union": avg_u, "avg_frac": avg_frac}
            loader.uninstall()

    elapsed  = time.perf_counter() - t_start
    avg_f1   = total_f1 / total if total > 0 else 0.0
    tps      = total_new_tokens / elapsed

    return avg_f1, tps, total, stats


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str,   default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    parser.add_argument("--device",      type=str,   default="cuda:0")
    parser.add_argument("--subset",      type=str,   default="qasper",
                        choices=["qasper", "multifieldqa_en", "2wikimqa", "hotpotqa", "narrativeqa"],
                        help="LongBench subset to evaluate")
    parser.add_argument("--context_len", type=int,   nargs="+", default=[2048],
                        help="Context length(s) to test (documents truncated to this length)")
    parser.add_argument("--batch_size",  type=int,   nargs="+", default=[8],
                        help="Batch size(s) to test")
    parser.add_argument("--k_values",    type=int,   nargs="+", default=[16],
                        help="K values to sweep")
    parser.add_argument("--n_examples",  type=int,   default=80,
                        help="Number of LongBench examples per condition")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="Max tokens to generate (LongBench answers are short)")
    parser.add_argument("--same_path_baseline", action="store_true",
                        help="Run baseline through loader with K=10000 (same code path)")
    args = parser.parse_args()

    print("Loading model...")
    model     = load_model(args.model, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    device    = args.device

    print(f"\nSubset: {args.subset}")

    all_results = {}

    for ctx_len in args.context_len:
        for batch_size in args.batch_size:
            print(f"\n{'='*60}")
            print(f"context_len={ctx_len}  batch_size={batch_size}")
            print(f"{'='*60}")

            torch.cuda.empty_cache()

            batches = load_longbench(
                args.subset, tokenizer, device,
                context_len=ctx_len,
                batch_size=batch_size,
                n_examples=args.n_examples,
            )

            k_values = [None] + list(args.k_values)
            results  = {}

            for k in k_values:
                label = str(k) if k is not None else "full"
                is_baseline = (k is None)
                print(f"  running K={label}...", flush=True)
                try:
                    f1, tps, n, stats = run_eval(
                        model, tokenizer, batches, k, device,
                        max_new_tokens=args.max_new_tokens,
                        same_path_baseline=(is_baseline and args.same_path_baseline),
                    )
                    results[k] = (f1, tps, stats)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"  K={label}: OOM — skip")
                    continue

            if None not in results:
                continue

            baseline_tps = results[None][1]
            baseline_f1  = results[None][0]

            print(f"\n  {'K':>6}  {'F1':>7}  {'F1 drop':>8}  {'tok/s':>8}  "
                  f"{'speedup':>9}  {'avg_N':>6}  {'union':>6}  {'frac':>6}")
            print("  " + "-" * 72)
            for k in k_values:
                if k not in results:
                    continue
                label = str(k) if k is not None else "full"
                f1, tps, stats = results[k]
                drop = f1 - baseline_f1
                if k is None:
                    speedup_str = "1.00x (baseline)"
                    stats_str   = f"{'—':>6}  {'—':>6}  {'—':>6}"
                else:
                    speedup_str = f"{tps/baseline_tps:.2f}x"
                    avg_n  = stats.get("avg_n", 0)
                    avg_u  = stats.get("avg_union", 0)
                    frac   = stats.get("avg_frac", 0)
                    stats_str = f"{avg_n:>6.1f}  {avg_u:>6.1f}  {frac:>5.1%}"
                print(f"  {label:>6}  {f1:>6.3f}  {drop:>+7.3f}  {tps:>8.1f}  "
                      f"{speedup_str:<18}  {stats_str}")

            all_results[(ctx_len, batch_size)] = results

    print(f"\n\nDone. Subset={args.subset}, n={args.n_examples} examples per condition.")


if __name__ == "__main__":
    main()
