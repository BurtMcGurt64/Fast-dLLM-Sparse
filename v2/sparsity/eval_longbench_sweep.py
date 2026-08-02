"""
LongBench length sweep: frac vs context length at fixed K.

Primary metric: frac = (union of selected blocks) / N at K=16.
If frac drops as context grows, top-K has value on real diverse data.
If frac is flat at ~70%, batched block sparsity is bounded by cross-sequence union.

Run:
    # dense baseline + K=16 at 2k/4k/8k, qasper + narrativeqa, B=8
    CUDA_VISIBLE_DEVICES=2 uv run v2/sparsity/eval_longbench_sweep.py

    # custom sweep
    CUDA_VISIBLE_DEVICES=2 uv run v2/sparsity/eval_longbench_sweep.py \
        --subsets qasper narrativeqa \
        --context_lens 2048 4096 8192 \
        --k 16 --batch_size 8 --n_examples 80
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../experiments"))

import re
import time
import types
import argparse
import zipfile
import shutil
import torch
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
from datasets import load_dataset
from transformers import AutoTokenizer

from helper import load_model
from generation_functions import Fast_dLLM_QwenForCausalLM as GenClass
from topk import TopKBlockLoader

BLOCK_SIZE = 32

TEMPLATES = {
    "qasper": (
        "You are given a scientific paper and a question. Answer the question "
        "in as few words as possible based on the paper.\n\n"
        "Paper:\n{context}\n\nQuestion: {input}\n\nAnswer:"
    ),
    "narrativeqa": (
        "You are given a story and a question. Answer the question as concisely "
        "as possible.\n\nStory:\n{context}\n\nQuestion: {input}\n\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. "
        "Only give me the answer and do not output any other words.\n\n"
        "Passages:\n{context}\n\nQuestion: {input}\n\nAnswer:"
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
    "gov_report": (
        "You are given a government report. Write a detailed summary of the report.\n\n"
        "Report:\n{context}\n\nSummary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query. "
        "Answer the query by summarizing the relevant parts of the transcript.\n\n"
        "Transcript:\n{context}\n\nQuery: {input}\n\nAnswer:"
    ),
    "multi_news": (
        "You are given several news articles. Write a summary of all the articles.\n\n"
        "Articles:\n{context}\n\nSummary:"
    ),
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def normalize(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


def f1_score(pred, gold):
    pred_t = normalize(pred).split()
    gold_t = normalize(gold).split()
    if not pred_t or not gold_t:
        return int(pred_t == gold_t)
    common = set(pred_t) & set(gold_t)
    if not common:
        return 0.0
    p = len(common) / len(pred_t)
    r = len(common) / len(gold_t)
    return 2 * p * r / (p + r)


def score_answer(prediction, answers):
    if not answers:
        return 0.0
    pred_norm = normalize(prediction[:300])
    return max(f1_score(pred_norm, normalize(a)) for a in answers)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_longbench_subset(subset, tokenizer, device, context_len, batch_size,
                           n_examples, seed=42):
    cache_dir = os.path.join(os.path.dirname(__file__), ".longbench_cache")
    jsonl_path = os.path.join(cache_dir, f"{subset}.jsonl")

    if not os.path.exists(jsonl_path):
        os.makedirs(cache_dir, exist_ok=True)
        zip_path = os.path.join(cache_dir, "data.zip")
        if not os.path.exists(zip_path):
            print("  Downloading LongBench data.zip from HuggingFace...", flush=True)
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id="THUDM/LongBench", filename="data.zip",
                repo_type="dataset", local_dir=cache_dir,
            )
        print("  Extracting data.zip...", flush=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(cache_dir)
        extracted = os.path.join(cache_dir, "data", f"{subset}.jsonl")
        if os.path.exists(extracted):
            shutil.move(extracted, jsonl_path)

    ds = load_dataset("json", data_files=jsonl_path, split="train")
    ds = ds.shuffle(seed=seed)
    if n_examples and n_examples < len(ds):
        ds = ds.select(range(n_examples))

    template = TEMPLATES.get(subset, TEMPLATES["qasper"])
    texts, all_answers, actual_lengths = [], [], []

    for ex in ds:
        context  = ex.get("context", "") or ex.get("input", "")
        question = ex.get("input", "")
        answers  = ex.get("answers", [])
        if isinstance(answers, str):
            answers = [answers]

        full_prompt = template.format(context=context, input=question)
        full_ids = tokenizer(full_prompt, return_tensors="pt").input_ids[0]

        if len(full_ids) > context_len:
            parts = template.split("{context}")
            prefix_tmpl = parts[0]
            suffix_tmpl = parts[1].format(input=question)
            prefix_len  = len(tokenizer(prefix_tmpl, return_tensors="pt").input_ids[0])
            suffix_len  = len(tokenizer(suffix_tmpl, return_tensors="pt").input_ids[0])
            budget      = max(64, context_len - prefix_len - suffix_len)
            ctx_ids     = tokenizer(context, return_tensors="pt").input_ids[0][:budget]
            context     = tokenizer.decode(ctx_ids, skip_special_tokens=True)
            full_prompt = template.format(context=context, input=question)

        msgs = [{"role": "user", "content": full_prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        texts.append(text)
        all_answers.append(answers)
        actual_lengths.append(len(tokenizer(text).input_ids))

    avg_len = sum(actual_lengths) / len(actual_lengths)
    print(f"  {len(texts)} examples, avg prompt={avg_len:.0f} tok (target {context_len})",
          flush=True)

    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

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

def run_eval(model, tokenizer, batches, k, device, max_new_tokens=64,
             same_path_baseline=False):
    """
    Returns (avg_f1, tps, n, stats).
    k=None → dense baseline (no loader installed).
    same_path_baseline: when k=None, install loader with K=10000 so baseline
      runs the same commit-step capture path — speedup then measures pruning only.
    """
    total_f1 = 0.0
    total = 0
    total_new_tokens = 0
    t_start = time.perf_counter()

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
                temperature      = 0.0,
            )

            for b, answers in enumerate(ground_truths):
                gen_ids  = output[b][prompt_len:]
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                if total == 0 and batch_idx == 0 and b == 0:
                    gold_str = answers[0][:60] if answers else "?"
                    print(f"    [debug] prompt_len={prompt_len} gen={len(gen_ids)} tok  "
                          f"pred={gen_text[:80]!r}  gold={gold_str!r}")
                total_f1 += score_answer(gen_text, answers)
                total    += 1
                total_new_tokens += len(gen_ids)

    finally:
        stats = {}
        if loader is not None:
            avg_n, avg_u, avg_frac = loader.avg_stats
            stats = {"avg_n": avg_n, "avg_union": avg_u, "avg_frac": avg_frac}
            loader.uninstall()

    elapsed = time.perf_counter() - t_start
    avg_f1  = total_f1 / total if total > 0 else 0.0
    tps     = total_new_tokens / elapsed
    # wall_per_example is the correct speedup metric when generation lengths differ
    # across conditions (tps conflates per-step speed with how many tokens were generated)
    wall_per_example = elapsed / total if total > 0 else 0.0
    stats["total_new_tokens"] = total_new_tokens
    stats["wall_per_example"] = wall_per_example
    stats["avg_gen_tokens"]   = total_new_tokens / total if total > 0 else 0.0
    return avg_f1, tps, total, stats


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        type=str,   default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    parser.add_argument("--device",       type=str,   default="cuda:0")
    parser.add_argument("--subsets",      type=str,   nargs="+",
                        default=["qasper", "narrativeqa"],
                        choices=list(TEMPLATES.keys()))
    parser.add_argument("--context_lens", type=int,   nargs="+", default=[2048, 4096, 8192])
    parser.add_argument("--k",            type=int,   default=16,
                        help="K for top-K block loading (fixed across sweep)")
    parser.add_argument("--batch_size",   type=int,   default=8)
    parser.add_argument("--n_examples",   type=int,   default=80)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--same_path_baseline", action="store_true",
                        help="Run baseline through loader with K=10000 so speedup "
                             "reflects pruning only, not commit-step overhead difference.")
    args = parser.parse_args()

    print("Loading model...")
    model     = load_model(args.model, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    device    = args.device

    # sanity check: verify max_position_embeddings
    max_pos = getattr(model.config, 'max_position_embeddings', None)
    print(f"\nmodel.config.max_position_embeddings = {max_pos}")
    for ctx in args.context_lens:
        if max_pos and ctx > max_pos:
            print(f"  WARNING: context_len={ctx} exceeds max_position_embeddings={max_pos} — skipping")

    valid_ctx = [c for c in args.context_lens if not max_pos or c <= max_pos]

    # results[subset][ctx_len] = {None: (f1, tps, stats), k: (f1, tps, stats)}
    all_results = {}

    for subset in args.subsets:
        print(f"\n{'='*70}")
        print(f"Subset: {subset}  K={args.k}  B={args.batch_size}  n={args.n_examples}")
        print(f"{'='*70}")
        all_results[subset] = {}

        for ctx_len in valid_ctx:
            print(f"\n  --- context_len={ctx_len} ---", flush=True)
            torch.cuda.empty_cache()

            try:
                batches = load_longbench_subset(
                    subset, tokenizer, device, ctx_len,
                    args.batch_size, args.n_examples, args.seed
                )
            except Exception as e:
                print(f"  Load failed: {e}")
                continue

            row = {}

            # dense baseline first — if F1 collapses here, model can't handle this length
            print(f"  running dense baseline...", flush=True)
            try:
                f1, tps, n, stats = run_eval(
                    model, tokenizer, batches, k=None, device=device,
                    max_new_tokens=args.max_new_tokens,
                    same_path_baseline=args.same_path_baseline,
                )
                row[None] = (f1, tps, n, stats)
                print(f"  dense baseline: F1={f1:.3f}  ({n} examples)")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  OOM at dense baseline for ctx={ctx_len} — skipping length")
                continue

            # top-K run
            print(f"  running K={args.k}...", flush=True)
            try:
                f1, tps, n, stats = run_eval(
                    model, tokenizer, batches, k=args.k, device=device,
                    max_new_tokens=args.max_new_tokens
                )
                row[args.k] = (f1, tps, n, stats)
                avg_n   = stats.get("avg_n", 0)
                avg_u   = stats.get("avg_union", 0)
                avg_frac = stats.get("avg_frac", 0)
                print(f"  K={args.k}: F1={f1:.3f}  N={avg_n:.1f}  union={avg_u:.1f}  frac={avg_frac:.1%}")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  OOM at K={args.k} for ctx={ctx_len}")

            all_results[subset][ctx_len] = row

        # ---------------------------------------------------------------
        # Per-subset summary table
        # ---------------------------------------------------------------
        print(f"\n  === {subset}: frac vs context length (K={args.k}) ===")
        print(f"  {'ctx':>6}  {'dense_F1':>9}  {'topK_F1':>9}  {'F1_drop':>8}  "
              f"{'spd(wall)':>10}  {'spd(tps)':>9}  {'gen_tok':>8}  {'frac':>7}")
        print("  " + "-" * 84)

        prev_f1_dense = None
        for ctx_len in valid_ctx:
            if ctx_len not in all_results[subset]:
                continue
            row = all_results[subset][ctx_len]
            if None not in row:
                continue

            dense_f1, dense_tps, _, dense_stats = row[None]
            dense_wall = dense_stats.get("wall_per_example", 0)
            dense_gen  = dense_stats.get("avg_gen_tokens", 0)
            collapse_flag = ""
            if prev_f1_dense is not None and dense_f1 < prev_f1_dense - 0.05:
                collapse_flag = " ← F1 collapse"
            prev_f1_dense = dense_f1

            if args.k in row:
                kf1, ktps, _, kstats = row[args.k]
                drop       = kf1 - dense_f1
                spd_tps    = ktps / dense_tps
                spd_wall   = dense_wall / kstats.get("wall_per_example", 1)
                avg_n      = kstats.get("avg_n", 0)
                avg_u      = kstats.get("avg_union", 0)
                frac       = kstats.get("avg_frac", 0)
                kgen       = kstats.get("avg_gen_tokens", 0)
                gen_str    = f"{dense_gen:.0f}→{kgen:.0f}"
                print(f"  {ctx_len:>6}  {dense_f1:>9.3f}  {kf1:>9.3f}  {drop:>+7.3f}  "
                      f"  {spd_wall:>8.2f}x  {spd_tps:>7.2f}x  {gen_str:>8}  {frac:>6.1%}{collapse_flag}")
            else:
                print(f"  {ctx_len:>6}  {dense_f1:>9.3f}  {'OOM':>9}  {'—':>8}  "
                      f"  {'—':>10}  {'—':>9}  {'—':>8}  {'—':>6}{collapse_flag}")

    # ---------------------------------------------------------------
    # Cross-subset frac summary — the key plot in text form
    # ---------------------------------------------------------------
    print(f"\n\n{'='*70}")
    print(f"KEY RESULT: frac vs context_len at K={args.k}  (primary metric)")
    print(f"If frac decreases as ctx grows → top-K value scales with context.")
    print(f"If frac flat at ~70% → union saturates, batched top-K bounded on diverse data.")
    print(f"{'='*70}")
    print(f"\n  {'ctx':>6}", end="")
    for subset in args.subsets:
        col = f"  {subset[:12]:>14}"
        print(col, end="")
    print()
    print("  " + "-" * (8 + 16 * len(args.subsets)))

    for ctx_len in valid_ctx:
        print(f"  {ctx_len:>6}", end="")
        for subset in args.subsets:
            row = all_results.get(subset, {}).get(ctx_len, {})
            if args.k in row:
                frac = row[args.k][3].get("avg_frac", 0)
                print(f"  {frac:>13.1%}", end="")
            else:
                print(f"  {'—':>13}", end="")
        print()

    print(f"\n  (n={args.n_examples} examples/condition, B={args.batch_size}, seed={args.seed})")


if __name__ == "__main__":
    main()
