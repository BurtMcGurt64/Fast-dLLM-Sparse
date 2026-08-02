"""
Measures attention sparsity at the block level across (batch, seq_len) operating points.

For each setting, reports how many of the N prefix blocks are needed to capture
X% of attention mass — the K that top-K block loading would need to use.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from helper import (load_model, load_gsm8k_stacked, enable_attn_capture,
                    disable_attn_capture, collect_attn_weights, generate_and_check)
from transformers import AutoTokenizer

BLOCK_SIZE           = 32
THRESHOLDS           = [0.50, 0.80, 0.90, 0.95]
BATCH_SIZES          = [1, 8, 16, 32]
QUESTIONS_PER_PROMPT = [2, 4, 8, 16]  # ~260, 520, 1040, 2080 tokens


def measure_sparsity(weights, thresholds):
    """
    Given weights dict {layer_idx: [B, H, Q, KV]}, compute per-threshold
    mean number of blocks needed to reach each coverage level.
    Averages across layers, heads, query positions, and batch.
    Returns {threshold: mean_K} and num_blocks.
    """
    all_blocks_needed = {t: [] for t in thresholds}
    num_blocks = None

    for w in weights.values():
        B, H, Q, KV = w.shape

        n_blocks = KV // BLOCK_SIZE
        if n_blocks == 0:
            continue
        num_blocks = n_blocks

        # trim to multiple of BLOCK_SIZE and sum within blocks
        w_trimmed = w[..., :n_blocks * BLOCK_SIZE]
        block_mass = w_trimmed.reshape(B, H, Q, n_blocks, BLOCK_SIZE).sum(dim=-1)
        # block_mass: [B, H, Q, n_blocks]

        sorted_mass = block_mass.sort(dim=-1, descending=True).values
        cumsum = sorted_mass.cumsum(dim=-1)  # [B, H, Q, n_blocks]

        for t in thresholds:
            # number of blocks needed to exceed threshold t
            k = (cumsum < t).sum(dim=-1) + 1  # [B, H, Q]
            all_blocks_needed[t].append(k.float().mean().item())

    return {t: sum(v) / len(v) for t, v in all_blocks_needed.items() if v}, num_blocks


def run_sweep(model, tokenizer, device, batch_sizes, questions_per_prompt_list):
    header = (f"{'batch':>6}  {'q_per_prompt':>13}  {'seq_len':>8}  {'N_blocks':>9}  "
              + "  ".join(f"K@{int(t*100)}%" for t in THRESHOLDS))
    print(header)
    print("-" * len(header))

    for questions_per_prompt in questions_per_prompt_list:
        for batch_size in batch_sizes:
            # need enough questions for at least one full batch
            n_questions = questions_per_prompt * batch_size
            batches = load_gsm8k_stacked(
                n_questions=n_questions,
                questions_per_prompt=questions_per_prompt,
                tokenizer=tokenizer,
                device=device,
                batch_size=batch_size,
            )
            batch = batches[0]
            seq_len = batch.input_ids.shape[1]

            torch.cuda.empty_cache()
            enable_attn_capture(model)
            try:
                with torch.no_grad():
                    model(
                        input_ids=batch.input_ids,
                        attention_mask=batch.attention_mask,
                        use_cache=True,
                        update_past_key_values=True,
                    )
            except torch.cuda.OutOfMemoryError:
                disable_attn_capture(model)
                torch.cuda.empty_cache()
                print(f"  {batch_size:>6}  {questions_per_prompt:>13}  {seq_len:>8}  OOM")
                continue
            weights = collect_attn_weights(model)
            disable_attn_capture(model)

            results, num_blocks = measure_sparsity(weights, THRESHOLDS)
            if num_blocks is None:
                print(f"  {batch_size:>6}  {questions_per_prompt:>13}  {seq_len:>8}  (too short)")
                continue

            k_strs = "  ".join(f"{results[t]:>6.1f}/{num_blocks}" for t in THRESHOLDS)
            print(f"  {batch_size:>6}  {questions_per_prompt:>13}  {seq_len:>8}  {num_blocks:>9}  {k_strs}")

            # accuracy check: run generation on one example (batch=1 only, others too slow)
            if batch_size == 1:
                correctness = generate_and_check(
                    model, tokenizer, questions_per_prompt,
                    question_offset=0, device=device,
                )
                symbols = ["✓" if c else ("✗" if c is False else "?") for c in correctness]
                print(f"  {'':>6}  {'accuracy check':>13}  " + "  ".join(f"Q{i+1}:{s}" for i, s in enumerate(symbols)))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",                type=str, default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    parser.add_argument("--device",               type=str, default="cuda:0")
    parser.add_argument("--batch_sizes",          type=int, nargs="+", default=BATCH_SIZES)
    parser.add_argument("--questions_per_prompt", type=int, nargs="+", default=QUESTIONS_PER_PROMPT)
    args = parser.parse_args()

    print("Loading model...")
    model = load_model(args.model, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    run_sweep(model, tokenizer, args.device, args.batch_sizes, args.questions_per_prompt)


if __name__ == "__main__":
    main()
