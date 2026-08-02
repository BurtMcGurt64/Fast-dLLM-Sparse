"""
CONTENT-SELECTION ABLATION — does the commit-score oracle earn its keep, or does a
StreamingLLM sink+recency baseline already capture the accuracy?

The frac90 analysis showed attention MASS is ~85% sink+recency and the content routing
is diffuse. But mass != task accuracy (a retrieval answer can live in a low-mass block).
So the decisive test is F1, not attention. Four variants, all sharing the same pinned
sink+recency set (n_sink, n_recency):

  A. dense            — full KV. Reference F1.
  B. structural only  — keep ONLY sink+recency (budget = n_sink+n_recency), no content.
  C. oracle top-K     — sink+recency + commit-score-selected content, budget = frac*N.
  D. random content   — sink+recency + RANDOM content blocks, same budget as C.

Read the verdict:
  C ~= A  and  B << A  and  C > D   -> content matters AND the oracle picks it well.
                                       The contribution is real (mass was a red herring).
  B ~= A                            -> sink+recency alone suffices -> oracle is dead weight
                                       -> story collapses to "StreamingLLM on block diffusion".
  C ~= D                            -> any content works, oracle adds nothing.

Run (QA tasks where a specific block holds the answer):
    CUDA_VISIBLE_DEVICES=0 uv run v2/sparsity/ablate_content.py \
        --subsets narrativeqa qasper --context_lens 4096 8192 \
        --frac 0.2 --batch_size 8 --n_examples 80

    # multi-hop / retrieval stress:
    CUDA_VISIBLE_DEVICES=0 uv run v2/sparsity/ablate_content.py \
        --subsets 2wikimqa hotpotqa --context_lens 4096 8192 --frac 0.2
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../experiments"))

import time
import types
import argparse
import torch
from transformers import AutoTokenizer

from helper import load_model
from generation_functions import Fast_dLLM_QwenForCausalLM as GenClass
from topk import TopKBlockLoader
from eval_longbench_sweep import load_longbench_subset, score_answer, TEMPLATES, BLOCK_SIZE


# ---------------------------------------------------------------------------
# Random-content loader: identical pin/budget/tail logic to TopKBlockLoader,
# but content blocks are chosen at RANDOM instead of by commit-score. Sink and
# recency are still pinned, so this isolates ORACLE content vs ANY content.
# ---------------------------------------------------------------------------

class RandomBlockLoader(TopKBlockLoader):
    def _compute_selected_indices(self, prefix_kv_len, past_key_values):
        n_blocks = prefix_kv_len // self.block_size
        if n_blocks == 0:
            return None

        # free any captured queries (unused here) to avoid holding memory
        for layer in self.model.model.layers:
            layer.self_attn._commit_query_mean = None

        try:
            keys = past_key_values[0][0]
            B, device = keys.shape[0], keys.device
        except Exception:
            return None

        block_scores = torch.rand(B, n_blocks, device=device)

        pad = self._padding_n_blocks
        if pad is not None:
            for b in range(B):
                pad_b = min(pad[b], n_blocks)
                if pad_b > 0:
                    block_scores[b, :pad_b] = float("-inf")

        # pin recency + sink exactly as the parent does
        if self.n_recency > 0:
            block_scores[:, max(0, n_blocks - self.n_recency):] = float("inf")
        if self.n_sink > 0:
            for b in range(B):
                s = min(pad[b], n_blocks) if pad is not None else 0
                block_scores[b, s:min(s + self.n_sink, n_blocks)] = float("inf")
        self._padding_n_blocks = None

        k_actual = self._budget(n_blocks)
        k_actual = min(n_blocks, max(k_actual, min(self.n_sink + self.n_recency, n_blocks)))

        topk_block_ids = torch.topk(block_scores, k_actual, dim=1).indices.sort(dim=1).values
        offsets = torch.arange(self.block_size, device=device)
        token_indices = (
            topk_block_ids.unsqueeze(-1) * self.block_size + offsets
        ).reshape(B, k_actual * self.block_size)

        tail_start = n_blocks * self.block_size
        if tail_start < prefix_kv_len:
            tail = torch.arange(tail_start, prefix_kv_len, device=device).unsqueeze(0).expand(B, -1)
            token_indices = torch.cat([token_indices, tail], dim=1)

        self.last_n_blocks  = n_blocks
        self.last_selected  = k_actual
        self._n_commits    += 1
        self._sum_n_blocks += n_blocks
        self._sum_selected += k_actual
        return token_indices, topk_block_ids


# ---------------------------------------------------------------------------
# Variant factory
# ---------------------------------------------------------------------------

def make_loader(variant, model, frac, k, n_sink, n_recency):
    """Return a fresh loader instance for the variant, or None for dense."""
    if variant == "A_dense":
        return None
    if variant == "B_structural":
        # keep ONLY the pinned sink+recency: budget == their count, no content room
        return TopKBlockLoader(model, k=n_sink + n_recency, n_sink=n_sink, n_recency=n_recency)
    if variant == "C_oracle":
        if frac is not None:
            return TopKBlockLoader(model, frac=frac, n_sink=n_sink, n_recency=n_recency)
        return TopKBlockLoader(model, k=k, n_sink=n_sink, n_recency=n_recency)
    if variant == "D_random":
        if frac is not None:
            return RandomBlockLoader(model, frac=frac, n_sink=n_sink, n_recency=n_recency)
        return RandomBlockLoader(model, k=k, n_sink=n_sink, n_recency=n_recency)
    raise ValueError(variant)


VARIANTS = ["A_dense", "B_structural", "C_oracle", "D_random"]


# ---------------------------------------------------------------------------
# Eval one variant over the batches
# ---------------------------------------------------------------------------

def run_variant(model, tokenizer, batches, loader, device, max_new_tokens):
    total_f1, total, total_new = 0.0, 0, 0
    t0 = time.perf_counter()
    if loader is not None:
        loader.install()
    sampler = types.MethodType(GenClass.batch_sample, model)
    try:
        for encoded, gts in batches:
            input_ids  = encoded.input_ids
            B          = input_ids.shape[0]
            prompt_len = input_ids.shape[1]
            seq_len_t  = torch.full((B,), prompt_len, dtype=torch.long, device=device)
            output = sampler(
                input_ids=input_ids, tokenizer=tokenizer, block_size=BLOCK_SIZE,
                max_new_tokens=max_new_tokens, small_block_size=8, min_len=prompt_len,
                seq_len=seq_len_t, threshold=0.95, temperature=0.0,
            )
            for b, answers in enumerate(gts):
                gen_ids  = output[b][prompt_len:]
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                total_f1 += score_answer(gen_text, answers)
                total    += 1
                total_new += len(gen_ids)
    finally:
        stats = {}
        if loader is not None:
            n, u, fr = loader.avg_stats
            stats = {"avg_n": n, "avg_selected": u, "avg_frac": fr}
            loader.uninstall()
    elapsed = time.perf_counter() - t0
    stats["f1"] = total_f1 / total if total else 0.0
    stats["n"] = total
    stats["avg_gen_tokens"] = total_new / total if total else 0.0
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         type=str, default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    parser.add_argument("--device",        type=str, default="cuda:0")
    parser.add_argument("--subsets",       type=str, nargs="+", default=["narrativeqa", "qasper"],
                        choices=list(TEMPLATES.keys()))
    parser.add_argument("--context_lens",  type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--frac",          type=float, default=0.2,
                        help="content budget as fraction of N (C and D). Set --k to use fixed k instead.")
    parser.add_argument("--k",             type=int, default=None,
                        help="fixed content budget (overrides --frac for C and D)")
    parser.add_argument("--n_sink",        type=int, default=2)
    parser.add_argument("--n_recency",     type=int, default=2)
    parser.add_argument("--batch_size",    type=int, default=8)
    parser.add_argument("--n_examples",    type=int, default=80)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    frac = None if args.k is not None else args.frac
    torch.manual_seed(args.seed)

    print("Loading model...")
    model     = load_model(args.model, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    device    = args.device
    max_pos   = getattr(model.config, "max_position_embeddings", 32768)
    ctx_lens  = [c for c in args.context_lens if c <= max_pos]

    budget_str = f"frac={args.frac}" if frac is not None else f"k={args.k}"
    print(f"Budget for C/D: {budget_str} + pinned sink={args.n_sink} recency={args.n_recency}\n")

    results = {}  # (subset, ctx) -> {variant: stats}

    for subset in args.subsets:
        for ctx_len in ctx_lens:
            print(f"\n{'='*70}\n{subset}  ctx={ctx_len}  B={args.batch_size}  n={args.n_examples}\n{'='*70}")
            torch.cuda.empty_cache()
            try:
                batches = load_longbench_subset(subset, tokenizer, device, ctx_len,
                                                args.batch_size, args.n_examples, args.seed)
            except Exception as e:
                print(f"  load failed: {e}")
                continue

            row = {}
            for variant in VARIANTS:
                torch.manual_seed(args.seed)  # keep D reproducible & data order fixed
                loader = make_loader(variant, model, frac, args.k, args.n_sink, args.n_recency)
                try:
                    stats = run_variant(model, tokenizer, batches, loader, device, args.max_new_tokens)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"  {variant}: OOM")
                    continue
                row[variant] = stats
                fr = stats.get("avg_frac")
                fr_str = f"  frac_kept={fr:.1%}" if fr is not None else ""
                print(f"  {variant:<14} F1={stats['f1']:.3f}  gen={stats['avg_gen_tokens']:.0f}tok"
                      f"{fr_str}  (n={stats['n']})")
            results[(subset, ctx_len)] = row

    # ------------------------------------------------------------------ tables
    print(f"\n\n{'='*70}\nF1 ABLATION SUMMARY\n{'='*70}")
    print(f"{'subset':<14}{'ctx':>6}  {'A dense':>8}  {'B struct':>8}  {'C oracle':>8}  "
          f"{'D random':>8}   {'B-A':>6}  {'C-A':>6}  {'C-D':>6}")
    print("-" * 92)
    for (subset, ctx_len), row in results.items():
        def g(v):
            return row.get(v, {}).get("f1")
        a, b, c, d = g("A_dense"), g("B_structural"), g("C_oracle"), g("D_random")
        def fmt(x):
            return f"{x:.3f}" if x is not None else "  —  "
        def dl(x, y):
            return f"{x - y:+.3f}" if (x is not None and y is not None) else "  —  "
        print(f"{subset:<14}{ctx_len:>6}  {fmt(a):>8}  {fmt(b):>8}  {fmt(c):>8}  {fmt(d):>8}   "
              f"{dl(b, a):>6}  {dl(c, a):>6}  {dl(c, d):>6}")

    # ------------------------------------------------------------------ verdict
    print(f"\n{'='*70}\nVERDICT\n{'='*70}")
    print("  B-A  (structural-only vs dense): if strongly negative, CONTENT matters for accuracy.")
    print("  C-A  (oracle vs dense):          near 0 means the method preserves quality.")
    print("  C-D  (oracle vs random content): if positive, the ORACLE beats keeping random blocks.")
    print("\n  -> Oracle earns its keep iff  B-A is clearly negative AND C-D is clearly positive.")
    print("  -> If B ~= A, sink+recency alone suffices and the oracle is not the contribution.")


if __name__ == "__main__":
    main()
