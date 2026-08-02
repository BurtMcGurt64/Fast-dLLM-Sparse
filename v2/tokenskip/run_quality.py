"""
Quality vs lock-rate sweep for FFN-freeze skipping on GSM8K.

Answers: how much can we skip (per metric/threshold) WITHOUT hurting accuracy, and is cosine or l2
safer per lock? Lock rate only matters if quality holds.

Uses the SAME few-shot GSM8K harness as prune_probe / nfe_probe_stage1 (dense ≈ 0.71) so the drops
are measured against a healthy baseline — NOT the zero-shot chat template (which lands ~0.31 and is
too noisy to read).

    CUDA_VISIBLE_DEVICES=<gpu> python v2/tokenskip/run_quality.py --n 150
    # plain `python` with venv active + LD_LIBRARY_PATH set, NOT `uv run`
"""
import sys, os, types, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "../experiments"))
sys.path.insert(0, os.path.join(HERE, "../train_scripts"))
import torch
from transformers import AutoTokenizer
from helper import load_model
from generation_functions import Fast_dLLM_QwenForCausalLM as GenClass
from nfe_probe_stage1 import EVALS               # few-shot gsm8k harness (dense ~0.71)
from ffn_skip import FFNSkipper

GRID = {
    "cosine":     [1e-3, 5e-3, 1e-2, 5e-2],
    "l2":         [1e-2, 5e-2, 1e-1, 2e-1],
    "confidence": [0.90, 0.95, 0.99],       # FREE baseline: lock when predicted-token conf >= thr
}


@torch.no_grad()
def eval_gsm8k(model, tok, sampler, data, ev, gen, device):
    correct = 0
    for q, gold in data:
        ids = tok(ev["prompt"](q), return_tensors="pt").input_ids.to(device)
        plen = ids.shape[1]
        out = sampler(input_ids=ids, tokenizer=tok, block_size=32, max_new_tokens=gen,
                      small_block_size=8, min_len=plen,
                      seq_len=torch.tensor([plen], device=device),
                      threshold=0.95, temperature=0.0, use_block_cache=True)
        pred = ev["parse_pred"](tok.decode(out[0][plen:], skip_special_tokens=True))
        correct += int(pred is not None and pred == gold)
    return correct / len(data) if data else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--gen", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print("Loading model...", flush=True)
    model = load_model(args.model, device=args.device).eval()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    sampler = types.MethodType(GenClass.batch_sample, model)

    ev = EVALS["gsm8k"]
    data = ev["load"](args.n, args.seed)          # [(question, gold)], few-shot prompt via ev["prompt"]

    print(f"\nDense baseline (few-shot, n={args.n})...", flush=True)
    dense = eval_gsm8k(model, tok, sampler, data, ev, args.gen, args.device)
    print(f"  dense acc = {dense:.3f}", flush=True)

    rows = []
    for metric, thrs in GRID.items():
        for thr in thrs:
            sk = FFNSkipper(model, metric=metric, threshold=thr).install()
            try:
                acc = eval_gsm8k(model, tok, sampler, data, ev, args.gen, args.device)
                lr = sk.mean_lock_rate()
            finally:
                sk.uninstall()
            rows.append((metric, thr, lr, acc))
            print(f"  {metric:>6} thr={thr:<7} lock={lr:.2f}  acc={acc:.3f}  "
                  f"drop={dense-acc:+.3f}", flush=True)

    print(f"\n{'='*60}\nQUALITY vs LOCK-RATE (GSM8K few-shot, n={args.n}, dense={dense:.3f})\n{'='*60}")
    print(f"  {'metric':>6} {'thr':>8} {'lock':>6} {'acc':>7} {'drop':>7}")
    for metric, thr, lr, acc in rows:
        print(f"  {metric:>6} {thr:>8} {lr:>6.2f} {acc:>7.3f} {dense-acc:>+7.3f}")
    print("\n  Read: highest lock rate with drop ~noise (<0.02) = usable skip rate. Compare cosine vs")
    print("  l2 at EQUAL lock — the one that holds accuracy is the safer signal (expect l2 > cosine).")


if __name__ == "__main__":
    main()
