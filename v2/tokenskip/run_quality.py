"""
Quality vs lock-rate sweep for FFN-freeze skipping on GSM8K.

Answers: how much can we skip (per metric/threshold) WITHOUT hurting accuracy, and is cosine or l2
safer per lock? This is the decisive experiment — lock rate only matters if quality holds.

    CUDA_VISIBLE_DEVICES=<gpu> python v2/tokenskip/run_quality.py --n 100
    # (use plain `python` with the venv active, not `uv run`, per the env notes)
"""
import sys, os, types, re, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "../experiments"))
import torch
from transformers import AutoTokenizer
from helper import load_model
from generation_functions import Fast_dLLM_QwenForCausalLM as GenClass
from ffn_skip import FFNSkipper

# metric -> thresholds to sweep (tight -> loose)
GRID = {
    "cosine": [1e-3, 5e-3, 1e-2, 5e-2],
    "l2":     [1e-2, 5e-2, 1e-1, 2e-1],
}


def gsm8k_gold(ans):
    m = re.search(r"####\s*(-?[\d,]+)", ans);  return m.group(1).replace(",", "") if m else None

def gsm8k_pred(text):
    m = re.search(r"####\s*(-?[\d,]+)", text)
    if m: return m.group(1).replace(",", "")
    nums = re.findall(r"-?[\d,]+", text);  return nums[-1].replace(",", "") if nums else None


@torch.no_grad()
def eval_gsm8k(model, tok, sampler, data, gen, device):
    correct = 0
    for q, gold in data:
        text = tok.apply_chat_template([{"role": "user", "content": q}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").input_ids.to(device)
        plen = ids.shape[1]
        out = sampler(input_ids=ids, tokenizer=tok, block_size=32, max_new_tokens=gen,
                      small_block_size=8, min_len=plen,
                      seq_len=torch.tensor([plen], device=device),
                      threshold=0.95, temperature=0.0, use_block_cache=True)
        pred = gsm8k_pred(tok.decode(out[0][plen:], skip_special_tokens=True))
        correct += int(pred is not None and pred == gold)
    return correct / len(data) if data else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--gen", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print("Loading model...", flush=True)
    model = load_model(args.model, device=args.device).eval()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    sampler = types.MethodType(GenClass.batch_sample, model)

    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test").shuffle(seed=args.seed).select(range(args.n))
    data = [(ex["question"], gsm8k_gold(ex["answer"])) for ex in ds]

    print(f"\nDense baseline (n={args.n})...", flush=True)
    dense = eval_gsm8k(model, tok, sampler, data, args.gen, args.device)
    print(f"  dense acc = {dense:.3f}", flush=True)

    rows = []
    for metric, thrs in GRID.items():
        for thr in thrs:
            sk = FFNSkipper(model, metric=metric, threshold=thr).install()
            try:
                acc = eval_gsm8k(model, tok, sampler, data, args.gen, args.device)
                lr = sk.mean_lock_rate()
            finally:
                sk.uninstall()
            rows.append((metric, thr, lr, acc))
            print(f"  {metric:>6} thr={thr:<7} lock={lr:.2f}  acc={acc:.3f}  "
                  f"drop={dense-acc:+.3f}", flush=True)

    print(f"\n{'='*60}\nQUALITY vs LOCK-RATE (GSM8K, n={args.n}, dense={dense:.3f})\n{'='*60}")
    print(f"  {'metric':>6} {'thr':>8} {'lock':>6} {'acc':>7} {'drop':>7}")
    for metric, thr, lr, acc in rows:
        print(f"  {metric:>6} {thr:>8} {lr:>6.2f} {acc:>7.3f} {dense-acc:>+7.3f}")
    print("\n  Read: find the HIGHEST lock rate whose drop is ~noise (<0.02). Compare cosine vs l2 at")
    print("  equal lock rate — whichever holds accuracy better is the safer signal (l2 should win if")
    print("  cosine over-locks on magnitude). That safe lock rate is what a real speedup would build on.")


if __name__ == "__main__":
    main()
