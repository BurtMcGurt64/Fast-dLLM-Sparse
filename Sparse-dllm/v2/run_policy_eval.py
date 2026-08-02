"""
Standalone driver for Xinze's token-policy method — NO lm_eval (bypasses the harness version drift).
Measures GSM8K accuracy + wall-clock tok/s for ORG vs token-policy, with cosine vs l2 staleness.

Run from Sparse-dllm/v2/ (venv active, LD_LIBRARY_PATH set), e.g.:
    CUDA_VISIBLE_DEVICES=<gpu> python run_policy_eval.py --n 60 --gen 256
"""
import sys, os, types, re, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

# --- main-repo compat patch: this model's DynamicCache uses .layers[i].keys/values ---
from transformers.cache_utils import DynamicCache
DynamicCache.__getitem__ = lambda self, i: (self.layers[i].keys, self.layers[i].values)
class _LayerProxy:
    def __init__(self, c, a): self._c, self._a = c, a
    def __getitem__(self, i): return getattr(self._c.layers[i], self._a)
    def __setitem__(self, i, v): setattr(self._c.layers[i], self._a, v)
    def __len__(self): return len(self._c.layers)
DynamicCache.key_cache = property(lambda self: _LayerProxy(self, "keys"))
DynamicCache.value_cache = property(lambda self: _LayerProxy(self, "values"))

from transformers import AutoTokenizer
from model.modeling import Fast_dLLM_QwenForCausalLM
import generation_functions as GF

GSM8K_FEWSHOT = """Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did she sell altogether in April and May?
Answer: In May she sold 48/2 = 24 clips. Altogether she sold 48 + 24 = 72 clips. #### 72

Question: Weng earns $12 an hour for babysitting. Yesterday she did 50 minutes of babysitting. How much did she earn?
Answer: Per minute she earns 12/60 = $0.2. For 50 minutes she earned 0.2 * 50 = $10. #### 10

Question: Betty is saving for a $100 wallet. She has half the money she needs. Her parents give her $15 and her grandparents twice as much as her parents. How much more does she need?
Answer: Betty has 100/2 = $50. Grandparents give 15*2 = $30. Now she has 50+15+30 = $95. She needs 100-95 = $5 more. #### 5

Question: Julie is reading a 120-page book. Yesterday she read 12 pages and today twice as many. If she wants to read half the remaining pages tomorrow, how many should she read?
Answer: Today she read 12*2 = 24. So far 12+24 = 36. Remaining 120-36 = 84. Half is 84/2 = 42. #### 42

"""

def gold(a):
    m = re.search(r"####\s*(-?[\d,]+)", a); return m.group(1).replace(",", "") if m else None
def pred(t):
    m = re.search(r"####\s*(-?[\d,]+)", t)
    if m: return m.group(1).replace(",", "")
    n = re.findall(r"-?[\d,]+", t); return n[-1].replace(",", "") if n else None


@torch.no_grad()
def run(model, tok, sampler, data, device, gen, **pk):
    correct, ntok = 0, 0
    torch.cuda.synchronize(device); t0 = time.perf_counter()
    for q, g in data:
        ids = tok(GSM8K_FEWSHOT + f"Question: {q}\nAnswer:", return_tensors="pt").input_ids.to(device)
        plen = ids.shape[1]
        out = sampler(input_ids=ids, tokenizer=tok, block_size=32, max_new_tokens=gen,
                      small_block_size=8, min_len=plen,
                      seq_len=torch.tensor([plen], device=device),
                      threshold=0.95, temperature=0.0, use_block_cache=True, **pk)
        seq = out[0] if not isinstance(out, tuple) else out[0][0]
        g_ids = seq[plen:]
        ntok += int((g_ids != tok.pad_token_id).sum()) if tok.pad_token_id is not None else g_ids.numel()
        correct += int((lambda p: p is not None and p == g)(pred(tok.decode(g_ids, skip_special_tokens=True))))
    torch.cuda.synchronize(device); dt = time.perf_counter() - t0
    return correct / len(data), ntok / dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--gen", type=int, default=256)
    p.add_argument("--freeze", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # from_pretrained SILENTLY SKIPS weight loading for this model -> manual load_state_dict.
    print("Loading model (manual state_dict load)...", flush=True)
    import json
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from model.configuration import Fast_dLLM_QwenConfig
    snap = snapshot_download(args.model, local_files_only=True)
    config = Fast_dLLM_QwenConfig.from_pretrained(snap)
    model = Fast_dLLM_QwenForCausalLM(config).to(torch.bfloat16)
    with open(os.path.join(snap, "model.safetensors.index.json")) as f:
        index = json.load(f)
    sd = {}
    for shard in sorted(set(index["weight_map"].values())):
        sd.update(load_file(os.path.join(snap, shard), device="cpu"))
    model.load_state_dict(sd, strict=True)
    model = model.to(args.device).eval()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test").shuffle(seed=args.seed).select(range(args.n))
    data = [(ex["question"], gold(ex["answer"])) for ex in ds]

    org = types.MethodType(GF.Fast_dLLM_QwenForCausalLM.batch_sample, model)
    tokp = types.MethodType(GF.Fast_dLLM_QwenForCausalLM.batch_sample_token_policy_flashattention, model)

    print(f"\nORG baseline (few-shot, n={args.n})...", flush=True)
    org_acc, org_tps = run(model, tok, org, data, args.device, args.gen)
    print(f"  ORG  acc={org_acc:.3f}  tok/s={org_tps:.1f}", flush=True)

    GRID = {"cosine": [0.95, 0.99], "l2": [0.02, 0.05, 0.1]}
    rows = []
    for metric, thrs in GRID.items():
        GF.set_staleness_metric(metric)
        for thr in thrs:
            acc, tps = run(model, tok, tokp, data, args.device, args.gen,
                           token_policy_threshold=thr, token_policy_freeze_step=args.freeze)
            spd = tps / org_tps if org_tps else 0.0
            rows.append((metric, thr, acc, tps, spd))
            print(f"  token/{metric} thr={thr:<6} acc={acc:.3f}  tok/s={tps:.1f}  "
                  f"speedup={spd:.2f}x  drop={org_acc-acc:+.3f}", flush=True)

    print(f"\n{'='*66}\nTOKEN POLICY: ORG acc={org_acc:.3f} tok/s={org_tps:.1f} (GSM8K n={args.n})\n{'='*66}")
    print(f"  {'metric':>7} {'thr':>6} {'acc':>7} {'tok/s':>8} {'speedup':>8} {'drop':>7}")
    for m, thr, acc, tps, spd in rows:
        print(f"  {m:>7} {thr:>6} {acc:>7.3f} {tps:>8.1f} {spd:>7.2f}x {org_acc-acc:>+7.3f}")
    print("\n  Read: which metric holds accuracy at the highest speedup. Wall-clock speedup at")
    print("  batch=1 will be modest (memory-bound); the accuracy/skip tradeoff is the main signal.")


if __name__ == "__main__":
    main()
