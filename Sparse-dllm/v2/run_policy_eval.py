"""
Standalone driver for Xinze's token-policy method — NO lm_eval. Measures GSM8K accuracy +
wall-clock tok/s + skip-fraction for ORG vs token-policy, across cosine/l2 and BATCH SIZES.

The batch sweep is the batch-union deliverable: skip-fraction should FALL as batch grows (a position
is skipped only if frozen in ALL sequences). Requires the token-policy sampler extended to batch>1
(union of active positions) with self._tp_active / self._tp_total counters.

Run from Sparse-dllm/v2/ (venv active, LD_LIBRARY_PATH set):
    CUDA_VISIBLE_DEVICES=<gpu> python run_policy_eval.py --n 64 --gen 256 --batch_sizes 1,4,8,16
"""
import sys, os, types, re, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

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

Question: Julie is reading a 120-page book. Yesterday she read 12 pages and today twice as many. If she reads half the remaining pages tomorrow, how many should she read?
Answer: Today she read 12*2 = 24. So far 12+24 = 36. Remaining 120-36 = 84. Half is 84/2 = 42. #### 42

"""

def gold(a):
    m = re.search(r"####\s*(-?[\d,]+)", a); return m.group(1).replace(",", "") if m else None
def pred(t):
    m = re.search(r"####\s*(-?[\d,]+)", t)
    if m: return m.group(1).replace(",", "")
    n = re.findall(r"-?[\d,]+", t); return n[-1].replace(",", "") if n else None


def make_batches(data, tok, bs, device, uniform_len=0):
    pad = tok.pad_token_id if tok.pad_token_id is not None else 151645
    tok.pad_token = tok.pad_token or tok.eos_token
    out = []
    for i in range(0, len(data), bs):
        chunk = data[i:i + bs]
        encs = [tok(GSM8K_FEWSHOT + f"Question: {q}\nAnswer:", return_tensors="pt").input_ids[0] for q, _ in chunk]
        lens = [e.shape[0] for e in encs]
        L = uniform_len if uniform_len else max(lens)
        ids = torch.full((len(encs), L), pad, dtype=torch.long)
        for j, e in enumerate(encs):
            e = e[-L:]                              # safety: keep the rightmost L if longer than uniform_len
            ids[j, L - e.shape[0]:] = e            # left-pad
        if uniform_len:                            # uniform: treat full padded length as the prompt -> clean, non-ragged batch
            seq_len = torch.full((len(encs),), L, device=device)
            min_len = L
        else:
            seq_len = torch.tensor(lens, device=device)
            min_len = min(lens)
        out.append((ids.to(device), seq_len, min_len, [g for _, g in chunk]))
    return out


@torch.no_grad()
def run(model, tok, sampler, batches, device, gen, **pk):
    correct = total = ntok = 0
    torch.cuda.synchronize(device); t0 = time.perf_counter()
    for ids, seq_len, min_len, golds in batches:
        out = sampler(input_ids=ids, tokenizer=tok, block_size=32, max_new_tokens=gen,
                      small_block_size=8, min_len=min_len, seq_len=seq_len,
                      threshold=0.95, temperature=0.0, use_block_cache=True, **pk)
        # samplers return finished_samples: dict {batch_idx: full_sequence_1d}
        for b in range(len(golds)):
            g_ids = out[b][ids.shape[1]:]                 # generated part (after left-padded prompt)
            ntok += int((g_ids != tok.pad_token_id).sum()) if tok.pad_token_id is not None else g_ids.numel()
            p = pred(tok.decode(g_ids, skip_special_tokens=True))
            correct += int(p is not None and p == golds[b]); total += 1
    torch.cuda.synchronize(device); dt = time.perf_counter() - t0
    skip = 1.0 - (getattr(model, "_tp_active", 0) / max(1, getattr(model, "_tp_total", 0)))
    return correct / total, ntok / dt, skip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--uniform", action="store_true",
                   help="pad ALL prompts to one length -> non-ragged batches -> trustworthy bs>1 accuracy")
    p.add_argument("--gen", type=int, default=256)
    p.add_argument("--freeze", type=int, default=3)
    p.add_argument("--batch_sizes", default="1,4,8,16")
    p.add_argument("--metric", default="l2", choices=["cosine", "l2"])
    p.add_argument("--thr", type=float, default=0.02, help="freeze threshold (l2~0.02, cosine~0.99)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print("Loading model (manual state_dict)...", flush=True)
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
    GF.set_staleness_metric(args.metric)

    print(f"\nBatch sweep — token policy ({args.metric} thr={args.thr}, freeze={args.freeze}), GSM8K n={args.n}\n"
          f"(accuracy at bs>1 may be unreliable — ragged prompts; skip-fraction + tok/s are the signal)")
    print(f"  {'bs':>3} {'ORG t/s':>8} {'pol t/s':>8} {'speedup':>8} {'skip%':>6} {'ORG acc':>8} {'pol acc':>8}")
    ulen = 0
    if args.uniform:
        ulen = max(tok(GSM8K_FEWSHOT + f"Question: {q}\nAnswer:", return_tensors="pt").input_ids.shape[1]
                   for q, _ in data)
        print(f"  (uniform batching: all prompts padded to {ulen} tokens)")
    for bs in [int(x) for x in args.batch_sizes.split(",")]:
        batches = make_batches(data, tok, bs, args.device, uniform_len=ulen)
        org_acc, org_tps, _ = run(model, tok, org, batches, args.device, args.gen)
        acc, tps, skip = run(model, tok, tokp, batches, args.device, args.gen,
                             token_policy_threshold=args.thr, token_policy_freeze_step=args.freeze)
        spd = tps / org_tps if org_tps else 0.0
        print(f"  {bs:>3} {org_tps:>8.1f} {tps:>8.1f} {spd:>7.2f}x {skip:>5.0%} {org_acc:>8.3f} {acc:>8.3f}",
              flush=True)
    print("\n  Read: does skip% hold as bs grows, and does speedup ever exceed 1? If skip% collapses")
    print("  with batch (batch-union) or speedup stays <=1, the union approach is capped -> a ragged")
    print("  (varlen) kernel would be needed for real per-sequence saving.")


if __name__ == "__main__":
    main()
