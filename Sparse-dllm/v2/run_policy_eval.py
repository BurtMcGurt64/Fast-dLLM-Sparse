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


def gsm8k_score(text, g):
    p = pred(text); return float(p is not None and p == g)


# ---- LongBench (long-context) ----------------------------------------------------------------
# Standard token-F1 (from the LongBench repo) for QA-style tasks.
def _normalize(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch.isalnum() or ch.isspace())
    return " ".join(s.split())

def qa_f1(pred_text, answers):
    best = 0.0
    pt = _normalize(pred_text).split()
    for ans in (answers if isinstance(answers, (list, tuple)) else [answers]):
        gt = _normalize(str(ans)).split()
        if not pt or not gt:
            best = max(best, float(pt == gt)); continue
        common = {}
        for w in pt: common[w] = min(pt.count(w), gt.count(w))
        ncommon = sum(common.values())
        if ncommon == 0: continue
        prec, rec = ncommon / len(pt), ncommon / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best

def load_longbench(lb_task, n, ctx_tokens, tok, seed):
    """Head+tail-truncate each example's (context+question) to ctx_tokens; return (enc_1d, answers).
    Reads the task jsonl straight out of the repo's data.zip (newer `datasets` refuses the
    LongBench.py loader script; the repo stores all tasks inside a single data.zip)."""
    import json, random, zipfile, io
    from huggingface_hub import hf_hub_download
    zpath = hf_hub_download("zai-org/LongBench", "data.zip", repo_type="dataset")
    with zipfile.ZipFile(zpath) as zf:
        member = next((m for m in zf.namelist() if m.rstrip("/").endswith(f"{lb_task}.jsonl")), None)
        if member is None:
            avail = sorted(m.split("/")[-1][:-6] for m in zf.namelist() if m.endswith(".jsonl"))
            raise ValueError(f"LongBench task '{lb_task}' not in data.zip. Available: {avail}")
        with zf.open(member) as f:
            rows = [json.loads(line) for line in io.TextIOWrapper(f, encoding="utf-8")]
    random.Random(seed).shuffle(rows)
    rows = rows[:n]
    out = []
    for ex in rows:
        prompt = f"{ex['context']}\n\nQuestion: {ex['input']}\nAnswer:"
        ids = tok(prompt, return_tensors="pt").input_ids[0]
        if ids.shape[0] > ctx_tokens:               # LongBench-style middle truncation
            h = ctx_tokens // 2
            ids = torch.cat([ids[:h], ids[-(ctx_tokens - h):]])
        out.append((ids, ex["answers"]))
    return out


def make_batches(data, tok, bs, device, uniform_len=0):
    """data: list of (enc_1d_tensor, gold). Left-pads each batch (or all, if uniform_len)."""
    pad = tok.pad_token_id if tok.pad_token_id is not None else 151645
    tok.pad_token = tok.pad_token or tok.eos_token
    out = []
    for i in range(0, len(data), bs):
        chunk = data[i:i + bs]
        encs = [e for e, _ in chunk]
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
def run(model, tok, sampler, batches, device, gen, score_fn, **pk):
    correct = 0.0; total = 0; ntok = 0
    torch.cuda.synchronize(device); t0 = time.perf_counter()
    for ids, seq_len, min_len, golds in batches:
        out = sampler(input_ids=ids, tokenizer=tok, block_size=32, max_new_tokens=gen,
                      small_block_size=8, min_len=min_len, seq_len=seq_len,
                      threshold=0.95, temperature=0.0, use_block_cache=True, **pk)
        # samplers return finished_samples: dict {batch_idx: full_sequence_1d}
        for b in range(len(golds)):
            g_ids = out[b][ids.shape[1]:]                 # generated part (after left-padded prompt)
            ntok += int((g_ids != tok.pad_token_id).sum()) if tok.pad_token_id is not None else g_ids.numel()
            correct += score_fn(tok.decode(g_ids, skip_special_tokens=True), golds[b]); total += 1
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
    p.add_argument("--kv_mode", default="none", choices=["none", "streaming", "compose", "fourway"],
                   help="none=token policy only; streaming=token+KV window; "
                        "compose=token-only vs token+KV per batch; "
                        "fourway=token-only vs KV-only vs composed (KV-only = streaming with freeze=0 "
                        "so nothing freezes -> dense queries + windowed KV, reproduces standalone KV top-K)")
    p.add_argument("--n_sink", type=int, default=2, help="streaming: prefix BLOCKS kept at the front")
    p.add_argument("--n_recency", type=int, default=4, help="streaming: prefix BLOCKS kept at the end")
    p.add_argument("--task", default="gsm8k", choices=["gsm8k", "longbench"],
                   help="gsm8k=short ctx (memory axis tiny, expect streaming~=no-op); "
                        "longbench=long ctx (memory axis real -> crossover can appear)")
    p.add_argument("--lb_task", default="hotpotqa", help="LongBench subset (hotpotqa/2wikimqa/narrativeqa/...)")
    p.add_argument("--lb_tasks", default="", help="comma-sep LongBench subsets; loops over each (overrides --lb_task). "
                                                 "Use with --kv_mode none for the headline token-only x task x batch table")
    p.add_argument("--ctx", type=int, default=4096, help="longbench: truncate each prompt to this many tokens")
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

    org = types.MethodType(GF.Fast_dLLM_QwenForCausalLM.batch_sample, model)
    tokp = types.MethodType(GF.Fast_dLLM_QwenForCausalLM.batch_sample_token_policy_flashattention, model)
    GF.set_staleness_metric(args.metric)

    # Which token-policy arms to run per batch. compose/fourway expose the roofline crossover.
    stream_kw = dict(kv_mode="streaming", n_sink=args.n_sink, n_recency=args.n_recency)
    kv_arms = [("none", None)]
    if args.kv_mode == "streaming":
        kv_arms = [("stream", dict(stream_kw))]
    elif args.kv_mode == "compose":
        kv_arms = [("token", None), ("compose", dict(stream_kw))]
    elif args.kv_mode == "fourway":
        # KV-only = streaming with freeze=0 (nothing stays frozen -> dense queries + windowed KV).
        kv_arms = [("token", None),
                   ("kv", dict(stream_kw, token_policy_freeze_step=0)),
                   ("compose", dict(stream_kw))]

    def build_task(spec):
        if spec == "gsm8k":
            from datasets import load_dataset
            ds = load_dataset("gsm8k", "main", split="test").shuffle(seed=args.seed).select(range(args.n))
            data = [(tok(GSM8K_FEWSHOT + f"Question: {ex['question']}\nAnswer:", return_tensors="pt").input_ids[0],
                     gold(ex["answer"])) for ex in ds]
            return data, gsm8k_score, f"GSM8K n={args.n}"
        data = load_longbench(spec, args.n, args.ctx, tok, args.seed)
        return data, qa_f1, f"LongBench/{spec} ctx<={args.ctx} n={args.n} (metric=token-F1)"

    # Task list: --lb_tasks (comma-sep longbench) loops; else single --task/--lb_task.
    if args.lb_tasks:
        task_specs = [t.strip() for t in args.lb_tasks.split(",") if t.strip()]
    else:
        task_specs = ["gsm8k"] if args.task == "gsm8k" else [args.lb_task]

    win = f", KV sink={args.n_sink}+rec={args.n_recency} blk" if args.kv_mode != "none" else ""
    for spec in task_specs:
        data, score_fn, task_desc = build_task(spec)
        print(f"\nBatch sweep — token policy ({args.metric} thr={args.thr}, freeze={args.freeze}{win}), {task_desc}")
        print("  compute axis = token policy (query rows); memory axis = streaming KV (prefix bytes).")
        hdr = f"  {'bs':>3} {'ORG t/s':>8} {'ORGacc':>6}"
        for name, _ in kv_arms:
            hdr += f" | {name:>7} t/s  spd  skip%  acc"
        print(hdr)
        ulen = 0
        if args.uniform:
            ulen = max(e.shape[0] for e, _ in data)
            print(f"  (uniform batching: all prompts padded to {ulen} tokens)")
        for bs in [int(x) for x in args.batch_sizes.split(",")]:
            batches = make_batches(data, tok, bs, args.device, uniform_len=ulen)
            org_acc, org_tps, _ = run(model, tok, org, batches, args.device, args.gen, score_fn)
            row = f"  {bs:>3} {org_tps:>8.1f} {org_acc:>6.3f}"
            for name, kvkw in kv_arms:
                extra = dict(token_policy_threshold=args.thr, token_policy_freeze_step=args.freeze)
                if kvkw:
                    extra.update(kvkw)
                acc, tps, skip = run(model, tok, tokp, batches, args.device, args.gen, score_fn, **extra)
                spd = tps / org_tps if org_tps else 0.0
                row += f" | {tps:>9.1f} {spd:>4.2f}x {skip:>4.0%} {acc:>5.3f}"
            print(row, flush=True)


if __name__ == "__main__":
    main()
