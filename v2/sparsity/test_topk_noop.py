"""
Noop test: TopKBlockLoader with K >> N_blocks must be bit-identical to baseline.

Also instruments model.forward to print past_key_values[0][0].float().sum()
after every commit step so we can pinpoint where baseline and topk diverge.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../experiments"))

import argparse
import types
import torch
from transformers import AutoTokenizer
from datasets import load_dataset

from helper import load_model
from generation_functions import Fast_dLLM_QwenForCausalLM as GenClass
from topk import TopKBlockLoader

BLOCK_SIZE = 32
_FILLER = "The quick brown fox jumps over the lazy dog. " * 200


def make_prompt(tokenizer, question, target_ctx_len=None):
    filler = ""
    if target_ctx_len:
        base = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
        )
        base_len = len(tokenizer(base, return_tensors="pt").input_ids[0])
        need = max(0, target_ctx_len - base_len)
        if need > 0:
            fids = tokenizer(_FILLER, return_tensors="pt").input_ids[0]
            filler = tokenizer.decode(fids[:need], skip_special_tokens=True) + " "
    content = filler + question
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )


def run_one(model, tokenizer, input_ids, device, k=None, label=""):
    sampler = types.MethodType(GenClass.batch_sample, model)
    plen = input_ids.shape[1]

    # --- KV checksum hook (wraps the actual model forward) ---
    actual_forward = model.forward
    kv_sums = []

    def _kv_hook(*args, **kwargs):
        pre_kv_len = 0
        try:
            pkv = kwargs.get('past_key_values', None)
            if pkv is not None:
                pre_kv_len = pkv.layers[0].keys.shape[-2]
        except Exception:
            pass
        is_commit = (kwargs.get('update_past_key_values', False)
                     and not kwargs.get('use_block_cache', False)
                     and pre_kv_len > 0)
        output = actual_forward(*args, **kwargs)
        if is_commit:
            s = output.past_key_values[0][0].float().sum().item()
            kv_sums.append((pre_kv_len, s))
            print(f"    [{label}] commit(pre={pre_kv_len}) kv0_sum={s:.4f}")
        return output

    model.forward = _kv_hook

    # --- TopK loader (wraps _kv_hook when installed) ---
    loader = TopKBlockLoader(model, k=k) if k is not None else None
    if loader:
        loader.install()

    try:
        out = sampler(
            input_ids=input_ids, tokenizer=tokenizer,
            block_size=BLOCK_SIZE, max_new_tokens=256,
            small_block_size=8, min_len=plen,
            seq_len=torch.tensor([plen], dtype=torch.long, device=device),
            threshold=0.95, temperature=0.0,
        )
    finally:
        if loader:
            loader.uninstall()
        model.forward = actual_forward   # remove hook

    tokens = out[0][plen:]
    text   = tokenizer.decode(tokens, skip_special_tokens=True)
    print(f"  [{label}] gen={len(tokens)}tok  {text[:80]!r}")
    return tokens, kv_sums


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str, default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    parser.add_argument("--device",      type=str, default="cuda:0")
    parser.add_argument("--context_len", type=int, default=None)
    parser.add_argument("--n_examples",  type=int, default=3)
    args = parser.parse_args()

    model     = load_model(args.model, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    ds = load_dataset("gsm8k", "main", split="test")

    all_passed = True
    for i in range(args.n_examples):
        text      = make_prompt(tokenizer, ds[i]["question"], args.context_len)
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(args.device)
        n_blocks  = input_ids.shape[1] // BLOCK_SIZE
        k_noop    = n_blocks + 10

        print(f"\n=== Example {i} (n_blocks={n_blocks}, K_noop={k_noop}) ===")
        base,  base_sums = run_one(model, tokenizer, input_ids, args.device, k=None,   label="baseline")
        topk_, topk_sums = run_one(model, tokenizer, input_ids, args.device, k=k_noop, label=f"K={k_noop}")

        # Compare KV checksums at each commit step
        for (pre_b, s_b), (pre_t, s_t) in zip(base_sums, topk_sums):
            match_kv = abs(s_b - s_t) < 1e-3
            print(f"  KV@{pre_b}: baseline={s_b:.4f} topk={s_t:.4f} {'OK' if match_kv else 'DIFFER'}")

        # Compare generated tokens
        min_len      = min(len(base), len(topk_))
        match        = torch.equal(base[:min_len], topk_[:min_len])
        length_match = len(base) == len(topk_)

        if match and length_match:
            print(f"  PASS — token-identical ({len(base)} tokens)")
        else:
            all_passed = False
            if not length_match:
                print(f"  FAIL length: baseline={len(base)} topk={len(topk_)}")
            if not match:
                j = next((j for j in range(min_len) if base[j] != topk_[j]), min_len)
                print(f"  FAIL — first token divergence at position {j}")
                print(f"    baseline[{j}]={base[j].item()}  topk[{j}]={topk_[j].item()}")

    print(f"\n{'ALL PASS' if all_passed else 'SOME FAILED'}")


if __name__ == "__main__":
    main()
