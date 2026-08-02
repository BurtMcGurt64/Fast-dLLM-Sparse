"""
Driver: measure token lock-rates on Fast-dLLM v2 (dry run — changes nothing, just measures).

    CUDA_VISIBLE_DEVICES=<gpu> uv run v2/tokenskip/run_dryrun.py --n 8 --gen 256

Prints, per staleness metric (cosine/l2) and threshold, the fraction of current-block tokens that
WOULD be locked, by step-in-block. High-and-early = lots of skippable FFN compute (green light for
building the real skip); low/late = little to skip.
"""
import sys, os, types, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))            # generation_functions
sys.path.insert(0, os.path.join(HERE, "../experiments"))  # helper
import torch
from transformers import AutoTokenizer
from helper import load_model
from generation_functions import Fast_dLLM_QwenForCausalLM as GenClass
from skip import LockInstrument


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n", type=int, default=8, help="# prompts")
    p.add_argument("--gen", type=int, default=256, help="max_new_tokens (need >=256 for enough steps)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print("Loading model...", flush=True)
    model = load_model(args.model, device=args.device).eval()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    sampler = types.MethodType(GenClass.batch_sample, model)

    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test").shuffle(seed=args.seed).select(range(args.n))
    prompts = []
    for ex in ds:
        msgs = [{"role": "user", "content": ex["question"]}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    inst = LockInstrument(model).install()
    try:
        with torch.no_grad():
            for i, text in enumerate(prompts):
                ids = tok(text, return_tensors="pt").input_ids.to(args.device)
                plen = ids.shape[1]
                sampler(input_ids=ids, tokenizer=tok, block_size=32, max_new_tokens=args.gen,
                        small_block_size=8, min_len=plen,
                        seq_len=torch.tensor([plen], device=args.device),
                        threshold=0.95, temperature=0.0, use_block_cache=True)
                print(f"  ran prompt {i+1}/{len(prompts)}", flush=True)
        inst.report()
    finally:
        inst.uninstall()


if __name__ == "__main__":
    main()
