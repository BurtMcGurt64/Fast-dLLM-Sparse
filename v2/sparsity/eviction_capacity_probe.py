"""
EVICTION CAPACITY PROBE — the go/no-go for "evict KV -> fit bigger batch -> 2x throughput".

The capacity story only works if the thing that limits batch size is the KV CACHE
(which eviction frees). But this repo already found the B=32/16k OOM is the PREFILL
FFN ACTIVATION SPIKE (~37 GB), not KV. If prefill is the binding constraint, evicting
KV during decode does NOT let you fit a bigger batch — you OOM before decode. This
script measures which constraint binds BEFORE you build eviction.

For each (context, batch) it measures:
  - prefill_peak : GPU peak memory right after the prefill forward (includes the
                   transient gate+up activation spike)
  - total_peak   : GPU peak over the whole run (prefill + decode + growing KV)
  - binding      : if total_peak ~= prefill_peak  -> PREFILL-BOUND (eviction won't help;
                   needs chunked prefill).   if total_peak >> prefill_peak -> KV/DECODE-BOUND
                   (eviction frees the binding memory -> capacity -> throughput).
  - throughput   : tok/s (to check where it saturates)
It also prints the analytic byte breakdown (weights / KV-per-seq / prefill spike) and
PROJECTS the batch you could fit if KV were evicted to `frac` — but only counts that
projection as real when the run is KV-bound.

Run:
  CUDA_VISIBLE_DEVICES=0 uv run v2/sparsity/eviction_capacity_probe.py \
      --context_lens 4096 8192 16384 --batch_sizes 8 16 32 --gen_len 256 --frac 0.2
"""
import sys
import os
HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "../experiments"))

import time
import types
import argparse
import contextlib
import torch
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

from transformers import AutoTokenizer
from helper import load_model
from generation_functions import Fast_dLLM_QwenForCausalLM as GenClass
from eval_longbench_sweep import load_longbench_subset

BLOCK_SIZE = 32
STOP_TOKEN = 151645

# Fast-dLLM v2 / Qwen2.5-7B dims
LAYERS, HIDDEN, FFN = 28, 3584, 18944
KV_HEADS, HEAD_DIM, BF16 = 4, 128, 2
KV_BYTES_PER_TOKEN = 2 * LAYERS * KV_HEADS * HEAD_DIM * BF16          # k+v, all layers = 57344 B
SPIKE_BYTES_PER_BL = 2 * FFN * BF16                                   # gate+up held together = 75776 B/(token·seq)


@contextlib.contextmanager
def suppress_stop_token(model):
    orig = model.sample_with_top_p
    def wrapped(logits, *a, **kw):
        logits[..., STOP_TOKEN] = float("-inf")
        return orig(logits, *a, **kw)
    model.sample_with_top_p = wrapped
    try:
        yield
    finally:
        model.sample_with_top_p = orig


class PrefillPeakHook:
    """Record GPU peak memory right after the FIRST prefill forward (q > block)."""
    def __init__(self, model, device):
        self.model, self.device, self._orig = model, device, model.forward
        self.prefill_peak = 0

    def __enter__(self):
        def wrapped(*a, **k):
            out = self._orig(*a, **k)
            if not self.prefill_peak:
                ids = k.get("input_ids")
                q = ids.shape[1] if ids is not None else 0
                if k.get("update_past_key_values", False) and q > BLOCK_SIZE:
                    torch.cuda.synchronize(self.device)
                    self.prefill_peak = torch.cuda.max_memory_allocated(self.device)
            return out
        self.model.forward = wrapped
        return self

    def __exit__(self, *x):
        self.model.forward = self._orig


def gb(x):
    return x / 1e9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--subset", default="narrativeqa")
    p.add_argument("--context_lens", type=int, nargs="+", default=[4096, 8192, 16384])
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--gen_len", type=int, default=256)
    p.add_argument("--frac", type=float, default=0.2, help="eviction retention ratio for the projection")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("Loading model...", flush=True)
    model = load_model(args.model, device=args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    sampler = types.MethodType(GenClass.batch_sample, model)
    device = args.device

    torch.cuda.reset_peak_memory_stats(device)
    weights_gb = gb(torch.cuda.memory_allocated(device))  # after load, before any run
    total_gpu_gb = gb(torch.cuda.get_device_properties(device).total_memory)
    print(f"weights resident ~{weights_gb:.1f} GB   |   GPU total {total_gpu_gb:.0f} GB\n", flush=True)

    hdr = (f"  {'ctx':>6} {'B':>4} | {'prefillPk':>9} {'totalPk':>8} {'binding':>9} | "
           f"{'KV/seq':>7} {'spike':>7} | {'tok/s':>7} | {'evict->maxB':>11}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for L in args.context_lens:
        for B in args.batch_sizes:
            kv_seq_gb = gb(KV_BYTES_PER_TOKEN * (L + args.gen_len))
            spike_gb = gb(SPIKE_BYTES_PER_BL * B * L)
            try:
                batches = load_longbench_subset(args.subset, tokenizer, device, L, B, B, args.seed)
            except Exception as e:
                print(f"  {L:>6} {B:>4} | data load failed: {e}", flush=True)
                continue

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            try:
                with suppress_stop_token(model), PrefillPeakHook(model, device) as hook, torch.no_grad():
                    enc = batches[0][0]
                    ids = enc.input_ids
                    plen = ids.shape[1]
                    t0 = time.perf_counter()
                    out = sampler(input_ids=ids, tokenizer=tokenizer, block_size=BLOCK_SIZE,
                                  max_new_tokens=args.gen_len, small_block_size=8, min_len=plen,
                                  seq_len=torch.full((B,), plen, dtype=torch.long, device=device),
                                  threshold=0.95, temperature=0.0, use_block_cache=True)
                    torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - t0
                    gen = sum(out[b].shape[0] - plen for b in range(B))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  {L:>6} {B:>4} | OOM  (prefill spike ~{spike_gb:.0f}GB + KV ~{kv_seq_gb*B:.0f}GB)",
                      flush=True)
                continue

            total_pk = gb(torch.cuda.max_memory_allocated(device))
            prefill_pk = gb(hook.prefill_peak)
            decode_growth = total_pk - prefill_pk
            binding = "PREFILL" if decode_growth < 0.05 * total_pk else "KV/DEC"
            tps = gen / elapsed if elapsed else 0.0

            # projection: if KV-bound, how big a batch could the freed memory buy?
            if binding == "KV/DEC":
                free = total_gpu_gb * 0.92 - weights_gb - spike_gb  # headroom, minus prefill spike
                evict_kv_seq = kv_seq_gb * args.frac
                proj_maxB = int(free / evict_kv_seq) if evict_kv_seq > 0 else 0
                proj = f"{proj_maxB}"
            else:
                proj = "n/a(prefill)"

            print(f"  {L:>6} {B:>4} | {prefill_pk:>8.1f}G {total_pk:>7.1f}G {binding:>9} | "
                  f"{kv_seq_gb:>6.2f}G {spike_gb:>6.1f}G | {tps:>7.0f} | {proj:>11}", flush=True)

    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    print("  For each row, read `binding`:")
    print("   - KV/DEC : eviction frees the constraint -> capacity -> throughput. GO for that regime.")
    print("             (`evict->maxB` = projected batch that fits after eviction; compare to current B.)")
    print("   - PREFILL: the prefill activation spike is the wall. Eviction alone buys NO capacity here;")
    print("             you MUST add chunked prefill first. Note the spike column dwarfs KV at long ctx.")
    print("\n  The eviction-for-2x-throughput story is only alive in the regimes marked KV/DEC.")
    print("  If everything is PREFILL-bound, the honest project is 'chunked prefill + eviction', not eviction alone.")


if __name__ == "__main__":
    main()
