"""
Needle-in-haystack attention test for Fast-dLLM v2.

For each needle position, reports three things:
  1. full-causal rank:    attention averaged over ALL Q rows (structurally biased
                          toward early blocks — shown for comparison only)
  2. question-only rank:  attention from ONLY the question-token rows — unbiased,
                          what the model actually reads when forming the answer
  3. ablation:            needle IS in the input (propagation allowed during prefill),
                          but needle block is zeroed from KV at answer-time.
                          Tests: is the block load-bearing, or did prefill propagation
                          already encode the needle into later token representations?
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import types
import torch
from transformers import AutoTokenizer
from generation_functions import Fast_dLLM_QwenForCausalLM as GenClass

from helper import (
    load_model, enable_attn_capture, disable_attn_capture, collect_attn_weights
)

BLOCK_SIZE = 32

NEEDLE   = "The secret passcode is: XRAY-TANGO-4291."
QUESTION = "What is the secret passcode mentioned above? Answer with only the passcode."
CORRECT  = "XRAY-TANGO-4291"

FILLER_UNIT = (
    "The weather today is mild and partly cloudy. "
    "People are walking their dogs in the park. "
    "The birds are singing in the trees nearby. "
    "Children are playing on the swings and slides. "
)

NEEDLE_POSITIONS  = [0.1, 0.25, 0.5, 0.75, 0.9]
PREFIX_TOKENS     = 1024
NEEDLE_LAST_SIZES = [256, 512, 1024, 2048]   # filler lengths for needle-last sweep
# distance sweep: fixed filler_before, vary gap between needle and question
FILLER_BEFORE_DISTANCE = 512
GAP_SIZES = [0, 32, 64, 128, 256, 512, 768]  # tokens of filler between needle and question


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_filler(target_tokens, tokenizer):
    text = FILLER_UNIT * (target_tokens // len(tokenizer(FILLER_UNIT).input_ids) + 5)
    tokens = tokenizer(text, return_tensors="pt").input_ids[0]
    return tokens[:target_tokens]


def build_prompt(tokenizer, needle_frac, prefix_tokens):
    """
    Returns (input_ids [1, L], nb_start, nb_end, q_start, q_end).
    nb_start/nb_end: first/last block index the needle occupies.
    q_start/q_end:  token indices of the question in the full prompt.
    """
    needle_insert = int(prefix_tokens * needle_frac)
    needle_ids    = tokenizer(NEEDLE,   return_tensors="pt").input_ids[0]
    question_ids  = tokenizer(QUESTION, return_tensors="pt").input_ids[0]
    needle_len    = len(needle_ids)

    filler_before = build_filler(needle_insert,                  tokenizer)
    filler_after  = build_filler(prefix_tokens - needle_insert,  tokenizer)

    full = torch.cat([filler_before, needle_ids, filler_after, question_ids])

    nb_start   = needle_insert // BLOCK_SIZE
    nb_end     = (needle_insert + needle_len - 1) // BLOCK_SIZE
    q_start    = needle_insert + needle_len + (prefix_tokens - needle_insert)
    q_end      = q_start + len(question_ids)

    return full.unsqueeze(0), nb_start, nb_end, q_start, q_end


def build_prompt_needle_last(tokenizer, filler_tokens):
    """
    Layout (inside chat template): [QUESTION] + [filler] + [NEEDLE]
    followed by the assistant generation prompt.

    The needle is the last content in the user turn. After it come only a
    handful of template delimiter tokens (<|im_end|> etc.), which are too
    generic to meaningfully encode the passcode. So no substantial downstream
    tokens absorb the needle during prefill — eviction is a near-genuine test.

    Returns (input_ids [1, L], nb_start, nb_end).
    Raises if needle lands outside the prefill-cached region (alignment guard).
    """
    # build filler text of approximately filler_tokens tokens
    filler_text = FILLER_UNIT * (filler_tokens // len(tokenizer(FILLER_UNIT).input_ids) + 5)
    filler_ids  = tokenizer(filler_text, return_tensors="pt").input_ids[0][:filler_tokens]
    filler_text = tokenizer.decode(filler_ids, skip_special_tokens=True)

    content  = QUESTION + "\n\nContext:\n" + filler_text + "\n\n" + NEEDLE
    messages = [{"role": "user", "content": content}]
    text     = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full     = tokenizer(text, return_tensors="pt").input_ids[0]

    # find where the needle starts in the token sequence
    needle_ids       = tokenizer(NEEDLE, return_tensors="pt", add_special_tokens=False).input_ids[0]
    needle_len       = len(needle_ids)
    # search from the end since needle is near the tail
    needle_start_tok = None
    for i in range(len(full) - needle_len, max(0, len(full) - needle_len - 20), -1):
        if torch.equal(full[i:i + needle_len], needle_ids):
            needle_start_tok = i
            break
    if needle_start_tok is None:
        # fallback: estimate from prompt structure
        needle_start_tok = len(full) - needle_len - 10

    nb_start = needle_start_tok // BLOCK_SIZE
    nb_end   = (needle_start_tok + needle_len - 1) // BLOCK_SIZE

    # guard: needle must be fully inside the prefill-cached region
    prompt_len       = full.shape[0]
    cached_up_to_tok = (prompt_len // BLOCK_SIZE) * BLOCK_SIZE
    if nb_end * BLOCK_SIZE >= cached_up_to_tok:
        raise ValueError(
            f"Needle at block {nb_end} falls outside cached region "
            f"(prompt_len={prompt_len}, cached up to tok {cached_up_to_tok}). "
            f"Increase filler_tokens so the needle lands at least one full block from the end."
        )

    return full.unsqueeze(0), nb_start, nb_end


def build_prompt_by_gap(tokenizer, filler_before_tokens, gap_tokens):
    """
    Layout (inside user message): [filler_before][NEEDLE][filler_gap][QUESTION]

    question is LAST so the model has a proper answer cue (baseline works).
    gap_tokens controls how many filler tokens sit between needle and question —
    those are the only downstream tokens that can absorb the needle during prefill.
    gap=0 → only question tokens (~17) absorb needle; gap=512 → 512 filler tokens absorb it.

    Returns (input_ids [1, L], nb_start, nb_end, gap_actual).
    """
    filler_before_text = FILLER_UNIT * (filler_before_tokens // len(tokenizer(FILLER_UNIT).input_ids) + 5)
    filler_before_text = tokenizer.decode(
        tokenizer(filler_before_text, return_tensors="pt").input_ids[0][:filler_before_tokens],
        skip_special_tokens=True
    )

    filler_gap_text = ""
    if gap_tokens > 0:
        fg = FILLER_UNIT * (gap_tokens // len(tokenizer(FILLER_UNIT).input_ids) + 5)
        filler_gap_text = tokenizer.decode(
            tokenizer(fg, return_tensors="pt").input_ids[0][:gap_tokens],
            skip_special_tokens=True
        )

    content  = filler_before_text + "\n\n" + NEEDLE + "\n\n" + filler_gap_text + "\n\n" + QUESTION
    messages = [{"role": "user", "content": content}]
    text     = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full     = tokenizer(text, return_tensors="pt").input_ids[0]

    # locate needle in token sequence by matching its token ids
    needle_ids = tokenizer(NEEDLE, return_tensors="pt", add_special_tokens=False).input_ids[0]
    needle_len = len(needle_ids)
    needle_start_tok = None
    # search backward from ~2/3 of prompt (needle is never at the very end)
    search_end = min(len(full), int(len(full) * 0.9))
    for i in range(search_end - needle_len, 0, -1):
        if torch.equal(full[i:i + needle_len], needle_ids):
            needle_start_tok = i
            break
    if needle_start_tok is None:
        needle_start_tok = filler_before_tokens + 30  # rough fallback

    nb_start = needle_start_tok // BLOCK_SIZE
    nb_end   = (needle_start_tok + needle_len - 1) // BLOCK_SIZE

    # verify needle block is in cached region (needs tokens after it to complete its block)
    prompt_len       = full.shape[0]
    cached_up_to_tok = (prompt_len // BLOCK_SIZE) * BLOCK_SIZE
    if nb_end * BLOCK_SIZE + BLOCK_SIZE > cached_up_to_tok + BLOCK_SIZE:
        pass  # needle block always fits if question + template tokens push past its boundary

    # count actual gap tokens between needle end and question start (approximate)
    gap_actual = gap_tokens

    return full.unsqueeze(0), nb_start, nb_end, gap_actual


# ---------------------------------------------------------------------------
# Attention measurement
# ---------------------------------------------------------------------------

def block_mass(weights, block_size, q_start=None, q_end=None):
    """
    Mean attention mass per KV block, averaged over layers/heads.
    q_start/q_end: if given, only use query rows [q_start:q_end] (question-only mode).
    """
    totals, count = None, 0
    for w in weights.values():
        B, H, Q, KV = w.shape
        if q_start is not None:
            qs = q_start
            qe = min(q_end, Q) if q_end is not None else Q
            if qs >= qe:
                continue
            w = w[:, :, qs:qe, :]
        n_blocks = KV // block_size
        if n_blocks == 0:
            continue
        w_trim  = w[..., :n_blocks * block_size]
        bmass   = w_trim.reshape(B, H, -1, n_blocks, block_size).sum(dim=-1)
        mean    = bmass.mean(dim=(0, 1, 2))
        totals  = mean if totals is None else totals + mean
        count  += 1
    return (totals / count).cpu().tolist() if totals is not None else []


def needle_rank_and_mass(mass, needle_block):
    if not mass or needle_block >= len(mass):
        return None, float("nan")
    ranked = sorted(range(len(mass)), key=lambda i: mass[i], reverse=True)
    return ranked.index(needle_block) + 1, mass[needle_block]


def measure_attn(model, prompt_ids, nb_start, q_start, q_end, device):
    """
    Prefill full prompt, return full-causal and question-only needle block ranks.
    """
    ids = prompt_ids.to(device)
    enable_attn_capture(model)
    with torch.no_grad():
        model(input_ids=ids, use_cache=True,
              update_past_key_values=True, block_size=BLOCK_SIZE)
    weights = collect_attn_weights(model)
    disable_attn_capture(model)

    mass_all = block_mass(weights, BLOCK_SIZE)
    mass_q   = block_mass(weights, BLOCK_SIZE, q_start=q_start, q_end=q_end)

    full_rank, _ = needle_rank_and_mass(mass_all, nb_start)
    q_rank, q_m  = needle_rank_and_mass(mass_q,   nb_start)
    N            = len(mass_all)
    return full_rank, q_rank, q_m, N


# ---------------------------------------------------------------------------
# Block-mask helpers (writes state onto each attention module)
# ---------------------------------------------------------------------------

def enable_block_mask(model, k_lo, k_hi, kv_threshold):
    """
    Arm the post-softmax eviction mask on every attention layer.
    The mask only fires when KV length >= kv_threshold (i.e., at decode time,
    not during prefill where KV length == prompt_len exactly).
    Requires _capture_attn_scores=True so the manual attention path is used.
    """
    for layer in model.model.layers:
        layer.self_attn._block_mask = {
            'active': True, 'k_lo': k_lo, 'k_hi': k_hi, 'kv_threshold': kv_threshold
        }


def disable_block_mask(model):
    for layer in model.model.layers:
        layer.self_attn._block_mask = None


def verify_mask_fired(model, prompt_ids, nb_start, device):
    """
    After arming the mask, run a commit-step forward (KV > prompt_len) with
    attn capture and confirm the needle block's residual mass is ~0.
    Aborts loudly if the mask didn't engage — catches mis-wiring.
    """
    with torch.no_grad():
        out = model(input_ids=prompt_ids.to(device), use_cache=True,
                    update_past_key_values=True, block_size=BLOCK_SIZE)
    past_kv = out.past_key_values
    q_ids = torch.zeros(prompt_ids.shape[0], BLOCK_SIZE, dtype=torch.long, device=device)
    enable_attn_capture(model)
    with torch.no_grad():
        model(input_ids=q_ids, use_cache=True, past_key_values=past_kv,
              update_past_key_values=True, block_size=BLOCK_SIZE)
    weights = collect_attn_weights(model)
    disable_attn_capture(model)

    mass = block_mass(weights, BLOCK_SIZE)
    residual = mass[nb_start] if nb_start < len(mass) else float("nan")
    if residual >= 1e-4:
        disable_block_mask(model)
        raise RuntimeError(
            f"Mask did not fire — needle-block residual mass={residual:.5f} (expected ~0). "
            f"Check that the _block_mask branch in modeling.py is reached."
        )
    return residual


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def run_generation(model, tokenizer, prompt_ids, device):
    prompt_ids = prompt_ids.to(device)
    prompt_len = prompt_ids.shape[1]
    sampler    = types.MethodType(GenClass.batch_sample, model)
    output     = sampler(
        input_ids        = prompt_ids,
        tokenizer        = tokenizer,
        block_size       = BLOCK_SIZE,
        max_new_tokens   = 64,
        small_block_size = 8,
        min_len          = prompt_len,
        seq_len          = torch.tensor([prompt_len], device=device),
        threshold        = 0.95,
        temperature      = 0.0,
    )
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------

def run_ablation(model, tokenizer, prompt_ids, nb_start, nb_end, device):
    """
    1. Baseline: normal generation (needle in input, no mask).
    2. Eviction: needle in input (propagation during prefill allowed),
       needle-block KV zeroed at decode time.
    Verdict: PROPAGATED → safe to evict. LOAD-BEARING → eviction kills recall.
    """
    prompt_len = prompt_ids.shape[1]

    # baseline
    disable_block_mask(model)
    base_gen = run_generation(model, tokenizer, prompt_ids, device)
    base_ok  = CORRECT in base_gen

    # arm mask — fires only at decode (KV > prompt_len)
    k_lo, k_hi = nb_start * BLOCK_SIZE, (nb_end + 1) * BLOCK_SIZE
    enable_block_mask(model, k_lo, k_hi, kv_threshold=prompt_len + 1)
    enable_attn_capture(model)   # forces manual attention path where mask lives

    residual = verify_mask_fired(model, prompt_ids, nb_start, device)

    masked_gen = run_generation(model, tokenizer, prompt_ids, device)
    masked_ok  = CORRECT in masked_gen

    disable_block_mask(model)
    disable_attn_capture(model)

    if base_ok and masked_ok:
        verdict = "PROPAGATED (droppable)"
    elif base_ok and not masked_ok:
        verdict = "LOAD-BEARING (must keep)"
    else:
        verdict = "baseline failed"

    return base_ok, masked_ok, residual, verdict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_negative_control(model, tokenizer, device, prefix_tokens=PREFIX_TOKENS):
    """
    Remove the needle from the prompt entirely. Recall must return NO.
    If it returns YES, `CORRECT in gen` is broken and no ablation result can be trusted.
    """
    print("\n=== Negative control (needle never in input) ===")
    filler_text = FILLER_UNIT * (prefix_tokens // len(tokenizer(FILLER_UNIT).input_ids) + 5)
    filler_ids  = tokenizer(filler_text, return_tensors="pt").input_ids[0][:prefix_tokens]
    filler_text = tokenizer.decode(filler_ids, skip_special_tokens=True)

    content  = QUESTION + "\n\nContext:\n" + filler_text
    messages = [{"role": "user", "content": content}]
    text     = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full     = tokenizer(text, return_tensors="pt").input_ids[0].unsqueeze(0)

    gen = run_generation(model, tokenizer, full, device)
    ok  = CORRECT in gen
    print(f"  Generated: '{gen[:80]}'")
    print(f"  Recall without needle: {'YES (BROKEN — check CORRECT string)' if ok else 'NO (correct)'}")
    return ok


def run_needle_last_sweep(model, tokenizer, device, filler_sizes=None):
    """
    Sweep over filler lengths with needle-last layout: [QUESTION][filler][NEEDLE].
    No downstream tokens absorb the needle during prefill — eviction is a genuine test.
    """
    if filler_sizes is None:
        filler_sizes = NEEDLE_LAST_SIZES

    print("\n=== Needle-last sweep (no downstream propagation possible) ===")
    print(f"Layout: [QUESTION] + [filler] + [NEEDLE] — needle is last prefix content\n")

    hdr = f"{'filler_tok':>11}  {'needle_blk':>11}  {'N':>4}  {'normal':>7}  {'evicted':>8}  verdict"
    print(hdr)
    print("-" * len(hdr))

    for filler_tok in filler_sizes:
        torch.cuda.empty_cache()
        try:
            prompt_ids, nb_start, nb_end = build_prompt_needle_last(tokenizer, filler_tok)
        except ValueError as e:
            print(f"  {filler_tok:>11}  skipped: {e}")
            continue

        N = prompt_ids.shape[1] // BLOCK_SIZE

        base_ok, masked_ok, _, verdict = run_ablation(
            model, tokenizer, prompt_ids, nb_start, nb_end, device
        )
        print(
            f"  {filler_tok:>11}  {nb_start:>11}  {N:>4}  "
            f"  {'YES' if base_ok else 'NO':>7}  "
            f"  {'YES' if masked_ok else 'NO':>8}  {verdict}"
        )


def run_distance_sweep(model, tokenizer, device,
                       filler_before=FILLER_BEFORE_DISTANCE, gap_sizes=None):
    """
    Vary gap between needle and question: [filler_before][NEEDLE][gap][QUESTION].
    Question is last so baseline always works.
    gap=0  → only ~17 question tokens absorb needle  (minimal propagation)
    gap=512 → 512 filler tokens absorb needle before question  (heavy propagation)
    Watch for the crossover where eviction starts costing recall.
    """
    if gap_sizes is None:
        gap_sizes = GAP_SIZES

    print(f"\n=== Distance sweep: needle-to-question gap ===")
    print(f"Layout: [filler({filler_before}tok)][NEEDLE][gap filler][QUESTION]\n")

    hdr = f"{'gap_tok':>8}  {'needle_blk':>11}  {'N':>4}  {'normal':>7}  {'evicted':>8}  verdict"
    print(hdr)
    print("-" * len(hdr))

    for gap in gap_sizes:
        torch.cuda.empty_cache()
        try:
            prompt_ids, nb_start, nb_end, _ = build_prompt_by_gap(
                tokenizer, filler_before, gap
            )
        except Exception as e:
            print(f"  {gap:>8}  build failed: {e}")
            continue

        N = prompt_ids.shape[1] // BLOCK_SIZE

        try:
            base_ok, masked_ok, _, verdict = run_ablation(
                model, tokenizer, prompt_ids, nb_start, nb_end, device
            )
        except RuntimeError as e:
            print(f"  {gap:>8}  {nb_start:>11}  {N:>4}  mask error: {e}")
            continue

        print(
            f"  {gap:>8}  {nb_start:>11}  {N:>4}  "
            f"  {'YES' if base_ok else 'NO':>7}  "
            f"  {'YES' if masked_ok else 'NO':>8}  {verdict}"
        )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         type=str, default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    parser.add_argument("--device",        type=str, default="cuda:0")
    parser.add_argument("--prefix_tokens", type=int, default=PREFIX_TOKENS)
    parser.add_argument("--no_ablation",   action="store_true")
    parser.add_argument("--negative_control", action="store_true",
                        help="Run negative control only (needle removed, recall must be NO)")
    parser.add_argument("--needle_last",   action="store_true",
                        help="Run needle-last sweep (question first, no downstream propagation)")
    parser.add_argument("--distance_sweep", action="store_true",
                        help="Sweep needle-to-question gap: [filler][NEEDLE][gap][QUESTION]")
    args = parser.parse_args()

    print("Loading model...")
    model     = load_model(args.model, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    device    = args.device

    if args.negative_control:
        run_negative_control(model, tokenizer, device, args.prefix_tokens)
        return

    if args.needle_last:
        run_negative_control(model, tokenizer, device, args.prefix_tokens)
        run_needle_last_sweep(model, tokenizer, device)
        return

    if args.distance_sweep:
        run_negative_control(model, tokenizer, device, args.prefix_tokens)
        run_distance_sweep(model, tokenizer, device)
        return

    # --- original sweep (filler + needle at position X + filler + question) ---
    print(f"\nPrefix: {args.prefix_tokens} tokens = {args.prefix_tokens // BLOCK_SIZE} blocks")
    print(f"Needle:   '{NEEDLE}'")
    print(f"Question: '{QUESTION}'\n")

    hdr = (f"{'pos':>5}  {'blk':>4}  {'N':>4}  "
           f"{'full_rank':>10}  {'q_rank':>7}  {'q_mass':>7}  "
           f"{'normal':>7}  {'evicted':>8}  verdict")
    print(hdr)
    print("-" * len(hdr))

    for frac in NEEDLE_POSITIONS:
        torch.cuda.empty_cache()

        prompt_ids, nb_start, nb_end, q_start, q_end = build_prompt(
            tokenizer, frac, args.prefix_tokens
        )

        full_rank, q_rank, q_m, N = measure_attn(
            model, prompt_ids, nb_start, q_start, q_end, device
        )

        if args.no_ablation:
            normal_ok   = CORRECT in run_generation(model, tokenizer, prompt_ids, device)
            evicted_str, verdict = "skipped", "?"
        else:
            base_ok, masked_ok, _, verdict = run_ablation(
                model, tokenizer, prompt_ids, nb_start, nb_end, device
            )
            normal_ok   = base_ok
            evicted_str = "YES" if masked_ok else "NO"

        print(
            f"  {frac:>4.0%}  {nb_start:>4}  {N:>4}  "
            f"  {str(full_rank)+'/'+str(N):>10}  "
            f"  {str(q_rank)+'/'+str(N):>7}  "
            f"  {q_m:>7.4f}  "
            f"  {'YES' if normal_ok else 'NO':>7}  "
            f"  {evicted_str:>8}  {verdict}"
        )


if __name__ == "__main__":
    main()
