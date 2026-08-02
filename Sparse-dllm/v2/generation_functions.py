from typing import Callable, Optional, Union
import torch
import types
from transformers.utils import auto_docstring, logging

# Constants for Fast_dLLM model
FAST_DLLM_MASK_ID = 151665
FAST_DLLM_STOP_TOKEN = 151645

MASK_COLOR = 0.5  
TOKEN_COLOR = -0.5  


import os
# --- staleness metric switch: cosine (default) or l2 (relative-L2 change) ---
# Set via env:  STALENESS_METRIC=l2 python eval_org_vs_new.py ...   (or call set_staleness_metric)
# NOTE: thresholds differ by metric — cosine freezes when similarity > thr (~0.99);
#       l2 freezes when change < thr (~0.05). Pass the metric-appropriate *_policy_threshold.
_STALENESS_METRIC = os.environ.get("STALENESS_METRIC", "cosine")


def set_staleness_metric(m):
    global _STALENESS_METRIC
    assert m in ("cosine", "l2"), f"bad metric {m!r}"
    _STALENESS_METRIC = m


def _token_is_stable(prev, cur, thr):
    """prev, cur: 1D float tensors. True = token settled -> freeze."""
    if _STALENESS_METRIC == "l2":
        d = (cur - prev).norm() / prev.norm().clamp_min(1e-6)
        return bool((d < thr).item())
    cos = torch.nn.functional.cosine_similarity(prev, cur, dim=-1)
    return bool((cos > thr).item())


def _layer_trigger(layer_cos, layer_l2, valid, thr):
    """bool tensor [num_layers]: which layers to freeze this step."""
    if _STALENESS_METRIC == "l2" and layer_l2 is not None:
        return valid & (layer_l2 < thr)
    return valid & (layer_cos > thr)


def _reset_decode_flops_estimator(model, policy_name: str, block_size: int):
    # Per-run estimator state (raw decode-step logs; FLOPs are computed only at finalize).
    model._decode_flops_estimator = {
        "policy_name": policy_name,
        "block_size": int(block_size),
        "events": [],
        "prefill_steps": 0,
        "full_block_steps": 0,
        "subblock_steps": 0,
        "sparse_steps": 0,
    }


def _record_decode_flops_step(
    model,
    *,
    step_kind: str,
    tokens_forwarded: int,
    active_layers: int,
    kv_length: int,
    cosine_token_pairs: int = 0,
    lm_head_tokens: Optional[int] = None,
):
    # Record one raw decode event; no FLOPs math is done here.
    state = getattr(model, "_decode_flops_estimator", None)
    if state is None:
        return

    t = int(tokens_forwarded)
    l = int(active_layers)
    kv_len = int(kv_length)
    cos_pairs = int(cosine_token_pairs)
    lm_t = int(t if lm_head_tokens is None else lm_head_tokens)

    state["events"].append(
        {
            "step_kind": step_kind,
            "tokens_forwarded": max(t, 0),
            "active_layers": max(l, 0),
            "kv_length": max(kv_len, 0),
            "cosine_token_pairs": max(cos_pairs, 0),
            "lm_head_tokens": max(lm_t, 0),
        }
    )

    if step_kind == "prefill":
        state["prefill_steps"] += 1
    elif step_kind == "full_block":
        state["full_block_steps"] += 1
    elif step_kind == "subblock":
        state["subblock_steps"] += 1
    elif step_kind == "sparse":
        state["sparse_steps"] += 1


def _finalize_decode_flops_estimator(model):
    # Expose estimator outputs for eval harness aggregation.
    state = getattr(model, "_decode_flops_estimator", None)
    if state is None:
        model.last_estimated_flops = 0.0
        model.last_flops_summary = None
        return

    h = int(model.config.hidden_size)
    i = int(model.config.intermediate_size)
    v = int(model.config.vocab_size)
    nh = int(getattr(model.config, "num_attention_heads", 0) or 0)
    nkv = int(getattr(model.config, "num_key_value_heads", nh) or nh)
    d = int(h // nh) if nh else h

    decoder_flops = 0.0
    lm_head_flops = 0.0
    cosine_flops = 0.0
    cosine_token_pairs = 0

    # FLOPs are computed only once here, after decoding is complete.
    for event in state["events"]:
        t = int(event["tokens_forwarded"])
        l = int(event["active_layers"])
        kv_len = int(event["kv_length"])
        lm_t = int(event["lm_head_tokens"])
        cos_pairs = int(event["cosine_token_pairs"])

        # Attention linear proj FLOPs with GQA: Q is full h, K/V use nkv heads.
        # Q: 2*t*h*h, K: 2*t*h*(nkv*d), V: 2*t*h*(nkv*d), O: 2*t*h*h
        attn_linear = (4.0 * t * h * h) + (4.0 * t * h * nkv * d)
        # Attention score + weighted sum: Q has t tokens, K/V has kv_len tokens.
        # score: 2*t*kv_len*h, weighted sum: 2*t*kv_len*h (same with GQA when summed over heads).
        attn_scores = 4.0 * t * kv_len * h
        # MLP (two linear layers + activation overhead): ~6*t*h*i
        mlp = 6.0 * t * h * i
        per_layer = attn_linear + mlp + attn_scores
        decoder_flops += float(max(l, 0)) * per_layer
        lm_head_flops += (2.0 * max(lm_t, 0) * h * v) if lm_t > 0 else 0.0
        cosine_flops += float(max(cos_pairs, 0)) * float((6 * h) + 6)
        cosine_token_pairs += max(cos_pairs, 0)

    total_flops = decoder_flops + lm_head_flops + cosine_flops
    model.last_estimated_flops = float(total_flops)
    model.last_flops_summary = {
        "policy_name": state["policy_name"],
        "block_size": state["block_size"],
        "prefill_steps": int(state["prefill_steps"]),
        "full_block_steps": int(state["full_block_steps"]),
        "subblock_steps": int(state["subblock_steps"]),
        "sparse_steps": int(state["sparse_steps"]),
        "cosine_token_pairs": int(cosine_token_pairs),
        "decoder_flops": float(decoder_flops),
        "lm_head_flops": float(lm_head_flops),
        "cosine_flops": float(cosine_flops),
        "total_flops": float(total_flops),
    }
    model._decode_flops_estimator = None


@auto_docstring
class Fast_dLLM_QwenForCausalLM:
    @torch.no_grad()
    def batch_sample(
        self,
        input_ids,
        tokenizer,
        block_size,
        max_new_tokens, 
        small_block_size,
        min_len,
        seq_len,
        mask_id=151665,
        threshold=0.95,
        stop_token=151645,
        use_block_cache=False,
        top_p=0.95,
        temperature=0.0,
        estimate_flops=False,
    ):
        num_blocks = max_new_tokens // block_size + seq_len.max().item() // block_size
        batch_size = input_ids.shape[0]
        num_layers = self.config.num_hidden_layers
        if estimate_flops and getattr(self, "_decode_flops_estimator", None) is None:
            _reset_decode_flops_estimator(self, "original", block_size)

        # only prefix cache if min_len > block_size
        if min_len > block_size:
            prefill_tokens = int(min_len // block_size * block_size)
            output = self.forward(input_ids=input_ids[:, :prefill_tokens], use_cache=True, update_past_key_values=True, block_size=block_size)
            logits, past_key_values = output.logits, output.past_key_values
            if estimate_flops:
                _record_decode_flops_step(self,
                    step_kind="prefill",
                    tokens_forwarded=prefill_tokens,
                    kv_length=prefill_tokens,  # self-attention, no prior cache
                    active_layers=num_layers,
                )
            # we only do AR steps if the next decoding block doesn't have an initial token
            # we need this token for full block forward and the first subblock forwawrd since
            # we shift the logits and pos0 will use pos0's logit. Therefore, we need to make
            # sure pos0 is not masked for the first block.
            if min_len % block_size == 0:
                predict_sample_idx = (seq_len == min_len)
                predict_logits = logits[predict_sample_idx, -1:, :]
                next_token = predict_logits.argmax(dim=-1)
                # case when we don't have right paddings
                if input_ids.shape[1] <= min_len:
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                # case we do have right paddings
                else:
                    input_ids[predict_sample_idx, min_len] = next_token.squeeze(dim=-1)
        else:
            past_key_values = None

        seq_block_idx = seq_len // block_size
        finished_flag = torch.zeros((batch_size), device=self.device, dtype=torch.bool)

        start_block_idx = min_len // block_size
        num_small_blocks = block_size // small_block_size

        sample_indices = torch.arange(batch_size, device=self.device)
        finished_samples = {}
        for block_idx in range(start_block_idx, num_blocks):
            if finished_flag.all():
                break
            if (seq_block_idx == block_idx).all():
                x_init = mask_id * torch.ones((input_ids.shape[0], block_size-input_ids.shape[1]%block_size), device=self.device, dtype=torch.long)
                x_init = torch.cat([input_ids, x_init], dim=1)
                input_ids = x_init
            else:
                x_init = input_ids[:, :(block_idx + 1)*block_size]

            x_init[finished_flag, -block_size:] = tokenizer.pad_token_id
            x_t = x_init.clone()
            step = 0
            block_past_key_values = None
            # Total KV length for attention: prefix cache + current block tokens.
            prefix_len = past_key_values.get_seq_length() if past_key_values is not None else 0
            block_kv_length = prefix_len + block_size
            while True:
                mask_idx = (x_t[:, -block_size:] == mask_id)
                if mask_idx.sum() == 0:
                    for sample_idx in range(x_t.shape[0]):
                        if finished_flag[sample_idx] and seq_len[sample_idx] < (block_idx + 1) * block_size:
                            stop_token_idx = (x_t[sample_idx, seq_len[sample_idx]:] == stop_token).nonzero()[0][0]
                            x_t[sample_idx, seq_len[sample_idx]+stop_token_idx+1:] = tokenizer.pad_token_id
                    if finished_flag.all():
                        break
                    # full block forward to update past_key_values for the next block and AR step to get the first token for the next block.
                    output = self.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=True, block_size=block_size)
                    logits, past_key_values = output.logits, output.past_key_values
                    if estimate_flops:
                        _record_decode_flops_step(self,
                            step_kind="full_block",
                            tokens_forwarded=int(x_t[:, -block_size:].shape[1]),
                            kv_length=block_kv_length,
                            active_layers=num_layers,
                        )
                    next_token = logits[:, -1:, :].argmax(dim=-1)
                    next_token[finished_flag] = tokenizer.pad_token_id
                    x_t = torch.cat([x_t, next_token], dim=1)
                    step += 1

                    break
                
                for small_block_idx in range(num_small_blocks):
                    small_block_start_idx = small_block_idx * small_block_size
                    small_block_end_idx = small_block_start_idx + small_block_size

                    start = -block_size + small_block_start_idx
                    end = None if block_size == small_block_end_idx else -block_size + small_block_end_idx
                    while True:
                        mask_idx = (x_t[:, -block_size:] == mask_id)
                        if mask_idx[:, start:end].sum() == 0:
                            break
                        
                        if use_block_cache:
                            if block_past_key_values is None or (x_t[:, -block_size+small_block_start_idx] == mask_id).any():
                                output = self.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=False, use_block_cache=True)
                                logits, block_past_key_values = output.logits, output.block_past_key_values
                                if estimate_flops:
                                    _record_decode_flops_step(self,
                                        step_kind="full_block",
                                        tokens_forwarded=int(x_t[:, -block_size:].shape[1]),
                                        kv_length=block_kv_length,
                                        active_layers=num_layers,
                                    )
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                                logits = logits[:, start:end]
                            else:
                                logits = self.forward(input_ids=x_t[:,start:end], use_cache=True, past_key_values=past_key_values, update_past_key_values=False, use_block_cache=True, block_past_key_values=block_past_key_values, replace_position=small_block_start_idx).logits
                                if estimate_flops:
                                    _record_decode_flops_step(self,
                                        step_kind="subblock",
                                        tokens_forwarded=int(x_t[:, start:end].shape[1]),
                                        kv_length=block_kv_length,
                                        active_layers=num_layers,
                                    )
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                        else:
                            logits = self.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=False).logits
                            if estimate_flops:
                                _record_decode_flops_step(self,
                                    step_kind="full_block",
                                    tokens_forwarded=int(x_t[:, -block_size:].shape[1]),
                                    kv_length=block_kv_length,
                                    active_layers=num_layers,
                                )
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                            logits = logits[:, start:end]
                        x_1, p_1t = self.sample_with_top_p(logits, top_p=top_p, temperature=temperature)
                        x1_p = torch.squeeze(torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1)
                        x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)

                        unmask_idx = (x1_p > threshold)
                        max_prob_idx = x1_p.argmax(dim=-1)
                        unmask_idx[torch.arange(x_1.shape[0]), max_prob_idx] = True
                        unmask_idx = unmask_idx & mask_idx[:, start:end]

                        x_t[:, start:end][unmask_idx] = x_1[unmask_idx]

                        finished_row_flags = ((x_1 == stop_token) & unmask_idx).any(dim=1) # shape: [B]
                        finished_flag = finished_flag | finished_row_flags

                        step += 1

            if input_ids.shape[1] ==  x_t.shape[1]:
                input_ids = x_t
            else:
                input_ids[:, :(block_idx + 1)*block_size] = x_t[:, :-1]
                if (seq_block_idx == block_idx).all():
                    input_ids = torch.cat([input_ids, x_t[:, -1:]], dim=1)
                else:
                    if input_ids.shape[1] <= (block_idx + 1)*block_size:
                        input_ids = x_t
                    else:
                        input_ids[seq_block_idx == block_idx, (block_idx + 1)*block_size] = x_t[seq_block_idx == block_idx, (block_idx + 1)*block_size]
            seq_block_idx[seq_block_idx == block_idx] = block_idx + 1
            if finished_flag.any():
                for sample_idx in range(x_t.shape[0]):
                    if finished_flag[sample_idx]:
                        original_idx = sample_indices[sample_idx].item()
                        finished_samples[original_idx] = x_t[sample_idx:sample_idx+1].clone().squeeze(dim=0)
                sample_indices = sample_indices[~finished_flag]
                input_ids = input_ids[~finished_flag]
                seq_block_idx = seq_block_idx[~finished_flag]
                seq_len = seq_len[~finished_flag]
                x_t = x_t[~finished_flag]

                for layer_id in range(len(past_key_values)):
                    past_key_values.key_cache[layer_id] = past_key_values.key_cache[layer_id][~finished_flag]
                    past_key_values.value_cache[layer_id] = past_key_values.value_cache[layer_id][~finished_flag]

                finished_flag = finished_flag[~finished_flag]



        # add not finished samples since max_new_tokens is reached
        if len(finished_samples) < batch_size:
            for sample_idx in range(x_t.shape[0]):
                original_idx = sample_indices[sample_idx].item()
                finished_samples[original_idx] = x_t[sample_idx:sample_idx+1].clone().squeeze(dim=0)
        
        assert len(finished_samples) == batch_size
        return finished_samples

    @torch.no_grad()
    def batch_sample_layer_policy(
        self,
        input_ids,
        tokenizer,
        block_size,
        max_new_tokens,
        small_block_size,
        min_len,
        seq_len,
        mask_id=151665,
        threshold=0.95,
        stop_token=151645,
        use_block_cache=False,
        top_p=0.95,
        temperature=0.0,
        layer_policy_threshold=0.99,
        layer_policy_freeze_step=3,
        track_layer_skip_stats=False,
        estimate_flops=False,
    ):
        """
        Layer-policy variant of batch_sample.
        If a layer's mean per-token cosine similarity (input vs output at a step) exceeds
        layer_policy_threshold, that layer is skipped for the next layer_policy_freeze_step
        decoding steps. Before the final full-block forward that writes into prefix cache,
        all layers are re-enabled.
        """
        num_blocks = max_new_tokens // block_size + seq_len.max().item() // block_size
        batch_size = input_ids.shape[0]
        if batch_size != 1:
            raise ValueError("batch_sample_layer_policy currently supports batch_size=1 only.")  # keep one global layer policy stream
        num_layers = self.config.num_hidden_layers  # number of layers to track/skip
        if estimate_flops and getattr(self, "_decode_flops_estimator", None) is None:
            _reset_decode_flops_estimator(self, "layer_policy", block_size)

        # Optional layer-skip counters:
        # - total_layer_skip_counts: how many decoding steps each layer is skipped globally.
        # - per_block_layer_skip_counts: same count but split by decoded block index.
        if track_layer_skip_stats:
            total_layer_skip_counts = torch.zeros((num_layers,), device=self.device, dtype=torch.long)  # global skipped-layer counter
            per_block_layer_skip_counts = {}  # block-level skipped-layer counters
        else:
            total_layer_skip_counts = None  # disable counter allocation when stats are not requested
            per_block_layer_skip_counts = None  # disable per-block counter storage when stats are not requested

        # only prefix cache if min_len > block_size
        if min_len > block_size:
            prefill_tokens = int(min_len // block_size * block_size)
            output = self.forward(
                input_ids=input_ids[:, :prefill_tokens],
                use_cache=True,
                update_past_key_values=True,
                block_size=block_size,
            )
            logits, past_key_values = output.logits, output.past_key_values
            if estimate_flops:
                _record_decode_flops_step(self,
                    step_kind="prefill",
                    tokens_forwarded=prefill_tokens,
                    kv_length=prefill_tokens,  # self-attention, no prior cache
                    active_layers=num_layers,
                )
            # we only do AR steps if the next decoding block doesn't have an initial token
            # we need this token for full block forward and the first subblock forwawrd since
            # we shift the logits and pos0 will use pos0's logit. Therefore, we need to make
            # sure pos0 is not masked for the first block.
            if min_len % block_size == 0:
                predict_sample_idx = (seq_len == min_len)
                predict_logits = logits[predict_sample_idx, -1:, :]
                next_token = predict_logits.argmax(dim=-1)
                # case when we don't have right paddings
                if input_ids.shape[1] <= min_len:
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                # case we do have right paddings
                else:
                    input_ids[predict_sample_idx, min_len] = next_token.squeeze(dim=-1)
        else:
            past_key_values = None

        seq_block_idx = seq_len // block_size
        finished_flag = torch.zeros((batch_size), device=self.device, dtype=torch.bool)

        start_block_idx = min_len // block_size
        num_small_blocks = block_size // small_block_size

        sample_indices = torch.arange(batch_size, device=self.device)
        finished_samples = {}
        for block_idx in range(start_block_idx, num_blocks):
            if finished_flag.all():
                break
            if (seq_block_idx == block_idx).all():
                x_init = mask_id * torch.ones(
                    (input_ids.shape[0], block_size - input_ids.shape[1] % block_size),
                    device=self.device,
                    dtype=torch.long,
                )
                x_init = torch.cat([input_ids, x_init], dim=1)
                input_ids = x_init
            else:
                x_init = input_ids[:, :(block_idx + 1) * block_size]

            x_init[finished_flag, -block_size:] = tokenizer.pad_token_id
            x_t = x_init.clone()
            step = 0
            block_past_key_values = None
            # Freeze counters for the current decoding block.
            layer_freeze_steps = torch.zeros((num_layers,), device=self.device, dtype=torch.long)  # remaining freeze steps per layer
            # Total KV length for attention: prefix cache + current block tokens.
            prefix_len = past_key_values.get_seq_length() if past_key_values is not None else 0
            block_kv_length = prefix_len + block_size
            # Per-block skip counter for reporting: layer i skipped k times in this block.
            if track_layer_skip_stats:
                block_layer_skip_counts = torch.zeros((num_layers,), device=self.device, dtype=torch.long)  # per-block skipped-layer counter
            else:
                block_layer_skip_counts = None  # disable per-block counter updates when stats are not requested
            while True:
                mask_idx = (x_t[:, -block_size:] == mask_id)
                if mask_idx.sum() == 0:
                    for sample_idx in range(x_t.shape[0]):
                        if finished_flag[sample_idx] and seq_len[sample_idx] < (block_idx + 1) * block_size:
                            stop_token_idx = (x_t[sample_idx, seq_len[sample_idx]:] == stop_token).nonzero()[0][0]
                            x_t[sample_idx, seq_len[sample_idx] + stop_token_idx + 1:] = tokenizer.pad_token_id
                    if finished_flag.all():
                        break
                    # IMPORTANT: re-enable all layers before final full-block forward
                    # that writes this block to prefix cache.
                    layer_freeze_steps.zero_()  # force all layers back before final prefix-cache write
                    # full block forward to update past_key_values for the next block and AR step to get the first token for the next block.
                    output = self.forward(
                        input_ids=x_t[:, -block_size:],
                        use_cache=True,
                        past_key_values=past_key_values,
                        update_past_key_values=True,
                        block_size=block_size,
                    )
                    logits, past_key_values = output.logits, output.past_key_values
                    if estimate_flops:
                        _record_decode_flops_step(self,
                            step_kind="full_block",
                            tokens_forwarded=int(x_t[:, -block_size:].shape[1]),
                            kv_length=block_kv_length,
                            active_layers=num_layers,
                        )
                    next_token = logits[:, -1:, :].argmax(dim=-1)
                    next_token[finished_flag] = tokenizer.pad_token_id
                    x_t = torch.cat([x_t, next_token], dim=1)
                    step += 1

                    break

                for small_block_idx in range(num_small_blocks):
                    small_block_start_idx = small_block_idx * small_block_size
                    small_block_end_idx = small_block_start_idx + small_block_size

                    start = -block_size + small_block_start_idx
                    end = None if block_size == small_block_end_idx else -block_size + small_block_end_idx
                    while True:
                        mask_idx = (x_t[:, -block_size:] == mask_id)
                        if mask_idx[:, start:end].sum() == 0:
                            break

                        # Layer policy (step-level):
                        # 1) build layer skip mask from freeze counters
                        # 2) run forward with skip_layer_mask and collect inline layer cosine
                        # 3) refresh freeze counters for future steps
                        skip_layer_mask_gpu = layer_freeze_steps > 0  # layers with freeze budget left are skipped now
                        if track_layer_skip_stats:
                            skip_counts_this_step = skip_layer_mask_gpu.to(dtype=torch.long)  # 1 if skipped at this step
                            block_layer_skip_counts += skip_counts_this_step  # accumulate block-level skip counts
                            total_layer_skip_counts += skip_counts_this_step  # accumulate global skip counts
                        skip_layer_mask = skip_layer_mask_gpu.detach().cpu()  # move mask once to CPU to avoid per-layer CUDA sync in model loop

                        if use_block_cache:
                            if block_past_key_values is None:
                                tokens_forwarded = int(x_t[:, -block_size:].shape[1])
                                output = self.forward(
                                    input_ids=x_t[:, -block_size:],
                                    use_cache=True,
                                    past_key_values=past_key_values,
                                    update_past_key_values=False,
                                    use_block_cache=True,
                                    skip_layer_mask=skip_layer_mask,  # actual layer skipping happens in model.forward
                                    collect_layer_policy_cos=True,  # collect per-layer mean cosine inline in model.forward
                                )
                                logits, block_past_key_values = output.logits, output.block_past_key_values
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                                logits = logits[:, start:end]
                            elif (x_t[:, -block_size + small_block_start_idx] == mask_id).any():
                                # Keep existing block cache and refresh full block by replacing from position 0.
                                tokens_forwarded = int(x_t[:, -block_size:].shape[1])
                                output = self.forward(
                                    input_ids=x_t[:, -block_size:],
                                    use_cache=True,
                                    past_key_values=past_key_values,
                                    update_past_key_values=False,
                                    use_block_cache=True,
                                    block_past_key_values=block_past_key_values,
                                    replace_position=0,
                                    skip_layer_mask=skip_layer_mask,  # actual layer skipping happens in model.forward
                                    collect_layer_policy_cos=True,  # collect per-layer mean cosine inline in model.forward
                                )
                                logits, block_past_key_values = output.logits, output.block_past_key_values
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                                logits = logits[:, start:end]
                            else:
                                tokens_forwarded = int(x_t[:, start:end].shape[1])
                                output = self.forward(
                                    input_ids=x_t[:, start:end],
                                    use_cache=True,
                                    past_key_values=past_key_values,
                                    update_past_key_values=False,
                                    use_block_cache=True,
                                    block_past_key_values=block_past_key_values,
                                    replace_position=small_block_start_idx,
                                    skip_layer_mask=skip_layer_mask,  # actual layer skipping happens in model.forward
                                    collect_layer_policy_cos=True,  # collect per-layer mean cosine inline in model.forward
                                )
                                logits = output.logits
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                        else:
                            tokens_forwarded = int(x_t[:, -block_size:].shape[1])
                            output = self.forward(
                                input_ids=x_t[:, -block_size:],
                                use_cache=True,
                                past_key_values=past_key_values,
                                update_past_key_values=False,
                                skip_layer_mask=skip_layer_mask,  # actual layer skipping happens in model.forward
                                collect_layer_policy_cos=True,  # collect per-layer mean cosine inline in model.forward
                            )
                            logits = output.logits
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                            logits = logits[:, start:end]

                        active_freeze = layer_freeze_steps > 0  # layers that were skipped this step
                        layer_freeze_steps[active_freeze] -= 1  # consume one freeze step for active frozen layers
                        layer_cos = output.layer_policy_cos  # shape [num_layers], NaN for skipped layers
                        valid_layer_count = 0
                        if layer_cos is not None:
                            valid_cos = ~torch.isnan(layer_cos)  # skipped layers keep NaN and are ignored for threshold trigger
                            valid_layer_count = int(valid_cos.sum().item())  # number of layers actually computed this step
                            trigger_freeze = _layer_trigger(layer_cos, output.layer_policy_l2, valid_cos, layer_policy_threshold)  # metric-aware (cos>thr or l2<thr)
                            layer_freeze_steps = torch.where(
                                trigger_freeze,
                                torch.full_like(layer_freeze_steps, int(layer_policy_freeze_step)),  # assign N freeze steps in one tensor op
                                layer_freeze_steps,
                            )
                        else:
                            valid_layer_count = int((~skip_layer_mask_gpu).sum().item())

                        if estimate_flops:
                            _record_decode_flops_step(self,
                                step_kind="full_block" if tokens_forwarded >= block_size else "subblock",
                                tokens_forwarded=tokens_forwarded,
                                kv_length=block_kv_length,
                                active_layers=valid_layer_count,
                                cosine_token_pairs=(tokens_forwarded * valid_layer_count * int(x_t.shape[0])),
                            )

                        x_1, p_1t = self.sample_with_top_p(logits, top_p=top_p, temperature=temperature)
                        x1_p = torch.squeeze(torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1)
                        x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)

                        unmask_idx = (x1_p > threshold)
                        max_prob_idx = x1_p.argmax(dim=-1)
                        unmask_idx[torch.arange(x_1.shape[0]), max_prob_idx] = True
                        unmask_idx = unmask_idx & mask_idx[:, start:end]

                        x_t[:, start:end][unmask_idx] = x_1[unmask_idx]

                        finished_row_flags = ((x_1 == stop_token) & unmask_idx).any(dim=1)  # shape: [B]
                        finished_flag = finished_flag | finished_row_flags

                        step += 1

            # Save per-block skip counts after the block decoding loop is complete.
            if track_layer_skip_stats:
                per_block_layer_skip_counts[block_idx] = block_layer_skip_counts.detach().cpu().tolist()  # persist per-block skip counts

            if input_ids.shape[1] == x_t.shape[1]:
                input_ids = x_t
            else:
                input_ids[:, :(block_idx + 1) * block_size] = x_t[:, :-1]
                if (seq_block_idx == block_idx).all():
                    input_ids = torch.cat([input_ids, x_t[:, -1:]], dim=1)
                else:
                    if input_ids.shape[1] <= (block_idx + 1) * block_size:
                        input_ids = x_t
                    else:
                        input_ids[seq_block_idx == block_idx, (block_idx + 1) * block_size] = x_t[
                            seq_block_idx == block_idx, (block_idx + 1) * block_size
                        ]
            seq_block_idx[seq_block_idx == block_idx] = block_idx + 1
            if finished_flag.any():
                for sample_idx in range(x_t.shape[0]):
                    if finished_flag[sample_idx]:
                        original_idx = sample_indices[sample_idx].item()
                        finished_samples[original_idx] = x_t[sample_idx:sample_idx + 1].clone().squeeze(dim=0)
                sample_indices = sample_indices[~finished_flag]
                input_ids = input_ids[~finished_flag]
                seq_block_idx = seq_block_idx[~finished_flag]
                seq_len = seq_len[~finished_flag]
                x_t = x_t[~finished_flag]

                for layer_id in range(len(past_key_values)):
                    past_key_values.key_cache[layer_id] = past_key_values.key_cache[layer_id][~finished_flag]
                    past_key_values.value_cache[layer_id] = past_key_values.value_cache[layer_id][~finished_flag]

                finished_flag = finished_flag[~finished_flag]

        # add not finished samples since max_new_tokens is reached
        if len(finished_samples) < batch_size:
            for sample_idx in range(x_t.shape[0]):
                original_idx = sample_indices[sample_idx].item()
                finished_samples[original_idx] = x_t[sample_idx:sample_idx + 1].clone().squeeze(dim=0)

        assert len(finished_samples) == batch_size
        # Export and print layer-skip stats only when explicitly requested.
        if track_layer_skip_stats:
            self.layer_policy_skip_stats = {  # expose counters for external inspection
                "total": {
                    f"layer_{layer_idx}": int(total_layer_skip_counts[layer_idx].item())
                    for layer_idx in range(num_layers)
                },
                "per_block": {
                    int(block_idx): {
                        f"layer_{layer_idx}": int(per_block_layer_skip_counts[block_idx][layer_idx])
                        for layer_idx in range(num_layers)
                    }
                    for block_idx in sorted(per_block_layer_skip_counts.keys())
                },
            }
            # Print total skipped counts per layer for quick terminal inspection.
            for layer_idx in range(num_layers):
                skip_cnt = int(total_layer_skip_counts[layer_idx].item())
                print(f"layer {layer_idx} got skipped {skip_cnt} times throughout the decoding")
        else:
            self.layer_policy_skip_stats = None  # clear stats when tracking is disabled
        return finished_samples

    @torch.no_grad()
    def batch_sample_token_policy_flashattention(
        self,
        input_ids,
        tokenizer,
        block_size,
        max_new_tokens,
        small_block_size,
        min_len,
        seq_len,
        mask_id=151665,
        threshold=0.95,
        stop_token=151645,
        use_block_cache=False,
        top_p=0.95,
        temperature=0.0,
        token_policy_threshold=0.99,
        token_policy_freeze_step=3,
        estimate_flops=False,
    ):
        """
        Token-policy decoding with a single sparse forward per step on active tokens.
        This uses Q for active tokens and full K/V from the block cache (FlashAttention via SDPA).
        """
        num_blocks = max_new_tokens // block_size + seq_len.max().item() // block_size
        batch_size = input_ids.shape[0]
        if batch_size != 1:
            # Sparse token-policy is implemented for single-request decoding.
            raise ValueError("batch_sample_token_policy_flashattention currently supports batch_size=1 only.")
        num_layers = self.config.num_hidden_layers
        if estimate_flops and getattr(self, "_decode_flops_estimator", None) is None:
            _reset_decode_flops_estimator(self, "token_policy", block_size)

        if min_len > block_size:
            prefill_tokens = int(min_len // block_size * block_size)
            output = self.forward(
                input_ids=input_ids[:, :prefill_tokens],
                use_cache=True,
                update_past_key_values=True,
                block_size=block_size,
            )
            logits, past_key_values = output.logits, output.past_key_values
            if estimate_flops:
                _record_decode_flops_step(self,
                    step_kind="prefill",
                    tokens_forwarded=prefill_tokens,
                    kv_length=prefill_tokens,  # self-attention, no prior cache
                    active_layers=num_layers,
                )
            if min_len % block_size == 0:
                predict_sample_idx = (seq_len == min_len)
                predict_logits = logits[predict_sample_idx, -1:, :]
                next_token = predict_logits.argmax(dim=-1)
                if input_ids.shape[1] <= min_len:
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                else:
                    input_ids[predict_sample_idx, min_len] = next_token.squeeze(dim=-1)
        else:
            past_key_values = None

        seq_block_idx = seq_len // block_size
        finished_flag = torch.zeros((batch_size), device=self.device, dtype=torch.bool)

        start_block_idx = min_len // block_size
        num_small_blocks = block_size // small_block_size

        sample_indices = torch.arange(batch_size, device=self.device)
        finished_samples = {}
        for block_idx in range(start_block_idx, num_blocks):
            if finished_flag.all():
                break

            if (seq_block_idx == block_idx).all():
                x_init = mask_id * torch.ones(
                    (input_ids.shape[0], block_size - input_ids.shape[1] % block_size),
                    device=self.device,
                    dtype=torch.long,
                )
                x_init = torch.cat([input_ids, x_init], dim=1)
                input_ids = x_init
            else:
                x_init = input_ids[:, :(block_idx + 1) * block_size]

            x_init[finished_flag, -block_size:] = tokenizer.pad_token_id
            x_t = x_init.clone()

            # Token-policy state for current block.
            freeze_steps_left = torch.zeros((block_size,), device=self.device, dtype=torch.long)
            prev_hidden = [None] * block_size
            cached_hidden = None
            cached_logits = None
            block_past_key_values = None
            step = 0
            # Total KV length for attention: prefix cache + current block tokens.
            prefix_len = past_key_values.get_seq_length() if past_key_values is not None else 0
            block_kv_length = prefix_len + block_size

            while True:
                mask_idx = (x_t[:, -block_size:] == mask_id)
                if mask_idx.sum() == 0:
                    # Release frozen tokens before the final full-block KV update.
                    freeze_steps_left.zero_()
                    for sample_idx in range(x_t.shape[0]):
                        if finished_flag[sample_idx] and seq_len[sample_idx] < (block_idx + 1) * block_size:
                            stop_token_idx = (x_t[sample_idx, seq_len[sample_idx]:] == stop_token).nonzero()[0][0]
                            x_t[sample_idx, seq_len[sample_idx] + stop_token_idx + 1:] = tokenizer.pad_token_id
                    if finished_flag.all():
                        break
                    output = self.forward(
                        input_ids=x_t[:, -block_size:],
                        use_cache=True,
                        past_key_values=past_key_values,
                        update_past_key_values=True,
                        block_size=block_size,
                    )
                    logits, past_key_values = output.logits, output.past_key_values
                    if estimate_flops:
                        _record_decode_flops_step(self,
                            step_kind="full_block",
                            tokens_forwarded=int(x_t[:, -block_size:].shape[1]),
                            kv_length=block_kv_length,
                            active_layers=num_layers,
                        )
                    next_token = logits[:, -1:, :].argmax(dim=-1)
                    next_token[finished_flag] = tokenizer.pad_token_id
                    x_t = torch.cat([x_t, next_token], dim=1)
                    step += 1
                    break

                for small_block_idx in range(num_small_blocks):
                    small_block_start_idx = small_block_idx * small_block_size
                    small_block_end_idx = small_block_start_idx + small_block_size
                    start = -block_size + small_block_start_idx
                    end = None if block_size == small_block_end_idx else -block_size + small_block_end_idx

                    while True:
                        mask_idx = (x_t[:, -block_size:] == mask_id)
                        if mask_idx[:, start:end].sum() == 0:
                            break

                        decoder_tokens_this_step = 0
                        cosine_pairs_this_step = 0
                        updated_positions = []
                        newly_frozen_positions = []
                        if block_past_key_values is None:
                            # First step in this block (both cache modes): full-block forward to initialize cache/hidden.
                            block_tokens = x_t[:, -block_size:]
                            decoder_tokens_this_step = int(block_tokens.shape[1])
                            output = self.forward(
                                input_ids=block_tokens,
                                use_cache=True,
                                past_key_values=past_key_values,
                                update_past_key_values=False,
                                use_block_cache=True,
                                output_hidden_states=True,
                            )
                            block_past_key_values = output.block_past_key_values
                            hidden_block = output.hidden_states[-1]
                            if hidden_block.dim() == 2:
                                hidden_block = hidden_block.unsqueeze(0)
                            if cached_hidden is None:
                                cached_hidden = hidden_block.new_zeros(
                                    (hidden_block.shape[0], block_size, hidden_block.shape[-1])
                                )
                            if cached_logits is None:
                                cached_logits = output.logits.new_zeros(
                                    (output.logits.shape[0], block_size, output.logits.shape[-1])
                                )
                            cached_hidden[:] = hidden_block.detach()
                            cached_logits[:] = output.logits.detach()
                            updated_positions = list(range(block_size))
                        elif use_block_cache and (
                            x_t[:, -block_size + small_block_start_idx] == mask_id
                        ).any():
                            # originally we do full block forward and reinitlaize the dynamic cache
                            # now, we don't reinitialize the cache but instead update it using sparse forard
                            # First subblock token still masked: refresh using active positions only.
                            block_tokens = x_t[:, -block_size:]
                            active_positions = (freeze_steps_left == 0).nonzero().flatten()
                            decoder_tokens_this_step = int(active_positions.numel())
                            if active_positions.numel() > 0:
                                output = self.forward_block_positions(
                                    input_ids=block_tokens[:, active_positions],
                                    past_key_values=past_key_values,
                                    block_past_key_values=block_past_key_values,
                                    replace_positions=active_positions,
                                    output_hidden_states=True,
                                )
                                block_past_key_values = output.block_past_key_values
                                hidden_block = output.hidden_states[-1]
                                if hidden_block.dim() == 2:
                                    hidden_block = hidden_block.unsqueeze(0)
                                if cached_hidden is None:
                                    cached_hidden = hidden_block.new_zeros(
                                        (hidden_block.shape[0], block_size, hidden_block.shape[-1])
                                    )
                                cached_hidden[:, active_positions, :] = hidden_block.detach()
                                cached_logits[:, active_positions, :] = output.logits.detach()
                                updated_positions = active_positions.tolist()
                        elif use_block_cache:
                            # Sparse forward: update active positions limited to current subblock.
                            active_positions = (freeze_steps_left == 0).nonzero().flatten()
                            active_positions = active_positions[
                                (active_positions >= small_block_start_idx)
                                & (active_positions < small_block_end_idx)
                            ]
                            # print(
                            #     f"[token_policy] active tokens number={int(active_positions.numel())}"
                            # )
                            if active_positions.numel() > 0:
                                decoder_tokens_this_step = int(active_positions.numel())
                                block_tokens = x_t[:, -block_size:]
                                output = self.forward_block_positions(
                                    input_ids=block_tokens[:, active_positions],
                                    past_key_values=past_key_values,
                                    block_past_key_values=block_past_key_values,
                                    replace_positions=active_positions,
                                    output_hidden_states=True,
                                )
                                block_past_key_values = output.block_past_key_values
                                hidden_active = output.hidden_states[-1]
                                if hidden_active.dim() == 2:
                                    hidden_active = hidden_active.unsqueeze(0)
                                cached_hidden[:, active_positions, :] = hidden_active.detach()
                                cached_logits[:, active_positions, :] = output.logits.detach()
                                updated_positions = active_positions.tolist()
                        else:
                            # use_block_cache=False: sparse forward across all active positions in the block.
                            active_positions = (freeze_steps_left == 0).nonzero().flatten()
                            # print(
                            #     f"[token_policy] active tokens number={int(active_positions.numel())}"
                            # )
                            if active_positions.numel() > 0:
                                decoder_tokens_this_step = int(active_positions.numel())
                                block_tokens = x_t[:, -block_size:]
                                output = self.forward_block_positions(
                                    input_ids=block_tokens[:, active_positions],
                                    past_key_values=past_key_values,
                                    block_past_key_values=block_past_key_values,
                                    replace_positions=active_positions,
                                    output_hidden_states=True,
                                )
                                block_past_key_values = output.block_past_key_values
                                hidden_active = output.hidden_states[-1]
                                if hidden_active.dim() == 2:
                                    hidden_active = hidden_active.unsqueeze(0)
                                cached_hidden[:, active_positions, :] = hidden_active.detach()
                                cached_logits[:, active_positions, :] = output.logits.detach()
                                updated_positions = active_positions.tolist()

                        # Update token policy based on cosine similarity for updated positions.
                        for pos in updated_positions:
                            cur = cached_hidden[0, pos]
                            if prev_hidden[pos] is not None:
                                cosine_pairs_this_step += 1
                                if _token_is_stable(prev_hidden[pos].float(), cur.float(), token_policy_threshold):
                                    # Mark token to be frozen for upcoming decoding steps.
                                    newly_frozen_positions.append(pos)
                            prev_hidden[pos] = cur.detach().clone()

                        if decoder_tokens_this_step > 0:
                            if estimate_flops:
                                _record_decode_flops_step(self,
                                    step_kind="full_block" if decoder_tokens_this_step >= block_size else "sparse",
                                    tokens_forwarded=decoder_tokens_this_step,
                                    kv_length=block_kv_length,
                                    active_layers=num_layers,
                                    cosine_token_pairs=cosine_pairs_this_step,
                                )

                        # Reuse cached logits (updated only at active positions) to avoid duplicate lm_head work.
                        if cached_logits is None:
                            raise RuntimeError("token policy cached_logits is None before sampling")
                        logits_full = cached_logits
                        logits_full = torch.cat([logits_full[:, :1, :], logits_full[:, :-1, :]], dim=1)
                        logits = logits_full[:, start:end]

                        x_1, p_1t = self.sample_with_top_p(logits, top_p=top_p, temperature=temperature)
                        x1_p = torch.squeeze(torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1)
                        x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)

                        unmask_idx = (x1_p > threshold)
                        max_prob_idx = x1_p.argmax(dim=-1)
                        unmask_idx[torch.arange(x_1.shape[0]), max_prob_idx] = True
                        unmask_idx = unmask_idx & mask_idx[:, start:end]

                        x_t[:, start:end][unmask_idx] = x_1[unmask_idx]
                        finished_row_flags = ((x_1 == stop_token) & unmask_idx).any(dim=1)
                        finished_flag = finished_flag | finished_row_flags

                        # Consume one freeze step only after an actual decode step happened.
                        freeze_steps_left = torch.where(
                            freeze_steps_left > 0, freeze_steps_left - 1, freeze_steps_left
                        )
                        # Apply new freezes after consuming current-step budget so N means N future steps.
                        if newly_frozen_positions:
                            freeze_steps_left[
                                torch.tensor(newly_frozen_positions, device=self.device, dtype=torch.long)
                            ] = token_policy_freeze_step
                        # print(
                        #     f"[token_policy] freeze_steps_left={freeze_steps_left.tolist()}"
                        # )

                        step += 1

            if input_ids.shape[1] == x_t.shape[1]:
                input_ids = x_t
            else:
                input_ids[:, :(block_idx + 1) * block_size] = x_t[:, :-1]
                if (seq_block_idx == block_idx).all():
                    input_ids = torch.cat([input_ids, x_t[:, -1:]], dim=1)
                else:
                    if input_ids.shape[1] <= (block_idx + 1) * block_size:
                        input_ids = x_t
                    else:
                        input_ids[
                            seq_block_idx == block_idx, (block_idx + 1) * block_size
                        ] = x_t[seq_block_idx == block_idx, (block_idx + 1) * block_size]
            seq_block_idx[seq_block_idx == block_idx] = block_idx + 1

            if finished_flag.any():
                for sample_idx in range(x_t.shape[0]):
                    if finished_flag[sample_idx]:
                        original_idx = sample_indices[sample_idx].item()
                        finished_samples[original_idx] = x_t[sample_idx:sample_idx + 1].clone().squeeze(dim=0)

                sample_indices = sample_indices[~finished_flag]
                input_ids = input_ids[~finished_flag]
                seq_block_idx = seq_block_idx[~finished_flag]
                seq_len = seq_len[~finished_flag]
                x_t = x_t[~finished_flag]

                for layer_id in range(len(past_key_values)):
                    past_key_values.key_cache[layer_id] = past_key_values.key_cache[layer_id][~finished_flag]
                    past_key_values.value_cache[layer_id] = past_key_values.value_cache[layer_id][~finished_flag]

                finished_flag = finished_flag[~finished_flag]

        if len(finished_samples) < batch_size:
            for sample_idx in range(x_t.shape[0]):
                original_idx = sample_indices[sample_idx].item()
                finished_samples[original_idx] = x_t[sample_idx:sample_idx + 1].clone().squeeze(dim=0)

        assert len(finished_samples) == batch_size
        return finished_samples

    @torch.no_grad()
    def batch_sample_layer_and_token_policy(
        self,
        input_ids,
        tokenizer,
        block_size,
        max_new_tokens,
        small_block_size,
        min_len,
        seq_len,
        mask_id=151665,
        threshold=0.95,
        stop_token=151645,
        use_block_cache=False,
        top_p=0.95,
        temperature=0.0,
        layer_policy_threshold=0.99,
        layer_policy_freeze_step=3,
        token_policy_threshold=0.99,
        token_policy_freeze_step=3,
        estimate_flops=False,
    ):
        """
        Combined layer+token policy decoding:
        - Layer policy: skip layers by per-layer mean cosine(input_i, output_i).
        - Token policy: sparse token updates by per-token cosine on final hidden states.
        """
        num_blocks = max_new_tokens // block_size + seq_len.max().item() // block_size
        batch_size = input_ids.shape[0]
        if batch_size != 1:
            raise ValueError("batch_sample_layer_and_token_policy currently supports batch_size=1 only.")
        num_layers = self.config.num_hidden_layers
        if estimate_flops and getattr(self, "_decode_flops_estimator", None) is None:
            _reset_decode_flops_estimator(self, "layer_and_token_policy", block_size)

        if min_len > block_size:
            prefill_tokens = int(min_len // block_size * block_size)
            output = self.forward(
                input_ids=input_ids[:, :prefill_tokens],
                use_cache=True,
                update_past_key_values=True,
                block_size=block_size,
            )
            logits, past_key_values = output.logits, output.past_key_values
            if estimate_flops:
                _record_decode_flops_step(
                    self,
                    step_kind="prefill",
                    tokens_forwarded=prefill_tokens,
                    kv_length=prefill_tokens,  # self-attention, no prior cache
                    active_layers=num_layers,
                )
            if min_len % block_size == 0:
                predict_sample_idx = (seq_len == min_len)
                predict_logits = logits[predict_sample_idx, -1:, :]
                next_token = predict_logits.argmax(dim=-1)
                if input_ids.shape[1] <= min_len:
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                else:
                    input_ids[predict_sample_idx, min_len] = next_token.squeeze(dim=-1)
        else:
            past_key_values = None

        seq_block_idx = seq_len // block_size
        finished_flag = torch.zeros((batch_size), device=self.device, dtype=torch.bool)
        start_block_idx = min_len // block_size
        num_small_blocks = block_size // small_block_size

        sample_indices = torch.arange(batch_size, device=self.device)
        finished_samples = {}
        for block_idx in range(start_block_idx, num_blocks):
            if finished_flag.all():
                break

            if (seq_block_idx == block_idx).all():
                x_init = mask_id * torch.ones(
                    (input_ids.shape[0], block_size - input_ids.shape[1] % block_size),
                    device=self.device,
                    dtype=torch.long,
                )
                x_init = torch.cat([input_ids, x_init], dim=1)
                input_ids = x_init
            else:
                x_init = input_ids[:, :(block_idx + 1) * block_size]

            x_init[finished_flag, -block_size:] = tokenizer.pad_token_id
            x_t = x_init.clone()

            # Token-policy state for current block.
            freeze_steps_left = torch.zeros((block_size,), device=self.device, dtype=torch.long)
            prev_hidden = [None] * block_size
            cached_hidden = None
            cached_logits = None
            # Layer-policy state for current block.
            layer_freeze_steps = torch.zeros((num_layers,), device=self.device, dtype=torch.long)
            block_past_key_values = None
            step = 0
            # Total KV length for attention: prefix cache + current block tokens.
            prefix_len = past_key_values.get_seq_length() if past_key_values is not None else 0
            block_kv_length = prefix_len + block_size

            while True:
                mask_idx = (x_t[:, -block_size:] == mask_id)
                if mask_idx.sum() == 0:
                    # Combined policy reset before final full-block prefix-cache write.
                    freeze_steps_left.zero_()
                    layer_freeze_steps.zero_()
                    for sample_idx in range(x_t.shape[0]):
                        if finished_flag[sample_idx] and seq_len[sample_idx] < (block_idx + 1) * block_size:
                            stop_token_idx = (x_t[sample_idx, seq_len[sample_idx]:] == stop_token).nonzero()[0][0]
                            x_t[sample_idx, seq_len[sample_idx] + stop_token_idx + 1:] = tokenizer.pad_token_id
                    if finished_flag.all():
                        break
                    output = self.forward(
                        input_ids=x_t[:, -block_size:],
                        use_cache=True,
                        past_key_values=past_key_values,
                        update_past_key_values=True,
                        block_size=block_size,
                    )
                    logits, past_key_values = output.logits, output.past_key_values
                    if estimate_flops:
                        _record_decode_flops_step(
                            self,
                            step_kind="full_block",
                            tokens_forwarded=int(x_t[:, -block_size:].shape[1]),
                            kv_length=block_kv_length,
                            active_layers=num_layers,
                        )
                    next_token = logits[:, -1:, :].argmax(dim=-1)
                    next_token[finished_flag] = tokenizer.pad_token_id
                    x_t = torch.cat([x_t, next_token], dim=1)
                    step += 1
                    break

                for small_block_idx in range(num_small_blocks):
                    small_block_start_idx = small_block_idx * small_block_size
                    small_block_end_idx = small_block_start_idx + small_block_size
                    start = -block_size + small_block_start_idx
                    end = None if block_size == small_block_end_idx else -block_size + small_block_end_idx

                    while True:
                        mask_idx = (x_t[:, -block_size:] == mask_id)
                        if mask_idx[:, start:end].sum() == 0:
                            break

                        # Layer-policy: freeze mask for this step; frozen layers are skipped in model.forward.
                        skip_layer_mask_gpu = layer_freeze_steps > 0
                        skip_layer_mask = skip_layer_mask_gpu.detach().cpu()

                        decoder_tokens_this_step = 0
                        cosine_pairs_this_step = 0
                        updated_positions = []
                        newly_frozen_positions = []
                        output = None

                        if block_past_key_values is None:
                            # First step in block: full-block forward initializes block cache + hidden/logits cache.
                            block_tokens = x_t[:, -block_size:]
                            decoder_tokens_this_step = int(block_tokens.shape[1])
                            output = self.forward(
                                input_ids=block_tokens,
                                use_cache=True,
                                past_key_values=past_key_values,
                                update_past_key_values=False,
                                use_block_cache=True,
                                output_hidden_states=True,
                                skip_layer_mask=skip_layer_mask,
                                collect_layer_policy_cos=True,
                            )
                            block_past_key_values = output.block_past_key_values
                            hidden_block = output.hidden_states[-1]
                            if hidden_block.dim() == 2:
                                hidden_block = hidden_block.unsqueeze(0)
                            if cached_hidden is None:
                                cached_hidden = hidden_block.new_zeros(
                                    (hidden_block.shape[0], block_size, hidden_block.shape[-1])
                                )
                            if cached_logits is None:
                                cached_logits = output.logits.new_zeros(
                                    (output.logits.shape[0], block_size, output.logits.shape[-1])
                                )
                            cached_hidden[:] = hidden_block.detach()
                            cached_logits[:] = output.logits.detach()
                            updated_positions = list(range(block_size))
                        elif use_block_cache and (x_t[:, -block_size + small_block_start_idx] == mask_id).any():
                            # Full-block sparse refresh when subblock head token is still masked.
                            block_tokens = x_t[:, -block_size:]
                            active_positions = (freeze_steps_left == 0).nonzero().flatten()
                            decoder_tokens_this_step = int(active_positions.numel())
                            if active_positions.numel() > 0:
                                output = self.forward(
                                    input_ids=block_tokens[:, active_positions],
                                    use_cache=True,
                                    past_key_values=past_key_values,
                                    update_past_key_values=False,
                                    use_block_cache=True,
                                    block_past_key_values=block_past_key_values,
                                    replace_positions=active_positions,
                                    output_hidden_states=True,
                                    skip_layer_mask=skip_layer_mask,
                                    collect_layer_policy_cos=True,
                                )
                                block_past_key_values = output.block_past_key_values
                                hidden_block = output.hidden_states[-1]
                                if hidden_block.dim() == 2:
                                    hidden_block = hidden_block.unsqueeze(0)
                                cached_hidden[:, active_positions, :] = hidden_block.detach()
                                cached_logits[:, active_positions, :] = output.logits.detach()
                                updated_positions = active_positions.tolist()
                        elif use_block_cache:
                            # Subblock sparse refresh: only active positions within current subblock.
                            active_positions = (freeze_steps_left == 0).nonzero().flatten()
                            active_positions = active_positions[
                                (active_positions >= small_block_start_idx)
                                & (active_positions < small_block_end_idx)
                            ]
                            if active_positions.numel() > 0:
                                decoder_tokens_this_step = int(active_positions.numel())
                                block_tokens = x_t[:, -block_size:]
                                output = self.forward(
                                    input_ids=block_tokens[:, active_positions],
                                    use_cache=True,
                                    past_key_values=past_key_values,
                                    update_past_key_values=False,
                                    use_block_cache=True,
                                    block_past_key_values=block_past_key_values,
                                    replace_positions=active_positions,
                                    output_hidden_states=True,
                                    skip_layer_mask=skip_layer_mask,
                                    collect_layer_policy_cos=True,
                                )
                                block_past_key_values = output.block_past_key_values
                                hidden_active = output.hidden_states[-1]
                                if hidden_active.dim() == 2:
                                    hidden_active = hidden_active.unsqueeze(0)
                                cached_hidden[:, active_positions, :] = hidden_active.detach()
                                cached_logits[:, active_positions, :] = output.logits.detach()
                                updated_positions = active_positions.tolist()
                        else:
                            # No-block-cache mode: sparse refresh across all active positions in block.
                            active_positions = (freeze_steps_left == 0).nonzero().flatten()
                            if active_positions.numel() > 0:
                                decoder_tokens_this_step = int(active_positions.numel())
                                block_tokens = x_t[:, -block_size:]
                                output = self.forward(
                                    input_ids=block_tokens[:, active_positions],
                                    use_cache=True,
                                    past_key_values=past_key_values,
                                    update_past_key_values=False,
                                    use_block_cache=True,
                                    block_past_key_values=block_past_key_values,
                                    replace_positions=active_positions,
                                    output_hidden_states=True,
                                    skip_layer_mask=skip_layer_mask,
                                    collect_layer_policy_cos=True,
                                )
                                block_past_key_values = output.block_past_key_values
                                hidden_active = output.hidden_states[-1]
                                if hidden_active.dim() == 2:
                                    hidden_active = hidden_active.unsqueeze(0)
                                cached_hidden[:, active_positions, :] = hidden_active.detach()
                                cached_logits[:, active_positions, :] = output.logits.detach()
                                updated_positions = active_positions.tolist()

                        # Token policy: compare updated token hidden with previous snapshot.
                        for pos in updated_positions:
                            cur = cached_hidden[0, pos]
                            if prev_hidden[pos] is not None:
                                cosine_pairs_this_step += 1
                                if _token_is_stable(prev_hidden[pos].float(), cur.float(), token_policy_threshold):
                                    newly_frozen_positions.append(pos)
                            prev_hidden[pos] = cur.detach().clone()

                        # Layer policy: consume freeze budget every decode step.
                        active_layer_freeze = layer_freeze_steps > 0
                        layer_freeze_steps[active_layer_freeze] -= 1
                        valid_layer_count = int((~skip_layer_mask_gpu).sum().item())
                        # Layer policy: trigger new freezes from this step's per-layer cosine.
                        if output is not None and output.layer_policy_cos is not None:
                            layer_cos = output.layer_policy_cos
                            valid_cos = ~torch.isnan(layer_cos)
                            valid_layer_count = int(valid_cos.sum().item())
                            trigger_freeze = _layer_trigger(layer_cos, output.layer_policy_l2, valid_cos, layer_policy_threshold)
                            layer_freeze_steps = torch.where(
                                trigger_freeze,
                                torch.full_like(layer_freeze_steps, int(layer_policy_freeze_step)),
                                layer_freeze_steps,
                            )

                        if decoder_tokens_this_step > 0 and estimate_flops:
                            _record_decode_flops_step(
                                self,
                                step_kind="full_block" if decoder_tokens_this_step >= block_size else "sparse",
                                tokens_forwarded=decoder_tokens_this_step,
                                kv_length=block_kv_length,
                                active_layers=valid_layer_count,
                                cosine_token_pairs=(
                                    cosine_pairs_this_step + (decoder_tokens_this_step * valid_layer_count * int(x_t.shape[0]))
                                ),
                            )

                        # Reuse cached logits (updated only at active positions) for sampling.
                        if cached_logits is None:
                            raise RuntimeError("combined policy cached_logits is None before sampling")
                        logits_full = cached_logits
                        logits_full = torch.cat([logits_full[:, :1, :], logits_full[:, :-1, :]], dim=1)
                        logits = logits_full[:, start:end]

                        x_1, p_1t = self.sample_with_top_p(logits, top_p=top_p, temperature=temperature)
                        x1_p = torch.squeeze(torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1)
                        x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)

                        unmask_idx = (x1_p > threshold)
                        max_prob_idx = x1_p.argmax(dim=-1)
                        unmask_idx[torch.arange(x_1.shape[0]), max_prob_idx] = True
                        unmask_idx = unmask_idx & mask_idx[:, start:end]
                        x_t[:, start:end][unmask_idx] = x_1[unmask_idx]

                        finished_row_flags = ((x_1 == stop_token) & unmask_idx).any(dim=1)
                        finished_flag = finished_flag | finished_row_flags

                        # Token policy: consume and then assign token freeze budget.
                        freeze_steps_left = torch.where(
                            freeze_steps_left > 0, freeze_steps_left - 1, freeze_steps_left
                        )
                        if newly_frozen_positions:
                            freeze_steps_left[
                                torch.tensor(newly_frozen_positions, device=self.device, dtype=torch.long)
                            ] = token_policy_freeze_step

                        step += 1

            if input_ids.shape[1] == x_t.shape[1]:
                input_ids = x_t
            else:
                input_ids[:, :(block_idx + 1) * block_size] = x_t[:, :-1]
                if (seq_block_idx == block_idx).all():
                    input_ids = torch.cat([input_ids, x_t[:, -1:]], dim=1)
                else:
                    if input_ids.shape[1] <= (block_idx + 1) * block_size:
                        input_ids = x_t
                    else:
                        input_ids[
                            seq_block_idx == block_idx, (block_idx + 1) * block_size
                        ] = x_t[seq_block_idx == block_idx, (block_idx + 1) * block_size]
            seq_block_idx[seq_block_idx == block_idx] = block_idx + 1

            if finished_flag.any():
                for sample_idx in range(x_t.shape[0]):
                    if finished_flag[sample_idx]:
                        original_idx = sample_indices[sample_idx].item()
                        finished_samples[original_idx] = x_t[sample_idx:sample_idx + 1].clone().squeeze(dim=0)

                sample_indices = sample_indices[~finished_flag]
                input_ids = input_ids[~finished_flag]
                seq_block_idx = seq_block_idx[~finished_flag]
                seq_len = seq_len[~finished_flag]
                x_t = x_t[~finished_flag]

                for layer_id in range(len(past_key_values)):
                    past_key_values.key_cache[layer_id] = past_key_values.key_cache[layer_id][~finished_flag]
                    past_key_values.value_cache[layer_id] = past_key_values.value_cache[layer_id][~finished_flag]

                finished_flag = finished_flag[~finished_flag]

        if len(finished_samples) < batch_size:
            for sample_idx in range(x_t.shape[0]):
                original_idx = sample_indices[sample_idx].item()
                finished_samples[original_idx] = x_t[sample_idx:sample_idx + 1].clone().squeeze(dim=0)

        assert len(finished_samples) == batch_size
        return finished_samples
    
    @torch.no_grad()
    def mdm_sample_with_visualization(
        self,
        input_ids,
        tokenizer,
        block_size=32,
        max_new_tokens=1024, 
        mask_id=FAST_DLLM_MASK_ID,
        threshold=0.95,
        small_block_size=32,
        stop_token=FAST_DLLM_STOP_TOKEN,
        temperature=0.0,
        top_p=0.95,
    ):
        """
        MDM sampling function with visualization
        with intermediate state output for Gradio visualization
        """
        nfe = 0
        self.model.bd_size = block_size
        num_blocks = max_new_tokens // block_size

        # Initialize state - show all positions as mask
        initial_state = []

        if input_ids.shape[1] > block_size:
            output = self.forward(input_ids=input_ids[:, :(input_ids.shape[1] // block_size * block_size)], use_cache=True, update_past_key_values=True)
            logits, past_key_values = output.logits, output.past_key_values
            nfe += 1
            if input_ids.shape[1] % block_size == 0:
                next_token = logits[:, -1:, :].argmax(dim=-1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        else:
            past_key_values = None

        num_small_blocks = block_size // small_block_size
        original_input_length = input_ids.shape[1]

        for block_idx in range(num_blocks):
            if stop_token in input_ids[:, original_input_length:]:
                break
            prompt_length = input_ids.shape[1]

            # Use the length of the first block to initialize state
            first_block_length = block_size - (input_ids.shape[1] % block_size)

            if len(initial_state) == 0:
                for i in range(first_block_length):
                    initial_state.append(("[MASK]", MASK_COLOR))
                yield initial_state
            else:
                for i in range(first_block_length):
                    current_state.append(("[MASK]", MASK_COLOR))
                yield current_state


            # Initialize x_init as mask_id
            x_init = mask_id * torch.ones((input_ids.shape[0], block_size-prompt_length%block_size), device=self.device, dtype=torch.long)
            x_init = torch.cat([input_ids, x_init], dim=1)
                
            x_t = x_init.clone()
            block_past_key_values = None
            step = 0
            
            while True:
                if stop_token in x_t[:, prompt_length:]:
                    stop_token_idx = (x_t[:, prompt_length:] == stop_token).nonzero()[0][1]
                    if (x_t[:, prompt_length:prompt_length+stop_token_idx] == mask_id).sum() == 0:
                        break
                mask_idx = (x_t[:, -block_size:] == mask_id)
                # Decode a complete block, update cache, and generate next token
                if mask_idx.sum() == 0:
                    nfe += 1
                    output = self.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=True)
                    logits, past_key_values = output.logits, output.past_key_values
                    next_token = logits[:, -1:, :].argmax(dim=-1)
                    x_t = torch.cat([x_t, next_token], dim=1)
                    token_text = tokenizer.decode([next_token[0].item()], skip_special_tokens=True)
                    # Handle special characters
                    token_text = token_text
                    current_state.append((token_text, TOKEN_COLOR))
                    yield current_state
                    break
                    
                for small_block_idx in range(num_small_blocks):
                    small_block_start_idx = small_block_idx * small_block_size
                    small_block_end_idx = small_block_start_idx + small_block_size

                    start = -block_size + small_block_start_idx
                    end = None if block_size == small_block_end_idx else -block_size + small_block_end_idx
                    while True:
                        mask_idx = (x_t[:, -block_size:] == mask_id)
                        if mask_idx[:, start:end].sum() == 0:
                            break
                        if stop_token in x_t[:, prompt_length:]:
                            stop_token_idx = (x_t[:, prompt_length:] == stop_token).nonzero()[0][1]
                            if (x_t[:, prompt_length:prompt_length+stop_token_idx] == mask_id).sum() == 0:
                                break

                        logits = self.forward(input_ids=x_t[:, -block_size:], use_cache=True, past_key_values=past_key_values, update_past_key_values=False).logits
                        logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                        logits = logits[:, start:end]
                            
                        step += 1
                        x_1, p_1t = self.sample_with_top_p(logits, top_p=top_p, temperature=temperature)

                        # Select tokens with probability greater than threshold in p_1t
                        x1_p = torch.squeeze(torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1)
                        x1_p = torch.where(mask_idx[:, small_block_start_idx:small_block_end_idx], x1_p, -torch.inf)
                        unmask_idx = (x1_p > threshold)
                        max_prob_idx = x1_p.argmax(dim=-1)
                        unmask_idx[torch.arange(x_1.shape[0]), max_prob_idx] = True
                        unmask_idx = unmask_idx & mask_idx[:, start:end]

                        x_t[:, start:end][unmask_idx] = x_1[unmask_idx]

                        # Generate visualization state
                        current_state = []
                        generated_tokens = x_t[0, original_input_length:]
                        
                        # Display generated tokens
                        for i, token_id in enumerate(generated_tokens):
                            if token_id == mask_id:
                                current_state.append(("[MASK]", MASK_COLOR))
                            else:
                                token_text = tokenizer.decode([token_id.item()], skip_special_tokens=True)
                                # Handle special characters
                                token_text = token_text
                                current_state.append((token_text, TOKEN_COLOR))
                        
                        yield current_state

            input_ids = x_t
            
        # Truncate stop_token
        if stop_token in input_ids[:, original_input_length:]:
            stop_token_idx = (input_ids[:, original_input_length:] == stop_token).nonzero()[0][1]
            input_ids = input_ids[:, :stop_token_idx+original_input_length+1]
            
        # Final state - display complete text
        final_state = []
        generated_tokens = input_ids[0, original_input_length:]
        for token_id in generated_tokens:
            token_text = tokenizer.decode([token_id.item()], skip_special_tokens=True)
            token_text = token_text
            final_state.append((token_text, TOKEN_COLOR))
        
        # Final state doesn't need mask padding, only show actually generated tokens
        
        yield final_state
        
        # Return final text
        final_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        yield final_text


def setup_model_with_custom_generation(model):
    """
    Set up custom generation functions for the model
    """
    # Add mdm_sample method with visualization
    model.mdm_sample_with_visualization = types.MethodType(Fast_dLLM_QwenForCausalLM.mdm_sample_with_visualization, model)
    return model
