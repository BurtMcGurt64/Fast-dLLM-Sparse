import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model.configuration import Fast_dLLM_QwenConfig
from model.modeling import Fast_dLLM_QwenForCausalLM


# -----------------------------
# Defaults and helper utilities
# -----------------------------
TASK_DEFAULT_FEWSHOT = {
    "mmlu": 5,
    "gpqa_main_n_shot": 0,
    "gsm8k": 0,
    "minerva_math": 0,
    "ifeval": 0,
}


def _parse_model_args(model_args: Optional[str]) -> Dict[str, str]:
    if not model_args:
        return {}
    out: Dict[str, str] = {}
    for part in model_args.split(","):
        if not part:
            continue
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


@dataclass
class BlockTrace:
    traces: Dict[int, List[Optional[float]]]
    subblock_steps: Dict[int, List[int]]
    steps: int


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    return float(F.cosine_similarity(a, b, dim=-1).item())


def _merge_intervals(steps: List[int]) -> str:
    if not steps:
        return ""
    steps = sorted(steps)
    ranges: List[Tuple[int, int]] = []
    start = prev = steps[0]
    for s in steps[1:]:
        if s == prev + 1:
            prev = s
        else:
            ranges.append((start, prev))
            start = prev = s
    ranges.append((start, prev))
    return ";".join([f"[{a},{b}]" for a, b in ranges])


def _steps_to_ranges(steps: List[int]) -> List[Tuple[int, int]]:
    if not steps:
        return []
    steps = sorted(steps)
    ranges: List[Tuple[int, int]] = []
    start = prev = steps[0]
    for s in steps[1:]:
        if s == prev + 1:
            prev = s
        else:
            ranges.append((start, prev))
            start = prev = s
    ranges.append((start, prev))
    return ranges


class _PromptDumpHarness:
    dump_path: Optional[Path] = None


# -----------------------------
# Prompt dump path (mirrors eval.py prompt rendering)
# -----------------------------
def _dump_prompt_via_lm_eval(
    tasks: str,
    model_args: str,
    num_fewshot: Optional[int],
    limit: int,
    fewshot_as_multiturn: bool,
    apply_chat_template: bool,
    confirm_run_unsafe_code: bool,
    dump_path: Path,
) -> None:
    try:
        from lm_eval.__main__ import cli_evaluate
        from lm_eval.api.registry import register_model
    except Exception as exc:
        raise RuntimeError("lm_eval is required to dump prompts via eval.py logic") from exc

    from eval import Fast_dLLM_v2EvalHarness

    @register_model("fast_dllm_v2_trace")
    class Fast_dLLM_v2TraceHarness(Fast_dLLM_v2EvalHarness, _PromptDumpHarness):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._dump_path = dump_path

        def _write_prompt(self, text: str) -> None:
            Path(self._dump_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self._dump_path).write_text(text)

        def get_loglikelihood(self, prefix, target):
            # Fallback: if loglikelihood override is bypassed, dump a decoded prompt here.
            try:
                text = self.tokenizer.decode(prefix, skip_special_tokens=True)
            except Exception:
                text = str(prefix)
            self._write_prompt(text)
            raise SystemExit(0)

        def loglikelihood(self, requests):
            # For multiple-choice tasks (e.g., MMLU), lm-eval uses loglikelihood.
            # Dump the first context prompt and exit.
            for req in requests:
                question = req.args[0]
                if req.task_name.startswith("minerva_math"):
                    question = question.replace(
                        "Solution:",
                        "Please reason step by step, and put your final answer within \\boxed{{}}.",
                    )
                elif req.task_name.startswith("gsm8k"):
                    question = question.replace(
                        "Answer:",
                        "Please reason step by step, and put your final answer within \\boxed{{}}.",
                    )
                self._write_prompt(question)
                break
            raise SystemExit(0)

        def generate_until(self, requests):
            for req in requests:
                question = req.args[0]
                if req.task_name.startswith("minerva_math"):
                    question = question.replace(
                        "Solution:",
                        "Please reason step by step, and put your final answer within \\boxed{{}}.",
                    )
                elif req.task_name.startswith("gsm8k"):
                    question = question.replace(
                        "Answer:",
                        "Please reason step by step, and put your final answer within \\boxed{{}}.",
                    )
                self._write_prompt(question)
                break
            raise SystemExit(0)

    Fast_dLLM_v2TraceHarness.dump_path = dump_path

    argv = [
        "trace_decode_task.py",
        "--tasks",
        tasks,
        "--batch_size",
        "1",
        "--limit",
        str(limit),
        "--model",
        "fast_dllm_v2_trace",
        "--model_args",
        model_args,
    ]
    if num_fewshot is not None:
        argv.extend(["--num_fewshot", str(num_fewshot)])
    if confirm_run_unsafe_code:
        argv.append("--confirm_run_unsafe_code")
    if fewshot_as_multiturn:
        argv.append("--fewshot_as_multiturn")
    if apply_chat_template:
        argv.append("--apply_chat_template")

    old_argv = sys.argv
    sys.argv = argv
    try:
        cli_evaluate()
    except SystemExit:
        pass
    finally:
        sys.argv = old_argv


def trace_single_request(
    prompt: str,
    model_path: str,
    output_csv: str,
    output_summary: str,
    block_size: int = 32,
    small_block_size: int = 8,
    max_new_tokens: int = 256,
    threshold: float = 0.9,
    use_block_cache: bool = False,
    temperature: float = 0.0,
    top_p: float = 0.95,
    mask_id: int = 151665,
    device: str = "cuda:0",
    seed: int = 1234,
    plot_blocks: Optional[List[int]] = None,
):
    # -----------------------------
    # Model/tokenizer setup
    # -----------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = Fast_dLLM_QwenConfig.from_pretrained(model_path)
    model = Fast_dLLM_QwenForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype="auto",
        device_map=device,
    )
    model.eval()

    input_ids = tokenizer([prompt], return_tensors="pt")["input_ids"].to(model.device)
    prompt_tokens = input_ids.shape[1]
    seq_len = torch.tensor([prompt_tokens], device=model.device)
    min_len = int(seq_len.min().item())

    num_blocks = max_new_tokens // block_size + int(seq_len.max().item()) // block_size
    num_layers = config.num_hidden_layers
    layer_trace_enabled = False
    layer_inputs: List[Optional[torch.Tensor]] = [None] * num_layers
    layer_outputs: List[Optional[torch.Tensor]] = [None] * num_layers
    layer_k_outputs: List[Optional[torch.Tensor]] = [None] * num_layers
    layer_v_outputs: List[Optional[torch.Tensor]] = [None] * num_layers
    # Per-block, per-layer step traces.
    layer_step_sims_by_block: Dict[int, Dict[int, List[Tuple[float, float, float]]]] = {}
    layer_step_delta_by_block: Dict[int, Dict[int, List[Tuple[float, float, float]]]] = {}

    # -----------------------------
    # Layer trace collection helpers
    # -----------------------------
    def _reset_layer_buffers() -> None:
        for i in range(num_layers):
            layer_inputs[i] = None
            layer_outputs[i] = None
            layer_k_outputs[i] = None
            layer_v_outputs[i] = None

    def _collect_layer_cos() -> List[Tuple[float, float, float]]:
        sims: List[Tuple[float, float, float]] = []
        for i in range(num_layers):
            inp = layer_inputs[i]
            out = layer_outputs[i]
            if inp is None or out is None:
                sims.append((float("nan"), float("nan"), float("nan")))
                continue
            # Min + mean cosine similarity across token positions for this forward step.
            per_token = F.cosine_similarity(inp.float(), out.float(), dim=-1)
            mean_cos = per_token.mean().item()
            min_cos = per_token.min().item()
            # Relative gap between min and mean: (B - A) / B.
            if mean_cos == 0:
                gap = float("nan")
            else:
                gap = (mean_cos - min_cos) / mean_cos
            sims.append((min_cos, mean_cos, gap))
        return sims

    def _collect_layer_delta() -> List[Tuple[float, float, float]]:
        deltas: List[Tuple[float, float, float]] = []
        for i in range(num_layers):
            inp = layer_inputs[i]
            out = layer_outputs[i]
            if inp is None or out is None:
                deltas.append((float("nan"), float("nan"), float("nan")))
                continue
            # Relative residual norm per token: ||out - inp|| / ||inp||.
            inp_f = inp.float()
            out_f = out.float()
            denom = inp_f.norm(dim=-1)
            delta = (out_f - inp_f).norm(dim=-1)
            rel = delta / denom.clamp_min(1e-8)
            mean_rel = rel.mean().item()
            p95_rel = torch.quantile(rel.flatten(), 0.95).item()
            if mean_rel == 0:
                gap = float("nan")
            else:
                gap = (p95_rel - mean_rel) / mean_rel
            deltas.append((p95_rel, mean_rel, gap))
        return deltas

    def _make_layer_hook(layer_idx: int):
        def _hook(_module, inputs, output):
            nonlocal layer_trace_enabled
            if not layer_trace_enabled:
                return
            if not inputs:
                return
            layer_inputs[layer_idx] = inputs[0].detach()
            layer_outputs[layer_idx] = output.detach()
        return _hook

    layer_hooks = [
        layer.register_forward_hook(_make_layer_hook(layer_idx))
        for layer_idx, layer in enumerate(model.model.layers)
    ]

    def _make_kv_hook(layer_idx: int, is_key: bool):
        def _hook(_module, inputs, output):
            if not layer_trace_enabled:
                return
            if is_key:
                layer_k_outputs[layer_idx] = output.detach()
            else:
                layer_v_outputs[layer_idx] = output.detach()
        return _hook

    kv_hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        kv_hooks.append(layer.self_attn.k_proj.register_forward_hook(_make_kv_hook(layer_idx, True)))
        kv_hooks.append(layer.self_attn.v_proj.register_forward_hook(_make_kv_hook(layer_idx, False)))

    # -----------------------------
    # Prefix-cache prefill (before decode blocks)
    # -----------------------------
    if min_len > block_size:
        output = model(
            input_ids=input_ids[:, :(min_len // block_size * block_size)],
            use_cache=True,
            update_past_key_values=True,
            block_size=block_size,
            output_hidden_states=True,
        )
        logits, past_key_values = output.logits, output.past_key_values
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
    start_block_idx = min_len // block_size
    num_small_blocks = block_size // small_block_size
    num_decode_blocks = max(1, num_blocks - start_block_idx)
    target_block_idx = int(start_block_idx + num_decode_blocks // 2)
    rng = random.Random(seed)
    first_third_start = 0
    first_third_end = max(0, block_size // 3)
    last_third_start = max(0, (2 * block_size) // 3)
    last_third_end = block_size
    mid_start = block_size // 3
    mid_end = max(mid_start + 1, (2 * block_size) // 3)

    candidate_positions = list(range(mid_start, mid_end)) if block_size >= 3 else list(range(block_size))
    mid_pos = rng.choice(candidate_positions) if candidate_positions else 0

    start_pos = min(block_size - 1, first_third_start + 4)
    end_pos = max(0, last_third_end - 5)
    target_positions: List[int] = []
    for pos in (start_pos, mid_pos, end_pos):
        if pos not in target_positions:
            target_positions.append(pos)
    if block_size >= 3:
        for pos in range(block_size):
            if len(target_positions) >= 3:
                break
            if pos not in target_positions:
                target_positions.append(pos)
    target_abs_positions = [target_block_idx * block_size + pos for pos in target_positions]

    finished_flag = torch.zeros((1,), device=model.device, dtype=torch.bool)
    sample_indices = torch.arange(1, device=model.device)
    finished_samples: Dict[int, torch.Tensor] = {}

    traces_by_block: Dict[int, BlockTrace] = {}
    target_hidden_steps: Dict[int, List[torch.Tensor]] = {pos: [] for pos in target_positions}
    target_k_steps: Dict[int, List[torch.Tensor]] = {pos: [] for pos in target_positions}
    target_v_steps: Dict[int, List[torch.Tensor]] = {pos: [] for pos in target_positions}
    target_unmask_steps: Dict[int, Optional[int]] = {pos: None for pos in target_positions}

    # -----------------------------
    # Main decoding loop (block -> subblock -> step)
    # -----------------------------
    for block_idx in range(start_block_idx, num_blocks):
        if finished_flag.all():
            break

        block_layer_step_sims: Dict[int, List[Tuple[float, float, float]]] = {i: [] for i in range(num_layers)}
        block_layer_step_delta: Dict[int, List[Tuple[float, float, float]]] = {i: [] for i in range(num_layers)}

        if (seq_block_idx == block_idx).all():
            pad_len = block_size - input_ids.shape[1] % block_size
            x_init = torch.full(
                (input_ids.shape[0], pad_len),
                fill_value=mask_id,
                device=model.device,
                dtype=torch.long,
            )
            x_init = torch.cat([input_ids, x_init], dim=1)
            input_ids = x_init
        else:
            x_init = input_ids[:, :(block_idx + 1) * block_size]

        x_init[finished_flag, -block_size:] = tokenizer.pad_token_id
        x_t = x_init.clone()

        traces: Dict[int, List[Optional[float]]] = {i: [] for i in range(block_size)}
        subblock_steps: Dict[int, List[int]] = {i: [] for i in range(block_size)}
        prev_hidden: List[Optional[torch.Tensor]] = [None] * block_size
        step_idx = 0
        block_past_key_values = None
        target_was_masked: Dict[int, bool] = {}
        if block_idx == target_block_idx:
            for pos in target_positions:
                target_was_masked[pos] = (x_t[0, -block_size + pos].item() == mask_id)

        # -----------------------------
        # Per-step trace recorders for this block
        # -----------------------------
        def record_step(
            hidden_block: torch.Tensor,
            full_block: bool,
            sub_start: Optional[int] = None,
            sub_end: Optional[int] = None,
            layer_cos: Optional[List[Tuple[float, float, float]]] = None,
            layer_delta: Optional[List[Tuple[float, float, float]]] = None,
        ) -> None:
            nonlocal step_idx, prev_hidden
            step_idx += 1
            if full_block:
                for pos in range(block_size):
                    cur = hidden_block[0, pos]
                    if prev_hidden[pos] is None:
                        sim = float("nan")
                    else:
                        sim = _cosine(prev_hidden[pos], cur)
                    traces[pos].append(sim)
                    prev_hidden[pos] = cur.detach()
                if block_idx == target_block_idx:
                    for pos in target_positions:
                        target_hidden_steps[pos].append(hidden_block[0, pos].detach())
                        k_vecs = [layer_k_outputs[li][0, pos] for li in range(num_layers)]
                        v_vecs = [layer_v_outputs[li][0, pos] for li in range(num_layers)]
                        target_k_steps[pos].append(torch.cat(k_vecs, dim=-1))
                        target_v_steps[pos].append(torch.cat(v_vecs, dim=-1))
            else:
                # Sub-block forward: only update positions inside the sub-block window.
                for pos in range(block_size):
                    traces[pos].append(None)
                assert sub_start is not None and sub_end is not None
                for pos in range(sub_start, sub_end):
                    cur = hidden_block[0, pos - sub_start]
                    if prev_hidden[pos] is None:
                        sim = float("nan")
                    else:
                        sim = _cosine(prev_hidden[pos], cur)
                    traces[pos][-1] = sim
                    prev_hidden[pos] = cur.detach()

            if sub_start is not None and sub_end is not None:
                for pos in range(sub_start, sub_end):
                    subblock_steps[pos].append(step_idx)
            if layer_cos is not None:
                for layer_idx, cos_val in enumerate(layer_cos):
                    block_layer_step_sims[layer_idx].append(cos_val)
                if layer_delta is not None:
                    for layer_idx, delta_val in enumerate(layer_delta):
                        block_layer_step_delta[layer_idx].append(delta_val)

        def maybe_mark_unmask() -> None:
            if block_idx != target_block_idx:
                return
            for pos in target_positions:
                if target_unmask_steps[pos] is None and target_was_masked.get(pos, False):
                    if x_t[0, -block_size + pos].item() != mask_id:
                        target_unmask_steps[pos] = step_idx

        while True:
            mask_idx = (x_t[:, -block_size:] == mask_id)
            if mask_idx.sum() == 0:
                if finished_flag.all():
                    break
                layer_cos = None
                layer_delta = None
                layer_trace_enabled = True
                _reset_layer_buffers()
                output = model(
                    input_ids=x_t[:, -block_size:],
                    use_cache=True,
                    past_key_values=past_key_values,
                    update_past_key_values=True,
                    block_size=block_size,
                    output_hidden_states=True,
                )
                layer_cos = _collect_layer_cos()
                layer_delta = _collect_layer_delta()
                layer_trace_enabled = False
                logits, past_key_values = output.logits, output.past_key_values
                hidden_block = output.hidden_states[-1]
                if hidden_block.dim() == 2:
                    hidden_block = hidden_block.unsqueeze(0)
                hidden_block = hidden_block[:, -block_size:, :]
                record_step(
                    hidden_block=hidden_block,
                    full_block=True,
                    layer_cos=layer_cos,
                    layer_delta=layer_delta,
                )

                next_token = logits[:, -1:, :].argmax(dim=-1)
                next_token[finished_flag] = tokenizer.pad_token_id
                x_t = torch.cat([x_t, next_token], dim=1)
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
                        if block_past_key_values is None or (x_t[:, -block_size + small_block_start_idx] == mask_id).any():
                            layer_cos = None
                            layer_delta = None
                            layer_trace_enabled = True
                            _reset_layer_buffers()
                            output = model(
                                input_ids=x_t[:, -block_size:],
                                use_cache=True,
                                past_key_values=past_key_values,
                                update_past_key_values=False,
                                use_block_cache=True,
                                output_hidden_states=True,
                            )
                            layer_cos = _collect_layer_cos()
                            layer_delta = _collect_layer_delta()
                            layer_trace_enabled = False
                            logits = output.logits
                            block_past_key_values = output.block_past_key_values
                            hidden_block = output.hidden_states[-1]
                            if hidden_block.dim() == 2:
                                hidden_block = hidden_block.unsqueeze(0)
                            hidden_block = hidden_block[:, -block_size:, :]
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                            logits = logits[:, start:end]
                            record_step(
                                hidden_block=hidden_block,
                                full_block=True,
                                sub_start=small_block_start_idx,
                                sub_end=small_block_end_idx,
                                layer_cos=layer_cos,
                                layer_delta=layer_delta,
                            )
                        else:
                            layer_cos = None
                            layer_delta = None
                            layer_trace_enabled = True
                            _reset_layer_buffers()
                            output = model(
                                input_ids=x_t[:, start:end],
                                use_cache=True,
                                past_key_values=past_key_values,
                                update_past_key_values=False,
                                use_block_cache=True,
                                block_past_key_values=block_past_key_values,
                                replace_position=small_block_start_idx,
                                output_hidden_states=True,
                            )
                            layer_cos = _collect_layer_cos()
                            layer_delta = _collect_layer_delta()
                            layer_trace_enabled = False
                            logits = output.logits
                            hidden_block = output.hidden_states[-1]
                            if hidden_block.dim() == 2:
                                hidden_block = hidden_block.unsqueeze(0)
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                            record_step(
                                hidden_block=hidden_block,
                                full_block=False,
                                sub_start=small_block_start_idx,
                                sub_end=small_block_end_idx,
                                layer_cos=layer_cos,
                                layer_delta=layer_delta,
                            )
                    else:
                        layer_cos = None
                        layer_delta = None
                        layer_trace_enabled = True
                        _reset_layer_buffers()
                        output = model(
                            input_ids=x_t[:, -block_size:],
                            use_cache=True,
                            past_key_values=past_key_values,
                            update_past_key_values=False,
                            output_hidden_states=True,
                        )
                        layer_cos = _collect_layer_cos()
                        layer_delta = _collect_layer_delta()
                        layer_trace_enabled = False
                        logits = output.logits
                        hidden_block = output.hidden_states[-1]
                        if hidden_block.dim() == 2:
                            hidden_block = hidden_block.unsqueeze(0)
                        hidden_block = hidden_block[:, -block_size:, :]
                        logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                        logits = logits[:, start:end]
                        record_step(
                            hidden_block=hidden_block,
                            full_block=True,
                            sub_start=small_block_start_idx,
                            sub_end=small_block_end_idx,
                            layer_cos=layer_cos,
                            layer_delta=layer_delta,
                        )

                    x_1, p_1t = model.sample_with_top_p(logits, top_p=top_p, temperature=temperature)
                    x1_p = torch.squeeze(
                        torch.gather(p_1t, dim=-1, index=torch.unsqueeze(x_1, -1)), -1
                    )
                    x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)
                    unmask_idx = (x1_p > threshold)
                    max_prob_idx = x1_p.argmax(dim=-1)
                    unmask_idx[torch.arange(x_1.shape[0]), max_prob_idx] = True
                    unmask_idx = unmask_idx & mask_idx[:, start:end]

                    x_t[:, start:end][unmask_idx] = x_1[unmask_idx]
                    finished_row_flags = ((x_1 == model.config.eos_token_id) & unmask_idx).any(dim=1)
                    finished_flag = finished_flag | finished_row_flags
                    maybe_mark_unmask()

        # -----------------------------
        # End-of-block state commit
        # -----------------------------
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

        traces_by_block[block_idx] = BlockTrace(
            traces=traces,
            subblock_steps=subblock_steps,
            steps=step_idx,
        )
        layer_step_sims_by_block[block_idx] = block_layer_step_sims
        layer_step_delta_by_block[block_idx] = block_layer_step_delta

    # -----------------------------
    # Final sample fallback
    # -----------------------------
    if len(finished_samples) < 1:
        finished_samples[0] = x_t[0:1].clone().squeeze(dim=0)

    # -----------------------------
    # Token-level trace CSV export
    # -----------------------------
    max_steps = max(bt.steps for bt in traces_by_block.values()) if traces_by_block else 0
    step_columns = [f"step_{i}" for i in range(1, max_steps + 1)]
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["block_idx", "pos_in_block", "abs_pos", *step_columns, "subblock_intervals"])
        for block_idx, bt in traces_by_block.items():
            for pos in range(block_size):
                abs_pos = block_idx * block_size + pos
                row = [block_idx, pos, abs_pos]
                vals = bt.traces[pos] + [None] * (max_steps - len(bt.traces[pos]))
                row.extend(vals)
                row.append(_merge_intervals(bt.subblock_steps[pos]))
                writer.writerow(row)

    # -----------------------------
    # Human-readable summary export
    # -----------------------------
    with open(output_summary, "w") as f:
        f.write(f"start_block_idx={start_block_idx}\n")
        f.write(f"prompt_tokens={prompt_tokens}\n")
        f.write(f"prompt_blocks={int(seq_len.max().item()) // block_size}\n")
        for block_idx, bt in traces_by_block.items():
            f.write(f"block_{block_idx}_steps={bt.steps}\n")
        f.write(f"heatmap_block_idx={target_block_idx}\n")
        for i, pos in enumerate(target_positions):
            f.write(f"heatmap_pos_in_block_{i}={pos}\n")
            f.write(f"heatmap_abs_pos_{i}={target_abs_positions[i]}\n")
            if target_block_idx in traces_by_block:
                intervals = _merge_intervals(traces_by_block[target_block_idx].subblock_steps[pos])
                f.write(f"heatmap_subblock_intervals_{i}={intervals}\n")
            else:
                f.write(f"heatmap_subblock_intervals_{i}=\n")
            unmask_step = target_unmask_steps.get(pos)
            f.write(f"heatmap_unmask_step_{i}={'' if unmask_step is None else unmask_step}\n")

    # -----------------------------
    # Optional token heatmaps (no block cache path)
    # -----------------------------
    if (not use_block_cache) and target_hidden_steps:
        try:
            import matplotlib.pyplot as plt
            from matplotlib import patches
        except Exception as exc:
            raise RuntimeError("matplotlib is required to generate the heatmap") from exc

        for pos, vec_list in target_hidden_steps.items():
            if not vec_list:
                continue
            vecs = torch.stack(vec_list).float()
            vecs = F.normalize(vecs, dim=1)
            sim_mat = (vecs @ vecs.T).cpu().numpy()

            fig, ax = plt.subplots(figsize=(6, 5))
            vmin = float(sim_mat.min())
            vmax = float(sim_mat.max())
            if vmin == vmax:
                vmin = vmin - 1e-6
                vmax = vmax + 1e-6
            im = ax.imshow(sim_mat, cmap="YlGnBu_r", vmin=vmin, vmax=vmax, origin="lower")
            ax.set_xlabel("step")
            ax.set_ylabel("step")
            ax.set_title(f"Block {target_block_idx} pos {pos}")
            fig.colorbar(im, ax=ax)
            if target_block_idx in traces_by_block:
                intervals = _steps_to_ranges(traces_by_block[target_block_idx].subblock_steps[pos])
                for a, b in intervals:
                    # Steps are 1-based; heatmap indices are 0-based.
                    rect = patches.Rectangle(
                        (a - 1, a - 1),
                        (b - a + 1),
                        (b - a + 1),
                        fill=False,
                        edgecolor="red",
                        linewidth=1.0,
                    )
                    ax.add_patch(rect)
            unmask_step = target_unmask_steps.get(pos)
            if unmask_step is not None:
                # Steps are 1-based; plot on 0-based heatmap indices.
                ax.scatter(unmask_step - 1, unmask_step - 1, c="red", s=18, zorder=5)
            heatmap_path = Path(output_csv).with_suffix("")
            heatmap_path = heatmap_path.parent / f"{heatmap_path.name}_heatmap_pos{pos}.png"
            fig.tight_layout()
            fig.savefig(heatmap_path, dpi=200)
            plt.close(fig)

        for kv_label, kv_steps_dict in [("K", target_k_steps), ("V", target_v_steps)]:
            for pos, vec_list in kv_steps_dict.items():
                if not vec_list:
                    continue
                vecs = torch.stack(vec_list).float()
                vecs = F.normalize(vecs, dim=1)
                sim_mat = (vecs @ vecs.T).cpu().numpy()

                fig, ax = plt.subplots(figsize=(6, 5))
                vmin = float(sim_mat.min())
                vmax = float(sim_mat.max())
                if vmin == vmax:
                    vmin = vmin - 1e-6
                    vmax = vmax + 1e-6
                im = ax.imshow(sim_mat, cmap="YlGnBu_r", vmin=vmin, vmax=vmax, origin="lower")
                ax.set_xlabel("step")
                ax.set_ylabel("step")
                ax.set_title(f"Block {target_block_idx} pos {pos} ({kv_label})")
                fig.colorbar(im, ax=ax)
                if target_block_idx in traces_by_block:
                    intervals = _steps_to_ranges(traces_by_block[target_block_idx].subblock_steps[pos])
                    for a, b in intervals:
                        rect = patches.Rectangle(
                            (a - 1, a - 1),
                            (b - a + 1),
                            (b - a + 1),
                            fill=False,
                            edgecolor="red",
                            linewidth=1.0,
                        )
                        ax.add_patch(rect)
                unmask_step = target_unmask_steps.get(pos)
                if unmask_step is not None:
                    ax.scatter(unmask_step - 1, unmask_step - 1, c="red", s=18, zorder=5)
                heatmap_path = Path(output_csv).with_suffix("")
                heatmap_path = heatmap_path.parent / f"{heatmap_path.name}_heatmap_{kv_label}_pos{pos}.png"
                fig.tight_layout()
                fig.savefig(heatmap_path, dpi=200)
                plt.close(fig)

    # -----------------------------
    # Layer-level CSV + plotting
    # -----------------------------
    # Per-layer traces for all decoded blocks; plotting below still uses selected blocks.
    if layer_step_sims_by_block:
        base_path = Path(output_csv).with_suffix("")
        base_name = base_path.name
        layer_csv = base_path.parent / f"{base_name}_layer_cos.csv"
        delta_csv = base_path.parent / f"{base_name}_layer_delta.csv"
        layer_steps = 0
        for block_layer_map in layer_step_sims_by_block.values():
            for vals in block_layer_map.values():
                layer_steps = max(layer_steps, len(vals))
        layer_step_cols = [f"step_{i}" for i in range(1, layer_steps + 1)]
        with open(layer_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["block_idx", "layer_idx", *layer_step_cols])
            for block_idx in sorted(layer_step_sims_by_block.keys()):
                for layer_idx in range(num_layers):
                    vals = layer_step_sims_by_block[block_idx][layer_idx]
                    vals = vals + [None] * (layer_steps - len(vals))
                    # Store each step as a tuple string "(min,mean,pct)" so it stays in one cell.
                    formatted = []
                    for v in vals:
                        if v is None:
                            formatted.append(None)
                        else:
                            formatted.append(f"({v[0]:.6g},{v[1]:.6g},{v[2]:.6g})")
                    writer.writerow([block_idx, layer_idx, *formatted])
        with open(delta_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["block_idx", "layer_idx", *layer_step_cols])
            for block_idx in sorted(layer_step_delta_by_block.keys()):
                for layer_idx in range(num_layers):
                    vals = layer_step_delta_by_block[block_idx][layer_idx]
                    vals = vals + [None] * (layer_steps - len(vals))
                    formatted = []
                    for v in vals:
                        if v is None:
                            formatted.append(None)
                        else:
                            formatted.append(f"({v[0]:.6g},{v[1]:.6g},{v[2]:.6g})")
                    writer.writerow([block_idx, layer_idx, *formatted])

        def _parse_tuple_cell(cell: Optional[str]) -> Optional[Tuple[float, float, float]]:
            if cell is None:
                return None
            cell = str(cell).strip()
            if not cell:
                return None
            if not (cell.startswith("(") and cell.endswith(")")):
                return None
            parts = [p.strip() for p in cell[1:-1].split(",")]
            if len(parts) != 3:
                return None
            return (float(parts[0]), float(parts[1]), float(parts[2]))

        def _load_block_series(csv_path: Path, block_idx: int) -> Dict[int, List[Tuple[float, float, float]]]:
            per_layer: Dict[int, List[Tuple[float, float, float]]] = {i: [] for i in range(num_layers)}
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if int(row["block_idx"]) != block_idx:
                        continue
                    layer_idx = int(row["layer_idx"])
                    vals: List[Tuple[float, float, float]] = []
                    for key, raw in row.items():
                        if key in ("block_idx", "layer_idx"):
                            continue
                        parsed = _parse_tuple_cell(raw)
                        if parsed is not None:
                            vals.append(parsed)
                    per_layer[layer_idx] = vals
            return per_layer

        # Plot block selection: default to the middle decode block (original behavior).
        if plot_blocks is None:
            selected_plot_blocks = [target_block_idx]
        else:
            selected_plot_blocks = [b for b in plot_blocks if b in layer_step_sims_by_block]
            if not selected_plot_blocks:
                selected_plot_blocks = [target_block_idx]
        selected_plot_blocks = sorted(selected_plot_blocks)

        layer_step_sims_selected = {
            b: _load_block_series(layer_csv, b) for b in selected_plot_blocks
        }
        layer_step_delta_selected = {
            b: _load_block_series(delta_csv, b) for b in selected_plot_blocks
        }

        # Pick layers: first, last, middle, 1/4, 3/4.
        middle_layer = num_layers // 2
        quarter_layer = num_layers // 4
        three_quarter_layer = (3 * num_layers) // 4
        layer_choices = []
        for idx in (0, quarter_layer, middle_layer, three_quarter_layer, num_layers - 1):
            if idx not in layer_choices:
                layer_choices.append(idx)
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            raise RuntimeError("matplotlib is required to plot layer similarity curves") from exc
        block_tag = "_".join(str(b) for b in selected_plot_blocks)
        for layer_idx in layer_choices:
            # Concatenate selected blocks on one timeline.
            series: List[Tuple[float, float, float]] = []
            boundaries: List[int] = []
            running = 0
            for b in selected_plot_blocks:
                part = layer_step_sims_selected[b][layer_idx]
                if not part:
                    continue
                series.extend(part)
                running += len(part)
                boundaries.append(running)
            if not series:
                continue
            # Plot A(min) and B(mean); indicate C(gap) as the shaded area between them.
            min_series = [val[0] for val in series]
            mean_series = [val[1] for val in series]
            steps = list(range(1, len(mean_series) + 1))
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(steps, min_series, marker="o", linewidth=1.0, label="min")
            ax.plot(steps, mean_series, marker="o", linewidth=1.2, label="mean")
            ax.fill_between(steps, min_series, mean_series, alpha=0.2, label="gap (mean-min)")
            ax.set_xlabel("step")
            ax.set_ylabel("value")
            ax.set_title(f"Blocks {selected_plot_blocks} layer {layer_idx}")
            for bnd in boundaries[:-1]:
                ax.axvline(bnd + 0.5, color="gray", linestyle="--", linewidth=0.8)
            ax.legend(loc="best")
            fig.tight_layout()
            plot_path = layer_csv.with_suffix("")
            plot_path = plot_path.parent / f"{base_name}_cos_layer{layer_idx}_blocks{block_tag}.png"
            fig.savefig(plot_path, dpi=200)
            plt.close(fig)
        for layer_idx in layer_choices:
            # Concatenate selected blocks on one timeline.
            series: List[Tuple[float, float, float]] = []
            boundaries: List[int] = []
            running = 0
            for b in selected_plot_blocks:
                part = layer_step_delta_selected[b][layer_idx]
                if not part:
                    continue
                series.extend(part)
                running += len(part)
                boundaries.append(running)
            if not series:
                continue
            # Plot A(p95) and B(mean); indicate C(gap) as the shaded area between them.
            p95_series = [val[0] for val in series]
            mean_series = [val[1] for val in series]
            steps = list(range(1, len(mean_series) + 1))
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(steps, p95_series, marker="o", linewidth=1.0, label="p95")
            ax.plot(steps, mean_series, marker="o", linewidth=1.2, label="mean")
            ax.fill_between(steps, p95_series, mean_series, alpha=0.2, label="gap (mean-p95)")
            ax.set_xlabel("step")
            ax.set_ylabel("value")
            ax.set_title(f"Blocks {selected_plot_blocks} layer {layer_idx} (delta)")
            for bnd in boundaries[:-1]:
                ax.axvline(bnd + 0.5, color="gray", linestyle="--", linewidth=0.8)
            ax.legend(loc="best")
            fig.tight_layout()
            plot_path = delta_csv.with_suffix("")
            plot_path = plot_path.parent / f"{base_name}_delta_layer{layer_idx}_blocks{block_tag}.png"
            fig.savefig(plot_path, dpi=200)
            plt.close(fig)

    # -----------------------------
    # Hook cleanup
    # -----------------------------
    for handle in layer_hooks:
        handle.remove()
    for handle in kv_hooks:
        handle.remove()


def main() -> None:
    # -----------------------------
    # CLI parsing
    # -----------------------------
    parser = argparse.ArgumentParser(description="Trace per-token cosine similarity for one task prompt.")
    parser.add_argument("--tasks", type=str, required=True, help="Task name (single task, eval.py-style)")
    parser.add_argument("--model_args", type=str, default="", help="Model args string, eval.py-style")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_fewshot", type=int, default=None)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--confirm_run_unsafe_code", action="store_true")
    parser.add_argument("--fewshot_as_multiturn", action="store_true")
    parser.add_argument("--apply_chat_template", action="store_true")
    parser.add_argument("--output_dir", type=str, default="traces", help="Output directory")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--small_block_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--use_block_cache", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--mask_id", type=int, default=151665)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--plot_blocks",
        type=str,
        default="",
        help="Comma-separated decode block indices to plot together (e.g., 7,8,9).",
    )
    args = parser.parse_args()

    task = args.tasks
    model_args = _parse_model_args(args.model_args)
    model_path = model_args.get("model_path", None)
    if model_path is None:
        raise ValueError("model_path is required in --model_args (e.g., model_path=... )")

    if "threshold" in model_args:
        args.threshold = float(model_args["threshold"])
    if "max_new_tokens" in model_args:
        args.max_new_tokens = int(model_args["max_new_tokens"])
    if "bd_size" in model_args:
        args.block_size = int(model_args["bd_size"])
    if "small_block_size" in model_args:
        args.small_block_size = int(model_args["small_block_size"])
    if "mask_id" in model_args:
        args.mask_id = int(model_args["mask_id"])
    if "use_block_cache" in model_args:
        args.use_block_cache = model_args["use_block_cache"].lower() in {"1", "true", "yes"}
    plot_blocks: Optional[List[int]] = None
    if args.plot_blocks.strip():
        plot_blocks = [int(x.strip()) for x in args.plot_blocks.split(",") if x.strip()]

    # -----------------------------
    # Output paths and prompt acquisition
    # -----------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"{task}.csv"
    output_summary = output_dir / f"{task}.summary.txt"
    prompt_path = output_dir / f"{task}.prompt.txt"

    num_fewshot = args.num_fewshot
    if num_fewshot is None:
        num_fewshot = TASK_DEFAULT_FEWSHOT.get(task, 0)

    _dump_prompt_via_lm_eval(
        tasks=task,
        model_args=args.model_args,
        num_fewshot=num_fewshot,
        limit=args.limit,
        fewshot_as_multiturn=args.fewshot_as_multiturn,
        apply_chat_template=args.apply_chat_template,
        confirm_run_unsafe_code=args.confirm_run_unsafe_code,
        dump_path=prompt_path,
    )
    prompt = prompt_path.read_text()

    # -----------------------------
    # Trace execution
    # -----------------------------
    trace_single_request(
        prompt=prompt,
        model_path=model_path,
        output_csv=str(output_csv),
        output_summary=str(output_summary),
        block_size=args.block_size,
        small_block_size=args.small_block_size,
        max_new_tokens=args.max_new_tokens,
        threshold=args.threshold,
        use_block_cache=args.use_block_cache,
        temperature=args.temperature,
        top_p=args.top_p,
        mask_id=args.mask_id,
        device=args.device,
        seed=args.seed,
        plot_blocks=plot_blocks,
    )


if __name__ == "__main__":
    main()
