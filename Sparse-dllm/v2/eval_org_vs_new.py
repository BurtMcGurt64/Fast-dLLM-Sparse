import argparse
import accelerate
import gc
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval import evaluator
from tqdm import tqdm
from transformers import AutoTokenizer

from model.modeling import Fast_dLLM_QwenForCausalLM
from model.configuration import Fast_dLLM_QwenConfig
import generation_functions
import types


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class _BaseFastDLLMEvalHarness(LM):
    """
    Shared eval harness used by both original and token-policy sampling.
    Adds decode-log based FLOPs estimation around mdm_sample calls.
    """

    last_instance = None

    def __init__(
        self,
        model_path="Efficient-Large-Model/Fast_dLLM_v2_7B",
        device="cuda",
        show_speed=False,
        max_new_tokens=2048,
        batch_size=1,
        mask_id=151665,
        use_block_cache=False,
        small_block_size=8,
        bd_size=32,
        threshold=0.9,
        token_policy_threshold=0.99,
        token_policy_freeze_step=3,
        layer_policy_threshold=0.99,
        layer_policy_freeze_step=3,
        profile_flops=False,
        **kwargs,
    ):
        super().__init__()

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None

        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({"device_map": {"": f"{self.accelerator.device}"}})

        config = Fast_dLLM_QwenConfig.from_pretrained(model_path)
        self.model = Fast_dLLM_QwenForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            **model_kwargs,
        )
        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f"{self.accelerator.device}")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model = self.model.to(device)
            self._rank = 0
            self._world_size = 1

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.show_speed = show_speed
        self.max_new_tokens = max_new_tokens
        self.batch_size = int(batch_size)
        self.mask_id = mask_id
        self.model_path = model_path
        self.use_block_cache = use_block_cache
        self.small_block_size = small_block_size
        self.threshold = threshold
        self.bd_size = bd_size
        self.token_policy_threshold = token_policy_threshold
        self.token_policy_freeze_step = int(token_policy_freeze_step)
        self.layer_policy_threshold = layer_policy_threshold
        self.layer_policy_freeze_step = int(layer_policy_freeze_step)

        # FLOP accounting (accumulated across generate_until calls).
        self.total_flops = 0.0
        self.flops_summary = None
        # Generation speed accounting (same naming style as eval.py additions).
        self.total_generated_tokens = 0
        self.total_generation_time = 0.0
        self.tokens_per_second = None
        # lm-eval passes model_args as strings; normalize to bool.
        if isinstance(profile_flops, str):
            self._profile_flops = profile_flops.strip().lower() in ("1", "true", "yes", "y")
        else:
            self._profile_flops = bool(profile_flops)

        type(self).last_instance = self

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def tokenizer_name(self):
        return self.model_path

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        return self.tokenizer.apply_chat_template(
            chat_history, add_generation_prompt=add_generation_prompt, tokenize=False
        )

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def _encode_pair(self, context, continuation):
        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape
        batch[:, prompt_index.sum()] = self.mask_id
        batch = torch.cat(
            [
                batch.to(self.device),
                torch.full(
                    (b, self.bd_size - batch.shape[1] % self.bd_size),
                    self.mask_id,
                    dtype=torch.long,
                    device=self.device,
                ),
            ],
            dim=1,
        )
        if batch.shape[1] > l:
            batch[:, l] = self.tokenizer.eos_token_id
        return batch

    @torch.no_grad()
    def get_logits(self, batch):
        logits = self.model(batch).logits
        logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
        return logits[:, : batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        loss_acc = []
        perturbed_seq = self._forward_process(seq.clone(), prompt_index)
        mask_indices = perturbed_seq == self.mask_id
        logits = self.get_logits(perturbed_seq)
        seq = torch.cat(
            [
                seq.to(self.device),
                torch.full(
                    (seq.shape[0], self.bd_size - seq.shape[1] % self.bd_size),
                    -100,
                    dtype=torch.long,
                    device=self.device,
                ),
            ],
            dim=1,
        )
        loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction="none")
        loss = loss.sum()
        loss_acc.append(loss.item())
        return -sum(loss_acc) / len(loss_acc)

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]
        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]
                ll = self.get_loglikelihood(prefix, target)
                out.append((ll, 0.0))
        torch.cuda.empty_cache()
        return out

    def _profile_mdm_sample(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)

    def generate_until(self, requests):
        output = [None] * len(requests)
        num_tokens = 0
        start_time = time.time()

        requests_with_indices = [(i, req) for i, req in enumerate(requests)]
        requests_with_indices.sort(key=lambda x: len(x[1].args[0]))

        batched_requests = []
        current_batch = []
        for i, req in requests_with_indices:
            current_batch.append((i, req))
            if len(current_batch) == self.batch_size:
                batched_requests.append(current_batch)
                current_batch = []

        if current_batch:
            batched_requests.append(current_batch)

        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(self.model)
        else:
            model = self.model

        for batch in tqdm(batched_requests, desc="Generating..."):
            batched_input_ids = []
            max_len = 0
            min_len = 1e9
            seq_len = []

            for orig_idx, req in batch:
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
                model_inputs = self.tokenizer([question], return_tensors="pt").to(self.device)
                batched_input_ids.append(model_inputs["input_ids"])
                max_len = max(max_len, model_inputs["input_ids"].shape[1])
                min_len = min(min_len, model_inputs["input_ids"].shape[1])
                seq_len.append(model_inputs["input_ids"].shape[1])

            # pad batched_input_ids to the same length
            batched_input_ids = [
                torch.cat(
                    [
                        input_ids,
                        torch.full(
                            (1, max_len - input_ids.shape[1]),
                            self.mask_id,
                            dtype=torch.long,
                            device=self.device,
                        ),
                    ],
                    dim=1,
                )
                for input_ids in batched_input_ids
            ]
            batched_input_ids = torch.cat(batched_input_ids, dim=0)
            batched_input_ids = batched_input_ids.to(self.device)

            with torch.no_grad():
                mdm_kwargs = dict(
                    tokenizer=self.tokenizer,
                    block_size=self.bd_size,
                    small_block_size=self.small_block_size,
                    max_new_tokens=self.max_new_tokens,
                    mask_id=self.mask_id,
                    min_len=min_len,
                    seq_len=torch.tensor(seq_len, device=self.device),
                    use_block_cache=self.use_block_cache,
                    threshold=self.threshold,
                    estimate_flops=self._profile_flops,
                )
                if getattr(self, "_use_token_policy", False):
                    mdm_kwargs["token_policy_threshold"] = self.token_policy_threshold
                    mdm_kwargs["token_policy_freeze_step"] = self.token_policy_freeze_step
                if getattr(self, "_use_layer_policy", False):
                    mdm_kwargs["layer_policy_threshold"] = self.layer_policy_threshold
                    mdm_kwargs["layer_policy_freeze_step"] = self.layer_policy_freeze_step
                generated_ids = self._profile_mdm_sample(
                    model.mdm_sample,
                    batched_input_ids,
                    **mdm_kwargs,
                )

            # extract new generated tokens, and keep original index order
            for batch_pos, (orig_idx, req) in enumerate(batch):
                generated_answer = self.tokenizer.decode(
                    generated_ids[batch_pos][seq_len[batch_pos] :], skip_special_tokens=True
                )
                num_tokens += int((generated_ids[batch_pos][seq_len[batch_pos] :] != self.mask_id).sum().item())

                output[orig_idx] = generated_answer

        if self._profile_flops:
            generation_functions._finalize_decode_flops_estimator(model)
            run_flops = getattr(model, "last_estimated_flops", None)
            if run_flops is not None:
                self.total_flops += float(run_flops)
            run_summary = getattr(model, "last_flops_summary", None)
            if run_summary is not None:
                if self.flops_summary is None:
                    self.flops_summary = dict(run_summary)
                else:
                    for key, value in run_summary.items():
                        if key in ("policy_name", "block_size"):
                            self.flops_summary[key] = value
                        elif isinstance(value, (int, float)):
                            self.flops_summary[key] = self.flops_summary.get(key, 0) + value

        end_time = time.time()
        elapsed = end_time - start_time
        self.total_generated_tokens = int(num_tokens)
        self.total_generation_time = float(elapsed)
        self.tokens_per_second = (
            (num_tokens / elapsed) if elapsed > 0 else 0.0
        )
        if self.show_speed:
            print(f"Total number of tokens generated: {num_tokens}")
            print(f"Total time taken: {elapsed} seconds")
            print(f"Tokens per second: {num_tokens / elapsed if elapsed > 0 else 0.0}")

        return output


@register_model("fast_dllm_v2_org")
class Fast_dLLM_v2OrgEvalHarness(_BaseFastDLLMEvalHarness):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Original sampling path (batch_sample).
        self.model.mdm_sample = types.MethodType(
            generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample, self.model
        )


@register_model("fast_dllm_v2_token_policy")
class Fast_dLLM_v2TokenPolicyEvalHarness(_BaseFastDLLMEvalHarness):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Token-policy flashattention sampling path.
        self.model.mdm_sample = types.MethodType(
            generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample_token_policy_flashattention,
            self.model,
        )
        self._use_token_policy = True


@register_model("fast_dllm_v2_layer_policy")
class Fast_dLLM_v2LayerPolicyEvalHarness(_BaseFastDLLMEvalHarness):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Layer-policy sampling path.
        self.model.mdm_sample = types.MethodType(
            generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample_layer_policy,
            self.model,
        )
        self._use_layer_policy = True


@register_model("fast_dllm_v2_combined_policy")
class Fast_dLLM_v2CombinedPolicyEvalHarness(_BaseFastDLLMEvalHarness):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Combined layer+token policy sampling path.
        self.model.mdm_sample = types.MethodType(
            generation_functions.Fast_dLLM_QwenForCausalLM.batch_sample_layer_and_token_policy,
            self.model,
        )
        self._use_token_policy = True
        self._use_layer_policy = True


def _extract_metrics(results, task_name):
    """
    Return the raw metrics dict for the task (same as cli_evaluate output).
    """
    task_metrics = results["results"].get(task_name, {})
    if not task_metrics:
        raise ValueError(f"No metrics found for task {task_name}. Keys: {list(results['results'].keys())}")
    return task_metrics


def _parse_token_policy_configs(args):
    def _parse_float_list(raw):
        return [float(x.strip()) for x in raw.split(",") if x.strip()]

    def _parse_int_list(raw):
        return [int(x.strip()) for x in raw.split(",") if x.strip()]

    thresholds = _parse_float_list(args.token_policy_thresholds)
    freeze_steps = _parse_int_list(args.token_policy_freeze_steps)

    if len(thresholds) != len(freeze_steps):
        raise ValueError(
            "token_policy_thresholds and token_policy_freeze_steps must have the same number of values."
        )

    configs = []
    for freeze_step, threshold in zip(freeze_steps, thresholds):
        configs.append({"freeze_step": int(freeze_step), "threshold": float(threshold)})
    return configs


def _parse_layer_policy_configs(args):
    def _parse_float_list(raw):
        return [float(x.strip()) for x in raw.split(",") if x.strip()]

    def _parse_int_list(raw):
        return [int(x.strip()) for x in raw.split(",") if x.strip()]

    thresholds = _parse_float_list(args.layer_policy_thresholds)
    freeze_steps = _parse_int_list(args.layer_policy_freeze_steps)

    if len(thresholds) != len(freeze_steps):
        raise ValueError(
            "layer_policy_thresholds and layer_policy_freeze_steps must have the same number of values."
        )

    configs = []
    for freeze_step, threshold in zip(freeze_steps, thresholds):
        configs.append({"freeze_step": int(freeze_step), "threshold": float(threshold)})
    return configs


def _parse_combined_policy_configs(args):
    def _parse_float_list(raw):
        return [float(x.strip()) for x in raw.split(",") if x.strip()]

    def _parse_int_list(raw):
        return [int(x.strip()) for x in raw.split(",") if x.strip()]

    token_thresholds = _parse_float_list(args.combined_token_policy_thresholds)
    token_freeze_steps = _parse_int_list(args.combined_token_policy_freeze_steps)
    layer_thresholds = _parse_float_list(args.combined_layer_policy_thresholds)
    layer_freeze_steps = _parse_int_list(args.combined_layer_policy_freeze_steps)

    n = len(token_thresholds)
    if not (
        n == len(token_freeze_steps)
        and n == len(layer_thresholds)
        and n == len(layer_freeze_steps)
    ):
        raise ValueError(
            "combined_* lists must have the same number of values "
            "(token thresholds, token freeze steps, layer thresholds, layer freeze steps)."
        )

    configs = []
    for i in range(n):
        configs.append(
            {
                "token_threshold": float(token_thresholds[i]),
                "token_freeze_step": int(token_freeze_steps[i]),
                "layer_threshold": float(layer_thresholds[i]),
                "layer_freeze_step": int(layer_freeze_steps[i]),
            }
        )
    return configs


def _policy_tag_from_configs(configs, policy_prefix="token_policy"):
    if policy_prefix == "combined_policy":
        if len(configs) == 1:
            cfg = configs[0]
            return (
                "combined_policy_"
                f"ls{cfg['layer_freeze_step']}_lt{cfg['layer_threshold']}_"
                f"ts{cfg['token_freeze_step']}_tt{cfg['token_threshold']}"
            )
        tag_parts = []
        for cfg in configs:
            lthr = str(cfg["layer_threshold"]).replace(".", "p")
            tthr = str(cfg["token_threshold"]).replace(".", "p")
            tag_parts.append(
                f"ls{cfg['layer_freeze_step']}lt{lthr}_ts{cfg['token_freeze_step']}tt{tthr}"
            )
        return "combined_policy_multi_" + "_".join(tag_parts)

    if len(configs) == 1:
        cfg = configs[0]
        return f"{policy_prefix}_{cfg['freeze_step']}_{cfg['threshold']}"

    tag_parts = []
    for cfg in configs:
        thr = str(cfg["threshold"]).replace(".", "p")
        tag_parts.append(f"s{cfg['freeze_step']}t{thr}")
    return f"{policy_prefix}_multi_" + "_".join(tag_parts)


def _plot_bars(
    task_name,
    metric_label,
    series_labels,
    acc_values,
    flops_values,
    output_dir,
    policy_tag,
    file_suffix="",
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Two categories: Accuracy (left axis) and TFLOPs (right axis)
    x = np.arange(2)
    n_series = len(series_labels)
    width = 0.8 / max(n_series, 1)

    fig, ax1 = plt.subplots(figsize=(7.0, 4.2))
    ax2 = ax1.twinx()

    # Use a larger palette to avoid repeated colors when many configs are plotted.
    if n_series <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_series))
    else:
        colors = plt.cm.hsv(np.linspace(0, 1, n_series, endpoint=False))
    offsets = [(i - (n_series - 1) / 2) * width for i in range(n_series)]

    acc_plot = [np.nan if v is None else v for v in acc_values]
    flops_plot = [np.nan if v is None else v for v in flops_values]

    # Accuracy bars (left axis) at x=0 and FLOPs bars (right axis) at x=1.
    for i, label in enumerate(series_labels):
        ax1.bar(x[0] + offsets[i], acc_plot[i], width, label=label, color=colors[i])
        ax2.bar(x[1] + offsets[i], flops_plot[i], width, color=colors[i])
    ax1.set_ylabel(metric_label)
    ax2.set_ylabel("Total FLOPs (TFLOPs)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(["Accuracy", "Total FLOPs"])
    title = f"{task_name}: Original vs Token-Policy"
    if metric_label:
        title = f"{task_name}: {metric_label} vs TFLOPs"
    fig.suptitle(title, y=0.995, fontsize=10)

    # Legend maps color to compared method/config.
    handles1, labels1 = ax1.get_legend_handles_labels()
    fig.legend(
        handles1,
        labels1,
        loc="upper left",
        bbox_to_anchor=(0.985, 1.0),
        ncol=2,
        fontsize=8,
        columnspacing=1.4,
        handlelength=1.4,
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0, 0.88, 1])
    suffix = f"_{file_suffix}" if file_suffix else ""
    out_path = output_dir / f"{task_name}_org_vs_{policy_tag}{suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_single_metric_bars(
    task_name,
    metric_label,
    series_labels,
    values,
    output_dir,
    policy_tag,
    file_suffix,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_vals = [np.nan if v is None else v for v in values]
    x = np.arange(len(series_labels))
    if len(series_labels) <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, len(series_labels)))
    else:
        colors = plt.cm.hsv(np.linspace(0, 1, len(series_labels), endpoint=False))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, plot_vals, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(series_labels, rotation=20, ha="right")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{task_name}: {metric_label}")
    fig.tight_layout()
    out_path = output_dir / f"{task_name}_org_vs_{policy_tag}_{file_suffix}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_eval(task_name, model_name, model_args, batch_size, num_fewshot, limit, confirm_run_unsafe_code, fewshot_as_multiturn, apply_chat_template):
    results = evaluator.simple_evaluate(
        model=model_name,
        model_args=model_args,
        tasks=[task_name],
        batch_size=batch_size,
        num_fewshot=num_fewshot,
        limit=limit,
        confirm_run_unsafe_code=confirm_run_unsafe_code,
        fewshot_as_multiturn=fewshot_as_multiturn,
        apply_chat_template=apply_chat_template,
    )
    return results


def _release_harness_instance(harness_cls):
    """
    Free the most recent lm-eval harness instance so sequential runs do not keep
    multiple full models alive on GPU.
    """
    inst = harness_cls.last_instance
    if inst is None:
        return
    try:
        if hasattr(inst, "model"):
            del inst.model
    except Exception:
        pass
    harness_cls.last_instance = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Compare original vs token-policy/layer-policy sampling.")
    parser.add_argument("--tasks", required=True, help="Task name: gsm8k or minerva_math")
    parser.add_argument("--model_path", default="Efficient-Large-Model/Fast_dLLM_v2_7B")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--subset_percent", type=float, default=100.0)
    parser.add_argument("--confirm_run_unsafe_code", action="store_true")
    parser.add_argument("--fewshot_as_multiturn", action="store_true")
    parser.add_argument("--apply_chat_template", action="store_true")
    parser.add_argument(
        "--enable_block_cache",
        action="store_true",
        help="Enable block cache for all evaluated generation functions.",
    )
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--show_speed", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--small_block_size", type=int, default=8)
    parser.add_argument("--bd_size", type=int, default=32)
    parser.add_argument(
        "--token_policy_thresholds",
        type=str,
        default="0.99",
        help="Comma-separated thresholds for multiple token-policy runs, e.g. 0.99,0.995",
    )
    parser.add_argument(
        "--token_policy_freeze_steps",
        type=str,
        default="3",
        help="Comma-separated freeze steps for multiple token-policy runs, e.g. 3,5",
    )
    parser.add_argument(
        "--layer_policy_thresholds",
        type=str,
        default="0.99",
        help="Comma-separated thresholds for multiple layer-policy runs, e.g. 0.99,0.995",
    )
    parser.add_argument(
        "--layer_policy_freeze_steps",
        type=str,
        default="3",
        help="Comma-separated freeze steps for multiple layer-policy runs, e.g. 3,5",
    )
    parser.add_argument(
        "--combined_token_policy_thresholds",
        type=str,
        default="0.99",
        help="Comma-separated token thresholds for combined-policy runs, e.g. 0.99,0.995",
    )
    parser.add_argument(
        "--combined_token_policy_freeze_steps",
        type=str,
        default="3",
        help="Comma-separated token freeze steps for combined-policy runs, e.g. 3,5",
    )
    parser.add_argument(
        "--combined_layer_policy_thresholds",
        type=str,
        default="0.99",
        help="Comma-separated layer thresholds for combined-policy runs, e.g. 0.99,0.995",
    )
    parser.add_argument(
        "--combined_layer_policy_freeze_steps",
        type=str,
        default="3",
        help="Comma-separated layer freeze steps for combined-policy runs, e.g. 3,5",
    )
    parser.add_argument(
        "--profile_flops",
        action="store_true",
        help="Enable decode-log based FLOPs estimation during generation.",
    )
    parser.add_argument("--output_dir", default="policy/plots")
    parser.add_argument("--results_dir", default="policy/results")
    parser.add_argument(
        "--token_policy_only",
        action="store_true",
        help="Only run the token-policy sampling (skip original).",
    )
    parser.add_argument(
        "--layer_policy_only",
        action="store_true",
        help="Only run the layer-policy sampling (skip original).",
    )
    parser.add_argument(
        "--combined_policy_only",
        action="store_true",
        help="Only run the combined layer+token policy sampling (skip original).",
    )
    parser.add_argument(
        "--enable_original",
        action="store_true",
        help="Also run the original sampling path and include it in JSON/plots.",
    )
    args = parser.parse_args()

    # Both methods are batch-size=1 only for this comparison.
    if args.batch_size != 1:
        raise ValueError("This comparison requires batch_size=1 for both methods.")

    task = args.tasks
    if "," in task:
        raise ValueError("Please run one dataset per invocation (e.g., --tasks gsm8k).")

    # Convert subset percent to lm-eval limit fraction.
    limit = None
    if args.subset_percent < 100.0:
        limit = args.subset_percent / 100.0

    mode_flags = [
        bool(args.token_policy_only),
        bool(args.layer_policy_only),
        bool(args.combined_policy_only),
    ]
    if sum(mode_flags) > 1:
        raise ValueError(
            "Please choose only one of --token_policy_only, --layer_policy_only, or --combined_policy_only."
        )

    if args.enable_block_cache:
        if args.results_dir == "policy/results":
            args.results_dir = "policy/results_use_block_cache"
        if args.output_dir == "policy/plots":
            args.output_dir = "policy/plots_use_block_cache"

    if args.combined_policy_only:
        policy_mode = "combined"
        policy_prefix = "combined_policy"
        policy_model_name = "fast_dllm_v2_combined_policy"
        policy_label = "combined_policy"
    elif args.layer_policy_only:
        policy_mode = "layer"
        policy_prefix = "layer_policy"
        policy_model_name = "fast_dllm_v2_layer_policy"
        policy_label = "layer_policy"
    else:
        policy_mode = "token"
        policy_prefix = "token_policy"
        policy_model_name = "fast_dllm_v2_token_policy"
        policy_label = "token_policy"
    run_original = bool(args.enable_original)
    if policy_mode == "layer":
        policy_configs = _parse_layer_policy_configs(args)
    elif policy_mode == "combined":
        policy_configs = _parse_combined_policy_configs(args)
    else:
        policy_configs = _parse_token_policy_configs(args)

    org_metrics = None
    org_flops = None
    org_tokens = None
    org_time = None
    org_tps = None
    org_flops_summary = None
    if run_original:
        # Original sampling
        org_model_args = (
            f"model_path={args.model_path},"
            f"threshold={args.threshold},"
            f"show_speed={args.show_speed},"
            f"max_new_tokens={args.max_new_tokens},"
            f"small_block_size={args.small_block_size},"
            f"bd_size={args.bd_size},"
            f"use_block_cache={args.enable_block_cache},"
            f"profile_flops={args.profile_flops}"
        )
        org_results = run_eval(
            task_name=task,
            model_name="fast_dllm_v2_org",
            model_args=org_model_args,
            batch_size=args.batch_size,
            num_fewshot=args.num_fewshot,
            limit=limit,
            confirm_run_unsafe_code=args.confirm_run_unsafe_code,
            fewshot_as_multiturn=args.fewshot_as_multiturn,
            apply_chat_template=args.apply_chat_template,
        )
        org_metrics = _extract_metrics(org_results, task)
        org_inst = Fast_dLLM_v2OrgEvalHarness.last_instance
        org_flops = org_inst.total_flops / 1e12
        org_tokens = org_inst.total_generated_tokens
        org_time = org_inst.total_generation_time
        org_tps = org_inst.tokens_per_second
        org_flops_summary = org_inst.flops_summary
        _release_harness_instance(Fast_dLLM_v2OrgEvalHarness)

    policy_runs = []
    # Policy sampling for each config in this run.
    for cfg in policy_configs:
        policy_model_args = (
            f"model_path={args.model_path},"
            f"threshold={args.threshold},"
            f"show_speed={args.show_speed},"
            f"max_new_tokens={args.max_new_tokens},"
            f"small_block_size={args.small_block_size},"
            f"bd_size={args.bd_size},"
            f"use_block_cache={args.enable_block_cache},"
            f"profile_flops={args.profile_flops}"
        )
        if policy_mode == "layer":
            policy_model_args += (
                f",layer_policy_threshold={cfg['threshold']},"
                f"layer_policy_freeze_step={cfg['freeze_step']}"
            )
        elif policy_mode == "combined":
            policy_model_args += (
                f",layer_policy_threshold={cfg['layer_threshold']},"
                f"layer_policy_freeze_step={cfg['layer_freeze_step']},"
                f"token_policy_threshold={cfg['token_threshold']},"
                f"token_policy_freeze_step={cfg['token_freeze_step']}"
            )
        else:
            policy_model_args += (
                f",token_policy_threshold={cfg['threshold']},"
                f"token_policy_freeze_step={cfg['freeze_step']}"
            )
        policy_results = run_eval(
            task_name=task,
            model_name=policy_model_name,
            model_args=policy_model_args,
            batch_size=args.batch_size,
            num_fewshot=args.num_fewshot,
            limit=limit,
            confirm_run_unsafe_code=args.confirm_run_unsafe_code,
            fewshot_as_multiturn=args.fewshot_as_multiturn,
            apply_chat_template=args.apply_chat_template,
        )
        metrics = _extract_metrics(policy_results, task)
        if policy_mode == "layer":
            policy_inst = Fast_dLLM_v2LayerPolicyEvalHarness.last_instance
        elif policy_mode == "combined":
            policy_inst = Fast_dLLM_v2CombinedPolicyEvalHarness.last_instance
        else:
            policy_inst = Fast_dLLM_v2TokenPolicyEvalHarness.last_instance
        flops = policy_inst.total_flops / 1e12
        policy_runs.append(
            {
                "config": cfg,
                "metrics": metrics,
                "tflops": flops,
                "total_generated_tokens": policy_inst.total_generated_tokens,
                "total_generation_time": policy_inst.total_generation_time,
                "tokens_per_second": policy_inst.tokens_per_second,
                "flops_summary": policy_inst.flops_summary,
            }
        )
        if policy_mode == "layer":
            _release_harness_instance(Fast_dLLM_v2LayerPolicyEvalHarness)
        elif policy_mode == "combined":
            _release_harness_instance(Fast_dLLM_v2CombinedPolicyEvalHarness)
        else:
            _release_harness_instance(Fast_dLLM_v2TokenPolicyEvalHarness)

    def _fmt_float(v):
        return f"{v:.4f}" if v is not None else "None"

    if org_metrics is not None:
        print(f"[{task}] original metrics: {org_metrics}")
        print(f"[{task}] original TFLOPs:   {org_flops:.4f}")
        print(f"[{task}] original total_generated_tokens: {org_tokens}")
        print(f"[{task}] original total_generation_time:  {_fmt_float(org_time)}")
        print(f"[{task}] original tokens_per_second:     {_fmt_float(org_tps)}")
    for run in policy_runs:
        cfg = run["config"]
        if policy_mode == "combined":
            cfg_tag = (
                f"ls={cfg['layer_freeze_step']}, lt={cfg['layer_threshold']}, "
                f"ts={cfg['token_freeze_step']}, tt={cfg['token_threshold']}"
            )
        else:
            cfg_tag = f"s={cfg['freeze_step']}, t={cfg['threshold']}"
        print(f"[{task}] {policy_label}({cfg_tag}) metrics: {run['metrics']}")
        print(f"[{task}] {policy_label}({cfg_tag}) TFLOPs:  {run['tflops']:.4f}")
        print(f"[{task}] {policy_label}({cfg_tag}) total_generated_tokens: {run['total_generated_tokens']}")
        print(f"[{task}] {policy_label}({cfg_tag}) total_generation_time:  {_fmt_float(run['total_generation_time'])}")
        tps = run["tokens_per_second"]
        print(f"[{task}] {policy_label}({cfg_tag}) tokens_per_second:     {_fmt_float(tps)}")

    policy_tag = _policy_tag_from_configs(policy_configs, policy_prefix=policy_prefix)

    # Persist JSON summary and bar plot for this dataset.
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"{task}_org_vs_{policy_tag}.json"
    with result_path.open("w") as f:
        json.dump(
            {
                "task": task,
                "subset_percent": args.subset_percent,
                "original": {
                    "metrics": org_metrics,
                    "tflops": org_flops,
                    "total_generated_tokens": org_tokens,
                    "total_generation_time": org_time,
                    "tokens_per_second": org_tps,
                    "flops_summary": org_flops_summary,
                },
                "policies": policy_runs,
            },
            f,
            indent=2,
        )

    # Extract flexible exact_match metric when available.
    def _metric_variant(metrics, key_name):
        if metrics is None:
            return None
        return float(metrics[key_name]) if key_name in metrics else None

    flexible_key = "exact_match,flexible-extract"

    series_labels = []
    acc_values = []
    flops_values = []
    tps_values = []

    if org_metrics is not None:
        series_labels.append("orig")
        acc_values.append(_metric_variant(org_metrics, flexible_key))
        flops_values.append(org_flops)
        tps_values.append(org_tps)

    for run in policy_runs:
        cfg = run["config"]
        if policy_mode == "combined":
            series_labels.append(
                f"{policy_label}_ls{cfg['layer_freeze_step']}_lt{cfg['layer_threshold']}_"
                f"ts{cfg['token_freeze_step']}_tt{cfg['token_threshold']}"
            )
        else:
            series_labels.append(f"{policy_label}_s{cfg['freeze_step']}_t{cfg['threshold']}")
        acc_values.append(_metric_variant(run["metrics"], flexible_key))
        flops_values.append(run["tflops"])
        tps_values.append(run["tokens_per_second"])

    # Plot flexible-extract metric if present.
    if any(v is not None for v in acc_values):
        _plot_bars(
            task,
            "Exact Match (flexible)",
            series_labels,
            acc_values,
            flops_values,
            args.output_dir,
            policy_tag=policy_tag,
            file_suffix="flexible",
        )

    # Plot tokens/sec as a separate graph for all compared series.
    if any(v is not None for v in tps_values):
        _plot_single_metric_bars(
            task_name=task,
            metric_label="Tokens per second",
            series_labels=series_labels,
            values=tps_values,
            output_dir=args.output_dir,
            policy_tag=policy_tag,
            file_suffix="tokens_per_second",
        )


if __name__ == "__main__":
    main()
