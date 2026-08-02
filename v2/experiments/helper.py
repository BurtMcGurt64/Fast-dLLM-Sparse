import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from model.configuration import Fast_dLLM_QwenConfig
from model.modeling import Fast_dLLM_QwenForCausalLM

from datasets import load_dataset
import datasets as _datasets
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache

# this model uses a non-standard DynamicCache API — .layers[idx].keys/values
# instead of standard .key_cache[idx] / .value_cache[idx]
DynamicCache.__getitem__ = lambda self, layer_idx: (self.layers[layer_idx].keys, self.layers[layer_idx].values)

class _LayerProxy:
    """Proxy so that cache.key_cache[i] / cache.value_cache[i] work via .layers."""
    def __init__(self, cache, attr):
        self._cache = cache
        self._attr = attr
    def __getitem__(self, i):
        return getattr(self._cache.layers[i], self._attr)
    def __setitem__(self, i, v):
        setattr(self._cache.layers[i], self._attr, v)
    def __len__(self):
        return len(self._cache.layers)

DynamicCache.key_cache   = property(lambda self: _LayerProxy(self, 'keys'))
DynamicCache.value_cache = property(lambda self: _LayerProxy(self, 'values'))

import torch
import os
import json
import re
import types

_datasets.logging.set_verbosity_error()  # suppress "Using cached version" noise

_GSM8K_CACHE = None

def _get_gsm8k():
    global _GSM8K_CACHE
    if _GSM8K_CACHE is None:
        _GSM8K_CACHE = load_dataset("gsm8k", "main")["test"]
    return _GSM8K_CACHE


import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_MODEL = "Efficient-Large-Model/Fast_dLLM_v2_7B"

def load_model(model_id: str = DEFAULT_MODEL, device: str = "cuda:0"):
    # from_pretrained skips weight loading for ;
    # use snapshot_download and manual load_state_dict
    snapshot = snapshot_download(model_id, local_files_only=True)
    config = Fast_dLLM_QwenConfig.from_pretrained(snapshot)
    model = Fast_dLLM_QwenForCausalLM(config).to(torch.bfloat16)

    index_path = os.path.join(snapshot, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    state_dict = {}
    for shard in sorted(set(index["weight_map"].values())):
        state_dict.update(load_file(os.path.join(snapshot, shard), device="cpu"))
    model.load_state_dict(state_dict, strict=True)

    return model.to(device).eval()

def load_gsm8k(n, tokenizer, device="cuda:0", batch_size=1):
    """
    Load 'n' GSM8K samples, default with batch_size=1.
    If batch=1 the output is a list of tensors with shape [1, seq_len_i]
    If batch>1 the output is a list of tensors with shape [batch_size, max_seq_len], where we left-pad
    """

    dataset = _get_gsm8k()
    questions = dataset[:n]["question"]  # list of n strings

    texts = []
    for q in questions:
        messages = [{"role": "user", "content": q}]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    if batch_size == 1:
        return [tokenizer(t, return_tensors="pt").input_ids.to(device) for t in texts]

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    batches = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        encoded = tokenizer(chunk, return_tensors="pt", padding=True).to(device)
        batches.append(encoded)

    tokenizer.padding_side = original_padding_side
    return batches


def load_gsm8k_stacked(n_questions, questions_per_prompt, tokenizer, device, batch_size=1):
    """
    Build prompts where each prompt contains questions_per_prompt GSM8K questions
    concatenated, asking the model to answer all of them.
    n_questions: total questions to draw from (must be >= questions_per_prompt * n_prompts)
    Returns same format as load_gsm8k: list of BatchEncodings (batch_size>1) or list of tensors.
    """
    dataset = _get_gsm8k()
    all_questions = dataset[:n_questions]["question"]

    # group into stacks of questions_per_prompt
    stacked_texts = []
    for i in range(0, len(all_questions) - questions_per_prompt + 1, questions_per_prompt):
        group = all_questions[i : i + questions_per_prompt]
        numbered = "\n\n".join(f"Problem {j+1}: {q}" for j, q in enumerate(group))
        prompt = numbered + f"\n\nPlease solve all {questions_per_prompt} problems above, answering each in order."
        messages = [{"role": "user", "content": prompt}]
        stacked_texts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    batches = []
    for i in range(0, len(stacked_texts), batch_size):
        chunk = stacked_texts[i : i + batch_size]
        encoded = tokenizer(chunk, return_tensors="pt", padding=True).to(device)
        batches.append(encoded)

    tokenizer.padding_side = original_padding_side
    return batches


def extract_answer(text):
    """Extract the numeric answer after #### in GSM8K format."""
    match = re.search(r'####\s*([\d,]+)', text)
    if match:
        return match.group(1).replace(',', '')
    return None


def check_stacked_answers(generated_text, ground_truth_answers, debug=False):
    """
    Parse answers for each sub-question from generated text and compare to ground truths.
    Tries #### format first; falls back to splitting by 'Problem N:' and taking the last
    number in each section (the model writes answers in prose without ####).
    Returns list of True/False/None (None = no answer found for that question).
    """
    # try #### format first
    predicted = re.findall(r'####\s*([\d,]+)', generated_text)
    if predicted:
        results = []
        for i, gt in enumerate(ground_truth_answers):
            if i < len(predicted):
                results.append(predicted[i].replace(',', '') == gt.replace(',', ''))
            else:
                results.append(None)
        return results

    # fall back: split on "Problem N:" or numbered list "N." headers
    # split on "Problem N:" or "N." headers; drop any leading preamble before first problem
    parts = re.split(r'(?:Problem\s+\d+[:.]?|\b\d+\.\s)', generated_text)
    # first part is preamble ("Sure, I'd be happy...") before Problem 1 — discard it
    sections = [s.strip() for s in parts[1:] if s.strip()]

    results = []
    for i, gt in enumerate(ground_truth_answers):
        if i < len(sections):
            nums = re.findall(r'\b\d+(?:,\d+)*\b', sections[i])
            pred = nums[-1].replace(',', '') if nums else None
            if debug:
                snippet = sections[i][:200].replace('\n', ' ')
                print(f"  Q{i+1}: gt={gt}  pred={pred}  | ...{snippet}...")
            results.append(pred == gt.replace(',', '') if pred else None)
        else:
            if debug:
                print(f"  Q{i+1}: gt={gt}  pred=None  | (no section found)")
            results.append(None)
    return results


def generate_and_check(model, tokenizer, questions_per_prompt, question_offset, device):
    """
    Run generation on one stacked prompt (batch=1) and check per-question accuracy.
    question_offset: which questions to use from the dataset (to match what the sweep used).
    Returns list of True/False/None per question.
    """
    from generation_functions import Fast_dLLM_QwenForCausalLM as GenClass

    dataset = _get_gsm8k()
    questions = dataset[question_offset : question_offset + questions_per_prompt]["question"]
    ground_truths = [
        extract_answer(dataset[question_offset + i]["answer"])
        for i in range(questions_per_prompt)
    ]

    numbered = "\n\n".join(f"Problem {j+1}: {q}" for j, q in enumerate(questions))
    prompt = numbered + f"\n\nPlease solve all {questions_per_prompt} problems above, answering each in order."
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    max_new_tokens = questions_per_prompt * 400

    sampler = types.MethodType(GenClass.batch_sample, model)
    output_ids = sampler(
        input_ids=input_ids,
        tokenizer=tokenizer,
        block_size=32,
        max_new_tokens=max_new_tokens,
        small_block_size=8,
        min_len=prompt_len,
        seq_len=torch.tensor([prompt_len], device=device),
        threshold=0.95,
        temperature=0.0,
    )

    generated_text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
    return check_stacked_answers(generated_text, ground_truths, debug=True)


def enable_attn_capture(model):
    for layer in model.model.layers:
        layer.self_attn._capture_attn_scores = True

def disable_attn_capture(model):
    for layer in model.model.layers:
        layer.self_attn._capture_attn_scores = False

def collect_attn_weights(model):
    # returns dict: layer_idx -> tensor [B, H, Q, KV]
    weights = {}
    for i, layer in enumerate(model.model.layers):
        w = getattr(layer.self_attn, '_last_attn_weights', None)
        if w is not None:
            weights[i] = w
    return weights
