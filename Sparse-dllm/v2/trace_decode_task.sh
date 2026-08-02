#!/usr/bin/env bash
set -euo pipefail

# Set the environment variables first before running the command.
export CUDA_VISIBLE_DEVICES=7
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
model_path=Efficient-Large-Model/Fast_dLLM_v2_7B

# task=mmlu
# accelerate launch trace_decode_task.py --tasks ${task} --batch_size 1 --num_fewshot 5 --limit 1 \
# --confirm_run_unsafe_code --fewshot_as_multiturn --apply_chat_template \
# --model_args model_path=${model_path}

# task=gpqa_main_n_shot
# accelerate launch trace_decode_task.py --tasks ${task} --batch_size 1 --limit 1 \
# --confirm_run_unsafe_code --fewshot_as_multiturn --apply_chat_template \
# --model_args model_path=${model_path}

task=gsm8k
accelerate launch trace_decode_task.py --tasks ${task} --batch_size 1 --num_fewshot 0 --limit 1 \
--confirm_run_unsafe_code --fewshot_as_multiturn --apply_chat_template \
--model_args model_path=${model_path},threshold=1,show_speed=True \
--plot_blocks 7,8,9

# task=minerva_math
# accelerate launch trace_decode_task.py --tasks ${task} --batch_size 1 --num_fewshot 0 --limit 1 \
# --confirm_run_unsafe_code --fewshot_as_multiturn --apply_chat_template \
# --model_args model_path=${model_path},threshold=1,show_speed=True
# ,use_block_cache=true

# task=ifeval
# accelerate launch trace_decode_task.py --tasks ${task} --batch_size 1 --limit 1 \
# --confirm_run_unsafe_code --fewshot_as_multiturn --apply_chat_template \
# --model_args model_path=${model_path},threshold=1,show_speed=True
