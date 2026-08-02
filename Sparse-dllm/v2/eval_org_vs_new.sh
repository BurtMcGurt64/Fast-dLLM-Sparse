#!/usr/bin/env bash
set -euo pipefail

# Set the environment variables first before running the command.
export CUDA_VISIBLE_DEVICES=5
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
model_path=Efficient-Large-Model/Fast_dLLM_v2_7B


## token policy
## sweep threshold (running for use block cache comp)
# token_policy_thresholds=0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99
# token_policy_freeze_steps=2,2,2,2,2,2,2,2,2,2

## sweep freeze steps (running for use block cache comp)
# token_policy_thresholds=0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99
# token_policy_freeze_steps=1,2,3,4,5,6,7,8


## layer policy
## sweep threshold (running for use block cache comp)
layer_policy_thresholds=0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99
layer_policy_freeze_steps=2,2,2,2,2,2,2,2,2,2

## sweep freeze steps (running for use block cache comp)
# layer_policy_thresholds=0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95
# layer_policy_freeze_steps=1,2,3,4,5,6,7,8

# Token policy - use_block_cache True
# task=gsm8k
# accelerate launch eval_org_vs_new.py --tasks ${task} --batch_size 1 --num_fewshot 0 --subset_percent 5 \
# --confirm_run_unsafe_code --fewshot_as_multiturn --apply_chat_template \
# --model_path ${model_path} --threshold 1 --show_speed --token_policy_only --profile_flops --enable_original --enable_block_cache \
# --token_policy_thresholds ${token_policy_thresholds} --token_policy_freeze_steps ${token_policy_freeze_steps}

# Layer policy - use_block_cache True
# task=gsm8k
# accelerate launch eval_org_vs_new.py --tasks ${task} --batch_size 1 --num_fewshot 0 --subset_percent 5 \
# --confirm_run_unsafe_code --fewshot_as_multiturn --apply_chat_template \
# --model_path ${model_path} --threshold 1 --show_speed --layer_policy_only --profile_flops --enable_original --enable_block_cache \
# --layer_policy_thresholds ${layer_policy_thresholds} --layer_policy_freeze_steps ${layer_policy_freeze_steps}

# Token policy - use_block_cache False
# task=gsm8k
# accelerate launch eval_org_vs_new.py --tasks ${task} --batch_size 1 --num_fewshot 0 --subset_percent 5 \
# --confirm_run_unsafe_code --fewshot_as_multiturn --apply_chat_template \
# --model_path ${model_path} --threshold 1 --show_speed --token_policy_only --profile_flops --enable_original \
# --token_policy_thresholds ${token_policy_thresholds} --token_policy_freeze_steps ${token_policy_freeze_steps}

# Layer policy - use_block_cache False
task=gsm8k
accelerate launch eval_org_vs_new.py --tasks ${task} --batch_size 1 --num_fewshot 0 --subset_percent 5 \
--confirm_run_unsafe_code --fewshot_as_multiturn --apply_chat_template \
--model_path ${model_path} --threshold 1 --show_speed --layer_policy_only --profile_flops --enable_original \
--layer_policy_thresholds ${layer_policy_thresholds} --layer_policy_freeze_steps ${layer_policy_freeze_steps}





# ------------------------------------------------combined policy runs (running for use block cache)------------------------------------------------



## combined policy (layer + token)
## example sweep with shared list length across all 4 knobs
# combined_layer_policy_thresholds=0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95
# combined_layer_policy_freeze_steps=2,2,2,2,2,2,2,2
# combined_token_policy_thresholds=0.99,0.99,0.99,0.99,0.99,0.99,0.99,0.99
# combined_token_policy_freeze_steps=1,2,3,4,5,6,7,8

# Combined policy - use_block_cache True
# task=gsm8k
# accelerate launch eval_org_vs_new.py --tasks ${task} --batch_size 1 --num_fewshot 0 --subset_percent 5 \
# --confirm_run_unsafe_code --fewshot_as_multiturn --apply_chat_template \
# --model_path ${model_path} --threshold 1 --show_speed --combined_policy_only --profile_flops --enable_original --enable_block_cache \
# --combined_layer_policy_thresholds ${combined_layer_policy_thresholds} --combined_layer_policy_freeze_steps ${combined_layer_policy_freeze_steps} \
# --combined_token_policy_thresholds ${combined_token_policy_thresholds} --combined_token_policy_freeze_steps ${combined_token_policy_freeze_steps}


# Combined policy - use_block_cache False
# task=gsm8k
# accelerate launch eval_org_vs_new.py --tasks ${task} --batch_size 1 --num_fewshot 0 --subset_percent 5 \
# --confirm_run_unsafe_code --fewshot_as_multiturn --apply_chat_template \
# --model_path ${model_path} --threshold 1 --show_speed --combined_policy_only --profile_flops --enable_original \
# --combined_layer_policy_thresholds ${combined_layer_policy_thresholds} --combined_layer_policy_freeze_steps ${combined_layer_policy_freeze_steps} \
# --combined_token_policy_thresholds ${combined_token_policy_thresholds} --combined_token_policy_freeze_steps ${combined_token_policy_freeze_steps}



# task=minerva_math
# accelerate launch eval_org_vs_new.py --tasks ${task} --batch_size 1 --num_fewshot 0 --subset_percent 5 \
# --confirm_run_unsafe_code --fewshot_as_multiturn --apply_chat_template \
# --model_path ${model_path} --threshold 1 --show_speed \
# --token_policy_thresholds ${token_policy_thresholds} --token_policy_freeze_steps ${token_policy_freeze_steps}
