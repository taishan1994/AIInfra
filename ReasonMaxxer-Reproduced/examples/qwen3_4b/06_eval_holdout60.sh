#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
MODEL_PATH=${MODEL_PATH:-/nfs/FM/gongoubo/checkpoints/Qwen/Qwen3-4B}
ROOT=${ROOT:-outputs/qwen3_4b_default}
RUN_NAME=${RUN_NAME:-qwen3_4b_reasonmaxxer_tau1p4_r32_lr1e4_ep2_seed42}
HOLDOUT_IDS_FILE=${HOLDOUT_IDS_FILE:-$ROOT/train/holdout60_ids_tau1p4.json}
RECORDS_FILE=${RECORDS_FILE:-$ROOT/sampled_records/records_l345_300.json}
export VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-1}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $PYTHON scripts/eval_checkpoints.py \
  --run_dir "$ROOT/checkpoints/$RUN_NAME" \
  --base_model "$MODEL_PATH" \
  --max_lora_rank 32 \
  --holdout_ids_file "$HOLDOUT_IDS_FILE" \
  --holdout_ids_key problem_ids \
  --dataset math500 \
  --records_file "$RECORDS_FILE" \
  --problem_set full \
  --max_problems 60 \
  --output_tag holdout60 \
  --output_dir "$ROOT/eval/${RUN_NAME}_holdout60_n1" \
  --num_generations 1 \
  --device 0 \
  --batch_size 4 \
  --temperature 0.6 \
  --top_p 0.95 \
  --max_tokens 8192 \
  --seed 42 \
  --match_timeout_s 0.2 \
  --prompt_style auto \
  --stop_profile auto \
  --math500_extractor auto \
  --qwen3_enable_thinking false \
  --force \
  --update_metrics_csv
