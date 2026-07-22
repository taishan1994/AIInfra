#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
MODEL_PATH=${MODEL_PATH:-/nfs/FM/gongoubo/checkpoints/Qwen/Qwen3-4B}
ROOT=${ROOT:-outputs/qwen3_4b_default}
CHECKPOINT=${CHECKPOINT:?set CHECKPOINT to the chosen LoRA checkpoint directory}
MAX_LORA_RANK=${MAX_LORA_RANK:-32}
NUM_GENERATIONS=${NUM_GENERATIONS:-1}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_TOKENS=${MAX_TOKENS:-8192}
SEED=${SEED:-42}
DATASETS=${DATASETS:-"math500 gsm8k"}
export VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-1}

mkdir -p "$ROOT/final_eval"

for DATASET in $DATASETS; do
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $PYTHON scripts/generate_rollouts.py \
    --model_path "$MODEL_PATH" \
    --lora_adapter "$CHECKPOINT" \
    --max_lora_rank "$MAX_LORA_RANK" \
    --condition_name "$(basename "$CHECKPOINT")_${DATASET}" \
    --dataset "$DATASET" \
    --problem_set full \
    --prompt_style auto \
    --stop_profile auto \
    --qwen3_enable_thinking false \
    --num_generations "$NUM_GENERATIONS" \
    --batch_size "$BATCH_SIZE" \
    --tensor_parallel_size 1 \
    --temperature 0.6 \
    --top_p 0.95 \
    --max_tokens "$MAX_TOKENS" \
    --seed "$SEED" \
    --math500_extractor auto \
    --output_dir "$ROOT/final_eval" \
    --output_name "$(basename "$CHECKPOINT")_${DATASET}.json" \
    --force
done

for DATASET in aime24 aime25 amc23 minerva_math olympiadbench; do
  RECORDS_FILE=${DATASET^^}_RECORDS_FILE
  RECORDS_PATH=${!RECORDS_FILE:-}
  if [ -n "$RECORDS_PATH" ]; then
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $PYTHON scripts/generate_rollouts.py \
      --model_path "$MODEL_PATH" \
      --lora_adapter "$CHECKPOINT" \
      --max_lora_rank "$MAX_LORA_RANK" \
      --condition_name "$(basename "$CHECKPOINT")_${DATASET}" \
      --dataset "$DATASET" \
      --records_file "$RECORDS_PATH" \
      --problem_set full \
      --prompt_style auto \
      --stop_profile auto \
      --qwen3_enable_thinking false \
      --num_generations "$NUM_GENERATIONS" \
      --batch_size "$BATCH_SIZE" \
      --tensor_parallel_size 1 \
      --temperature 0.6 \
      --top_p 0.95 \
      --max_tokens "$MAX_TOKENS" \
      --seed "$SEED" \
      --math500_extractor auto \
      --output_dir "$ROOT/final_eval" \
      --output_name "$(basename "$CHECKPOINT")_${DATASET}.json" \
      --force
  fi
done
