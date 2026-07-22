#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
MODEL_PATH=${MODEL_PATH:-/nfs/FM/gongoubo/checkpoints/Qwen/Qwen3-4B}
ROOT=${ROOT:-outputs/qwen3_4b_default}
NUM_GENERATIONS=${NUM_GENERATIONS:-1}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_TOKENS=${MAX_TOKENS:-8192}
SEED=${SEED:-42}
DATASETS=${DATASETS:-"math500 gsm8k"}
export VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-1}

mkdir -p "$ROOT/base_eval"

for DATASET in $DATASETS; do
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $PYTHON scripts/generate_rollouts.py \
    --model_path "$MODEL_PATH" \
    --condition_name "qwen3_4b_base_${DATASET}" \
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
    --output_dir "$ROOT/base_eval" \
    --output_name "qwen3_4b_base_${DATASET}.json" \
    --force
done
