#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
MODEL_PATH=${MODEL_PATH:-/nfs/FM/gongoubo/checkpoints/Qwen/Qwen3-4B}
ROOT=${ROOT:-outputs/qwen3_4b_default}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-4}
SCORE_DTYPE=${SCORE_DTYPE:-bfloat16}
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-1}

mkdir -p "$ROOT/gen" "$ROOT/score"
export VLLM_ENFORCE_EAGER

if [ -d "${HOME}/.cache/vllm/torch_compile_cache" ]; then
  rm -rf "${HOME}/.cache/vllm/torch_compile_cache"
fi

for SHARD in 0 1 2; do
  CUDA_VISIBLE_DEVICES=${SHARD} $PYTHON scripts/generate_rollouts.py \
    --model_path "$MODEL_PATH" \
    --condition_name "qwen3_4b_shard${SHARD}_n20" \
    --dataset math500 \
    --records_file "$ROOT/sampled_records/records_l345_300_shard${SHARD}.json" \
    --problem_set full \
    --prompt_style auto \
    --stop_profile auto \
    --qwen3_enable_thinking false \
    --num_generations 20 \
    --batch_size "$GEN_BATCH_SIZE" \
    --tensor_parallel_size 1 \
    --temperature 0.6 \
    --top_p 0.95 \
    --max_tokens 8192 \
    --max_model_len 8192 \
    --seed 42 \
    --output_dir "$ROOT/gen" \
    --output_name "qwen3_4b_shard${SHARD}_n20.json" \
    --force &
done
wait

for SHARD in 0 1 2; do
  CUDA_VISIBLE_DEVICES=${SHARD} $PYTHON scripts/score_rollouts.py \
    --input_json "$ROOT/gen/qwen3_4b_shard${SHARD}_n20.json" \
    --output_file "$ROOT/score/qwen3_4b_shard${SHARD}_n20_entropy.json" \
    --model_path "$MODEL_PATH" \
    --dataset math500 \
    --prompt_style auto \
    --qwen3_enable_thinking false \
    --dtype "$SCORE_DTYPE" \
    --hf_attn_implementation sdpa \
    --max_seq_len 8192 \
    --n_gens 20 \
    --force &
done
wait
