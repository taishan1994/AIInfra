#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
ROOT=${ROOT:-outputs/qwen3_4b_default}

$PYTHON scripts/make_holdout_split.py \
  --sampled_ids_file "$ROOT/sampled_records/ids_l345_300.json" \
  --selected_target_ids_file "$ROOT/train/target_ids_tau1p4.json" \
  --holdout_size 60 \
  --seed 42 \
  --output_file "$ROOT/train/holdout60_ids_tau1p4.json"
