#!/usr/bin/env bash
set -euo pipefail

TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$TRAIN_DIR/../../../.." && pwd)"
DATASET_ROOT="$PROJECT_ROOT/real_image_process/FPK_PJ_fullflow/assets/datasets"
SCRIPT_DIR="$TRAIN_DIR/scripts/fintuning_dataset_generate"
SEED="${SEED:-42}"
VAL_SIZE="${VAL_SIZE:-0.05}"

python3 "$SCRIPT_DIR/split_real_v1_datasets.py" \
  --dataset-dir "$DATASET_ROOT/dataset_json/v4" \
  --output-root "$DATASET_ROOT/dataset_json/splits/real_v4_seed42" \
  --val-size "$VAL_SIZE" \
  --seed "$SEED" \
  "$@"
