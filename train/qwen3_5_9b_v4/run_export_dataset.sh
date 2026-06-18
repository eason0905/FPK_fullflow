#!/usr/bin/env bash
set -euo pipefail

TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$TRAIN_DIR/../../../.." && pwd)"
DATASET_ROOT="$PROJECT_ROOT/real_image_process/FPK_PJ_fullflow/assets/datasets"
SCRIPT_DIR="$TRAIN_DIR/scripts/fintuning_dataset_generate"

export FPK_DATASET_ROOT="$DATASET_ROOT"

PYTHONPATH="$SCRIPT_DIR" python3 "$SCRIPT_DIR/export_dataset.py" \
  --input "$DATASET_ROOT/dataset_full_v4" \
  --output-dir "$DATASET_ROOT/dataset_json/v4" \
  "$@"
