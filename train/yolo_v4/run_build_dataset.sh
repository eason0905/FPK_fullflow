#!/usr/bin/env bash
set -euo pipefail

TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FULLFLOW_ROOT="$(cd "$TRAIN_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python3}"

"$PYTHON" "$TRAIN_DIR/scripts/build_dataset.py" \
  --input-root "$FULLFLOW_ROOT/assets/datasets/dataset_full_v4" \
  --output-root "$FULLFLOW_ROOT/assets/datasets/yolo/v4_seed42" \
  --dataset-yaml "$FULLFLOW_ROOT/assets/datasets/yolo/v4_seed42/dataset.yaml" \
  "$@"
