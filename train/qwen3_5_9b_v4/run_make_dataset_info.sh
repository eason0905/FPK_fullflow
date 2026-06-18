#!/usr/bin/env bash
set -euo pipefail

TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$TRAIN_DIR/../../../.." && pwd)"
DATASET_ROOT="$PROJECT_ROOT/real_image_process/FPK_PJ_fullflow/assets/datasets"

python3 "$TRAIN_DIR/make_dataset_info.py" \
  --output "$DATASET_ROOT/dataset_json/splits/real_v4_seed42/dataset_info.json" \
  --prefix "real_v4_seed42"
