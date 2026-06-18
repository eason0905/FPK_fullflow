#!/usr/bin/env bash
set -euo pipefail

TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$TRAIN_DIR/../../../.." && pwd)"
DATASET_ROOT="$PROJECT_ROOT/real_image_process/FPK_PJ_fullflow/assets/datasets"

find "$DATASET_ROOT/dataset_json/v4/task345_overlay_images" -type f -size 0 -delete 2>/dev/null || true

bash "$TRAIN_DIR/run_export_dataset.sh"
bash "$TRAIN_DIR/run_split_dataset.sh"
bash "$TRAIN_DIR/run_make_dataset_info.sh"
