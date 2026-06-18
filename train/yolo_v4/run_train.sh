#!/usr/bin/env bash
set -euo pipefail

TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FULLFLOW_ROOT="$(cd "$TRAIN_DIR/../.." && pwd)"
RUN_ID="${RUN_ID:-v4_detector_$(date +%Y%m%d_%H%M%S)}"
PYTHON="${PYTHON:-python3}"

"$PYTHON" "$TRAIN_DIR/scripts/train_detector.py" \
  --data "$FULLFLOW_ROOT/assets/datasets/yolo/v4_seed42/dataset.yaml" \
  --model "$FULLFLOW_ROOT/assets/models/yolo/base/best.pt" \
  --project "$TRAIN_DIR/runs" \
  --name "$RUN_ID" \
  --export-dir "$FULLFLOW_ROOT/assets/models/yolo/v4" \
  "$@"
