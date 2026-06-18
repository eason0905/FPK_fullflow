#!/usr/bin/env bash
set -euo pipefail

TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$TRAIN_DIR/../../../.." && pwd)"
CONFIG="$TRAIN_DIR/configs/qwen3_5_9b_sft_split_v4_px2400_cutoff8192.yaml"
DEFAULT_CLI="/home/114/pohua1010/miniconda3/envs/llamafactory/bin/llamafactory-cli"
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-$DEFAULT_CLI}"
if [[ ! -x "$LLAMAFACTORY_CLI" ]]; then
  LLAMAFACTORY_CLI="$(command -v llamafactory-cli)"
fi

PYTHONPATH="$PROJECT_ROOT/LlamaFactory/src" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$LLAMAFACTORY_CLI" train "$CONFIG" "$@"
