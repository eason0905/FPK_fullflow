from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REAL_IMAGE_ROOT = PROJECT_ROOT / "real_image_process"
FULLFLOW_ROOT = REAL_IMAGE_ROOT / "FPK_PJ_fullflow"

CONFIG_ROOT = FULLFLOW_ROOT / "configs"
ASSET_ROOT = FULLFLOW_ROOT / "assets"
MANIFEST_ROOT = FULLFLOW_ROOT / "manifests"
RUNS_ROOT = FULLFLOW_ROOT / "runs"

DEFAULT_CONFIG_PATH = CONFIG_ROOT / "default_v3.json"
DEFAULT_SOURCE_DATASET_ROOT = REAL_IMAGE_ROOT / "dataset" / "dataset_full_v3"
DEFAULT_ASSET_DATASET_ROOT = ASSET_ROOT / "datasets" / "dataset_full_v3"
DEFAULT_SOURCE_MODEL_PATH = PROJECT_ROOT / "models" / "Qwen3-VL-8B-Instruct"
DEFAULT_ASSET_MODEL_PATH = ASSET_ROOT / "models" / "Qwen3-VL-8B-Instruct"
DEFAULT_SOURCE_ADAPTER_PATH = (
    PROJECT_ROOT
    / "saves"
    / "qwen3-vl-8b-instruct"
    / "lora"
    / "real-task12345-sft-split-20260516-v3-px2400-cutoff8192"
)
DEFAULT_ASSET_ADAPTER_PATH = ASSET_ROOT / "adapters" / "real-task12345-sft-split-20260516-v3-px2400-cutoff8192"

DEFAULT_EVAL_DATASET_DIR = REAL_IMAGE_ROOT / "dataset" / "dataset_json" / "splits" / "real_v3_seed42" / "val"
DEFAULT_KNOWN_ISSUES_SOURCE = REAL_IMAGE_ROOT / "package_graph" / "review_known_data_issues.jsonl"
DEFAULT_KNOWN_ISSUES_PATH = MANIFEST_ROOT / "known_data_issues.jsonl"
DEFAULT_TABLE_REVIEW_LOG_SOURCE = REAL_IMAGE_ROOT / "package_graph" / "table_lookup_review_log.jsonl"
DEFAULT_TABLE_REVIEW_LOG_PATH = MANIFEST_ROOT / "table_lookup_review_log.jsonl"

