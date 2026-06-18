from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import paths
from ..run_context import FullflowConfig, fail_if_missing, summarize_dataset, write_json


def resolve_project_path(value: Any) -> Path | None:
    if value is None or str(value) == "":
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (paths.PROJECT_ROOT / path).resolve()


def check_inputs(config: FullflowConfig, output_path: Path | None = None, require_assets: bool = True) -> dict[str, Any]:
    dataset_root = config.asset_dataset_root if require_assets else config.source_dataset_root
    checks = [
        ("dataset_root", dataset_root),
        ("eval_dataset_dir", config.eval_dataset_dir),
        ("known_issues_path", config.known_issues_path),
        ("table_review_log_path", config.table_review_log_path),
    ]
    if require_assets:
        checks.extend(
            [
                ("asset_model_path", config.asset_model_path),
                ("asset_adapter_path", config.asset_adapter_path),
            ]
        )
    else:
        checks.extend(
            [
                ("source_model_path", config.source_model_path),
                ("source_adapter_path", config.source_adapter_path),
            ]
        )
    yolo_review = dict(config.yolo_review or {})
    yolo_model_path = resolve_project_path(yolo_review.get("model_path"))
    yolo_data_yaml = resolve_project_path(yolo_review.get("data_yaml"))
    if bool(yolo_review.get("enabled", False)):
        checks.extend(
            [
                ("yolo_review.model_path", yolo_model_path),
                ("yolo_review.data_yaml", yolo_data_yaml),
            ]
        )
    missing = fail_if_missing(checks)
    summary = {
        "status": "success" if not missing else "failed",
        "require_assets": require_assets,
        "missing": missing,
        "dataset": summarize_dataset(dataset_root),
        "model_path": str(config.asset_model_path if require_assets else config.source_model_path),
        "adapter_path": str(config.asset_adapter_path if require_assets else config.source_adapter_path),
        "eval_dataset_dir": str(config.eval_dataset_dir),
        "known_issues_path": str(config.known_issues_path),
        "table_review_log_path": str(config.table_review_log_path),
        "yolo_review": {
            "enabled": bool(yolo_review.get("enabled", False)),
            "model_path": str(yolo_model_path) if yolo_model_path is not None else "",
            "data_yaml": str(yolo_data_yaml) if yolo_data_yaml is not None else "",
        },
    }
    if output_path is not None:
        write_json(output_path, summary)
    if missing:
        raise FileNotFoundError("Missing fullflow inputs: " + "; ".join(missing))
    return summary
