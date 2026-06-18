from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import paths
from ..run_context import RunContext, run_stage_command


def yolo_review_dir(context: RunContext) -> Path:
    return context.outputs_dir / "review" / "yolo_errors"


def config_path(value: Any) -> Path | None:
    if value is None or str(value) == "":
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (paths.PROJECT_ROOT / path).resolve()


def build_yolo_review_command(context: RunContext) -> list[str]:
    config = context.config.yolo_review
    model_path = config_path(config.get("model_path"))
    data_yaml = config_path(config.get("data_yaml"))
    if model_path is None:
        raise ValueError("yolo_review.model_path is required when yolo_review is enabled")
    if data_yaml is None:
        raise ValueError("yolo_review.data_yaml is required when yolo_review is enabled")

    command = [
        str(context.config.python),
        "-m",
        "real_image_process.FPK_PJ_fullflow.review.cli",
        "yolo-errors",
        "--model",
        str(model_path),
        "--data",
        str(data_yaml),
        "--output-root",
        str(yolo_review_dir(context)),
        "--split",
        str(config.get("split", "val")),
        "--conf",
        str(config.get("conf", 0.25)),
        "--iou",
        str(config.get("iou", 0.5)),
        "--imgsz",
        str(config.get("imgsz", 1280)),
        "--device",
        str(config.get("device", "0")),
    ]
    max_images = int(config.get("max_images", 0) or 0)
    if max_images > 0:
        command.extend(["--max-images", str(max_images)])
    return command


def run_yolo_review(context: RunContext, *, dry_run: bool = False) -> dict[str, Any]:
    if not bool(context.config.yolo_review.get("enabled", False)):
        payload = {"status": "skipped", "reason": "yolo_review.enabled is false"}
        context.update_status("yolo_review", payload)
        return payload

    expected = {
        "gallery_dir": str(yolo_review_dir(context)),
        "index": str(yolo_review_dir(context) / "index.html"),
        "summary": str(yolo_review_dir(context) / "summary.json"),
        "all_cases": str(yolo_review_dir(context) / "all" / "cases.jsonl"),
    }
    return run_stage_command(
        context,
        "yolo_review",
        build_yolo_review_command(context),
        expected_outputs=expected,
        dry_run=dry_run,
    )
