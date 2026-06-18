from __future__ import annotations

from pathlib import Path
from typing import Any

from ..run_context import RunContext, run_stage_command
from .package_graph_overlay import package_graph_overlay_run_dir
from .reconstruct_graph import reconstruction_run_dir


def score_merge_gt_run_dir(context: RunContext) -> Path:
    return context.outputs_dir / "scoring" / "merge_gt_metrics"


def score_merge_gt_review_dir(context: RunContext) -> Path:
    return context.outputs_dir / "review" / "merge_gt_score_gallery"


def build_score_merge_gt_command(context: RunContext) -> list[str]:
    scoring_config: dict[str, Any] = dict(context.config.scoring or {})
    weights = dict(scoring_config.get("weights") or {})
    command = [
        str(context.config.python),
        "-m",
        "real_image_process.FPK_PJ_fullflow.scoring.merge_gt_metrics",
        "--merge-root",
        str(package_graph_overlay_run_dir(context)),
        "--dataset-root",
        str(context.config.asset_dataset_root),
        "--output-dir",
        str(score_merge_gt_run_dir(context)),
        "--review-dir",
        str(score_merge_gt_review_dir(context)),
        "--table-missing-graph-root",
        str(reconstruction_run_dir(context) / "graphs"),
        "--weight-iou-ic",
        str(float(weights.get("iou_ic", 0.25))),
        "--weight-pin-count",
        str(float(weights.get("pin_count", 0.25))),
        "--weight-d-pin",
        str(float(weights.get("d_pin", 0.25))),
        "--weight-iou-pin",
        str(float(weights.get("iou_pin", 0.25))),
    ]
    return command


def run_score_merge_gt(context: RunContext, *, dry_run: bool = False) -> dict[str, Any]:
    expected = {
        "scoring_dir": str(score_merge_gt_run_dir(context)),
        "summary": str(score_merge_gt_run_dir(context) / "summary.json"),
        "records": str(score_merge_gt_run_dir(context) / "records.jsonl"),
        "gallery_dir": str(score_merge_gt_review_dir(context)),
        "index": str(score_merge_gt_review_dir(context) / "index.html"),
    }
    return run_stage_command(
        context,
        "score_merge_gt",
        build_score_merge_gt_command(context),
        expected_outputs=expected,
        dry_run=dry_run,
    )

