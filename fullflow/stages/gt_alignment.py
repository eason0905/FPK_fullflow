from __future__ import annotations

import json
from pathlib import Path

from .. import paths
from ..run_context import RunContext, run_stage_command
from .multiview_integrate import multiview_run_dir


def gt_alignment_run_dir(context: RunContext) -> Path:
    return context.outputs_dir / "eval" / "gt_alignment"


def gt_alignment_review_dir(context: RunContext) -> Path:
    return context.outputs_dir / "review" / "gt_alignment"


def final_comparison_review_dir(context: RunContext) -> Path:
    return context.outputs_dir / "review" / "final_comparison"


def build_gt_alignment_command(context: RunContext, *, limit: int = 0) -> list[str]:
    command = [
        str(context.config.python),
        "-m",
        "real_image_process.FPK_PJ_fullflow.gt_alignment.cli",
        "--dataset-root",
        str(context.config.asset_dataset_root),
        "--multiview-root",
        str(multiview_run_dir(context)),
        "--output-root",
        str(gt_alignment_run_dir(context)),
        "--config-json",
        json.dumps(context.config.gt_alignment, ensure_ascii=False),
    ]
    if limit > 0:
        command.extend(["--limit", str(limit)])
    return command


def build_gt_alignment_review_command(context: RunContext) -> list[str]:
    return [
        str(context.config.python),
        "-m",
        "real_image_process.FPK_PJ_fullflow.review.cli",
        "gt-alignment",
        "--alignment-root",
        str(gt_alignment_run_dir(context)),
        "--output-root",
        str(gt_alignment_review_dir(context)),
        "--fullflow-root",
        str(paths.FULLFLOW_ROOT),
        "--run-id",
        context.run_id,
    ]


def build_final_comparison_review_command(context: RunContext) -> list[str]:
    return [
        str(context.config.python),
        "-m",
        "real_image_process.FPK_PJ_fullflow.review.cli",
        "final-comparison",
        "--alignment-root",
        str(gt_alignment_run_dir(context)),
        "--output-root",
        str(final_comparison_review_dir(context)),
        "--fullflow-root",
        str(paths.FULLFLOW_ROOT),
        "--run-id",
        context.run_id,
    ]


def run_gt_alignment(context: RunContext, *, limit: int = 0, dry_run: bool = False) -> dict:
    if not bool(context.config.gt_alignment.get("enabled", True)):
        payload = {"status": "skipped", "reason": "gt_alignment.enabled is false"}
        context.update_status("gt_alignment", payload)
        return {"eval": payload, "review": payload}

    eval_expected = {
        "alignment_dir": str(gt_alignment_run_dir(context)),
        "summary": str(gt_alignment_run_dir(context) / "summary.json"),
        "mismatches": str(gt_alignment_run_dir(context) / "mismatches.jsonl"),
    }
    evaluation = run_stage_command(
        context,
        "gt_alignment",
        build_gt_alignment_command(context, limit=limit),
        expected_outputs=eval_expected,
        dry_run=dry_run,
    )
    review_expected = {
        "gallery_dir": str(gt_alignment_review_dir(context)),
        "index": str(gt_alignment_review_dir(context) / "index.html"),
        "notes": str(gt_alignment_review_dir(context) / "data" / "notes.json"),
        "cases": str(gt_alignment_review_dir(context) / "data" / "cases.json"),
    }
    review = run_stage_command(
        context,
        "gt_alignment_review",
        build_gt_alignment_review_command(context),
        expected_outputs=review_expected,
        dry_run=dry_run,
    )
    final_comparison_expected = {
        "gallery_dir": str(final_comparison_review_dir(context)),
        "index": str(final_comparison_review_dir(context) / "index.html"),
        "notes": str(final_comparison_review_dir(context) / "data" / "notes.json"),
        "cases": str(final_comparison_review_dir(context) / "data" / "cases.json"),
    }
    final_comparison = run_stage_command(
        context,
        "final_comparison_review",
        build_final_comparison_review_command(context),
        expected_outputs=final_comparison_expected,
        dry_run=dry_run,
    )
    return {"eval": evaluation, "review": review, "final_comparison": final_comparison}
