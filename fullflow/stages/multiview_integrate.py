from __future__ import annotations

import json
from pathlib import Path

from .. import paths
from ..run_context import RunContext, run_stage_command
from .reconstruct_graph import reconstruction_run_dir


def multiview_run_dir(context: RunContext) -> Path:
    return context.outputs_dir / "multiview"


def multiview_review_dir(context: RunContext) -> Path:
    return context.outputs_dir / "review" / "multiview"


def build_multiview_integrate_command(context: RunContext, *, limit: int = 0) -> list[str]:
    command = [
        str(context.config.python),
        "-m",
        "real_image_process.FPK_PJ_fullflow.multiview.cli",
        "integrate",
        "--graph-input",
        str(reconstruction_run_dir(context)),
        "--dataset-root",
        str(context.config.asset_dataset_root),
        "--output-root",
        str(multiview_run_dir(context)),
        "--config-json",
        json.dumps(context.config.multiview, ensure_ascii=False),
    ]
    if limit > 0:
        command.extend(["--limit", str(limit)])
    return command


def build_multiview_review_command(context: RunContext) -> list[str]:
    return [
        str(context.config.python),
        "-m",
        "real_image_process.FPK_PJ_fullflow.review.cli",
        "multiview",
        "--multiview-root",
        str(multiview_run_dir(context)),
        "--output-root",
        str(multiview_review_dir(context)),
        "--fullflow-root",
        str(paths.FULLFLOW_ROOT),
        "--run-id",
        context.run_id,
    ]


def run_multiview(context: RunContext, *, limit: int = 0, dry_run: bool = False) -> dict:
    if not bool(context.config.multiview.get("enabled", True)):
        payload = {"status": "skipped", "reason": "multiview.enabled is false"}
        context.update_status("multiview_integrate", payload)
        return {"integrate": payload, "review": payload}

    integrate_expected = {
        "multiview_dir": str(multiview_run_dir(context)),
        "summary": str(multiview_run_dir(context) / "summary.json"),
        "parts": str(multiview_run_dir(context) / "parts"),
    }
    integrate = run_stage_command(
        context,
        "multiview_integrate",
        build_multiview_integrate_command(context, limit=limit),
        expected_outputs=integrate_expected,
        dry_run=dry_run,
    )
    review_expected = {
        "gallery_dir": str(multiview_review_dir(context)),
        "index": str(multiview_review_dir(context) / "index.html"),
        "notes": str(multiview_review_dir(context) / "data" / "notes.json"),
        "cases": str(multiview_review_dir(context) / "data" / "cases.json"),
    }
    review = run_stage_command(
        context,
        "multiview_review",
        build_multiview_review_command(context),
        expected_outputs=review_expected,
        dry_run=dry_run,
    )
    return {"integrate": integrate, "review": review}
