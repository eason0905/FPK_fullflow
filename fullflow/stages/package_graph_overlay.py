from __future__ import annotations

from pathlib import Path

from ..run_context import RunContext, run_stage_command
from .multiview_integrate import multiview_run_dir
from .reconstruct_graph import reconstruction_run_dir


def package_graph_overlay_run_dir(context: RunContext) -> Path:
    return context.outputs_dir / "review" / "package_graph_overlay_gallery"


def build_package_graph_overlay_command(context: RunContext, *, limit: int = 0) -> list[str]:
    command = [
        str(context.config.python),
        "-m",
        "real_image_process.FPK_PJ_fullflow.review.package_graph_overlay",
        "--graph-root",
        str(reconstruction_run_dir(context) / "graphs"),
        "--output-dir",
        str(package_graph_overlay_run_dir(context)),
        "--dataset-root",
        str(context.config.asset_dataset_root),
        "--multiview-root",
        str(multiview_run_dir(context)),
    ]
    if limit > 0:
        command.extend(["--limit", str(limit)])
    return command


def run_package_graph_overlay(context: RunContext, *, limit: int = 0, dry_run: bool = False) -> dict:
    expected = {
        "gallery_dir": str(package_graph_overlay_run_dir(context)),
        "index": str(package_graph_overlay_run_dir(context) / "index.html"),
        "summary": str(package_graph_overlay_run_dir(context) / "gallery_summary.json"),
    }
    return run_stage_command(
        context,
        "package_graph_overlay_gallery",
        build_package_graph_overlay_command(context, limit=limit),
        expected_outputs=expected,
        dry_run=dry_run,
    )
