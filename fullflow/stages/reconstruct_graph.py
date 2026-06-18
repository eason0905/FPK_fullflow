from __future__ import annotations

from pathlib import Path

from .. import paths
from ..run_context import RunContext, run_stage_command


def reconstruction_run_dir(context: RunContext) -> Path:
    return context.outputs_dir / "reconstruction" / context.run_id


def visualization_run_dir(context: RunContext) -> Path:
    return context.outputs_dir / "visualization" / f"reconstruction_{context.run_id}"


def build_reconstruct_command(context: RunContext, *, limit: int = 0, layout: str = "split_vertical") -> list[str]:
    command = [
        str(context.config.python),
        str(paths.REAL_IMAGE_ROOT / "package_graph" / "cli" / "run_reconstruction_and_visualize.py"),
        "--input",
        str(context.config.asset_dataset_root),
        "--timestamp",
        context.run_id,
        "--recon-root",
        str(context.outputs_dir / "reconstruction"),
        "--viz-root",
        str(context.outputs_dir / "visualization"),
        "--layout",
        layout,
    ]
    if limit > 0:
        command.extend(["--limit", str(limit)])
    return command


def run_reconstruct(context: RunContext, *, limit: int = 0, layout: str = "split_vertical", dry_run: bool = False) -> dict:
    command = build_reconstruct_command(context, limit=limit, layout=layout)
    expected = {
        "reconstruction_dir": str(reconstruction_run_dir(context)),
        "summary": str(reconstruction_run_dir(context) / "summary.json"),
        "graphs": str(reconstruction_run_dir(context) / "graphs"),
        "visualization_dir": str(visualization_run_dir(context)),
    }
    return run_stage_command(context, "reconstruction", command, expected_outputs=expected, dry_run=dry_run)

