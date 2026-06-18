from __future__ import annotations

from pathlib import Path

from .. import paths
from ..run_context import RunContext, run_stage_command
from .reconstruct_graph import reconstruction_run_dir


def table_lookup_gallery_dir(context: RunContext) -> Path:
    return context.outputs_dir / "diagnosis" / "table_lookup_missing"


def table_lookup_diagnosis_dir(context: RunContext) -> Path:
    return context.outputs_dir / "diagnosis" / "table_lookup_reasons"


def build_table_lookup_gallery_command(context: RunContext, *, include_no_table: bool = True, limit: int = 0) -> list[str]:
    command = [
        str(context.config.python),
        str(paths.REAL_IMAGE_ROOT / "package_graph" / "cli" / "make_table_lookup_missing_gallery.py"),
        "--input",
        str(context.config.asset_dataset_root),
        "--graph-input",
        str(reconstruction_run_dir(context) / "graphs"),
        "--output-root",
        str(context.outputs_dir / "diagnosis"),
        "--timestamp",
        "table_lookup_missing",
    ]
    for view in context.config.include_views:
        command.extend(["--include-view", view])
    if include_no_table:
        command.append("--include-no-table")
    if limit > 0:
        command.extend(["--limit", str(limit)])
    return command


def build_table_lookup_diagnosis_command(context: RunContext) -> list[str]:
    return [
        str(context.config.python),
        str(paths.REAL_IMAGE_ROOT / "package_graph" / "cli" / "diagnose_table_lookup_missing.py"),
        "--input",
        str(table_lookup_gallery_dir(context) / "table_lookup_missing.jsonl"),
        "--output-root",
        str(context.outputs_dir / "diagnosis"),
        "--timestamp",
        "table_lookup_reasons",
    ]


def run_table_lookup_gallery(
    context: RunContext,
    *,
    include_no_table: bool = True,
    limit: int = 0,
    dry_run: bool = False,
) -> dict:
    command = build_table_lookup_gallery_command(context, include_no_table=include_no_table, limit=limit)
    expected = {
        "gallery_dir": str(table_lookup_gallery_dir(context)),
        "index": str(table_lookup_gallery_dir(context) / "index.html"),
        "jsonl": str(table_lookup_gallery_dir(context) / "table_lookup_missing.jsonl"),
        "csv": str(table_lookup_gallery_dir(context) / "table_lookup_missing.csv"),
    }
    return run_stage_command(context, "diagnose_table_lookup_gallery", command, expected_outputs=expected, dry_run=dry_run)


def run_table_lookup_diagnosis(context: RunContext, *, dry_run: bool = False) -> dict:
    command = build_table_lookup_diagnosis_command(context)
    expected = {
        "diagnosis_dir": str(table_lookup_diagnosis_dir(context)),
        "index": str(table_lookup_diagnosis_dir(context) / "index.html"),
        "jsonl": str(table_lookup_diagnosis_dir(context) / "table_lookup_diagnosis.jsonl"),
        "csv": str(table_lookup_diagnosis_dir(context) / "table_lookup_diagnosis.csv"),
    }
    return run_stage_command(context, "diagnose_table_lookup_reasons", command, expected_outputs=expected, dry_run=dry_run)
