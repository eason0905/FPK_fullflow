from __future__ import annotations

from pathlib import Path

from .. import paths
from ..run_context import RunContext, run_stage_command
from .reconstruct_graph import reconstruction_run_dir, visualization_run_dir


def gallery_run_dir(context: RunContext) -> Path:
    return context.outputs_dir / "review" / "top_bottom_land_filtered"


def package_graph_review_dir(context: RunContext) -> Path:
    return context.outputs_dir / "review" / "package_graph"


def all_view_gallery_run_dir(context: RunContext) -> Path:
    return context.outputs_dir / "review" / "package_graph_all_views_source"


def package_graph_all_views_review_dir(context: RunContext) -> Path:
    return context.outputs_dir / "review" / "package_graph_all_views"


def build_gallery_command(context: RunContext, *, max_items: int = 0, limit: int = 0) -> list[str]:
    return build_legacy_gallery_command(
        context,
        timestamp="top_bottom_land_filtered",
        include_config_views=True,
        max_items=max_items,
        limit=limit,
    )


def build_all_view_gallery_command(context: RunContext, *, max_items: int = 0, limit: int = 0) -> list[str]:
    return build_legacy_gallery_command(
        context,
        timestamp="package_graph_all_views_source",
        include_config_views=False,
        max_items=max_items,
        limit=limit,
    )


def build_legacy_gallery_command(
    context: RunContext,
    *,
    timestamp: str,
    include_config_views: bool,
    max_items: int = 0,
    limit: int = 0,
) -> list[str]:
    config = context.config
    command = [
        str(config.python),
        str(paths.REAL_IMAGE_ROOT / "package_graph" / "cli" / "make_review_gallery.py"),
        "--input",
        str(reconstruction_run_dir(context)),
        "--visualization-root",
        str(visualization_run_dir(context)),
        "--output-root",
        str(context.outputs_dir / "review"),
        "--timestamp",
        timestamp,
        "--max-items",
        str(max_items),
    ]
    if include_config_views:
        for view in config.include_views:
            command.extend(["--include-view", view])
    for value_source in config.exclude_value_sources:
        command.extend(["--exclude-value-source", value_source])
    if config.known_issues_path.exists():
        command.extend(["--exclude-known-issues", str(config.known_issues_path)])
    if limit > 0:
        command.extend(["--limit", str(limit)])
    return command


def build_package_graph_review_command(context: RunContext, *, pages: int = 5) -> list[str]:
    return build_package_graph_review_from_risk_report_command(
        context,
        risk_report_path=gallery_run_dir(context) / "risk_report.jsonl",
        output_root=package_graph_review_dir(context),
        pages=pages,
    )


def build_package_graph_all_views_review_command(context: RunContext, *, pages: int = 5) -> list[str]:
    return build_package_graph_review_from_risk_report_command(
        context,
        risk_report_path=all_view_gallery_run_dir(context) / "risk_report.jsonl",
        output_root=package_graph_all_views_review_dir(context),
        pages=pages,
    )


def build_package_graph_review_from_risk_report_command(
    context: RunContext,
    *,
    risk_report_path: Path,
    output_root: Path,
    pages: int = 5,
) -> list[str]:
    return [
        str(context.config.python),
        "-m",
        "real_image_process.FPK_PJ_fullflow.review.cli",
        "package-graph",
        "--risk-report",
        str(risk_report_path),
        "--output-root",
        str(output_root),
        "--fullflow-root",
        str(paths.FULLFLOW_ROOT),
        "--run-id",
        context.run_id,
        "--split-by",
        "view",
    ]


def run_gallery(context: RunContext, *, max_items: int = 0, limit: int = 0, dry_run: bool = False) -> dict:
    command = build_gallery_command(context, max_items=max_items, limit=limit)
    expected = {
        "gallery_dir": str(gallery_run_dir(context)),
        "index": str(gallery_run_dir(context) / "index.html"),
        "risk_report": str(gallery_run_dir(context) / "risk_report.jsonl"),
    }
    legacy_gallery = run_stage_command(context, "gallery", command, expected_outputs=expected, dry_run=dry_run)

    review_expected = {
        "gallery_dir": str(package_graph_review_dir(context)),
        "index": str(package_graph_review_dir(context) / "index.html"),
        "notes": str(package_graph_review_dir(context) / "data" / "notes.json"),
        "cases": str(package_graph_review_dir(context) / "data" / "cases.json"),
    }
    package_graph_review = run_stage_command(
        context,
        "package_graph_review",
        build_package_graph_review_command(context),
        expected_outputs=review_expected,
        dry_run=dry_run,
    )

    all_view_command = build_all_view_gallery_command(context, max_items=max_items, limit=limit)
    all_view_expected = {
        "gallery_dir": str(all_view_gallery_run_dir(context)),
        "index": str(all_view_gallery_run_dir(context) / "index.html"),
        "risk_report": str(all_view_gallery_run_dir(context) / "risk_report.jsonl"),
    }
    all_view_gallery = run_stage_command(
        context,
        "package_graph_all_views_source",
        all_view_command,
        expected_outputs=all_view_expected,
        dry_run=dry_run,
    )

    all_view_review_expected = {
        "gallery_dir": str(package_graph_all_views_review_dir(context)),
        "index": str(package_graph_all_views_review_dir(context) / "index.html"),
        "notes": str(package_graph_all_views_review_dir(context) / "data" / "notes.json"),
        "cases": str(package_graph_all_views_review_dir(context) / "data" / "cases.json"),
    }
    package_graph_all_views_review = run_stage_command(
        context,
        "package_graph_all_views_review",
        build_package_graph_all_views_review_command(context),
        expected_outputs=all_view_review_expected,
        dry_run=dry_run,
    )
    return {
        "legacy_gallery": legacy_gallery,
        "package_graph_review": package_graph_review,
        "package_graph_all_views_source": all_view_gallery,
        "package_graph_all_views_review": package_graph_all_views_review,
    }
