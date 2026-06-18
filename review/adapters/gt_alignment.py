from __future__ import annotations

import html
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ..common.gallery_renderer import page_relative_url, render_card, render_review_index
from ..common.gallery_renderer import page as render_page_shell
from ..common.schemas import ReviewItem, ReviewMedia
from ..schema import slugify
from ...gt_alignment.evaluator import alignment_review_quality_score
from ...multiview.integrator import (
    MultiviewOptions,
    extract_objects,
    extract_outline,
    regularize_two_column_package_pad_x_geometry,
)
from .package_graph import find_fullflow_root, infer_run_id, restore_notes_from_history, root_relative_url
from .package_graph import static_url_prefix


MAIN_POSTPROCESSED_VIEWS = ("top", "bottom", "land")
MAIN_POSTPROCESSED_VIEW_COLORS = {
    "top": "#2563eb",
    "bottom": "#16a34a",
    "land": "#f97316",
}
MULTIVIEW_OVERLAY_VIEW_COLORS = {
    "top": "#2563eb",
    "bottom": "#16a34a",
    "land": "#f97316",
    "front": "#e11d48",
    "side": "#0f766e",
    "lead": "#9333ea",
    "land_detail": "#7c3aed",
    "lateral": "#e11d48",
    "lead_detail": "#9333ea",
    "scan_result": "#64748b",
    "unknown": "#94a3b8",
}
DISPLAY_SVG_WIDTH = 1280.0
DISPLAY_SVG_HEIGHT = 920.0
DISPLAY_SVG_TARGET = (64.0, 96.0, 1216.0, 856.0)


def build_gt_alignment_review(
    *,
    alignment_root: Path,
    output_root: Path,
    fullflow_root: Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    alignment_root = alignment_root.resolve()
    output_root = output_root.resolve()
    fullflow_root = (fullflow_root or find_fullflow_root(output_root)).resolve()
    run_id = run_id or infer_run_id(output_root)

    rows = read_mismatch_rows(alignment_root / "mismatches.jsonl")
    items = [
        item_from_row(row, alignment_root=alignment_root, fullflow_root=fullflow_root, case_prefix="gt_alignment")
        for row in rows
    ]
    return write_alignment_review(
        alignment_root=alignment_root,
        output_root=output_root,
        fullflow_root=fullflow_root,
        run_id=run_id,
        items=items,
        title="GT Alignment Review",
        description="Review mismatches between ScanResultFormat GT/reference and unified multiview layers.",
        gallery_id="gt_alignment",
        include_risk_pages=False,
        include_status_pages=False,
    )


def build_final_comparison_review(
    *,
    alignment_root: Path,
    output_root: Path,
    fullflow_root: Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    alignment_root = alignment_root.resolve()
    output_root = output_root.resolve()
    fullflow_root = (fullflow_root or find_fullflow_root(output_root)).resolve()
    run_id = run_id or infer_run_id(output_root)

    rows = read_summary_rows(alignment_root / "summary.json")
    items = [
        item_from_row(row, alignment_root=alignment_root, fullflow_root=fullflow_root, case_prefix="final_comparison")
        for row in rows
    ]
    return write_alignment_review(
        alignment_root=alignment_root,
        output_root=output_root,
        fullflow_root=fullflow_root,
        run_id=run_id,
        items=items,
        title="Final Comparison Review",
        description=(
            "Review every part with source views, ScanResultFormat GT, aligned multiview result, "
            "and GT/result comparison."
        ),
        gallery_id="final_comparison",
        include_risk_pages=True,
        include_status_pages=True,
    )


def write_alignment_review(
    *,
    alignment_root: Path,
    output_root: Path,
    fullflow_root: Path,
    run_id: str,
    items: list[ReviewItem],
    title: str,
    description: str,
    gallery_id: str,
    include_risk_pages: bool,
    include_status_pages: bool,
) -> dict[str, Any]:
    data_dir = output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    cases_path = data_dir / "cases.json"
    notes_path = data_dir / "notes.json"
    history_path = data_dir / "notes_history.jsonl"
    summary_path = data_dir / "summary.json"
    cases_path.write_text(json.dumps([item.to_dict() for item in items], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not notes_path.exists():
        notes_path.write_text(
            json.dumps({"gallery_id": gallery_id, "run_id": run_id, "updated_at": None, "items": {}}, indent=2)
            + "\n",
            encoding="utf-8",
        )
    history_path.touch(exist_ok=True)
    restore_notes_from_history(notes_path, history_path)
    update_notes_header(notes_path, gallery_id=gallery_id, run_id=run_id)
    clear_generated_page_dirs(output_root)

    notes_rel_path = root_relative_url(notes_path, fullflow_root)
    history_rel_path = root_relative_url(history_path, fullflow_root)
    static_prefix = static_url_prefix(fullflow_root)
    risk_pages = []
    status_pages = []
    if include_risk_pages:
        risk_pages = write_pages(
            output_root=output_root,
            subdir="by_risk",
            grouped=group_by_risk(items),
            notes_rel_path=notes_rel_path,
            history_rel_path=history_rel_path,
            run_id=run_id,
            gallery_id=gallery_id,
            title_prefix=title,
            static_prefix=static_prefix,
        )
    if include_status_pages:
        status_pages = write_pages(
            output_root=output_root,
            subdir="by_status",
            grouped=group_by_status(items),
            notes_rel_path=notes_rel_path,
            history_rel_path=history_rel_path,
            run_id=run_id,
            gallery_id=gallery_id,
            title_prefix=title,
            static_prefix=static_prefix,
        )
    reason_pages = write_pages(
        output_root=output_root,
        subdir="by_reason",
        grouped=group_by_reason(items),
        notes_rel_path=notes_rel_path,
        history_rel_path=history_rel_path,
        run_id=run_id,
        gallery_id=gallery_id,
        title_prefix=title,
        static_prefix=static_prefix,
    )
    source_pages = write_pages(
        output_root=output_root,
        subdir="by_source",
        grouped=group_by_source(items),
        notes_rel_path=notes_rel_path,
        history_rel_path=history_rel_path,
        run_id=run_id,
        gallery_id=gallery_id,
        title_prefix=title,
        static_prefix=static_prefix,
    )
    objective_source_pages = write_pages(
        output_root=output_root,
        subdir="by_objective_source",
        grouped=group_by_objective_source(items),
        notes_rel_path=notes_rel_path,
        history_rel_path=history_rel_path,
        run_id=run_id,
        gallery_id=gallery_id,
        title_prefix=title,
        static_prefix=static_prefix,
    )
    stage_pages = write_pages(
        output_root=output_root,
        subdir="by_stage_hint",
        grouped=group_by_stage_hint(items),
        notes_rel_path=notes_rel_path,
        history_rel_path=history_rel_path,
        run_id=run_id,
        gallery_id=gallery_id,
        title_prefix=title,
        static_prefix=static_prefix,
    )
    check_pages = write_pages(
        output_root=output_root,
        subdir="by_check",
        grouped=group_by_check(items),
        notes_rel_path=notes_rel_path,
        history_rel_path=history_rel_path,
        run_id=run_id,
        gallery_id=gallery_id,
        title_prefix=title,
        static_prefix=static_prefix,
    )
    review_bucket_pages = write_pages(
        output_root=output_root,
        subdir="by_review_bucket",
        grouped=group_by_review_bucket(items),
        notes_rel_path=notes_rel_path,
        history_rel_path=history_rel_path,
        run_id=run_id,
        gallery_id=gallery_id,
        title_prefix=title,
        static_prefix=static_prefix,
    )
    page_paths = (
        risk_pages
        + status_pages
        + reason_pages
        + source_pages
        + objective_source_pages
        + stage_pages
        + check_pages
        + review_bucket_pages
    )
    render_review_index(
        output_root=output_root,
        title=title,
        description=description,
        items=items,
        page_paths=page_paths,
        notes_rel_path=notes_rel_path,
        run_id=run_id,
        gallery_id=gallery_id,
        page_labels=[path.stem for path in page_paths],
        static_prefix=static_prefix,
    )

    summary = {
        "output_root": str(output_root),
        "index_path": str(output_root / "index.html"),
        "gallery_url": "/" + root_relative_url(output_root / "index.html", fullflow_root).lstrip("/"),
        "cases_path": str(cases_path),
        "notes_path": str(notes_path),
        "notes_history_path": str(history_path),
        "summary_path": str(summary_path),
        "alignment_root": str(alignment_root),
        "run_id": run_id,
        "total_items": len(items),
        "risk_pages": [str(path) for path in risk_pages],
        "status_pages": [str(path) for path in status_pages],
        "reason_pages": [str(path) for path in reason_pages],
        "source_pages": [str(path) for path in source_pages],
        "objective_source_pages": [str(path) for path in objective_source_pages],
        "stage_pages": [str(path) for path in stage_pages],
        "check_pages": [str(path) for path in check_pages],
        "review_bucket_pages": [str(path) for path in review_bucket_pages],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def clear_generated_page_dirs(output_root: Path) -> None:
    for subdir in (
        "by_risk",
        "by_status",
        "by_reason",
        "by_source",
        "by_objective_source",
        "by_stage_hint",
        "by_check",
        "by_review_bucket",
    ):
        path = output_root / subdir
        if path.exists():
            shutil.rmtree(path)


def update_notes_header(notes_path: Path, *, gallery_id: str, run_id: str) -> None:
    payload = json.loads(notes_path.read_text(encoding="utf-8"))
    if payload.get("gallery_id") == gallery_id and payload.get("run_id") == run_id:
        return
    payload["gallery_id"] = gallery_id
    payload["run_id"] = run_id
    notes_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_mismatch_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_summary_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("parts") or []
    return [row for row in rows if isinstance(row, dict)]


def item_from_row(
    row: dict[str, Any],
    *,
    alignment_root: Path,
    fullflow_root: Path,
    case_prefix: str,
) -> ReviewItem:
    part_number = str(row.get("part_number") or "")
    alignment_path = Path(str(row.get("alignment_path") or ""))
    if not alignment_path.exists():
        alignment_path = alignment_root / "parts" / slugify(part_number) / "alignment.json"
    canonical_path = Path(str(row.get("unified_multiview_layers_path") or ""))
    media = []
    source_image_evidence = source_image_evidence_from_canonical(canonical_path, fullflow_root)
    if not source_image_evidence:
        source_image_evidence = source_image_evidence_from_dataset_part(
            Path(str(row.get("dataset_part_dir") or "")),
            fullflow_root,
        )
    if not source_image_evidence:
        source_image_evidence = [source_image_unavailable_placeholder(part_number)]
    adopted_media_count = 0
    for source in source_image_evidence:
        if not is_adopted_source_overlay(source):
            continue
        raw_view = str(source.get("raw_view") or "unknown")
        media.append(
            ReviewMedia(
                label=f"Adopted graph - {raw_view}",
                path=source["path"],
                url=source["url"],
            )
        )
        adopted_media_count += 1
    if adopted_media_count == 0:
        for source in source_image_evidence:
            media.append(ReviewMedia(label=source["label"], path=source["path"], url=source["url"]))
    main_postprocessed_evidence = main_postprocessed_graph_evidence_from_canonical(canonical_path, fullflow_root)
    for graph in main_postprocessed_evidence:
        media.append(
            ReviewMedia(
                label=f"Source package graph - {graph['raw_view']}",
                path=graph["path"],
                url=graph["url"],
            )
        )
    display_context = display_scale_context_from_canonical(canonical_path)
    main_view_overlay = main_view_overlay_evidence_from_canonical(canonical_path, fullflow_root, display_context=display_context)
    if main_view_overlay:
        media.append(
            ReviewMedia(
                label="Overlay graph - top/bottom/land",
                path=main_view_overlay["path"],
                url=main_view_overlay["url"],
            )
        )
    multiview_overlay = multiview_overlay_evidence_from_canonical(
        canonical_path,
        fullflow_root,
        display_context=display_context,
    )
    if multiview_overlay:
        media.append(
            ReviewMedia(
                label="Overlay graph - multiview",
                path=multiview_overlay["path"],
                url=multiview_overlay["url"],
            )
        )
    alignment_graph_path = Path(str(row.get("alignment_graph_svg_path") or alignment_path.with_name("alignment_graph.svg")))
    if alignment_graph_path.exists():
        media.append(
            ReviewMedia(
                label="Dimension-scaled graph",
                path=str(alignment_graph_path),
                url=root_relative_url(alignment_graph_path, fullflow_root),
            )
        )
    scan_svg_path = Path(str(row.get("scan_result_svg_path") or alignment_path.with_name("scan_result.svg")))
    scan_display = scan_result_display_evidence_from_canonical(canonical_path, fullflow_root, display_context=display_context)
    if scan_display:
        media.append(ReviewMedia(label="GT reference", path=scan_display["path"], url=scan_display["url"]))
    elif scan_svg_path.exists():
        media.append(ReviewMedia(label="GT reference", path=str(scan_svg_path), url=root_relative_url(scan_svg_path, fullflow_root)))
    comparison_svg_path = Path(str(row.get("comparison_svg_path") or canonical_path.with_name("comparison.svg")))
    scan_path = Path(str(row.get("scan_result_path") or ""))
    final_graph_path = Path(str(row.get("final_graph_path") or canonical_path.with_name("final_graph.json")))
    graph_review_context = load_graph_review_context(canonical_path, final_graph_path)
    case_id = f"{case_prefix}:{slugify(part_number)}"
    reasons = [str(reason) for reason in row.get("reasons") or []]
    score_diagnostics = [str(reason) for reason in row.get("score_diagnostics") or []]
    review_reasons = reasons + score_diagnostics
    error_sources = sorted(
        {
            str(source)
            for source in (row.get("error_sources") or []) + (row.get("score_error_sources") or [])
        }
    )
    objective_error_sources = sorted(
        {
            str(source)
            for source in (row.get("objective_error_sources") or []) + (row.get("score_objective_error_sources") or [])
        }
    )
    stage_hints = sorted(
        {
            str(stage_hint)
            for stage_hint in (row.get("stage_hints") or []) + (row.get("score_stage_hints") or [])
        }
    )
    checks = row.get("checks") or []
    mismatch_checks = [check for check in checks if check.get("status") != "aligned"]
    alignment_scores = row.get("alignment_scores") or {}
    review_score = review_risk_score(alignment_scores, review_reasons)
    row_review_risk_level = str(row.get("review_risk_level") or "")
    if row_review_risk_level not in {"high", "medium", "low"}:
        row_review_risk_level = review_risk_level(alignment_scores, review_reasons)
    return ReviewItem(
        case_id=case_id,
        title=part_number,
        rank=0,
        part_number=part_number,
        file_name="alignment.json",
        view="gt_alignment",
        risk_score=review_score,
        risk_level=row_review_risk_level,
        risk_reasons=review_reasons or [str(row.get("status") or "mismatch")],
        media=media,
        links={
            "alignment": root_relative_url(alignment_path, fullflow_root),
            "scan_result": root_relative_url(scan_path, fullflow_root),
            "scan_result_svg": root_relative_url(scan_svg_path, fullflow_root),
            "canonical_gt_overlay": root_relative_url(comparison_svg_path, fullflow_root),
            "unified_multiview_layers": root_relative_url(canonical_path, fullflow_root),
            "final_graph": root_relative_url(final_graph_path, fullflow_root),
        },
        metrics={
            "status": row.get("status"),
            "alignment_scores": alignment_scores,
            "overall_score": alignment_scores.get("overall_score"),
            "selected_quality_score": alignment_scores.get("selected_quality_score") or alignment_scores.get("quality_score"),
            "review_quality_score": alignment_scores.get("review_quality_score"),
            "iou": {
                "outline": alignment_scores.get("outline_iou"),
                "land": alignment_scores.get("land_iou"),
                "lead": alignment_scores.get("lead_iou"),
            },
            "pad_count_match": {
                "land": alignment_scores.get("land_pad_count_match"),
                "lead": alignment_scores.get("lead_count_match"),
                "checks": alignment_scores.get("count_checks"),
            },
            "layout_score": alignment_scores.get("pad_layout_score"),
            "dimension_mismatch": {
                "count": alignment_scores.get("dimension_mismatch_count"),
                "dimension_count": alignment_scores.get("dimension_count"),
                "score": alignment_scores.get("dimension_value_score"),
            },
            "final_coordinate_system": graph_review_context.get("final_coordinate_system"),
            "alignment_transform": graph_review_context.get("alignment_transform"),
            "evidence_summary": graph_review_context.get("evidence_summary"),
            "conflicts": graph_review_context.get("conflicts"),
            "mismatch_checks": mismatch_checks,
            "score_diagnostics": score_diagnostics,
            "score_diagnostic_details": row.get("score_diagnostic_details") or [],
            "error_sources": error_sources,
            "objective_error_sources": objective_error_sources,
            "review_bucket": row.get("review_bucket"),
            "review_risk_level": row.get("review_risk_level"),
            "gt": row.get("gt"),
            "graph": row.get("graph"),
        },
        metadata={
            "checks": checks,
            "mismatch_checks": mismatch_checks,
            "score_diagnostics": score_diagnostics,
            "score_diagnostic_details": row.get("score_diagnostic_details") or [],
            "error_sources": error_sources,
            "objective_error_sources": objective_error_sources,
            "stage_hints": stage_hints,
            "review_bucket": row.get("review_bucket"),
            "review_risk_level": row.get("review_risk_level"),
            "source_image_evidence": source_image_evidence,
            "main_postprocessed_graph_evidence": main_postprocessed_evidence,
            "main_view_overlay": main_view_overlay,
            "evidence_refs": graph_review_context.get("evidence_refs"),
            "dataset_part_dir": row.get("dataset_part_dir"),
        },
    )


def load_graph_review_context(canonical_path: Path, final_graph_path: Path) -> dict[str, Any]:
    context: dict[str, Any] = {
        "final_coordinate_system": None,
        "alignment_transform": None,
        "evidence_summary": None,
        "evidence_refs": [],
        "conflicts": [],
    }
    canonical = read_json_or_empty(canonical_path)
    if canonical:
        context["evidence_summary"] = (canonical.get("summary") or {}).get("evidence_summary") or {}
        context["evidence_refs"] = canonical.get("evidence_refs") or []
        context["conflicts"] = canonical.get("conflicts") or []
    final_graph = read_json_or_empty(final_graph_path)
    if final_graph:
        context["final_coordinate_system"] = final_graph.get("coordinate_system")
        context["alignment_transform"] = final_graph.get("alignment_transform")
        context["conflicts"] = final_graph.get("conflicts") or context["conflicts"]
    return context


def read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def review_risk_score(alignment_scores: dict[str, Any], reasons: list[str]) -> float:
    score = alignment_review_quality_score(alignment_scores)
    if isinstance(score, (int, float)):
        return round(max(0.0, 1.0 - float(score)) * 100.0, 3)
    return float(len(reasons) * 10)


def review_risk_level(alignment_scores: dict[str, Any], reasons: list[str]) -> str:
    score = alignment_review_quality_score(alignment_scores)
    if isinstance(score, (int, float)):
        if float(score) < 0.5:
            return "high"
        if float(score) < 0.8:
            return "medium"
        if reasons:
            return "medium"
        return "low"
    return "high" if len(reasons) >= 3 else "medium"


def source_image_evidence_from_canonical(canonical_path: Path, fullflow_root: Path) -> list[dict[str, str]]:
    if not canonical_path.is_file():
        return []
    try:
        canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    overlays = source_overlay_manifest(canonical_path)
    sources = []
    seen: set[str] = set()
    for ref in canonical.get("evidence_refs") or []:
        if str(ref.get("evidence_type") or "") != "package_graph":
            continue
        image_path = Path(str(ref.get("image_path") or ""))
        if not image_path.exists():
            continue
        resolved = str(image_path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        raw_view = str(ref.get("raw_view") or "unknown")
        canonical_view = str(ref.get("canonical_view") or "unknown")
        graph_path = str(ref.get("graph_path") or "")
        overlay = overlays.get(graph_path)
        if overlay:
            overlay_path = Path(str(overlay.get("path") or ""))
            if overlay_path.exists():
                sources.append(
                    {
                        "label": f"Source {raw_view} adopted overlay ({canonical_view})",
                        "path": str(overlay_path),
                        "url": root_relative_url(overlay_path, fullflow_root),
                        "raw_view": raw_view,
                        "canonical_view": canonical_view,
                        "graph_path": graph_path,
                        "original_image_path": str(image_path),
                    }
                )
                continue
        sources.append(
            {
                "label": f"Source {raw_view} ({canonical_view})",
                "path": str(image_path),
                "url": root_relative_url(image_path, fullflow_root),
                "raw_view": raw_view,
                "canonical_view": canonical_view,
                "graph_path": graph_path,
            }
        )
    return sources


def is_adopted_source_overlay(source: dict[str, str]) -> bool:
    label = str(source.get("label") or "").lower()
    path = str(source.get("path") or "").lower()
    return "adopted overlay" in label or path.endswith(".adopted.svg")


def source_overlay_manifest(canonical_path: Path) -> dict[str, dict[str, Any]]:
    manifest_path = canonical_path.parent / "source_overlays" / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    overlays = {}
    for item in payload.get("overlays") or []:
        graph_path = str(item.get("graph_path") or "")
        if graph_path:
            overlays[graph_path] = item
    return overlays


def main_postprocessed_graph_evidence_from_canonical(canonical_path: Path, fullflow_root: Path) -> list[dict[str, str]]:
    if not canonical_path.is_file():
        return []
    try:
        canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    sources = []
    seen: set[str] = set()
    for ref in canonical.get("evidence_refs") or []:
        if str(ref.get("evidence_type") or "") != "package_graph":
            continue
        raw_view = str(ref.get("raw_view") or "").strip().lower()
        if raw_view not in MAIN_POSTPROCESSED_VIEWS:
            continue
        graph_path = Path(str(ref.get("graph_path") or ""))
        visualization_path = package_graph_visualization_path(graph_path)
        if visualization_path is None or not visualization_path.exists():
            continue
        resolved = str(visualization_path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        sources.append(
            {
                "label": f"Postprocessed {raw_view} package graph",
                "path": str(visualization_path),
                "url": root_relative_url(visualization_path, fullflow_root),
                "raw_view": raw_view,
                "graph_path": str(graph_path),
            }
        )
    order = {view: index for index, view in enumerate(MAIN_POSTPROCESSED_VIEWS)}
    return sorted(sources, key=lambda item: (order.get(item["raw_view"], 99), item["path"]))


def package_graph_visualization_path(graph_path: Path) -> Path | None:
    if not str(graph_path):
        return None
    parts = graph_path.parts
    for index in range(len(parts) - 4):
        if (
            parts[index] == "outputs"
            and parts[index + 1] == "reconstruction"
            and parts[index + 3] == "graphs"
        ):
            run_id = parts[index + 2]
            relative_graph = Path(*parts[index + 4 :])
            if not relative_graph.name:
                return None
            output_root = Path(*parts[: index + 1])
            return output_root / "visualization" / f"reconstruction_{run_id}" / relative_graph.with_suffix(".png")
    return None


def main_view_overlay_evidence_from_canonical(
    canonical_path: Path,
    fullflow_root: Path,
    *,
    display_context: dict[str, Any] | None = None,
) -> dict[str, str] | None:
    payload = multiview_overlay_payload_from_canonical(canonical_path)
    if payload and payload.get("layers"):
        overlay_path = canonical_path.parent / "top_bottom_land_overlay.svg"
        display_context = display_context or display_scale_context_from_canonical(canonical_path)
        write_normalized_multiview_overlay_svg(
            overlay_path,
            payload,
            title="main-view overlay",
            subtitle="top/bottom/land geometry; coordinates are materialized by multiview integration",
            include_extra=False,
            display_context=display_context,
        )
        return {
            "label": "Main-view overlay",
            "path": str(overlay_path),
            "url": root_relative_url(overlay_path, fullflow_root),
        }

    graph_items = main_view_graph_items_from_canonical(canonical_path)
    if not graph_items:
        return None
    overlay_path = canonical_path.parent / "top_bottom_land_overlay.svg"
    write_main_view_overlay_svg(overlay_path, graph_items, display_context=display_context)
    return {
        "label": "Main-view overlay",
        "path": str(overlay_path),
        "url": root_relative_url(overlay_path, fullflow_root),
    }


def multiview_overlay_evidence_from_canonical(
    canonical_path: Path,
    fullflow_root: Path,
    *,
    display_context: dict[str, Any] | None = None,
) -> dict[str, str] | None:
    payload = multiview_overlay_payload_from_canonical(canonical_path)
    if payload and (payload.get("layers") or payload.get("extra_objects")):
        overlay_path = canonical_path.parent / "multi_view_overlay.svg"
        display_context = display_context or display_scale_context_from_canonical(canonical_path)
        write_normalized_multiview_overlay_svg(
            overlay_path,
            payload,
            title="multi-view overlay",
            subtitle="top/bottom/land plus materialized partial evidence; display stage only scales",
            include_extra=True,
            display_context=display_context,
        )
        return {
            "label": "Multi-view overlay",
            "path": str(overlay_path),
            "url": root_relative_url(overlay_path, fullflow_root),
        }

    graph_items = main_view_graph_items_from_canonical(canonical_path)
    extra_objects = multiview_overlay_extra_objects_from_canonical(canonical_path, graph_items)
    if not graph_items and not extra_objects:
        return None
    overlay_path = canonical_path.parent / "multi_view_overlay.svg"
    write_multiview_overlay_svg(
        overlay_path,
        graph_items,
        extra_objects=extra_objects,
        display_context=display_context,
    )
    return {
        "label": "Multi-view overlay",
        "path": str(overlay_path),
        "url": root_relative_url(overlay_path, fullflow_root),
    }


def multiview_overlay_extra_objects_from_canonical(
    canonical_path: Path,
    graph_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    canonical = read_json_or_empty(canonical_path)
    # Geometry must already be materialized by the multiview integration stage.
    # Review SVG generation is display-only and must not synthesize a second
    # coordinate path from side/front/lead partial graphs.
    extras = []
    for obj in canonical.get("lead_pads") or []:
        if str(obj.get("source_type") or "") == "scan_result_format":
            continue
        if not str(obj.get("source_graph") or ""):
            continue
        source_package_bbox = obj.get("source_package_pad_bbox")
        if not source_package_bbox:
            continue
        extras.append(obj)
    for obj in canonical.get("inner_land_pads") or []:
        if str(obj.get("source_type") or "") == "scan_result_format":
            continue
        if not str(obj.get("source_graph") or ""):
            continue
        source_land_bbox = obj.get("source_land_pad_bbox")
        if not source_land_bbox:
            continue
        extras.append(obj)
    return dedupe_overlay_extra_objects(extras)


def multiview_overlay_extra_objects_from_partial_graphs(
    canonical: dict[str, Any],
    graph_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project side/front/lead partial dimensions onto top/bottom/land pads.

    Coordinate system: output bboxes use the selected main-view package graph
    reconstructed coordinates.  Dimension values are physical units; conversion
    uses the selected main-view graph unit scale in physical units per graph
    pixel.
    """
    base_item = select_overlay_base_graph_item(graph_items)
    if base_item is None:
        return []
    base_graph = base_item["graph"]
    unit_scales = graph_contact_unit_scales(base_graph)
    if not unit_scales:
        return []
    terminal_pads = terminal_overlay_pad_objects(base_graph)
    if not terminal_pads:
        return []
    outline_bbox = graph_frame(base_graph) or union_boxes([bbox for _obj, bbox in terminal_pads])
    if outline_bbox is None:
        return []

    dimensions = multiview_partial_contact_dimensions(canonical)
    if not dimensions:
        return []
    center_x = (outline_bbox[0] + outline_bbox[2]) / 2.0
    center_y = (outline_bbox[1] + outline_bbox[3]) / 2.0
    extras = []
    for selected in dimensions:
        dimension_value = positive_float(selected.get("value"))
        if dimension_value is None:
            continue
        base_semantics = str(selected.get("overlay_semantics") or "")
        for index, (pad, bbox) in enumerate(terminal_pads):
            x1, y1, x2, y2 = bbox
            pad_cx = (x1 + x2) / 2.0
            pad_cy = (y1 + y2) / 2.0
            radial_axis = "x" if abs(pad_cx - center_x) >= abs(pad_cy - center_y) else "y"
            projection_axis = overlay_projection_axis(selected, radial_axis)
            unit_scale = unit_scales.get(projection_axis)
            if unit_scale is None or unit_scale <= 0.0:
                continue
            semantics = effective_partial_dimension_semantics(
                dim=selected,
                base_semantics=base_semantics,
                projection_axis=projection_axis,
                dimension_value=dimension_value,
                bbox=bbox,
                unit_scale=unit_scale,
            )
            length = dimension_value / unit_scale
            if length <= 0.0:
                continue
            new_bbox = partial_dimension_overlay_bbox(
                bbox=bbox,
                package_center=(center_x, center_y),
                package_frame=outline_bbox,
                projection_axis=projection_axis,
                length=length,
                semantics=semantics,
            )
            extras.append(
                {
                    "role": overlay_extra_role(semantics),
                    "label": overlay_extra_role(semantics),
                    "source_label": pad.get("source_label") or pad.get("label"),
                    "bbox": new_bbox,
                    "source_type": "derived_partial_evidence_display",
                    "source_graph": str(base_item.get("graph_path") or ""),
                    "source_package_pad_id": pad.get("id"),
                    "source_package_pad_bbox": list(bbox),
                    "source_package_pad_index": index,
                    "lead_contact_length": dimension_value,
                    "lead_contact_length_axis": str(selected.get("axis") or ""),
                    "partial_dimension_semantics": semantics,
                    "partial_dimension_base_semantics": base_semantics,
                    "lead_contact_length_source": {
                        "id": selected.get("id"),
                        "dimension_id": selected.get("dimension_id"),
                        "text": selected.get("text"),
                        "value": selected.get("value"),
                        "raw_view": selected.get("raw_view"),
                        "canonical_view": selected.get("canonical_view"),
                        "source_graph": selected.get("source_graph"),
                        "annotation_path": selected.get("annotation_path"),
                    },
                    "radial_axis": radial_axis,
                    "projection_axis": projection_axis,
                    "coordinate_unit_scale": unit_scale,
                }
            )
    return extras


def effective_partial_dimension_semantics(
    *,
    dim: dict[str, Any],
    base_semantics: str,
    projection_axis: str,
    dimension_value: float,
    bbox: tuple[float, float, float, float],
    unit_scale: float,
) -> str:
    """Resolve partial evidence semantics after mapping it to a main-view pad.

    Coordinate system: bbox is in the selected main-view package graph
    reconstructed coordinates. unit_scale is physical units per graph pixel on
    projection_axis. If a front-view width is larger than the main-view pad
    extent, treat it as contact length hidden under the body and align it to
    the pad's outer edge instead of centering it as pad width.
    """
    if base_semantics != "pad_width":
        return base_semantics
    raw_view = str(dim.get("raw_view") or "").strip().lower()
    if raw_view != "front":
        return base_semantics
    x1, y1, x2, y2 = bbox
    pad_extent = (x2 - x1) if projection_axis == "x" else (y2 - y1)
    pad_physical_extent = pad_extent * unit_scale
    if pad_physical_extent > 0.0 and dimension_value > pad_physical_extent:
        return "lead_ground_contact_length"
    return base_semantics


def partial_dimension_overlay_bbox(
    *,
    bbox: tuple[float, float, float, float],
    package_center: tuple[float, float],
    package_frame: tuple[float, float, float, float] | None = None,
    projection_axis: str,
    length: float,
    semantics: str,
) -> list[float]:
    x1, y1, x2, y2 = bbox
    pad_cx = (x1 + x2) / 2.0
    pad_cy = (y1 + y2) / 2.0
    center_x, center_y = package_center
    if semantics == "pad_width":
        if projection_axis == "x":
            return [pad_cx - length / 2.0, y1, pad_cx + length / 2.0, y2]
        return [x1, pad_cy - length / 2.0, x2, pad_cy + length / 2.0]
    if semantics == "lead_pad_length":
        if projection_axis == "x":
            return [pad_cx - length / 2.0, y1, pad_cx + length / 2.0, y2]
        return [x1, pad_cy - length / 2.0, x2, pad_cy + length / 2.0]
    radial_axis = "x" if abs(pad_cx - center_x) >= abs(pad_cy - center_y) else "y"
    if semantics == "lead_ground_contact_length" and projection_axis != radial_axis:
        outside_side = pad_outside_side(bbox, package_frame)
        if projection_axis == "y" and outside_side == "top":
            return [x1, y1, x2, y1 + length]
        if projection_axis == "y" and outside_side == "bottom":
            return [x1, y2 - length, x2, y2]
        if projection_axis == "x" and outside_side == "left":
            return [x1, y1, x1 + length, y2]
        if projection_axis == "x" and outside_side == "right":
            return [x2 - length, y1, x2, y2]
        if projection_axis == "x":
            return [pad_cx - length / 2.0, y1, pad_cx + length / 2.0, y2]
        return [x1, pad_cy - length / 2.0, x2, pad_cy + length / 2.0]
    if projection_axis == "x":
        if pad_cx <= center_x:
            return [x1, y1, x1 + length, y2]
        return [x2 - length, y1, x2, y2]
    if pad_cy <= center_y:
        return [x1, y1, x2, y1 + length]
    return [x1, y2 - length, x2, y2]


def pad_outside_side(
    bbox: tuple[float, float, float, float],
    package_frame: tuple[float, float, float, float] | None,
) -> str:
    if package_frame is None:
        return ""
    x1, y1, x2, y2 = bbox
    frame_x1, frame_y1, frame_x2, frame_y2 = package_frame
    pad_cx = (x1 + x2) / 2.0
    pad_cy = (y1 + y2) / 2.0
    if pad_cy < frame_y1:
        return "top"
    if pad_cy > frame_y2:
        return "bottom"
    if pad_cx < frame_x1:
        return "left"
    if pad_cx > frame_x2:
        return "right"
    return ""


def overlay_extra_role(semantics: str) -> str:
    if semantics == "pad_width":
        return "partial_pad_width"
    if semantics == "lead_pad_length":
        return "partial_lead_pad_length"
    return "lead_pad"


def overlay_projection_axis(dim: dict[str, Any], radial_axis: str) -> str:
    semantics = str(dim.get("overlay_semantics") or "")
    raw_view = str(dim.get("raw_view") or "").strip().lower()
    if semantics in {"pad_width", "lead_pad_length", "lead_ground_contact_length"}:
        if raw_view == "side":
            return "y"
        if raw_view == "front":
            return "x"
        return radial_axis
    return radial_axis


def select_overlay_base_graph_item(graph_items: list[dict[str, Any]]) -> dict[str, Any] | None:
    order = {"bottom": 0, "top": 1, "land": 2}
    candidates = []
    for item in graph_items:
        pads = terminal_overlay_pad_objects(item["graph"])
        if not pads:
            continue
        raw_view = str(item.get("raw_view") or "")
        candidates.append((order.get(raw_view, 99), -len(pads), str(item.get("graph_path") or ""), item))
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: row[:3])[0][3]


def multiview_partial_contact_dimensions(canonical: dict[str, Any]) -> list[dict[str, Any]]:
    dimensions = []
    for ref in canonical.get("evidence_refs") or []:
        if str(ref.get("evidence_type") or "") != "package_graph":
            continue
        raw_view = str(ref.get("raw_view") or "").strip().lower()
        canonical_view = normalized_multiview_color_key(str(ref.get("canonical_view") or raw_view))
        if canonical_view not in {"lateral", "lead_detail"}:
            continue
        graph_path = Path(str(ref.get("graph_path") or ""))
        if not graph_path.is_file():
            continue
        graph = read_json_or_empty(graph_path)
        if not graph:
            continue
        objects_by_id = {str(obj.get("id")): obj for obj in graph.get("objects") or [] if obj.get("id") is not None}
        for dim in graph.get("dimensions") or []:
            semantics = overlay_dimension_semantics(dim, raw_view)
            if not is_overlay_contact_dimension(dim, objects_by_id, raw_view, semantics):
                continue
            enriched = dict(dim)
            enriched["raw_view"] = raw_view
            enriched["canonical_view"] = canonical_view
            enriched["source_graph"] = str(graph_path)
            enriched["annotation_path"] = ref.get("annotation_path")
            enriched["target_labels"] = dimension_target_labels(dim, objects_by_id)
            enriched["overlay_semantics"] = semantics
            dimensions.append(enriched)
    return dimensions


def overlay_dimension_semantics(dim: dict[str, Any], raw_view: str) -> str:
    axis = str(dim.get("axis") or "").lower()
    if axis != "x":
        return ""
    target_ids = list(dim.get("target_ids") or [])
    if len(target_ids) != 1:
        return ""
    anchors = [str(anchor or "").lower() for anchor in dim.get("anchors") or []]
    has_left_or_right = any(anchor in {"left_edge", "right_edge"} for anchor in anchors)
    if "center" in anchors and has_left_or_right:
        return "lead_ground_contact_length"
    if set(anchors) == {"left_edge", "right_edge"}:
        if raw_view == "lead":
            return "lead_pad_length"
        if raw_view == "side":
            return "lead_ground_contact_length"
        return "pad_width"
    return ""


def is_overlay_contact_dimension(
    dim: dict[str, Any],
    objects_by_id: dict[str, dict[str, Any]],
    raw_view: str,
    semantics: str,
) -> bool:
    if str(dim.get("status") or "") != "accepted":
        return False
    if str(dim.get("kind") or "") != "size":
        return False
    if semantics not in {"lead_ground_contact_length", "pad_width", "lead_pad_length"}:
        return False
    if positive_float(dim.get("value")) is None:
        return False
    labels = dimension_target_labels(dim, objects_by_id)
    if not labels:
        return False
    label_text = " ".join(labels).lower()
    if "outline" in label_text or "package" in label_text:
        return False
    return any(token in label_text for token in ("pad", "lead", "rect", "circle", "dshape"))


def dimension_target_labels(dim: dict[str, Any], objects_by_id: dict[str, dict[str, Any]]) -> list[str]:
    labels = []
    for target_id in dim.get("target_ids") or []:
        obj = objects_by_id.get(str(target_id))
        if obj is None:
            continue
        label = " ".join(str(obj.get(key) or "") for key in ("label", "source_label", "shape")).strip()
        if label:
            labels.append(label.lower())
    return sorted(set(labels))


def terminal_overlay_pad_objects(graph: dict[str, Any]) -> list[tuple[dict[str, Any], tuple[float, float, float, float]]]:
    pads = []
    for obj in graph.get("objects") or []:
        bbox = object_bbox(obj)
        if bbox is None or not object_is_pad_like(obj):
            continue
        pads.append((obj, bbox))
    if len(pads) < 5:
        return pads
    areas = sorted((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for _obj, bbox in pads)
    median_area = areas[len(areas) // 2]
    extent = union_boxes([bbox for _obj, bbox in pads])
    if median_area <= 0.0 or extent is None:
        return pads
    x1, y1, x2, y2 = extent
    width = x2 - x1
    height = y2 - y1
    if width <= 0.0 or height <= 0.0:
        return pads
    terminals = []
    for obj, bbox in pads:
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        x_ratio = (cx - x1) / width
        y_ratio = (cy - y1) / height
        central = 0.2 <= x_ratio <= 0.8 and 0.2 <= y_ratio <= 0.8
        if central and area >= median_area * 1.8:
            continue
        terminals.append((obj, bbox))
    return terminals or pads


def graph_contact_unit_scales(graph: dict[str, Any]) -> dict[str, float]:
    metrics = graph.get("metrics") or {}
    global_scale = positive_float(metrics.get("global_scale"))
    x_scale = positive_float(metrics.get("axis_scale_x")) or global_scale
    y_scale = positive_float(metrics.get("axis_scale_y")) or global_scale
    dimension_scales = graph_dimension_unit_scales(graph)
    x_scale = x_scale or positive_float(dimension_scales.get("x"))
    y_scale = y_scale or positive_float(dimension_scales.get("y"))
    if x_scale is None or y_scale is None:
        return {}
    return {"x": x_scale, "y": y_scale}


def dedupe_overlay_extra_objects(extras: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for obj in extras:
        bbox = object_bbox(obj)
        if bbox is None:
            continue
        key = (
            str(obj.get("role") or ""),
            str(obj.get("source_graph") or ""),
            tuple(round(value, 3) for value in bbox),
            partial_evidence_view_key(obj),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(obj)
    return deduped


def scan_result_display_evidence_from_canonical(
    canonical_path: Path,
    fullflow_root: Path,
    *,
    display_context: dict[str, Any] | None = None,
) -> dict[str, str] | None:
    objects = scan_result_gt_objects_from_canonical(canonical_path)
    if not objects:
        return None
    display_context = display_context or display_scale_context_from_canonical(canonical_path)
    output_path = canonical_path.parent / "scan_result_gt_display.svg"
    write_scan_result_gt_display_svg(output_path, objects, display_context=display_context)
    return {
        "label": "ScanResult GT",
        "path": str(output_path),
        "url": root_relative_url(output_path, fullflow_root),
    }


def scan_result_gt_objects_from_canonical(canonical_path: Path) -> list[dict[str, Any]]:
    final_graph = read_json_or_empty(canonical_path.with_name("final_graph.json"))
    objects = (final_graph.get("gt_reference") or {}).get("objects") or []
    return [obj for obj in objects if object_bbox(obj) is not None]


def final_result_display_evidence_from_canonical(
    canonical_path: Path,
    fullflow_root: Path,
    *,
    display_context: dict[str, Any] | None = None,
) -> dict[str, str] | None:
    objects = final_result_objects_from_canonical(canonical_path)
    if not objects:
        return None
    display_context = display_context or display_scale_context_from_canonical(canonical_path)
    output_path = canonical_path.parent / "multi_view_overlay.svg"
    write_final_result_display_svg(output_path, objects, display_context=display_context)
    return {
        "label": "Multi-view overlay",
        "path": str(output_path),
        "url": root_relative_url(output_path, fullflow_root),
    }


def final_result_objects_from_canonical(canonical_path: Path) -> list[dict[str, Any]]:
    final_graph = read_json_or_empty(canonical_path.with_name("final_graph.json"))
    objects = final_graph.get("objects") or []
    return [obj for obj in objects if object_bbox(obj) is not None]


def multiview_overlay_payload_from_canonical(canonical_path: Path) -> dict[str, Any] | None:
    canonical = read_json_or_empty(canonical_path)
    payload = canonical.get("multiview_overlay")
    if not isinstance(payload, dict):
        return None
    if str(payload.get("coordinate_mode") or "") != "dimension_scaled_centered":
        return None
    return payload


def normalized_overlay_frames_from_payload(payload: dict[str, Any]) -> list[tuple[float, float, float, float]]:
    frames = []
    frame = object_bbox({"bbox": payload.get("frame") or []})
    if frame is not None:
        frames.append(frame)
    for layer in payload.get("layers") or []:
        frame = object_bbox({"bbox": layer.get("normalized_frame") or []})
        if frame is not None:
            frames.append(frame)
    for obj in payload.get("extra_objects") or []:
        bbox = object_bbox(obj)
        if bbox is not None:
            frames.append(bbox)
    return frames


def display_scale_context_from_canonical(canonical_path: Path) -> dict[str, Any]:
    payload = multiview_overlay_payload_from_canonical(canonical_path)
    normalized_frames = normalized_overlay_frames_from_payload(payload) if payload else []
    if normalized_frames:
        gt_objects = scan_result_gt_objects_from_canonical(canonical_path)
        gt_frame = scan_result_gt_scale_frame(gt_objects)
        frames = list(normalized_frames)
        if gt_frame is not None:
            frames.append(gt_frame)
        scale = overlay_common_scale(frames, DISPLAY_SVG_TARGET)
        return {
            "width": DISPLAY_SVG_WIDTH,
            "height": DISPLAY_SVG_HEIGHT,
            "target": DISPLAY_SVG_TARGET,
            "scale": scale,
            "scale_source": (
                "gt_and_multiview_overlay_normalized"
                if gt_frame is not None
                else "multiview_overlay_normalized"
            ),
        }

    graph_items = main_view_graph_items_from_canonical(canonical_path)
    overlay_frames = []
    for item in graph_items:
        frame = graph_display_frame(item["graph"])
        if frame is None:
            continue
        overlay_frames.append(calibrated_frame(frame, graph_dimension_unit_scales(item["graph"])))
    gt_objects = scan_result_gt_objects_from_canonical(canonical_path)
    gt_frame = scan_result_gt_scale_frame(gt_objects)
    frames = list(overlay_frames)
    if gt_frame is not None:
        frames.append(gt_frame)
    scale = overlay_common_scale(frames, DISPLAY_SVG_TARGET) if frames else 1.0
    return {
        "width": DISPLAY_SVG_WIDTH,
        "height": DISPLAY_SVG_HEIGHT,
        "target": DISPLAY_SVG_TARGET,
        "scale": scale,
        "scale_source": "gt_and_dimension_calibrated_overlay" if gt_frame is not None and overlay_frames else "available_display_geometry",
    }


def main_view_graph_items_from_canonical(canonical_path: Path) -> list[dict[str, Any]]:
    if not canonical_path.exists():
        return []
    try:
        canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    items = []
    seen: set[str] = set()
    for ref in canonical.get("evidence_refs") or []:
        if str(ref.get("evidence_type") or "") != "package_graph":
            continue
        raw_view = str(ref.get("raw_view") or "").strip().lower()
        if raw_view not in MAIN_POSTPROCESSED_VIEWS:
            continue
        graph_path_text = str(ref.get("graph_path") or "").strip()
        if not graph_path_text:
            continue
        graph_path = Path(graph_path_text)
        if not graph_path.is_file():
            continue
        resolved = str(graph_path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        graph = read_json_or_empty(graph_path)
        if not graph:
            continue
        graph = regularized_main_view_graph_for_overlay(graph, raw_view)
        items.append({"raw_view": raw_view, "graph_path": str(graph_path), "graph": graph})
    order = {view: index for index, view in enumerate(MAIN_POSTPROCESSED_VIEWS)}
    return sorted(items, key=lambda item: (order.get(item["raw_view"], 99), item["graph_path"]))


def regularized_main_view_graph_for_overlay(graph: dict[str, Any], raw_view: str) -> dict[str, Any]:
    """Return graph geometry after applying bottom pad x regularization.

    Coordinate system: package graph reconstructed pixels.  This is display-only
    normalization for review overlays; it mirrors the multiview integration
    regularizer so source overlays show the same two-column pad geometry that
    unified_multiview_layers.json uses.
    """
    options = MultiviewOptions()
    if raw_view == "land":
        return regularized_land_view_graph_for_overlay(graph, options)
    if raw_view != "bottom":
        return graph
    outline = extract_outline(graph, options)
    package_pads = extract_objects(graph, "package_pad", options)
    adjusted = regularize_two_column_package_pad_x_geometry(package_pads, outline, graph)
    adjusted_by_id = {
        pad.get("source_object_id"): pad
        for pad in adjusted
        if str(pad.get("geometry_adjusted_reason") or "") == "dimension_regularized_package_pad_x_grid"
    }
    if not adjusted_by_id:
        return graph

    new_graph = dict(graph)
    new_objects = []
    for obj in graph.get("objects") or []:
        adjusted_pad = adjusted_by_id.get(obj.get("id"))
        if adjusted_pad is None:
            new_objects.append(obj)
            continue
        bbox = obj.get("bbox_reconstructed") or obj.get("bbox")
        if not bbox:
            new_objects.append(obj)
            continue
        new_obj = dict(obj)
        if obj.get("bbox_reconstructed") is not None:
            new_obj["bbox_reconstructed"] = list(adjusted_pad["bbox"])
        else:
            new_obj["bbox"] = list(adjusted_pad["bbox"])
        new_obj["bbox_before_overlay_regularization"] = list(bbox)
        new_obj["overlay_geometry_adjusted_reason"] = adjusted_pad["geometry_adjusted_reason"]
        new_obj["overlay_dimension_regularization_axis"] = adjusted_pad.get("dimension_regularization_axis")
        new_objects.append(new_obj)
    new_graph["objects"] = new_objects
    return new_graph


def regularized_land_view_graph_for_overlay(graph: dict[str, Any], options: MultiviewOptions) -> dict[str, Any]:
    """Regularize two-column land pad x geometry for review display only.

    Coordinate system: reconstructed graph pixels.  This intentionally changes
    only x1/x2 in overlay SVG generation; y placement, source graph JSON, and
    canonical integration output are preserved.
    """
    land_pads = extract_objects(graph, "land_pad", options)
    adjusted = regularize_two_column_overlay_pad_x_geometry(land_pads)
    adjusted_by_id = {
        pad.get("source_object_id"): pad
        for pad in adjusted
        if str(pad.get("geometry_adjusted_reason") or "") == "display_regularized_two_column_land_pad_x_grid"
    }
    if not adjusted_by_id:
        return graph

    new_graph = dict(graph)
    new_objects = []
    for obj in graph.get("objects") or []:
        adjusted_pad = adjusted_by_id.get(obj.get("id"))
        if adjusted_pad is None:
            new_objects.append(obj)
            continue
        bbox = obj.get("bbox_reconstructed") or obj.get("bbox")
        if not bbox:
            new_objects.append(obj)
            continue
        new_obj = dict(obj)
        if obj.get("bbox_reconstructed") is not None:
            new_obj["bbox_reconstructed"] = list(adjusted_pad["bbox"])
        else:
            new_obj["bbox"] = list(adjusted_pad["bbox"])
        new_obj["bbox_before_overlay_regularization"] = list(bbox)
        new_obj["overlay_geometry_adjusted_reason"] = adjusted_pad["geometry_adjusted_reason"]
        new_obj["overlay_dimension_regularization_axis"] = adjusted_pad.get("dimension_regularization_axis")
        new_objects.append(new_obj)
    new_graph["objects"] = new_objects
    return new_graph


def regularize_two_column_overlay_pad_x_geometry(pads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(pads) < 4 or len(pads) % 2 != 0:
        return pads
    pad_boxes = []
    for pad in pads:
        bbox = object_bbox(pad)
        if bbox is None:
            return pads
        width = bbox[2] - bbox[0]
        if width <= 0:
            return pads
        pad_boxes.append((pad, bbox, (bbox[0] + bbox[2]) / 2.0, width))
    widths = [item[3] for item in pad_boxes]
    min_width = min(widths)
    max_width = max(widths)
    if min_width <= 0 or max_width / min_width > 1.25:
        return pads

    pad_boxes.sort(key=lambda item: item[2])
    midpoint = len(pad_boxes) // 2
    columns = (pad_boxes[:midpoint], pad_boxes[midpoint:])
    if len(columns[0]) != len(columns[1]):
        return pads

    target_width = median_float(widths)
    adjusted_by_id: dict[Any, dict[str, Any]] = {}
    for column in columns:
        centers = [item[2] for item in column]
        if max(centers) - min(centers) > max(target_width * 0.5, 8.0):
            return pads
        target_center = median_float(centers)
        new_x1 = round(target_center - target_width / 2.0, 6)
        new_x2 = round(target_center + target_width / 2.0, 6)
        for pad, bbox, _center, _width in column:
            adjusted_by_id[pad.get("source_object_id")] = dict(
                pad,
                bbox=[new_x1, bbox[1], new_x2, bbox[3]],
                bbox_before_overlay_regularization=list(bbox),
                geometry_adjusted_reason="display_regularized_two_column_land_pad_x_grid",
                dimension_regularization_axis="x",
            )

    return [
        adjusted_by_id.get(pad.get("source_object_id"), pad)
        for pad in pads
    ]


def median_float(values: list[float]) -> float:
    ordered = sorted(float(value) for value in values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def write_main_view_overlay_svg(
    path: Path,
    graph_items: list[dict[str, Any]],
    *,
    display_context: dict[str, Any] | None = None,
) -> None:
    frames = [graph_display_frame(item["graph"]) for item in graph_items]
    drawable = [(item, frame) for item, frame in zip(graph_items, frames) if frame is not None]
    if not drawable:
        path.write_text(main_view_empty_overlay_svg(), encoding="utf-8")
        return
    display_context = display_context or {}
    width = float(display_context.get("width") or DISPLAY_SVG_WIDTH)
    height = float(display_context.get("height") or DISPLAY_SVG_HEIGHT)
    target = tuple(display_context.get("target") or DISPLAY_SVG_TARGET)
    elements = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        '<text x="24" y="38" font-size="20" font-family="monospace" fill="#0f172a">'
        "top / bottom / land postprocessed overlay</text>",
        '<text x="24" y="64" font-size="13" font-family="monospace" fill="#64748b">'
        "dimension-calibrated scale from accepted dimensions; graph pixels are fallback only</text>",
    ]
    legend_x = 650.0
    for index, view in enumerate(MAIN_POSTPROCESSED_VIEWS):
        color = MAIN_POSTPROCESSED_VIEW_COLORS[view]
        y = 30.0 + index * 24.0
        elements.append(f'<rect x="{legend_x}" y="{y - 12}" width="16" height="16" fill="{color}" opacity="0.24" stroke="{color}"/>')
        elements.append(f'<text x="{legend_x + 24}" y="{y + 1}" font-size="14" font-family="monospace" fill="#334155">{view}</text>')
    calibrated = [(item, frame, graph_dimension_unit_scales(item["graph"])) for item, frame in drawable]
    common_scale = float(display_context.get("scale") or overlay_common_scale([calibrated_frame(frame, unit_scales) for _, frame, unit_scales in calibrated], target))
    elements.extend(main_view_overlay_graph_elements(calibrated, target, common_scale))
    path.write_text(
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{int(width)}" height="{int(height)}">'
            + "".join(elements)
            + "</svg>\n"
        ),
        encoding="utf-8",
    )


def write_multiview_overlay_svg(
    path: Path,
    graph_items: list[dict[str, Any]],
    *,
    extra_objects: list[dict[str, Any]],
    display_context: dict[str, Any] | None = None,
) -> None:
    frames = [graph_display_frame(item["graph"]) for item in graph_items]
    drawable = [(item, frame) for item, frame in zip(graph_items, frames) if frame is not None]
    display_context = display_context or {}
    width = float(display_context.get("width") or DISPLAY_SVG_WIDTH)
    height = float(display_context.get("height") or DISPLAY_SVG_HEIGHT)
    target = tuple(display_context.get("target") or DISPLAY_SVG_TARGET)
    calibrated = [(item, frame, graph_dimension_unit_scales(item["graph"])) for item, frame in drawable]
    if not calibrated and not extra_objects:
        path.write_text(main_view_empty_overlay_svg(), encoding="utf-8")
        return
    common_scale = float(display_context.get("scale") or overlay_common_scale([calibrated_frame(frame, unit_scales) for _, frame, unit_scales in calibrated], target))
    elements = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        '<text x="24" y="38" font-size="20" font-family="monospace" fill="#0f172a">'
        "multi-view overlay</text>",
        '<text x="24" y="64" font-size="13" font-family="monospace" fill="#64748b">'
        "top/bottom/land source geometry plus projected partial evidence only</text>",
    ]
    elements.extend(multiview_overlay_legend_svg(extra_objects))
    elements.extend(main_view_overlay_graph_elements(calibrated, target, common_scale))
    elements.extend(multiview_extra_object_elements(extra_objects, calibrated, target, common_scale))
    path.write_text(
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{int(width)}" height="{int(height)}">'
            + "".join(elements)
            + "</svg>\n"
        ),
        encoding="utf-8",
    )


def write_normalized_multiview_overlay_svg(
    path: Path,
    payload: dict[str, Any],
    *,
    title: str,
    subtitle: str,
    include_extra: bool,
    display_context: dict[str, Any],
) -> None:
    layers = list(payload.get("layers") or [])
    extra_objects = list(payload.get("extra_objects") or []) if include_extra else []
    if not layers and not extra_objects:
        path.write_text(main_view_empty_overlay_svg(), encoding="utf-8")
        return

    width = float(display_context.get("width") or DISPLAY_SVG_WIDTH)
    height = float(display_context.get("height") or DISPLAY_SVG_HEIGHT)
    target = tuple(display_context.get("target") or DISPLAY_SVG_TARGET)
    frames = normalized_overlay_frames_from_payload(
        {"layers": layers, "extra_objects": extra_objects, "frame": payload.get("frame") or []}
    )
    display_scale = float(display_context.get("scale") or overlay_common_scale(frames, target))
    elements = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="24" y="38" font-size="20" font-family="monospace" fill="#0f172a">{html.escape(title)}</text>',
        f'<text x="24" y="64" font-size="13" font-family="monospace" fill="#64748b">{html.escape(subtitle)}</text>',
        '<text x="24" y="86" font-size="12" font-family="monospace" fill="#94a3b8">'
        "coordinate_mode=dimension_scaled_centered; display-only scale</text>",
    ]
    elements.extend(normalized_multiview_overlay_legend_svg(layers, extra_objects))
    elements.extend(normalized_multiview_layer_elements(layers, target, display_scale))
    elements.extend(normalized_multiview_extra_object_elements(extra_objects, target, display_scale))
    path.write_text(
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{int(width)}" height="{int(height)}" '
            f'data-coordinate-mode="dimension_scaled_centered" data-display-only="true">'
            + "".join(elements)
            + "</svg>\n"
        ),
        encoding="utf-8",
    )


def normalized_multiview_layer_elements(
    layers: list[dict[str, Any]],
    target: tuple[float, float, float, float],
    display_scale: float,
) -> list[str]:
    elements = []
    for layer in layers:
        view = str(layer.get("raw_view") or "")
        color = MAIN_POSTPROCESSED_VIEW_COLORS.get(view, "#64748b")
        graph_name = html.escape(Path(str(layer.get("graph_path") or "")).name)
        unit_scales = layer.get("unit_scales") or {}
        elements.append(
            f'<g data-view="{html.escape(view)}" data-graph="{graph_name}" '
            'data-coordinate-mode="dimension_scaled_centered" data-display-only="true" '
            f'data-scale-source="{html.escape(str(unit_scales.get("source") or ""))}" '
            f'data-display-scale="{display_scale:.8f}" '
            f'data-unit-per-px-x="{float(unit_scales.get("x") or 1.0):.8f}" '
            f'data-unit-per-px-y="{float(unit_scales.get("y") or 1.0):.8f}">'
        )
        for obj in layer.get("objects") or []:
            bbox = object_bbox(obj)
            if bbox is None:
                continue
            mapped = normalized_bbox_to_display(bbox, target, display_scale)
            elements.append(overlay_object_svg(obj, mapped, color=color))
        elements.append("</g>")
    return elements


def normalized_multiview_extra_object_elements(
    extra_objects: list[dict[str, Any]],
    target: tuple[float, float, float, float],
    display_scale: float,
) -> list[str]:
    elements = []
    for obj in extra_objects:
        bbox = object_bbox(obj)
        if bbox is None:
            continue
        mapped = normalized_bbox_to_display(bbox, target, display_scale)
        view_key = partial_evidence_view_key(obj)
        color = MULTIVIEW_OVERLAY_VIEW_COLORS.get(view_key, MULTIVIEW_OVERLAY_VIEW_COLORS["unknown"])
        elements.append(
            overlay_rect(
                mapped,
                stroke=color,
                width=4,
                fill=color,
                fill_opacity=0.30,
                data_attrs={
                    "role": str(obj.get("role") or ""),
                    "source_view": view_key,
                    "canonical_view": final_result_object_view_key(obj),
                    "coordinate_mode": str(obj.get("coordinate_mode") or ""),
                    "projection_axis": str(obj.get("projection_axis") or ""),
                    "partial_dimension_semantics": str(obj.get("partial_dimension_semantics") or ""),
                    "source_graph": Path(str(obj.get("source_graph") or "")).name,
                },
            )
        )
    return elements


def normalized_bbox_to_display(
    bbox: tuple[float, float, float, float],
    target: tuple[float, float, float, float],
    display_scale: float,
) -> tuple[float, float, float, float]:
    tx1, ty1, tx2, ty2 = target
    target_cx = (tx1 + tx2) / 2.0
    target_cy = (ty1 + ty2) / 2.0
    x1, y1, x2, y2 = bbox
    return (
        target_cx + x1 * display_scale,
        target_cy + y1 * display_scale,
        target_cx + x2 * display_scale,
        target_cy + y2 * display_scale,
    )


def normalized_multiview_overlay_legend_svg(
    layers: list[dict[str, Any]],
    extra_objects: list[dict[str, Any]],
) -> list[str]:
    used_views = {str(layer.get("raw_view") or "") for layer in layers}
    used_views.update(partial_evidence_view_key(obj) for obj in extra_objects)
    used_views.discard("")
    order = ["top", "bottom", "land", "front", "side", "lead", "land_detail", "lateral", "lead_detail", "unknown"]
    labels = {
        "top": "top",
        "bottom": "bottom",
        "land": "land",
        "front": "front partial evidence",
        "side": "side partial evidence",
        "lead": "lead partial evidence",
        "land_detail": "land detail partial evidence",
        "lateral": "lateral partial evidence",
        "lead_detail": "lead detail partial evidence",
        "unknown": "unknown partial evidence",
    }
    elements = []
    x = 650.0
    y = 30.0
    for index, view in enumerate([item for item in order if item in used_views]):
        color = MAIN_POSTPROCESSED_VIEW_COLORS.get(view) or MULTIVIEW_OVERLAY_VIEW_COLORS.get(view, "#94a3b8")
        item_y = y + index * 24.0
        elements.append(
            f'<rect x="{x}" y="{item_y - 12}" width="16" height="16" '
            f'fill="{color}" opacity="0.24" stroke="{color}"/>'
        )
        elements.append(
            f'<text x="{x + 24}" y="{item_y + 1}" font-size="14" '
            f'font-family="monospace" fill="#334155">{html.escape(labels.get(view, view))}</text>'
        )
    return elements


def main_view_overlay_graph_elements(
    calibrated: list[tuple[dict[str, Any], tuple[float, float, float, float], dict[str, Any]]],
    target: tuple[float, float, float, float],
    common_scale: float,
) -> list[str]:
    elements = []
    for item, frame, unit_scales in calibrated:
        view = str(item["raw_view"])
        color = MAIN_POSTPROCESSED_VIEW_COLORS.get(view, "#64748b")
        transform = graph_to_overlay_transform(frame, target, scale=common_scale, unit_scales=unit_scales)
        graph_name = html.escape(Path(str(item.get("graph_path") or "")).name)
        elements.append(
            f'<g data-view="{html.escape(view)}" data-graph="{graph_name}" '
            f'data-scale-source="{html.escape(str(unit_scales["source"]))}" '
            f'data-display-scale="{common_scale:.8f}" '
            f'data-unit-per-px-x="{unit_scales["x"]:.8f}" data-unit-per-px-y="{unit_scales["y"]:.8f}">'
        )
        for obj in item["graph"].get("objects") or []:
            bbox = object_bbox(obj)
            if bbox is None:
                continue
            mapped = map_bbox(bbox, transform)
            elements.append(overlay_object_svg(obj, mapped, color=color))
        elements.append("</g>")
    return elements


def multiview_extra_object_elements(
    extra_objects: list[dict[str, Any]],
    calibrated: list[tuple[dict[str, Any], tuple[float, float, float, float], dict[str, Any]]],
    target: tuple[float, float, float, float],
    common_scale: float,
) -> list[str]:
    by_graph = {
        str(item.get("graph_path") or ""): (frame, unit_scales)
        for item, frame, unit_scales in calibrated
    }
    elements = []
    for obj in extra_objects:
        source_graph = str(obj.get("source_graph") or "")
        if source_graph not in by_graph:
            continue
        bbox = object_bbox(obj)
        if bbox is None:
            continue
        frame, unit_scales = by_graph[source_graph]
        transform = graph_to_overlay_transform(frame, target, scale=common_scale, unit_scales=unit_scales)
        mapped = map_bbox(bbox, transform)
        view_key = partial_evidence_view_key(obj)
        color = MULTIVIEW_OVERLAY_VIEW_COLORS.get(view_key, MULTIVIEW_OVERLAY_VIEW_COLORS["unknown"])
        elements.append(
            overlay_rect(
                mapped,
                stroke=color,
                width=4,
                fill=color,
                fill_opacity=0.30,
                data_attrs={
                    "role": str(obj.get("role") or ""),
                    "source_view": view_key,
                    "canonical_view": final_result_object_view_key(obj),
                    "projection_axis": str(obj.get("projection_axis") or ""),
                    "partial_dimension_semantics": str(obj.get("partial_dimension_semantics") or ""),
                    "source_graph": Path(source_graph).name,
                },
            )
        )
    return elements


def multiview_overlay_legend_svg(extra_objects: list[dict[str, Any]]) -> list[str]:
    used_views = set(MAIN_POSTPROCESSED_VIEWS)
    used_views.update(partial_evidence_view_key(obj) for obj in extra_objects)
    used_views.discard("scan_result")
    order = ["top", "bottom", "land", "front", "side", "lead", "land_detail", "lateral", "lead_detail", "unknown"]
    labels = {
        "top": "top",
        "bottom": "bottom",
        "land": "land",
        "front": "front partial evidence",
        "side": "side partial evidence",
        "lead": "lead partial evidence",
        "land_detail": "land detail partial evidence",
        "lateral": "lateral partial evidence",
        "lead_detail": "lead detail partial evidence",
        "unknown": "unknown partial evidence",
    }
    elements = []
    x = 650.0
    y = 30.0
    for index, view in enumerate([item for item in order if item in used_views]):
        color = MAIN_POSTPROCESSED_VIEW_COLORS.get(view) or MULTIVIEW_OVERLAY_VIEW_COLORS.get(view, "#94a3b8")
        item_y = y + index * 24.0
        elements.append(
            f'<rect x="{x}" y="{item_y - 12}" width="16" height="16" '
            f'fill="{color}" opacity="0.24" stroke="{color}"/>'
        )
        elements.append(
            f'<text x="{x + 24}" y="{item_y + 1}" font-size="14" '
            f'font-family="monospace" fill="#334155">{html.escape(labels.get(view, view))}</text>'
        )
    return elements


def write_scan_result_gt_display_svg(
    path: Path,
    objects: list[dict[str, Any]],
    *,
    display_context: dict[str, Any],
) -> None:
    frame = scan_result_gt_display_frame(objects)
    if frame is None:
        path.write_text(main_view_empty_overlay_svg(), encoding="utf-8")
        return
    width = float(display_context.get("width") or DISPLAY_SVG_WIDTH)
    height = float(display_context.get("height") or DISPLAY_SVG_HEIGHT)
    target = tuple(display_context.get("target") or DISPLAY_SVG_TARGET)
    display_scale = float(display_context.get("scale") or overlay_common_scale([frame], target))
    transform = graph_to_overlay_transform(frame, target, scale=display_scale, unit_scales={"x": 1.0, "y": 1.0})
    elements = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        '<text x="24" y="38" font-size="20" font-family="monospace" fill="#0f172a">ScanResult GT</text>',
        '<text x="24" y="64" font-size="13" font-family="monospace" fill="#64748b">'
        "shared display scale with top/bottom/land overlay</text>",
        f'<g data-view="scan_result_gt" data-display-scale="{display_scale:.8f}">',
    ]
    for obj in objects:
        bbox = object_bbox(obj)
        if bbox is None:
            continue
        mapped = map_bbox(bbox, transform)
        elements.append(gt_display_object_svg(obj, mapped))
    elements.append("</g>")
    path.write_text(
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{int(width)}" height="{int(height)}">'
            + "".join(elements)
            + "</svg>\n"
        ),
        encoding="utf-8",
    )


def write_final_result_display_svg(
    path: Path,
    objects: list[dict[str, Any]],
    *,
    display_context: dict[str, Any],
) -> None:
    frame = final_result_display_frame(objects)
    if frame is None:
        path.write_text(main_view_empty_overlay_svg(), encoding="utf-8")
        return
    width = float(display_context.get("width") or DISPLAY_SVG_WIDTH)
    height = float(display_context.get("height") or DISPLAY_SVG_HEIGHT)
    target = tuple(display_context.get("target") or DISPLAY_SVG_TARGET)
    display_scale = float(display_context.get("scale") or overlay_common_scale([frame], target))
    transform = graph_to_overlay_transform(frame, target, scale=display_scale, unit_scales={"x": 1.0, "y": 1.0})
    elements = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        '<text x="24" y="38" font-size="20" font-family="monospace" fill="#0f172a">Multi-view overlay</text>',
        '<text x="24" y="64" font-size="13" font-family="monospace" fill="#64748b">'
        "result objects only; colors indicate source view, not GT reference geometry</text>",
        final_result_legend_svg(objects),
        f'<g data-view="final_result" data-display-scale="{display_scale:.8f}">',
    ]
    for obj in objects:
        bbox = object_bbox(obj)
        if bbox is None:
            continue
        mapped = map_bbox(bbox, transform)
        elements.append(final_result_object_svg(obj, mapped))
    elements.append("</g>")
    path.write_text(
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{int(width)}" height="{int(height)}">'
            + "".join(elements)
            + "</svg>\n"
        ),
        encoding="utf-8",
    )


def gt_display_object_svg(obj: dict[str, Any], bbox: tuple[float, float, float, float]) -> str:
    role = str(obj.get("role") or "unknown")
    color = {
        "land": "#15803d",
        "lead": "#b45309",
        "shape": "#dc2626",
        "outline_or_line": "#dc2626",
    }.get(role, "#ef4444")
    x1, y1, x2, y2 = bbox
    width = max(x2 - x1, 0.0)
    height = max(y2 - y1, 0.0)
    return (
        f'<rect x="{x1:.3f}" y="{y1:.3f}" width="{width:.3f}" height="{height:.3f}" '
        f'fill="{color}" fill-opacity="0.08" stroke="{color}" stroke-width="2" opacity="0.95"/>'
    )


def final_result_object_svg(obj: dict[str, Any], bbox: tuple[float, float, float, float]) -> str:
    role = str(obj.get("role") or "unknown")
    view_key = final_result_object_view_key(obj)
    color = MULTIVIEW_OVERLAY_VIEW_COLORS.get(view_key, MULTIVIEW_OVERLAY_VIEW_COLORS["unknown"])
    fill = "none" if role in {"outline", "outline_2d"} else color
    fill_opacity = "0.00" if fill == "none" else "0.16"
    stroke_width = "3" if role in {"outline", "outline_2d"} else "2"
    if role in {"lead_pad", "inner_land_pad"}:
        fill_opacity = "0.28"
        stroke_width = "3"
    x1, y1, x2, y2 = bbox
    width = max(x2 - x1, 0.0)
    height = max(y2 - y1, 0.0)
    return (
        f'<rect x="{x1:.3f}" y="{y1:.3f}" width="{width:.3f}" height="{height:.3f}" '
        f'fill="{fill}" fill-opacity="{fill_opacity}" stroke="{color}" '
        f'stroke-width="{stroke_width}" opacity="0.95" data-role="{html.escape(role)}" '
        f'data-source-view="{html.escape(view_key)}"/>'
    )


def final_result_object_view_key(obj: dict[str, Any]) -> str:
    source_type = str(obj.get("source_type") or "")
    if source_type == "scan_result_format":
        return "scan_result"
    if str(obj.get("role") or "") == "lead_pad":
        source = obj.get("lead_contact_length_source") or {}
        source_view = normalized_multiview_color_key(str(source.get("canonical_view") or source.get("raw_view") or ""))
        if source_view != "unknown":
            return source_view
    if str(obj.get("role") or "") == "inner_land_pad":
        source = obj.get("inner_land_pad_source") or {}
        raw_view = str(source.get("raw_view") or "").strip().lower()
        if raw_view in MULTIVIEW_OVERLAY_VIEW_COLORS:
            return raw_view
        source_view = normalized_multiview_color_key(str(source.get("canonical_view") or raw_view or ""))
        if source_view != "unknown":
            return source_view
    view = normalized_multiview_color_key(str(obj.get("canonical_view") or obj.get("raw_view") or ""))
    return view


def partial_evidence_view_key(obj: dict[str, Any]) -> str:
    if str(obj.get("source_type") or "") == "scan_result_format":
        return "scan_result"
    source = obj.get("inner_land_pad_source") or obj.get("lead_contact_length_source") or {}
    raw_view = str(source.get("raw_view") or obj.get("raw_view") or "").strip().lower()
    if raw_view in {"front", "side", "lead", "land_detail"}:
        return raw_view
    return final_result_object_view_key(obj)


def normalized_multiview_color_key(view: str) -> str:
    value = view.strip().lower()
    if value in {"side", "front"}:
        return "lateral"
    if value in {"lead", "land_detail"}:
        return "lead_detail"
    if value in MULTIVIEW_OVERLAY_VIEW_COLORS:
        return value
    return "unknown"


def final_result_legend_svg(objects: list[dict[str, Any]]) -> str:
    used_views = []
    seen = set()
    for obj in objects:
        view_key = final_result_object_view_key(obj)
        if view_key in seen:
            continue
        seen.add(view_key)
        used_views.append(view_key)
    order = ["top", "bottom", "land", "land_detail", "lateral", "lead_detail", "scan_result", "unknown"]
    used_views = sorted(used_views, key=lambda value: order.index(value) if value in order else len(order))
    if not used_views:
        return ""
    elements = []
    x = 24.0
    y = 96.0
    for index, view_key in enumerate(used_views):
        color = MULTIVIEW_OVERLAY_VIEW_COLORS.get(view_key, MULTIVIEW_OVERLAY_VIEW_COLORS["unknown"])
        item_x = x + (index % 4) * 260.0
        item_y = y + (index // 4) * 24.0
        label = {
            "top": "top",
            "bottom": "bottom",
            "land": "land",
            "land_detail": "land detail",
            "lateral": "lateral (side/front)",
            "lead_detail": "lead detail",
            "scan_result": "ScanResult fallback",
            "unknown": "unknown",
        }.get(view_key, view_key)
        elements.append(
            f'<rect x="{item_x:.1f}" y="{item_y - 12:.1f}" width="16" height="16" '
            f'fill="{color}" fill-opacity="0.22" stroke="{color}" stroke-width="2"/>'
        )
        elements.append(
            f'<text x="{item_x + 24:.1f}" y="{item_y + 1:.1f}" font-size="14" '
            f'font-family="monospace" fill="#334155">{html.escape(label)}</text>'
        )
    return "".join(elements)


def graph_frame(graph: dict[str, Any]) -> tuple[float, float, float, float] | None:
    outline_boxes = []
    object_boxes = []
    for obj in graph.get("objects") or []:
        bbox = object_bbox(obj)
        if bbox is None:
            continue
        object_boxes.append(bbox)
        label = " ".join(str(obj.get(key) or "") for key in ("label", "source_label")).lower()
        if "outline" in label or "package" in label:
            outline_boxes.append(bbox)
    return union_boxes(outline_boxes or object_boxes)


def graph_display_frame(graph: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Return the footprint frame used only for review display zoom.

    Coordinate system: package graph reconstructed coordinates.  Pad-like
    objects determine the display zoom so large outlines do not make the
    terminal geometry unreadably small.  Falls back to graph_frame when there
    are no pad-like objects.
    """
    pad_boxes = []
    for obj in graph.get("objects") or []:
        bbox = object_bbox(obj)
        if bbox is None:
            continue
        if object_is_pad_like(obj):
            pad_boxes.append(bbox)
    return union_boxes(pad_boxes) or graph_frame(graph)


def scan_result_gt_display_frame(objects: list[dict[str, Any]]) -> tuple[float, float, float, float] | None:
    """Return the ScanResult GT frame used only for review display.

    Coordinate system: ScanResultFormat reference units.  The frame must
    include all GT roles so body/shape geometry is not clipped when pads are
    only on one side of the package.
    """
    all_boxes = []
    for obj in objects:
        bbox = object_bbox(obj)
        if bbox is None:
            continue
        all_boxes.append(bbox)
    return union_boxes(all_boxes)


def scan_result_gt_scale_frame(objects: list[dict[str, Any]]) -> tuple[float, float, float, float] | None:
    """Return the ScanResult footprint frame used for shared review scale.

    Coordinate system: ScanResultFormat reference units.  Land and lead
    geometry determine zoom so terminal geometry remains inspectable.  This is
    only for selecting shared display scale; drawing uses the full GT frame.
    """
    footprint_boxes = []
    all_boxes = []
    for obj in objects:
        bbox = object_bbox(obj)
        if bbox is None:
            continue
        all_boxes.append(bbox)
        if str(obj.get("role") or "") in {"land", "lead"}:
            footprint_boxes.append(bbox)
    return union_boxes(footprint_boxes) or union_boxes(all_boxes)


def final_result_display_frame(objects: list[dict[str, Any]]) -> tuple[float, float, float, float] | None:
    footprint_boxes = []
    all_boxes = []
    for obj in objects:
        bbox = object_bbox(obj)
        if bbox is None:
            continue
        all_boxes.append(bbox)
        if str(obj.get("role") or "") in {
            "package_pad",
            "land",
            "land_pad",
            "lead",
            "lead_contact",
            "lead_pad",
            "inner_land_pad",
        }:
            footprint_boxes.append(bbox)
    return union_boxes(footprint_boxes) or union_boxes(all_boxes)


def object_bbox(obj: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = obj.get("bbox_reconstructed") or obj.get("bbox") or []
    if len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def object_is_pad_like(obj: dict[str, Any]) -> bool:
    label = " ".join(str(obj.get(key) or "") for key in ("label", "source_label", "shape")).lower()
    if "outline" in label or "package" in label:
        return False
    return any(token in label for token in ("pad", "rect", "circle", "dshape"))


def union_boxes(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float] | None:
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def graph_dimension_unit_scales(graph: dict[str, Any]) -> dict[str, Any]:
    """Estimate graph coordinate units from accepted dimensions.

    Coordinate system: package graph reconstructed coordinates are measured in
    graph pixels. Dimension values are treated as the physical unit currently
    carried by the package graph value parser/table lookup. The return values
    are physical units per graph pixel on x/y axes.
    """
    objects_by_id = {str(obj.get("id")): obj for obj in graph.get("objects") or [] if obj.get("id") is not None}
    samples: dict[str, list[float]] = {"x": [], "y": []}
    for dim in graph.get("dimensions") or []:
        if str(dim.get("status") or "") != "accepted":
            continue
        axis = str(dim.get("axis") or "").lower()
        if axis not in samples:
            continue
        value = positive_float(dim.get("value"))
        if value is None:
            continue
        pixel_distance = dimension_pixel_distance(dim, objects_by_id)
        if pixel_distance is None or pixel_distance <= 0.0:
            continue
        samples[axis].append(value / pixel_distance)

    all_samples = samples["x"] + samples["y"]
    fallback = median_positive(all_samples) or 1.0
    x_scale = median_positive(samples["x"]) or fallback
    y_scale = median_positive(samples["y"]) or fallback
    return {
        "x": x_scale,
        "y": y_scale,
        "source": "accepted_dimensions" if all_samples else "graph_pixels",
        "x_sample_count": len(samples["x"]),
        "y_sample_count": len(samples["y"]),
    }


def positive_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result <= 0.0:
        return None
    return result


def median_positive(values: list[float]) -> float | None:
    positives = sorted(value for value in values if value > 0.0)
    if not positives:
        return None
    mid = len(positives) // 2
    if len(positives) % 2:
        return positives[mid]
    return (positives[mid - 1] + positives[mid]) / 2.0


def dimension_pixel_distance(dim: dict[str, Any], objects_by_id: dict[str, dict[str, Any]]) -> float | None:
    axis = str(dim.get("axis") or "").lower()
    target_ids = [str(item) for item in dim.get("target_ids") or []]
    if axis not in {"x", "y"} or not target_ids:
        return None
    target_boxes = [object_bbox(objects_by_id[target_id]) for target_id in target_ids if target_id in objects_by_id]
    target_boxes = [bbox for bbox in target_boxes if bbox is not None]
    if not target_boxes:
        return None
    if len(target_boxes) == 1:
        x1, y1, x2, y2 = target_boxes[0]
        return (x2 - x1) if axis == "x" else (y2 - y1)

    anchors = list(dim.get("anchors") or [])
    anchor_a = str(anchors[0] if len(anchors) >= 1 else "center")
    anchor_b = str(anchors[1] if len(anchors) >= 2 else "center")
    first = anchor_coordinate(target_boxes[0], axis, anchor_a)
    second = anchor_coordinate(target_boxes[1], axis, anchor_b)
    return abs(second - first)


def anchor_coordinate(bbox: tuple[float, float, float, float], axis: str, anchor: str) -> float:
    x1, y1, x2, y2 = bbox
    normalized = anchor.lower()
    if axis == "x":
        if normalized == "left_edge":
            return x1
        if normalized == "right_edge":
            return x2
        return (x1 + x2) / 2.0
    if normalized == "top_edge":
        return y1
    if normalized == "bottom_edge":
        return y2
    return (y1 + y2) / 2.0


def calibrated_frame(
    frame: tuple[float, float, float, float],
    unit_scales: dict[str, Any],
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = frame
    unit_x = float(unit_scales["x"])
    unit_y = float(unit_scales["y"])
    return (x1 * unit_x, y1 * unit_y, x2 * unit_x, y2 * unit_y)


def overlay_common_scale(
    frames: list[tuple[float, float, float, float]],
    target: tuple[float, float, float, float],
) -> float:
    """Return one SVG scale shared by all overlaid views.

    Coordinate system: source frames are reconstructed package graph coordinates;
    target is SVG pixel coordinates.  The scale is based on the maximum source
    width/height across all frames so smaller views remain visibly smaller
    instead of being independently normalized.
    """
    tx1, ty1, tx2, ty2 = target
    target_w = max(tx2 - tx1, 1.0)
    target_h = max(ty2 - ty1, 1.0)
    min_source_size = 1e-9
    max_source_w = max(max(frame[2] - frame[0], min_source_size) for frame in frames)
    max_source_h = max(max(frame[3] - frame[1], min_source_size) for frame in frames)
    return min(target_w / max_source_w, target_h / max_source_h)


def graph_to_overlay_transform(
    frame: tuple[float, float, float, float],
    target: tuple[float, float, float, float],
    *,
    scale: float | None = None,
    unit_scales: dict[str, Any] | None = None,
) -> dict[str, float]:
    fx1, fy1, fx2, fy2 = frame
    tx1, ty1, tx2, ty2 = target
    source_w = max(fx2 - fx1, 1.0)
    source_h = max(fy2 - fy1, 1.0)
    target_w = max(tx2 - tx1, 1.0)
    target_h = max(ty2 - ty1, 1.0)
    scale = scale if scale is not None else min(target_w / source_w, target_h / source_h)
    source_cx = (fx1 + fx2) / 2.0
    source_cy = (fy1 + fy2) / 2.0
    target_cx = (tx1 + tx2) / 2.0
    target_cy = (ty1 + ty2) / 2.0
    unit_scales = unit_scales or {"x": 1.0, "y": 1.0}
    return {
        "scale": scale,
        "unit_x": float(unit_scales["x"]),
        "unit_y": float(unit_scales["y"]),
        "source_cx": source_cx,
        "source_cy": source_cy,
        "target_cx": target_cx,
        "target_cy": target_cy,
    }


def map_bbox(
    bbox: tuple[float, float, float, float],
    transform: dict[str, float],
) -> tuple[float, float, float, float]:
    scale = transform["scale"]
    unit_x = transform["unit_x"]
    unit_y = transform["unit_y"]
    source_cx = transform["source_cx"]
    source_cy = transform["source_cy"]
    target_cx = transform["target_cx"]
    target_cy = transform["target_cy"]
    x1, y1, x2, y2 = bbox
    return (
        target_cx + (x1 - source_cx) * unit_x * scale,
        target_cy + (y1 - source_cy) * unit_y * scale,
        target_cx + (x2 - source_cx) * unit_x * scale,
        target_cy + (y2 - source_cy) * unit_y * scale,
    )


def overlay_object_svg(obj: dict[str, Any], bbox: tuple[float, float, float, float], *, color: str) -> str:
    x1, y1, x2, y2 = bbox
    width = max(x2 - x1, 0.0)
    height = max(y2 - y1, 0.0)
    label = " ".join(str(obj.get(key) or "") for key in ("label", "source_label", "shape")).lower()
    is_outline = "outline" in label or "package" in label
    fill = "none" if is_outline else color
    opacity = "0.95" if is_outline else "0.20"
    stroke_width = "3" if is_outline else "2"
    if "circle" in label and not is_outline:
        return (
            f'<ellipse cx="{x1 + width / 2.0:.3f}" cy="{y1 + height / 2.0:.3f}" '
            f'rx="{width / 2.0:.3f}" ry="{height / 2.0:.3f}" fill="{fill}" '
            f'stroke="{color}" stroke-width="{stroke_width}" opacity="{opacity}"/>'
        )
    return (
        f'<rect x="{x1:.3f}" y="{y1:.3f}" width="{width:.3f}" height="{height:.3f}" '
        f'fill="{fill}" stroke="{color}" stroke-width="{stroke_width}" opacity="{opacity}"/>'
    )


def overlay_rect(
    bbox: tuple[float, float, float, float],
    *,
    stroke: str,
    width: int,
    fill: str = "none",
    fill_opacity: float = 0.0,
    dash: str = "",
    data_attrs: dict[str, str] | None = None,
) -> str:
    x1, y1, x2, y2 = bbox
    rect_width = max(x2 - x1, 0.0)
    rect_height = max(y2 - y1, 0.0)
    dash_attr = f' stroke-dasharray="{html.escape(dash)}"' if dash else ""
    data_attr_text = ""
    for key, value in (data_attrs or {}).items():
        data_attr_text += f' data-{html.escape(key.replace("_", "-"))}="{html.escape(str(value))}"'
    return (
        f'<rect x="{x1:.3f}" y="{y1:.3f}" width="{rect_width:.3f}" height="{rect_height:.3f}" '
        f'fill="{fill}" fill-opacity="{fill_opacity:.3f}" stroke="{stroke}" '
        f'stroke-width="{width}" opacity="0.95"{dash_attr}{data_attr_text}/>'
    )


def main_view_empty_overlay_svg() -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 360">'
        '<rect width="720" height="360" fill="#ffffff"/>'
        '<text x="360" y="180" text-anchor="middle" font-family="monospace" font-size="18" fill="#64748b">'
        "no top/bottom/land package graph geometry"
        "</text></svg>\n"
    )


def source_image_evidence_from_dataset_part(dataset_part_dir: Path, fullflow_root: Path, *, limit: int = 8) -> list[dict[str, str]]:
    if not str(dataset_part_dir):
        return []
    extract_dir = dataset_part_dir / "extract_image"
    if not extract_dir.exists():
        return []
    sources = []
    for image_path in sorted(extract_dir.iterdir(), key=lambda path: path.name):
        if len(sources) >= limit:
            break
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        raw_view = view_from_source_image_name(image_path)
        sources.append(
            {
                "label": f"Source {raw_view} (dataset)",
                "path": str(image_path),
                "url": root_relative_url(image_path, fullflow_root),
                "raw_view": raw_view,
                "canonical_view": "dataset",
                "graph_path": "",
            }
        )
    return sources


def view_from_source_image_name(path: Path) -> str:
    parts = path.stem.split("_")
    if len(parts) >= 2:
        return parts[-2].lower()
    return "unknown"


def source_image_unavailable_placeholder(part_number: str) -> dict[str, str]:
    safe_part = html.escape(str(part_number or "unknown"))
    svg = (
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"720\" height=\"420\" viewBox=\"0 0 720 420\">"
        "<rect width=\"720\" height=\"420\" fill=\"#f8fafc\"/>"
        "<rect x=\"24\" y=\"24\" width=\"672\" height=\"372\" fill=\"none\" stroke=\"#94a3b8\" stroke-width=\"3\"/>"
        "<text x=\"360\" y=\"190\" text-anchor=\"middle\" font-family=\"monospace\" font-size=\"28\" fill=\"#334155\">"
        "Source image unavailable"
        "</text>"
        f"<text x=\"360\" y=\"235\" text-anchor=\"middle\" font-family=\"monospace\" font-size=\"20\" fill=\"#64748b\">{safe_part}</text>"
        "</svg>"
    )
    return {
        "label": "Source unavailable (dataset)",
        "path": "",
        "url": "data:image/svg+xml;charset=utf-8," + quote(svg),
        "raw_view": "unavailable",
        "canonical_view": "dataset",
        "graph_path": "",
    }


def group_by_reason(items: list[ReviewItem]) -> dict[str, list[ReviewItem]]:
    grouped: dict[str, list[ReviewItem]] = defaultdict(list)
    for item in items:
        reasons = item.risk_reasons or ["unknown"]
        for reason in reasons:
            grouped[reason].append(item)
    return dict(sorted(grouped.items()))


def group_by_risk(items: list[ReviewItem]) -> dict[str, list[ReviewItem]]:
    grouped: dict[str, list[ReviewItem]] = defaultdict(list)
    for item in items:
        grouped[item.risk_level or "unknown"].append(item)
    return {key: grouped[key] for key in ("high", "medium", "low") if key in grouped}


def group_by_status(items: list[ReviewItem]) -> dict[str, list[ReviewItem]]:
    grouped: dict[str, list[ReviewItem]] = defaultdict(list)
    for item in items:
        grouped[str(item.metrics.get("status") or "unknown")].append(item)
    return dict(sorted(grouped.items()))


def group_by_source(items: list[ReviewItem]) -> dict[str, list[ReviewItem]]:
    grouped: dict[str, list[ReviewItem]] = defaultdict(list)
    for item in items:
        sources = item.metadata.get("error_sources") or ["unknown"]
        for source in sources:
            grouped[str(source)].append(item)
    return dict(sorted(grouped.items()))


def group_by_objective_source(items: list[ReviewItem]) -> dict[str, list[ReviewItem]]:
    grouped: dict[str, list[ReviewItem]] = defaultdict(list)
    for item in items:
        sources = item.metadata.get("objective_error_sources") or ["unknown"]
        for source in sources:
            grouped[str(source)].append(item)
    return dict(sorted(grouped.items()))


def group_by_stage_hint(items: list[ReviewItem]) -> dict[str, list[ReviewItem]]:
    grouped: dict[str, list[ReviewItem]] = defaultdict(list)
    for item in items:
        stage_hints = item.metadata.get("stage_hints") or ["unknown"]
        for stage_hint in stage_hints:
            grouped[str(stage_hint)].append(item)
    return dict(sorted(grouped.items()))


def group_by_check(items: list[ReviewItem]) -> dict[str, list[ReviewItem]]:
    grouped: dict[str, list[ReviewItem]] = defaultdict(list)
    for item in items:
        mismatch_checks = item.metadata.get("mismatch_checks") or []
        if not mismatch_checks:
            grouped["unknown"].append(item)
            continue
        for check in mismatch_checks:
            check_name = str(check.get("name") or "unknown")
            stage_hint = str(check.get("stage_hint") or "unknown")
            grouped[f"{check_name}__{stage_hint}"].append(item)
    return dict(sorted(grouped.items()))


def group_by_review_bucket(items: list[ReviewItem]) -> dict[str, list[ReviewItem]]:
    grouped: dict[str, list[ReviewItem]] = defaultdict(list)
    for item in items:
        grouped[str(item.metadata.get("review_bucket") or "unknown")].append(item)
    return dict(sorted(grouped.items()))


def write_pages(
    *,
    output_root: Path,
    subdir: str,
    grouped: dict[str, list[ReviewItem]],
    notes_rel_path: str,
    history_rel_path: str,
    run_id: str,
    gallery_id: str,
    title_prefix: str,
    static_prefix: str,
) -> list[Path]:
    page_paths = []
    for label, items in grouped.items():
        ordered_items = sort_review_items_by_risk(items)
        page_path = output_root / subdir / f"{slugify(label)}.html"
        page_path.parent.mkdir(parents=True, exist_ok=True)
        page_static_prefix = page_relative_url(static_prefix, page_path.parent)
        cards = "\n".join(render_card(item, page_dir=page_path.parent) for item in enumerate_items(ordered_items))
        config = {
            "notesPath": notes_rel_path,
            "historyPath": history_rel_path,
            "runId": run_id,
            "galleryId": gallery_id,
            "pageIndex": label,
        }
        body = f"""
  <header>
    <div class="topbar">
      <div><h1>{title_prefix}: {label}</h1></div>
      <div class="actions"><a class="button" href="../index.html">Index</a></div>
    </div>
    <div class="chips">
      <span class="chip">Items: {len(items)}</span>
      <span class="chip">Reviewed: <span data-reviewed-count>0</span></span>
      <span class="chip" id="review-server-state">notes server ready</span>
    </div>
  </header>
  <main><section class="cases">{cards or '<p class="muted">No items.</p>'}</section></main>
  <script>window.REVIEW_CONFIG = {json.dumps(config, ensure_ascii=False)};</script>
  <script src="{page_static_prefix}/review.js"></script>
"""
        page_path.write_text(
            render_page_shell(f"{title_prefix}: {label}", body, static_prefix=page_static_prefix),
            encoding="utf-8",
        )
        page_paths.append(page_path)
    return page_paths


def sort_review_items_by_risk(items: list[ReviewItem]) -> list[ReviewItem]:
    return sorted(items, key=lambda item: (-float(item.risk_score or 0.0), item.part_number, item.file_name))


def enumerate_items(items: list[ReviewItem]) -> list[ReviewItem]:
    enumerated = []
    for index, item in enumerate(items, start=1):
        enumerated.append(
            ReviewItem(
                case_id=item.case_id,
                title=item.title,
                rank=index,
                part_number=item.part_number,
                file_name=item.file_name,
                view=item.view,
                risk_score=item.risk_score,
                risk_level=item.risk_level,
                risk_reasons=item.risk_reasons,
                media=item.media,
                links=item.links,
                metrics=item.metrics,
                metadata=item.metadata,
            )
        )
    return enumerated
