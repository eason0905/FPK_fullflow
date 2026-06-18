from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from real_image_process.FPK_PJ_fullflow.review.schema import slugify


OBJECTIVE_ERROR_SOURCE_KEYS = (
    "model_prediction",
    "table_lookup",
    "package_graph_reconstruction",
    "multiview_alignment",
    "scan_result_parsing",
    "gt_annotation_issue",
)

MAIN_ALIGNMENT_VIEWS = ("top", "bottom", "land")
FOLLOW_ALIGNMENT_VIEWS = {
    "side": ("top", "bottom"),
    "front": ("top", "bottom"),
    "lead": ("top", "bottom"),
    "land_detail": ("land",),
}
ROTATION_CANDIDATES = (0, 90, 270, 180)
ALIGNMENT_FALLBACK_PRIORITY_BY_VIEW = {
    "land": 0,
    "land_detail": 0,
    "front": 1,
    "side": 1,
    "lead": 1,
    "bottom": 2,
    "top": 3,
}
ALIGNMENT_VIEW_ORDER = ("land", "land_detail", "front", "side", "lead", "bottom", "top", "unknown")
UNIFIED_MULTIVIEW_LAYERS_FILENAME = "unified_multiview_layers.json"


@dataclass(frozen=True)
class AlignmentOptions:
    count_tolerance: int = 0
    bbox_rel_tol: float = 0.15
    bbox_abs_tol: float = 0.1
    scan_dedupe_center_tol: float = 0.01
    known_issues_path: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "AlignmentOptions":
        payload = payload or {}
        return cls(
            count_tolerance=int(payload.get("count_tolerance", 0)),
            bbox_rel_tol=float(payload.get("bbox_rel_tol", 0.15)),
            bbox_abs_tol=float(payload.get("bbox_abs_tol", 0.1)),
            scan_dedupe_center_tol=float(payload.get("scan_dedupe_center_tol", 0.01)),
            known_issues_path=str(payload.get("known_issues_path") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "count_tolerance": self.count_tolerance,
            "bbox_rel_tol": self.bbox_rel_tol,
            "bbox_abs_tol": self.bbox_abs_tol,
            "scan_dedupe_center_tol": self.scan_dedupe_center_tol,
            "known_issues_path": self.known_issues_path,
        }


def evaluate_alignment(
    *,
    dataset_root: Path,
    multiview_root: Path,
    output_root: Path,
    options: AlignmentOptions | None = None,
    limit: int = 0,
) -> dict[str, Any]:
    options = options or AlignmentOptions()
    output_root.mkdir(parents=True, exist_ok=True)
    part_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    if limit > 0:
        part_dirs = part_dirs[:limit]
    known_issue_rows = load_known_issue_rows(Path(options.known_issues_path)) if options.known_issues_path else []

    summaries = []
    mismatch_rows = []
    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    stage_hint_counts: Counter[str] = Counter()
    error_source_counts: Counter[str] = Counter()
    objective_error_source_counts: Counter[str] = Counter()
    score_diagnostic_counts: Counter[str] = Counter()
    score_stage_hint_counts: Counter[str] = Counter()
    score_objective_error_source_counts: Counter[str] = Counter()
    alignment_transform_strategy_counts: Counter[str] = Counter()
    review_bucket_counts: Counter[str] = Counter()
    review_bucket_risk_counts: dict[str, Counter[str]] = {}
    mapping_counts: Counter[str] = Counter()
    mismatch_check_counts: Counter[str] = Counter()
    count_delta_histograms: dict[str, Counter[int]] = {}
    stage_hint_reason_counts: Counter[str] = Counter()
    for dataset_part_dir in part_dirs:
        part_number = dataset_part_dir.name
        part_multiview_dir = multiview_root / "parts" / slugify(part_number)
        result = evaluate_part(
            part_number=part_number,
            dataset_part_dir=dataset_part_dir,
            canonical_path=multiview_layers_path(part_multiview_dir),
            output_part_dir=output_root / "parts" / slugify(part_number),
            options=options,
        )
        summary = result["summary"]
        attach_known_data_issues(summary, known_issue_rows)
        review_bucket = review_bucket_for_summary(summary)
        review_risk_level = review_risk_level_for_summary(summary)
        summary["review_bucket"] = review_bucket
        summary["review_risk_level"] = review_risk_level
        review_bucket_counts[review_bucket] += 1
        review_bucket_risk_counts.setdefault(review_bucket, Counter())[review_risk_level] += 1
        summaries.append(summary)
        status_counts[summary["status"]] += 1
        for reason in summary.get("reasons") or []:
            reason_counts[reason] += 1
        for stage_hint in summary.get("stage_hints") or []:
            stage_hint_counts[stage_hint] += 1
        for error_source in summary.get("error_sources") or []:
            error_source_counts[error_source] += 1
        for error_source in summary.get("objective_error_sources") or []:
            objective_error_source_counts[error_source] += 1
        for reason in summary.get("score_diagnostics") or []:
            score_diagnostic_counts[reason] += 1
        for stage_hint in summary.get("score_stage_hints") or []:
            score_stage_hint_counts[stage_hint] += 1
        for error_source in summary.get("score_objective_error_sources") or []:
            score_objective_error_source_counts[error_source] += 1
        transform_strategy = (summary.get("alignment_transform") or {}).get("strategy")
        if transform_strategy:
            alignment_transform_strategy_counts[str(transform_strategy)] += 1
        for check in summary.get("checks") or []:
            if check.get("selected_mapping") and check.get("selected_mapping") != "direct":
                mapping_counts[str(check.get("selected_mapping"))] += 1
            if check.get("status") != "aligned":
                check_name = str(check.get("name") or "")
                reason = str(check.get("reason") or "")
                stage_hint = str(check.get("stage_hint") or "")
                mismatch_check_counts[check_name] += 1
                stage_hint_reason_counts[f"{stage_hint}|{reason}"] += 1
                delta = check.get("delta")
                if isinstance(delta, int):
                    count_delta_histograms.setdefault(check_name, Counter())[delta] += 1
        if summary["status"] != "aligned":
            mismatch_rows.append(summary)

    conflict_count = sum(int((summary.get("graph") or {}).get("conflict_count") or 0) for summary in summaries)
    conflict_case_count = sum(1 for summary in summaries if int((summary.get("graph") or {}).get("conflict_count") or 0) > 0)
    missing_canonical_view_counts: Counter[str] = Counter()
    missing_view_case_count = 0
    for summary in summaries:
        missing_views = [
            str(view)
            for view in (summary.get("graph") or {}).get("missing_canonical_views") or []
            if view
        ]
        if missing_views:
            missing_view_case_count += 1
            missing_canonical_view_counts.update(missing_views)

    mismatches_path = output_root / "mismatches.jsonl"
    with mismatches_path.open("w", encoding="utf-8") as file:
        for row in mismatch_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    review_bucket_manifest_path = output_root / "review_bucket_manifest.jsonl"
    review_bucket_manifest_count = write_review_bucket_manifest(review_bucket_manifest_path, summaries)
    alignment_score_summary = summarize_alignment_scores(summaries)
    payload = {
        "output_root": str(output_root),
        "summary_path": str(output_root / "summary.json"),
        "mismatches_path": str(mismatches_path),
        "review_bucket_manifest_path": str(review_bucket_manifest_path),
        "review_bucket_manifest_count": review_bucket_manifest_count,
        "dataset_root": str(dataset_root),
        "multiview_root": str(multiview_root),
        "options": options.to_dict(),
        "total_parts": len(summaries),
        "valid_case_count": status_counts.get("aligned", 0) + status_counts.get("mismatch", 0),
        "aligned_parts": status_counts.get("aligned", 0),
        "mismatch_parts": len(mismatch_rows),
        "missing_gt_parts": status_counts.get("missing_gt", 0),
        "missing_canonical_parts": status_counts.get("missing_canonical", 0),
        "gallery_path": str(infer_final_comparison_gallery_path(output_root)),
        "gallery_url": workspace_server_url(infer_final_comparison_gallery_path(output_root)),
        "conflict_count": conflict_count,
        "conflict_case_count": conflict_case_count,
        "missing_view_case_count": missing_view_case_count,
        "missing_canonical_view_counts": dict(sorted(missing_canonical_view_counts.items())),
        "missing_canonical_view_total_count": sum(missing_canonical_view_counts.values()),
        "status_counts": dict(sorted(status_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "stage_hint_counts": dict(sorted(stage_hint_counts.items())),
        "error_source_counts": dict(sorted(error_source_counts.items())),
        "objective_error_source_keys": list(OBJECTIVE_ERROR_SOURCE_KEYS),
        "objective_error_source_counts": dict(sorted(objective_error_source_counts.items())),
        "score_diagnostic_counts": dict(sorted(score_diagnostic_counts.items())),
        "score_stage_hint_counts": dict(sorted(score_stage_hint_counts.items())),
        "score_objective_error_source_counts": dict(sorted(score_objective_error_source_counts.items())),
        "alignment_transform_strategy_counts": dict(sorted(alignment_transform_strategy_counts.items())),
        "review_bucket_counts": dict(sorted(review_bucket_counts.items())),
        "review_bucket_risk_counts": {
            bucket: {level: int(counts.get(level, 0)) for level in ("high", "medium", "low", "unscored")}
            for bucket, counts in sorted(review_bucket_risk_counts.items())
        },
        "algorithm_evaluable_high_risk_count": int(review_bucket_risk_counts.get("algorithm_evaluable", Counter()).get("high", 0)),
        "algorithm_evaluable_medium_risk_count": int(review_bucket_risk_counts.get("algorithm_evaluable", Counter()).get("medium", 0)),
        "data_issue_count": int(review_bucket_counts.get("data_or_gt_issue", 0)),
        "excluded_data_issue_count": int(review_bucket_counts.get("data_or_gt_issue", 0) + review_bucket_counts.get("scan_result_issue", 0)),
        "evidence_limited_high_risk_count": int(review_bucket_risk_counts.get("evidence_limited", Counter()).get("high", 0)),
        "mapping_counts": dict(sorted(mapping_counts.items())),
        "mismatch_check_counts": dict(sorted(mismatch_check_counts.items())),
        "count_delta_histograms": {
            name: {str(delta): count for delta, count in sorted(histogram.items())}
            for name, histogram in sorted(count_delta_histograms.items())
        },
        "stage_hint_reason_counts": dict(sorted(stage_hint_reason_counts.items())),
        "alignment_score_summary": alignment_score_summary,
        "risk_counts": alignment_score_summary["risk_counts"],
        "representative_score_cases": representative_score_cases(summaries),
        "parts": summaries,
    }
    (output_root / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def infer_final_comparison_gallery_path(output_root: Path) -> Path:
    resolved = output_root.resolve()
    if resolved.name == "gt_alignment" and resolved.parent.name == "eval":
        return resolved.parent.parent / "review" / "final_comparison" / "index.html"
    return resolved / "review" / "final_comparison" / "index.html"


def workspace_server_url(path: Path, workspace_root: Path | None = None) -> str:
    resolved = path.resolve()
    parts = resolved.parts
    if "FPK_PJ_fullflow" in parts:
        index = parts.index("FPK_PJ_fullflow")
        return "/" + Path(*parts[index + 1 :]).as_posix()
    workspace = (workspace_root or Path.cwd()).resolve()
    try:
        return "/" + resolved.relative_to(workspace).as_posix()
    except ValueError:
        return "/" + resolved.as_posix().lstrip("/")


def write_review_bucket_manifest(path: Path, summaries: list[dict[str, Any]]) -> int:
    rows = review_bucket_manifest_rows(summaries)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def review_bucket_manifest_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for summary in summaries:
        review_risk_level = str(summary.get("review_risk_level") or review_risk_level_for_summary(summary))
        if review_risk_level not in {"high", "medium"}:
            continue
        alignment_score_payload = summary.get("alignment_scores") or {}
        review_quality_score = alignment_review_quality_score(alignment_score_payload)
        selected_quality_score = alignment_quality_score(alignment_score_payload)
        rows.append(
            {
                "part_number": summary.get("part_number"),
                "status": summary.get("status"),
                "review_bucket": summary.get("review_bucket") or review_bucket_for_summary(summary),
                "review_risk_level": review_risk_level,
                "quality_score": review_quality_score,
                "review_quality_score": review_quality_score,
                "selected_quality_score": selected_quality_score,
                "overall_score": alignment_score_payload.get("overall_score"),
                "reasons": summary.get("reasons") or [],
                "stage_hints": summary.get("stage_hints") or [],
                "score_diagnostics": summary.get("score_diagnostics") or [],
                "score_stage_hints": summary.get("score_stage_hints") or [],
                "error_sources": summary.get("error_sources") or [],
                "objective_error_sources": summary.get("objective_error_sources") or [],
                "score_objective_error_sources": summary.get("score_objective_error_sources") or [],
                "dataset_part_dir": summary.get("dataset_part_dir"),
                "scan_result_path": summary.get("scan_result_path"),
                "unified_multiview_layers_path": summary.get("unified_multiview_layers_path"),
                "alignment_path": summary.get("alignment_path"),
                "final_graph_path": summary.get("final_graph_path"),
            }
        )
    risk_rank = {"high": 0, "medium": 1}
    return sorted(
        rows,
        key=lambda row: (
            risk_rank.get(str(row.get("review_risk_level")), 9),
            str(row.get("review_bucket") or ""),
            float(row["review_quality_score"]) if isinstance(row.get("review_quality_score"), (int, float)) else 999.0,
            str(row.get("part_number") or ""),
        ),
    )


def summarize_alignment_scores(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    scores: list[float] = []
    overall_scores: list[float] = []
    risk_counts: Counter[str] = Counter()
    unscored = 0
    for summary in summaries:
        alignment_score_payload = summary.get("alignment_scores") or {}
        score = alignment_review_quality_score(alignment_score_payload)
        overall_score = alignment_score_payload.get("overall_score")
        if isinstance(score, (int, float)):
            numeric_score = float(score)
            scores.append(numeric_score)
            risk_counts[alignment_risk_level(numeric_score)] += 1
            if isinstance(overall_score, (int, float)):
                overall_scores.append(float(overall_score))
            continue
        unscored += 1
        risk_counts["high"] += 1
    return {
        "total_parts": len(summaries),
        "scored_parts": len(scores),
        "unscored_parts": unscored,
        "min_score": round(min(scores), 6) if scores else None,
        "mean_score": round(sum(scores) / len(scores), 6) if scores else None,
        "max_score": round(max(scores), 6) if scores else None,
        "overall_min_score": round(min(overall_scores), 6) if overall_scores else None,
        "overall_mean_score": round(sum(overall_scores) / len(overall_scores), 6) if overall_scores else None,
        "overall_max_score": round(max(overall_scores), 6) if overall_scores else None,
        "risk_counts": {level: int(risk_counts.get(level, 0)) for level in ("high", "medium", "low")},
    }


def review_risk_level_for_summary(summary: dict[str, Any]) -> str:
    score = alignment_review_quality_score(summary.get("alignment_scores") or {})
    if not isinstance(score, (int, float)):
        return "unscored"
    return alignment_risk_level(float(score))


def review_bucket_for_summary(summary: dict[str, Any]) -> str:
    """Classify a part for review triage without changing score/risk.

    This is intentionally deterministic and conservative.  A case remains high
    risk in the gallery even when it is classified as evidence-limited or
    upstream; this bucket only answers whether the current failure is a direct
    multiview-alignment algorithm target.
    """
    risk_level = review_risk_level_for_summary(summary)
    if risk_level == "low":
        return "low_risk"
    if summary.get("known_data_issues"):
        return "data_or_gt_issue"

    hints = {
        str(hint)
        for hint in (summary.get("stage_hints") or []) + (summary.get("score_stage_hints") or [])
        if hint
    }
    sources = {
        str(source)
        for source in (summary.get("objective_error_sources") or [])
        + (summary.get("score_objective_error_sources") or [])
        if source
    }

    if any(is_data_or_gt_hint(hint) for hint in hints) or "gt_annotation_issue" in sources or "table_lookup" in sources:
        return "data_or_gt_issue"
    if has_primary_pad_source_coverage_gap(summary, hints):
        return "data_or_gt_issue"
    if sources & {"model_prediction", "package_graph_reconstruction"} and any(
        is_package_graph_reconstruction_hint(hint) for hint in hints
    ):
        return "upstream_prediction_or_reconstruction"
    if any(is_evidence_limited_hint(hint) for hint in hints):
        return "evidence_limited"
    if "scan_result_parsing" in sources:
        return "scan_result_issue"
    if sources & {"model_prediction", "package_graph_reconstruction"}:
        return "upstream_prediction_or_reconstruction"
    if "multiview_alignment" in sources:
        return "algorithm_evaluable"
    return "unclassified"


def load_known_issue_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def attach_known_data_issues(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    matches = [row for row in rows if known_issue_matches_summary(row, summary)]
    if not matches:
        return
    summary["known_data_issues"] = matches
    summary["known_data_issue_types"] = sorted({str(row.get("issue_type") or "unknown") for row in matches})


def known_issue_matches_summary(row: dict[str, Any], summary: dict[str, Any]) -> bool:
    if str(row.get("part_number") or "") != str(summary.get("part_number") or ""):
        return False
    file_name = str(row.get("file") or Path(str(row.get("annotation_path") or "")).name)
    if not file_name:
        return True
    return normalized_issue_file_token(file_name) in summary_source_file_tokens(summary)


def normalized_issue_file_token(file_name: str) -> str:
    path = Path(file_name)
    name = path.name
    if name.endswith(".package_graph.json"):
        return name.removesuffix(".package_graph.json") + ".json"
    return name


def summary_source_file_tokens(summary: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    graph = summary.get("graph") or {}
    source_selection = graph.get("source_selection") or {}
    for selection in source_selection.values():
        if not isinstance(selection, dict):
            continue
        graph_path = str(selection.get("graph_path") or "")
        if graph_path:
            tokens.add(normalized_issue_file_token(graph_path))

    canonical_path = summary.get("unified_multiview_layers_path")
    if canonical_path:
        try:
            canonical = json.loads(Path(str(canonical_path)).read_text(encoding="utf-8"))
        except Exception:
            canonical = {}
        for dimension in canonical.get("dimensions") or []:
            annotation_path = str(dimension.get("annotation_path") or "")
            if annotation_path:
                tokens.add(normalized_issue_file_token(annotation_path))
    return tokens


def is_data_or_gt_hint(stage_hint: str) -> bool:
    return bool(
        stage_hint.startswith("data_missing_")
        or stage_hint.startswith("low_score_data_missing_")
        or stage_hint == "scan_result_land_count_exceeds_visible_land_annotation"
    )


def is_evidence_limited_hint(stage_hint: str) -> bool:
    return stage_hint in {
        "low_score_multiview_package_pad_fallback_geometry",
        "low_score_multiview_partial_lead_detail_layout",
        "low_score_multiview_land_pad_proxy_size_mismatch",
        "low_score_multiview_duplicate_lead_geometry_sources",
        "low_score_multiview_lateral_lead_projection_excluded",
    }


def is_package_graph_reconstruction_hint(stage_hint: str) -> bool:
    return stage_hint.startswith("low_score_package_graph_")


def has_primary_pad_source_coverage_gap(summary: dict[str, Any], hints: set[str]) -> bool:
    if "low_score_multiview_package_pad_fallback_geometry" not in hints:
        return False
    missing_views = set(summary.get("graph", {}).get("missing_canonical_views") or [])
    return {"land", "lead_detail"}.issubset(missing_views)


def representative_score_cases(
    summaries: list[dict[str, Any]],
    *,
    limit_per_reason: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    """Pick deterministic examples for each score diagnostic reason.

    Coordinates are not recomputed here.  This summary only points reviewers to
    already-written GT/aligned/comparison artifacts.  Sorting prioritizes the
    worst component value, then the conservative quality score.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for summary in summaries:
        alignment_scores_payload = summary.get("alignment_scores") or {}
        quality_score = alignment_review_quality_score(alignment_scores_payload)
        overall_score = alignment_scores_payload.get("overall_score")
        for detail in summary.get("score_diagnostic_details") or []:
            reason = str(detail.get("reason") or "")
            if not reason:
                continue
            grouped.setdefault(reason, []).append(
                {
                    "part_number": summary.get("part_number"),
                    "status": summary.get("status"),
                    "quality_score": rounded_number_or_none(quality_score),
                    "overall_score": rounded_number_or_none(overall_score),
                    "metric": detail.get("metric"),
                    "metric_value": detail.get("value"),
                    "threshold": detail.get("threshold"),
                    "stage_hint": detail.get("stage_hint"),
                    "objective_error_sources": detail.get("objective_error_sources") or [],
                    "score_diagnostics": summary.get("score_diagnostics") or [],
                    "gt_reference_svg_path": summary.get("gt_reference_svg_path"),
                    "aligned_result_svg_path": summary.get("aligned_result_svg_path"),
                    "comparison_svg_path": summary.get("comparison_svg_path"),
        "unified_multiview_layers_path": summary.get("unified_multiview_layers_path"),
                    "scan_result_path": summary.get("scan_result_path"),
                    "final_graph_path": summary.get("final_graph_path"),
                }
            )

    return {
        reason: sorted(cases, key=representative_case_sort_key)[:limit_per_reason]
        for reason, cases in sorted(grouped.items())
    }


def representative_case_sort_key(row: dict[str, Any]) -> tuple[float, float, str]:
    value = row.get("metric_value")
    metric_rank = -1.0 if value is None else float(value) if isinstance(value, (int, float)) else 2.0
    quality = row.get("quality_score")
    quality_rank = 2.0 if quality is None else float(quality) if isinstance(quality, (int, float)) else 2.0
    return (metric_rank, quality_rank, str(row.get("part_number") or ""))


def rounded_number_or_none(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return round(float(value), 6)


def alignment_risk_level(score: float) -> str:
    if score < 0.5:
        return "high"
    if score < 0.8:
        return "medium"
    return "low"


def alignment_quality_score(alignment_scores: dict[str, Any]) -> float | None:
    """Conservative score for review risk.

    `overall_score` remains the arithmetic aggregate for broad tracking.  Review
    risk must not average away a broken pad layout, so quality uses the minimum
    defined critical component score in the same 0..1 coordinate.
    """
    critical_keys = (
        "overall_score",
        "source_independence_score",
        "outline_iou",
        "land_iou",
        "lead_iou",
        "land_pad_iou_score",
        "lead_pad_iou_score",
        "pad_layout_score",
        "dimension_value_score",
    )
    values = [float(alignment_scores[key]) for key in critical_keys if isinstance(alignment_scores.get(key), (int, float))]
    for aligned in (alignment_scores.get("count_checks") or {}).values():
        if isinstance(aligned, bool):
            values.append(1.0 if aligned else 0.0)
    if not values:
        return None
    return min(values)


def alignment_review_quality_score(alignment_scores: dict[str, Any]) -> float | None:
    review_score = alignment_scores.get("review_quality_score")
    if isinstance(review_score, (int, float)):
        return float(review_score)
    selected_score = alignment_scores.get("quality_score")
    if isinstance(selected_score, (int, float)):
        return float(selected_score)
    return alignment_quality_score(alignment_scores)


def evaluate_part(
    *,
    part_number: str,
    dataset_part_dir: Path,
    canonical_path: Path,
    output_part_dir: Path,
    options: AlignmentOptions,
) -> dict[str, Any]:
    scan_path = dataset_part_dir / "ScanResultFormat.txt"
    output_part_dir.mkdir(parents=True, exist_ok=True)
    reasons: list[str] = []
    if not canonical_path.exists():
        gt = parse_scan_result(scan_path, dedupe_center_tol=options.scan_dedupe_center_tol) if scan_path.exists() else {}
        summary = base_summary(part_number, dataset_part_dir, scan_path, canonical_path, "missing_canonical", ["missing_unified_multiview_layers"])
        summary["stage_hints"] = ["package_graph_reconstruction_missing"]
        summary["error_sources"] = error_sources_for_stage_hints(summary["stage_hints"])
        summary["objective_error_sources"] = objective_error_sources_for_stage_hints(summary["stage_hints"])
        if gt:
            summary["gt"] = gt["summary"]
        write_part_files(output_part_dir, summary, gt, {}, [])
        return {"summary": summary}

    gt = parse_scan_result(scan_path, dedupe_center_tol=options.scan_dedupe_center_tol) if scan_path.exists() else {}
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    if str(canonical.get("status") or "") == "missing_graphs":
        failure_reason = str(canonical.get("failure_reason") or "no_package_graph_for_part")
        summary = base_summary(part_number, dataset_part_dir, scan_path, canonical_path, "missing_canonical", [failure_reason])
        summary["failure_reason"] = failure_reason
        summary["stage_hints"] = ["package_graph_reconstruction_missing"]
        summary["error_sources"] = error_sources_for_stage_hints(summary["stage_hints"])
        summary["objective_error_sources"] = objective_error_sources_for_stage_hints(summary["stage_hints"])
        if gt:
            summary["gt"] = gt["summary"]
        summary["graph"] = canonical_features(canonical)["summary"]
        write_part_files(output_part_dir, summary, gt, canonical, [])
        return {"summary": summary}
    graph_features = canonical_features(canonical)
    layer_alignment = align_overlay_layers(canonical)
    checks = layer_alignment.get("checks") or []
    reasons.extend(check["reason"] for check in checks if check.get("status") != "aligned")
    stage_hints = sorted({check["stage_hint"] for check in checks if check["status"] != "aligned" and check.get("stage_hint")})
    status = "aligned" if not reasons else "mismatch"
    summary = base_summary(part_number, dataset_part_dir, scan_path, canonical_path, status, reasons)
    summary["stage_hints"] = stage_hints
    summary["error_sources"] = error_sources_for_stage_hints(stage_hints)
    summary["objective_error_sources"] = objective_error_sources_for_stage_hints(stage_hints)
    if gt:
        summary["gt"] = gt["summary"]
    summary["graph"] = graph_features["summary"]
    summary["checks"] = checks
    write_part_files(output_part_dir, summary, gt, canonical, checks)
    return {"summary": summary}


def base_summary(
    part_number: str,
    dataset_part_dir: Path,
    scan_path: Path,
    canonical_path: Path,
    status: str,
    reasons: list[str],
) -> dict[str, Any]:
    return {
        "part_number": part_number,
        "status": status,
        "reasons": reasons,
        "stage_hints": [],
        "error_sources": [],
        "dataset_part_dir": str(dataset_part_dir),
        "scan_result_path": str(scan_path),
        "unified_multiview_layers_path": str(canonical_path),
        "alignment_path": "",
    }


def multiview_layers_path(part_dir: Path) -> Path:
    return part_dir / UNIFIED_MULTIVIEW_LAYERS_FILENAME


def parse_scan_result(path: Path, *, dedupe_center_tol: float = 0.01) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    objects = []
    raw_role_counts: Counter[str] = Counter()
    node_counts: Counter[str] = Counter()
    for item in payload.get("Object") or []:
        points = item.get("PointList") or []
        bbox = bbox_from_points(points)
        role = gt_role(item)
        node = str(item.get("NodeName") or "unknown")
        raw_role_counts[role] += 1
        node_counts[node] += 1
        objects.append(
            {
                "id": item.get("ID"),
                "role": role,
                "raw_role": role,
                "node_name": node,
                "geometry": item.get("Geometry"),
                "bbox": bbox,
                "point_count": len(points),
            }
        )
    normalize_scan_result_roles(objects)
    group_items = payload.get("GroupItems") or []
    role_counts = deduped_role_counts(
        objects,
        dedupe_center_tol=dedupe_center_tol,
        skip_large_overlap=bool(group_items),
    )
    effective_objects = effective_scan_result_raw_objects(objects, role_counts, dedupe_center_tol)
    count_candidates = scan_count_candidates(group_items, objects)
    features = {
        "objects": objects,
        "effective_objects": effective_objects,
        "dedupe_center_tol": dedupe_center_tol,
        "raw_role_counts": dict(sorted(raw_role_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "count_candidates": count_candidates,
        "node_counts": dict(sorted(node_counts.items())),
        "bbox": union_bbox([obj["bbox"] for obj in objects if obj.get("bbox")]),
        "bbox_candidates": scan_bbox_candidates(effective_objects),
        "raw_bbox_candidates": scan_bbox_candidates(objects),
    }
    return {
        "path": str(path),
        "features": features,
        "summary": {
            "object_count": len(objects),
            "raw_role_counts": features["raw_role_counts"],
            "role_counts": features["role_counts"],
            "count_candidates": features["count_candidates"],
            "node_counts": features["node_counts"],
            "bbox": features["bbox"],
        },
    }


def normalize_scan_result_roles(objects: list[dict[str, Any]]) -> None:
    """Remove non-terminal markers from effective lead counts.

    Coordinate system: ScanResultFormat physical/reference units. This function
    only uses relative bbox areas within one ScanResult, so the absolute unit is
    irrelevant.

    Some ScanResult files attach LeadData to pin-1 circles or exposed-pad/body
    rectangles. When a package already has many DShape lead terminals, those
    circle/large-rectangle LeadData objects should not be counted as terminal
    lead contacts.
    """
    dshape_leads = [
        obj
        for obj in objects
        if str(obj.get("role") or "") == "lead" and str(obj.get("node_name") or "") == "DShape"
    ]
    if len(dshape_leads) < 8:
        return
    dshape_areas = sorted(bbox_area(obj.get("bbox")) for obj in dshape_leads if bbox_area(obj.get("bbox")) > 0)
    if not dshape_areas:
        return
    median_dshape_area = dshape_areas[len(dshape_areas) // 2]
    if median_dshape_area <= 0:
        return
    for obj in objects:
        if str(obj.get("role") or "") != "lead":
            continue
        node = str(obj.get("node_name") or "")
        area = bbox_area(obj.get("bbox"))
        if node == "Circle":
            obj["role"] = "shape"
            obj["role_override_reason"] = "lead_circle_marker_with_dshape_terminal_array"
        elif node == "Rectangle" and area >= median_dshape_area * 4.0:
            obj["role"] = "shape"
            obj["role_override_reason"] = "large_lead_rectangle_marker_with_dshape_terminal_array"


def gt_role(item: dict[str, Any]) -> str:
    if "LandData" in item:
        return "land"
    if "LeadData" in item:
        return "lead"
    node = str(item.get("NodeName") or "").lower()
    if "line" in node:
        return "outline_or_line"
    if "rectangle" in node or "circle" in node:
        return "shape"
    return "unknown"


def deduped_role_counts(
    objects: list[dict[str, Any]],
    *,
    dedupe_center_tol: float,
    skip_large_overlap: bool = True,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    seen: set[tuple[str, int, int] | tuple[str, str, int]] = set()
    lead_bbox_areas = sorted(
        bbox_area(obj.get("bbox"))
        for obj in objects
        if str(obj.get("role") or "") == "lead" and bbox_area(obj.get("bbox")) > 0.0
    )
    min_lead_area = lead_bbox_areas[0] if lead_bbox_areas else None
    thermal_overlap_center_keys = (
        large_land_lead_overlap_center_keys(objects, min_lead_area, dedupe_center_tol)
        if skip_large_overlap
        else set()
    )
    for index, obj in enumerate(objects):
        role = str(obj.get("role") or "unknown")
        if role not in {"land", "lead"}:
            counts[role] += 1
            continue
        bbox = obj.get("bbox")
        if (
            role in {"land", "lead"}
            and center_dedupe_key(bbox, dedupe_center_tol) in thermal_overlap_center_keys
            and min_lead_area is not None
            and bbox_area(bbox) > min_lead_area * 2.0
        ):
            continue
        if bbox and len(bbox) >= 4 and dedupe_center_tol > 0:
            center_x = (float(bbox[0]) + float(bbox[2])) / 2.0
            center_y = (float(bbox[1]) + float(bbox[3])) / 2.0
            key: tuple[str, int, int] | tuple[str, str, int] = (
                role,
                round(center_x / dedupe_center_tol),
                round(center_y / dedupe_center_tol),
            )
        else:
            key = (role, "raw", index)
        if key in seen:
            continue
        seen.add(key)
        counts[role] += 1
    return counts


def large_land_lead_overlap_center_keys(
    objects: list[dict[str, Any]],
    min_lead_area: float | None,
    tolerance: float,
) -> set[tuple[int, int] | None]:
    if min_lead_area is None:
        return set()
    threshold = min_lead_area * 2.0
    large_land_by_bbox = {
        bbox_dedupe_key(obj.get("bbox"), tolerance): center_dedupe_key(obj.get("bbox"), tolerance)
        for obj in objects
        if str(obj.get("role") or "") == "land" and bbox_area(obj.get("bbox")) > threshold
    }
    large_lead_bbox_keys = {
        bbox_dedupe_key(obj.get("bbox"), tolerance)
        for obj in objects
        if str(obj.get("role") or "") == "lead" and bbox_area(obj.get("bbox")) > threshold
    }
    large_land_by_bbox.pop(None, None)
    large_lead_bbox_keys.discard(None)
    return {large_land_by_bbox[key] for key in large_land_by_bbox.keys() & large_lead_bbox_keys}


def center_dedupe_key(bbox: list[float] | None, tolerance: float) -> tuple[int, int] | None:
    if not bbox or len(bbox) < 4 or tolerance <= 0:
        return None
    center_x = (float(bbox[0]) + float(bbox[2])) / 2.0
    center_y = (float(bbox[1]) + float(bbox[3])) / 2.0
    return (round(center_x / tolerance), round(center_y / tolerance))


def bbox_dedupe_key(bbox: list[float] | None, tolerance: float) -> tuple[int, int, int, int] | None:
    if not bbox or len(bbox) < 4 or tolerance <= 0:
        return None
    return tuple(round(float(value) / tolerance) for value in bbox[:4])


def bbox_area(bbox: list[float] | None) -> float:
    if not bbox or len(bbox) < 4:
        return 0.0
    return max(float(bbox[2]) - float(bbox[0]), 0.0) * max(float(bbox[3]) - float(bbox[1]), 0.0)


def scan_count_candidates(group_items: list[dict[str, Any]], objects: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    objects_by_id = {obj.get("id"): obj for obj in objects}
    candidates: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, int, int, str]] = set()
    for group_index, group in enumerate(group_items):
        cells = scan_group_cells(group.get("FirstMartixPinIDs"))
        raw_role_counts: Counter[str] = Counter()
        cell_role_counts: Counter[str] = Counter()
        object_ids_by_role: dict[str, list[Any]] = {}
        for cell in cells:
            roles = []
            for object_id in cell:
                obj = objects_by_id.get(object_id)
                if not obj:
                    continue
                role = str(obj.get("role") or "unknown")
                if role in {"land", "lead"}:
                    raw_role_counts[role] += 1
                    object_ids_by_role.setdefault(role, []).append(object_id)
                    roles.append(role)
            for role in sorted(set(roles)):
                cell_role_counts[role] += 1
        for candidate_type, role_counts in (
            ("group_raw_object_count", raw_role_counts),
            ("group_matrix_cell_count", cell_role_counts),
        ):
            for role, count in role_counts.items():
                if count <= 0:
                    continue
                key = (role, group_index, count, candidate_type)
                if key in seen:
                    continue
                seen.add(key)
                candidates.setdefault(role, []).append(
                    {
                        "source": "GroupItems",
                        "candidate_type": candidate_type,
                        "group_index": group_index,
                        "count": count,
                        "matrix_qx": group.get("FirstMartixQX"),
                        "matrix_qy": group.get("FirstMartixQY"),
                        "object_ids": object_ids_by_role.get(role, []),
                    }
                )
        matrix_total = matrix_total_count(group)
        if matrix_total is None:
            continue
        for role in sorted(raw_role_counts):
            if raw_role_counts[role] <= 0:
                continue
            key = (role, group_index, matrix_total, "group_matrix_total_count")
            if key in seen:
                continue
            seen.add(key)
            candidates.setdefault(role, []).append(
                {
                    "source": "GroupItems",
                    "candidate_type": "group_matrix_total_count",
                    "group_index": group_index,
                    "count": matrix_total,
                    "matrix_qx": group.get("FirstMartixQX"),
                    "matrix_qy": group.get("FirstMartixQY"),
                    "object_ids": object_ids_by_role.get(role, []),
                }
            )
    return {role: values for role, values in sorted(candidates.items())}


def matrix_total_count(group: dict[str, Any]) -> int | None:
    qx = numeric(group.get("FirstMartixQX"))
    qy = numeric(group.get("FirstMartixQY"))
    if qx is None or qy is None:
        return None
    total = int(round(qx)) * int(round(qy))
    return total if total > 0 else None


def scan_group_cells(value: Any) -> list[list[Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        return [[value]]
    cells: list[list[Any]] = []
    for item in value:
        if item is None:
            continue
        if isinstance(item, list):
            if any(isinstance(child, list) or child is None for child in item):
                cells.extend(scan_group_cells(item))
            else:
                cell = [child for child in item if child is not None]
                if cell:
                    cells.append(cell)
        else:
            cells.append([item])
    return cells


def scan_bbox_candidates(objects: list[dict[str, Any]]) -> dict[str, list[float] | None]:
    by_role: dict[str, list[list[float]]] = {}
    for obj in objects:
        bbox = obj.get("bbox")
        if not bbox:
            continue
        by_role.setdefault(str(obj.get("role") or "unknown"), []).append(bbox)
    return {
        "all": union_bbox([box for boxes in by_role.values() for box in boxes]),
        "shape": union_bbox(by_role.get("shape", [])),
        "conductive": union_bbox(by_role.get("land", []) + by_role.get("lead", [])),
        "land": union_bbox(by_role.get("land", [])),
        "lead": union_bbox(by_role.get("lead", [])),
    }


def bbox_from_points(points: list[dict[str, Any]]) -> list[float] | None:
    coords = [(numeric(point.get("PointX")), numeric(point.get("PointY"))) for point in points]
    coords = [(x, y) for x, y in coords if x is not None and y is not None]
    if not coords:
        return None
    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]
    return [min(xs), min(ys), max(xs), max(ys)]


def canonical_features(canonical: dict[str, Any]) -> dict[str, Any]:
    objects = []
    for role, key in (("outline", "outline_2d"),):
        obj = canonical.get(key) or {}
        if obj:
            objects.append({"role": role, "bbox": obj.get("bbox"), "raw_view": obj.get("raw_view"), "canonical_view": obj.get("canonical_view")})
    for role, key in (("package_pad", "package_pads"), ("land", "land_pads"), ("lead", "lead_contacts")):
        for obj in canonical.get(key) or []:
            objects.append(
                {
                    "role": role,
                    "bbox": obj.get("bbox"),
                    "raw_view": obj.get("raw_view"),
                    "canonical_view": obj.get("canonical_view"),
                    "label": obj.get("label"),
                    "source_label": obj.get("source_label"),
                }
            )
    role_counts = Counter(obj["role"] for obj in objects)
    features = {
        "objects": objects,
        "role_counts": dict(sorted(role_counts.items())),
        "bbox": union_bbox([obj["bbox"] for obj in objects if obj.get("bbox")]),
        "bbox_candidates": canonical_bbox_candidates(objects),
    }
    package_pad_count = int(role_counts.get("package_pad", 0))
    package_pad_rect_count = sum(
        1 for obj in objects if obj.get("role") == "package_pad" and str(obj.get("label") or "").lower() == "rect"
    )
    land_pad_count = int(role_counts.get("land", 0))
    lead_contact_count = int(role_counts.get("lead", 0))
    lead_pad_count = len(canonical.get("lead_pads") or [])
    inner_land_pad_count = len(canonical.get("inner_land_pads") or [])
    package_pad_boxes = [obj["bbox"] for obj in objects if obj.get("role") == "package_pad" and obj.get("bbox")]
    thermal_package_pad_count = thermal_like_package_pad_count(package_pad_boxes)
    terminal_package_pad_count = max(0, package_pad_count - thermal_package_pad_count)
    land_pad_boxes = [obj["bbox"] for obj in objects if obj.get("role") == "land" and obj.get("bbox")]
    thermal_land_pad_count = thermal_like_package_pad_count(land_pad_boxes)
    terminal_land_pad_count = max(0, land_pad_count - thermal_land_pad_count)
    lead_equivalent_count = terminal_package_pad_count if package_pad_count > 0 else lead_contact_count
    land_detail_lead_contact_count = sum(
        1 for obj in objects if obj.get("role") == "lead" and str(obj.get("raw_view") or "") == "land_detail"
    )
    features["summary"] = {
        "object_count": len(objects),
        "role_counts": features["role_counts"],
        "package_pad_count": package_pad_count,
        "package_pad_rect_count": package_pad_rect_count,
        "terminal_package_pad_count": terminal_package_pad_count,
        "thermal_package_pad_count": thermal_package_pad_count,
        "land_pad_count": land_pad_count,
        "terminal_land_pad_count": terminal_land_pad_count,
        "thermal_land_pad_count": thermal_land_pad_count,
        "lead_contact_count": lead_contact_count,
        "lead_pad_count": lead_pad_count,
        "inner_land_pad_count": inner_land_pad_count,
        "lead_equivalent_count": lead_equivalent_count,
        "land_detail_lead_contact_count": land_detail_lead_contact_count,
        "land_detail_padlike_count_candidates": land_detail_padlike_count_candidates(canonical),
        "supplemental_land_padlike_count_candidates": supplemental_land_padlike_count_candidates(canonical),
        "bbox": features["bbox"],
        "source_views": canonical.get("source_views"),
        "canonical_source_views": canonical.get("canonical_source_views"),
        "missing_canonical_views": canonical.get("missing_canonical_views"),
        "source_selection": canonical.get("source_selection"),
        "selected_package_graph_pad_like_count": selected_graph_pad_like_count(canonical, "package_pads"),
        "selected_land_graph_pad_like_count": selected_graph_pad_like_count(canonical, "land_pads"),
        "conflict_count": len(canonical.get("conflicts") or []),
    }
    return features


def selected_graph_pad_like_count(canonical: dict[str, Any], role: str) -> int | None:
    selection = ((canonical.get("source_selection") or {}).get(role) or {})
    graph_path = str(selection.get("graph_path") or "")
    if not graph_path:
        return None
    for ref in canonical.get("evidence_refs") or []:
        if str(ref.get("evidence_type") or "") != "package_graph":
            continue
        if str(ref.get("graph_path") or "") != graph_path:
            continue
        value = ref.get("pad_like_count")
        return int(value) if isinstance(value, int) else None
    return None


def supplemental_land_padlike_count_candidates(canonical: dict[str, Any]) -> list[dict[str, Any]]:
    selected_land_graph = str(
        ((canonical.get("source_selection") or {}).get("land_pads") or {}).get("graph_path") or ""
    )
    candidates = []
    for ref in canonical.get("evidence_refs") or []:
        if str(ref.get("evidence_type") or "") != "package_graph":
            continue
        if str(ref.get("canonical_view") or "") != "land":
            continue
        graph_path = str(ref.get("graph_path") or "")
        if graph_path == selected_land_graph:
            continue
        terminal_count = int(ref.get("terminal_pad_like_count") or ref.get("pad_like_count") or 0)
        if terminal_count <= 0:
            continue
        candidates.append(
            {
                "source": "package_graph_evidence",
                "candidate_type": "supplemental_land_terminal_padlike_count",
                "count": terminal_count,
                "raw_pad_like_count": int(ref.get("pad_like_count") or 0),
                "thermal_pad_like_count": int(ref.get("thermal_pad_like_count") or 0),
                "raw_view": ref.get("raw_view"),
                "canonical_view": ref.get("canonical_view"),
                "graph_path": graph_path,
            }
        )
    return candidates


def thermal_like_package_pad_count(package_pad_boxes: list[list[float]]) -> int:
    """Count large internal package pads that are not perimeter lead terminals.

    Coordinate system: image/model bbox coordinates in pixels.  This only uses
    relative bbox area and center position, so units do not matter.  A package
    pad is treated as thermal-like when it is much larger than the median pad
    area and its center lies in the central body band.
    """
    return len(thermal_like_bbox_keys(package_pad_boxes))


def thermal_like_bbox_keys(package_pad_boxes: list[list[float]]) -> set[tuple[int, int, int, int]]:
    """Return bbox keys for large internal pads.

    Coordinate system: image/model bbox coordinates, x right, y down. The
    returned keys use bbox_dedupe_key tolerance 0.001 and are only intended for
    filtering boxes from the same coordinate system.
    """
    if len(package_pad_boxes) < 5:
        return set()
    areas = [bbox_area(box) for box in package_pad_boxes]
    positive_areas = sorted(area for area in areas if area > 0)
    if not positive_areas:
        return set()
    median_area = positive_areas[len(positive_areas) // 2]
    if median_area <= 0:
        return set()
    widths = sorted(abs(box[2] - box[0]) for box in package_pad_boxes if bbox_area(box) > 0)
    heights = sorted(abs(box[3] - box[1]) for box in package_pad_boxes if bbox_area(box) > 0)
    median_width = widths[len(widths) // 2] if widths else 0.0
    median_height = heights[len(heights) // 2] if heights else 0.0
    extent = union_bbox(package_pad_boxes)
    if not extent:
        return set()
    x1, y1, x2, y2 = extent
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return set()

    large_central_found = False
    candidates: list[tuple[list[float], bool, bool]] = []
    for box, area in zip(package_pad_boxes, areas):
        box_width = abs(box[2] - box[0])
        box_height = abs(box[3] - box[1])
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        x_ratio = (cx - x1) / width
        y_ratio = (cy - y1) / height
        central = 0.2 <= x_ratio <= 0.8 and 0.2 <= y_ratio <= 0.8
        clearly_larger = area >= median_area * 8.0
        compact_larger = (
            area >= median_area * 1.8
            and (median_width <= 0 or box_width >= median_width * 0.8)
            and (median_height <= 0 or box_height >= median_height * 0.8)
        )
        elongated_larger = (
            area >= median_area * 1.8
            and (
                (median_width > 0 and box_width >= median_width * 1.5)
                or (median_height > 0 and box_height >= median_height * 1.5)
            )
        )
        central_nonterminal = central and (clearly_larger or compact_larger or elongated_larger)
        if central_nonterminal:
            large_central_found = True
        side_internal_larger = (
            0.2 <= y_ratio <= 0.8
            and (x_ratio < 0.25 or x_ratio > 0.75)
            and area >= median_area * 1.8
            and (
                (median_width > 0 and box_width >= median_width * 1.8)
                or (median_height > 0 and box_height >= median_height * 1.8)
            )
        )
        candidates.append((box, central_nonterminal, side_internal_larger))
    keys = set()
    for box, central_nonterminal, side_internal_larger in candidates:
        if central_nonterminal or (large_central_found and side_internal_larger):
            key = bbox_dedupe_key(box, 0.001)
            if key is not None:
                keys.add(key)
    return keys


def land_detail_padlike_count_candidates(canonical: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    for ref in canonical.get("evidence_refs") or []:
        if str(ref.get("evidence_type") or "") != "package_graph":
            continue
        if str(ref.get("raw_view") or "") != "land_detail":
            continue
        count = int(ref.get("pad_like_count") or 0)
        if count <= 0:
            continue
        candidates.append(
            {
                "source": "package_graph_evidence",
                "candidate_type": "land_detail_padlike_count",
                "count": count,
                "raw_view": ref.get("raw_view"),
                "canonical_view": ref.get("canonical_view"),
                "graph_path": ref.get("graph_path"),
            }
        )
    return candidates


def canonical_bbox_candidates(objects: list[dict[str, Any]]) -> dict[str, list[float] | None]:
    by_role: dict[str, list[list[float]]] = {}
    for obj in objects:
        bbox = obj.get("bbox")
        if not bbox:
            continue
        by_role.setdefault(str(obj.get("role") or "unknown"), []).append(bbox)
    package_pad_boxes = by_role.get("package_pad", [])
    lead_boxes = by_role.get("lead", [])
    package_label_boxes: dict[str, list[list[float]]] = {}
    for obj in objects:
        if str(obj.get("role") or "") != "package_pad":
            continue
        bbox = obj.get("bbox")
        label = package_pad_label_key(obj)
        if bbox and label:
            package_label_boxes.setdefault(label, []).append(bbox)
    return {
        "all": union_bbox([box for boxes in by_role.values() for box in boxes]),
        "outline": union_bbox(by_role.get("outline", [])),
        "package": union_bbox(by_role.get("outline", []) + package_pad_boxes),
        "package_dshape": union_bbox(package_label_boxes.get("dshape", [])),
        "package_rect": union_bbox(package_label_boxes.get("rect", [])),
        "package_circle": union_bbox(package_label_boxes.get("circle", [])),
        "package_pad": union_bbox(package_label_boxes.get("pad", [])),
        "conductive": union_bbox(package_pad_boxes + by_role.get("land", []) + lead_boxes),
        "land": union_bbox(by_role.get("land", [])),
        "lead": union_bbox(lead_boxes if lead_boxes else package_pad_boxes),
    }


def package_pad_label_key(obj: dict[str, Any]) -> str:
    label = f"{obj.get('label') or ''} {obj.get('source_label') or ''}".lower()
    if "dshape" in label:
        return "dshape"
    if "circle" in label:
        return "circle"
    if "rect" in label:
        return "rect"
    if "pad" in label:
        return "pad"
    return ""


def compare_features(gt: dict[str, Any], graph: dict[str, Any], options: AlignmentOptions) -> list[dict[str, Any]]:
    checks = []
    gt_land = int(gt.get("role_counts", {}).get("land", 0))
    graph_land = int(graph.get("role_counts", {}).get("land", 0))

    gt_lead = int(gt.get("role_counts", {}).get("lead", 0))
    graph_lead = int((graph.get("summary") or {}).get("lead_equivalent_count", 0))
    count_checks = semantic_count_checks(
        gt_land,
        gt_lead,
        graph_land,
        graph_lead,
        graph,
        options,
        gt_count_candidates=gt.get("count_candidates") or {},
        land_detail_padlike_count_candidates=(graph.get("summary") or {}).get("land_detail_padlike_count_candidates") or [],
        supplemental_land_padlike_count_candidates=(graph.get("summary") or {}).get("supplemental_land_padlike_count_candidates") or [],
    )
    annotate_scan_count_candidate_mismatches(count_checks, gt.get("count_candidates") or {})
    checks.extend(count_checks)

    bbox_check = compare_bbox_candidates(gt, graph, options)
    if bbox_check:
        checks.append(bbox_check)
    for check in checks:
        check["stage_hint"] = infer_stage_hint(check, graph)
    return checks


def semantic_count_checks(
    gt_land: int,
    gt_lead: int,
    graph_land: int,
    graph_lead: int,
    graph: dict[str, Any],
    options: AlignmentOptions,
    gt_count_candidates: dict[str, list[dict[str, Any]]] | None = None,
    land_detail_padlike_count_candidates: list[dict[str, Any]] | None = None,
    supplemental_land_padlike_count_candidates: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    direct_land = count_check("land_count", gt_land, graph_land, options.count_tolerance)
    direct_lead = count_check("lead_count", gt_lead, graph_lead, options.count_tolerance)
    package_pad_count = int((graph.get("summary") or {}).get("package_pad_count", 0))
    package_pad_rect_count = int((graph.get("summary") or {}).get("package_pad_rect_count", 0))
    terminal_package_pad_count = int((graph.get("summary") or {}).get("terminal_package_pad_count", package_pad_count))
    terminal_land_pad_count = int((graph.get("summary") or {}).get("terminal_land_pad_count", graph_land))
    land_detail_lead_contact_count = int((graph.get("summary") or {}).get("land_detail_lead_contact_count", 0))
    group_land = scan_group_count_check(
        "land_count",
        "land",
        gt_land,
        graph_land,
        gt_count_candidates,
        options.count_tolerance,
        actual_role="land_count",
    )
    if direct_land["status"] != "aligned" and group_land:
        direct_land = group_land
    if (
        direct_land["status"] != "aligned"
        and gt_land > 0
        and graph_land > 0
        and terminal_land_pad_count != graph_land
    ):
        group_terminal_land = scan_group_count_check(
            "land_count",
            "land",
            gt_land,
            terminal_land_pad_count,
            gt_count_candidates,
            options.count_tolerance,
            actual_role="terminal_land_pad_count",
        )
        if group_terminal_land:
            group_terminal_land["selected_mapping"] = "scan_group_land_count_to_terminal_land_pad_count"
            group_terminal_land["direct_actual"] = graph_land
            group_terminal_land["raw_land_pad_count"] = graph_land
            direct_land = group_terminal_land
    if (
        direct_land["status"] != "aligned"
        and gt_land > 0
        and graph_land > 0
        and terminal_land_pad_count != graph_land
    ):
        terminal_land = count_check("land_count", gt_land, terminal_land_pad_count, options.count_tolerance)
        if terminal_land["status"] == "aligned":
            terminal_land["selected_mapping"] = "scan_land_to_terminal_land_pad_count"
            terminal_land["actual_role"] = "terminal_land_pad_count"
            terminal_land["direct_actual"] = graph_land
            terminal_land["raw_land_pad_count"] = graph_land
            direct_land = terminal_land
    group_lead = scan_group_count_check(
        "lead_count",
        "lead",
        gt_lead,
        graph_lead,
        gt_count_candidates,
        options.count_tolerance,
        actual_role="lead_equivalent_count",
    )
    if direct_lead["status"] != "aligned" and group_lead:
        direct_lead = group_lead

    if direct_lead["status"] != "aligned" and package_pad_rect_count > 0:
        rect_group_lead = scan_group_count_check(
            "lead_count",
            "lead",
            gt_lead,
            package_pad_rect_count,
            gt_count_candidates,
            options.count_tolerance,
            actual_role="package_pad_rect_count",
        )
        if rect_group_lead:
            rect_group_lead["selected_mapping"] = "scan_group_lead_count_to_package_pad_rect_count"
            rect_group_lead["direct_actual"] = graph_lead
            rect_group_lead["package_pad_count"] = package_pad_count
            direct_lead = rect_group_lead

    if (
        direct_lead["status"] != "aligned"
        and gt_lead > 0
        and package_pad_count > 0
        and terminal_package_pad_count != package_pad_count
    ):
        raw_package_lead = count_check("lead_count", gt_lead, package_pad_count, options.count_tolerance)
        if raw_package_lead["status"] == "aligned":
            raw_package_lead["selected_mapping"] = "scan_lead_to_raw_package_pad_count_including_thermal"
            raw_package_lead["actual_role"] = "package_pad_count"
            raw_package_lead["direct_actual"] = graph_lead
            raw_package_lead["terminal_package_pad_count"] = terminal_package_pad_count
            direct_lead = raw_package_lead

    if (
        direct_lead["status"] != "aligned"
        and gt_lead > 0
        and terminal_package_pad_count > 0
        and land_detail_lead_contact_count > 0
    ):
        terminal_package_plus_detail = count_check(
            "lead_count",
            gt_lead,
            terminal_package_pad_count + land_detail_lead_contact_count,
            options.count_tolerance,
        )
        if terminal_package_plus_detail["status"] == "aligned":
            terminal_package_plus_detail["selected_mapping"] = (
                "scan_lead_to_terminal_package_pad_plus_land_detail_lead_contacts"
            )
            terminal_package_plus_detail["actual_role"] = (
                "terminal_package_pad_count_plus_land_detail_lead_contact_count"
            )
            terminal_package_plus_detail["direct_actual"] = graph_lead
            terminal_package_plus_detail["terminal_package_pad_count"] = terminal_package_pad_count
            terminal_package_plus_detail["supplemental_actual"] = land_detail_lead_contact_count
            direct_lead = terminal_package_plus_detail

    if direct_land["status"] == "aligned" and direct_lead["status"] == "aligned":
        return [direct_land, direct_lead]

    # ScanResult sometimes stores package-side pads under LandData and PCB lands
    # under LeadData. Only accept this alternate mapping when both direct count
    # checks fail and the cross mapping matches exactly within count tolerance.
    if (
        direct_land["status"] != "aligned"
        and direct_lead["status"] != "aligned"
        and gt_land > 0
        and gt_lead > 0
        and package_pad_count > 0
        and graph_land > 0
    ):
        cross_land = count_check("land_count", gt_land, terminal_package_pad_count, options.count_tolerance)
        cross_lead = count_check("lead_count", gt_lead, terminal_land_pad_count, options.count_tolerance)
        if cross_land["status"] == "aligned" and cross_lead["status"] == "aligned":
            mapping = "scan_land_to_terminal_package_pad_scan_lead_to_terminal_land"
            cross_land["selected_mapping"] = mapping
            cross_land["actual_role"] = "terminal_package_pad_count"
            cross_land["direct_actual"] = graph_land
            cross_land["raw_package_pad_count"] = package_pad_count
            cross_lead["selected_mapping"] = mapping
            cross_lead["actual_role"] = "terminal_land_pad_count"
            cross_lead["direct_actual"] = graph_lead
            cross_lead["raw_land_pad_count"] = graph_land
            return [cross_land, cross_lead]

    if (
        direct_land["status"] != "aligned"
        and gt_land > 0
        and land_detail_lead_contact_count > 0
        and terminal_land_pad_count != graph_land
    ):
        terminal_land_plus_detail = count_check(
            "land_count",
            gt_land,
            terminal_land_pad_count + land_detail_lead_contact_count,
            options.count_tolerance,
        )
        if terminal_land_plus_detail["status"] == "aligned":
            terminal_land_plus_detail["selected_mapping"] = "scan_land_to_terminal_land_plus_land_detail_lead_contacts"
            terminal_land_plus_detail["actual_role"] = "terminal_land_pad_count_plus_land_detail_lead_contact_count"
            terminal_land_plus_detail["direct_actual"] = graph_land
            terminal_land_plus_detail["terminal_land_pad_count"] = terminal_land_pad_count
            terminal_land_plus_detail["supplemental_actual"] = land_detail_lead_contact_count
            set_direct_lead_mapping_if_missing(direct_lead)
            return [terminal_land_plus_detail, direct_lead]
    if direct_land["status"] != "aligned" and gt_land > 0 and land_detail_lead_contact_count > 0:
        land_plus_detail = count_check(
            "land_count",
            gt_land,
            graph_land + land_detail_lead_contact_count,
            options.count_tolerance,
        )
        if land_plus_detail["status"] == "aligned":
            land_plus_detail["selected_mapping"] = "scan_land_to_land_plus_land_detail_lead_contacts"
            land_plus_detail["actual_role"] = "land_count_plus_land_detail_lead_contact_count"
            land_plus_detail["direct_actual"] = graph_land
            land_plus_detail["supplemental_actual"] = land_detail_lead_contact_count
            set_direct_lead_mapping_if_missing(direct_lead)
            return [land_plus_detail, direct_lead]

    if direct_land["status"] != "aligned" and gt_land > 0 and graph_land > 0:
        evidence_land = land_plus_land_detail_padlike_count_check(
            gt_land,
            graph_land,
            land_detail_padlike_count_candidates,
            options.count_tolerance,
        )
        if evidence_land:
            set_direct_lead_mapping_if_missing(direct_lead)
            return [evidence_land, direct_lead]

    if direct_land["status"] != "aligned" and gt_land > 0 and graph_land > 0:
        supplemental_land = land_plus_supplemental_land_padlike_count_check(
            gt_land,
            graph_land,
            supplemental_land_padlike_count_candidates,
            options.count_tolerance,
        )
        if supplemental_land:
            set_direct_lead_mapping_if_missing(direct_lead)
            return [supplemental_land, direct_lead]

    if (
        direct_lead["status"] != "aligned"
        and gt_lead > 0
        and graph_land > 0
    ):
        lead_from_land = count_check("lead_count", gt_lead, graph_land, options.count_tolerance)
        if lead_from_land["status"] == "aligned":
            missing_views = set((graph.get("summary") or {}).get("missing_canonical_views") or [])
            mapping = (
                "scan_lead_to_land_count_missing_package_view"
                if package_pad_count == 0 or "bottom" in missing_views
                else "scan_lead_to_land_count"
            )
            direct_land["selected_mapping"] = "direct"
            direct_land["actual_role"] = "land_count"
            lead_from_land["selected_mapping"] = mapping
            lead_from_land["actual_role"] = "land_count"
            lead_from_land["direct_actual"] = graph_lead
            return [direct_land, lead_from_land]

    if direct_land["status"] != "aligned" and gt_land > 0 and graph_land == 0 and package_pad_count == gt_land:
        fallback_land = count_check("land_count", gt_land, package_pad_count, options.count_tolerance)
        fallback_land["selected_mapping"] = "scan_land_to_package_pad_missing_land_view"
        fallback_land["actual_role"] = "package_pad_count"
        fallback_land["direct_actual"] = graph_land
        fallback_land["mapping_note"] = "land view is missing; ScanResult LandData count matches package pad count"
        set_direct_lead_mapping_if_missing(direct_lead)
        return [fallback_land, direct_lead]

    if "selected_mapping" not in direct_land:
        direct_land["selected_mapping"] = "direct"
    if "actual_role" not in direct_land:
        direct_land["actual_role"] = "land_count"
    set_direct_lead_mapping_if_missing(direct_lead)
    return [direct_land, direct_lead]


def set_direct_lead_mapping_if_missing(check: dict[str, Any]) -> None:
    if "selected_mapping" not in check:
        check["selected_mapping"] = "direct"
    if "actual_role" not in check:
        check["actual_role"] = "lead_equivalent_count"


def annotate_scan_count_candidate_mismatches(
    checks: list[dict[str, Any]],
    candidates_by_role: dict[str, list[dict[str, Any]]],
) -> None:
    for check in checks:
        if check.get("status") == "aligned":
            continue
        role = "land" if check.get("name") == "land_count" else "lead" if check.get("name") == "lead_count" else ""
        if not role:
            continue
        counts = sorted(
            {
                int(candidate.get("count") or 0)
                for candidate in candidates_by_role.get(role, [])
                if int(candidate.get("count") or 0) > 0
            }
        )
        if counts:
            check["scan_count_candidate_counts"] = counts


def land_plus_supplemental_land_padlike_count_check(
    expected_land: int,
    graph_land: int,
    candidates: list[dict[str, Any]] | None,
    tolerance: int,
) -> dict[str, Any] | None:
    for candidate in candidates or []:
        supplemental = int(candidate.get("count") or 0)
        check = count_check("land_count", expected_land, graph_land + supplemental, tolerance)
        if check["status"] != "aligned":
            continue
        check["selected_mapping"] = "scan_land_to_land_plus_supplemental_land_terminal_padlike_evidence"
        check["actual_role"] = "land_count_plus_supplemental_land_terminal_padlike_evidence_count"
        check["direct_actual"] = graph_land
        check["supplemental_actual"] = supplemental
        check["candidate"] = candidate
        return check
    return None


def land_plus_land_detail_padlike_count_check(
    expected_land: int,
    graph_land: int,
    candidates: list[dict[str, Any]] | None,
    tolerance: int,
) -> dict[str, Any] | None:
    for candidate in candidates or []:
        supplemental = int(candidate.get("count") or 0)
        check = count_check("land_count", expected_land, graph_land + supplemental, tolerance)
        if check["status"] != "aligned":
            continue
        check["selected_mapping"] = "scan_land_to_land_plus_land_detail_padlike_evidence"
        check["actual_role"] = "land_count_plus_land_detail_padlike_evidence_count"
        check["direct_actual"] = graph_land
        check["supplemental_actual"] = supplemental
        check["candidate"] = candidate
        return check
    return None


def scan_group_count_check(
    name: str,
    role: str,
    direct_expected: int,
    actual: int,
    candidates_by_role: dict[str, list[dict[str, Any]]] | None,
    tolerance: int,
    *,
    actual_role: str,
) -> dict[str, Any] | None:
    for candidate in candidates_by_role.get(role, []) if candidates_by_role else []:
        candidate_count = int(candidate.get("count") or 0)
        check = count_check(name, candidate_count, actual, tolerance)
        if check["status"] != "aligned":
            continue
        check["selected_mapping"] = f"scan_group_{role}_count_candidate"
        check["actual_role"] = actual_role
        check["direct_expected"] = direct_expected
        check["candidate"] = candidate
        return check
    return None


def count_check(name: str, expected: int, actual: int, tolerance: int) -> dict[str, Any]:
    delta = abs(expected - actual)
    return {
        "name": name,
        "expected": expected,
        "actual": actual,
        "delta": delta,
        "tolerance": tolerance,
        "status": "aligned" if delta <= tolerance else "mismatch",
        "reason": f"{name}_mismatch",
    }


def infer_stage_hint(check: dict[str, Any], graph: dict[str, Any]) -> str:
    if check.get("status") == "aligned":
        return "aligned"
    name = str(check.get("name") or "")
    expected = int(check.get("expected") or 0) if isinstance(check.get("expected"), int) else 0
    actual = int(check.get("actual") or 0) if isinstance(check.get("actual"), int) else 0
    missing_views = set((graph.get("summary") or {}).get("missing_canonical_views") or [])
    source_views = set((graph.get("summary") or {}).get("source_views") or [])
    package_pad_count = int((graph.get("summary") or {}).get("package_pad_count", 0) or 0)
    if name == "land_count" and expected > 0 and actual == 0 and "land" in missing_views and package_pad_count == expected:
        return "multiview_missing_land_view_package_pad_count_matches"
    if name == "land_count" and expected > 0 and actual == 0 and "land" in missing_views:
        if "land" not in source_views:
            return "data_missing_land_view"
        return "multiview_missing_land_view"
    selected_land_pad_count = (graph.get("summary") or {}).get("selected_land_graph_pad_like_count")
    if (
        name == "land_count"
        and expected > actual
        and actual > 0
        and isinstance(selected_land_pad_count, int)
        and expected > selected_land_pad_count
        and actual <= selected_land_pad_count
    ):
        return "scan_result_land_count_exceeds_visible_land_annotation"
    if name == "land_count" and expected > 0 and actual > 0 and "land" not in missing_views:
        return "package_graph_land_reconstruction_count_mismatch"
    if name == "land_count":
        return "package_graph_land_or_multiview_count_mismatch"
    if name == "lead_count" and expected > 0 and "bottom" in missing_views:
        if "bottom" not in source_views:
            return "data_missing_package_pad_source_view"
        return "multiview_missing_package_pad_source_view"
    if name == "lead_count" and expected > 0 and actual == 0 and {"lateral", "lead_detail"} & missing_views:
        return "multiview_missing_lateral_or_lead_detail_view"
    scan_candidate_counts = {
        int(value)
        for value in (check.get("scan_count_candidate_counts") or [])
        if isinstance(value, int)
    }
    if (
        name == "lead_count"
        and expected > 0
        and actual > 0
        and scan_candidate_counts
        and expected not in scan_candidate_counts
        and actual not in scan_candidate_counts
    ):
        if max(scan_candidate_counts) < expected and actual > expected:
            return "package_graph_package_pad_reconstruction_count_mismatch"
        return "scan_result_lead_count_ambiguous_with_graph_count_mismatch"
    if name == "lead_count" and expected > 0 and actual > 0:
        return "package_graph_package_pad_reconstruction_count_mismatch"
    if name == "lead_count":
        return "package_graph_package_pad_or_multiview_count_mismatch"
    if name == "bbox_aspect" and bbox_aspect_uses_fallback_package_source(check, graph):
        return "multiview_fallback_package_pad_source_geometry_mismatch"
    if name == "bbox_aspect":
        return "scan_result_alignment_or_reconstruction_geometry_mismatch"
    return "gt_alignment_mismatch"


def bbox_aspect_uses_fallback_package_source(check: dict[str, Any], graph: dict[str, Any]) -> bool:
    selected_pair = check.get("selected_pair") or {}
    graph_pair = str(selected_pair.get("graph") or "")
    if graph_pair not in {"package", "lead", "conductive", "all"}:
        return False
    source_selection = (graph.get("summary") or {}).get("source_selection") or {}
    package_source = source_selection.get("package_pads") or {}
    return bool(package_source.get("used_fallback") or package_source.get("missing_primary"))


def error_sources_for_stage_hints(stage_hints: list[str]) -> list[str]:
    sources: set[str] = set()
    for stage_hint in stage_hints:
        sources.update(error_sources_for_stage_hint(stage_hint))
    return sorted(sources)


def error_sources_for_stage_hint(stage_hint: str) -> list[str]:
    if stage_hint == "aligned":
        return []
    if stage_hint.startswith("review_note_multiview_"):
        return ["multiview_integration"]
    if stage_hint.startswith("low_score_scan_result_"):
        return ["multiview_integration", "scan_result_parsing_alignment"]
    if stage_hint.startswith("low_score_data_missing_"):
        return ["data_coverage"]
    if stage_hint.startswith("low_score_multiview_"):
        return ["multiview_integration"]
    if stage_hint.startswith("low_score_package_graph_"):
        return ["package_graph_reconstruction"]
    if stage_hint in {"scan_result_missing"}:
        return ["annotation_gt_mismatch", "scan_result_parsing_alignment"]
    if stage_hint == "scan_result_land_count_exceeds_visible_land_annotation":
        return ["annotation_gt_mismatch", "scan_result_parsing_alignment"]
    if stage_hint == "scan_result_lead_count_ambiguous_with_graph_count_mismatch":
        return ["package_graph_reconstruction", "scan_result_parsing_alignment"]
    if stage_hint.startswith("data_missing_"):
        return ["data_coverage"]
    if stage_hint.startswith("multiview_"):
        return ["multiview_integration"]
    if stage_hint.startswith("package_graph_"):
        return ["package_graph_reconstruction"]
    if stage_hint == "scan_result_alignment_or_reconstruction_geometry_mismatch":
        return ["package_graph_reconstruction", "scan_result_parsing_alignment"]
    return ["scan_result_parsing_alignment"]


def objective_error_sources_for_stage_hints(stage_hints: list[str]) -> list[str]:
    sources: set[str] = set()
    for stage_hint in stage_hints:
        sources.update(objective_error_sources_for_stage_hint(stage_hint))
    return sorted(sources)


def objective_error_sources_for_stage_hint(stage_hint: str) -> list[str]:
    if stage_hint == "aligned":
        return []
    if stage_hint.startswith("review_note_multiview_"):
        return ["multiview_alignment"]
    if stage_hint.startswith("low_score_scan_result_"):
        return ["multiview_alignment", "scan_result_parsing"]
    if stage_hint.startswith("low_score_data_missing_"):
        return ["gt_annotation_issue"]
    if stage_hint.startswith("low_score_multiview_"):
        return ["multiview_alignment"]
    if stage_hint.startswith("low_score_package_graph_"):
        return ["model_prediction", "package_graph_reconstruction"]
    if stage_hint in {"scan_result_missing"}:
        return ["gt_annotation_issue", "scan_result_parsing"]
    if stage_hint == "scan_result_land_count_exceeds_visible_land_annotation":
        return ["gt_annotation_issue", "scan_result_parsing"]
    if stage_hint == "scan_result_lead_count_ambiguous_with_graph_count_mismatch":
        return ["package_graph_reconstruction", "scan_result_parsing"]
    if stage_hint.startswith("data_missing_"):
        return ["gt_annotation_issue"]
    if stage_hint.startswith("multiview_"):
        return ["multiview_alignment"]
    if stage_hint in {
        "package_graph_land_reconstruction_count_mismatch",
        "package_graph_package_pad_reconstruction_count_mismatch",
    }:
        return ["model_prediction", "package_graph_reconstruction"]
    if stage_hint.startswith("package_graph_"):
        return ["package_graph_reconstruction"]
    if stage_hint == "scan_result_alignment_or_reconstruction_geometry_mismatch":
        return ["package_graph_reconstruction", "scan_result_parsing"]
    return ["scan_result_parsing"]


def compare_bbox_candidates(gt: dict[str, Any], graph: dict[str, Any], options: AlignmentOptions) -> dict[str, Any] | None:
    # ScanResult coordinates are physical/reference units while graph bboxes are
    # reconstruction coordinates, so this compares only unitless aspect ratios.
    candidate_pairs = [
        ("all", "all"),
        ("all", "package"),
        ("shape", "outline"),
        ("shape", "package"),
        ("conductive", "package"),
        ("conductive", "conductive"),
        ("land", "package"),
        ("land", "land"),
        ("lead", "package"),
        ("lead", "lead"),
    ]
    gt_candidates = gt.get("bbox_candidates") or {"all": gt.get("bbox")}
    graph_candidates = graph.get("bbox_candidates") or {"all": graph.get("bbox")}
    best: dict[str, Any] | None = None
    for gt_name, graph_name in candidate_pairs:
        candidate = compare_bbox(
            gt_candidates.get(gt_name),
            graph_candidates.get(graph_name),
            options,
            selected_pair={"gt": gt_name, "graph": graph_name},
        )
        if candidate is None:
            continue
        if best is None or candidate["delta"]["aspect"] < best["delta"]["aspect"]:
            best = candidate
    return best


def compare_bbox(
    gt_bbox: list[float] | None,
    graph_bbox: list[float] | None,
    options: AlignmentOptions,
    *,
    selected_pair: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    if not gt_bbox or not graph_bbox:
        return None
    gt_w = max(float(gt_bbox[2]) - float(gt_bbox[0]), 0.0)
    gt_h = max(float(gt_bbox[3]) - float(gt_bbox[1]), 0.0)
    graph_w = max(float(graph_bbox[2]) - float(graph_bbox[0]), 0.0)
    graph_h = max(float(graph_bbox[3]) - float(graph_bbox[1]), 0.0)
    if gt_w <= 0.0 or gt_h <= 0.0 or graph_w <= 0.0 or graph_h <= 0.0:
        return None
    gt_aspect = gt_w / gt_h
    graph_aspect = graph_w / graph_h
    aspect_delta = abs(gt_aspect - graph_aspect)
    aspect_tol = max(options.bbox_abs_tol, gt_aspect * options.bbox_rel_tol)
    status = "aligned" if aspect_delta <= aspect_tol else "mismatch"
    return {
        "name": "bbox_aspect",
        "expected": {"width": gt_w, "height": gt_h, "aspect": gt_aspect},
        "actual": {"width": graph_w, "height": graph_h, "aspect": graph_aspect},
        "delta": {"aspect": aspect_delta},
        "tolerance": {"aspect": aspect_tol},
        "status": status,
        "reason": "bbox_aspect_mismatch",
        "selected_pair": selected_pair or {"gt": "all", "graph": "all"},
    }


def union_bbox(boxes: list[list[float] | None]) -> list[float] | None:
    valid = [box for box in boxes if box and len(box) >= 4]
    if not valid:
        return None
    return [
        min(float(box[0]) for box in valid),
        min(float(box[1]) for box in valid),
        max(float(box[2]) for box in valid),
        max(float(box[3]) for box in valid),
    ]


def write_part_files(
    output_part_dir: Path,
    summary: dict[str, Any],
    gt: dict[str, Any],
    canonical: dict[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    summary["alignment_path"] = str(output_part_dir / "alignment.json")
    scan_svg_path = output_part_dir / "scan_result.svg"
    if write_scan_result_svg(gt, scan_svg_path):
        summary["scan_result_svg_path"] = str(scan_svg_path)
    write_multiview_alignment_svgs(summary, gt, canonical, checks)
    payload = {
        "part_number": summary["part_number"],
        "status": summary["status"],
        "reasons": summary["reasons"],
        "metrics": summary.get("alignment_scores") or {},
        "summary": summary,
        "gt": gt,
        "canonical": canonical,
        "checks": checks,
    }
    output_part_dir.mkdir(parents=True, exist_ok=True)
    (output_part_dir / "alignment.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def align_overlay_layers(canonical: dict[str, Any]) -> dict[str, Any]:
    """Rotate source package-graph layers and center them in graph coordinates.

    Coordinate system: layer bboxes use multiview/reconstructed graph
    coordinates, x right and y down. Each layer is first normalized by its
    accepted-dimension unit scale into the selected reference layer scale, then
    translated so the scaled frame center matches the reference frame center.
    Rotation is selected after this scale-and-center step. No GT/reference
    remapping or SVG display scaling is applied to the main output.

    Rotation scoring uses canonical multiview objects, not raw detector boxes.
    Reference priority is:
    land/inner-land pads, lateral lead pads, bottom package pads, then top
    package pads. Layers without canonical match boxes are skipped, because
    alignment must consume the multiview result instead of creating a second
    raw-graph path.
    """
    items = package_graph_layer_items(canonical)
    alignable_items = [item for item in items if item["match_boxes"]]
    reference = first_reference_layer(alignable_items)
    if reference is None:
        return {
            "objects": [],
            "layers": [],
            "transforms": {},
            "scores": {
                "overall_score": 0.0,
                "quality_score": 0.0,
                "review_quality_score": 0.0,
                "layer_alignment_score": 0.0,
                "layer_alignment_ref_view": None,
            },
            "summary": {
                    "strategy": "dimension_scaled_layer_rotation",
                    "status": "missing_reference_layer",
                    "reference_view": None,
                    "transforms": {},
            },
            "checks": [
                {
                    "name": "layer_reference",
                    "status": "mismatch",
                    "reason": "missing_top_bottom_land_reference_layer",
                    "stage_hint": "multiview_layer_alignment_missing_reference",
                }
            ],
        }

    transforms: dict[str, dict[str, Any]] = {}
    reference_key = reference["key"]
    transforms[reference_key] = layer_transform_payload(
        reference,
        target_frame=reference["frame"],
        target_unit_scales=reference["unit_scales"],
        rotation_degrees=0,
        rotation_iou=1.0,
        followed_view="",
        strategy="reference",
        alignment_stage="reference_seed",
    )
    checks: list[dict[str, Any]] = []
    rotation_scores = [1.0]
    reference_boxes = [list(box) for box in reference["match_boxes"]]
    processed_keys = {reference_key}

    for item, stage_name in staged_rotation_items(reference, alignable_items):
        if item["key"] in processed_keys:
            continue
        best = best_layer_rotation(
            reference_boxes=reference_boxes,
            reference_frame=reference["frame"],
            reference_unit_scales=reference["unit_scales"],
            candidate_boxes=item["match_boxes"],
            candidate_frame=item["frame"],
            candidate_unit_scales=item["unit_scales"],
        )
        transforms[item["key"]] = layer_transform_payload(
            item,
            target_frame=reference["frame"],
            target_unit_scales=reference["unit_scales"],
            rotation_degrees=int(best["rotation_degrees"]),
            rotation_iou=float(best["iou"]),
            followed_view=str(reference["raw_view"]),
            strategy="rotated_to_reference",
            alignment_stage=stage_name,
        )
        processed_keys.add(item["key"])
        rotation_scores.append(float(best["iou"]))
        reference_boxes.extend(transformed_match_boxes(item, transforms[item["key"]]))

    transforms_by_view = transforms_by_raw_view(items, transforms)
    for item in items:
        if item["key"] in transforms:
            continue
        followed = followed_transform_for_layer(item, transforms_by_view)
        if followed is None:
            transforms[item["key"]] = layer_transform_payload(
                item,
                target_frame=reference["frame"],
                target_unit_scales=reference["unit_scales"],
                rotation_degrees=0,
                rotation_iou=None,
                followed_view=str(reference["raw_view"]),
                strategy="fallback_reference",
                alignment_stage="no_match_boxes",
            )
            continue
        transforms[item["key"]] = layer_transform_payload(
            item,
            target_frame=followed["target_frame"],
            target_unit_scales=followed["target_unit_scales"],
            rotation_degrees=int(followed["rotation_degrees"]),
            rotation_iou=None,
            followed_view=str(followed["raw_view"]),
            strategy="follow_primary_view",
            alignment_stage="no_match_boxes",
        )

    rotation_only_objects = rotation_only_layer_objects(items, transforms)
    rotation_only_layers = grouped_layer_objects(rotation_only_objects)
    layer_score = sum(rotation_scores) / len(rotation_scores) if rotation_scores else 0.0
    status_score = 1.0 if not checks else 0.0
    scores = {
        "overall_score": round(status_score, 6),
        "quality_score": round(status_score, 6),
        "review_quality_score": round(status_score, 6),
        "layer_alignment_score": round(status_score, 6),
        "layer_alignment_rotation_iou_mean": round(layer_score, 6),
        "layer_alignment_ref_view": reference["raw_view"],
        "layer_alignment_rotation_count": sum(
            1
            for transform in transforms.values()
            if int(transform.get("rotation_degrees") or 0) % 360 != 0
        ),
        "layer_alignment_transforms": {
            key: transform_summary_payload(transform)
            for key, transform in sorted(transforms.items())
        },
    }
    return {
        "objects": rotation_only_objects,
        "layers": rotation_only_layers,
        "rotation_only_objects": rotation_only_objects,
        "rotation_only_layers": rotation_only_layers,
        "transforms": transforms,
        "scores": scores,
        "summary": {
            "strategy": "dimension_scaled_layer_rotation",
            "status": "aligned" if not checks else "mismatch",
            "reference_view": reference["raw_view"],
            "reference_graph": reference["graph_path"],
            "transforms": scores["layer_alignment_transforms"],
        },
        "checks": checks,
    }


def package_graph_layer_items(canonical: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    seen: set[str] = set()
    canonical_layers = canonical_alignment_layers_by_graph(canonical)
    for ref in canonical.get("evidence_refs") or []:
        evidence_type = str(ref.get("evidence_type") or ref.get("kind") or "")
        if evidence_type != "package_graph":
            continue
        raw_view = str(ref.get("raw_view") or "").strip().lower()
        canonical_view = str(ref.get("canonical_view") or raw_view).strip().lower()
        if raw_view not in set(MAIN_ALIGNMENT_VIEWS) | set(FOLLOW_ALIGNMENT_VIEWS):
            continue
        graph_path = str(ref.get("graph_path") or ref.get("path") or "").strip()
        if not graph_path or graph_path in seen:
            continue
        seen.add(graph_path)
        path = Path(graph_path)
        if not path.is_file():
            continue
        try:
            graph = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        canonical_layer = canonical_layers.get(graph_path) or {}
        match_boxes = canonical_layer_match_boxes(canonical_layer)
        render_objects = list(canonical_layer.get("objects") or [])
        frame = union_bbox(match_boxes) if match_boxes else None
        match_source = str(canonical_layer.get("match_source") or "")
        match_priority = canonical_layer.get("match_priority")
        match_role_counts = canonical_layer.get("match_role_counts") or {}
        if not match_boxes:
            continue
        if not render_objects:
            continue
        if frame is None:
            continue
        items.append(
            {
                "key": graph_path,
                "raw_view": raw_view,
                "canonical_view": canonical_view,
                "graph_path": graph_path,
                "annotation_path": ref.get("annotation_path"),
                "graph": graph,
                "frame": frame,
                "unit_scales": graph_dimension_unit_scales(graph),
                "match_boxes": match_boxes,
                "match_source": match_source,
                "match_priority": int(match_priority) if isinstance(match_priority, (int, float)) else fallback_alignment_priority(raw_view),
                "match_role_counts": dict(match_role_counts),
                "render_objects": render_objects,
            }
        )
    for synthetic_item in synthetic_canonical_layer_items(canonical):
        items.append(synthetic_item)
    attach_alignment_partial_overlay_objects(canonical, items)
    order = {view: index for index, view in enumerate(ALIGNMENT_VIEW_ORDER)}
    return sorted(items, key=lambda item: (item["match_priority"], order.get(item["raw_view"], 99), item["graph_path"]))


def synthetic_canonical_layer_items(canonical: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    package_item = synthetic_canonical_no_source_layer_item(
        canonical,
        key="__canonical__/package",
        raw_view="bottom",
        roles={"package_pad", "lead_pad", "partial_pad_width", "partial_lead_pad_length"},
        sources=("package_pads", "lead_pads"),
        match_roles={"package_pad", "lead_pad"},
        match_priority=1,
        match_source="canonical_no_source_package_or_lead",
    )
    if package_item is not None:
        items.append(package_item)
    land_item = synthetic_canonical_no_source_layer_item(
        canonical,
        key="__canonical__/land",
        raw_view="land",
        roles={"land_pad", "land", "inner_land_pad"},
        sources=("land_pads", "inner_land_pads"),
        match_roles={"land_pad", "land", "inner_land_pad"},
        match_priority=0,
        match_source="canonical_no_source_land_or_inner_land",
    )
    if land_item is not None:
        items.append(land_item)
    return items


def synthetic_canonical_no_source_layer_item(
    canonical: dict[str, Any],
    *,
    key: str,
    raw_view: str,
    roles: set[str],
    sources: tuple[str, ...],
    match_roles: set[str],
    match_priority: int,
    match_source: str,
) -> dict[str, Any] | None:
    objects = []
    for source in sources:
        for obj in canonical.get(source) or []:
            if str(obj.get("source_graph") or ""):
                continue
            if str(obj.get("source_type") or "") == "scan_result_format":
                continue
            role = str(obj.get("role") or "")
            if role not in roles:
                continue
            bbox = graph_object_bbox(obj)
            if bbox is None:
                continue
            render_obj = dict(obj)
            render_obj["bbox"] = list(bbox)
            render_obj.setdefault("raw_view", raw_view)
            render_obj.setdefault("canonical_view", raw_view)
            render_obj.setdefault("source_graph", key)
            objects.append(render_obj)
    if not objects:
        return None
    match_boxes = [
        list(graph_object_bbox(obj) or [])
        for obj in objects
        if str(obj.get("role") or "") in match_roles and graph_object_bbox(obj) is not None
    ]
    frame = union_bbox(match_boxes)
    if frame is None:
        return None
    role_counts = Counter(str(obj.get("role") or "") for obj in objects)
    return {
        "key": key,
        "raw_view": raw_view,
        "canonical_view": raw_view,
        "graph_path": key,
        "annotation_path": "",
        "graph": {"objects": objects, "dimensions": [], "metrics": {}},
        "frame": frame,
        "unit_scales": {"x": 1.0, "y": 1.0, "source": "canonical_units", "x_sample_count": 0, "y_sample_count": 0},
        "match_boxes": match_boxes,
        "match_source": match_source,
        "match_priority": match_priority,
        "match_role_counts": dict(role_counts),
        "render_objects": objects,
    }


def canonical_alignment_layers_by_graph(canonical: dict[str, Any]) -> dict[str, dict[str, Any]]:
    # Coordinate system: canonical object bboxes are multiview/reconstructed
    # graph coordinates, x-right and y-down. These are the same objects drawn
    # in the multiview overlay, so rotation scoring uses the visible
    # post-processed evidence instead of raw detector boxes.
    layers: dict[str, dict[str, Any]] = {}
    for obj, priority, match_role, source in canonical_alignment_objects(canonical):
        graph_path = str(obj.get("source_graph") or "")
        if not graph_path:
            continue
        bbox = graph_object_bbox(obj)
        if bbox is None:
            continue
        layer = layers.setdefault(
            graph_path,
            {
                "objects": [],
                "match_candidates": [],
            },
        )
        render_obj = dict(obj)
        render_obj.setdefault("role", match_role)
        layer["objects"].append(render_obj)
        layer["match_candidates"].append(
            {
                "bbox": list(bbox),
                "priority": int(priority),
                "role": match_role,
                "source": source,
            }
        )
    for layer in layers.values():
        candidates = layer.get("match_candidates") or []
        if not candidates:
            continue
        best_priority = min(int(candidate["priority"]) for candidate in candidates)
        selected = [candidate for candidate in candidates if int(candidate["priority"]) == best_priority]
        layer["match_boxes"] = [candidate["bbox"] for candidate in selected]
        layer["match_priority"] = best_priority
        layer["match_source"] = canonical_alignment_source_label(best_priority, selected)
        layer["match_role_counts"] = dict(Counter(str(candidate.get("role") or "") for candidate in selected))
    return layers


def canonical_alignment_objects(canonical: dict[str, Any]) -> list[tuple[dict[str, Any], int, str, str]]:
    objects: list[tuple[dict[str, Any], int, str, str]] = []
    for obj in canonical.get("land_pads") or []:
        objects.append((obj, 0, "land", "canonical_land_pad"))
    for obj in canonical.get("inner_land_pads") or []:
        objects.append((obj, 0, "inner_land_pad", "canonical_inner_land_pad"))
    for obj in canonical.get("lead_pads") or []:
        raw_view = str(obj.get("raw_view") or "").strip().lower()
        if raw_view in {"front", "side", "lead"}:
            objects.append((obj, 1, "lead_pad", "canonical_lateral_lead_pad"))
    for obj in canonical.get("lead_contacts") or []:
        raw_view = str(obj.get("raw_view") or "").strip().lower()
        if raw_view in {"front", "side", "lead"}:
            objects.append((obj, 1, "lead_pad", "canonical_lateral_lead_contact"))
    for obj in canonical.get("package_pads") or []:
        raw_view = str(obj.get("raw_view") or "").strip().lower()
        if raw_view == "bottom":
            objects.append((obj, 2, "package_pad", "canonical_bottom_package_pad"))
        elif raw_view == "top":
            objects.append((obj, 3, "package_pad", "canonical_top_package_pad"))
    return objects


def canonical_layer_match_boxes(layer: dict[str, Any]) -> list[list[float]]:
    return [list(box) for box in layer.get("match_boxes") or [] if len(box) >= 4]


def canonical_alignment_source_label(priority: int, candidates: list[dict[str, Any]]) -> str:
    sources = sorted({str(candidate.get("source") or "") for candidate in candidates if candidate.get("source")})
    if sources:
        return "+".join(sources)
    return {
        0: "canonical_land_or_inner_land",
        1: "canonical_lateral_lead_pad",
        2: "canonical_bottom_package_pad",
        3: "canonical_top_package_pad",
    }.get(priority, "canonical_unknown")


def fallback_alignment_priority(raw_view: str) -> int:
    return ALIGNMENT_FALLBACK_PRIORITY_BY_VIEW.get(str(raw_view), 9)


def source_graph_render_objects(graph: dict[str, Any], raw_view: str, graph_path: str) -> list[dict[str, Any]]:
    objects = []
    for obj in graph.get("objects") or []:
        role = layer_object_role(obj, raw_view)
        if not role:
            continue
        bbox = graph_object_bbox(obj)
        if bbox is None:
            continue
        render_obj = dict(obj)
        render_obj["role"] = role
        render_obj["bbox"] = list(bbox)
        render_obj["source_graph"] = graph_path
        objects.append(render_obj)
    return objects


def attach_alignment_partial_overlay_objects(canonical: dict[str, Any], items: list[dict[str, Any]]) -> None:
    extras = alignment_partial_overlay_objects_from_canonical(canonical, items)
    if not extras:
        return
    by_graph = {str(item.get("graph_path") or ""): item for item in items}
    for extra in extras:
        graph_path = str(extra.get("source_graph") or "")
        item = by_graph.get(graph_path)
        if item is None:
            continue
        item["render_objects"].append(extra)
    for item in items:
        item["render_objects"] = dedupe_alignment_render_objects(item.get("render_objects") or [])


def alignment_partial_overlay_objects_from_canonical(
    canonical: dict[str, Any],
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return dedupe_alignment_render_objects(canonical_extra_objects_with_source_graph(canonical))


def canonical_extra_objects_with_source_graph(canonical: dict[str, Any]) -> list[dict[str, Any]]:
    extras = []
    for obj in canonical.get("lead_pads") or []:
        if str(obj.get("source_type") or "") == "scan_result_format":
            continue
        if not str(obj.get("source_graph") or ""):
            continue
        if not obj.get("source_package_pad_bbox"):
            continue
        extras.append(dict(obj))
    for obj in canonical.get("inner_land_pads") or []:
        if str(obj.get("source_type") or "") == "scan_result_format":
            continue
        if not str(obj.get("source_graph") or ""):
            continue
        if not obj.get("source_land_pad_bbox"):
            continue
        extras.append(dict(obj))
    return extras


def alignment_partial_overlay_objects_from_partial_graphs(
    canonical: dict[str, Any],
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base_item = select_alignment_overlay_base_item(items)
    if base_item is None:
        return []
    base_graph = base_item["graph"]
    unit_scales = alignment_graph_contact_unit_scales(base_graph)
    if not unit_scales:
        return []
    terminal_pads = terminal_alignment_pad_objects(base_graph)
    if not terminal_pads:
        return []
    outline_bbox = layer_frame(base_graph) or union_bbox([list(bbox) for _obj, bbox in terminal_pads])
    if outline_bbox is None:
        return []

    dimensions = alignment_partial_contact_dimensions(canonical, items)
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
            projection_axis = alignment_overlay_projection_axis(selected, radial_axis)
            unit_scale = unit_scales.get(projection_axis)
            if unit_scale is None or unit_scale <= 0.0:
                continue
            semantics = effective_alignment_partial_dimension_semantics(
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
            new_bbox = alignment_partial_dimension_overlay_bbox(
                bbox=bbox,
                package_center=(center_x, center_y),
                package_frame=tuple(outline_bbox),
                projection_axis=projection_axis,
                length=length,
                semantics=semantics,
            )
            extras.append(
                {
                    "role": alignment_overlay_extra_role(semantics),
                    "label": alignment_overlay_extra_role(semantics),
                    "source_label": pad.get("source_label") or pad.get("label"),
                    "bbox": new_bbox,
                    "source_type": "derived_partial_evidence_alignment",
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
                    "raw_view": selected.get("raw_view"),
                    "canonical_view": selected.get("canonical_view"),
                    "radial_axis": radial_axis,
                    "projection_axis": projection_axis,
                    "coordinate_unit_scale": unit_scale,
                }
            )
    return extras


def select_alignment_overlay_base_item(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    order = {"bottom": 0, "top": 1, "land": 2}
    candidates = []
    for item in items:
        if str(item.get("raw_view") or "") not in order:
            continue
        pads = terminal_alignment_pad_objects(item["graph"])
        if not pads:
            continue
        candidates.append((order[str(item.get("raw_view") or "")], -len(pads), str(item.get("graph_path") or ""), item))
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: row[:3])[0][3]


def terminal_alignment_pad_objects(graph: dict[str, Any]) -> list[tuple[dict[str, Any], tuple[float, float, float, float]]]:
    pads = []
    for obj in graph.get("objects") or []:
        bbox = graph_object_bbox(obj)
        if bbox is None or not graph_object_is_pad_like(obj):
            continue
        pads.append((obj, bbox))
    if len(pads) < 5:
        return pads
    areas = sorted((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for _obj, bbox in pads)
    median_area = areas[len(areas) // 2]
    extent = union_bbox([list(bbox) for _obj, bbox in pads])
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


def alignment_graph_contact_unit_scales(graph: dict[str, Any]) -> dict[str, float]:
    metrics = graph.get("metrics") or {}
    global_scale = positive_float(metrics.get("global_scale"))
    dimension_scales = graph_dimension_unit_scales(graph)
    x_scale = positive_float(metrics.get("axis_scale_x")) or global_scale or positive_float(dimension_scales.get("x"))
    y_scale = positive_float(metrics.get("axis_scale_y")) or global_scale or positive_float(dimension_scales.get("y"))
    if x_scale is None or y_scale is None:
        return {}
    return {"x": x_scale, "y": y_scale}


def alignment_partial_contact_dimensions(
    canonical: dict[str, Any],
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    items_by_graph = {str(item.get("graph_path") or ""): item for item in items}
    dimensions = []
    for ref in canonical.get("evidence_refs") or []:
        if str(ref.get("evidence_type") or "") != "package_graph":
            continue
        raw_view = str(ref.get("raw_view") or "").strip().lower()
        canonical_view = normalized_alignment_view(str(ref.get("canonical_view") or raw_view))
        if canonical_view not in {"lateral", "lead_detail"}:
            continue
        graph_path = str(ref.get("graph_path") or ref.get("path") or "").strip()
        item = items_by_graph.get(graph_path)
        if item is None:
            continue
        graph = item.get("graph") or {}
        objects_by_id = {str(obj.get("id")): obj for obj in graph.get("objects") or [] if obj.get("id") is not None}
        for dim in graph.get("dimensions") or []:
            semantics = alignment_overlay_dimension_semantics(dim, raw_view)
            if not is_alignment_overlay_contact_dimension(dim, objects_by_id, semantics):
                continue
            enriched = dict(dim)
            enriched["raw_view"] = raw_view
            enriched["canonical_view"] = canonical_view
            enriched["source_graph"] = graph_path
            enriched["annotation_path"] = ref.get("annotation_path")
            enriched["target_labels"] = alignment_dimension_target_labels(dim, objects_by_id)
            enriched["overlay_semantics"] = semantics
            dimensions.append(enriched)
    return dimensions


def alignment_overlay_dimension_semantics(dim: dict[str, Any], raw_view: str) -> str:
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


def is_alignment_overlay_contact_dimension(
    dim: dict[str, Any],
    objects_by_id: dict[str, dict[str, Any]],
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
    labels = alignment_dimension_target_labels(dim, objects_by_id)
    if not labels:
        return False
    label_text = " ".join(labels).lower()
    if "outline" in label_text or "package" in label_text:
        return False
    return any(token in label_text for token in ("pad", "lead", "rect", "circle", "dshape"))


def alignment_dimension_target_labels(dim: dict[str, Any], objects_by_id: dict[str, dict[str, Any]]) -> list[str]:
    labels = []
    for target_id in dim.get("target_ids") or []:
        obj = objects_by_id.get(str(target_id))
        if obj is None:
            continue
        label = " ".join(str(obj.get(key) or "") for key in ("label", "source_label", "shape")).strip()
        if label:
            labels.append(label.lower())
    return sorted(set(labels))


def alignment_overlay_projection_axis(dim: dict[str, Any], radial_axis: str) -> str:
    semantics = str(dim.get("overlay_semantics") or "")
    raw_view = str(dim.get("raw_view") or "").strip().lower()
    if semantics in {"pad_width", "lead_pad_length", "lead_ground_contact_length"}:
        if raw_view == "side":
            return "y"
        if raw_view == "front":
            return "x"
        return radial_axis
    return radial_axis


def effective_alignment_partial_dimension_semantics(
    *,
    dim: dict[str, Any],
    base_semantics: str,
    projection_axis: str,
    dimension_value: float,
    bbox: tuple[float, float, float, float],
    unit_scale: float,
) -> str:
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


def alignment_partial_dimension_overlay_bbox(
    *,
    bbox: tuple[float, float, float, float],
    package_center: tuple[float, float],
    package_frame: tuple[float, float, float, float] | None,
    projection_axis: str,
    length: float,
    semantics: str,
) -> list[float]:
    x1, y1, x2, y2 = bbox
    pad_cx = (x1 + x2) / 2.0
    pad_cy = (y1 + y2) / 2.0
    center_x, center_y = package_center
    if semantics in {"pad_width", "lead_pad_length"}:
        if projection_axis == "x":
            return [pad_cx - length / 2.0, y1, pad_cx + length / 2.0, y2]
        return [x1, pad_cy - length / 2.0, x2, pad_cy + length / 2.0]
    radial_axis = "x" if abs(pad_cx - center_x) >= abs(pad_cy - center_y) else "y"
    if semantics == "lead_ground_contact_length" and projection_axis != radial_axis:
        outside_side = alignment_pad_outside_side(bbox, package_frame)
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


def alignment_pad_outside_side(
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


def alignment_overlay_extra_role(semantics: str) -> str:
    if semantics == "pad_width":
        return "partial_pad_width"
    if semantics == "lead_pad_length":
        return "partial_lead_pad_length"
    return "lead_pad"


def normalized_alignment_view(view: str) -> str:
    value = view.strip().lower()
    if value in {"side", "front"}:
        return "lateral"
    if value in {"lead", "land_detail"}:
        return "lead_detail"
    return value or "unknown"


def dedupe_alignment_render_objects(objects: list[dict[str, Any]], *, bbox_tol: float = 0.001) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for obj in objects:
        bbox = graph_object_bbox(obj)
        if bbox is None:
            continue
        source = obj.get("lead_contact_length_source") or obj.get("inner_land_pad_source") or {}
        key = (
            str(obj.get("role") or ""),
            str(obj.get("source_graph") or ""),
            str(source.get("raw_view") or obj.get("raw_view") or ""),
            round(bbox[0] / bbox_tol),
            round(bbox[1] / bbox_tol),
            round(bbox[2] / bbox_tol),
            round(bbox[3] / bbox_tol),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(obj)
    return deduped


def first_reference_layer(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not items:
        return None
    order = {view: index for index, view in enumerate(ALIGNMENT_VIEW_ORDER)}
    return sorted(
        items,
        key=lambda item: (
            int(item["match_priority"]) if isinstance(item.get("match_priority"), (int, float)) else 9,
            order.get(str(item.get("raw_view") or "unknown"), 99),
            -len(item.get("match_boxes") or []),
            str(item.get("graph_path") or ""),
        ),
    )[0]


def staged_rotation_items(
    reference: dict[str, Any],
    items: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], str]]:
    # This stage only chooses rotation degrees. It does not accept/reject
    # matches and does not synthesize geometry. Later stages may interpret
    # whether the rotated layers are semantically compatible.
    reference_key = reference["key"]
    remaining = [item for item in items if item["key"] != reference_key]
    lateral_views = {"side", "front", "lead", "land_detail"}
    lateral = [item for item in remaining if str(item.get("raw_view") or "") in lateral_views]
    main = [item for item in remaining if str(item.get("raw_view") or "") in {"bottom", "top"}]
    other = [item for item in remaining if item not in lateral and item not in main]

    order = {view: index for index, view in enumerate(ALIGNMENT_VIEW_ORDER)}
    lateral_sorted = sorted(
        lateral,
        key=lambda item: (
            int(item["match_priority"]) if isinstance(item.get("match_priority"), (int, float)) else 9,
            order.get(str(item.get("raw_view") or "unknown"), 99),
            str(item.get("graph_path") or ""),
        ),
    )
    main_sorted = sorted(
        main,
        key=lambda item: (
            -len(item.get("match_boxes") or []),
            0 if str(item.get("raw_view") or "") == "bottom" else 1,
            str(item.get("graph_path") or ""),
        ),
    )
    other_sorted = sorted(
        other,
        key=lambda item: (
            int(item["match_priority"]) if isinstance(item.get("match_priority"), (int, float)) else 9,
            order.get(str(item.get("raw_view") or "unknown"), 99),
            str(item.get("graph_path") or ""),
        ),
    )
    return (
        [(item, "rotate_lateral_to_seed") for item in lateral_sorted]
        + [(item, "rotate_main_to_merged_seed") for item in main_sorted]
        + [(item, "rotate_remaining_to_seed") for item in other_sorted]
    )


def transformed_match_boxes(item: dict[str, Any], transform: dict[str, Any]) -> list[list[float]]:
    boxes = []
    for box in item.get("match_boxes") or []:
        transformed = transform_bbox_dimension_scaled_centered_rotated(list(box), transform)
        if transformed is not None:
            boxes.append(transformed)
    return boxes


def best_layer_rotation(
    *,
    reference_boxes: list[list[float]],
    reference_frame: list[float],
    reference_unit_scales: dict[str, Any],
    candidate_boxes: list[list[float]],
    candidate_frame: list[float],
    candidate_unit_scales: dict[str, Any],
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    scaled_frame = scale_bbox_to_target_units(candidate_frame, candidate_frame, candidate_unit_scales, reference_unit_scales)
    translation = center_translation(scaled_frame, reference_frame)
    aligned_frame = translate_bbox(scaled_frame, *translation)
    for rotation in ROTATION_CANDIDATES:
        transformed = [
            transformed
            for box in candidate_boxes
            if (
                transformed := rotate_bbox_in_box(
                    translate_bbox(
                        scale_bbox_to_target_units(box, candidate_frame, candidate_unit_scales, reference_unit_scales),
                        *translation,
                    ),
                    aligned_frame,
                    rotation,
                )
            )
            is not None
        ]
        score = matched_box_iou_score(reference_boxes, transformed)
        score_value = float(score) if isinstance(score, (int, float)) else 0.0
        candidate = {
            "rotation_degrees": int(rotation),
            "iou": score_value,
            "rotation_cost": rotation_cost(rotation),
        }
        if best is None or layer_rotation_sort_key(candidate) > layer_rotation_sort_key(best):
            best = candidate
    return best or {"rotation_degrees": 0, "iou": 0.0, "rotation_cost": 0}


def layer_rotation_sort_key(candidate: dict[str, Any]) -> tuple[float, int, int]:
    return (
        float(candidate.get("iou") or 0.0),
        -int(candidate.get("rotation_cost") or 0),
        -int(candidate.get("rotation_degrees") or 0),
    )


def rotation_cost(rotation_degrees: int) -> int:
    turns = (int(rotation_degrees) // 90) % 4
    return min(turns, 4 - turns)


def bbox_center_xy(box: list[float] | tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def center_translation(
    source_frame: list[float] | tuple[float, float, float, float],
    target_frame: list[float] | tuple[float, float, float, float],
) -> tuple[float, float]:
    source_x, source_y = bbox_center_xy(source_frame)
    target_x, target_y = bbox_center_xy(target_frame)
    return (target_x - source_x, target_y - source_y)


def translate_bbox(bbox: list[float], dx: float, dy: float) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    return [x1 + dx, y1 + dy, x2 + dx, y2 + dy]


def graph_dimension_unit_scales(graph: dict[str, Any]) -> dict[str, Any]:
    """Estimate physical units per graph pixel from accepted dimensions.

    Coordinate system: graph object bboxes are reconstructed graph coordinates
    with x-right/y-down axes. Dimension values are treated as physical package
    units. The returned x/y scales convert graph pixels into physical units.
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
    return result if result > 0.0 else None


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
    target_boxes = [graph_object_bbox(objects_by_id[target_id]) for target_id in target_ids if target_id in objects_by_id]
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


def scale_bbox_to_target_units(
    bbox: list[float],
    source_frame: list[float],
    source_unit_scales: dict[str, Any],
    target_unit_scales: dict[str, Any],
) -> list[float]:
    source_x = positive_float(source_unit_scales.get("x")) or 1.0
    source_y = positive_float(source_unit_scales.get("y")) or 1.0
    target_x = positive_float(target_unit_scales.get("x")) or 1.0
    target_y = positive_float(target_unit_scales.get("y")) or 1.0
    ratio_x = source_x / target_x
    ratio_y = source_y / target_y
    cx, cy = bbox_center_xy(source_frame)
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    return [
        cx + (x1 - cx) * ratio_x,
        cy + (y1 - cy) * ratio_y,
        cx + (x2 - cx) * ratio_x,
        cy + (y2 - cy) * ratio_y,
    ]


def transform_bbox_dimension_scaled_centered_rotated(
    bbox: list[float],
    transform: dict[str, Any],
) -> list[float] | None:
    scaled = scale_bbox_to_target_units(
        bbox,
        transform["source_frame"],
        transform["source_unit_scales"],
        transform["target_unit_scales"],
    )
    centered = translate_bbox(scaled, *transform["center_translation"])
    return rotate_bbox_in_box(centered, transform["aligned_source_frame"], int(transform["rotation_degrees"]))


def layer_transform_payload(
    item: dict[str, Any],
    *,
    target_frame: list[float],
    target_unit_scales: dict[str, Any],
    rotation_degrees: int,
    rotation_iou: float | None,
    followed_view: str,
    strategy: str,
    alignment_stage: str,
) -> dict[str, Any]:
    scaled_frame = scale_bbox_to_target_units(item["frame"], item["frame"], item["unit_scales"], target_unit_scales)
    translation = center_translation(scaled_frame, target_frame)
    aligned_frame = translate_bbox(scaled_frame, *translation)
    return {
        "raw_view": item["raw_view"],
        "canonical_view": item["canonical_view"],
        "graph_path": item["graph_path"],
        "source_frame": item["frame"],
        "target_frame": target_frame,
        "source_unit_scales": item["unit_scales"],
        "target_unit_scales": target_unit_scales,
        "scaled_source_frame": scaled_frame,
        "aligned_source_frame": aligned_frame,
        "rotation_degrees": int(rotation_degrees),
        "rotation_cost": rotation_cost(rotation_degrees),
        "rotation_iou": rotation_iou,
        "followed_view": followed_view,
        "strategy": strategy,
        "alignment_stage": alignment_stage,
        "match_source": item.get("match_source"),
        "match_priority": item.get("match_priority"),
        "match_role_counts": item.get("match_role_counts") or {},
        "center_translation": list(translation),
        "coordinate_mode": "dimension_scale_center_translate_then_rotate",
    }


def transforms_by_raw_view(
    items: list[dict[str, Any]],
    transforms: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_view = {}
    for item in items:
        transform = transforms.get(item["key"])
        if transform is None:
            continue
        view = str(item["raw_view"])
        current = by_view.get(view)
        if current is None or transform_priority_key(transform) > transform_priority_key(current):
            by_view[view] = transform
    return by_view


def transform_priority_key(transform: dict[str, Any]) -> tuple[int, float, int]:
    strategy_rank = 2 if transform.get("strategy") == "reference" else 1
    iou = transform.get("rotation_iou")
    return (strategy_rank, float(iou) if isinstance(iou, (int, float)) else 0.0, -int(transform.get("rotation_cost") or 0))


def followed_transform_for_layer(
    item: dict[str, Any],
    transforms_by_view: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    for view in FOLLOW_ALIGNMENT_VIEWS.get(str(item["raw_view"]), ()):
        transform = transforms_by_view.get(view)
        if transform is not None:
            return transform
    return None


def transform_summary_payload(transform: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_view": transform.get("raw_view"),
        "canonical_view": transform.get("canonical_view"),
        "graph_path": transform.get("graph_path"),
        "rotation_degrees": transform.get("rotation_degrees"),
        "rotation_cost": transform.get("rotation_cost"),
        "rotation_iou": transform.get("rotation_iou"),
        "followed_view": transform.get("followed_view"),
        "strategy": transform.get("strategy"),
        "alignment_stage": transform.get("alignment_stage"),
        "match_source": transform.get("match_source"),
        "match_priority": transform.get("match_priority"),
        "match_role_counts": transform.get("match_role_counts") or {},
        "source_unit_scales": transform.get("source_unit_scales"),
        "target_unit_scales": transform.get("target_unit_scales"),
        "scaled_source_frame": transform.get("scaled_source_frame"),
        "aligned_source_frame": transform.get("aligned_source_frame"),
        "center_translation": transform.get("center_translation"),
        "coordinate_mode": transform.get("coordinate_mode"),
    }


def aligned_layer_objects(items: list[dict[str, Any]], transforms: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    objects = []
    for item in items:
        transform = transforms.get(item["key"])
        if transform is None:
            continue
        for obj in item.get("render_objects") or []:
            role = str(obj.get("role") or layer_object_role(obj, item["raw_view"]))
            if not role:
                continue
            bbox = graph_object_bbox(obj)
            if bbox is None:
                continue
            transformed = transform_bbox_with_source_orientation(
                list(bbox),
                transform["source_frame"],
                transform["target_frame"],
                "",
                int(transform["rotation_degrees"]),
            )
            if transformed is None:
                continue
            objects.append(
                {
                    "role": role,
                    "bbox": transformed,
                    "label": str(obj.get("label") or obj.get("source_label") or obj.get("shape") or role),
                    "source_label": obj.get("source_label"),
                    "source_type": obj.get("source_type") or "multiview_canonical_layer",
                    "source_path": obj.get("source_path") or item["graph_path"],
                    "source_graph": obj.get("source_graph") or item["graph_path"],
                    "source_object_id": obj.get("source_object_id", obj.get("id")),
                    "raw_view": obj.get("raw_view") or item["raw_view"],
                    "canonical_view": obj.get("canonical_view") or item["canonical_view"],
                    "alignment_rotation_degrees": int(transform["rotation_degrees"]),
                    "alignment_rotation_iou": transform.get("rotation_iou"),
                    "alignment_followed_view": transform.get("followed_view"),
                    "alignment_strategy": transform.get("strategy"),
                    "alignment_match_source": transform.get("match_source"),
                    "alignment_match_priority": transform.get("match_priority"),
                    "alignment_source_bbox": transform["source_frame"],
                    "alignment_target_bbox": transform["target_frame"],
                }
            )
    return objects


def rotation_only_layer_objects(items: list[dict[str, Any]], transforms: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    objects = []
    for item in items:
        transform = transforms.get(item["key"])
        if transform is None:
            continue
        for obj in item.get("render_objects") or []:
            role = str(obj.get("role") or layer_object_role(obj, item["raw_view"]))
            if not role:
                continue
            bbox = graph_object_bbox(obj)
            if bbox is None:
                continue
            transformed = transform_bbox_dimension_scaled_centered_rotated(list(bbox), transform)
            if transformed is None:
                continue
            objects.append(
                {
                    "role": role,
                    "bbox": transformed,
                    "label": str(obj.get("label") or obj.get("source_label") or obj.get("shape") or role),
                    "source_label": obj.get("source_label"),
                    "source_type": obj.get("source_type") or "multiview_canonical_layer",
                    "source_path": obj.get("source_path") or item["graph_path"],
                    "source_graph": obj.get("source_graph") or item["graph_path"],
                    "source_object_id": obj.get("source_object_id", obj.get("id")),
                    "raw_view": obj.get("raw_view") or item["raw_view"],
                    "canonical_view": obj.get("canonical_view") or item["canonical_view"],
                    "alignment_rotation_degrees": int(transform["rotation_degrees"]),
                    "alignment_rotation_iou": transform.get("rotation_iou"),
                    "alignment_followed_view": transform.get("followed_view"),
                    "alignment_strategy": transform.get("strategy"),
                    "alignment_match_source": transform.get("match_source"),
                    "alignment_match_priority": transform.get("match_priority"),
                    "alignment_display_mode": "dimension_scaled_centered_rotated",
                    "alignment_source_bbox": transform["source_frame"],
                    "alignment_target_bbox": transform["target_frame"],
                    "alignment_center_translation": transform.get("center_translation"),
                    "alignment_scaled_source_bbox": transform.get("scaled_source_frame"),
                    "alignment_aligned_source_bbox": transform.get("aligned_source_frame"),
                    "source_package_pad_id": obj.get("source_package_pad_id"),
                    "source_package_pad_bbox": obj.get("source_package_pad_bbox"),
                    "source_package_pad_index": obj.get("source_package_pad_index"),
                    "lead_contact_length": obj.get("lead_contact_length"),
                    "lead_contact_length_axis": obj.get("lead_contact_length_axis"),
                    "lead_contact_length_source": obj.get("lead_contact_length_source"),
                    "partial_dimension_semantics": obj.get("partial_dimension_semantics"),
                    "partial_dimension_base_semantics": obj.get("partial_dimension_base_semantics"),
                    "projection_axis": obj.get("projection_axis"),
                    "radial_axis": obj.get("radial_axis"),
                    "coordinate_unit_scale": obj.get("coordinate_unit_scale"),
                }
            )
    return objects


def grouped_layer_objects(objects: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for obj in objects:
        grouped.setdefault(str(obj.get("raw_view") or "unknown"), []).append(obj)
    order = {view: index for index, view in enumerate(ALIGNMENT_VIEW_ORDER)}
    return [(view, grouped[view]) for view in sorted(grouped, key=lambda value: (order.get(value, 99), value))]


def layer_match_boxes(graph: dict[str, Any]) -> list[list[float]]:
    return [list(bbox) for obj in graph.get("objects") or [] if (bbox := graph_object_bbox(obj)) and graph_object_is_pad_like(obj)]


def layer_frame(graph: dict[str, Any]) -> list[float] | None:
    match_boxes = layer_match_boxes(graph)
    if match_boxes:
        return union_bbox(match_boxes)
    boxes = [list(bbox) for obj in graph.get("objects") or [] if (bbox := graph_object_bbox(obj))]
    return union_bbox(boxes)


def layer_object_role(obj: dict[str, Any], raw_view: str) -> str:
    label = graph_object_label(obj)
    if "outline" in label or "package" in label:
        return "outline"
    if not graph_object_is_pad_like(obj):
        return ""
    if raw_view == "land":
        return "land"
    if raw_view == "land_detail":
        return "inner_land_pad"
    if raw_view in {"side", "front", "lead"}:
        return "lead_pad"
    return "package_pad"


def graph_object_is_pad_like(obj: dict[str, Any]) -> bool:
    label = graph_object_label(obj)
    if "outline" in label or "package" in label:
        return False
    return any(token in label for token in ("pad", "rect", "circle", "dshape"))


def graph_object_label(obj: dict[str, Any]) -> str:
    return " ".join(str(obj.get(key) or "") for key in ("label", "source_label", "shape")).lower()


def graph_object_bbox(obj: dict[str, Any]) -> tuple[float, float, float, float] | None:
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


def write_multiview_alignment_svgs(
    summary: dict[str, Any],
    gt: dict[str, Any],
    canonical: dict[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    canonical_path = Path(str(summary.get("unified_multiview_layers_path") or ""))
    if not canonical_path.parent.exists():
        return
    part_dir = canonical_path.parent
    gt_reference_path = part_dir / "gt_reference.svg"
    default_aligned_result_path = part_dir / "default_aligned_result.svg"
    default_comparison_path = part_dir / "default_comparison.svg"
    aligned_result_path = part_dir / "aligned_result.svg"
    alignment_graph_path = part_dir / "alignment_graph.svg"
    comparison_path = part_dir / "comparison.svg"
    final_graph_path = part_dir / "final_graph.json"
    gt_objects = scan_result_objects(gt) if gt else []
    layer_alignment = align_overlay_layers(canonical) if canonical else {"objects": [], "layers": [], "scores": {}, "summary": {}}
    result_objects = layer_alignment.get("objects") or []
    result_layers = layer_alignment.get("layers") or [("result", result_objects)]
    score_payload = layer_alignment.get("scores") or {}
    transform_summary = layer_alignment.get("summary") or {}
    write_scene_svg(
        gt_reference_path,
        title=f"{summary.get('part_number')}: GT reference",
        layers=[("gt", gt_objects)],
        fallback_label=f"{summary.get('part_number')}: no GT reference geometry",
    )
    write_scene_svg(
        default_aligned_result_path,
        title=f"{summary.get('part_number')}: rotation-centered overlay",
        layers=result_layers,
        fallback_label=f"{summary.get('part_number')}: no rotation-centered overlay geometry",
    )
    write_scene_svg(
        default_comparison_path,
        title=f"{summary.get('part_number')}: rotation-centered overlay",
        layers=result_layers,
        fallback_label=f"{summary.get('part_number')}: no rotation-centered overlay geometry",
    )
    write_scene_svg(
        aligned_result_path,
        title=f"{summary.get('part_number')}: rotation-centered overlay",
        layers=result_layers,
        fallback_label=f"{summary.get('part_number')}: no rotation-centered overlay geometry",
    )
    write_scene_svg(
        alignment_graph_path,
        title=f"{summary.get('part_number')}: rotation-centered overlay",
        layers=result_layers,
        fallback_label=f"{summary.get('part_number')}: no rotation-centered overlay geometry",
    )
    write_scene_svg(
        comparison_path,
        title=f"{summary.get('part_number')}: GT vs rotation-centered overlay",
        layers=[("gt", gt_objects), *list(result_layers)],
        fallback_label=f"{summary.get('part_number')}: no comparison geometry",
    )
    summary["gt_reference_svg_path"] = str(gt_reference_path)
    summary["default_aligned_result_svg_path"] = str(default_aligned_result_path)
    summary["default_comparison_svg_path"] = str(default_comparison_path)
    summary["aligned_result_svg_path"] = str(aligned_result_path)
    summary["alignment_graph_svg_path"] = str(alignment_graph_path)
    summary["comparison_svg_path"] = str(comparison_path)
    summary["final_graph_path"] = str(final_graph_path)
    summary["alignment_scores"] = score_payload
    summary["alignment_transform"] = transform_summary
    apply_score_diagnostics(summary, canonical)
    write_final_graph_json(final_graph_path, summary, gt_objects, canonical, result_objects, score_payload, transform_summary)


def write_final_graph_json(
    output_path: Path,
    summary: dict[str, Any],
    gt_objects: list[dict[str, Any]],
    canonical: dict[str, Any],
    result_objects: list[dict[str, Any]],
    alignment_scores_payload: dict[str, Any],
    transform_summary: dict[str, Any],
) -> None:
    objects = [final_graph_object(obj) for obj in result_objects]
    by_role: dict[str, list[dict[str, Any]]] = {}
    for obj in objects:
        by_role.setdefault(str(obj.get("role") or "unknown"), []).append(obj)
    outline = by_role.get("outline", [])
    land_pads = [*by_role.get("land", []), *by_role.get("land_pad", [])]
    lead_contacts = [*by_role.get("lead", []), *by_role.get("lead_contact", [])]
    lead_pads = [
        *by_role.get("lead_pad", []),
        *by_role.get("partial_pad_width", []),
        *by_role.get("partial_lead_pad_length", []),
    ]
    payload = {
        "part_number": summary.get("part_number"),
        "status": summary.get("status"),
        "coordinate_system": {
            "name": "multiview_dimension_scaled_centered_rotated_2d",
            "unit": "selected reference package-graph pixels after dimension-scale normalization",
            "x_axis": "right",
            "y_axis": "down",
            "source_stage": "gt_alignment",
            "note": "Source package-graph layers are copied from the multiview result, normalized by accepted-dimension unit scales into the selected reference layer scale, translated so scaled layer frame centers share the reference center, then rotated by 0/90/180/270 degrees. No GT/reference-frame remapping or SVG display scaling is applied. ScanResultFormat GT is preserved only as review reference.",
        },
        "source_unified_multiview_layers_path": summary.get("unified_multiview_layers_path"),
        "alignment_transform": transform_summary,
        "alignment_scores": alignment_scores_payload,
        "source_views": canonical.get("source_views") or (canonical.get("summary") or {}).get("source_views") or [],
        "canonical_source_views": canonical.get("canonical_source_views")
        or (canonical.get("summary") or {}).get("canonical_source_views")
        or [],
        "outline_2d": outline[0] if outline else {},
        "package_pads": by_role.get("package_pad", []),
        "land_pads": land_pads,
        "lead_contacts": lead_contacts,
        "lead_pads": lead_pads,
        "inner_land_pads": by_role.get("inner_land_pad", []),
        "dimensions": canonical.get("dimensions") or [],
        "objects": objects,
        "evidence_refs": canonical.get("evidence_refs") or [],
        "conflicts": canonical.get("conflicts") or [],
        "gt_reference": {
            "scan_result_path": summary.get("scan_result_path"),
            "objects": [final_graph_object(obj) for obj in gt_objects],
        },
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def final_graph_object(obj: dict[str, Any]) -> dict[str, Any]:
    keep_keys = (
        "role",
        "bbox",
        "label",
        "source",
        "source_type",
        "source_path",
        "source_graph",
        "source_object_id",
        "source_label",
        "raw_view",
        "canonical_view",
        "alignment_display_mode",
        "alignment_source_bbox",
        "alignment_target_bbox",
        "alignment_scaled_source_bbox",
        "alignment_aligned_source_bbox",
        "alignment_center_translation",
        "alignment_rotation_degrees",
        "alignment_rotation_iou",
        "alignment_followed_view",
        "alignment_strategy",
        "alignment_match_source",
        "alignment_match_priority",
        "alignment_package_pad_flip",
        "alignment_package_pad_rotation",
        "source_package_pad_id",
        "source_package_pad_bbox",
        "source_package_pad_index",
        "lead_contact_length",
        "lead_contact_length_axis",
        "lead_contact_length_source",
        "partial_dimension_semantics",
        "partial_dimension_base_semantics",
        "projection_axis",
        "radial_axis",
        "coordinate_unit_scale",
    )
    return {key: obj[key] for key in keep_keys if key in obj and obj[key] is not None}


def scan_result_objects(gt: dict[str, Any]) -> list[dict[str, Any]]:
    objects = []
    for obj in effective_scan_result_objects(gt):
        bbox = obj.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        objects.append(
            {
                "source_object_id": obj.get("id"),
                "role": str(obj.get("role") or "unknown"),
                "bbox": [float(value) for value in bbox[:4]],
                "label": str(obj.get("node_name") or obj.get("role") or "gt"),
                "source": "gt",
            }
        )
    return objects


def effective_scan_result_objects(gt: dict[str, Any]) -> list[dict[str, Any]]:
    features = gt.get("features") or {}
    objects = features.get("objects") or []
    effective_objects = features.get("effective_objects")
    if isinstance(effective_objects, list):
        return effective_objects
    role_counts = features.get("role_counts") or {}
    tolerance = float(features.get("dedupe_center_tol") or 0.01)
    return effective_scan_result_raw_objects(objects, role_counts, tolerance)


def effective_scan_result_raw_objects(
    objects: list[dict[str, Any]],
    role_counts: dict[str, Any] | Counter[str],
    tolerance: float,
) -> list[dict[str, Any]]:
    if not role_counts:
        return objects
    raw_roles = {str(obj.get("role") or "") for obj in objects}
    effective_by_role = {
        role: {id(obj) for obj in effective_scan_result_role_objects(objects, role, int(role_counts.get(role) or 0), tolerance)}
        for role in ("land", "lead")
        if role in raw_roles
    }
    effective = []
    for obj in objects:
        role = str(obj.get("role") or "")
        if role in effective_by_role and id(obj) not in effective_by_role[role]:
            continue
        effective.append(obj)
    return effective


def effective_scan_result_role_objects(
    objects: list[dict[str, Any]],
    role: str,
    effective_count: int,
    tolerance: float,
) -> list[dict[str, Any]]:
    role_objects = [
        obj
        for obj in objects
        if str(obj.get("role") or "") == role and len(obj.get("bbox") or []) >= 4
    ]
    if effective_count <= 0:
        return []
    if len(role_objects) <= effective_count:
        return role_objects

    lead_bbox_areas = sorted(
        bbox_area(obj.get("bbox"))
        for obj in objects
        if str(obj.get("role") or "") == "lead" and bbox_area(obj.get("bbox")) > 0.0
    )
    min_lead_area = lead_bbox_areas[0] if lead_bbox_areas else None
    overlap_centers = large_land_lead_overlap_center_keys(objects, min_lead_area, tolerance)
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int] | tuple[str, str, int]] = set()
    for index, obj in enumerate(role_objects):
        bbox = obj.get("bbox")
        if (
            min_lead_area is not None
            and center_dedupe_key(bbox, tolerance) in overlap_centers
            and bbox_area(bbox) > min_lead_area * 2.0
        ):
            continue
        if bbox and len(bbox) >= 4 and tolerance > 0:
            center_x = (float(bbox[0]) + float(bbox[2])) / 2.0
            center_y = (float(bbox[1]) + float(bbox[3])) / 2.0
            key: tuple[str, int, int] | tuple[str, str, int] = (
                role,
                round(center_x / tolerance),
                round(center_y / tolerance),
            )
        else:
            key = (role, "raw", index)
        if key in seen:
            continue
        seen.add(key)
        selected.append(obj)

    if len(selected) > effective_count:
        selected = sorted(selected, key=lambda obj: (bbox_area(obj.get("bbox")), object_id_sort_key(obj)))[:effective_count]
        selected = sorted(selected, key=object_id_sort_key)
    if len(selected) < effective_count:
        selected_ids = {id(obj) for obj in selected}
        for obj in role_objects:
            if id(obj) in selected_ids:
                continue
            selected.append(obj)
            selected_ids.add(id(obj))
            if len(selected) >= effective_count:
                break
    return selected


def object_id_sort_key(obj: dict[str, Any]) -> tuple[str, float, float]:
    bbox = obj.get("bbox") or []
    if len(bbox) >= 4:
        return (str(obj.get("id") or ""), float(bbox[1]), float(bbox[0]))
    return (str(obj.get("id") or ""), 0.0, 0.0)


def select_aligned_result(
    gt: dict[str, Any],
    canonical: dict[str, Any],
    checks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    default_objects = aligned_canonical_objects(gt, canonical)
    default_scores = alignment_scores(gt, canonical, default_objects, checks)
    default_summary = {
        "strategy": "default",
        "package_pad_source_order": ["package", "lead", "all"],
        "overall_score": default_scores.get("overall_score"),
        "quality_score": default_scores.get("quality_score"),
    }
    best_objects = default_objects
    best_scores = default_scores
    best_summary = default_summary
    flip_candidates = package_pad_flip_candidates(default_scores, canonical)
    for flip_axis in flip_candidates:
        for exclude_partial_leads in partial_lead_exclusion_options(gt, canonical):
            candidate_objects = aligned_canonical_objects(
                gt,
                canonical,
                package_pad_flip=flip_axis,
                exclude_lead_contacts=exclude_partial_leads,
            )
            candidate_scores = alignment_scores(gt, canonical, candidate_objects, checks)
            candidate_summary = {
                "strategy": (
                    "package_pad_flip_without_partial_lead_detail"
                    if exclude_partial_leads
                    else "package_pad_flip_for_low_score"
                ),
                "package_pad_source_order": ["package", "lead", "all"],
                "package_pad_flip": flip_axis,
                "excluded_partial_lead_contacts": exclude_partial_leads,
                "default_overall_score": default_scores.get("overall_score"),
                "default_quality_score": default_scores.get("quality_score"),
                "selected_overall_score": candidate_scores.get("overall_score"),
                "selected_quality_score": candidate_scores.get("quality_score"),
            }
            if alignment_candidate_is_better(candidate_scores, candidate_objects, best_scores, best_objects, gt):
                best_objects = candidate_objects
                best_scores = candidate_scores
                best_summary = candidate_summary
    for rotation in package_pad_rotation_candidates(default_scores, canonical):
        for flip_axis in ["", *flip_candidates]:
            for exclude_partial_leads in partial_lead_exclusion_options(gt, canonical):
                candidate_objects = aligned_canonical_objects(
                    gt,
                    canonical,
                    package_pad_flip=flip_axis,
                    package_pad_rotation=rotation,
                    exclude_lead_contacts=exclude_partial_leads,
                )
                candidate_scores = alignment_scores(gt, canonical, candidate_objects, checks)
                candidate_summary = {
                    "strategy": (
                        "package_pad_rotation_without_partial_lead_detail"
                        if exclude_partial_leads
                        else "package_pad_rotation_for_low_score"
                    ),
                    "package_pad_source_order": ["package", "lead", "all"],
                    "package_pad_rotation": rotation,
                    "excluded_partial_lead_contacts": exclude_partial_leads,
                    "default_overall_score": default_scores.get("overall_score"),
                    "default_quality_score": default_scores.get("quality_score"),
                    "selected_overall_score": candidate_scores.get("overall_score"),
                    "selected_quality_score": candidate_scores.get("quality_score"),
                }
                if flip_axis:
                    candidate_summary["package_pad_flip"] = flip_axis
                if alignment_candidate_is_better(candidate_scores, candidate_objects, best_scores, best_objects, gt):
                    best_objects = candidate_objects
                    best_scores = candidate_scores
                    best_summary = candidate_summary
    for label in package_pad_terminal_label_candidates(gt, canonical):
        candidate_objects = aligned_canonical_objects(
            gt,
            canonical,
            package_pad_source_order=(f"package_{label}", "package", "lead", "all"),
            package_pad_label_filter=label,
        )
        candidate_scores = alignment_scores(gt, canonical, candidate_objects, checks)
        candidate_summary = {
            "strategy": "package_pad_terminal_label_subset_for_low_score",
            "package_pad_source_order": [f"package_{label}", "package", "lead", "all"],
            "package_pad_label_filter": label,
            "default_overall_score": default_scores.get("overall_score"),
            "default_quality_score": default_scores.get("quality_score"),
            "selected_overall_score": candidate_scores.get("overall_score"),
            "selected_quality_score": candidate_scores.get("quality_score"),
        }
        if alignment_score_tuple(candidate_scores) > alignment_score_tuple(best_scores):
            best_objects = candidate_objects
            best_scores = candidate_scores
            best_summary = candidate_summary

    for source_order in package_pad_bbox_source_order_candidates(default_scores, canonical):
        for flip_axis in ["", *flip_candidates]:
            for exclude_partial_leads in partial_lead_exclusion_options(gt, canonical):
                candidate_objects = aligned_canonical_objects(
                    gt,
                    canonical,
                    package_pad_source_order=source_order,
                    package_pad_flip=flip_axis,
                    exclude_lead_contacts=exclude_partial_leads,
                )
                candidate_scores = alignment_scores(gt, canonical, candidate_objects, checks)
                candidate_summary = {
                    "strategy": (
                        "package_pad_conductive_bbox_without_partial_lead_detail"
                        if exclude_partial_leads
                        else "package_pad_conductive_bbox_for_low_score"
                    ),
                    "package_pad_source_order": list(source_order),
                    "default_overall_score": default_scores.get("overall_score"),
                    "default_quality_score": default_scores.get("quality_score"),
                    "selected_overall_score": candidate_scores.get("overall_score"),
                    "selected_quality_score": candidate_scores.get("quality_score"),
                }
                if flip_axis:
                    candidate_summary["package_pad_flip"] = flip_axis
                if exclude_partial_leads:
                    candidate_summary["excluded_partial_lead_contacts"] = True
                if alignment_candidate_is_better(candidate_scores, candidate_objects, best_scores, best_objects, gt):
                    best_objects = candidate_objects
                    best_scores = candidate_scores
                    best_summary = candidate_summary

    if not should_try_package_pad_lead_bbox_alignment(default_scores, canonical):
        best_objects, best_scores, best_summary = prefer_excluding_thermal_land_pads(
            gt, canonical, checks, best_objects, best_scores, best_summary
        )
        best_objects, best_scores, best_summary = prefer_excluding_thermal_package_pads(
            gt, canonical, checks, best_objects, best_scores, best_summary
        )
        best_objects, best_scores, best_summary = prefer_limiting_duplicate_land_pads(
            gt, canonical, checks, best_objects, best_scores, best_summary
        )
        best_objects, best_scores, best_summary = prefer_land_pad_orientation(
            gt, canonical, checks, best_objects, best_scores, best_summary
        )
        best_objects, best_scores, best_summary = prefer_land_pads_as_package_pad_proxy(
            gt, canonical, checks, best_objects, best_scores, best_summary
        )
        best_objects, best_scores, best_summary = prefer_terminal_land_pads_as_package_pad_proxy(
            gt, canonical, checks, best_objects, best_scores, best_summary
        )
        best_objects, best_scores, best_summary = prefer_excluding_lateral_lead_contacts(
            gt, canonical, checks, best_objects, best_scores, best_summary
        )
        best_objects, best_scores, best_summary = prefer_land_pads_as_package_pad_proxy(
            gt, canonical, checks, best_objects, best_scores, best_summary
        )
        best_objects, best_scores, best_summary = prefer_terminal_land_pads_as_package_pad_proxy(
            gt, canonical, checks, best_objects, best_scores, best_summary
        )
        best_objects, best_scores, best_summary = prefer_excluding_unreliable_partial_leads(
            gt, canonical, checks, best_objects, best_scores, best_summary
        )
        return best_objects, annotate_scores_with_transform_flags(best_scores, best_summary), best_summary

    alternate_objects = aligned_canonical_objects(
        gt,
        canonical,
        package_pad_source_order=("lead", "package", "all"),
    )
    alternate_scores = alignment_scores(gt, canonical, alternate_objects, checks)
    alternate_summary = {
        "strategy": "package_pad_lead_bbox_for_low_score",
        "package_pad_source_order": ["lead", "package", "all"],
        "default_overall_score": default_scores.get("overall_score"),
        "default_quality_score": default_scores.get("quality_score"),
        "selected_overall_score": alternate_scores.get("overall_score"),
        "selected_quality_score": alternate_scores.get("quality_score"),
    }
    if alignment_score_tuple(alternate_scores) > alignment_score_tuple(best_scores):
        best_objects = alternate_objects
        best_scores = alternate_scores
        best_summary = alternate_summary
    best_objects, best_scores, best_summary = prefer_excluding_thermal_land_pads(
        gt, canonical, checks, best_objects, best_scores, best_summary
    )
    best_objects, best_scores, best_summary = prefer_excluding_thermal_package_pads(
        gt, canonical, checks, best_objects, best_scores, best_summary
    )
    best_objects, best_scores, best_summary = prefer_limiting_duplicate_land_pads(
        gt, canonical, checks, best_objects, best_scores, best_summary
    )
    best_objects, best_scores, best_summary = prefer_land_pad_orientation(
        gt, canonical, checks, best_objects, best_scores, best_summary
    )
    best_objects, best_scores, best_summary = prefer_land_pads_as_package_pad_proxy(
        gt, canonical, checks, best_objects, best_scores, best_summary
    )
    best_objects, best_scores, best_summary = prefer_terminal_land_pads_as_package_pad_proxy(
        gt, canonical, checks, best_objects, best_scores, best_summary
    )
    best_objects, best_scores, best_summary = prefer_excluding_lateral_lead_contacts(
        gt, canonical, checks, best_objects, best_scores, best_summary
    )
    best_objects, best_scores, best_summary = prefer_land_pads_as_package_pad_proxy(
        gt, canonical, checks, best_objects, best_scores, best_summary
    )
    best_objects, best_scores, best_summary = prefer_terminal_land_pads_as_package_pad_proxy(
        gt, canonical, checks, best_objects, best_scores, best_summary
    )
    best_objects, best_scores, best_summary = prefer_excluding_unreliable_partial_leads(
        gt, canonical, checks, best_objects, best_scores, best_summary
    )
    return best_objects, annotate_scores_with_transform_flags(best_scores, best_summary), best_summary


def alignment_score_tuple(alignment_scores_payload: dict[str, Any]) -> tuple[float, float]:
    quality = alignment_quality_score(alignment_scores_payload)
    overall = alignment_scores_payload.get("overall_score")
    return (
        float(quality) if isinstance(quality, (int, float)) else -1.0,
        float(overall) if isinstance(overall, (int, float)) else -1.0,
    )


def alignment_candidate_is_better(
    candidate_scores: dict[str, Any],
    candidate_objects: list[dict[str, Any]],
    best_scores: dict[str, Any],
    best_objects: list[dict[str, Any]],
    gt: dict[str, Any],
) -> bool:
    if alignment_score_tuple(candidate_scores) <= alignment_score_tuple(best_scores):
        return False
    candidate_pad_iou = candidate_scores.get("lead_pad_iou_score")
    best_pad_iou = best_scores.get("lead_pad_iou_score")
    if (
        isinstance(candidate_pad_iou, (int, float))
        and float(candidate_pad_iou) >= 0.8
        and (
            not isinstance(best_pad_iou, (int, float))
            or float(best_pad_iou) < 0.5
        )
    ):
        return True
    candidate_mismatch = package_pad_lattice_mismatch_score(gt, candidate_objects)
    best_mismatch = package_pad_lattice_mismatch_score(gt, best_objects)
    if candidate_mismatch is None or best_mismatch is None:
        return True
    return candidate_mismatch <= best_mismatch


def package_pad_lattice_mismatch_score(gt: dict[str, Any], objects: list[dict[str, Any]]) -> int | None:
    gt_boxes = [list(obj.get("bbox") or []) for obj in scan_result_objects(gt) if str(obj.get("role") or "") == "lead"]
    result_boxes = [list(obj.get("bbox") or []) for obj in objects if str(obj.get("role") or "") == "package_pad"]
    if len(gt_boxes) != len(result_boxes) or len(gt_boxes) < 8:
        return None
    gt_signature = lattice_signature(gt_boxes)
    result_signature = lattice_signature(result_boxes)
    if gt_signature is None or result_signature is None:
        return None
    gt_rows, gt_cols = gt_signature
    result_rows, result_cols = result_signature
    return count_pattern_distance(gt_rows, result_rows) + count_pattern_distance(gt_cols, result_cols)


def lattice_signature(bboxes: list[list[float]]) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    clean_boxes = [bbox for bbox in bboxes if len(bbox) >= 4]
    if len(clean_boxes) < 8:
        return None
    sizes = sorted(min(abs(float(bbox[2]) - float(bbox[0])), abs(float(bbox[3]) - float(bbox[1]))) for bbox in clean_boxes)
    median_size = sizes[len(sizes) // 2]
    tolerance = max(median_size * 0.7, 1e-9)
    centers = [((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0) for bbox in clean_boxes]
    return (
        tuple(sorted(group_counts_by_axis(centers, axis=1, tolerance=tolerance))),
        tuple(sorted(group_counts_by_axis(centers, axis=0, tolerance=tolerance))),
    )


def group_counts_by_axis(centers: list[tuple[float, float]], *, axis: int, tolerance: float) -> list[int]:
    groups: list[tuple[float, int]] = []
    for center in sorted(centers, key=lambda item: item[axis]):
        value = center[axis]
        if not groups or abs(groups[-1][0] - value) > tolerance:
            groups.append((value, 1))
            continue
        old_value, old_count = groups[-1]
        new_count = old_count + 1
        groups[-1] = ((old_value * old_count + value) / new_count, new_count)
    return [count for _, count in groups]


def count_pattern_distance(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    distance = abs(len(left) - len(right))
    for lhs, rhs in zip(left, right):
        distance += abs(lhs - rhs)
    if len(left) > len(right):
        distance += sum(left[len(right) :])
    elif len(right) > len(left):
        distance += sum(right[len(left) :])
    return distance


def annotate_scores_with_transform_flags(scores: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    annotated = dict(scores)
    if summary.get("package_pad_from_land_proxy"):
        annotated["package_pad_from_land_proxy"] = True
    review_quality = conservative_transform_quality_score(scores, summary)
    selected_quality = alignment_quality_score(scores)
    if isinstance(review_quality, (int, float)):
        annotated["review_quality_score"] = round(float(review_quality), 6)
    if isinstance(selected_quality, (int, float)):
        annotated["selected_quality_score"] = round(float(selected_quality), 6)
    return annotated


def conservative_transform_quality_score(scores: dict[str, Any], summary: dict[str, Any]) -> float | None:
    # Review risk is for the selected final graph, not for intermediate
    # candidates that were intentionally rejected. Pre/default quality remains
    # available in alignment_transform metadata for audit.
    return alignment_quality_score(scores)


def prefer_excluding_thermal_land_pads(
    gt: dict[str, Any],
    canonical: dict[str, Any],
    checks: list[dict[str, Any]],
    objects: list[dict[str, Any]],
    scores: dict[str, Any],
    summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if summary.get("excluded_thermal_land_pads"):
        return objects, scores, summary
    if not should_exclude_thermal_land_pads(checks, canonical):
        return objects, scores, summary
    candidate_objects = aligned_canonical_objects(
        gt,
        canonical,
        package_pad_source_order=tuple(summary.get("package_pad_source_order") or ["package", "lead", "all"]),
        package_pad_label_filter=summary.get("package_pad_label_filter"),
        package_pad_flip=str(summary.get("package_pad_flip") or ""),
        package_pad_rotation=int(summary.get("package_pad_rotation") or 0),
        exclude_lead_contacts=bool(summary.get("excluded_partial_lead_contacts")),
        exclude_lateral_lead_contacts=bool(summary.get("excluded_lateral_lead_contacts")),
        exclude_thermal_land_pads=True,
        exclude_thermal_package_pads=bool(summary.get("excluded_thermal_package_pads")),
        land_pad_limit=int(summary.get("land_pad_limit") or 0),
        package_pad_from_land_proxy=bool(summary.get("package_pad_from_land_proxy")),
        package_pad_proxy_size_source=str(summary.get("package_pad_proxy_size_source") or ""),
        package_pad_proxy_land_filter=str(summary.get("package_pad_proxy_land_filter") or ""),
    )
    candidate_scores = alignment_scores(gt, canonical, candidate_objects, checks)
    current_land_score = scores.get("land_pad_iou_score")
    candidate_land_score = candidate_scores.get("land_pad_iou_score")
    land_improved = (
        isinstance(candidate_land_score, (int, float))
        and (
            not isinstance(current_land_score, (int, float))
            or float(candidate_land_score) > float(current_land_score)
        )
    )
    if not land_improved and alignment_score_tuple(candidate_scores) <= alignment_score_tuple(scores):
        return objects, scores, summary
    candidate_summary = dict(summary)
    candidate_summary["strategy"] = f"{summary.get('strategy', 'alignment')}_excluding_thermal_land_pads"
    candidate_summary["excluded_thermal_land_pads"] = True
    candidate_summary["thermal_land_exclusion_reason"] = "land_count_uses_terminal_land_pad_count"
    candidate_summary["pre_land_exclusion_overall_score"] = scores.get("overall_score")
    candidate_summary["pre_land_exclusion_quality_score"] = scores.get("quality_score")
    candidate_summary["selected_overall_score"] = candidate_scores.get("overall_score")
    candidate_summary["selected_quality_score"] = candidate_scores.get("quality_score")
    return candidate_objects, candidate_scores, candidate_summary


def prefer_excluding_thermal_package_pads(
    gt: dict[str, Any],
    canonical: dict[str, Any],
    checks: list[dict[str, Any]],
    objects: list[dict[str, Any]],
    scores: dict[str, Any],
    summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if summary.get("excluded_thermal_package_pads"):
        return objects, scores, summary
    if not should_exclude_thermal_package_pads(checks, canonical):
        return objects, scores, summary
    candidate_objects = aligned_canonical_objects(
        gt,
        canonical,
        package_pad_source_order=tuple(summary.get("package_pad_source_order") or ["package", "lead", "all"]),
        package_pad_label_filter=summary.get("package_pad_label_filter"),
        package_pad_flip=str(summary.get("package_pad_flip") or ""),
        package_pad_rotation=int(summary.get("package_pad_rotation") or 0),
        exclude_lead_contacts=bool(summary.get("excluded_partial_lead_contacts")),
        exclude_lateral_lead_contacts=bool(summary.get("excluded_lateral_lead_contacts")),
        exclude_thermal_land_pads=bool(summary.get("excluded_thermal_land_pads")),
        exclude_thermal_package_pads=True,
        land_pad_limit=int(summary.get("land_pad_limit") or 0),
        package_pad_from_land_proxy=bool(summary.get("package_pad_from_land_proxy")),
        package_pad_proxy_size_source=str(summary.get("package_pad_proxy_size_source") or ""),
        package_pad_proxy_land_filter=str(summary.get("package_pad_proxy_land_filter") or ""),
    )
    candidate_scores = alignment_scores(gt, canonical, candidate_objects, checks)
    if alignment_score_tuple(candidate_scores) <= alignment_score_tuple(scores):
        return objects, scores, summary
    candidate_summary = dict(summary)
    candidate_summary["strategy"] = f"{summary.get('strategy', 'alignment')}_excluding_thermal_package_pads"
    candidate_summary["excluded_thermal_package_pads"] = True
    candidate_summary["thermal_package_exclusion_reason"] = "lead_count_uses_terminal_package_pad_count"
    candidate_summary["pre_package_exclusion_overall_score"] = scores.get("overall_score")
    candidate_summary["pre_package_exclusion_quality_score"] = scores.get("quality_score")
    candidate_summary["selected_overall_score"] = candidate_scores.get("overall_score")
    candidate_summary["selected_quality_score"] = candidate_scores.get("quality_score")
    return candidate_objects, candidate_scores, candidate_summary


def prefer_land_pads_as_package_pad_proxy(
    gt: dict[str, Any],
    canonical: dict[str, Any],
    checks: list[dict[str, Any]],
    objects: list[dict[str, Any]],
    scores: dict[str, Any],
    summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if summary.get("package_pad_from_land_proxy"):
        return objects, scores, summary
    if not should_try_land_pads_as_package_pad_proxy(scores):
        return objects, scores, summary
    source_order = tuple(summary.get("package_pad_source_order") or ["package", "lead", "all"])
    best_candidate_objects = None
    best_candidate_scores = None
    best_size_source = ""
    for size_source in ("", "package_pad_median"):
        candidate_objects = aligned_canonical_objects(
            gt,
            canonical,
            package_pad_source_order=source_order,
            package_pad_label_filter=summary.get("package_pad_label_filter"),
            package_pad_flip=str(summary.get("package_pad_flip") or ""),
            package_pad_rotation=int(summary.get("package_pad_rotation") or 0),
            land_pad_flip=str(summary.get("land_pad_flip") or ""),
            land_pad_rotation=int(summary.get("land_pad_rotation") or 0),
            exclude_lead_contacts=bool(summary.get("excluded_partial_lead_contacts")),
            exclude_lateral_lead_contacts=bool(summary.get("excluded_lateral_lead_contacts")),
            exclude_thermal_land_pads=bool(summary.get("excluded_thermal_land_pads")),
            exclude_thermal_package_pads=bool(summary.get("excluded_thermal_package_pads")),
            land_pad_limit=int(summary.get("land_pad_limit") or 0),
            package_pad_from_land_proxy=True,
            package_pad_proxy_size_source=size_source,
        )
        candidate_scores = alignment_scores(gt, canonical, candidate_objects, checks)
        if best_candidate_scores is None or alignment_score_tuple(candidate_scores) > alignment_score_tuple(best_candidate_scores):
            best_candidate_objects = candidate_objects
            best_candidate_scores = candidate_scores
            best_size_source = size_source
    if best_candidate_objects is None or best_candidate_scores is None:
        return objects, scores, summary
    if alignment_score_tuple(best_candidate_scores) <= alignment_score_tuple(scores):
        return objects, scores, summary
    candidate_summary = dict(summary)
    candidate_summary["strategy"] = f"{summary.get('strategy', 'alignment')}_using_land_pads_as_package_pad_proxy"
    candidate_summary["package_pad_from_land_proxy"] = True
    candidate_summary["package_pad_proxy_reason"] = "land_and_lead_counts_match_land_layout_outscores_package_layout"
    if best_size_source:
        candidate_summary["package_pad_proxy_size_source"] = best_size_source
    candidate_summary["pre_package_pad_proxy_overall_score"] = scores.get("overall_score")
    candidate_summary["pre_package_pad_proxy_quality_score"] = scores.get("quality_score")
    candidate_summary["overall_score"] = best_candidate_scores.get("overall_score")
    candidate_summary["quality_score"] = best_candidate_scores.get("quality_score")
    candidate_summary["selected_overall_score"] = best_candidate_scores.get("overall_score")
    candidate_summary["selected_quality_score"] = best_candidate_scores.get("quality_score")
    return best_candidate_objects, best_candidate_scores, candidate_summary


def prefer_terminal_land_pads_as_package_pad_proxy(
    gt: dict[str, Any],
    canonical: dict[str, Any],
    checks: list[dict[str, Any]],
    objects: list[dict[str, Any]],
    scores: dict[str, Any],
    summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if summary.get("package_pad_from_land_proxy"):
        return objects, scores, summary
    if not should_try_terminal_land_pads_as_package_pad_proxy(scores, canonical):
        return objects, scores, summary
    source_order = tuple(summary.get("package_pad_source_order") or ["package", "lead", "all"])
    best_candidate_objects = None
    best_candidate_scores = None
    best_size_source = ""
    for size_source in ("", "package_pad_median"):
        candidate_objects = aligned_canonical_objects(
            gt,
            canonical,
            package_pad_source_order=source_order,
            package_pad_label_filter=summary.get("package_pad_label_filter"),
            package_pad_flip=str(summary.get("package_pad_flip") or ""),
            package_pad_rotation=int(summary.get("package_pad_rotation") or 0),
            land_pad_flip=str(summary.get("land_pad_flip") or ""),
            land_pad_rotation=int(summary.get("land_pad_rotation") or 0),
            exclude_lead_contacts=bool(summary.get("excluded_partial_lead_contacts")),
            exclude_lateral_lead_contacts=bool(summary.get("excluded_lateral_lead_contacts")),
            exclude_thermal_land_pads=bool(summary.get("excluded_thermal_land_pads")),
            exclude_thermal_package_pads=bool(summary.get("excluded_thermal_package_pads")),
            land_pad_limit=int(summary.get("land_pad_limit") or 0),
            package_pad_from_land_proxy=True,
            package_pad_proxy_size_source=size_source,
            package_pad_proxy_land_filter="terminal_rect",
        )
        candidate_scores = alignment_scores(gt, canonical, candidate_objects, checks)
        if best_candidate_scores is None or alignment_score_tuple(candidate_scores) > alignment_score_tuple(best_candidate_scores):
            best_candidate_objects = candidate_objects
            best_candidate_scores = candidate_scores
            best_size_source = size_source
    if best_candidate_objects is None or best_candidate_scores is None:
        return objects, scores, summary
    if alignment_score_tuple(best_candidate_scores) <= alignment_score_tuple(scores):
        return objects, scores, summary
    candidate_summary = dict(summary)
    candidate_summary["strategy"] = f"{summary.get('strategy', 'alignment')}_using_terminal_land_pads_as_package_pad_proxy"
    candidate_summary["package_pad_from_land_proxy"] = True
    candidate_summary["package_pad_proxy_land_filter"] = "terminal_rect"
    candidate_summary["package_pad_proxy_reason"] = "terminal_land_rect_count_matches_scan_lead_count_and_outscores_package_layout"
    if best_size_source:
        candidate_summary["package_pad_proxy_size_source"] = best_size_source
    candidate_summary["pre_package_pad_proxy_overall_score"] = scores.get("overall_score")
    candidate_summary["pre_package_pad_proxy_quality_score"] = scores.get("quality_score")
    candidate_summary["overall_score"] = best_candidate_scores.get("overall_score")
    candidate_summary["quality_score"] = best_candidate_scores.get("quality_score")
    candidate_summary["selected_overall_score"] = best_candidate_scores.get("overall_score")
    candidate_summary["selected_quality_score"] = best_candidate_scores.get("quality_score")
    return best_candidate_objects, best_candidate_scores, candidate_summary


def prefer_land_pad_orientation(
    gt: dict[str, Any],
    canonical: dict[str, Any],
    checks: list[dict[str, Any]],
    objects: list[dict[str, Any]],
    scores: dict[str, Any],
    summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if summary.get("land_pad_rotation") or summary.get("land_pad_flip"):
        return objects, scores, summary
    if not should_try_land_pad_orientation(scores, canonical):
        return objects, scores, summary
    best_objects = objects
    best_scores = scores
    best_summary = summary
    source_order = tuple(summary.get("package_pad_source_order") or ["package", "lead", "all"])
    rotations = [0, *land_pad_rotation_candidates(scores, canonical)]
    flips = ["", *land_pad_flip_candidates(scores, canonical)]
    for rotation in rotations:
        for flip_axis in flips:
            if rotation == 0 and not flip_axis:
                continue
            candidate_objects = aligned_canonical_objects(
                gt,
                canonical,
                package_pad_source_order=source_order,
                package_pad_label_filter=summary.get("package_pad_label_filter"),
                package_pad_flip=str(summary.get("package_pad_flip") or ""),
                package_pad_rotation=int(summary.get("package_pad_rotation") or 0),
                land_pad_flip=flip_axis,
                land_pad_rotation=rotation,
                exclude_lead_contacts=bool(summary.get("excluded_partial_lead_contacts")),
                exclude_lateral_lead_contacts=bool(summary.get("excluded_lateral_lead_contacts")),
                exclude_thermal_land_pads=bool(summary.get("excluded_thermal_land_pads")),
                exclude_thermal_package_pads=bool(summary.get("excluded_thermal_package_pads")),
                land_pad_limit=int(summary.get("land_pad_limit") or 0),
                package_pad_from_land_proxy=bool(summary.get("package_pad_from_land_proxy")),
                package_pad_proxy_size_source=str(summary.get("package_pad_proxy_size_source") or ""),
                package_pad_proxy_land_filter=str(summary.get("package_pad_proxy_land_filter") or ""),
            )
            candidate_scores = alignment_scores(gt, canonical, candidate_objects, checks)
            if alignment_score_tuple(candidate_scores) <= alignment_score_tuple(best_scores):
                continue
            candidate_summary = dict(summary)
            candidate_summary["strategy"] = f"{summary.get('strategy', 'alignment')}_orienting_land_pads"
            if rotation:
                candidate_summary["land_pad_rotation"] = rotation
            if flip_axis:
                candidate_summary["land_pad_flip"] = flip_axis
            candidate_summary["land_pad_orientation_reason"] = "land_view_orientation_differs_from_scan_result_reference"
            candidate_summary["pre_land_orientation_overall_score"] = scores.get("overall_score")
            candidate_summary["pre_land_orientation_quality_score"] = scores.get("quality_score")
            candidate_summary["selected_overall_score"] = candidate_scores.get("overall_score")
            candidate_summary["selected_quality_score"] = candidate_scores.get("quality_score")
            best_objects = candidate_objects
            best_scores = candidate_scores
            best_summary = candidate_summary
    return best_objects, best_scores, best_summary


def should_try_land_pad_orientation(scores: dict[str, Any], canonical: dict[str, Any]) -> bool:
    land_pad_count = int(scores.get("land_pad_count") or 0)
    gt_land_count = int(scores.get("gt_land_count") or 0)
    if land_pad_count <= 1 or gt_land_count <= 1:
        return False
    if not canonical.get("land_pads"):
        return False
    source_selection = (canonical.get("summary") or {}).get("source_selection") or {}
    land_source = source_selection.get("land_pads") or {}
    if land_source.get("source_type") == "scan_result_format":
        return False
    land_score = scores.get("land_pad_iou_score")
    return isinstance(land_score, (int, float)) and float(land_score) < 0.8


def land_pad_flip_candidates(alignment_scores: dict[str, Any], canonical: dict[str, Any]) -> list[str]:
    if not should_try_land_pad_orientation(alignment_scores, canonical):
        return []
    return ["flip_y", "flip_x", "flip_xy"]


def land_pad_rotation_candidates(alignment_scores: dict[str, Any], canonical: dict[str, Any]) -> list[int]:
    if not should_try_land_pad_orientation(alignment_scores, canonical):
        return []
    return [90, 180, 270]


def should_try_land_pads_as_package_pad_proxy(scores: dict[str, Any]) -> bool:
    gt_land_count = int(scores.get("gt_land_count") or 0)
    gt_lead_count = int(scores.get("gt_lead_count") or 0)
    package_pad_count = int(scores.get("package_pad_count") or 0)
    land_pad_count = int(scores.get("land_pad_count") or 0)
    package_score = scores.get("lead_pad_iou_package_only")
    land_score = scores.get("land_pad_iou_score")
    if gt_land_count <= 0 or gt_land_count != gt_lead_count:
        return False
    if land_pad_count != gt_lead_count:
        return False
    if not isinstance(package_score, (int, float)) or not isinstance(land_score, (int, float)):
        return False
    if package_pad_count != gt_lead_count:
        return float(package_score) < 0.5
    return float(package_score) < 0.5 and float(land_score) >= 0.6


def should_try_terminal_land_pads_as_package_pad_proxy(scores: dict[str, Any], canonical: dict[str, Any]) -> bool:
    gt_lead_count = int(scores.get("gt_lead_count") or 0)
    gt_land_count = int(scores.get("gt_land_count") or 0)
    land_pad_count = int(scores.get("land_pad_count") or 0)
    package_score = scores.get("lead_pad_iou_package_only")
    if gt_lead_count <= 0 or gt_land_count <= gt_lead_count:
        return False
    if land_pad_count < gt_lead_count:
        return False
    if not isinstance(package_score, (int, float)) or float(package_score) >= 0.5:
        return False
    return terminal_land_rect_proxy_count(canonical) == gt_lead_count


def terminal_land_rect_proxy_count(canonical: dict[str, Any]) -> int:
    objects = canonical_source_objects(canonical)
    land_objects = [obj for obj in objects if obj.get("role") == "land" and obj.get("bbox")]
    thermal_keys = thermal_like_bbox_keys([obj["bbox"] for obj in land_objects])
    return sum(1 for obj in land_objects if should_proxy_terminal_land_rect(obj, thermal_keys))


def should_proxy_terminal_land_rect(
    obj: dict[str, Any],
    thermal_keys: set[tuple[int, int, int, int]],
) -> bool:
    if str(obj.get("role") or "") != "land":
        return False
    if str(obj.get("source_label") or "") != "pad":
        return False
    key = bbox_dedupe_key(obj.get("bbox"), 0.001)
    return key is not None and key not in thermal_keys


def prefer_limiting_duplicate_land_pads(
    gt: dict[str, Any],
    canonical: dict[str, Any],
    checks: list[dict[str, Any]],
    objects: list[dict[str, Any]],
    scores: dict[str, Any],
    summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if summary.get("land_pad_limit"):
        return objects, scores, summary
    land_limit = duplicate_land_pad_limit(gt, canonical, scores, checks)
    if land_limit <= 0:
        return objects, scores, summary
    source_order = tuple(summary.get("package_pad_source_order") or ["package", "lead", "all"])
    candidate_objects = aligned_canonical_objects(
        gt,
        canonical,
        package_pad_source_order=source_order,
        package_pad_label_filter=summary.get("package_pad_label_filter"),
        package_pad_flip=str(summary.get("package_pad_flip") or ""),
        package_pad_rotation=int(summary.get("package_pad_rotation") or 0),
        exclude_lead_contacts=bool(summary.get("excluded_partial_lead_contacts")),
        exclude_lateral_lead_contacts=bool(summary.get("excluded_lateral_lead_contacts")),
        exclude_thermal_land_pads=bool(summary.get("excluded_thermal_land_pads")),
        exclude_thermal_package_pads=bool(summary.get("excluded_thermal_package_pads")),
        land_pad_limit=land_limit,
        package_pad_from_land_proxy=bool(summary.get("package_pad_from_land_proxy")),
        package_pad_proxy_size_source=str(summary.get("package_pad_proxy_size_source") or ""),
        package_pad_proxy_land_filter=str(summary.get("package_pad_proxy_land_filter") or ""),
    )
    candidate_scores = alignment_scores(gt, canonical, candidate_objects, checks)
    current_land_score = scores.get("land_pad_iou_score")
    candidate_land_score = candidate_scores.get("land_pad_iou_score")
    land_improved = (
        isinstance(candidate_land_score, (int, float))
        and (
            not isinstance(current_land_score, (int, float))
            or float(candidate_land_score) > float(current_land_score)
        )
    )
    if not land_improved and alignment_score_tuple(candidate_scores) <= alignment_score_tuple(scores):
        return objects, scores, summary
    candidate_summary = dict(summary)
    candidate_summary["strategy"] = f"{summary.get('strategy', 'alignment')}_limiting_duplicate_land_pads"
    candidate_summary["land_pad_limit"] = land_limit
    candidate_summary["land_pad_limit_reason"] = "scan_result_direct_land_count_with_low_land_iou"
    candidate_summary["pre_land_limit_overall_score"] = scores.get("overall_score")
    candidate_summary["pre_land_limit_quality_score"] = scores.get("quality_score")
    candidate_summary["selected_overall_score"] = candidate_scores.get("overall_score")
    candidate_summary["selected_quality_score"] = candidate_scores.get("quality_score")
    return candidate_objects, candidate_scores, candidate_summary


def duplicate_land_pad_limit(
    gt: dict[str, Any],
    canonical: dict[str, Any],
    scores: dict[str, Any],
    checks: list[dict[str, Any]],
) -> int:
    land_check = next((check for check in checks if check.get("name") == "land_count"), {})
    if land_check.get("status") != "aligned":
        return 0
    direct_land_count = gt_direct_role_count(gt, "land")
    if direct_land_count <= 0:
        return 0
    land_pads = canonical.get("land_pads") or []
    if len(land_pads) <= direct_land_count:
        return 0
    land_score = scores.get("land_pad_iou_score")
    if isinstance(land_score, (int, float)) and float(land_score) >= 0.8:
        return 0
    return direct_land_count


def gt_direct_role_count(gt: dict[str, Any], role: str) -> int:
    features = gt.get("features") or {}
    role_counts = features.get("role_counts") or {}
    return int(role_counts.get(role) or 0)


def prefer_excluding_lateral_lead_contacts(
    gt: dict[str, Any],
    canonical: dict[str, Any],
    checks: list[dict[str, Any]],
    objects: list[dict[str, Any]],
    scores: dict[str, Any],
    summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if summary.get("excluded_lateral_lead_contacts"):
        return objects, scores, summary
    if not should_try_excluding_lateral_lead_contacts(canonical):
        return objects, scores, summary
    source_order = tuple(summary.get("package_pad_source_order") or ["package", "lead", "all"])
    candidate_objects = aligned_canonical_objects(
        gt,
        canonical,
        package_pad_source_order=source_order,
        package_pad_label_filter=summary.get("package_pad_label_filter"),
        package_pad_flip=str(summary.get("package_pad_flip") or ""),
        package_pad_rotation=int(summary.get("package_pad_rotation") or 0),
        exclude_lead_contacts=bool(summary.get("excluded_partial_lead_contacts")),
        exclude_lateral_lead_contacts=True,
        exclude_thermal_land_pads=bool(summary.get("excluded_thermal_land_pads")),
        exclude_thermal_package_pads=bool(summary.get("excluded_thermal_package_pads")),
        land_pad_limit=int(summary.get("land_pad_limit") or 0),
        package_pad_from_land_proxy=bool(summary.get("package_pad_from_land_proxy")),
        package_pad_proxy_size_source=str(summary.get("package_pad_proxy_size_source") or ""),
        package_pad_proxy_land_filter=str(summary.get("package_pad_proxy_land_filter") or ""),
    )
    candidate_scores = alignment_scores(gt, canonical, candidate_objects, checks)
    if alignment_score_tuple(candidate_scores) <= alignment_score_tuple(scores):
        return objects, scores, summary
    candidate_summary = dict(summary)
    candidate_summary["strategy"] = f"{summary.get('strategy', 'alignment')}_excluding_lateral_lead_contacts"
    candidate_summary["excluded_lateral_lead_contacts"] = True
    candidate_summary["lateral_lead_exclusion_reason"] = "lateral_projection_not_drawn_as_2d_terminal"
    candidate_summary["pre_lateral_lead_exclusion_overall_score"] = scores.get("overall_score")
    candidate_summary["pre_lateral_lead_exclusion_quality_score"] = scores.get("quality_score")
    candidate_summary["overall_score"] = candidate_scores.get("overall_score")
    candidate_summary["quality_score"] = candidate_scores.get("quality_score")
    candidate_summary["selected_overall_score"] = candidate_scores.get("overall_score")
    candidate_summary["selected_quality_score"] = candidate_scores.get("quality_score")
    return candidate_objects, candidate_scores, candidate_summary


def should_try_excluding_lateral_lead_contacts(canonical: dict[str, Any]) -> bool:
    if not canonical.get("package_pads"):
        return False
    return any(
        str(obj.get("canonical_view") or "") == "lateral"
        for obj in canonical.get("lead_contacts") or []
    )


def should_exclude_thermal_land_pads(checks: list[dict[str, Any]], canonical: dict[str, Any]) -> bool:
    thermal_land_count = int((canonical_features(canonical).get("summary") or {}).get("thermal_land_pad_count") or 0)
    if thermal_land_count <= 0:
        return False
    for check in checks:
        if check.get("name") != "land_count" or check.get("status") != "aligned":
            continue
        if check.get("actual_role") == "terminal_land_pad_count":
            return True
        if str(check.get("selected_mapping") or "").startswith("scan_land_to_terminal_land"):
            return True
        if str(check.get("selected_mapping") or "").startswith("scan_group_land_count_to_terminal_land"):
            return True
    return False


def should_exclude_thermal_package_pads(checks: list[dict[str, Any]], canonical: dict[str, Any]) -> bool:
    summary = canonical_features(canonical).get("summary") or {}
    if int(summary.get("thermal_package_pad_count") or 0) <= 0:
        return False
    for check in checks:
        if check.get("name") != "lead_count" or check.get("status") != "aligned":
            continue
        if check.get("actual_role") in {"terminal_package_pad_count", "lead_equivalent_count"}:
            return True
        selected_mapping = str(check.get("selected_mapping") or "")
        if selected_mapping.startswith("scan_lead_to_terminal_package_pad"):
            return True
        if selected_mapping.startswith("scan_group_lead_count_to_terminal_package"):
            return True
    return False


def prefer_excluding_unreliable_partial_leads(
    gt: dict[str, Any],
    canonical: dict[str, Any],
    checks: list[dict[str, Any]],
    objects: list[dict[str, Any]],
    scores: dict[str, Any],
    summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if not has_partial_lead_detail_layout(gt, canonical):
        return objects, scores, summary
    if summary.get("excluded_partial_lead_contacts"):
        return objects, scores, summary
    lead_score = scores.get("lead_pad_iou_score")
    lead_contact_score = scores.get("lead_pad_iou_lead_contact_only")
    lead_score_is_low = isinstance(lead_score, (int, float)) and float(lead_score) < 0.5
    lead_contact_score_is_low = isinstance(lead_contact_score, (int, float)) and float(lead_contact_score) < 0.5
    if not lead_score_is_low and not lead_contact_score_is_low:
        return objects, scores, summary
    source_order = tuple(summary.get("package_pad_source_order") or ["package", "lead", "all"])
    candidate_objects = aligned_canonical_objects(
        gt,
        canonical,
        package_pad_source_order=source_order,
        package_pad_label_filter=summary.get("package_pad_label_filter"),
        package_pad_flip=str(summary.get("package_pad_flip") or ""),
        package_pad_rotation=int(summary.get("package_pad_rotation") or 0),
        exclude_lead_contacts=True,
        exclude_lateral_lead_contacts=bool(summary.get("excluded_lateral_lead_contacts")),
        exclude_thermal_land_pads=bool(summary.get("excluded_thermal_land_pads")),
        exclude_thermal_package_pads=bool(summary.get("excluded_thermal_package_pads")),
        land_pad_limit=int(summary.get("land_pad_limit") or 0),
        package_pad_from_land_proxy=bool(summary.get("package_pad_from_land_proxy")),
        package_pad_proxy_size_source=str(summary.get("package_pad_proxy_size_source") or ""),
        package_pad_proxy_land_filter=str(summary.get("package_pad_proxy_land_filter") or ""),
    )
    candidate_scores = alignment_scores(gt, canonical, candidate_objects, checks)
    candidate_summary = dict(summary)
    candidate_summary["strategy"] = f"{summary.get('strategy', 'alignment')}_excluding_unreliable_partial_lead_detail"
    candidate_summary["excluded_partial_lead_contacts"] = True
    candidate_summary["partial_lead_exclusion_reason"] = (
        "lead_contact_score_below_0.5" if lead_contact_score_is_low else "lead_score_below_0.5"
    )
    candidate_summary["pre_exclusion_overall_score"] = scores.get("overall_score")
    candidate_summary["pre_exclusion_quality_score"] = scores.get("quality_score")
    candidate_summary["overall_score"] = candidate_scores.get("overall_score")
    candidate_summary["quality_score"] = candidate_scores.get("quality_score")
    candidate_summary["selected_overall_score"] = candidate_scores.get("overall_score")
    candidate_summary["selected_quality_score"] = candidate_scores.get("quality_score")
    return candidate_objects, candidate_scores, candidate_summary


def package_pad_terminal_label_candidates(gt: dict[str, Any], canonical: dict[str, Any]) -> list[str]:
    features = gt.get("features") or {}
    lead_count = int((features.get("role_counts") or {}).get("lead") or 0)
    if lead_count <= 0:
        lead_count = sum(1 for obj in scan_result_objects(gt) if str(obj.get("role") or "") == "lead")
    if lead_count <= 0:
        return []
    counts: Counter[str] = Counter()
    for obj in canonical_source_objects(canonical):
        if str(obj.get("role") or "") != "package_pad":
            continue
        label = package_pad_label_key(obj)
        if label:
            counts[label] += 1
    priority = ("dshape", "rect", "circle", "pad")
    return [label for label in priority if counts.get(label) == lead_count]


def package_pad_flip_candidates(alignment_scores: dict[str, Any], canonical: dict[str, Any]) -> list[str]:
    lead_pad_score = alignment_scores.get("lead_pad_iou_score")
    if not isinstance(lead_pad_score, (int, float)) or float(lead_pad_score) >= 0.5:
        return []
    if not canonical.get("package_pads"):
        return []
    source_selection = (canonical.get("summary") or {}).get("source_selection") or {}
    package_source = source_selection.get("package_pads") or {}
    if package_source.get("source_type") == "scan_result_format":
        return []
    return ["flip_y", "flip_x", "flip_xy"]


def package_pad_rotation_candidates(alignment_scores: dict[str, Any], canonical: dict[str, Any]) -> list[int]:
    lead_pad_score = alignment_scores.get("lead_pad_iou_score")
    if not isinstance(lead_pad_score, (int, float)) or float(lead_pad_score) >= 0.5:
        return []
    if not canonical.get("package_pads"):
        return []
    source_selection = (canonical.get("summary") or {}).get("source_selection") or {}
    package_source = source_selection.get("package_pads") or {}
    if package_source.get("source_type") == "scan_result_format":
        return []
    return [90, 180, 270]


def package_pad_bbox_source_order_candidates(
    alignment_scores: dict[str, Any],
    canonical: dict[str, Any],
) -> list[tuple[str, ...]]:
    lead_pad_score = alignment_scores.get("lead_pad_iou_score")
    if not isinstance(lead_pad_score, (int, float)) or float(lead_pad_score) >= 0.5:
        return []
    if not canonical.get("package_pads"):
        return []
    source_selection = (canonical.get("summary") or {}).get("source_selection") or {}
    package_source = source_selection.get("package_pads") or {}
    if package_source.get("source_type") == "scan_result_format":
        return []
    features = canonical_features(canonical)
    candidates = features.get("bbox_candidates") or {}
    has_circle_array = bool(candidates.get("package_circle"))
    if not has_circle_array:
        return []
    return [("package_circle", "conductive", "package", "lead", "all")]


def partial_lead_exclusion_options(gt: dict[str, Any], canonical: dict[str, Any]) -> list[bool]:
    if has_partial_lead_detail_layout(gt, canonical):
        return [False, True]
    return [False]


def has_partial_lead_detail_layout(gt: dict[str, Any], canonical: dict[str, Any]) -> bool:
    features = gt.get("features") or {}
    gt_lead_count = int((features.get("role_counts") or {}).get("lead") or 0)
    if gt_lead_count <= 0:
        gt_lead_count = sum(1 for obj in scan_result_objects(gt) if str(obj.get("role") or "") == "lead")
    lead_contact_count = len(canonical.get("lead_contacts") or [])
    return gt_lead_count > 0 and 0 < lead_contact_count < gt_lead_count


def should_try_package_pad_lead_bbox_alignment(alignment_scores: dict[str, Any], canonical: dict[str, Any]) -> bool:
    overall_score = alignment_scores.get("overall_score")
    if not isinstance(overall_score, (int, float)) or float(overall_score) >= 0.8:
        return False
    lead_pad_score = alignment_scores.get("lead_pad_iou_score")
    if not isinstance(lead_pad_score, (int, float)) or float(lead_pad_score) >= 0.5:
        return False
    source_selection = (canonical.get("summary") or {}).get("source_selection") or {}
    package_source = source_selection.get("package_pads") or {}
    return package_source.get("source_type") != "scan_result_format"


def aligned_canonical_objects(
    gt: dict[str, Any],
    canonical: dict[str, Any],
    *,
    package_pad_source_order: tuple[str, ...] = ("package", "lead", "all"),
    package_pad_label_filter: str | None = None,
    package_pad_flip: str = "",
    package_pad_rotation: int = 0,
    land_pad_flip: str = "",
    land_pad_rotation: int = 0,
    exclude_lead_contacts: bool = False,
    exclude_lateral_lead_contacts: bool = False,
    exclude_thermal_land_pads: bool = False,
    exclude_thermal_package_pads: bool = False,
    land_pad_limit: int = 0,
    package_pad_from_land_proxy: bool = False,
    package_pad_proxy_size_source: str = "",
    package_pad_proxy_land_filter: str = "",
) -> list[dict[str, Any]]:
    if not canonical:
        return []
    features = canonical_features(canonical)
    gt_candidates = (gt.get("features") or {}).get("bbox_candidates") or {}
    graph_candidates = features.get("bbox_candidates") or {}
    objects = canonical_source_objects(canonical)
    thermal_land_keys = set()
    if exclude_thermal_land_pads:
        thermal_land_keys = thermal_like_bbox_keys([obj["bbox"] for obj in objects if obj.get("role") == "land" and obj.get("bbox")])
    thermal_package_keys = set()
    if exclude_thermal_package_pads:
        thermal_package_keys = thermal_like_bbox_keys(
            [obj["bbox"] for obj in objects if obj.get("role") == "package_pad" and obj.get("bbox")]
        )
    limited_land_ids = set()
    if land_pad_limit > 0:
        land_objects = [obj for obj in objects if obj.get("role") == "land" and obj.get("bbox")]
        if len(land_objects) > land_pad_limit:
            kept = sorted(land_objects, key=lambda obj: (-bbox_area(obj.get("bbox")), object_id_sort_key(obj)))[
                :land_pad_limit
            ]
            limited_land_ids = {id(obj) for obj in kept}
    package_proxy_size = None
    if package_pad_from_land_proxy and package_pad_proxy_size_source == "package_pad_median":
        package_proxy_size = transformed_package_pad_median_size(
            objects,
            graph_candidates,
            gt_candidates,
            package_pad_source_order=package_pad_source_order,
            package_pad_label_filter=package_pad_label_filter,
            package_pad_flip=package_pad_flip,
            package_pad_rotation=package_pad_rotation,
        )
    terminal_land_proxy_keys = set()
    if package_pad_from_land_proxy and package_pad_proxy_land_filter == "terminal_rect":
        terminal_land_objects = [obj for obj in objects if obj.get("role") == "land" and obj.get("bbox")]
        thermal_land_proxy_keys = thermal_like_bbox_keys([obj["bbox"] for obj in terminal_land_objects])
        terminal_land_proxy_keys = {
            bbox_dedupe_key(obj.get("bbox"), 0.001)
            for obj in terminal_land_objects
            if should_proxy_terminal_land_rect(obj, thermal_land_proxy_keys)
        }
        terminal_land_proxy_keys.discard(None)
    aligned = []
    for obj in objects:
        if (
            package_pad_label_filter
            and str(obj.get("role") or "") == "package_pad"
            and package_pad_label_key(obj) != package_pad_label_filter
        ):
            continue
        role = str(obj.get("role") or "")
        if package_pad_from_land_proxy and role == "package_pad":
            continue
        if exclude_lead_contacts and role == "lead":
            continue
        if exclude_lateral_lead_contacts and role == "lead" and str(obj.get("canonical_view") or "") == "lateral":
            continue
        if (
            exclude_thermal_land_pads
            and role == "land"
            and bbox_dedupe_key(obj.get("bbox"), 0.001) in thermal_land_keys
        ):
            continue
        if (
            exclude_thermal_package_pads
            and role == "package_pad"
            and bbox_dedupe_key(obj.get("bbox"), 0.001) in thermal_package_keys
        ):
            continue
        if limited_land_ids and role == "land" and id(obj) not in limited_land_ids:
            continue
        bbox = obj.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        source_bbox = [float(value) for value in bbox[:4]]
        if str(obj.get("source_type") or "") == "scan_result_format":
            # ScanResultFormat bboxes are already in the GT physical coordinate
            # system; applying the reconstruction bbox transform again distorts
            # pad size and spacing.
            source_box = source_bbox
            target_box = source_bbox
            transformed = source_bbox
        else:
            source_box, target_box = transform_boxes_for_role(
                role,
                graph_candidates,
                gt_candidates,
                package_pad_source_order=package_pad_source_order,
            )
            transformed = transform_bbox(source_bbox, source_box, target_box)
            if role == "package_pad" and package_pad_flip:
                transformed = flip_bbox_in_box(transformed, target_box, package_pad_flip)
            if role == "package_pad" and package_pad_rotation:
                transformed = rotate_bbox_in_box(transformed, target_box, package_pad_rotation)
            if role == "land" and (land_pad_flip or land_pad_rotation):
                transformed = transform_bbox_with_source_orientation(source_bbox, source_box, target_box, land_pad_flip, land_pad_rotation)
        if transformed is None:
            continue
        aligned_obj = dict(obj)
        aligned_obj["bbox"] = transformed
        aligned_obj["source"] = "result"
        aligned_obj["alignment_source_bbox"] = source_box
        aligned_obj["alignment_target_bbox"] = target_box
        if role == "package_pad" and package_pad_flip:
            aligned_obj["alignment_package_pad_flip"] = package_pad_flip
        if role == "package_pad" and package_pad_rotation:
            aligned_obj["alignment_package_pad_rotation"] = package_pad_rotation
        if role == "land" and land_pad_flip:
            aligned_obj["alignment_land_pad_flip"] = land_pad_flip
        if role == "land" and land_pad_rotation:
            aligned_obj["alignment_land_pad_rotation"] = land_pad_rotation
        aligned.append(aligned_obj)
        if package_pad_from_land_proxy and role == "land":
            if terminal_land_proxy_keys and bbox_dedupe_key(source_bbox, 0.001) not in terminal_land_proxy_keys:
                continue
            proxy_obj = dict(aligned_obj)
            proxy_obj["role"] = "package_pad"
            proxy_obj["label"] = f"land_proxy_{aligned_obj.get('label', '')}"
            proxy_obj["source_role"] = "land"
            proxy_obj["package_pad_proxy_source"] = "land_pad"
            if package_pad_proxy_land_filter:
                proxy_obj["package_pad_proxy_land_filter"] = package_pad_proxy_land_filter
            if package_proxy_size:
                proxy_obj["bbox"] = resize_bbox_about_center(proxy_obj["bbox"], *package_proxy_size)
                proxy_obj["package_pad_proxy_size_source"] = "package_pad_median"
            aligned.append(proxy_obj)
    return aligned


def transformed_package_pad_median_size(
    objects: list[dict[str, Any]],
    graph_candidates: dict[str, Any],
    gt_candidates: dict[str, Any],
    *,
    package_pad_source_order: tuple[str, ...],
    package_pad_label_filter: str | None,
    package_pad_flip: str,
    package_pad_rotation: int,
) -> tuple[float, float] | None:
    sizes: list[tuple[float, float]] = []
    for obj in objects:
        if str(obj.get("role") or "") != "package_pad":
            continue
        if package_pad_label_filter and package_pad_label_key(obj) != package_pad_label_filter:
            continue
        bbox = obj.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        source_bbox = [float(value) for value in bbox[:4]]
        source_box, target_box = transform_boxes_for_role(
            "package_pad",
            graph_candidates,
            gt_candidates,
            package_pad_source_order=package_pad_source_order,
        )
        transformed = transform_bbox(source_bbox, source_box, target_box)
        if transformed is None:
            continue
        if package_pad_flip:
            transformed = flip_bbox_in_box(transformed, target_box, package_pad_flip)
        if package_pad_rotation:
            transformed = rotate_bbox_in_box(transformed, target_box, package_pad_rotation)
        width = float(transformed[2]) - float(transformed[0])
        height = float(transformed[3]) - float(transformed[1])
        if width > 0 and height > 0:
            sizes.append((width, height))
    if not sizes:
        return None
    widths = sorted(width for width, _ in sizes)
    heights = sorted(height for _, height in sizes)
    midpoint = len(sizes) // 2
    if len(sizes) % 2:
        return widths[midpoint], heights[midpoint]
    return (widths[midpoint - 1] + widths[midpoint]) / 2.0, (heights[midpoint - 1] + heights[midpoint]) / 2.0


def resize_bbox_about_center(bbox: list[float], width: float, height: float) -> list[float]:
    cx = (float(bbox[0]) + float(bbox[2])) / 2.0
    cy = (float(bbox[1]) + float(bbox[3])) / 2.0
    half_w = float(width) / 2.0
    half_h = float(height) / 2.0
    return [cx - half_w, cy - half_h, cx + half_w, cy + half_h]


def canonical_source_objects(canonical: dict[str, Any]) -> list[dict[str, Any]]:
    objects = []
    outline = canonical.get("outline_2d") or {}
    if outline:
        objects.append({"role": "outline", "bbox": outline.get("bbox"), "label": "outline"})
    for role, key in (
        ("package_pad", "package_pads"),
        ("land", "land_pads"),
        ("lead", "lead_contacts"),
        ("lead_pad", "lead_pads"),
        ("inner_land_pad", "inner_land_pads"),
    ):
        for index, obj in enumerate(canonical.get(key) or []):
            payload = {
                "role": role,
                "bbox": obj.get("bbox"),
                "label": str(obj.get("source_object_id") if obj.get("source_object_id") is not None else index),
                "source_type": obj.get("source_type"),
                "source_path": obj.get("source_path"),
                "source_object_id": obj.get("source_object_id"),
                "source_label": obj.get("source_label"),
                "raw_view": obj.get("raw_view"),
                "canonical_view": obj.get("canonical_view"),
            }
            for metadata_key in (
                "source_package_pad_id",
                "source_package_pad_bbox",
                "source_package_pad_index",
                "lead_contact_length",
                "lead_contact_length_axis",
                "lead_contact_length_source",
                "radial_axis",
                "coordinate_unit_scale",
                "source_land_pad_id",
                "source_land_pad_bbox",
                "source_land_pad_index",
                "inner_land_pad_source",
            ):
                if obj.get(metadata_key) is not None:
                    payload[metadata_key] = obj.get(metadata_key)
            objects.append(payload)
    return objects


def transform_boxes_for_role(
    role: str,
    graph_candidates: dict[str, list[float] | None],
    gt_candidates: dict[str, list[float] | None],
    *,
    package_pad_source_order: tuple[str, ...] = ("package", "lead", "all"),
) -> tuple[list[float] | None, list[float] | None]:
    fallback_source = graph_candidates.get("all")
    fallback_target = gt_candidates.get("all")
    if role == "outline":
        return first_bbox(graph_candidates, "outline", "package", "all"), first_bbox(gt_candidates, "shape", "all")
    if role == "package_pad":
        return first_bbox(graph_candidates, *package_pad_source_order), first_bbox(gt_candidates, "lead", "shape", "all")
    if role == "land":
        return first_bbox(graph_candidates, "land", "conductive", "all"), first_bbox(gt_candidates, "land", "conductive", "all")
    if role == "lead":
        return first_bbox(graph_candidates, "lead", "package", "all"), first_bbox(gt_candidates, "lead", "conductive", "all")
    if role == "lead_pad":
        return first_bbox(graph_candidates, *package_pad_source_order), first_bbox(gt_candidates, "lead", "conductive", "all")
    return fallback_source, fallback_target


def first_bbox(candidates: dict[str, list[float] | None], *names: str) -> list[float] | None:
    for name in names:
        bbox = candidates.get(name)
        if bbox and len(bbox) >= 4:
            return [float(value) for value in bbox[:4]]
    return None


def transform_bbox(bbox: list[float], source_box: list[float] | None, target_box: list[float] | None) -> list[float] | None:
    if not source_box or not target_box:
        return None
    sx1, sy1, sx2, sy2 = source_box[:4]
    tx1, ty1, tx2, ty2 = target_box[:4]
    source_w = sx2 - sx1
    source_h = sy2 - sy1
    target_w = tx2 - tx1
    target_h = ty2 - ty1
    if source_w == 0.0 or source_h == 0.0:
        return None
    x1, y1, x2, y2 = bbox[:4]
    return [
        tx1 + (x1 - sx1) / source_w * target_w,
        ty1 + (y1 - sy1) / source_h * target_h,
        tx1 + (x2 - sx1) / source_w * target_w,
        ty1 + (y2 - sy1) / source_h * target_h,
    ]


def transform_bbox_with_source_orientation(
    bbox: list[float],
    source_box: list[float] | None,
    target_box: list[float] | None,
    flip_axis: str,
    rotation_degrees: int,
) -> list[float] | None:
    # Coordinate system: bbox/source_box/target_box are axis-aligned x-right,
    # y-down boxes. Orientation is applied in normalized source coordinates
    # before mapping to the target box. This handles separate source views
    # whose pad layout is rotated relative to the reference view.
    if not source_box or not target_box:
        return None
    sx1, sy1, sx2, sy2 = [float(value) for value in source_box[:4]]
    tx1, ty1, tx2, ty2 = [float(value) for value in target_box[:4]]
    source_w = sx2 - sx1
    source_h = sy2 - sy1
    target_w = tx2 - tx1
    target_h = ty2 - ty1
    if source_w == 0.0 or source_h == 0.0:
        return None
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    normalized_points = [
        ((x - sx1) / source_w, (y - sy1) / source_h)
        for x, y in ((x1, y1), (x1, y2), (x2, y1), (x2, y2))
    ]
    turns = (int(rotation_degrees) // 90) % 4
    oriented_points = []
    for nx, ny in normalized_points:
        if flip_axis in {"flip_x", "flip_xy"}:
            nx = 1.0 - nx
        if flip_axis in {"flip_y", "flip_xy"}:
            ny = 1.0 - ny
        dx = nx - 0.5
        dy = ny - 0.5
        if turns == 1:
            nx, ny = 0.5 + dy, 0.5 - dx
        elif turns == 2:
            nx, ny = 0.5 - dx, 0.5 - dy
        elif turns == 3:
            nx, ny = 0.5 - dy, 0.5 + dx
        oriented_points.append((tx1 + nx * target_w, ty1 + ny * target_h))
    xs = [point[0] for point in oriented_points]
    ys = [point[1] for point in oriented_points]
    return [min(xs), min(ys), max(xs), max(ys)]


def flip_bbox_in_box(
    bbox: list[float] | None,
    target_box: list[float] | None,
    flip_axis: str,
) -> list[float] | None:
    if not bbox or not target_box:
        return bbox
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    tx1, ty1, tx2, ty2 = [float(value) for value in target_box[:4]]
    if flip_axis in {"flip_x", "flip_xy"}:
        x1, x2 = tx1 + tx2 - x2, tx1 + tx2 - x1
    if flip_axis in {"flip_y", "flip_xy"}:
        y1, y2 = ty1 + ty2 - y2, ty1 + ty2 - y1
    return [x1, y1, x2, y2]


def rotate_bbox_in_box(
    bbox: list[float] | None,
    target_box: list[float] | None,
    rotation_degrees: int,
) -> list[float] | None:
    # Coordinate system: same image/graph coordinates as bbox fields, x right,
    # y down. Rotation is clockwise around the target bbox center.
    if not bbox or not target_box:
        return bbox
    turns = (int(rotation_degrees) // 90) % 4
    if turns == 0:
        return bbox
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    tx1, ty1, tx2, ty2 = [float(value) for value in target_box[:4]]
    cx = (tx1 + tx2) / 2.0
    cy = (ty1 + ty2) / 2.0
    rotated_points = []
    for x, y in ((x1, y1), (x1, y2), (x2, y1), (x2, y2)):
        dx = x - cx
        dy = y - cy
        if turns == 1:
            rx, ry = cx + dy, cy - dx
        elif turns == 2:
            rx, ry = cx - dx, cy - dy
        else:
            rx, ry = cx - dy, cy + dx
        rotated_points.append((rx, ry))
    xs = [point[0] for point in rotated_points]
    ys = [point[1] for point in rotated_points]
    return [min(xs), min(ys), max(xs), max(ys)]


def alignment_scores(
    gt: dict[str, Any],
    canonical: dict[str, Any],
    result_objects: list[dict[str, Any]],
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    gt_objects = score_gt_objects_for_selected_count_mappings(gt, checks)
    result_by_role: dict[str, list[list[float]]] = {}
    for obj in result_objects:
        result_by_role.setdefault(str(obj.get("role") or "unknown"), []).append(obj["bbox"])
    gt_by_role: dict[str, list[list[float]]] = {}
    for obj in gt_objects:
        gt_by_role.setdefault(str(obj.get("role") or "unknown"), []).append(obj["bbox"])
    ious = {
        "outline_iou": bbox_iou(union_bbox(gt_by_role.get("shape", [])), union_bbox(result_by_role.get("outline", []))),
        "land_iou": bbox_iou(union_bbox(gt_by_role.get("land", [])), union_bbox(result_by_role.get("land", []))),
        "lead_iou": bbox_iou(union_bbox(gt_by_role.get("lead", [])), union_bbox(result_by_role.get("lead", []) + result_by_role.get("package_pad", []))),
    }
    land_pad_iou_score = matched_box_iou_score(gt_by_role.get("land", []), result_by_role.get("land", []))
    lead_pad_iou_package_only = matched_box_iou_score(gt_by_role.get("lead", []), result_by_role.get("package_pad", []))
    lead_pad_iou_lead_contact_only = matched_box_iou_score(gt_by_role.get("lead", []), result_by_role.get("lead", []))
    lead_pad_iou_score = matched_box_iou_score(
        gt_by_role.get("lead", []),
        result_by_role.get("lead", []) + result_by_role.get("package_pad", []),
    )
    pad_layout_score = average_defined([land_pad_iou_score, lead_pad_iou_score])
    dimension_mismatch_count = count_dimension_conflicts(canonical)
    dimension_count = len(canonical.get("dimensions") or []) if canonical else 0
    dimension_value_score = dimension_score(dimension_count, dimension_mismatch_count)
    input_count_statuses = {
        str(check.get("name")): check.get("status") == "aligned"
        for check in checks
        if str(check.get("name") or "").endswith("_count")
    }
    count_statuses = selected_result_count_statuses(gt_by_role, result_by_role, input_count_statuses)
    scan_result_fallback_count = scan_result_fallback_object_count(result_objects)
    scan_result_fallback_roles = scan_result_fallback_role_counts(result_objects)
    source_independence_score = 0.0 if scan_result_fallback_count else 1.0
    numeric_scores = [value for value in ious.values() if value is not None]
    numeric_scores.extend(value for value in (pad_layout_score, dimension_value_score, source_independence_score) if value is not None)
    numeric_scores.extend(1.0 if aligned else 0.0 for aligned in count_statuses.values())
    overall_score = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0.0
    payload = {
        "overall_score": round(overall_score, 6),
        "source_independence_score": round(source_independence_score, 6),
        "scan_result_fallback_object_count": scan_result_fallback_count,
        "scan_result_fallback_role_counts": scan_result_fallback_roles,
        "gt_land_count": len(gt_by_role.get("land", [])),
        "gt_lead_count": len(gt_by_role.get("lead", [])),
        **{key: None if value is None else round(value, 6) for key, value in ious.items()},
        "land_pad_iou_score": None if land_pad_iou_score is None else round(land_pad_iou_score, 6),
        "lead_pad_iou_package_only": None if lead_pad_iou_package_only is None else round(lead_pad_iou_package_only, 6),
        "lead_pad_iou_lead_contact_only": None if lead_pad_iou_lead_contact_only is None else round(lead_pad_iou_lead_contact_only, 6),
        "lead_pad_iou_score": None if lead_pad_iou_score is None else round(lead_pad_iou_score, 6),
        "pad_layout_score": None if pad_layout_score is None else round(pad_layout_score, 6),
        "dimension_value_score": None if dimension_value_score is None else round(dimension_value_score, 6),
        "dimension_mismatch_count": dimension_mismatch_count,
        "dimension_count": dimension_count,
        "package_pad_count": len(canonical.get("package_pads") or []) if canonical else 0,
        "land_pad_count": len(canonical.get("land_pads") or []) if canonical else 0,
        "lead_contact_count": len(canonical.get("lead_contacts") or []) if canonical else 0,
        "lead_pad_count": len(canonical.get("lead_pads") or []) if canonical else 0,
        "inner_land_pad_count": len(canonical.get("inner_land_pads") or []) if canonical else 0,
        "result_package_pad_count": len(result_by_role.get("package_pad", [])),
        "result_lead_contact_count": len(result_by_role.get("lead", [])),
        "result_lead_pad_count": len(result_by_role.get("lead_pad", [])),
        "result_inner_land_pad_count": len(result_by_role.get("inner_land_pad", [])),
        "land_pad_count_match": count_statuses.get("land_count"),
        "lead_count_match": count_statuses.get("lead_count"),
        "count_checks": count_statuses,
        "input_count_checks": input_count_statuses,
    }
    quality_score = alignment_quality_score(payload)
    payload["quality_score"] = None if quality_score is None else round(quality_score, 6)
    return payload


def score_gt_objects_for_selected_count_mappings(gt: dict[str, Any], checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gt_objects = scan_result_objects(gt)
    object_ids_by_role = selected_group_object_ids_by_role(checks)
    if not object_ids_by_role:
        return gt_objects
    filtered = []
    for obj in gt_objects:
        role = str(obj.get("role") or "unknown")
        selected_ids = object_ids_by_role.get(role)
        object_id = obj.get("source_object_id", obj.get("id"))
        if selected_ids is not None and object_id not in selected_ids:
            continue
        filtered.append(obj)
    return filtered


def selected_group_object_ids_by_role(checks: list[dict[str, Any]]) -> dict[str, set[Any]]:
    selected: dict[str, set[Any]] = {}
    for check in checks:
        name = str(check.get("name") or "")
        role = "land" if name == "land_count" else "lead" if name == "lead_count" else ""
        if not role:
            continue
        if str(check.get("selected_mapping") or "") != f"scan_group_{role}_count_candidate":
            continue
        candidate = check.get("candidate") or {}
        object_ids = candidate.get("object_ids") or []
        expected = int(check.get("expected") or 0)
        if expected <= 0 or len(object_ids) != expected:
            continue
        selected[role] = set(object_ids)
    return selected


def selected_result_count_statuses(
    gt_by_role: dict[str, list[list[float]]],
    result_by_role: dict[str, list[list[float]]],
    input_count_statuses: dict[str, bool],
) -> dict[str, bool]:
    statuses = dict(input_count_statuses)
    if statuses.get("land_count") is False:
        statuses["land_count"] = len(gt_by_role.get("land", [])) == len(result_by_role.get("land", []))
    if statuses.get("lead_count") is False:
        result_lead_equivalent_count = len(result_by_role.get("lead", [])) + len(result_by_role.get("package_pad", []))
        statuses["lead_count"] = len(gt_by_role.get("lead", [])) == result_lead_equivalent_count
    return statuses


def scan_result_fallback_object_count(result_objects: list[dict[str, Any]]) -> int:
    return sum(
        1
        for obj in result_objects
        if str(obj.get("source_type") or "") == "scan_result_format"
        and str(obj.get("role") or "") in {"outline", "package_pad", "land", "lead"}
    )


def scan_result_fallback_role_counts(result_objects: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for obj in result_objects:
        if str(obj.get("source_type") or "") != "scan_result_format":
            continue
        role = str(obj.get("role") or "")
        if role in {"outline", "package_pad", "land", "lead"}:
            counts[role] += 1
    return dict(sorted(counts.items()))


def apply_score_diagnostics(summary: dict[str, Any], canonical: dict[str, Any]) -> None:
    diagnostics = score_diagnostic_details(summary.get("alignment_scores") or {}, canonical)
    add_count_mapping_direct_mismatch_diagnostics(diagnostics, summary)
    add_alignment_transform_explanation_diagnostics(diagnostics, summary)
    reasons = [str(item["reason"]) for item in diagnostics]
    stage_hints = sorted({str(item["stage_hint"]) for item in diagnostics if item.get("stage_hint")})
    objective_sources = sorted(
        {
            str(source)
            for item in diagnostics
            for source in item.get("objective_error_sources", [])
        }
    )
    error_sources = sorted(
        {
            str(source)
            for item in diagnostics
            for source in item.get("error_sources", [])
        }
    )
    summary["score_diagnostic_details"] = diagnostics
    summary["score_diagnostics"] = reasons
    summary["score_stage_hints"] = stage_hints
    summary["score_error_sources"] = error_sources
    summary["score_objective_error_sources"] = objective_sources


def add_count_mapping_direct_mismatch_diagnostics(
    details: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    for check in summary.get("checks") or []:
        selected_mapping = str(check.get("selected_mapping") or "")
        if not selected_mapping or selected_mapping == "direct":
            continue
        direct_expected = check.get("direct_expected")
        direct_actual = check.get("direct_actual")
        if not isinstance(direct_expected, int) or not isinstance(direct_actual, int):
            continue
        direct_delta = abs(direct_actual - direct_expected)
        if direct_delta <= 0:
            continue
        check_name = str(check.get("name") or "count")
        add_score_diagnostic(
            details,
            reason="count_mapping_direct_mismatch",
            metric=f"{check_name}_direct_delta",
            value=float(direct_delta),
            threshold=float(check.get("tolerance") or 0),
            stage_hint="low_score_scan_result_package_pad_alignment",
            extra={
                "check_name": check_name,
                "selected_mapping": selected_mapping,
                "expected_after_mapping": check.get("expected"),
                "actual_after_mapping": check.get("actual"),
                "direct_expected": direct_expected,
                "direct_actual": direct_actual,
                "actual_role": check.get("actual_role"),
                "candidate": check.get("candidate"),
            },
        )


def add_alignment_transform_explanation_diagnostics(
    details: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    transform = summary.get("alignment_transform") or {}
    if not isinstance(transform, dict):
        return
    strategy = str(transform.get("strategy") or "")
    label_filter = str(transform.get("package_pad_label_filter") or "")
    if "subset" not in strategy and not label_filter:
        return
    alignment_scores = summary.get("alignment_scores") or {}
    package_pad_count = alignment_scores.get("package_pad_count")
    result_package_pad_count = alignment_scores.get("result_package_pad_count")
    if not isinstance(package_pad_count, int) or not isinstance(result_package_pad_count, int):
        return
    default_quality_score = transform.get("default_quality_score")
    selected_quality_score = transform.get("selected_quality_score")
    quality_gain = None
    if isinstance(default_quality_score, (int, float)) and isinstance(selected_quality_score, (int, float)):
        quality_gain = float(selected_quality_score) - float(default_quality_score)
    count_changed = package_pad_count != result_package_pad_count
    quality_changed = quality_gain is not None and quality_gain >= 0.1
    if not count_changed and not quality_changed:
        return
    metric = "result_package_pad_count" if count_changed else "selected_quality_score"
    value = float(result_package_pad_count) if count_changed else float(selected_quality_score)
    threshold = float(package_pad_count) if count_changed else float(default_quality_score)
    add_score_diagnostic(
        details,
        reason="package_pad_subset_filter_applied",
        metric=metric,
        value=value,
        threshold=threshold,
        stage_hint="review_note_multiview_package_pad_subset_filter",
        extra={
            "strategy": strategy,
            "package_pad_label_filter": label_filter,
            "package_pad_count": package_pad_count,
            "result_package_pad_count": result_package_pad_count,
            "default_quality_score": default_quality_score,
            "selected_quality_score": selected_quality_score,
            "quality_gain": quality_gain,
        },
    )


def score_diagnostic_details(alignment_scores: dict[str, Any], canonical: dict[str, Any]) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    add_missing_view_score_diagnostics(details, alignment_scores, canonical)
    source_independence_score = alignment_scores.get("source_independence_score")
    if isinstance(source_independence_score, (int, float)) and float(source_independence_score) < 1.0:
        fallback_role_counts = alignment_scores.get("scan_result_fallback_role_counts") or {}
        extra: dict[str, Any] = {"fallback_role_counts": fallback_role_counts}
        fallback_sources = scan_result_fallback_source_summary(canonical, fallback_role_counts)
        if fallback_sources:
            extra["fallback_sources"] = fallback_sources
        add_score_diagnostic(
            details,
            reason="scan_result_geometry_fallback",
            metric="source_independence_score",
            value=float(source_independence_score),
            threshold=1.0,
            stage_hint="low_score_scan_result_geometry_fallback",
            extra=extra,
        )
    add_duplicate_lead_geometry_source_diagnostic(details, alignment_scores)
    add_package_pad_proxy_size_mismatch_diagnostic(details, alignment_scores, canonical)
    add_land_pad_proxy_size_mismatch_diagnostic(details, alignment_scores)
    add_array_pad_layout_mismatch_diagnostic(details, alignment_scores)
    add_low_component_score_diagnostic(
        details,
        alignment_scores,
        canonical,
        key="outline_iou",
        reason="low_outline_iou",
        component="outline",
    )
    add_low_component_score_diagnostic(
        details,
        alignment_scores,
        canonical,
        key="land_pad_iou_score",
        reason="low_land_pad_iou",
        component="land",
    )
    add_low_component_score_diagnostic(
        details,
        alignment_scores,
        canonical,
        key="lead_pad_iou_score",
        reason="low_lead_pad_iou",
        component="package_pad",
    )
    combined_lead_score = alignment_scores.get("lead_pad_iou_score")
    package_only_lead_score = alignment_scores.get("lead_pad_iou_package_only")
    lead_contact_only_score = alignment_scores.get("lead_pad_iou_lead_contact_only")
    if (
        isinstance(combined_lead_score, (int, float))
        and isinstance(package_only_lead_score, (int, float))
        and isinstance(lead_contact_only_score, (int, float))
        and max(float(package_only_lead_score), float(lead_contact_only_score)) - float(combined_lead_score) >= 0.1
        and float(combined_lead_score) <= 0.5
    ):
        add_score_diagnostic(
            details,
            reason="duplicate_lead_geometry_sources",
            metric="lead_pad_iou_score",
            value=float(combined_lead_score),
            threshold=max(float(package_only_lead_score), float(lead_contact_only_score)),
            stage_hint="low_score_multiview_duplicate_lead_geometry_sources",
        )
    add_low_component_score_diagnostic(
        details,
        alignment_scores,
        canonical,
        key="pad_layout_score",
        reason="low_pad_layout_score",
        component="package_pad",
    )
    add_borderline_component_score_diagnostic(
        details,
        alignment_scores,
        canonical,
        key="land_pad_iou_score",
        reason="borderline_land_pad_iou",
        component="land",
    )
    add_borderline_component_score_diagnostic(
        details,
        alignment_scores,
        canonical,
        key="lead_pad_iou_score",
        reason="borderline_lead_pad_iou",
        component="package_pad",
    )
    add_borderline_component_score_diagnostic(
        details,
        alignment_scores,
        canonical,
        key="pad_layout_score",
        reason="borderline_pad_layout_score",
        component="package_pad",
    )
    dimension_score_value = alignment_scores.get("dimension_value_score")
    if isinstance(dimension_score_value, (int, float)) and float(dimension_score_value) < 1.0:
        add_score_diagnostic(
            details,
            reason="dimension_value_conflict",
            metric="dimension_value_score",
            value=float(dimension_score_value),
            threshold=1.0,
            stage_hint="low_score_multiview_dimension_conflict",
        )
    return details


def scan_result_fallback_source_summary(
    canonical: dict[str, Any],
    fallback_role_counts: dict[str, Any],
) -> list[dict[str, Any]]:
    source_selection = canonical_source_selection(canonical)
    role_to_selection_key = {
        "outline": "outline_2d",
        "package_pad": "package_pads",
        "land": "land_pads",
        "lead": "lead_contacts",
    }
    summaries = []
    for object_role, count in sorted((fallback_role_counts or {}).items()):
        selection_key = role_to_selection_key.get(str(object_role), str(object_role))
        selection = source_selection.get(selection_key) or {}
        summary: dict[str, Any] = {
            "object_role": str(object_role),
            "fallback_object_count": count,
            "selection_role": selection_key,
            "source_type": str(selection.get("source_type") or ""),
            "used_fallback": bool(selection.get("used_fallback")),
            "missing_primary": bool(selection.get("missing_primary")),
            "fallback_reason": str(selection.get("fallback_reason") or ""),
            "selected_raw_view": str(selection.get("selected_raw_view") or ""),
            "selected_canonical_view": str(selection.get("selected_canonical_view") or ""),
        }
        for key in (
            "object_count",
            "dimension_count",
            "previous_terminal_package_pad_count",
            "scan_result_lead_count",
            "scan_result_effective_land_count",
            "scan_result_raw_land_count",
        ):
            if key in selection:
                output_key = "selection_object_count" if key == "object_count" else key
                summary[output_key] = selection.get(key)
        for key in ("graph_path", "previous_graph_path", "scan_result_path"):
            value = str(selection.get(key) or "")
            if value:
                summary[key.replace("_path", "_file")] = Path(value).name
        summaries.append(summary)
    return summaries


def canonical_source_selection(canonical: dict[str, Any]) -> dict[str, Any]:
    direct = canonical.get("source_selection")
    if isinstance(direct, dict):
        return direct
    summary = canonical.get("summary") or {}
    nested = summary.get("source_selection")
    return nested if isinstance(nested, dict) else {}


def add_duplicate_lead_geometry_source_diagnostic(
    details: list[dict[str, Any]],
    alignment_scores: dict[str, Any],
) -> None:
    combined_score = alignment_scores.get("lead_pad_iou_score")
    package_only_score = alignment_scores.get("lead_pad_iou_package_only")
    lead_only_score = alignment_scores.get("lead_pad_iou_lead_contact_only")
    package_count = int(alignment_scores.get("result_package_pad_count") or 0)
    lead_count = int(alignment_scores.get("result_lead_contact_count") or 0)
    if package_count <= 0 or lead_count <= 0:
        return
    if not isinstance(combined_score, (int, float)):
        return
    single_scores = [score for score in (package_only_score, lead_only_score) if isinstance(score, (int, float))]
    if not single_scores:
        return
    best_single_score = max(float(score) for score in single_scores)
    # If either source alone matches much better than the combined set, the
    # aligned result is likely drawing duplicate geometry for the same leads.
    if float(combined_score) < 0.5 and best_single_score - float(combined_score) >= 0.1:
        add_score_diagnostic(
            details,
            reason="duplicate_lead_geometry_sources",
            metric="lead_pad_iou_score",
            value=float(combined_score),
            threshold=best_single_score,
            stage_hint="low_score_multiview_duplicate_lead_geometry_sources",
        )


def add_package_pad_proxy_size_mismatch_diagnostic(
    details: list[dict[str, Any]],
    alignment_scores: dict[str, Any],
    canonical: dict[str, Any],
) -> None:
    gt_lead_count = int(alignment_scores.get("gt_lead_count") or 0)
    result_package_count = int(alignment_scores.get("result_package_pad_count") or 0)
    result_lead_count = int(alignment_scores.get("result_lead_contact_count") or 0)
    package_only_score = alignment_scores.get("lead_pad_iou_package_only")
    land_score = alignment_scores.get("land_pad_iou_score")
    if gt_lead_count <= 0 or result_package_count != gt_lead_count or result_lead_count != 0:
        return
    if not isinstance(package_only_score, (int, float)) or float(package_only_score) >= 0.8:
        return
    if not isinstance(land_score, (int, float)) or float(land_score) < 0.8:
        return
    stage_hint = "low_score_package_graph_package_pad_geometry"
    canonical_lead_count = len(canonical.get("lead_contacts") or [])
    if canonical_lead_count > 0 and result_lead_count == 0:
        stage_hint = "low_score_multiview_lateral_lead_projection_excluded"
    add_score_diagnostic(
        details,
        reason="package_pad_proxy_size_mismatch",
        metric="lead_pad_iou_package_only",
        value=float(package_only_score),
        threshold=0.8,
        stage_hint=stage_hint,
        extra={
            "gt_lead_count": gt_lead_count,
            "result_package_pad_count": result_package_count,
            "result_lead_contact_count": result_lead_count,
            "canonical_lead_contact_count": canonical_lead_count,
            "land_pad_iou_score": float(land_score),
        },
    )


def add_array_pad_layout_mismatch_diagnostic(
    details: list[dict[str, Any]],
    alignment_scores: dict[str, Any],
) -> None:
    gt_lead_count = int(alignment_scores.get("gt_lead_count") or 0)
    lead_iou = alignment_scores.get("lead_iou")
    lead_pad_score = alignment_scores.get("lead_pad_iou_score")
    if gt_lead_count < 16:
        return
    if not isinstance(lead_iou, (int, float)) or float(lead_iou) < 0.7:
        return
    if not isinstance(lead_pad_score, (int, float)) or float(lead_pad_score) >= 0.8:
        return
    add_score_diagnostic(
        details,
        reason="array_pad_layout_mismatch",
        metric="lead_pad_iou_score",
        value=float(lead_pad_score),
        threshold=0.8,
        stage_hint="low_score_package_graph_array_pad_layout",
        extra={"gt_lead_count": gt_lead_count, "lead_iou": float(lead_iou)},
    )


def add_land_pad_proxy_size_mismatch_diagnostic(
    details: list[dict[str, Any]],
    alignment_scores: dict[str, Any],
) -> None:
    if not alignment_scores.get("package_pad_from_land_proxy"):
        return
    lead_score = alignment_scores.get("lead_pad_iou_score")
    if not isinstance(lead_score, (int, float)) or float(lead_score) >= 0.8:
        return
    add_score_diagnostic(
        details,
        reason="land_pad_proxy_size_mismatch",
        metric="lead_pad_iou_score",
        value=float(lead_score),
        threshold=0.8,
        stage_hint="low_score_multiview_land_pad_proxy_size_mismatch",
        extra={
            "package_pad_from_land_proxy": True,
            "gt_lead_count": int(alignment_scores.get("gt_lead_count") or 0),
            "result_package_pad_count": int(alignment_scores.get("result_package_pad_count") or 0),
            "result_lead_contact_count": int(alignment_scores.get("result_lead_contact_count") or 0),
        },
    )


def add_missing_view_score_diagnostics(
    details: list[dict[str, Any]],
    alignment_scores: dict[str, Any],
    canonical: dict[str, Any],
) -> None:
    summary = canonical.get("summary") or {}
    missing_views = set(summary.get("missing_canonical_views") or [])
    gt_land_count = int(alignment_scores.get("gt_land_count") or 0)
    gt_lead_count = int(alignment_scores.get("gt_lead_count") or 0)
    lead_contact_count = int(summary.get("lead_contact_count") or len(canonical.get("lead_contacts") or []))
    if (
        "land" in missing_views
        and gt_land_count > 0
        and alignment_scores.get("land_iou") is None
        and alignment_scores.get("land_pad_iou_score") is None
    ):
        add_score_diagnostic(
            details,
            reason="missing_land_view_for_land_iou",
            metric="land_iou",
            value=None,
            threshold=None,
            stage_hint="low_score_data_missing_land_view",
        )
    if "lateral" in missing_views and alignment_scores.get("lead_iou") is None:
        add_score_diagnostic(
            details,
            reason="missing_lateral_view_for_lead_iou",
            metric="lead_iou",
            value=None,
            threshold=None,
            stage_hint="low_score_data_missing_lateral_view",
        )
    value = alignment_scores.get("lead_pad_iou_score")
    missing_or_low_lead_score = not isinstance(value, (int, float)) or float(value) < 0.5
    if "lead_detail" in missing_views and lead_contact_count == 0 and missing_or_low_lead_score:
        add_score_diagnostic(
            details,
            reason="missing_lead_detail_view_for_lead_layout",
            metric="lead_pad_iou_score",
            value=float(value) if isinstance(value, (int, float)) else None,
            threshold=None,
            stage_hint="low_score_data_missing_lead_detail_view",
        )
    if (
        gt_lead_count > 0
        and 0 < lead_contact_count < gt_lead_count
    ):
        add_score_diagnostic(
            details,
            reason="partial_lead_detail_layout",
            metric="lead_contact_count",
            value=float(lead_contact_count),
            threshold=float(gt_lead_count),
            stage_hint="low_score_multiview_partial_lead_detail_layout",
        )


def add_low_component_score_diagnostic(
    details: list[dict[str, Any]],
    alignment_scores: dict[str, Any],
    canonical: dict[str, Any],
    *,
    key: str,
    reason: str,
    component: str,
    threshold: float = 0.8,
) -> None:
    value = alignment_scores.get(key)
    if not isinstance(value, (int, float)) or float(value) >= threshold:
        return
    add_score_diagnostic(
        details,
        reason=reason,
        metric=key,
        value=float(value),
        threshold=threshold,
        stage_hint=score_stage_hint_for_component(component, canonical),
    )


def add_borderline_component_score_diagnostic(
    details: list[dict[str, Any]],
    alignment_scores: dict[str, Any],
    canonical: dict[str, Any],
    *,
    key: str,
    reason: str,
    component: str,
    lower_bound: float = 0.8,
    warning_threshold: float = 0.9,
) -> None:
    value = alignment_scores.get(key)
    if not isinstance(value, (int, float)):
        return
    numeric_value = float(value)
    if numeric_value < lower_bound or numeric_value >= warning_threshold:
        return
    add_score_diagnostic(
        details,
        reason=reason,
        metric=key,
        value=numeric_value,
        threshold=warning_threshold,
        stage_hint=score_stage_hint_for_component(component, canonical),
        extra={"lower_bound": lower_bound},
    )


def add_score_diagnostic(
    details: list[dict[str, Any]],
    *,
    reason: str,
    metric: str,
    value: float | None,
    threshold: float | None,
    stage_hint: str,
    extra: dict[str, Any] | None = None,
) -> None:
    if any(item.get("reason") == reason and item.get("metric") == metric for item in details):
        return
    item = {
        "reason": reason,
        "metric": metric,
        "value": value,
        "threshold": threshold,
        "stage_hint": stage_hint,
        "error_sources": error_sources_for_stage_hints([stage_hint]),
        "objective_error_sources": objective_error_sources_for_stage_hints([stage_hint]),
    }
    if extra:
        item.update(extra)
    details.append(item)


def score_stage_hint_for_component(component: str, canonical: dict[str, Any]) -> str:
    summary = canonical.get("summary") or {}
    source_selection = summary.get("source_selection") or {}
    selection_key = {
        "outline": "outline_2d",
        "land": "land_pads",
        "package_pad": "package_pads",
    }.get(component, "")
    selection = source_selection.get(selection_key) or {}
    if selection.get("source_type") == "scan_result_format":
        return f"low_score_scan_result_{component}_alignment"
    if selection.get("missing_primary"):
        return f"low_score_data_missing_{component}_primary_view"
    if selection.get("used_fallback"):
        return f"low_score_multiview_{component}_fallback_geometry"
    if component in {"land", "package_pad"}:
        return f"low_score_package_graph_{component}_geometry"
    return f"low_score_multiview_{component}_alignment"


def matched_box_iou_score(gt_boxes: list[list[float]], result_boxes: list[list[float]]) -> float | None:
    if not gt_boxes and not result_boxes:
        return None
    denominator = max(len(gt_boxes), len(result_boxes))
    if denominator == 0:
        return None
    gt_sorted = sorted_boxes(gt_boxes)
    result_sorted = sorted_boxes(result_boxes)
    candidates = []
    for gt_index, gt_box in enumerate(gt_sorted):
        for result_index, result_box in enumerate(result_sorted):
            iou = bbox_iou(gt_box, result_box) or 0.0
            candidates.append((-iou, gt_index, result_index))
    matched_gt: set[int] = set()
    matched_result: set[int] = set()
    matched_scores = []
    for negative_iou, gt_index, result_index in sorted(candidates):
        if gt_index in matched_gt or result_index in matched_result:
            continue
        matched_gt.add(gt_index)
        matched_result.add(result_index)
        matched_scores.append(-negative_iou)
    unmatched_count = denominator - len(matched_scores)
    matched_scores.extend([0.0] * unmatched_count)
    return sum(matched_scores) / denominator


def sorted_boxes(boxes: list[list[float]]) -> list[list[float]]:
    def key(box: list[float]) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = [float(value) for value in box[:4]]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return (round(cy, 6), round(cx, 6), round(y2 - y1, 6), round(x2 - x1, 6))

    return sorted(([float(value) for value in box[:4]] for box in boxes if len(box) >= 4), key=key)


def average_defined(values: list[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    if not defined:
        return None
    return sum(defined) / len(defined)


def count_dimension_conflicts(canonical: dict[str, Any]) -> int:
    return sum(1 for conflict in canonical.get("conflicts") or [] if conflict.get("type") == "dimension_value_conflict")


def dimension_score(dimension_count: int, mismatch_count: int) -> float | None:
    if dimension_count <= 0:
        return None
    return max(0.0, 1.0 - mismatch_count / dimension_count)


def bbox_iou(box_a: list[float] | None, box_b: list[float] | None) -> float | None:
    if not box_a or not box_b:
        return None
    ax1, ay1, ax2, ay2 = box_a[:4]
    bx1, by1, bx2, by2 = box_b[:4]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return None
    return intersection / union


def write_scene_svg(
    output_path: Path,
    *,
    title: str,
    layers: list[tuple[str, list[dict[str, Any]]]],
    fallback_label: str,
) -> None:
    boxes = [obj.get("bbox") for _, objects in layers for obj in objects if len(obj.get("bbox") or []) >= 4]
    canvas = union_bbox(boxes)
    if not canvas:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(empty_scene_svg(fallback_label), encoding="utf-8")
        return
    x1, y1, x2, y2 = [float(value) for value in canvas[:4]]
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    pad = max(width, height) * 0.08
    stroke_width = max(width, height) * 0.006
    view_box = f"{x1 - pad:.6g} {y1 - pad:.6g} {width + pad * 2:.6g} {height + pad * 2:.6g}"
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="720" '
        f'viewBox="{view_box}">',
        '<rect x="{}" y="{}" width="{}" height="{}" fill="#ffffff"/>'.format(
            f"{x1 - pad:.6g}", f"{y1 - pad:.6g}", f"{width + pad * 2:.6g}", f"{height + pad * 2:.6g}"
        ),
        f'<text x="{x1:.6g}" y="{y1 - pad * 0.35:.6g}" font-size="{max(width, height) * 0.03:.6g}" fill="#172033">{escape_xml(title)}</text>',
    ]
    for layer_name, objects in layers:
        for obj in objects:
            bbox = obj.get("bbox") or []
            if len(bbox) < 4:
                continue
            bx1, by1, bx2, by2 = [float(value) for value in bbox[:4]]
            role = str(obj.get("role") or "unknown")
            color = scene_color(layer_name, role)
            fill_opacity = "0.08" if layer_name == "gt" else "0.18"
            dash = ' stroke-dasharray="0.05 0.05"' if layer_name == "gt" else ""
            lines.append(
                scene_svg_shape(
                    obj,
                    [bx1, by1, bx2, by2],
                    color=color,
                    stroke_width=stroke_width,
                    fill_opacity=fill_opacity,
                    extra_attrs=dash,
                )
            )
    lines.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def scene_color(layer_name: str, role: str) -> str:
    if layer_name == "gt":
        return {
            "land": "#15803d",
            "lead": "#b45309",
            "shape": "#dc2626",
            "outline_or_line": "#dc2626",
        }.get(role, "#ef4444")
    if layer_name in {
        "top",
        "bottom",
        "land",
        "side",
        "front",
        "lead",
        "land_detail",
    }:
        return {
            "top": "#2563eb",
            "bottom": "#16a34a",
            "land": "#f97316",
            "side": "#0f766e",
            "front": "#e11d48",
            "lead": "#9333ea",
            "land_detail": "#7c3aed",
        }[layer_name]
    return {
        "outline": "#0284c7",
        "package_pad": "#0ea5e9",
        "land": "#16a34a",
        "lead": "#f97316",
        "lead_pad": "#e11d48",
    }.get(role, "#64748b")


def empty_scene_svg(label: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 120">'
        '<rect width="320" height="120" fill="#ffffff"/>'
        f'<text x="20" y="60" font-size="14" fill="#64748b">{escape_xml(label)}</text>'
        "</svg>\n"
    )


def write_scan_result_svg(gt: dict[str, Any], output_path: Path) -> bool:
    objects = list((gt.get("features") or {}).get("objects") or [])
    boxes = [obj.get("bbox") for obj in objects if obj.get("bbox")]
    canvas = union_bbox(boxes)
    if not canvas:
        return False
    x1, y1, x2, y2 = [float(value) for value in canvas]
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    pad = max(width, height) * 0.08
    view_box = f"{x1 - pad:.6g} {y1 - pad:.6g} {width + pad * 2:.6g} {height + pad * 2:.6g}"
    stroke_width = max(width, height) * 0.006
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="720" '
        f'viewBox="{view_box}">',
        '<rect x="{}" y="{}" width="{}" height="{}" fill="#ffffff"/>'.format(
            f"{x1 - pad:.6g}", f"{y1 - pad:.6g}", f"{width + pad * 2:.6g}", f"{height + pad * 2:.6g}"
        ),
    ]
    for obj in objects:
        bbox = obj.get("bbox")
        if not bbox:
            continue
        bx1, by1, bx2, by2 = [float(value) for value in bbox[:4]]
        role = str(obj.get("role") or "unknown")
        color = scan_role_color(role)
        lines.append(scan_result_svg_shape(obj, [bx1, by1, bx2, by2], color=color, stroke_width=stroke_width))
    lines.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def scan_result_svg_shape(obj: dict[str, Any], bbox: list[float], *, color: str, stroke_width: float) -> str:
    """Render one ScanResult object using its native 2D shape when available.

    Coordinate system: ScanResultFormat reference units, x right, y down.
    The bbox is still the geometric extent used by scoring; this helper only
    changes visualization so review images do not turn every GT object into a
    rectangle.
    """
    return scene_svg_shape(
        obj,
        bbox,
        color=color,
        stroke_width=stroke_width,
        fill_opacity="0.16",
    )


def scene_svg_shape(
    obj: dict[str, Any],
    bbox: list[float],
    *,
    color: str,
    stroke_width: float,
    fill_opacity: str,
    extra_attrs: str = "",
) -> str:
    """Render an aligned review object with the known 2D pad shape.

    Coordinate system: same as the input bbox, x right and y down. The bbox is
    not changed; only the SVG primitive changes for review readability.
    """
    bx1, by1, bx2, by2 = [float(value) for value in bbox[:4]]
    width = max(bx2 - bx1, 0.0)
    height = max(by2 - by1, 0.0)
    shape_text = " ".join(
        str(obj.get(key) or "")
        for key in ("node_name", "NodeName", "label", "source_label")
    ).lower()
    if "circle" in shape_text:
        return (
            '<ellipse cx="{:.6g}" cy="{:.6g}" rx="{:.6g}" ry="{:.6g}" '
            'fill="{}" fill-opacity="{}" stroke="{}" stroke-width="{:.6g}"{} />'
        ).format(
            bx1 + width / 2.0,
            by1 + height / 2.0,
            width / 2.0,
            height / 2.0,
            color,
            fill_opacity,
            color,
            stroke_width,
            extra_attrs,
        )
    if "dshape" in shape_text:
        radius = min(width, height) / 2.0
        if width >= height:
            path = (
                f"M {bx1:.6g} {by1:.6g} "
                f"L {max(bx2 - radius, bx1):.6g} {by1:.6g} "
                f"A {radius:.6g} {radius:.6g} 0 0 1 {max(bx2 - radius, bx1):.6g} {by2:.6g} "
                f"L {bx1:.6g} {by2:.6g} Z"
            )
        else:
            path = (
                f"M {bx1:.6g} {by1:.6g} "
                f"L {bx2:.6g} {by1:.6g} "
                f"L {bx2:.6g} {max(by2 - radius, by1):.6g} "
                f"A {radius:.6g} {radius:.6g} 0 0 1 {bx1:.6g} {max(by2 - radius, by1):.6g} "
                f"Z"
            )
        return (
            '<path d="{}" fill="{}" fill-opacity="{}" '
            'stroke="{}" stroke-width="{:.6g}"{} />'
        ).format(path, color, fill_opacity, color, stroke_width, extra_attrs)
    return (
        '<rect x="{:.6g}" y="{:.6g}" width="{:.6g}" height="{:.6g}" '
        'fill="{}" fill-opacity="{}" stroke="{}" stroke-width="{:.6g}"{} />'
    ).format(bx1, by1, width, height, color, fill_opacity, color, stroke_width, extra_attrs)


def scan_role_color(role: str) -> str:
    return {
        "land": "#00a36c",
        "lead": "#2f80ed",
        "shape": "#8e44ad",
        "outline_or_line": "#111827",
        "unknown": "#f59e0b",
    }.get(role, "#f59e0b")


def escape_xml(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def numeric(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number
