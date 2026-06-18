from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from real_image_process.FPK_PJ_fullflow.review.schema import slugify


RISK_RANK = {"high": 0, "medium": 1, "low": 2, "unknown": 3}


def build_auto_improve_queue(
    *,
    run_dir: Path,
    output_root: Path,
    limit: int = 0,
) -> dict[str, Any]:
    alignment_summary_path = run_dir / "outputs" / "eval" / "gt_alignment" / "summary.json"
    alignment_summary = read_json(alignment_summary_path)
    parts = alignment_summary.get("parts") or []
    cases = [case_from_part(run_dir, part) for part in parts]
    cases = [case for case in cases if case is not None]
    cases.sort(key=case_sort_key)
    if limit > 0:
        cases = cases[:limit]

    output_root.mkdir(parents=True, exist_ok=True)
    reviewed_cases_path = output_root / "reviewed_cases.jsonl"
    score_history_path = output_root / "score_history.jsonl"
    iteration_summary_path = output_root / "iteration_summary.json"
    write_jsonl(reviewed_cases_path, cases)
    write_jsonl(score_history_path, [score_history_row(case) for case in cases])

    payload = {
        "stage": "auto_improve_queue",
        "status": "success",
        "run_dir": str(run_dir),
        "output_root": str(output_root),
        "alignment_summary_path": str(alignment_summary_path),
        "reviewed_cases_path": str(reviewed_cases_path),
        "score_history_path": str(score_history_path),
        "iteration_summary_path": str(iteration_summary_path),
        "total_alignment_parts": len(parts),
        "queued_cases": len(cases),
        "risk_counts": risk_counts(cases),
        "lowest_score_cases": [
            {
                "part_number": case["part_number"],
                "overall_score": case["overall_score"],
                "risk_level": case["risk_level"],
                "status": case["status"],
                "reasons": case["reasons"],
                "primary_error_sources": case["error_sources"],
                "objective_error_sources": case["objective_error_sources"],
            }
            for case in cases[:10]
        ],
    }
    iteration_summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def case_from_part(run_dir: Path, part: dict[str, Any]) -> dict[str, Any] | None:
    part_number = str(part.get("part_number") or "")
    if not part_number:
        return None
    scores = part.get("alignment_scores") or {}
    score = scores.get("overall_score")
    numeric_score = float(score) if isinstance(score, (int, float)) else None
    risk_level = risk_level_for_score(numeric_score)
    part_slug = slugify(part_number)
    multiview_part_dir = run_dir / "outputs" / "multiview" / "parts" / part_slug
    alignment_part_dir = run_dir / "outputs" / "eval" / "gt_alignment" / "parts" / part_slug
    layers_path = Path(
        str(
            part.get("unified_multiview_layers_path")
            or multiview_part_dir / "unified_multiview_layers.json"
        )
    )
    evidence_refs = evidence_refs_from_path(multiview_part_dir / "evidence.json")
    return {
        "part_number": part_number,
        "status": part.get("status"),
        "risk_level": risk_level,
        "overall_score": numeric_score,
        "alignment_scores": scores,
        "reasons": list(part.get("reasons") or []),
        "stage_hints": list(part.get("stage_hints") or []),
        "error_sources": list(part.get("error_sources") or []),
        "objective_error_sources": list(part.get("objective_error_sources") or []),
        "checks": list(part.get("checks") or []),
        "paths": {
            "dataset_part_dir": part.get("dataset_part_dir"),
            "scan_result": part.get("scan_result_path"),
            "unified_multiview_layers": str(layers_path),
            "evidence": str(multiview_part_dir / "evidence.json"),
            "conflicts": str(multiview_part_dir / "conflicts.json"),
            "alignment": part.get("alignment_path") or str(alignment_part_dir / "alignment.json"),
            "gt_reference_svg": part.get("gt_reference_svg_path") or str(multiview_part_dir / "gt_reference.svg"),
            "aligned_result_svg": part.get("aligned_result_svg_path") or str(multiview_part_dir / "aligned_result.svg"),
            "comparison_svg": part.get("comparison_svg_path") or str(multiview_part_dir / "comparison.svg"),
        },
        "source_images": source_images_from_evidence(evidence_refs),
        "evidence_refs": evidence_refs,
        "suggested_action": suggested_action(part),
    }


def risk_level_for_score(score: float | None) -> str:
    if score is None:
        return "high"
    if score < 0.5:
        return "high"
    if score < 0.8:
        return "medium"
    return "low"


def suggested_action(part: dict[str, Any]) -> str:
    objective_sources = set(str(source) for source in part.get("objective_error_sources") or [])
    sources = set(str(source) for source in part.get("error_sources") or [])
    hints = set(str(hint) for hint in part.get("stage_hints") or [])
    if "gt_annotation_issue" in objective_sources:
        return "review_dataset_or_gt_annotation"
    if "table_lookup" in objective_sources:
        return "inspect_table_lookup"
    if "package_graph_reconstruction" in objective_sources or "model_prediction" in objective_sources:
        return "inspect_package_graph_reconstruction"
    if "multiview_alignment" in objective_sources:
        return "inspect_multiview_alignment"
    if "scan_result_parsing" in objective_sources:
        return "inspect_scan_result_parser_or_alignment"
    if "data_coverage" in sources:
        return "review_dataset_coverage"
    if "package_graph_reconstruction" in sources:
        return "inspect_package_graph_reconstruction"
    if "multiview_integration" in sources:
        return "inspect_multiview_alignment"
    if "scan_result_parsing_alignment" in sources:
        return "inspect_scan_result_parser_or_alignment"
    if any(hint.startswith("data_missing_") for hint in hints):
        return "review_dataset_coverage"
    return "manual_review"


def case_sort_key(case: dict[str, Any]) -> tuple[int, float, str]:
    score = case.get("overall_score")
    numeric_score = float(score) if isinstance(score, (int, float)) else -1.0
    return (RISK_RANK.get(str(case.get("risk_level") or "unknown"), 3), numeric_score, str(case.get("part_number") or ""))


def score_history_row(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "part_number": case["part_number"],
        "overall_score": case["overall_score"],
        "risk_level": case["risk_level"],
        "status": case["status"],
        "reasons": case["reasons"],
        "error_sources": case["error_sources"],
        "objective_error_sources": case["objective_error_sources"],
    }


def risk_counts(cases: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"high": 0, "medium": 0, "low": 0}
    for case in cases:
        level = str(case.get("risk_level") or "high")
        counts[level] = counts.get(level, 0) + 1
    return {level: counts.get(level, 0) for level in ("high", "medium", "low")}


def evidence_refs_from_path(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    refs = payload.get("evidence_refs") or []
    return [ref for ref in refs if isinstance(ref, dict)]


def source_images_from_evidence(refs: list[dict[str, Any]]) -> list[dict[str, str]]:
    images = []
    seen = set()
    for ref in refs:
        image_path = str(ref.get("image_path") or "")
        if not image_path or image_path in seen:
            continue
        seen.add(image_path)
        images.append(
            {
                "raw_view": str(ref.get("raw_view") or ""),
                "canonical_view": str(ref.get("canonical_view") or ""),
                "image_path": image_path,
                "annotation_path": str(ref.get("annotation_path") or ""),
                "graph_path": str(ref.get("graph_path") or ""),
            }
        )
    return images


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
