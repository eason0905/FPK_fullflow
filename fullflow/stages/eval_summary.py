from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..run_context import RunContext, write_json


ERROR_SOURCE_KEYS = (
    "model_prediction",
    "table_lookup",
    "data_coverage",
    "annotation_gt_mismatch",
    "multiview_integration",
    "package_graph_reconstruction",
    "scan_result_parsing_alignment",
)

OBJECTIVE_ERROR_SOURCE_KEYS = (
    "model_prediction",
    "table_lookup",
    "package_graph_reconstruction",
    "multiview_alignment",
    "scan_result_parsing",
    "gt_annotation_issue",
)

MULTIVIEW_REQUIRED_PART_FILES = (
    "unified_multiview_layers.json",
    "evidence.json",
    "conflicts.json",
    "unified_multiview_layers.svg",
)


def run_eval_summary(context: RunContext, *, dry_run: bool = False) -> dict[str, Any]:
    output_root = context.outputs_dir / "eval" / "fullflow_summary"
    summary_path = output_root / "summary.json"
    stable_summary_path = context.outputs_dir / "eval" / "summary.json"
    if dry_run:
        payload = {
            "stage": "eval_summary",
            "status": "dry_run",
            "summary_path": str(summary_path),
            "stable_summary_path": str(stable_summary_path),
        }
        context.update_status("eval_summary", payload)
        return payload

    payload = build_eval_summary(context)
    write_json(summary_path, payload)
    write_json(stable_summary_path, payload)
    result = {
        "stage": "eval_summary",
        "status": "success",
        "summary_path": str(summary_path),
        "stable_summary_path": str(stable_summary_path),
        "output_root": str(output_root),
    }
    context.update_status("eval_summary", result)
    return result


def build_eval_summary(context: RunContext) -> dict[str, Any]:
    outputs = context.outputs_dir
    check_inputs = read_json(outputs / "check_inputs" / "summary.json")
    predictions = read_json(outputs / "predictions" / "run_summary.json")
    prediction_targets = read_json(outputs / "predictions" / "target_summary.json")
    reconstruction = read_json(outputs / "reconstruction" / context.run_id / "summary.json")
    multiview = read_json(outputs / "multiview" / "summary.json")
    gt_alignment = read_json(outputs / "eval" / "gt_alignment" / "summary.json")
    package_graph_overlay = read_json(outputs / "review" / "package_graph_overlay_gallery" / "gallery_summary.json")
    llm_eval = latest_llm_eval_summary(outputs / "eval")
    llm_review = llm_review_summary(outputs / "review" / "llm_errors")
    yolo_review_path = outputs / "review" / "yolo_errors" / "summary.json"
    yolo_review = read_json(yolo_review_path)
    if yolo_review and not yolo_review.get("summary_path"):
        yolo_review["_summary_path"] = str(yolo_review_path)
    table_lookup = table_lookup_summary(outputs / "diagnosis")
    auto_improve = auto_improve_summary(outputs / "auto_improve")

    error_source_overview = {
        "model_prediction": {
            "llm_eval": llm_eval_summary(llm_eval),
            "llm_review": llm_review,
            "yolo_review": yolo_review_summary(yolo_review),
        },
        "table_lookup": table_lookup,
        "data_coverage": source_count(gt_alignment, "data_coverage"),
        "annotation_gt_mismatch": {
            "known_issue_manifest": str(context.config.known_issues_path),
            "known_issue_count": count_jsonl_lines(context.config.known_issues_path),
        },
        "multiview_integration": source_count(gt_alignment, "multiview_integration"),
        "package_graph_reconstruction": source_count(gt_alignment, "package_graph_reconstruction"),
        "scan_result_parsing_alignment": source_count(gt_alignment, "scan_result_parsing_alignment"),
    }

    return {
        "run_id": context.run_id,
        "run_dir": str(context.run_dir),
        "summary_path": str(outputs / "eval" / "fullflow_summary" / "summary.json"),
        "dataset_root": str(context.config.asset_dataset_root),
        "error_source_keys": list(ERROR_SOURCE_KEYS),
        "objective_error_source_keys": list(OBJECTIVE_ERROR_SOURCE_KEYS),
        "objective_error_source_counts": gt_alignment.get("objective_error_source_counts") or {},
        "alignment_risk_counts": gt_alignment.get("risk_counts")
        or (gt_alignment.get("alignment_score_summary") or {}).get("risk_counts")
        or {},
        "error_source_overview": error_source_overview,
        "stage_summaries": {
            "check_inputs": compact_check_inputs(check_inputs),
            "predictions": compact_predictions(predictions, prediction_targets),
            "llm_eval": llm_eval_summary(llm_eval),
            "llm_review": llm_review,
            "yolo_review": yolo_review_summary(yolo_review),
            "reconstruction": compact_reconstruction(reconstruction),
            "multiview": compact_multiview(multiview),
            "package_graph_overlay": compact_package_graph_overlay(package_graph_overlay),
            "gt_alignment": compact_gt_alignment(gt_alignment),
            "table_lookup": table_lookup,
            "auto_improve": auto_improve,
        },
        "artifacts": artifact_paths(context),
        "output_completeness": output_completeness(context),
    }


def latest_llm_eval_summary(eval_root: Path) -> dict[str, Any]:
    candidates = sorted(
        path
        for path in eval_root.glob("*/overall_summary.json")
        if path.parent.name != "gt_alignment" and path.parent.name != "fullflow_summary"
    )
    if not candidates:
        return {}
    payload = read_json(candidates[-1])
    payload["_summary_path"] = str(candidates[-1])
    return payload


def llm_eval_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return {"available": False}
    task_metrics = {}
    for task_name, metric in (payload.get("task_metrics") or {}).items():
        if isinstance(metric, dict):
            task_metrics[task_name] = {
                "accuracy": metric.get("accuracy"),
                "total": metric.get("total"),
                "correct": metric.get("correct"),
            }
    for metric in payload.get("datasets") or []:
        if isinstance(metric, dict) and metric.get("dataset"):
            task_metrics[str(metric.get("dataset"))] = {
                "accuracy": metric.get("accuracy"),
                "total": metric.get("total"),
                "correct": metric.get("ref_exact_match") or metric.get("clean_exact_match"),
            }
    return {
        "available": True,
        "summary_path": payload.get("_summary_path") or payload.get("summary_path"),
        "overall_accuracy": payload.get("overall_accuracy"),
        "task_metrics": task_metrics,
    }


def yolo_review_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return {"available": False}
    return {
        "available": True,
        "index_path": payload.get("index_path"),
        "total_items": payload.get("total_items") or payload.get("total_images"),
        "error_cases": payload.get("error_cases"),
        "model_path": payload.get("model_path"),
        "data_yaml": payload.get("data_yaml"),
        "split": payload.get("split"),
        "summary_path": payload.get("summary_path") or payload.get("_summary_path"),
    }


def llm_review_summary(review_root: Path) -> dict[str, Any]:
    index_path = review_root / "index.html"
    all_cases_path = review_root / "all" / "cases.jsonl"
    task_case_counts: dict[str, int] = {}
    for cases_path in sorted(review_root.glob("task*/cases.jsonl")):
        task_case_counts[cases_path.parent.name] = count_jsonl_lines(cases_path)

    return {
        "available": index_path.exists(),
        "index_path": str(index_path),
        "all_cases_path": str(all_cases_path),
        "total_items": count_jsonl_lines(all_cases_path),
        "task_case_counts": task_case_counts,
    }


def table_lookup_summary(diagnosis_root: Path) -> dict[str, Any]:
    missing_jsonl = diagnosis_root / "table_lookup_missing" / "table_lookup_missing.jsonl"
    reasons_jsonl = diagnosis_root / "table_lookup_reasons" / "table_lookup_diagnosis.jsonl"
    return {
        "available": missing_jsonl.exists() or reasons_jsonl.exists(),
        "missing_count": count_jsonl_lines(missing_jsonl),
        "diagnosis_count": count_jsonl_lines(reasons_jsonl),
        "missing_jsonl": str(missing_jsonl),
        "diagnosis_jsonl": str(reasons_jsonl),
    }


def auto_improve_summary(auto_improve_root: Path) -> dict[str, Any]:
    summary_path = auto_improve_root / "iteration_summary.json"
    payload = read_json(summary_path)
    if not payload:
        return {
            "available": False,
            "summary_path": str(summary_path),
            "reviewed_cases_path": str(auto_improve_root / "reviewed_cases.jsonl"),
            "score_history_path": str(auto_improve_root / "score_history.jsonl"),
        }
    return {
        "available": True,
        "summary_path": str(summary_path),
        "reviewed_cases_path": payload.get("reviewed_cases_path"),
        "score_history_path": payload.get("score_history_path"),
        "queued_cases": payload.get("queued_cases"),
        "risk_counts": payload.get("risk_counts"),
        "lowest_score_cases": payload.get("lowest_score_cases"),
    }


def source_count(gt_alignment: dict[str, Any], source: str) -> dict[str, Any]:
    counts = gt_alignment.get("error_source_counts") or {}
    return {
        "count": int(counts.get(source, 0) or 0),
        "gt_alignment_summary": gt_alignment.get("summary_path"),
    }


def compact_check_inputs(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "available": bool(payload),
        "dataset": payload.get("dataset") or payload.get("asset_dataset"),
        "model": payload.get("model"),
        "adapter": payload.get("adapter"),
    }


def compact_predictions(run_summary: dict[str, Any], target_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "available": bool(run_summary or target_summary),
        "run_summary": {
            "target_count": first_present(run_summary, "target_count", "num_group_count"),
            "success_count": first_present(run_summary, "success_count", "valid_predictions"),
            "failure_count": first_present(run_summary, "failure_count", "failed_predictions"),
            "file_count": run_summary.get("file_count"),
            "num_group_count": run_summary.get("num_group_count"),
            "write_enabled": run_summary.get("write_enabled"),
            "valid_predictions": run_summary.get("valid_predictions"),
            "written_predictions": run_summary.get("written_predictions"),
            "failed_predictions": run_summary.get("failed_predictions"),
            "skipped_predictions": run_summary.get("skipped_predictions"),
            "predictions_path": run_summary.get("predictions_path"),
            "views": run_summary.get("views"),
        },
        "target_summary": target_summary,
    }


def first_present(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload.get(key)
    return None


def compact_reconstruction(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "available": bool(payload),
        "processed": payload.get("processed"),
        "failed": payload.get("failed"),
        "skipped": payload.get("skipped"),
        "graph_output": payload.get("graph_output") or payload.get("output_dir"),
    }


def compact_multiview(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "available": bool(payload),
        "total_parts": payload.get("total_parts"),
        "part_outputs": payload.get("part_outputs"),
        "canonical_parts": payload.get("canonical_parts"),
        "graph_based_parts": payload.get("graph_based_parts"),
        "failure_reason_parts": payload.get("failure_reason_parts"),
        "missing_graph_parts": payload.get("missing_graph_parts"),
        "status_counts": payload.get("status_counts"),
        "risk_counts": payload.get("risk_counts"),
        "dimension_value_source_counts": payload.get("dimension_value_source_counts"),
        "dimension_role_counts": payload.get("dimension_role_counts"),
        "dimension_canonical_view_counts": payload.get("dimension_canonical_view_counts"),
        "evidence_type_counts": payload.get("evidence_type_counts"),
        "summary_path": payload.get("summary_path"),
    }


def compact_gt_alignment(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "available": bool(payload),
        "total_parts": payload.get("total_parts"),
        "aligned_parts": payload.get("aligned_parts"),
        "mismatch_parts": payload.get("mismatch_parts"),
        "missing_gt_parts": payload.get("missing_gt_parts"),
        "missing_canonical_parts": payload.get("missing_canonical_parts"),
        "reason_counts": payload.get("reason_counts"),
        "stage_hint_counts": payload.get("stage_hint_counts"),
        "error_source_counts": payload.get("error_source_counts"),
        "mapping_counts": payload.get("mapping_counts"),
        "mismatch_check_counts": payload.get("mismatch_check_counts"),
        "count_delta_histograms": payload.get("count_delta_histograms"),
        "stage_hint_reason_counts": payload.get("stage_hint_reason_counts"),
        "alignment_score_summary": payload.get("alignment_score_summary"),
        "risk_counts": payload.get("risk_counts")
        or (payload.get("alignment_score_summary") or {}).get("risk_counts"),
        "summary_path": payload.get("summary_path"),
        "mismatches_path": payload.get("mismatches_path"),
    }


def compact_package_graph_overlay(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "available": bool(payload),
        "part_count": payload.get("part_count"),
        "skipped_count": payload.get("skipped_count"),
        "index": payload.get("index"),
        "output_dir": payload.get("output_dir"),
        "method": payload.get("method"),
    }


def artifact_paths(context: RunContext) -> dict[str, str]:
    outputs = context.outputs_dir
    return {
        "predictions": str(outputs / "predictions"),
        "reconstruction": str(outputs / "reconstruction" / context.run_id),
        "multiview": str(outputs / "multiview"),
        "package_graph_overlay": str(outputs / "review" / "package_graph_overlay_gallery" / "index.html"),
        "gt_alignment_legacy": str(outputs / "eval" / "gt_alignment"),
        "llm_review": str(outputs / "review" / "llm_errors" / "index.html"),
        "yolo_review": str(outputs / "review" / "yolo_errors" / "index.html"),
        "package_graph_review": str(outputs / "review" / "package_graph" / "index.html"),
        "package_graph_all_views_review": str(outputs / "review" / "package_graph_all_views" / "index.html"),
        "multiview_review": str(outputs / "review" / "multiview" / "index.html"),
        "gt_alignment_review_legacy": str(outputs / "review" / "gt_alignment" / "index.html"),
        "final_comparison_review_legacy": str(outputs / "review" / "final_comparison" / "index.html"),
        "auto_improve": str(outputs / "auto_improve" / "iteration_summary.json"),
    }


def output_completeness(context: RunContext) -> dict[str, Any]:
    outputs = context.outputs_dir
    required_paths = {
        "predictions_dir": outputs / "predictions",
        "predictions_run_summary": outputs / "predictions" / "run_summary.json",
        "predictions_target_summary": outputs / "predictions" / "target_summary.json",
        "predictions_jsonl": outputs / "predictions" / "predictions.jsonl",
        "reconstruction_summary": outputs / "reconstruction" / context.run_id / "summary.json",
        "reconstruction_graphs_dir": outputs / "reconstruction" / context.run_id / "graphs",
        "multiview_summary": outputs / "multiview" / "summary.json",
        "package_graph_overlay_index": outputs / "review" / "package_graph_overlay_gallery" / "index.html",
        "package_graph_overlay_summary": outputs / "review" / "package_graph_overlay_gallery" / "gallery_summary.json",
        "llm_review_index": outputs / "review" / "llm_errors" / "index.html",
        "yolo_review_index": outputs / "review" / "yolo_errors" / "index.html",
        "package_graph_review_index": outputs / "review" / "package_graph" / "index.html",
        "package_graph_all_views_review_index": outputs / "review" / "package_graph_all_views" / "index.html",
        "multiview_review_index": outputs / "review" / "multiview" / "index.html",
        "auto_improve_iteration_summary": outputs / "auto_improve" / "iteration_summary.json",
        "auto_improve_reviewed_cases": outputs / "auto_improve" / "reviewed_cases.jsonl",
        "auto_improve_score_history": outputs / "auto_improve" / "score_history.jsonl",
    }
    artifacts = {name: path_status(path) for name, path in required_paths.items()}
    missing = [name for name, item in artifacts.items() if not item["exists"]]
    parts_root = outputs / "multiview" / "parts"
    multiview_parts = multiview_part_file_status(parts_root)
    return {
        "all_required_present": not missing and multiview_parts["incomplete_part_count"] == 0,
        "missing_required_artifacts": missing,
        "required_artifacts": artifacts,
        "multiview_part_files": multiview_parts,
        "multiview_evidence": multiview_evidence_completeness(parts_root),
        "eval_summary_paths": {
            "stable_summary": str(outputs / "eval" / "summary.json"),
            "nested_summary": str(outputs / "eval" / "fullflow_summary" / "summary.json"),
        },
    }


def path_status(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "kind": "dir" if path.is_dir() else "file" if path.is_file() else "missing",
    }


def multiview_part_file_status(parts_root: Path) -> dict[str, Any]:
    if not parts_root.exists():
        return {
            "parts_root": str(parts_root),
            "required_files": list(MULTIVIEW_REQUIRED_PART_FILES),
            "total_part_dirs": 0,
            "complete_part_count": 0,
            "incomplete_part_count": 0,
            "incomplete_samples": [],
        }
    total = 0
    complete = 0
    incomplete_samples = []
    for part_dir in sorted(path for path in parts_root.iterdir() if path.is_dir()):
        total += 1
        missing = [name for name in MULTIVIEW_REQUIRED_PART_FILES if not (part_dir / name).exists()]
        if not missing:
            complete += 1
            continue
        if len(incomplete_samples) < 20:
            incomplete_samples.append(
                {
                    "part_number": part_dir.name,
                    "part_dir": str(part_dir),
                    "missing_files": missing,
                }
            )
    return {
        "parts_root": str(parts_root),
        "required_files": list(MULTIVIEW_REQUIRED_PART_FILES),
        "total_part_dirs": total,
        "complete_part_count": complete,
        "incomplete_part_count": total - complete,
        "incomplete_samples": incomplete_samples,
    }


def multiview_evidence_completeness(parts_root: Path) -> dict[str, Any]:
    if not parts_root.exists():
        return {
            "parts_root": str(parts_root),
            "total_part_dirs": 0,
            "scan_result_evidence_count": 0,
            "missing_scan_result_evidence_count": 0,
            "package_graph_evidence_expected_count": 0,
            "package_graph_evidence_count": 0,
            "missing_package_graph_evidence_count": 0,
            "missing_scan_result_samples": [],
            "missing_package_graph_samples": [],
        }

    total = 0
    scan_result_count = 0
    missing_scan_samples = []
    package_expected = 0
    package_count = 0
    missing_package_samples = []
    for part_dir in sorted(path for path in parts_root.iterdir() if path.is_dir()):
        total += 1
        canonical_path = multiview_layers_path(part_dir)
        canonical = read_json(canonical_path)
        refs = canonical.get("evidence_refs") or []
        if any(str(ref.get("evidence_type") or "") == "scan_result_format" for ref in refs):
            scan_result_count += 1
        elif len(missing_scan_samples) < 20:
            missing_scan_samples.append({"part_number": part_dir.name, "unified_multiview_layers": str(canonical_path)})

        if str(canonical.get("status") or "") == "missing_graphs":
            continue
        package_expected += 1
        if any(str(ref.get("evidence_type") or "") == "package_graph" for ref in refs):
            package_count += 1
        elif len(missing_package_samples) < 20:
            missing_package_samples.append({"part_number": part_dir.name, "unified_multiview_layers": str(canonical_path)})

    return {
        "parts_root": str(parts_root),
        "total_part_dirs": total,
        "scan_result_evidence_count": scan_result_count,
        "missing_scan_result_evidence_count": total - scan_result_count,
        "package_graph_evidence_expected_count": package_expected,
        "package_graph_evidence_count": package_count,
        "missing_package_graph_evidence_count": package_expected - package_count,
        "missing_scan_result_samples": missing_scan_samples,
        "missing_package_graph_samples": missing_package_samples,
    }


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def multiview_layers_path(part_dir: Path) -> Path:
    return part_dir / "unified_multiview_layers.json"


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())
