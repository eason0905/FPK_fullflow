#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from real_image_process.FPK_PJ_fullflow.gt_alignment.evaluator import parse_scan_result


PRED_BASE_PIN_ROLES = {
    "merged_pad",
    "unmerged_pad",
    "land_pad",
    "package_pad",
    "lead_pad",
    "partial_pad_width",
    "partial_lead_pad_length",
}
PRED_OPTIONAL_PIN_ROLES = {"inner_land_pad"}
GT_PIN_ROLES = {"land", "lead"}


@dataclass(frozen=True)
class ScoreWeights:
    iou_ic: float = 0.25
    pin_count: float = 0.25
    d_pin: float = 0.25
    iou_pin: float = 0.25

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ScoreWeights":
        payload = payload or {}
        return cls(
            iou_ic=float(payload.get("iou_ic", 0.25)),
            pin_count=float(payload.get("pin_count", 0.25)),
            d_pin=float(payload.get("d_pin", 0.25)),
            iou_pin=float(payload.get("iou_pin", 0.25)),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "iou_ic": self.iou_ic,
            "pin_count": self.pin_count,
            "d_pin": self.d_pin,
            "iou_pin": self.iou_pin,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate merge-stage footprint geometry against ScanResultFormat GT using "
            "paper-style pin count, pin center, pin dimension, and whole-layout IoU metrics."
        )
    )
    parser.add_argument("--merge-root", type=Path, required=True, help="Gallery root containing part/*/mergy_pad.json")
    parser.add_argument("--dataset-root", type=Path, required=True, help="dataset_full_v5 root containing ScanResultFormat.txt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, default=None, help="Optional low-score-first gallery output directory.")
    parser.add_argument(
        "--table-missing-graph-root",
        type=Path,
        default=None,
        help="Optional reconstruction graph root; parts containing table_lookup_missing are excluded from metric summary.",
    )
    parser.add_argument("--weight-iou-ic", type=float, default=0.25)
    parser.add_argument("--weight-pin-count", type=float, default=0.25)
    parser.add_argument("--weight-d-pin", type=float, default=0.25)
    parser.add_argument("--weight-iou-pin", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_merge_root(
        merge_root=args.merge_root,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        review_dir=args.review_dir,
        table_missing_graph_root=args.table_missing_graph_root,
        weights=ScoreWeights(
            iou_ic=args.weight_iou_ic,
            pin_count=args.weight_pin_count,
            d_pin=args.weight_d_pin,
            iou_pin=args.weight_iou_pin,
        ),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def evaluate_merge_root(
    *,
    merge_root: Path,
    dataset_root: Path,
    output_dir: Path,
    review_dir: Path | None = None,
    table_missing_graph_root: Path | None = None,
    weights: ScoreWeights | None = None,
) -> dict[str, Any]:
    weights = weights or ScoreWeights()
    output_dir.mkdir(parents=True, exist_ok=True)
    table_missing_parts = load_table_missing_parts(table_missing_graph_root) if table_missing_graph_root else set()
    records = []
    for merge_path in sorted(merge_root.glob("*/mergy_pad.json")):
        part_number = merge_path.parent.name
        scan_path = dataset_root / part_number / "ScanResultFormat.txt"
        record = evaluate_part(
            merge_path=merge_path,
            scan_path=scan_path,
            part_number=part_number,
            weights=weights,
        )
        if part_number in table_missing_parts:
            record["status"] = "excluded_table_lookup_missing"
            record["excluded_from_metrics"] = True
            record["exclusion_reason"] = "table_lookup_missing"
        records.append(record)
    summary = summarize_records(
        records,
        merge_root=merge_root,
        dataset_root=dataset_root,
        table_missing_graph_root=table_missing_graph_root,
        weights=weights,
        review_dir=review_dir,
    )
    (output_dir / "records.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if review_dir is not None:
        write_score_gallery(review_dir, records, summary=summary)
    return summary


def load_table_missing_parts(graph_root: Path | None) -> set[str]:
    if graph_root is None or not graph_root.is_dir():
        return set()
    parts = set()
    for path in graph_root.glob("*/*.json"):
        if json_has_value_source(path, "table_lookup_missing"):
            parts.add(path.parent.name)
    return parts


def json_has_value_source(path: Path, value_source: str) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return has_value_source(payload, value_source)


def has_value_source(payload: Any, value_source: str) -> bool:
    if isinstance(payload, dict):
        if payload.get("value_source") == value_source:
            return True
        return any(has_value_source(value, value_source) for value in payload.values())
    if isinstance(payload, list):
        return any(has_value_source(value, value_source) for value in payload)
    return False


def evaluate_part(
    *,
    merge_path: Path,
    scan_path: Path,
    part_number: str,
    weights: ScoreWeights | None = None,
) -> dict[str, Any]:
    weights = weights or ScoreWeights()
    merge_payload = json.loads(merge_path.read_text(encoding="utf-8"))
    pred_objects = [obj for obj in merge_payload.get("objects") or [] if object_bbox(obj) is not None]
    gt_objects = load_gt_pin_objects(scan_path)
    base_pred = pin_objects(pred_objects, include_inner_land=False)
    inner_pred = pin_objects(pred_objects, include_inner_land=True)
    base_metrics = geometry_metrics(base_pred, gt_objects)
    inner_metrics = geometry_metrics(inner_pred, gt_objects)
    metric_selection = select_inner_land_by_metric(base_metrics, inner_metrics)
    selected_metrics = build_selected_metrics(base_metrics, inner_metrics, metric_selection)
    score_payload = weighted_score(selected_metrics, weights)
    use_inner = any(metric_selection.values())
    status = "ok" if gt_objects and max(base_metrics["pred_pin_count"], inner_metrics["pred_pin_count"]) > 0 else "unscored"
    return {
        "part_number": part_number,
        "merge_path": str(merge_path),
        "scan_result_path": str(scan_path) if scan_path.exists() else "",
        "status": status,
        "inner_land_policy": "per_metric_included_if_that_metric_improves",
        "inner_land_included": use_inner,
        "inner_land_metric_selection": metric_selection,
        "base_metrics": base_metrics,
        "with_inner_land_metrics": inner_metrics,
        "selected_metrics": selected_metrics,
        "metric_scores": score_payload["metric_scores"],
        "weighted_score": score_payload["weighted_score"] if status == "ok" else None,
        "score_weights": weights.to_dict(),
    }


def load_gt_pin_objects(scan_path: Path) -> list[dict[str, Any]]:
    if not scan_path.is_file():
        return []
    gt = parse_scan_result(scan_path)
    features = gt.get("features") or {}
    objects = features.get("effective_objects") or features.get("objects") or []
    result = []
    for obj in objects:
        role = str(obj.get("role") or "")
        if role not in GT_PIN_ROLES:
            continue
        bbox = object_bbox(obj)
        if bbox is None:
            continue
        result.append({"role": role, "bbox": bbox, "source_object_id": obj.get("id")})
    return result


def pin_objects(objects: list[dict[str, Any]], *, include_inner_land: bool) -> list[dict[str, Any]]:
    roles = set(PRED_BASE_PIN_ROLES)
    if include_inner_land:
        roles |= PRED_OPTIONAL_PIN_ROLES
    result = []
    for obj in objects:
        role = str(obj.get("role") or "")
        if role not in roles:
            continue
        bbox = object_bbox(obj)
        if bbox is None:
            continue
        result.append({"role": role, "bbox": bbox, "source_object_id": obj.get("source_object_id")})
    return result


def geometry_metrics(pred_objects: list[dict[str, Any]], gt_objects: list[dict[str, Any]]) -> dict[str, Any]:
    pred_boxes = [obj["bbox"] for obj in pred_objects]
    gt_boxes = [obj["bbox"] for obj in gt_objects]
    transformed_pred_boxes, transform = align_pred_boxes_to_gt_with_quarter_turns(pred_boxes, gt_boxes)
    pairs = match_boxes_by_center(transformed_pred_boxes, gt_boxes)
    center_distances = [center_distance(transformed_pred_boxes[pred_index], gt_boxes[gt_index]) for pred_index, gt_index in pairs]
    pin_ious = [bbox_iou(transformed_pred_boxes[pred_index], gt_boxes[gt_index]) for pred_index, gt_index in pairs]
    pin_denominator = max(len(transformed_pred_boxes), len(gt_boxes), 1)
    mean_pin_iou = sum(pin_ious) / pin_denominator if pin_denominator else None
    d_pin = sum(center_distances) / len(center_distances) if center_distances else None
    count_error = len(pred_boxes) - len(gt_boxes)
    gt_frame = union_bbox(gt_boxes)
    gt_diagonal = bbox_diagonal(gt_frame)
    d_pin_normalized = d_pin / gt_diagonal if isinstance(d_pin, (int, float)) and gt_diagonal > 0.0 else None
    return {
        "pred_pin_count": len(pred_boxes),
        "gt_pin_count": len(gt_boxes),
        "pin_count_error": count_error,
        "pin_count_abs_error": abs(count_error),
        "pin_count_sq_error": count_error * count_error,
        "iou_ic": union_iou(transformed_pred_boxes, gt_boxes),
        "d_pin": d_pin,
        "d_pin_normalized": d_pin_normalized,
        "iou_pin": mean_pin_iou,
        "matched_pin_count": len(pairs),
        "unmatched_pred_count": max(len(pred_boxes) - len(pairs), 0),
        "unmatched_gt_count": max(len(gt_boxes) - len(pairs), 0),
        "alignment_transform": transform,
    }


def align_pred_boxes_to_gt_with_quarter_turns(
    pred_boxes: list[list[float]],
    gt_boxes: list[list[float]],
) -> tuple[list[list[float]], dict[str, Any]]:
    """Align predicted pin boxes to GT physical coordinates.

    Coordinate system: all input and output boxes are axis-aligned
    [x1, y1, x2, y2] in each stage's physical coordinate units.  The
    evaluation transform may rotate the predicted layout by 0/90/180/270
    degrees around its own pin-layout center before the existing frame
    scale/translate.  A rotated candidate is used only when pin count is
    unchanged and IoU_IC, d_pin, and IoU_pin are all no worse than the
    unrotated candidate.
    """
    candidates = []
    for quarter_turns in range(4):
        rotated_boxes = rotate_boxes_by_quarter_turns(pred_boxes, quarter_turns)
        transformed_boxes, transform = align_pred_boxes_to_gt(rotated_boxes, gt_boxes)
        metrics = alignment_candidate_metrics(transformed_boxes, gt_boxes)
        transform = dict(transform)
        transform["quarter_turns"] = quarter_turns
        transform["rotation_degrees"] = quarter_turns * 90
        candidates.append((transformed_boxes, transform, metrics))
    if not candidates:
        return [], {"status": "missing_frame"}
    best_boxes, best_transform, best_metrics = candidates[0]
    best_score = alignment_metric_score(best_metrics)
    for transformed_boxes, transform, metrics in candidates[1:]:
        if not alignment_metrics_no_worse(metrics, best_metrics):
            continue
        score = alignment_metric_score(metrics)
        if score > best_score:
            best_boxes = transformed_boxes
            best_transform = transform
            best_metrics = metrics
            best_score = score
    if best_transform.get("quarter_turns"):
        best_transform["status"] = "quarter_turn_axis_aligned_frame_scale_translate"
    return best_boxes, best_transform


def rotate_boxes_by_quarter_turns(boxes: list[list[float]], quarter_turns: int) -> list[list[float]]:
    """Rotate axis-aligned boxes by quarter turns around the union-frame center."""
    quarter_turns = quarter_turns % 4
    if quarter_turns == 0:
        return [list(box) for box in boxes]
    frame = union_bbox(boxes)
    if frame is None:
        return []
    center_x = (frame[0] + frame[2]) / 2.0
    center_y = (frame[1] + frame[3]) / 2.0
    rotated = []
    for box in boxes:
        points = ((box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3]))
        rotated_points = []
        for x, y in points:
            dx = x - center_x
            dy = y - center_y
            if quarter_turns == 1:
                rx, ry = -dy, dx
            elif quarter_turns == 2:
                rx, ry = -dx, -dy
            else:
                rx, ry = dy, -dx
            rotated_points.append((center_x + rx, center_y + ry))
        xs = [point[0] for point in rotated_points]
        ys = [point[1] for point in rotated_points]
        rotated.append([min(xs), min(ys), max(xs), max(ys)])
    return rotated


def alignment_candidate_metrics(transformed_pred_boxes: list[list[float]], gt_boxes: list[list[float]]) -> dict[str, Any]:
    pairs = match_boxes_by_center(transformed_pred_boxes, gt_boxes)
    center_distances = [center_distance(transformed_pred_boxes[pred_index], gt_boxes[gt_index]) for pred_index, gt_index in pairs]
    pin_ious = [bbox_iou(transformed_pred_boxes[pred_index], gt_boxes[gt_index]) for pred_index, gt_index in pairs]
    pin_denominator = max(len(transformed_pred_boxes), len(gt_boxes), 1)
    return {
        "iou_ic": union_iou(transformed_pred_boxes, gt_boxes),
        "d_pin": sum(center_distances) / len(center_distances) if center_distances else None,
        "iou_pin": sum(pin_ious) / pin_denominator if pin_denominator else None,
        "matched_pin_count": len(pairs),
        "pred_pin_count": len(transformed_pred_boxes),
        "gt_pin_count": len(gt_boxes),
    }


def alignment_metrics_no_worse(candidate: dict[str, Any], base: dict[str, Any]) -> bool:
    return (
        metric_no_worse(candidate.get("iou_ic"), base.get("iou_ic"), higher_is_better=True)
        and metric_no_worse(candidate.get("d_pin"), base.get("d_pin"), higher_is_better=False)
        and metric_no_worse(candidate.get("iou_pin"), base.get("iou_pin"), higher_is_better=True)
        and int(candidate.get("pred_pin_count") or 0) == int(base.get("pred_pin_count") or 0)
        and int(candidate.get("gt_pin_count") or 0) == int(base.get("gt_pin_count") or 0)
    )


def metric_no_worse(candidate: Any, base: Any, *, higher_is_better: bool) -> bool:
    if not isinstance(candidate, (int, float)):
        return not isinstance(base, (int, float))
    if not isinstance(base, (int, float)):
        return True
    if higher_is_better:
        return float(candidate) >= float(base)
    return float(candidate) <= float(base)


def alignment_metric_score(metrics: dict[str, Any]) -> float:
    score = 0.0
    for key in ("iou_ic", "iou_pin"):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            score += float(value)
    value = metrics.get("d_pin")
    if isinstance(value, (int, float)):
        score -= float(value)
    return score


def align_pred_boxes_to_gt(
    pred_boxes: list[list[float]],
    gt_boxes: list[list[float]],
) -> tuple[list[list[float]], dict[str, Any]]:
    pred_frame = union_bbox(pred_boxes)
    gt_frame = union_bbox(gt_boxes)
    if pred_frame is None or gt_frame is None:
        return [], {"status": "missing_frame"}
    pred_w = pred_frame[2] - pred_frame[0]
    pred_h = pred_frame[3] - pred_frame[1]
    gt_w = gt_frame[2] - gt_frame[0]
    gt_h = gt_frame[3] - gt_frame[1]
    if pred_w <= 0.0 or pred_h <= 0.0 or gt_w <= 0.0 or gt_h <= 0.0:
        return [], {"status": "degenerate_frame", "pred_frame": pred_frame, "gt_frame": gt_frame}
    scale_x = gt_w / pred_w
    scale_y = gt_h / pred_h
    transformed = [
        [
            gt_frame[0] + (box[0] - pred_frame[0]) * scale_x,
            gt_frame[1] + (box[1] - pred_frame[1]) * scale_y,
            gt_frame[0] + (box[2] - pred_frame[0]) * scale_x,
            gt_frame[1] + (box[3] - pred_frame[1]) * scale_y,
        ]
        for box in pred_boxes
    ]
    return transformed, {
        "status": "axis_aligned_frame_scale_translate",
        "pred_frame": pred_frame,
        "gt_frame": gt_frame,
        "scale_x": scale_x,
        "scale_y": scale_y,
    }


def select_inner_land_by_metric(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, bool]:
    return {
        "iou_ic": metric_is_better(candidate.get("iou_ic"), base.get("iou_ic"), higher_is_better=True),
        "pin_count": metric_is_better(
            candidate.get("pin_count_abs_error"),
            base.get("pin_count_abs_error"),
            higher_is_better=False,
        ),
        "d_pin": metric_is_better(candidate.get("d_pin"), base.get("d_pin"), higher_is_better=False),
        "iou_pin": metric_is_better(candidate.get("iou_pin"), base.get("iou_pin"), higher_is_better=True),
    }


def build_selected_metrics(
    base: dict[str, Any],
    candidate: dict[str, Any],
    metric_selection: dict[str, bool],
) -> dict[str, Any]:
    selected = dict(base)
    if metric_selection.get("iou_ic"):
        selected["iou_ic"] = candidate.get("iou_ic")
    if metric_selection.get("pin_count"):
        for key in ("pred_pin_count", "gt_pin_count", "pin_count_error", "pin_count_abs_error", "pin_count_sq_error"):
            selected[key] = candidate.get(key)
    if metric_selection.get("d_pin"):
        selected["d_pin"] = candidate.get("d_pin")
        selected["d_pin_normalized"] = candidate.get("d_pin_normalized")
    if metric_selection.get("iou_pin"):
        selected["iou_pin"] = candidate.get("iou_pin")
    return selected


def metric_is_better(candidate: Any, base: Any, *, higher_is_better: bool) -> bool:
    if not isinstance(candidate, (int, float)):
        return False
    if not isinstance(base, (int, float)):
        return True
    if higher_is_better:
        return float(candidate) > float(base)
    return float(candidate) < float(base)


def weighted_score(metrics: dict[str, Any], weights: ScoreWeights) -> dict[str, Any]:
    metric_scores = {
        "iou_ic": clamp01(metrics.get("iou_ic")),
        "pin_count": pin_count_score(metrics),
        "d_pin": d_pin_score(metrics),
        "iou_pin": clamp01(metrics.get("iou_pin")),
    }
    total_weight = 0.0
    weighted_total = 0.0
    for key, score in metric_scores.items():
        weight = weights.to_dict()[key]
        if score is None or weight <= 0.0:
            continue
        weighted_total += weight * score
        total_weight += weight
    return {
        "metric_scores": metric_scores,
        "weighted_score": weighted_total / total_weight if total_weight > 0.0 else None,
    }


def pin_count_score(metrics: dict[str, Any]) -> float | None:
    pred = metrics.get("pred_pin_count")
    gt = metrics.get("gt_pin_count")
    if not isinstance(pred, int) or not isinstance(gt, int):
        return None
    denominator = max(pred, gt, 1)
    return max(0.0, 1.0 - abs(pred - gt) / denominator)


def d_pin_score(metrics: dict[str, Any]) -> float | None:
    value = metrics.get("d_pin_normalized")
    if not isinstance(value, (int, float)):
        return None
    return max(0.0, 1.0 - float(value))


def clamp01(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return max(0.0, min(1.0, float(value)))


def summarize_records(
    records: list[dict[str, Any]],
    *,
    merge_root: Path,
    dataset_root: Path,
    table_missing_graph_root: Path | None = None,
    weights: ScoreWeights | None = None,
    review_dir: Path | None = None,
) -> dict[str, Any]:
    weights = weights or ScoreWeights()
    scored = [record for record in records if record.get("status") == "ok"]
    excluded = [record for record in records if record.get("status") == "excluded_table_lookup_missing"]
    selected = [record["selected_metrics"] for record in scored]
    abs_errors = [float(metric["pin_count_abs_error"]) for metric in selected]
    sq_errors = [float(metric["pin_count_sq_error"]) for metric in selected]
    weighted_scores = [
        float(record["weighted_score"])
        for record in scored
        if isinstance(record.get("weighted_score"), (int, float))
    ]
    return {
        "merge_root": str(merge_root),
        "dataset_root": str(dataset_root),
        "table_missing_graph_root": str(table_missing_graph_root) if table_missing_graph_root else "",
        "review_gallery_path": str(review_dir / "index.html") if review_dir is not None else "",
        "metric_source": {
            "paper": "A Large Language Model Powered Integrated Circuit Footprint Geometry Understanding",
            "arxiv": "2508.03725v1",
            "metrics": ["IoU_IC", "Task1_MAE", "Task1_RMSE", "Task2_d_pin", "Task3_IoU_pin"],
        },
        "weighted_score_policy": {
            "weights": weights.to_dict(),
            "subscores": {
                "iou_ic": "IoU_IC",
                "pin_count": "1 - abs(pred_count - gt_count) / max(pred_count, gt_count, 1)",
                "d_pin": "1 - d_pin / GT_pin_layout_diagonal",
                "iou_pin": "IoU_pin",
            },
        },
        "coordinate_policy": (
            "merge bboxes are evaluation-only axis-aligned scaled/transformed into the GT pin-layout frame; "
            "formal merge JSON is not modified"
        ),
        "inner_land_policy": "per metric, include inner_land_pad only when that metric improves",
        "part_count": len(records),
        "scored_part_count": len(scored),
        "excluded_table_lookup_missing_part_count": len(excluded),
        "excluded_table_lookup_missing_parts_sample": sorted(str(record.get("part_number") or "") for record in excluded)[:20],
        "unscored_part_count": len(records) - len(scored) - len(excluded),
        "inner_land_included_part_count": sum(1 for record in scored if record.get("inner_land_included")),
        "inner_land_used_by_metric_counts": {
            key: sum(1 for record in scored if (record.get("inner_land_metric_selection") or {}).get(key))
            for key in ("iou_ic", "pin_count", "d_pin", "iou_pin")
        },
        "weighted_score": mean_values(weighted_scores),
        "weighted_score_std": std_values(weighted_scores),
        "IoU_IC": mean_metric(selected, "iou_ic"),
        "IoU_IC_std": std_metric(selected, "iou_ic"),
        "Task1_MAE": sum(abs_errors) / len(abs_errors) if abs_errors else None,
        "Task1_RMSE": math.sqrt(sum(sq_errors) / len(sq_errors)) if sq_errors else None,
        "Task1_abs_error_std": std_values(abs_errors),
        "Task2_d_pin": mean_metric(selected, "d_pin"),
        "Task2_d_pin_std": std_metric(selected, "d_pin"),
        "Task3_IoU_pin": mean_metric(selected, "iou_pin"),
        "Task3_IoU_pin_std": std_metric(selected, "iou_pin"),
        "matched_pin_count": sum(int(metric.get("matched_pin_count") or 0) for metric in selected),
        "pred_pin_count": sum(int(metric.get("pred_pin_count") or 0) for metric in selected),
        "gt_pin_count": sum(int(metric.get("gt_pin_count") or 0) for metric in selected),
    }


def write_score_gallery(review_dir: Path, records: list[dict[str, Any]], *, summary: dict[str, Any]) -> Path:
    review_dir.mkdir(parents=True, exist_ok=True)
    scored = [record for record in records if record.get("status") == "ok"]
    scored = sorted(scored, key=lambda record: (float(record.get("weighted_score") or 0.0), str(record.get("part_number") or "")))
    index_path = review_dir / "index.html"
    css = """
body { margin: 0; font-family: Arial, sans-serif; color: #111827; background: #f8fafc; }
header { position: sticky; top: 0; z-index: 2; padding: 16px 20px; background: #fff; border-bottom: 1px solid #d1d5db; }
h1 { margin: 0 0 8px; font-size: 22px; }
.meta { color: #475569; font-size: 13px; line-height: 1.45; }
.case { margin: 16px 20px; padding: 14px; background: #fff; border: 1px solid #d1d5db; border-radius: 6px; }
.case h2 { margin: 0 0 10px; font-size: 18px; }
.metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 8px; margin-bottom: 12px; }
.metric { padding: 8px; background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 4px; }
.metric span { display: block; color: #64748b; font-size: 12px; }
.metric strong { display: block; font-size: 16px; margin-top: 3px; }
.media-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 12px; }
.panel { border: 1px solid #e5e7eb; border-radius: 6px; overflow: hidden; background: #fff; }
.panel h3 { margin: 0; padding: 8px 10px; font-size: 13px; background: #f1f5f9; border-bottom: 1px solid #e5e7eb; }
.media { padding: 8px; }
.media img { width: 100%; max-height: 520px; object-fit: contain; border: 1px solid #e5e7eb; background: transparent; }
a { color: #2563eb; }
code { background: #f1f5f9; padding: 2px 4px; border-radius: 3px; }
"""
    lines = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Merge GT Score Gallery</title>",
        f"<style>{css}</style>",
        "</head>",
        "<body>",
        "<header>",
        "<h1>Merge GT Score Gallery</h1>",
        f'<div class="meta">sorted low to high weighted score | scored: {summary.get("scored_part_count")} | excluded table_lookup_missing: {summary.get("excluded_table_lookup_missing_part_count")}</div>',
        f'<div class="meta">IoUIC: {format_metric(summary.get("IoU_IC"))} | Task1 MAE: {format_metric(summary.get("Task1_MAE"))} | Task2 dpin: {format_metric(summary.get("Task2_d_pin"))} | Task3 IoUpin: {format_metric(summary.get("Task3_IoU_pin"))}</div>',
        f'<div class="meta">weights: <code>{html.escape(json.dumps(summary.get("weighted_score_policy", {}).get("weights", {}), ensure_ascii=False))}</code></div>',
        "</header>",
    ]
    for index, record in enumerate(scored, start=1):
        lines.extend(render_score_case(index, record, html_dir=review_dir))
    lines.extend(["</body>", "</html>"])
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index_path


def render_score_case(index: int, record: dict[str, Any], *, html_dir: Path) -> list[str]:
    part = str(record.get("part_number") or "")
    metrics = record.get("selected_metrics") or {}
    scores = record.get("metric_scores") or {}
    merge_path = Path(str(record.get("merge_path") or ""))
    part_dir = merge_path.parent
    media = [
        ("Merge result", part_dir / "mergy_pad.svg"),
        ("Alignment / GT reference", part_dir / "multiview_alignment.svg"),
        ("All-view overlay", part_dir / "package_graph_multi_view_overlay.svg"),
    ]
    media_panels = []
    for title, path in media:
        if path.is_file():
            rel = relative_href(path, html_dir)
            media_panels.append(
                '<div class="panel">'
                f"<h3>{escape(title)}</h3>"
                '<div class="media">'
                f'<a href="{escape(rel)}"><img src="{escape(rel)}" loading="lazy"></a>'
                "</div></div>"
            )
    part_index = part_dir / "index.html"
    part_link = f' | <a href="{escape(relative_href(part_index, html_dir))}">part page</a>' if part_index.is_file() else ""
    return [
        '<section class="case">',
        f"<h2>#{index} {escape(part)} - weighted score {format_metric(record.get('weighted_score'))}{part_link}</h2>",
        '<div class="metrics">',
        metric_box("IoUIC", metrics.get("iou_ic"), scores.get("iou_ic")),
        metric_box("Task1 count", f"{metrics.get('pred_pin_count')} / {metrics.get('gt_pin_count')}", scores.get("pin_count")),
        metric_box("Task2 dpin", metrics.get("d_pin"), scores.get("d_pin")),
        metric_box("Task3 IoUpin", metrics.get("iou_pin"), scores.get("iou_pin")),
        metric_box("matched pins", metrics.get("matched_pin_count"), None),
        "</div>",
        '<div class="media-row">',
        *media_panels,
        "</div>",
        "</section>",
    ]


def metric_box(label: str, value: Any, score: Any) -> str:
    score_text = "" if score is None else f"score {format_metric(score)}"
    return (
        '<div class="metric">'
        f"<span>{escape(label)} {escape(score_text)}</span>"
        f"<strong>{escape(format_metric(value))}</strong>"
        "</div>"
    )


def format_metric(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    if value is None:
        return "None"
    return str(value)


def relative_href(path: Path, html_dir: Path) -> str:
    return os.path.relpath(path.resolve(), start=html_dir.resolve())


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def mean_metric(metrics: list[dict[str, Any]], key: str) -> float | None:
    values = [float(metric[key]) for metric in metrics if isinstance(metric.get(key), (int, float))]
    return mean_values(values)


def std_metric(metrics: list[dict[str, Any]], key: str) -> float | None:
    values = [float(metric[key]) for metric in metrics if isinstance(metric.get(key), (int, float))]
    return std_values(values)


def mean_values(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def std_values(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def match_boxes_by_center(pred_boxes: list[list[float]], gt_boxes: list[list[float]]) -> list[tuple[int, int]]:
    candidates = []
    for pred_index, pred_box in enumerate(pred_boxes):
        for gt_index, gt_box in enumerate(gt_boxes):
            candidates.append((center_distance(pred_box, gt_box), pred_index, gt_index))
    pairs = []
    used_pred = set()
    used_gt = set()
    for _distance, pred_index, gt_index in sorted(candidates):
        if pred_index in used_pred or gt_index in used_gt:
            continue
        used_pred.add(pred_index)
        used_gt.add(gt_index)
        pairs.append((pred_index, gt_index))
    return pairs


def union_iou(a_boxes: list[list[float]], b_boxes: list[list[float]]) -> float | None:
    if not a_boxes or not b_boxes:
        return None
    area_a = rectangle_union_area(a_boxes)
    area_b = rectangle_union_area(b_boxes)
    intersections = []
    for a in a_boxes:
        for b in b_boxes:
            box = intersection_bbox(a, b)
            if box is not None:
                intersections.append(box)
    intersection = rectangle_union_area(intersections)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return None
    return intersection / union


def rectangle_union_area(boxes: list[list[float]]) -> float:
    valid = [normalize_box(box) for box in boxes if normalize_box(box) is not None]
    if not valid:
        return 0.0
    xs = sorted({x for box in valid for x in (box[0], box[2])})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        if right <= left:
            continue
        intervals = []
        for x1, y1, x2, y2 in valid:
            if x1 < right and x2 > left:
                intervals.append((y1, y2))
        area += (right - left) * union_interval_length(intervals)
    return area


def union_interval_length(intervals: list[tuple[float, float]]) -> float:
    total = 0.0
    current_start = None
    current_end = None
    for start, end in sorted(intervals):
        if current_start is None:
            current_start, current_end = start, end
            continue
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_start is not None:
        total += current_end - current_start
    return total


def bbox_iou(a: list[float], b: list[float]) -> float:
    intersection = intersection_bbox(a, b)
    intersection_area = bbox_area(intersection) if intersection else 0.0
    union = bbox_area(a) + bbox_area(b) - intersection_area
    return intersection_area / union if union > 0.0 else 0.0


def intersection_bbox(a: list[float], b: list[float]) -> list[float] | None:
    a_norm = normalize_box(a)
    b_norm = normalize_box(b)
    if a_norm is None or b_norm is None:
        return None
    x1 = max(a_norm[0], b_norm[0])
    y1 = max(a_norm[1], b_norm[1])
    x2 = min(a_norm[2], b_norm[2])
    y2 = min(a_norm[3], b_norm[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def union_bbox(boxes: list[list[float]]) -> list[float] | None:
    valid = [normalize_box(box) for box in boxes if normalize_box(box) is not None]
    if not valid:
        return None
    return [
        min(box[0] for box in valid),
        min(box[1] for box in valid),
        max(box[2] for box in valid),
        max(box[3] for box in valid),
    ]


def object_bbox(obj: dict[str, Any]) -> list[float] | None:
    return normalize_box(obj.get("bbox") or [])


def normalize_box(box: list[float] | tuple[float, ...]) -> list[float] | None:
    if len(box) < 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def bbox_area(box: list[float] | None) -> float:
    if box is None:
        return 0.0
    norm = normalize_box(box)
    if norm is None:
        return 0.0
    return (norm[2] - norm[0]) * (norm[3] - norm[1])


def bbox_diagonal(box: list[float] | None) -> float:
    norm = normalize_box(box or [])
    if norm is None:
        return 0.0
    return math.hypot(norm[2] - norm[0], norm[3] - norm[1])


def center_distance(a: list[float], b: list[float]) -> float:
    ax = (a[0] + a[2]) / 2.0
    ay = (a[1] + a[3]) / 2.0
    bx = (b[0] + b[2]) / 2.0
    by = (b[1] + b[3]) / 2.0
    return math.hypot(ax - bx, ay - by)


if __name__ == "__main__":
    main()
