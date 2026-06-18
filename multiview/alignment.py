#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from real_image_process.FPK_PJ_fullflow.gt_alignment.evaluator import parse_scan_result


RESULT_COLORS = {
    "outline_2d": "#0284c7",
    "package_pad": "#2563eb",
    "land_pad": "#16a34a",
    "lead_pad": "#dc2626",
    "partial_pad_width": "#dc2626",
    "partial_lead_pad_length": "#dc2626",
    "inner_land_pad": "#7c3aed",
    "unknown": "#64748b",
}
GT_COLORS = {
    "shape": "#dc2626",
    "land": "#15803d",
    "lead": "#b45309",
    "outline_or_line": "#dc2626",
    "unknown": "#64748b",
}
ALIGNABLE_OUTLINE_VIEWS = {"top", "bottom"}
OUTLINE_EDGE_PAD_ROLES = {"package_pad", "lead_pad", "partial_pad_width", "partial_lead_pad_length"}
# Coordinates here are dimension_scaled_centered units. Edge-lock uses a small
# outline-relative tolerance so exact graph contacts and minor reconstruction
# jitter are treated the same without moving interior pads.
OUTLINE_EDGE_TOLERANCE_RATIO = 0.015


@dataclass(frozen=True)
class FrameTransform:
    source: tuple[float, float, float, float]
    target: tuple[float, float, float, float]
    scale: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a standalone multiview alignment review artifact. This stage groups multiview "
            "pad/outline evidence non-destructively and optionally draws ScanResultFormat GT as "
            "a review-only reference layer."
        )
    )
    parser.add_argument("--unified-layers", type=Path, required=True)
    parser.add_argument("--scan-result", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--part", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_multiview_alignment(
        unified_layers_path=args.unified_layers,
        scan_result_path=args.scan_result,
        output_dir=args.output_dir,
        part=args.part,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_multiview_alignment(
    *,
    unified_layers_path: Path,
    scan_result_path: Path | None,
    output_dir: Path,
    part: str = "",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(unified_layers_path.read_text(encoding="utf-8"))
    part_number = part or str(payload.get("part_number") or unified_layers_path.parent.name)
    result_objects = multiview_result_objects(payload)
    alignment_objects, excluded_alignment_objects = filter_alignment_objects(result_objects)
    outline_alignment = build_outline_alignment(alignment_objects, payload.get("dimensions") or [])
    outline_aligned_objects = apply_outline_alignment(alignment_objects, outline_alignment)
    top_bottom_package_pad_alignment = build_top_bottom_package_pad_alignment(
        outline_aligned_objects,
        payload.get("dimensions") or [],
    )
    package_aligned_objects = apply_top_bottom_package_pad_alignment(
        outline_aligned_objects,
        top_bottom_package_pad_alignment,
    )
    pad_center_alignment = build_pad_center_alignment(package_aligned_objects)
    aligned_result_objects = apply_pad_center_alignment(package_aligned_objects, pad_center_alignment)
    land_package_pair_diagnostics = build_land_package_pair_diagnostics(aligned_result_objects, pad_center_alignment)
    pad_stacks = build_pad_stacks(aligned_result_objects)
    gt_objects = load_gt_objects(scan_result_path)
    alignment = {
        "part_number": part_number,
        "coordinate_policy": (
            "non_destructive_multiview_alignment_review; result coordinates are unified "
            "dimension_scaled_centered units; GT is drawn in its own reference panel only"
        ),
        "inputs": {
            "unified_layers": str(unified_layers_path),
            "scan_result": str(scan_result_path) if scan_result_path else "",
        },
        "result_object_count": len(result_objects),
        "alignment_object_count": len(alignment_objects),
        "excluded_alignment_object_count": len(excluded_alignment_objects),
        "excluded_alignment_objects": excluded_alignment_objects,
        "gt_object_count": len(gt_objects),
        "outline_alignment": outline_alignment,
        "top_bottom_package_pad_alignment": top_bottom_package_pad_alignment,
        "pad_center_alignment": pad_center_alignment,
        "land_package_pair_diagnostics": land_package_pair_diagnostics,
        "pad_stack_count": len(pad_stacks),
        "pad_stacks": pad_stacks,
    }
    json_path = output_dir / "multiview_alignment.json"
    svg_path = output_dir / "multiview_alignment.svg"
    aligned_layers_path = output_dir / "aligned_multiview_layers.json"
    aligned_layers = aligned_multiview_payload(
        part_number=part_number,
        unified_layers_path=unified_layers_path,
        result_objects=aligned_result_objects,
        excluded_alignment_objects=excluded_alignment_objects,
        outline_alignment=outline_alignment,
        top_bottom_package_pad_alignment=top_bottom_package_pad_alignment,
        pad_center_alignment=pad_center_alignment,
        pad_stacks=pad_stacks,
    )
    json_path.write_text(json.dumps(alignment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    aligned_layers_path.write_text(json.dumps(aligned_layers, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_multiview_alignment_svg(
        svg_path,
        part_number=part_number,
        result_objects=aligned_result_objects,
        pad_stacks=pad_stacks,
        land_package_pair_diagnostics=land_package_pair_diagnostics,
        gt_objects=gt_objects,
    )
    return {
        "part": part_number,
        "alignment_json": str(json_path),
        "alignment_svg": str(svg_path),
        "aligned_multiview_layers_json": str(aligned_layers_path),
        "result_object_count": len(result_objects),
        "alignment_object_count": len(alignment_objects),
        "excluded_alignment_object_count": len(excluded_alignment_objects),
        "gt_object_count": len(gt_objects),
        "pad_stack_count": len(pad_stacks),
    }


def aligned_multiview_payload(
    *,
    part_number: str,
    unified_layers_path: Path,
    result_objects: list[dict[str, Any]],
    excluded_alignment_objects: list[dict[str, Any]],
    outline_alignment: dict[str, Any],
    top_bottom_package_pad_alignment: dict[str, Any],
    pad_center_alignment: dict[str, Any],
    pad_stacks: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "part_number": part_number,
        "coordinate_mode": "dimension_scaled_centered",
        "coordinate_policy": (
            "alignment_stage_applied_coordinates; bboxes are unified multiview coordinates "
            "after outline_alignment, top_bottom_package_pad_alignment, and pad_center_alignment; "
            "display scale is not applied"
        ),
        "inputs": {
            "unified_layers": str(unified_layers_path),
        },
        "alignment_steps": {
            "outline_alignment": alignment_step_summary(outline_alignment),
            "top_bottom_package_pad_alignment": alignment_step_summary(top_bottom_package_pad_alignment),
            "pad_center_alignment": alignment_step_summary(pad_center_alignment),
        },
        "object_count": len(result_objects),
        "objects": [aligned_multiview_object(obj) for obj in result_objects],
        "excluded_alignment_object_count": len(excluded_alignment_objects),
        "excluded_alignment_objects": excluded_alignment_objects,
        "pad_stack_count": len(pad_stacks),
        "pad_stacks": pad_stacks,
    }


def alignment_step_summary(step: dict[str, Any]) -> dict[str, Any]:
    adjustments = step.get("object_adjustments") or {}
    result = {
        "status": step.get("status"),
        "skip_reason": step.get("skip_reason", ""),
        "object_adjustment_count": len(adjustments),
    }
    for key in (
        "matched_side_count",
        "matched_pair_count",
        "anchor_role",
        "strategy",
        "main_outline_selection_reason",
        "anchor_view",
    ):
        if key in step:
            result[key] = step.get(key)
    return result


def aligned_multiview_object(obj: dict[str, Any]) -> dict[str, Any]:
    result = {
        "source_object_id": obj.get("source_object_id"),
        "role": obj.get("role"),
        "label": obj.get("label"),
        "source_label": obj.get("source_label"),
        "shape_family": obj.get("shape_family"),
        "bbox": list(obj.get("bbox") or []),
        "raw_view": obj.get("raw_view"),
        "canonical_view": obj.get("canonical_view"),
        "source_graph": obj.get("source_graph"),
        "source_package_pad_id": obj.get("source_package_pad_id"),
        "source_land_pad_id": obj.get("source_land_pad_id"),
        "source_quality": obj.get("source_quality"),
        "dimension_refs": obj.get("dimension_refs") or [],
    }
    for key in (
        "bbox_before_outline_adjust",
        "bbox_after_outline_adjust",
        "outline_edge_lock",
        "outline_adjustment_type",
        "bbox_before_top_bottom_package_pad_align",
        "bbox_after_top_bottom_package_pad_align",
        "top_bottom_package_pad_alignment_anchor_view",
        "top_bottom_package_pad_alignment_type",
        "bbox_before_pad_center_align",
        "bbox_after_pad_center_align",
        "pad_center_alignment_side",
        "pad_center_alignment_axis",
        "pad_center_alignment_anchor_role",
        "pad_center_alignment_type",
        "matched_package_object_id",
        "matched_land_object_id",
    ):
        if key in obj:
            result[key] = obj[key]
    return result


def multiview_result_objects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    overlay = payload.get("multiview_overlay") or {}
    formal_source_keys = formal_alignment_source_keys(payload)
    objects = []
    for layer in overlay.get("layers") or []:
        for obj in layer.get("objects") or []:
            item = result_object(obj, source_layer=layer)
            if item and formal_alignment_object_allowed(item, formal_source_keys):
                objects.append(item)
    for obj in overlay.get("extra_objects") or []:
        item = result_object(obj, source_layer=obj.get("source_overlay_layer") or {})
        if item and formal_alignment_object_allowed(item, formal_source_keys):
            objects.append(item)
    return sorted(objects, key=lambda obj: (bbox_center(obj["bbox"])[1], bbox_center(obj["bbox"])[0], obj["role"]))


def formal_alignment_source_keys(payload: dict[str, Any]) -> dict[str, set[tuple[str, str]]]:
    """Build source keys for formal multiview graph objects.

    Coordinates are not read from these objects here; the alignment stage keeps
    overlay-normalized coordinates but filters out evidence that the formal
    unified multiview JSON has already rejected.
    """
    keys: dict[str, set[tuple[str, str]]] = {}
    add_formal_role_keys(keys, "outline_2d", [payload.get("outline_2d") or {}])
    add_formal_role_keys(keys, "package_pad", payload.get("package_pads") or [])
    add_formal_role_keys(keys, "land_pad", payload.get("land_pads") or [])
    add_formal_role_keys(keys, "inner_land_pad", payload.get("inner_land_pads") or [])
    for obj in payload.get("lead_pads") or []:
        role = str(obj.get("role") or "lead_pad")
        if role in RESULT_COLORS:
            add_formal_role_keys(keys, role, [obj])
    return keys


def add_formal_role_keys(keys: dict[str, set[tuple[str, str]]], role: str, objects: list[dict[str, Any]]) -> None:
    for obj in objects:
        key = formal_source_key(obj)
        if key is None:
            continue
        keys.setdefault(role, set()).add(key)


def formal_alignment_object_allowed(obj: dict[str, Any], formal_source_keys: dict[str, set[tuple[str, str]]]) -> bool:
    role = str(obj.get("role") or "")
    allowed_keys = formal_source_keys.get(role)
    if not allowed_keys:
        return True
    key = formal_source_key(obj)
    return key in allowed_keys


def formal_source_key(obj: dict[str, Any]) -> tuple[str, str] | None:
    source_object_id = obj.get("source_object_id")
    if source_object_id is None:
        return None
    return (str(obj.get("source_graph") or ""), json.dumps(source_object_id, sort_keys=True))


def result_object(obj: dict[str, Any], *, source_layer: dict[str, Any]) -> dict[str, Any] | None:
    bbox = object_bbox(obj)
    if bbox is None:
        return None
    role = str(obj.get("role") or "unknown")
    return {
        "source_object_id": obj.get("source_object_id"),
        "role": role,
        "label": str(obj.get("label") or ""),
        "source_label": str(obj.get("source_label") or ""),
        "shape_family": shape_family(obj),
        "bbox": list(bbox),
        "raw_view": str(obj.get("raw_view") or source_layer.get("raw_view") or ""),
        "canonical_view": str(obj.get("canonical_view") or source_layer.get("canonical_view") or ""),
        "source_graph": str(obj.get("source_graph") or source_layer.get("graph_path") or ""),
        "source_package_pad_id": obj.get("source_package_pad_id"),
        "source_land_pad_id": obj.get("source_land_pad_id"),
        "source_quality": source_quality(obj),
        "dimension_refs": dimension_refs(obj),
    }


def filter_alignment_objects(objects: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    included = []
    excluded = []
    for index, obj in enumerate(objects):
        if is_land_outline(obj):
            excluded.append(
                {
                    "source_object_index": index,
                    "source_object_id": obj.get("source_object_id"),
                    "role": obj.get("role"),
                    "raw_view": obj.get("raw_view"),
                    "canonical_view": obj.get("canonical_view"),
                    "source_graph": obj.get("source_graph"),
                    "bbox": obj.get("bbox"),
                    "excluded_reason": "land_outline_excluded_from_alignment_stage",
                }
            )
            continue
        included.append(obj)
    return included, excluded


def is_land_outline(obj: dict[str, Any]) -> bool:
    if str(obj.get("role") or "") != "outline_2d":
        return False
    view = str(obj.get("canonical_view") or obj.get("raw_view") or "").strip().lower()
    return view == "land"


def build_outline_alignment(objects: list[dict[str, Any]], dimensions: list[dict[str, Any]]) -> dict[str, Any]:
    indexed_outlines = [
        (index, obj)
        for index, obj in enumerate(objects)
        if str(obj.get("role") or "") == "outline_2d" and len(obj.get("bbox") or []) >= 4
    ]
    if not indexed_outlines:
        return {
            "status": "no_outline",
            "main_outline": None,
            "main_outline_selection_reason": "no_outline_2d_objects",
            "outline_count": 0,
            "outline_alignment": [],
            "object_adjustments": {},
        }

    dimension_refs_by_key = outline_dimension_refs_by_key(dimensions)
    candidates = [(index, obj) for index, obj in indexed_outlines if is_alignable_outline_view(obj)]
    if not candidates:
        candidates = indexed_outlines
    scored = [
        (
            outline_selection_score(obj, dimension_refs_by_key.get(source_object_key(obj), [])),
            index,
            obj,
            dimension_refs_by_key.get(source_object_key(obj), []),
        )
        for index, obj in candidates
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    _score, main_index, main_outline, main_dimension_refs = scored[0]
    main_bbox = list(main_outline["bbox"])
    object_adjustments: dict[str, dict[str, Any]] = {}
    alignments = []

    for source_index, source_outline in indexed_outlines:
        if source_index == main_index:
            continue
        if not is_alignable_outline_view(source_outline):
            continue
        source_bbox = list(source_outline["bbox"])
        outline_adjustment = {
            "source_object_index": source_index,
            "source_object_id": source_outline.get("source_object_id"),
            "role": source_outline.get("role"),
            "raw_view": source_outline.get("raw_view"),
            "canonical_view": source_outline.get("canonical_view"),
            "source_graph": source_outline.get("source_graph"),
            "bbox_before_outline_adjust": source_bbox,
            "bbox_after_outline_adjust": main_bbox,
            "outline_edge_lock": ["outline_frame"],
            "adjustment_type": "outline_to_main_outline",
        }
        object_adjustments[str(source_index)] = outline_adjustment
        edge_adjusted_pad_count = 0
        unchanged_pad_count = 0
        skipped_pad_count = 0
        tolerance = outline_edge_tolerance(source_bbox)
        for object_index, obj in enumerate(objects):
            if object_index == source_index:
                continue
            if not is_outline_adjustable_pad(obj):
                if is_same_outline_source(obj, source_outline):
                    skipped_pad_count += 1
                continue
            if not is_same_outline_source(obj, source_outline):
                continue
            locks = locked_outline_edges(list(obj["bbox"]), source_bbox, tolerance=tolerance)
            if not locks:
                unchanged_pad_count += 1
                continue
            adjusted_bbox = adjust_bbox_to_outline_edges(list(obj["bbox"]), locks, main_bbox)
            object_adjustments[str(object_index)] = {
                "source_object_index": object_index,
                "source_object_id": obj.get("source_object_id"),
                "role": obj.get("role"),
                "raw_view": obj.get("raw_view"),
                "canonical_view": obj.get("canonical_view"),
                "source_graph": obj.get("source_graph"),
                "bbox_before_outline_adjust": list(obj["bbox"]),
                "bbox_after_outline_adjust": adjusted_bbox,
                "outline_edge_lock": locks,
                "adjustment_type": "edge_locked_pad_to_main_outline",
            }
            edge_adjusted_pad_count += 1
        alignments.append(
            {
                "source_outline": outline_summary(source_index, source_outline, dimension_refs_by_key.get(source_object_key(source_outline), [])),
                "target_outline": outline_summary(main_index, main_outline, main_dimension_refs),
                "bbox_before_outline_adjust": source_bbox,
                "bbox_after_outline_adjust": main_bbox,
                "edge_tolerance": tolerance,
                "edge_adjusted_pad_count": edge_adjusted_pad_count,
                "unchanged_pad_count": unchanged_pad_count,
                "skipped_pad_count": skipped_pad_count,
            }
        )

    return {
        "status": "ok",
        "main_outline": outline_summary(main_index, main_outline, main_dimension_refs),
        "main_outline_selection_reason": main_outline_selection_reason(main_outline, main_dimension_refs),
        "outline_count": len(indexed_outlines),
        "aligned_outline_count": len(alignments),
        "outline_alignment": alignments,
        "object_adjustments": object_adjustments,
    }


def apply_outline_alignment(objects: list[dict[str, Any]], outline_alignment: dict[str, Any]) -> list[dict[str, Any]]:
    adjustments = outline_alignment.get("object_adjustments") or {}
    aligned = []
    for index, obj in enumerate(objects):
        item = dict(obj)
        adjustment = adjustments.get(str(index))
        if adjustment:
            item["bbox_before_outline_adjust"] = list(adjustment["bbox_before_outline_adjust"])
            item["bbox"] = list(adjustment["bbox_after_outline_adjust"])
            item["bbox_after_outline_adjust"] = list(adjustment["bbox_after_outline_adjust"])
            item["outline_edge_lock"] = list(adjustment.get("outline_edge_lock") or [])
            item["outline_adjustment_type"] = str(adjustment.get("adjustment_type") or "")
        aligned.append(item)
    return aligned


def build_top_bottom_package_pad_alignment(objects: list[dict[str, Any]], dimensions: list[dict[str, Any]]) -> dict[str, Any]:
    package_items = [
        item
        for item in indexed_pad_items(objects, role="package_pad")
        if normalized_view(item["object"]) in {"top", "bottom"}
    ]
    by_view = {
        "top": [item for item in package_items if normalized_view(item["object"]) == "top"],
        "bottom": [item for item in package_items if normalized_view(item["object"]) == "bottom"],
    }
    if not by_view["top"] or not by_view["bottom"]:
        return top_bottom_package_pad_alignment_skip(by_view, "missing_top_or_bottom_package_pads")
    anchor_view = top_bottom_package_anchor_view(dimensions)
    moving_view = "bottom" if anchor_view == "top" else "top"
    anchor_terminals = terminal_pad_items_for_alignment(by_view[anchor_view])
    moving_terminals = terminal_pad_items_for_alignment(by_view[moving_view])
    if len(anchor_terminals) != len(moving_terminals):
        return top_bottom_package_pad_alignment_skip(
            by_view,
            "terminal_package_pad_count_mismatch",
            anchor_view=anchor_view,
        )
    if len(anchor_terminals) < 2:
        return top_bottom_package_pad_alignment_skip(by_view, "insufficient_terminal_package_pad_count", anchor_view=anchor_view)
    pairs = package_pad_side_pairs(anchor_terminals, moving_terminals)
    if len(pairs) != len(anchor_terminals):
        return top_bottom_package_pad_alignment_skip(by_view, "package_pad_pairing_incomplete", anchor_view=anchor_view)

    object_adjustments: dict[str, dict[str, Any]] = {}
    pair_payload = []
    deltas = []
    for pair in pairs:
        moving_item = pair["moving_item"]
        anchor_item = pair["anchor_item"]
        moving_center = bbox_center(moving_item["bbox"])
        anchor_center = bbox_center(anchor_item["bbox"])
        delta_x = clean_float(anchor_center[0] - moving_center[0])
        delta_y = clean_float(anchor_center[1] - moving_center[1])
        adjusted_bbox = translate_bbox(moving_item["bbox"], delta_x=delta_x, delta_y=delta_y)
        object_adjustments[str(moving_item["index"])] = top_bottom_package_pad_adjustment(
            moving_item,
            anchor_item,
            adjusted_bbox,
            delta_x=delta_x,
            delta_y=delta_y,
            adjustment_type="package_pad_center_to_dimension_anchor_package_pad",
            anchor_view=anchor_view,
        )
        add_package_source_follow_adjustments_2d(
            objects,
            object_adjustments,
            package_item=moving_item,
            anchor_item=anchor_item,
            delta_x=delta_x,
            delta_y=delta_y,
            anchor_view=anchor_view,
        )
        deltas.append(math.hypot(delta_x, delta_y))
        pair_payload.append(
            {
                "anchor_package_object_index": anchor_item["index"],
                "anchor_package_object_id": anchor_item["object"].get("source_object_id"),
                "moving_package_object_index": moving_item["index"],
                "moving_package_object_id": moving_item["object"].get("source_object_id"),
                "delta_x": delta_x,
                "delta_y": delta_y,
                "center_distance_before": clean_float(pair["distance"]),
            }
        )
    return {
        "status": "ok",
        "skip_reason": "",
        "anchor_view": anchor_view,
        "moving_view": moving_view,
        "anchor_selection_reason": "pad_spacing_dimension_preferred",
        "top_package_pad_count": len(by_view["top"]),
        "bottom_package_pad_count": len(by_view["bottom"]),
        "matched_pair_count": len(pairs),
        "center_delta_min": clean_float(min(deltas)),
        "center_delta_max": clean_float(max(deltas)),
        "center_delta_mean": clean_float(sum(deltas) / len(deltas)),
        "pairs": pair_payload,
        "object_adjustments": object_adjustments,
    }


def apply_top_bottom_package_pad_alignment(
    objects: list[dict[str, Any]],
    alignment: dict[str, Any],
) -> list[dict[str, Any]]:
    adjustments = alignment.get("object_adjustments") or {}
    aligned = []
    for index, obj in enumerate(objects):
        item = dict(obj)
        adjustment = adjustments.get(str(index))
        if adjustment:
            item["bbox_before_top_bottom_package_pad_align"] = list(adjustment["bbox_before_top_bottom_package_pad_align"])
            item["bbox"] = list(adjustment["bbox_after_top_bottom_package_pad_align"])
            item["bbox_after_top_bottom_package_pad_align"] = list(adjustment["bbox_after_top_bottom_package_pad_align"])
            item["top_bottom_package_pad_alignment_anchor_view"] = str(adjustment.get("anchor_view") or "")
            item["top_bottom_package_pad_alignment_type"] = str(adjustment.get("adjustment_type") or "")
            item["matched_anchor_package_object_id"] = adjustment.get("matched_anchor_package_object_id")
        aligned.append(item)
    return aligned


def top_bottom_package_anchor_view(dimensions: list[dict[str, Any]]) -> str:
    scores = {"top": 0, "bottom": 0}
    for dimension in dimensions:
        if str(dimension.get("role") or "") != "pad_spacing":
            continue
        view = normalize_view(str(dimension.get("canonical_view") or dimension.get("raw_view") or ""))
        if view in scores:
            scores[view] += 1
    if scores["top"] >= scores["bottom"]:
        return "top"
    return "bottom"


def top_bottom_package_pad_alignment_skip(
    by_view: dict[str, list[dict[str, Any]]],
    reason: str,
    *,
    anchor_view: str = "",
) -> dict[str, Any]:
    return {
        "status": "skipped",
        "skip_reason": reason,
        "anchor_view": anchor_view,
        "top_package_pad_count": len(by_view.get("top") or []),
        "bottom_package_pad_count": len(by_view.get("bottom") or []),
        "matched_pair_count": 0,
        "pairs": [],
        "object_adjustments": {},
    }


def package_pad_side_pairs(anchor_items: list[dict[str, Any]], moving_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anchor_sides = classify_pad_sides(anchor_items)
    moving_sides = classify_pad_sides(moving_items)
    pairs = []
    for side in ("left", "right", "top", "bottom"):
        anchor_side = sorted(anchor_sides.get(side, []), key=lambda item: side_sort_key(item, side))
        moving_side = sorted(moving_sides.get(side, []), key=lambda item: side_sort_key(item, side))
        if len(anchor_side) != len(moving_side):
            return []
        for anchor_item, moving_item in zip(anchor_side, moving_side):
            ac = bbox_center(anchor_item["bbox"])
            mc = bbox_center(moving_item["bbox"])
            pairs.append(
                {
                    "anchor_item": anchor_item,
                    "moving_item": moving_item,
                    "side": side,
                    "distance": math.hypot(ac[0] - mc[0], ac[1] - mc[1]),
                }
            )
    return pairs


def top_bottom_package_pad_adjustment(
    moving_item: dict[str, Any],
    anchor_item: dict[str, Any],
    adjusted_bbox: list[float],
    *,
    delta_x: float,
    delta_y: float,
    adjustment_type: str,
    anchor_view: str,
) -> dict[str, Any]:
    moving_obj = moving_item["object"]
    anchor_obj = anchor_item["object"]
    return {
        "source_object_index": moving_item["index"],
        "source_object_id": moving_obj.get("source_object_id"),
        "role": moving_obj.get("role"),
        "raw_view": moving_obj.get("raw_view"),
        "canonical_view": moving_obj.get("canonical_view"),
        "source_graph": moving_obj.get("source_graph"),
        "bbox_before_top_bottom_package_pad_align": list(moving_item["bbox"]),
        "bbox_after_top_bottom_package_pad_align": adjusted_bbox,
        "delta_x": delta_x,
        "delta_y": delta_y,
        "anchor_view": anchor_view,
        "matched_anchor_package_object_index": anchor_item["index"],
        "matched_anchor_package_object_id": anchor_obj.get("source_object_id"),
        "adjustment_type": adjustment_type,
    }


def add_package_source_follow_adjustments_2d(
    objects: list[dict[str, Any]],
    object_adjustments: dict[str, dict[str, Any]],
    *,
    package_item: dict[str, Any],
    anchor_item: dict[str, Any],
    delta_x: float,
    delta_y: float,
    anchor_view: str,
) -> None:
    package_obj = package_item["object"]
    package_id = package_obj.get("source_object_id")
    source_graph = str(package_obj.get("source_graph") or "")
    if package_id is None or not source_graph:
        return
    for index, obj in enumerate(objects):
        if str(obj.get("role") or "") not in {"lead_pad", "partial_pad_width", "partial_lead_pad_length"}:
            continue
        if obj.get("source_package_pad_id") != package_id:
            continue
        if str(obj.get("source_graph") or "") != source_graph:
            continue
        bbox = list(obj.get("bbox") or [])
        if len(bbox) < 4:
            continue
        object_adjustments[str(index)] = top_bottom_package_pad_adjustment(
            {"index": index, "object": obj, "bbox": bbox},
            anchor_item,
            translate_bbox(bbox, delta_x=delta_x, delta_y=delta_y),
            delta_x=delta_x,
            delta_y=delta_y,
            adjustment_type="package_derived_pad_follow_top_bottom_package_pad_alignment",
            anchor_view=anchor_view,
        )


def build_pad_center_alignment(objects: list[dict[str, Any]]) -> dict[str, Any]:
    package_items = indexed_pad_items(objects, role="package_pad")
    land_items = indexed_pad_items(objects, role="land_pad")
    if not package_items or not land_items:
        return {
            "status": "skipped",
            "skip_reason": "missing_package_or_land_pads",
            "side_alignments": [],
            "object_adjustments": {},
        }

    package_sides = classify_pad_sides(package_items)
    land_sides = classify_pad_sides(land_items)
    anchor_role = pad_center_anchor_role(land_items)
    if anchor_role == "land_pad" and should_use_2d_land_center_alignment(land_items):
        center_alignment = build_circle_land_center_alignment(objects, package_items, land_items)
        if center_alignment["status"] == "ok":
            return center_alignment
    object_adjustments: dict[str, dict[str, Any]] = {}
    side_alignments = []
    for side in ("top", "bottom", "left", "right"):
        package_side = sorted(package_sides.get(side, []), key=lambda item: side_sort_key(item, side))
        land_side = sorted(land_sides.get(side, []), key=lambda item: side_sort_key(item, side))
        side_result = {
            "side": side,
            "package_pad_count": len(package_side),
            "land_pad_count": len(land_side),
            "matched_pair_count": 0,
            "status": "skipped",
            "skip_reason": "",
            "axis": "x" if side in {"top", "bottom"} else "y",
            "anchor_role": anchor_role,
            "center_delta_min": None,
            "center_delta_max": None,
            "center_delta_mean": None,
        }
        if not package_side and not land_side:
            side_result["skip_reason"] = "no_side_pads"
            side_alignments.append(side_result)
            continue
        if len(package_side) != len(land_side):
            side_result["skip_reason"] = "side_pad_count_mismatch"
            side_alignments.append(side_result)
            continue
        if len(package_side) < 2:
            side_result["skip_reason"] = "insufficient_side_pad_count"
            side_alignments.append(side_result)
            continue
        pitch_check = side_pitch_check(package_side, land_side, side)
        side_result["pitch_check"] = pitch_check
        if not pitch_check["matched"]:
            side_result["skip_reason"] = "side_pitch_mismatch"
            side_alignments.append(side_result)
            continue
        deltas = []
        pairs = []
        for package_item, land_item in zip(package_side, land_side):
            if anchor_role == "land_pad":
                adjusted_bbox, delta = adjust_pad_center(package_item["bbox"], land_item["bbox"], side)
                adjusted_item = package_item
                adjustment_type = "package_pad_center_to_land_pad_center"
            else:
                adjusted_bbox, delta = adjust_pad_center(land_item["bbox"], package_item["bbox"], side)
                adjusted_item = land_item
                adjustment_type = "land_pad_center_to_package_pad_center"
            deltas.append(delta)
            pairs.append(
                {
                    "package_object_index": package_item["index"],
                    "package_object_id": package_item["object"].get("source_object_id"),
                    "land_object_index": land_item["index"],
                    "land_object_id": land_item["object"].get("source_object_id"),
                    "center_delta": delta,
                }
            )
            object_adjustments[str(adjusted_item["index"])] = {
                "source_object_index": adjusted_item["index"],
                "source_object_id": adjusted_item["object"].get("source_object_id"),
                "role": adjusted_item["object"].get("role"),
                "raw_view": adjusted_item["object"].get("raw_view"),
                "canonical_view": adjusted_item["object"].get("canonical_view"),
                "source_graph": adjusted_item["object"].get("source_graph"),
                "bbox_before_pad_center_align": list(adjusted_item["bbox"]),
                "bbox_after_pad_center_align": adjusted_bbox,
                "pad_center_alignment_side": side,
                "pad_center_alignment_axis": side_result["axis"],
                "matched_package_object_index": package_item["index"],
                "matched_package_object_id": package_item["object"].get("source_object_id"),
                "matched_land_object_index": land_item["index"],
                "matched_land_object_id": land_item["object"].get("source_object_id"),
                "pad_center_alignment_anchor_role": anchor_role,
                "adjustment_type": adjustment_type,
            }
            if anchor_role == "land_pad":
                add_package_derived_pad_adjustments(
                    objects,
                    object_adjustments,
                    package_item=package_item,
                    land_item=land_item,
                    delta=delta,
                    side=side,
                    axis=side_result["axis"],
                )
            else:
                add_inner_land_pad_adjustments(
                    objects,
                    object_adjustments,
                    land_item=land_item,
                    delta=delta,
                    side=side,
                    axis=side_result["axis"],
                    package_item=package_item,
                )
        side_result.update(
            {
                "status": "matched",
                "skip_reason": "",
                "matched_pair_count": len(pairs),
                "center_delta_min": min(deltas),
                "center_delta_max": max(deltas),
                "center_delta_mean": sum(deltas) / len(deltas),
                "pairs": pairs,
            }
        )
        side_alignments.append(side_result)

    matched_side_count = sum(1 for item in side_alignments if item.get("status") == "matched")
    if matched_side_count == 1:
        return {
            "status": "skipped",
            "skip_reason": "insufficient_matched_side_count",
            "anchor_role": anchor_role,
            "matched_side_count": matched_side_count,
            "side_alignments": side_alignments,
            "object_adjustments": {},
        }
    return {
        "status": "ok" if matched_side_count else "skipped",
        "skip_reason": "" if matched_side_count else "no_matched_sides",
        "anchor_role": anchor_role,
        "matched_side_count": matched_side_count,
        "side_alignments": side_alignments,
        "object_adjustments": object_adjustments,
    }


def apply_pad_center_alignment(objects: list[dict[str, Any]], pad_center_alignment: dict[str, Any]) -> list[dict[str, Any]]:
    adjustments = pad_center_alignment.get("object_adjustments") or {}
    aligned = []
    for index, obj in enumerate(objects):
        item = dict(obj)
        adjustment = adjustments.get(str(index))
        if adjustment:
            item["bbox_before_pad_center_align"] = list(adjustment["bbox_before_pad_center_align"])
            item["bbox"] = list(adjustment["bbox_after_pad_center_align"])
            item["bbox_after_pad_center_align"] = list(adjustment["bbox_after_pad_center_align"])
            item["pad_center_alignment_side"] = str(adjustment.get("pad_center_alignment_side") or "")
            item["pad_center_alignment_axis"] = str(adjustment.get("pad_center_alignment_axis") or "")
            item["pad_center_alignment_anchor_role"] = str(adjustment.get("pad_center_alignment_anchor_role") or "")
            item["pad_center_alignment_type"] = str(adjustment.get("adjustment_type") or "")
            if adjustment.get("matched_package_object_id") is not None:
                item["matched_package_object_id"] = adjustment.get("matched_package_object_id")
            if adjustment.get("matched_land_object_id") is not None:
                item["matched_land_object_id"] = adjustment.get("matched_land_object_id")
        aligned.append(item)
    return aligned


def build_land_package_pair_diagnostics(
    objects: list[dict[str, Any]],
    pad_center_alignment: dict[str, Any],
) -> dict[str, Any]:
    """Report post-alignment land_pad/package_pad center residuals.

    Coordinates are dimension_scaled_centered units. For side-based alignment,
    only the declared alignment axis is a required match: x for top/bottom
    rows, y for left/right columns. For center_2d alignment, xy residual is the
    Euclidean distance between centers.
    """
    pairs = []
    by_index = {index: obj for index, obj in enumerate(objects)}
    for side_alignment in pad_center_alignment.get("side_alignments") or []:
        axis = str(side_alignment.get("axis") or "")
        side = str(side_alignment.get("side") or "")
        for pair in side_alignment.get("pairs") or []:
            package_index = pair.get("package_object_index")
            land_index = pair.get("land_object_index")
            package_obj = by_index.get(package_index)
            land_obj = by_index.get(land_index)
            if not package_obj or not land_obj:
                continue
            if package_obj.get("role") != "package_pad" or land_obj.get("role") != "land_pad":
                continue
            package_bbox = list(package_obj.get("bbox") or [])
            land_bbox = list(land_obj.get("bbox") or [])
            if len(package_bbox) < 4 or len(land_bbox) < 4:
                continue
            package_center = bbox_center(package_bbox)
            land_center = bbox_center(land_bbox)
            dx = land_center[0] - package_center[0]
            dy = land_center[1] - package_center[1]
            center_distance = math.hypot(dx, dy)
            if axis == "x":
                axis_residual = abs(dx)
            elif axis == "y":
                axis_residual = abs(dy)
            else:
                axis_residual = center_distance
            pairs.append(
                {
                    "side": side,
                    "axis": axis,
                    "package_object_index": package_index,
                    "package_object_id": package_obj.get("source_object_id"),
                    "land_object_index": land_index,
                    "land_object_id": land_obj.get("source_object_id"),
                    "package_center": [clean_float(package_center[0]), clean_float(package_center[1])],
                    "land_center": [clean_float(land_center[0]), clean_float(land_center[1])],
                    "delta_x": clean_float(dx),
                    "delta_y": clean_float(dy),
                    "axis_residual": clean_float(axis_residual),
                    "center_distance": clean_float(center_distance),
                }
            )
    axis_residuals = [float(pair["axis_residual"]) for pair in pairs]
    center_distances = [float(pair["center_distance"]) for pair in pairs]
    return {
        "pair_count": len(pairs),
        "max_axis_residual": clean_float(max(axis_residuals) if axis_residuals else 0.0),
        "mean_axis_residual": clean_float(sum(axis_residuals) / len(axis_residuals) if axis_residuals else 0.0),
        "max_center_distance": clean_float(max(center_distances) if center_distances else 0.0),
        "pairs": pairs,
    }


def indexed_pad_items(objects: list[dict[str, Any]], *, role: str) -> list[dict[str, Any]]:
    items = []
    for index, obj in enumerate(objects):
        if str(obj.get("role") or "") != role:
            continue
        bbox = obj.get("bbox") or []
        if len(bbox) < 4:
            continue
        items.append({"index": index, "object": obj, "bbox": list(bbox)})
    return items


def pad_center_anchor_role(land_items: list[dict[str, Any]]) -> str:
    terminal_items = terminal_pad_items_for_alignment(land_items)
    if not terminal_items:
        return "package_pad"
    return "land_pad"


def should_use_2d_land_center_alignment(land_items: list[dict[str, Any]]) -> bool:
    terminal_items = terminal_pad_items_for_alignment(land_items)
    if not terminal_items:
        return False
    circle_count = sum(1 for item in terminal_items if pad_shape_family(item["object"]) == "circle")
    return circle_count / len(terminal_items) >= 0.8


def build_circle_land_center_alignment(
    objects: list[dict[str, Any]],
    package_items: list[dict[str, Any]],
    land_items: list[dict[str, Any]],
) -> dict[str, Any]:
    package_terminals = terminal_pad_items_for_alignment(package_items)
    land_terminals = terminal_pad_items_for_alignment(land_items)
    package_match_items = package_items if len(package_items) == len(land_items) else package_terminals
    land_match_items = land_items if len(package_items) == len(land_items) else land_terminals
    if len(package_match_items) != len(land_match_items):
        return circle_land_center_alignment_skip(
            package_match_items,
            land_match_items,
            "terminal_pad_count_mismatch",
        )
    if len(package_match_items) < 2:
        return circle_land_center_alignment_skip(
            package_match_items,
            land_match_items,
            "insufficient_terminal_pad_count",
        )
    pitch = median_nearest_neighbor_distance([item["bbox"] for item in package_terminals])
    if pitch <= 1e-9:
        return circle_land_center_alignment_skip(package_match_items, land_match_items, "missing_package_pitch")
    pairs = greedy_center_pairs(package_match_items, land_match_items)
    if len(pairs) != len(package_match_items):
        return circle_land_center_alignment_skip(package_match_items, land_match_items, "center_pairing_incomplete")
    max_distance = max(pair["distance"] for pair in pairs)
    if max_distance > pitch * 0.35:
        result = circle_land_center_alignment_skip(package_terminals, land_terminals, "center_distance_too_large")
        result["max_center_distance"] = max_distance
        result["median_package_pitch"] = pitch
        return result

    object_adjustments: dict[str, dict[str, Any]] = {}
    pair_payload = []
    deltas = []
    for pair in pairs:
        package_item = pair["package_item"]
        land_item = pair["land_item"]
        package_center = bbox_center(package_item["bbox"])
        land_center = bbox_center(land_item["bbox"])
        delta_x = clean_float(land_center[0] - package_center[0])
        delta_y = clean_float(land_center[1] - package_center[1])
        adjusted_bbox = translate_bbox(package_item["bbox"], delta_x=delta_x, delta_y=delta_y)
        deltas.append(math.hypot(delta_x, delta_y))
        object_adjustments[str(package_item["index"])] = {
            "source_object_index": package_item["index"],
            "source_object_id": package_item["object"].get("source_object_id"),
            "role": package_item["object"].get("role"),
            "raw_view": package_item["object"].get("raw_view"),
            "canonical_view": package_item["object"].get("canonical_view"),
            "source_graph": package_item["object"].get("source_graph"),
            "bbox_before_pad_center_align": list(package_item["bbox"]),
            "bbox_after_pad_center_align": adjusted_bbox,
            "pad_center_alignment_side": "center_2d",
            "pad_center_alignment_axis": "xy",
            "matched_package_object_index": package_item["index"],
            "matched_package_object_id": package_item["object"].get("source_object_id"),
            "matched_land_object_index": land_item["index"],
            "matched_land_object_id": land_item["object"].get("source_object_id"),
            "pad_center_alignment_anchor_role": "land_pad",
            "adjustment_type": "package_pad_center_to_land_pad_center_2d",
        }
        add_package_derived_pad_adjustments_2d(
            objects,
            object_adjustments,
            package_item=package_item,
            land_item=land_item,
            delta_x=delta_x,
            delta_y=delta_y,
        )
        pair_payload.append(
            {
                "package_object_index": package_item["index"],
                "package_object_id": package_item["object"].get("source_object_id"),
                "land_object_index": land_item["index"],
                "land_object_id": land_item["object"].get("source_object_id"),
                "center_distance": clean_float(pair["distance"]),
                "delta_x": delta_x,
                "delta_y": delta_y,
            }
        )
    return {
        "status": "ok",
        "skip_reason": "",
        "anchor_role": "land_pad",
        "strategy": "circle_land_center_2d",
        "matched_side_count": 0,
        "matched_pair_count": len(pairs),
        "median_package_pitch": pitch,
        "max_center_distance": max_distance,
        "center_delta_min": min(deltas),
        "center_delta_max": max(deltas),
        "center_delta_mean": sum(deltas) / len(deltas),
        "side_alignments": [
            {
                "side": "center_2d",
                "status": "matched",
                "anchor_role": "land_pad",
                "package_pad_count": len(package_match_items),
                "land_pad_count": len(land_match_items),
                "matched_pair_count": len(pairs),
                "axis": "xy",
                "pairs": pair_payload,
            }
        ],
        "object_adjustments": object_adjustments,
    }


def circle_land_center_alignment_skip(
    package_terminals: list[dict[str, Any]],
    land_terminals: list[dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    return {
        "status": "skipped",
        "skip_reason": reason,
        "anchor_role": "land_pad",
        "strategy": "circle_land_center_2d",
        "matched_side_count": 0,
        "side_alignments": [
            {
                "side": "center_2d",
                "status": "skipped",
                "anchor_role": "land_pad",
                "package_pad_count": len(package_terminals),
                "land_pad_count": len(land_terminals),
                "matched_pair_count": 0,
                "axis": "xy",
                "skip_reason": reason,
            }
        ],
        "object_adjustments": {},
    }


def greedy_center_pairs(package_items: list[dict[str, Any]], land_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for package_index, package_item in enumerate(package_items):
        package_center = bbox_center(package_item["bbox"])
        for land_index, land_item in enumerate(land_items):
            land_center = bbox_center(land_item["bbox"])
            distance = math.hypot(package_center[0] - land_center[0], package_center[1] - land_center[1])
            candidates.append((distance, package_index, land_index))
    used_packages: set[int] = set()
    used_lands: set[int] = set()
    pairs = []
    for distance, package_index, land_index in sorted(candidates):
        if package_index in used_packages or land_index in used_lands:
            continue
        used_packages.add(package_index)
        used_lands.add(land_index)
        pairs.append(
            {
                "distance": distance,
                "package_item": package_items[package_index],
                "land_item": land_items[land_index],
            }
        )
    return pairs


def median_nearest_neighbor_distance(boxes: list[list[float]]) -> float:
    centers = [bbox_center(box) for box in boxes]
    distances = []
    for index, center in enumerate(centers):
        other_distances = [
            math.hypot(center[0] - other[0], center[1] - other[1])
            for other_index, other in enumerate(centers)
            if other_index != index
        ]
        if other_distances:
            distances.append(min(other_distances))
    return median_value(distances)


def terminal_pad_items_for_alignment(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not items:
        return []
    areas = [bbox_area(item["bbox"]) for item in items]
    median_area = median_value(areas)
    if median_area <= 0.0:
        return items
    return [item for item, area in zip(items, areas) if area <= median_area * 6.0]


def pad_shape_family(obj: dict[str, Any]) -> str:
    shape = str(obj.get("shape_family") or "").lower()
    if shape:
        return shape
    text = " ".join(str(obj.get(key) or "") for key in ("label", "source_label")).lower()
    if "circle" in text:
        return "circle"
    if "dshape" in text:
        return "dshape"
    return "rect"


def classify_pad_sides(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    if not items:
        return {"top": [], "bottom": [], "left": [], "right": []}
    centers = [bbox_center(item["bbox"]) for item in items]
    widths = [abs(item["bbox"][2] - item["bbox"][0]) for item in items]
    heights = [abs(item["bbox"][3] - item["bbox"][1]) for item in items]
    areas = [widths[index] * heights[index] for index in range(len(items))]
    median_area = median_value(areas)
    min_x = min(center[0] for center in centers)
    max_x = max(center[0] for center in centers)
    min_y = min(center[1] for center in centers)
    max_y = max(center[1] for center in centers)
    range_x = max(max_x - min_x, 1e-9)
    range_y = max(max_y - min_y, 1e-9)
    median_minor = median_value([min(width, height) for width, height in zip(widths, heights)])
    edge_band_x = max(range_x * 0.10, median_minor * 2.5)
    edge_band_y = max(range_y * 0.10, median_minor * 2.5)
    sides: dict[str, list[dict[str, Any]]] = {"top": [], "bottom": [], "left": [], "right": []}
    for index, item in enumerate(items):
        if median_area > 0.0 and areas[index] > median_area * 6.0:
            continue
        width = widths[index]
        height = heights[index]
        cx, cy = centers[index]
        side = ""
        if height >= width * 1.25:
            if cy - min_y <= edge_band_y:
                side = "top"
            elif max_y - cy <= edge_band_y:
                side = "bottom"
        elif width >= height * 1.25:
            if cx - min_x <= edge_band_x:
                side = "left"
            elif max_x - cx <= edge_band_x:
                side = "right"
        else:
            distances = {
                "top": cy - min_y,
                "bottom": max_y - cy,
                "left": cx - min_x,
                "right": max_x - cx,
            }
            side = min(distances, key=distances.get)
            if distances[side] > (edge_band_y if side in {"top", "bottom"} else edge_band_x):
                side = ""
        if side:
            sides[side].append(item)
    return sides


def side_sort_key(item: dict[str, Any], side: str) -> float:
    cx, cy = bbox_center(item["bbox"])
    return cx if side in {"top", "bottom"} else cy


def side_pitch_check(package_side: list[dict[str, Any]], land_side: list[dict[str, Any]], side: str) -> dict[str, Any]:
    package_values = [side_sort_key(item, side) for item in package_side]
    land_values = [side_sort_key(item, side) for item in land_side]
    package_gaps = normalized_gaps(package_values)
    land_gaps = normalized_gaps(land_values)
    if len(package_gaps) != len(land_gaps) or not package_gaps:
        return {"matched": False, "reason": "missing_pitch_gaps"}
    max_gap_delta = max(abs(a - b) for a, b in zip(package_gaps, land_gaps))
    return {
        "matched": max_gap_delta <= 0.18,
        "reason": "" if max_gap_delta <= 0.18 else "normalized_pitch_gap_delta_too_large",
        "max_normalized_gap_delta": max_gap_delta,
    }


def normalized_gaps(values: list[float]) -> list[float]:
    if len(values) < 2:
        return []
    span = max(values) - min(values)
    if span <= 1e-9:
        return []
    return [(values[index + 1] - values[index]) / span for index in range(len(values) - 1)]


def adjust_pad_center(moving_bbox: list[float], anchor_bbox: list[float], side: str) -> tuple[list[float], float]:
    moving_cx, moving_cy = bbox_center(moving_bbox)
    anchor_cx, anchor_cy = bbox_center(anchor_bbox)
    if side in {"top", "bottom"}:
        delta = anchor_cx - moving_cx
        return clean_bbox([moving_bbox[0] + delta, moving_bbox[1], moving_bbox[2] + delta, moving_bbox[3]]), clean_float(delta)
    delta = anchor_cy - moving_cy
    return clean_bbox([moving_bbox[0], moving_bbox[1] + delta, moving_bbox[2], moving_bbox[3] + delta]), clean_float(delta)


def add_package_derived_pad_adjustments(
    objects: list[dict[str, Any]],
    object_adjustments: dict[str, dict[str, Any]],
    *,
    package_item: dict[str, Any],
    land_item: dict[str, Any],
    delta: float,
    side: str,
    axis: str,
) -> None:
    package_obj = package_item["object"]
    package_id = package_obj.get("source_object_id")
    source_graph = str(package_obj.get("source_graph") or "")
    if package_id is None or not source_graph:
        return
    for index, obj in enumerate(objects):
        if str(obj.get("role") or "") not in {"lead_pad", "partial_pad_width", "partial_lead_pad_length"}:
            continue
        if obj.get("source_package_pad_id") != package_id:
            continue
        if str(obj.get("source_graph") or "") != source_graph:
            continue
        bbox = list(obj.get("bbox") or [])
        if len(bbox) < 4:
            continue
        adjusted = (
            clean_bbox([bbox[0] + delta, bbox[1], bbox[2] + delta, bbox[3]])
            if axis == "x"
            else clean_bbox([bbox[0], bbox[1] + delta, bbox[2], bbox[3] + delta])
        )
        object_adjustments[str(index)] = {
            "source_object_index": index,
            "source_object_id": obj.get("source_object_id"),
            "role": obj.get("role"),
            "raw_view": obj.get("raw_view"),
            "canonical_view": obj.get("canonical_view"),
            "source_graph": obj.get("source_graph"),
            "bbox_before_pad_center_align": bbox,
            "bbox_after_pad_center_align": adjusted,
            "pad_center_alignment_side": side,
            "pad_center_alignment_axis": axis,
            "matched_package_object_index": package_item["index"],
            "matched_package_object_id": package_id,
            "matched_land_object_index": land_item["index"],
            "matched_land_object_id": land_item["object"].get("source_object_id"),
            "pad_center_alignment_anchor_role": "land_pad",
            "adjustment_type": "package_derived_pad_follow_package_pad_center_alignment",
        }


def add_package_derived_pad_adjustments_2d(
    objects: list[dict[str, Any]],
    object_adjustments: dict[str, dict[str, Any]],
    *,
    package_item: dict[str, Any],
    land_item: dict[str, Any],
    delta_x: float,
    delta_y: float,
) -> None:
    package_obj = package_item["object"]
    package_id = package_obj.get("source_object_id")
    source_graph = str(package_obj.get("source_graph") or "")
    if package_id is None or not source_graph:
        return
    for index, obj in enumerate(objects):
        if str(obj.get("role") or "") not in {"lead_pad", "partial_pad_width", "partial_lead_pad_length"}:
            continue
        if obj.get("source_package_pad_id") != package_id:
            continue
        if str(obj.get("source_graph") or "") != source_graph:
            continue
        bbox = list(obj.get("bbox") or [])
        if len(bbox) < 4:
            continue
        object_adjustments[str(index)] = {
            "source_object_index": index,
            "source_object_id": obj.get("source_object_id"),
            "role": obj.get("role"),
            "raw_view": obj.get("raw_view"),
            "canonical_view": obj.get("canonical_view"),
            "source_graph": obj.get("source_graph"),
            "bbox_before_pad_center_align": bbox,
            "bbox_after_pad_center_align": translate_bbox(bbox, delta_x=delta_x, delta_y=delta_y),
            "pad_center_alignment_side": "center_2d",
            "pad_center_alignment_axis": "xy",
            "matched_package_object_index": package_item["index"],
            "matched_package_object_id": package_id,
            "matched_land_object_index": land_item["index"],
            "matched_land_object_id": land_item["object"].get("source_object_id"),
            "pad_center_alignment_anchor_role": "land_pad",
            "adjustment_type": "package_derived_pad_follow_package_pad_center_alignment_2d",
        }


def translate_bbox(bbox: list[float], *, delta_x: float, delta_y: float) -> list[float]:
    return clean_bbox([bbox[0] + delta_x, bbox[1] + delta_y, bbox[2] + delta_x, bbox[3] + delta_y])


def add_inner_land_pad_adjustments(
    objects: list[dict[str, Any]],
    object_adjustments: dict[str, dict[str, Any]],
    *,
    land_item: dict[str, Any],
    delta: float,
    side: str,
    axis: str,
    package_item: dict[str, Any],
) -> None:
    land_obj = land_item["object"]
    land_id = land_obj.get("source_object_id")
    source_graph = str(land_obj.get("source_graph") or "")
    if land_id is None or not source_graph:
        return
    for index, obj in enumerate(objects):
        if str(obj.get("role") or "") != "inner_land_pad":
            continue
        if obj.get("source_land_pad_id") != land_id:
            continue
        if str(obj.get("source_graph") or "") != source_graph:
            continue
        bbox = list(obj.get("bbox") or [])
        if len(bbox) < 4:
            continue
        adjusted = (
            clean_bbox([bbox[0] + delta, bbox[1], bbox[2] + delta, bbox[3]])
            if axis == "x"
            else clean_bbox([bbox[0], bbox[1] + delta, bbox[2], bbox[3] + delta])
        )
        object_adjustments[str(index)] = {
            "source_object_index": index,
            "source_object_id": obj.get("source_object_id"),
            "role": obj.get("role"),
            "raw_view": obj.get("raw_view"),
            "canonical_view": obj.get("canonical_view"),
            "source_graph": obj.get("source_graph"),
            "bbox_before_pad_center_align": bbox,
            "bbox_after_pad_center_align": adjusted,
            "pad_center_alignment_side": side,
            "pad_center_alignment_axis": axis,
            "matched_package_object_index": package_item["index"],
            "matched_package_object_id": package_item["object"].get("source_object_id"),
            "matched_land_object_index": land_item["index"],
            "matched_land_object_id": land_id,
            "adjustment_type": "inner_land_pad_follow_land_pad_center_alignment",
        }


def median_value(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def clean_bbox(bbox: list[float]) -> list[float]:
    return [clean_float(value) for value in bbox]


def clean_float(value: float) -> float:
    rounded = round(float(value), 12)
    return 0.0 if rounded == -0.0 else rounded


def normalized_view(obj: dict[str, Any]) -> str:
    return normalize_view(str(obj.get("canonical_view") or obj.get("raw_view") or ""))


def normalize_view(view: str) -> str:
    return view.strip().lower() or "unknown"


def outline_dimension_refs_by_key(dimensions: list[dict[str, Any]]) -> dict[tuple[str, Any], list[dict[str, Any]]]:
    refs_by_key: dict[tuple[str, Any], list[dict[str, Any]]] = {}
    for dimension in dimensions:
        if not is_outline_dimension(dimension):
            continue
        source_graph = str(dimension.get("source_graph") or "")
        if not source_graph:
            continue
        for target_id in dimension.get("target_ids") or []:
            refs_by_key.setdefault((source_graph, target_id), []).append(
                {
                    "dimension_id": dimension.get("dimension_id", dimension.get("id")),
                    "role": dimension.get("role"),
                    "kind": dimension.get("kind"),
                    "axis": dimension.get("axis"),
                    "value": dimension.get("value"),
                }
            )
    return refs_by_key


def is_outline_dimension(dimension: dict[str, Any]) -> bool:
    role = str(dimension.get("role") or "").lower()
    if role.startswith("outline_"):
        return True
    labels = {str(label).strip().lower() for label in dimension.get("target_labels") or []}
    return bool(labels.intersection({"outline", "package"}))


def source_object_key(obj: dict[str, Any]) -> tuple[str, Any]:
    return (str(obj.get("source_graph") or ""), obj.get("source_object_id"))


def outline_selection_score(obj: dict[str, Any], dimension_refs: list[dict[str, Any]]) -> tuple[int, int, int, float]:
    size_dimension_count = sum(1 for ref in dimension_refs if str(ref.get("role") or "") == "outline_size")
    dimension_rank = 2 if size_dimension_count else (1 if dimension_refs else 0)
    view_rank = {"bottom": 2, "top": 1}.get(outline_view(obj), 0)
    return (dimension_rank, len(dimension_refs), view_rank, bbox_area(list(obj["bbox"])))


def main_outline_selection_reason(obj: dict[str, Any], dimension_refs: list[dict[str, Any]]) -> str:
    if any(str(ref.get("role") or "") == "outline_size" for ref in dimension_refs):
        return "dimension_supported_outline_size_preferred"
    if dimension_refs:
        return "dimension_supported_outline_relation_preferred"
    view = outline_view(obj)
    if view == "bottom":
        return "bottom_outline_preferred_without_dimensions"
    if view == "top":
        return "top_outline_preferred_without_dimensions"
    return "largest_available_outline_without_dimensions"


def outline_summary(index: int, obj: dict[str, Any], dimension_refs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "source_object_index": index,
        "source_object_id": obj.get("source_object_id"),
        "raw_view": obj.get("raw_view"),
        "canonical_view": obj.get("canonical_view"),
        "source_graph": obj.get("source_graph"),
        "bbox": list(obj["bbox"]),
        "dimension_ref_count": len(dimension_refs),
        "dimension_refs": dimension_refs,
    }


def is_alignable_outline_view(obj: dict[str, Any]) -> bool:
    return outline_view(obj) in ALIGNABLE_OUTLINE_VIEWS


def outline_view(obj: dict[str, Any]) -> str:
    return str(obj.get("canonical_view") or obj.get("raw_view") or "").lower()


def is_outline_adjustable_pad(obj: dict[str, Any]) -> bool:
    return str(obj.get("role") or "") in OUTLINE_EDGE_PAD_ROLES


def is_same_outline_source(obj: dict[str, Any], outline: dict[str, Any]) -> bool:
    obj_graph = str(obj.get("source_graph") or "")
    outline_graph = str(outline.get("source_graph") or "")
    if obj_graph and outline_graph:
        return obj_graph == outline_graph
    return outline_view(obj) == outline_view(outline)


def outline_edge_tolerance(outline_bbox: list[float]) -> float:
    return max(abs(outline_bbox[2] - outline_bbox[0]), abs(outline_bbox[3] - outline_bbox[1]), 1e-9) * OUTLINE_EDGE_TOLERANCE_RATIO


def locked_outline_edges(bbox: list[float], outline_bbox: list[float], *, tolerance: float) -> list[str]:
    locks = []
    if abs(bbox[0] - outline_bbox[0]) <= tolerance:
        locks.append("left")
    if abs(bbox[2] - outline_bbox[2]) <= tolerance:
        locks.append("right")
    if abs(bbox[1] - outline_bbox[1]) <= tolerance:
        locks.append("top")
    if abs(bbox[3] - outline_bbox[3]) <= tolerance:
        locks.append("bottom")
    return locks


def adjust_bbox_to_outline_edges(bbox: list[float], locks: list[str], target_outline_bbox: list[float]) -> list[float]:
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x1, y1, x2, y2 = bbox
    if "left" in locks and "right" in locks:
        x1, x2 = target_outline_bbox[0], target_outline_bbox[2]
    elif "left" in locks:
        x1 = target_outline_bbox[0]
        x2 = x1 + width
    elif "right" in locks:
        x2 = target_outline_bbox[2]
        x1 = x2 - width
    if "top" in locks and "bottom" in locks:
        y1, y2 = target_outline_bbox[1], target_outline_bbox[3]
    elif "top" in locks:
        y1 = target_outline_bbox[1]
        y2 = y1 + height
    elif "bottom" in locks:
        y2 = target_outline_bbox[3]
        y1 = y2 - height
    return clean_bbox([x1, y1, x2, y2])


def build_pad_stacks(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        obj
        for obj in objects
        if obj["role"] in {"package_pad", "land_pad", "lead_pad", "partial_pad_width", "partial_lead_pad_length", "inner_land_pad"}
    ]
    source_groups = source_key_groups(candidates)
    stacks: list[dict[str, Any]] = []
    used: set[int] = set()
    for index, obj in enumerate(candidates):
        if index in used:
            continue
        group_indices = set(source_groups.get(index) or [index])
        if len(group_indices) == 1:
            group_indices = {index}
            for other_index, other in enumerate(candidates[index + 1 :], start=index + 1):
                if other_index in used:
                    continue
                if same_pad_family(obj, other):
                    group_indices.add(other_index)
        group = []
        for group_index in sorted(group_indices):
            if group_index in used:
                continue
            group.append(candidates[group_index])
            used.add(group_index)
        if not group:
            continue
        representative = select_representative(group)
        stacks.append(
            {
                "id": f"pad_stack_{len(stacks)}",
                "shape_family": representative["shape_family"],
                "canonical_bbox": representative["bbox"],
                "canonical_role": representative["role"],
                "canonical_source_quality": representative["source_quality"],
                "selection_reason": selection_reason(representative),
                "member_count": len(group),
                "members": [
                    pad_stack_member(member)
                    for member in sorted(group, key=lambda item: quality_rank(item), reverse=True)
                ],
            }
        )
    return stacks


def pad_stack_member(member: dict[str, Any]) -> dict[str, Any]:
    result = {
                        "role": member["role"],
                        "shape_family": member["shape_family"],
                        "bbox": member["bbox"],
                        "source_quality": member["source_quality"],
                        "raw_view": member["raw_view"],
                        "source_graph": member["source_graph"],
                        "source_object_id": member["source_object_id"],
                        "source_package_pad_id": member.get("source_package_pad_id"),
                        "source_land_pad_id": member.get("source_land_pad_id"),
    }
    for key in (
        "bbox_before_outline_adjust",
        "bbox_after_outline_adjust",
        "outline_edge_lock",
        "outline_adjustment_type",
        "bbox_before_pad_center_align",
        "bbox_after_pad_center_align",
        "pad_center_alignment_side",
        "pad_center_alignment_axis",
        "pad_center_alignment_type",
        "matched_package_object_id",
    ):
        if key in member:
            result[key] = member[key]
    return result


def source_key_groups(candidates: list[dict[str, Any]]) -> dict[int, list[int]]:
    by_key: dict[tuple[str, str, Any], list[int]] = {}
    for index, obj in enumerate(candidates):
        for key in source_group_keys(obj):
            by_key.setdefault(key, []).append(index)
    result: dict[int, list[int]] = {}
    for indices in by_key.values():
        if len(indices) < 2:
            continue
        merged = sorted(set(indices))
        for index in merged:
            result[index] = merged
    return result


def source_group_keys(obj: dict[str, Any]) -> list[tuple[str, str, Any]]:
    role = str(obj.get("role") or "")
    source_graph = str(obj.get("source_graph") or "")
    keys = []
    if role == "package_pad":
        keys.append(("package", source_graph, obj.get("source_object_id")))
    elif role in {"lead_pad", "partial_pad_width", "partial_lead_pad_length"}:
        keys.append(("package", source_graph, obj.get("source_package_pad_id")))
    elif role == "land_pad":
        keys.append(("land", source_graph, obj.get("source_object_id")))
    elif role == "inner_land_pad":
        keys.append(("land", source_graph, obj.get("source_land_pad_id")))
    return [key for key in keys if key[1] and key[2] is not None]


def same_pad_family(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if geometry_group_role(a) != geometry_group_role(b):
        return False
    if a["shape_family"] != b["shape_family"]:
        return False
    ac = bbox_center(a["bbox"])
    bc = bbox_center(b["bbox"])
    if ac is None or bc is None:
        return False
    tolerance = max(min_bbox_size(a["bbox"]), min_bbox_size(b["bbox"]), 1e-6) * 0.45
    if math.hypot(ac[0] - bc[0], ac[1] - bc[1]) > tolerance:
        return False
    return contains_or_overlaps(a["bbox"], b["bbox"])


def geometry_group_role(obj: dict[str, Any]) -> str:
    role = str(obj.get("role") or "")
    if role in {"lead_pad", "partial_pad_width", "partial_lead_pad_length"}:
        return "lead_pad"
    return role


def contains_or_overlaps(a: list[float], b: list[float]) -> bool:
    return bbox_contains(a, b, tolerance=0.05) or bbox_contains(b, a, tolerance=0.05) or bbox_iou(a, b) >= 0.35


def select_representative(group: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(group, key=lambda item: (quality_rank(item), role_rank(item), bbox_area(item["bbox"])), reverse=True)[0]


def quality_rank(obj: dict[str, Any]) -> int:
    quality = str(obj.get("source_quality") or "")
    if quality == "dimension_supported":
        return 3
    if quality == "derived_from_dimension":
        return 2
    if quality == "graph_object":
        return 1
    return 0


def role_rank(obj: dict[str, Any]) -> int:
    return {
        "inner_land_pad": 4,
        "land_pad": 3,
        "lead_pad": 3,
        "partial_pad_width": 3,
        "partial_lead_pad_length": 3,
        "package_pad": 2,
    }.get(str(obj.get("role") or ""), 1)


def selection_reason(obj: dict[str, Any]) -> str:
    quality = str(obj.get("source_quality") or "")
    if quality == "dimension_supported":
        return "dimension_supported_preferred_over_prediction"
    if quality == "derived_from_dimension":
        return "derived_dimension_geometry_preferred"
    return "best_available_graph_geometry"


def source_quality(obj: dict[str, Any]) -> str:
    if obj.get("inner_land_pad_source") or obj.get("source_type") == "derived_inner_land_pad":
        return "derived_from_dimension"
    if dimension_refs(obj):
        return "dimension_supported"
    return "graph_object"


def dimension_refs(obj: dict[str, Any]) -> list[Any]:
    refs = []
    for key in ("dimension_id", "dimension_ids"):
        value = obj.get(key)
        if isinstance(value, list):
            refs.extend(value)
        elif value is not None:
            refs.append(value)
    source = obj.get("inner_land_pad_source") or {}
    if source.get("dimension_id") is not None:
        refs.append(source.get("dimension_id"))
    return refs


def load_gt_objects(scan_result_path: Path | None) -> list[dict[str, Any]]:
    if scan_result_path is None or not scan_result_path.is_file():
        return []
    gt = parse_scan_result(scan_result_path)
    features = gt.get("features") or {}
    objects = features.get("effective_objects") or features.get("objects") or []
    result = []
    for obj in objects:
        bbox = object_bbox(obj)
        if bbox is None:
            continue
        result.append(
            {
                "source_object_id": obj.get("id"),
                "role": str(obj.get("role") or "unknown"),
                "label": str(obj.get("node_name") or obj.get("geometry") or ""),
                "shape_family": shape_family(obj),
                "bbox": list(bbox),
            }
        )
    return result


def write_multiview_alignment_svg(
    path: Path,
    *,
    part_number: str,
    result_objects: list[dict[str, Any]],
    pad_stacks: list[dict[str, Any]],
    land_package_pair_diagnostics: dict[str, Any],
    gt_objects: list[dict[str, Any]],
) -> None:
    width = 1280.0
    height = 900.0
    result_panel = (44.0, 112.0, 618.0, 830.0)
    gt_panel = (662.0, 112.0, 1236.0, 830.0)
    result_transform = transform_for_objects(result_objects, result_panel)
    gt_transform = transform_for_objects(gt_objects, gt_panel)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(width)}" height="{int(height)}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="28" y="38" font-family="monospace" font-size="21" fill="#0f172a">{escape(part_number)} multiview alignment</text>',
        '<text x="28" y="64" font-family="monospace" font-size="13" fill="#64748b">'
        "left: multiview result + land/package pad-center diagnostics; right: ScanResultFormat GT reference only</text>",
        panel_rect(result_panel, "Multiview result / land-package alignment"),
        panel_rect(gt_panel, "GT reference"),
    ]
    if result_transform:
        for obj in result_objects:
            lines.append(draw_result_object(obj, result_transform))
        for stack in visible_pad_stacks(pad_stacks):
            lines.append(draw_stack_outline(stack, result_transform))
        lines.extend(draw_land_package_pair_diagnostics(land_package_pair_diagnostics, result_transform))
    else:
        lines.append(empty_panel_text(result_panel, "No multiview result geometry"))
    if gt_transform:
        for obj in gt_objects:
            lines.append(draw_gt_object(obj, gt_transform))
    else:
        lines.append(empty_panel_text(gt_panel, "No GT reference geometry"))
    lines.extend(legend())
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def visible_pad_stacks(pad_stacks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [stack for stack in pad_stacks if int(stack.get("member_count") or 0) > 1]


def transform_for_objects(objects: list[dict[str, Any]], target: tuple[float, float, float, float]) -> FrameTransform | None:
    boxes = [obj["bbox"] for obj in objects if len(obj.get("bbox") or []) >= 4]
    if not boxes:
        return None
    frame = union_boxes(boxes)
    fw = max(frame[2] - frame[0], 1e-9)
    fh = max(frame[3] - frame[1], 1e-9)
    tw = max(target[2] - target[0], 1e-9)
    th = max(target[3] - target[1], 1e-9)
    return FrameTransform(source=frame, target=target, scale=min(tw / fw, th / fh) * 0.92)


def map_bbox(bbox: list[float], transform: FrameTransform) -> list[float]:
    sx1, sy1, sx2, sy2 = transform.source
    tx1, ty1, tx2, ty2 = transform.target
    source_cx = (sx1 + sx2) / 2.0
    source_cy = (sy1 + sy2) / 2.0
    target_cx = (tx1 + tx2) / 2.0
    target_cy = (ty1 + ty2) / 2.0
    return [
        target_cx + (bbox[0] - source_cx) * transform.scale,
        target_cy + (bbox[1] - source_cy) * transform.scale,
        target_cx + (bbox[2] - source_cx) * transform.scale,
        target_cy + (bbox[3] - source_cy) * transform.scale,
    ]


def draw_result_object(obj: dict[str, Any], transform: FrameTransform) -> str:
    color = RESULT_COLORS.get(str(obj.get("role") or ""), RESULT_COLORS["unknown"])
    bbox = map_bbox(obj["bbox"], transform)
    return shape_svg(obj, bbox, stroke=color, fill=color, fill_opacity=0.16, stroke_width=2.2)


def draw_gt_object(obj: dict[str, Any], transform: FrameTransform) -> str:
    color = GT_COLORS.get(str(obj.get("role") or ""), GT_COLORS["unknown"])
    bbox = map_bbox(obj["bbox"], transform)
    return shape_svg(obj, bbox, stroke=color, fill=color, fill_opacity=0.08, stroke_width=2.2, dash="5 4")


def draw_stack_outline(stack: dict[str, Any], transform: FrameTransform) -> str:
    bbox = map_bbox(list(stack["canonical_bbox"]), transform)
    attrs = {
        "data-stack-id": str(stack.get("id") or ""),
        "data-member-count": str(stack.get("member_count") or ""),
        "data-selection-reason": str(stack.get("selection_reason") or ""),
    }
    return rect_svg(bbox, stroke="#111827", fill="none", fill_opacity=0.0, stroke_width=1.4, dash="2 5", attrs=attrs)


def draw_land_package_pair_diagnostics(
    diagnostics: dict[str, Any],
    transform: FrameTransform,
) -> list[str]:
    lines = []
    for pair in diagnostics.get("pairs") or []:
        package_center = pair.get("package_center") or []
        land_center = pair.get("land_center") or []
        if len(package_center) < 2 or len(land_center) < 2:
            continue
        px, py = map_point(float(package_center[0]), float(package_center[1]), transform)
        lx, ly = map_point(float(land_center[0]), float(land_center[1]), transform)
        attrs = {
            "data-package-object-id": svg_data_value(pair.get("package_object_id")),
            "data-land-object-id": svg_data_value(pair.get("land_object_id")),
            "data-axis-residual": str(pair.get("axis_residual") or 0.0),
            "data-center-distance": str(pair.get("center_distance") or 0.0),
        }
        attr_text = "".join(f' {escape(key)}="{escape(value)}"' for key, value in attrs.items())
        lines.append(f'<circle cx="{px:.3f}" cy="{py:.3f}" r="2.8" fill="#2563eb" stroke="#ffffff" stroke-width="0.8"{attr_text}/>')
        lines.append(f'<circle cx="{lx:.3f}" cy="{ly:.3f}" r="1.7" fill="#16a34a" stroke="#ffffff" stroke-width="0.7"{attr_text}/>')
        if float(pair.get("axis_residual") or 0.0) > 1e-6:
            lines.append(
                f'<line x1="{px:.3f}" y1="{py:.3f}" x2="{lx:.3f}" y2="{ly:.3f}" '
                f'stroke="#ef4444" stroke-width="1.2" stroke-opacity="0.65"{attr_text}/>'
            )
    return lines


def map_point(x: float, y: float, transform: FrameTransform) -> tuple[float, float]:
    sx1, sy1, sx2, sy2 = transform.source
    tx1, ty1, tx2, ty2 = transform.target
    source_cx = (sx1 + sx2) / 2.0
    source_cy = (sy1 + sy2) / 2.0
    target_cx = (tx1 + tx2) / 2.0
    target_cy = (ty1 + ty2) / 2.0
    return (
        target_cx + (x - source_cx) * transform.scale,
        target_cy + (y - source_cy) * transform.scale,
    )


def svg_data_value(value: Any) -> str:
    return "" if value is None else str(value)


def shape_svg(
    obj: dict[str, Any],
    bbox: list[float],
    *,
    stroke: str,
    fill: str,
    fill_opacity: float,
    stroke_width: float,
    dash: str = "",
) -> str:
    if str(obj.get("shape_family") or "") == "circle":
        x1, y1, x2, y2 = bbox
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        return (
            f'<ellipse cx="{(x1 + x2) / 2.0:.3f}" cy="{(y1 + y2) / 2.0:.3f}" '
            f'rx="{abs(x2 - x1) / 2.0:.3f}" ry="{abs(y2 - y1) / 2.0:.3f}" '
            f'fill="{fill}" fill-opacity="{fill_opacity}" stroke="{stroke}" '
            f'stroke-width="{stroke_width}"{dash_attr}/>'
        )
    return rect_svg(bbox, stroke=stroke, fill=fill, fill_opacity=fill_opacity, stroke_width=stroke_width, dash=dash)


def rect_svg(
    bbox: list[float],
    *,
    stroke: str,
    fill: str,
    fill_opacity: float,
    stroke_width: float,
    dash: str = "",
    attrs: dict[str, str] | None = None,
) -> str:
    x1, y1, x2, y2 = bbox
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    attr_text = "".join(f' {escape(key)}="{escape(value)}"' for key, value in (attrs or {}).items())
    return (
        f'<rect x="{min(x1, x2):.3f}" y="{min(y1, y2):.3f}" '
        f'width="{abs(x2 - x1):.3f}" height="{abs(y2 - y1):.3f}" '
        f'fill="{fill}" fill-opacity="{fill_opacity}" stroke="{stroke}" '
        f'stroke-width="{stroke_width}"{dash_attr}{attr_text}/>'
    )


def panel_rect(panel: tuple[float, float, float, float], title: str) -> str:
    x1, y1, x2, y2 = panel
    return (
        f'<rect x="{x1}" y="{y1}" width="{x2 - x1}" height="{y2 - y1}" fill="#f8fafc" stroke="#cbd5e1"/>'
        f'<text x="{x1 + 12}" y="{y1 - 12}" font-family="monospace" font-size="14" fill="#334155">{escape(title)}</text>'
    )


def empty_panel_text(panel: tuple[float, float, float, float], text: str) -> str:
    x1, y1, x2, y2 = panel
    return (
        f'<text x="{(x1 + x2) / 2.0}" y="{(y1 + y2) / 2.0}" text-anchor="middle" '
        f'font-family="monospace" font-size="14" fill="#64748b">{escape(text)}</text>'
    )


def legend() -> list[str]:
    items = [
        ("#2563eb", "package pad"),
        ("#16a34a", "land pad"),
        ("#7c3aed", "inner land pad"),
        ("#111827", "pad stack representative"),
        ("#b45309", "GT lead"),
        ("#15803d", "GT land"),
    ]
    lines = []
    x = 32
    y = 868
    for color, label in items:
        lines.append(f'<rect x="{x}" y="{y - 11}" width="14" height="14" fill="{color}" fill-opacity="0.20" stroke="{color}"/>')
        lines.append(f'<text x="{x + 20}" y="{y}" font-family="monospace" font-size="12" fill="#334155">{escape(label)}</text>')
        x += 190
    return lines


def object_bbox(obj: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = obj.get("bbox") or []
    if len(bbox) < 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    if x1 == x2 or y1 == y2:
        return None
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def shape_family(obj: dict[str, Any]) -> str:
    text = " ".join(str(obj.get(key) or "") for key in ("label", "source_label", "node_name", "geometry")).lower()
    if "circle" in text:
        return "circle"
    if "dshape" in text:
        return "dshape"
    return "rect"


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def min_bbox_size(bbox: list[float]) -> float:
    return min(abs(bbox[2] - bbox[0]), abs(bbox[3] - bbox[1]))


def bbox_area(bbox: list[float]) -> float:
    return abs((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


def bbox_contains(outer: list[float], inner: list[float], *, tolerance: float) -> bool:
    return (
        inner[0] >= outer[0] - tolerance
        and inner[1] >= outer[1] - tolerance
        and inner[2] <= outer[2] + tolerance
        and inner[3] <= outer[3] + tolerance
    )


def bbox_iou(a: list[float], b: list[float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    union = bbox_area(a) + bbox_area(b) - intersection
    return intersection / union if union > 0.0 else 0.0


def union_boxes(boxes: list[list[float]]) -> tuple[float, float, float, float]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    main()
