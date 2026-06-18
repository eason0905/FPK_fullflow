#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MERGE_SOURCE_ROLES = {"lead_pad", "partial_pad_width", "partial_lead_pad_length"}
DRAW_COLORS = {
    "merged_pad": "#dc2626",
    "unmerged_pad": "#64748b",
    "package_pad": "#2563eb",
    "land_pad": "#16a34a",
    "inner_land_pad": "#9333ea",
    "outline_2d": "#0f172a",
}


@dataclass(frozen=True)
class FrameTransform:
    source: tuple[float, float, float, float]
    target: tuple[float, float, float, float]
    scale: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge aligned front/side pad evidence without changing upstream multiview coordinates."
    )
    parser.add_argument("--aligned-layers", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--part", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_multiview_mergy_pad(
        aligned_layers_path=args.aligned_layers,
        output_dir=args.output_dir,
        part=args.part,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_multiview_mergy_pad(
    *,
    aligned_layers_path: Path,
    output_dir: Path,
    part: str = "",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(aligned_layers_path.read_text(encoding="utf-8"))
    part_number = part or str(payload.get("part_number") or aligned_layers_path.parent.name)
    result = build_mergy_pad_payload(payload, part_number=part_number, aligned_layers_path=aligned_layers_path)
    json_path = output_dir / "mergy_pad.json"
    svg_path = output_dir / "mergy_pad.svg"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_mergy_pad_svg(svg_path, result)
    return {
        "part": part_number,
        "mergy_pad_json": str(json_path),
        "mergy_pad_svg": str(svg_path),
        "merged_pad_count": len(result["merged_pads"]),
        "unmerged_pad_count": len(result["unmerged_pads"]),
        "candidate_group_count": result["candidate_group_count"],
        "unresolved_count": result["unresolved_count"],
    }


def build_mergy_pad_payload(
    payload: dict[str, Any],
    *,
    part_number: str,
    aligned_layers_path: Path | None = None,
) -> dict[str, Any]:
    objects = list(payload.get("objects") or [])
    groups = grouped_pad_evidence(objects)
    merged = []
    unmerged = []
    unresolved_count = 0
    candidate_count = 0
    for key, items in sorted(groups.items(), key=lambda item: (item[0][0], str(item[0][1]))):
        source_pair = select_merge_source_pair(items)
        if source_pair is None:
            source = select_unmerged_member(items)
            if source is not None:
                unmerged.append(unmerged_pad(key, source, len(unmerged), "single_source_or_no_merge_pair"))
            continue
        candidate_count += 1
        x_source, y_source, policy = source_pair
        bbox = source_x_source_y_bbox(x_source["bbox"], y_source["bbox"])
        if bbox is None:
            unresolved_count += 1
            source = select_unmerged_member(items)
            if source is not None:
                unmerged.append(unmerged_pad(key, source, len(unmerged), "invalid_merge_bbox"))
            continue
        merged.append(merged_pad(key, bbox, len(merged), policy))
    merge_graph_objects = build_merge_graph_objects(objects, merged, unmerged)
    return {
        "part_number": part_number,
        "coordinate_mode": str(payload.get("coordinate_mode") or "dimension_scaled_centered"),
        "coordinate_policy": (
            "mergy_pad_stage_applied_coordinates; input bboxes are aligned_multiview_layers coordinates; "
            "merged bboxes use one source x-range and one source y-range in the same coordinate system; SVG applies display scale only"
        ),
        "inputs": {
            "aligned_multiview_layers": str(aligned_layers_path) if aligned_layers_path else "",
        },
        "merge_scope": "pad evidence groups with front+side, front+lead, or lead+side",
        "merge_policy": "front_x_side_y_or_lead_fills_missing_axis",
        "candidate_group_count": candidate_count,
        "object_count": len(merge_graph_objects),
        "merged_pad_count": len(merged),
        "unmerged_pad_count": len(unmerged),
        "unresolved_count": unresolved_count,
        "objects": merge_graph_objects,
        "merged_pads": merged,
        "unmerged_pads": unmerged,
    }


def grouped_pad_evidence(objects: list[dict[str, Any]]) -> dict[tuple[str, Any], list[dict[str, Any]]]:
    groups: dict[tuple[str, Any], list[dict[str, Any]]] = {}
    for obj in objects:
        if str(obj.get("role") or "") not in MERGE_SOURCE_ROLES:
            continue
        view = normalized_view(obj)
        if view not in {"front", "side", "lead"}:
            continue
        source_graph = str(obj.get("source_graph") or "")
        source_package_pad_id = obj.get("source_package_pad_id")
        bbox = obj.get("bbox") or []
        if not source_graph or source_package_pad_id is None or len(bbox) < 4:
            continue
        groups.setdefault((source_graph, source_package_pad_id), []).append(obj)
    return groups


def normalized_view(obj: dict[str, Any]) -> str:
    return str(obj.get("raw_view") or obj.get("canonical_view") or "").strip().lower()


def build_merge_graph_objects(
    source_objects: list[dict[str, Any]],
    merged_pads: list[dict[str, Any]],
    unmerged_pads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    derived_package_pad_keys = package_pad_keys_with_derived_lead_pads(source_objects)
    objects = [
        merge_graph_context_object(obj)
        for obj in source_objects
        if str(obj.get("role") or "") not in MERGE_SOURCE_ROLES
        and not is_derived_source_package_pad(obj, derived_package_pad_keys)
        and valid_bbox(obj.get("bbox") or [])
    ]
    objects = [obj for obj in objects if obj is not None]
    objects.extend(dict(item) for item in unmerged_pads)
    objects.extend(dict(item) for item in merged_pads)
    return sorted(objects, key=merge_graph_object_sort_key)


def package_pad_keys_with_derived_lead_pads(source_objects: list[dict[str, Any]]) -> set[tuple[str, str]]:
    keys = set()
    for obj in source_objects:
        if str(obj.get("role") or "") not in MERGE_SOURCE_ROLES:
            continue
        source_graph = str(obj.get("source_graph") or "")
        source_package_pad_id = obj.get("source_package_pad_id")
        if not source_graph or source_package_pad_id is None:
            continue
        keys.add((source_graph, json.dumps(source_package_pad_id, sort_keys=True)))
    return keys


def is_derived_source_package_pad(obj: dict[str, Any], derived_package_pad_keys: set[tuple[str, str]]) -> bool:
    if str(obj.get("role") or "") != "package_pad":
        return False
    source_graph = str(obj.get("source_graph") or "")
    source_object_id = obj.get("source_object_id")
    if not source_graph or source_object_id is None:
        return False
    return (source_graph, json.dumps(source_object_id, sort_keys=True)) in derived_package_pad_keys


def merge_graph_context_object(obj: dict[str, Any]) -> dict[str, Any] | None:
    role = str(obj.get("role") or "")
    bbox = obj.get("bbox") or []
    if not role or not valid_bbox(bbox):
        return None
    result = {
        "role": role,
        "bbox": clean_bbox(bbox),
        "source_graph": obj.get("source_graph"),
        "source_object_id": obj.get("source_object_id"),
        "source_package_pad_id": obj.get("source_package_pad_id"),
        "source_land_pad_id": obj.get("source_land_pad_id"),
        "raw_view": obj.get("raw_view"),
        "canonical_view": obj.get("canonical_view"),
        "source_quality": obj.get("source_quality"),
    }
    return {key: value for key, value in result.items() if value is not None}


def merge_graph_object_sort_key(obj: dict[str, Any]) -> tuple[int, str, str]:
    role_rank_value = {
        "outline_2d": 0,
        "package_pad": 1,
        "land_pad": 2,
        "inner_land_pad": 3,
        "unmerged_pad": 4,
        "merged_pad": 5,
    }.get(str(obj.get("role") or ""), 9)
    return (
        role_rank_value,
        str(obj.get("source_graph") or ""),
        str(obj.get("source_package_pad_id") or obj.get("source_object_id") or obj.get("merge_id") or ""),
    )


def select_merge_member(items: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        items,
        key=lambda obj: (
            bbox_area(obj.get("bbox") or []),
            role_rank(str(obj.get("role") or "")),
            str(obj.get("source_object_id") or ""),
        ),
    )[0]


def select_unmerged_member(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [item for item in items if valid_bbox(item.get("bbox") or [])]
    if not valid:
        return None
    return select_merge_member(valid)


def role_rank(role: str) -> int:
    return {"partial_pad_width": 0, "lead_pad": 1, "partial_lead_pad_length": 2}.get(role, 9)


def select_merge_source_pair(items: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], str] | None:
    by_view = {
        view: [item for item in items if normalized_view(item) == view]
        for view in ("front", "side", "lead")
    }
    if by_view["front"] and by_view["side"]:
        return (
            select_merge_member(by_view["front"]),
            select_merge_member(by_view["side"]),
            "front_x_side_y",
        )
    if by_view["front"] and by_view["lead"]:
        return (
            select_merge_member(by_view["front"]),
            select_merge_member(by_view["lead"]),
            "front_x_lead_y",
        )
    if by_view["lead"] and by_view["side"]:
        return (
            select_merge_member(by_view["lead"]),
            select_merge_member(by_view["side"]),
            "lead_x_side_y",
        )
    return None


def merged_pad(
    key: tuple[str, Any],
    bbox: list[float],
    index: int,
    merge_policy: str,
) -> dict[str, Any]:
    return {
        "merge_id": f"merged_pad_{index:04d}",
        "role": "merged_pad",
        "bbox": clean_bbox(bbox),
        "source_graph": key[0],
        "source_package_pad_id": key[1],
        "merge_policy": merge_policy,
    }


def unmerged_pad(
    key: tuple[str, Any],
    source: dict[str, Any],
    index: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "merge_id": f"unmerged_pad_{index:04d}",
        "role": "unmerged_pad",
        "bbox": clean_bbox(source["bbox"]),
        "source_graph": key[0],
        "source_package_pad_id": key[1],
        "source_view": normalized_view(source),
        "source_role": str(source.get("role") or ""),
        "merge_policy": "unmerged",
        "unmerged_reason": reason,
    }


def source_x_source_y_bbox(x_source_bbox: list[float], y_source_bbox: list[float]) -> list[float] | None:
    if len(x_source_bbox) < 4 or len(y_source_bbox) < 4:
        return None
    x1 = float(x_source_bbox[0])
    x2 = float(x_source_bbox[2])
    y1 = float(y_source_bbox[1])
    y2 = float(y_source_bbox[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def bbox_area(bbox: list[float]) -> float:
    if len(bbox) < 4:
        return 0.0
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def valid_bbox(bbox: list[float]) -> bool:
    if len(bbox) < 4:
        return False
    return float(bbox[2]) > float(bbox[0]) and float(bbox[3]) > float(bbox[1])


def clean_bbox(bbox: list[float]) -> list[float]:
    return [clean_float(value) for value in bbox]


def clean_float(value: float) -> float:
    rounded = round(float(value), 12)
    return 0.0 if rounded == -0.0 else rounded


def write_mergy_pad_svg(path: Path, payload: dict[str, Any]) -> None:
    width = 1100.0
    height = 820.0
    panel = (40.0, 96.0, 1060.0, 780.0)
    drawable = drawable_objects(payload)
    transform = transform_for_objects(drawable, panel)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(width)}" height="{int(height)}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="28" y="38" font-family="monospace" font-size="21" fill="#0f172a">{escape(str(payload.get("part_number") or ""))} merge pad</text>',
        '<text x="28" y="64" font-family="monospace" font-size="13" fill="#64748b">'
        "merged when the same package pad has front/side, front/lead, or lead/side pad evidence; JSON coordinates are not display-scaled</text>",
        panel_rect(panel, "Merge pad"),
    ]
    if transform is None:
        lines.append(empty_text(panel, "No merge-stage pads"))
    else:
        for obj in payload.get("objects") or []:
            lines.append(draw_merge_graph_object(obj, transform))
    lines.extend(legend())
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def drawable_objects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    objects = list(payload.get("objects") or [])
    return [obj for obj in objects if len(obj.get("bbox") or []) >= 4]


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


def union_boxes(boxes: list[list[float]]) -> tuple[float, float, float, float]:
    return (
        min(float(box[0]) for box in boxes),
        min(float(box[1]) for box in boxes),
        max(float(box[2]) for box in boxes),
        max(float(box[3]) for box in boxes),
    )


def map_bbox(bbox: list[float], transform: FrameTransform) -> list[float]:
    sx1, sy1, sx2, sy2 = transform.source
    tx1, ty1, tx2, ty2 = transform.target
    source_cx = (sx1 + sx2) / 2.0
    source_cy = (sy1 + sy2) / 2.0
    target_cx = (tx1 + tx2) / 2.0
    target_cy = (ty1 + ty2) / 2.0
    return [
        target_cx + (float(bbox[0]) - source_cx) * transform.scale,
        target_cy + (float(bbox[1]) - source_cy) * transform.scale,
        target_cx + (float(bbox[2]) - source_cx) * transform.scale,
        target_cy + (float(bbox[3]) - source_cy) * transform.scale,
    ]


def draw_box(
    bbox: list[float],
    transform: FrameTransform,
    color: str,
    *,
    fill_opacity: float,
    stroke_width: float,
    dash: str = "",
) -> str:
    x1, y1, x2, y2 = map_bbox(bbox, transform)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x1:.3f}" y="{y1:.3f}" width="{x2 - x1:.3f}" height="{y2 - y1:.3f}" '
        f'fill="{color}" fill-opacity="{fill_opacity}" stroke="{color}" stroke-width="{stroke_width}"{dash_attr}/>'
    )


def draw_merge_graph_object(obj: dict[str, Any], transform: FrameTransform) -> str:
    role = str(obj.get("role") or "")
    color = DRAW_COLORS.get(role, "#475569")
    if role == "outline_2d":
        return draw_box(obj["bbox"], transform, color, fill_opacity=0.0, stroke_width=1.4, dash="6 5")
    if role == "merged_pad":
        return draw_box(obj["bbox"], transform, color, fill_opacity=0.26, stroke_width=3.0)
    if role == "unmerged_pad":
        return draw_box(obj["bbox"], transform, color, fill_opacity=0.10, stroke_width=1.8, dash="5 4")
    if role == "inner_land_pad":
        return draw_box(obj["bbox"], transform, color, fill_opacity=0.14, stroke_width=1.6)
    return draw_box(obj["bbox"], transform, color, fill_opacity=0.10, stroke_width=1.4)


def panel_rect(panel: tuple[float, float, float, float], title: str) -> str:
    x1, y1, x2, y2 = panel
    return (
        f'<rect x="{x1}" y="{y1}" width="{x2 - x1}" height="{y2 - y1}" fill="#f8fafc" stroke="#cbd5e1"/>'
        f'<text x="{x1 + 12}" y="{y1 + 24}" font-family="monospace" font-size="14" fill="#334155">{escape(title)}</text>'
    )


def empty_text(panel: tuple[float, float, float, float], message: str) -> str:
    x1, y1, x2, y2 = panel
    return (
        f'<text x="{(x1 + x2) / 2.0:.3f}" y="{(y1 + y2) / 2.0:.3f}" text-anchor="middle" '
        f'font-family="monospace" font-size="16" fill="#64748b">{escape(message)}</text>'
    )


def legend() -> list[str]:
    items = [
        ("package pad", DRAW_COLORS["package_pad"]),
        ("land pad", DRAW_COLORS["land_pad"]),
        ("merged pad", DRAW_COLORS["merged_pad"]),
        ("unmerged pad", DRAW_COLORS["unmerged_pad"]),
    ]
    lines = []
    x = 42.0
    y = 798.0
    for label, color in items:
        lines.append(f'<rect x="{x}" y="{y - 10}" width="16" height="10" fill="{color}" fill-opacity="0.28" stroke="{color}"/>')
        lines.append(f'<text x="{x + 22}" y="{y}" font-family="monospace" font-size="12" fill="#334155">{escape(label)}</text>')
        x += 170.0
    return lines


def escape(value: str) -> str:
    return html.escape(value, quote=True)


if __name__ == "__main__":
    main()
