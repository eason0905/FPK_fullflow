from __future__ import annotations

import json
import math
import base64
import mimetypes
import re
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from real_image_process.FPK_PJ_fullflow.review.schema import slugify


PAD_LABELS = {"pad", "pad_circle", "pad_dshape", "rect", "circle"}
OUTLINE_LABELS = {"outline", "package"}
LEAD_CONTACT_TARGET_LABELS = PAD_LABELS | {"lead"}
EXPECTED_CANONICAL_VIEWS = ("bottom", "land", "lateral", "lead_detail")
BODY_SIZE_DIMENSION_SYMBOLS = {"D", "E", "D1", "E1"}
OVERDENSE_TOP_LAND_COUNT_FALLBACK_REASON = "primary_package_pad_layout_overdense_top_matches_land_count"
UNIFIED_MULTIVIEW_LAYERS_FILENAME = "unified_multiview_layers.json"
UNIFIED_MULTIVIEW_LAYERS_SVG_FILENAME = "unified_multiview_layers.svg"


@dataclass(frozen=True)
class MultiviewOptions:
    input_mode: str = "gt"
    group_by: str = "part_number"
    primary_package_pad_view: str = "bottom"
    primary_land_pad_view: str = "land"
    lateral_views: tuple[str, ...] = ("side", "front")
    conflict_abs_tol: float = 0.05
    conflict_rel_tol: float = 0.05
    ignore_lateral_height: bool = True
    preserve_raw_view: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "MultiviewOptions":
        payload = payload or {}
        return cls(
            input_mode=str(payload.get("input_mode") or "gt"),
            group_by=str(payload.get("group_by") or "part_number"),
            primary_package_pad_view=str(payload.get("primary_package_pad_view") or "bottom"),
            primary_land_pad_view=str(payload.get("primary_land_pad_view") or "land"),
            lateral_views=tuple(payload.get("lateral_views") or ("side", "front")),
            conflict_abs_tol=float(payload.get("conflict_abs_tol", 0.05)),
            conflict_rel_tol=float(payload.get("conflict_rel_tol", 0.05)),
            ignore_lateral_height=bool(payload.get("ignore_lateral_height", True)),
            preserve_raw_view=bool(payload.get("preserve_raw_view", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_mode": self.input_mode,
            "group_by": self.group_by,
            "primary_package_pad_view": self.primary_package_pad_view,
            "primary_land_pad_view": self.primary_land_pad_view,
            "lateral_views": list(self.lateral_views),
            "conflict_abs_tol": self.conflict_abs_tol,
            "conflict_rel_tol": self.conflict_rel_tol,
            "ignore_lateral_height": self.ignore_lateral_height,
            "preserve_raw_view": self.preserve_raw_view,
        }


def normalize_view(view: str, options: MultiviewOptions | None = None) -> str:
    raw = str(view or "").strip().lower()
    options = options or MultiviewOptions()
    if raw in {item.lower() for item in options.lateral_views}:
        return "lateral"
    if raw in {"lead", "land_detail"}:
        return "lead_detail"
    return raw or "unknown"


def resolve_graph_root(graph_input: Path) -> Path:
    graph_input = graph_input.resolve()
    if graph_input.is_file():
        return graph_input.parent
    if (graph_input / "graphs").is_dir():
        return graph_input / "graphs"
    return graph_input


def load_info_view_map(dataset_root: Path | None) -> dict[tuple[str, str], str]:
    if dataset_root is None or not dataset_root.exists():
        return {}
    view_map: dict[tuple[str, str], str] = {}
    for info_path in dataset_root.glob("*/extract_image/info.json"):
        part_number = info_path.parent.parent.name
        try:
            payload = json.loads(info_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for item in payload.get("images") or []:
            file_name = str(item.get("file_name") or "")
            view = str(item.get("view") or "")
            if file_name and view:
                view_map[(part_number, Path(file_name).stem)] = view
    return view_map


def read_graphs(graph_input: Path, dataset_root: Path | None = None, limit: int = 0) -> list[dict[str, Any]]:
    graph_root = resolve_graph_root(graph_input)
    files = sorted(graph_root.rglob("*.package_graph.json"))
    if limit > 0:
        files = files[:limit]
    info_views = load_info_view_map(dataset_root)
    graphs: list[dict[str, Any]] = []
    for path in files:
        graph = json.loads(path.read_text(encoding="utf-8"))
        part_number = str(graph.get("part_number") or path.parent.name)
        raw_view = str(graph.get("view") or "")
        if not raw_view:
            raw_view = info_views.get((part_number, path.stem.replace(".package_graph", "")), "")
        graph["_graph_path"] = str(path)
        graph["_part_number"] = part_number
        graph["_raw_view"] = raw_view.lower()
        graphs.append(graph)
    return graphs


def integrate_graphs(
    graph_input: Path,
    output_root: Path,
    *,
    dataset_root: Path | None = None,
    options: MultiviewOptions | None = None,
    limit: int = 0,
) -> dict[str, Any]:
    options = options or MultiviewOptions()
    output_root.mkdir(parents=True, exist_ok=True)
    graphs = read_graphs(graph_input, dataset_root=dataset_root, limit=limit)
    by_part: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for graph in graphs:
        by_part[str(graph.get("_part_number") or "")].append(graph)

    summaries = []
    raw_view_counts: Counter[str] = Counter()
    canonical_view_counts: Counter[str] = Counter()
    dimension_value_source_counts: Counter[str] = Counter()
    dimension_role_counts: Counter[str] = Counter()
    dimension_canonical_view_counts: Counter[str] = Counter()
    evidence_type_counts: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    missing_graph_parts: list[str] = []
    for part_number in sorted(by_part):
        canonical = integrate_part(part_number, by_part[part_number], options, dataset_root=dataset_root)
        part_dir = output_root / "parts" / slugify(part_number)
        write_part_outputs(part_dir, canonical)
        summary = canonical["summary"]
        summaries.append(summary)
        raw_view_counts.update(summary["raw_view_counts"])
        canonical_view_counts.update(summary["canonical_view_counts"])
        evidence_summary = summary.get("evidence_summary") or {}
        dimension_value_source_counts.update(evidence_summary.get("dimension_value_source_counts") or {})
        dimension_role_counts.update(evidence_summary.get("dimension_role_counts") or {})
        dimension_canonical_view_counts.update(evidence_summary.get("dimension_canonical_view_counts") or {})
        evidence_type_counts.update(evidence_summary.get("evidence_type_counts") or {})
        risk_counts[summary["risk_level"]] += 1
        status_counts[str(summary.get("status") or "canonical")] += 1

    if dataset_root is not None and dataset_root.exists():
        for part_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
            part_number = part_dir.name
            if part_number in by_part:
                continue
            canonical = missing_graph_canonical(part_number, dataset_root=dataset_root)
            write_part_outputs(output_root / "parts" / slugify(part_number), canonical)
            summary = canonical["summary"]
            summaries.append(summary)
            evidence_summary = summary.get("evidence_summary") or {}
            evidence_type_counts.update(evidence_summary.get("evidence_type_counts") or {})
            risk_counts[summary["risk_level"]] += 1
            status_counts[str(summary.get("status") or "canonical")] += 1
            if str(summary.get("status") or "") == "missing_graphs":
                missing_graph_parts.append(part_number)

    graph_based_parts = len(by_part)
    canonical_parts = len(summaries) - len(missing_graph_parts)
    payload = {
        "output_root": str(output_root),
        "summary_path": str(output_root / "summary.json"),
        "graph_input": str(resolve_graph_root(graph_input)),
        "dataset_root": str(dataset_root) if dataset_root else None,
        "options": options.to_dict(),
        "total_graphs": len(graphs),
        "total_parts": len(summaries),
        "part_outputs": len(summaries),
        "canonical_parts": canonical_parts,
        "graph_based_parts": graph_based_parts,
        "scan_result_only_parts": 0,
        "scan_result_only_part_numbers": [],
        "failure_reason_parts": len(missing_graph_parts),
        "missing_graph_parts": len(missing_graph_parts),
        "missing_graph_part_numbers": missing_graph_parts,
        "raw_view_counts": dict(sorted(raw_view_counts.items())),
        "canonical_view_counts": dict(sorted(canonical_view_counts.items())),
        "dimension_value_source_counts": dict(sorted(dimension_value_source_counts.items())),
        "dimension_role_counts": dict(sorted(dimension_role_counts.items())),
        "dimension_canonical_view_counts": dict(sorted(dimension_canonical_view_counts.items())),
        "evidence_type_counts": dict(sorted(evidence_type_counts.items())),
        "risk_counts": dict(sorted(risk_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "parts": summaries,
    }
    (output_root / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def integrate_part(
    part_number: str,
    graphs: list[dict[str, Any]],
    options: MultiviewOptions,
    *,
    dataset_root: Path | None = None,
) -> dict[str, Any]:
    ordered = sorted(graphs, key=lambda graph: (view_priority(str(graph.get("_raw_view") or "")), str(graph.get("_graph_path"))))
    graph_evidence_refs = [build_evidence_ref(graph, options) for graph in ordered]
    evidence_refs = graph_evidence_refs
    raw_views = sorted({item["raw_view"] for item in graph_evidence_refs if item["raw_view"]})
    canonical_views = sorted({item["canonical_view"] for item in graph_evidence_refs if item["canonical_view"]})
    raw_view_counts = Counter(item["raw_view"] for item in graph_evidence_refs if item["raw_view"])
    canonical_view_counts = Counter(item["canonical_view"] for item in graph_evidence_refs if item["canonical_view"])

    package_selection = select_graph_with_metadata(
        ordered,
        unique_view_order([options.primary_package_pad_view, "top", "bottom"]),
        primary_view=options.primary_package_pad_view,
        role="package_pads",
    )
    land_selection = select_graph_with_metadata(
        ordered,
        [options.primary_land_pad_view],
        primary_view=options.primary_land_pad_view,
        role="land_pads",
    )
    package_selection, land_selection = apply_secondary_bottom_role_split(
        ordered,
        package_selection,
        land_selection,
        options,
    )
    land_graph = land_selection["graph"]
    land_pads = extract_objects(land_graph, "land_pad", options) if land_graph else []
    land_pads, oversized_filtered_land_pads = filter_oversized_pad_like_outliers(land_pads)
    package_selection = prefer_package_source_with_land_count_consistency(
        ordered,
        package_selection,
        land_pads,
        options,
    )
    package_graph = package_selection["graph"]
    package_pads = extract_objects(package_graph, "package_pad", options) if package_graph else []
    package_pads, oversized_filtered_package_pads = filter_oversized_pad_like_outliers(package_pads)
    body_dimension_filtered_package_pads: list[dict[str, Any]] = []
    if str(package_selection.get("fallback_reason") or "") == OVERDENSE_TOP_LAND_COUNT_FALLBACK_REASON:
        package_pads, body_dimension_filtered_package_pads = filter_body_dimension_package_pads(package_pads, package_graph)
    outline_selection = (
        source_selection_from_graph(package_graph, role="outline_2d", selected_from="package_pads")
        if package_graph
        else select_graph_with_metadata(ordered, ["top", "bottom", "land"], primary_view="top", role="outline_2d")
    )
    outline = extract_outline(outline_selection["graph"], options)
    package_pads = regularize_two_column_package_pad_x_geometry(package_pads, outline, package_graph)
    package_pads, filtered_package_pads = filter_remote_detail_inset_package_pads(package_pads, outline)
    filtered_package_pads = oversized_filtered_package_pads + body_dimension_filtered_package_pads + filtered_package_pads
    land_pads, filtered_land_pads = filter_remote_detail_land_pads(land_pads)
    filtered_land_pads = oversized_filtered_land_pads + filtered_land_pads
    lead_contacts = [
        obj
        for graph in ordered
        if normalize_view(str(graph.get("_raw_view") or graph.get("view") or ""), options) in {"lateral", "lead_detail"}
        for obj in extract_objects(graph, "lead_contact", options)
    ]

    accepted_dimensions: list[dict[str, Any]] = []
    ignored_evidence: list[dict[str, Any]] = []
    for graph in ordered:
        for dim in graph.get("dimensions") or []:
            if str(dim.get("status") or "") != "accepted" and not lead_edge_to_center_contact_dimension(dim, graph, options):
                continue
            enriched = enrich_dimension(dim, graph, options)
            if should_ignore_lateral_height(enriched, options):
                enriched["ignored_reason"] = "lateral_height_or_vertical_dimension"
                ignored_evidence.append(enriched)
                continue
            accepted_dimensions.append(enriched)

    lead_pads = synthesize_lead_pads(package_pads, outline, accepted_dimensions, package_graph)
    land_detail_graphs = [
        graph for graph in ordered if str(graph.get("_raw_view") or graph.get("view") or "").lower() == "land_detail"
    ]
    inner_land_pads = synthesize_inner_land_pads(land_pads, land_detail_graphs, options)
    multiview_overlay = build_multiview_overlay_payload(ordered, lead_pads, inner_land_pads, options)
    conflicts = detect_conflicts(accepted_dimensions, options)
    match_report = match_package_and_land_pads(package_pads, land_pads)
    if match_report["status"] != "matched" and match_report["status"] != "not_applicable":
        conflicts.append(match_report)
    table_evidence_refs = build_table_evidence_refs(part_number, dataset_root, accepted_dimensions + ignored_evidence)
    evidence_refs.extend(table_evidence_refs)

    missing_views = [view for view in EXPECTED_CANONICAL_VIEWS if view not in canonical_views]
    active_conflict_count = len(active_conflicts(conflicts))
    risk_score = score_part(conflicts, missing_views)
    risk_level = risk_level_for_score(risk_score)
    risk_reasons = risk_reasons_for_part(conflicts, missing_views)
    evidence_summary = build_evidence_summary(evidence_refs, accepted_dimensions, ignored_evidence)

    source_selection = {
        "package_pads": public_source_selection(package_selection),
        "land_pads": public_source_selection(land_selection),
        "outline_2d": public_source_selection(outline_selection),
    }

    canonical = {
        "part_number": part_number,
        "status": "canonical",
        "source_views": raw_views,
        "canonical_source_views": canonical_views,
        "source_selection": source_selection,
        "outline_2d": outline,
        "package_pads": package_pads,
        "filtered_package_pads": filtered_package_pads,
        "land_pads": land_pads,
        "filtered_land_pads": filtered_land_pads,
        "lead_contacts": lead_contacts,
        "lead_pads": lead_pads,
        "inner_land_pads": inner_land_pads,
        "multiview_overlay": multiview_overlay,
        "dimensions": accepted_dimensions,
        "evidence_refs": evidence_refs,
        "ignored_evidence": ignored_evidence,
        "missing_canonical_views": missing_views,
        "pad_matching": match_report,
        "conflicts": conflicts,
        "evidence_summary": evidence_summary,
    }
    canonical["summary"] = {
        "part_number": part_number,
        "part_dir": "",
        "status": "canonical",
        "source_views": raw_views,
        "canonical_source_views": canonical_views,
        "raw_view_counts": dict(sorted(raw_view_counts.items())),
        "canonical_view_counts": dict(sorted(canonical_view_counts.items())),
        "graph_count": len(ordered),
        "package_pad_count": len(package_pads),
        "filtered_package_pad_count": len(filtered_package_pads),
        "land_pad_count": len(land_pads),
        "filtered_land_pad_count": len(filtered_land_pads),
        "lead_contact_count": len(lead_contacts),
        "lead_pad_count": len(lead_pads),
        "inner_land_pad_count": len(inner_land_pads),
        "multiview_overlay_layer_count": len(multiview_overlay.get("layers") or []),
        "multiview_overlay_extra_object_count": len(multiview_overlay.get("extra_objects") or []),
        "dimension_count": len(accepted_dimensions),
        "ignored_evidence_count": len(ignored_evidence),
        "conflict_count": active_conflict_count,
        "missing_canonical_views": missing_views,
        "source_selection": source_selection,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_reasons": risk_reasons,
        "scan_result_path": "",
        "has_scan_result": False,
        "table_evidence_count": len(table_evidence_refs),
        "evidence_summary": evidence_summary,
    }
    return canonical


def missing_graph_canonical(part_number: str, *, dataset_root: Path | None) -> dict[str, Any]:
    evidence_refs = []
    missing_views = list(EXPECTED_CANONICAL_VIEWS)
    risk_score = 100.0
    failure_reason = "no_package_graph_for_part"
    canonical = {
        "part_number": part_number,
        "status": "missing_graphs",
        "failure_reason": failure_reason,
        "source_views": [],
        "canonical_source_views": [],
        "outline_2d": {},
        "package_pads": [],
        "land_pads": [],
        "filtered_land_pads": [],
        "lead_contacts": [],
        "lead_pads": [],
        "inner_land_pads": [],
        "dimensions": [],
        "evidence_refs": evidence_refs,
        "ignored_evidence": [],
        "missing_canonical_views": missing_views,
        "pad_matching": {"status": "not_applicable", "reason": "missing_package_graphs"},
        "conflicts": [{"status": "missing_graphs", "reason": failure_reason}],
        "evidence_summary": build_evidence_summary(evidence_refs, [], []),
    }
    canonical["summary"] = {
        "part_number": part_number,
        "part_dir": "",
        "status": "missing_graphs",
        "failure_reason": failure_reason,
        "source_views": [],
        "canonical_source_views": [],
        "raw_view_counts": {},
        "canonical_view_counts": {},
        "graph_count": 0,
        "package_pad_count": 0,
        "land_pad_count": 0,
        "filtered_land_pad_count": 0,
        "lead_contact_count": 0,
        "lead_pad_count": 0,
        "inner_land_pad_count": 0,
        "dimension_count": 0,
        "ignored_evidence_count": 0,
        "conflict_count": 1,
        "missing_canonical_views": missing_views,
        "risk_score": risk_score,
        "risk_level": "high",
        "risk_reasons": ["missing package graph reconstruction for this part"],
        "scan_result_path": "",
        "has_scan_result": False,
        "evidence_summary": canonical["evidence_summary"],
    }
    return canonical


def build_evidence_ref(graph: dict[str, Any], options: MultiviewOptions) -> dict[str, Any]:
    raw_view = str(graph.get("_raw_view") or graph.get("view") or "").lower()
    labels = [object_label(obj) for obj in graph.get("objects") or []]
    label_counts = Counter(labels)
    pad_like_counts = evidence_pad_like_geometry_counts(graph)
    return {
        "evidence_type": "package_graph",
        "part_number": str(graph.get("_part_number") or graph.get("part_number") or ""),
        "raw_view": raw_view,
        "canonical_view": normalize_view(raw_view, options),
        "graph_path": str(graph.get("_graph_path") or ""),
        "annotation_path": str(graph.get("annotation_path") or ""),
        "image_path": str((graph.get("image") or {}).get("path") or ""),
        "object_count": len(graph.get("objects") or []),
        "pad_like_count": pad_like_counts["pad_like_count"],
        "terminal_pad_like_count": pad_like_counts["terminal_pad_like_count"],
        "thermal_pad_like_count": pad_like_counts["thermal_pad_like_count"],
        "object_label_counts": dict(sorted(label_counts.items())),
        "dimension_count": len(graph.get("dimensions") or []),
    }


def evidence_pad_like_geometry_counts(graph: dict[str, Any]) -> dict[str, int]:
    pads = []
    for obj in graph.get("objects") or []:
        if object_label(obj) not in PAD_LABELS:
            continue
        bbox = obj.get("bbox_reconstructed") or obj.get("bbox") or []
        if len(bbox) < 4:
            pads.append({"bbox": []})
            continue
        pads.append({"bbox": [float(value) for value in bbox[:4]]})
    thermal_count = thermal_like_pad_count(pads)
    pad_count = len(pads)
    return {
        "pad_like_count": pad_count,
        "terminal_pad_like_count": max(0, pad_count - thermal_count),
        "thermal_pad_like_count": thermal_count,
    }


def thermal_like_pad_count(pads: list[dict[str, Any]]) -> int:
    """Count large internal pads for evidence-only terminal counts.

    Coordinate system: reconstructed graph bbox coordinates in pixels.  The
    heuristic is relative to the graph's own pad bbox extent and only affects
    count evidence; it does not remove objects from unified multiview layers.
    """
    boxes = [pad.get("bbox") for pad in pads if len(pad.get("bbox") or []) >= 4]
    if len(boxes) < 5:
        return 0
    areas = sorted(abs((box[2] - box[0]) * (box[3] - box[1])) for box in boxes)
    median_area = areas[len(areas) // 2]
    if median_area <= 0:
        return 0
    widths = sorted(abs(box[2] - box[0]) for box in boxes)
    heights = sorted(abs(box[3] - box[1]) for box in boxes)
    median_width = widths[len(widths) // 2] if widths else 0.0
    median_height = heights[len(heights) // 2] if heights else 0.0
    extent = [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]
    width = extent[2] - extent[0]
    height = extent[3] - extent[1]
    if width <= 0 or height <= 0:
        return 0
    large_central_found = False
    candidates: list[tuple[bool, bool]] = []
    for box in boxes:
        area = abs((box[2] - box[0]) * (box[3] - box[1]))
        box_width = abs(box[2] - box[0])
        box_height = abs(box[3] - box[1])
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        x_ratio = (cx - extent[0]) / width
        y_ratio = (cy - extent[1]) / height
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
        candidates.append((central_nonterminal, side_internal_larger))
    count = 0
    for central_nonterminal, side_internal_larger in candidates:
        if central_nonterminal or (large_central_found and side_internal_larger):
            count += 1
    return count


def build_table_evidence_refs(
    part_number: str,
    dataset_root: Path | None,
    dimensions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if dataset_root is None:
        return []
    if not any(str(dim.get("value_source") or "") == "table_lookup" for dim in dimensions):
        return []
    table_dir = dataset_root / part_number / "table"
    if not table_dir.exists():
        return [
            {
                "evidence_type": "table_lookup_files",
                "part_number": part_number,
                "table_dir": str(table_dir),
                "status": "missing_table_dir",
                "files": [],
            }
        ]
    files = [
        str(path)
        for path in sorted(table_dir.iterdir())
        if path.is_file() and path.suffix.lower() in {".xlsx", ".xls", ".csv", ".png", ".jpg", ".jpeg"}
    ]
    return [
        {
            "evidence_type": "table_lookup_files",
            "part_number": part_number,
            "table_dir": str(table_dir),
            "status": "available" if files else "empty_table_dir",
            "files": files,
        }
    ]


def build_evidence_summary(
    evidence_refs: list[dict[str, Any]],
    dimensions: list[dict[str, Any]],
    ignored_evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "evidence_ref_count": len(evidence_refs),
        "evidence_type_counts": count_by_field(evidence_refs, "evidence_type"),
        "dimension_count": len(dimensions),
        "ignored_evidence_count": len(ignored_evidence),
        "dimension_value_source_counts": count_by_field(dimensions, "value_source"),
        "dimension_role_counts": count_by_field(dimensions, "role"),
        "dimension_canonical_view_counts": count_by_field(dimensions, "canonical_view"),
        "ignored_evidence_reason_counts": count_by_field(ignored_evidence, "ignored_reason"),
        "table_lookup_dimension_count": sum(1 for dim in dimensions if str(dim.get("value_source") or "") == "table_lookup"),
        "text_parser_dimension_count": sum(1 for dim in dimensions if str(dim.get("value_source") or "") == "text_parser"),
        "scan_result_format_ref_count": sum(1 for ref in evidence_refs if str(ref.get("evidence_type") or "") == "scan_result_format"),
        "table_lookup_file_ref_count": sum(1 for ref in evidence_refs if str(ref.get("evidence_type") or "") == "table_lookup_files"),
        "source_graph_count": len({str(dim.get("source_graph") or "") for dim in dimensions if dim.get("source_graph")}),
    }


def count_by_field(items: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in items:
        value = str(item.get(field) or "unknown")
        counts[value] += 1
    return dict(sorted(counts.items()))


def terminal_pad_count(package_pads: list[dict[str, Any]]) -> int:
    thermal_count = thermal_like_pad_count(package_pads)
    return max(0, len(package_pads) - thermal_count)


def select_graph(graphs: list[dict[str, Any]], preferred_views: list[str]) -> dict[str, Any] | None:
    for view in preferred_views:
        candidates = [
            graph
            for graph in graphs
            if str(graph.get("_raw_view") or graph.get("view") or "").lower() == view
        ]
        if candidates:
            return max(candidates, key=graph_selection_key)
    return None


def unique_view_order(views: list[str]) -> list[str]:
    ordered = []
    seen = set()
    for view in views:
        normalized = str(view or "").lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def select_graph_with_metadata(
    graphs: list[dict[str, Any]],
    preferred_views: list[str],
    *,
    primary_view: str,
    role: str,
) -> dict[str, Any]:
    primary_exists = any(
        str(graph.get("_raw_view") or graph.get("view") or "").lower() == primary_view for graph in graphs
    )
    for view in preferred_views:
        candidates = [
            graph
            for graph in graphs
            if str(graph.get("_raw_view") or graph.get("view") or "").lower() == view
        ]
        if candidates:
            graph = max(candidates, key=graph_selection_key)
            fallback_reason = ""
            if role == "package_pads" and view == primary_view:
                fallback_graph = best_fallback_graph(graphs, preferred_views, start_after=view)
                if should_use_package_pad_fallback(graph, fallback_graph):
                    graph = fallback_graph
                    fallback_reason = "primary_package_pad_layout_sparse"
            raw_view = str(graph.get("_raw_view") or graph.get("view") or "").lower()
            return {
                "role": role,
                "graph": graph,
                "selected_raw_view": raw_view,
                "selected_canonical_view": normalize_view(raw_view),
                "primary_view": primary_view,
                "preferred_views": preferred_views,
                "used_fallback": raw_view != primary_view,
                "missing_primary": not primary_exists,
                "fallback_reason": fallback_reason,
                "graph_path": str(graph.get("_graph_path") or ""),
                "object_count": len(graph.get("objects") or []),
                "dimension_count": len(graph.get("dimensions") or []),
            }
    return {
        "role": role,
        "graph": None,
        "selected_raw_view": "",
        "selected_canonical_view": "",
        "primary_view": primary_view,
        "preferred_views": preferred_views,
        "used_fallback": False,
        "missing_primary": True,
        "fallback_reason": "",
        "graph_path": "",
        "object_count": 0,
        "dimension_count": 0,
    }


def source_selection_from_graph(graph: dict[str, Any] | None, *, role: str, selected_from: str) -> dict[str, Any]:
    if not graph:
        return {
            "role": role,
            "graph": None,
            "selected_raw_view": "",
            "selected_canonical_view": "",
            "primary_view": "",
            "preferred_views": [],
            "used_fallback": False,
            "missing_primary": True,
            "fallback_reason": "",
            "graph_path": "",
            "object_count": 0,
            "dimension_count": 0,
            "selected_from": selected_from,
        }
    raw_view = str(graph.get("_raw_view") or graph.get("view") or "").lower()
    return {
        "role": role,
        "graph": graph,
        "selected_raw_view": raw_view,
        "selected_canonical_view": normalize_view(raw_view),
        "primary_view": raw_view,
        "preferred_views": [raw_view],
        "used_fallback": False,
        "missing_primary": False,
        "fallback_reason": "",
        "graph_path": str(graph.get("_graph_path") or ""),
        "object_count": len(graph.get("objects") or []),
        "dimension_count": len(graph.get("dimensions") or []),
        "selected_from": selected_from,
    }


def public_source_selection(selection: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in selection.items() if key != "graph"}


def apply_secondary_bottom_role_split(
    graphs: list[dict[str, Any]],
    package_selection: dict[str, Any],
    land_selection: dict[str, Any],
    options: MultiviewOptions,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split multiple bottom diagrams into package and footprint roles.

    Coordinates and labels are from per-image reconstructed package graphs.
    This rule only activates when one bottom graph is shape-heavy
    (dshape/circle-like package terminals) and another is rect-heavy
    (footprint/land-like terminals). It keeps explicit land views primary.
    """
    pair = secondary_bottom_package_land_pair(graphs)
    if pair is None:
        return package_selection, land_selection

    package_graph, land_graph = pair
    selected_package = package_selection.get("graph")
    if selected_package is not package_graph:
        package_selection = selection_for_graph(
            package_graph,
            role="package_pads",
            primary_view=options.primary_package_pad_view,
            preferred_views=unique_view_order([options.primary_package_pad_view, "top", "bottom"]),
            used_fallback=True,
            missing_primary=False,
            fallback_reason="secondary_bottom_shape_layout_selected_for_package_pads",
        )

    if land_selection.get("graph") is None:
        land_selection = selection_for_graph(
            land_graph,
            role="land_pads",
            primary_view=options.primary_land_pad_view,
            preferred_views=[options.primary_land_pad_view],
            used_fallback=True,
            missing_primary=True,
            fallback_reason="missing_land_view_used_secondary_bottom_rect_layout",
        )
    return package_selection, land_selection


def secondary_bottom_package_land_pair(graphs: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    bottom_graphs = [
        graph
        for graph in graphs
        if str(graph.get("_raw_view") or graph.get("view") or "").lower() == "bottom"
    ]
    if len(bottom_graphs) < 2:
        return None

    package_candidates = [graph for graph in bottom_graphs if is_shape_heavy_bottom_graph(graph)]
    land_candidates = [graph for graph in bottom_graphs if is_rect_heavy_bottom_graph(graph)]
    pairs = [
        (package_graph, land_graph)
        for package_graph in package_candidates
        for land_graph in land_candidates
        if graph_identity(package_graph) != graph_identity(land_graph)
    ]
    if not pairs:
        return None
    return max(pairs, key=secondary_bottom_pair_key)


def secondary_bottom_pair_key(pair: tuple[dict[str, Any], dict[str, Any]]) -> tuple[int, int, int, str, str]:
    package_graph, land_graph = pair
    package_profile = graph_pad_shape_profile(package_graph)
    land_profile = graph_pad_shape_profile(land_graph)
    return (
        package_profile["shape_like_count"],
        land_profile["rect_like_count"],
        min(package_profile["pad_like_count"], land_profile["pad_like_count"]),
        graph_identity(package_graph),
        graph_identity(land_graph),
    )


def is_shape_heavy_bottom_graph(graph: dict[str, Any]) -> bool:
    profile = graph_pad_shape_profile(graph)
    pad_count = profile["pad_like_count"]
    if pad_count < 4:
        return False
    return profile["shape_like_count"] >= max(4, math.ceil(pad_count * 0.5))


def is_rect_heavy_bottom_graph(graph: dict[str, Any]) -> bool:
    profile = graph_pad_shape_profile(graph)
    pad_count = profile["pad_like_count"]
    if pad_count < 4:
        return False
    return profile["rect_like_count"] >= math.ceil(pad_count * 0.8)


def graph_pad_shape_profile(graph: dict[str, Any]) -> dict[str, int]:
    pad_like_count = 0
    rect_like_count = 0
    shape_like_count = 0
    for obj in graph.get("objects") or []:
        if object_label(obj) not in PAD_LABELS:
            continue
        pad_like_count += 1
        label_text = f"{obj.get('label') or ''} {obj.get('source_label') or ''}".lower()
        if "rect" in label_text:
            rect_like_count += 1
        if "dshape" in label_text or "circle" in label_text:
            shape_like_count += 1
    return {
        "pad_like_count": pad_like_count,
        "rect_like_count": rect_like_count,
        "shape_like_count": shape_like_count,
    }


def graph_identity(graph: dict[str, Any]) -> str:
    return str(graph.get("_graph_path") or id(graph))


def selection_for_graph(
    graph: dict[str, Any],
    *,
    role: str,
    primary_view: str,
    preferred_views: list[str],
    used_fallback: bool,
    missing_primary: bool,
    fallback_reason: str,
) -> dict[str, Any]:
    raw_view = str(graph.get("_raw_view") or graph.get("view") or "").lower()
    return {
        "role": role,
        "graph": graph,
        "selected_raw_view": raw_view,
        "selected_canonical_view": normalize_view(raw_view),
        "primary_view": primary_view,
        "preferred_views": preferred_views,
        "used_fallback": used_fallback,
        "missing_primary": missing_primary,
        "fallback_reason": fallback_reason,
        "graph_path": str(graph.get("_graph_path") or ""),
        "object_count": len(graph.get("objects") or []),
        "dimension_count": len(graph.get("dimensions") or []),
    }


def graph_selection_key(graph: dict[str, Any]) -> tuple[int, int, str]:
    labels = [object_label(obj) for obj in graph.get("objects") or []]
    pad_like_count = sum(1 for label in labels if label in PAD_LABELS)
    outline_count = sum(1 for label in labels if label in OUTLINE_LABELS)
    return (pad_like_count, outline_count, str(graph.get("_graph_path") or ""))


def best_fallback_graph(
    graphs: list[dict[str, Any]],
    preferred_views: list[str],
    *,
    start_after: str,
) -> dict[str, Any] | None:
    try:
        start_index = preferred_views.index(start_after) + 1
    except ValueError:
        start_index = 0
    for view in preferred_views[start_index:]:
        candidates = [
            graph
            for graph in graphs
            if str(graph.get("_raw_view") or graph.get("view") or "").lower() == view
        ]
        if candidates:
            return max(candidates, key=graph_selection_key)
    return None


def should_use_package_pad_fallback(primary_graph: dict[str, Any], fallback_graph: dict[str, Any] | None) -> bool:
    if not fallback_graph:
        return False
    primary_pad_count, primary_outline_count = graph_pad_outline_counts(primary_graph)
    fallback_pad_count, fallback_outline_count = graph_pad_outline_counts(fallback_graph)
    return bool(
        primary_pad_count <= 2
        and primary_outline_count == 0
        and fallback_pad_count >= 4
        and fallback_pad_count >= primary_pad_count * 2
        and fallback_outline_count > 0
    )


def prefer_package_source_with_land_count_consistency(
    graphs: list[dict[str, Any]],
    package_selection: dict[str, Any],
    land_pads: list[dict[str, Any]],
    options: MultiviewOptions,
) -> dict[str, Any]:
    """Prefer a top package source when bottom is an over-dense detail view.

    Coordinates are still graph coordinates at this stage. Counts are canonical
    object counts after local dedupe/merge.
    """
    package_graph = package_selection.get("graph")
    if not package_graph:
        return package_selection
    if str(package_selection.get("selected_raw_view") or "") != str(package_selection.get("primary_view") or ""):
        return package_selection
    if str(package_selection.get("selected_raw_view") or "") != "bottom":
        return package_selection
    land_count = len(land_pads)
    if land_count < 3:
        return package_selection
    preferred_views = [str(view) for view in package_selection.get("preferred_views") or []]
    fallback_graph = best_fallback_graph(graphs, preferred_views, start_after="bottom")
    if not fallback_graph:
        return package_selection
    if str(fallback_graph.get("_raw_view") or fallback_graph.get("view") or "").lower() != "top":
        return package_selection
    primary_count = terminal_pad_count(package_pads_for_source_selection(package_graph, options))
    fallback_count = terminal_pad_count(package_pads_for_source_selection(fallback_graph, options))
    if fallback_count != land_count:
        return package_selection
    dense_threshold = max(land_count + 4, int(math.ceil(land_count * 2.0)))
    if primary_count < dense_threshold:
        return package_selection
    return selection_for_graph(
        fallback_graph,
        role=str(package_selection.get("role") or "package_pads"),
        primary_view=str(package_selection.get("primary_view") or "bottom"),
        preferred_views=preferred_views,
        used_fallback=True,
        missing_primary=bool(package_selection.get("missing_primary")),
        fallback_reason=OVERDENSE_TOP_LAND_COUNT_FALLBACK_REASON,
    )


def package_pads_for_source_selection(graph: dict[str, Any], options: MultiviewOptions) -> list[dict[str, Any]]:
    pads = extract_objects(graph, "package_pad", options)
    kept, _ = filter_body_dimension_package_pads(pads, graph)
    return kept


def graph_pad_outline_counts(graph: dict[str, Any]) -> tuple[int, int]:
    labels = [object_label(obj) for obj in graph.get("objects") or []]
    pad_like_count = sum(1 for label in labels if label in PAD_LABELS)
    outline_count = sum(1 for label in labels if label in OUTLINE_LABELS)
    return pad_like_count, outline_count


def view_priority(view: str) -> int:
    order = {"bottom": 0, "land": 1, "top": 2, "side": 3, "front": 4, "lead": 5, "land_detail": 6}
    return order.get(str(view or "").lower(), 99)


def extract_outline(graph: dict[str, Any] | None, options: MultiviewOptions) -> dict[str, Any]:
    if not graph:
        return {}
    for obj in graph.get("objects") or []:
        if object_label(obj) in OUTLINE_LABELS:
            return object_payload(obj, graph, "outline_2d", options)
    return {}


def extract_objects(graph: dict[str, Any], role: str, options: MultiviewOptions) -> list[dict[str, Any]]:
    objects = []
    for obj in graph.get("objects") or []:
        label = object_label(obj)
        if label in OUTLINE_LABELS:
            continue
        if role == "lead_contact" and label not in PAD_LABELS | {"lead"}:
            continue
        if role in {"package_pad", "land_pad"} and label not in PAD_LABELS:
            continue
        objects.append(object_payload(obj, graph, role, options))
    return dedupe_objects_by_bbox(objects)


def filter_remote_detail_inset_package_pads(
    package_pads: list[dict[str, Any]],
    outline: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove remote circle-only detail inset pads from dense package layouts.

    Coordinates are reconstructed package-graph image coordinates. This is a
    conservative filter for BGA-like bottom views: when almost all package pads
    are circle-like and only a few centers fall far outside the selected outline,
    those outside pads are treated as detail inset evidence instead of package
    pads. Rect/lead-heavy layouts are intentionally left unchanged.
    """
    if len(package_pads) < 20:
        return package_pads, []
    outline_bbox = normalized_bbox(outline)
    if outline_bbox is None:
        return package_pads, []
    circle_like_count = sum(1 for pad in package_pads if is_circle_like_pad(pad))
    if circle_like_count / len(package_pads) < 0.8:
        return package_pads, []

    expanded = expand_bbox(outline_bbox, rel_margin=0.10)
    inside = []
    outside = []
    for pad in package_pads:
        center = bbox_center(pad)
        if center is not None and point_in_bbox(center, expanded):
            inside.append(pad)
        else:
            outside.append(pad)
    if not outside:
        return package_pads, []
    max_detail_count = max(5, math.ceil(len(package_pads) * 0.10))
    if len(outside) > max_detail_count or len(inside) < 4:
        return package_pads, []
    filtered = [dict(pad, filtered_reason="remote_detail_inset_outside_outline") for pad in outside]
    return sorted(inside, key=object_sort_key), sorted(filtered, key=object_sort_key)


def filter_body_dimension_package_pads(
    package_pads: list[dict[str, Any]],
    graph: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter pad-like objects directly targeted by package body dimensions.

    Coordinates are graph/reconstruction coordinates. The rule uses source
    graph dimension symbols only.
    """
    target_ids = body_size_dimension_target_ids(graph)
    if not target_ids:
        return package_pads, []
    kept = []
    filtered = []
    for pad in package_pads:
        if pad.get("source_object_id") in target_ids:
            filtered.append(dict(pad, filtered_reason="body_dimension_target"))
        else:
            kept.append(pad)
    return kept, filtered


def filter_oversized_pad_like_outliers(
    pads: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove obvious oversized pad-like objects from dense package/land grids.

    Coordinates are reconstructed package-graph pixels. The rule is deliberately
    inactive for sparse layouts; it only removes a small minority whose area or
    side length is far larger than the median pad geometry.
    """
    if len(pads) < 8:
        return pads, []
    dims = [(pad, bbox_dimensions(pad)) for pad in pads]
    dims = [(pad, dim) for pad, dim in dims if dim is not None and dim[0] > 0.0 and dim[1] > 0.0]
    if len(dims) < 8:
        return pads, []
    widths = sorted(dim[0] for _, dim in dims)
    heights = sorted(dim[1] for _, dim in dims)
    areas = sorted(dim[0] * dim[1] for _, dim in dims)
    median_width = median_sorted(widths)
    median_height = median_sorted(heights)
    median_area = median_sorted(areas)
    if median_width <= 0.0 or median_height <= 0.0 or median_area <= 0.0:
        return pads, []

    oversized = []
    regular = []
    for pad in pads:
        dim = bbox_dimensions(pad)
        if dim is None:
            regular.append(pad)
            continue
        width, height = dim
        area = width * height
        is_outlier = area > median_area * 6.0 or width > median_width * 4.0 or height > median_height * 4.0
        if is_outlier:
            oversized.append(pad)
        else:
            regular.append(pad)
    if not oversized:
        return pads, []

    regular_centers = [bbox_center(pad) for pad in regular]
    regular_centers = [center for center in regular_centers if center is not None]
    if len(regular_centers) < 4:
        return pads, []

    xs = [center[0] for center in regular_centers]
    ys = [center[1] for center in regular_centers]
    center_frame = expand_bbox((min(xs), min(ys), max(xs), max(ys)), rel_margin=0.10)
    kept = regular[:]
    filtered = []
    for pad in oversized:
        center = bbox_center(pad)
        if center is not None and point_in_bbox(center, center_frame):
            kept.append(pad)
        else:
            filtered.append(dict(pad, filtered_reason="oversized_pad_like_outlier"))
    if not filtered:
        return pads, []
    max_filtered = max(2, math.ceil(len(pads) * 0.20))
    if len(filtered) > max_filtered or len(kept) < 4:
        return pads, []
    return sorted(kept, key=object_sort_key), sorted(filtered, key=object_sort_key)


def synthesize_lead_pads(
    package_pads: list[dict[str, Any]],
    outline: dict[str, Any],
    dimensions: list[dict[str, Any]],
    package_graph: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Derive lead-pad contact geometry from lateral/lead-detail dimensions.

    Coordinate system: output bboxes use the same graph/reconstruction
    coordinate system as `package_pads`. The dimension value is converted to
    reconstruction pixels using the selected package graph's axis_scale_x/y,
    where axis_scale is physical units per pixel.

    The lead contact length is applied to every terminal package pad.  The
    cross-axis width stays identical to the source package pad, while the
    radial length is replaced by the contact length and anchored to the outer
    side of the pad, extending toward the package center.
    """
    if not package_pads:
        return []
    contact_dimensions = lead_partial_dimensions(dimensions)
    if not contact_dimensions:
        return []

    outline_bbox = normalized_bbox(outline) or union_object_bbox(package_pads)
    if outline_bbox is None:
        return []
    center_x = (outline_bbox[0] + outline_bbox[2]) / 2.0
    center_y = (outline_bbox[1] + outline_bbox[3]) / 2.0

    unit_scales = lead_pad_coordinate_unit_scales(package_pads, package_graph)
    if not unit_scales:
        return []

    terminal_pads = terminal_package_pads_for_lead_synthesis(package_pads)
    left_right_only_layout = has_left_right_only_terminal_layout(terminal_pads, outline_bbox)
    terminal_band_layout = terminal_pad_band_layout(terminal_pads)
    contact_dimensions = select_lead_partial_dimensions_for_terminal_layout(
        contact_dimensions,
        terminal_band_layout=terminal_band_layout,
    )
    uniform_cross_axis_length = uniform_perimeter_terminal_cross_axis_length(
        terminal_pads,
        outline_bbox,
        terminal_band_layout=terminal_band_layout,
    )
    lead_pads = []
    for contact_dimension in contact_dimensions:
        contact_length = numeric(contact_dimension.get("value"))
        if contact_length is None or contact_length <= 0:
            continue
        base_semantics = str(contact_dimension.get("overlay_semantics") or "")
        for index, pad in enumerate(terminal_pads):
            bbox = normalized_bbox(pad)
            center = bbox_center(pad)
            if bbox is None or center is None:
                continue
            radial_axis = lead_pad_radial_axis(
                center=center,
                package_center=(center_x, center_y),
                terminal_band_layout=terminal_band_layout,
            )
            projection_axis = lead_partial_projection_axis(contact_dimension, radial_axis)
            force_pad_width_semantics = False
            pad_width_alignment = "center"
            if (
                terminal_band_layout == "top_bottom_rows"
                and base_semantics == "pad_width"
                and str(contact_dimension.get("raw_view") or "").strip().lower() == "front"
            ):
                projection_axis = "x"
                force_pad_width_semantics = True
            elif (
                left_right_only_layout
                and base_semantics == "pad_width"
                and str(contact_dimension.get("raw_view") or "").strip().lower() == "front"
                and radial_axis == "x"
            ):
                projection_axis = "x"
                force_pad_width_semantics = True
                pad_width_alignment = "left_right_outer_edge"
            unit_scale = unit_scales.get(projection_axis)
            if unit_scale is None or unit_scale <= 0:
                continue
            source_bbox = normalized_lead_source_bbox(
                bbox,
                radial_axis=radial_axis,
                uniform_cross_axis_length=uniform_cross_axis_length,
            )
            semantics = (
                "pad_width"
                if force_pad_width_semantics
                else effective_lead_partial_semantics(
                    dim=contact_dimension,
                    base_semantics=base_semantics,
                    projection_axis=projection_axis,
                    dimension_value=contact_length,
                    bbox=source_bbox,
                    unit_scale=unit_scale,
                )
            )
            length = contact_length / unit_scale
            if length <= 0:
                continue
            new_bbox = lead_partial_bbox(
                bbox=source_bbox,
                package_center=(center_x, center_y),
                package_frame=outline_bbox,
                projection_axis=projection_axis,
                length=length,
                semantics=semantics,
                pad_width_alignment=pad_width_alignment,
            )
            role = lead_partial_role(semantics)
            lead_pads.append(
                {
                    "source_object_id": pad.get("source_object_id"),
                    "role": role,
                    "label": role,
                    "source_label": pad.get("source_label"),
                    "raw_view": contact_dimension.get("raw_view"),
                    "canonical_view": contact_dimension.get("canonical_view"),
                    "bbox": normalized_bbox_list(new_bbox),
                    "source_graph": pad.get("source_graph"),
                    "source_type": "derived_partial_evidence_multiview",
                    "source_package_pad_id": pad.get("source_object_id"),
                    "source_package_pad_bbox": list(bbox),
                    "source_package_pad_index": index,
                    "lead_contact_length": contact_length,
                    "lead_contact_length_axis": str(contact_dimension.get("axis") or ""),
                    "lead_contact_length_source": lead_contact_dimension_ref(contact_dimension),
                    "partial_dimension_semantics": semantics,
                    "partial_dimension_base_semantics": base_semantics,
                    "radial_axis": radial_axis,
                    "projection_axis": projection_axis,
                    "coordinate_unit_scale": unit_scale,
                    **(
                        {
                            "lead_source_bbox": normalized_bbox_list(source_bbox),
                            "uniform_cross_axis_source": "terminal_minor_median",
                            "uniform_cross_axis_length": uniform_cross_axis_length,
                        }
                        if uniform_cross_axis_length is not None
                        else {}
                    ),
                }
            )
    return sorted(dedupe_objects_by_bbox(lead_pads), key=object_sort_key)


def select_lead_partial_dimensions_for_terminal_layout(
    dimensions: list[dict[str, Any]],
    *,
    terminal_band_layout: str,
) -> list[dict[str, Any]]:
    """Select unambiguous lateral partial dimensions for a terminal layout.

    Coordinates are not modified here.  For two-row packages, front-view
    left/right pad-width dimensions describe the terminal width along the row.
    When the same front graph supplies multiple accepted widths, keep the
    narrowest one before pad synthesis so concentric bbox merging cannot
    promote the terminal width to a wider body/lead-bend dimension.
    """
    if terminal_band_layout != "top_bottom_rows":
        return dimensions
    best_by_source: dict[tuple[str, str], tuple[float, int, dict[str, Any]]] = {}
    passthrough: list[tuple[int, dict[str, Any]]] = []
    for index, dim in enumerate(dimensions):
        raw_view = str(dim.get("raw_view") or "").strip().lower()
        semantics = str(dim.get("overlay_semantics") or "")
        if raw_view != "front" or semantics != "pad_width":
            passthrough.append((index, dim))
            continue
        value = numeric(dim.get("value"))
        if value is None or value <= 0:
            passthrough.append((index, dim))
            continue
        key = (str(dim.get("source_graph") or ""), semantics)
        current = best_by_source.get(key)
        if current is None or (value, index) < (current[0], current[1]):
            best_by_source[key] = (value, index, dim)
    selected = passthrough + [(index, dim) for _value, index, dim in best_by_source.values()]
    return [dim for _index, dim in sorted(selected, key=lambda item: item[0])]


def synthesize_inner_land_pads(
    land_pads: list[dict[str, Any]],
    land_detail_graphs: list[dict[str, Any]],
    options: MultiviewOptions,
) -> list[dict[str, Any]]:
    """Project land-detail inner pad evidence onto selected land pads.

    Coordinate system: output bboxes use the same reconstructed coordinate
    system as `land_pads`.  The land_detail graph describes one pad shape
    family.  Its inner pad template is applied only to matching land-pad
    shapes, and dimension-derived templates use an inward physical margin.
    """
    if not land_pads or not land_detail_graphs:
        return []
    templates = inner_land_pad_templates(land_detail_graphs, options)
    if not templates:
        return []

    inner_pads = []
    applicable_land_pads = inner_land_pad_applicable_land_pads(land_pads)
    for template_index, template in enumerate(templates):
        ratio = template["relative_bbox"]
        for land_index, land_pad in applicable_land_pads:
            if not inner_land_template_matches_land_pad(template, land_pad):
                continue
            land_bbox = normalized_bbox(land_pad)
            if land_bbox is None:
                continue
            inner_bbox = apply_relative_bbox_to_bbox(land_bbox, ratio)
            inner_pads.append(
                {
                    "source_object_id": land_pad.get("source_object_id"),
                    "role": "inner_land_pad",
                    "label": "inner_land_pad",
                    "source_label": land_pad.get("source_label"),
                    "raw_view": land_pad.get("raw_view"),
                    "canonical_view": land_pad.get("canonical_view"),
                    "bbox": normalized_bbox_list(inner_bbox),
                    "source_graph": land_pad.get("source_graph"),
                    "source_type": "derived_inner_land_pad",
                    "source_land_pad_id": land_pad.get("source_object_id"),
                    "source_land_pad_bbox": list(land_bbox),
                    "source_land_pad_index": land_index,
                    "inner_land_template_index": template_index,
                    "inner_land_pad_source": template["source"],
                }
            )
    return dedupe_inner_land_pads(sorted(inner_pads, key=object_sort_key))


def inner_land_pad_applicable_land_pads(land_pads: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    """Select land pads that may receive a land_detail inner inset.

    Inner land pads are only valid for perimeter/terminal land pads and circle
    pads. Large central rectangular pads, such as exposed thermal pads, are not
    land_detail inset targets.
    """
    if not land_pads:
        return []
    circle_items = [(index, pad) for index, pad in enumerate(land_pads) if pad_shape_family(pad) == "circle"]
    rect_items = [(index, pad) for index, pad in enumerate(land_pads) if pad_shape_family(pad) != "circle"]
    if len(rect_items) < 5:
        return [(index, pad) for index, pad in enumerate(land_pads)]
    terminal_rects = terminal_package_pads_for_lead_synthesis([pad for _index, pad in rect_items])
    terminal_rect_ids = {id(pad) for pad in terminal_rects}
    return circle_items + [(index, pad) for index, pad in rect_items if id(pad) in terminal_rect_ids]


def build_multiview_overlay_payload(
    graphs: list[dict[str, Any]],
    lead_pads: list[dict[str, Any]],
    inner_land_pads: list[dict[str, Any]],
    options: MultiviewOptions,
) -> dict[str, Any]:
    """Materialize multiview overlay coordinates for review rendering.

    Coordinate system: output bboxes are in dimension units, not source pixels.
    Each source layer is centered on its own pad-like display frame before
    storage. Review rendering may only multiply these coordinates by a display
    scale and translate to the SVG canvas center.
    """
    layers = []
    layer_by_graph: dict[str, dict[str, Any]] = {}
    for graph in graphs:
        layer = multiview_overlay_layer_from_graph(graph, options)
        if layer is None:
            continue
        layers.append(layer)
        layer_by_graph[str(layer.get("graph_path") or "")] = layer

    extra_objects = []
    for obj in list(lead_pads) + list(inner_land_pads):
        normalized = multiview_overlay_extra_object(obj, layer_by_graph)
        if normalized is not None:
            extra_objects.append(normalized)

    source_views = {str(graph.get("_raw_view") or graph.get("view") or "").strip().lower() for graph in graphs}
    rotation_summary = normalize_multiview_overlay_layer_rotations(layers, extra_objects, source_views=source_views)

    frames = [
        tuple(layer["normalized_frame"])
        for layer in layers
        if len(layer.get("normalized_frame") or []) >= 4
    ]
    frames.extend(
        tuple(obj["bbox"])
        for obj in extra_objects
        if len(obj.get("bbox") or []) >= 4
    )
    return {
        "coordinate_mode": "dimension_scaled_centered",
        "rotation_normalization": rotation_summary,
        "layers": layers,
        "extra_objects": sorted(extra_objects, key=object_sort_key),
        "frame": normalized_bbox_list(union_bbox_values(frames)) if frames else [],
    }


def multiview_overlay_layer_from_graph(
    graph: dict[str, Any],
    options: MultiviewOptions,
) -> dict[str, Any] | None:
    raw_view = str(graph.get("_raw_view") or graph.get("view") or "").strip().lower()
    if raw_view not in {"top", "bottom", "land"}:
        return None
    objects = multiview_overlay_objects_from_graph(graph, raw_view, options)
    if not objects:
        return None
    frame = multiview_overlay_source_frame(objects)
    if frame is None:
        return None
    unit_scales = multiview_overlay_unit_scales(graph)
    normalized_objects = [
        normalized_multiview_overlay_object(obj, frame, unit_scales)
        for obj in objects
        if normalized_bbox(obj) is not None
    ]
    normalized_objects = [obj for obj in normalized_objects if obj is not None]
    if not normalized_objects:
        return None
    normalized_frames = [tuple(obj["bbox"]) for obj in normalized_objects]
    return {
        "raw_view": raw_view,
        "canonical_view": normalize_view(raw_view, options),
        "graph_path": str(graph.get("_graph_path") or ""),
        "coordinate_mode": "dimension_scaled_centered",
        "source_frame": normalized_bbox_list(frame),
        "normalized_frame": normalized_bbox_list(union_bbox_values(normalized_frames)),
        "unit_scales": unit_scales,
        "objects": sorted(normalized_objects, key=object_sort_key),
    }


def normalize_multiview_overlay_layer_rotations(
    layers: list[dict[str, Any]],
    extra_objects: list[dict[str, Any]],
    *,
    source_views: set[str] | None = None,
) -> dict[str, Any]:
    reference = select_rotation_reference_layer(layers, source_views=source_views)
    if reference is None:
        return {
            "status": "missing_reference_layer",
            "reference_view": None,
            "reference_graph_path": None,
            "rotation_candidates": [0, 90, 180, 270],
            "layer_rotations": [],
            "extra_object_rotations": [],
        }

    reference_boxes = layer_rotation_boxes(reference)
    if len(reference_boxes) < 2:
        return {
            "status": "insufficient_reference_boxes",
            "reference_view": reference.get("raw_view"),
            "reference_graph_path": reference.get("graph_path"),
            "rotation_candidates": [0, 90, 180, 270],
            "layer_rotations": [],
            "extra_object_rotations": [],
        }

    layer_rotations: list[dict[str, Any]] = []
    rotation_by_graph: dict[str, int] = {}
    for layer in layers:
        raw_view = str(layer.get("raw_view") or "")
        layer_rotation = best_multiview_layer_rotation(reference_boxes, reference, layer)
        rotation_by_graph[str(layer.get("graph_path") or "")] = layer_rotation["rotation_degrees"]
        apply_rotation_to_layer(layer, int(layer_rotation["rotation_degrees"]))
        layer_rotations.append(
            {
                "raw_view": raw_view,
                "canonical_view": layer.get("canonical_view"),
                "graph_path": layer.get("graph_path"),
                "rotation_degrees": int(layer_rotation["rotation_degrees"]),
                "rotation_iou": layer_rotation["iou"],
                "rotation_candidates": layer_rotation["scores"],
            }
        )

    extra_object_rotations: list[dict[str, Any]] = []
    for obj in extra_objects:
        source_graph = str(obj.get("source_graph") or "")
        rotation = rotation_by_graph.get(source_graph, 0)
        rotated_bbox = rotate_bbox_around_origin(normalized_bbox(obj), rotation)
        if rotated_bbox is not None:
            obj["bbox"] = normalized_bbox_list(rotated_bbox)
        obj["rotation_degrees"] = rotation
        extra_object_rotations.append(
            {
                "source_graph": source_graph,
                "rotation_degrees": rotation,
            }
        )

    return {
        "status": "aligned",
        "reference_view": reference.get("raw_view"),
        "reference_graph_path": reference.get("graph_path"),
        "rotation_candidates": [0, 90, 180, 270],
        "layer_rotations": layer_rotations,
        "extra_object_rotations": extra_object_rotations,
    }


def is_top_package_rotation_anchor(layer: dict[str, Any]) -> bool:
    if str(layer.get("raw_view") or "") != "top":
        return False
    has_outline = False
    has_package_pad = False
    for obj in layer.get("objects") or []:
        if is_outline_rotation_object(obj):
            has_outline = True
        if str(obj.get("role") or "") == "package_pad":
            has_package_pad = True
    return has_outline and has_package_pad


def select_rotation_reference_layer(
    layers: list[dict[str, Any]],
    *,
    source_views: set[str] | None = None,
) -> dict[str, Any] | None:
    preferred = {"land": 0, "bottom": 1, "top": 2}
    candidates = [layer for layer in layers if len(layer.get("normalized_frame") or []) >= 4]
    if not candidates:
        return None
    if source_views == {"top", "land"}:
        top_package_candidates = [layer for layer in candidates if is_top_package_rotation_anchor(layer)]
        if top_package_candidates:
            return min(
                top_package_candidates,
                key=lambda layer: (
                    -len(layer.get("objects") or []),
                    str(layer.get("graph_path") or ""),
                ),
            )
    return min(
        candidates,
        key=lambda layer: (
            preferred.get(str(layer.get("raw_view") or ""), 9),
            -len(layer.get("objects") or []),
            str(layer.get("graph_path") or ""),
        ),
    )


OUTLINE_ROTATION_TIE_TOLERANCE = 1e-9
MIN_NONZERO_ROTATION_IOU = 0.05


def is_outline_rotation_object(obj: dict[str, Any]) -> bool:
    role = str(obj.get("role") or "").strip().lower()
    if role == "outline_2d":
        return True
    label = str(obj.get("source_label") or obj.get("label") or "").strip().lower()
    return label == "outline"


def layer_rotation_boxes(
    layer: dict[str, Any],
    *,
    include_outline: bool = True,
    outline_only: bool = False,
) -> list[list[float]]:
    boxes = []
    for obj in layer.get("objects") or []:
        is_outline = is_outline_rotation_object(obj)
        if outline_only and not is_outline:
            continue
        if not include_outline and is_outline:
            continue
        bbox = normalized_bbox(obj)
        if bbox is not None:
            boxes.append(normalized_bbox_list(bbox))
    boxes.sort(key=lambda box: (round((box[1] + box[3]) / 2.0, 6), round((box[0] + box[2]) / 2.0, 6)))
    return boxes


def best_multiview_layer_rotation(
    reference_boxes: list[list[float]],
    reference_layer: dict[str, Any],
    candidate_layer: dict[str, Any],
) -> dict[str, Any]:
    reference_score_boxes = layer_rotation_boxes(reference_layer, include_outline=False)
    candidate_boxes = layer_rotation_boxes(candidate_layer, include_outline=False)
    if len(reference_score_boxes) < 2:
        reference_score_boxes = reference_boxes
    if len(candidate_boxes) < 2:
        candidate_boxes = layer_rotation_boxes(candidate_layer)
    if len(candidate_boxes) < 2:
        return {"rotation_degrees": 0, "iou": None, "scores": []}

    reference_outline_boxes = layer_rotation_boxes(reference_layer, outline_only=True)
    candidate_outline_boxes = layer_rotation_boxes(candidate_layer, outline_only=True)
    outline_scores: dict[int, float | None] = {}
    allowed_rotations = (0, 90, 180, 270)
    if reference_outline_boxes and candidate_outline_boxes:
        for rotation in allowed_rotations:
            rotated_outline = [rotate_bbox_around_origin(box, rotation) for box in candidate_outline_boxes]
            rotated_outline = [box for box in rotated_outline if box is not None]
            outline_score = matched_box_iou_score(reference_outline_boxes, rotated_outline)
            outline_scores[rotation] = float(outline_score) if isinstance(outline_score, (int, float)) else 0.0
        best_outline_iou = max(outline_scores.values())
        allowed_rotations = tuple(
            rotation
            for rotation, score in outline_scores.items()
            if score is not None and abs(float(score) - best_outline_iou) <= OUTLINE_ROTATION_TIE_TOLERANCE
        )
        if not allowed_rotations:
            allowed_rotations = (0, 90, 180, 270)

    best: dict[str, Any] | None = None
    scores = []
    for rotation in (0, 90, 180, 270):
        rotated = [rotate_bbox_around_origin(box, rotation) for box in candidate_boxes]
        rotated = [box for box in rotated if box is not None]
        score = matched_box_iou_score(reference_score_boxes, rotated)
        score_value = float(score) if isinstance(score, (int, float)) else 0.0
        candidate = {
            "rotation_degrees": rotation,
            "iou": score_value,
            "outline_iou": outline_scores.get(rotation),
            "eligible_by_outline": rotation in allowed_rotations,
        }
        scores.append(candidate)
        if rotation not in allowed_rotations:
            continue
        if best is None or candidate["iou"] > best["iou"] or (
            candidate["iou"] == best["iou"] and abs(rotation) < abs(int(best["rotation_degrees"]))
        ):
            best = candidate

    assert best is not None
    zero_candidate = next((item for item in scores if int(item["rotation_degrees"]) == 0), None)
    if (
        int(best["rotation_degrees"]) != 0
        and best["iou"] < MIN_NONZERO_ROTATION_IOU
        and zero_candidate is not None
        and zero_candidate["eligible_by_outline"]
    ):
        best = dict(zero_candidate)
        best["suppressed_low_confidence_rotation"] = True
    return {"rotation_degrees": int(best["rotation_degrees"]), "iou": best["iou"], "scores": scores}


def apply_rotation_to_layer(layer: dict[str, Any], rotation_degrees: int) -> None:
    objects = layer.get("objects") or []
    rotated_objects = []
    for obj in objects:
        bbox = normalized_bbox(obj)
        if bbox is None:
            continue
        rotated = rotate_bbox_around_origin(bbox, rotation_degrees)
        if rotated is None:
            continue
        updated = dict(obj)
        updated["bbox"] = normalized_bbox_list(rotated)
        updated["rotation_degrees"] = rotation_degrees
        rotated_objects.append(updated)
    layer["objects"] = sorted(rotated_objects, key=object_sort_key)
    frame = union_bbox_values([tuple(obj["bbox"]) for obj in rotated_objects if len(obj.get("bbox") or []) >= 4])
    if frame:
        layer["normalized_frame"] = normalized_bbox_list(frame)
    layer["rotation_degrees"] = rotation_degrees


def rotate_bbox_around_origin(
    bbox: list[float] | tuple[float, float, float, float] | None,
    rotation_degrees: int,
) -> list[float] | None:
    if not bbox or len(bbox) < 4:
        return None
    turns = (int(rotation_degrees) // 90) % 4
    if turns == 0:
        return [float(value) for value in bbox[:4]]
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    points = []
    for x, y in ((x1, y1), (x1, y2), (x2, y1), (x2, y2)):
        if turns == 1:
            rx, ry = y, -x
        elif turns == 2:
            rx, ry = -x, -y
        else:
            rx, ry = -y, x
        points.append((rx, ry))
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


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


def multiview_overlay_objects_from_graph(
    graph: dict[str, Any],
    raw_view: str,
    options: MultiviewOptions,
) -> list[dict[str, Any]]:
    objects = []
    outline = extract_outline(graph, options)
    if outline and normalized_bbox(outline) is not None:
        objects.append(outline)
    role = "land_pad" if raw_view == "land" else "package_pad"
    pads = extract_objects(graph, role, options)
    if not pads:
        return []
    objects.extend(pads)
    return [obj for obj in objects if normalized_bbox(obj) is not None]


def multiview_overlay_source_frame(objects: list[dict[str, Any]]) -> tuple[float, float, float, float] | None:
    pad_like = [
        obj
        for obj in objects
        if str(obj.get("role") or "") in {"package_pad", "land_pad"}
        or is_pad_like_object(obj)
    ]
    return union_object_bbox(pad_like) or union_object_bbox(objects)


def multiview_overlay_unit_scales(graph: dict[str, Any]) -> dict[str, Any]:
    scales = graph_axis_unit_scales(graph)
    x_scale = scales.get("x") or scales.get("y") or 1.0
    y_scale = scales.get("y") or scales.get("x") or 1.0
    source = "accepted_dimensions" if scales else "graph_pixels"
    return {"x": float(x_scale), "y": float(y_scale), "source": source}


def normalized_multiview_overlay_object(
    obj: dict[str, Any],
    frame: tuple[float, float, float, float],
    unit_scales: dict[str, Any],
) -> dict[str, Any] | None:
    bbox = normalized_bbox(obj)
    if bbox is None:
        return None
    result = dict(obj)
    result["source_bbox"] = normalized_bbox_list(bbox)
    result["bbox"] = normalize_bbox_to_multiview_frame(bbox, frame, unit_scales)
    result["coordinate_mode"] = "dimension_scaled_centered"
    return result


def multiview_overlay_extra_object(
    obj: dict[str, Any],
    layer_by_graph: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    source_graph = str(obj.get("source_graph") or "")
    layer = layer_by_graph.get(source_graph)
    if layer is None:
        return None
    bbox = normalized_bbox(obj)
    if bbox is None:
        return None
    source_frame = tuple(layer.get("source_frame") or [])
    if len(source_frame) < 4:
        return None
    unit_scales = layer.get("unit_scales") or {}
    result = dict(obj)
    result["source_bbox"] = normalized_bbox_list(bbox)
    result["bbox"] = normalize_bbox_to_multiview_frame(bbox, source_frame, unit_scales)
    result["coordinate_mode"] = "dimension_scaled_centered"
    result["source_overlay_layer"] = {
        "raw_view": layer.get("raw_view"),
        "canonical_view": layer.get("canonical_view"),
        "graph_path": layer.get("graph_path"),
    }
    return result


def normalize_bbox_to_multiview_frame(
    bbox: tuple[float, float, float, float],
    frame: tuple[float, float, float, float],
    unit_scales: dict[str, Any],
) -> list[float]:
    fx1, fy1, fx2, fy2 = [float(value) for value in frame[:4]]
    source_cx = (fx1 + fx2) / 2.0
    source_cy = (fy1 + fy2) / 2.0
    unit_x = float(unit_scales.get("x") or 1.0)
    unit_y = float(unit_scales.get("y") or 1.0)
    x1, y1, x2, y2 = bbox
    return normalized_bbox_list(
        [
            (x1 - source_cx) * unit_x,
            (y1 - source_cy) * unit_y,
            (x2 - source_cx) * unit_x,
            (y2 - source_cy) * unit_y,
        ]
    )


def union_bbox_values(
    boxes: list[tuple[float, float, float, float]],
) -> list[float]:
    valid = [box for box in boxes if len(box) >= 4]
    if not valid:
        return []
    return [
        min(float(box[0]) for box in valid),
        min(float(box[1]) for box in valid),
        max(float(box[2]) for box in valid),
        max(float(box[3]) for box in valid),
    ]


def inner_land_pad_templates(
    land_detail_graphs: list[dict[str, Any]],
    options: MultiviewOptions,
) -> list[dict[str, Any]]:
    templates = []
    seen: set[tuple[int, int, int, int]] = set()
    for graph in land_detail_graphs:
        template = inner_land_pad_template_from_graph(graph, options)
        if template is None:
            continue
        ratio = template["relative_bbox"]
        key = tuple(round(float(value) * 10000) for value in ratio)
        if key in seen:
            continue
        seen.add(key)
        templates.append(template)
    return templates


def inner_land_pad_template_from_graph(
    graph: dict[str, Any],
    options: MultiviewOptions,
) -> dict[str, Any] | None:
    candidates = [
        object_payload(obj, graph, "land_detail_pad", options)
        for obj in graph.get("objects") or []
        if object_label(obj) in PAD_LABELS
    ]
    candidates = [obj for obj in candidates if normalized_bbox(obj) is not None]
    if len(candidates) < 2:
        return None
    candidates = sorted(candidates, key=bbox_area, reverse=True)
    outer = candidates[0]
    outer_bbox = normalized_bbox(outer)
    outer_area = bbox_area(outer)
    if outer_bbox is None or outer_area <= 0.0:
        return None

    dimension_template = dimension_inset_inner_land_pad_template_from_graph(graph, candidates, outer, outer_bbox, options)
    if dimension_template is not None:
        return dimension_template

    inner_candidates = []
    for candidate in candidates[1:]:
        candidate_bbox = normalized_bbox(candidate)
        candidate_area = bbox_area(candidate)
        if candidate_bbox is None or candidate_area <= 0.0:
            continue
        if candidate_area >= outer_area * 0.98:
            continue
        if not bbox_inside_bbox(candidate_bbox, outer_bbox, tolerance=0.05):
            continue
        inner_candidates.append(candidate)
    if not inner_candidates:
        return None

    inner = sorted(inner_candidates, key=lambda obj: (-bbox_area(obj), str(obj.get("source_object_id") or "")))[0]
    inner_bbox = normalized_bbox(inner)
    if inner_bbox is None:
        return None
    ratio = relative_bbox(outer_bbox, inner_bbox)
    if ratio is None:
        return None
    return {
        "relative_bbox": ratio,
        "source": {
            "template_type": "bbox_ratio",
            "shape_family": pad_shape_family(outer),
            "source_graph": str(graph.get("_graph_path") or ""),
            "annotation_path": str(graph.get("annotation_path") or ""),
            "raw_view": str(graph.get("_raw_view") or graph.get("view") or "").lower(),
            "canonical_view": normalize_view(str(graph.get("_raw_view") or graph.get("view") or ""), options),
            "source_object_id": inner.get("source_object_id"),
            "source_bbox": list(inner_bbox),
            "outer_source_object_id": outer.get("source_object_id"),
            "outer_bbox": list(outer_bbox),
            "relative_bbox": ratio,
        },
    }


def dimension_inset_inner_land_pad_template_from_graph(
    graph: dict[str, Any],
    candidates: list[dict[str, Any]],
    outer: dict[str, Any],
    outer_bbox: tuple[float, float, float, float],
    options: MultiviewOptions,
) -> dict[str, Any] | None:
    scales = graph_axis_unit_scales(graph)
    unit_x = scales.get("x") or scales.get("y")
    unit_y = scales.get("y") or scales.get("x")
    if unit_x is None or unit_x <= 0.0 or unit_y is None or unit_y <= 0.0:
        return None
    inset_dimensions = accepted_land_detail_inset_dimensions(graph, candidates)
    if not inset_dimensions:
        return None

    outer_w = outer_bbox[2] - outer_bbox[0]
    outer_h = outer_bbox[3] - outer_bbox[1]
    if outer_w <= 0.0 or outer_h <= 0.0:
        return None
    dimension = sorted(
        inset_dimensions,
        key=lambda dim: (
            float(dim.get("value") or 0.0),
            str(dim.get("dimension_id") or dim.get("id") or ""),
        ),
    )[0]
    inset_value = numeric(dimension.get("value"))
    if inset_value is None or inset_value <= 0.0:
        return None

    margin_x = inset_value / unit_x
    margin_y = inset_value / unit_y
    if margin_x <= 0.0 or margin_y <= 0.0:
        return None
    if margin_x * 2.0 >= outer_w or margin_y * 2.0 >= outer_h:
        return None
    inner_bbox = (
        outer_bbox[0] + margin_x,
        outer_bbox[1] + margin_y,
        outer_bbox[2] - margin_x,
        outer_bbox[3] - margin_y,
    )
    ratio = relative_bbox(outer_bbox, inner_bbox)
    if ratio is None:
        return None

    return {
        "relative_bbox": ratio,
        "source": {
            "template_type": "dimension_inset",
            "shape_family": pad_shape_family(outer),
            "source_graph": str(graph.get("_graph_path") or ""),
            "annotation_path": str(graph.get("annotation_path") or ""),
            "raw_view": str(graph.get("_raw_view") or graph.get("view") or "").lower(),
            "canonical_view": normalize_view(str(graph.get("_raw_view") or graph.get("view") or ""), options),
            "source_object_id": first_dimension_target_id(dimension),
            "source_bbox": normalized_bbox_list(list(inner_bbox)),
            "outer_source_object_id": outer.get("source_object_id"),
            "outer_bbox": list(outer_bbox),
            "relative_bbox": ratio,
            "dimension_id": dimension.get("dimension_id"),
            "dimension_text": dimension.get("text"),
            "dimension_value": inset_value,
            "dimension_kind": dimension.get("kind"),
            "inset_value": inset_value,
            "inset_margin_x": margin_x,
            "inset_margin_y": margin_y,
            "coordinate_unit_scale_x": unit_x,
            "coordinate_unit_scale_y": unit_y,
        },
    }


def accepted_land_detail_inset_dimensions(graph: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidate_ids = {candidate.get("source_object_id") for candidate in candidates}
    dimensions = []
    for dim in graph.get("dimensions") or []:
        if str(dim.get("status") or "") != "accepted":
            continue
        if str(dim.get("kind") or "").lower() != "distance":
            continue
        value = numeric(dim.get("value"))
        if value is None or value <= 0.0:
            continue
        target_ids = set(dim.get("target_ids") or [])
        if target_ids and not (target_ids & candidate_ids):
            continue
        dimensions.append(dim)
    return dimensions


def inner_land_template_matches_land_pad(template: dict[str, Any], land_pad: dict[str, Any]) -> bool:
    expected = str((template.get("source") or {}).get("shape_family") or "")
    if not expected:
        return True
    return pad_shape_family(land_pad) == expected


def pad_shape_family(obj: dict[str, Any]) -> str:
    if is_circle_like_pad(obj):
        return "circle"
    if is_dshape_like_pad(obj):
        return "dshape"
    return "rect"


def first_dimension_target_id(dim: dict[str, Any]) -> Any:
    target_ids = list(dim.get("target_ids") or [])
    return target_ids[0] if target_ids else None


def bbox_inside_bbox(
    inner: tuple[float, float, float, float],
    outer: tuple[float, float, float, float],
    *,
    tolerance: float,
) -> bool:
    return (
        inner[0] >= outer[0] - tolerance
        and inner[1] >= outer[1] - tolerance
        and inner[2] <= outer[2] + tolerance
        and inner[3] <= outer[3] + tolerance
    )


def relative_bbox(
    outer: tuple[float, float, float, float],
    inner: tuple[float, float, float, float],
) -> list[float] | None:
    outer_w = outer[2] - outer[0]
    outer_h = outer[3] - outer[1]
    if outer_w <= 0.0 or outer_h <= 0.0:
        return None
    ratio = [
        (inner[0] - outer[0]) / outer_w,
        (inner[1] - outer[1]) / outer_h,
        (inner[2] - outer[0]) / outer_w,
        (inner[3] - outer[1]) / outer_h,
    ]
    if ratio[2] <= ratio[0] or ratio[3] <= ratio[1]:
        return None
    return [min(max(float(value), 0.0), 1.0) for value in ratio]


def apply_relative_bbox_to_bbox(
    bbox: tuple[float, float, float, float],
    ratio: list[float],
) -> list[float]:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    return [
        x1 + ratio[0] * width,
        y1 + ratio[1] * height,
        x1 + ratio[2] * width,
        y1 + ratio[3] * height,
    ]


def dedupe_inner_land_pads(objects: list[dict[str, Any]], *, bbox_tol: float = 0.001) -> list[dict[str, Any]]:
    deduped = []
    seen: set[tuple[str, int, int, int, int]] = set()
    for obj in objects:
        bbox = normalized_bbox(obj)
        if bbox is None:
            continue
        key = (
            str(obj.get("source_land_pad_id") or obj.get("source_land_pad_index") or ""),
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


def select_lead_contact_length_dimension(dimensions: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = []
    for dim in dimensions:
        if str(dim.get("canonical_view") or "") not in {"lateral", "lead_detail"}:
            continue
        if str(dim.get("kind") or "") != "size" or str(dim.get("axis") or "").lower() != "x":
            continue
        labels = {str(label).lower() for label in dim.get("target_labels") or [] if str(label)}
        if not labels or not labels <= LEAD_CONTACT_TARGET_LABELS:
            continue
        value = numeric(dim.get("value"))
        if value is None or value <= 0:
            continue
        candidates.append((value, dim))
    if not candidates:
        return None
    values = sorted(value for value, _dim in candidates)
    median_value = values[len(values) // 2]
    return sorted(candidates, key=lambda item: (abs(item[0] - median_value), str(item[1].get("source_graph") or "")))[0][1]


def lead_partial_dimensions(dimensions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for dim in dimensions:
        if str(dim.get("canonical_view") or "") not in {"lateral", "lead_detail"}:
            continue
        semantics = lead_partial_dimension_semantics(dim)
        if semantics not in {"lead_ground_contact_length", "pad_width", "lead_pad_length"}:
            continue
        enriched = corrected_lateral_dual_unit_dimension(dim)
        if not lead_partial_dimension_is_usable(enriched):
            continue
        enriched["overlay_semantics"] = semantics
        candidates.append(enriched)
    candidates = prefer_lead_contact_dimensions(candidates)
    return sorted(
        candidates,
        key=lambda dim: (
            str(dim.get("raw_view") or ""),
            str(dim.get("source_graph") or ""),
            str(dim.get("dimension_id") or dim.get("id") or ""),
        ),
    )


def prefer_lead_contact_dimensions(dimensions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lead_contact_keys = {
        (str(dim.get("source_graph") or ""), tuple(dim.get("target_ids") or []))
        for dim in dimensions
        if str(dim.get("overlay_semantics") or "") == "lead_ground_contact_length"
        and is_center_edge_lead_dimension(dim)
    }
    if not lead_contact_keys:
        return dimensions
    kept = []
    for dim in dimensions:
        key = (str(dim.get("source_graph") or ""), tuple(dim.get("target_ids") or []))
        if (
            str(dim.get("overlay_semantics") or "") in {"lead_ground_contact_length", "lead_pad_length"}
            and key in lead_contact_keys
            and not is_center_edge_lead_dimension(dim)
        ):
            continue
        kept.append(dim)
    return kept


def corrected_lateral_dual_unit_dimension(dim: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with inch value selected for reversed inch/mm lateral text.

    Coordinate/unit convention: multiview geometry currently uses the inch-like
    value from dual-unit dimension text.  Some OCR strings put the millimeter
    value first, e.g. ``24X1.143 .045``.  This helper is intentionally scoped to
    lateral/lead-detail partial evidence and only rewrites when two positive
    numeric tokens form an inch/mm ratio close to 25.4 and the current value is
    the larger token.
    """
    current = numeric(dim.get("value"))
    if current is None or current <= 0:
        return dict(dim)
    text = str(dim.get("text") or "")
    tokens = dimension_numeric_tokens_without_repeat_counts(text)
    if len(tokens) < 2:
        return dict(dim)
    best_pair = None
    for first_index, first in enumerate(tokens):
        for second in tokens[first_index + 1 :]:
            small = min(first, second)
            large = max(first, second)
            if small <= 0:
                continue
            ratio = large / small
            if 20.0 <= ratio <= 30.0:
                best_pair = (small, large, ratio)
                break
        if best_pair is not None:
            break
    if best_pair is None:
        return dict(dim)
    small, large, ratio = best_pair
    if not math.isclose(current, large, rel_tol=1e-3, abs_tol=1e-6):
        return dict(dim)
    corrected = dict(dim)
    corrected["value"] = small
    corrected["value_midpoint"] = small
    corrected["value_unit_correction"] = "dual_unit_reversed_inch_mm"
    corrected["value_unit_correction_original_value"] = current
    corrected["value_unit_correction_ratio"] = ratio
    return corrected


def dimension_numeric_tokens_without_repeat_counts(text: str) -> list[float]:
    text_without_counts = re.sub(r"(?i)(?<![A-Za-z])\d+\s*x", " ", text)
    tokens = []
    for match in re.finditer(r"(?<![A-Za-z])[-+]?(?:\d+\.\d+|\.\d+|\d+)", text_without_counts):
        try:
            value = float(match.group(0))
        except ValueError:
            continue
        if value > 0:
            tokens.append(value)
    return tokens


def is_center_edge_lead_dimension(dim: dict[str, Any]) -> bool:
    anchors = {str(anchor or "").lower() for anchor in dim.get("anchors") or []}
    return "center" in anchors and bool(anchors & {"left_edge", "right_edge"})


def lead_partial_dimension_semantics(dim: dict[str, Any]) -> str:
    if str(dim.get("kind") or "") != "size":
        return ""
    if str(dim.get("axis") or "").lower() != "x":
        return ""
    if len(list(dim.get("target_ids") or [])) != 1:
        return ""
    raw_view = str(dim.get("raw_view") or "").strip().lower()
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


def lead_edge_to_center_contact_dimension(
    dim: dict[str, Any],
    graph: dict[str, Any],
    options: MultiviewOptions,
) -> bool:
    raw_view = str(graph.get("_raw_view") or graph.get("view") or "").strip().lower()
    if normalize_view(raw_view, options) != "lead_detail":
        return False
    if str(dim.get("kind") or "") != "size":
        return False
    if str(dim.get("axis") or "").lower() != "x":
        return False
    if len(list(dim.get("target_ids") or [])) != 1:
        return False
    anchors = {str(anchor or "").lower() for anchor in dim.get("anchors") or []}
    if "center" not in anchors or not (anchors & {"left_edge", "right_edge"}):
        return False
    value = numeric(dim.get("value"))
    if value is None or value <= 0:
        return False
    labels = {str(label).lower() for label in target_object_labels(graph, list(dim.get("target_ids") or []))}
    return bool(labels) and labels <= LEAD_CONTACT_TARGET_LABELS


def lead_partial_dimension_is_usable(dim: dict[str, Any]) -> bool:
    value = numeric(dim.get("value"))
    if value is None or value <= 0:
        return False
    labels = {str(label).lower() for label in dim.get("target_labels") or [] if str(label)}
    if not labels or not labels <= LEAD_CONTACT_TARGET_LABELS:
        return False
    return True


def lead_partial_projection_axis(dim: dict[str, Any], radial_axis: str) -> str:
    semantics = str(dim.get("overlay_semantics") or "")
    raw_view = str(dim.get("raw_view") or "").strip().lower()
    if semantics == "pad_width":
        return "y" if radial_axis == "x" else "x"
    if semantics in {"lead_pad_length", "lead_ground_contact_length"}:
        if raw_view == "side":
            return "y"
        if raw_view == "front":
            return "x"
        return radial_axis
    return radial_axis


def lead_pad_radial_axis(
    *,
    center: tuple[float, float],
    package_center: tuple[float, float],
    terminal_band_layout: str,
) -> str:
    """Choose the package edge axis for a terminal pad.

    Coordinates are graph/reconstruction coordinates. Two-row terminal layouts
    use the row axis even for corner pads, so outer pads are not mistaken for
    left/right-side terminals only because their x distance is larger.
    """
    if terminal_band_layout == "top_bottom_rows":
        return "y"
    if terminal_band_layout == "left_right_columns":
        return "x"
    return "x" if abs(center[0] - package_center[0]) >= abs(center[1] - package_center[1]) else "y"


def has_left_right_only_terminal_layout(
    package_pads: list[dict[str, Any]],
    outline_bbox: tuple[float, float, float, float],
) -> bool:
    """Return true when terminal package pads occupy left/right sides only.

    Coordinates are graph/reconstruction coordinates. The classifier uses pad
    centers relative to the package outline center; it does not infer from file
    names or nominal view labels.
    """
    if not package_pads:
        return False
    frame_x1, frame_y1, frame_x2, frame_y2 = outline_bbox
    width = frame_x2 - frame_x1
    height = frame_y2 - frame_y1
    if width <= 0.0 or height <= 0.0:
        return False
    center_x = (frame_x1 + frame_x2) / 2.0
    center_y = (frame_y1 + frame_y2) / 2.0
    left_or_right = 0
    top_or_bottom = 0
    for pad in package_pads:
        bbox = normalized_bbox(pad)
        center = bbox_center(pad)
        if bbox is None or center is None:
            continue
        dx = abs(center[0] - center_x) / width
        dy = abs(center[1] - center_y) / height
        if dx >= dy:
            left_or_right += 1
        else:
            top_or_bottom += 1
    return left_or_right > 0 and top_or_bottom == 0


def uniform_perimeter_terminal_cross_axis_length(
    package_pads: list[dict[str, Any]],
    outline_bbox: tuple[float, float, float, float],
    *,
    terminal_band_layout: str,
) -> float | None:
    """Return a common terminal minor dimension for four-side pad layouts.

    Coordinates are graph/reconstruction pixels. This is only applied when pads
    exist on both radial axes; two-row and two-column packages keep their source
    pad cross-axis extents.
    """
    if terminal_band_layout or len(package_pads) < 8:
        return None
    frame_x1, frame_y1, frame_x2, frame_y2 = outline_bbox
    width = frame_x2 - frame_x1
    height = frame_y2 - frame_y1
    if width <= 0.0 or height <= 0.0:
        return None
    center_x = (frame_x1 + frame_x2) / 2.0
    center_y = (frame_y1 + frame_y2) / 2.0
    x_axis_count = 0
    y_axis_count = 0
    minor_lengths = []
    for pad in package_pads:
        bbox = normalized_bbox(pad)
        center = bbox_center(pad)
        dim = bbox_dimensions(pad)
        if bbox is None or center is None or dim is None:
            continue
        if abs(center[0] - center_x) >= abs(center[1] - center_y):
            x_axis_count += 1
        else:
            y_axis_count += 1
        minor_lengths.append(min(dim))
    if x_axis_count < 2 or y_axis_count < 2 or len(minor_lengths) < 8:
        return None
    value = median_sorted(sorted(minor_lengths))
    return value if value > 0.0 else None


def normalized_lead_source_bbox(
    bbox: tuple[float, float, float, float],
    *,
    radial_axis: str,
    uniform_cross_axis_length: float | None,
) -> tuple[float, float, float, float]:
    """Normalize lead source-pad cross-axis extent for four-side layouts.

    Coordinates are graph/reconstruction pixels. The radial extent remains from
    the package pad; only the perpendicular source extent is replaced.
    """
    if uniform_cross_axis_length is None or uniform_cross_axis_length <= 0.0:
        return bbox
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half = uniform_cross_axis_length / 2.0
    if radial_axis == "x":
        return (x1, cy - half, x2, cy + half)
    return (cx - half, y1, cx + half, y2)


def terminal_pad_band_layout(package_pads: list[dict[str, Any]]) -> str:
    """Classify simple two-band terminal layouts from pad centers.

    Coordinates are graph/reconstruction coordinates. The result is used only
    to choose the axis for front pad-width evidence; it does not move pads.
    """
    if len(package_pads) < 4:
        return ""
    centers = []
    widths = []
    heights = []
    for pad in package_pads:
        bbox = normalized_bbox(pad)
        center = bbox_center(pad)
        if bbox is None or center is None:
            continue
        centers.append(center)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
    if len(centers) < 4:
        return ""
    x_values = [center[0] for center in centers]
    y_values = [center[1] for center in centers]
    x_span = max(x_values) - min(x_values)
    y_span = max(y_values) - min(y_values)
    if x_span <= 0.0 or y_span <= 0.0:
        return ""
    median_width = median_sorted(sorted(widths))
    median_height = median_sorted(sorted(heights))
    y_clusters = axis_center_clusters(y_values, tolerance=max(median_height * 0.75, y_span * 0.08))
    x_clusters = axis_center_clusters(x_values, tolerance=max(median_width * 0.75, x_span * 0.08))
    if len(y_clusters) == 2 and len(x_clusters) > 2 and all(len(cluster) >= 2 for cluster in y_clusters):
        return "top_bottom_rows"
    if len(x_clusters) == 2 and len(y_clusters) > 2 and all(len(cluster) >= 2 for cluster in x_clusters):
        return "left_right_columns"
    return ""


def axis_center_clusters(values: list[float], *, tolerance: float) -> list[list[float]]:
    if tolerance <= 0.0:
        return [[value] for value in sorted(values)]
    clusters: list[list[float]] = []
    for value in sorted(values):
        if not clusters or abs(value - clusters[-1][-1]) > tolerance:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    return clusters


def effective_lead_partial_semantics(
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
    if str(dim.get("raw_view") or "").strip().lower() != "front":
        return base_semantics
    x1, y1, x2, y2 = bbox
    pad_extent = (x2 - x1) if projection_axis == "x" else (y2 - y1)
    pad_physical_extent = pad_extent * unit_scale
    if pad_physical_extent > 0.0 and dimension_value > pad_physical_extent * 1.25:
        return "lead_ground_contact_length"
    return base_semantics


def lead_partial_bbox(
    *,
    bbox: tuple[float, float, float, float],
    package_center: tuple[float, float],
    package_frame: tuple[float, float, float, float],
    projection_axis: str,
    length: float,
    semantics: str,
    pad_width_alignment: str = "center",
) -> list[float]:
    x1, y1, x2, y2 = bbox
    pad_cx = (x1 + x2) / 2.0
    pad_cy = (y1 + y2) / 2.0
    center_x, center_y = package_center
    if semantics in {"pad_width", "lead_pad_length"}:
        if semantics == "pad_width" and pad_width_alignment == "left_right_outer_edge" and projection_axis == "x":
            if pad_cx <= center_x:
                return [x1, y1, x1 + length, y2]
            return [x2 - length, y1, x2, y2]
        if projection_axis == "x":
            return [pad_cx - length / 2.0, y1, pad_cx + length / 2.0, y2]
        return [x1, pad_cy - length / 2.0, x2, pad_cy + length / 2.0]
    radial_axis = "x" if abs(pad_cx - center_x) >= abs(pad_cy - center_y) else "y"
    if semantics == "lead_ground_contact_length" and projection_axis != radial_axis:
        outside_side = lead_partial_outside_side(bbox, package_frame)
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


def lead_partial_outside_side(
    bbox: tuple[float, float, float, float],
    package_frame: tuple[float, float, float, float],
) -> str:
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


def lead_partial_role(semantics: str) -> str:
    if semantics == "pad_width":
        return "partial_pad_width"
    if semantics == "lead_pad_length":
        return "partial_lead_pad_length"
    return "lead_pad"


def lead_pad_coordinate_unit_scales(
    package_pads: list[dict[str, Any]],
    package_graph: dict[str, Any] | None,
) -> dict[str, float]:
    scales = graph_axis_unit_scales(package_graph)
    x_scale = scales.get("x") or scales.get("y")
    y_scale = scales.get("y") or scales.get("x")
    if x_scale is None or y_scale is None:
        return {}
    return {"x": x_scale, "y": y_scale}


def graph_axis_unit_scales(graph: dict[str, Any] | None) -> dict[str, float]:
    if not graph:
        return {}
    axis_values: dict[str, list[float]] = {"x": [], "y": []}
    metrics = graph.get("metrics") or {}
    for axis in ("x", "y"):
        value = numeric(metrics.get(f"axis_scale_{axis}"))
        if value is not None and value > 0:
            axis_values[axis].append(value)
    global_scale = numeric(metrics.get("global_scale"))
    for axis in ("x", "y"):
        if not axis_values[axis] and global_scale is not None and global_scale > 0:
            axis_values[axis].append(global_scale)
    for dim in graph.get("dimensions") or []:
        if str(dim.get("status") or "") != "accepted":
            continue
        axis = str(dim.get("axis") or "").lower()
        if axis not in axis_values:
            continue
        value = numeric(dim.get("axis_scale"))
        if value is not None and value > 0:
            axis_values[axis].append(value)
    return {axis: median_sorted(sorted(values)) for axis, values in axis_values.items() if values}


def terminal_package_pads_for_lead_synthesis(package_pads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    boxes = [normalized_bbox(pad) for pad in package_pads]
    valid_boxes = [box for box in boxes if box is not None]
    if len(valid_boxes) < 5:
        return package_pads
    areas = sorted(abs((box[2] - box[0]) * (box[3] - box[1])) for box in valid_boxes)
    median_area = areas[len(areas) // 2]
    if median_area <= 0:
        return package_pads
    extent = (
        min(box[0] for box in valid_boxes),
        min(box[1] for box in valid_boxes),
        max(box[2] for box in valid_boxes),
        max(box[3] for box in valid_boxes),
    )
    width = extent[2] - extent[0]
    height = extent[3] - extent[1]
    if width <= 0 or height <= 0:
        return package_pads
    terminals = []
    for pad, box in zip(package_pads, boxes):
        if box is None:
            continue
        area = abs((box[2] - box[0]) * (box[3] - box[1]))
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        x_ratio = (cx - extent[0]) / width
        y_ratio = (cy - extent[1]) / height
        central = 0.2 <= x_ratio <= 0.8 and 0.2 <= y_ratio <= 0.8
        if central and area >= median_area * 1.8:
            continue
        terminals.append(pad)
    return terminals or package_pads


def lead_contact_dimension_ref(dim: dict[str, Any]) -> dict[str, Any]:
    ref = {
        "id": dim.get("id"),
        "dimension_id": dim.get("dimension_id"),
        "text": dim.get("text"),
        "value": dim.get("value"),
        "raw_view": dim.get("raw_view"),
        "canonical_view": dim.get("canonical_view"),
        "source_graph": dim.get("source_graph"),
        "annotation_path": dim.get("annotation_path"),
    }
    for key in (
        "value_unit_correction",
        "value_unit_correction_original_value",
        "value_unit_correction_ratio",
    ):
        if key in dim:
            ref[key] = dim.get(key)
    return ref


def union_object_bbox(objects: list[dict[str, Any]]) -> tuple[float, float, float, float] | None:
    boxes = [normalized_bbox(obj) for obj in objects]
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def normalized_bbox_list(bbox: list[float]) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def regularize_two_column_package_pad_x_geometry(
    package_pads: list[dict[str, Any]],
    outline: dict[str, Any],
    graph: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Align two-column package pads using accepted x-axis dimensions.

    Coordinates are reconstructed graph pixels. Dimension values are physical
    drawing units. The x scale is inferred only from accepted pad width
    dimensions in the same graph, then used to apply an accepted outline-pad
    edge distance. The rule intentionally changes x geometry only; y placement,
    pad height, source selection, and non-two-column layouts are preserved.
    """
    if not graph or len(package_pads) < 4 or len(package_pads) % 2 != 0:
        return package_pads
    outline_bbox = normalized_bbox(outline)
    if outline_bbox is None:
        return package_pads

    id_to_pad = {pad.get("source_object_id"): pad for pad in package_pads}
    id_to_label = {obj.get("id"): object_label(obj) for obj in graph.get("objects") or []}
    scales: list[float] = []
    width_values: list[float] = []
    for dim in graph.get("dimensions") or []:
        if str(dim.get("status") or "") != "accepted":
            continue
        if str(dim.get("kind") or "") != "size" or str(dim.get("axis") or "").lower() != "x":
            continue
        value = numeric(dim.get("value"))
        if value is None or value <= 0:
            continue
        for target_id in dim.get("target_ids") or []:
            if id_to_label.get(target_id) not in PAD_LABELS:
                continue
            pad = id_to_pad.get(target_id)
            dims = bbox_dimensions(pad or {})
            if dims is None or dims[0] <= 0:
                continue
            scales.append(dims[0] / value)
            width_values.append(value)
    if not scales or not width_values:
        return package_pads

    px_per_unit = median_sorted(scales)
    target_width = median_sorted(width_values) * px_per_unit
    x1, _y1, x2, _y2 = outline_bbox
    outline_width = abs(x2 - x1)
    if target_width <= 0 or outline_width <= 0 or target_width >= outline_width * 0.5:
        return package_pads

    margins = package_pad_outline_x_margins(graph, px_per_unit)
    left_margin = margins.get("left")
    right_margin = margins.get("right")
    if left_margin is None and right_margin is None:
        return package_pads
    if left_margin is None:
        left_margin = right_margin
    if right_margin is None:
        right_margin = left_margin
    if left_margin is None or right_margin is None or left_margin < 0 or right_margin < 0:
        return package_pads

    columns = split_two_aligned_x_columns(package_pads, target_width)
    if columns is None:
        return package_pads
    left_ids = {pad.get("source_object_id") for pad in columns[0]}
    right_ids = {pad.get("source_object_id") for pad in columns[1]}
    left_box = (x1 + left_margin, x1 + left_margin + target_width)
    right_box = (x2 - right_margin - target_width, x2 - right_margin)
    if left_box[0] < x1 or left_box[1] > x2 or right_box[0] < x1 or right_box[1] > x2:
        return package_pads
    if left_box[1] >= right_box[0]:
        return package_pads

    adjusted = []
    for pad in package_pads:
        bbox = normalized_bbox(pad)
        if bbox is None:
            adjusted.append(pad)
            continue
        new_x1, new_x2 = left_box if pad.get("source_object_id") in left_ids else right_box
        if pad.get("source_object_id") not in left_ids and pad.get("source_object_id") not in right_ids:
            adjusted.append(pad)
            continue
        adjusted.append(
            dict(
                pad,
                bbox=[new_x1, bbox[1], new_x2, bbox[3]],
                bbox_before_dimension_regularization=list(bbox),
                geometry_adjusted_reason="dimension_regularized_package_pad_x_grid",
                dimension_regularization_axis="x",
            )
        )
    return sorted(adjusted, key=object_sort_key)


def package_pad_outline_x_margins(graph: dict[str, Any], px_per_unit: float) -> dict[str, float]:
    id_to_label = {obj.get("id"): object_label(obj) for obj in graph.get("objects") or []}
    margins: dict[str, float] = {}
    for dim in graph.get("dimensions") or []:
        if str(dim.get("status") or "") != "accepted":
            continue
        if str(dim.get("kind") or "") != "distance" or str(dim.get("axis") or "").lower() != "x":
            continue
        value = numeric(dim.get("value"))
        if value is None or value <= 0:
            continue
        target_ids = list(dim.get("target_ids") or [])
        anchors = list(dim.get("anchors") or [])
        if len(target_ids) != 2 or len(anchors) != 2:
            continue
        labels = [id_to_label.get(target_id) for target_id in target_ids]
        if not (any(label in OUTLINE_LABELS for label in labels) and any(label in PAD_LABELS for label in labels)):
            continue
        outline_index = 0 if labels[0] in OUTLINE_LABELS else 1
        pad_index = 1 - outline_index
        outline_anchor = str(anchors[outline_index] or "").lower()
        pad_anchor = str(anchors[pad_index] or "").lower()
        if outline_anchor == pad_anchor == "left_edge":
            margins["left"] = value * px_per_unit
        elif outline_anchor == pad_anchor == "right_edge":
            margins["right"] = value * px_per_unit
    return margins


def split_two_aligned_x_columns(
    package_pads: list[dict[str, Any]],
    target_width: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    centers = []
    for pad in package_pads:
        center = bbox_center(pad)
        if center is None:
            return None
        centers.append((center[0], pad))
    centers.sort(key=lambda item: item[0])
    midpoint = len(centers) // 2
    left = [pad for _x, pad in centers[:midpoint]]
    right = [pad for _x, pad in centers[midpoint:]]
    if len(left) != len(right) or not left:
        return None

    def column_spread(pads: list[dict[str, Any]]) -> float:
        xs = [bbox_center(pad)[0] for pad in pads if bbox_center(pad) is not None]
        return max(xs) - min(xs) if xs else math.inf

    max_allowed_spread = max(target_width * 0.5, 8.0)
    if column_spread(left) > max_allowed_spread or column_spread(right) > max_allowed_spread:
        return None
    return (left, right)


def body_size_dimension_target_ids(graph: dict[str, Any] | None) -> set[Any]:
    if not graph:
        return set()
    target_ids: set[Any] = set()
    for dim in graph.get("dimensions") or []:
        if str(dim.get("kind") or "") != "size":
            continue
        if not (dimension_lookup_symbols(dim) & BODY_SIZE_DIMENSION_SYMBOLS):
            continue
        target_ids.update(dim.get("target_ids") or [])
    return target_ids


def dimension_lookup_symbols(dim: dict[str, Any]) -> set[str]:
    symbols = {str(symbol) for symbol in dim.get("lookup_symbols") or [] if str(symbol)}
    if dim.get("lookup_symbol"):
        symbols.add(str(dim.get("lookup_symbol")))
    text = str(dim.get("text") or "")
    if text:
        symbols.add(text)
    return symbols


def filter_remote_detail_land_pads(
    land_pads: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove isolated footprint detail pads from dense land drawings.

    Coordinates are reconstructed graph pixel coordinates. This filter targets
    detail insets copied beside the main land pattern: a small rect-like tail
    after a large x-axis gap. It is intentionally disabled for small layouts,
    circle/dshape pads, and tails larger than a few pads.
    """
    if len(land_pads) < 20:
        return land_pads, []
    items = []
    for index, pad in enumerate(land_pads):
        center = bbox_center(pad)
        dims = bbox_dimensions(pad)
        if center is None or dims is None:
            return land_pads, []
        items.append((center[0], center[1], index, pad))
    items.sort(key=lambda item: item[0])
    gaps = [(items[index + 1][0] - items[index][0], index) for index in range(len(items) - 1)]
    positive_gaps = [gap for gap, _ in gaps if gap > 1.0]
    if len(positive_gaps) < 4:
        return land_pads, []
    pitch = median_lower_quantile(positive_gaps, quantile=0.75)
    if pitch <= 0:
        return land_pads, []
    largest_gap, gap_index = max(gaps, key=lambda item: item[0])
    if largest_gap < max(80.0, pitch * 3.0):
        return land_pads, []

    left = list(range(0, gap_index + 1))
    right = list(range(gap_index + 1, len(items)))
    tail = right if len(right) < len(left) else left
    main = left if tail is right else right
    if len(tail) > max(2, math.ceil(len(items) * 0.06)):
        return land_pads, []
    if len(main) < len(items) * 0.85:
        return land_pads, []
    tail_pads = [items[index][3] for index in tail]
    if any(is_circle_like_pad(pad) or is_dshape_like_pad(pad) for pad in tail_pads):
        return land_pads, []

    main_widths = sorted(bbox_dimensions(items[index][3])[0] for index in main if bbox_dimensions(items[index][3]))
    main_heights = sorted(bbox_dimensions(items[index][3])[1] for index in main if bbox_dimensions(items[index][3]))
    median_width = median_sorted(main_widths)
    median_height = median_sorted(main_heights)
    for pad in tail_pads:
        dims = bbox_dimensions(pad)
        if dims is None:
            return land_pads, []
        if median_width > 0 and dims[0] > median_width * 3.0:
            return land_pads, []
        if median_height > 0 and dims[1] > median_height * 3.0:
            return land_pads, []

    filtered_indices = {items[index][2] for index in tail}
    kept = [pad for index, pad in enumerate(land_pads) if index not in filtered_indices]
    filtered = [
        dict(pad, filtered_reason="remote_detail_land_pad_tail_gap")
        for index, pad in enumerate(land_pads)
        if index in filtered_indices
    ]
    return sorted(kept, key=object_sort_key), sorted(filtered, key=object_sort_key)


def median_lower_quantile(values: list[float], *, quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    limit = max(1, math.ceil(len(ordered) * quantile))
    return median_sorted(ordered[:limit])


def median_sorted(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(values[len(values) // 2])


def dedupe_objects_by_bbox(objects: list[dict[str, Any]], *, bbox_tol: float = 0.001) -> list[dict[str, Any]]:
    """Remove exact duplicate reconstructed boxes.

    Bboxes are in the package graph reconstruction coordinate system, usually
    image pixels after solver alignment. The tolerance is intentionally tiny:
    it only catches duplicate detections with the same center and size, not
    nearby pads. Circle-like pads sharing the same center in the same graph are
    merged afterward because ScanResult and the canonical 2D graph treat
    concentric pad rings as one pad with nested geometry evidence.
    """
    deduped = []
    seen: set[tuple[str, str, int, int, int, int] | tuple[str, str, str, str]] = set()
    for obj in objects:
        bbox = obj.get("bbox") or []
        role = str(obj.get("role") or "")
        label = str(obj.get("source_label") or obj.get("label") or "")
        if len(bbox) >= 4:
            x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            key: tuple[str, str, int, int, int, int] | tuple[str, str, str, str] = (
                role,
                label,
                round(center_x / bbox_tol),
                round(center_y / bbox_tol),
                round(width / bbox_tol),
                round(height / bbox_tol),
            )
        else:
            key = (role, label, str(obj.get("source_graph") or ""), str(obj.get("source_object_id") or ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(obj)
    return merge_near_concentric_pads(deduped)


def normalized_bbox(obj: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = obj.get("bbox") or []
    if len(bbox) < 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def expand_bbox(bbox: tuple[float, float, float, float], *, rel_margin: float) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    margin_x = abs(x2 - x1) * rel_margin
    margin_y = abs(y2 - y1) * rel_margin
    return (x1 - margin_x, y1 - margin_y, x2 + margin_x, y2 + margin_y)


def point_in_bbox(point: tuple[float, float], bbox: tuple[float, float, float, float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def is_circle_like_pad(obj: dict[str, Any]) -> bool:
    label = f"{obj.get('label') or ''} {obj.get('source_label') or ''}".lower()
    return "circle" in label


def is_dshape_like_pad(obj: dict[str, Any]) -> bool:
    label = f"{obj.get('label') or ''} {obj.get('source_label') or ''}".lower()
    return "dshape" in label


def merge_near_concentric_pads(objects: list[dict[str, Any]], *, center_tol: float = 5.0) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    passthrough: list[dict[str, Any]] = []
    for obj in objects:
        bbox = obj.get("bbox") or []
        if len(bbox) < 4 or not is_pad_like_object(obj):
            passthrough.append(obj)
            continue
        key = (
            str(obj.get("role") or ""),
            str(obj.get("raw_view") or ""),
            str(obj.get("source_graph") or ""),
        )
        grouped[key].append(obj)

    merged = passthrough[:]
    for candidates in grouped.values():
        for group in cluster_near_concentric_pads(candidates, center_tol=center_tol):
            if len(group) == 1:
                merged.append(group[0])
                continue
            primary = dict(select_concentric_merge_primary(group))
            primary["merged_source_object_ids"] = [obj.get("source_object_id") for obj in group]
            primary["nested_bboxes"] = [obj.get("bbox") for obj in sorted(group, key=bbox_area, reverse=True)]
            primary["merged_bbox_count"] = len(group)
            largest = max(group, key=bbox_area)
            if largest is not primary and largest.get("source_object_id") != primary.get("source_object_id"):
                primary["outer_mask_source_object_id"] = largest.get("source_object_id")
                primary["outer_mask_bbox"] = largest.get("bbox")
            merged.append(primary)
    return sorted(merged, key=object_sort_key)


def select_concentric_merge_primary(group: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose the canonical bbox for concentric pad geometries.

    Coordinates are package graph image/reconstruction coordinates.  Nested
    circle-only rings keep the largest ring.  For land patterns that include an
    outer solder-mask/metal-covered rectangle plus an inner real pad rectangle
    and via circle, use the inner rectangle as the canonical pad bbox while
    preserving all nested evidence.
    """
    by_area = sorted(group, key=bbox_area, reverse=True)
    if len(by_area) < 3:
        return by_area[0]
    rects = [obj for obj in by_area if is_rect_like_pad(obj)]
    circles = [obj for obj in by_area if is_circle_like_pad(obj)]
    if len(rects) < 2 or not circles:
        return by_area[0]
    largest_rect, inner_rect = rects[0], rects[1]
    largest_area = bbox_area(largest_rect)
    inner_area = bbox_area(inner_rect)
    if inner_area <= 0:
        return by_area[0]
    if largest_area >= inner_area * 2.0:
        return inner_rect
    return by_area[0]


def is_rect_like_pad(obj: dict[str, Any]) -> bool:
    label = f"{obj.get('label') or ''} {obj.get('source_label') or ''}".lower()
    if "circle" in label or "dshape" in label:
        return False
    return "rect" in label or "pad" in label


def cluster_near_concentric_pads(objects: list[dict[str, Any]], *, center_tol: float) -> list[list[dict[str, Any]]]:
    clusters: list[list[dict[str, Any]]] = []
    for obj in objects:
        for cluster in clusters:
            if any(near_concentric(obj, other, center_tol=center_tol) for other in cluster):
                cluster.append(obj)
                break
        else:
            clusters.append([obj])
    return clusters


def near_concentric(a: dict[str, Any], b: dict[str, Any], *, center_tol: float) -> bool:
    center_a = bbox_center(a)
    center_b = bbox_center(b)
    if center_a is None or center_b is None:
        return False
    distance = math.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1])
    return distance <= pad_center_merge_tolerance(a, b, center_tol=center_tol)


def pad_center_merge_tolerance(a: dict[str, Any], b: dict[str, Any], *, center_tol: float) -> float:
    dims_a = bbox_dimensions(a)
    dims_b = bbox_dimensions(b)
    if dims_a is None or dims_b is None:
        return 0.0
    min_dim = min(dims_a[0], dims_a[1], dims_b[0], dims_b[1])
    return min(center_tol, min_dim * 0.25)


def bbox_center(obj: dict[str, Any]) -> tuple[float, float] | None:
    bbox = obj.get("bbox") or []
    if len(bbox) < 4:
        return None
    return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)


def bbox_dimensions(obj: dict[str, Any]) -> tuple[float, float] | None:
    bbox = obj.get("bbox") or []
    if len(bbox) < 4:
        return None
    return (abs(float(bbox[2]) - float(bbox[0])), abs(float(bbox[3]) - float(bbox[1])))


def is_pad_like_object(obj: dict[str, Any]) -> bool:
    label = f"{obj.get('label') or ''} {obj.get('source_label') or ''}".lower()
    return any(name in label for name in PAD_LABELS)


def bbox_area(obj: dict[str, Any]) -> float:
    bbox = obj.get("bbox") or []
    if len(bbox) < 4:
        return 0.0
    return abs((float(bbox[2]) - float(bbox[0])) * (float(bbox[3]) - float(bbox[1])))


def object_sort_key(obj: dict[str, Any]) -> tuple[float, float, str]:
    bbox = obj.get("bbox") or []
    if len(bbox) >= 4:
        return (float(bbox[1]), float(bbox[0]), str(obj.get("source_object_id") or ""))
    return (0.0, 0.0, str(obj.get("source_object_id") or ""))


def object_payload(obj: dict[str, Any], graph: dict[str, Any], role: str, options: MultiviewOptions) -> dict[str, Any]:
    raw_view = str(graph.get("_raw_view") or graph.get("view") or "").lower()
    bbox = obj.get("bbox_reconstructed") or obj.get("bbox") or []
    return {
        "source_object_id": obj.get("id"),
        "role": role,
        "label": str(obj.get("label") or ""),
        "source_label": str(obj.get("source_label") or ""),
        "raw_view": raw_view,
        "canonical_view": normalize_view(raw_view, options),
        "bbox": bbox,
        "bbox_yolo": obj.get("bbox_yolo"),
        "group_id": obj.get("group_id"),
        "source_graph": str(graph.get("_graph_path") or ""),
    }


def object_label(obj: dict[str, Any]) -> str:
    return str(obj.get("source_label") or obj.get("label") or "").lower()


def enrich_dimension(dim: dict[str, Any], graph: dict[str, Any], options: MultiviewOptions) -> dict[str, Any]:
    raw_view = str(graph.get("_raw_view") or graph.get("view") or "").lower()
    target_ids = list(dim.get("target_ids") or [])
    target_labels = target_object_labels(graph, target_ids)
    role = dimension_role(dim, target_labels)
    return {
        "id": dim.get("id"),
        "dimension_id": dim.get("dimension_id"),
        "text": str(dim.get("text") or ""),
        "kind": str(dim.get("kind") or ""),
        "axis": str(dim.get("axis") or ""),
        "status": str(dim.get("status") or ""),
        "value": dim.get("value"),
        "value_lower": dim.get("value_lower"),
        "value_upper": dim.get("value_upper"),
        "value_source": dim.get("value_source"),
        "target_ids": target_ids,
        "target_labels": target_labels,
        "anchors": list(dim.get("anchors") or []),
        "role": role,
        "raw_view": raw_view,
        "canonical_view": normalize_view(raw_view, options),
        "source_graph": str(graph.get("_graph_path") or ""),
        "annotation_path": str(graph.get("annotation_path") or ""),
        "bbox": dim.get("bbox"),
        "bbox_norm_1000": dim.get("bbox_norm_1000"),
    }


def target_object_labels(graph: dict[str, Any], target_ids: list[Any]) -> list[str]:
    by_id = {obj.get("id"): object_label(obj) for obj in graph.get("objects") or []}
    return [by_id.get(target_id, "unknown") for target_id in target_ids]


def dimension_role(dim: dict[str, Any], target_labels: list[str]) -> str:
    labels = set(target_labels)
    kind = str(dim.get("kind") or "")
    if labels & OUTLINE_LABELS:
        return "outline_size" if kind == "size" else "outline_relation"
    if labels and labels <= PAD_LABELS | {"lead"}:
        if kind == "size":
            return "pad_size"
        return "pad_spacing"
    return "dimension"


def should_ignore_lateral_height(dim: dict[str, Any], options: MultiviewOptions) -> bool:
    return bool(
        options.ignore_lateral_height
        and dim.get("canonical_view") == "lateral"
        and str(dim.get("axis") or "").lower() == "y"
    )


def detect_conflicts(dimensions: list[dict[str, Any]], options: MultiviewOptions) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for dim in dimensions:
        value = numeric(dim.get("value"))
        if value is None:
            continue
        key = dimension_conflict_key(dim)
        grouped[key].append(dim)

    conflicts = []
    for key, dims in grouped.items():
        if len(dims) < 2:
            continue
        if not is_pad_dimension_conflict_key(key):
            by_view: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for dim in dims:
                by_view[str(dim.get("canonical_view") or "")].append(dim)
            if len(by_view) < 2:
                continue
        primary = choose_primary_dimension(dims)
        primary_value = numeric(primary.get("value"))
        if primary_value is None:
            continue
        tolerance = max(options.conflict_abs_tol, abs(primary_value) * options.conflict_rel_tol)
        for dim in dims:
            if dim is primary:
                continue
            value = numeric(dim.get("value"))
            if value is None:
                continue
            delta = abs(value - primary_value)
            if delta > tolerance:
                conflicts.append(
                    {
                        "type": "dimension_value_conflict",
                        "signature": list(key),
                        "primary": conflict_dimension_ref(primary),
                        "other": conflict_dimension_ref(dim),
                        "delta": delta,
                        "tolerance": tolerance,
                    }
                )
    return conflicts


def dimension_conflict_key(dim: dict[str, Any]) -> tuple[Any, ...]:
    """Return a conservative semantic key for dimension conflict checks.

    Global outline dimensions can be compared across views. Pad dimensions are
    local to a specific pad group unless matching has already proven the same
    physical target, so this key only compares pad dimensions that reference
    the same targets within the same source graph.
    """
    role = str(dim.get("role") or "")
    base = (
        role,
        str(dim.get("kind") or ""),
        str(dim.get("axis") or ""),
        tuple(str(anchor) for anchor in dim.get("anchors") or []),
    )
    if role in {"pad_size", "pad_spacing"}:
        return (
            *base,
            "source_graph",
            str(dim.get("source_graph") or ""),
            "target_ids",
            tuple(str(target_id) for target_id in dim.get("target_ids") or []),
        )
    return base


def is_pad_dimension_conflict_key(key: tuple[Any, ...]) -> bool:
    return bool(key and key[0] in {"pad_size", "pad_spacing"})


def choose_primary_dimension(dims: list[dict[str, Any]]) -> dict[str, Any]:
    priority = {"bottom": 0, "land": 1, "top": 2, "lateral": 3, "lead_detail": 4}
    return sorted(dims, key=lambda dim: priority.get(str(dim.get("canonical_view") or ""), 99))[0]


def conflict_dimension_ref(dim: dict[str, Any]) -> dict[str, Any]:
    return {
        "value": dim.get("value"),
        "text": dim.get("text"),
        "raw_view": dim.get("raw_view"),
        "canonical_view": dim.get("canonical_view"),
        "source_graph": dim.get("source_graph"),
    }


def match_package_and_land_pads(package_pads: list[dict[str, Any]], land_pads: list[dict[str, Any]]) -> dict[str, Any]:
    if not package_pads or not land_pads:
        return {"status": "not_applicable", "reason": "missing_package_or_land_pads"}
    if len(package_pads) != len(land_pads):
        return {
            "status": "not_applicable",
            "reason": "package_land_count_differs",
            "package_pad_count": len(package_pads),
            "land_pad_count": len(land_pads),
        }
    package_order = sorted_object_ids(package_pads)
    land_order = sorted_object_ids(land_pads)
    if len(set(package_order)) != len(package_order) or len(set(land_order)) != len(land_order):
        return {"status": "ambiguous_match", "reason": "non_unique_object_order"}
    return {
        "status": "matched",
        "strategy": "row_major_bbox_order",
        "pairs": [
            {"package_object_id": package_id, "land_object_id": land_id}
            for package_id, land_id in zip(package_order, land_order)
        ],
    }


def sorted_object_ids(objects: list[dict[str, Any]]) -> list[Any]:
    def key(obj: dict[str, Any]) -> tuple[float, float, str]:
        bbox = obj.get("bbox") or [0, 0, 0, 0]
        if len(bbox) < 4:
            return (0.0, 0.0, str(obj.get("source_object_id")))
        cx = (float(bbox[0]) + float(bbox[2])) / 2.0
        cy = (float(bbox[1]) + float(bbox[3])) / 2.0
        return (round(cy, 3), round(cx, 3), str(obj.get("source_object_id")))

    return [obj.get("source_object_id") for obj in sorted(objects, key=key)]


def active_conflicts(conflicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [conflict for conflict in conflicts if conflict.get("status") != "applied"]


def score_part(conflicts: list[dict[str, Any]], missing_views: list[str]) -> float:
    score = 0.0
    for conflict in active_conflicts(conflicts):
        if conflict.get("type") == "dimension_value_conflict":
            score += 10.0
        elif conflict.get("status") == "ambiguous_match":
            score += 15.0
        elif conflict.get("status") == "pad_count_mismatch":
            score += 20.0
        else:
            score += 5.0
    score += 4.0 * len(missing_views)
    return round(score, 3)


def risk_level_for_score(score: float) -> str:
    if score >= 30.0:
        return "high"
    if score >= 10.0:
        return "medium"
    return "low"


def risk_reasons_for_part(conflicts: list[dict[str, Any]], missing_views: list[str]) -> list[str]:
    reasons = []
    active = active_conflicts(conflicts)
    if active:
        reasons.append(f"{len(active)} conflicts")
    if missing_views:
        reasons.append("missing views: " + ", ".join(missing_views))
    if not reasons:
        reasons.append("no obvious multiview risk signals")
    return reasons


def write_part_outputs(part_dir: Path, canonical: dict[str, Any]) -> None:
    part_dir.mkdir(parents=True, exist_ok=True)
    canonical["summary"]["part_dir"] = str(part_dir)
    (part_dir / UNIFIED_MULTIVIEW_LAYERS_FILENAME).write_text(
        json.dumps(canonical, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (part_dir / "evidence.json").write_text(
        json.dumps(
            {
                "part_number": canonical["part_number"],
                "summary": canonical.get("evidence_summary") or {},
                "evidence_refs": canonical["evidence_refs"],
                "ignored_evidence": canonical["ignored_evidence"],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (part_dir / "conflicts.json").write_text(
        json.dumps(canonical["conflicts"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_svg(part_dir / UNIFIED_MULTIVIEW_LAYERS_SVG_FILENAME, canonical)
    write_source_overlays(part_dir / "source_overlays", canonical)


def write_svg(path: Path, canonical: dict[str, Any]) -> None:
    objects = []
    if canonical.get("outline_2d"):
        objects.append(canonical["outline_2d"])
    objects.extend(canonical.get("package_pads") or [])
    objects.extend(canonical.get("land_pads") or [])
    objects.extend(canonical.get("lead_pads") or [])
    objects.extend(canonical.get("inner_land_pads") or [])
    if not objects:
        path.write_text(empty_svg(canonical["part_number"]), encoding="utf-8")
        return
    boxes = [obj.get("bbox") for obj in objects if len(obj.get("bbox") or []) >= 4]
    if not boxes:
        path.write_text(empty_svg(canonical["part_number"]), encoding="utf-8")
        return
    min_x = min(float(box[0]) for box in boxes)
    min_y = min(float(box[1]) for box in boxes)
    max_x = max(float(box[2]) for box in boxes)
    max_y = max(float(box[3]) for box in boxes)
    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    pad = 20.0
    elements = [
        f'<text x="{min_x}" y="{min_y - 8}" font-size="12" fill="#172033">{escape_xml(canonical["part_number"])}</text>'
    ]
    for obj in objects:
        bbox = obj.get("bbox") or []
        if len(bbox) < 4:
            continue
        x, y, x2, y2 = [float(value) for value in bbox[:4]]
        role = str(obj.get("role") or "")
        color = (
            "#0284c7"
            if role == "outline_2d"
            else "#16a34a"
            if role == "land_pad"
            else "#e11d48"
            if role == "lead_pad"
            else "#7c3aed"
            if role == "inner_land_pad"
            else "#0ea5e9"
        )
        fill = (
            "none"
            if role == "outline_2d"
            else "#fecdd3"
            if role == "lead_pad"
            else "#ddd6fe"
            if role == "inner_land_pad"
            else "#bfdbfe"
        )
        elements.append(canonical_svg_shape(obj, [x, y, x2, y2], fill=fill, stroke=color))
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{min_x - pad} {min_y - pad} {width + pad * 2} {height + pad * 2}">'
        '<rect x="-100000" y="-100000" width="200000" height="200000" fill="#ffffff"/>'
        + "".join(elements)
        + "</svg>\n"
    )
    path.write_text(svg, encoding="utf-8")


def write_source_overlays(output_dir: Path, canonical: dict[str, Any]) -> list[dict[str, Any]]:
    """Draw adopted multiview evidence on each original source image.

    Coordinate system: all adopted object and dimension bboxes are package graph
    source-image pixel coordinates, x right and y down.  The SVG embeds the
    original raster image and overlays only evidence selected into the final
    canonical multiview graph.
    """
    refs = [
        ref
        for ref in canonical.get("evidence_refs") or []
        if str(ref.get("evidence_type") or "") == "package_graph"
    ]
    if not refs:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    objects_by_graph = adopted_objects_by_source_graph(canonical)
    dimensions_by_graph = defaultdict(list)
    for dim in canonical.get("dimensions") or []:
        graph_path = str(dim.get("source_graph") or "")
        if graph_path:
            dimensions_by_graph[graph_path].append(dim)

    manifest = []
    for ref in refs:
        graph_path = str(ref.get("graph_path") or "")
        image_path = Path(str(ref.get("image_path") or ""))
        if not graph_path or not image_path.exists():
            continue
        objects = objects_by_graph.get(graph_path, [])
        dimensions = dimensions_by_graph.get(graph_path, [])
        if not objects and not dimensions:
            continue
        overlay_name = f"{slugify(Path(graph_path).stem.replace('.package_graph', ''))}.adopted.svg"
        overlay_path = output_dir / overlay_name
        write_source_overlay_svg(
            overlay_path,
            image_path=image_path,
            graph_path=graph_path,
            raw_view=str(ref.get("raw_view") or "unknown"),
            canonical_view=str(ref.get("canonical_view") or "unknown"),
            objects=objects,
            dimensions=dimensions,
        )
        manifest.append(
            {
                "graph_path": graph_path,
                "image_path": str(image_path),
                "path": str(overlay_path),
                "raw_view": str(ref.get("raw_view") or "unknown"),
                "canonical_view": str(ref.get("canonical_view") or "unknown"),
                "adopted_object_count": len(objects),
                "adopted_dimension_count": len(dimensions),
            }
        )
    if manifest:
        (output_dir / "manifest.json").write_text(
            json.dumps({"part_number": canonical.get("part_number"), "overlays": manifest}, indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
    return manifest


def adopted_objects_by_source_graph(canonical: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_graph: dict[str, list[dict[str, Any]]] = defaultdict(list)
    objects: list[dict[str, Any]] = []
    if canonical.get("outline_2d"):
        objects.append(canonical["outline_2d"])
    objects.extend(canonical.get("package_pads") or [])
    objects.extend(canonical.get("land_pads") or [])
    objects.extend(canonical.get("lead_contacts") or [])
    objects.extend(canonical.get("lead_pads") or [])
    objects.extend(canonical.get("inner_land_pads") or [])
    for obj in objects:
        graph_path = str(obj.get("source_graph") or "")
        if graph_path:
            by_graph[graph_path].append(obj)
    return by_graph


def write_source_overlay_svg(
    path: Path,
    *,
    image_path: Path,
    graph_path: str,
    raw_view: str,
    canonical_view: str,
    objects: list[dict[str, Any]],
    dimensions: list[dict[str, Any]],
) -> None:
    image_width, image_height = source_image_size(image_path, objects, dimensions)
    object_by_id = {obj.get("source_object_id"): obj for obj in objects}
    href = source_image_href(image_path)
    elements = [
        f'<image href="{escape_xml(href)}" x="0" y="0" width="{image_width}" height="{image_height}" '
        'preserveAspectRatio="none" opacity="0.88"/>',
        '<rect x="0" y="0" width="100%" height="100%" fill="none" stroke="#94a3b8" stroke-width="1"/>',
        (
            f'<text x="12" y="24" font-size="18" font-family="monospace" fill="#0f172a">'
            f'adopted evidence: {escape_xml(raw_view)} -> {escape_xml(canonical_view)}</text>'
        ),
    ]
    for obj in objects:
        bbox = normalized_bbox(obj)
        if bbox is None:
            continue
        role = str(obj.get("role") or "")
        stroke = "#0284c7" if role == "outline_2d" else "#7c3aed" if role == "inner_land_pad" else "#16a34a"
        fill = "none"
        elements.append(
            canonical_svg_shape(
                obj,
                list(bbox),
                fill=fill,
                stroke=stroke,
                stroke_width=4,
                opacity=0.95,
            )
        )
        x1, y1, _, _ = bbox
        label = f"obj:{obj.get('source_object_id')} {role}"
        elements.append(overlay_label(x1, y1, label, fill="#16a34a"))

    for dim in dimensions:
        bbox = normalized_bbox(dim)
        if bbox is not None:
            elements.append(overlay_rect(bbox, stroke="#f97316", width=4, dash="10 6"))
            x1, y1, _, _ = bbox
            elements.append(overlay_label(x1, y1, dimension_overlay_label(dim), fill="#f97316"))
        target_bbox = target_union_bbox(dim, object_by_id)
        if target_bbox is not None:
            expanded = expand_bbox(target_bbox, rel_margin=0.08)
            elements.append(overlay_rect(expanded, stroke="#f59e0b", width=3, dash="5 5"))

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {image_width} {image_height}" '
        f'width="{image_width}" height="{image_height}">'
        + "".join(elements)
        + "</svg>\n"
    )
    path.write_text(svg, encoding="utf-8")


def source_image_size(
    image_path: Path,
    objects: list[dict[str, Any]],
    dimensions: list[dict[str, Any]],
) -> tuple[int, int]:
    parsed = parse_image_size(image_path)
    if parsed:
        return parsed
    boxes = []
    for item in objects + dimensions:
        bbox = normalized_bbox(item)
        if bbox is not None:
            boxes.append(bbox)
    if not boxes:
        return (1000, 1000)
    max_x = max(box[2] for box in boxes)
    max_y = max(box[3] for box in boxes)
    return (max(1, int(math.ceil(max_x + 40))), max(1, int(math.ceil(max_y + 40))))


def parse_image_size(path: Path) -> tuple[int, int] | None:
    try:
        with path.open("rb") as file:
            header = file.read(32)
            if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
                return struct.unpack(">II", header[16:24])
            if header[:2] == b"\xff\xd8":
                return parse_jpeg_size(file)
    except OSError:
        return None
    return None


def parse_jpeg_size(file: Any) -> tuple[int, int] | None:
    file.seek(2)
    while True:
        marker_start = file.read(1)
        if not marker_start:
            return None
        if marker_start != b"\xff":
            continue
        marker = file.read(1)
        while marker == b"\xff":
            marker = file.read(1)
        if not marker:
            return None
        marker_value = marker[0]
        if marker_value in {0xD8, 0xD9}:
            continue
        length_bytes = file.read(2)
        if len(length_bytes) != 2:
            return None
        segment_length = struct.unpack(">H", length_bytes)[0]
        if segment_length < 2:
            return None
        if 0xC0 <= marker_value <= 0xC3:
            data = file.read(5)
            if len(data) != 5:
                return None
            height, width = struct.unpack(">HH", data[1:5])
            return (width, height)
        file.seek(segment_length - 2, 1)


def source_image_href(image_path: Path) -> str:
    """Return a self-contained data URI for SVG overlays.

    SVGs are displayed through HTML <img>.  Browsers commonly block nested
    external image references in that mode, so the source raster is embedded.
    """
    try:
        data = image_path.read_bytes()
    except OSError:
        return ""
    mime_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def target_union_bbox(dim: dict[str, Any], object_by_id: dict[Any, dict[str, Any]]) -> tuple[float, float, float, float] | None:
    boxes = []
    for target_id in dim.get("target_ids") or []:
        target = object_by_id.get(target_id)
        if not target:
            continue
        bbox = normalized_bbox(target)
        if bbox is not None:
            boxes.append(bbox)
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def dimension_overlay_label(dim: dict[str, Any]) -> str:
    dim_id = dim.get("dimension_id")
    text = str(dim.get("text") or "")
    return f"dim:{dim_id} {text}".strip()


def overlay_rect(
    bbox: tuple[float, float, float, float],
    *,
    stroke: str,
    width: int,
    dash: str = "",
) -> str:
    x1, y1, x2, y2 = bbox
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x1}" y="{y1}" width="{max(x2 - x1, 0.0)}" height="{max(y2 - y1, 0.0)}" '
        f'fill="none" stroke="{stroke}" stroke-width="{width}"{dash_attr} opacity="0.95"/>'
    )


def overlay_label(x: float, y: float, text: str, *, fill: str) -> str:
    safe = escape_xml(text)
    label_y = max(14.0, y - 6.0)
    return (
        f'<text x="{x}" y="{label_y}" font-size="16" font-family="monospace" '
        f'font-weight="700" fill="{fill}" stroke="#ffffff" stroke-width="3" paint-order="stroke">{safe}</text>'
    )


def canonical_svg_shape(
    obj: dict[str, Any],
    bbox: list[float],
    *,
    fill: str,
    stroke: str,
    stroke_width: int = 2,
    opacity: float = 0.85,
) -> str:
    """Render unified layer geometry using pad type labels when available.

    Coordinate system: unified multiview layer units, x right and y down. The
    bbox is the geometry used by downstream stages; this helper only changes
    the SVG primitive for review readability.
    """
    x, y, x2, y2 = [float(value) for value in bbox[:4]]
    width = max(x2 - x, 0.0)
    height = max(y2 - y, 0.0)
    label_text = " ".join(
        str(obj.get(key) or "")
        for key in ("label", "source_label", "shape", "node_name", "NodeName")
    ).lower()
    if "circle" in label_text:
        return (
            f'<ellipse cx="{x + width / 2.0}" cy="{y + height / 2.0}" '
            f'rx="{width / 2.0}" ry="{height / 2.0}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}" opacity="{opacity}" />'
        )
    if "dshape" in label_text:
        radius = min(width, height) / 2.0
        if width >= height:
            path = (
                f"M {x} {y} "
                f"L {max(x2 - radius, x)} {y} "
                f"A {radius} {radius} 0 0 1 {max(x2 - radius, x)} {y2} "
                f"L {x} {y2} Z"
            )
        else:
            path = (
                f"M {x} {y} "
                f"L {x2} {y} "
                f"L {x2} {max(y2 - radius, y)} "
                f"A {radius} {radius} 0 0 1 {x} {max(y2 - radius, y)} "
                f"Z"
            )
        return f'<path d="{path}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" opacity="{opacity}" />'
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{stroke_width}" opacity="{opacity}" />'
    )


def empty_svg(part_number: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 120">'
        '<rect width="320" height="120" fill="#ffffff"/>'
        f'<text x="20" y="60" font-size="14" fill="#64748b">{escape_xml(part_number)}: no geometry</text>'
        "</svg>\n"
    )


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
