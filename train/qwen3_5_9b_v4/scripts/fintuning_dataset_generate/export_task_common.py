from __future__ import annotations

import importlib.util
import inspect
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from process_dataset_for_fintuning import bbox_to_qwen_1000, decode_dimension_pairs


def _find_project_root(start: Path) -> Path:
    for parent in start.parents:
        if (parent / "real_image_process" / "dataset").is_dir():
            return parent
    return start.parents[6]


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
DATASET_ROOT = Path(
    os.environ.get(
        "FPK_DATASET_ROOT",
        PROJECT_ROOT / "real_image_process" / "FPK_PJ_fullflow" / "assets" / "datasets",
    )
).resolve()
REAL_IMAGE_ROOT = DATASET_ROOT.parent
DEFAULT_INPUT = (
    DATASET_ROOT / "dataset_full_v1"
    if (DATASET_ROOT / "dataset_full_v1").exists()
    else DATASET_ROOT / "dataset_full"
)
DEFAULT_OUTPUT_DIR = DATASET_ROOT / "dataset_json" / "v1"

OBJECT_LABELS = {
    "pad",
    "pad_circle",
    "pad_dshape",
    "outline",
    "reference_object",
    "package",
    "lead",
}

OVERLAY_OBJECT_LABELS = {
    "outline",
    "pad",
    "pad_circle",
    "pad_dshape",
    "lead",
    "package",
}
ALLOWED_VIEWS = {"top", "bottom", "land", "front", "land_detail", "side", "lead"}

OBJECT_COLORS = {
    "outline": "#2E86DE",
    "pad": "#16A085",
    "pad_circle": "#27AE60",
    "pad_dshape": "#1ABC9C",
    "lead": "#D35400",
    "package": "#8E44AD",
}

DIMENSION_COLOR = "#E74C3C"
DIMENSION_TEXT_BG = "#FFF3CD"
OBJECT_TEXT_COLOR = "#111111"

_INFO_CACHE: Dict[Path, dict] = {}
_APO_RENDER_MODULE = None


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def tok(text: str) -> str:
    return f"<|object_ref_start|>{text}<|object_ref_end|>"


def box_str(box: Iterable[int]) -> str:
    x1, y1, x2, y2 = [int(v) for v in box]
    return f"<|box_start|>{x1},{y1},{x2},{y2}<|box_end|>"


def _load_apo_render_module():
    global _APO_RENDER_MODULE
    if _APO_RENDER_MODULE is not None:
        return _APO_RENDER_MODULE

    candidate_paths = [
        PROJECT_ROOT / "real_image_process" / "APO" / "apo-miprov2" / "build_task345_dataset.py",
        PROJECT_ROOT / "real_image_process" / "apo-miprov2" / "build_task345_dataset.py",
    ]
    module_path = next((path for path in candidate_paths if path.exists()), candidate_paths[0])
    spec = importlib.util.spec_from_file_location("apo_build_task345_dataset", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load APO render module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _APO_RENDER_MODULE = module
    return module


def _bbox_from_points(points: Iterable[Iterable[float]]) -> List[float]:
    pts = [list(point) for point in points]
    xs = [float(point[0]) for point in pts]
    ys = [float(point[1]) for point in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def _resolve_image_name(annotation_path: Path, raw_record: dict) -> str:
    image_name = str(raw_record.get("imagePath") or "").strip()
    if image_name:
        candidate = annotation_path.parent / image_name
        if candidate.exists():
            return candidate.name

    for suffix in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        candidate = annotation_path.with_suffix(suffix)
        if candidate.exists():
            return candidate.name

    # Fall back to the annotation stem with png for visibility if the image is missing.
    return annotation_path.with_suffix(".png").name


def _image_relative_path(annotation_path: Path, raw_record: dict) -> str:
    image_name = _resolve_image_name(annotation_path, raw_record)
    return str(annotation_path.parent.relative_to(DATASET_ROOT) / image_name)


def _build_image_info(annotation_path: Path, raw_record: dict) -> dict:
    image_name = _resolve_image_name(annotation_path, raw_record)
    info = _load_info(annotation_path.parent / "info.json")
    view_info = _lookup_view_info(info, image_name)
    return {
        "file_name": image_name,
        "relative_path": _image_relative_path(annotation_path, raw_record),
        "width": int(raw_record.get("imageWidth") or 0),
        "height": int(raw_record.get("imageHeight") or 0),
        "view_info": view_info,
    }


def _load_info(info_path: Path) -> dict:
    cached = _INFO_CACHE.get(info_path)
    if cached is not None:
        return cached
    if not info_path.exists():
        payload: dict = {}
    else:
        payload = load_json(info_path)
    _INFO_CACHE[info_path] = payload
    return payload


def _lookup_view_info(info: dict, image_name: str) -> Optional[str]:
    for image_info in info.get("images", []):
        if str(image_info.get("file_name") or "").strip() == image_name:
            return str(image_info.get("view") or "").strip() or None
    return None


def resolve_image_path(record: dict) -> str:
    rel = record.get("image", {}).get("relative_path") or ""
    return str((DATASET_ROOT / rel).resolve())


def normalize_label(label: Optional[str]) -> str:
    mapping = {
        "pad": "pad_rect",
        "pad_circle": "pad_circle",
        "pad_dshape": "pad_dshape",
        "outline": "outline",
        "reference_object": "reference_object",
        "package": "package",
        "lead": "lead",
    }
    return mapping.get((label or "").strip(), (label or "").strip() or "unknown")


def normalize_dataset_text(text: Optional[str]) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    return re.sub(r"(?i)theta", "θ", raw)


def anchor_label(anchor: str) -> str:
    mapping = {
        "top": "top_edge",
        "bottom": "bottom_edge",
        "left": "left_edge",
        "right": "right_edge",
        "horizontal_centerline": "horizontal_centerline",
        "vertical_centerline": "vertical_centerline",
    }
    return mapping.get(anchor, anchor)


def split_relation_name(relation_name: Optional[str]) -> Optional[Tuple[str, str]]:
    if not relation_name or "_to_" not in relation_name:
        return None
    left, right = relation_name.split("_to_", 1)
    return anchor_label(left), anchor_label(right)


def is_exportable_record(record: dict) -> bool:
    return (
        record.get("annotation_status") == "full"
        and not record.get("warnings")
        and is_allowed_view_record(record)
        and has_all_num_group_linking_record(record)
        and has_nonempty_num_group_record(record)
    )


def is_top_view_record(record: dict) -> bool:
    return (record.get("image", {}).get("view_info") or "").strip().lower() == "top"


def is_allowed_view_record(record: dict) -> bool:
    return (record.get("image", {}).get("view_info") or "").strip().lower() in ALLOWED_VIEWS


def has_nonempty_num_group_record(record: dict) -> bool:
    return any(str(dim.get("text") or "").strip() for dim in record.get("dimensions", []))


def has_all_num_group_linking_record(record: dict) -> bool:
    return bool(record.get("all_num_group_linked"))


def is_noise_or_invalid_dimension(dim: dict) -> bool:
    return bool(dim.get("is_noise")) or not dim.get("target_ids") or not dim.get("text")


def dimension_task_type(dim: dict) -> Optional[str]:
    if is_noise_or_invalid_dimension(dim):
        return None
    target_count = len(dim.get("target_ids") or [])
    if target_count == 1:
        return "size"
    if target_count == 2:
        return "distance"
    return None


def dimension_task3_type(dim: dict) -> Optional[str]:
    if is_noise_or_invalid_dimension(dim):
        return None

    target_count = len(dim.get("target_ids") or [])
    if target_count == 2:
        return "distance"
    if target_count != 1:
        return None

    geometry_label = str(dim.get("geometry_label") or "").strip()
    if geometry_label == "diameter":
        return "diameter"
    if geometry_label in {"vertical_length", "horizontal_length"}:
        return "size"
    if geometry_label in {
        "lead_thickness",
        "corner_radius_or_notch",
        "groove_depth_or_protrusion_length",
    }:
        return "other"

    return "size"


def build_object_map(record: dict) -> Dict[int, dict]:
    return {int(obj["id"]): obj for obj in record.get("objects", [])}


def thinking_header() -> List[str]:
    return [
        "你是 IC 封裝圖幾何理解助手。",
        "你可以先在心中分析圖片、dimension 文字與 bbox 的相對位置，再輸出答案。",
        "不要輸出思考過程，不要解釋理由，只輸出指定格式。",
    ]


def dimension_block(dim: dict) -> List[str]:
    return [
        "DIMENSION_BEGIN",
        f"{tok(str(dim.get('text') or ''))} {box_str(dim.get('bbox_norm_1000') or [0, 0, 0, 0])}",
        "DIMENSION_END",
    ]


def dimension_block_plain_text(dim: dict) -> List[str]:
    return [
        "DIMENSION_BEGIN",
        f"{str(dim.get('text') or '')} {box_str(dim.get('bbox_norm_1000') or [0, 0, 0, 0])}",
        "DIMENSION_END",
    ]


def human_turn(question_lines: List[str]) -> str:
    return "<image>\n" + "\n".join(question_lines).strip()


def record(image_path: str, question_lines: List[str], answer_lines: List[str]) -> dict:
    return {
        "conversations": [
            {"from": "human", "value": human_turn(question_lines)},
            {"from": "gpt", "value": "\n".join(answer_lines).strip()},
        ],
        "images": [image_path],
    }


def pixel_box_from_norm_1000(box: Iterable[int], image_width: int, image_height: int) -> List[int]:
    x1, y1, x2, y2 = [int(v) for v in box]
    return [
        int(round(x1 * image_width / 1000.0)),
        int(round(y1 * image_height / 1000.0)),
        int(round(x2 * image_width / 1000.0)),
        int(round(y2 * image_height / 1000.0)),
    ]


def overlay_sort_key(box: Iterable[int]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = [int(v) for v in box]
    return ((y1 + y2) / 2.0, (x1 + x2) / 2.0, y1, x1)


def build_overlay_objects(record_data: dict) -> List[dict]:
    objects = [
        obj for obj in record_data.get("objects", [])
        if obj.get("label") in OVERLAY_OBJECT_LABELS
    ]
    ordered = sorted(
        objects,
        key=lambda obj: overlay_sort_key(
            obj.get("bbox_xyxy")
            or obj.get("bbox_norm_1000")
            or [0, 0, 0, 0]
        ),
    )
    overlay_objects: List[dict] = []
    for overlay_id, obj in enumerate(ordered, start=1):
        payload = dict(obj)
        payload["overlay_id"] = overlay_id
        overlay_objects.append(payload)
    return overlay_objects


def build_overlay_object_map(record_data: dict) -> Dict[int, dict]:
    return {int(obj["id"]): obj for obj in build_overlay_objects(record_data)}


def ensure_task345_overlay_image(
    record_data: dict,
    dim: dict,
    dim_index: int,
    output_dir: Path,
) -> str:
    apo_module = _load_apo_render_module()
    source_image_path = Path(resolve_image_path(record_data))
    apo_objects = [
        apo_module.ApoObject(
            raw_id=int(obj["id"]),
            object_id=int(obj["overlay_id"]),
            label=str(obj["label"]),
            bbox_xyxy=[int(v) for v in (obj.get("bbox_xyxy") or [])],
        )
        for obj in build_overlay_objects(record_data)
    ]
    dimension_box = [int(v) for v in (dim.get("bbox_xyxy") or [])]

    image_name = Path(record_data.get("image", {}).get("file_name") or "image.png").stem
    overlay_rel_path = Path("task345_overlay_images") / f"{image_name}__dim{dim_index + 1:03d}.png"
    overlay_abs_path = output_dir / overlay_rel_path
    overlay_abs_path.parent.mkdir(parents=True, exist_ok=True)
    if overlay_abs_path.exists():
        if overlay_abs_path.stat().st_size > 0:
            return str(overlay_abs_path.resolve())
        overlay_abs_path.unlink()

    original_compute_render_scale = apo_module.compute_render_scale

    def capped_compute_render_scale(objects, image_width, image_height):
        return min(original_compute_render_scale(objects, image_width, image_height), 1.5)

    apo_module.compute_render_scale = capped_compute_render_scale
    try:
        apo_module.render_overlay_image(
            source_image_path=source_image_path,
            output_path=overlay_abs_path,
            objects=apo_objects,
            dimension_box=dimension_box,
        )
    finally:
        apo_module.compute_render_scale = original_compute_render_scale
    return str(overlay_abs_path.resolve())


def write_markdown(records: List[dict], output_path: Path) -> None:
    md_path = output_path.with_suffix(".md")
    lines: List[str] = [
        "# 資料集可讀性版本",
        "",
        f"共 {len(records)} 筆記錄",
        "",
        "---",
        "",
    ]
    for idx, rec in enumerate(records, 1):
        lines.append(f"## 記錄 {idx}")
        lines.append("")
        for turn in rec.get("conversations", []):
            title = "Human" if turn.get("from") == "human" else "Assistant"
            lines.append(f"### {title}")
            lines.append("")
            lines.append("```")
            lines.append(turn.get("value", ""))
            lines.append("```")
            lines.append("")
        for img in rec.get("images", []):
            lines.append(f"- `{img}`")
        lines.append("")
        lines.append("---")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def _normalize_object(shape: dict, image_width: int, image_height: int) -> Optional[dict]:
    group_id = shape.get("group_id")
    if group_id is None or shape.get("label") not in OBJECT_LABELS:
        return None

    bbox = _bbox_from_points(shape.get("points") or [])
    return {
        "id": int(group_id),
        "label": shape.get("label"),
        "bbox_xyxy": [int(round(v)) for v in bbox],
        "bbox_norm_1000": bbox_to_qwen_1000(bbox, image_width, image_height),
    }


def _normalize_dimension(shape: dict, image_width: int, image_height: int) -> Optional[dict]:
    if shape.get("label") != "num_group":
        return None

    text = normalize_dataset_text(shape.get("description"))
    if not text:
        return None

    bbox = _bbox_from_points(shape.get("points") or [])
    normalized = {
        "text": text,
        "description": text,
        "bbox_xyxy": [int(round(v)) for v in bbox],
        "bbox_norm_1000": bbox_to_qwen_1000(bbox, image_width, image_height),
        "is_noise": False,
        "target_ids": [],
        "relation_type": None,
        "relation_name": None,
        "geometry_label": None,
        "dimension_orientation": "unknown",
    }

    kie_linking = shape.get("kie_linking") or []
    if len(kie_linking) >= 2:
        try:
            decoded = decode_dimension_pairs(kie_linking[0], kie_linking[1])
        except ValueError:
            decoded = None
        if decoded is not None:
            normalized["target_ids"] = list(decoded.get("target_ids") or [])
            normalized["relation_type"] = decoded.get("relation_type")
            normalized["is_noise"] = decoded.get("geometry_label") == "noise"
            normalized["geometry_label"] = decoded.get("geometry_label")
            normalized["dimension_orientation"] = decoded.get("dimension_orientation") or "unknown"
            if decoded.get("relation_type") == "anchors":
                normalized["relation_name"] = (
                    f"{decoded['start_anchor']}_to_{decoded['end_anchor']}"
                )

    return normalized


def _normalize_raw_record(annotation_path: Path) -> dict:
    raw_record = load_json(annotation_path)
    image_width = int(raw_record.get("imageWidth") or 0)
    image_height = int(raw_record.get("imageHeight") or 0)
    num_group_shapes = [
        shape for shape in (raw_record.get("shapes") or [])
        if shape.get("label") == "num_group"
    ]

    objects: List[dict] = []
    dimensions: List[dict] = []

    for shape in raw_record.get("shapes", []):
        normalized_object = _normalize_object(shape, image_width, image_height)
        if normalized_object is not None:
            objects.append(normalized_object)

        normalized_dimension = _normalize_dimension(shape, image_width, image_height)
        if normalized_dimension is not None:
            dimensions.append(normalized_dimension)

    return {
        "annotation_status": "full",
        "warnings": [],
        "image": _build_image_info(annotation_path, raw_record),
        "objects": objects,
        "dimensions": dimensions,
        "all_num_group_linked": bool(num_group_shapes) and all(
            bool(shape.get("kie_linking") or []) for shape in num_group_shapes
        ),
    }


def iter_dataset_records(dataset_root: Path, limit: int = 0) -> Iterator[dict]:
    processed = 0
    for annotation_path in sorted(dataset_root.glob("*/extract_image/*.json")):
        if annotation_path.name == "info.json":
            continue
        yield _normalize_raw_record(annotation_path)
        processed += 1
        if limit > 0 and processed >= limit:
            break


def export_records(
    input_json: Path,
    output_path: Path,
    builder,
    limit: int = 0,
) -> Dict[str, int]:
    output_records: List[dict] = []
    processed = 0

    builder_param_count = len(inspect.signature(builder).parameters)

    for rec in iter_dataset_records(input_json, limit=limit):
        if not is_exportable_record(rec):
            continue

        obj_map = build_object_map(rec)
        for dim_index, dim in enumerate(rec.get("dimensions", [])):
            if builder_param_count >= 5:
                built = builder(rec, dim, obj_map, dim_index, output_path.parent)
            else:
                built = builder(rec, dim, obj_map)
            if built:
                output_records.append(built)

        processed += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(output_path, output_records)
    write_markdown(output_records, output_path)
    return {
        "output_records": len(output_records),
        "processed_records": processed,
    }
