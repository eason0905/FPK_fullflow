from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

from export_task_common import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT_DIR,
    build_overlay_object_map,
    dimension_block_plain_text,
    dimension_task3_type,
    ensure_task345_overlay_image,
    export_records,
    is_allowed_view_record,
    record,
    split_relation_name,
    thinking_header,
    tok,
)


OUTPUT_NAME = "task5_dim_start_anchor.json"


def _coarse_geometry_label(label: str) -> str:
    mapping = {
        "vertical_length": "vertical_line",
        "horizontal_length": "horizontal_line",
        "lead_thickness": "other",
        "diameter": "diameter",
        "corner_radius_or_notch": "other",
        "groove_depth_or_protrusion_length": "other",
    }
    return mapping.get(label, "other")


def _coarse_anchor_label(label: str) -> str:
    if label in {"horizontal_centerline", "vertical_centerline"}:
        return "center"
    return label


def build_task5_record(
    record_data: dict,
    dim: dict,
    obj_map: Dict[int, dict],
    dim_index: int,
    output_dir: Path,
) -> Optional[dict]:
    if not is_allowed_view_record(record_data):
        return None

    task3_type = dimension_task3_type(dim)
    if task3_type is None or task3_type == "diameter":
        return None

    target_ids = list(dim.get("target_ids") or [])
    relation_type = dim.get("relation_type")
    overlay_obj_map = build_overlay_object_map(record_data)
    image_path = ensure_task345_overlay_image(record_data, dim, dim_index, output_dir)

    question_lines = thinking_header() + [
        "",
        "圖片中每個物件都已框出並標上 ID，目標 num_group 也已用紅框標示。",
        "以下提供一個 dimension 的文字內容與 bbox，還有它對應的參考物件 ID。",
        "請判斷這個 dimension 的『起始語意標籤』。",
        "如果它描述的是兩個物件之間的距離，答案應是第一個物件上的起點 anchor。",
        "如果它描述的是單一物件的特殊幾何尺寸，答案應是該尺寸本身的幾何類型。",
        "你可以參考 dimension 線方向、文字位置、箭頭指向、以及物件相對位置。",
        "",
    ] + dimension_block_plain_text(dim) + [""]

    if len(target_ids) == 1:
        obj = obj_map.get(int(target_ids[0]))
        if obj is None:
            return None
        overlay_obj = overlay_obj_map.get(int(target_ids[0]))
        if overlay_obj is None:
            return None

        geometry_label = dim.get("geometry_label")
        if not geometry_label or geometry_label == "noise":
            return None

        coarse_geometry_label = _coarse_geometry_label(geometry_label)

        question_lines += [
            "已知這個 dimension 描述的是單一物件尺寸。",
            "當 dimension 屬於特殊情況時，請直接輸出幾何語意名稱，而不是 edge 名稱。",
            "",
            "參考物件如下：",
            f"{tok('object_id')} {tok(str(overlay_obj['overlay_id']))}",
            "",
            "請只輸出 1 行答案。",
            "特殊幾何答案只會有 3 種：vertical_line / horizontal_line / other",
        ]
        answer_lines = [tok(coarse_geometry_label)]
        return record(image_path, question_lines, answer_lines)

    if len(target_ids) != 2 or relation_type != "anchors":
        return None

    anchors = split_relation_name(dim.get("relation_name"))
    if anchors is None:
        return None

    obj_a = obj_map.get(int(target_ids[0]))
    obj_b = obj_map.get(int(target_ids[1]))
    if obj_a is None or obj_b is None:
        return None
    overlay_obj_a = overlay_obj_map.get(int(target_ids[0]))
    overlay_obj_b = overlay_obj_map.get(int(target_ids[1]))
    if overlay_obj_a is None or overlay_obj_b is None:
        return None

    start_anchor, end_anchor = anchors
    coarse_start_anchor = _coarse_anchor_label(start_anchor)
    coarse_end_anchor = _coarse_anchor_label(end_anchor)
    question_lines += [
        "已知這個 dimension 描述的是兩個物件之間的距離。",
        "請分別輸出第一個物件上的起點 anchor 與第二個物件上的終點 anchor。",
        "注意要保留順序：先起點，再終點。",
        "",
        "參考物件如下：",
        f"{tok('object_a_id')} {tok(str(overlay_obj_a['overlay_id']))}",
        f"{tok('object_b_id')} {tok(str(overlay_obj_b['overlay_id']))}",
        "",
        "請依固定格式輸出 2 行答案：",
        "1) 起點 anchor",
        "2) 終點 anchor",
        "一般 anchor 可能答案：top_edge / bottom_edge / left_edge / right_edge / center",
        "如果原本是 horizontal_centerline 或 vertical_centerline，都統一輸出 center",
    ]
    answer_lines = [tok(coarse_start_anchor), tok(coarse_end_anchor)]
    return record(image_path, question_lines, answer_lines)


def export_dataset(input_json: Path, output_dir: Path, limit: int = 0) -> Dict[str, int]:
    return export_records(input_json, output_dir / OUTPUT_NAME, build_task5_record, limit=limit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export task5 dimension anchor dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="dataset_full directory path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--limit", type=int, default=0, help="Optional record limit for debugging.")
    args = parser.parse_args()

    summary = export_dataset(args.input.resolve(), args.output_dir.resolve(), limit=args.limit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
