from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

from export_task_common import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT_DIR,
    build_overlay_object_map,
    dimension_block,
    dimension_task3_type,
    ensure_task345_overlay_image,
    export_records,
    is_allowed_view_record,
    record,
    thinking_header,
    tok,
)


OUTPUT_NAME = "task4_dim_target.json"


def build_task4_record(
    record_data: dict,
    dim: dict,
    obj_map: Dict[int, dict],
    dim_index: int,
    output_dir: Path,
) -> Optional[dict]:
    if not is_allowed_view_record(record_data):
        return None

    task_type = dimension_task3_type(dim)
    if task_type is None:
        return None
    if task_type == "other":
        return None

    overlay_obj_map = build_overlay_object_map(record_data)
    image_path = ensure_task345_overlay_image(record_data, dim, dim_index, output_dir)

    target_ids = list(dim.get("target_ids") or [])
    target_objs = [obj_map.get(int(obj_id)) for obj_id in target_ids]
    if any(obj is None for obj in target_objs):
        return None

    target_overlay_ids = []
    for obj_id in target_ids:
        overlay_obj = overlay_obj_map.get(int(obj_id))
        if overlay_obj is None:
            return None
        target_overlay_ids.append(int(overlay_obj["overlay_id"]))

    question_lines = thinking_header() + [""] + dimension_block(dim) + [""]

    if task_type in {"size", "diameter"}:
        question_lines += [
            "圖片中每個物件都已框出並標上 ID，目標 num_group 也已用紅框標示。",
            "已知這個 dimension 描述的是單一物件尺寸。",
            "請指出它描述的是哪個物件，並輸出該物件的 ID。",
            "",
            "請只輸出 1 行答案：",
            "1) 物件 ID",
        ]
        answer_lines = [tok(str(target_overlay_ids[0]))]
        return record(image_path, question_lines, answer_lines)

    question_lines += [
        "圖片中每個物件都已框出並標上 ID，目標 num_group 也已用紅框標示。",
        "已知這個 dimension 描述的是兩個物件之間的距離。",
        "請指出它描述的是哪兩個物件，並輸出兩個物件的 ID。",
        "注意要保留順序。",
        "",
        "請依固定格式輸出 2 行：",
        "1) 第一個物件 ID",
        "2) 第二個物件 ID",
    ]
    answer_lines = [
        tok(str(target_overlay_ids[0])),
        tok(str(target_overlay_ids[1])),
    ]
    return record(image_path, question_lines, answer_lines)


def export_dataset(input_json: Path, output_dir: Path, limit: int = 0) -> Dict[str, int]:
    return export_records(input_json, output_dir / OUTPUT_NAME, build_task4_record, limit=limit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export task4 dimension target dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="dataset_full directory path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--limit", type=int, default=0, help="Optional record limit for debugging.")
    args = parser.parse_args()

    summary = export_dataset(args.input.resolve(), args.output_dir.resolve(), limit=args.limit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
