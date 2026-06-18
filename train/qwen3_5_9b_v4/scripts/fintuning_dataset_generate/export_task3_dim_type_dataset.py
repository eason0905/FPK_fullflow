from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

from export_task_common import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT_DIR,
    dimension_block,
    dimension_task3_type,
    ensure_task345_overlay_image,
    export_records,
    is_allowed_view_record,
    record,
    thinking_header,
    tok,
)


OUTPUT_NAME = "task3_dim_type.json"


def build_task3_record(
    record_data: dict,
    dim: dict,
    _: Dict[int, dict],
    dim_index: int,
    output_dir: Path,
) -> Optional[dict]:
    if not is_allowed_view_record(record_data):
        return None

    task_type = dimension_task3_type(dim)
    if task_type is None:
        return None

    image_path = ensure_task345_overlay_image(record_data, dim, dim_index, output_dir)

    question_lines = thinking_header() + [
        "",
        "圖片中每個物件都已框出並標上 ID，目標 num_group 也已用紅框標示。",
        "以下提供一個 dimension 的文字內容與 bbox。",
        "請判斷它的 dimension 類型。",
        "",
    ] + dimension_block(dim) + [
        "",
        "請只輸出 1 行答案。",
        "可能答案只有 4 種：size / distance / other / diameter",
        "size: 單一物件的一般長寬尺寸。",
        "distance: 兩個物件之間的距離。",
        "other: 單一物件的其他特殊幾何尺寸，但不包含 diameter。",
        "diameter: 單一物件的直徑。",
    ]
    answer_lines = [tok(task_type)]
    return record(image_path, question_lines, answer_lines)


def export_dataset(input_json: Path, output_dir: Path, limit: int = 0) -> Dict[str, int]:
    return export_records(input_json, output_dir / OUTPUT_NAME, build_task3_record, limit=limit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export task3 dimension type dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="dataset_full directory path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--limit", type=int, default=0, help="Optional record limit for debugging.")
    args = parser.parse_args()

    summary = export_dataset(args.input.resolve(), args.output_dir.resolve(), limit=args.limit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
