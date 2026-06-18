from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from export_task_common import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT_DIR,
    box_str,
    dump_json,
    has_nonempty_num_group_record,
    iter_dataset_records,
    is_allowed_view_record,
    record,
    resolve_image_path,
    thinking_header,
    write_markdown,
)


def _ground_truth_text(dim: dict) -> str:
    return str(dim.get("description") or dim.get("text") or "").strip()


OUTPUT_NAME = "task2_num_group_recognition.json"


def _build_task2_record(record_data: dict, dim: dict) -> Optional[dict]:
    answer_text = _ground_truth_text(dim)
    if not answer_text:
        return None

    question_lines = thinking_header() + [
        "",
        "以下提供一個 num_group 的 bbox。",
        "請讀出這個 bbox 內的完整文字內容。",
        "不要補字，不要解釋，只輸出辨識結果。",
        "",
        "NUM_GROUP_BEGIN",
        box_str(dim.get("bbox_norm_1000") or [0, 0, 0, 0]),
        "NUM_GROUP_END",
        "",
        "請只輸出 1 行文字答案。",
    ]
    answer_lines = [answer_text]
    return record(resolve_image_path(record_data), question_lines, answer_lines)


def export_dataset(input_json: Path, output_dir: Path, limit: int = 0) -> Dict[str, int]:
    task2_records: List[dict] = []

    processed = 0
    for record in iter_dataset_records(input_json, limit=limit):
        if not is_allowed_view_record(record) or not has_nonempty_num_group_record(record):
            continue

        for dim in record.get("dimensions", []):
            rec2 = _build_task2_record(record, dim)
            if rec2:
                task2_records.append(rec2)
        processed += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / OUTPUT_NAME
    dump_json(out_path, task2_records)
    write_markdown(task2_records, out_path)

    summary = {
        "output_records": len(task2_records),
        "processed_records": processed,
    }
    dump_json(output_dir / "task2_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export task2 num_group recognition dataset from dataset_full."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="dataset_full directory path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--limit", type=int, default=0, help="Optional record limit for debugging.")
    args = parser.parse_args()

    summary = export_dataset(args.input.resolve(), args.output_dir.resolve(), limit=args.limit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
