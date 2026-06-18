from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from export_task_common import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT_DIR,
    dump_json,
    iter_dataset_records,
    is_allowed_view_record,
    record,
    resolve_image_path,
    thinking_header,
    tok,
    write_markdown,
)


OUTPUT_NAME = "task1_view_classification.json"


def _build_task1_record(record_data: dict) -> dict | None:
    view = str(record_data.get("image", {}).get("view_info") or "").strip().lower()
    if not view:
        return None

    question_lines = thinking_header() + [
        "",
        "請判斷這張 IC 封裝圖的視角類型。",
        "只需根據整張圖的視覺內容回答，不要解釋。",
        "",
        "請只輸出 1 行答案。",
        "答案只會是以下其中一種：top / bottom / land / front / land_detail / side / lead。",
    ]
    answer_lines = [tok(view)]
    return record(resolve_image_path(record_data), question_lines, answer_lines)


def export_dataset(input_json: Path, output_dir: Path, limit: int = 0) -> Dict[str, int]:
    task1_records: List[dict] = []

    processed = 0
    for rec in iter_dataset_records(input_json, limit=limit):
        if not is_allowed_view_record(rec):
            continue
        built = _build_task1_record(rec)
        if built:
            task1_records.append(built)
        processed += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / OUTPUT_NAME
    dump_json(out_path, task1_records)
    write_markdown(task1_records, out_path)

    summary = {
        "output_records": len(task1_records),
        "processed_records": processed,
    }
    dump_json(output_dir / "task1_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export task1 view classification dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="dataset_full directory path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--limit", type=int, default=0, help="Optional record limit for debugging.")
    args = parser.parse_args()

    summary = export_dataset(args.input.resolve(), args.output_dir.resolve(), limit=args.limit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
