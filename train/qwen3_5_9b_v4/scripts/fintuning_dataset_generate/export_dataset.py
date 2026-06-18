from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from export_task1_view_classification_dataset import export_dataset as export_task1_dataset
from export_task2_num_group_recognition_dataset import export_dataset as export_task2_dataset
from export_task3_dim_type_dataset import export_dataset as export_task3_dataset
from export_task4_dim_target_dataset import export_dataset as export_task4_dataset
from export_task5_dim_start_anchor_dataset import export_dataset as export_task5_dataset
from export_task_common import DEFAULT_INPUT, DEFAULT_OUTPUT_DIR, dump_json


def export_datasets(input_json: Path, output_dir: Path, limit: int = 0) -> Dict[str, int]:
    task1_summary = export_task1_dataset(input_json, output_dir, limit=limit)
    task2_summary = export_task2_dataset(input_json, output_dir, limit=limit)
    task3_summary = export_task3_dataset(input_json, output_dir, limit=limit)
    task4_summary = export_task4_dataset(input_json, output_dir, limit=limit)
    task5_summary = export_task5_dataset(input_json, output_dir, limit=limit)

    summary = {
        "task1_view_classification": task1_summary["output_records"],
        "task2_num_group_recognition": task2_summary["output_records"],
        "task3_dim_type": task3_summary["output_records"],
        "task4_dim_target": task4_summary["output_records"],
        "task5_dim_start_anchor": task5_summary["output_records"],
        "processed_records": max(
            task1_summary["processed_records"],
            task2_summary["processed_records"],
            task3_summary["processed_records"],
            task4_summary["processed_records"],
            task5_summary["processed_records"],
        ),
    }
    dump_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export five real-image finetuning datasets for Qwen3-VL Thinking / LLaMAFactory."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="dataset_full directory path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--limit", type=int, default=0, help="Optional record limit for debugging.")
    args = parser.parse_args()

    summary = export_datasets(args.input.resolve(), args.output_dir.resolve(), limit=args.limit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
