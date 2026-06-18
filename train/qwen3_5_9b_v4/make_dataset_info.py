from __future__ import annotations

import argparse
import json
from pathlib import Path


TASKS = [
    "task1_view_classification",
    "task2_num_group_recognition",
    "task3_dim_type",
    "task4_dim_target",
    "task5_dim_start_anchor",
]


def dataset_entry(file_name: str) -> dict:
    return {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "images": "images",
        },
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
            "system_tag": "system",
        },
    }


def build_dataset_info(prefix: str) -> dict:
    payload = {}
    for split in ("train", "val"):
        for task in TASKS:
            payload[f"{prefix}_{split}_{task}"] = dataset_entry(f"{split}/{task}.json")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Write LlamaFactory dataset_info.json for split task datasets.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", default="real_v4_seed42")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(build_dataset_info(args.prefix), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
