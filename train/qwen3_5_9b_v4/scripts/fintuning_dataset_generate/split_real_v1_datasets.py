from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from datasets import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "real_image_process" / "dataset" / "dataset_json" / "v1"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "real_image_process" / "dataset" / "dataset_json" / "splits" / "real_v1_seed42"
DEFAULT_DATASETS = [
    "task1_view_classification.json",
    "task2_num_group_recognition.json",
    "task3_dim_type.json",
    "task4_dim_target.json",
    "task5_dim_start_anchor.json",
]


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def split_dataset_file(dataset_path: Path, output_root: Path, val_size: float, seed: int) -> Dict[str, int]:
    records: List[dict] = _load_json(dataset_path)
    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(test_size=val_size, seed=seed)

    train_records = list(split["train"])
    val_records = list(split["test"])

    train_path = output_root / "train" / dataset_path.name
    val_path = output_root / "val" / dataset_path.name
    _dump_json(train_path, train_records)
    _dump_json(val_path, val_records)

    return {
        "train": len(train_records),
        "val": len(val_records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split real-image v1 task datasets into train/val with the same seed logic as LlamaFactory."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR, help="Directory containing task JSON files.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output root for split datasets.")
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated dataset filenames to split.",
    )
    parser.add_argument("--val-size", type=float, default=0.05, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Split seed.")
    args = parser.parse_args()

    dataset_names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    summary: Dict[str, Dict[str, int]] = {}
    for name in dataset_names:
        dataset_path = args.dataset_dir.resolve() / name
        summary[name] = split_dataset_file(dataset_path, args.output_root.resolve(), args.val_size, args.seed)

    overall = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "output_root": str(args.output_root.resolve()),
        "val_size": args.val_size,
        "seed": args.seed,
        "datasets": summary,
    }
    _dump_json(args.output_root.resolve() / "summary.json", overall)
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
