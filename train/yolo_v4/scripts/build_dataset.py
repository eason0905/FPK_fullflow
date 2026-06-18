from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_ROOT = SCRIPT_DIR.parent
FULLFLOW_ROOT = SCRIPT_DIR.parents[2]
DATASET_ROOT = FULLFLOW_ROOT / "assets" / "datasets" / "dataset_full_v4"
OUTPUT_ROOT = FULLFLOW_ROOT / "assets" / "datasets" / "yolo" / "v4_seed42"

CLASS_NAMES = [
    "lead",
    "outline",
    "package",
    "pad",
    "pad_circle",
    "pad_dshape",
    "num_group",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _bbox_from_points(points: Iterable[Iterable[float]]) -> list[float]:
    pts = [list(point) for point in points]
    xs = [float(point[0]) for point in pts]
    ys = [float(point[1]) for point in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def _resolve_image_path(annotation_path: Path, record: dict) -> Path | None:
    image_name = str(record.get("imagePath") or "").strip()
    if image_name:
        candidate = annotation_path.parent / image_name
        if candidate.exists():
            return candidate

    for suffix in IMAGE_SUFFIXES:
        candidate = annotation_path.with_suffix(suffix)
        if candidate.exists():
            return candidate

    return None


def _yolo_line(class_id: int, bbox: list[float], width: int, height: int) -> str | None:
    x1, y1, x2, y2 = bbox
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    if width <= 0 or height <= 0 or box_w <= 0 or box_h <= 0:
        return None

    x_center = ((x1 + x2) / 2.0) / width
    y_center = ((y1 + y2) / 2.0) / height
    norm_w = box_w / width
    norm_h = box_h / height

    x_center = min(max(x_center, 0.0), 1.0)
    y_center = min(max(y_center, 0.0), 1.0)
    norm_w = min(max(norm_w, 0.0), 1.0)
    norm_h = min(max(norm_h, 0.0), 1.0)
    if norm_w <= 0 or norm_h <= 0:
        return None

    return f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"


def _link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_images:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def _dataset_image_name(annotation_path: Path, image_path: Path) -> str:
    part_name = annotation_path.parent.parent.name
    return f"{part_name}__{image_path.name}"


def _write_dataset_yaml(path: Path, dataset_root: Path) -> None:
    lines = [
        f"path: {dataset_root.resolve()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(CLASS_NAMES)}",
        "",
        "names:",
    ]
    for idx, name in enumerate(CLASS_NAMES):
        lines.append(f"  {idx}: {name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_dataset(
    input_root: Path,
    output_root: Path,
    dataset_yaml: Path,
    val_ratio: float,
    seed: int,
    copy_images: bool,
) -> dict:
    annotation_paths = sorted(
        path for path in input_root.glob("*/extract_image/*.json") if path.name != "info.json"
    )
    rng = random.Random(seed)
    shuffled_paths = annotation_paths[:]
    rng.shuffle(shuffled_paths)

    val_count = int(len(shuffled_paths) * val_ratio)
    val_set = {path for path in shuffled_paths[:val_count]}

    images_train = output_root / "images" / "train"
    images_val = output_root / "images" / "val"
    labels_train = output_root / "labels" / "train"
    labels_val = output_root / "labels" / "val"

    if output_root.exists():
        shutil.rmtree(output_root)
    images_train.mkdir(parents=True, exist_ok=True)
    images_val.mkdir(parents=True, exist_ok=True)
    labels_train.mkdir(parents=True, exist_ok=True)
    labels_val.mkdir(parents=True, exist_ok=True)

    written_images = 0
    written_labels = 0
    class_counts = {name: 0 for name in CLASS_NAMES}
    skipped_missing_image = 0
    skipped_empty = 0

    for annotation_path in annotation_paths:
        record = _load_json(annotation_path)
        image_path = _resolve_image_path(annotation_path, record)
        if image_path is None:
            skipped_missing_image += 1
            continue

        width = int(record.get("imageWidth") or 0)
        height = int(record.get("imageHeight") or 0)
        label_lines: list[str] = []
        for shape in record.get("shapes", []):
            label = str(shape.get("label") or "").strip()
            if label not in CLASS_TO_ID:
                continue
            points = shape.get("points") or []
            if not points:
                continue
            line = _yolo_line(CLASS_TO_ID[label], _bbox_from_points(points), width, height)
            if line is None:
                continue
            label_lines.append(line)
            class_counts[label] += 1

        if not label_lines:
            skipped_empty += 1
            continue

        split = "val" if annotation_path in val_set else "train"
        image_dst_dir = images_val if split == "val" else images_train
        label_dst_dir = labels_val if split == "val" else labels_train

        dataset_image_name = _dataset_image_name(annotation_path, image_path)
        image_dst = image_dst_dir / dataset_image_name
        label_dst = label_dst_dir / f"{Path(dataset_image_name).stem}.txt"
        _link_or_copy(image_path, image_dst, copy_images=copy_images)
        label_dst.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

        written_images += 1
        written_labels += len(label_lines)

    dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
    _write_dataset_yaml(dataset_yaml, output_root)

    summary = {
        "input_root": str(input_root.resolve()),
        "output_root": str(output_root.resolve()),
        "dataset_yaml": str(dataset_yaml.resolve()),
        "total_annotations": len(annotation_paths),
        "written_images": written_images,
        "written_boxes": written_labels,
        "skipped_missing_image": skipped_missing_image,
        "skipped_empty": skipped_empty,
        "val_ratio": val_ratio,
        "seed": seed,
        "copy_images": copy_images,
        "class_counts": class_counts,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a YOLO detection dataset from dataset_full.")
    parser.add_argument("--input-root", type=Path, default=DATASET_ROOT, help="Path to dataset_full.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT, help="Path to YOLO dataset output.")
    parser.add_argument(
        "--dataset-yaml",
        type=Path,
        default=OUTPUT_ROOT / "dataset.yaml",
        help="Path to write YOLO dataset yaml.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split.")
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of creating symlinks.",
    )
    args = parser.parse_args()

    summary = build_dataset(
        input_root=args.input_root.resolve(),
        output_root=args.output_root.resolve(),
        dataset_yaml=args.dataset_yaml.resolve(),
        val_ratio=args.val_ratio,
        seed=args.seed,
        copy_images=args.copy_images,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
