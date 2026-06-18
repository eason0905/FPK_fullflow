from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..render import render_root_index, render_task_gallery
from ..schema import ReviewCase, relative_path, slugify
from .llm_errors import direct_annotation_info, write_cases


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class YoloBox:
    cls_id: int
    xyxy: tuple[float, float, float, float]
    conf: float | None = None


def build_yolo_error_review(
    *,
    model_path: Path,
    data_yaml: Path,
    output_root: Path,
    split: str = "val",
    conf: float = 0.25,
    iou: float = 0.5,
    imgsz: int = 1280,
    device: str = "0",
    max_images: int = 0,
) -> dict[str, Any]:
    import cv2
    from ultralytics import YOLO

    data_cfg = load_dataset_yaml(data_yaml)
    dataset_root = Path(data_cfg["path"]).resolve()
    class_names = normalize_class_names(data_cfg["names"])
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    image_paths = [
        path for path in sorted(image_dir.iterdir()) if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if max_images > 0:
        image_paths = image_paths[:max_images]

    model = YOLO(str(model_path.resolve()))
    cases: list[ReviewCase] = []
    for index, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        image_h, image_w = image.shape[:2]
        gt_boxes = load_gt_boxes(label_dir / f"{image_path.stem}.txt", image_w, image_h)
        pred_boxes = predict_boxes(model, image_path, conf=conf, imgsz=imgsz, device=device)
        matched_gt, matched_pred = match_boxes(gt_boxes, pred_boxes, iou)
        unmatched_gt = [idx for idx in range(len(gt_boxes)) if idx not in matched_gt]
        unmatched_pred = [idx for idx in range(len(pred_boxes)) if idx not in matched_pred]
        if not unmatched_gt and not unmatched_pred:
            continue

        case_id = f"yolo_{split}:{index}"
        visual_rel_path = draw_error_visual(
            image=image,
            image_path=image_path,
            output_root=output_root,
            case_id=case_id,
            class_names=class_names,
            gt_boxes=gt_boxes,
            pred_boxes=pred_boxes,
            unmatched_gt=unmatched_gt,
            unmatched_pred=unmatched_pred,
        )
        cases.append(
            ReviewCase(
                case_id=case_id,
                stage="yolo_eval",
                task=f"yolo_{split}",
                index=index,
                image_path=str(image_path.resolve()),
                image_rel_path=visual_rel_path,
                expected=format_boxes([gt_boxes[idx] for idx in unmatched_gt], class_names, prefix="GT"),
                predicted=format_boxes([pred_boxes[idx] for idx in unmatched_pred], class_names, prefix="PRED"),
                reason=infer_reason(bool(unmatched_gt), bool(unmatched_pred)),
                source_path=str(data_yaml.resolve()),
                tags=infer_tags(bool(unmatched_gt), bool(unmatched_pred)),
                metadata={
                    "model_path": str(model_path.resolve()),
                    "data_yaml": str(data_yaml.resolve()),
                    "split": split,
                    "conf": conf,
                    "iou": iou,
                    "imgsz": imgsz,
                    "device": device,
                    "gt_count": len(gt_boxes),
                    "pred_count": len(pred_boxes),
                    "unmatched_gt_count": len(unmatched_gt),
                    "unmatched_pred_count": len(unmatched_pred),
                    "unmatched_gt": boxes_payload([gt_boxes[idx] for idx in unmatched_gt], class_names),
                    "unmatched_pred": boxes_payload([pred_boxes[idx] for idx in unmatched_pred], class_names),
                    **source_info_from_yolo_image(image_path),
                },
            )
        )

    cases_by_task = {f"yolo_{split}": cases}
    write_cases(output_root / f"yolo_{split}", cases)
    write_cases(output_root / "all", cases)
    render_task_gallery(f"yolo_{split}", cases, output_root / f"yolo_{split}" / "index.html", gallery_root=output_root)
    render_task_gallery("all", cases, output_root / "all" / "index.html", gallery_root=output_root)
    render_root_index(
        output_root / "index.html",
        cases_by_task,
        title="YOLO Error Review",
        description="YOLO detection error reviewer；綠色是 missing GT，紅色是 extra prediction。",
    )

    summary = {
        "output_root": str(output_root),
        "index_path": str(output_root / "index.html"),
        "task_index_path": str(output_root / f"yolo_{split}" / "index.html"),
        "model_path": str(model_path.resolve()),
        "data_yaml": str(data_yaml.resolve()),
        "split": split,
        "total_images": len(image_paths),
        "error_cases": len(cases),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def load_dataset_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def normalize_class_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {idx: str(value) for idx, value in enumerate(names or [])}


def load_gt_boxes(label_path: Path, image_w: int, image_h: int) -> list[YoloBox]:
    boxes: list[YoloBox] = []
    if not label_path.exists():
        return boxes
    with label_path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(float(parts[0]))
            x_center, y_center, w, h = map(float, parts[1:])
            box_w = w * image_w
            box_h = h * image_h
            cx = x_center * image_w
            cy = y_center * image_h
            boxes.append(
                YoloBox(
                    cls_id=cls_id,
                    xyxy=(cx - box_w / 2.0, cy - box_h / 2.0, cx + box_w / 2.0, cy + box_h / 2.0),
                )
            )
    return boxes


def predict_boxes(model: Any, image_path: Path, *, conf: float, imgsz: int, device: str) -> list[YoloBox]:
    result = model.predict(
        source=str(image_path),
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )[0]
    boxes: list[YoloBox] = []
    if result.boxes is None:
        return boxes
    xyxy = result.boxes.xyxy.tolist()
    confs = result.boxes.conf.tolist()
    classes = result.boxes.cls.tolist()
    for box, score, cls_id in zip(xyxy, confs, classes):
        boxes.append(
            YoloBox(
                cls_id=int(cls_id),
                xyxy=tuple(float(value) for value in box),
                conf=float(score),
            )
        )
    return boxes


def iou_xyxy(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0.0 else inter / denom


def match_boxes(gt_boxes: list[YoloBox], pred_boxes: list[YoloBox], iou_threshold: float) -> tuple[list[int], list[int]]:
    candidates: list[tuple[float, int, int]] = []
    for gt_idx, gt_box in enumerate(gt_boxes):
        for pred_idx, pred_box in enumerate(pred_boxes):
            if gt_box.cls_id != pred_box.cls_id:
                continue
            score = iou_xyxy(gt_box.xyxy, pred_box.xyxy)
            if score >= iou_threshold:
                candidates.append((score, gt_idx, pred_idx))

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    for _, gt_idx, pred_idx in sorted(candidates, reverse=True):
        if gt_idx in matched_gt or pred_idx in matched_pred:
            continue
        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)
    return sorted(matched_gt), sorted(matched_pred)


def draw_error_visual(
    *,
    image: Any,
    image_path: Path,
    output_root: Path,
    case_id: str,
    class_names: dict[int, str],
    gt_boxes: list[YoloBox],
    pred_boxes: list[YoloBox],
    unmatched_gt: Iterable[int],
    unmatched_pred: Iterable[int],
) -> str:
    import cv2

    visual = image.copy()
    for idx in unmatched_gt:
        draw_box(visual, gt_boxes[idx], f"GT:{class_names.get(gt_boxes[idx].cls_id, gt_boxes[idx].cls_id)}", (30, 180, 70))
    for idx in unmatched_pred:
        pred = pred_boxes[idx]
        label = f"PRED:{class_names.get(pred.cls_id, pred.cls_id)}"
        if pred.conf is not None:
            label += f" {pred.conf:.2f}"
        draw_box(visual, pred, label, (40, 40, 230))

    dest_dir = output_root / "assets" / "images" / "yolo_errors"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{slugify(case_id)}__{image_path.name}"
    cv2.imwrite(str(dest), visual)
    return relative_path(dest, output_root)


def draw_box(image: Any, box: YoloBox, label: str, color: tuple[int, int, int]) -> None:
    import cv2

    x1, y1, x2, y2 = [int(round(value)) for value in box.xyxy]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        image,
        label,
        (x1, max(18, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def format_boxes(boxes: Iterable[YoloBox], class_names: dict[int, str], *, prefix: str) -> str:
    lines = []
    for box in boxes:
        label = class_names.get(box.cls_id, str(box.cls_id))
        score = "" if box.conf is None else f" conf={box.conf:.3f}"
        xyxy = ", ".join(f"{value:.1f}" for value in box.xyxy)
        lines.append(f"{prefix} {label}{score} [{xyxy}]")
    return "\n".join(lines)


def boxes_payload(boxes: Iterable[YoloBox], class_names: dict[int, str]) -> list[dict[str, Any]]:
    payload = []
    for box in boxes:
        payload.append(
            {
                "label": class_names.get(box.cls_id, str(box.cls_id)),
                "cls_id": box.cls_id,
                "conf": box.conf,
                "xyxy": [round(value, 3) for value in box.xyxy],
            }
        )
    return payload


def infer_reason(has_unmatched_gt: bool, has_unmatched_pred: bool) -> str:
    if has_unmatched_gt and has_unmatched_pred:
        return "missing_and_extra_detection"
    if has_unmatched_gt:
        return "missing_detection"
    return "extra_detection"


def infer_tags(has_unmatched_gt: bool, has_unmatched_pred: bool) -> list[str]:
    tags = ["yolo_detection"]
    if has_unmatched_gt:
        tags.append("missing_gt")
    if has_unmatched_pred:
        tags.append("extra_prediction")
    return tags


def source_info_from_yolo_image(image_path: Path) -> dict[str, str]:
    resolved_image = image_path.resolve()
    info = direct_annotation_info(resolved_image)
    if info:
        return info
    if "__" not in image_path.name:
        return {"source_file_name": image_path.name, "source_image_stem": image_path.stem}

    part_number, source_name = image_path.name.split("__", 1)
    annotation_path = resolved_image.with_suffix(".json")
    return {
        "part_number": part_number,
        "annotation_path": str(annotation_path) if annotation_path.exists() else "",
        "annotation_file_name": annotation_path.name if annotation_path.exists() else "",
        "source_file_name": source_name,
        "source_image_stem": Path(source_name).stem,
    }


def copy_source_image(image_path: Path, output_root: Path, case_id: str) -> str:
    dest_dir = output_root / "assets" / "source_images" / "yolo_errors"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{slugify(case_id)}__{image_path.name}"
    shutil.copy2(image_path, dest)
    return relative_path(dest, output_root)
