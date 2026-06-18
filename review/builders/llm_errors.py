from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from ..schema import ReviewCase, relative_path, slugify


DEFAULT_TASKS = (
    "task1_view_classification",
    "task2_num_group_recognition",
    "task3_dim_type",
    "task4_dim_target",
    "task5_dim_start_anchor",
)


def build_llm_error_review(
    eval_dir: Path,
    output_root: Path,
    *,
    tasks: Iterable[str] = DEFAULT_TASKS,
    copy_images: bool = True,
) -> dict[str, Any]:
    eval_dir = eval_dir.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    cases_by_task: dict[str, list[ReviewCase]] = {}
    for task in tasks:
        prediction_path = eval_dir / task / "predictions.jsonl"
        if not prediction_path.exists():
            cases_by_task[task] = []
            continue
        task_cases = list(
            iter_llm_error_cases(
                prediction_path,
                output_root=output_root,
                copy_images=copy_images,
            )
        )
        cases_by_task[task] = task_cases
        write_cases(output_root / task, task_cases)

    all_cases = [case for task_cases in cases_by_task.values() for case in task_cases]
    write_cases(output_root / "all", all_cases)
    write_root_index(output_root, cases_by_task)
    return {
        "output_root": str(output_root),
        "index_path": str(output_root / "index.html"),
        "total_cases": len(all_cases),
        "tasks": {task: len(cases) for task, cases in cases_by_task.items()},
    }


def iter_llm_error_cases(
    prediction_path: Path,
    *,
    output_root: Path,
    copy_images: bool = True,
) -> Iterable[ReviewCase]:
    task = prediction_path.parent.name
    with prediction_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            if bool(row.get("clean_exact_match")):
                continue
            yield build_case_from_prediction_row(
                row,
                prediction_path=prediction_path,
                output_root=output_root,
                copy_images=copy_images,
            )


def build_case_from_prediction_row(
    row: dict[str, Any],
    *,
    prediction_path: Path,
    output_root: Path,
    copy_images: bool = True,
) -> ReviewCase:
    task = str(row.get("dataset") or prediction_path.parent.name)
    index = int(row.get("index") or 0)
    case_id = f"{task}:{index}"
    image_path = Path(str(row.get("image_path") or ""))
    image_rel_path = ""
    if image_path.exists() and copy_images:
        image_rel_path = copy_case_image(image_path, output_root, task, case_id)
    elif image_path.exists():
        image_rel_path = image_path.resolve().as_posix()

    expected = expected_text(row)
    predicted = predicted_text(row)
    source_info = resolve_source_info(image_path)
    return ReviewCase(
        case_id=case_id,
        stage="llm_eval",
        task=task,
        index=index,
        image_path=str(image_path),
        image_rel_path=image_rel_path,
        expected=expected,
        predicted=predicted,
        reason=infer_reason(task, row),
        source_path=str(prediction_path),
        prompt=str(row.get("prompt") or ""),
        tags=infer_tags(task, row),
        metadata={
            "gold_answer": row.get("gold_answer"),
            "prediction": row.get("prediction"),
            "gold_refs": row.get("gold_refs"),
            "pred_refs": row.get("pred_refs"),
            "gold_text": row.get("gold_text"),
            "pred_text": row.get("pred_text"),
            "ref_exact_match": row.get("ref_exact_match"),
            "text_exact_match": row.get("text_exact_match"),
            **source_info,
        },
    )


def resolve_source_info(image_path: Path) -> dict[str, str]:
    direct = direct_annotation_info(image_path)
    if direct:
        return direct

    source_stem = overlay_source_stem(image_path)
    if not source_stem:
        return {}

    annotation_path = find_overlay_annotation(image_path, source_stem)
    if annotation_path is None:
        return {"source_image_stem": source_stem}

    return {
        "part_number": annotation_path.parent.parent.name,
        "annotation_path": str(annotation_path.resolve()),
        "annotation_file_name": annotation_path.name,
        "source_file_name": annotation_path.with_suffix(image_path.suffix).name,
        "source_image_stem": source_stem,
    }


def direct_annotation_info(image_path: Path) -> dict[str, str]:
    parts = image_path.parts
    part_number = ""
    for idx, part in enumerate(parts):
        if part.startswith("dataset_full_") and idx + 1 < len(parts):
            part_number = parts[idx + 1]
            break
    if not part_number:
        return {}

    annotation_path = image_path.with_suffix(".json")
    return {
        "part_number": part_number,
        "annotation_path": str(annotation_path.resolve()) if annotation_path.exists() else "",
        "annotation_file_name": annotation_path.name if annotation_path.exists() else "",
        "source_file_name": image_path.name,
        "source_image_stem": image_path.stem,
    }


def overlay_source_stem(image_path: Path) -> str:
    stem = image_path.stem
    if "__dim" not in stem:
        return ""
    return stem.split("__dim", 1)[0]


@lru_cache(maxsize=2048)
def find_overlay_annotation(image_path: Path, source_stem: str) -> Path | None:
    for dataset_root in candidate_dataset_full_roots(image_path):
        matches = sorted(dataset_root.glob(f"*/extract_image/{source_stem}.json"))
        if matches:
            return matches[0]
    return None


def candidate_dataset_full_roots(image_path: Path) -> list[Path]:
    roots: list[Path] = []
    parts = image_path.parts
    if "dataset_json" in parts:
        idx = parts.index("dataset_json")
        datasets_root = Path(*parts[:idx])
        version = parts[idx + 1] if idx + 1 < len(parts) else ""
        if version:
            roots.append(datasets_root / f"dataset_full_{version}")
        roots.append(datasets_root / "dataset_full_v4")

    for parent in image_path.parents:
        if parent.name == "datasets":
            roots.append(parent / "dataset_full_v4")
            roots.append(parent / "dataset_full_v3")

    unique_roots: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen and root.exists():
            unique_roots.append(root)
            seen.add(key)
    return unique_roots


def copy_case_image(image_path: Path, output_root: Path, task: str, case_id: str) -> str:
    suffix = image_path.suffix or ".png"
    dest_dir = output_root / "assets" / "images" / slugify(task)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{slugify(case_id)}{suffix}"
    shutil.copy2(image_path, dest)
    return relative_path(dest, output_root)


def expected_text(row: dict[str, Any]) -> str:
    if row.get("gold_refs") is not None:
        return "\n".join(str(item) for item in row.get("gold_refs") or [])
    if row.get("gold_text") is not None:
        return str(row.get("gold_text") or "")
    return str(row.get("gold_answer_clean") or row.get("gold_answer") or "")


def predicted_text(row: dict[str, Any]) -> str:
    if row.get("pred_refs") is not None:
        return "\n".join(str(item) for item in row.get("pred_refs") or [])
    if row.get("pred_text") is not None:
        return str(row.get("pred_text") or "")
    return str(row.get("prediction_clean") or row.get("prediction") or "")


def infer_reason(task: str, row: dict[str, Any]) -> str:
    expected = expected_text(row).splitlines()
    predicted = predicted_text(row).splitlines()
    if task == "task1_view_classification":
        return "wrong_view"
    if task == "task2_num_group_recognition":
        return "wrong_text"
    if task == "task3_dim_type":
        return "wrong_dimension_type"
    if task == "task4_dim_target":
        return "wrong_target_count" if len(expected) != len(predicted) else "wrong_target"
    if task == "task5_dim_start_anchor":
        if len(expected) != len(predicted):
            return "wrong_anchor_count"
        labels = {"vertical_line", "horizontal_line", "other"}
        return "wrong_geometry" if set(expected) & labels or set(predicted) & labels else "wrong_anchor"
    return "wrong_prediction"


def infer_tags(task: str, row: dict[str, Any]) -> list[str]:
    tags = [task.replace("task", "task_")]
    reason = infer_reason(task, row)
    tags.append(reason)
    if row.get("ref_exact_match") is False:
        tags.append("ref_mismatch")
    if row.get("text_exact_match") is False:
        tags.append("text_mismatch")
    return tags


def write_cases(output_dir: Path, cases: list[ReviewCase]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "cases.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as file:
        for case in cases:
            file.write(json.dumps(case.to_dict(), ensure_ascii=False))
            file.write("\n")
    write_cases_csv(output_dir / "cases.csv", cases)


def write_cases_csv(path: Path, cases: list[ReviewCase]) -> None:
    fieldnames = [
        "case_id",
        "task",
        "index",
        "reason",
        "expected",
        "predicted",
        "part_number",
        "image_path",
        "annotation_path",
        "source_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            payload = case.to_dict()
            payload["part_number"] = case.metadata.get("part_number", "")
            payload["annotation_path"] = case.metadata.get("annotation_path", "")
            writer.writerow({key: payload.get(key, "") for key in fieldnames})


def write_root_index(output_root: Path, cases_by_task: dict[str, list[ReviewCase]]) -> None:
    from ..render import render_root_index, render_task_gallery

    for task, cases in cases_by_task.items():
        render_task_gallery(task, cases, output_root / task / "index.html", gallery_root=output_root)
    all_cases = [case for cases in cases_by_task.values() for case in cases]
    render_task_gallery("all", all_cases, output_root / "all" / "index.html", gallery_root=output_root)
    render_root_index(output_root / "index.html", cases_by_task)


def group_by_task(cases: Iterable[ReviewCase]) -> dict[str, list[ReviewCase]]:
    grouped: dict[str, list[ReviewCase]] = defaultdict(list)
    for case in cases:
        grouped[case.task].append(case)
    return dict(grouped)
