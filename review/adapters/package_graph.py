from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..common.gallery_renderer import render_review_index, render_review_page
from ..common.schemas import ReviewItem, ReviewMedia
from ..schema import slugify


def build_package_graph_review(
    *,
    risk_report_path: Path,
    output_root: Path,
    fullflow_root: Path | None = None,
    run_id: str | None = None,
    page_count: int = 5,
    split_by: str = "view",
) -> dict[str, Any]:
    risk_report_path = risk_report_path.resolve()
    output_root = output_root.resolve()
    fullflow_root = (fullflow_root or find_fullflow_root(output_root)).resolve()
    run_id = run_id or infer_run_id(output_root)

    rows = read_jsonl(risk_report_path)
    items = [item_from_row(row, fullflow_root=fullflow_root) for row in rows]
    page_groups = split_items_for_review(items, split_by=split_by, page_count=page_count)

    data_dir = output_root / "data"
    pages_dir = output_root / "pages"
    data_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    cases_path = data_dir / "cases.json"
    notes_path = data_dir / "notes.json"
    history_path = data_dir / "notes_history.jsonl"
    summary_path = data_dir / "summary.json"
    cases_path.write_text(
        json.dumps([item.to_dict() for item in items], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not notes_path.exists():
        notes_path.write_text(
            json.dumps(
                {
                    "gallery_id": "package_graph",
                    "run_id": run_id,
                    "updated_at": None,
                    "items": {},
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    history_path.touch(exist_ok=True)
    restore_notes_from_history(notes_path, history_path)

    page_paths = [pages_dir / f"{slugify(label)}.html" for label, _ in page_groups]
    page_labels = [label for label, _ in page_groups]
    notes_rel_path = root_relative_url(notes_path, fullflow_root)
    history_rel_path = root_relative_url(history_path, fullflow_root)
    static_prefix = static_url_prefix(fullflow_root)
    for index, (page_path, (_, page_items)) in enumerate(zip(page_paths, page_groups), start=1):
        render_review_page(
            output_path=page_path,
            output_root=output_root,
            title=f"Package Graph Review: {page_labels[index - 1]}",
            description="Write issue descriptions per graph. Notes are persisted by the fullflow review server.",
            items=page_items,
            all_page_paths=page_paths,
            current_page=index,
            notes_rel_path=notes_rel_path,
            history_rel_path=history_rel_path,
            run_id=run_id,
            gallery_id="package_graph",
            page_labels=page_labels,
            static_prefix=static_prefix,
        )

    render_review_index(
        output_root=output_root,
        title="Package Graph Review",
        description="Package graph reviewer grouped by view with persisted issue descriptions.",
        items=items,
        page_paths=page_paths,
        notes_rel_path=notes_rel_path,
        run_id=run_id,
        gallery_id="package_graph",
        page_labels=page_labels,
        static_prefix=static_prefix,
    )

    summary = {
        "output_root": str(output_root),
        "index_path": str(output_root / "index.html"),
        "cases_path": str(cases_path),
        "notes_path": str(notes_path),
        "notes_history_path": str(history_path),
        "summary_path": str(summary_path),
        "risk_report_path": str(risk_report_path),
        "run_id": run_id,
        "total_items": len(items),
        "split_by": split_by,
        "page_count": len(page_groups),
        "page_sizes": {label: len(page_items) for label, page_items in page_groups},
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def restore_notes_from_history(notes_path: Path, history_path: Path) -> None:
    if not history_path.exists():
        return
    notes = json.loads(notes_path.read_text(encoding="utf-8")) if notes_path.exists() else {}
    items = dict(notes.get("items") or {})
    updated_at = notes.get("updated_at")

    for line in history_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        note = row.get("note") or {}
        case_id = str(row.get("case_id") or note.get("case_id") or "")
        if not case_id:
            continue
        row_updated = str(row.get("updated_at") or note.get("updated_at") or "")
        existing = items.get(case_id) or {}
        existing_updated = str(existing.get("updated_at") or "")
        if existing_updated and row_updated and row_updated < existing_updated:
            continue

        has_content = bool(
            str(note.get("issue_text") or "").strip()
            or str(note.get("category") or "").strip()
            or str(note.get("status") or "").strip()
        )
        if has_content:
            items[case_id] = note
        else:
            items.pop(case_id, None)
        updated_at = row_updated or updated_at

    notes["items"] = items
    notes["updated_at"] = updated_at
    notes_path.write_text(json.dumps(notes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def item_from_row(row: dict[str, Any], *, fullflow_root: Path) -> ReviewItem:
    rank = int(row.get("rank") or 0)
    part_number = str(row.get("part_number") or "")
    graph_path = Path(str(row.get("graph_path") or ""))
    annotation_path = Path(str(row.get("annotation_path") or ""))
    file_name = annotation_path.name or graph_path.name
    title = f"{part_number} / {file_name}"
    case_key = graph_path.relative_to(fullflow_root).as_posix() if graph_path.exists() else title
    case_id = f"package_graph:{slugify(case_key)}"
    risk_score = float(row.get("risk_score") or 0.0)
    media = [
        ReviewMedia(
            label="Graph rendering",
            path=str(row.get("overlay_path") or ""),
            url=root_relative_url(Path(str(row.get("overlay_path") or "")), fullflow_root),
        ),
    ]
    links = {
        "graph_json": root_relative_url(graph_path, fullflow_root),
        "annotation_json": root_relative_url(annotation_path, fullflow_root),
        "overlay_png": media[0].url,
    }
    return ReviewItem(
        case_id=case_id,
        title=title,
        rank=rank,
        part_number=part_number,
        file_name=file_name,
        view=str(row.get("view") or ""),
        risk_score=risk_score,
        risk_level=risk_level(risk_score),
        risk_reasons=[str(reason) for reason in row.get("risk_reasons") or []],
        media=media,
        links=links,
        metrics=dict(row.get("metrics") or {}),
        metadata={
            "graph_path": str(row.get("graph_path") or ""),
            "annotation_path": str(row.get("annotation_path") or ""),
            "image_path": str(row.get("image_path") or ""),
            "overlay_path": str(row.get("overlay_path") or ""),
            "object_count": row.get("object_count"),
            "dimension_count": row.get("dimension_count"),
            "latent_count": row.get("latent_count"),
            "constraint_status": row.get("constraint_status"),
            "dimension_status": row.get("dimension_status"),
            "annotation_stats": row.get("annotation_stats"),
        },
    )


def fullflow_url_prefix(fullflow_root: Path, workspace_root: Path | None = None) -> str:
    """Return the browser path prefix from the HTTP server root to FPK_PJ_fullflow.

    Fullflow review galleries are served with `FPK_PJ_fullflow` as the HTTP
    root.  Artifact URLs should therefore be rooted at `/runs`, `/assets`, and
    `/review`, not at the repository workspace path.
    """
    _ = fullflow_root, workspace_root
    return ""


def static_url_prefix(fullflow_root: Path) -> str:
    prefix = fullflow_url_prefix(fullflow_root)
    return "/" + "/".join(part for part in (prefix, "review/common/static") if part)


def root_relative_url(path: Path, fullflow_root: Path, workspace_root: Path | None = None) -> str:
    if not str(path):
        return ""
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(fullflow_root.resolve()).as_posix()
    except ValueError:
        workspace = (workspace_root or Path.cwd()).resolve()
        try:
            return resolved.relative_to(workspace).as_posix()
        except ValueError:
            return resolved.as_posix().lstrip("/")
    prefix = fullflow_url_prefix(fullflow_root, workspace_root=workspace_root)
    return "/".join(part for part in (prefix, rel) if part)


def risk_level(score: float) -> str:
    if score >= 30.0:
        return "high"
    if score >= 10.0:
        return "medium"
    return "low"


def split_items_for_review(
    items: list[ReviewItem],
    *,
    split_by: str = "view",
    page_count: int = 5,
) -> list[tuple[str, list[ReviewItem]]]:
    if split_by == "view":
        return split_items_by_view(items)
    if split_by == "count":
        return [(f"page_{index:03d}", page) for index, page in enumerate(split_items_by_count(items, page_count), start=1)]
    raise ValueError(f"Unsupported split_by: {split_by}")


def split_items_by_view(items: list[ReviewItem]) -> list[tuple[str, list[ReviewItem]]]:
    grouped: dict[str, list[ReviewItem]] = defaultdict(list)
    for item in items:
        grouped[item.view or "unknown"].append(item)

    preferred = ["bottom", "land", "top", "side", "front", "lead", "land_detail", "lateral", "lead_detail"]
    labels = [label for label in preferred if label in grouped]
    labels.extend(sorted(label for label in grouped if label not in set(preferred)))
    return [(label, grouped[label]) for label in labels]


def split_items_by_count(items: list[ReviewItem], page_count: int) -> list[list[ReviewItem]]:
    if page_count <= 1:
        return [items]
    page_size = max(1, math.ceil(len(items) / page_count))
    pages = [items[index : index + page_size] for index in range(0, len(items), page_size)]
    while len(pages) < page_count:
        pages.append([])
    return pages[:page_count]


def find_fullflow_root(path: Path) -> Path:
    resolved = path.resolve()
    for parent in [resolved, *resolved.parents]:
        if parent.name == "FPK_PJ_fullflow":
            return parent
    raise ValueError(f"Cannot infer FPK_PJ_fullflow root from {path}")


def infer_run_id(path: Path) -> str:
    parts = path.resolve().parts
    if "runs" in parts:
        idx = parts.index("runs")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return ""
