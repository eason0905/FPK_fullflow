from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..common.gallery_renderer import page_relative_url, render_card, render_review_index
from ..common.gallery_renderer import page as render_page_shell
from ..common.schemas import ReviewItem, ReviewMedia
from ..schema import slugify
from .package_graph import find_fullflow_root, infer_run_id, restore_notes_from_history, root_relative_url
from .package_graph import static_url_prefix
from .gt_alignment import (
    display_scale_context_from_canonical,
    main_view_overlay_evidence_from_canonical,
    multiview_overlay_evidence_from_canonical,
)


UNIFIED_MULTIVIEW_LAYERS_FILENAME = "unified_multiview_layers.json"
UNIFIED_MULTIVIEW_LAYERS_SVG_FILENAME = "unified_multiview_layers.svg"


def build_multiview_review(
    *,
    multiview_root: Path,
    output_root: Path,
    fullflow_root: Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    multiview_root = multiview_root.resolve()
    output_root = output_root.resolve()
    fullflow_root = (fullflow_root or find_fullflow_root(output_root)).resolve()
    run_id = run_id or infer_run_id(output_root)

    items = [item_from_part(part_dir, fullflow_root=fullflow_root) for part_dir in sorted((multiview_root / "parts").iterdir()) if part_dir.is_dir()]
    data_dir = output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    cases_path = data_dir / "cases.json"
    notes_path = data_dir / "notes.json"
    history_path = data_dir / "notes_history.jsonl"
    summary_path = data_dir / "summary.json"
    cases_path.write_text(json.dumps([item.to_dict() for item in items], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not notes_path.exists():
        notes_path.write_text(
            json.dumps({"gallery_id": "multiview", "run_id": run_id, "updated_at": None, "items": {}}, indent=2)
            + "\n",
            encoding="utf-8",
        )
    history_path.touch(exist_ok=True)
    restore_notes_from_history(notes_path, history_path)

    notes_rel_path = root_relative_url(notes_path, fullflow_root)
    history_rel_path = root_relative_url(history_path, fullflow_root)
    static_prefix = static_url_prefix(fullflow_root)
    risk_pages = write_pages(
        output_root=output_root,
        subdir="by_risk",
        grouped=group_by_risk(items),
        notes_rel_path=notes_rel_path,
        history_rel_path=history_rel_path,
        run_id=run_id,
        title_prefix="Multiview Review Risk",
        static_prefix=static_prefix,
    )
    view_pages = write_pages(
        output_root=output_root,
        subdir="by_view",
        grouped=group_by_view(items),
        notes_rel_path=notes_rel_path,
        history_rel_path=history_rel_path,
        run_id=run_id,
        title_prefix="Multiview Review View",
        static_prefix=static_prefix,
    )
    render_review_index(
        output_root=output_root,
        title="Multiview 2D Graph Review",
        description="Review unified per-part multiview layers, conflicts, missing views, and ignored lateral dimensions.",
        items=items,
        page_paths=risk_pages + view_pages,
        notes_rel_path=notes_rel_path,
        run_id=run_id,
        gallery_id="multiview",
        page_labels=[path.stem for path in risk_pages + view_pages],
        static_prefix=static_prefix,
    )

    summary = {
        "output_root": str(output_root),
        "index_path": str(output_root / "index.html"),
        "cases_path": str(cases_path),
        "notes_path": str(notes_path),
        "notes_history_path": str(history_path),
        "summary_path": str(summary_path),
        "multiview_root": str(multiview_root),
        "run_id": run_id,
        "total_items": len(items),
        "risk_pages": [str(path) for path in risk_pages],
        "view_pages": [str(path) for path in view_pages],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def item_from_part(part_dir: Path, *, fullflow_root: Path) -> ReviewItem:
    canonical_path = multiview_layers_path(part_dir)
    evidence_path = part_dir / "evidence.json"
    conflicts_path = part_dir / "conflicts.json"
    svg_path = multiview_layers_svg_path(part_dir)
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    conflicts = json.loads(conflicts_path.read_text(encoding="utf-8"))
    part_number = str(canonical.get("part_number") or part_dir.name)
    summary = summarize_case(canonical, conflicts)
    media = [ReviewMedia(label="Unified multiview layers", path=str(svg_path), url=root_relative_url(svg_path, fullflow_root))]
    display_context = display_scale_context_from_canonical(canonical_path)
    for overlay in (
        main_view_overlay_evidence_from_canonical(canonical_path, fullflow_root, display_context=display_context),
        multiview_overlay_evidence_from_canonical(canonical_path, fullflow_root, display_context=display_context),
    ):
        if overlay is not None:
            media.append(ReviewMedia(label=overlay["label"], path=overlay["path"], url=overlay["url"]))
    for ref in evidence.get("evidence_refs") or []:
        image_path = Path(str(ref.get("image_path") or ""))
        if image_path.exists():
            label = f"{ref.get('raw_view') or 'view'} source"
            media.append(ReviewMedia(label=label, path=str(image_path), url=root_relative_url(image_path, fullflow_root)))
    case_id = f"multiview:{slugify(part_number)}"
    return ReviewItem(
        case_id=case_id,
        title=part_number,
        rank=0,
        part_number=part_number,
        file_name=canonical_path.name,
        view=",".join(canonical.get("canonical_source_views") or []),
        risk_score=summary["risk_score"],
        risk_level=summary["risk_level"],
        risk_reasons=summary["risk_reasons"],
        media=media,
        links={
            "unified_multiview_layers": root_relative_url(canonical_path, fullflow_root),
            "evidence": root_relative_url(evidence_path, fullflow_root),
            "conflicts": root_relative_url(conflicts_path, fullflow_root),
        },
        metrics={
            "source_views": canonical.get("source_views"),
            "canonical_source_views": canonical.get("canonical_source_views"),
            "package_pad_count": len(canonical.get("package_pads") or []),
            "land_pad_count": len(canonical.get("land_pads") or []),
            "lead_contact_count": len(canonical.get("lead_contacts") or []),
            "lead_pad_count": len(canonical.get("lead_pads") or []),
            "inner_land_pad_count": len(canonical.get("inner_land_pads") or []),
            "dimension_count": len(canonical.get("dimensions") or []),
            "ignored_evidence_count": len(canonical.get("ignored_evidence") or []),
            "missing_canonical_views": canonical.get("missing_canonical_views"),
            "pad_matching": canonical.get("pad_matching"),
            "source_selection": canonical.get("source_selection"),
            "evidence_summary": evidence.get("summary") or canonical.get("evidence_summary"),
        },
        metadata={
            "part_dir": str(part_dir),
            "source_selection": canonical.get("source_selection"),
            "evidence_summary": evidence.get("summary") or canonical.get("evidence_summary"),
            "conflicts": conflicts,
            "ignored_evidence": canonical.get("ignored_evidence"),
            "evidence_refs": evidence.get("evidence_refs"),
        },
    )


def multiview_layers_path(part_dir: Path) -> Path:
    return part_dir / UNIFIED_MULTIVIEW_LAYERS_FILENAME


def multiview_layers_svg_path(part_dir: Path) -> Path:
    return part_dir / UNIFIED_MULTIVIEW_LAYERS_SVG_FILENAME


def summarize_case(canonical: dict[str, Any], conflicts: list[dict[str, Any]]) -> dict[str, Any]:
    score = float((canonical.get("summary") or {}).get("risk_score") or 0.0)
    level = str((canonical.get("summary") or {}).get("risk_level") or risk_level(score))
    reasons = list((canonical.get("summary") or {}).get("risk_reasons") or [])
    if not reasons:
        reasons = ["no obvious multiview risk signals"]
    if conflicts and not any("conflict" in reason for reason in reasons):
        reasons.append(f"{len(conflicts)} conflicts")
    return {"risk_score": score, "risk_level": level, "risk_reasons": reasons}


def risk_level(score: float) -> str:
    if score >= 30.0:
        return "high"
    if score >= 10.0:
        return "medium"
    return "low"


def group_by_risk(items: list[ReviewItem]) -> dict[str, list[ReviewItem]]:
    groups = {"high": [], "medium": [], "low": []}
    for item in items:
        groups.setdefault(item.risk_level, []).append(item)
    return {label: groups[label] for label in ("high", "medium", "low") if groups.get(label)}


def group_by_view(items: list[ReviewItem]) -> dict[str, list[ReviewItem]]:
    groups: dict[str, list[ReviewItem]] = defaultdict(list)
    for item in items:
        views = [view for view in item.view.split(",") if view] or ["unknown"]
        for view in views:
            groups[view].append(item)
    preferred = ["bottom", "land", "lateral", "lead_detail", "top", "unknown"]
    ordered = {label: groups[label] for label in preferred if groups.get(label)}
    for label in sorted(groups):
        if label not in ordered:
            ordered[label] = groups[label]
    return ordered


def write_pages(
    *,
    output_root: Path,
    subdir: str,
    grouped: dict[str, list[ReviewItem]],
    notes_rel_path: str,
    history_rel_path: str,
    run_id: str,
    title_prefix: str,
    static_prefix: str,
) -> list[Path]:
    page_paths = []
    for label, items in grouped.items():
        page_path = output_root / subdir / f"{slugify(label)}.html"
        page_path.parent.mkdir(parents=True, exist_ok=True)
        page_static_prefix = page_relative_url(static_prefix, page_path.parent)
        cards = "\n".join(render_card(item, page_dir=page_path.parent) for item in enumerate_items(items))
        config = {
            "notesPath": notes_rel_path,
            "historyPath": history_rel_path,
            "runId": run_id,
            "galleryId": "multiview",
            "pageIndex": label,
        }
        body = f"""
  <header>
    <div class="topbar">
      <div><h1>{title_prefix}: {label}</h1></div>
      <div class="actions"><a class="button" href="../index.html">Index</a></div>
    </div>
    <div class="chips">
      <span class="chip">Items: {len(items)}</span>
      <span class="chip">Reviewed: <span data-reviewed-count>0</span></span>
      <span class="chip" id="review-server-state">notes server ready</span>
    </div>
  </header>
  <main><section class="cases">{cards or '<p class="muted">No items.</p>'}</section></main>
  <script>window.REVIEW_CONFIG = {json.dumps(config, ensure_ascii=False)};</script>
  <script src="{page_static_prefix}/review.js"></script>
"""
        page_path.write_text(
            render_page_shell(f"{title_prefix}: {label}", body, static_prefix=page_static_prefix),
            encoding="utf-8",
        )
        page_paths.append(page_path)
    return page_paths


def enumerate_items(items: list[ReviewItem]) -> list[ReviewItem]:
    enumerated = []
    for index, item in enumerate(items, start=1):
        enumerated.append(
            ReviewItem(
                case_id=item.case_id,
                title=item.title,
                rank=index,
                part_number=item.part_number,
                file_name=item.file_name,
                view=item.view,
                risk_score=item.risk_score,
                risk_level=item.risk_level,
                risk_reasons=item.risk_reasons,
                media=item.media,
                links=item.links,
                metrics=item.metrics,
                metadata=item.metadata,
            )
        )
    return enumerated
