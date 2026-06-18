from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .schemas import ReviewItem


CATEGORIES = ("algorithm_error", "annotation_error", "table_issue", "model_error", "ambiguous", "ignore")
STATUSES = ("open", "fixed", "skip", "need_check")


def render_review_index(
    *,
    output_root: Path,
    title: str,
    description: str,
    items: list[ReviewItem],
    page_paths: list[Path],
    notes_rel_path: str,
    run_id: str = "",
    gallery_id: str = "review",
    page_labels: list[str] | None = None,
    static_prefix: str = "/review/common/static",
) -> None:
    page_dir = output_root
    page_static_prefix = page_relative_url(static_prefix, page_dir)
    notes_href = page_relative_url(notes_rel_path, page_dir)
    rows = []
    for index, page_path in enumerate(page_paths, start=1):
        rel = page_path.relative_to(output_root).as_posix()
        label = page_labels[index - 1] if page_labels and index - 1 < len(page_labels) else f"Page {index}"
        rows.append(f"<a class=\"page-link\" href=\"{html.escape(rel)}\">{html.escape(label)}</a>")

    view_counts = Counter(item.view for item in items)
    risk_counts = Counter(item.risk_level for item in items)
    config = {
        "notesPath": notes_rel_path,
        "historyPath": "",
        "runId": run_id,
        "galleryId": gallery_id,
        "pageIndex": None,
    }
    body = f"""
  <header>
    <div class="topbar">
      <div>
        <h1>{html.escape(title)}</h1>
        <p class="muted">{html.escape(description)}</p>
      </div>
      <div class="actions">
        <a class="button" href="{html.escape(notes_href)}" target="_blank">Open notes.json</a>
      </div>
    </div>
    <div class="chips">
      <span class="chip">Total: {len(items)}</span>
      <span class="chip">Reviewed: <span data-reviewed-count>0</span></span>
      <span class="chip">High: {risk_counts.get("high", 0)}</span>
      <span class="chip">Medium: {risk_counts.get("medium", 0)}</span>
      <span class="chip">Low: {risk_counts.get("low", 0)}</span>
    </div>
  </header>
  <main>
    <section class="overview">
      <h2>Pages</h2>
      <p class="muted">Open one page at a time. Notes are saved into <code>{html.escape(notes_rel_path)}</code>.</p>
      <div class="pager">{''.join(rows)}</div>
      <table>
        <thead><tr><th>View</th><th>Count</th></tr></thead>
        <tbody>{''.join(f"<tr><td>{html.escape(view)}</td><td>{count}</td></tr>" for view, count in sorted(view_counts.items()))}</tbody>
      </table>
    </section>
  </main>
  <script>window.REVIEW_CONFIG = {json.dumps(config, ensure_ascii=False)};</script>
"""
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "index.html").write_text(
        page(title, body, include_script=True, static_prefix=page_static_prefix),
        encoding="utf-8",
    )


def render_review_page(
    *,
    output_path: Path,
    output_root: Path,
    title: str,
    description: str,
    items: list[ReviewItem],
    all_page_paths: list[Path],
    current_page: int,
    notes_rel_path: str,
    history_rel_path: str,
    run_id: str,
    gallery_id: str,
    page_labels: list[str] | None = None,
    categories: tuple[str, ...] = CATEGORIES,
    statuses: tuple[str, ...] = STATUSES,
    static_prefix: str = "/review/common/static",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page_dir = output_path.parent
    page_static_prefix = page_relative_url(static_prefix, page_dir)
    cards = "\n".join(
        render_card(item, categories=categories, statuses=statuses, page_dir=page_dir)
        for item in items
    )
    pager = render_pager(output_root, all_page_paths, current_page, page_labels=page_labels)
    config = {
        "notesPath": notes_rel_path,
        "historyPath": history_rel_path,
        "runId": run_id,
        "galleryId": gallery_id,
        "pageIndex": current_page,
    }
    body = f"""
  <header>
    <div class="topbar">
      <div>
        <h1>{html.escape(title)}</h1>
      </div>
      <div class="actions">
        <a class="button" href="../index.html">Index</a>
        <button type="button" data-action="download-notes">Download notes</button>
      </div>
    </div>
    <div class="compact-nav">
      <div class="chips">
        <span class="chip">Page: {html.escape(page_labels[current_page - 1] if page_labels and current_page - 1 < len(page_labels) else str(current_page))}</span>
        <span class="chip">Items: {len(items)}</span>
        <span class="chip">Reviewed: <span data-reviewed-count>0</span></span>
        <span class="chip" id="review-server-state">notes server ready</span>
      </div>
      <div class="pager">{pager}</div>
    </div>
    <details class="filter-panel">
      <summary>Filters</summary>
      <div class="filters">
        <label>Search <input id="filter-text" data-filter type="search" placeholder="part / file / note"></label>
        <label>Status <select id="filter-status" data-filter><option value="">all</option>{options(statuses)}</select></label>
        <label>Category <select id="filter-category" data-filter><option value="">all</option>{options(categories)}</select></label>
      </div>
    </details>
  </header>
  <main>
    <section class="cases">
      {cards or '<p class="muted">No items on this page.</p>'}
    </section>
  </main>
  <script>window.REVIEW_CONFIG = {json.dumps(config, ensure_ascii=False)};</script>
  <script src="{html.escape(page_static_prefix)}/review.js"></script>
"""
    output_path.write_text(page(title, body, static_prefix=page_static_prefix), encoding="utf-8")


def render_card(
    item: ReviewItem,
    *,
    categories: tuple[str, ...] = CATEGORIES,
    statuses: tuple[str, ...] = STATUSES,
    page_dir: Path | None = None,
) -> str:
    media = render_media_grid(item.media, page_dir=page_dir)
    reasons = "".join(f"<li>{html.escape(reason)}</li>" for reason in item.risk_reasons)
    diagnostic_details = render_diagnostic_details((item.metrics or {}).get("score_diagnostic_details") or [])
    metrics = html.escape(json.dumps(item.metrics, ensure_ascii=False, indent=2))
    metadata = html.escape(json.dumps(item.metadata, ensure_ascii=False, indent=2))
    links = " | ".join(
        f"<a href=\"{html.escape(page_relative_url(path, page_dir))}\" target=\"_blank\">{html.escape(label)}</a>"
        for label, path in item.links.items()
        if path
    )
    return f"""
<article class="review-card {html.escape(item.risk_level)}"
  data-case-id="{html.escape(item.case_id)}"
  data-title="{html.escape(item.title)}"
  data-part-number="{html.escape(item.part_number)}"
  data-file-name="{html.escape(item.file_name)}"
  data-view="{html.escape(item.view)}">
  <div class="card-head">
    <div>
      <h2>#{item.rank} {html.escape(item.title)}</h2>
      <p class="muted">{html.escape(item.part_number)} / {html.escape(item.file_name)} / {html.escape(item.view)}</p>
    </div>
    <span class="risk {html.escape(item.risk_level)}">{html.escape(item.risk_level)} {item.risk_score:.3f}</span>
  </div>
  <div class="card-body">
    <div class="media-grid">{media}</div>
    <div class="details">
      <div class="meta-grid">
        <div><h3>Risk reasons</h3><ul class="reason-list">{reasons}</ul></div>
        <div><h3>Links</h3><p>{links or '-'}</p></div>
      </div>
      {diagnostic_details}
      <details><summary>Metrics</summary><pre>{metrics}</pre></details>
      <details><summary>Metadata</summary><pre>{metadata}</pre></details>
      <div class="note-grid">
        <div class="note-controls">
          <label><h3>Category</h3><select data-field="category"><option value=""></option>{options(categories)}</select></label>
          <label><h3>Status</h3><select data-field="status"><option value=""></option>{options(statuses)}</select></label>
        </div>
        <label><h3>Issue Description</h3><textarea data-field="issue_text" placeholder="Describe what looks wrong in this package graph..."></textarea></label>
        <div class="save-row">
          <button type="button" data-action="save-now">Save now</button>
          <span class="save-state">not reviewed</span>
        </div>
      </div>
    </div>
  </div>
</article>
"""


def render_diagnostic_details(details: list[dict[str, Any]]) -> str:
    if not details:
        return ""
    rows = []
    for detail in details:
        reason = str(detail.get("reason") or "")
        metric = str(detail.get("metric") or "")
        value = detail.get("value")
        threshold = detail.get("threshold")
        stage_hint = str(detail.get("stage_hint") or "")
        extras = {
            key: value
            for key, value in detail.items()
            if key
            not in {
                "reason",
                "metric",
                "value",
                "threshold",
                "stage_hint",
                "error_sources",
                "objective_error_sources",
            }
        }
        value_text = "-" if value is None else str(value)
        threshold_text = "-" if threshold is None else str(threshold)
        extras_text = ""
        if extras:
            extras_text = f"<code>{html.escape(json.dumps(extras, ensure_ascii=False, sort_keys=True))}</code>"
        rows.append(
            "<tr>"
            f"<td>{html.escape(reason)}</td>"
            f"<td>{html.escape(metric)}</td>"
            f"<td>{html.escape(value_text)}</td>"
            f"<td>{html.escape(threshold_text)}</td>"
            f"<td>{html.escape(stage_hint)}</td>"
            f"<td>{extras_text}</td>"
            "</tr>"
        )
    return f"""
      <div class="diagnostic-details">
        <h3>Risk details</h3>
        <table>
          <thead><tr><th>Reason</th><th>Metric</th><th>Value</th><th>Threshold</th><th>Stage</th><th>Extra</th></tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
"""


def render_media_grid(media_items: list[Any], *, page_dir: Path | None = None) -> str:
    adopted_items = [item for item in media_items if is_adopted_graph_media(item)]
    source_graph_items = [item for item in media_items if is_source_package_graph_media(item)]
    overlay_items = [item for item in media_items if is_overlay_graph_media(item)]
    alignment_gt_items = [item for item in media_items if is_alignment_or_gt_media(item)]
    alignment_graph_items = [item for item in media_items if is_alignment_graph_media(item)]
    if adopted_items or source_graph_items or overlay_items or alignment_graph_items:
        known = set(id(item) for item in adopted_items + source_graph_items + overlay_items + alignment_gt_items)
        other_items = [item for item in media_items if id(item) not in known]
        sections = []
        if adopted_items:
            sections.append(render_media_section("Adopted graph", adopted_items, "adopted", page_dir=page_dir))
        if source_graph_items:
            sections.append(render_media_section("Source package graph", source_graph_items, "source-package", page_dir=page_dir))
        if overlay_items:
            sections.append(render_media_section("Overlay graph", overlay_items, "overlay", page_dir=page_dir))
        if alignment_gt_items:
            sections.append(render_media_section("Dimension-scaled graph / GT reference", alignment_gt_items, "alignment-gt", page_dir=page_dir))
        if other_items:
            sections.append(render_media_section("Other", other_items, "other", page_dir=page_dir))
        return "\n".join(sections)

    source_items = [item for item in media_items if is_source_media(item)]
    postprocessed_items = [item for item in media_items if is_postprocessed_media(item)]
    result_items = [
        item
        for item in media_items
        if not is_source_media(item) and not is_postprocessed_media(item)
    ]
    sections = []
    if source_items:
        sections.append(render_media_section("Source views", source_items, "source", page_dir=page_dir))
    if postprocessed_items:
        sections.append(render_media_section("Main postprocessed views", postprocessed_items, "postprocessed", page_dir=page_dir))
    if result_items:
        sections.append(render_media_section("GT / Result / Comparison", result_items, "result", page_dir=page_dir))
    return "\n".join(sections)


def render_media_section(
    title: str,
    media_items: list[Any],
    section_kind: str,
    *,
    page_dir: Path | None = None,
) -> str:
    media = "\n".join(render_media(media_item, page_dir=page_dir) for media_item in media_items)
    return f"""
<section class="media-section media-section--{html.escape(section_kind)}">
  <h3 class="media-section-title">{html.escape(title)}</h3>
  <div class="media-row">{media}</div>
</section>
"""


def is_source_media(media_item: Any) -> bool:
    label = str(getattr(media_item, "label", "") or "").strip().lower()
    return bool(label.startswith("source ") or label.endswith(" source") or label.startswith("original "))


def is_adopted_graph_media(media_item: Any) -> bool:
    label = str(getattr(media_item, "label", "") or "").strip().lower()
    return label == "adopted graph" or label.startswith("adopted graph -")


def is_source_package_graph_media(media_item: Any) -> bool:
    label = str(getattr(media_item, "label", "") or "").strip().lower()
    return label.startswith("source package graph")


def is_overlay_graph_media(media_item: Any) -> bool:
    label = str(getattr(media_item, "label", "") or "").strip().lower()
    return label.startswith("overlay graph")


def is_alignment_or_gt_media(media_item: Any) -> bool:
    label = str(getattr(media_item, "label", "") or "").strip().lower()
    return (
        label.startswith("alignment graph")
        or label.startswith("dimension-scaled graph")
        or label.startswith("rotation-only graph")
        or label.startswith("rotation-centered graph")
        or label.startswith("gt reference")
    )


def is_alignment_graph_media(media_item: Any) -> bool:
    label = str(getattr(media_item, "label", "") or "").strip().lower()
    return (
        label.startswith("alignment graph")
        or label.startswith("dimension-scaled graph")
        or label.startswith("rotation-only graph")
        or label.startswith("rotation-centered graph")
    )


def is_postprocessed_media(media_item: Any) -> bool:
    label = str(getattr(media_item, "label", "") or "").strip().lower()
    return label.startswith("postprocessed ")


def render_media(media_item: Any, *, page_dir: Path | None = None) -> str:
    url = page_relative_url(str(media_item.url), page_dir)
    return f"""
<figure class="media">
  <figcaption class="media-title">{html.escape(media_item.label)}</figcaption>
  <a href="{html.escape(url)}" target="_blank">
    <img src="{html.escape(url)}" loading="lazy" alt="{html.escape(media_item.label)}">
  </a>
</figure>
"""


def page_relative_url(url: str, page_dir: Path | None) -> str:
    """Resolve local review asset paths relative to the current HTML page."""
    raw = str(url or "")
    if not raw:
        return ""
    if raw.startswith(("http://", "https://", "data:", "blob:", "mailto:", "#")):
        return raw
    fullflow_prefix = "real_image_process/FPK_PJ_fullflow/"
    if raw.startswith(fullflow_prefix):
        return "/" + raw[len(fullflow_prefix) :]
    clean = raw.lstrip("/")
    if clean.startswith(("runs/", "assets/", "review/")):
        return "/" + clean
    if page_dir is None:
        return raw
    workspace = Path.cwd().resolve()
    fullflow_root = workspace / "real_image_process/FPK_PJ_fullflow"

    def relative_to_page(path: Path) -> str:
        return path.resolve().relative_to(page_dir.resolve()).as_posix()

    def relpath_to_page(path: Path) -> str:
        import os

        return os.path.relpath(path.resolve(), page_dir.resolve()).replace("\\", "/")

    if raw.startswith("/"):
        absolute_candidate = Path(raw).resolve()
        if absolute_candidate.exists():
            try:
                return "/" + absolute_candidate.relative_to(fullflow_root.resolve()).as_posix()
            except ValueError:
                return relpath_to_page(absolute_candidate)
        for root in (fullflow_root, workspace):
            candidate = (root / clean).resolve()
            if candidate.exists():
                return relpath_to_page(candidate)
        return raw

    for root in (fullflow_root, workspace):
        candidate = (root / clean).resolve()
        if candidate.exists():
            try:
                return "/" + candidate.relative_to(fullflow_root.resolve()).as_posix()
            except ValueError:
                return relpath_to_page(candidate)
    local_candidate = (page_dir / clean).resolve()
    if local_candidate.exists():
        return relative_to_page(local_candidate)
    return clean


def render_pager(
    output_root: Path,
    page_paths: list[Path],
    current_page: int,
    *,
    page_labels: list[str] | None = None,
) -> str:
    links = []
    for index, path in enumerate(page_paths, start=1):
        rel = path.relative_to(output_root / "pages").as_posix()
        current = " current" if index == current_page else ""
        label = page_labels[index - 1] if page_labels and index - 1 < len(page_labels) else str(index)
        links.append(f"<a class=\"page-link{current}\" href=\"{html.escape(rel)}\">{html.escape(label)}</a>")
    return "".join(links)


def options(values: tuple[str, ...]) -> str:
    return "".join(f"<option value=\"{html.escape(value)}\">{html.escape(value)}</option>" for value in values)


def page(
    title: str,
    body: str,
    *,
    include_script: bool = False,
    static_prefix: str = "/review/common/static",
) -> str:
    script = f'<script src="{html.escape(static_prefix)}/review.js"></script>' if include_script else ""
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <link rel="stylesheet" href="{html.escape(static_prefix)}/review.css?v=compact_header">
</head>
<body>
{body}
{script}
</body>
</html>
"""
