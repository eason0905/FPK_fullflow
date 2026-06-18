#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from real_image_process.FPK_PJ_fullflow.multiview.integrator import (
    MultiviewOptions,
    integrate_part as integrate_part_in_memory,
)
from real_image_process.FPK_PJ_fullflow.review.adapters.gt_alignment import (
    write_normalized_multiview_overlay_svg,
)
from real_image_process.FPK_PJ_fullflow.review.schema import slugify as multiview_slugify
from real_image_process.FPK_PJ_fullflow.multiview.alignment import build_multiview_alignment
from real_image_process.FPK_PJ_fullflow.multiview.merge_pads import build_multiview_mergy_pad


PAD_LABELS = {"pad", "pad_circle", "pad_dshape", "rect", "circle", "lead"}
OUTLINE_LABELS = {"outline", "package"}
VIEW_COLORS = {
    "top": "#2563eb",
    "bottom": "#16a34a",
    "land": "#dc2626",
    "land_detail": "#9333ea",
    "front": "#ea580c",
    "side": "#0891b2",
    "lead": "#7c3aed",
    "unknown": "#475569",
}
GALLERY_PAGE_SIZE = 120


@dataclass(frozen=True)
class OverlayObject:
    view: str
    source_file: str
    label: str
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class OverlayLayer:
    view: str
    source_file: str
    frame: tuple[float, float, float, float]
    objects: list[OverlayObject]
    unit_scales: dict[str, Any]
    unit_scales: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate package-graph-only multiview overlays without ScanResult or canonical_graph inputs."
    )
    parser.add_argument(
        "--graph-root",
        type=Path,
        required=True,
        help="Directory containing per-part *.package_graph.json files, or the graphs root containing part dirs.",
    )
    parser.add_argument("--part", default="", help="Part number / graph subdirectory name. Omit to build a gallery.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("real_image_process/FPK_PJ_fullflow/assets/datasets/dataset_full_v5"),
        help="Dataset root used only to draw ScanResultFormat GT in the multiview_alignment review panel.",
    )
    parser.add_argument(
        "--multiview-root",
        type=Path,
        default=Path("real_image_process/FPK_PJ_fullflow/runs/v5_fullflow_20260608_115410/outputs/multiview"),
        help="Multiview output root containing parts/*/unified_multiview_layers.json for the alignment review panel.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Gallery mode: maximum number of parts. 0 means all.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.part:
        summary = build_gallery(
            args.graph_root,
            args.output_dir,
            dataset_root=args.dataset_root,
            multiview_root=args.multiview_root,
            limit=args.limit,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return
    summary = build_part_outputs(
        args.graph_root,
        args.part,
        args.output_dir,
        dataset_root=args.dataset_root,
        multiview_root=args.multiview_root,
    )
    visualization_root = infer_visualization_root(args.graph_root)
    part_page = write_part_html(
        args.output_dir,
        {
            "part": summary["part"],
            "views": summary["views"],
            "graph_count": summary["input_graph_count"],
            "package_graph_images": package_graph_images(visualization_root, summary["part"]),
            "top_bottom_land_overlay": Path(summary["outputs"]["top_bottom_land_overlay"]),
            "multi_view_overlay": Path(summary["outputs"]["multi_view_overlay"]),
            "multiview_alignment": Path(summary["outputs"]["multiview_alignment"])
            if summary["outputs"].get("multiview_alignment")
            else None,
            "mergy_pad": Path(summary["outputs"]["mergy_pad"])
            if summary["outputs"].get("mergy_pad")
            else None,
        },
    )
    summary["outputs"]["part_html"] = str(part_page)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_gallery(
    graph_root: Path,
    output_dir: Path,
    *,
    dataset_root: Path | None = None,
    multiview_root: Path | None = None,
    limit: int = 0,
) -> dict[str, Any]:
    part_dirs = sorted(path for path in graph_root.iterdir() if path.is_dir())
    if limit > 0:
        part_dirs = part_dirs[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization_root = infer_visualization_root(graph_root)
    rows = []
    skipped = []
    for part_dir in part_dirs:
        try:
            part_summary = build_part_outputs(
                graph_root,
                part_dir.name,
                output_dir,
                dataset_root=dataset_root,
                multiview_root=multiview_root,
            )
        except SystemExit as exc:
            skipped.append({"part": part_dir.name, "reason": str(exc)})
            continue
        rows.append(
            {
                "part": part_dir.name,
                "views": part_summary["views"],
                "graph_count": part_summary["input_graph_count"],
                "package_graph_images": package_graph_images(visualization_root, part_dir.name),
                "top_bottom_land_overlay": Path(part_summary["outputs"]["top_bottom_land_overlay"]),
                "multi_view_overlay": Path(part_summary["outputs"]["multi_view_overlay"]),
                "multiview_alignment": Path(part_summary["outputs"]["multiview_alignment"])
                if part_summary["outputs"].get("multiview_alignment")
                else None,
                "mergy_pad": Path(part_summary["outputs"]["mergy_pad"])
                if part_summary["outputs"].get("mergy_pad")
                else None,
            }
        )
    index_path = output_dir / "index.html"
    write_gallery_html(index_path, rows, graph_root=graph_root, visualization_root=visualization_root)
    summary = {
        "graph_root": str(graph_root),
        "output_dir": str(output_dir),
        "index": str(index_path),
        "part_count": len(rows),
        "skipped_count": len(skipped),
        "skipped": skipped[:20],
        "method": (
            "Package-graph-only gallery. Package graph PNGs come from reconstruction visualization; "
            "overlays are generated directly from *.package_graph.json. ScanResultFormat.txt is only "
            "used as an optional GT reference in the alignment panel. No canonical_graph.json is read."
        ),
    }
    (output_dir / "gallery_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def build_part_outputs(
    graph_root: Path,
    part: str,
    output_dir: Path,
    *,
    dataset_root: Path | None = None,
    multiview_root: Path | None = None,
) -> dict[str, Any]:
    part_graph_dir = graph_root / part
    if not part_graph_dir.is_dir():
        part_graph_dir = graph_root
    graph_paths = sorted(part_graph_dir.glob("*.package_graph.json"))
    if not graph_paths:
        raise SystemExit(f"No package graph files found under {part_graph_dir}")

    layers = [layer_from_graph(path) for path in graph_paths]
    layers = [layer for layer in layers if layer.objects]
    if not layers:
        raise SystemExit("No drawable package graph objects found.")

    out_dir = output_dir / slug(part)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_multiview_integrator_overlay(
        out_dir / "package_graph_top_bottom_land_overlay.svg",
        part=part,
        graph_paths=graph_paths,
        title=f"{part} package-graph top/bottom/land overlay",
        include_extra=False,
    )
    write_multiview_integrator_overlay(
        out_dir / "package_graph_multi_view_overlay.svg",
        part=part,
        graph_paths=graph_paths,
        title=f"{part} package-graph multi-view overlay",
        include_extra=True,
    )
    alignment_summary = build_part_multiview_alignment(
        part=part,
        multiview_part=package_graph_part_number(graph_paths, fallback=part),
        out_dir=out_dir,
        dataset_root=dataset_root,
        multiview_root=multiview_root,
    )
    mergy_pad_summary = build_part_mergy_pad(
        aligned_layers_path=Path(alignment_summary["aligned_multiview_layers_json"])
        if alignment_summary and alignment_summary.get("aligned_multiview_layers_json")
        else None,
        out_dir=out_dir,
        part=package_graph_part_number(graph_paths, fallback=part),
    )
    summary = {
        "part": part,
        "graph_root": str(graph_root),
        "output_dir": str(out_dir),
        "input_graph_count": len(graph_paths),
        "drawn_layer_count": len(layers),
        "views": sorted({layer.view for layer in layers}),
        "outputs": {
            "top_bottom_land_overlay": str(out_dir / "package_graph_top_bottom_land_overlay.svg"),
            "multi_view_overlay": str(out_dir / "package_graph_multi_view_overlay.svg"),
            "multiview_alignment": alignment_summary.get("alignment_svg") if alignment_summary else "",
            "multiview_alignment_json": alignment_summary.get("alignment_json") if alignment_summary else "",
            "aligned_multiview_layers": alignment_summary.get("aligned_multiview_layers_json") if alignment_summary else "",
            "mergy_pad": mergy_pad_summary.get("mergy_pad_svg") if mergy_pad_summary else "",
            "mergy_pad_json": mergy_pad_summary.get("mergy_pad_json") if mergy_pad_summary else "",
        },
        "multiview_alignment": alignment_summary,
        "mergy_pad": mergy_pad_summary,
        "method": (
            "Package-graph-only probe. Main overlay keeps reconstructed package graph coordinates "
            "with one common dimension-calibrated display scale. All-view overlay reuses the "
            "multiview integrator in memory to synthesize lead pads and inner land pads. "
            "ScanResultFormat.txt is only used as an optional GT reference in the alignment panel. "
            "No canonical_graph.json is read."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def build_part_mergy_pad(
    *,
    aligned_layers_path: Path | None,
    out_dir: Path,
    part: str,
) -> dict[str, Any] | None:
    if aligned_layers_path is None or not aligned_layers_path.is_file():
        return None
    return build_multiview_mergy_pad(
        aligned_layers_path=aligned_layers_path,
        output_dir=out_dir,
        part=part,
    )


def build_part_multiview_alignment(
    *,
    part: str,
    multiview_part: str,
    out_dir: Path,
    dataset_root: Path | None,
    multiview_root: Path | None,
) -> dict[str, Any] | None:
    if multiview_root is None:
        return None
    unified_layers = multiview_root / "parts" / multiview_slugify(multiview_part) / "unified_multiview_layers.json"
    if not unified_layers.is_file():
        return None
    scan_result = dataset_scan_result_path(dataset_root, part, multiview_part)
    if scan_result is not None and not scan_result.is_file():
        scan_result = None
    return build_multiview_alignment(
        unified_layers_path=unified_layers,
        scan_result_path=scan_result,
        output_dir=out_dir,
        part=multiview_part,
    )


def package_graph_part_number(graph_paths: list[Path], *, fallback: str) -> str:
    for graph_path in graph_paths:
        try:
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        part_number = str(graph.get("part_number") or graph.get("_part_number") or "").strip()
        if part_number:
            return part_number
    return fallback


def dataset_scan_result_path(dataset_root: Path | None, part: str, multiview_part: str) -> Path | None:
    if dataset_root is None:
        return None
    for candidate_part in (multiview_part, part):
        candidate = dataset_root / candidate_part / "ScanResultFormat.txt"
        if candidate.is_file():
            return candidate
    return dataset_root / multiview_part / "ScanResultFormat.txt"


def infer_visualization_root(graph_root: Path) -> Path | None:
    parts = graph_root.parts
    for index in range(len(parts) - 3):
        if (
            parts[index] == "outputs"
            and parts[index + 1] == "reconstruction"
            and index + 2 < len(parts)
            and index + 3 < len(parts)
        ):
            run_id = parts[index + 2]
            output_root = Path(*parts[: index + 1])
            candidate = output_root / "visualization" / f"reconstruction_{run_id}"
            if candidate.is_dir():
                return candidate
    return None


def package_graph_images(visualization_root: Path | None, part: str) -> list[Path]:
    if visualization_root is None:
        return []
    part_dir = visualization_root / part
    if not part_dir.is_dir():
        return []
    return sorted(part_dir.glob("*.package_graph.png"))


def write_gallery_html(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    graph_root: Path,
    visualization_root: Path | None,
) -> None:
    css = """
body { margin: 0; font-family: Arial, sans-serif; color: #111827; background: #f8fafc; }
header { position: sticky; top: 0; z-index: 2; padding: 16px 20px; background: #ffffff; border-bottom: 1px solid #d1d5db; }
h1 { margin: 0 0 6px; font-size: 20px; }
.meta { color: #475569; font-size: 13px; }
.toolbar { padding: 12px 20px; display: flex; gap: 10px; align-items: center; background: #f8fafc; border-bottom: 1px solid #e5e7eb; }
.toolbar input { width: min(520px, 80vw); padding: 8px 10px; border: 1px solid #cbd5e1; border-radius: 4px; font-size: 14px; }
.pagination { margin: 14px 20px 0; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.pagination a, .pagination strong { padding: 7px 10px; border: 1px solid #cbd5e1; border-radius: 4px; background: #ffffff; color: #111827; text-decoration: none; font-size: 13px; }
.pagination strong { background: #e5e7eb; }
.case-list { margin: 14px 20px 32px; display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 8px; }
.case-link { display: block; padding: 10px 12px; background: #ffffff; border: 1px solid #d1d5db; border-radius: 6px; color: #111827; text-decoration: none; }
.case-link:hover { border-color: #64748b; background: #f8fafc; }
.case-title { display: block; font-size: 14px; font-weight: 700; }
.case-meta { display: block; margin-top: 3px; color: #64748b; font-size: 12px; }
.paged-cases { margin-bottom: 32px; }
.case { margin: 18px 20px; padding: 14px; background: #ffffff; border: 1px solid #d1d5db; border-radius: 6px; }
.case h2 { margin: 0 0 10px; font-size: 16px; }
.case-layout { display: grid; grid-template-columns: 1fr; gap: 12px; }
.overlay-row { display: grid; grid-template-columns: minmax(320px, 1fr) minmax(320px, 1fr); gap: 12px; align-items: start; }
.panel { border: 1px solid #e5e7eb; border-radius: 6px; overflow: hidden; background: #ffffff; }
.panel h3 { margin: 0; padding: 8px 10px; font-size: 13px; background: #f1f5f9; border-bottom: 1px solid #e5e7eb; }
.media { padding: 8px; }
.media img { display: block; width: 100%; max-height: 620px; object-fit: contain; background: transparent; border: 1px solid #e5e7eb; }
.thumbs { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 8px; }
.caption { margin-top: 4px; font-size: 11px; color: #64748b; word-break: break-all; }
.empty { padding: 16px; color: #64748b; font-size: 13px; }
@media (max-width: 1100px) { .overlay-row { grid-template-columns: 1fr; } }
"""
    lines = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Package Graph Multiview Overlay Gallery</title>",
        f"<style>{css}</style>",
        "</head>",
        "<body>",
        "<header>",
        "<h1>Package Graph Multiview Overlay Gallery</h1>",
        f'<div class="meta">parts: {len(rows)} | graph root: {escape(str(graph_root))}</div>',
        f'<div class="meta">visualization root: {escape(str(visualization_root) if visualization_root else "not found")}</div>',
        '<div class="meta">Inputs: reconstruction *.package_graph.json and package graph PNGs. ScanResultFormat.txt is optional GT reference for alignment panel. No canonical_graph.json.</div>',
        "</header>",
        '<div class="toolbar"><input id="filter" type="search" placeholder="Filter part number" autofocus></div>',
        '<main class="case-list" id="cases">',
    ]
    for row in rows:
        part_page = write_part_html(path.parent, row)
        row["part_page"] = part_page
    page_paths = write_gallery_page_html(
        output_dir=path.parent,
        rows=rows,
        graph_root=graph_root,
        visualization_root=visualization_root,
        css=css,
    )
    lines.append('<nav class="pagination">')
    for page_index, page_path in enumerate(page_paths, start=1):
        lines.append(f'<a href="{escape(relative_href(page_path, path.parent))}">Page {page_index}</a>')
    lines.append("</nav>")
    lines.extend(
        [
            '<p class="empty">Open one of the paged lists above. Each part page loads only that part media.</p>',
            "</main>",
            "</body>",
            "</html>",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_gallery_page_html(
    *,
    output_dir: Path,
    rows: list[dict[str, Any]],
    graph_root: Path,
    visualization_root: Path | None,
    css: str,
) -> list[Path]:
    page_paths = []
    total_pages = max(1, (len(rows) + GALLERY_PAGE_SIZE - 1) // GALLERY_PAGE_SIZE)
    for page_index in range(total_pages):
        start = page_index * GALLERY_PAGE_SIZE
        page_rows = rows[start : start + GALLERY_PAGE_SIZE]
        page_path = output_dir / f"page_{page_index + 1}.html"
        page_paths.append(page_path)
        lines = [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>Package Graph Overlay Page {page_index + 1}</title>",
            f"<style>{css}</style>",
            "</head>",
            "<body>",
            "<header>",
            '<div class="meta"><a href="index.html">Back to gallery index</a></div>',
            f"<h1>Package Graph Multiview Overlay Gallery - Page {page_index + 1}/{total_pages}</h1>",
            f'<div class="meta">parts: {len(rows)} | showing {start + 1}-{start + len(page_rows)} | graph root: {escape(str(graph_root))}</div>',
            f'<div class="meta">visualization root: {escape(str(visualization_root) if visualization_root else "not found")}</div>',
            "</header>",
            '<div class="toolbar"><input id="filter" type="search" placeholder="Filter this page" autofocus></div>',
            '<nav class="pagination">',
        ]
        for nav_index in range(total_pages):
            label = f"Page {nav_index + 1}"
            href = f"page_{nav_index + 1}.html"
            if nav_index == page_index:
                lines.append(f'<strong>{escape(label)}</strong>')
            else:
                lines.append(f'<a href="{escape(href)}">{escape(label)}</a>')
        lines.extend(['</nav>', '<main class="paged-cases" id="cases">'])
        for row in page_rows:
            lines.extend(render_row(row, page_path.parent))
        lines.extend(
            [
                "</main>",
                "<script>",
                "const filter = document.getElementById('filter');",
                "const cases = [...document.querySelectorAll('.case')];",
                "filter.addEventListener('input', () => {",
                "  const q = filter.value.toLowerCase();",
                "  for (const item of cases) item.style.display = item.dataset.part.includes(q) ? '' : 'none';",
                "});",
                "</script>",
                "</body>",
                "</html>",
            ]
        )
        page_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return page_paths


def write_part_html(output_dir: Path, row: dict[str, Any]) -> Path:
    part = str(row["part"])
    part_dir = output_dir / slug(part)
    part_path = part_dir / "index.html"
    css = """
body { margin: 0; font-family: Arial, sans-serif; color: #111827; background: #f8fafc; }
header { padding: 16px 20px; background: #ffffff; border-bottom: 1px solid #d1d5db; }
a { color: #2563eb; }
h1 { margin: 0 0 6px; font-size: 20px; }
.meta { color: #475569; font-size: 13px; }
.case { margin: 18px 20px; padding: 14px; background: #ffffff; border: 1px solid #d1d5db; border-radius: 6px; }
.case h2 { margin: 0 0 10px; font-size: 16px; }
.case-layout { display: grid; grid-template-columns: 1fr; gap: 12px; }
.overlay-row { display: grid; grid-template-columns: minmax(320px, 1fr) minmax(320px, 1fr); gap: 12px; align-items: start; }
.panel { border: 1px solid #e5e7eb; border-radius: 6px; overflow: hidden; background: #ffffff; }
.panel h3 { margin: 0; padding: 8px 10px; font-size: 13px; background: #f1f5f9; border-bottom: 1px solid #e5e7eb; }
.media { padding: 8px; }
.media img, .media object { display: block; width: 100%; max-height: 620px; object-fit: contain; background: transparent; border: 1px solid #e5e7eb; }
.thumbs { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 8px; }
.caption { margin-top: 4px; font-size: 11px; color: #64748b; word-break: break-all; }
.empty { padding: 16px; color: #64748b; font-size: 13px; }
@media (max-width: 1100px) { .overlay-row { grid-template-columns: 1fr; } }
"""
    lines = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{escape(part)} Package Graph Overlay</title>",
        f"<style>{css}</style>",
        "</head>",
        "<body>",
        "<header>",
        f'<div class="meta"><a href="../index.html">Back to gallery index</a></div>',
        f"<h1>{escape(part)}</h1>",
        f'<div class="meta">graphs: {int(row["graph_count"])} | views: {escape(", ".join(row["views"]))}</div>',
        "</header>",
        *render_row(row, part_dir),
        "</body>",
        "</html>",
    ]
    part_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return part_path


def render_index_link(row: dict[str, Any], part_page: Path, html_dir: Path) -> str:
    part = str(row["part"])
    views = ", ".join(row["views"])
    href = relative_href(part_page, html_dir)
    return (
        f'<a class="case-link" data-part="{escape(part.lower())}" href="{escape(href)}">'
        f'<span class="case-title">{escape(part)}</span>'
        f'<span class="case-meta">graphs: {int(row["graph_count"])} | views: {escape(views)}</span>'
        "</a>"
    )


def render_row(row: dict[str, Any], html_dir: Path) -> list[str]:
    part = str(row["part"])
    views = ", ".join(row["views"])
    graph_count = int(row["graph_count"])
    return [
        f'<section class="case" data-part="{escape(part.lower())}">',
        f"<h2>{escape(part)} <span class=\"meta\">graphs: {graph_count} | views: {escape(views)}</span></h2>",
        '<div class="case-layout">',
        render_package_graph_panel(row["package_graph_images"], html_dir),
        '<div class="overlay-row">',
        render_svg_panel("Main-view overlay", row["top_bottom_land_overlay"], html_dir),
        render_svg_panel("All-view overlay", row["multi_view_overlay"], html_dir),
        "</div>",
        '<div class="overlay-row">',
        render_alignment_panel(row.get("multiview_alignment"), html_dir),
        render_mergy_pad_panel(row.get("mergy_pad"), html_dir),
        "</div>",
        "</div>",
        "</section>",
    ]


def render_package_graph_panel(paths: list[Path], html_dir: Path) -> str:
    if not paths:
        body = '<div class="empty">No package graph PNG found.</div>'
    else:
        items = []
        for media_path in paths:
            rel = media_href(media_path, html_dir)
            items.append(
                '<div>'
                f'<a href="{escape(rel)}"><img src="{escape(rel)}" loading="lazy"></a>'
                f'<div class="caption">{escape(media_path.name)}</div>'
                "</div>"
            )
        body = '<div class="thumbs">' + "\n".join(items) + "</div>"
    return f'<div class="panel"><h3>1. Package graph</h3><div class="media">{body}</div></div>'


def render_svg_panel(title: str, media_path: Path, html_dir: Path) -> str:
    rel = media_href(media_path, html_dir)
    body = (
        f'<a href="{escape(rel)}"><img src="{escape(rel)}" loading="lazy"></a>'
        f'<div class="caption">{escape(media_path.name)}</div>'
    )
    return f'<div class="panel"><h3>{escape(title)}</h3><div class="media">{body}</div></div>'


def render_alignment_panel(media_path: Path | None, html_dir: Path) -> str:
    if media_path is None:
        body = '<div class="empty">No multiview alignment output found.</div>'
    else:
        rel = media_href(media_path, html_dir)
        body = (
            f'<a href="{escape(rel)}"><img src="{escape(rel)}" loading="lazy"></a>'
            f'<div class="caption">{escape(media_path.name)}</div>'
        )
    return f'<div class="panel"><h3>3. Alignment / GT</h3><div class="media">{body}</div></div>'


def render_mergy_pad_panel(media_path: Path | None, html_dir: Path) -> str:
    if media_path is None:
        body = '<div class="empty">No merge pad output found.</div>'
    else:
        rel = media_href(media_path, html_dir)
        body = (
            f'<a href="{escape(rel)}"><img src="{escape(rel)}" loading="lazy"></a>'
            f'<div class="caption">{escape(media_path.name)}</div>'
        )
    return f'<div class="panel"><h3>3. Merge pad</h3><div class="media">{body}</div></div>'


def relative_href(path: Path, base_dir: Path) -> str:
    return os.path.relpath(path.resolve(), base_dir.resolve())


def media_href(path: Path, base_dir: Path) -> str:
    href = relative_href(path, base_dir)
    try:
        version = path.stat().st_mtime_ns
    except OSError:
        return href
    return f"{href}?v={version}"


def layer_from_graph(path: Path) -> OverlayLayer:
    graph = json.loads(path.read_text(encoding="utf-8"))
    view = normalize_view(str(graph.get("view") or "unknown"))
    raw_objects = list(graph.get("objects") or [])
    frame = graph_display_frame(raw_objects) or graph_frame(raw_objects)
    if frame is None:
        return OverlayLayer(
            view=view,
            source_file=path.name,
            frame=(0.0, 0.0, 1.0, 1.0),
            objects=[],
            unit_scales={"x": 1.0, "y": 1.0, "source": "graph_pixels"},
        )
    objects = []
    for obj in raw_objects:
        bbox = object_bbox(obj)
        if bbox is None:
            continue
        label = str(obj.get("source_label") or obj.get("label") or "object")
        if not should_draw_object(label):
            continue
        objects.append(
            OverlayObject(
                view=view,
                source_file=path.name,
                label=label,
                bbox=bbox,
            )
        )
    return OverlayLayer(
        view=view,
        source_file=path.name,
        frame=frame,
        objects=objects,
        unit_scales=graph_dimension_unit_scales(graph),
    )


def normalize_view(view: str) -> str:
    value = view.strip().lower()
    if value in VIEW_COLORS:
        return value
    return value or "unknown"


def object_bbox(obj: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = obj.get("bbox_reconstructed") or obj.get("bbox") or []
    if len(bbox) < 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    if x2 == x1 or y2 == y1:
        return None
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def should_draw_object(label: str) -> bool:
    normalized = label.strip().lower()
    return normalized in PAD_LABELS or normalized in OUTLINE_LABELS


def graph_frame(objects: list[dict[str, Any]]) -> tuple[float, float, float, float] | None:
    outline_boxes = []
    object_boxes = []
    for obj in objects:
        bbox = object_bbox(obj)
        if bbox is None:
            continue
        label = str(obj.get("source_label") or obj.get("label") or "").lower()
        object_boxes.append(bbox)
        if label in OUTLINE_LABELS:
            outline_boxes.append(bbox)
    boxes = outline_boxes or object_boxes
    if not boxes:
        return None
    return union_boxes(boxes)


def graph_display_frame(objects: list[dict[str, Any]]) -> tuple[float, float, float, float] | None:
    pad_boxes = []
    for obj in objects:
        bbox = object_bbox(obj)
        if bbox is None:
            continue
        label = str(obj.get("source_label") or obj.get("label") or "").lower()
        if label in PAD_LABELS:
            pad_boxes.append(bbox)
    return union_boxes(pad_boxes) if pad_boxes else None


def union_boxes(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def graph_dimension_unit_scales(graph: dict[str, Any]) -> dict[str, Any]:
    objects_by_id = {str(obj.get("id")): obj for obj in graph.get("objects") or [] if obj.get("id") is not None}
    samples: dict[str, list[float]] = {"x": [], "y": []}
    for dim in graph.get("dimensions") or []:
        if str(dim.get("status") or "") != "accepted":
            continue
        axis = str(dim.get("axis") or "").lower()
        if axis not in samples:
            continue
        value = positive_float(dim.get("value"))
        if value is None:
            continue
        pixel_distance = dimension_pixel_distance(dim, objects_by_id)
        if pixel_distance is None or pixel_distance <= 0.0:
            continue
        samples[axis].append(value / pixel_distance)

    all_samples = samples["x"] + samples["y"]
    fallback = median_positive(all_samples) or positive_float(graph.get("global_scale")) or 1.0
    x_scale = median_positive(samples["x"]) or positive_float(((graph.get("metrics") or {}).get("axis_scale_x"))) or fallback
    y_scale = median_positive(samples["y"]) or positive_float(((graph.get("metrics") or {}).get("axis_scale_y"))) or fallback
    return {
        "x": x_scale,
        "y": y_scale,
        "source": "accepted_dimensions" if all_samples else "graph_metrics_or_pixels",
        "x_sample_count": len(samples["x"]),
        "y_sample_count": len(samples["y"]),
    }


def positive_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result <= 0.0:
        return None
    return result


def median_positive(values: list[float]) -> float | None:
    positives = sorted(value for value in values if value > 0.0)
    if not positives:
        return None
    mid = len(positives) // 2
    if len(positives) % 2:
        return positives[mid]
    return (positives[mid - 1] + positives[mid]) / 2.0


def dimension_pixel_distance(dim: dict[str, Any], objects_by_id: dict[str, dict[str, Any]]) -> float | None:
    axis = str(dim.get("axis") or "").lower()
    target_ids = [str(item) for item in dim.get("target_ids") or []]
    if axis not in {"x", "y"} or not target_ids:
        return None
    target_boxes = [object_bbox(objects_by_id[target_id]) for target_id in target_ids if target_id in objects_by_id]
    target_boxes = [bbox for bbox in target_boxes if bbox is not None]
    if not target_boxes:
        return None
    if len(target_boxes) == 1:
        x1, y1, x2, y2 = target_boxes[0]
        return (x2 - x1) if axis == "x" else (y2 - y1)

    anchors = list(dim.get("anchors") or [])
    anchor_a = str(anchors[0] if len(anchors) >= 1 else "center")
    anchor_b = str(anchors[1] if len(anchors) >= 2 else "center")
    first = anchor_coordinate(target_boxes[0], axis, anchor_a)
    second = anchor_coordinate(target_boxes[1], axis, anchor_b)
    return abs(second - first)


def anchor_coordinate(bbox: tuple[float, float, float, float], axis: str, anchor: str) -> float:
    x1, y1, x2, y2 = bbox
    normalized = anchor.lower()
    if axis == "x":
        if normalized == "left_edge":
            return x1
        if normalized == "right_edge":
            return x2
        return (x1 + x2) / 2.0
    if normalized == "top_edge":
        return y1
    if normalized == "bottom_edge":
        return y2
    return (y1 + y2) / 2.0


def calibrated_frame(
    frame: tuple[float, float, float, float],
    unit_scales: dict[str, Any],
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = frame
    unit_x = float(unit_scales["x"])
    unit_y = float(unit_scales["y"])
    return (x1 * unit_x, y1 * unit_y, x2 * unit_x, y2 * unit_y)


def overlay_common_scale(
    frames: list[tuple[float, float, float, float]],
    target: tuple[float, float, float, float],
) -> float:
    tx1, ty1, tx2, ty2 = target
    target_w = max(tx2 - tx1, 1.0)
    target_h = max(ty2 - ty1, 1.0)
    max_source_w = max(max(frame[2] - frame[0], 1e-9) for frame in frames)
    max_source_h = max(max(frame[3] - frame[1], 1e-9) for frame in frames)
    return min(target_w / max_source_w, target_h / max_source_h)


def graph_to_overlay_transform(
    frame: tuple[float, float, float, float],
    target: tuple[float, float, float, float],
    scale: float,
    unit_scales: dict[str, Any],
) -> dict[str, float]:
    fx1, fy1, fx2, fy2 = frame
    tx1, ty1, tx2, ty2 = target
    return {
        "scale": scale,
        "unit_x": float(unit_scales["x"]),
        "unit_y": float(unit_scales["y"]),
        "source_cx": (fx1 + fx2) / 2.0,
        "source_cy": (fy1 + fy2) / 2.0,
        "target_cx": (tx1 + tx2) / 2.0,
        "target_cy": (ty1 + ty2) / 2.0,
    }


def map_bbox(
    bbox: tuple[float, float, float, float],
    transform: dict[str, float],
) -> tuple[float, float, float, float]:
    scale = transform["scale"]
    unit_x = transform["unit_x"]
    unit_y = transform["unit_y"]
    source_cx = transform["source_cx"]
    source_cy = transform["source_cy"]
    target_cx = transform["target_cx"]
    target_cy = transform["target_cy"]
    x1, y1, x2, y2 = bbox
    return (
        target_cx + (x1 - source_cx) * unit_x * scale,
        target_cy + (y1 - source_cy) * unit_y * scale,
        target_cx + (x2 - source_cx) * unit_x * scale,
        target_cy + (y2 - source_cy) * unit_y * scale,
    )


def write_overlay_svg(path: Path, layers: list[OverlayLayer], *, title: str) -> None:
    width = 1100
    height = 900
    target = (120.0, 110.0, 840.0, 830.0)
    common_scale = overlay_common_scale([calibrated_frame(layer.frame, layer.unit_scales) for layer in layers], target)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="28" y="40" font-family="monospace" font-size="22" fill="#0f172a">{escape(title)}</text>',
        '<text x="28" y="66" font-family="monospace" font-size="13" fill="#64748b">'
        "package_graph only; reconstructed coordinates with common dimension-calibrated display scale; no GT/canonical input</text>",
    ]
    for index, layer in enumerate(layers):
        color = VIEW_COLORS.get(layer.view, VIEW_COLORS["unknown"])
        transform = graph_to_overlay_transform(layer.frame, target, common_scale, layer.unit_scales)
        y = 120 + index * 24
        elements.append(f'<rect x="880" y="{y - 11}" width="13" height="13" fill="{color}" opacity="0.35" stroke="{color}"/>')
        elements.append(
            f'<text x="902" y="{y}" font-family="monospace" font-size="12" fill="#334155">'
            f'{escape(layer.view)} {escape(layer.source_file)} scale={escape(str(layer.unit_scales.get("source")))}</text>'
        )
        for obj in layer.objects:
            x1, y1, x2, y2 = map_bbox(obj.bbox, transform)
            x = x1
            y = y1
            w = max(x2 - x1, 1.0)
            h = max(y2 - y1, 1.0)
            elements.append(gallery_overlay_object_svg(obj, x, y, w, h, color))
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def gallery_overlay_object_svg(obj: OverlayObject, x: float, y: float, width: float, height: float, color: str) -> str:
    label = obj.label.lower()
    is_outline = label in OUTLINE_LABELS
    fill = "none" if is_outline else color
    opacity = "0.95" if is_outline else "0.20"
    stroke_width = 3 if is_outline else 2
    attrs = (
        f'data-view="{escape(obj.view)}" data-label="{escape(obj.label)}" '
        f'data-source-file="{escape(obj.source_file)}"'
    )
    if "circle" in label and not is_outline:
        return (
            f'<ellipse cx="{x + width / 2.0:.3f}" cy="{y + height / 2.0:.3f}" '
            f'rx="{width / 2.0:.3f}" ry="{height / 2.0:.3f}" fill="{fill}" '
            f'stroke="{color}" stroke-width="{stroke_width}" opacity="{opacity}" {attrs}/>'
        )
    return (
        f'<rect x="{x:.3f}" y="{y:.3f}" width="{width:.3f}" height="{height:.3f}" '
        f'fill="{fill}" stroke="{color}" stroke-width="{stroke_width}" opacity="{opacity}" {attrs}/>'
    )


def write_multiview_integrator_overlay(
    path: Path,
    *,
    part: str,
    graph_paths: list[Path],
    title: str,
    include_extra: bool,
) -> None:
    graphs = []
    for graph_path in graph_paths:
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        graph["_graph_path"] = str(graph_path)
        graph["_part_number"] = part
        graph["_raw_view"] = normalize_view(str(graph.get("view") or "unknown"))
        graphs.append(graph)
    canonical = integrate_part_in_memory(
        part,
        graphs,
        MultiviewOptions(),
        dataset_root=None,
    )
    payload = canonical.get("multiview_overlay") or {}
    if payload.get("layers") or payload.get("extra_objects"):
        write_normalized_multiview_overlay_svg(
            path,
            payload,
            title=title,
            subtitle=(
                "package_graph only; multiview integrator synthesized lead pads "
                "and inner land pads in memory; no GT/canonical file input"
            ),
            include_extra=include_extra,
            display_context={
                "width": 1100,
                "height": 900,
                "target": (120.0, 110.0, 840.0, 830.0),
            },
        )
        return
    write_overlay_svg(path, [layer_from_graph(graph_path) for graph_path in graph_paths], title=title)


def slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def escape(value: str) -> str:
    return html.escape(value, quote=True)


if __name__ == "__main__":
    main()
