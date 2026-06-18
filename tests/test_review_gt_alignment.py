from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from real_image_process.FPK_PJ_fullflow.review.adapters.gt_alignment import (
    DISPLAY_SVG_HEIGHT,
    DISPLAY_SVG_TARGET,
    DISPLAY_SVG_WIDTH,
    build_final_comparison_review,
    build_gt_alignment_review,
    display_scale_context_from_canonical,
    effective_partial_dimension_semantics,
    item_from_row,
    main_view_overlay_evidence_from_canonical,
    main_view_graph_items_from_canonical,
    main_postprocessed_graph_evidence_from_canonical,
    multiview_overlay_evidence_from_canonical,
    package_graph_visualization_path,
    overlay_common_scale,
    overlay_projection_axis,
    partial_dimension_overlay_bbox,
    scan_result_display_evidence_from_canonical,
    scan_result_gt_display_frame,
    source_image_evidence_from_canonical,
    sort_review_items_by_risk,
)
from real_image_process.FPK_PJ_fullflow.review.common.schemas import ReviewItem
from real_image_process.FPK_PJ_fullflow.review.schema import slugify


class GTAlignmentReviewTests(unittest.TestCase):
    def test_sort_review_items_by_risk_keeps_highest_risk_first(self) -> None:
        items = [
            review_item("LOW", 1.0),
            review_item("HIGH", 90.0),
            review_item("MID", 50.0),
        ]

        ordered = sort_review_items_by_risk(items)

        self.assertEqual([item.part_number for item in ordered], ["HIGH", "MID", "LOW"])

    def test_build_gt_alignment_review_writes_reason_pages_and_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "FPK_PJ_fullflow"
            alignment_root = root / "runs" / "run1" / "outputs" / "eval" / "gt_alignment"
            part_dir = alignment_root / "parts" / "PART"
            scan_svg = part_dir / "scan_result.svg"
            canonical = root / "runs" / "run1" / "outputs" / "multiview" / "parts" / "PART" / "unified_multiview_layers.json"
            source_image = root / "assets" / "datasets" / "dataset_full_v5" / "PART" / "extract_image" / "part_top.png"
            scan = root / "assets" / "datasets" / "dataset_full_v5" / "PART" / "ScanResultFormat.txt"
            for path in (
                part_dir / "alignment.json",
                    scan_svg,
                    canonical.with_name("gt_reference.svg"),
                    canonical.with_name("default_aligned_result.svg"),
                    canonical.with_name("default_comparison.svg"),
                    canonical.with_name("aligned_result.svg"),
                    canonical.with_name("comparison.svg"),
                    canonical.with_name("unified_multiview_layers.svg"),
                    canonical.with_name("final_graph.json"),
                    scan,
                    source_image,
                ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            canonical.parent.mkdir(parents=True, exist_ok=True)
            canonical.write_text(
                json.dumps(
                    {
                        "summary": {
                            "evidence_summary": {
                                "evidence_ref_count": 1,
                            }
                        },
                        "evidence_refs": [
                            {
                                "evidence_type": "package_graph",
                                "raw_view": "top",
                                "canonical_view": "top",
                                "image_path": str(source_image),
                                "graph_path": str(canonical.with_name("top.package_graph.json")),
                            }
                        ],
                        "conflicts": [{"type": "dimension_value_conflict"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            canonical.with_name("final_graph.json").write_text(
                json.dumps(
                    {
                        "coordinate_system": {"name": "scan_result_reference_aligned_2d"},
                        "alignment_transform": {"strategy": "default"},
                        "conflicts": [{"type": "dimension_value_conflict"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            row = {
                "part_number": "PART",
                "status": "mismatch",
                "reasons": ["lead_count_mismatch"],
                "error_sources": ["package_graph_reconstruction"],
                "objective_error_sources": ["model_prediction", "package_graph_reconstruction"],
                "stage_hints": ["package_graph_package_pad_reconstruction_count_mismatch"],
                "alignment_path": str(part_dir / "alignment.json"),
                "scan_result_svg_path": str(scan_svg),
                "scan_result_path": str(scan),
                "unified_multiview_layers_path": str(canonical),
                "final_graph_path": str(canonical.with_name("final_graph.json")),
                "alignment_scores": {
                    "overall_score": 0.4,
                    "outline_iou": 0.9,
                    "land_iou": None,
                    "lead_iou": 0.5,
                    "pad_layout_score": 0.25,
                    "dimension_mismatch_count": 2,
                    "dimension_count": 5,
                    "dimension_value_score": 0.6,
                    "land_pad_count_match": False,
                    "lead_count_match": True,
                    "count_checks": {"land_count": False, "lead_count": True},
                },
                "gt": {"role_counts": {"lead": 1}},
                "graph": {"role_counts": {}},
                "checks": [
                    {
                        "name": "lead_count",
                        "status": "mismatch",
                        "reason": "lead_count_mismatch",
                        "stage_hint": "package_graph_package_pad_reconstruction_count_mismatch",
                    }
                ],
            }
            alignment_root.mkdir(parents=True, exist_ok=True)
            (alignment_root / "mismatches.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            output_root = root / "runs" / "run1" / "outputs" / "review" / "gt_alignment"
            result = build_gt_alignment_review(
                alignment_root=alignment_root,
                output_root=output_root,
                fullflow_root=root,
                run_id="run1",
            )

            self.assertEqual(result["total_items"], 1)
            self.assertTrue((output_root / "index.html").exists())
            self.assertTrue((output_root / "by_reason" / "lead_count_mismatch.html").exists())
            self.assertTrue((output_root / "by_source" / "package_graph_reconstruction.html").exists())
            self.assertTrue((output_root / "by_objective_source" / "model_prediction.html").exists())
            self.assertTrue((output_root / "by_objective_source" / "package_graph_reconstruction.html").exists())
            self.assertTrue((output_root / "by_stage_hint" / "package_graph_package_pad_reconstruction_count_mismatch.html").exists())
            self.assertTrue(
                (
                    output_root
                    / "by_check"
                    / f"{slugify('lead_count__package_graph_package_pad_reconstruction_count_mismatch')}.html"
                ).exists()
            )
            self.assertTrue((output_root / "data" / "notes.json").exists())
            self.assertTrue((output_root / "data" / "cases.json").exists())
            self.assertEqual(len(result["source_pages"]), 1)
            self.assertEqual(len(result["objective_source_pages"]), 2)
            self.assertEqual(len(result["stage_pages"]), 1)
            self.assertEqual(len(result["check_pages"]), 1)
            cases = json.loads((output_root / "data" / "cases.json").read_text(encoding="utf-8"))
            self.assertEqual(cases[0]["metadata"]["error_sources"], ["package_graph_reconstruction"])
            self.assertEqual(cases[0]["metadata"]["objective_error_sources"], ["model_prediction", "package_graph_reconstruction"])
            self.assertEqual(cases[0]["metrics"]["objective_error_sources"], ["model_prediction", "package_graph_reconstruction"])
            self.assertEqual(cases[0]["metadata"]["mismatch_checks"][0]["name"], "lead_count")
            self.assertEqual(cases[0]["media"][0]["label"], "Source top (top)")
            self.assertEqual(cases[0]["media"][1]["label"], "GT reference")
            self.assertEqual(cases[0]["risk_score"], 100.0)
            self.assertEqual(cases[0]["risk_level"], "high")
            self.assertEqual(cases[0]["metrics"]["overall_score"], 0.4)
            self.assertIn("final_graph", cases[0]["links"])
            self.assertEqual(cases[0]["metrics"]["final_coordinate_system"], {"name": "scan_result_reference_aligned_2d"})
            self.assertEqual(cases[0]["metrics"]["alignment_transform"], {"strategy": "default"})
            self.assertEqual(cases[0]["metrics"]["evidence_summary"], {"evidence_ref_count": 1})
            self.assertEqual(cases[0]["metrics"]["conflicts"], [{"type": "dimension_value_conflict"}])
            self.assertEqual(cases[0]["metadata"]["evidence_refs"][0]["raw_view"], "top")
            self.assertEqual(cases[0]["metrics"]["iou"], {"outline": 0.9, "land": None, "lead": 0.5})
            self.assertEqual(cases[0]["metrics"]["layout_score"], 0.25)
            self.assertEqual(cases[0]["metrics"]["pad_count_match"]["land"], False)
            self.assertEqual(cases[0]["metrics"]["pad_count_match"]["lead"], True)
            self.assertEqual(cases[0]["metrics"]["dimension_mismatch"]["count"], 2)
            self.assertEqual(cases[0]["metadata"]["source_image_evidence"][0]["raw_view"], "top")

    def test_source_image_evidence_prefers_adopted_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "FPK_PJ_fullflow"
            canonical = root / "runs" / "run1" / "outputs" / "multiview" / "parts" / "PART" / "unified_multiview_layers.json"
            image = root / "assets" / "datasets" / "dataset_full_v5" / "PART" / "extract_image" / "bottom.png"
            overlay = canonical.parent / "source_overlays" / "bottom.adopted.svg"
            graph_path = str(root / "runs" / "run1" / "outputs" / "reconstruction" / "graphs" / "PART" / "bottom.package_graph.json")
            for path in (canonical, image, overlay):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            canonical.write_text(
                json.dumps(
                    {
                        "evidence_refs": [
                            {
                                "evidence_type": "package_graph",
                                "raw_view": "bottom",
                                "canonical_view": "bottom",
                                "image_path": str(image),
                                "graph_path": graph_path,
                            }
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (overlay.parent / "manifest.json").write_text(
                json.dumps({"overlays": [{"graph_path": graph_path, "path": str(overlay)}]}) + "\n",
                encoding="utf-8",
            )

            sources = source_image_evidence_from_canonical(canonical, root)

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["label"], "Source bottom adopted overlay (bottom)")
        self.assertEqual(sources[0]["path"], str(overlay))
        self.assertEqual(sources[0]["original_image_path"], str(image))

    def test_item_from_row_uses_adopted_overlay_as_adopted_graph_media(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "FPK_PJ_fullflow"
            alignment_root = root / "runs" / "run1" / "outputs" / "eval" / "gt_alignment"
            part_dir = alignment_root / "parts" / "PART"
            canonical = root / "runs" / "run1" / "outputs" / "multiview" / "parts" / "PART" / "unified_multiview_layers.json"
            image = root / "assets" / "datasets" / "dataset_full_v5" / "PART" / "extract_image" / "bottom.png"
            overlay = canonical.parent / "source_overlays" / "bottom.adopted.svg"
            graph_path = str(root / "runs" / "run1" / "outputs" / "reconstruction" / "graphs" / "PART" / "bottom.package_graph.json")
            for path in (part_dir / "alignment.json", canonical, image, overlay, canonical.with_name("unified_multiview_layers.svg")):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            canonical.write_text(
                json.dumps(
                    {
                        "evidence_refs": [
                            {
                                "evidence_type": "package_graph",
                                "raw_view": "bottom",
                                "canonical_view": "bottom",
                                "image_path": str(image),
                                "graph_path": graph_path,
                            }
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (overlay.parent / "manifest.json").write_text(
                json.dumps({"overlays": [{"graph_path": graph_path, "path": str(overlay)}]}) + "\n",
                encoding="utf-8",
            )

            item = item_from_row(
                {
                    "part_number": "PART",
                    "status": "aligned",
                    "reasons": [],
                    "alignment_path": str(part_dir / "alignment.json"),
                    "unified_multiview_layers_path": str(canonical),
                    "alignment_scores": {},
                    "checks": [],
                },
                alignment_root=alignment_root,
                fullflow_root=root,
                case_prefix="final_comparison",
            )

        self.assertEqual(item.media[0].label, "Adopted graph - bottom")
        self.assertTrue(item.media[0].url.endswith("runs/run1/outputs/multiview/parts/PART/source_overlays/bottom.adopted.svg"))
        self.assertNotEqual(item.media[0].url, "/runs/run1/outputs/multiview/parts/PART/unified_multiview_layers.svg")

    def test_main_postprocessed_graph_evidence_lists_only_top_bottom_land_visualizations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "FPK_PJ_fullflow"
            canonical = root / "runs" / "run1" / "outputs" / "multiview" / "parts" / "PART" / "unified_multiview_layers.json"
            graph_root = root / "runs" / "run1" / "outputs" / "reconstruction" / "run1" / "graphs" / "PART"
            visualization_root = root / "runs" / "run1" / "outputs" / "visualization" / "reconstruction_run1" / "PART"
            for view in ("Top", "Bottom", "Land", "Front"):
                graph_path = graph_root / f"part_{view}.package_graph.json"
                graph_path.parent.mkdir(parents=True, exist_ok=True)
                graph_path.write_text("{}\n", encoding="utf-8")
                png_path = visualization_root / f"part_{view}.package_graph.png"
                png_path.parent.mkdir(parents=True, exist_ok=True)
                png_path.write_text("png\n", encoding="utf-8")
            canonical.parent.mkdir(parents=True, exist_ok=True)
            canonical.write_text(
                json.dumps(
                    {
                        "evidence_refs": [
                            {
                                "evidence_type": "package_graph",
                                "raw_view": "top",
                                "graph_path": str(graph_root / "part_Top.package_graph.json"),
                            },
                            {
                                "evidence_type": "package_graph",
                                "raw_view": "bottom",
                                "graph_path": str(graph_root / "part_Bottom.package_graph.json"),
                            },
                            {
                                "evidence_type": "package_graph",
                                "raw_view": "land",
                                "graph_path": str(graph_root / "part_Land.package_graph.json"),
                            },
                            {
                                "evidence_type": "package_graph",
                                "raw_view": "front",
                                "graph_path": str(graph_root / "part_Front.package_graph.json"),
                            },
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            sources = main_postprocessed_graph_evidence_from_canonical(canonical, root)

        self.assertEqual(
            [item["label"] for item in sources],
            [
                "Postprocessed top package graph",
                "Postprocessed bottom package graph",
                "Postprocessed land package graph",
            ],
        )
        self.assertTrue(all("part_Front" not in item["path"] for item in sources))
        self.assertEqual(
            package_graph_visualization_path(graph_root / "part_Top.package_graph.json"),
            visualization_root / "part_Top.package_graph.png",
        )

    def test_main_view_overlay_stacks_top_bottom_land_package_graphs_by_color(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "FPK_PJ_fullflow"
            canonical = root / "runs" / "run1" / "outputs" / "multiview" / "parts" / "PART" / "unified_multiview_layers.json"
            graph_root = root / "runs" / "run1" / "outputs" / "reconstruction" / "run1" / "graphs" / "PART"
            refs = []
            for index, view in enumerate(("top", "top", "bottom", "land", "front", "side")):
                graph_path = graph_root / f"part_{view}_{index}.package_graph.json"
                graph_path.parent.mkdir(parents=True, exist_ok=True)
                outline_width = 20.0 if view == "bottom" else 10.0
                object_source_label = "lead" if view in {"front", "side"} else "pad"
                dimension_value = 0.01 if view == "front" else 0.25 if view == "side" else 1.0
                dimension_anchors = ["left_edge", "right_edge"]
                graph_path.write_text(
                    json.dumps(
                        {
                            "objects": [
                                {
                                    "id": 0,
                                    "label": "outline",
                                    "source_label": "outline",
                                    "bbox_reconstructed": [0.0, 0.0, outline_width, 6.0],
                                },
                                {
                                    "id": 1,
                                    "label": "rect",
                                    "source_label": object_source_label,
                                    "bbox_reconstructed": [1.0 + index, 1.0, 2.0 + index, 2.0],
                                },
                                {
                                    "id": 2,
                                    "label": "rect",
                                    "source_label": object_source_label,
                                    "bbox_reconstructed": [9.0, 0.2, 11.0, 1.0],
                                },
                            ],
                            "dimensions": [
                                {
                                    "id": 0,
                                    "dimension_id": 0,
                                    "text": "1.0",
                                    "kind": "size",
                                    "axis": "x",
                                    "value": dimension_value,
                                    "status": "accepted",
                                    "target_ids": [1 if view in {"front", "side"} else 0],
                                    "anchors": dimension_anchors,
                                }
                            ],
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                refs.append(
                    {
                        "evidence_type": "package_graph",
                        "raw_view": view,
                        "graph_path": str(graph_path),
                    }
                )
            canonical.parent.mkdir(parents=True, exist_ok=True)
            bottom_graph_path = graph_root / "part_bottom_2.package_graph.json"
            canonical.write_text(
                json.dumps(
                    {
                        "evidence_refs": refs,
                        "lead_pads": [
                            {
                                "role": "partial_pad_width",
                                "label": "partial_pad_width",
                                "raw_view": "front",
                                "canonical_view": "lateral",
                                "bbox": [1.2, 1.0, 1.8, 2.0],
                                "source_type": "derived_partial_evidence_multiview",
                                "source_graph": str(bottom_graph_path),
                                "source_package_pad_bbox": [1.0, 1.0, 2.0, 2.0],
                                "projection_axis": "x",
                                "partial_dimension_semantics": "pad_width",
                            },
                            {
                                "role": "lead_pad",
                                "label": "lead_pad",
                                "raw_view": "side",
                                "canonical_view": "lateral",
                                "bbox": [9.0, 0.2, 11.0, 0.7],
                                "source_type": "derived_partial_evidence_multiview",
                                "source_graph": str(bottom_graph_path),
                                "source_package_pad_bbox": [9.0, 0.2, 11.0, 1.0],
                                "projection_axis": "y",
                                "partial_dimension_semantics": "lead_ground_contact_length",
                            },
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            canonical.with_name("final_graph.json").write_text(
                json.dumps(
                    {
                        "gt_reference": {
                            "objects": [
                                {
                                    "role": "land",
                                    "bbox": [0.0, 0.0, 1.0, 1.0],
                                },
                                {
                                    "role": "shape",
                                    "bbox": [0.0, 0.0, 100.0, 100.0],
                                }
                            ]
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            display_context = display_scale_context_from_canonical(canonical)
            overlay = main_view_overlay_evidence_from_canonical(canonical, root, display_context=display_context)
            multiview_overlay = multiview_overlay_evidence_from_canonical(canonical, root, display_context=display_context)
            gt_display = scan_result_display_evidence_from_canonical(canonical, root, display_context=display_context)

            self.assertIsNotNone(overlay)
            self.assertIsNotNone(multiview_overlay)
            self.assertIsNotNone(gt_display)
            self.assertEqual(overlay["label"], "Main-view overlay")
            self.assertEqual(multiview_overlay["label"], "Multi-view overlay")
            svg = Path(overlay["path"]).read_text(encoding="utf-8")
            multiview_svg = Path(multiview_overlay["path"]).read_text(encoding="utf-8")
            gt_svg = Path(gt_display["path"]).read_text(encoding="utf-8")
            self.assertIn("#2563eb", svg)
            self.assertIn("#16a34a", svg)
            self.assertIn("#f97316", svg)
            self.assertGreater(svg.count("#2563eb"), svg.count("#16a34a"))
            self.assertNotIn("front", svg)
            self.assertNotIn("ScanResult fallback", svg)
            self.assertNotIn("scan_result", svg)
            self.assertIn("front partial evidence", multiview_svg)
            self.assertIn("side partial evidence", multiview_svg)
            self.assertIn('data-role="lead_pad"', multiview_svg)
            self.assertIn('data-role="partial_pad_width"', multiview_svg)
            self.assertIn('data-source-view="front"', multiview_svg)
            self.assertIn('data-source-view="side"', multiview_svg)
            self.assertIn('data-canonical-view="lateral"', multiview_svg)
            self.assertIn('data-projection-axis="x"', multiview_svg)
            self.assertIn('data-projection-axis="y"', multiview_svg)
            self.assertIn('data-partial-dimension-semantics="pad_width"', multiview_svg)
            self.assertIn('data-partial-dimension-semantics="lead_ground_contact_length"', multiview_svg)
            self.assertNotIn("ScanResult fallback", multiview_svg)
            self.assertNotIn('data-source-view="scan_result"', multiview_svg)
            self.assertIn("dimension-calibrated scale", svg)
            self.assertIn(f'width="{int(DISPLAY_SVG_WIDTH)}" height="{int(DISPLAY_SVG_HEIGHT)}"', svg)
            self.assertIn(f'width="{int(DISPLAY_SVG_WIDTH)}" height="{int(DISPLAY_SVG_HEIGHT)}"', gt_svg)
            self.assertIn('data-scale-source="accepted_dimensions"', svg)
            overlay_scale = re.search(r'data-display-scale="([0-9.]+)"', svg)
            gt_scale = re.search(r'data-display-scale="([0-9.]+)"', gt_svg)
            self.assertIsNotNone(overlay_scale)
            self.assertIsNotNone(gt_scale)
            self.assertEqual(overlay_scale.group(1), gt_scale.group(1))
            self.assertGreater(float(overlay_scale.group(1)), 100.0)
            top_group = re.search(r'<g data-view="top"[^>]*>(.*?)</g>', svg)
            bottom_group = re.search(r'<g data-view="bottom"[^>]*>(.*?)</g>', svg)
            self.assertIsNotNone(top_group)
            self.assertIsNotNone(bottom_group)
            top_outline = re.search(r'<rect [^>]*width="([0-9.]+)"[^>]*fill="none"[^>]*stroke="#2563eb"', top_group.group(1))
            bottom_outline = re.search(r'<rect [^>]*width="([0-9.]+)"[^>]*fill="none"[^>]*stroke="#16a34a"', bottom_group.group(1))
            gt_box = re.search(r'<rect [^>]*width="([0-9.]+)"[^>]*fill="#15803d"', gt_svg)
            self.assertIsNotNone(top_outline)
            self.assertIsNotNone(bottom_outline)
            self.assertIsNotNone(gt_box)
            self.assertAlmostEqual(float(bottom_outline.group(1)), float(top_outline.group(1)), places=3)
            self.assertAlmostEqual(float(gt_box.group(1)), float(top_outline.group(1)), places=3)

    def test_multiview_overlay_uses_materialized_normalized_coordinates_without_graph_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "FPK_PJ_fullflow"
            canonical = root / "runs" / "run1" / "outputs" / "multiview" / "parts" / "PART" / "unified_multiview_layers.json"
            canonical.parent.mkdir(parents=True, exist_ok=True)
            canonical.write_text(
                json.dumps(
                    {
                        "evidence_refs": [],
                        "multiview_overlay": {
                            "coordinate_mode": "dimension_scaled_centered",
                            "frame": [-2.0, -1.0, 2.0, 1.0],
                            "layers": [
                                {
                                    "raw_view": "bottom",
                                    "canonical_view": "bottom",
                                    "graph_path": "/tmp/PART/bottom.package_graph.json",
                                    "coordinate_mode": "dimension_scaled_centered",
                                    "source_frame": [0.0, 0.0, 8.0, 4.0],
                                    "normalized_frame": [-2.0, -1.0, 2.0, 1.0],
                                    "unit_scales": {"x": 0.5, "y": 0.5, "source": "accepted_dimensions"},
                                    "objects": [
                                        {
                                            "role": "package_pad",
                                            "label": "rect",
                                            "source_label": "rect",
                                            "bbox": [-2.0, -1.0, -1.0, 1.0],
                                            "coordinate_mode": "dimension_scaled_centered",
                                        }
                                    ],
                                }
                            ],
                            "extra_objects": [
                                {
                                    "role": "lead_pad",
                                    "label": "lead_pad",
                                    "raw_view": "side",
                                    "canonical_view": "lateral",
                                    "bbox": [-2.0, -0.5, -1.5, 0.5],
                                    "coordinate_mode": "dimension_scaled_centered",
                                    "source_graph": "/tmp/PART/bottom.package_graph.json",
                                    "projection_axis": "x",
                                    "partial_dimension_semantics": "lead_ground_contact_length",
                                }
                            ],
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            display_context = display_scale_context_from_canonical(canonical)
            overlay = main_view_overlay_evidence_from_canonical(canonical, root, display_context=display_context)
            multiview_overlay = multiview_overlay_evidence_from_canonical(canonical, root, display_context=display_context)

            self.assertIsNotNone(overlay)
            self.assertIsNotNone(multiview_overlay)
            main_svg = Path(overlay["path"]).read_text(encoding="utf-8")
            multiview_svg = Path(multiview_overlay["path"]).read_text(encoding="utf-8")
            self.assertIn('data-coordinate-mode="dimension_scaled_centered"', main_svg)
            self.assertIn('data-display-only="true"', main_svg)
            self.assertIn('data-scale-source="accepted_dimensions"', main_svg)
            self.assertIn('data-role="lead_pad"', multiview_svg)
            self.assertIn('data-coordinate-mode="dimension_scaled_centered"', multiview_svg)
            self.assertIn('data-source-view="side"', multiview_svg)
            self.assertNotIn("dimension-calibrated scale", main_svg)
            self.assertEqual(display_context["scale_source"], "multiview_overlay_normalized")

    def test_bottom_main_view_overlay_uses_regularized_two_column_pad_x_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "FPK_PJ_fullflow"
            canonical = root / "runs" / "run1" / "outputs" / "multiview" / "parts" / "PART" / "unified_multiview_layers.json"
            graph_path = root / "runs" / "run1" / "outputs" / "reconstruction" / "run1" / "graphs" / "PART" / "bottom.package_graph.json"
            graph_path.parent.mkdir(parents=True, exist_ok=True)
            graph_path.write_text(
                json.dumps(
                    {
                        "objects": [
                            {"id": 0, "label": "outline", "source_label": "outline", "bbox_reconstructed": [0.0, 0.0, 10.0, 10.0]},
                            {"id": 1, "label": "rect", "source_label": "pad", "bbox_reconstructed": [0.0, 1.0, 2.0, 2.0]},
                            {"id": 2, "label": "rect", "source_label": "pad", "bbox_reconstructed": [8.0, 1.0, 10.0, 2.0]},
                            {"id": 3, "label": "rect", "source_label": "pad", "bbox_reconstructed": [0.0, 4.0, 2.0, 5.0]},
                            {"id": 4, "label": "rect", "source_label": "pad", "bbox_reconstructed": [8.0, 4.0, 10.0, 5.0]},
                            {"id": 5, "label": "rect", "source_label": "pad", "bbox_reconstructed": [0.0, 7.0, 2.0, 8.0]},
                            {"id": 6, "label": "rect", "source_label": "pad", "bbox_reconstructed": [8.0, 7.0, 10.0, 8.0]},
                        ],
                        "dimensions": [
                            {
                                "id": 10,
                                "kind": "size",
                                "axis": "x",
                                "value": 1.0,
                                "status": "accepted",
                                "target_ids": [1],
                                "anchors": ["left_edge", "right_edge"],
                            },
                            {
                                "id": 11,
                                "kind": "distance",
                                "axis": "x",
                                "value": 1.0,
                                "status": "accepted",
                                "target_ids": [0, 1],
                                "anchors": ["left_edge", "left_edge"],
                            },
                            {
                                "id": 12,
                                "kind": "distance",
                                "axis": "x",
                                "value": 1.0,
                                "status": "accepted",
                                "target_ids": [2, 0],
                                "anchors": ["right_edge", "right_edge"],
                            },
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            canonical.parent.mkdir(parents=True, exist_ok=True)
            canonical.write_text(
                json.dumps(
                    {
                        "evidence_refs": [
                            {
                                "evidence_type": "package_graph",
                                "raw_view": "bottom",
                                "graph_path": str(graph_path),
                            }
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            items = main_view_graph_items_from_canonical(canonical)

        objects = {obj["id"]: obj for obj in items[0]["graph"]["objects"]}
        left_x_ranges = [objects[obj_id]["bbox_reconstructed"][0:3:2] for obj_id in (1, 3, 5)]
        right_x_ranges = [objects[obj_id]["bbox_reconstructed"][0:3:2] for obj_id in (2, 4, 6)]
        self.assertEqual(left_x_ranges, [[2.0, 4.0], [2.0, 4.0], [2.0, 4.0]])
        self.assertEqual(right_x_ranges, [[6.0, 8.0], [6.0, 8.0], [6.0, 8.0]])
        self.assertEqual(objects[1]["bbox_before_overlay_regularization"], [0.0, 1.0, 2.0, 2.0])
        self.assertEqual(objects[1]["overlay_geometry_adjusted_reason"], "dimension_regularized_package_pad_x_grid")

    def test_land_main_view_overlay_regularizes_two_column_pad_x_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "FPK_PJ_fullflow"
            canonical = root / "runs" / "run1" / "outputs" / "multiview" / "parts" / "PART" / "unified_multiview_layers.json"
            graph_path = root / "runs" / "run1" / "outputs" / "reconstruction" / "run1" / "graphs" / "PART" / "land.package_graph.json"
            graph_path.parent.mkdir(parents=True, exist_ok=True)
            graph_path.write_text(
                json.dumps(
                    {
                        "objects": [
                            {"id": 1, "label": "rect", "source_label": "pad", "bbox_reconstructed": [0.0, 1.0, 2.0, 2.0]},
                            {"id": 2, "label": "rect", "source_label": "pad", "bbox_reconstructed": [8.0, 1.0, 10.0, 2.0]},
                            {"id": 3, "label": "rect", "source_label": "pad", "bbox_reconstructed": [0.3, 4.0, 2.3, 5.0]},
                            {"id": 4, "label": "rect", "source_label": "pad", "bbox_reconstructed": [7.7, 4.0, 9.7, 5.0]},
                            {"id": 5, "label": "rect", "source_label": "pad", "bbox_reconstructed": [0.1, 7.0, 2.1, 8.0]},
                            {"id": 6, "label": "rect", "source_label": "pad", "bbox_reconstructed": [7.9, 7.0, 9.9, 8.0]},
                        ],
                        "dimensions": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            canonical.parent.mkdir(parents=True, exist_ok=True)
            canonical.write_text(
                json.dumps(
                    {
                        "evidence_refs": [
                            {
                                "evidence_type": "package_graph",
                                "raw_view": "land",
                                "graph_path": str(graph_path),
                            }
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            items = main_view_graph_items_from_canonical(canonical)

        objects = {obj["id"]: obj for obj in items[0]["graph"]["objects"]}
        left_x_ranges = [objects[obj_id]["bbox_reconstructed"][0:3:2] for obj_id in (1, 3, 5)]
        right_x_ranges = [objects[obj_id]["bbox_reconstructed"][0:3:2] for obj_id in (2, 4, 6)]
        self.assertEqual(left_x_ranges, [[0.1, 2.1], [0.1, 2.1], [0.1, 2.1]])
        self.assertEqual(right_x_ranges, [[7.9, 9.9], [7.9, 9.9], [7.9, 9.9]])
        self.assertEqual(objects[1]["bbox_before_overlay_regularization"], [0.0, 1.0, 2.0, 2.0])
        self.assertEqual(objects[1]["overlay_geometry_adjusted_reason"], "display_regularized_two_column_land_pad_x_grid")

    def test_front_partial_width_larger_than_main_pad_becomes_outer_aligned_contact_length(self) -> None:
        bbox = (10.0, 2.0, 12.0, 4.0)

        semantics = effective_partial_dimension_semantics(
            dim={"raw_view": "front"},
            base_semantics="pad_width",
            projection_axis="x",
            dimension_value=0.30,
            bbox=bbox,
            unit_scale=0.10,
        )

        self.assertEqual(semantics, "lead_ground_contact_length")
        lead_bbox = partial_dimension_overlay_bbox(
            bbox=bbox,
            package_center=(20.0, 3.0),
            projection_axis="x",
            length=3.0,
            semantics=semantics,
        )
        self.assertEqual(lead_bbox, [10.0, 2.0, 13.0, 4.0])

    def test_front_partial_width_within_main_pad_stays_centered_pad_width(self) -> None:
        bbox = (10.0, 2.0, 12.0, 4.0)

        semantics = effective_partial_dimension_semantics(
            dim={"raw_view": "front"},
            base_semantics="pad_width",
            projection_axis="x",
            dimension_value=0.15,
            bbox=bbox,
            unit_scale=0.10,
        )

        self.assertEqual(semantics, "pad_width")

    def test_scan_result_gt_display_frame_includes_body_shape_not_only_pads(self) -> None:
        frame = scan_result_gt_display_frame(
            [
                {"role": "land", "bbox": [10.0, 1.0, 11.0, 2.0]},
                {"role": "lead", "bbox": [10.0, 3.0, 11.0, 4.0]},
                {"role": "shape", "bbox": [0.0, 0.0, 12.0, 5.0]},
            ]
        )

        self.assertEqual(frame, (0.0, 0.0, 12.0, 5.0))

    def test_side_partial_contact_uses_vertical_centerline_for_left_right_outline_pads(self) -> None:
        projection_axis = overlay_projection_axis(
            {"raw_view": "side", "overlay_semantics": "lead_ground_contact_length"},
            radial_axis="x",
        )

        self.assertEqual(projection_axis, "y")
        self.assertEqual(
            partial_dimension_overlay_bbox(
                bbox=(1.0, 2.0, 3.0, 4.0),
                package_center=(10.0, 3.0),
                projection_axis=projection_axis,
                length=5.0,
                semantics="lead_ground_contact_length",
            ),
            [1.0, 0.5, 3.0, 5.5],
        )

    def test_side_partial_contact_aligns_bottom_edge_for_bottom_outline_pads(self) -> None:
        projection_axis = overlay_projection_axis(
            {"raw_view": "side", "overlay_semantics": "lead_ground_contact_length"},
            radial_axis="x",
        )

        self.assertEqual(projection_axis, "y")
        self.assertEqual(
            partial_dimension_overlay_bbox(
                bbox=(13.0, 6.0, 15.0, 8.0),
                package_center=(5.0, 4.0),
                package_frame=(0.0, 2.0, 10.0, 6.0),
                projection_axis=projection_axis,
                length=3.0,
                semantics="lead_ground_contact_length",
            ),
            [13.0, 5.0, 15.0, 8.0],
        )

    def test_display_scale_uses_part_size_without_one_unit_floor(self) -> None:
        scale = overlay_common_scale([(0.0, 0.0, 0.167, 0.048)], DISPLAY_SVG_TARGET)
        target_width = DISPLAY_SVG_TARGET[2] - DISPLAY_SVG_TARGET[0]

        self.assertAlmostEqual(0.167 * scale, target_width, places=6)
        self.assertGreater(scale, 1000.0)

    def test_build_final_comparison_review_includes_aligned_and_mismatch_parts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "FPK_PJ_fullflow"
            alignment_root = root / "runs" / "run1" / "outputs" / "eval" / "gt_alignment"
            rows = []
            for part, status, score, reasons in (
                ("PART_BAD", "mismatch", 0.4, ["lead_count_mismatch"]),
                ("PART_OK", "aligned", 0.95, []),
            ):
                part_dir = alignment_root / "parts" / part
                canonical = root / "runs" / "run1" / "outputs" / "multiview" / "parts" / part / "unified_multiview_layers.json"
                source_image = root / "assets" / "datasets" / "dataset_full_v5" / part / "extract_image" / "source.png"
                scan = root / "assets" / "datasets" / "dataset_full_v5" / part / "ScanResultFormat.txt"
                for path in (
                    part_dir / "alignment.json",
                    part_dir / "scan_result.svg",
                    canonical.with_name("gt_reference.svg"),
                    canonical.with_name("default_aligned_result.svg"),
                    canonical.with_name("default_comparison.svg"),
                    canonical.with_name("aligned_result.svg"),
                    canonical.with_name("comparison.svg"),
                    canonical.with_name("unified_multiview_layers.svg"),
                    canonical.with_name("final_graph.json"),
                    scan,
                    source_image,
                ):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("{}\n", encoding="utf-8")
                canonical.write_text(
                    json.dumps(
                        {
                            "evidence_refs": [
                                {
                                    "evidence_type": "package_graph",
                                    "raw_view": "bottom",
                                    "canonical_view": "bottom",
                                    "image_path": str(source_image),
                                }
                            ]
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                rows.append(
                    {
                        "part_number": part,
                        "status": status,
                        "reasons": reasons,
                        "objective_error_sources": ["package_graph_reconstruction"] if reasons else [],
                        "stage_hints": ["package_graph_package_pad_reconstruction_count_mismatch"] if reasons else [],
                        "review_bucket": "upstream_prediction_or_reconstruction" if reasons else "low_risk",
                        "review_risk_level": "high" if reasons else "low",
                        "alignment_path": str(part_dir / "alignment.json"),
                        "scan_result_svg_path": str(part_dir / "scan_result.svg"),
                        "scan_result_path": str(scan),
                        "unified_multiview_layers_path": str(canonical),
                        "final_graph_path": str(canonical.with_name("final_graph.json")),
                        "alignment_scores": {"overall_score": score, "outline_iou": 1.0, "pad_layout_score": score},
                        "checks": [
                            {
                                "name": "lead_count",
                                "status": "mismatch",
                                "reason": "lead_count_mismatch",
                                "stage_hint": "package_graph_package_pad_reconstruction_count_mismatch",
                            }
                        ]
                        if reasons
                        else [],
                    }
                )
            alignment_root.mkdir(parents=True, exist_ok=True)
            (alignment_root / "summary.json").write_text(
                json.dumps({"parts": rows}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            output_root = root / "runs" / "run1" / "outputs" / "review" / "final_comparison"
            stale_page = output_root / "by_reason" / "stale.html"
            stale_page.parent.mkdir(parents=True, exist_ok=True)
            stale_page.write_text("stale\n", encoding="utf-8")
            result = build_final_comparison_review(
                alignment_root=alignment_root,
                output_root=output_root,
                fullflow_root=root,
                run_id="run1",
            )

            self.assertEqual(result["total_items"], 2)
            self.assertEqual(result["gallery_url"], "/runs/run1/outputs/review/final_comparison/index.html")
            self.assertTrue((output_root / "index.html").exists())
            self.assertFalse(stale_page.exists())
            self.assertTrue((output_root / "by_risk" / "high.html").exists())
            self.assertTrue((output_root / "by_risk" / "low.html").exists())
            self.assertTrue((output_root / "by_status" / "aligned.html").exists())
            self.assertTrue((output_root / "by_status" / "mismatch.html").exists())
            self.assertTrue((output_root / "by_review_bucket" / "upstream_prediction_or_reconstruction.html").exists())
            self.assertTrue((output_root / "by_review_bucket" / "low_risk.html").exists())
            cases = json.loads((output_root / "data" / "cases.json").read_text(encoding="utf-8"))
            notes = json.loads((output_root / "data" / "notes.json").read_text(encoding="utf-8"))
            self.assertEqual(notes["gallery_id"], "final_comparison")
            self.assertEqual([case["case_id"] for case in cases], ["final_comparison:PART_BAD", "final_comparison:PART_OK"])
            self.assertEqual(cases[0]["metadata"]["review_bucket"], "upstream_prediction_or_reconstruction")
            self.assertEqual(cases[0]["metrics"]["review_bucket"], "upstream_prediction_or_reconstruction")
            self.assertEqual(cases[0]["media"][0]["label"], "Source bottom (bottom)")
            self.assertEqual(cases[0]["media"][1]["label"], "GT reference")
            self.assertEqual(cases[1]["risk_level"], "low")
            self.assertEqual(cases[1]["risk_reasons"], ["aligned"])

    def test_final_comparison_review_uses_score_diagnostics_for_aligned_medium_parts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "FPK_PJ_fullflow"
            alignment_root = root / "runs" / "run1" / "outputs" / "eval" / "gt_alignment"
            part_dir = alignment_root / "parts" / "PART_MEDIUM"
            canonical = root / "runs" / "run1" / "outputs" / "multiview" / "parts" / "PART_MEDIUM" / "unified_multiview_layers.json"
            scan = root / "assets" / "datasets" / "dataset_full_v5" / "PART_MEDIUM" / "ScanResultFormat.txt"
            for path in (
                part_dir / "alignment.json",
                part_dir / "scan_result.svg",
                canonical.with_name("gt_reference.svg"),
                canonical.with_name("default_aligned_result.svg"),
                canonical.with_name("default_comparison.svg"),
                canonical.with_name("aligned_result.svg"),
                canonical.with_name("comparison.svg"),
                canonical.with_name("unified_multiview_layers.svg"),
                canonical.with_name("final_graph.json"),
                scan,
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            canonical.write_text(json.dumps({"evidence_refs": []}) + "\n", encoding="utf-8")
            row = {
                "part_number": "PART_MEDIUM",
                "status": "aligned",
                "reasons": [],
                "alignment_path": str(part_dir / "alignment.json"),
                "scan_result_svg_path": str(part_dir / "scan_result.svg"),
                "scan_result_path": str(scan),
                "unified_multiview_layers_path": str(canonical),
                "final_graph_path": str(canonical.with_name("final_graph.json")),
                "alignment_scores": {"overall_score": 0.7, "lead_pad_iou_score": 0.0},
                "score_diagnostics": ["low_lead_pad_iou"],
                "score_stage_hints": ["low_score_package_graph_package_pad_geometry"],
                "score_error_sources": ["package_graph_reconstruction"],
                "score_objective_error_sources": ["model_prediction", "package_graph_reconstruction"],
                "score_diagnostic_details": [
                    {
                        "reason": "low_lead_pad_iou",
                        "metric": "lead_pad_iou_score",
                        "value": 0.0,
                        "threshold": 0.5,
                        "stage_hint": "low_score_package_graph_package_pad_geometry",
                    }
                ],
                "checks": [],
            }
            alignment_root.mkdir(parents=True, exist_ok=True)
            (alignment_root / "summary.json").write_text(
                json.dumps({"parts": [row]}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            output_root = root / "runs" / "run1" / "outputs" / "review" / "final_comparison"
            result = build_final_comparison_review(
                alignment_root=alignment_root,
                output_root=output_root,
                fullflow_root=root,
                run_id="run1",
            )

            self.assertEqual(result["total_items"], 1)
            self.assertTrue((output_root / "by_risk" / "high.html").exists())
            self.assertTrue((output_root / "by_reason" / "low_lead_pad_iou.html").exists())
            self.assertTrue((output_root / "by_stage_hint" / "low_score_package_graph_package_pad_geometry.html").exists())
            self.assertTrue((output_root / "by_objective_source" / "model_prediction.html").exists())
            cases = json.loads((output_root / "data" / "cases.json").read_text(encoding="utf-8"))
            self.assertEqual(cases[0]["risk_reasons"], ["low_lead_pad_iou"])
            self.assertEqual(cases[0]["metadata"]["stage_hints"], ["low_score_package_graph_package_pad_geometry"])
            self.assertEqual(
                cases[0]["metrics"]["objective_error_sources"],
                ["model_prediction", "package_graph_reconstruction"],
            )
            self.assertEqual(cases[0]["metrics"]["score_diagnostics"], ["low_lead_pad_iou"])
            media_labels = [item["label"] for item in cases[0]["media"]]
            self.assertIn("GT reference", media_labels)
            self.assertNotIn("Canonical graph", media_labels)
            self.assertNotIn("Canonical over ScanResult GT", media_labels)
            self.assertNotIn("ScanResult GT", media_labels)
            self.assertNotIn("Default aligned result", media_labels)
            self.assertNotIn("Selected aligned result", media_labels)
            self.assertNotIn("GT vs selected result", media_labels)

    def test_final_comparison_review_falls_back_to_dataset_source_images_when_canonical_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "FPK_PJ_fullflow"
            alignment_root = root / "runs" / "run1" / "outputs" / "eval" / "gt_alignment"
            dataset_part = root / "assets" / "datasets" / "dataset_full_v5" / "PART_MISSING"
            source_image = dataset_part / "extract_image" / "123-page1_Top_0.png"
            source_image.parent.mkdir(parents=True, exist_ok=True)
            source_image.write_text("image\n", encoding="utf-8")
            part_dir = alignment_root / "parts" / "PART_MISSING"
            part_dir.mkdir(parents=True, exist_ok=True)
            row = {
                "part_number": "PART_MISSING",
                "status": "missing_canonical",
                "reasons": ["no_package_graph_for_part"],
                "dataset_part_dir": str(dataset_part),
                "alignment_path": str(part_dir / "alignment.json"),
                "unified_multiview_layers_path": str(root / "runs" / "run1" / "outputs" / "multiview" / "parts" / "PART_MISSING" / "unified_multiview_layers.json"),
                "alignment_scores": {},
                "checks": [],
            }
            alignment_root.mkdir(parents=True, exist_ok=True)
            (alignment_root / "summary.json").write_text(
                json.dumps({"parts": [row]}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            output_root = root / "runs" / "run1" / "outputs" / "review" / "final_comparison"
            build_final_comparison_review(
                alignment_root=alignment_root,
                output_root=output_root,
                fullflow_root=root,
                run_id="run1",
            )

            cases = json.loads((output_root / "data" / "cases.json").read_text(encoding="utf-8"))
            self.assertEqual(cases[0]["media"][0]["label"], "Source top (dataset)")
            self.assertEqual(cases[0]["metadata"]["source_image_evidence"][0]["raw_view"], "top")

    def test_final_comparison_review_uses_source_placeholder_when_no_source_image_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "FPK_PJ_fullflow"
            alignment_root = root / "runs" / "run1" / "outputs" / "eval" / "gt_alignment"
            dataset_part = root / "assets" / "datasets" / "dataset_full_v5" / "PART_NO_SOURCE"
            (dataset_part / "extract_image").mkdir(parents=True, exist_ok=True)
            part_dir = alignment_root / "parts" / "PART_NO_SOURCE"
            part_dir.mkdir(parents=True, exist_ok=True)
            row = {
                "part_number": "PART_NO_SOURCE",
                "status": "missing_canonical",
                "reasons": ["no_package_graph_for_part"],
                "dataset_part_dir": str(dataset_part),
                "alignment_path": str(part_dir / "alignment.json"),
                "unified_multiview_layers_path": str(root / "runs" / "run1" / "outputs" / "multiview" / "parts" / "PART_NO_SOURCE" / "unified_multiview_layers.json"),
                "alignment_scores": {},
                "checks": [],
            }
            alignment_root.mkdir(parents=True, exist_ok=True)
            (alignment_root / "summary.json").write_text(
                json.dumps({"parts": [row]}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            output_root = root / "runs" / "run1" / "outputs" / "review" / "final_comparison"
            build_final_comparison_review(
                alignment_root=alignment_root,
                output_root=output_root,
                fullflow_root=root,
                run_id="run1",
            )

            cases = json.loads((output_root / "data" / "cases.json").read_text(encoding="utf-8"))
            self.assertEqual(cases[0]["media"][0]["label"], "Source unavailable (dataset)")
            self.assertTrue(cases[0]["media"][0]["url"].startswith("data:image/svg+xml"))
            self.assertEqual(cases[0]["metadata"]["source_image_evidence"][0]["raw_view"], "unavailable")

    def test_final_comparison_review_promotes_low_score_with_diagnostics_to_medium(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "FPK_PJ_fullflow"
            alignment_root = root / "runs" / "run1" / "outputs" / "eval" / "gt_alignment"
            part_dir = alignment_root / "parts" / "PART_WARNING"
            canonical = root / "runs" / "run1" / "outputs" / "multiview" / "parts" / "PART_WARNING" / "unified_multiview_layers.json"
            scan = root / "assets" / "datasets" / "dataset_full_v5" / "PART_WARNING" / "ScanResultFormat.txt"
            for path in (
                part_dir / "alignment.json",
                part_dir / "scan_result.svg",
                canonical.with_name("gt_reference.svg"),
                canonical.with_name("default_aligned_result.svg"),
                canonical.with_name("default_comparison.svg"),
                canonical.with_name("aligned_result.svg"),
                canonical.with_name("comparison.svg"),
                canonical.with_name("unified_multiview_layers.svg"),
                canonical.with_name("final_graph.json"),
                scan,
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            canonical.write_text(json.dumps({"evidence_refs": []}) + "\n", encoding="utf-8")
            row = {
                "part_number": "PART_WARNING",
                "status": "aligned",
                "reasons": [],
                "alignment_path": str(part_dir / "alignment.json"),
                "scan_result_svg_path": str(part_dir / "scan_result.svg"),
                "scan_result_path": str(scan),
                "unified_multiview_layers_path": str(canonical),
                "final_graph_path": str(canonical.with_name("final_graph.json")),
                "alignment_scores": {"review_quality_score": 0.85, "quality_score": 0.9},
                "score_diagnostics": ["borderline_lead_pad_iou"],
                "checks": [],
            }
            alignment_root.mkdir(parents=True, exist_ok=True)
            (alignment_root / "summary.json").write_text(
                json.dumps({"parts": [row]}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            output_root = root / "runs" / "run1" / "outputs" / "review" / "final_comparison"
            build_final_comparison_review(
                alignment_root=alignment_root,
                output_root=output_root,
                fullflow_root=root,
                run_id="run1",
            )

            cases = json.loads((output_root / "data" / "cases.json").read_text(encoding="utf-8"))
            self.assertEqual(cases[0]["risk_level"], "medium")
            self.assertEqual(cases[0]["risk_reasons"], ["borderline_lead_pad_iou"])
            self.assertTrue((output_root / "by_risk" / "medium.html").exists())

    def test_final_comparison_review_uses_summary_review_risk_level(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "FPK_PJ_fullflow"
            alignment_root = root / "runs" / "run1" / "outputs" / "eval" / "gt_alignment"
            part_dir = alignment_root / "parts" / "PART_REVIEW_LOW"
            canonical = root / "runs" / "run1" / "outputs" / "multiview" / "parts" / "PART_REVIEW_LOW" / "unified_multiview_layers.json"
            scan = root / "assets" / "datasets" / "dataset_full_v5" / "PART_REVIEW_LOW" / "ScanResultFormat.txt"
            for path in (
                part_dir / "alignment.json",
                part_dir / "scan_result.svg",
                canonical.with_name("gt_reference.svg"),
                canonical.with_name("default_aligned_result.svg"),
                canonical.with_name("default_comparison.svg"),
                canonical.with_name("aligned_result.svg"),
                canonical.with_name("comparison.svg"),
                canonical.with_name("unified_multiview_layers.svg"),
                canonical.with_name("final_graph.json"),
                scan,
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            canonical.write_text(json.dumps({"evidence_refs": []}) + "\n", encoding="utf-8")
            row = {
                "part_number": "PART_REVIEW_LOW",
                "status": "aligned",
                "reasons": [],
                "review_bucket": "low_risk",
                "review_risk_level": "low",
                "alignment_path": str(part_dir / "alignment.json"),
                "scan_result_svg_path": str(part_dir / "scan_result.svg"),
                "scan_result_path": str(scan),
                "unified_multiview_layers_path": str(canonical),
                "final_graph_path": str(canonical.with_name("final_graph.json")),
                "alignment_scores": {"review_quality_score": 0.85, "quality_score": 0.9},
                "score_diagnostics": ["borderline_lead_pad_iou"],
                "checks": [],
            }
            alignment_root.mkdir(parents=True, exist_ok=True)
            (alignment_root / "summary.json").write_text(
                json.dumps({"parts": [row]}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            output_root = root / "runs" / "run1" / "outputs" / "review" / "final_comparison"
            build_final_comparison_review(
                alignment_root=alignment_root,
                output_root=output_root,
                fullflow_root=root,
                run_id="run1",
            )

            cases = json.loads((output_root / "data" / "cases.json").read_text(encoding="utf-8"))
            self.assertEqual(cases[0]["risk_level"], "low")
            self.assertTrue((output_root / "by_risk" / "low.html").exists())
            self.assertFalse((output_root / "by_risk" / "medium.html").exists())


def review_item(part_number: str, risk_score: float) -> ReviewItem:
    return ReviewItem(
        case_id=f"case:{part_number}",
        title=part_number,
        rank=0,
        part_number=part_number,
        file_name="",
        view="",
        risk_score=risk_score,
        risk_level="high",
        risk_reasons=["test"],
        media=[],
    )


if __name__ == "__main__":
    unittest.main()
