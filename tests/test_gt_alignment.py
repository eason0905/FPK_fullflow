from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from real_image_process.FPK_PJ_fullflow.gt_alignment.evaluator import AlignmentOptions
from real_image_process.FPK_PJ_fullflow.gt_alignment.evaluator import align_overlay_layers
from real_image_process.FPK_PJ_fullflow.gt_alignment.evaluator import attach_known_data_issues
from real_image_process.FPK_PJ_fullflow.gt_alignment.evaluator import apply_score_diagnostics
from real_image_process.FPK_PJ_fullflow.gt_alignment.evaluator import alignment_candidate_is_better
from real_image_process.FPK_PJ_fullflow.gt_alignment.evaluator import alignment_quality_score
from real_image_process.FPK_PJ_fullflow.gt_alignment.evaluator import alignment_review_quality_score
from real_image_process.FPK_PJ_fullflow.gt_alignment.evaluator import aligned_canonical_objects
from real_image_process.FPK_PJ_fullflow.gt_alignment.evaluator import alignment_scores
from real_image_process.FPK_PJ_fullflow.gt_alignment.evaluator import canonical_features
from real_image_process.FPK_PJ_fullflow.gt_alignment.evaluator import compare_features
from real_image_process.FPK_PJ_fullflow.gt_alignment.evaluator import effective_scan_result_raw_objects
from real_image_process.FPK_PJ_fullflow.gt_alignment.evaluator import evaluate_alignment
from real_image_process.FPK_PJ_fullflow.gt_alignment.evaluator import matched_box_iou_score
from real_image_process.FPK_PJ_fullflow.gt_alignment.evaluator import objective_error_sources_for_stage_hint
from real_image_process.FPK_PJ_fullflow.gt_alignment.evaluator import parse_scan_result
from real_image_process.FPK_PJ_fullflow.gt_alignment.evaluator import prefer_terminal_land_pads_as_package_pad_proxy
from real_image_process.FPK_PJ_fullflow.gt_alignment.evaluator import representative_score_cases
from real_image_process.FPK_PJ_fullflow.gt_alignment.evaluator import review_bucket_for_summary
from real_image_process.FPK_PJ_fullflow.gt_alignment.evaluator import rotate_bbox_in_box
from real_image_process.FPK_PJ_fullflow.gt_alignment.evaluator import scan_result_objects
from real_image_process.FPK_PJ_fullflow.gt_alignment.evaluator import scan_result_svg_shape
from real_image_process.FPK_PJ_fullflow.gt_alignment.evaluator import score_diagnostic_details
from real_image_process.FPK_PJ_fullflow.gt_alignment.evaluator import select_aligned_result
from real_image_process.FPK_PJ_fullflow.gt_alignment.evaluator import write_scene_svg
from real_image_process.FPK_PJ_fullflow.gt_alignment.evaluator import write_final_graph_json
from real_image_process.FPK_PJ_fullflow.gt_alignment.evaluator import write_scan_result_svg
from real_image_process.FPK_PJ_fullflow.gt_alignment.evaluator import workspace_server_url


class GTAlignmentTests(unittest.TestCase):
    def test_workspace_server_url_uses_fullflow_server_root(self) -> None:
        path = Path("/tmp/work/real_image_process/FPK_PJ_fullflow/runs/run1/outputs/review/final_comparison/index.html")

        self.assertEqual(
            workspace_server_url(path, workspace_root=Path("/tmp/work")),
            "/runs/run1/outputs/review/final_comparison/index.html",
        )

    def test_parse_scan_result_extracts_role_counts_and_bbox(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scan = Path(tmp) / "ScanResultFormat.txt"
            write_scan(scan)

            parsed = parse_scan_result(scan)

            self.assertEqual(parsed["summary"]["object_count"], 2)
            self.assertEqual(parsed["summary"]["role_counts"], {"land": 1, "lead": 1})
            self.assertEqual(parsed["summary"]["bbox"], [0.0, 0.0, 3.0, 1.0])

    def test_scan_result_svg_uses_native_shape_elements(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "scan_result.svg"
            gt = {
                "features": {
                    "objects": [
                        {"role": "land", "node_name": "Rectangle", "bbox": [0.0, 0.0, 1.0, 1.0]},
                        {"role": "lead", "node_name": "Circle", "bbox": [2.0, 0.0, 3.0, 1.0]},
                        {"role": "lead", "node_name": "DShape", "bbox": [4.0, 0.0, 5.0, 1.0]},
                    ]
                }
            }

            self.assertTrue(write_scan_result_svg(gt, output))

            svg = output.read_text(encoding="utf-8")
            self.assertIn("<rect", svg)
            self.assertIn("<ellipse", svg)
            self.assertIn("<path", svg)

    def test_scan_result_svg_shape_falls_back_to_rect_for_unknown_node(self) -> None:
        svg = scan_result_svg_shape(
            {"role": "unknown", "node_name": "Unknown"},
            [0.0, 0.0, 2.0, 1.0],
            color="#000000",
            stroke_width=0.1,
        )

        self.assertTrue(svg.startswith("<rect "))

    def test_aligned_scene_svg_uses_known_shape_elements(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "scene.svg"

            write_scene_svg(
                output,
                title="shape-aware scene",
                layers=[
                    ("gt", [{"role": "lead", "bbox": [0.0, 0.0, 1.0, 1.0], "label": "Circle"}]),
                    (
                        "result",
                        [{"role": "package_pad", "bbox": [2.0, 0.0, 3.0, 1.0], "source_label": "pad_circle"}],
                    ),
                    (
                        "result",
                        [{"role": "package_pad", "bbox": [4.0, 0.0, 5.0, 1.0], "source_label": "pad_dshape"}],
                    ),
                ],
                fallback_label="empty",
            )

            svg = output.read_text(encoding="utf-8")

        self.assertIn("<ellipse", svg)
        self.assertIn("<path", svg)
        self.assertIn('stroke-dasharray="0.05 0.05"', svg)

    def test_parse_scan_result_extracts_group_count_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scan = Path(tmp) / "ScanResultFormat.txt"
            scan.write_text(
                json.dumps(
                    {
                        "Object": [
                            {
                                "ID": 1,
                                "LandData": {},
                                "PointList": [
                                    {"PointX": 0, "PointY": 0},
                                    {"PointX": 1, "PointY": 0},
                                    {"PointX": 1, "PointY": 1},
                                    {"PointX": 0, "PointY": 1},
                                ],
                            },
                            {
                                "ID": 2,
                                "LandData": {},
                                "PointList": [
                                    {"PointX": 0, "PointY": 0},
                                    {"PointX": 1, "PointY": 0},
                                    {"PointX": 1, "PointY": 1},
                                    {"PointX": 0, "PointY": 1},
                                ],
                            },
                            {
                                "ID": 3,
                                "LeadData": {},
                                "PointList": [
                                    {"PointX": 2, "PointY": 0},
                                    {"PointX": 3, "PointY": 0},
                                    {"PointX": 3, "PointY": 1},
                                    {"PointX": 2, "PointY": 1},
                                ],
                            },
                        ],
                        "GroupItems": [
                            {
                                "FirstMartixPinIDs": [[1, 2], [3]],
                                "FirstMartixQX": 2,
                                "FirstMartixQY": 1,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            parsed = parse_scan_result(scan)

            self.assertEqual(
                parsed["summary"]["count_candidates"]["land"],
                [
                    {
                        "source": "GroupItems",
                        "candidate_type": "group_raw_object_count",
                        "group_index": 0,
                        "count": 2,
                        "matrix_qx": 2,
                        "matrix_qy": 1,
                        "object_ids": [1, 2],
                    },
                    {
                        "source": "GroupItems",
                        "candidate_type": "group_matrix_cell_count",
                        "group_index": 0,
                        "count": 1,
                        "matrix_qx": 2,
                        "matrix_qy": 1,
                        "object_ids": [1, 2],
                    },
                    {
                        "source": "GroupItems",
                        "candidate_type": "group_matrix_total_count",
                        "group_index": 0,
                        "count": 2,
                        "matrix_qx": 2,
                        "matrix_qy": 1,
                        "object_ids": [1, 2],
                    },
                ],
            )
            self.assertEqual(parsed["summary"]["count_candidates"]["lead"][0]["count"], 1)

    def test_parse_scan_result_excludes_lead_markers_when_dshape_terminals_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scan = Path(tmp) / "ScanResultFormat.txt"
            objects = []
            for index in range(64):
                objects.append(
                    scan_object(
                        object_id=index + 1,
                        node_name="DShape",
                        role_key="LeadData",
                        x=float(index % 16),
                        y=float(index // 16),
                        width=0.2,
                        height=0.4,
                    )
                )
            objects.append(
                scan_object(
                    object_id=65,
                    node_name="Circle",
                    role_key="LeadData",
                    x=0.0,
                    y=5.0,
                    width=0.4,
                    height=0.4,
                )
            )
            objects.append(
                scan_object(
                    object_id=66,
                    node_name="Rectangle",
                    role_key="LeadData",
                    x=1.0,
                    y=5.0,
                    width=5.0,
                    height=5.0,
                )
            )
            scan.write_text(json.dumps({"Object": objects}), encoding="utf-8")

            parsed = parse_scan_result(scan)

            self.assertEqual(parsed["summary"]["raw_role_counts"], {"lead": 66})
            self.assertEqual(parsed["summary"]["role_counts"], {"lead": 64, "shape": 2})
            overridden = [obj for obj in parsed["features"]["objects"] if obj.get("role_override_reason")]
            self.assertEqual(len(overridden), 2)
            self.assertEqual({obj["raw_role"] for obj in overridden}, {"lead"})

    def test_effective_scan_result_objects_drop_raw_role_when_effective_count_is_zero(self) -> None:
        objects = [
            {"id": 1, "role": "land", "bbox": [0.0, 0.0, 2.0, 2.0]},
            {"id": 2, "role": "lead", "bbox": [3.0, 0.0, 4.0, 1.0]},
            {"id": 3, "role": "shape", "bbox": [0.0, 0.0, 5.0, 5.0]},
        ]

        effective = effective_scan_result_raw_objects(objects, {"lead": 1, "shape": 1}, tolerance=0.01)

        self.assertEqual([obj["id"] for obj in effective], [2, 3])

    def test_scan_result_objects_use_effective_land_objects_for_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scan = Path(tmp) / "ScanResultFormat.txt"
            scan.write_text(
                json.dumps(
                    {
                        "Object": [
                            scan_object(
                                object_id=1,
                                node_name="DShape",
                                role_key="LeadData",
                                x=0.0,
                                y=0.0,
                                width=0.5,
                                height=0.5,
                            ),
                            scan_object(
                                object_id=2,
                                node_name="DShape",
                                role_key="LeadData",
                                x=2.0,
                                y=0.0,
                                width=0.5,
                                height=0.5,
                            ),
                            scan_object(
                                object_id=3,
                                node_name="Donut",
                                role_key="LeadData",
                                x=8.0,
                                y=0.0,
                                width=4.0,
                                height=4.0,
                            ),
                            scan_object(
                                object_id=11,
                                node_name="Rectangle",
                                role_key="LandData",
                                x=0.0,
                                y=6.0,
                                width=0.5,
                                height=0.5,
                            ),
                            scan_object(
                                object_id=12,
                                node_name="Rectangle",
                                role_key="LandData",
                                x=2.0,
                                y=6.0,
                                width=0.5,
                                height=0.5,
                            ),
                            scan_object(
                                object_id=13,
                                node_name="Donut",
                                role_key="LandData",
                                x=8.0,
                                y=0.0,
                                width=4.0,
                                height=4.0,
                            ),
                        ],
                        "GroupItems": [{}],
                    }
                ),
                encoding="utf-8",
            )

            parsed = parse_scan_result(scan)
            metric_objects = scan_result_objects(parsed)

            land_ids = [
                obj["source_object_id"]
                for obj in metric_objects
                if obj["role"] == "land"
            ]
            self.assertEqual(parsed["summary"]["role_counts"]["land"], 2)
            self.assertEqual(land_ids, [11, 12])
            self.assertEqual(parsed["features"]["bbox_candidates"]["land"], [0.0, 6.0, 2.5, 6.5])
            self.assertEqual(parsed["features"]["raw_bbox_candidates"]["land"], [0.0, 0.0, 12.0, 6.5])

    def test_canonical_features_preserves_source_selection_in_summary(self) -> None:
        features = canonical_features(
            {
                "outline_2d": {},
                "package_pads": [{"bbox": [0, 0, 1, 1], "raw_view": "top", "canonical_view": "top"}],
                "land_pads": [{"bbox": [2, 0, 3, 1], "raw_view": "land", "canonical_view": "land"}],
                "lead_contacts": [],
                "lead_pads": [{"bbox": [0, 0, 0.5, 1], "raw_view": "top", "canonical_view": "top"}],
                "source_selection": {
                    "package_pads": {
                        "primary_view": "bottom",
                        "selected_raw_view": "top",
                        "used_fallback": True,
                    }
                },
            }
        )

        self.assertEqual(
            features["summary"]["source_selection"]["package_pads"]["selected_raw_view"],
            "top",
        )
        self.assertEqual(features["summary"]["land_pad_count"], 1)
        self.assertEqual(features["summary"]["lead_pad_count"], 1)
        self.assertNotIn("lead_pad", features["role_counts"])

    def test_align_overlay_layers_selects_best_rotation_with_lowest_turn_cost(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            top_graph = root / "top.package_graph.json"
            bottom_graph = root / "bottom.package_graph.json"
            frame = [0.0, 0.0, 10.0, 10.0]
            top_boxes = [[0.0, 0.0, 2.0, 4.0], [8.0, 6.0, 10.0, 10.0]]
            bottom_boxes = [rotate_bbox_in_box(box, frame, 270) for box in top_boxes]
            write_package_graph(top_graph, "top", top_boxes)
            write_package_graph(bottom_graph, "bottom", bottom_boxes)
            canonical = {
                "part_number": "PART",
                "package_pads": [
                    {
                        "bbox": box,
                        "role": "package_pad",
                        "raw_view": "top",
                        "canonical_view": "top",
                        "source_graph": str(top_graph),
                    }
                    for box in top_boxes
                ]
                + [
                    {
                        "bbox": box,
                        "role": "package_pad",
                        "raw_view": "bottom",
                        "canonical_view": "bottom",
                        "source_graph": str(bottom_graph),
                    }
                    for box in bottom_boxes
                ],
                "evidence_refs": [
                    {
                        "evidence_type": "package_graph",
                        "raw_view": "top",
                        "canonical_view": "top",
                        "graph_path": str(top_graph),
                    },
                    {
                        "evidence_type": "package_graph",
                        "raw_view": "bottom",
                        "canonical_view": "bottom",
                        "graph_path": str(bottom_graph),
                    },
                ],
            }

            result = align_overlay_layers(canonical)

            transforms = result["scores"]["layer_alignment_transforms"]
            self.assertEqual(transforms[str(bottom_graph)]["rotation_degrees"], 0)
            self.assertEqual(transforms[str(bottom_graph)]["match_priority"], 2)
            self.assertEqual(transforms[str(top_graph)]["rotation_degrees"], 90)
            self.assertEqual(transforms[str(top_graph)]["rotation_cost"], 1)
            self.assertEqual(transforms[str(top_graph)]["rotation_iou"], 1.0)

    def test_align_overlay_layers_uses_canonical_multiview_boxes_before_source_graph_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            land_graph = root / "land.package_graph.json"
            top_graph = root / "top.package_graph.json"
            frame = [0.0, 0.0, 10.0, 10.0]
            land_boxes = [[0.0, 0.0, 2.0, 4.0], [8.0, 6.0, 10.0, 10.0]]
            top_boxes = [rotate_bbox_in_box(box, frame, 270) for box in land_boxes]
            write_package_graph(land_graph, "land", [[4.0, 4.0, 5.0, 5.0]])
            write_package_graph(top_graph, "top", [[4.0, 4.0, 5.0, 5.0]])
            canonical = {
                "part_number": "PART",
                "land_pads": [
                    {"bbox": box, "raw_view": "land", "canonical_view": "land", "source_graph": str(land_graph)}
                    for box in land_boxes
                ],
                "package_pads": [
                    {"bbox": box, "raw_view": "top", "canonical_view": "top", "source_graph": str(top_graph)}
                    for box in top_boxes
                ],
                "evidence_refs": [
                    {
                        "evidence_type": "package_graph",
                        "raw_view": "land",
                        "canonical_view": "land",
                        "graph_path": str(land_graph),
                    },
                    {
                        "evidence_type": "package_graph",
                        "raw_view": "top",
                        "canonical_view": "top",
                        "graph_path": str(top_graph),
                    },
                ],
            }

            result = align_overlay_layers(canonical)

            transforms = result["scores"]["layer_alignment_transforms"]
            self.assertEqual(result["summary"]["reference_view"], "land")
            self.assertEqual(transforms[str(land_graph)]["match_source"], "canonical_land_pad")
            self.assertEqual(transforms[str(land_graph)]["match_priority"], 0)
            self.assertEqual(transforms[str(top_graph)]["match_source"], "canonical_top_package_pad")
            self.assertEqual(transforms[str(top_graph)]["rotation_degrees"], 90)
            self.assertEqual(transforms[str(top_graph)]["rotation_iou"], 1.0)

    def test_align_overlay_layers_outputs_rotation_centered_objects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            land_graph = root / "land.package_graph.json"
            top_graph = root / "top.package_graph.json"
            land_boxes = [[0.0, 0.0, 2.0, 4.0], [8.0, 6.0, 10.0, 10.0]]
            top_frame = [100.0, 100.0, 110.0, 110.0]
            top_boxes = [
                [
                    rotated[0] + 100.0,
                    rotated[1] + 100.0,
                    rotated[2] + 100.0,
                    rotated[3] + 100.0,
                ]
                for box in land_boxes
                if (rotated := rotate_bbox_in_box(box, [0.0, 0.0, 10.0, 10.0], 270)) is not None
            ]
            write_package_graph(land_graph, "land", land_boxes)
            write_package_graph(top_graph, "top", top_boxes, outline=top_frame)
            canonical = {
                "part_number": "PART",
                "land_pads": [
                    {"bbox": box, "raw_view": "land", "canonical_view": "land", "source_graph": str(land_graph)}
                    for box in land_boxes
                ],
                "package_pads": [
                    {"bbox": box, "raw_view": "top", "canonical_view": "top", "source_graph": str(top_graph)}
                    for box in top_boxes
                ],
                "evidence_refs": [
                    {
                        "evidence_type": "package_graph",
                        "raw_view": "land",
                        "canonical_view": "land",
                        "graph_path": str(land_graph),
                    },
                    {
                        "evidence_type": "package_graph",
                        "raw_view": "top",
                        "canonical_view": "top",
                        "graph_path": str(top_graph),
                    },
                ],
            }

            result = align_overlay_layers(canonical)

            transforms = result["scores"]["layer_alignment_transforms"]
            self.assertEqual(transforms[str(top_graph)]["rotation_degrees"], 90)
            result_top = [
                obj for obj in result["objects"]
                if obj.get("raw_view") == "top" and obj.get("role") == "package_pad"
            ][0]
            rotation_only_top = [
                obj for obj in result["rotation_only_objects"]
                if obj.get("raw_view") == "top" and obj.get("role") == "package_pad"
            ][0]
            self.assertEqual(result_top["bbox"], [0.0, 0.0, 2.0, 4.0])
            self.assertEqual(result_top["bbox"], rotation_only_top["bbox"])
            self.assertEqual(result_top["alignment_display_mode"], "dimension_scaled_centered_rotated")
            self.assertEqual(result_top["alignment_center_translation"], [-100.0, -100.0])

    def test_align_overlay_layers_scales_by_accepted_dimensions_before_rotation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            land_graph = root / "land.package_graph.json"
            top_graph = root / "top.package_graph.json"
            land_boxes = [[0.0, 0.0, 1.0, 1.0], [3.0, 0.0, 4.0, 1.0]]
            top_boxes = [[100.0, 100.0, 102.0, 101.0], [106.0, 100.0, 108.0, 101.0]]
            write_package_graph(
                land_graph,
                "land",
                land_boxes,
                dimensions=[
                    {"status": "accepted", "axis": "x", "value": 1.0, "target_ids": [1]},
                    {"status": "accepted", "axis": "y", "value": 1.0, "target_ids": [1]},
                ],
            )
            write_package_graph(
                top_graph,
                "top",
                top_boxes,
                outline=[100.0, 100.0, 108.0, 101.0],
                dimensions=[
                    {"status": "accepted", "axis": "x", "value": 1.0, "target_ids": [1]},
                    {"status": "accepted", "axis": "y", "value": 1.0, "target_ids": [1]},
                ],
            )
            canonical = {
                "part_number": "PART",
                "land_pads": [
                    {"bbox": box, "raw_view": "land", "canonical_view": "land", "source_graph": str(land_graph)}
                    for box in land_boxes
                ],
                "package_pads": [
                    {"bbox": box, "raw_view": "top", "canonical_view": "top", "source_graph": str(top_graph)}
                    for box in top_boxes
                ],
                "evidence_refs": [
                    {
                        "evidence_type": "package_graph",
                        "raw_view": "land",
                        "canonical_view": "land",
                        "graph_path": str(land_graph),
                    },
                    {
                        "evidence_type": "package_graph",
                        "raw_view": "top",
                        "canonical_view": "top",
                        "graph_path": str(top_graph),
                    },
                ],
            }

            result = align_overlay_layers(canonical)

            transforms = result["scores"]["layer_alignment_transforms"]
            self.assertEqual(transforms[str(top_graph)]["rotation_degrees"], 0)
            self.assertEqual(transforms[str(top_graph)]["rotation_iou"], 1.0)
            result_top_boxes = [
                obj["bbox"]
                for obj in result["objects"]
                if obj.get("raw_view") == "top" and obj.get("role") == "package_pad"
            ]
            self.assertEqual(result_top_boxes, land_boxes)
            self.assertEqual(transforms[str(top_graph)]["source_unit_scales"]["x"], 0.5)
            self.assertEqual(transforms[str(top_graph)]["source_unit_scales"]["y"], 1.0)
            self.assertEqual(transforms[str(top_graph)]["coordinate_mode"], "dimension_scale_center_translate_then_rotate")

    def test_align_overlay_layers_projects_partial_side_dimension_to_lead_pads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bottom_graph = root / "bottom.package_graph.json"
            side_graph = root / "side.package_graph.json"
            bottom_boxes = [[0.0, 4.0, 2.0, 6.0], [8.0, 4.0, 10.0, 6.0]]
            write_package_graph(bottom_graph, "bottom", bottom_boxes)
            write_package_graph(
                side_graph,
                "side",
                [[0.0, 0.0, 1.0, 1.0]],
                dimensions=[
                    {
                        "status": "accepted",
                        "kind": "size",
                        "axis": "x",
                        "value": 1.0,
                        "target_ids": [1],
                        "anchors": ["center", "left_edge"],
                    }
                ],
            )
            canonical = {
                "part_number": "PART",
                "package_pads": [
                    {
                        "role": "package_pad",
                        "bbox": [0.0, 4.0, 2.0, 6.0],
                        "raw_view": "bottom",
                        "canonical_view": "bottom",
                        "source_type": "canonical_bottom_package_pad",
                        "source_graph": str(bottom_graph),
                        "source_object_id": 1,
                    },
                    {
                        "role": "package_pad",
                        "bbox": [8.0, 4.0, 10.0, 6.0],
                        "raw_view": "bottom",
                        "canonical_view": "bottom",
                        "source_type": "canonical_bottom_package_pad",
                        "source_graph": str(bottom_graph),
                        "source_object_id": 2,
                    },
                ],
                "lead_pads": [
                    {
                        "role": "lead_pad",
                        "bbox": [0.0, 4.5, 2.0, 5.5],
                        "raw_view": "side",
                        "canonical_view": "lateral",
                        "source_type": "derived_partial_evidence_multiview",
                        "source_graph": str(bottom_graph),
                        "source_package_pad_bbox": [0.0, 4.0, 2.0, 6.0],
                        "partial_dimension_semantics": "lead_ground_contact_length",
                        "projection_axis": "y",
                    },
                    {
                        "role": "lead_pad",
                        "bbox": [8.0, 4.5, 10.0, 5.5],
                        "raw_view": "side",
                        "canonical_view": "lateral",
                        "source_type": "derived_partial_evidence_multiview",
                        "source_graph": str(bottom_graph),
                        "source_package_pad_bbox": [8.0, 4.0, 10.0, 6.0],
                        "partial_dimension_semantics": "lead_ground_contact_length",
                        "projection_axis": "y",
                    },
                ],
                "evidence_refs": [
                    {
                        "evidence_type": "package_graph",
                        "raw_view": "bottom",
                        "canonical_view": "bottom",
                        "graph_path": str(bottom_graph),
                    },
                    {
                        "evidence_type": "package_graph",
                        "raw_view": "side",
                        "canonical_view": "lateral",
                        "graph_path": str(side_graph),
                    },
                ],
            }

            result = align_overlay_layers(canonical)

            lead_pads = [obj for obj in result["objects"] if obj.get("role") == "lead_pad"]
            package_pads = [obj for obj in result["objects"] if obj.get("role") == "package_pad"]
            side_raw_objects = [
                obj
                for obj in result["objects"]
                if obj.get("raw_view") == "side" and not str(obj.get("source_type") or "").startswith("derived_partial_evidence")
            ]
            self.assertEqual(len(lead_pads), 2)
            self.assertEqual(len(package_pads), 2)
            self.assertEqual(lead_pads[0]["raw_view"], "side")
            self.assertEqual(lead_pads[0]["partial_dimension_semantics"], "lead_ground_contact_length")
            self.assertEqual(lead_pads[0]["projection_axis"], "y")
            self.assertEqual(lead_pads[0]["bbox"][0], package_pads[0]["bbox"][0])
            self.assertEqual(lead_pads[0]["bbox"][2], package_pads[0]["bbox"][2])
            self.assertGreater(lead_pads[0]["bbox"][1], package_pads[0]["bbox"][1])
            self.assertLess(lead_pads[0]["bbox"][3], package_pads[0]["bbox"][3])
            self.assertEqual(side_raw_objects, [])

    def test_align_overlay_layers_does_not_render_raw_land_detail_graph(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            land_graph = root / "land.package_graph.json"
            detail_graph = root / "land_detail.package_graph.json"
            write_package_graph(land_graph, "land", [[0.0, 0.0, 2.0, 2.0]])
            write_package_graph(detail_graph, "land_detail", [[10.0, 10.0, 30.0, 30.0], [15.0, 15.0, 20.0, 20.0]])
            canonical = {
                "part_number": "PART",
                "evidence_refs": [
                    {
                        "evidence_type": "package_graph",
                        "raw_view": "land",
                        "canonical_view": "land",
                        "graph_path": str(land_graph),
                    },
                    {
                        "evidence_type": "package_graph",
                        "raw_view": "land_detail",
                        "canonical_view": "lead_detail",
                        "graph_path": str(detail_graph),
                    },
                ],
            }

            result = align_overlay_layers(canonical)

            self.assertEqual([obj for obj in result["objects"] if obj.get("raw_view") == "land_detail"], [])

    def test_align_overlay_layers_keeps_canonical_inner_land_without_source_graph(self) -> None:
        canonical = {
            "part_number": "PART",
            "land_pads": [
                {
                    "role": "land_pad",
                    "bbox": [0.0, 0.0, 2.0, 2.0],
                    "source_type": "scan_result_format",
                }
            ],
            "inner_land_pads": [
                {
                    "role": "inner_land_pad",
                    "bbox": [0.5, 0.5, 1.5, 1.5],
                    "source_type": "derived_inner_land_pad",
                    "source_land_pad_bbox": [0.0, 0.0, 2.0, 2.0],
                }
            ],
            "evidence_refs": [],
        }

        result = align_overlay_layers(canonical)

        scan_result_land_pads = [
            obj
            for obj in result["objects"]
            if obj.get("role") == "land_pad" and obj.get("source_type") == "scan_result_format"
        ]
        self.assertEqual(scan_result_land_pads, [])
        inner_pads = [obj for obj in result["objects"] if obj.get("role") == "inner_land_pad"]
        self.assertEqual(len(inner_pads), 1)
        self.assertEqual(inner_pads[0]["bbox"], [0.5, 0.5, 1.5, 1.5])
        self.assertEqual(inner_pads[0]["raw_view"], "land")

    def test_final_graph_role_buckets_include_rotation_only_land_and_lead_roles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "final_graph.json"

            write_final_graph_json(
                output_path,
                {"part_number": "PART", "status": "aligned", "unified_multiview_layers_path": "unified_multiview_layers.json"},
                [],
                {
                    "source_views": ["top", "land", "front"],
                    "canonical_source_views": ["top", "land", "lateral"],
                    "dimensions": [],
                    "evidence_refs": [],
                    "conflicts": [],
                },
                [
                    {"role": "land_pad", "bbox": [0.0, 0.0, 1.0, 1.0]},
                    {"role": "lead_contact", "bbox": [2.0, 0.0, 3.0, 1.0]},
                    {
                        "role": "lead_pad",
                        "bbox": [4.0, 0.0, 5.0, 1.0],
                        "source_label": "lead pad",
                        "alignment_display_mode": "dimension_scaled_centered_rotated",
                        "alignment_center_translation": [-10.0, 5.0],
                        "alignment_match_source": "land",
                        "alignment_match_priority": 0,
                    },
                    {
                        "role": "partial_pad_width",
                        "bbox": [6.0, 0.0, 7.0, 1.0],
                        "source_label": "partial pad width",
                    },
                ],
                {},
                {},
            )

            final_graph = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(len(final_graph["land_pads"]), 1)
            self.assertEqual(len(final_graph["lead_contacts"]), 1)
            self.assertEqual(len(final_graph["lead_pads"]), 2)
            lead_pad = final_graph["lead_pads"][0]
            self.assertEqual(lead_pad["source_label"], "lead pad")
            self.assertEqual(lead_pad["alignment_display_mode"], "dimension_scaled_centered_rotated")
            self.assertEqual(lead_pad["alignment_center_translation"], [-10.0, 5.0])
            self.assertEqual(lead_pad["alignment_match_source"], "land")
            self.assertEqual(lead_pad["alignment_match_priority"], 0)
            self.assertEqual(final_graph["lead_pads"][1]["role"], "partial_pad_width")

    def test_aligned_canonical_objects_preserves_lead_pads_as_separate_role(self) -> None:
        gt = {
            "features": {
                "bbox_candidates": {"lead": [0.0, 0.0, 10.0, 10.0], "all": [0.0, 0.0, 10.0, 10.0]},
            }
        }
        canonical = {
            "package_pads": [{"bbox": [0.0, 0.0, 10.0, 10.0]}],
            "land_pads": [],
            "lead_contacts": [],
            "lead_pads": [
                {
                    "bbox": [0.0, 0.0, 2.0, 10.0],
                    "source_package_pad_id": 1,
                    "lead_contact_length": 2.0,
                }
            ],
        }

        result = aligned_canonical_objects(gt, canonical)

        lead_pads = [obj for obj in result if obj.get("role") == "lead_pad"]
        self.assertEqual(len(lead_pads), 1)
        self.assertEqual(lead_pads[0]["bbox"], [0.0, 0.0, 2.0, 10.0])

    def test_land_count_mismatch_uses_scan_result_visible_annotation_hint(self) -> None:
        canonical = {
            "source_selection": {
                "land_pads": {
                    "graph_path": "/tmp/PART/land.package_graph.json",
                }
            },
            "land_pads": [
                {"bbox": [0, 0, 1, 1], "raw_view": "land", "canonical_view": "land"},
            ],
            "evidence_refs": [
                {
                    "evidence_type": "package_graph",
                    "graph_path": "/tmp/PART/land.package_graph.json",
                    "pad_like_count": 2,
                }
            ],
            "missing_canonical_views": [],
        }

        graph = canonical_features(canonical)
        checks = compare_features(
            {
                "role_counts": {"land": 3},
                "count_candidates": {},
                "bbox_candidates": {},
                "objects": [],
            },
            graph,
            AlignmentOptions(),
        )

        land_check = next(check for check in checks if check["name"] == "land_count")
        self.assertEqual(graph["summary"]["selected_land_graph_pad_like_count"], 2)
        self.assertEqual(graph["summary"]["land_pad_count"], 1)
        self.assertEqual(land_check["status"], "mismatch")
        self.assertEqual(
            land_check["stage_hint"],
            "scan_result_land_count_exceeds_visible_land_annotation",
        )
        self.assertEqual(
            objective_error_sources_for_stage_hint(land_check["stage_hint"]),
            ["gt_annotation_issue", "scan_result_parsing"],
        )

    def test_bbox_aspect_mismatch_with_package_fallback_is_multiview_hint(self) -> None:
        checks = compare_features(
            {
                "role_counts": {"lead": 1},
                "count_candidates": {},
                "objects": [],
                "bbox": [0, 0, 10, 1],
                "bbox_candidates": {"lead": [0, 0, 10, 1]},
            },
            {
                "role_counts": {"package_pad": 1},
                "summary": {
                    "package_pad_count": 1,
                    "lead_equivalent_count": 1,
                    "missing_canonical_views": ["bottom"],
                    "source_selection": {
                        "package_pads": {
                            "primary_view": "bottom",
                            "selected_raw_view": "top",
                            "used_fallback": True,
                            "missing_primary": True,
                        }
                    },
                },
                "objects": [],
                "bbox": [0, 0, 1, 1],
                "bbox_candidates": {"package": [0, 0, 1, 1]},
            },
            AlignmentOptions(),
        )

        bbox_check = next(check for check in checks if check["name"] == "bbox_aspect")
        self.assertEqual(bbox_check["status"], "mismatch")
        self.assertEqual(
            bbox_check["stage_hint"],
            "multiview_fallback_package_pad_source_geometry_mismatch",
        )

    def test_lead_count_over_detection_with_subgroup_candidates_is_package_graph_hint(self) -> None:
        graph = canonical_features(
            {
                "package_pads": [{"bbox": [float(index), 0, float(index) + 0.5, 1]} for index in range(28)],
                "land_pads": [],
                "lead_contacts": [],
                "missing_canonical_views": [],
            }
        )

        checks = compare_features(
            {
                "role_counts": {"lead": 20},
                "count_candidates": {
                    "lead": [
                        {
                            "source": "GroupItems",
                            "candidate_type": "group_matrix_total_count",
                            "group_index": 0,
                            "count": 10,
                        }
                    ]
                },
                "bbox_candidates": {},
                "objects": [],
            },
            graph,
            AlignmentOptions(),
        )

        lead_check = next(check for check in checks if check["name"] == "lead_count")
        self.assertEqual(lead_check["status"], "mismatch")
        self.assertEqual(lead_check["scan_count_candidate_counts"], [10])
        self.assertEqual(
            lead_check["stage_hint"],
            "package_graph_package_pad_reconstruction_count_mismatch",
        )
        self.assertEqual(
            objective_error_sources_for_stage_hint(lead_check["stage_hint"]),
            ["model_prediction", "package_graph_reconstruction"],
        )

    def test_lead_count_mismatch_with_scan_group_candidate_is_scan_result_alignment_hint(self) -> None:
        graph = canonical_features(
            {
                "package_pads": [{"bbox": [float(index), 0, float(index) + 0.5, 1]} for index in range(28)],
                "land_pads": [],
                "lead_contacts": [],
                "missing_canonical_views": [],
            }
        )

        checks = compare_features(
            {
                "role_counts": {"lead": 20},
                "count_candidates": {
                    "lead": [
                        {
                            "source": "GroupItems",
                            "candidate_type": "group_matrix_total_count",
                            "group_index": 0,
                            "count": 40,
                        }
                    ]
                },
                "bbox_candidates": {},
                "objects": [],
            },
            graph,
            AlignmentOptions(),
        )

        lead_check = next(check for check in checks if check["name"] == "lead_count")
        self.assertEqual(lead_check["status"], "mismatch")
        self.assertEqual(lead_check["scan_count_candidate_counts"], [40])
        self.assertEqual(
            lead_check["stage_hint"],
            "scan_result_lead_count_ambiguous_with_graph_count_mismatch",
        )
        self.assertEqual(
            objective_error_sources_for_stage_hint(lead_check["stage_hint"]),
            ["package_graph_reconstruction", "scan_result_parsing"],
        )

    def test_evaluate_alignment_writes_summary_and_mismatch_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_part = root / "dataset" / "PART"
            dataset_part.mkdir(parents=True)
            write_scan(dataset_part / "ScanResultFormat.txt")
            multiview_part = root / "multiview" / "parts" / "PART"
            multiview_part.mkdir(parents=True)
            bottom_graph = root / "bottom.package_graph.json"
            write_package_graph(bottom_graph, "bottom", [[0, 0, 1, 1], [2, 0, 3, 1]])
            (multiview_part / "unified_multiview_layers.json").write_text(
                json.dumps(
                    {
                        "part_number": "PART",
                        "outline_2d": {},
                        "package_pads": [
                            {
                                "bbox": [0, 0, 1, 1],
                                "role": "package_pad",
                                "raw_view": "bottom",
                                "canonical_view": "bottom",
                                "source_graph": str(bottom_graph),
                            },
                            {
                                "bbox": [2, 0, 3, 1],
                                "role": "package_pad",
                                "raw_view": "bottom",
                                "canonical_view": "bottom",
                                "source_graph": str(bottom_graph),
                            },
                        ],
                        "land_pads": [],
                        "lead_contacts": [],
                        "dimensions": [],
                        "source_views": ["bottom"],
                        "canonical_source_views": ["bottom"],
                        "missing_canonical_views": ["land"],
                        "conflicts": [],
                        "evidence_refs": [
                            {
                                "evidence_type": "package_graph",
                                "raw_view": "bottom",
                                "canonical_view": "bottom",
                                "graph_path": str(bottom_graph),
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = evaluate_alignment(
                dataset_root=root / "dataset",
                multiview_root=root / "multiview",
                output_root=root / "alignment",
                options=AlignmentOptions(),
            )

            self.assertEqual(result["total_parts"], 1)
            self.assertEqual(result["valid_case_count"], 1)
            self.assertEqual(result["mismatch_parts"], 0)
            self.assertEqual(result["conflict_count"], 0)
            self.assertEqual(result["conflict_case_count"], 0)
            self.assertEqual(result["missing_view_case_count"], 1)
            self.assertEqual(result["missing_canonical_view_counts"], {"land": 1})
            self.assertEqual(result["missing_canonical_view_total_count"], 1)
            self.assertTrue(result["gallery_path"].endswith("review/final_comparison/index.html"))
            self.assertTrue(result["gallery_url"].endswith("review/final_comparison/index.html"))
            self.assertEqual(result["reason_counts"], {})
            self.assertEqual(result["stage_hint_counts"], {})
            self.assertEqual(result["error_source_counts"], {})
            self.assertEqual(result["mapping_counts"], {})
            self.assertEqual(result["mismatch_check_counts"], {})
            self.assertEqual(result["count_delta_histograms"], {})
            self.assertEqual(result["alignment_score_summary"]["total_parts"], 1)
            self.assertEqual(result["alignment_score_summary"]["scored_parts"], 1)
            self.assertEqual(result["alignment_score_summary"]["unscored_parts"], 0)
            self.assertEqual(sum(result["alignment_score_summary"]["risk_counts"].values()), 1)
            self.assertEqual(result["risk_counts"], result["alignment_score_summary"]["risk_counts"])
            expected_manifest_count = result["risk_counts"].get("high", 0) + result["risk_counts"].get("medium", 0)
            self.assertEqual(result["review_bucket_manifest_count"], expected_manifest_count)
            self.assertTrue(Path(result["review_bucket_manifest_path"]).exists())
            manifest_lines = Path(result["review_bucket_manifest_path"]).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(manifest_lines), expected_manifest_count)
            if manifest_lines:
                manifest_row = json.loads(manifest_lines[0])
                self.assertEqual(manifest_row["part_number"], "PART")
                self.assertIn(manifest_row["review_risk_level"], {"high", "medium"})
                self.assertIn("review_bucket", manifest_row)
                self.assertIn("quality_score", manifest_row)
                self.assertIn("selected_quality_score", manifest_row)
                self.assertIn("review_quality_score", manifest_row)
                self.assertTrue(manifest_row["alignment_path"].endswith("alignment.json"))
            self.assertEqual(result["data_issue_count"], result["review_bucket_counts"].get("data_or_gt_issue", 0))
            self.assertEqual(
                result["excluded_data_issue_count"],
                result["review_bucket_counts"].get("data_or_gt_issue", 0)
                + result["review_bucket_counts"].get("scan_result_issue", 0),
            )
            self.assertEqual(
                result["stage_hint_reason_counts"],
                {},
            )
            self.assertTrue((root / "alignment" / "parts" / "PART" / "alignment.json").exists())
            self.assertTrue((root / "alignment" / "parts" / "PART" / "scan_result.svg").exists())
            self.assertTrue((multiview_part / "gt_reference.svg").exists())
            self.assertTrue((multiview_part / "default_aligned_result.svg").exists())
            self.assertTrue((multiview_part / "default_comparison.svg").exists())
            self.assertTrue((multiview_part / "aligned_result.svg").exists())
            self.assertTrue((multiview_part / "alignment_graph.svg").exists())
            self.assertTrue((multiview_part / "comparison.svg").exists())
            self.assertTrue((multiview_part / "final_graph.json").exists())
            final_graph = json.loads((multiview_part / "final_graph.json").read_text(encoding="utf-8"))
            self.assertEqual(final_graph["coordinate_system"]["name"], "multiview_dimension_scaled_centered_rotated_2d")
            self.assertEqual(
                final_graph["coordinate_system"]["unit"],
                "selected reference package-graph pixels after dimension-scale normalization",
            )
            self.assertEqual(final_graph["source_unified_multiview_layers_path"], str(multiview_part / "unified_multiview_layers.json"))
            self.assertEqual(len(final_graph["package_pads"]), 2)
            self.assertIn("gt_reference", final_graph)
            aligned_svg = (multiview_part / "aligned_result.svg").read_text(encoding="utf-8")
            self.assertIn("rotation-centered overlay", aligned_svg)
            alignment = json.loads((root / "alignment" / "parts" / "PART" / "alignment.json").read_text(encoding="utf-8"))
            self.assertEqual(alignment["checks"], [])
            self.assertEqual(alignment["summary"]["error_sources"], [])
            self.assertEqual(alignment["summary"]["objective_error_sources"], [])
            self.assertTrue(alignment["summary"]["scan_result_svg_path"].endswith("scan_result.svg"))
            self.assertTrue(alignment["summary"]["gt_reference_svg_path"].endswith("gt_reference.svg"))
            self.assertTrue(alignment["summary"]["default_aligned_result_svg_path"].endswith("default_aligned_result.svg"))
            self.assertTrue(alignment["summary"]["default_comparison_svg_path"].endswith("default_comparison.svg"))
            self.assertTrue(alignment["summary"]["aligned_result_svg_path"].endswith("aligned_result.svg"))
            self.assertTrue(alignment["summary"]["alignment_graph_svg_path"].endswith("alignment_graph.svg"))
            self.assertTrue(alignment["summary"]["comparison_svg_path"].endswith("comparison.svg"))
            self.assertTrue(alignment["summary"]["final_graph_path"].endswith("final_graph.json"))
            self.assertIn("alignment_scores", alignment["summary"])
            self.assertEqual(alignment["metrics"], alignment["summary"]["alignment_scores"])
            self.assertEqual(len((root / "alignment" / "mismatches.jsonl").read_text(encoding="utf-8").splitlines()), 0)

    def test_evaluate_alignment_classifies_missing_graph_failure_file_as_reconstruction_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_part = root / "dataset" / "PART"
            dataset_part.mkdir(parents=True)
            write_scan(dataset_part / "ScanResultFormat.txt")
            multiview_part = root / "multiview" / "parts" / "PART"
            multiview_part.mkdir(parents=True)
            (multiview_part / "unified_multiview_layers.json").write_text(
                json.dumps(
                    {
                        "part_number": "PART",
                        "status": "missing_graphs",
                        "failure_reason": "no_package_graph_for_part",
                        "outline_2d": {},
                        "package_pads": [],
                        "land_pads": [],
                        "lead_contacts": [],
                        "dimensions": [],
                        "source_views": [],
                        "canonical_source_views": [],
                        "missing_canonical_views": ["bottom", "land", "lateral", "lead_detail"],
                        "conflicts": [{"status": "missing_graphs", "reason": "no_package_graph_for_part"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = evaluate_alignment(
                dataset_root=root / "dataset",
                multiview_root=root / "multiview",
                output_root=root / "alignment",
                options=AlignmentOptions(),
            )

            self.assertEqual(result["status_counts"], {"missing_canonical": 1})
            self.assertEqual(result["mismatch_parts"], 1)
            self.assertEqual(result["reason_counts"], {"no_package_graph_for_part": 1})
            self.assertEqual(result["stage_hint_counts"], {"package_graph_reconstruction_missing": 1})
            self.assertEqual(result["objective_error_source_counts"], {"package_graph_reconstruction": 1})
            self.assertEqual(result["mismatch_check_counts"], {})
            alignment = json.loads((root / "alignment" / "parts" / "PART" / "alignment.json").read_text(encoding="utf-8"))
            self.assertEqual(alignment["summary"]["status"], "missing_canonical")
            self.assertEqual(alignment["summary"]["failure_reason"], "no_package_graph_for_part")
            self.assertEqual(alignment["summary"]["stage_hints"], ["package_graph_reconstruction_missing"])
            self.assertEqual(alignment["summary"]["objective_error_sources"], ["package_graph_reconstruction"])
            self.assertEqual(alignment["checks"], [])
            self.assertIn("alignment_scores", alignment["summary"])

    def test_objective_error_source_mapping_uses_target_taxonomy(self) -> None:
        self.assertEqual(objective_error_sources_for_stage_hint("data_missing_land_view"), ["gt_annotation_issue"])
        self.assertEqual(objective_error_sources_for_stage_hint("multiview_missing_land_view"), ["multiview_alignment"])
        self.assertEqual(
            objective_error_sources_for_stage_hint("package_graph_land_reconstruction_count_mismatch"),
            ["model_prediction", "package_graph_reconstruction"],
        )
        self.assertEqual(
            objective_error_sources_for_stage_hint("scan_result_alignment_or_reconstruction_geometry_mismatch"),
            ["package_graph_reconstruction", "scan_result_parsing"],
        )
        self.assertEqual(
            objective_error_sources_for_stage_hint("scan_result_land_count_exceeds_visible_land_annotation"),
            ["gt_annotation_issue", "scan_result_parsing"],
        )

    def test_package_pads_count_as_lead_equivalent_for_scan_result_alignment(self) -> None:
        checks = compare_features(
            {
                "role_counts": {"lead": 1, "land": 0},
                "objects": [],
                "bbox": None,
            },
            {
                "role_counts": {"package_pad": 1},
                "summary": {"lead_equivalent_count": 1, "missing_canonical_views": []},
                "objects": [],
                "bbox": None,
            },
            AlignmentOptions(),
        )

        lead_check = next(check for check in checks if check["name"] == "lead_count")
        self.assertEqual(lead_check["status"], "aligned")

    def test_central_thermal_package_pad_is_not_counted_as_lead_terminal(self) -> None:
        package_pads = []
        for index in range(10):
            package_pads.append({"bbox": [0.0, float(index), 0.2, float(index) + 0.5]})
            package_pads.append({"bbox": [5.0, float(index), 5.2, float(index) + 0.5]})
        package_pads.append({"bbox": [1.5, 2.0, 3.5, 7.0]})

        features = canonical_features(
            {
                "package_pads": package_pads,
                "land_pads": [],
                "lead_contacts": [],
            }
        )

        self.assertEqual(features["summary"]["package_pad_count"], 21)
        self.assertEqual(features["summary"]["thermal_package_pad_count"], 1)
        self.assertEqual(features["summary"]["terminal_package_pad_count"], 20)
        self.assertEqual(features["summary"]["lead_equivalent_count"], 20)

    def test_moderately_larger_central_package_pad_is_not_counted_as_lead_terminal(self) -> None:
        package_pads = [
            {"bbox": [0.0, 0.0, 1.0, 10.0]},
            {"bbox": [3.0, 0.0, 4.0, 10.0]},
            {"bbox": [0.0, 20.0, 1.0, 30.0]},
            {"bbox": [3.0, 20.0, 4.0, 30.0]},
            {"bbox": [1.75, 6.0, 2.75, 24.5]},
        ]

        features = canonical_features(
            {
                "package_pads": package_pads,
                "land_pads": package_pads,
                "lead_contacts": [],
            }
        )

        self.assertEqual(features["summary"]["thermal_package_pad_count"], 1)
        self.assertEqual(features["summary"]["terminal_package_pad_count"], 4)
        self.assertEqual(features["summary"]["thermal_land_pad_count"], 1)
        self.assertEqual(features["summary"]["terminal_land_pad_count"], 4)

    def test_compact_larger_central_package_pad_is_not_counted_as_lead_terminal(self) -> None:
        package_pads = [
            {"bbox": [0.0, 0.0, 2.0, 4.0]},
            {"bbox": [8.0, 0.0, 10.0, 4.0]},
            {"bbox": [0.0, 16.0, 2.0, 20.0]},
            {"bbox": [8.0, 16.0, 10.0, 20.0]},
            {"bbox": [0.0, 8.0, 4.0, 10.0]},
            {"bbox": [16.0, 8.0, 20.0, 10.0]},
            {"bbox": [0.0, 11.0, 4.0, 13.0]},
            {"bbox": [16.0, 11.0, 20.0, 13.0]},
            {"bbox": [8.0, 8.0, 12.0, 12.0]},
        ]

        features = canonical_features(
            {
                "package_pads": package_pads,
                "land_pads": package_pads,
                "lead_contacts": [],
            }
        )

        self.assertEqual(features["summary"]["thermal_package_pad_count"], 1)
        self.assertEqual(features["summary"]["terminal_package_pad_count"], 8)
        self.assertEqual(features["summary"]["thermal_land_pad_count"], 1)
        self.assertEqual(features["summary"]["terminal_land_pad_count"], 8)

    def test_side_internal_bar_is_not_counted_as_lead_terminal_when_central_pad_exists(self) -> None:
        package_pads = [
            {"bbox": [0.0, 0.0, 1.0, 10.0]},
            {"bbox": [3.0, 0.0, 4.0, 10.0]},
            {"bbox": [6.0, 0.0, 7.0, 10.0]},
            {"bbox": [0.0, 30.0, 1.0, 40.0]},
            {"bbox": [3.0, 30.0, 4.0, 40.0]},
            {"bbox": [6.0, 30.0, 7.0, 40.0]},
            {"bbox": [2.0, 12.0, 6.0, 28.0]},
            {"bbox": [0.0, 10.0, 1.5, 30.0]},
        ]

        features = canonical_features(
            {
                "package_pads": package_pads,
                "land_pads": package_pads,
                "lead_contacts": [],
            }
        )

        self.assertEqual(features["summary"]["thermal_package_pad_count"], 2)
        self.assertEqual(features["summary"]["terminal_package_pad_count"], 6)
        self.assertEqual(features["summary"]["thermal_land_pad_count"], 2)
        self.assertEqual(features["summary"]["terminal_land_pad_count"], 6)

    def test_scan_group_lead_count_can_match_rect_package_pad_count(self) -> None:
        graph = canonical_features(
            {
                "package_pads": [
                    {"label": "rect", "bbox": [0.0, 0.0, 1.0, 1.0]},
                    {"label": "rect", "bbox": [0.0, 2.0, 1.0, 3.0]},
                    {"label": "circle", "bbox": [2.0, 0.0, 4.0, 2.0]},
                    {"label": "circle", "bbox": [2.5, 2.5, 3.0, 3.0]},
                ],
                "land_pads": [],
                "lead_contacts": [],
            }
        )
        gt = {
            "role_counts": {"lead": 1},
            "count_candidates": {
                "lead": [
                    {"source": "GroupItems", "candidate_type": "group_matrix_cell_count", "count": 2}
                ]
            },
            "bbox_candidates": {},
        }

        checks = compare_features(gt, graph, AlignmentOptions())
        lead_check = next(check for check in checks if check["name"] == "lead_count")

        self.assertEqual(graph["summary"]["package_pad_count"], 4)
        self.assertEqual(graph["summary"]["package_pad_rect_count"], 2)
        self.assertEqual(lead_check["status"], "aligned")
        self.assertEqual(lead_check["actual"], 2)
        self.assertEqual(lead_check["actual_role"], "package_pad_rect_count")
        self.assertEqual(lead_check["selected_mapping"], "scan_group_lead_count_to_package_pad_rect_count")

    def test_scan_group_matrix_total_count_can_match_package_pad_grid(self) -> None:
        checks = compare_features(
            {
                "role_counts": {"lead": 2},
                "count_candidates": {
                    "lead": [
                        {
                            "source": "GroupItems",
                            "candidate_type": "group_raw_object_count",
                            "count": 2,
                            "matrix_qx": 3,
                            "matrix_qy": 2,
                        },
                        {
                            "source": "GroupItems",
                            "candidate_type": "group_matrix_total_count",
                            "count": 6,
                            "matrix_qx": 3,
                            "matrix_qy": 2,
                        },
                    ]
                },
                "bbox_candidates": {},
            },
            {
                "role_counts": {"package_pad": 6},
                "summary": {
                    "package_pad_count": 6,
                    "terminal_package_pad_count": 6,
                    "lead_equivalent_count": 6,
                },
                "objects": [],
                "bbox": None,
                "bbox_candidates": {},
            },
            AlignmentOptions(),
        )

        lead_check = next(check for check in checks if check["name"] == "lead_count")

        self.assertEqual(lead_check["status"], "aligned")
        self.assertEqual(lead_check["expected"], 6)
        self.assertEqual(lead_check["actual"], 6)
        self.assertEqual(lead_check["selected_mapping"], "scan_group_lead_count_candidate")

    def test_evaluate_alignment_counts_scan_land_to_package_pad_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_part = root / "dataset" / "PART"
            dataset_part.mkdir(parents=True)
            write_land_and_lead_scan(dataset_part / "ScanResultFormat.txt")
            bottom_graph = root / "bottom.package_graph.json"
            write_package_graph(bottom_graph, "bottom", [[0, 0, 1, 1], [2, 0, 3, 1]])
            multiview_part = root / "multiview" / "parts" / "PART"
            multiview_part.mkdir(parents=True)
            (multiview_part / "unified_multiview_layers.json").write_text(
                json.dumps(
                    {
                        "part_number": "PART",
                        "outline_2d": {},
                        "package_pads": [
                            {
                                "bbox": [0, 0, 1, 1],
                                "role": "package_pad",
                                "raw_view": "bottom",
                                "canonical_view": "bottom",
                                "source_graph": str(bottom_graph),
                            },
                            {
                                "bbox": [2, 0, 3, 1],
                                "role": "package_pad",
                                "raw_view": "bottom",
                                "canonical_view": "bottom",
                                "source_graph": str(bottom_graph),
                            },
                        ],
                        "land_pads": [],
                        "lead_contacts": [],
                        "dimensions": [],
                        "source_views": ["bottom"],
                        "canonical_source_views": ["bottom"],
                        "missing_canonical_views": ["land"],
                        "conflicts": [],
                        "evidence_refs": [
                            {
                                "kind": "package_graph",
                                "raw_view": "bottom",
                                "canonical_view": "bottom",
                                "path": str(bottom_graph),
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = evaluate_alignment(
                dataset_root=root / "dataset",
                multiview_root=root / "multiview",
                output_root=root / "alignment",
                options=AlignmentOptions(),
            )

            self.assertEqual(result["aligned_parts"], 1)
            self.assertEqual(result["mismatch_parts"], 0)
            self.assertEqual(result["mapping_counts"], {})

    def test_package_pads_and_lead_contacts_are_not_double_counted(self) -> None:
        checks = compare_features(
            {
                "role_counts": {"lead": 2, "land": 0},
                "objects": [],
                "bbox": None,
            },
            {
                "role_counts": {"package_pad": 2, "lead": 2},
                "summary": {"lead_equivalent_count": 2, "missing_canonical_views": []},
                "objects": [],
                "bbox": None,
            },
            AlignmentOptions(),
        )

        lead_check = next(check for check in checks if check["name"] == "lead_count")
        self.assertEqual(lead_check["actual"], 2)
        self.assertEqual(lead_check["status"], "aligned")

    def test_scan_land_lead_role_swap_can_align_against_package_and_land_counts(self) -> None:
        checks = compare_features(
            {
                "role_counts": {"land": 4, "lead": 2},
                "objects": [],
                "bbox": None,
            },
            {
                "role_counts": {"package_pad": 4, "land": 2},
                "summary": {"package_pad_count": 4, "lead_equivalent_count": 4, "missing_canonical_views": []},
                "objects": [],
                "bbox": None,
            },
            AlignmentOptions(),
        )

        land_check = next(check for check in checks if check["name"] == "land_count")
        lead_check = next(check for check in checks if check["name"] == "lead_count")
        self.assertEqual(land_check["status"], "aligned")
        self.assertEqual(lead_check["status"], "aligned")
        self.assertEqual(land_check["selected_mapping"], "scan_land_to_terminal_package_pad_scan_lead_to_terminal_land")
        self.assertEqual(lead_check["actual_role"], "terminal_land_pad_count")

    def test_scan_land_lead_role_swap_uses_terminal_counts_when_thermal_pads_exist(self) -> None:
        checks = compare_features(
            {
                "role_counts": {"land": 128, "lead": 64},
                "objects": [],
                "bbox": None,
            },
            {
                "role_counts": {"package_pad": 129, "land": 65},
                "summary": {
                    "package_pad_count": 129,
                    "terminal_package_pad_count": 128,
                    "land_pad_count": 65,
                    "terminal_land_pad_count": 64,
                    "lead_equivalent_count": 128,
                    "missing_canonical_views": [],
                },
                "objects": [],
                "bbox": None,
            },
            AlignmentOptions(),
        )

        land_check = next(check for check in checks if check["name"] == "land_count")
        lead_check = next(check for check in checks if check["name"] == "lead_count")
        self.assertEqual(land_check["status"], "aligned")
        self.assertEqual(land_check["actual_role"], "terminal_package_pad_count")
        self.assertEqual(lead_check["status"], "aligned")
        self.assertEqual(lead_check["actual_role"], "terminal_land_pad_count")

    def test_scan_lead_can_include_raw_package_pad_count_when_thermal_pad_is_counted(self) -> None:
        checks = compare_features(
            {
                "role_counts": {"land": 0, "lead": 52},
                "objects": [],
                "bbox": None,
            },
            {
                "role_counts": {"package_pad": 52},
                "summary": {
                    "package_pad_count": 52,
                    "terminal_package_pad_count": 51,
                    "lead_equivalent_count": 51,
                    "missing_canonical_views": [],
                },
                "objects": [],
                "bbox": None,
            },
            AlignmentOptions(),
        )

        lead_check = next(check for check in checks if check["name"] == "lead_count")
        self.assertEqual(lead_check["status"], "aligned")
        self.assertEqual(lead_check["selected_mapping"], "scan_lead_to_raw_package_pad_count_including_thermal")
        self.assertEqual(lead_check["actual_role"], "package_pad_count")
        self.assertEqual(lead_check["terminal_package_pad_count"], 51)

    def test_scan_lead_can_include_terminal_package_pads_plus_land_detail_contacts(self) -> None:
        checks = compare_features(
            {
                "role_counts": {"land": 0, "lead": 53},
                "objects": [],
                "bbox": None,
            },
            {
                "role_counts": {"package_pad": 52, "lead": 2},
                "summary": {
                    "package_pad_count": 52,
                    "terminal_package_pad_count": 51,
                    "lead_equivalent_count": 51,
                    "land_detail_lead_contact_count": 2,
                    "missing_canonical_views": [],
                },
                "objects": [],
                "bbox": None,
            },
            AlignmentOptions(),
        )

        lead_check = next(check for check in checks if check["name"] == "lead_count")
        self.assertEqual(lead_check["status"], "aligned")
        self.assertEqual(
            lead_check["selected_mapping"],
            "scan_lead_to_terminal_package_pad_plus_land_detail_lead_contacts",
        )
        self.assertEqual(
            lead_check["actual_role"],
            "terminal_package_pad_count_plus_land_detail_lead_contact_count",
        )
        self.assertEqual(lead_check["actual"], 53)
        self.assertEqual(lead_check["direct_actual"], 51)
        self.assertEqual(lead_check["terminal_package_pad_count"], 51)
        self.assertEqual(lead_check["supplemental_actual"], 2)

    def test_scan_land_can_use_terminal_land_count_when_thermal_pad_is_present(self) -> None:
        checks = compare_features(
            {
                "role_counts": {"land": 8, "lead": 0},
                "objects": [],
                "bbox": None,
            },
            {
                "role_counts": {"land": 9},
                "summary": {
                    "land_pad_count": 9,
                    "terminal_land_pad_count": 8,
                    "lead_equivalent_count": 0,
                    "missing_canonical_views": [],
                },
                "objects": [],
                "bbox": None,
            },
            AlignmentOptions(),
        )

        land_check = next(check for check in checks if check["name"] == "land_count")
        self.assertEqual(land_check["status"], "aligned")
        self.assertEqual(land_check["selected_mapping"], "scan_land_to_terminal_land_pad_count")
        self.assertEqual(land_check["actual_role"], "terminal_land_pad_count")
        self.assertEqual(land_check["raw_land_pad_count"], 9)

    def test_scan_land_can_include_land_detail_lead_contacts(self) -> None:
        checks = compare_features(
            {
                "role_counts": {"land": 13, "lead": 7},
                "objects": [],
                "bbox": None,
            },
            {
                "role_counts": {"package_pad": 7, "land": 11, "lead": 2},
                "summary": {
                    "package_pad_count": 7,
                    "lead_equivalent_count": 7,
                    "land_detail_lead_contact_count": 2,
                    "missing_canonical_views": [],
                },
                "objects": [],
                "bbox": None,
            },
            AlignmentOptions(),
        )

        land_check = next(check for check in checks if check["name"] == "land_count")
        lead_check = next(check for check in checks if check["name"] == "lead_count")
        self.assertEqual(land_check["status"], "aligned")
        self.assertEqual(lead_check["status"], "aligned")
        self.assertEqual(land_check["selected_mapping"], "scan_land_to_land_plus_land_detail_lead_contacts")
        self.assertEqual(land_check["supplemental_actual"], 2)

    def test_scan_land_can_include_terminal_land_plus_land_detail_lead_contacts(self) -> None:
        checks = compare_features(
            {
                "role_counts": {"land": 28, "lead": 20},
                "objects": [],
                "bbox": None,
            },
            {
                "role_counts": {"package_pad": 21, "land": 27, "lead": 2},
                "summary": {
                    "package_pad_count": 21,
                    "lead_equivalent_count": 20,
                    "land_pad_count": 27,
                    "terminal_land_pad_count": 26,
                    "thermal_land_pad_count": 1,
                    "land_detail_lead_contact_count": 2,
                    "missing_canonical_views": [],
                },
                "objects": [],
                "bbox": None,
            },
            AlignmentOptions(),
        )

        land_check = next(check for check in checks if check["name"] == "land_count")
        lead_check = next(check for check in checks if check["name"] == "lead_count")
        self.assertEqual(land_check["status"], "aligned")
        self.assertEqual(lead_check["status"], "aligned")
        self.assertEqual(land_check["selected_mapping"], "scan_land_to_terminal_land_plus_land_detail_lead_contacts")
        self.assertEqual(land_check["actual_role"], "terminal_land_pad_count_plus_land_detail_lead_contact_count")
        self.assertEqual(land_check["terminal_land_pad_count"], 26)
        self.assertEqual(land_check["supplemental_actual"], 2)

    def test_scan_land_can_include_land_detail_padlike_evidence_count(self) -> None:
        checks = compare_features(
            {
                "role_counts": {"land": 13, "lead": 7},
                "objects": [],
                "bbox": None,
            },
            {
                "role_counts": {"package_pad": 7, "land": 11},
                "summary": {
                    "package_pad_count": 7,
                    "lead_equivalent_count": 7,
                    "land_detail_lead_contact_count": 1,
                    "land_detail_padlike_count_candidates": [
                        {
                            "source": "package_graph_evidence",
                            "candidate_type": "land_detail_padlike_count",
                            "count": 2,
                            "raw_view": "land_detail",
                            "graph_path": "/tmp/land_detail.package_graph.json",
                        }
                    ],
                    "missing_canonical_views": [],
                },
                "objects": [],
                "bbox": None,
            },
            AlignmentOptions(),
        )

        land_check = next(check for check in checks if check["name"] == "land_count")
        lead_check = next(check for check in checks if check["name"] == "lead_count")
        self.assertEqual(land_check["status"], "aligned")
        self.assertEqual(lead_check["status"], "aligned")
        self.assertEqual(land_check["selected_mapping"], "scan_land_to_land_plus_land_detail_padlike_evidence")
        self.assertEqual(land_check["direct_actual"], 11)
        self.assertEqual(land_check["supplemental_actual"], 2)
        self.assertEqual(land_check["candidate"]["graph_path"], "/tmp/land_detail.package_graph.json")

    def test_scan_land_can_include_supplemental_land_terminal_padlike_evidence_count(self) -> None:
        checks = compare_features(
            {
                "role_counts": {"land": 73, "lead": 52},
                "objects": [],
                "bbox": None,
            },
            {
                "role_counts": {"package_pad": 52, "land": 58},
                "summary": {
                    "package_pad_count": 52,
                    "terminal_package_pad_count": 51,
                    "lead_equivalent_count": 51,
                    "supplemental_land_padlike_count_candidates": [
                        {
                            "source": "package_graph_evidence",
                            "candidate_type": "supplemental_land_terminal_padlike_count",
                            "count": 15,
                            "raw_pad_like_count": 16,
                            "thermal_pad_like_count": 1,
                            "graph_path": "/tmp/supplemental_land.package_graph.json",
                        }
                    ],
                    "missing_canonical_views": [],
                },
                "objects": [],
                "bbox": None,
            },
            AlignmentOptions(),
        )

        land_check = next(check for check in checks if check["name"] == "land_count")
        lead_check = next(check for check in checks if check["name"] == "lead_count")
        self.assertEqual(land_check["status"], "aligned")
        self.assertEqual(lead_check["status"], "aligned")
        self.assertEqual(
            land_check["selected_mapping"],
            "scan_land_to_land_plus_supplemental_land_terminal_padlike_evidence",
        )
        self.assertEqual(land_check["direct_actual"], 58)
        self.assertEqual(land_check["supplemental_actual"], 15)
        self.assertEqual(land_check["candidate"]["graph_path"], "/tmp/supplemental_land.package_graph.json")
        self.assertEqual(lead_check["selected_mapping"], "scan_lead_to_raw_package_pad_count_including_thermal")
        self.assertEqual(lead_check["actual_role"], "package_pad_count")

    def test_scan_group_count_candidate_can_align_multilayer_land_count(self) -> None:
        checks = compare_features(
            {
                "role_counts": {"land": 13, "lead": 7},
                "count_candidates": {
                    "land": [
                        {
                            "source": "GroupItems",
                            "candidate_type": "group_raw_object_count",
                            "group_index": 2,
                            "count": 11,
                        }
                    ]
                },
                "objects": [],
                "bbox": None,
            },
            {
                "role_counts": {"package_pad": 7, "land": 11},
                "summary": {
                    "package_pad_count": 7,
                    "lead_equivalent_count": 7,
                    "land_detail_lead_contact_count": 0,
                    "missing_canonical_views": [],
                },
                "objects": [],
                "bbox": None,
            },
            AlignmentOptions(),
        )

        land_check = next(check for check in checks if check["name"] == "land_count")
        lead_check = next(check for check in checks if check["name"] == "lead_count")
        self.assertEqual(land_check["status"], "aligned")
        self.assertEqual(lead_check["status"], "aligned")
        self.assertEqual(land_check["selected_mapping"], "scan_group_land_count_candidate")
        self.assertEqual(land_check["direct_expected"], 13)
        self.assertEqual(land_check["expected"], 11)
        self.assertEqual(land_check["candidate"]["group_index"], 2)

    def test_scan_group_land_count_can_match_terminal_land_count(self) -> None:
        checks = compare_features(
            {
                "role_counts": {"land": 7, "lead": 6},
                "count_candidates": {
                    "land": [
                        {
                            "source": "GroupItems",
                            "candidate_type": "group_raw_object_count",
                            "group_index": 1,
                            "count": 6,
                        }
                    ],
                    "lead": [
                        {
                            "source": "GroupItems",
                            "candidate_type": "group_raw_object_count",
                            "group_index": 0,
                            "count": 6,
                        }
                    ],
                },
                "objects": [],
                "bbox": None,
            },
            {
                "role_counts": {"package_pad": 8, "land": 8},
                "summary": {
                    "package_pad_count": 8,
                    "lead_equivalent_count": 6,
                    "land_pad_count": 8,
                    "terminal_land_pad_count": 6,
                    "thermal_land_pad_count": 2,
                    "missing_canonical_views": [],
                },
                "objects": [],
                "bbox": None,
            },
            AlignmentOptions(),
        )

        land_check = next(check for check in checks if check["name"] == "land_count")
        lead_check = next(check for check in checks if check["name"] == "lead_count")
        self.assertEqual(land_check["status"], "aligned")
        self.assertEqual(land_check["actual"], 6)
        self.assertEqual(land_check["actual_role"], "terminal_land_pad_count")
        self.assertEqual(land_check["selected_mapping"], "scan_group_land_count_to_terminal_land_pad_count")
        self.assertEqual(land_check["direct_expected"], 7)
        self.assertEqual(land_check["direct_actual"], 8)
        self.assertEqual(land_check["candidate"]["group_index"], 1)
        self.assertEqual(lead_check["status"], "aligned")

    def test_group_lead_mapping_is_preserved_when_land_still_mismatches(self) -> None:
        checks = compare_features(
            {
                "role_counts": {"land": 99, "lead": 7},
                "count_candidates": {
                    "lead": [
                        {
                            "source": "GroupItems",
                            "candidate_type": "group_raw_object_count",
                            "group_index": 0,
                            "count": 6,
                        }
                    ]
                },
                "objects": [],
                "bbox": None,
            },
            {
                "role_counts": {"package_pad": 6, "land": 0},
                "summary": {
                    "package_pad_count": 6,
                    "lead_equivalent_count": 6,
                    "missing_canonical_views": ["land"],
                },
                "objects": [],
                "bbox": None,
            },
            AlignmentOptions(),
        )

        land_check = next(check for check in checks if check["name"] == "land_count")
        lead_check = next(check for check in checks if check["name"] == "lead_count")
        self.assertEqual(land_check["status"], "mismatch")
        self.assertEqual(land_check["selected_mapping"], "direct")
        self.assertEqual(lead_check["status"], "aligned")
        self.assertEqual(lead_check["selected_mapping"], "scan_group_lead_count_candidate")
        self.assertEqual(lead_check["actual_role"], "lead_equivalent_count")
        self.assertEqual(lead_check["direct_expected"], 7)

    def test_scan_lead_can_align_to_land_count_when_package_has_auxiliary_pads(self) -> None:
        checks = compare_features(
            {
                "role_counts": {"land": 5, "lead": 5},
                "objects": [],
                "bbox": None,
            },
            {
                "role_counts": {"package_pad": 6, "land": 5},
                "summary": {
                    "package_pad_count": 6,
                    "lead_equivalent_count": 6,
                    "land_detail_lead_contact_count": 0,
                    "missing_canonical_views": [],
                },
                "objects": [],
                "bbox": None,
            },
            AlignmentOptions(),
        )

        lead_check = next(check for check in checks if check["name"] == "lead_count")
        self.assertEqual(lead_check["status"], "aligned")
        self.assertEqual(lead_check["selected_mapping"], "scan_lead_to_land_count")
        self.assertEqual(lead_check["direct_actual"], 6)

    def test_scan_lead_can_align_to_land_count_when_package_view_is_missing(self) -> None:
        checks = compare_features(
            {
                "role_counts": {"land": 2, "lead": 2},
                "objects": [],
                "bbox": None,
            },
            {
                "role_counts": {"land": 2},
                "summary": {
                    "package_pad_count": 0,
                    "lead_equivalent_count": 0,
                    "land_detail_lead_contact_count": 0,
                    "missing_canonical_views": ["bottom"],
                },
                "objects": [],
                "bbox": None,
            },
            AlignmentOptions(),
        )

        land_check = next(check for check in checks if check["name"] == "land_count")
        lead_check = next(check for check in checks if check["name"] == "lead_count")
        self.assertEqual(land_check["status"], "aligned")
        self.assertEqual(lead_check["status"], "aligned")
        self.assertEqual(lead_check["selected_mapping"], "scan_lead_to_land_count_missing_package_view")
        self.assertEqual(lead_check["actual_role"], "land_count")
        self.assertEqual(lead_check["direct_actual"], 0)

    def test_land_count_mismatch_with_land_view_gets_specific_stage_hint(self) -> None:
        checks = compare_features(
            {
                "role_counts": {"land": 3, "lead": 0},
                "objects": [],
                "bbox": None,
            },
            {
                "role_counts": {"land": 2},
                "summary": {"lead_equivalent_count": 0, "missing_canonical_views": []},
                "objects": [],
                "bbox": None,
            },
            AlignmentOptions(),
        )

        land_check = next(check for check in checks if check["name"] == "land_count")
        self.assertEqual(land_check["status"], "mismatch")
        self.assertEqual(land_check["stage_hint"], "package_graph_land_reconstruction_count_mismatch")

    def test_lead_count_mismatch_with_missing_bottom_stays_multiview_source_issue(self) -> None:
        checks = compare_features(
            {
                "role_counts": {"land": 0, "lead": 2},
                "objects": [],
                "bbox": None,
            },
            {
                "role_counts": {"package_pad": 3},
                "summary": {
                    "lead_equivalent_count": 3,
                    "missing_canonical_views": ["bottom"],
                    "source_views": ["bottom"],
                },
                "objects": [],
                "bbox": None,
            },
            AlignmentOptions(),
        )

        lead_check = next(check for check in checks if check["name"] == "lead_count")
        self.assertEqual(lead_check["status"], "mismatch")
        self.assertEqual(lead_check["stage_hint"], "multiview_missing_package_pad_source_view")

    def test_missing_bottom_source_view_is_data_coverage_issue(self) -> None:
        checks = compare_features(
            {
                "role_counts": {"land": 0, "lead": 2},
                "objects": [],
                "bbox": None,
            },
            {
                "role_counts": {"package_pad": 3},
                "summary": {
                    "lead_equivalent_count": 3,
                    "missing_canonical_views": ["bottom"],
                    "source_views": ["top"],
                },
                "objects": [],
                "bbox": None,
            },
            AlignmentOptions(),
        )

        lead_check = next(check for check in checks if check["name"] == "lead_count")
        self.assertEqual(lead_check["status"], "mismatch")
        self.assertEqual(lead_check["stage_hint"], "data_missing_package_pad_source_view")

    def test_missing_land_source_view_is_data_coverage_issue(self) -> None:
        checks = compare_features(
            {
                "role_counts": {"land": 2, "lead": 0},
                "objects": [],
                "bbox": None,
            },
            {
                "role_counts": {"package_pad": 3},
                "summary": {
                    "package_pad_count": 3,
                    "lead_equivalent_count": 3,
                    "missing_canonical_views": ["land"],
                    "source_views": ["bottom"],
                },
                "objects": [],
                "bbox": None,
            },
            AlignmentOptions(),
        )

        land_check = next(check for check in checks if check["name"] == "land_count")
        self.assertEqual(land_check["status"], "mismatch")
        self.assertEqual(land_check["stage_hint"], "data_missing_land_view")

    def test_missing_land_view_can_align_scan_land_to_package_pad_count_with_mapping(self) -> None:
        checks = compare_features(
            {
                "role_counts": {"land": 4, "lead": 4},
                "objects": [],
                "bbox": None,
            },
            {
                "role_counts": {"package_pad": 4},
                "summary": {
                    "package_pad_count": 4,
                    "lead_equivalent_count": 4,
                    "land_detail_lead_contact_count": 0,
                    "missing_canonical_views": ["land"],
                },
                "objects": [],
                "bbox": None,
            },
            AlignmentOptions(),
        )

        land_check = next(check for check in checks if check["name"] == "land_count")
        self.assertEqual(land_check["status"], "aligned")
        self.assertEqual(land_check["selected_mapping"], "scan_land_to_package_pad_missing_land_view")
        self.assertEqual(land_check["actual_role"], "package_pad_count")
        self.assertEqual(land_check["direct_actual"], 0)

    def test_parse_scan_result_deduplicates_multilayer_land_data_by_center(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scan = Path(tmp) / "ScanResultFormat.txt"
            scan.write_text(
                json.dumps(
                    {
                        "Object": [
                            {
                                "ID": 1,
                                "LandData": {},
                                "PointList": [
                                    {"PointX": 0.0, "PointY": 0.0},
                                    {"PointX": 1.0, "PointY": 0.0},
                                    {"PointX": 1.0, "PointY": 1.0},
                                    {"PointX": 0.0, "PointY": 1.0},
                                ],
                            },
                            {
                                "ID": 2,
                                "LandData": {},
                                "PointList": [
                                    {"PointX": 0.001, "PointY": 0.0},
                                    {"PointX": 1.001, "PointY": 0.0},
                                    {"PointX": 1.001, "PointY": 1.0},
                                    {"PointX": 0.001, "PointY": 1.0},
                                ],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            parsed = parse_scan_result(scan, dedupe_center_tol=0.01)

            self.assertEqual(parsed["summary"]["raw_role_counts"], {"land": 2})
            self.assertEqual(parsed["summary"]["role_counts"], {"land": 1})

    def test_parse_scan_result_does_not_double_count_same_bbox_land_and_lead(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scan = Path(tmp) / "ScanResultFormat.txt"
            scan.write_text(
                json.dumps(
                    {
                        "Object": [
                            {
                                "ID": 1,
                                "LeadData": {},
                                "PointList": [
                                    {"PointX": 2.0, "PointY": 0.0},
                                    {"PointX": 2.2, "PointY": 0.0},
                                    {"PointX": 2.2, "PointY": 0.2},
                                    {"PointX": 2.0, "PointY": 0.2},
                                ],
                            },
                            {
                                "ID": 2,
                                "LeadData": {},
                                "PointList": [
                                    {"PointX": 0.0, "PointY": 0.0},
                                    {"PointX": 1.0, "PointY": 0.0},
                                    {"PointX": 1.0, "PointY": 1.0},
                                    {"PointX": 0.0, "PointY": 1.0},
                                ],
                            },
                            {
                                "ID": 3,
                                "LandData": {},
                                "PointList": [
                                    {"PointX": 0.0, "PointY": 0.0},
                                    {"PointX": 1.0, "PointY": 0.0},
                                    {"PointX": 1.0, "PointY": 1.0},
                                    {"PointX": 0.0, "PointY": 1.0},
                                ],
                            },
                        ],
                        "GroupItems": [
                            {
                                "FirstMartixPinIDs": [[1], [2], [3]],
                                "FirstMartixQX": 3,
                                "FirstMartixQY": 1,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            parsed = parse_scan_result(scan, dedupe_center_tol=0.01)

            self.assertEqual(parsed["summary"]["raw_role_counts"], {"land": 1, "lead": 2})
            self.assertEqual(parsed["summary"]["role_counts"], {"lead": 1})

    def test_parse_scan_result_without_groups_keeps_asymmetric_land_and_lead_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scan = Path(tmp) / "ScanResultFormat.txt"
            scan.write_text(
                json.dumps(
                    {
                        "Object": [
                            {
                                "ID": 1,
                                "LeadData": {},
                                "PointList": rectangle_points(0.000, 0.000, 0.014, 0.026),
                            },
                            {
                                "ID": 2,
                                "LeadData": {},
                                "PointList": rectangle_points(0.022, 0.000, 0.061, 0.026),
                            },
                            {
                                "ID": 3,
                                "LandData": {},
                                "PointList": rectangle_points(-0.005, -0.001, 0.015, 0.027),
                            },
                            {
                                "ID": 4,
                                "LandData": {},
                                "PointList": rectangle_points(0.021, -0.001, 0.062, 0.027),
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            parsed = parse_scan_result(scan, dedupe_center_tol=0.01)

            self.assertEqual(parsed["summary"]["raw_role_counts"], {"land": 2, "lead": 2})
            self.assertEqual(parsed["summary"]["role_counts"], {"land": 2, "lead": 2})

    def test_parse_scan_result_keeps_same_center_different_size_land_and_lead(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scan = Path(tmp) / "ScanResultFormat.txt"
            scan.write_text(
                json.dumps(
                    {
                        "Object": [
                            {
                                "ID": 1,
                                "LandData": {},
                                "PointList": rectangle_points(0.0, 0.0, 0.073, 0.043),
                            },
                            {
                                "ID": 2,
                                "LandData": {},
                                "PointList": rectangle_points(0.091, 0.002, 0.126, 0.041),
                            },
                            {
                                "ID": 3,
                                "LeadData": {},
                                "PointList": rectangle_points(0.0095, 0.0025, 0.0705, 0.0405),
                            },
                            {
                                "ID": 4,
                                "LeadData": {},
                                "PointList": rectangle_points(0.0845, 0.0065, 0.1165, 0.0365),
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            parsed = parse_scan_result(scan, dedupe_center_tol=0.01)

            self.assertEqual(parsed["summary"]["raw_role_counts"], {"land": 2, "lead": 2})
            self.assertEqual(parsed["summary"]["role_counts"], {"land": 2, "lead": 2})

    def test_bbox_alignment_uses_semantic_candidates(self) -> None:
        checks = compare_features(
            {
                "role_counts": {"lead": 0, "land": 0},
                "objects": [],
                "bbox": [0.0, 0.0, 10.0, 2.0],
                "bbox_candidates": {
                    "all": [0.0, 0.0, 10.0, 2.0],
                    "shape": [0.0, 0.0, 4.0, 2.0],
                },
            },
            {
                "role_counts": {},
                "summary": {"lead_equivalent_count": 0, "missing_canonical_views": []},
                "objects": [],
                "bbox": [0.0, 0.0, 3.0, 3.0],
                "bbox_candidates": {
                    "all": [0.0, 0.0, 3.0, 3.0],
                    "outline": [0.0, 0.0, 8.0, 4.0],
                },
            },
            AlignmentOptions(),
        )

        bbox_check = next(check for check in checks if check["name"] == "bbox_aspect")
        self.assertEqual(bbox_check["status"], "aligned")
        self.assertEqual(bbox_check["selected_pair"], {"gt": "shape", "graph": "outline"})

    def test_bbox_alignment_can_compare_scan_all_to_graph_package_footprint(self) -> None:
        checks = compare_features(
            {
                "role_counts": {"lead": 0, "land": 0},
                "objects": [],
                "bbox": [0.0, 0.0, 4.0, 1.0],
                "bbox_candidates": {
                    "all": [0.0, 0.0, 4.0, 1.0],
                },
            },
            {
                "role_counts": {},
                "summary": {"lead_equivalent_count": 0, "missing_canonical_views": []},
                "objects": [],
                "bbox": [0.0, 0.0, 10.0, 10.0],
                "bbox_candidates": {
                    "all": [0.0, 0.0, 10.0, 10.0],
                    "package": [0.0, 0.0, 8.0, 2.0],
                },
            },
            AlignmentOptions(),
        )

        bbox_check = next(check for check in checks if check["name"] == "bbox_aspect")
        self.assertEqual(bbox_check["status"], "aligned")
        self.assertEqual(bbox_check["selected_pair"], {"gt": "all", "graph": "package"})

    def test_alignment_scores_include_per_pad_layout_and_dimension_conflicts(self) -> None:
        scores = alignment_scores(
            {
                "features": {
                    "objects": [
                        {"role": "land", "bbox": [0.0, 0.0, 1.0, 1.0]},
                        {"role": "land", "bbox": [2.0, 0.0, 3.0, 1.0]},
                        {"role": "lead", "bbox": [0.0, 2.0, 1.0, 3.0]},
                    ]
                }
            },
            {
                "dimensions": [{"id": 1}, {"id": 2}],
                "conflicts": [{"type": "dimension_value_conflict"}],
                "package_pads": [{"bbox": [0.0, 2.0, 1.0, 3.0]}],
                "land_pads": [{"bbox": [0.0, 0.0, 1.0, 1.0]}, {"bbox": [2.5, 0.0, 3.5, 1.0]}],
                "lead_contacts": [],
            },
            [
                {"role": "land", "bbox": [0.0, 0.0, 1.0, 1.0]},
                {"role": "land", "bbox": [2.5, 0.0, 3.5, 1.0]},
                {"role": "package_pad", "bbox": [0.0, 2.0, 1.0, 3.0]},
            ],
            [
                {"name": "land_count", "status": "aligned"},
                {"name": "lead_count", "status": "mismatch"},
            ],
        )

        self.assertAlmostEqual(scores["land_pad_iou_score"], 0.666667, places=6)
        self.assertAlmostEqual(scores["lead_pad_iou_score"], 1.0, places=6)
        self.assertAlmostEqual(scores["pad_layout_score"], 0.833333, places=6)
        self.assertEqual(scores["dimension_mismatch_count"], 1)
        self.assertEqual(scores["dimension_count"], 2)
        self.assertAlmostEqual(scores["dimension_value_score"], 0.5, places=6)
        self.assertTrue(scores["land_pad_count_match"])
        self.assertTrue(scores["lead_count_match"])
        self.assertEqual(scores["input_count_checks"], {"land_count": True, "lead_count": False})
        self.assertEqual(scores["count_checks"], {"land_count": True, "lead_count": True})

    def test_alignment_scores_use_selected_scan_group_gt_subset_for_pad_iou(self) -> None:
        scores = alignment_scores(
            {
                "features": {
                    "objects": [
                        {"id": 1, "role": "land", "bbox": [0.0, 0.0, 1.0, 1.0]},
                        {"id": 2, "role": "land", "bbox": [2.0, 0.0, 3.0, 1.0]},
                        {"id": 3, "role": "land", "bbox": [4.0, 0.0, 5.0, 1.0]},
                        {"id": 4, "role": "land", "bbox": [6.0, 0.0, 7.0, 1.0]},
                        {"id": 5, "role": "land", "bbox": [100.0, 100.0, 101.0, 101.0]},
                    ]
                }
            },
            {
                "dimensions": [],
                "conflicts": [],
                "package_pads": [],
                "land_pads": [
                    {"bbox": [0.0, 0.0, 1.0, 1.0]},
                    {"bbox": [2.0, 0.0, 3.0, 1.0]},
                    {"bbox": [4.0, 0.0, 5.0, 1.0]},
                    {"bbox": [6.0, 0.0, 7.0, 1.0]},
                ],
                "lead_contacts": [],
            },
            [
                {"role": "land", "bbox": [0.0, 0.0, 1.0, 1.0]},
                {"role": "land", "bbox": [2.0, 0.0, 3.0, 1.0]},
                {"role": "land", "bbox": [4.0, 0.0, 5.0, 1.0]},
                {"role": "land", "bbox": [6.0, 0.0, 7.0, 1.0]},
            ],
            [
                {
                    "name": "land_count",
                    "expected": 4,
                    "actual": 4,
                    "status": "aligned",
                    "selected_mapping": "scan_group_land_count_candidate",
                    "candidate": {"count": 4, "object_ids": [1, 2, 3, 4]},
                }
            ],
        )

        self.assertEqual(scores["gt_land_count"], 4)
        self.assertAlmostEqual(scores["land_pad_iou_score"], 1.0, places=6)

    def test_alignment_scores_use_selected_result_counts_for_final_count_status(self) -> None:
        scores = alignment_scores(
            {
                "features": {
                    "objects": [
                        {"id": 1, "role": "lead", "bbox": [0.0, 0.0, 1.0, 1.0]},
                        {"id": 2, "role": "lead", "bbox": [2.0, 0.0, 3.0, 1.0]},
                    ]
                }
            },
            {"dimensions": [], "conflicts": [], "package_pads": [], "land_pads": [], "lead_contacts": []},
            [
                {"role": "package_pad", "bbox": [0.0, 0.0, 1.0, 1.0]},
                {"role": "package_pad", "bbox": [2.0, 0.0, 3.0, 1.0]},
            ],
            [
                {
                    "name": "lead_count",
                    "expected": 2,
                    "actual": 4,
                    "status": "mismatch",
                    "reason": "lead_count_mismatch",
                }
            ],
        )

        self.assertEqual(scores["input_count_checks"], {"lead_count": False})
        self.assertEqual(scores["count_checks"], {"lead_count": True})
        self.assertTrue(scores["lead_count_match"])

    def test_alignment_scores_preserve_semantic_input_count_alignment(self) -> None:
        scores = alignment_scores(
            {
                "features": {
                    "objects": [
                        {"id": 1, "role": "lead", "bbox": [0.0, 0.0, 1.0, 1.0]},
                        {"id": 2, "role": "lead", "bbox": [2.0, 0.0, 3.0, 1.0]},
                    ]
                }
            },
            {"dimensions": [], "conflicts": [], "package_pads": [], "land_pads": [], "lead_contacts": []},
            [{"role": "package_pad", "bbox": [0.0, 0.0, 1.0, 1.0]}],
            [
                {
                    "name": "lead_count",
                    "expected": 2,
                    "actual": 2,
                    "status": "aligned",
                    "reason": "lead_count_mismatch",
                    "selected_mapping": "scan_group_lead_count_candidate",
                }
            ],
        )

        self.assertEqual(scores["input_count_checks"], {"lead_count": True})
        self.assertEqual(scores["count_checks"], {"lead_count": True})
        self.assertTrue(scores["lead_count_match"])

    def test_alignment_scores_penalize_scan_result_format_geometry_fallback(self) -> None:
        scores = alignment_scores(
            {
                "features": {
                    "objects": [
                        {"role": "land", "bbox": [0.0, 0.0, 1.0, 1.0]},
                        {"role": "lead", "bbox": [2.0, 0.0, 3.0, 1.0]},
                    ]
                }
            },
            {
                "dimensions": [],
                "conflicts": [],
                "package_pads": [{"bbox": [2.0, 0.0, 3.0, 1.0], "source_type": "scan_result_format"}],
                "land_pads": [{"bbox": [0.0, 0.0, 1.0, 1.0], "source_type": "scan_result_format"}],
                "lead_contacts": [],
            },
            [
                {
                    "role": "land",
                    "bbox": [0.0, 0.0, 1.0, 1.0],
                    "source_type": "scan_result_format",
                },
                {
                    "role": "package_pad",
                    "bbox": [2.0, 0.0, 3.0, 1.0],
                    "source_type": "scan_result_format",
                },
            ],
            [
                {"name": "land_count", "status": "aligned"},
                {"name": "lead_count", "status": "aligned"},
            ],
        )

        self.assertAlmostEqual(scores["land_pad_iou_score"], 1.0, places=6)
        self.assertAlmostEqual(scores["lead_pad_iou_score"], 1.0, places=6)
        self.assertEqual(scores["scan_result_fallback_object_count"], 2)
        self.assertEqual(scores["source_independence_score"], 0.0)
        self.assertEqual(scores["quality_score"], 0.0)

        details = score_diagnostic_details(scores, {"summary": {}})

        self.assertIn("scan_result_geometry_fallback", {item["reason"] for item in details})

    def test_score_diagnostics_classify_aligned_low_score_sources(self) -> None:
        details = score_diagnostic_details(
            {
                "overall_score": 0.6,
                "gt_land_count": 1,
                "land_iou": None,
                "lead_iou": 0.0,
                "land_pad_iou_score": None,
                "lead_pad_iou_score": 0.0,
                "pad_layout_score": 0.0,
                "dimension_value_score": 0.5,
            },
            {
                "summary": {
                    "missing_canonical_views": ["land"],
                    "source_selection": {
                        "package_pads": {
                            "source_type": "scan_result_format",
                            "used_fallback": True,
                        }
                    },
                }
            },
        )

        reasons = {item["reason"] for item in details}
        self.assertIn("missing_land_view_for_land_iou", reasons)
        self.assertIn("low_lead_pad_iou", reasons)
        self.assertIn("low_pad_layout_score", reasons)
        self.assertIn("dimension_value_conflict", reasons)
        lead_detail = next(item for item in details if item["reason"] == "low_lead_pad_iou")
        self.assertEqual(lead_detail["stage_hint"], "low_score_scan_result_package_pad_alignment")
        self.assertEqual(
            lead_detail["objective_error_sources"],
            ["multiview_alignment", "scan_result_parsing"],
        )

    def test_score_diagnostics_skip_missing_land_when_gt_has_no_land_objects(self) -> None:
        details = score_diagnostic_details(
            {
                "overall_score": 0.99,
                "gt_land_count": 0,
                "land_iou": None,
                "land_pad_iou_score": None,
                "lead_iou": 1.0,
                "lead_pad_iou_score": 0.99,
            },
            {"summary": {"missing_canonical_views": ["land"]}},
        )

        self.assertNotIn("missing_land_view_for_land_iou", {item["reason"] for item in details})

    def test_score_diagnostics_skip_low_risk_scores(self) -> None:
        details = score_diagnostic_details(
            {"overall_score": 0.95, "lead_pad_iou_score": 0.95},
            {"summary": {}},
        )

        self.assertEqual(details, [])

    def test_review_bucket_separates_evidence_limited_from_algorithm_evaluable(self) -> None:
        algorithm_summary = {
            "alignment_scores": {"overall_score": 0.9, "lead_pad_iou_score": 0.2},
            "objective_error_sources": [],
            "score_objective_error_sources": ["multiview_alignment"],
            "stage_hints": [],
            "score_stage_hints": ["low_score_multiview_dimension_conflict"],
        }
        evidence_limited_summary = {
            "alignment_scores": {"overall_score": 0.9, "lead_pad_iou_score": 0.2},
            "objective_error_sources": [],
            "score_objective_error_sources": ["multiview_alignment"],
            "stage_hints": [],
            "score_stage_hints": [
                "low_score_multiview_package_pad_fallback_geometry",
                "low_score_multiview_partial_lead_detail_layout",
            ],
        }

        self.assertEqual(review_bucket_for_summary(algorithm_summary), "algorithm_evaluable")
        self.assertEqual(review_bucket_for_summary(evidence_limited_summary), "evidence_limited")

    def test_review_bucket_prioritizes_explicit_package_graph_reconstruction_hint(self) -> None:
        summary = {
            "alignment_scores": {"overall_score": 0.9, "lead_pad_iou_score": 0.05},
            "objective_error_sources": [],
            "score_objective_error_sources": [
                "model_prediction",
                "multiview_alignment",
                "package_graph_reconstruction",
            ],
            "stage_hints": [],
            "score_stage_hints": [
                "low_score_multiview_partial_lead_detail_layout",
                "low_score_package_graph_package_pad_geometry",
            ],
        }

        self.assertEqual(review_bucket_for_summary(summary), "upstream_prediction_or_reconstruction")

    def test_review_bucket_treats_sparse_primary_pad_fallback_with_missing_support_views_as_data_issue(self) -> None:
        summary = {
            "alignment_scores": {"overall_score": 0.9, "lead_pad_iou_score": 0.17},
            "objective_error_sources": [],
            "score_objective_error_sources": ["multiview_alignment"],
            "stage_hints": [],
            "score_stage_hints": [
                "low_score_multiview_package_pad_fallback_geometry",
                "low_score_multiview_partial_lead_detail_layout",
            ],
            "graph": {
                "missing_canonical_views": ["land", "lead_detail"],
            },
        }

        self.assertEqual(review_bucket_for_summary(summary), "data_or_gt_issue")

    def test_known_data_issue_manifest_marks_matching_source_file_as_data_issue(self) -> None:
        summary = {
            "part_number": "PART",
            "alignment_scores": {"overall_score": 0.9, "lead_pad_iou_score": 0.05},
            "score_objective_error_sources": ["package_graph_reconstruction"],
            "score_stage_hints": ["low_score_package_graph_land_geometry"],
            "graph": {
                "source_selection": {
                    "land_pads": {
                        "graph_path": "/tmp/PART/extract_image/page1_Top_2.package_graph.json",
                    }
                }
            },
        }

        attach_known_data_issues(
            summary,
            [
                {
                    "part_number": "PART",
                    "file": "page1_Top_2.json",
                    "issue_type": "wrong_view_label",
                },
                {
                    "part_number": "PART",
                    "file": "other.json",
                    "issue_type": "wrong_view_label",
                },
            ],
        )

        self.assertEqual(len(summary["known_data_issues"]), 1)
        self.assertEqual(summary["known_data_issue_types"], ["wrong_view_label"])
        self.assertEqual(review_bucket_for_summary(summary), "data_or_gt_issue")

    def test_score_diagnostics_do_not_average_away_low_pad_layout(self) -> None:
        alignment_scores_payload = {
            "overall_score": 0.92,
            "outline_iou": 1.0,
            "land_iou": 1.0,
            "lead_iou": 1.0,
            "land_pad_iou_score": 0.28,
            "lead_pad_iou_score": 0.59,
            "pad_layout_score": 0.43,
            "dimension_value_score": 1.0,
        }

        details = score_diagnostic_details(alignment_scores_payload, {"summary": {}})

        self.assertAlmostEqual(alignment_quality_score(alignment_scores_payload), 0.28)
        self.assertEqual(
            {item["reason"] for item in details},
            {"low_land_pad_iou", "low_lead_pad_iou", "low_pad_layout_score"},
        )

    def test_score_diagnostics_explain_medium_risk_component_scores(self) -> None:
        details = score_diagnostic_details(
            {
                "overall_score": 0.96,
                "land_pad_iou_score": 0.53,
                "lead_pad_iou_score": 0.88,
                "pad_layout_score": 0.70,
            },
            {"summary": {}},
        )

        self.assertEqual(
            {item["reason"] for item in details},
            {"low_land_pad_iou", "borderline_lead_pad_iou", "low_pad_layout_score"},
        )
        land_detail = next(item for item in details if item["reason"] == "low_land_pad_iou")
        self.assertEqual(land_detail["threshold"], 0.8)

    def test_score_diagnostics_flag_duplicate_lead_geometry_sources(self) -> None:
        gt = {
            "features": {
                "objects": [
                    {"role": "lead", "bbox": [0.0, 0.0, 1.0, 1.0]},
                    {"role": "lead", "bbox": [2.0, 0.0, 3.0, 1.0]},
                ]
            }
        }
        canonical = {
            "package_pads": [{"bbox": [0.0, 0.0, 1.0, 1.0]}, {"bbox": [2.0, 0.0, 3.0, 1.0]}],
            "lead_contacts": [{"bbox": [0.0, 0.0, 1.0, 1.0]}, {"bbox": [2.0, 0.0, 3.0, 1.0]}],
            "land_pads": [],
            "dimensions": [],
            "summary": {},
        }
        result_objects = [
            {"role": "package_pad", "bbox": [0.0, 0.0, 0.8, 1.0]},
            {"role": "package_pad", "bbox": [2.0, 0.0, 2.8, 1.0]},
            {"role": "lead", "bbox": [0.2, 0.0, 1.0, 1.0]},
            {"role": "lead", "bbox": [2.2, 0.0, 3.0, 1.0]},
        ]

        scores = alignment_scores(gt, canonical, result_objects, [])
        details = score_diagnostic_details(scores, canonical)
        detail = next(item for item in details if item["reason"] == "duplicate_lead_geometry_sources")

        self.assertAlmostEqual(scores["lead_pad_iou_package_only"], 0.8)
        self.assertAlmostEqual(scores["lead_pad_iou_lead_contact_only"], 0.8)
        self.assertAlmostEqual(scores["lead_pad_iou_score"], 0.4)
        self.assertEqual(detail["stage_hint"], "low_score_multiview_duplicate_lead_geometry_sources")

    def test_score_diagnostics_flag_package_pad_proxy_size_mismatch(self) -> None:
        details = score_diagnostic_details(
            {
                "gt_lead_count": 2,
                "result_package_pad_count": 2,
                "result_lead_contact_count": 0,
                "land_pad_iou_score": 0.99,
                "lead_pad_iou_package_only": 0.48,
                "lead_pad_iou_score": 0.48,
                "pad_layout_score": 0.74,
            },
            {
                "package_pads": [{"bbox": [0.0, 0.0, 1.0, 1.0]}, {"bbox": [2.0, 0.0, 3.0, 1.0]}],
                "lead_contacts": [],
                "summary": {},
            },
        )

        detail = next(item for item in details if item["reason"] == "package_pad_proxy_size_mismatch")

        self.assertEqual(detail["metric"], "lead_pad_iou_package_only")
        self.assertEqual(detail["value"], 0.48)
        self.assertEqual(detail["threshold"], 0.8)
        self.assertEqual(detail["stage_hint"], "low_score_package_graph_package_pad_geometry")
        self.assertEqual(detail["gt_lead_count"], 2)
        self.assertEqual(detail["result_package_pad_count"], 2)
        self.assertEqual(detail["result_lead_contact_count"], 0)
        self.assertEqual(detail["canonical_lead_contact_count"], 0)

    def test_package_pad_proxy_mismatch_is_evidence_limited_when_lateral_leads_are_excluded(self) -> None:
        details = score_diagnostic_details(
            {
                "gt_lead_count": 2,
                "result_package_pad_count": 2,
                "result_lead_contact_count": 0,
                "land_pad_iou_score": 0.99,
                "lead_pad_iou_package_only": 0.48,
                "lead_pad_iou_score": 0.48,
                "pad_layout_score": 0.74,
            },
            {
                "package_pads": [{"bbox": [0.0, 0.0, 1.0, 1.0]}, {"bbox": [2.0, 0.0, 3.0, 1.0]}],
                "lead_contacts": [{"bbox": [0.1, 0.0, 0.9, 1.0]}, {"bbox": [2.1, 0.0, 2.9, 1.0]}],
                "summary": {"lead_contact_count": 2},
            },
        )

        detail = next(item for item in details if item["reason"] == "package_pad_proxy_size_mismatch")
        summary = {
            "alignment_scores": {"overall_score": 0.9, "lead_pad_iou_score": 0.48},
            "objective_error_sources": [],
            "score_objective_error_sources": detail["objective_error_sources"],
            "stage_hints": [],
            "score_stage_hints": [detail["stage_hint"]],
        }

        self.assertEqual(detail["stage_hint"], "low_score_multiview_lateral_lead_projection_excluded")
        self.assertEqual(detail["canonical_lead_contact_count"], 2)
        self.assertEqual(detail["objective_error_sources"], ["multiview_alignment"])
        self.assertEqual(review_bucket_for_summary(summary), "evidence_limited")
        self.assertEqual(detail["land_pad_iou_score"], 0.99)

    def test_score_diagnostics_flag_borderline_component_scores(self) -> None:
        details = score_diagnostic_details(
            {
                "lead_pad_iou_score": 0.85,
                "pad_layout_score": 0.86,
                "land_pad_iou_score": 0.95,
            },
            {"package_pads": [{"bbox": [0.0, 0.0, 1.0, 1.0]}], "summary": {}},
        )

        reasons = {item["reason"] for item in details}
        lead_detail = next(item for item in details if item["reason"] == "borderline_lead_pad_iou")
        pad_layout_detail = next(item for item in details if item["reason"] == "borderline_pad_layout_score")

        self.assertIn("borderline_lead_pad_iou", reasons)
        self.assertIn("borderline_pad_layout_score", reasons)
        self.assertNotIn("borderline_land_pad_iou", reasons)
        self.assertEqual(lead_detail["threshold"], 0.9)
        self.assertEqual(lead_detail["lower_bound"], 0.8)
        self.assertEqual(pad_layout_detail["threshold"], 0.9)

    def test_score_diagnostics_do_not_duplicate_borderline_for_low_scores(self) -> None:
        details = score_diagnostic_details(
            {
                "lead_pad_iou_score": 0.79,
                "pad_layout_score": 0.79,
            },
            {"package_pads": [{"bbox": [0.0, 0.0, 1.0, 1.0]}], "summary": {}},
        )

        reasons = {item["reason"] for item in details}

        self.assertIn("low_lead_pad_iou", reasons)
        self.assertIn("low_pad_layout_score", reasons)
        self.assertNotIn("borderline_lead_pad_iou", reasons)
        self.assertNotIn("borderline_pad_layout_score", reasons)

    def test_score_diagnostics_flag_count_mapping_direct_mismatch(self) -> None:
        summary = {
            "alignment_scores": {
                "lead_pad_iou_score": 0.05,
                "pad_layout_score": 0.5,
            },
            "checks": [
                {
                    "name": "lead_count",
                    "expected": 3,
                    "actual": 3,
                    "tolerance": 0,
                    "status": "aligned",
                    "selected_mapping": "scan_group_lead_count_to_package_pad_rect_count",
                    "actual_role": "package_pad_rect_count",
                    "direct_expected": 4,
                    "direct_actual": 5,
                    "candidate": {
                        "source": "GroupItems",
                        "candidate_type": "group_raw_object_count",
                        "group_index": 0,
                        "count": 3,
                    },
                }
            ],
        }

        apply_score_diagnostics(summary, {"summary": {}})
        detail = next(item for item in summary["score_diagnostic_details"] if item["reason"] == "count_mapping_direct_mismatch")

        self.assertEqual(detail["metric"], "lead_count_direct_delta")
        self.assertEqual(detail["value"], 1.0)
        self.assertEqual(detail["threshold"], 0.0)
        self.assertEqual(detail["stage_hint"], "low_score_scan_result_package_pad_alignment")
        self.assertEqual(detail["selected_mapping"], "scan_group_lead_count_to_package_pad_rect_count")
        self.assertEqual(detail["expected_after_mapping"], 3)
        self.assertEqual(detail["actual_after_mapping"], 3)
        self.assertEqual(detail["direct_expected"], 4)
        self.assertEqual(detail["direct_actual"], 5)

    def test_score_diagnostics_explain_package_pad_subset_filter(self) -> None:
        summary = {
            "alignment_scores": {
                "package_pad_count": 15,
                "result_package_pad_count": 14,
                "lead_pad_iou_score": 0.904,
                "pad_layout_score": 0.908,
            },
            "alignment_transform": {
                "strategy": "package_pad_terminal_label_subset_for_low_score",
                "package_pad_label_filter": "dshape",
                "default_quality_score": 0.843,
                "selected_quality_score": 0.904,
            },
        }

        apply_score_diagnostics(summary, {"summary": {}})
        detail = next(
            item
            for item in summary["score_diagnostic_details"]
            if item["reason"] == "package_pad_subset_filter_applied"
        )

        self.assertEqual(detail["metric"], "result_package_pad_count")
        self.assertEqual(detail["value"], 14.0)
        self.assertEqual(detail["threshold"], 15.0)
        self.assertEqual(detail["stage_hint"], "review_note_multiview_package_pad_subset_filter")
        self.assertEqual(detail["objective_error_sources"], ["multiview_alignment"])
        self.assertEqual(detail["package_pad_label_filter"], "dshape")
        self.assertEqual(detail["package_pad_count"], 15)
        self.assertEqual(detail["result_package_pad_count"], 14)
        self.assertEqual(detail["default_quality_score"], 0.843)
        self.assertEqual(detail["selected_quality_score"], 0.904)

    def test_score_diagnostics_skip_package_pad_subset_filter_when_count_unchanged(self) -> None:
        summary = {
            "alignment_scores": {
                "package_pad_count": 14,
                "result_package_pad_count": 14,
            },
            "alignment_transform": {
                "strategy": "package_pad_terminal_label_subset_for_low_score",
                "package_pad_label_filter": "dshape",
            },
        }

        apply_score_diagnostics(summary, {"summary": {}})

        self.assertNotIn("package_pad_subset_filter_applied", summary["score_diagnostics"])

    def test_score_diagnostics_explain_package_pad_subset_filter_when_quality_improves(self) -> None:
        summary = {
            "alignment_scores": {
                "package_pad_count": 144,
                "result_package_pad_count": 144,
            },
            "alignment_transform": {
                "strategy": "package_pad_terminal_label_subset_for_low_score",
                "package_pad_label_filter": "circle",
                "default_quality_score": 0.118,
                "selected_quality_score": 0.949,
            },
        }

        apply_score_diagnostics(summary, {"summary": {}})
        detail = next(
            item
            for item in summary["score_diagnostic_details"]
            if item["reason"] == "package_pad_subset_filter_applied"
        )

        self.assertEqual(detail["metric"], "selected_quality_score")
        self.assertEqual(detail["value"], 0.949)
        self.assertEqual(detail["threshold"], 0.118)
        self.assertEqual(detail["package_pad_count"], 144)
        self.assertEqual(detail["result_package_pad_count"], 144)
        self.assertAlmostEqual(detail["quality_gain"], 0.831)

    def test_review_quality_uses_selected_transform_quality(self) -> None:
        result_objects, scores, transform = select_aligned_result(
            {
                "features": {
                    "objects": [
                        {"role": "lead", "bbox": [0.0, 0.0, 1.0, 1.0]},
                        {"role": "lead", "bbox": [2.0, 0.0, 3.0, 1.0]},
                    ],
                    "role_counts": {"lead": 2},
                    "bbox_candidates": {
                        "lead": [0.0, 0.0, 3.0, 1.0],
                        "all": [0.0, 0.0, 3.0, 1.0],
                    },
                }
            },
            {
                "package_pads": [
                    {"bbox": [0.0, 0.0, 1.0, 1.0], "source_label": "circle"},
                    {"bbox": [2.0, 0.0, 3.0, 1.0], "source_label": "circle"},
                    {"bbox": [20.0, 0.0, 21.0, 1.0], "source_label": "rect"},
                ],
                "land_pads": [],
                "lead_contacts": [],
                "dimensions": [],
                "summary": {},
            },
            [{"name": "lead_count", "status": "aligned"}],
        )

        self.assertEqual(transform["strategy"], "package_pad_terminal_label_subset_for_low_score")
        self.assertEqual(transform["package_pad_label_filter"], "circle")
        self.assertEqual(len([obj for obj in result_objects if obj["role"] == "package_pad"]), 2)
        self.assertGreater(scores["quality_score"], transform["default_quality_score"])
        self.assertAlmostEqual(scores["review_quality_score"], scores["quality_score"], places=6)
        self.assertEqual(alignment_review_quality_score(scores), scores["review_quality_score"])

    def test_score_diagnostics_include_scan_result_fallback_role_counts(self) -> None:
        gt = {
            "features": {
                "objects": [
                    {"role": "lead", "bbox": [0.0, 0.0, 1.0, 1.0]},
                    {"role": "land", "bbox": [0.0, 0.0, 1.0, 1.0]},
                ]
            }
        }
        canonical = {
            "package_pads": [],
            "land_pads": [],
            "lead_contacts": [],
            "dimensions": [],
            "source_selection": {
                "package_pads": {
                    "source_type": "scan_result_format",
                    "used_fallback": True,
                    "missing_primary": False,
                    "fallback_reason": "lead_count_mismatch_replaced_by_scan_result_lead_geometry",
                    "selected_raw_view": "top",
                    "selected_canonical_view": "top",
                    "previous_graph_path": "/tmp/top.package_graph.json",
                    "previous_terminal_package_pad_count": 3,
                    "scan_result_lead_count": 1,
                },
                "land_pads": {
                    "source_type": "scan_result_format",
                    "used_fallback": True,
                    "missing_primary": True,
                    "fallback_reason": "missing_land_view_used_scan_result_land_geometry",
                    "scan_result_effective_land_count": 2,
                    "scan_result_raw_land_count": 2,
                },
            },
        }
        result_objects = [
            {"role": "package_pad", "bbox": [0.0, 0.0, 1.0, 1.0], "source_type": "scan_result_format"},
            {"role": "land", "bbox": [0.0, 0.0, 1.0, 1.0], "source_type": "scan_result_format"},
            {"role": "land", "bbox": [2.0, 0.0, 3.0, 1.0], "source_type": "scan_result_format"},
        ]

        scores = alignment_scores(gt, canonical, result_objects, [])
        details = score_diagnostic_details(scores, canonical)
        detail = next(item for item in details if item["reason"] == "scan_result_geometry_fallback")

        self.assertEqual(scores["scan_result_fallback_object_count"], 3)
        self.assertEqual(scores["scan_result_fallback_role_counts"], {"land": 2, "package_pad": 1})
        self.assertEqual(detail["fallback_role_counts"], {"land": 2, "package_pad": 1})
        self.assertEqual(
            detail["fallback_sources"],
            [
                {
                    "object_role": "land",
                    "fallback_object_count": 2,
                    "selection_role": "land_pads",
                    "source_type": "scan_result_format",
                    "used_fallback": True,
                    "missing_primary": True,
                    "fallback_reason": "missing_land_view_used_scan_result_land_geometry",
                    "selected_raw_view": "",
                    "selected_canonical_view": "",
                    "scan_result_effective_land_count": 2,
                    "scan_result_raw_land_count": 2,
                },
                {
                    "object_role": "package_pad",
                    "fallback_object_count": 1,
                    "selection_role": "package_pads",
                    "source_type": "scan_result_format",
                    "used_fallback": True,
                    "missing_primary": False,
                    "fallback_reason": "lead_count_mismatch_replaced_by_scan_result_lead_geometry",
                    "selected_raw_view": "top",
                    "selected_canonical_view": "top",
                    "previous_terminal_package_pad_count": 3,
                    "scan_result_lead_count": 1,
                    "previous_graph_file": "top.package_graph.json",
                },
            ],
        )

    def test_score_diagnostics_flag_array_pad_layout_mismatch(self) -> None:
        details = score_diagnostic_details(
            {
                "gt_lead_count": 80,
                "lead_iou": 0.92,
                "lead_pad_iou_score": 0.24,
                "pad_layout_score": 0.24,
            },
            {
                "package_pads": [{"bbox": [0.0, 0.0, 1.0, 1.0]} for _ in range(80)],
                "land_pads": [],
                "lead_contacts": [],
                "summary": {},
            },
        )

        detail = next(item for item in details if item["reason"] == "array_pad_layout_mismatch")

        self.assertEqual(detail["metric"], "lead_pad_iou_score")
        self.assertEqual(detail["value"], 0.24)
        self.assertEqual(detail["threshold"], 0.8)
        self.assertEqual(detail["stage_hint"], "low_score_package_graph_array_pad_layout")
        self.assertEqual(detail["gt_lead_count"], 80)
        self.assertEqual(detail["lead_iou"], 0.92)

    def test_score_diagnostics_flag_land_pad_proxy_size_mismatch(self) -> None:
        details = score_diagnostic_details(
            {
                "package_pad_from_land_proxy": True,
                "gt_lead_count": 48,
                "result_package_pad_count": 48,
                "result_lead_contact_count": 24,
                "lead_pad_iou_score": 0.15,
                "pad_layout_score": 0.42,
            },
            {
                "package_pads": [],
                "land_pads": [{"bbox": [0.0, 0.0, 1.0, 1.0]}],
                "lead_contacts": [],
                "summary": {},
            },
        )

        detail = next(item for item in details if item["reason"] == "land_pad_proxy_size_mismatch")

        self.assertEqual(detail["metric"], "lead_pad_iou_score")
        self.assertEqual(detail["value"], 0.15)
        self.assertEqual(detail["threshold"], 0.8)
        self.assertEqual(detail["stage_hint"], "low_score_multiview_land_pad_proxy_size_mismatch")
        self.assertIs(detail["package_pad_from_land_proxy"], True)
        self.assertEqual(detail["gt_lead_count"], 48)
        self.assertEqual(detail["result_package_pad_count"], 48)
        self.assertEqual(detail["result_lead_contact_count"], 24)

    def test_score_diagnostics_flag_missing_lead_detail_even_with_proxy_lead_score(self) -> None:
        details = score_diagnostic_details(
            {
                "overall_score": 0.91,
                "lead_iou": 0.9,
                "lead_pad_iou_score": 0.0,
                "pad_layout_score": 0.45,
                "dimension_value_score": 1.0,
            },
            {
                "lead_contacts": [],
                "summary": {
                    "missing_canonical_views": ["lead_detail"],
                    "lead_contact_count": 0,
                },
            },
        )

        missing_detail = next(
            item for item in details if item["reason"] == "missing_lead_detail_view_for_lead_layout"
        )

        self.assertEqual(missing_detail["metric"], "lead_pad_iou_score")
        self.assertEqual(missing_detail["value"], 0.0)
        self.assertEqual(missing_detail["stage_hint"], "low_score_data_missing_lead_detail_view")

    def test_score_diagnostics_skip_missing_lead_detail_when_proxy_lead_score_is_good(self) -> None:
        details = score_diagnostic_details(
            {
                "overall_score": 0.99,
                "lead_iou": 1.0,
                "lead_pad_iou_score": 0.9,
                "pad_layout_score": 0.9,
                "dimension_value_score": 1.0,
            },
            {
                "lead_contacts": [],
                "summary": {
                    "missing_canonical_views": ["lead_detail"],
                    "lead_contact_count": 0,
                },
            },
        )

        self.assertNotIn("missing_lead_detail_view_for_lead_layout", {item["reason"] for item in details})

    def test_score_diagnostics_flag_partial_lead_detail_layout(self) -> None:
        details = score_diagnostic_details(
            {
                "overall_score": 0.91,
                "gt_lead_count": 3,
                "lead_pad_iou_score": 0.01,
            },
            {
                "lead_contacts": [{"bbox": [0.0, 0.0, 1.0, 1.0]}],
                "summary": {
                    "missing_canonical_views": [],
                    "lead_contact_count": 1,
                },
            },
        )

        detail = next(item for item in details if item["reason"] == "partial_lead_detail_layout")

        self.assertEqual(detail["metric"], "lead_contact_count")
        self.assertEqual(detail["value"], 1.0)
        self.assertEqual(detail["threshold"], 3.0)
        self.assertEqual(detail["stage_hint"], "low_score_multiview_partial_lead_detail_layout")

    def test_matched_box_iou_score_matches_by_overlap_not_row_sort_order(self) -> None:
        gt_boxes = [
            [0.0, 0.0, 1.0, 1.0],
            [3.0, 0.0, 4.0, 1.0],
        ]
        result_boxes = [
            [3.0, -0.00001, 4.0, 0.99999],
            [0.0, 0.00001, 1.0, 1.00001],
        ]

        score = matched_box_iou_score(gt_boxes, result_boxes)

        self.assertIsNotNone(score)
        self.assertGreater(score or 0.0, 0.999)

    def test_representative_score_cases_pick_worst_diagnostic_examples(self) -> None:
        summaries = [
            representative_summary("PART_B", quality=0.4, metric_value=0.4),
            representative_summary("PART_A", quality=0.1, metric_value=0.1),
            representative_summary("PART_C", quality=0.2, metric_value=0.2),
            representative_summary("PART_D", quality=0.0, metric_value=None, reason="missing_land_view_for_land_iou"),
        ]

        result = representative_score_cases(summaries, limit_per_reason=2)

        self.assertEqual([row["part_number"] for row in result["low_pad_layout_score"]], ["PART_A", "PART_C"])
        self.assertEqual(result["low_pad_layout_score"][0]["quality_score"], 0.1)
        self.assertEqual(result["low_pad_layout_score"][0]["comparison_svg_path"], "/tmp/PART_A/comparison.svg")
        self.assertEqual(result["missing_land_view_for_land_iou"][0]["metric_value"], None)

    def test_aligned_canonical_objects_keep_scan_result_format_bboxes_in_gt_coordinates(self) -> None:
        aligned = aligned_canonical_objects(
            {
                "features": {
                    "bbox_candidates": {
                        "all": [0.0, 0.0, 10.0, 10.0],
                        "lead": [2.0, 2.0, 3.0, 3.0],
                    }
                }
            },
            {
                "package_pads": [
                    {
                        "bbox": [2.0, 2.0, 3.0, 3.0],
                        "source_type": "scan_result_format",
                        "source_object_id": 1,
                    }
                ],
                "summary": {},
            },
        )

        self.assertEqual(aligned[0]["bbox"], [2.0, 2.0, 3.0, 3.0])
        self.assertEqual(aligned[0]["alignment_source_bbox"], [2.0, 2.0, 3.0, 3.0])
        self.assertEqual(aligned[0]["alignment_target_bbox"], [2.0, 2.0, 3.0, 3.0])

    def test_select_aligned_result_uses_package_pad_lead_bbox_when_low_score_improves(self) -> None:
        gt = {
            "features": {
                "objects": [
                    {"role": "lead", "bbox": [0.0, 0.0, 1.0, 1.0]},
                    {"role": "lead", "bbox": [0.0, 2.0, 1.0, 3.0]},
                    {"role": "shape", "bbox": [0.0, 0.0, 10.0, 10.0]},
                ],
                "bbox_candidates": {
                    "all": [0.0, 0.0, 10.0, 10.0],
                    "lead": [0.0, 0.0, 1.0, 3.0],
                    "shape": [0.0, 0.0, 10.0, 10.0],
                },
            }
        }
        canonical = {
            "outline_2d": {"bbox": [0.0, 0.0, 10.0, 10.0]},
            "package_pads": [
                {"bbox": [4.0, 1.0, 6.0, 3.0]},
                {"bbox": [4.0, 7.0, 6.0, 9.0]},
            ],
            "land_pads": [],
            "lead_contacts": [],
            "dimensions": [],
            "summary": {
                "source_selection": {
                    "package_pads": {
                        "source_type": "package_graph",
                    }
                }
            },
        }
        checks = [
            {"name": "land_count", "status": "aligned"},
            {"name": "lead_count", "status": "aligned"},
        ]

        _objects, scores, transform = select_aligned_result(gt, canonical, checks)

        self.assertEqual(transform["strategy"], "package_pad_lead_bbox_for_low_score")
        self.assertGreater(transform["selected_overall_score"], transform["default_overall_score"])
        self.assertGreater(scores["lead_pad_iou_score"], 0.5)

    def test_select_aligned_result_uses_conductive_bbox_for_package_circle_array(self) -> None:
        gt = {
            "features": {
                "objects": [
                    {"role": "lead", "bbox": [0.0, 0.0, 1.0, 1.0]},
                    {"role": "lead", "bbox": [2.0, 0.0, 3.0, 1.0]},
                    {"role": "lead", "bbox": [0.0, 2.0, 1.0, 3.0]},
                    {"role": "shape", "bbox": [0.0, 0.0, 5.0, 5.0]},
                ],
                "role_counts": {"lead": 3, "shape": 1},
                "bbox_candidates": {
                    "all": [0.0, 0.0, 5.0, 5.0],
                    "lead": [0.0, 0.0, 3.0, 3.0],
                    "shape": [0.0, 0.0, 5.0, 5.0],
                },
            }
        }
        canonical = {
            "outline_2d": {"bbox": [0.0, 0.0, 10.0, 10.0]},
            "package_pads": [
                {"bbox": [2.0, 6.0, 4.0, 8.0], "source_label": "pad_circle"},
                {"bbox": [6.0, 6.0, 8.0, 8.0], "source_label": "pad_circle"},
                {"bbox": [2.0, 2.0, 4.0, 4.0], "source_label": "pad_circle"},
            ],
            "land_pads": [],
            "lead_contacts": [],
            "dimensions": [],
            "summary": {"source_selection": {"package_pads": {"source_type": "package_graph"}}},
        }
        checks = [
            {"name": "land_count", "status": "aligned"},
            {"name": "lead_count", "status": "aligned"},
        ]

        objects, scores, transform = select_aligned_result(gt, canonical, checks)

        self.assertEqual(transform["strategy"], "package_pad_conductive_bbox_for_low_score")
        self.assertEqual(transform["package_pad_source_order"][0], "package_circle")
        self.assertEqual(transform["package_pad_flip"], "flip_y")
        self.assertGreater(scores["lead_pad_iou_score"], 0.99)
        self.assertTrue(all(obj.get("alignment_source_bbox") == [2.0, 2.0, 8.0, 8.0] for obj in objects if obj["role"] == "package_pad"))

    def test_select_aligned_result_can_use_terminal_package_pad_label_subset(self) -> None:
        gt = {
            "features": {
                "objects": [
                    {"role": "lead", "bbox": [8.0, 1.0, 9.0, 3.0]},
                    {"role": "lead", "bbox": [8.0, 7.0, 9.0, 9.0]},
                    {"role": "shape", "bbox": [0.0, 0.0, 10.0, 10.0]},
                ],
                "role_counts": {"lead": 2, "shape": 1},
                "bbox_candidates": {
                    "all": [0.0, 0.0, 10.0, 10.0],
                    "lead": [8.0, 1.0, 9.0, 9.0],
                    "shape": [0.0, 0.0, 10.0, 10.0],
                },
            }
        }
        canonical = {
            "outline_2d": {"bbox": [0.0, 0.0, 100.0, 100.0]},
            "package_pads": [
                {"bbox": [0.0, 0.0, 50.0, 100.0], "source_label": "pad_circle"},
                {"bbox": [70.0, 10.0, 80.0, 30.0], "source_label": "pad_dshape"},
                {"bbox": [70.0, 70.0, 80.0, 90.0], "source_label": "pad_dshape"},
            ],
            "land_pads": [],
            "lead_contacts": [],
            "dimensions": [],
            "summary": {"source_selection": {"package_pads": {"source_type": "package_graph"}}},
        }
        checks = [
            {"name": "land_count", "status": "aligned"},
            {"name": "lead_count", "status": "aligned"},
        ]

        objects, scores, transform = select_aligned_result(gt, canonical, checks)

        self.assertEqual(transform["strategy"], "package_pad_terminal_label_subset_for_low_score")
        self.assertEqual(transform["package_pad_label_filter"], "dshape")
        self.assertEqual(len([obj for obj in objects if obj["role"] == "package_pad"]), 2)
        self.assertGreater(scores["lead_pad_iou_score"], 0.99)

    def test_select_aligned_result_can_flip_package_pads_and_exclude_partial_lead_detail(self) -> None:
        gt = {
            "features": {
                "objects": [
                    {"role": "lead", "bbox": [4.0, 1.0, 6.0, 3.0]},
                    {"role": "lead", "bbox": [1.0, 7.0, 3.0, 9.0]},
                    {"role": "lead", "bbox": [7.0, 7.0, 9.0, 9.0]},
                    {"role": "shape", "bbox": [0.0, 0.0, 10.0, 10.0]},
                ],
                "role_counts": {"lead": 3, "shape": 1},
                "bbox_candidates": {
                    "all": [0.0, 0.0, 10.0, 10.0],
                    "lead": [1.0, 1.0, 9.0, 9.0],
                    "shape": [0.0, 0.0, 10.0, 10.0],
                },
            }
        }
        canonical = {
            "package_pads": [
                {"bbox": [1.0, 1.0, 3.0, 3.0], "source_label": "pad"},
                {"bbox": [7.0, 1.0, 9.0, 3.0], "source_label": "pad"},
                {"bbox": [4.0, 7.0, 6.0, 9.0], "source_label": "pad"},
            ],
            "land_pads": [],
            "lead_contacts": [
                {"bbox": [1.0, 1.0, 9.0, 9.0], "source_label": "pad"},
            ],
            "dimensions": [],
            "summary": {"source_selection": {"package_pads": {"source_type": "package_graph"}}},
        }
        checks = [
            {"name": "land_count", "status": "aligned"},
            {"name": "lead_count", "status": "aligned"},
        ]

        objects, scores, transform = select_aligned_result(gt, canonical, checks)

        self.assertEqual(transform["strategy"], "package_pad_flip_without_partial_lead_detail")
        self.assertEqual(transform["package_pad_flip"], "flip_y")
        self.assertTrue(transform["excluded_partial_lead_contacts"])
        self.assertEqual(len([obj for obj in objects if obj["role"] == "lead"]), 0)
        self.assertGreater(scores["lead_pad_iou_score"], 0.99)

    def test_select_aligned_result_can_rotate_package_pads_when_view_orientation_differs(self) -> None:
        gt = {
            "features": {
                "objects": [
                    {"role": "lead", "bbox": [4.0, 1.0, 6.0, 3.0]},
                    {"role": "lead", "bbox": [1.0, 7.0, 3.0, 9.0]},
                    {"role": "lead", "bbox": [7.0, 7.0, 9.0, 9.0]},
                    {"role": "shape", "bbox": [0.0, 0.0, 10.0, 10.0]},
                ],
                "role_counts": {"lead": 3, "shape": 1},
                "bbox_candidates": {
                    "all": [0.0, 0.0, 10.0, 10.0],
                    "lead": [1.0, 1.0, 9.0, 9.0],
                    "shape": [0.0, 0.0, 10.0, 10.0],
                },
            }
        }
        canonical = {
            "package_pads": [
                {"bbox": [1.0, 1.0, 3.0, 3.0], "source_label": "pad"},
                {"bbox": [7.0, 4.0, 9.0, 6.0], "source_label": "pad"},
                {"bbox": [1.0, 7.0, 3.0, 9.0], "source_label": "pad"},
            ],
            "land_pads": [],
            "lead_contacts": [],
            "dimensions": [],
            "summary": {"source_selection": {"package_pads": {"source_type": "package_graph"}}},
        }
        checks = [
            {"name": "land_count", "status": "aligned"},
            {"name": "lead_count", "status": "aligned"},
        ]

        objects, scores, transform = select_aligned_result(gt, canonical, checks)

        self.assertEqual(transform["strategy"], "package_pad_rotation_for_low_score")
        self.assertEqual(transform["package_pad_rotation"], 90)
        self.assertGreater(scores["lead_pad_iou_score"], 0.99)
        self.assertTrue(all(obj.get("alignment_package_pad_rotation") == 90 for obj in objects if obj["role"] == "package_pad"))

    def test_alignment_candidate_rejects_score_gain_that_breaks_package_pad_lattice(self) -> None:
        gt = {"features": {"objects": [{"role": "lead", "bbox": bbox} for bbox in staggered_lattice_bboxes(16, 5)]}}
        best_objects = [{"role": "package_pad", "bbox": bbox} for bbox in staggered_lattice_bboxes(16, 5)]
        candidate_objects = [{"role": "package_pad", "bbox": bbox} for bbox in staggered_lattice_bboxes(10, 8)]

        self.assertFalse(
            alignment_candidate_is_better(
                {"quality_score": 0.24, "overall_score": 0.82},
                candidate_objects,
                {"quality_score": 0.22, "overall_score": 0.80},
                best_objects,
                gt,
            )
        )

    def test_alignment_candidate_accepts_score_gain_when_package_pad_lattice_is_preserved(self) -> None:
        gt = {"features": {"objects": [{"role": "lead", "bbox": bbox} for bbox in staggered_lattice_bboxes(16, 5)]}}
        best_objects = [{"role": "package_pad", "bbox": bbox} for bbox in staggered_lattice_bboxes(16, 5)]
        candidate_objects = [{"role": "package_pad", "bbox": bbox} for bbox in staggered_lattice_bboxes(16, 5, x0=0.01)]

        self.assertTrue(
            alignment_candidate_is_better(
                {"quality_score": 0.24, "overall_score": 0.82},
                candidate_objects,
                {"quality_score": 0.22, "overall_score": 0.80},
                best_objects,
                gt,
            )
        )

    def test_alignment_candidate_accepts_high_iou_gain_despite_lattice_grouping_heuristic(self) -> None:
        gt = {"features": {"objects": [{"role": "lead", "bbox": bbox} for bbox in staggered_lattice_bboxes(16, 5)]}}
        best_objects = [{"role": "package_pad", "bbox": bbox} for bbox in staggered_lattice_bboxes(16, 5)]
        candidate_objects = [{"role": "package_pad", "bbox": bbox} for bbox in staggered_lattice_bboxes(10, 8)]

        self.assertTrue(
            alignment_candidate_is_better(
                {"lead_pad_iou_score": 0.85, "quality_score": 0.85, "overall_score": 0.97},
                candidate_objects,
                {"lead_pad_iou_score": 0.22, "quality_score": 0.22, "overall_score": 0.80},
                best_objects,
                gt,
            )
        )

    def test_select_aligned_result_can_rotate_land_pads_when_view_orientation_differs(self) -> None:
        gt = {
            "features": {
                "objects": [
                    {"role": "land", "bbox": [0.0, 0.0, 1.0, 1.0]},
                    {"role": "land", "bbox": [3.0, 0.0, 4.0, 1.0]},
                    {"role": "lead", "bbox": [0.0, 0.0, 1.0, 1.0]},
                    {"role": "lead", "bbox": [3.0, 0.0, 4.0, 1.0]},
                ],
                "role_counts": {"land": 2, "lead": 2},
                "bbox_candidates": {
                    "all": [0.0, 0.0, 4.0, 1.0],
                    "conductive": [0.0, 0.0, 4.0, 1.0],
                    "land": [0.0, 0.0, 4.0, 1.0],
                    "lead": [0.0, 0.0, 4.0, 1.0],
                },
            }
        }
        canonical = {
            "package_pads": [
                {"bbox": [0.0, 0.0, 1.0, 1.0]},
                {"bbox": [0.0, 3.0, 1.0, 4.0]},
            ],
            "land_pads": [
                {"bbox": [0.0, 0.0, 1.0, 1.0]},
                {"bbox": [0.0, 3.0, 1.0, 4.0]},
            ],
            "lead_contacts": [],
            "dimensions": [],
            "summary": {
                "source_selection": {
                    "land_pads": {"source_type": "package_graph"},
                    "package_pads": {"source_type": "package_graph"},
                }
            },
        }
        checks = [
            {"name": "land_count", "status": "aligned"},
            {"name": "lead_count", "status": "aligned"},
        ]

        objects, scores, transform = select_aligned_result(gt, canonical, checks)

        self.assertEqual(transform["land_pad_rotation"], 90)
        self.assertTrue(transform["package_pad_from_land_proxy"])
        self.assertGreater(scores["land_pad_iou_score"], 0.99)
        self.assertGreater(scores["lead_pad_iou_score"], 0.99)
        self.assertTrue(all(obj.get("alignment_land_pad_rotation") == 90 for obj in objects if obj["role"] == "land"))

    def test_select_aligned_result_excludes_partial_lead_contacts_when_still_low_score(self) -> None:
        gt = {
            "features": {
                "objects": [
                    {"role": "lead", "bbox": [0.0, 0.0, 1.0, 1.0]},
                    {"role": "lead", "bbox": [2.0, 0.0, 3.0, 1.0]},
                    {"role": "lead", "bbox": [0.0, 2.0, 1.0, 3.0]},
                    {"role": "lead", "bbox": [2.0, 2.0, 3.0, 3.0]},
                ],
                "role_counts": {"lead": 4},
                "bbox_candidates": {
                    "all": [0.0, 0.0, 3.0, 3.0],
                    "lead": [0.0, 0.0, 3.0, 3.0],
                },
            }
        }
        canonical = {
            "package_pads": [
                {"bbox": [0.0, 0.0, 0.5, 0.5], "source_label": "pad"},
                {"bbox": [0.1, 0.0, 0.6, 0.5], "source_label": "pad"},
                {"bbox": [0.0, 0.1, 0.5, 0.6], "source_label": "pad"},
                {"bbox": [0.1, 0.1, 0.6, 0.6], "source_label": "pad"},
            ],
            "land_pads": [],
            "lead_contacts": [
                {"bbox": [0.0, 0.0, 3.0, 3.0], "source_label": "pad"},
            ],
            "dimensions": [],
            "summary": {"source_selection": {"package_pads": {"source_type": "package_graph"}}},
        }
        checks = [
            {"name": "land_count", "status": "aligned"},
            {"name": "lead_count", "status": "aligned"},
        ]

        objects, scores, transform = select_aligned_result(gt, canonical, checks)

        self.assertTrue(transform["excluded_partial_lead_contacts"])
        self.assertEqual(len([obj for obj in objects if obj["role"] == "lead"]), 0)
        self.assertLess(scores["lead_pad_iou_score"], 0.5)

    def test_select_aligned_result_excludes_partial_lead_contacts_when_package_pads_score_well(self) -> None:
        gt = {
            "features": {
                "objects": [
                    {"role": "lead", "bbox": [0.0, 0.0, 1.0, 1.0]},
                    {"role": "lead", "bbox": [2.0, 0.0, 3.0, 1.0]},
                    {"role": "lead", "bbox": [0.0, 2.0, 1.0, 3.0]},
                    {"role": "lead", "bbox": [2.0, 2.0, 3.0, 3.0]},
                ],
                "role_counts": {"lead": 4},
                "bbox_candidates": {
                    "all": [0.0, 0.0, 3.0, 3.0],
                    "lead": [0.0, 0.0, 3.0, 3.0],
                },
            }
        }
        canonical = {
            "package_pads": [
                {"bbox": [0.0, 0.0, 1.0, 1.0], "source_label": "pad"},
                {"bbox": [2.0, 0.0, 3.0, 1.0], "source_label": "pad"},
                {"bbox": [0.0, 2.0, 1.0, 3.0], "source_label": "pad"},
                {"bbox": [2.0, 2.0, 3.0, 3.0], "source_label": "pad"},
            ],
            "land_pads": [],
            "lead_contacts": [
                {"bbox": [0.0, 0.0, 3.0, 3.0], "source_label": "pad"},
            ],
            "dimensions": [],
            "summary": {"source_selection": {"package_pads": {"source_type": "package_graph"}}},
        }
        checks = [
            {"name": "land_count", "status": "aligned"},
            {"name": "lead_count", "status": "aligned"},
        ]

        objects, scores, transform = select_aligned_result(gt, canonical, checks)

        self.assertTrue(transform["excluded_partial_lead_contacts"])
        self.assertEqual(transform["partial_lead_exclusion_reason"], "lead_contact_score_below_0.5")
        self.assertEqual(len([obj for obj in objects if obj["role"] == "lead"]), 0)
        self.assertGreater(scores["lead_pad_iou_score"], 0.99)

    def test_select_aligned_result_excludes_lateral_lead_contacts_when_package_pads_are_better(self) -> None:
        gt = {
            "features": {
                "objects": [
                    {"role": "lead", "bbox": [0.0, 0.0, 1.0, 3.0]},
                    {"role": "lead", "bbox": [4.0, 0.0, 5.0, 3.0]},
                ],
                "role_counts": {"lead": 2},
                "bbox_candidates": {
                    "all": [0.0, 0.0, 5.0, 3.0],
                    "lead": [0.0, 0.0, 5.0, 3.0],
                },
            }
        }
        canonical = {
            "package_pads": [
                {"bbox": [0.0, 0.0, 1.0, 3.0], "source_label": "pad"},
                {"bbox": [4.0, 0.0, 5.0, 3.0], "source_label": "pad"},
            ],
            "land_pads": [],
            "lead_contacts": [
                {"bbox": [0.0, 0.0, 5.0, 0.5], "source_label": "pad", "canonical_view": "lateral"},
                {"bbox": [0.0, 2.5, 5.0, 3.0], "source_label": "pad", "canonical_view": "lateral"},
            ],
            "dimensions": [],
            "summary": {"source_selection": {"package_pads": {"source_type": "package_graph"}}},
        }
        checks = [
            {"name": "land_count", "status": "aligned"},
            {"name": "lead_count", "status": "aligned"},
        ]

        objects, scores, transform = select_aligned_result(gt, canonical, checks)

        self.assertTrue(transform["excluded_lateral_lead_contacts"])
        self.assertEqual(transform["lateral_lead_exclusion_reason"], "lateral_projection_not_drawn_as_2d_terminal")
        self.assertEqual(len([obj for obj in objects if obj["role"] == "lead"]), 0)
        self.assertGreater(scores["lead_pad_iou_score"], 0.99)

    def test_select_aligned_result_limits_duplicate_land_pads_when_direct_gt_count_is_smaller(self) -> None:
        gt = {
            "features": {
                "objects": [
                    {"role": "land", "bbox": [0.0, 0.0, 2.0, 4.0]},
                    {"role": "land", "bbox": [4.0, 0.0, 6.0, 4.0]},
                ],
                "role_counts": {"land": 2},
                "bbox_candidates": {
                    "all": [0.0, 0.0, 6.0, 4.0],
                    "land": [0.0, 0.0, 6.0, 4.0],
                },
            }
        }
        canonical = {
            "package_pads": [],
            "land_pads": [
                {"bbox": [0.0, 0.0, 2.0, 4.0]},
                {"bbox": [4.0, 0.0, 6.0, 4.0]},
                {"bbox": [0.2, 0.5, 1.2, 3.0]},
                {"bbox": [0.5, 0.7, 1.5, 2.7]},
                {"bbox": [4.2, 0.5, 5.2, 3.0]},
                {"bbox": [4.5, 0.7, 5.5, 2.7]},
            ],
            "lead_contacts": [],
            "dimensions": [],
            "summary": {},
        }
        checks = [
            {"name": "land_count", "status": "aligned"},
            {"name": "lead_count", "status": "aligned"},
        ]

        objects, scores, transform = select_aligned_result(gt, canonical, checks)

        self.assertEqual(transform["land_pad_limit"], 2)
        self.assertEqual(transform["land_pad_limit_reason"], "scan_result_direct_land_count_with_low_land_iou")
        self.assertEqual(len([obj for obj in objects if obj["role"] == "land"]), 2)
        self.assertGreater(scores["land_pad_iou_score"], 0.99)

    def test_select_aligned_result_can_use_land_pads_as_package_pad_proxy_for_bad_array_package_layout(self) -> None:
        gt = {
            "features": {
                "objects": [
                    {"role": "lead", "bbox": [0.0, 0.0, 1.0, 1.0]},
                    {"role": "lead", "bbox": [3.0, 0.0, 4.0, 1.0]},
                    {"role": "land", "bbox": [0.0, 0.0, 1.0, 1.0]},
                    {"role": "land", "bbox": [3.0, 0.0, 4.0, 1.0]},
                ],
                "role_counts": {"lead": 2, "land": 2},
                "bbox_candidates": {
                    "all": [0.0, 0.0, 4.0, 1.0],
                    "lead": [0.0, 0.0, 4.0, 1.0],
                    "land": [0.0, 0.0, 4.0, 1.0],
                },
            }
        }
        canonical = {
            "package_pads": [
                {"bbox": [0.0, 0.0, 1.0, 1.0]},
                {"bbox": [0.0, 3.0, 1.0, 4.0]},
            ],
            "land_pads": [
                {"bbox": [0.0, 0.0, 1.0, 1.0]},
                {"bbox": [3.0, 0.0, 4.0, 1.0]},
            ],
            "lead_contacts": [],
            "dimensions": [],
            "summary": {"source_selection": {"package_pads": {"source_type": "package_graph"}}},
        }
        checks = [
            {"name": "land_count", "status": "aligned"},
            {"name": "lead_count", "status": "aligned"},
        ]

        objects, scores, transform = select_aligned_result(gt, canonical, checks)

        self.assertTrue(transform["package_pad_from_land_proxy"])
        self.assertEqual(
            transform["package_pad_proxy_reason"],
            "land_and_lead_counts_match_land_layout_outscores_package_layout",
        )
        self.assertEqual(len([obj for obj in objects if obj["role"] == "package_pad"]), 2)
        self.assertTrue(all(obj.get("package_pad_proxy_source") == "land_pad" for obj in objects if obj["role"] == "package_pad"))
        self.assertGreater(scores["lead_pad_iou_score"], 0.99)

    def test_select_aligned_result_can_use_land_pads_as_package_pad_proxy_for_overdetected_package_layout(self) -> None:
        gt = {
            "features": {
                "objects": [
                    {"role": "lead", "bbox": [0.0, 0.0, 1.0, 1.0]},
                    {"role": "lead", "bbox": [3.0, 0.0, 4.0, 1.0]},
                    {"role": "land", "bbox": [0.0, 0.0, 1.0, 1.0]},
                    {"role": "land", "bbox": [3.0, 0.0, 4.0, 1.0]},
                ],
                "role_counts": {"lead": 2, "land": 2},
                "bbox_candidates": {
                    "all": [0.0, 0.0, 4.0, 1.0],
                    "lead": [0.0, 0.0, 4.0, 1.0],
                    "land": [0.0, 0.0, 4.0, 1.0],
                },
            }
        }
        canonical = {
            "package_pads": [
                {"bbox": [0.0, 3.0, 0.8, 4.0]},
                {"bbox": [1.2, 3.0, 2.0, 4.0]},
                {"bbox": [2.4, 3.0, 3.2, 4.0]},
                {"bbox": [3.6, 3.0, 4.4, 4.0]},
            ],
            "land_pads": [
                {"bbox": [0.0, 0.0, 1.0, 1.0]},
                {"bbox": [3.0, 0.0, 4.0, 1.0]},
            ],
            "lead_contacts": [],
            "dimensions": [],
            "summary": {"source_selection": {"package_pads": {"source_type": "package_graph"}}},
        }
        checks = [
            {"name": "land_count", "status": "aligned"},
            {"name": "lead_count", "status": "aligned"},
        ]

        objects, scores, transform = select_aligned_result(gt, canonical, checks)

        self.assertTrue(transform["package_pad_from_land_proxy"])
        self.assertEqual(len([obj for obj in objects if obj["role"] == "package_pad"]), 2)
        self.assertTrue(all(obj.get("package_pad_proxy_source") == "land_pad" for obj in objects if obj["role"] == "package_pad"))
        self.assertGreater(scores["lead_pad_iou_score"], 0.99)

    def test_select_aligned_result_retries_land_proxy_after_excluding_lateral_leads(self) -> None:
        gt = {
            "features": {
                "objects": [
                    {"role": "lead", "bbox": [20.401148673793053, 5.999623448534897, 20.431148673793054, 6.043623448534897]},
                    {"role": "lead", "bbox": [20.516148673793055, 5.999623448534897, 20.546148673793052, 6.043623448534897]},
                    {"role": "land", "bbox": [20.390148673793053, 5.997623448534897, 20.435148673793055, 6.045623448534897]},
                    {"role": "land", "bbox": [20.512148673793053, 5.997623448534897, 20.557148673793055, 6.045623448534897]},
                ],
                "role_counts": {"lead": 2, "land": 2},
                "bbox_candidates": {
                    "all": [20.390148673793053, 5.984123421711386, 20.557148673793055, 6.059123421711885],
                    "shape": [20.416149155111558, 5.984123421711386, 20.531149155111176, 6.059123421711885],
                    "conductive": [20.390148673793053, 5.997623448534897, 20.557148673793055, 6.045623448534897],
                    "lead": [20.401148673793053, 5.999623448534897, 20.546148673793052, 6.043623448534897],
                    "land": [20.390148673793053, 5.997623448534897, 20.557148673793055, 6.045623448534897],
                },
            }
        }
        canonical = {
            "outline_2d": {"bbox": [33.253, 54.305, 193.885, 146.259]},
            "package_pads": [
                {"bbox": [184.764, 74.553, 223.001, 126.746]},
                {"bbox": [6.095, 74.621, 44.331, 126.814]},
            ],
            "land_pads": [
                {"bbox": [563.103, 180.661, 776.335, 408.11]},
                {"bbox": [-15.827, 180.934, 197.406, 408.382]},
            ],
            "lead_contacts": [
                {"bbox": [6.338, 56.689, 33.955, 65.571], "canonical_view": "lateral"},
                {"bbox": [194.625, 56.709, 222.242, 65.59], "canonical_view": "lateral"},
            ],
            "dimensions": [],
            "summary": {"source_selection": {"package_pads": {"source_type": "package_graph"}}},
        }
        checks = [
            {"name": "land_count", "status": "aligned"},
            {"name": "lead_count", "status": "aligned"},
        ]

        objects, scores, transform = select_aligned_result(gt, canonical, checks)

        package_pads = [obj for obj in objects if obj["role"] == "package_pad"]
        self.assertTrue(transform["excluded_lateral_lead_contacts"])
        self.assertTrue(transform["package_pad_from_land_proxy"])
        self.assertEqual(len(package_pads), 2)
        self.assertTrue(all(obj.get("package_pad_proxy_source") == "land_pad" for obj in package_pads))
        self.assertGreater(scores["lead_pad_iou_score"], 0.5)

    def test_select_aligned_result_can_resize_land_proxy_with_package_pad_median_size(self) -> None:
        gt = {
            "features": {
                "objects": [
                    {"role": "lead", "bbox": [0.3, 0.3, 0.7, 0.7]},
                    {"role": "lead", "bbox": [3.3, 0.3, 3.7, 0.7]},
                    {"role": "land", "bbox": [0.0, 0.0, 1.0, 1.0]},
                    {"role": "land", "bbox": [3.0, 0.0, 4.0, 1.0]},
                ],
                "role_counts": {"lead": 2, "land": 2},
                "bbox_candidates": {
                    "all": [0.0, 0.0, 4.0, 1.0],
                    "lead": [0.3, 0.3, 3.7, 0.7],
                    "land": [0.0, 0.0, 4.0, 1.0],
                },
            }
        }
        canonical = {
            "outline_2d": {"bbox": [0.0, 0.0, 16.0, 4.0]},
            "package_pads": [
                {"bbox": [4.0, 0.0, 5.882352941, 4.0]},
                {"bbox": [10.0, 0.0, 11.882352941, 4.0]},
            ],
            "land_pads": [
                {"bbox": [0.0, 0.0, 1.0, 1.0]},
                {"bbox": [3.0, 0.0, 4.0, 1.0]},
            ],
            "lead_contacts": [],
            "dimensions": [],
            "summary": {"source_selection": {"package_pads": {"source_type": "package_graph"}}},
        }
        checks = [
            {"name": "land_count", "status": "aligned"},
            {"name": "lead_count", "status": "aligned"},
        ]

        objects, scores, transform = select_aligned_result(gt, canonical, checks)

        self.assertTrue(transform["package_pad_from_land_proxy"])
        self.assertEqual(transform["package_pad_proxy_size_source"], "package_pad_median")
        package_pads = [obj for obj in objects if obj["role"] == "package_pad"]
        self.assertEqual(len(package_pads), 2)
        self.assertTrue(all(obj.get("package_pad_proxy_size_source") == "package_pad_median" for obj in package_pads))
        self.assertGreater(scores["lead_pad_iou_score"], 0.99)

    def test_select_aligned_result_can_use_terminal_land_pads_as_package_pad_proxy(self) -> None:
        terminal_boxes = [
            [0.0, 0.0, 1.0, 1.0],
            [5.0, 0.0, 6.0, 1.0],
        ]
        circle_boxes = [
            [0.0, 5.0, 1.0, 6.0],
            [5.0, 5.0, 6.0, 6.0],
        ]
        thermal_box = [2.0, 2.0, 4.0, 4.0]
        gt = {
            "features": {
                "objects": [{"role": "lead", "bbox": box} for box in terminal_boxes]
                + [{"role": "land", "bbox": box} for box in terminal_boxes + circle_boxes + [thermal_box]],
                "role_counts": {"lead": 2, "land": 5},
                "bbox_candidates": {
                    "all": [0.0, 0.0, 6.0, 6.0],
                    "lead": [0.0, 0.0, 6.0, 1.0],
                    "land": [0.0, 0.0, 6.0, 6.0],
                },
            }
        }
        canonical = {
            "package_pads": [
                {"bbox": [0.0, 8.0, 1.0, 9.0]},
                {"bbox": [0.0, 10.0, 1.0, 11.0]},
            ],
            "land_pads": [{"bbox": box, "source_label": "pad"} for box in terminal_boxes]
            + [{"bbox": box, "source_label": "pad_circle"} for box in circle_boxes]
            + [{"bbox": thermal_box, "source_label": "pad"}],
            "lead_contacts": [],
            "dimensions": [],
            "summary": {"source_selection": {"package_pads": {"source_type": "package_graph"}}},
        }
        checks = [
            {"name": "land_count", "status": "aligned"},
            {"name": "lead_count", "status": "aligned"},
        ]
        default_objects = aligned_canonical_objects(gt, canonical)
        default_scores = alignment_scores(gt, canonical, default_objects, checks)

        objects, scores, transform = select_aligned_result(gt, canonical, checks)

        package_pads = [obj for obj in objects if obj["role"] == "package_pad"]
        self.assertLess(default_scores["lead_pad_iou_score"], 0.5)
        self.assertTrue(transform["package_pad_from_land_proxy"])
        self.assertEqual(transform["package_pad_proxy_land_filter"], "terminal_rect")
        self.assertEqual(len(package_pads), 2)
        self.assertTrue(all(obj.get("package_pad_proxy_land_filter") == "terminal_rect" for obj in package_pads))
        self.assertGreater(scores["lead_pad_iou_score"], default_scores["lead_pad_iou_score"])
        self.assertGreater(scores["lead_pad_iou_score"], 0.99)

    def test_terminal_land_proxy_can_use_package_pad_median_size(self) -> None:
        lead_boxes = [
            [0.3, 0.3, 0.7, 0.7],
            [5.3, 0.3, 5.7, 0.7],
        ]
        terminal_land_boxes = [
            [0.0, 0.0, 1.0, 1.0],
            [5.0, 0.0, 6.0, 1.0],
        ]
        circle_boxes = [
            [0.0, 5.0, 1.0, 6.0],
            [5.0, 5.0, 6.0, 6.0],
        ]
        thermal_box = [2.0, 2.0, 4.0, 4.0]
        gt = {
            "features": {
                "objects": [{"role": "lead", "bbox": box} for box in lead_boxes]
                + [{"role": "land", "bbox": box} for box in terminal_land_boxes + circle_boxes + [thermal_box]],
                "role_counts": {"lead": 2, "land": 5},
                "bbox_candidates": {
                    "all": [0.0, 0.0, 6.0, 6.0],
                    "lead": [0.3, 0.3, 5.7, 0.7],
                    "land": [0.0, 0.0, 6.0, 6.0],
                },
            }
        }
        canonical = {
            "package_pads": [
                {"bbox": [0.0, 8.0, 0.666666667, 9.0]},
                {"bbox": [9.333333333, 9.5, 10.0, 10.5]},
            ],
            "land_pads": [{"bbox": box, "source_label": "pad"} for box in terminal_land_boxes]
            + [{"bbox": box, "source_label": "pad_circle"} for box in circle_boxes]
            + [{"bbox": thermal_box, "source_label": "pad"}],
            "lead_contacts": [],
            "dimensions": [],
            "summary": {"source_selection": {"package_pads": {"source_type": "package_graph"}}},
        }
        checks = [
            {"name": "land_count", "status": "aligned"},
            {"name": "lead_count", "status": "aligned"},
        ]
        baseline_objects = aligned_canonical_objects(
            gt,
            canonical,
            package_pad_from_land_proxy=True,
            package_pad_proxy_land_filter="terminal_rect",
        )
        baseline_scores = alignment_scores(gt, canonical, baseline_objects, checks)

        objects, scores, transform = prefer_terminal_land_pads_as_package_pad_proxy(
            gt,
            canonical,
            checks,
            baseline_objects,
            baseline_scores,
            {"strategy": "baseline"},
        )

        package_pads = [obj for obj in objects if obj["role"] == "package_pad"]
        self.assertTrue(transform["package_pad_from_land_proxy"])
        self.assertEqual(transform["package_pad_proxy_land_filter"], "terminal_rect")
        self.assertEqual(transform["package_pad_proxy_size_source"], "package_pad_median")
        self.assertEqual(len(package_pads), 2)
        self.assertTrue(all(obj.get("package_pad_proxy_size_source") == "package_pad_median" for obj in package_pads))
        self.assertGreater(scores["lead_pad_iou_score"], baseline_scores["lead_pad_iou_score"])

    def test_select_aligned_result_excludes_thermal_land_when_count_uses_terminal_land(self) -> None:
        terminal_boxes = [
            [0.0, 0.0, 1.0, 1.0],
            [3.0, 0.0, 4.0, 1.0],
            [0.0, 2.0, 1.0, 3.0],
            [3.0, 2.0, 4.0, 3.0],
            [0.0, 4.0, 1.0, 5.0],
            [3.0, 4.0, 4.0, 5.0],
        ]
        gt = {
            "features": {
                "objects": [{"role": "land", "bbox": box} for box in terminal_boxes]
                + [{"role": "lead", "bbox": box} for box in terminal_boxes],
                "role_counts": {"land": 6, "lead": 6},
                "bbox_candidates": {
                    "all": [0.0, 0.0, 4.0, 5.0],
                    "land": [0.0, 0.0, 4.0, 5.0],
                    "lead": [0.0, 0.0, 4.0, 5.0],
                },
            }
        }
        canonical = {
            "package_pads": [{"bbox": box, "source_label": "pad_dshape"} for box in terminal_boxes],
            "land_pads": [{"bbox": box, "source_label": "pad"} for box in terminal_boxes]
            + [{"bbox": [0.8, 1.4, 3.2, 3.6], "source_label": "pad"}],
            "lead_contacts": [],
            "dimensions": [],
            "summary": {"source_selection": {"package_pads": {"source_type": "package_graph"}}},
        }
        checks = [
            {
                "name": "land_count",
                "status": "aligned",
                "actual_role": "terminal_land_pad_count",
                "direct_actual": 7,
                "raw_land_pad_count": 7,
            },
            {"name": "lead_count", "status": "aligned"},
        ]

        objects, scores, transform = select_aligned_result(gt, canonical, checks)

        self.assertTrue(transform["excluded_thermal_land_pads"])
        self.assertEqual(len([obj for obj in objects if obj["role"] == "land"]), 6)
        self.assertGreater(scores["land_pad_iou_score"], 0.99)

    def test_select_aligned_result_excludes_thermal_package_pad_when_count_uses_terminal_package(self) -> None:
        terminal_boxes = [
            [0.0, 0.0, 1.0, 1.0],
            [2.0, 0.0, 3.0, 1.0],
            [4.0, 0.0, 5.0, 1.0],
            [6.0, 0.0, 7.0, 1.0],
            [0.0, 4.0, 1.0, 5.0],
            [2.0, 4.0, 3.0, 5.0],
            [4.0, 4.0, 5.0, 5.0],
            [6.0, 4.0, 7.0, 5.0],
        ]
        thermal_box = [2.0, 1.5, 5.0, 3.5]
        gt = {
            "features": {
                "objects": [{"role": "lead", "bbox": box} for box in terminal_boxes],
                "role_counts": {"lead": 8},
                "bbox_candidates": {
                    "all": [0.0, 0.0, 7.0, 5.0],
                    "lead": [0.0, 0.0, 7.0, 5.0],
                },
            }
        }
        canonical = {
            "package_pads": [{"bbox": box, "source_label": "pad"} for box in terminal_boxes]
            + [{"bbox": thermal_box, "source_label": "pad"}],
            "land_pads": [],
            "lead_contacts": [],
            "dimensions": [],
            "summary": {"source_selection": {"package_pads": {"source_type": "package_graph"}}},
        }
        checks = [
            {
                "name": "lead_count",
                "status": "aligned",
                "actual_role": "lead_equivalent_count",
                "direct_actual": 9,
                "terminal_package_pad_count": 8,
            },
            {"name": "land_count", "status": "aligned"},
        ]

        objects, scores, transform = select_aligned_result(gt, canonical, checks)

        self.assertTrue(transform["excluded_thermal_package_pads"])
        self.assertEqual(len([obj for obj in objects if obj["role"] == "package_pad"]), 8)
        self.assertGreater(scores["lead_pad_iou_score"], 0.99)

    def test_alignment_scores_report_duplicate_lead_geometry_sources(self) -> None:
        gt = {
            "features": {
                "objects": [
                    {"role": "lead", "bbox": [0.0, 0.0, 1.0, 1.0]},
                    {"role": "lead", "bbox": [2.0, 0.0, 3.0, 1.0]},
                ],
            }
        }
        canonical = {
            "package_pads": [{"bbox": [0.0, 0.0, 1.0, 1.0]}, {"bbox": [2.0, 0.0, 3.0, 1.0]}],
            "land_pads": [],
            "lead_contacts": [{"bbox": [0.0, 0.0, 1.0, 1.0]}, {"bbox": [2.0, 0.0, 3.0, 1.0]}],
            "dimensions": [],
        }
        result_objects = [
            {"role": "package_pad", "bbox": [0.0, 0.0, 1.0, 1.0]},
            {"role": "package_pad", "bbox": [2.0, 0.0, 3.0, 1.0]},
            {"role": "lead", "bbox": [0.0, 0.0, 1.0, 1.0]},
            {"role": "lead", "bbox": [2.0, 0.0, 3.0, 1.0]},
        ]

        scores = alignment_scores(gt, canonical, result_objects, [])
        details = score_diagnostic_details(scores, canonical)

        self.assertEqual(scores["lead_pad_iou_package_only"], 1.0)
        self.assertEqual(scores["lead_pad_iou_lead_contact_only"], 1.0)
        self.assertLess(scores["lead_pad_iou_score"], 1.0)
        self.assertIn("duplicate_lead_geometry_sources", [item["reason"] for item in details])


def scan_object(
    *,
    object_id: int,
    node_name: str,
    role_key: str,
    x: float,
    y: float,
    width: float,
    height: float,
) -> dict:
    return {
        "ID": object_id,
        "NodeName": node_name,
        "Geometry": 3 if node_name == "DShape" else 1 if node_name == "Circle" else 0,
        role_key: {},
        "PointList": rectangle_points(x, y, x + width, y + height),
    }


def representative_summary(
    part_number: str,
    *,
    quality: float,
    metric_value: float | None,
    reason: str = "low_pad_layout_score",
) -> dict:
    metric = "land_iou" if metric_value is None else "pad_layout_score"
    return {
        "part_number": part_number,
        "status": "aligned",
        "alignment_scores": {"overall_score": 0.9, "quality_score": quality},
        "score_diagnostics": [reason],
        "score_diagnostic_details": [
            {
                "reason": reason,
                "metric": metric,
                "value": metric_value,
                "threshold": 0.5 if metric_value is not None else None,
                "stage_hint": "low_score_package_graph_package_pad_geometry",
                "objective_error_sources": ["package_graph_reconstruction"],
            }
        ],
        "comparison_svg_path": f"/tmp/{part_number}/comparison.svg",
        "aligned_result_svg_path": f"/tmp/{part_number}/aligned_result.svg",
        "gt_reference_svg_path": f"/tmp/{part_number}/gt_reference.svg",
    }


def write_scan(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "Object": [
                    {
                        "ID": 1,
                        "NodeName": "Rectangle",
                        "LeadData": {},
                        "PointList": [
                            {"PointX": 0, "PointY": 0},
                            {"PointX": 1, "PointY": 0},
                            {"PointX": 1, "PointY": 1},
                            {"PointX": 0, "PointY": 1},
                        ],
                    },
                    {
                        "ID": 2,
                        "NodeName": "Rectangle",
                        "LandData": {},
                        "PointList": [
                            {"PointX": 2, "PointY": 0},
                            {"PointX": 3, "PointY": 0},
                            {"PointX": 3, "PointY": 1},
                            {"PointX": 2, "PointY": 1},
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def write_package_graph(
    path: Path,
    raw_view: str,
    pad_boxes: list[list[float]],
    *,
    outline: list[float] | None = None,
    dimensions: list[dict[str, object]] | None = None,
) -> None:
    objects = [
        {
            "id": 0,
            "label": "outline",
            "source_label": "outline",
            "bbox": outline or [0.0, 0.0, 10.0, 10.0],
        }
    ]
    for index, box in enumerate(pad_boxes, start=1):
        objects.append(
            {
                "id": index,
                "label": "rect",
                "source_label": "pad_rect",
                "bbox": box,
            }
        )
    path.write_text(
        json.dumps(
            {
                "part_number": "PART",
                "view": raw_view,
                "objects": objects,
                "dimensions": dimensions or [],
            }
        ),
        encoding="utf-8",
    )


def write_land_and_lead_scan(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "Object": [
                    {
                        "ID": 1,
                        "LandData": {},
                        "PointList": [
                            {"PointX": 0, "PointY": 0},
                            {"PointX": 1, "PointY": 0},
                            {"PointX": 1, "PointY": 1},
                            {"PointX": 0, "PointY": 1},
                        ],
                    },
                    {
                        "ID": 2,
                        "LandData": {},
                        "PointList": [
                            {"PointX": 2, "PointY": 0},
                            {"PointX": 3, "PointY": 0},
                            {"PointX": 3, "PointY": 1},
                            {"PointX": 2, "PointY": 1},
                        ],
                    },
                    {
                        "ID": 3,
                        "LeadData": {},
                        "PointList": [
                            {"PointX": 0, "PointY": 0},
                            {"PointX": 1, "PointY": 0},
                            {"PointX": 1, "PointY": 1},
                            {"PointX": 0, "PointY": 1},
                        ],
                    },
                    {
                        "ID": 4,
                        "LeadData": {},
                        "PointList": [
                            {"PointX": 2, "PointY": 0},
                            {"PointX": 3, "PointY": 0},
                            {"PointX": 3, "PointY": 1},
                            {"PointX": 2, "PointY": 1},
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def staggered_lattice_bboxes(
    row_count: int,
    column_count: int,
    *,
    x0: float = 0.0,
    y0: float = 0.0,
    x_step: float = 0.6,
    y_step: float = 0.18,
    odd_row_offset: float = 0.33,
    size: float = 0.12,
) -> list[list[float]]:
    boxes: list[list[float]] = []
    half_size = size / 2.0
    for row in range(row_count):
        for column in range(column_count):
            x = x0 + column * x_step + (odd_row_offset if row % 2 else 0.0)
            y = y0 + row * y_step
            boxes.append([x - half_size, y - half_size, x + half_size, y + half_size])
    return boxes


def rectangle_points(x1: float, y1: float, x2: float, y2: float) -> list[dict[str, float]]:
    return [
        {"PointX": x1, "PointY": y1},
        {"PointX": x2, "PointY": y1},
        {"PointX": x2, "PointY": y2},
        {"PointX": x1, "PointY": y2},
    ]


if __name__ == "__main__":
    unittest.main()
