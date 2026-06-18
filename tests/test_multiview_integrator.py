from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from real_image_process.FPK_PJ_fullflow.multiview.integrator import (
    MultiviewOptions,
    best_multiview_layer_rotation,
    risk_reasons_for_part,
    score_part,
    integrate_graphs,
    integrate_part,
    layer_rotation_boxes,
    match_package_and_land_pads,
    normalize_view,
    select_rotation_reference_layer,
    write_source_overlays,
    write_svg,
)


class MultiviewIntegratorTests(unittest.TestCase):
    def assert_bbox_close(self, actual: list[float], expected: list[float], places: int = 6) -> None:
        self.assertEqual(len(actual), len(expected))
        for actual_value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(actual_value, expected_value, places=places)

    def test_view_normalization_preserves_lateral_and_lead_detail_classes(self) -> None:
        options = MultiviewOptions()

        self.assertEqual(normalize_view("front", options), "lateral")
        self.assertEqual(normalize_view("side", options), "lateral")
        self.assertEqual(normalize_view("lead", options), "lead_detail")
        self.assertEqual(normalize_view("land_detail", options), "lead_detail")

    def test_canonical_svg_uses_pad_type_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "unified_multiview_layers.svg"
            write_svg(
                output,
                {
                    "part_number": "PART",
                    "outline_2d": {"role": "outline_2d", "bbox": [0.0, 0.0, 10.0, 5.0]},
                    "package_pads": [
                        {"role": "package_pad", "bbox": [1.0, 1.0, 2.0, 2.0], "source_label": "pad_circle"},
                        {"role": "package_pad", "bbox": [3.0, 1.0, 4.0, 2.0], "source_label": "pad_dshape"},
                    ],
                    "land_pads": [],
                },
            )

            svg = output.read_text(encoding="utf-8")

        self.assertIn("<ellipse", svg)
        self.assertIn("<path", svg)

    def test_source_overlay_marks_adopted_objects_and_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "PART" / "extract_image" / "bottom.png"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(
                b"\x89PNG\r\n\x1a\n"
                + b"\x00\x00\x00\rIHDR"
                + (200).to_bytes(4, "big")
                + (120).to_bytes(4, "big")
                + b"\x08\x02\x00\x00\x00"
            )
            graph_path = str(root / "PART" / "bottom.package_graph.json")
            canonical = {
                "part_number": "PART",
                "outline_2d": {
                    "source_object_id": 100,
                    "role": "outline_2d",
                    "source_label": "outline",
                    "bbox": [10.0, 10.0, 180.0, 100.0],
                    "source_graph": graph_path,
                },
                "package_pads": [
                    {
                        "source_object_id": 1,
                        "role": "package_pad",
                        "source_label": "pad_circle",
                        "bbox": [30.0, 40.0, 60.0, 70.0],
                        "source_graph": graph_path,
                    }
                ],
                "land_pads": [],
                "lead_contacts": [],
                "dimensions": [
                    {
                        "dimension_id": 5,
                        "text": "1.00",
                        "target_ids": [1],
                        "bbox": [80.0, 20.0, 120.0, 36.0],
                        "source_graph": graph_path,
                    }
                ],
                "evidence_refs": [
                    {
                        "evidence_type": "package_graph",
                        "raw_view": "bottom",
                        "canonical_view": "bottom",
                        "graph_path": graph_path,
                        "image_path": str(image_path),
                    }
                ],
            }

            manifest = write_source_overlays(root / "out" / "source_overlays", canonical)
            svg = Path(manifest[0]["path"]).read_text(encoding="utf-8")

        self.assertEqual(len(manifest), 1)
        self.assertIn("<image", svg)
        self.assertIn("data:image/png;base64,", svg)
        self.assertNotIn(str(image_path), svg)
        self.assertIn("obj:1 package_pad", svg)
        self.assertIn("dim:5 1.00", svg)
        self.assertIn("<ellipse", svg)
        self.assertIn('stroke="#f97316"', svg)

    def test_toy_gt_integration_keeps_2d_evidence_and_ignores_lateral_height(self) -> None:
        options = MultiviewOptions()
        canonical = integrate_part(
            "PART",
            [
                toy_graph("PART", "bottom", pad_count=4, outline=True, spacing_value=1.0),
                toy_graph("PART", "land", pad_count=4, outline=False, spacing_value=1.0),
                toy_graph("PART", "front", pad_count=0, outline=False, spacing_value=None, y_height_value=0.4),
            ],
            options,
        )

        self.assertEqual(canonical["part_number"], "PART")
        self.assertEqual(len(canonical["package_pads"]), 4)
        self.assertEqual(len(canonical["land_pads"]), 4)
        self.assertEqual(canonical["evidence_refs"][2]["raw_view"], "front")
        self.assertEqual(canonical["evidence_refs"][2]["canonical_view"], "lateral")
        self.assertEqual(len(canonical["ignored_evidence"]), 1)
        self.assertEqual(canonical["ignored_evidence"][0]["ignored_reason"], "lateral_height_or_vertical_dimension")
        self.assertEqual(canonical["evidence_summary"]["dimension_value_source_counts"], {"text_parser": 2})
        self.assertEqual(canonical["evidence_summary"]["dimension_canonical_view_counts"], {"bottom": 1, "land": 1})
        self.assertEqual(canonical["evidence_summary"]["ignored_evidence_reason_counts"], {"lateral_height_or_vertical_dimension": 1})
        self.assertEqual(canonical["summary"]["evidence_summary"], canonical["evidence_summary"])

    def test_lateral_contact_length_generates_lead_pads_for_all_package_pads(self) -> None:
        bottom = toy_graph("PART", "bottom", pad_count=0, outline=True, spacing_value=None)
        bottom["objects"] = [
            {"id": 100, "label": "outline", "source_label": "outline", "bbox_reconstructed": [0.0, 0.0, 10.0, 10.0]},
            rect_pad_object(1, [0.0, 4.0, 1.0, 6.0]),
            rect_pad_object(2, [9.0, 4.0, 10.0, 6.0]),
            rect_pad_object(3, [4.0, 0.0, 6.0, 1.0]),
            rect_pad_object(4, [4.0, 9.0, 6.0, 10.0]),
        ]
        bottom["metrics"] = {"axis_scale_x": 1.0, "axis_scale_y": 1.0}
        lead = toy_graph("PART", "lead", pad_count=0, outline=False, spacing_value=None)
        lead["objects"] = [
            {"id": 10, "label": "rect", "source_label": "lead", "bbox_reconstructed": [0.0, 0.0, 2.0, 1.0]}
        ]
        lead["dimensions"] = [
            {
                "id": 20,
                "dimension_id": 20,
                "text": "0.5",
                "kind": "size",
                "axis": "x",
                "target_ids": [10],
                "anchors": ["center", "left_edge"],
                "value": 0.5,
                "status": "accepted",
                "value_source": "text_parser",
            }
        ]

        canonical = integrate_part("PART", [bottom, lead], MultiviewOptions())
        lead_pads = {pad["source_package_pad_id"]: pad for pad in canonical["lead_pads"]}

        self.assertEqual(len(canonical["package_pads"]), 4)
        self.assertEqual(len(lead_pads), 4)
        self.assertEqual(lead_pads[1]["bbox"], [0.0, 4.0, 0.5, 6.0])
        self.assertEqual(lead_pads[2]["bbox"], [9.5, 4.0, 10.0, 6.0])
        self.assertEqual(lead_pads[3]["bbox"], [4.0, 0.0, 6.0, 0.5])
        self.assertEqual(lead_pads[4]["bbox"], [4.0, 9.5, 6.0, 10.0])
        self.assertEqual(canonical["summary"]["lead_pad_count"], 4)
        package_pads = {pad["source_object_id"]: pad for pad in canonical["package_pads"]}
        self.assertEqual(package_pads[1]["bbox"], [0.0, 4.0, 1.0, 6.0])
        overlay = canonical["multiview_overlay"]
        self.assertEqual(overlay["coordinate_mode"], "dimension_scaled_centered")
        self.assertEqual(canonical["summary"]["multiview_overlay_layer_count"], 1)
        self.assertEqual(canonical["summary"]["multiview_overlay_extra_object_count"], 4)
        bottom_layer = overlay["layers"][0]
        self.assertEqual(bottom_layer["raw_view"], "bottom")
        self.assertEqual(bottom_layer["source_frame"], [0.0, 0.0, 10.0, 10.0])
        self.assertEqual(bottom_layer["normalized_frame"], [-5.0, -5.0, 5.0, 5.0])
        lead_extra = {
            obj["source_package_pad_id"]: obj
            for obj in overlay["extra_objects"]
            if obj["role"] == "lead_pad"
        }
        self.assertEqual(lead_extra[1]["bbox"], [-5.0, -1.0, -4.5, 1.0])
        self.assertEqual(lead_extra[1]["coordinate_mode"], "dimension_scaled_centered")
        self.assertEqual(lead_extra[1]["source_bbox"], [0.0, 4.0, 0.5, 6.0])

    def test_lead_edge_to_center_contact_dimension_preferred_over_full_lead_span(self) -> None:
        bottom = toy_graph("PART", "bottom", pad_count=0, outline=True, spacing_value=None)
        bottom["objects"] = [
            {"id": 100, "label": "outline", "source_label": "outline", "bbox_reconstructed": [0.0, 0.0, 10.0, 10.0]},
            rect_pad_object(1, [0.0, 4.0, 1.0, 6.0]),
            rect_pad_object(2, [9.0, 4.0, 10.0, 6.0]),
        ]
        bottom["metrics"] = {"axis_scale_x": 1.0, "axis_scale_y": 1.0}
        lead = toy_graph("PART", "lead", pad_count=0, outline=False, spacing_value=None)
        lead["objects"] = [
            {"id": 10, "label": "rect", "source_label": "lead", "bbox_reconstructed": [0.0, 0.0, 2.0, 1.0]}
        ]
        lead["dimensions"] = [
            {
                "id": 20,
                "dimension_id": 20,
                "text": "0.75 0.50",
                "kind": "size",
                "axis": "x",
                "target_ids": [10],
                "anchors": ["left_edge", "center"],
                "value": 0.5,
                "status": "downweighted",
                "value_source": "text_parser",
            },
            {
                "id": 21,
                "dimension_id": 21,
                "text": "(1)",
                "kind": "size",
                "axis": "x",
                "target_ids": [10],
                "anchors": ["left_edge", "right_edge"],
                "value": 1.0,
                "status": "accepted",
                "value_source": "text_parser",
            },
        ]

        canonical = integrate_part("PART", [bottom, lead], MultiviewOptions())
        lead_pads = {pad["source_package_pad_id"]: pad for pad in canonical["lead_pads"]}

        self.assertEqual(len(lead_pads), 2)
        self.assertEqual(lead_pads[1]["bbox"], [0.0, 4.0, 0.5, 6.0])
        self.assertEqual(lead_pads[2]["bbox"], [9.5, 4.0, 10.0, 6.0])
        self.assertEqual({pad["lead_contact_length"] for pad in lead_pads.values()}, {0.5})
        self.assertEqual({pad["partial_dimension_semantics"] for pad in lead_pads.values()}, {"lead_ground_contact_length"})

    def test_side_center_edge_lead_dimension_preferred_over_l1_full_span(self) -> None:
        bottom = toy_graph("PART", "bottom", pad_count=0, outline=True, spacing_value=None)
        bottom["objects"] = [
            {"id": 100, "label": "outline", "source_label": "outline", "bbox_reconstructed": [0.0, 0.0, 10.0, 10.0]},
            rect_pad_object(1, [4.0, 0.0, 6.0, 1.0]),
            rect_pad_object(2, [4.0, 9.0, 6.0, 10.0]),
        ]
        bottom["metrics"] = {"axis_scale_x": 1.0, "axis_scale_y": 1.0}
        side = toy_graph("PART", "side", pad_count=0, outline=False, spacing_value=None)
        side["objects"] = [
            {"id": 10, "label": "rect", "source_label": "lead", "bbox_reconstructed": [0.0, 0.0, 2.0, 1.0]}
        ]
        side["dimensions"] = [
            {
                "id": 20,
                "dimension_id": 20,
                "text": "L",
                "kind": "size",
                "axis": "x",
                "target_ids": [10],
                "anchors": ["center", "right_edge"],
                "value": 0.35,
                "status": "accepted",
                "value_source": "table_lookup",
            },
            {
                "id": 21,
                "dimension_id": 21,
                "text": "L1",
                "kind": "size",
                "axis": "x",
                "target_ids": [10],
                "anchors": ["left_edge", "right_edge"],
                "value": 0.4,
                "status": "accepted",
                "value_source": "table_lookup",
            },
        ]

        canonical = integrate_part("PART", [bottom, side], MultiviewOptions())
        lead_pads = {pad["source_package_pad_id"]: pad for pad in canonical["lead_pads"]}

        self.assertEqual(len(lead_pads), 2)
        self.assertEqual(lead_pads[1]["bbox"], [4.0, 0.0, 6.0, 0.35])
        self.assertEqual(lead_pads[2]["bbox"], [4.0, 9.65, 6.0, 10.0])
        self.assertEqual({pad["lead_contact_length"] for pad in lead_pads.values()}, {0.35})
        self.assertEqual({pad["lead_contact_length_source"]["text"] for pad in lead_pads.values()}, {"L"})
        self.assertEqual({pad["lead_contact_length_source"]["dimension_id"] for pad in lead_pads.values()}, {20})

    def test_lateral_dual_unit_dimension_uses_inch_value_when_metric_is_first(self) -> None:
        bottom = toy_graph("PART", "bottom", pad_count=0, outline=True, spacing_value=None)
        bottom["objects"] = [
            {"id": 100, "label": "outline", "source_label": "outline", "bbox_reconstructed": [0.0, 0.0, 10.0, 10.0]},
            rect_pad_object(1, [4.0, 0.0, 6.0, 0.2]),
            rect_pad_object(2, [7.0, 0.0, 9.0, 0.2]),
            rect_pad_object(3, [4.0, 9.8, 6.0, 10.0]),
            rect_pad_object(4, [7.0, 9.8, 9.0, 10.0]),
        ]
        bottom["metrics"] = {"axis_scale_x": 1.0, "axis_scale_y": 1.0}
        side = toy_graph("PART", "side", pad_count=0, outline=False, spacing_value=None)
        side["objects"] = [
            {"id": 20, "label": "rect", "source_label": "lead", "bbox_reconstructed": [0.0, 0.0, 2.0, 1.0]}
        ]
        side["dimensions"] = [
            {
                "id": 30,
                "dimension_id": 30,
                "text": "24X1.143 .045",
                "kind": "size",
                "axis": "x",
                "target_ids": [20],
                "anchors": ["left_edge", "right_edge"],
                "value": 1.143,
                "status": "accepted",
                "value_source": "text_parser",
            }
        ]

        canonical = integrate_part("PART", [bottom, side], MultiviewOptions())
        lead_pads = {pad["source_package_pad_id"]: pad for pad in canonical["lead_pads"]}

        self.assertEqual(len(lead_pads), 4)
        self.assert_bbox_close(lead_pads[1]["bbox"], [4.0, 0.0, 6.0, 0.045])
        self.assert_bbox_close(lead_pads[3]["bbox"], [4.0, 9.955, 6.0, 10.0])
        self.assertEqual({pad["lead_contact_length"] for pad in lead_pads.values()}, {0.045})
        self.assertEqual(
            {pad["lead_contact_length_source"]["value_unit_correction"] for pad in lead_pads.values()},
            {"dual_unit_reversed_inch_mm"},
        )
        self.assertEqual(
            {pad["lead_contact_length_source"]["value_unit_correction_original_value"] for pad in lead_pads.values()},
            {1.143},
        )

    def test_front_pad_width_projects_to_tangential_axis_for_side_pads(self) -> None:
        bottom = toy_graph("PART", "bottom", pad_count=0, outline=True, spacing_value=None)
        bottom["objects"] = [
            {"id": 100, "label": "outline", "source_label": "outline", "bbox_reconstructed": [0.0, 0.0, 10.0, 10.0]},
            rect_pad_object(1, [0.0, 4.0, 1.0, 6.0]),
            rect_pad_object(2, [9.0, 4.0, 10.0, 6.0]),
            rect_pad_object(3, [4.0, 0.0, 6.0, 1.0]),
            rect_pad_object(4, [4.0, 9.0, 6.0, 10.0]),
        ]
        bottom["metrics"] = {"axis_scale_x": 1.0, "axis_scale_y": 1.0}
        front = toy_graph("PART", "front", pad_count=0, outline=False, spacing_value=None)
        front["objects"] = [
            {"id": 10, "label": "rect", "source_label": "pad", "bbox_reconstructed": [0.0, 0.0, 2.0, 1.0]}
        ]
        front["dimensions"] = [
            {
                "id": 20,
                "dimension_id": 20,
                "text": "b",
                "kind": "size",
                "axis": "x",
                "target_ids": [10],
                "anchors": ["left_edge", "right_edge"],
                "value": 0.5,
                "status": "accepted",
                "value_source": "table_lookup",
            }
        ]

        canonical = integrate_part("PART", [bottom, front], MultiviewOptions())
        lead_pads = {pad["source_package_pad_id"]: pad for pad in canonical["lead_pads"]}

        self.assertEqual(len(lead_pads), 4)
        self.assertEqual(lead_pads[1]["bbox"], [0.0, 4.75, 1.0, 5.25])
        self.assertEqual(lead_pads[2]["bbox"], [9.0, 4.75, 10.0, 5.25])
        self.assertEqual(lead_pads[3]["bbox"], [4.75, 0.0, 5.25, 1.0])
        self.assertEqual(lead_pads[4]["bbox"], [4.75, 9.0, 5.25, 10.0])
        self.assertEqual({pad["projection_axis"] for pad in lead_pads.values()}, {"x", "y"})
        self.assertEqual({pad["partial_dimension_semantics"] for pad in lead_pads.values()}, {"pad_width"})

    def test_four_side_lead_pads_use_uniform_terminal_minor_cross_axis(self) -> None:
        bottom = toy_graph("PART", "bottom", pad_count=0, outline=True, spacing_value=None)
        bottom["objects"] = [
            {"id": 100, "label": "outline", "source_label": "outline", "bbox_reconstructed": [0.0, 0.0, 10.0, 10.0]},
            rect_pad_object(1, [0.0, 2.0, 1.0, 4.0]),
            rect_pad_object(2, [0.0, 6.0, 1.0, 8.0]),
            rect_pad_object(3, [9.0, 2.0, 10.0, 4.0]),
            rect_pad_object(4, [9.0, 6.0, 10.0, 8.0]),
            rect_pad_object(5, [2.0, 0.0, 4.0, 1.0]),
            rect_pad_object(6, [6.0, 0.0, 8.0, 1.0]),
            rect_pad_object(7, [2.0, 9.0, 4.0, 10.0]),
            rect_pad_object(8, [6.0, 9.0, 8.0, 10.0]),
        ]
        bottom["metrics"] = {"axis_scale_x": 1.0, "axis_scale_y": 1.0}
        lead = toy_graph("PART", "lead", pad_count=0, outline=False, spacing_value=None)
        lead["objects"] = [
            {"id": 30, "label": "rect", "source_label": "lead", "bbox_reconstructed": [0.0, 0.0, 1.0, 1.0]}
        ]
        lead["dimensions"] = [
            {
                "id": 40,
                "dimension_id": 40,
                "text": "L",
                "kind": "size",
                "axis": "x",
                "target_ids": [30],
                "anchors": ["center", "left_edge"],
                "value": 0.5,
                "status": "accepted",
                "value_source": "text_parser",
            }
        ]

        canonical = integrate_part("PART", [bottom, lead], MultiviewOptions())
        lead_pads = {pad["source_package_pad_id"]: pad for pad in canonical["lead_pads"]}

        self.assertEqual(len(lead_pads), 8)
        self.assertEqual(lead_pads[1]["bbox"], [0.0, 2.5, 0.5, 3.5])
        self.assertEqual(lead_pads[5]["bbox"], [2.5, 0.0, 3.5, 0.5])
        self.assertEqual({pad["uniform_cross_axis_source"] for pad in lead_pads.values()}, {"terminal_minor_median"})
        self.assertEqual({pad["uniform_cross_axis_length"] for pad in lead_pads.values()}, {1.0})

    def test_front_pad_width_preserves_vertical_length_for_left_right_only_pads(self) -> None:
        bottom = toy_graph("PART", "bottom", pad_count=0, outline=True, spacing_value=None)
        bottom["objects"] = [
            {"id": 100, "label": "outline", "source_label": "outline", "bbox_reconstructed": [0.0, 0.0, 10.0, 10.0]},
            rect_pad_object(1, [0.0, 2.0, 1.0, 8.0]),
            rect_pad_object(2, [9.0, 2.0, 10.0, 8.0]),
        ]
        bottom["metrics"] = {"axis_scale_x": 1.0, "axis_scale_y": 1.0}
        front = toy_graph("PART", "front", pad_count=0, outline=False, spacing_value=None)
        front["objects"] = [
            {"id": 10, "label": "rect", "source_label": "pad", "bbox_reconstructed": [0.0, 0.0, 2.0, 1.0]}
        ]
        front["dimensions"] = [
            {
                "id": 20,
                "dimension_id": 20,
                "text": "b",
                "kind": "size",
                "axis": "x",
                "target_ids": [10],
                "anchors": ["left_edge", "right_edge"],
                "value": 0.5,
                "status": "accepted",
                "value_source": "table_lookup",
            }
        ]

        canonical = integrate_part("PART", [bottom, front], MultiviewOptions())
        lead_pads = {pad["source_package_pad_id"]: pad for pad in canonical["lead_pads"]}

        self.assertEqual(len(lead_pads), 2)
        self.assertEqual(lead_pads[1]["bbox"], [0.0, 2.0, 0.5, 8.0])
        self.assertEqual(lead_pads[2]["bbox"], [9.5, 2.0, 10.0, 8.0])
        self.assertEqual({pad["projection_axis"] for pad in lead_pads.values()}, {"x"})
        self.assertEqual({pad["partial_dimension_semantics"] for pad in lead_pads.values()}, {"pad_width"})

    def test_front_pad_width_left_right_only_does_not_promote_to_contact_length(self) -> None:
        bottom = toy_graph("PART", "bottom", pad_count=0, outline=True, spacing_value=None)
        bottom["objects"] = [
            {"id": 100, "label": "outline", "source_label": "outline", "bbox_reconstructed": [0.0, 0.0, 10.0, 10.0]},
            rect_pad_object(1, [0.0, 2.0, 0.2, 8.0]),
            rect_pad_object(2, [9.8, 2.0, 10.0, 8.0]),
        ]
        bottom["metrics"] = {"axis_scale_x": 1.0, "axis_scale_y": 1.0}
        front = toy_graph("PART", "front", pad_count=0, outline=False, spacing_value=None)
        front["objects"] = [
            {"id": 10, "label": "rect", "source_label": "pad", "bbox_reconstructed": [0.0, 0.0, 2.0, 1.0]}
        ]
        front["dimensions"] = [
            {
                "id": 20,
                "dimension_id": 20,
                "text": "b",
                "kind": "size",
                "axis": "x",
                "target_ids": [10],
                "anchors": ["left_edge", "right_edge"],
                "value": 0.6,
                "status": "accepted",
                "value_source": "table_lookup",
            }
        ]

        canonical = integrate_part("PART", [bottom, front], MultiviewOptions())
        lead_pads = {pad["source_package_pad_id"]: pad for pad in canonical["lead_pads"]}

        self.assertEqual(len(lead_pads), 2)
        self.assert_bbox_close(lead_pads[1]["bbox"], [0.0, 2.0, 0.6, 8.0])
        self.assert_bbox_close(lead_pads[2]["bbox"], [9.4, 2.0, 10.0, 8.0])
        self.assertEqual({pad["projection_axis"] for pad in lead_pads.values()}, {"x"})
        self.assertEqual({pad["partial_dimension_semantics"] for pad in lead_pads.values()}, {"pad_width"})

    def test_front_pad_width_uses_row_axis_for_top_bottom_pad_rows(self) -> None:
        bottom = toy_graph("PART", "bottom", pad_count=0, outline=True, spacing_value=None)
        bottom["objects"] = [
            {"id": 100, "label": "outline", "source_label": "outline", "bbox_reconstructed": [0.0, 0.0, 20.0, 10.0]},
        ]
        for index, x in enumerate([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0], start=1):
            bottom["objects"].append(rect_pad_object(index, [x - 0.25, 0.0, x + 0.25, 1.5]))
        for index, x in enumerate([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0], start=11):
            bottom["objects"].append(rect_pad_object(index, [x - 0.25, 8.5, x + 0.25, 10.0]))
        bottom["metrics"] = {"axis_scale_x": 1.0, "axis_scale_y": 1.0}
        front = toy_graph("PART", "front", pad_count=0, outline=False, spacing_value=None)
        front["objects"] = [
            {"id": 30, "label": "rect", "source_label": "pad", "bbox_reconstructed": [0.0, 0.0, 1.0, 1.0]}
        ]
        front["dimensions"] = [
            {
                "id": 40,
                "dimension_id": 40,
                "text": "b",
                "kind": "size",
                "axis": "x",
                "target_ids": [30],
                "anchors": ["left_edge", "right_edge"],
                "value": 0.2,
                "status": "accepted",
                "value_source": "text_parser",
            }
        ]

        canonical = integrate_part("PART", [bottom, front], MultiviewOptions())
        lead_pads = {pad["source_package_pad_id"]: pad for pad in canonical["lead_pads"]}

        self.assertEqual(len(lead_pads), 20)
        self.assertEqual({pad["projection_axis"] for pad in lead_pads.values()}, {"x"})
        self.assertEqual(lead_pads[1]["bbox"], [0.9, 0.0, 1.1, 1.5])
        self.assertEqual(lead_pads[10]["bbox"], [18.9, 0.0, 19.1, 1.5])
        self.assertEqual(lead_pads[11]["bbox"], [0.9, 8.5, 1.1, 10.0])
        self.assertEqual(lead_pads[20]["bbox"], [18.9, 8.5, 19.1, 10.0])

    def test_front_pad_width_top_bottom_rows_prefers_narrow_terminal_dimension(self) -> None:
        bottom = toy_graph("PART", "bottom", pad_count=0, outline=True, spacing_value=None)
        bottom["objects"] = [
            {"id": 100, "label": "outline", "source_label": "outline", "bbox_reconstructed": [0.0, 0.0, 20.0, 10.0]},
        ]
        for index, x in enumerate([1.0, 3.0, 5.0, 7.0], start=1):
            bottom["objects"].append(rect_pad_object(index, [x - 0.25, 0.0, x + 0.25, 1.5]))
        for index, x in enumerate([1.0, 3.0, 5.0, 7.0], start=5):
            bottom["objects"].append(rect_pad_object(index, [x - 0.25, 8.5, x + 0.25, 10.0]))
        bottom["metrics"] = {"axis_scale_x": 1.0, "axis_scale_y": 1.0}
        front = toy_graph("PART", "front", pad_count=0, outline=False, spacing_value=None)
        front["objects"] = [
            rect_pad_object(30, [0.0, 0.0, 1.0, 1.0]),
            rect_pad_object(31, [2.0, 0.0, 3.0, 1.0]),
        ]
        front["dimensions"] = [
            {
                "id": 40,
                "dimension_id": 40,
                "text": "terminal",
                "kind": "size",
                "axis": "x",
                "target_ids": [30],
                "anchors": ["left_edge", "right_edge"],
                "value": 0.2,
                "status": "accepted",
                "value_source": "text_parser",
            },
            {
                "id": 41,
                "dimension_id": 41,
                "text": "body lead bend",
                "kind": "size",
                "axis": "x",
                "target_ids": [31],
                "anchors": ["left_edge", "right_edge"],
                "value": 0.6,
                "status": "accepted",
                "value_source": "text_parser",
            },
        ]

        canonical = integrate_part("PART", [bottom, front], MultiviewOptions())
        lead_pads = {pad["source_package_pad_id"]: pad for pad in canonical["lead_pads"]}

        self.assertEqual(len(lead_pads), 8)
        self.assertEqual({pad["lead_contact_length"] for pad in lead_pads.values()}, {0.2})
        self.assertEqual({pad["lead_contact_length_source"]["dimension_id"] for pad in lead_pads.values()}, {40})
        self.assertEqual(lead_pads[1]["bbox"], [0.9, 0.0, 1.1, 1.5])
        self.assertEqual(lead_pads[5]["bbox"], [0.9, 8.5, 1.1, 10.0])

    def test_dense_central_thermal_pad_is_optional_inner_land_pad(self) -> None:
        bottom = toy_graph("PART", "bottom", pad_count=0, outline=True, spacing_value=None)
        bottom["objects"] = [
            {"id": 100, "label": "outline", "source_label": "outline", "bbox_reconstructed": [-5.0, -5.0, 5.0, 5.0]},
        ]
        object_id = 0
        for x in [-4.0, -2.4, -0.8, 0.8, 2.4, 4.0]:
            bottom["objects"].append(rect_pad_object(object_id, [x - 0.2, -4.4, x + 0.2, -4.0]))
            object_id += 1
            bottom["objects"].append(rect_pad_object(object_id, [x - 0.2, 4.0, x + 0.2, 4.4]))
            object_id += 1
        for y in [-3.2, -1.9, -0.6, 0.6, 1.9, 3.2]:
            bottom["objects"].append(rect_pad_object(object_id, [-4.4, y - 0.2, -4.0, y + 0.2]))
            object_id += 1
            bottom["objects"].append(rect_pad_object(object_id, [4.0, y - 0.2, 4.4, y + 0.2]))
            object_id += 1
        bottom["objects"].append(rect_pad_object(999, [-2.0, -2.0, 2.0, 2.0]))

        canonical = integrate_part("PART", [bottom], MultiviewOptions())

        self.assertEqual(len(canonical["package_pads"]), 24)
        self.assertEqual(len(canonical["inner_land_pads"]), 1)
        self.assertEqual(canonical["inner_land_pads"][0]["source_object_id"], 999)
        self.assertEqual(canonical["inner_land_pads"][0]["reclassified_reason"], "central_thermal_package_pad")

    def test_lead_detail_uses_row_axis_for_top_bottom_corner_pads(self) -> None:
        bottom = toy_graph("PART", "bottom", pad_count=0, outline=True, spacing_value=None)
        bottom["objects"] = [
            {"id": 100, "label": "outline", "source_label": "outline", "bbox_reconstructed": [0.0, 0.0, 20.0, 10.0]},
        ]
        for index, x in enumerate([1.0, 3.0, 5.0, 15.0, 17.0, 19.0], start=1):
            bottom["objects"].append(rect_pad_object(index, [x - 0.25, 0.0, x + 0.25, 1.5]))
        for index, x in enumerate([1.0, 3.0, 5.0, 15.0, 17.0, 19.0], start=7):
            bottom["objects"].append(rect_pad_object(index, [x - 0.25, 8.5, x + 0.25, 10.0]))
        bottom["metrics"] = {"axis_scale_x": 1.0, "axis_scale_y": 1.0}
        lead = toy_graph("PART", "lead", pad_count=0, outline=False, spacing_value=None)
        lead["objects"] = [
            {"id": 30, "label": "rect", "source_label": "lead", "bbox_reconstructed": [0.0, 0.0, 1.0, 1.0]}
        ]
        lead["dimensions"] = [
            {
                "id": 40,
                "dimension_id": 40,
                "text": "0.5",
                "kind": "size",
                "axis": "x",
                "target_ids": [30],
                "anchors": ["center", "left_edge"],
                "value": 0.5,
                "status": "accepted",
                "value_source": "text_parser",
            }
        ]

        canonical = integrate_part("PART", [bottom, lead], MultiviewOptions())
        lead_pads = {pad["source_package_pad_id"]: pad for pad in canonical["lead_pads"]}

        self.assertEqual(len(lead_pads), 12)
        self.assertEqual({pad["radial_axis"] for pad in lead_pads.values()}, {"y"})
        self.assertEqual({pad["projection_axis"] for pad in lead_pads.values()}, {"y"})
        self.assertEqual(lead_pads[1]["bbox"], [0.75, 0.5, 1.25, 1.0])
        self.assertEqual(lead_pads[6]["bbox"], [18.75, 0.5, 19.25, 1.0])
        self.assertEqual(lead_pads[7]["bbox"], [0.75, 9.0, 1.25, 9.5])
        self.assertEqual(lead_pads[12]["bbox"], [18.75, 9.0, 19.25, 9.5])

    def test_source_graphless_derived_lead_pads_are_not_forced_into_multiview_overlay(self) -> None:
        bottom = toy_graph("PART", "bottom", pad_count=0, outline=True, spacing_value=None)
        bottom["objects"] = [
            {"id": 100, "label": "outline", "source_label": "outline", "bbox_reconstructed": [0.0, 0.0, 10.0, 6.0]},
        ]
        lead_pads = [
            {
                "source_object_id": 1,
                "role": "partial_pad_width",
                "label": "partial_pad_width",
                "source_label": "Rectangle",
                "raw_view": "front",
                "canonical_view": "lateral",
                "bbox": [0.2, 2.0, 0.8, 4.0],
                "source_graph": "",
                "source_type": "derived_partial_evidence_multiview",
                "source_package_pad_id": 1,
                "source_package_pad_bbox": [0.0, 2.0, 1.0, 4.0],
            },
            {
                "source_object_id": 2,
                "role": "partial_pad_width",
                "label": "partial_pad_width",
                "source_label": "Rectangle",
                "raw_view": "front",
                "canonical_view": "lateral",
                "bbox": [9.2, 2.0, 9.8, 4.0],
                "source_graph": "",
                "source_type": "derived_partial_evidence_multiview",
                "source_package_pad_id": 2,
                "source_package_pad_bbox": [9.0, 2.0, 10.0, 4.0],
            },
        ]

        from real_image_process.FPK_PJ_fullflow.multiview.integrator import build_multiview_overlay_payload

        overlay = build_multiview_overlay_payload([bottom], lead_pads, [], MultiviewOptions())
        extras = overlay["extra_objects"]

        self.assertEqual(extras, [])

    def test_land_detail_generates_inner_land_pads_for_selected_land_pads(self) -> None:
        bottom = toy_graph("PART", "bottom", pad_count=1, outline=True, spacing_value=None)
        land = toy_graph("PART", "land", pad_count=0, outline=False, spacing_value=None)
        land["objects"] = [
            rect_pad_object(10, [10.0, 20.0, 20.0, 40.0]),
            rect_pad_object(11, [30.0, 20.0, 40.0, 40.0]),
        ]
        land_detail = toy_graph("PART", "land_detail", pad_count=0, outline=False, spacing_value=None)
        land_detail["objects"] = [
            rect_pad_object(20, [0.0, 0.0, 10.0, 10.0]),
            rect_pad_object(21, [2.0, 3.0, 8.0, 7.0]),
        ]

        canonical = integrate_part("PART", [bottom, land, land_detail], MultiviewOptions())
        inner_land_pads = {pad["source_land_pad_id"]: pad for pad in canonical["inner_land_pads"]}

        self.assertEqual(len(canonical["land_pads"]), 2)
        self.assertEqual(len(inner_land_pads), 2)
        self.assertEqual(inner_land_pads[10]["bbox"], [12.0, 26.0, 18.0, 34.0])
        self.assertEqual(inner_land_pads[11]["bbox"], [32.0, 26.0, 38.0, 34.0])
        self.assertEqual(inner_land_pads[10]["role"], "inner_land_pad")
        self.assertEqual(inner_land_pads[10]["inner_land_pad_source"]["raw_view"], "land_detail")
        self.assertEqual(canonical["summary"]["inner_land_pad_count"], 2)
        self.assertEqual(canonical["land_pads"][0]["bbox"], [10.0, 20.0, 20.0, 40.0])
        inner_extras = {
            obj["source_land_pad_id"]: obj
            for obj in canonical["multiview_overlay"]["extra_objects"]
            if obj["role"] == "inner_land_pad"
        }
        self.assertEqual(inner_extras[10]["bbox"], [-13.0, -4.0, -7.0, 4.0])
        self.assertEqual(inner_extras[11]["bbox"], [7.0, -4.0, 13.0, 4.0])

    def test_land_detail_does_not_generate_inner_land_pad_for_central_large_rect_pad(self) -> None:
        bottom = toy_graph("PART", "bottom", pad_count=1, outline=True, spacing_value=None)
        land = toy_graph("PART", "land", pad_count=0, outline=False, spacing_value=None)
        land["objects"] = [
            rect_pad_object(10, [0.0, 4.0, 1.0, 6.0]),
            rect_pad_object(11, [9.0, 4.0, 10.0, 6.0]),
            rect_pad_object(12, [0.0, 6.5, 1.0, 8.5]),
            rect_pad_object(13, [9.0, 6.5, 10.0, 8.5]),
            rect_pad_object(14, [3.0, 3.0, 7.0, 9.0]),
        ]
        land_detail = toy_graph("PART", "land_detail", pad_count=0, outline=False, spacing_value=None)
        land_detail["objects"] = [
            rect_pad_object(20, [0.0, 0.0, 10.0, 10.0]),
            rect_pad_object(21, [2.0, 3.0, 8.0, 7.0]),
        ]

        canonical = integrate_part("PART", [bottom, land, land_detail], MultiviewOptions())
        inner_ids = {pad["source_land_pad_id"] for pad in canonical["inner_land_pads"]}

        self.assertEqual(inner_ids, {10, 11, 12, 13})
        self.assertNotIn(14, inner_ids)
        self.assertEqual(canonical["summary"]["inner_land_pad_count"], 4)

    def test_land_detail_distance_dimension_insets_matching_circle_land_pads(self) -> None:
        bottom = toy_graph("PART", "bottom", pad_count=1, outline=True, spacing_value=None)
        land = toy_graph("PART", "land", pad_count=0, outline=False, spacing_value=None)
        land["objects"] = [
            {"id": 10, "label": "circle", "source_label": "pad_circle", "bbox_reconstructed": [0.0, 0.0, 10.0, 10.0]},
            rect_pad_object(11, [20.0, 0.0, 30.0, 10.0]),
        ]
        land_detail = toy_graph("PART", "land_detail", pad_count=0, outline=False, spacing_value=None)
        land_detail["objects"] = [
            {"id": 20, "label": "circle", "source_label": "pad_circle", "bbox_reconstructed": [0.0, 0.0, 10.0, 10.0]},
            {"id": 21, "label": "circle", "source_label": "pad_circle", "bbox_reconstructed": [0.1, 0.1, 10.1, 10.1]},
        ]
        land_detail["metrics"] = {"axis_scale_x": 0.1, "axis_scale_y": 0.1, "global_scale": 0.1}
        land_detail["dimensions"] = [
            {
                "id": 30,
                "dimension_id": 30,
                "text": "0.20 MAX",
                "kind": "distance",
                "axis": "x",
                "target_ids": [21],
                "anchors": ["edge", "edge"],
                "value": 0.2,
                "status": "accepted",
                "value_source": "text_parser",
            },
            {
                "id": 31,
                "dimension_id": 31,
                "text": "(0.50) METAL",
                "kind": "diameter",
                "axis": None,
                "target_ids": [21],
                "anchors": ["center", "center"],
                "value": 0.5,
                "status": "accepted",
                "value_source": "text_parser",
            }
        ]

        canonical = integrate_part("PART", [bottom, land, land_detail], MultiviewOptions())

        self.assertEqual(len(canonical["inner_land_pads"]), 1)
        inner = canonical["inner_land_pads"][0]
        self.assertEqual(inner["source_land_pad_id"], 10)
        self.assertEqual(inner["bbox"], [2.0, 2.0, 8.0, 8.0])
        self.assertEqual(inner["inner_land_pad_source"]["template_type"], "dimension_inset")
        self.assertEqual(inner["inner_land_pad_source"]["shape_family"], "circle")
        self.assertEqual(inner["inner_land_pad_source"]["dimension_value"], 0.2)
        self.assertEqual(inner["inner_land_pad_source"]["inset_margin_x"], 2.0)
        self.assertEqual(inner["inner_land_pad_source"]["inset_margin_y"], 2.0)
        self.assertEqual(inner["inner_land_pad_source"]["coordinate_unit_scale_x"], 0.1)
        self.assertEqual(inner["inner_land_pad_source"]["coordinate_unit_scale_y"], 0.1)

    def test_multiview_overlay_rotates_layers_to_reference_orientation(self) -> None:
        options = MultiviewOptions()
        bottom = toy_graph("PART", "bottom", pad_count=3, outline=True, spacing_value=None)
        land = toy_graph("PART", "land", pad_count=3, outline=True, spacing_value=None)
        bottom["objects"] = [
            {"id": 100, "label": "outline", "source_label": "outline", "bbox_reconstructed": [0.0, 0.0, 10.0, 20.0]},
            rect_pad_object(1, [0.0, 0.0, 2.0, 2.0]),
            rect_pad_object(2, [8.0, 0.0, 10.0, 2.0]),
            rect_pad_object(3, [0.0, 18.0, 2.0, 20.0]),
        ]
        land["objects"] = [
            {"id": 200, "label": "outline", "source_label": "outline", "bbox_reconstructed": [0.0, 0.0, 20.0, 10.0]},
            rect_pad_object(11, [0.0, 0.0, 2.0, 2.0]),
            rect_pad_object(12, [0.0, 8.0, 2.0, 10.0]),
            rect_pad_object(13, [18.0, 0.0, 20.0, 2.0]),
        ]
        bottom["metrics"] = {"axis_scale_x": 1.0, "axis_scale_y": 1.0}
        land["metrics"] = {"axis_scale_x": 1.0, "axis_scale_y": 1.0}

        canonical = integrate_part("PART", [bottom, land], options)
        overlay = canonical["multiview_overlay"]

        self.assertEqual(overlay["rotation_normalization"]["status"], "aligned")
        rotations = {item["raw_view"]: item["rotation_degrees"] for item in overlay["rotation_normalization"]["layer_rotations"]}
        self.assertEqual(rotations["land"], 0)
        self.assertEqual(rotations["bottom"], 90)
        bottom_layer = next(layer for layer in overlay["layers"] if layer["raw_view"] == "bottom")
        self.assertEqual(bottom_layer["rotation_degrees"], 90)
        rotated_pads = {obj["source_object_id"]: obj for obj in bottom_layer["objects"] if obj.get("role") == "package_pad"}
        self.assertEqual(rotated_pads[1]["bbox"], [-10.0, 3.0, -8.0, 5.0])
        self.assertEqual(overlay["extra_objects"], [])

    def test_multiview_rotation_uses_outline_as_candidate_gate(self) -> None:
        reference = {
            "objects": [
                {"role": "outline_2d", "bbox": [-10.0, -5.0, 10.0, 5.0]},
                {"role": "land_pad", "bbox": [-0.5, -0.5, 0.5, 0.5]},
                {"role": "land_pad", "bbox": [2.5, -0.5, 3.5, 0.5]},
            ]
        }
        candidate = {
            "objects": [
                {"role": "outline_2d", "bbox": [-10.0, -5.0, 10.0, 5.0]},
                {"role": "package_pad", "bbox": [-0.5, -0.5, 0.5, 0.5]},
                {"role": "package_pad", "bbox": [-0.5, 2.5, 0.5, 3.5]},
            ]
        }

        result = best_multiview_layer_rotation(layer_rotation_boxes(reference), reference, candidate)

        self.assertEqual(result["rotation_degrees"], 0)
        candidate_by_rotation = {item["rotation_degrees"]: item for item in result["scores"]}
        self.assertTrue(candidate_by_rotation[0]["eligible_by_outline"])
        self.assertFalse(candidate_by_rotation[90]["eligible_by_outline"])
        self.assertFalse(candidate_by_rotation[270]["eligible_by_outline"])
        self.assertGreater(candidate_by_rotation[90]["iou"], candidate_by_rotation[0]["iou"])

    def test_multiview_rotation_keeps_pad_layout_when_outline_is_near_tie(self) -> None:
        reference = {
            "objects": [
                {"role": "outline_2d", "bbox": [-6.80180348307, -8.14945902817, 6.79819603389, 8.1505384041]},
                {"role": "land_pad", "bbox": [5.10276832637, -7.91323965623, 5.776857958154, -5.975732543195]},
                {"role": "land_pad", "bbox": [5.102139120118, 5.984763777209, 5.776228751901, 7.922270890243]},
            ]
        }
        candidate = {
            "objects": [
                {"role": "outline_2d", "bbox": [-6.748009998713, -5.993196167039, 6.851992033578, 6.006793912167]},
                {"role": "package_pad", "bbox": [5.19912093167, -7.51196431447, 5.666857798595, -6.007293829268]},
                {"role": "package_pad", "bbox": [5.199082554835, 6.007293829268, 5.66681942176, 7.51196431447]},
            ]
        }

        result = best_multiview_layer_rotation(layer_rotation_boxes(reference), reference, candidate)

        self.assertEqual(result["rotation_degrees"], 0)
        candidate_by_rotation = {item["rotation_degrees"]: item for item in result["scores"]}
        self.assertGreater(candidate_by_rotation[90]["outline_iou"], candidate_by_rotation[0]["outline_iou"])
        self.assertGreater(candidate_by_rotation[0]["iou"], candidate_by_rotation[90]["iou"])
        self.assertTrue(candidate_by_rotation[0]["eligible_by_outline"])
        self.assertTrue(candidate_by_rotation[90]["eligible_by_outline"])

    def test_multiview_rotation_accepts_low_confidence_nonzero_when_zero_has_no_overlap(self) -> None:
        reference = {
            "objects": [
                {"role": "land_pad", "bbox": [-0.032527053674, -0.085856579169, 0.032527053674, -0.051101038202]},
                {"role": "land_pad", "bbox": [-0.032527053674, 0.051101038202, 0.032527053674, 0.085856579169]},
            ]
        }
        candidate = {
            "objects": [
                {"role": "package_pad", "bbox": [-0.100913202846, -0.029500220238, -0.084912928816, 0.029500220238]},
                {"role": "package_pad", "bbox": [0.084912928816, -0.058915552803, 0.100913202846, 0.058915552803]},
            ]
        }

        result = best_multiview_layer_rotation(layer_rotation_boxes(reference), reference, candidate)

        self.assertEqual(result["rotation_degrees"], 90)
        candidate_by_rotation = {item["rotation_degrees"]: item for item in result["scores"]}
        self.assertEqual(candidate_by_rotation[0]["iou"], 0.0)
        self.assertGreater(candidate_by_rotation[90]["iou"], candidate_by_rotation[0]["iou"])
        self.assertLess(candidate_by_rotation[90]["iou"], 0.05)

    def test_multiview_rotation_suppresses_low_confidence_nonzero_when_zero_has_overlap(self) -> None:
        reference = {
            "objects": [
                {"role": "land_pad", "bbox": [-0.0325, -0.0858, 0.0325, -0.0511]},
                {"role": "land_pad", "bbox": [-0.0325, 0.0511, 0.0325, 0.0858]},
            ]
        }
        candidate = {
            "objects": [
                {"role": "package_pad", "bbox": [-0.1009, -0.0295, -0.0849, 0.0295]},
                {"role": "package_pad", "bbox": [0.0849, -0.0589, 0.1009, 0.0589]},
                {"role": "package_pad", "bbox": [-0.0325, -0.0858, -0.0315, -0.0511]},
            ]
        }

        result = best_multiview_layer_rotation(layer_rotation_boxes(reference), reference, candidate)

        self.assertEqual(result["rotation_degrees"], 0)
        candidate_by_rotation = {item["rotation_degrees"]: item for item in result["scores"]}
        self.assertGreater(candidate_by_rotation[0]["iou"], 0.0)
        self.assertGreater(candidate_by_rotation[90]["iou"], candidate_by_rotation[0]["iou"])
        self.assertLess(candidate_by_rotation[90]["iou"], 0.05)

    def test_multiview_rotation_prefers_top_anchor_only_for_top_land_source_views(self) -> None:
        top = {
            "raw_view": "top",
            "graph_path": "top.package_graph.json",
            "normalized_frame": [-3.0, -1.0, 3.0, 1.0],
            "objects": [
                {"role": "outline_2d", "bbox": [-3.0, -1.0, 3.0, 1.0]},
                {"role": "package_pad", "bbox": [-3.5, -0.5, -2.5, 0.5]},
                {"role": "package_pad", "bbox": [2.5, -0.5, 3.5, 0.5]},
            ],
        }
        land = {
            "raw_view": "land",
            "graph_path": "land.package_graph.json",
            "normalized_frame": [-0.5, -3.5, 0.5, 3.5],
            "objects": [
                {"role": "land_pad", "bbox": [-0.5, -3.5, 0.5, -2.5]},
                {"role": "land_pad", "bbox": [-0.5, 2.5, 0.5, 3.5]},
            ],
        }

        top_land_reference = select_rotation_reference_layer([land, top], source_views={"top", "land"})
        with_side_reference = select_rotation_reference_layer([land, top], source_views={"top", "land", "side"})

        self.assertIs(top_land_reference, top)
        self.assertIs(with_side_reference, land)

    def test_multiview_overlay_skips_planar_outline_only_layers(self) -> None:
        bottom = toy_graph("PART", "bottom", pad_count=2, outline=False, spacing_value=None)
        top = toy_graph("PART", "top", pad_count=0, outline=True, spacing_value=None)

        canonical = integrate_part("PART", [bottom, top], MultiviewOptions())
        overlay = canonical["multiview_overlay"]

        self.assertEqual([layer["raw_view"] for layer in overlay["layers"]], ["bottom"])
        self.assertEqual(canonical["summary"]["multiview_overlay_layer_count"], 1)

    def test_conflict_detection_records_cross_view_dimension_mismatch(self) -> None:
        options = MultiviewOptions(conflict_abs_tol=0.05, conflict_rel_tol=0.05)
        canonical = integrate_part(
            "PART",
            [
                toy_graph("PART", "bottom", pad_count=0, outline=True, spacing_value=None, y_height_value=1.0),
                toy_graph("PART", "top", pad_count=0, outline=True, spacing_value=None, y_height_value=1.2),
            ],
            options,
        )

        conflicts = canonical["conflicts"]
        self.assertTrue(any(item.get("type") == "dimension_value_conflict" for item in conflicts))
        conflict = next(item for item in conflicts if item.get("type") == "dimension_value_conflict")
        self.assertEqual(conflict["primary"]["value"], 1.0)
        self.assertEqual(conflict["other"]["value"], 1.2)

    def test_conflict_detection_does_not_compare_unmatched_pad_dimensions_across_views(self) -> None:
        options = MultiviewOptions(conflict_abs_tol=0.05, conflict_rel_tol=0.05)
        bottom = toy_graph("PART", "bottom", pad_count=1, outline=True, spacing_value=None)
        bottom["dimensions"].append(
            {
                "id": 10,
                "dimension_id": 10,
                "text": "1.0",
                "kind": "size",
                "axis": "x",
                "target_ids": [0],
                "anchors": ["left_edge", "right_edge"],
                "value": 1.0,
                "status": "accepted",
                "value_source": "text_parser",
            }
        )
        land = toy_graph("PART", "land", pad_count=1, outline=False, spacing_value=None)
        land["dimensions"].append(
            {
                "id": 11,
                "dimension_id": 11,
                "text": "1.4",
                "kind": "size",
                "axis": "x",
                "target_ids": [0],
                "anchors": ["left_edge", "right_edge"],
                "value": 1.4,
                "status": "accepted",
                "value_source": "text_parser",
            }
        )

        canonical = integrate_part("PART", [bottom, land], options)

        self.assertFalse(any(item.get("type") == "dimension_value_conflict" for item in canonical["conflicts"]))

    def test_missing_views_do_not_block_unified_multiview_layers(self) -> None:
        canonical = integrate_part("PART", [toy_graph("PART", "bottom", pad_count=2, outline=True, spacing_value=1.0)], MultiviewOptions())

        self.assertEqual(canonical["part_number"], "PART")
        self.assertEqual(len(canonical["package_pads"]), 2)
        self.assertEqual(canonical["missing_canonical_views"], ["land", "lateral", "lead_detail"])
        self.assertGreater(canonical["summary"]["risk_score"], 0.0)

    def test_package_pad_source_selection_records_top_fallback_when_bottom_missing(self) -> None:
        canonical = integrate_part("PART", [toy_graph("PART", "top", pad_count=2, outline=True, spacing_value=1.0)], MultiviewOptions())

        selection = canonical["source_selection"]["package_pads"]
        self.assertEqual(selection["primary_view"], "bottom")
        self.assertEqual(selection["selected_raw_view"], "top")
        self.assertEqual(selection["preferred_views"], ["bottom", "top"])
        self.assertTrue(selection["used_fallback"])
        self.assertTrue(selection["missing_primary"])
        self.assertEqual(canonical["summary"]["source_selection"]["package_pads"], selection)

    def test_package_pad_source_selection_uses_top_when_bottom_layout_is_sparse(self) -> None:
        sparse_bottom = toy_graph("PART", "bottom", pad_count=2, outline=False, spacing_value=1.0)
        complete_top = toy_graph("PART", "top", pad_count=6, outline=True, spacing_value=1.0)

        canonical = integrate_part("PART", [sparse_bottom, complete_top], MultiviewOptions())

        selection = canonical["source_selection"]["package_pads"]
        self.assertEqual(selection["selected_raw_view"], "top")
        self.assertTrue(selection["used_fallback"])
        self.assertFalse(selection["missing_primary"])
        self.assertEqual(selection["fallback_reason"], "primary_package_pad_layout_sparse")
        self.assertEqual(len(canonical["package_pads"]), 6)

    def test_package_pad_source_selection_uses_top_when_bottom_is_overdense_and_top_matches_land_count(self) -> None:
        overdense_bottom = toy_graph("PART", "bottom", pad_count=19, outline=True, spacing_value=1.0)
        complete_top = toy_graph("PART", "top", pad_count=5, outline=True, spacing_value=1.0)
        complete_top["objects"].append(rect_pad_object(99, [1.0, 3.0, 9.0, 5.0]))
        complete_top["dimensions"].extend(
            [
                {
                    "id": 20,
                    "dimension_id": 20,
                    "text": "D1",
                    "kind": "size",
                    "axis": "y",
                    "target_ids": [99],
                    "anchors": ["top_edge", "bottom_edge"],
                    "value": 4.0,
                    "status": "accepted",
                    "lookup_symbol": "D1",
                    "lookup_symbols": ["D1"],
                    "value_source": "table_lookup",
                },
                {
                    "id": 21,
                    "dimension_id": 21,
                    "text": "E1",
                    "kind": "size",
                    "axis": "x",
                    "target_ids": [99],
                    "anchors": ["left_edge", "right_edge"],
                    "value": 8.0,
                    "status": "accepted",
                    "lookup_symbol": "E1",
                    "lookup_symbols": ["E1"],
                    "value_source": "table_lookup",
                },
            ]
        )
        land = toy_graph("PART", "land", pad_count=5, outline=False, spacing_value=1.0)

        canonical = integrate_part("PART", [overdense_bottom, complete_top, land], MultiviewOptions())

        selection = canonical["source_selection"]["package_pads"]
        self.assertEqual(selection["selected_raw_view"], "top")
        self.assertTrue(selection["used_fallback"])
        self.assertFalse(selection["missing_primary"])
        self.assertEqual(selection["fallback_reason"], "primary_package_pad_layout_overdense_top_matches_land_count")
        self.assertEqual(len(canonical["package_pads"]), 5)
        self.assertEqual(len(canonical["land_pads"]), 5)
        self.assertEqual(len(canonical["filtered_package_pads"]), 1)
        self.assertEqual(canonical["filtered_package_pads"][0]["source_object_id"], 99)
        self.assertEqual(canonical["filtered_package_pads"][0]["filtered_reason"], "body_dimension_target")

    def test_body_dimension_target_filter_does_not_apply_without_overdense_top_fallback(self) -> None:
        bottom = toy_graph("PART", "bottom", pad_count=5, outline=True, spacing_value=1.0)
        bottom["objects"].append(rect_pad_object(99, [1.0, 3.0, 9.0, 5.0]))
        bottom["dimensions"].append(
            {
                "id": 20,
                "dimension_id": 20,
                "text": "D1",
                "kind": "size",
                "axis": "y",
                "target_ids": [99],
                "anchors": ["top_edge", "bottom_edge"],
                "value": 4.0,
                "status": "accepted",
                "lookup_symbol": "D1",
                "lookup_symbols": ["D1"],
                "value_source": "table_lookup",
            }
        )

        canonical = integrate_part("PART", [bottom], MultiviewOptions())

        self.assertEqual(canonical["source_selection"]["package_pads"]["selected_raw_view"], "bottom")
        self.assertEqual(len(canonical["package_pads"]), 6)
        self.assertEqual(len(canonical["filtered_package_pads"]), 0)

    def test_package_pad_source_selection_keeps_sparse_bottom_when_no_complete_fallback_exists(self) -> None:
        sparse_bottom = toy_graph("PART", "bottom", pad_count=2, outline=False, spacing_value=1.0)
        sparse_top = toy_graph("PART", "top", pad_count=2, outline=True, spacing_value=1.0)

        canonical = integrate_part("PART", [sparse_bottom, sparse_top], MultiviewOptions())

        selection = canonical["source_selection"]["package_pads"]
        self.assertEqual(selection["selected_raw_view"], "bottom")
        self.assertFalse(selection["used_fallback"])
        self.assertFalse(selection["missing_primary"])
        self.assertEqual(selection["fallback_reason"], "")
        self.assertEqual(len(canonical["package_pads"]), 2)

    def test_same_view_selection_uses_most_complete_pad_graph(self) -> None:
        sparse_top = toy_graph("PART", "top", pad_count=2, outline=True, spacing_value=1.0)
        sparse_top["_graph_path"] = "/tmp/PART/top_a.package_graph.json"
        dense_top = toy_graph("PART", "top", pad_count=5, outline=True, spacing_value=1.0)
        dense_top["_graph_path"] = "/tmp/PART/top_b.package_graph.json"
        sparse_land = toy_graph("PART", "land", pad_count=3, outline=False, spacing_value=1.0)
        sparse_land["_graph_path"] = "/tmp/PART/land_a.package_graph.json"
        dense_land = toy_graph("PART", "land", pad_count=6, outline=False, spacing_value=1.0)
        dense_land["_graph_path"] = "/tmp/PART/land_b.package_graph.json"

        canonical = integrate_part("PART", [sparse_top, dense_top, sparse_land, dense_land], MultiviewOptions())

        self.assertEqual(len(canonical["package_pads"]), 5)
        self.assertEqual(len(canonical["land_pads"]), 6)
        self.assertTrue(all(pad["source_graph"] == "/tmp/PART/top_b.package_graph.json" for pad in canonical["package_pads"]))
        self.assertTrue(all(pad["source_graph"] == "/tmp/PART/land_b.package_graph.json" for pad in canonical["land_pads"]))

    def test_multiple_bottom_graphs_split_shape_package_and_rect_land_when_land_view_missing(self) -> None:
        shape_bottom = toy_graph("PART", "bottom", pad_count=4, outline=True, spacing_value=1.0)
        shape_bottom["_graph_path"] = "/tmp/PART/bottom_a_shape.package_graph.json"
        set_pad_labels(shape_bottom, label="pad_dshape", source_label="pad_dshape")
        rect_bottom = toy_graph("PART", "bottom", pad_count=4, outline=True, spacing_value=1.0)
        rect_bottom["_graph_path"] = "/tmp/PART/bottom_z_rect.package_graph.json"
        set_pad_labels(rect_bottom, label="rect", source_label="rect")

        canonical = integrate_part("PART", [shape_bottom, rect_bottom], MultiviewOptions())

        package_selection = canonical["source_selection"]["package_pads"]
        land_selection = canonical["source_selection"]["land_pads"]
        self.assertEqual(package_selection["graph_path"], "/tmp/PART/bottom_a_shape.package_graph.json")
        self.assertEqual(package_selection["fallback_reason"], "secondary_bottom_shape_layout_selected_for_package_pads")
        self.assertTrue(package_selection["used_fallback"])
        self.assertEqual(land_selection["graph_path"], "/tmp/PART/bottom_z_rect.package_graph.json")
        self.assertEqual(land_selection["fallback_reason"], "missing_land_view_used_secondary_bottom_rect_layout")
        self.assertTrue(land_selection["used_fallback"])
        self.assertTrue(land_selection["missing_primary"])
        self.assertEqual(len(canonical["package_pads"]), 4)
        self.assertEqual(len(canonical["land_pads"]), 4)
        self.assertTrue(
            all(pad["source_graph"] == "/tmp/PART/bottom_a_shape.package_graph.json" for pad in canonical["package_pads"])
        )
        self.assertTrue(
            all(pad["source_graph"] == "/tmp/PART/bottom_z_rect.package_graph.json" for pad in canonical["land_pads"])
        )

    def test_secondary_bottom_land_split_keeps_explicit_land_view_primary(self) -> None:
        shape_bottom = toy_graph("PART", "bottom", pad_count=4, outline=True, spacing_value=1.0)
        shape_bottom["_graph_path"] = "/tmp/PART/bottom_a_shape.package_graph.json"
        set_pad_labels(shape_bottom, label="pad_dshape", source_label="pad_dshape")
        rect_bottom = toy_graph("PART", "bottom", pad_count=4, outline=True, spacing_value=1.0)
        rect_bottom["_graph_path"] = "/tmp/PART/bottom_z_rect.package_graph.json"
        set_pad_labels(rect_bottom, label="rect", source_label="rect")
        land = toy_graph("PART", "land", pad_count=2, outline=False, spacing_value=1.0)
        land["_graph_path"] = "/tmp/PART/land.package_graph.json"

        canonical = integrate_part("PART", [shape_bottom, rect_bottom, land], MultiviewOptions())

        package_selection = canonical["source_selection"]["package_pads"]
        land_selection = canonical["source_selection"]["land_pads"]
        self.assertEqual(package_selection["graph_path"], "/tmp/PART/bottom_a_shape.package_graph.json")
        self.assertEqual(land_selection["graph_path"], "/tmp/PART/land.package_graph.json")
        self.assertEqual(land_selection["selected_raw_view"], "land")
        self.assertFalse(land_selection["used_fallback"])
        self.assertEqual(len(canonical["land_pads"]), 2)

    def test_duplicate_pad_bboxes_are_deduplicated(self) -> None:
        graph = toy_graph("PART", "bottom", pad_count=2, outline=True, spacing_value=1.0)
        duplicate = dict(graph["objects"][1])
        duplicate["id"] = 99
        graph["objects"].append(duplicate)

        canonical = integrate_part("PART", [graph], MultiviewOptions())

        self.assertEqual(len(canonical["package_pads"]), 2)
        self.assertEqual(canonical["summary"]["package_pad_count"], 2)

    def test_concentric_circle_pads_are_merged_as_one_canonical_pad(self) -> None:
        graph = toy_graph("PART", "bottom", pad_count=1, outline=True, spacing_value=None)
        graph["objects"] = graph["objects"][:1] + [
            {
                "id": 1,
                "label": "pad_circle",
                "source_label": "pad_circle",
                "bbox_reconstructed": [0.0, 0.0, 10.0, 10.0],
            },
            {
                "id": 2,
                "label": "pad_circle",
                "source_label": "pad_circle",
                "bbox_reconstructed": [2.0, 2.0, 8.0, 8.0],
            },
            {
                "id": 3,
                "label": "pad_circle",
                "source_label": "pad_circle",
                "bbox_reconstructed": [4.0, 4.0, 6.0, 6.0],
            },
            {
                "id": 4,
                "label": "pad_circle",
                "source_label": "pad_circle",
                "bbox_reconstructed": [20.0, 0.0, 30.0, 10.0],
            },
        ]

        canonical = integrate_part("PART", [graph], MultiviewOptions())

        self.assertEqual(len(canonical["package_pads"]), 2)
        merged = next(pad for pad in canonical["package_pads"] if pad.get("merged_bbox_count"))
        self.assertEqual(merged["bbox"], [0.0, 0.0, 10.0, 10.0])
        self.assertEqual(merged["merged_source_object_ids"], [1, 2, 3])
        self.assertEqual(len(merged["nested_bboxes"]), 3)

    def test_near_concentric_pad_geometries_are_merged_as_one_canonical_pad(self) -> None:
        graph = toy_graph("PART", "land", pad_count=0, outline=False, spacing_value=None)
        graph["objects"] = [
            {
                "id": 1,
                "label": "pad",
                "source_label": "pad",
                "bbox_reconstructed": [0.0, 0.0, 100.0, 100.0],
            },
            {
                "id": 2,
                "label": "pad_circle",
                "source_label": "pad_circle",
                "bbox_reconstructed": [40.0, 40.0, 62.0, 62.0],
            },
            {
                "id": 3,
                "label": "pad",
                "source_label": "pad",
                "bbox_reconstructed": [150.0, 0.0, 250.0, 100.0],
            },
        ]

        canonical = integrate_part("PART", [graph], MultiviewOptions())

        self.assertEqual(len(canonical["land_pads"]), 2)
        merged = next(pad for pad in canonical["land_pads"] if pad.get("merged_bbox_count"))
        self.assertEqual(merged["bbox"], [0.0, 0.0, 100.0, 100.0])
        self.assertEqual(merged["merged_source_object_ids"], [1, 2])

    def test_concentric_land_mask_pad_keeps_inner_rect_as_canonical_bbox(self) -> None:
        graph = toy_graph("PART", "land", pad_count=0, outline=False, spacing_value=None)
        graph["objects"] = [
            {
                "id": 1,
                "label": "pad",
                "source_label": "pad",
                "bbox_reconstructed": [0.0, 0.0, 120.0, 120.0],
            },
            {
                "id": 2,
                "label": "pad",
                "source_label": "pad",
                "bbox_reconstructed": [35.0, 35.0, 85.0, 85.0],
            },
            {
                "id": 3,
                "label": "pad_circle",
                "source_label": "pad_circle",
                "bbox_reconstructed": [55.0, 55.0, 65.0, 65.0],
            },
            {
                "id": 4,
                "label": "pad",
                "source_label": "pad",
                "bbox_reconstructed": [200.0, 0.0, 250.0, 50.0],
            },
        ]

        canonical = integrate_part("PART", [graph], MultiviewOptions())

        self.assertEqual(len(canonical["land_pads"]), 2)
        merged = next(pad for pad in canonical["land_pads"] if pad.get("merged_bbox_count"))
        self.assertEqual(merged["bbox"], [35.0, 35.0, 85.0, 85.0])
        self.assertEqual(merged["merged_source_object_ids"], [1, 2, 3])
        self.assertEqual(merged["outer_mask_source_object_id"], 1)
        self.assertEqual(merged["outer_mask_bbox"], [0.0, 0.0, 120.0, 120.0])

    def test_remote_circle_detail_inset_package_pads_are_filtered_by_outline(self) -> None:
        graph = toy_graph("PART", "bottom", pad_count=0, outline=True, spacing_value=None)
        objects = [graph["objects"][0]]
        object_id = 1
        for row in range(4):
            for col in range(5):
                x = 1.0 + col * 1.5
                y = 1.0 + row * 1.5
                objects.append(circle_object(object_id, [x, y, x + 0.5, y + 0.5]))
                object_id += 1
        for index in range(3):
            x = 20.0 + index * 1.0
            objects.append(circle_object(object_id, [x, 1.0, x + 0.5, 1.5]))
            object_id += 1
        graph["objects"] = objects

        canonical = integrate_part("PART", [graph], MultiviewOptions())

        self.assertEqual(len(canonical["package_pads"]), 20)
        self.assertEqual(len(canonical["filtered_package_pads"]), 3)
        self.assertEqual(canonical["summary"]["package_pad_count"], 20)
        self.assertEqual(canonical["summary"]["filtered_package_pad_count"], 3)
        self.assertEqual(
            {pad["filtered_reason"] for pad in canonical["filtered_package_pads"]},
            {"remote_detail_inset_outside_outline"},
        )

    def test_remote_rect_package_pads_are_not_filtered_by_outline(self) -> None:
        graph = toy_graph("PART", "bottom", pad_count=0, outline=True, spacing_value=None)
        objects = [graph["objects"][0]]
        object_id = 1
        for row in range(4):
            for col in range(5):
                x = 1.0 + col * 1.5
                y = 1.0 + row * 1.5
                objects.append(rect_pad_object(object_id, [x, y, x + 0.5, y + 0.5]))
                object_id += 1
        for index in range(3):
            x = 20.0 + index * 1.0
            objects.append(rect_pad_object(object_id, [x, 1.0, x + 0.5, 1.5]))
            object_id += 1
        graph["objects"] = objects

        canonical = integrate_part("PART", [graph], MultiviewOptions())

        self.assertEqual(len(canonical["package_pads"]), 23)
        self.assertEqual(len(canonical["filtered_package_pads"]), 0)

    def test_remote_oversized_package_pad_like_outliers_are_filtered_from_dense_grid(self) -> None:
        graph = toy_graph("PART", "bottom", pad_count=0, outline=True, spacing_value=None)
        objects = [graph["objects"][0]]
        object_id = 1
        for row in range(4):
            for col in range(5):
                x = 1.0 + col * 2.0
                y = 1.0 + row * 2.0
                objects.append(rect_pad_object(object_id, [x, y, x + 0.5, y + 1.0]))
                object_id += 1
        objects.append(rect_pad_object(object_id, [14.0, 14.0, 19.0, 19.0]))
        graph["objects"] = objects

        canonical = integrate_part("PART", [graph], MultiviewOptions())

        self.assertEqual(len(canonical["package_pads"]), 20)
        self.assertEqual(len(canonical["filtered_package_pads"]), 1)
        self.assertEqual(canonical["filtered_package_pads"][0]["filtered_reason"], "oversized_pad_like_outlier")

    def test_center_oversized_package_pad_is_kept_in_dense_grid(self) -> None:
        graph = toy_graph("PART", "bottom", pad_count=0, outline=True, spacing_value=None)
        objects = [graph["objects"][0]]
        object_id = 1
        for row in range(4):
            for col in range(5):
                x = 1.0 + col * 2.0
                y = 1.0 + row * 2.0
                objects.append(rect_pad_object(object_id, [x, y, x + 0.5, y + 1.0]))
                object_id += 1
        objects.append(rect_pad_object(object_id, [2.0, 2.0, 7.0, 7.0]))
        graph["objects"] = objects

        canonical = integrate_part("PART", [graph], MultiviewOptions())

        self.assertEqual(len(canonical["package_pads"]), 21)
        self.assertEqual(len(canonical["filtered_package_pads"]), 0)

    def test_two_column_package_pads_use_dimension_width_and_outline_margin(self) -> None:
        graph = toy_graph("PART", "bottom", pad_count=0, outline=True, spacing_value=None)
        graph["objects"] = [
            {"id": 100, "label": "outline", "source_label": "outline", "bbox_reconstructed": [0.0, 0.0, 100.0, 100.0]},
            rect_pad_object(1, [0.0, 10.0, 20.0, 25.0]),
            rect_pad_object(2, [80.0, 10.0, 100.0, 25.0]),
            rect_pad_object(3, [5.0, 40.0, 35.0, 55.0]),
            rect_pad_object(4, [65.0, 40.0, 95.0, 55.0]),
            rect_pad_object(5, [0.0, 75.0, 20.0, 90.0]),
            rect_pad_object(6, [80.0, 75.0, 100.0, 90.0]),
        ]
        graph["dimensions"] = [
            {
                "id": 1,
                "dimension_id": 1,
                "text": "1.0",
                "kind": "size",
                "axis": "x",
                "target_ids": [5],
                "anchors": ["left_edge", "right_edge"],
                "value": 1.0,
                "status": "accepted",
                "value_source": "text_parser",
            },
            {
                "id": 2,
                "dimension_id": 2,
                "text": "1.0",
                "kind": "size",
                "axis": "x",
                "target_ids": [6],
                "anchors": ["left_edge", "right_edge"],
                "value": 1.0,
                "status": "accepted",
                "value_source": "text_parser",
            },
            {
                "id": 3,
                "dimension_id": 3,
                "text": "0.5",
                "kind": "distance",
                "axis": "x",
                "target_ids": [2, 100],
                "anchors": ["right_edge", "right_edge"],
                "value": 0.5,
                "status": "accepted",
                "value_source": "text_parser",
            },
        ]

        canonical = integrate_part("PART", [graph], MultiviewOptions())
        pads = {pad["source_object_id"]: pad for pad in canonical["package_pads"]}

        for pad_id in (1, 3, 5):
            self.assertEqual(pads[pad_id]["bbox"][0], 10.0)
            self.assertEqual(pads[pad_id]["bbox"][2], 30.0)
        for pad_id in (2, 4, 6):
            self.assertEqual(pads[pad_id]["bbox"][0], 70.0)
            self.assertEqual(pads[pad_id]["bbox"][2], 90.0)
        self.assertEqual(pads[1]["bbox"][1:], [10.0, 30.0, 25.0])
        self.assertTrue(
            all(
                pad.get("geometry_adjusted_reason") == "dimension_regularized_package_pad_x_grid"
                for pad in canonical["package_pads"]
            )
        )

    def test_remote_land_detail_tail_pads_are_filtered_from_dense_land_layout(self) -> None:
        graph = toy_graph("PART", "land", pad_count=0, outline=False, spacing_value=None)
        objects = []
        object_id = 1
        for row, y in enumerate((0.0, 30.0)):
            for col in range(10):
                x = col * 10.0
                objects.append(rect_pad_object(object_id, [x, y, x + 2.0, y + 8.0]))
                object_id += 1
        objects.append(rect_pad_object(object_id, [220.0, 0.0, 222.0, 8.0]))
        object_id += 1
        objects.append(rect_pad_object(object_id, [220.0, 30.0, 222.0, 38.0]))
        graph["objects"] = objects

        canonical = integrate_part("PART", [graph], MultiviewOptions())

        self.assertEqual(len(canonical["land_pads"]), 20)
        self.assertEqual(len(canonical["filtered_land_pads"]), 2)
        self.assertEqual(canonical["summary"]["land_pad_count"], 20)
        self.assertEqual(canonical["summary"]["filtered_land_pad_count"], 2)
        self.assertEqual(
            {pad["filtered_reason"] for pad in canonical["filtered_land_pads"]},
            {"remote_detail_land_pad_tail_gap"},
        )

    def test_scan_result_format_is_not_used_as_multiview_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = Path(tmp)
            part_dir = dataset_root / "PART"
            part_dir.mkdir()
            scan_path = part_dir / "ScanResultFormat.txt"
            scan_path.write_text('{"Object": []}\n', encoding="utf-8")

            canonical = integrate_part(
                "PART",
                [toy_graph("PART", "bottom", pad_count=2, outline=True, spacing_value=1.0)],
                MultiviewOptions(),
                dataset_root=dataset_root,
            )

            scan_refs = [ref for ref in canonical["evidence_refs"] if ref.get("evidence_type") == "scan_result_format"]
            self.assertEqual(scan_refs, [])
            self.assertEqual(canonical["summary"]["scan_result_path"], "")
            self.assertFalse(canonical["summary"]["has_scan_result"])

    def test_scan_result_terminal_geometry_does_not_replace_package_graph(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = Path(tmp)
            part_dir = dataset_root / "PART"
            part_dir.mkdir()
            scan_path = part_dir / "ScanResultFormat.txt"
            scan_path.write_text(json.dumps(scan_result_payload(lead_count=10)), encoding="utf-8")

            canonical = integrate_part(
                "PART",
                [toy_graph("PART", "bottom", pad_count=4, outline=True, spacing_value=1.0)],
                MultiviewOptions(),
                dataset_root=dataset_root,
            )

            selection = canonical["source_selection"]["package_pads"]
            self.assertEqual(len(canonical["package_pads"]), 4)
            self.assertEqual(canonical["summary"]["package_pad_count"], 4)
            self.assertNotEqual(selection.get("source_type"), "scan_result_format")
            self.assertEqual(selection["fallback_reason"], "")
            self.assertTrue(all(pad.get("source_type") != "scan_result_format" for pad in canonical["package_pads"]))
            self.assertNotEqual(canonical["outline_2d"].get("source_type"), "scan_result_format")

    def test_scan_result_land_geometry_does_not_fill_missing_land(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = Path(tmp)
            part_dir = dataset_root / "PART"
            part_dir.mkdir()
            scan_path = part_dir / "ScanResultFormat.txt"
            scan_path.write_text(json.dumps(scan_result_payload(lead_count=2, land_count=4)), encoding="utf-8")

            canonical = integrate_part(
                "PART",
                [toy_graph("PART", "bottom", pad_count=2, outline=True, spacing_value=1.0)],
                MultiviewOptions(),
                dataset_root=dataset_root,
            )

            selection = canonical["source_selection"]["land_pads"]
            self.assertEqual(canonical["land_pads"], [])
            self.assertNotEqual(selection.get("source_type"), "scan_result_format")
            self.assertEqual(selection["fallback_reason"], "")

    def test_package_graph_evidence_records_pad_like_counts(self) -> None:
        canonical = integrate_part(
            "PART",
            [
                toy_graph("PART", "land", pad_count=3, outline=False, spacing_value=1.0),
                toy_graph("PART", "land_detail", pad_count=2, outline=False, spacing_value=None),
            ],
            MultiviewOptions(),
        )

        refs = {
            ref["raw_view"]: ref
            for ref in canonical["evidence_refs"]
            if ref.get("evidence_type") == "package_graph"
        }
        self.assertEqual(refs["land"]["pad_like_count"], 3)
        self.assertEqual(refs["land_detail"]["pad_like_count"], 2)
        self.assertEqual(refs["land_detail"]["object_label_counts"], {"pad": 2})

    def test_package_graph_evidence_records_terminal_pad_like_counts(self) -> None:
        graph = toy_graph("PART", "land", pad_count=4, outline=False, spacing_value=None)
        graph["objects"].append(
            {
                "id": 99,
                "label": "rect",
                "source_label": "pad",
                "bbox_reconstructed": [1.0, -1.0, 7.0, 5.0],
            }
        )

        canonical = integrate_part("PART", [graph], MultiviewOptions())
        land_ref = next(ref for ref in canonical["evidence_refs"] if ref.get("raw_view") == "land")

        self.assertEqual(land_ref["pad_like_count"], 5)
        self.assertEqual(land_ref["thermal_pad_like_count"], 1)
        self.assertEqual(land_ref["terminal_pad_like_count"], 4)

    def test_package_graph_evidence_detects_moderately_larger_central_thermal_pad(self) -> None:
        graph = toy_graph("PART", "land", pad_count=0, outline=False, spacing_value=None)
        graph["objects"] = [
            {"id": 1, "label": "rect", "source_label": "pad", "bbox_reconstructed": [0.0, 0.0, 1.0, 10.0]},
            {"id": 2, "label": "rect", "source_label": "pad", "bbox_reconstructed": [3.0, 0.0, 4.0, 10.0]},
            {"id": 3, "label": "rect", "source_label": "pad", "bbox_reconstructed": [0.0, 20.0, 1.0, 30.0]},
            {"id": 4, "label": "rect", "source_label": "pad", "bbox_reconstructed": [3.0, 20.0, 4.0, 30.0]},
            {"id": 5, "label": "rect", "source_label": "pad", "bbox_reconstructed": [1.75, 6.0, 2.75, 24.5]},
        ]

        canonical = integrate_part("PART", [graph], MultiviewOptions())
        land_ref = next(ref for ref in canonical["evidence_refs"] if ref.get("raw_view") == "land")

        self.assertEqual(land_ref["pad_like_count"], 5)
        self.assertEqual(land_ref["thermal_pad_like_count"], 1)
        self.assertEqual(land_ref["terminal_pad_like_count"], 4)

    def test_package_graph_evidence_detects_compact_larger_central_thermal_pad(self) -> None:
        graph = toy_graph("PART", "land", pad_count=0, outline=False, spacing_value=None)
        graph["objects"] = [
            {"id": 1, "label": "rect", "source_label": "pad", "bbox_reconstructed": [0.0, 0.0, 2.0, 4.0]},
            {"id": 2, "label": "rect", "source_label": "pad", "bbox_reconstructed": [8.0, 0.0, 10.0, 4.0]},
            {"id": 3, "label": "rect", "source_label": "pad", "bbox_reconstructed": [0.0, 16.0, 2.0, 20.0]},
            {"id": 4, "label": "rect", "source_label": "pad", "bbox_reconstructed": [8.0, 16.0, 10.0, 20.0]},
            {"id": 5, "label": "rect", "source_label": "pad", "bbox_reconstructed": [0.0, 8.0, 4.0, 10.0]},
            {"id": 6, "label": "rect", "source_label": "pad", "bbox_reconstructed": [16.0, 8.0, 20.0, 10.0]},
            {"id": 7, "label": "rect", "source_label": "pad", "bbox_reconstructed": [0.0, 11.0, 4.0, 13.0]},
            {"id": 8, "label": "rect", "source_label": "pad", "bbox_reconstructed": [16.0, 11.0, 20.0, 13.0]},
            {"id": 9, "label": "rect", "source_label": "pad", "bbox_reconstructed": [8.0, 8.0, 12.0, 12.0]},
        ]

        canonical = integrate_part("PART", [graph], MultiviewOptions())
        land_ref = next(ref for ref in canonical["evidence_refs"] if ref.get("raw_view") == "land")

        self.assertEqual(land_ref["pad_like_count"], 9)
        self.assertEqual(land_ref["thermal_pad_like_count"], 1)
        self.assertEqual(land_ref["terminal_pad_like_count"], 8)

    def test_package_graph_evidence_detects_side_internal_bar_when_central_pad_exists(self) -> None:
        graph = toy_graph("PART", "land", pad_count=0, outline=False, spacing_value=None)
        graph["objects"] = [
            {"id": 1, "label": "rect", "source_label": "pad", "bbox_reconstructed": [0.0, 0.0, 1.0, 10.0]},
            {"id": 2, "label": "rect", "source_label": "pad", "bbox_reconstructed": [3.0, 0.0, 4.0, 10.0]},
            {"id": 3, "label": "rect", "source_label": "pad", "bbox_reconstructed": [6.0, 0.0, 7.0, 10.0]},
            {"id": 4, "label": "rect", "source_label": "pad", "bbox_reconstructed": [0.0, 30.0, 1.0, 40.0]},
            {"id": 5, "label": "rect", "source_label": "pad", "bbox_reconstructed": [3.0, 30.0, 4.0, 40.0]},
            {"id": 6, "label": "rect", "source_label": "pad", "bbox_reconstructed": [6.0, 30.0, 7.0, 40.0]},
            {"id": 7, "label": "rect", "source_label": "pad", "bbox_reconstructed": [2.0, 12.0, 6.0, 28.0]},
            {"id": 8, "label": "rect", "source_label": "pad", "bbox_reconstructed": [0.0, 10.0, 1.5, 30.0]},
        ]

        canonical = integrate_part("PART", [graph], MultiviewOptions())
        land_ref = next(ref for ref in canonical["evidence_refs"] if ref.get("raw_view") == "land")

        self.assertEqual(land_ref["pad_like_count"], 8)
        self.assertEqual(land_ref["thermal_pad_like_count"], 2)
        self.assertEqual(land_ref["terminal_pad_like_count"], 6)

    def test_table_lookup_dimensions_preserve_table_file_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = Path(tmp)
            part_dir = dataset_root / "PART"
            table_dir = part_dir / "table"
            table_dir.mkdir(parents=True)
            table_file = table_dir / "table1.xlsx"
            table_file.write_text("", encoding="utf-8")
            graph = toy_graph("PART", "bottom", pad_count=2, outline=True, spacing_value=1.0)
            graph["dimensions"][0]["value_source"] = "table_lookup"

            canonical = integrate_part("PART", [graph], MultiviewOptions(), dataset_root=dataset_root)

            table_refs = [ref for ref in canonical["evidence_refs"] if ref.get("evidence_type") == "table_lookup_files"]
            self.assertEqual(len(table_refs), 1)
            self.assertEqual(table_refs[0]["status"], "available")
            self.assertEqual(table_refs[0]["files"], [str(table_file)])
            self.assertEqual(canonical["summary"]["table_evidence_count"], 1)
            self.assertEqual(canonical["evidence_summary"]["table_lookup_dimension_count"], 1)
            self.assertEqual(canonical["evidence_summary"]["evidence_type_counts"]["table_lookup_files"], 1)

    def test_integrate_graphs_writes_explicit_failure_for_dataset_part_without_graphs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "dataset"
            for part_number in ("PART_A", "PART_B"):
                part_dir = dataset_root / part_number
                part_dir.mkdir(parents=True)
                (part_dir / "ScanResultFormat.txt").write_text('{"Object": []}\n', encoding="utf-8")
            graph_dir = root / "graphs" / "PART_A"
            graph_dir.mkdir(parents=True)
            (graph_dir / "bottom.package_graph.json").write_text(
                json.dumps(toy_graph("PART_A", "bottom", pad_count=2, outline=True, spacing_value=1.0)),
                encoding="utf-8",
            )

            summary = integrate_graphs(
                root / "graphs",
                root / "out",
                dataset_root=dataset_root,
                options=MultiviewOptions(),
            )

            self.assertEqual(summary["total_parts"], 2)
            self.assertEqual(summary["part_outputs"], 2)
            self.assertEqual(summary["canonical_parts"], 1)
            self.assertEqual(summary["graph_based_parts"], 1)
            self.assertEqual(summary["failure_reason_parts"], 1)
            self.assertEqual(summary["missing_graph_parts"], 1)
            self.assertEqual(summary["status_counts"], {"canonical": 1, "missing_graphs": 1})
            self.assertEqual(summary["dimension_value_source_counts"], {"text_parser": 1})
            self.assertEqual(summary["evidence_type_counts"], {"package_graph": 1})
            canonical = json.loads((root / "out" / "parts" / "PART_A" / "unified_multiview_layers.json").read_text(encoding="utf-8"))
            self.assertEqual(canonical["status"], "canonical")
            evidence = json.loads((root / "out" / "parts" / "PART_A" / "evidence.json").read_text(encoding="utf-8"))
            self.assertEqual(evidence["summary"], canonical["evidence_summary"])
            missing = json.loads((root / "out" / "parts" / "PART_B" / "unified_multiview_layers.json").read_text(encoding="utf-8"))
            self.assertEqual(missing["status"], "missing_graphs")
            self.assertEqual(missing["failure_reason"], "no_package_graph_for_part")
            self.assertEqual(missing["summary"]["failure_reason"], "no_package_graph_for_part")
            self.assertEqual(missing["summary"]["risk_level"], "high")

    def test_integrate_graphs_does_not_use_scan_result_geometry_when_graphs_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "dataset"
            part_dir = dataset_root / "PART"
            part_dir.mkdir(parents=True)
            (part_dir / "ScanResultFormat.txt").write_text(
                json.dumps(scan_result_payload(lead_count=4, land_count=4)),
                encoding="utf-8",
            )
            graph_root = root / "graphs"
            graph_root.mkdir()

            summary = integrate_graphs(graph_root, root / "out", dataset_root=dataset_root, options=MultiviewOptions())

            self.assertEqual(summary["total_parts"], 1)
            self.assertEqual(summary["canonical_parts"], 0)
            self.assertEqual(summary["graph_based_parts"], 0)
            self.assertEqual(summary["scan_result_only_parts"], 0)
            self.assertEqual(summary["missing_graph_parts"], 1)
            self.assertEqual(summary["status_counts"], {"missing_graphs": 1})
            canonical = json.loads((root / "out" / "parts" / "PART" / "unified_multiview_layers.json").read_text(encoding="utf-8"))
            self.assertEqual(canonical["status"], "missing_graphs")
            self.assertEqual(len(canonical["package_pads"]), 0)
            self.assertEqual(len(canonical["land_pads"]), 0)
            self.assertEqual(canonical["summary"]["package_pad_count"], 0)
            self.assertEqual(canonical["summary"]["land_pad_count"], 0)

    def test_package_land_count_mismatch_is_not_forced(self) -> None:
        package_pads = [
            {"source_object_id": 1, "bbox": [0, 0, 1, 1]},
            {"source_object_id": 2, "bbox": [2, 0, 3, 1]},
        ]
        land_pads = [
            {"source_object_id": 11, "bbox": [0, 0, 1, 1]},
            {"source_object_id": 12, "bbox": [2, 0, 3, 1]},
            {"source_object_id": 13, "bbox": [4, 0, 5, 1]},
        ]

        result = match_package_and_land_pads(package_pads, land_pads)

        self.assertEqual(result["status"], "not_applicable")
        self.assertEqual(result["reason"], "package_land_count_differs")
        self.assertEqual(result["package_pad_count"], 2)
        self.assertEqual(result["land_pad_count"], 3)

    def test_applied_scan_result_events_do_not_count_as_active_conflicts(self) -> None:
        conflicts = [
            {"type": "scan_result_reference_mismatch", "status": "applied"},
            {"type": "dimension_value_conflict"},
        ]

        self.assertEqual(score_part(conflicts, []), 10.0)
        self.assertEqual(risk_reasons_for_part(conflicts, []), ["1 conflicts"])

    def test_ambiguous_pad_matching_is_not_forced(self) -> None:
        package_pads = [{"source_object_id": None, "bbox": [0, 0, 1, 1]}, {"source_object_id": None, "bbox": [2, 0, 3, 1]}]
        land_pads = [{"source_object_id": None, "bbox": [0, 0, 1, 1]}, {"source_object_id": None, "bbox": [2, 0, 3, 1]}]

        result = match_package_and_land_pads(package_pads, land_pads)

        self.assertEqual(result["status"], "ambiguous_match")


def toy_graph(
    part_number: str,
    view: str,
    *,
    pad_count: int,
    outline: bool,
    spacing_value: float | None,
    y_height_value: float | None = None,
) -> dict:
    objects = []
    if outline:
        objects.append(
            {
                "id": 100,
                "label": "outline",
                "source_label": "outline",
                "bbox_reconstructed": [0, 0, 10, 6],
            }
        )
    for index in range(pad_count):
        objects.append(
            {
                "id": index,
                "label": "rect",
                "source_label": "pad",
                "bbox_reconstructed": [index * 2.0, 1.0, index * 2.0 + 1.0, 2.0],
            }
        )

    dimensions = []
    if spacing_value is not None and pad_count >= 2:
        dimensions.append(
            {
                "id": 1,
                "dimension_id": 1,
                "text": str(spacing_value),
                "kind": "distance",
                "axis": "x",
                "target_ids": [0, 1],
                "anchors": ["center", "center"],
                "value": spacing_value,
                "status": "accepted",
                "value_source": "text_parser",
            }
        )
    if y_height_value is not None:
        dimensions.append(
            {
                "id": 2,
                "dimension_id": 2,
                "text": str(y_height_value),
                "kind": "size",
                "axis": "y",
                "target_ids": [100],
                "anchors": ["top_edge", "bottom_edge"],
                "value": y_height_value,
                "status": "accepted",
                "value_source": "text_parser",
            }
        )
    return {
        "part_number": part_number,
        "_part_number": part_number,
        "view": view,
        "_raw_view": view,
        "_graph_path": f"/tmp/{part_number}/{view}.package_graph.json",
        "annotation_path": f"/tmp/{part_number}/{view}.json",
        "image": {"path": f"/tmp/{part_number}/{view}.png"},
        "objects": objects,
        "dimensions": dimensions,
    }


def circle_object(object_id: int, bbox: list[float]) -> dict:
    return {
        "id": object_id,
        "label": "pad_circle",
        "source_label": "pad_circle",
        "bbox_reconstructed": bbox,
    }


def rect_pad_object(object_id: int, bbox: list[float]) -> dict:
    return {
        "id": object_id,
        "label": "rect",
        "source_label": "pad",
        "bbox_reconstructed": bbox,
    }


def set_pad_labels(graph: dict, *, label: str, source_label: str) -> None:
    for obj in graph["objects"]:
        if obj.get("source_label") == "outline":
            continue
        obj["label"] = label
        obj["source_label"] = source_label


def scan_result_payload(*, lead_count: int, land_count: int = 0) -> dict:
    objects = [
        scan_result_object(
            100,
            "Rectangle",
            [0.0, 0.0, 12.0, 6.0],
            role_payload={},
        )
    ]
    for index in range(lead_count):
        x = float(index)
        objects.append(
            scan_result_object(
                index,
                "DShape",
                [x, 1.0, x + 0.5, 1.5],
                role_payload={"LeadData": {}},
            )
        )
    for index in range(land_count):
        x = float(index)
        objects.append(
            scan_result_object(
                1000 + index,
                "Rectangle",
                [x, 3.0, x + 0.5, 3.5],
                role_payload={"LandData": {}},
            )
        )
    return {"Object": objects}


def scan_result_object(object_id: int, node_name: str, bbox: list[float], *, role_payload: dict) -> dict:
    x1, y1, x2, y2 = bbox
    payload = {
        "ID": object_id,
        "NodeName": node_name,
        "Geometry": node_name,
        "PointList": [
            {"PointX": x1, "PointY": y1},
            {"PointX": x2, "PointY": y1},
            {"PointX": x2, "PointY": y2},
            {"PointX": x1, "PointY": y2},
        ],
    }
    payload.update(role_payload)
    return payload


if __name__ == "__main__":
    unittest.main()
