from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tmp.multiview_alignment import (
    apply_pad_center_alignment,
    apply_outline_alignment,
    apply_top_bottom_package_pad_alignment,
    build_multiview_alignment,
    build_land_package_pair_diagnostics,
    build_outline_alignment,
    build_pad_center_alignment,
    build_top_bottom_package_pad_alignment,
    filter_alignment_objects,
    visible_pad_stacks,
)


class MultiviewAlignmentTests(unittest.TestCase):
    def test_dimension_backed_outline_is_main_outline(self) -> None:
        objects = [
            outline(0, "bottom", "bottom_graph", [0.0, 0.0, 10.0, 10.0]),
            outline(1, "top", "top_graph", [-1.0, -1.0, 11.0, 11.0]),
        ]
        dimensions = [
            {
                "dimension_id": 100,
                "role": "outline_size",
                "kind": "size",
                "axis": "x",
                "target_ids": [1],
                "target_labels": ["outline"],
                "source_graph": "top_graph",
            }
        ]

        alignment = build_outline_alignment(objects, dimensions)

        self.assertEqual(alignment["main_outline"]["source_object_id"], 1)
        self.assertEqual(alignment["main_outline_selection_reason"], "dimension_supported_outline_size_preferred")
        self.assertEqual(alignment["outline_alignment"][0]["bbox_after_outline_adjust"], [-1.0, -1.0, 11.0, 11.0])

    def test_bottom_outline_is_main_when_no_outline_dimensions_exist(self) -> None:
        objects = [
            outline(0, "top", "top_graph", [-1.0, -1.0, 11.0, 11.0]),
            outline(1, "bottom", "bottom_graph", [0.0, 0.0, 10.0, 10.0]),
        ]

        alignment = build_outline_alignment(objects, [])

        self.assertEqual(alignment["main_outline"]["source_object_id"], 1)
        self.assertEqual(alignment["main_outline_selection_reason"], "bottom_outline_preferred_without_dimensions")

    def test_edge_locked_pads_follow_adjusted_outline_and_interior_pads_stay_fixed(self) -> None:
        objects = [
            outline(10, "bottom", "bottom_graph", [0.0, 0.0, 10.0, 10.0]),
            package_pad(11, "bottom", "bottom_graph", [0.0, 3.0, 1.0, 5.0]),
            package_pad(12, "bottom", "bottom_graph", [4.0, 4.0, 6.0, 6.0]),
            land_pad(13, "bottom", "bottom_graph", [0.0, 1.0, 2.0, 3.0]),
            lead_pad(14, "bottom", "bottom_graph", [8.0, 9.5, 9.0, 10.0]),
            outline(20, "top", "top_graph", [-2.0, -2.0, 12.0, 12.0]),
        ]
        dimensions = [
            {
                "dimension_id": 200,
                "role": "outline_size",
                "target_ids": [20],
                "target_labels": ["outline"],
                "source_graph": "top_graph",
            }
        ]

        alignment = build_outline_alignment(objects, dimensions)
        aligned = apply_outline_alignment(objects, alignment)

        self.assertEqual(aligned[0]["bbox"], [-2.0, -2.0, 12.0, 12.0])
        self.assertEqual(aligned[1]["bbox"], [-2.0, 3.0, -1.0, 5.0])
        self.assertEqual(aligned[1]["outline_edge_lock"], ["left"])
        self.assertEqual(aligned[2]["bbox"], [4.0, 4.0, 6.0, 6.0])
        self.assertNotIn("outline_edge_lock", aligned[2])
        self.assertEqual(aligned[3]["bbox"], [0.0, 1.0, 2.0, 3.0])
        self.assertNotIn("outline_edge_lock", aligned[3])
        self.assertEqual(aligned[4]["bbox"], [8.0, 11.5, 9.0, 12.0])
        self.assertEqual(aligned[4]["outline_edge_lock"], ["bottom"])
        self.assertEqual(alignment["outline_alignment"][0]["edge_adjusted_pad_count"], 2)
        self.assertEqual(alignment["outline_alignment"][0]["unchanged_pad_count"], 1)

    def test_land_outline_is_excluded_from_alignment_stage_but_land_pad_remains(self) -> None:
        objects = [
            outline(1, "bottom", "bottom_graph", [0.0, 0.0, 10.0, 10.0]),
            outline(2, "land", "land_graph", [-5.0, -5.0, 15.0, 15.0]),
            land_pad(3, "land", "land_graph", [1.0, 1.0, 2.0, 2.0]),
            outline(4, "top", "top_graph", [0.0, 0.0, 9.0, 9.0]),
        ]
        dimensions = [
            {
                "dimension_id": 300,
                "role": "outline_size",
                "target_ids": [2],
                "target_labels": ["outline"],
                "source_graph": "land_graph",
            }
        ]

        alignment_objects, excluded_objects = filter_alignment_objects(objects)
        alignment = build_outline_alignment(alignment_objects, dimensions)

        self.assertEqual(len(excluded_objects), 1)
        self.assertEqual(excluded_objects[0]["excluded_reason"], "land_outline_excluded_from_alignment_stage")
        self.assertEqual([obj["source_object_id"] for obj in alignment_objects], [1, 3, 4])
        self.assertEqual(alignment["outline_count"], 2)
        self.assertEqual(alignment["main_outline"]["source_object_id"], 1)
        self.assertEqual(alignment["main_outline_selection_reason"], "bottom_outline_preferred_without_dimensions")

    def test_land_pad_centers_align_by_side_axis_only(self) -> None:
        objects = [
            package_pad(1, "bottom", "package_graph", [-2.1, -3.0, -1.9, -2.5]),
            package_pad(2, "bottom", "package_graph", [-0.1, -3.0, 0.1, -2.5]),
            package_pad(3, "bottom", "package_graph", [1.9, -3.0, 2.1, -2.5]),
            package_pad(4, "bottom", "package_graph", [-2.1, 2.5, -1.9, 3.0]),
            package_pad(5, "bottom", "package_graph", [-0.1, 2.5, 0.1, 3.0]),
            package_pad(6, "bottom", "package_graph", [1.9, 2.5, 2.1, 3.0]),
            package_pad(7, "bottom", "package_graph", [-3.0, -2.1, -2.5, -1.9]),
            package_pad(8, "bottom", "package_graph", [-3.0, -0.1, -2.5, 0.1]),
            package_pad(9, "bottom", "package_graph", [-3.0, 1.9, -2.5, 2.1]),
            package_pad(10, "bottom", "package_graph", [2.5, -2.1, 3.0, -1.9]),
            package_pad(11, "bottom", "package_graph", [2.5, -0.1, 3.0, 0.1]),
            package_pad(12, "bottom", "package_graph", [2.5, 1.9, 3.0, 2.1]),
            land_pad(101, "land", "land_graph", [-2.3, -3.2, -2.1, -2.6]),
            land_pad(102, "land", "land_graph", [0.2, -3.2, 0.4, -2.6]),
            land_pad(103, "land", "land_graph", [1.6, -3.2, 1.8, -2.6]),
            land_pad(104, "land", "land_graph", [-2.3, 2.6, -2.1, 3.2]),
            land_pad(105, "land", "land_graph", [0.2, 2.6, 0.4, 3.2]),
            land_pad(106, "land", "land_graph", [1.6, 2.6, 1.8, 3.2]),
            land_pad(107, "land", "land_graph", [-3.2, -2.3, -2.6, -2.1]),
            land_pad(108, "land", "land_graph", [-3.2, 0.2, -2.6, 0.4]),
            land_pad(109, "land", "land_graph", [-3.2, 1.6, -2.6, 1.8]),
            land_pad(110, "land", "land_graph", [2.6, -2.3, 3.2, -2.1]),
            land_pad(111, "land", "land_graph", [2.6, 0.2, 3.2, 0.4]),
            land_pad(112, "land", "land_graph", [2.6, 1.6, 3.2, 1.8]),
            land_pad(113, "land", "land_graph", [-1.0, -1.0, 1.0, 1.0]),
        ]

        alignment = build_pad_center_alignment(objects)
        aligned = apply_pad_center_alignment(objects, alignment)

        self.assertEqual(alignment["matched_side_count"], 4)
        self.assertEqual(alignment["anchor_role"], "land_pad")
        self.assertEqual(aligned[0]["bbox"], [-2.3, -3.0, -2.1, -2.5])
        self.assertEqual(aligned[1]["bbox"], [0.2, -3.0, 0.4, -2.5])
        self.assertEqual(aligned[2]["bbox"], [1.6, -3.0, 1.8, -2.5])
        self.assertEqual(aligned[6]["bbox"], [-3.0, -2.3, -2.5, -2.1])
        self.assertEqual(aligned[7]["bbox"], [-3.0, 0.2, -2.5, 0.4])
        self.assertEqual(aligned[8]["bbox"], [-3.0, 1.6, -2.5, 1.8])
        self.assertEqual(aligned[12]["bbox"], [-2.3, -3.2, -2.1, -2.6])
        self.assertEqual(aligned[18]["bbox"], [-3.2, -2.3, -2.6, -2.1])
        self.assertEqual(aligned[24]["bbox"], [-1.0, -1.0, 1.0, 1.0])
        self.assertEqual(aligned[0]["pad_center_alignment_axis"], "x")
        self.assertEqual(aligned[6]["pad_center_alignment_axis"], "y")
        self.assertNotIn("pad_center_alignment_axis", aligned[12])

    def test_land_pad_center_alignment_skips_count_mismatch(self) -> None:
        objects = [
            package_pad(1, "bottom", "package_graph", [-2.1, -3.0, -1.9, -2.5]),
            package_pad(2, "bottom", "package_graph", [-0.1, -3.0, 0.1, -2.5]),
            package_pad(3, "bottom", "package_graph", [1.9, -3.0, 2.1, -2.5]),
            land_pad(101, "land", "land_graph", [-2.3, -3.2, -2.1, -2.6]),
            land_pad(102, "land", "land_graph", [0.2, -3.2, 0.4, -2.6]),
        ]

        alignment = build_pad_center_alignment(objects)
        aligned = apply_pad_center_alignment(objects, alignment)

        self.assertEqual(alignment["status"], "skipped")
        self.assertEqual(alignment["side_alignments"][0]["skip_reason"], "side_pad_count_mismatch")
        self.assertEqual(aligned[3]["bbox"], [-2.3, -3.2, -2.1, -2.6])
        self.assertNotIn("pad_center_alignment_axis", aligned[3])

    def test_pad_center_alignment_skips_single_side_left_right_only_ambiguity(self) -> None:
        objects = [
            package_pad(1, "top", "package_graph", [-2.0, -1.0, -1.8, 1.0]),
            package_pad(2, "top", "package_graph", [1.8, -1.0, 2.0, 1.0]),
            land_pad(101, "land", "land_graph", [-2.2, -2.0, -1.2, 2.0]),
            land_pad(102, "land", "land_graph", [1.2, -2.0, 2.2, 2.0]),
            lead_pad_with_source_package(201, 1, "top", "package_graph", [-2.0, -1.0, -1.5, 1.0]),
        ]

        alignment = build_pad_center_alignment(objects)
        aligned = apply_pad_center_alignment(objects, alignment)

        self.assertEqual(alignment["status"], "skipped")
        self.assertEqual(alignment["skip_reason"], "insufficient_matched_side_count")
        self.assertEqual(aligned[0]["bbox"], [-2.0, -1.0, -1.8, 1.0])
        self.assertEqual(aligned[1]["bbox"], [1.8, -1.0, 2.0, 1.0])
        self.assertEqual(aligned[4]["bbox"], [-2.0, -1.0, -1.5, 1.0])
        self.assertNotIn("pad_center_alignment_type", aligned[0])
        self.assertNotIn("pad_center_alignment_type", aligned[4])

    def test_package_derived_pad_follows_matched_package_pad_axis_shift(self) -> None:
        objects = [
            package_pad(1, "bottom", "package_graph", [-2.1, -3.0, -1.9, -2.5]),
            package_pad(2, "bottom", "package_graph", [-0.1, -3.0, 0.1, -2.5]),
            package_pad(3, "bottom", "package_graph", [-2.1, 2.5, -1.9, 3.0]),
            package_pad(4, "bottom", "package_graph", [-0.1, 2.5, 0.1, 3.0]),
            land_pad(101, "land", "land_graph", [-2.3, -3.2, -2.1, -2.6]),
            land_pad(102, "land", "land_graph", [0.2, -3.2, 0.4, -2.6]),
            land_pad(103, "land", "land_graph", [-2.3, 2.6, -2.1, 3.2]),
            land_pad(104, "land", "land_graph", [0.2, 2.6, 0.4, 3.2]),
            lead_pad_with_source_package(301, 1, "bottom", "package_graph", [-2.1, -3.0, -1.9, -2.7]),
            inner_land_pad(201, 101, "land", "land_graph", [-2.25, -3.05, -2.15, -2.75]),
        ]

        alignment = build_pad_center_alignment(objects)
        aligned = apply_pad_center_alignment(objects, alignment)

        self.assertEqual(aligned[0]["bbox"], [-2.3, -3.0, -2.1, -2.5])
        self.assertEqual(aligned[4]["bbox"], [-2.3, -3.2, -2.1, -2.6])
        self.assertEqual(aligned[8]["bbox"], [-2.3, -3.0, -2.1, -2.7])
        self.assertEqual(aligned[9]["bbox"], [-2.25, -3.05, -2.15, -2.75])
        self.assertEqual(aligned[8]["pad_center_alignment_type"], "package_derived_pad_follow_package_pad_center_alignment")
        self.assertNotIn("pad_center_alignment_type", aligned[9])

    def test_circle_land_pads_are_anchor_and_package_pads_move(self) -> None:
        objects = [
            package_pad(1, "bottom", "package_graph", [-2.1, -3.0, -1.9, -2.5]),
            package_pad(2, "bottom", "package_graph", [-0.1, -3.0, 0.1, -2.5]),
            land_circle_pad(101, "land", "land_graph", [-2.3, -3.2, -2.1, -3.0]),
            land_circle_pad(102, "land", "land_graph", [0.2, -3.2, 0.4, -3.0]),
            lead_pad_with_source_package(201, 1, "bottom", "package_graph", [-2.1, -3.0, -1.9, -2.7]),
        ]

        alignment = build_pad_center_alignment(objects)
        aligned = apply_pad_center_alignment(objects, alignment)

        self.assertEqual(alignment["anchor_role"], "land_pad")
        self.assertEqual(alignment["strategy"], "circle_land_center_2d")
        self.assertEqual(aligned[0]["bbox"], [-2.3, -3.35, -2.1, -2.85])
        self.assertEqual(aligned[1]["bbox"], [0.2, -3.35, 0.4, -2.85])
        self.assertEqual(aligned[2]["bbox"], [-2.3, -3.2, -2.1, -3.0])
        self.assertEqual(aligned[4]["bbox"], [-2.3, -3.35, -2.1, -3.05])
        self.assertEqual(aligned[0]["pad_center_alignment_type"], "package_pad_center_to_land_pad_center_2d")
        self.assertEqual(aligned[4]["pad_center_alignment_type"], "package_derived_pad_follow_package_pad_center_alignment_2d")

    def test_circle_land_anchor_uses_2d_center_matching_when_side_counts_do_not_match(self) -> None:
        objects = []
        for index, (x, y) in enumerate([(0.0, -1.0), (1.0, -1.0), (0.0, 0.0), (1.0, 0.0)], start=1):
            objects.append(package_pad(index, "bottom", "package_graph", [x - 0.1, y - 0.1, x + 0.1, y + 0.1]))
        for index, bbox in enumerate(
            [[0.1, -1.2, 0.3, -1.0], [1.1, -1.2, 1.3, -1.0], [0.1, -0.2, 0.3, 0.0], [1.1, -0.2, 1.3, 0.0]],
            start=101,
        ):
            objects.append(land_circle_pad(index, "land", "land_graph", bbox))

        alignment = build_pad_center_alignment(objects)
        aligned = apply_pad_center_alignment(objects, alignment)

        self.assertEqual(alignment["anchor_role"], "land_pad")
        self.assertEqual(alignment["strategy"], "circle_land_center_2d")
        self.assertEqual(alignment["matched_pair_count"], 4)
        self.assertEqual(aligned[0]["bbox"], [0.1, -1.2, 0.3, -1.0])
        self.assertEqual(aligned[1]["bbox"], [1.1, -1.2, 1.3, -1.0])
        self.assertEqual(aligned[4]["bbox"], [0.1, -1.2, 0.3, -1.0])
        self.assertEqual(aligned[0]["pad_center_alignment_type"], "package_pad_center_to_land_pad_center_2d")

    def test_top_bottom_package_pad_alignment_uses_pad_spacing_anchor_and_moves_derived_pads(self) -> None:
        objects = [
            package_pad(1, "top", "top_graph", [-2.1, -1.0, -1.9, -0.8]),
            package_pad(2, "top", "top_graph", [1.9, -1.0, 2.1, -0.8]),
            package_pad(3, "top", "top_graph", [-2.1, 0.8, -1.9, 1.0]),
            package_pad(4, "top", "top_graph", [1.9, 0.8, 2.1, 1.0]),
            package_pad(11, "bottom", "bottom_graph", [-1.8, -0.7, -1.6, -0.5]),
            package_pad(12, "bottom", "bottom_graph", [2.2, -0.7, 2.4, -0.5]),
            package_pad(13, "bottom", "bottom_graph", [-1.8, 1.1, -1.6, 1.3]),
            package_pad(14, "bottom", "bottom_graph", [2.2, 1.1, 2.4, 1.3]),
            lead_pad_with_source_package(101, 11, "bottom", "bottom_graph", [-1.8, -0.7, -1.6, -0.6]),
        ]
        dimensions = [
            {
                "dimension_id": 1,
                "role": "pad_spacing",
                "kind": "distance",
                "axis": "y",
                "target_ids": [1, 3],
                "target_labels": ["pad", "pad"],
                "source_graph": "top_graph",
                "canonical_view": "top",
            }
        ]

        alignment = build_top_bottom_package_pad_alignment(objects, dimensions)
        aligned = apply_top_bottom_package_pad_alignment(objects, alignment)

        self.assertEqual(alignment["status"], "ok")
        self.assertEqual(alignment["anchor_view"], "top")
        self.assertEqual(alignment["moving_view"], "bottom")
        self.assertEqual(alignment["matched_pair_count"], 4)
        self.assertEqual(aligned[4]["bbox"], [-2.1, -1.0, -1.9, -0.8])
        self.assertEqual(aligned[5]["bbox"], [1.9, -1.0, 2.1, -0.8])
        self.assertEqual(aligned[8]["bbox"], [-2.1, -1.0, -1.9, -0.9])
        self.assertEqual(aligned[8]["top_bottom_package_pad_alignment_type"], "package_derived_pad_follow_top_bottom_package_pad_alignment")

    def test_land_package_pair_diagnostics_reports_only_matched_package_land_pairs(self) -> None:
        objects = [
            package_pad(1, "bottom", "package_graph", [0.0, 0.0, 1.0, 1.0]),
            land_pad(101, "land", "land_graph", [0.2, 2.0, 1.2, 3.0]),
            lead_pad(201, "bottom", "package_graph", [0.0, 0.0, 1.0, 0.5]),
        ]
        alignment = {
            "side_alignments": [
                {
                    "side": "top",
                    "axis": "x",
                    "pairs": [
                        {
                            "package_object_index": 0,
                            "package_object_id": 1,
                            "land_object_index": 1,
                            "land_object_id": 101,
                        },
                        {
                            "package_object_index": 2,
                            "package_object_id": 201,
                            "land_object_index": 1,
                            "land_object_id": 101,
                        },
                    ],
                }
            ]
        }

        diagnostics = build_land_package_pair_diagnostics(objects, alignment)

        self.assertEqual(diagnostics["pair_count"], 1)
        self.assertAlmostEqual(diagnostics["pairs"][0]["axis_residual"], 0.2)
        self.assertAlmostEqual(diagnostics["pairs"][0]["center_distance"], (0.2**2 + 2.0**2) ** 0.5)
        self.assertEqual(diagnostics["pairs"][0]["package_object_id"], 1)
        self.assertEqual(diagnostics["pairs"][0]["land_object_id"], 101)

    def test_visible_pad_stacks_hides_singletons(self) -> None:
        stacks = [
            {"id": "singleton", "member_count": 1},
            {"id": "merged", "member_count": 2},
        ]

        self.assertEqual([stack["id"] for stack in visible_pad_stacks(stacks)], ["merged"])

    def test_build_multiview_alignment_writes_aligned_multiview_layers_json(self) -> None:
        payload = {
            "part_number": "PART",
            "dimensions": [],
            "multiview_overlay": {
                "layers": [
                    {
                        "raw_view": "bottom",
                        "canonical_view": "bottom",
                        "graph_path": "package_graph",
                        "objects": [
                            {"source_object_id": 1, "role": "package_pad", "label": "pad", "bbox": [-2.1, -3.0, -1.9, -2.5]},
                            {"source_object_id": 2, "role": "package_pad", "label": "pad", "bbox": [-0.1, -3.0, 0.1, -2.5]},
                            {"source_object_id": 3, "role": "package_pad", "label": "pad", "bbox": [-2.1, 2.5, -1.9, 3.0]},
                            {"source_object_id": 4, "role": "package_pad", "label": "pad", "bbox": [-0.1, 2.5, 0.1, 3.0]},
                            {
                                "source_object_id": 301,
                                "role": "lead_pad",
                                "label": "lead_pad",
                                "bbox": [-2.1, -3.0, -1.9, -2.7],
                                "source_package_pad_id": 1,
                            },
                        ],
                    },
                    {
                        "raw_view": "land",
                        "canonical_view": "land",
                        "graph_path": "land_graph",
                        "objects": [
                            {"source_object_id": 101, "role": "land_pad", "label": "pad", "bbox": [-2.3, -3.2, -2.1, -2.6]},
                            {"source_object_id": 102, "role": "land_pad", "label": "pad", "bbox": [0.2, -3.2, 0.4, -2.6]},
                            {"source_object_id": 103, "role": "land_pad", "label": "pad", "bbox": [-2.3, 2.6, -2.1, 3.2]},
                            {"source_object_id": 104, "role": "land_pad", "label": "pad", "bbox": [0.2, 2.6, 0.4, 3.2]},
                        ],
                    },
                ],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            unified = root / "unified_multiview_layers.json"
            out_dir = root / "out"
            unified.write_text(json.dumps(payload), encoding="utf-8")

            summary = build_multiview_alignment(
                unified_layers_path=unified,
                scan_result_path=None,
                output_dir=out_dir,
                part="PART",
            )

            aligned_path = Path(summary["aligned_multiview_layers_json"])
            aligned = json.loads(aligned_path.read_text(encoding="utf-8"))

        by_id = {obj["source_object_id"]: obj for obj in aligned["objects"]}
        self.assertEqual(aligned["coordinate_mode"], "dimension_scaled_centered")
        self.assertEqual(aligned["alignment_steps"]["pad_center_alignment"]["status"], "ok")
        self.assertEqual(by_id[1]["bbox"], [-2.3, -3.0, -2.1, -2.5])
        self.assertEqual(by_id[301]["bbox"], [-2.3, -3.0, -2.1, -2.7])
        self.assertEqual(by_id[101]["bbox"], [-2.3, -3.2, -2.1, -2.6])
        self.assertEqual(by_id[1]["bbox_before_pad_center_align"], [-2.1, -3.0, -1.9, -2.5])
        self.assertEqual(by_id[301]["pad_center_alignment_type"], "package_derived_pad_follow_package_pad_center_alignment")

    def test_alignment_uses_formal_multiview_kept_objects(self) -> None:
        payload = {
            "part_number": "PART",
            "dimensions": [],
            "package_pads": [
                {"source_object_id": 1, "role": "package_pad", "source_graph": "package_graph", "bbox": [0.0, 0.0, 1.0, 1.0]},
            ],
            "filtered_package_pads": [
                {
                    "source_object_id": 99,
                    "role": "package_pad",
                    "source_graph": "package_graph",
                    "bbox": [0.0, 0.0, 5.0, 5.0],
                    "filtered_reason": "oversized_pad_like_outlier",
                },
            ],
            "multiview_overlay": {
                "layers": [
                    {
                        "raw_view": "bottom",
                        "canonical_view": "bottom",
                        "graph_path": "package_graph",
                        "objects": [
                            {"source_object_id": 1, "role": "package_pad", "label": "pad", "bbox": [0.0, 0.0, 1.0, 1.0]},
                            {"source_object_id": 99, "role": "package_pad", "label": "pad", "bbox": [0.0, 0.0, 5.0, 5.0]},
                        ],
                    },
                ],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            unified = root / "unified_multiview_layers.json"
            out_dir = root / "out"
            unified.write_text(json.dumps(payload), encoding="utf-8")

            summary = build_multiview_alignment(
                unified_layers_path=unified,
                scan_result_path=None,
                output_dir=out_dir,
                part="PART",
            )

            aligned_path = Path(summary["aligned_multiview_layers_json"])
            aligned = json.loads(aligned_path.read_text(encoding="utf-8"))

        self.assertEqual([obj["source_object_id"] for obj in aligned["objects"]], [1])


def outline(source_object_id: int, view: str, graph: str, bbox: list[float]) -> dict[str, object]:
    return {
        "source_object_id": source_object_id,
        "role": "outline_2d",
        "shape_family": "rect",
        "bbox": bbox,
        "raw_view": view,
        "canonical_view": view,
        "source_graph": graph,
    }


def package_pad(source_object_id: int, view: str, graph: str, bbox: list[float]) -> dict[str, object]:
    return pad(source_object_id, "package_pad", view, graph, bbox)


def lead_pad(source_object_id: int, view: str, graph: str, bbox: list[float]) -> dict[str, object]:
    return pad(source_object_id, "lead_pad", view, graph, bbox)


def land_pad(source_object_id: int, view: str, graph: str, bbox: list[float]) -> dict[str, object]:
    return pad(source_object_id, "land_pad", view, graph, bbox)


def land_circle_pad(source_object_id: int, view: str, graph: str, bbox: list[float]) -> dict[str, object]:
    item = pad(source_object_id, "land_pad", view, graph, bbox)
    item["shape_family"] = "circle"
    item["source_label"] = "pad_circle"
    return item


def inner_land_pad(source_object_id: int, source_land_pad_id: int, view: str, graph: str, bbox: list[float]) -> dict[str, object]:
    item = pad(source_object_id, "inner_land_pad", view, graph, bbox)
    item["source_land_pad_id"] = source_land_pad_id
    return item


def lead_pad_with_source_package(source_object_id: int, source_package_pad_id: int, view: str, graph: str, bbox: list[float]) -> dict[str, object]:
    item = pad(source_object_id, "lead_pad", view, graph, bbox)
    item["source_package_pad_id"] = source_package_pad_id
    return item


def pad(source_object_id: int, role: str, view: str, graph: str, bbox: list[float]) -> dict[str, object]:
    return {
        "source_object_id": source_object_id,
        "role": role,
        "shape_family": "rect",
        "bbox": bbox,
        "raw_view": view,
        "canonical_view": view,
        "source_graph": graph,
    }


if __name__ == "__main__":
    unittest.main()
