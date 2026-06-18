from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from real_image_process.FPK_PJ_fullflow.multiview.merge_pads import (
    build_mergy_pad_payload,
    build_multiview_mergy_pad,
)


class MultiviewMergyPadTests(unittest.TestCase):
    def test_front_side_same_package_pad_merges_front_x_and_side_y(self) -> None:
        payload = aligned_payload(
            [
                package_pad(10, "bottom", "graph", [0.0, 0.0, 10.0, 10.0]),
                lead_pad(1, "front", "graph", 10, [0.0, 0.0, 4.0, 4.0]),
                lead_pad(2, "side", "graph", 10, [2.0, 1.0, 6.0, 5.0]),
            ]
        )

        result = build_mergy_pad_payload(payload, part_number="PART")

        self.assertEqual(result["candidate_group_count"], 1)
        self.assertEqual(result["merged_pad_count"], 1)
        self.assertEqual(result["unmerged_pad_count"], 0)
        self.assertEqual(result["unresolved_count"], 0)
        self.assertEqual(result["merged_pads"][0]["bbox"], [0.0, 1.0, 4.0, 5.0])
        self.assertEqual(result["merged_pads"][0]["merge_policy"], "front_x_side_y")
        self.assertEqual(result["object_count"], 1)
        self.assertEqual([obj["role"] for obj in result["objects"]], ["merged_pad"])
        self.assertNotIn("members", result["merged_pads"][0])
        self.assertNotIn("context_objects", result)
        self.assertNotIn("unresolved_evidence", result)

    def test_package_pad_with_derived_lead_pad_is_removed_from_merge_graph(self) -> None:
        payload = aligned_payload(
            [
                package_pad(10, "bottom", "graph", [0.0, 0.0, 10.0, 10.0]),
                package_pad(11, "bottom", "graph", [20.0, 0.0, 30.0, 10.0]),
                lead_pad(1, "side", "graph", 10, [0.0, 0.0, 1.0, 1.0]),
            ]
        )

        result = build_mergy_pad_payload(payload, part_number="PART")

        self.assertEqual([obj["role"] for obj in result["objects"]], ["package_pad", "unmerged_pad"])
        self.assertEqual(result["objects"][0]["source_object_id"], 11)
        self.assertEqual(result["unmerged_pads"][0]["source_package_pad_id"], 10)

    def test_front_lead_same_package_pad_merges_front_x_and_lead_y(self) -> None:
        payload = aligned_payload(
            [
                lead_pad(1, "front", "graph", 10, [0.0, 0.0, 4.0, 4.0]),
                lead_pad(2, "lead", "graph", 10, [2.0, 1.0, 6.0, 5.0]),
            ]
        )

        result = build_mergy_pad_payload(payload, part_number="PART")

        self.assertEqual(result["candidate_group_count"], 1)
        self.assertEqual(result["merged_pad_count"], 1)
        self.assertEqual(result["merged_pads"][0]["bbox"], [0.0, 1.0, 4.0, 5.0])
        self.assertEqual(result["merged_pads"][0]["merge_policy"], "front_x_lead_y")

    def test_lead_side_same_package_pad_merges_lead_x_and_side_y(self) -> None:
        payload = aligned_payload(
            [
                lead_pad(1, "lead", "graph", 10, [0.0, 0.0, 4.0, 4.0]),
                lead_pad(2, "side", "graph", 10, [2.0, 1.0, 6.0, 5.0]),
            ]
        )

        result = build_mergy_pad_payload(payload, part_number="PART")

        self.assertEqual(result["candidate_group_count"], 1)
        self.assertEqual(result["merged_pad_count"], 1)
        self.assertEqual(result["merged_pads"][0]["bbox"], [0.0, 1.0, 4.0, 5.0])
        self.assertEqual(result["merged_pads"][0]["merge_policy"], "lead_x_side_y")

    def test_front_side_is_preferred_when_lead_also_exists(self) -> None:
        payload = aligned_payload(
            [
                lead_pad(1, "front", "graph", 10, [0.0, 0.0, 4.0, 4.0]),
                lead_pad(2, "side", "graph", 10, [2.0, 1.0, 6.0, 5.0]),
                lead_pad(3, "lead", "graph", 10, [8.0, 8.0, 9.0, 9.0]),
            ]
        )

        result = build_mergy_pad_payload(payload, part_number="PART")

        self.assertEqual(result["candidate_group_count"], 1)
        self.assertEqual(result["merged_pad_count"], 1)
        self.assertEqual(result["merged_pads"][0]["bbox"], [0.0, 1.0, 4.0, 5.0])
        self.assertEqual(result["merged_pads"][0]["merge_policy"], "front_x_side_y")

    def test_invalid_front_or_side_bbox_is_unresolved(self) -> None:
        payload = aligned_payload(
            [
                lead_pad(1, "front", "graph", 10, [1.0, 0.0, 1.0, 1.0]),
                lead_pad(2, "side", "graph", 10, [2.0, 2.0, 3.0, 3.0]),
            ]
        )

        result = build_mergy_pad_payload(payload, part_number="PART")

        self.assertEqual(result["candidate_group_count"], 1)
        self.assertEqual(result["merged_pad_count"], 0)
        self.assertEqual(result["unresolved_count"], 1)
        self.assertNotIn("unresolved_evidence", result)

    def test_single_view_pad_evidence_is_not_a_candidate(self) -> None:
        payload = aligned_payload(
            [
                lead_pad(1, "front", "graph", 10, [0.0, 0.0, 1.0, 1.0]),
                lead_pad(2, "front", "graph", 11, [2.0, 2.0, 3.0, 3.0]),
            ]
        )

        result = build_mergy_pad_payload(payload, part_number="PART")

        self.assertEqual(result["candidate_group_count"], 0)
        self.assertEqual(result["merged_pad_count"], 0)
        self.assertEqual(result["unmerged_pad_count"], 2)
        self.assertEqual(result["unresolved_count"], 0)
        self.assertEqual(result["unmerged_pads"][0]["bbox"], [0.0, 0.0, 1.0, 1.0])
        self.assertEqual(result["unmerged_pads"][0]["merge_policy"], "unmerged")
        self.assertEqual([obj["role"] for obj in result["objects"]], ["unmerged_pad", "unmerged_pad"])
        self.assertNotIn("members", result["unmerged_pads"][0])

    def test_eight_front_partial_width_pads_use_square_width_fallback(self) -> None:
        payload = aligned_payload(
            [
                lead_pad(
                    index,
                    "front",
                    "graph",
                    index,
                    [index * 10.0, 0.0, index * 10.0 + 2.0, 8.0],
                    role="partial_pad_width",
                )
                for index in range(8)
            ]
        )

        result = build_mergy_pad_payload(payload, part_number="PART")

        self.assertEqual(result["merged_pad_count"], 0)
        self.assertEqual(result["unmerged_pad_count"], 8)
        for index, pad in enumerate(result["unmerged_pads"]):
            self.assertEqual(pad["bbox"], [index * 10.0, 3.0, index * 10.0 + 2.0, 5.0])
            self.assertEqual(pad["original_unmerged_bbox"], [index * 10.0, 0.0, index * 10.0 + 2.0, 8.0])
            self.assertEqual(pad["merge_policy"], "unmerged_front_width_square_fallback")
            self.assertEqual(pad["fallback_reason"], "eight_pin_front_only_partial_pad_width")
        self.assertEqual(
            sorted(obj["bbox"] for obj in result["objects"]),
            [[index * 10.0, 3.0, index * 10.0 + 2.0, 5.0] for index in range(8)],
        )

    def test_two_front_partial_width_pads_keep_original_height(self) -> None:
        payload = aligned_payload(
            [
                lead_pad(1, "front", "graph", 1, [0.0, 0.0, 2.0, 8.0], role="partial_pad_width"),
                lead_pad(2, "front", "graph", 2, [10.0, 0.0, 12.0, 8.0], role="partial_pad_width"),
            ]
        )

        result = build_mergy_pad_payload(payload, part_number="PART")

        self.assertEqual(result["unmerged_pads"][0]["bbox"], [0.0, 0.0, 2.0, 8.0])
        self.assertEqual(result["unmerged_pads"][0]["merge_policy"], "unmerged")
        self.assertNotIn("fallback_reason", result["unmerged_pads"][0])

    def test_different_package_pad_ids_are_not_merged_together(self) -> None:
        payload = aligned_payload(
            [
                lead_pad(1, "front", "graph", 10, [0.0, 0.0, 4.0, 4.0]),
                lead_pad(2, "side", "graph", 11, [2.0, 1.0, 6.0, 5.0]),
            ]
        )

        result = build_mergy_pad_payload(payload, part_number="PART")

        self.assertEqual(result["candidate_group_count"], 0)
        self.assertEqual(result["merged_pad_count"], 0)
        self.assertEqual(result["unmerged_pad_count"], 2)

    def test_invalid_merge_bbox_preserves_valid_unmerged_pad(self) -> None:
        payload = aligned_payload(
            [
                lead_pad(1, "front", "graph", 10, [1.0, 0.0, 1.0, 1.0]),
                lead_pad(2, "side", "graph", 10, [2.0, 2.0, 3.0, 3.0]),
            ]
        )

        result = build_mergy_pad_payload(payload, part_number="PART")

        self.assertEqual(result["candidate_group_count"], 1)
        self.assertEqual(result["merged_pad_count"], 0)
        self.assertEqual(result["unmerged_pad_count"], 1)
        self.assertEqual(result["unmerged_pads"][0]["bbox"], [2.0, 2.0, 3.0, 3.0])
        self.assertEqual(result["unmerged_pads"][0]["unmerged_reason"], "invalid_merge_bbox")

    def test_build_multiview_mergy_pad_writes_json_and_svg(self) -> None:
        payload = aligned_payload(
            [
                lead_pad(1, "front", "graph", 10, [0.0, 0.0, 4.0, 4.0]),
                lead_pad(2, "side", "graph", 10, [2.0, 1.0, 6.0, 5.0]),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aligned_path = root / "aligned_multiview_layers.json"
            out_dir = root / "out"
            aligned_path.write_text(json.dumps(payload), encoding="utf-8")

            summary = build_multiview_mergy_pad(
                aligned_layers_path=aligned_path,
                output_dir=out_dir,
                part="PART",
            )

            self.assertTrue(Path(summary["mergy_pad_json"]).is_file())
            self.assertTrue(Path(summary["mergy_pad_svg"]).is_file())
            written = json.loads(Path(summary["mergy_pad_json"]).read_text(encoding="utf-8"))
            self.assertEqual(written["merged_pads"][0]["bbox"], [0.0, 1.0, 4.0, 5.0])
            self.assertEqual(written["unmerged_pads"], [])
            self.assertEqual([obj["role"] for obj in written["objects"]], ["merged_pad"])
            self.assertNotIn("members", written["merged_pads"][0])
            svg = Path(summary["mergy_pad_svg"]).read_text(encoding="utf-8")
            self.assertNotIn("front evidence", svg)
            self.assertNotIn("side evidence", svg)
            self.assertNotIn("Front/side mergy pad", svg)
            self.assertIn("merged pad", svg)


def aligned_payload(objects: list[dict[str, object]]) -> dict[str, object]:
    return {
        "part_number": "PART",
        "coordinate_mode": "dimension_scaled_centered",
        "objects": objects,
    }


def package_pad(source_object_id: int, view: str, graph: str, bbox: list[float]) -> dict[str, object]:
    return {
        "source_object_id": source_object_id,
        "role": "package_pad",
        "bbox": bbox,
        "raw_view": view,
        "canonical_view": view,
        "source_graph": graph,
    }


def lead_pad(
    source_object_id: int,
    view: str,
    graph: str,
    source_package_pad_id: int,
    bbox: list[float],
    *,
    role: str = "lead_pad",
) -> dict[str, object]:
    return {
        "source_object_id": source_object_id,
        "role": role,
        "bbox": bbox,
        "raw_view": view,
        "canonical_view": "lateral" if view in {"front", "side"} else view,
        "source_graph": graph,
        "source_package_pad_id": source_package_pad_id,
    }


if __name__ == "__main__":
    unittest.main()
