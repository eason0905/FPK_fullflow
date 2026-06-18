from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from real_image_process.FPK_PJ_fullflow.scoring.merge_gt_metrics import (
    ScoreWeights,
    align_pred_boxes_to_gt,
    alignment_candidate_metrics,
    build_selected_metrics,
    evaluate_merge_root,
    evaluate_part,
    geometry_metrics,
    load_table_missing_parts,
    select_inner_land_by_metric,
    summarize_records,
    weighted_score,
)


class MergeGtMetricsTests(unittest.TestCase):
    def test_geometry_metrics_perfect_match(self) -> None:
        pred = [
            {"role": "merged_pad", "bbox": [0.0, 0.0, 1.0, 1.0]},
            {"role": "land_pad", "bbox": [2.0, 0.0, 3.0, 1.0]},
        ]
        gt = [
            {"role": "lead", "bbox": [0.0, 0.0, 1.0, 1.0]},
            {"role": "land", "bbox": [2.0, 0.0, 3.0, 1.0]},
        ]

        metrics = geometry_metrics(pred, gt)

        self.assertEqual(metrics["pred_pin_count"], 2)
        self.assertEqual(metrics["gt_pin_count"], 2)
        self.assertEqual(metrics["pin_count_abs_error"], 0)
        self.assertAlmostEqual(metrics["iou_ic"], 1.0)
        self.assertAlmostEqual(metrics["d_pin"], 0.0)
        self.assertAlmostEqual(metrics["iou_pin"], 1.0)

    def test_geometry_metrics_selects_quarter_turn_when_all_metrics_improve(self) -> None:
        gt = [
            {"role": "lead", "bbox": [0.0, 0.0, 1.0, 4.0]},
            {"role": "lead", "bbox": [3.0, 0.0, 4.0, 4.0]},
            {"role": "lead", "bbox": [0.0, 7.0, 1.0, 9.0]},
        ]
        pred = [
            {"role": "package_pad", "bbox": [0.0, 0.0, 4.0, 1.0]},
            {"role": "package_pad", "bbox": [0.0, 3.0, 4.0, 4.0]},
            {"role": "package_pad", "bbox": [-5.0, 0.0, -3.0, 1.0]},
        ]

        pred_boxes = [obj["bbox"] for obj in pred]
        gt_boxes = [obj["bbox"] for obj in gt]
        unrotated_boxes, _transform = align_pred_boxes_to_gt(pred_boxes, gt_boxes)
        unrotated_metrics = alignment_candidate_metrics(unrotated_boxes, gt_boxes)
        metrics = geometry_metrics(pred, gt)

        self.assertEqual(metrics["alignment_transform"]["quarter_turns"], 3)
        self.assertGreater(metrics["iou_ic"], unrotated_metrics["iou_ic"])
        self.assertLess(metrics["d_pin"], unrotated_metrics["d_pin"])
        self.assertGreater(metrics["iou_pin"], unrotated_metrics["iou_pin"])
        self.assertAlmostEqual(metrics["iou_ic"], 1.0)
        self.assertAlmostEqual(metrics["d_pin"], 0.0)
        self.assertAlmostEqual(metrics["iou_pin"], 1.0)

    def test_evaluate_part_includes_inner_land_when_it_improves_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            merge_path = root / "mergy_pad.json"
            merge_path.write_text(
                json.dumps(
                    {
                        "objects": [
                            {"role": "package_pad", "bbox": [0.0, 0.0, 1.0, 1.0]},
                            {"role": "inner_land_pad", "bbox": [2.0, 0.0, 3.0, 1.0]},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            gt = [
                {"role": "lead", "bbox": [0.0, 0.0, 1.0, 1.0]},
                {"role": "land", "bbox": [2.0, 0.0, 3.0, 1.0]},
            ]

            with patch("real_image_process.FPK_PJ_fullflow.scoring.merge_gt_metrics.load_gt_pin_objects", return_value=gt):
                record = evaluate_part(merge_path=merge_path, scan_path=root / "ScanResultFormat.txt", part_number="PART")

        self.assertTrue(record["inner_land_included"])
        self.assertEqual(record["selected_metrics"]["pred_pin_count"], 2)
        self.assertAlmostEqual(record["selected_metrics"]["iou_ic"], 1.0)
        self.assertAlmostEqual(record["selected_metrics"]["iou_pin"], 1.0)

    def test_evaluate_part_excludes_inner_land_when_it_worsens_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            merge_path = root / "mergy_pad.json"
            merge_path.write_text(
                json.dumps(
                    {
                        "objects": [
                            {"role": "package_pad", "bbox": [0.0, 0.0, 1.0, 1.0]},
                            {"role": "land_pad", "bbox": [2.0, 0.0, 3.0, 1.0]},
                            {"role": "inner_land_pad", "bbox": [10.0, 10.0, 11.0, 11.0]},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            gt = [
                {"role": "lead", "bbox": [0.0, 0.0, 1.0, 1.0]},
                {"role": "land", "bbox": [2.0, 0.0, 3.0, 1.0]},
            ]

            with patch("real_image_process.FPK_PJ_fullflow.scoring.merge_gt_metrics.load_gt_pin_objects", return_value=gt):
                record = evaluate_part(merge_path=merge_path, scan_path=root / "ScanResultFormat.txt", part_number="PART")

        self.assertFalse(record["inner_land_included"])
        self.assertEqual(record["selected_metrics"]["pred_pin_count"], 2)
        self.assertAlmostEqual(record["selected_metrics"]["iou_ic"], 1.0)

    def test_inner_land_selection_is_independent_per_metric(self) -> None:
        base = {
            "pred_pin_count": 10,
            "gt_pin_count": 10,
            "pin_count_error": 0,
            "pin_count_abs_error": 0,
            "pin_count_sq_error": 0,
            "iou_ic": 0.8,
            "d_pin": 0.2,
            "iou_pin": 0.7,
        }
        with_inner = {
            "pred_pin_count": 15,
            "gt_pin_count": 10,
            "pin_count_error": 5,
            "pin_count_abs_error": 5,
            "pin_count_sq_error": 25,
            "iou_ic": 0.9,
            "d_pin": 0.1,
            "iou_pin": 0.6,
        }

        selection = select_inner_land_by_metric(base, with_inner)
        selected = build_selected_metrics(base, with_inner, selection)

        self.assertEqual(selection, {"iou_ic": True, "pin_count": False, "d_pin": True, "iou_pin": False})
        self.assertEqual(selected["pred_pin_count"], 10)
        self.assertEqual(selected["pin_count_abs_error"], 0)
        self.assertAlmostEqual(selected["iou_ic"], 0.9)
        self.assertAlmostEqual(selected["d_pin"], 0.1)
        self.assertAlmostEqual(selected["iou_pin"], 0.7)

    def test_summarize_records_reports_paper_metrics_and_std(self) -> None:
        records = [
            scored_record(abs_error=1, sq_error=1, iou_ic=0.5, d_pin=2.0, iou_pin=0.25),
            scored_record(abs_error=2, sq_error=4, iou_ic=1.0, d_pin=4.0, iou_pin=0.75),
        ]

        summary = summarize_records(records, merge_root=Path("merge"), dataset_root=Path("dataset"))

        self.assertEqual(summary["part_count"], 2)
        self.assertEqual(summary["scored_part_count"], 2)
        self.assertAlmostEqual(summary["IoU_IC"], 0.75)
        self.assertAlmostEqual(summary["IoU_IC_std"], 0.25)
        self.assertAlmostEqual(summary["Task1_MAE"], 1.5)
        self.assertAlmostEqual(summary["Task1_RMSE"], math.sqrt(2.5))
        self.assertAlmostEqual(summary["Task1_abs_error_std"], 0.5)
        self.assertAlmostEqual(summary["Task2_d_pin"], 3.0)
        self.assertAlmostEqual(summary["Task2_d_pin_std"], 1.0)
        self.assertAlmostEqual(summary["Task3_IoU_pin"], 0.5)
        self.assertAlmostEqual(summary["Task3_IoU_pin_std"], 0.25)
        self.assertIn("weighted_score", summary)

    def test_weighted_score_uses_all_paper_metric_families(self) -> None:
        metrics = {
            "pred_pin_count": 8,
            "gt_pin_count": 10,
            "pin_count_abs_error": 2,
            "iou_ic": 0.5,
            "d_pin_normalized": 0.25,
            "iou_pin": 0.75,
        }

        score = weighted_score(metrics, ScoreWeights(iou_ic=0.25, pin_count=0.25, d_pin=0.25, iou_pin=0.25))

        self.assertAlmostEqual(score["metric_scores"]["pin_count"], 0.8)
        self.assertAlmostEqual(score["metric_scores"]["d_pin"], 0.75)
        self.assertAlmostEqual(score["weighted_score"], (0.5 + 0.8 + 0.75 + 0.75) / 4.0)

    def test_load_table_missing_parts_finds_value_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "A").mkdir()
            (root / "B").mkdir()
            (root / "A" / "graph.json").write_text(
                json.dumps({"constraints": [{"value_source": "table_lookup_missing"}]}),
                encoding="utf-8",
            )
            (root / "B" / "graph.json").write_text(
                json.dumps({"constraints": [{"value_source": "table_lookup"}]}),
                encoding="utf-8",
            )

            self.assertEqual(load_table_missing_parts(root), {"A"})

    def test_evaluate_merge_root_excludes_table_lookup_missing_from_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            merge_root = root / "merge"
            dataset_root = root / "dataset"
            graph_root = root / "graphs"
            output_dir = root / "out"
            for part in ("GOOD", "MISSING"):
                (merge_root / part).mkdir(parents=True)
                (dataset_root / part).mkdir(parents=True)
                (graph_root / part).mkdir(parents=True)
                (merge_root / part / "mergy_pad.json").write_text(
                    json.dumps({"objects": [{"role": "package_pad", "bbox": [0.0, 0.0, 1.0, 1.0]}]}),
                    encoding="utf-8",
                )
            (graph_root / "GOOD" / "graph.json").write_text(json.dumps({"value_source": "table_lookup"}), encoding="utf-8")
            (graph_root / "MISSING" / "graph.json").write_text(
                json.dumps({"value_source": "table_lookup_missing"}),
                encoding="utf-8",
            )
            gt = [{"role": "lead", "bbox": [0.0, 0.0, 1.0, 1.0]}]

            with patch("real_image_process.FPK_PJ_fullflow.scoring.merge_gt_metrics.load_gt_pin_objects", return_value=gt):
                summary = evaluate_merge_root(
                    merge_root=merge_root,
                    dataset_root=dataset_root,
                    output_dir=output_dir,
                    table_missing_graph_root=graph_root,
                )

            self.assertEqual(summary["part_count"], 2)
            self.assertEqual(summary["scored_part_count"], 1)
            self.assertEqual(summary["excluded_table_lookup_missing_part_count"], 1)
            self.assertEqual(summary["unscored_part_count"], 0)
            records = [json.loads(line) for line in (output_dir / "records.jsonl").read_text().splitlines()]
            statuses = {record["part_number"]: record["status"] for record in records}
            self.assertEqual(statuses, {"GOOD": "ok", "MISSING": "excluded_table_lookup_missing"})


def scored_record(
    *,
    abs_error: int,
    sq_error: int,
    iou_ic: float,
    d_pin: float,
    iou_pin: float,
) -> dict[str, object]:
    return {
        "status": "ok",
        "inner_land_included": False,
        "selected_metrics": {
            "pin_count_abs_error": abs_error,
            "pin_count_sq_error": sq_error,
            "iou_ic": iou_ic,
            "d_pin": d_pin,
            "iou_pin": iou_pin,
            "matched_pin_count": 1,
            "pred_pin_count": 1,
            "gt_pin_count": 1,
        },
    }


if __name__ == "__main__":
    unittest.main()
