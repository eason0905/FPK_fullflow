from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from real_image_process.FPK_PJ_fullflow.review.adapters.package_graph import build_package_graph_review
from real_image_process.FPK_PJ_fullflow.review.adapters.package_graph import fullflow_url_prefix
from real_image_process.FPK_PJ_fullflow.review.adapters.package_graph import restore_notes_from_history
from real_image_process.FPK_PJ_fullflow.review.adapters.package_graph import risk_level
from real_image_process.FPK_PJ_fullflow.review.adapters.package_graph import root_relative_url
from real_image_process.FPK_PJ_fullflow.review.adapters.package_graph import split_items_by_view
from real_image_process.FPK_PJ_fullflow.review.common.review_server import save_note
from real_image_process.FPK_PJ_fullflow.review.common.schemas import ReviewItem


class PackageGraphReviewTests(unittest.TestCase):
    def test_risk_level_matches_package_graph_risk_report_thresholds(self) -> None:
        self.assertEqual(risk_level(9.999), "low")
        self.assertEqual(risk_level(10.0), "medium")
        self.assertEqual(risk_level(29.999), "medium")
        self.assertEqual(risk_level(30.0), "high")

    def test_build_package_graph_review_writes_pages_and_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "FPK_PJ_fullflow"
            run_root = root / "runs" / "run1" / "outputs"
            graph = run_root / "reconstruction" / "run1" / "graphs" / "PART" / "sample.package_graph.json"
            image = root / "assets" / "datasets" / "dataset_full_v4" / "PART" / "extract_image" / "sample.png"
            overlay = run_root / "visualization" / "reconstruction_run1" / "PART" / "sample.package_graph.png"
            annotation = image.with_suffix(".json")
            for path in (graph, image, overlay, annotation):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
            risk_report = run_root / "review" / "top_bottom_land_filtered" / "risk_report.jsonl"
            risk_report.parent.mkdir(parents=True, exist_ok=True)
            risk_report.write_text(
                json.dumps(
                    {
                        "rank": 1,
                        "risk_score": 5,
                        "risk_reasons": ["1 ignored dimensions"],
                        "part_number": "PART",
                        "view": "bottom",
                        "graph_path": str(graph),
                        "image_path": str(image),
                        "overlay_path": str(overlay),
                        "annotation_path": str(annotation),
                        "metrics": {"mean_iou_vs_yolo": 0.8},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            output_root = run_root / "review" / "package_graph"
            result = build_package_graph_review(
                risk_report_path=risk_report,
                output_root=output_root,
                fullflow_root=root,
                run_id="run1",
            )

            self.assertEqual(result["total_items"], 1)
            self.assertEqual(result["split_by"], "view")
            self.assertEqual(result["page_sizes"], {"bottom": 1})
            self.assertTrue((output_root / "index.html").exists())
            self.assertTrue((output_root / "pages" / "bottom.html").exists())
            self.assertTrue((output_root / "data" / "cases.json").exists())
            self.assertTrue((output_root / "data" / "notes.json").exists())
            cases = json.loads((output_root / "data" / "cases.json").read_text(encoding="utf-8"))
            self.assertEqual([media["label"] for media in cases[0]["media"]], ["Graph rendering"])

    def test_split_items_by_view_orders_multiview_pages_stably(self) -> None:
        items = [
            ReviewItem(
                case_id=view,
                title=view,
                rank=0,
                part_number="PART",
                file_name=f"{view}.json",
                view=view,
                risk_score=0.0,
                risk_level="low",
                risk_reasons=[],
                media=[],
            )
            for view in ("front", "bottom", "land_detail", "top")
        ]

        page_groups = split_items_by_view(items)

        self.assertEqual([label for label, _ in page_groups], ["bottom", "top", "front", "land_detail"])

    def test_root_relative_url_uses_fullflow_root_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            fullflow_root = workspace / "real_image_process" / "FPK_PJ_fullflow"
            asset = fullflow_root / "assets" / "datasets" / "dataset_full_v5" / "PART" / "image.png"
            asset.parent.mkdir(parents=True)
            asset.write_text("", encoding="utf-8")

            self.assertEqual(
                fullflow_url_prefix(fullflow_root, workspace_root=workspace),
                "",
            )
            self.assertEqual(
                root_relative_url(asset, fullflow_root, workspace_root=workspace),
                "assets/datasets/dataset_full_v5/PART/image.png",
            )

    def test_save_note_updates_notes_json_and_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "FPK_PJ_fullflow"
            notes = root / "runs" / "run1" / "outputs" / "review" / "package_graph" / "data" / "notes.json"
            history = notes.with_name("notes_history.jsonl")
            notes.parent.mkdir(parents=True)
            notes.write_text('{"gallery_id":"package_graph","run_id":"run1","updated_at":null,"items":{}}\n')

            result = save_note(
                root,
                {
                    "notes_path": notes.relative_to(root).as_posix(),
                    "history_path": history.relative_to(root).as_posix(),
                    "gallery_id": "package_graph",
                    "run_id": "run1",
                    "payload": {
                        "case_id": "case1",
                        "issue_text": "pad is outside outline",
                        "category": "algorithm_error",
                        "status": "open",
                    },
                },
            )

            self.assertIn("case1", result["items"])
            self.assertEqual(result["items"]["case1"]["issue_text"], "pad is outside outline")
            self.assertTrue(history.exists())
            self.assertEqual(len(history.read_text(encoding="utf-8").splitlines()), 1)

    def test_restore_notes_from_history_keeps_notes_when_cases_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            notes = root / "notes.json"
            history = root / "notes_history.jsonl"
            notes.write_text(
                json.dumps(
                    {
                        "gallery_id": "package_graph",
                        "run_id": "run1",
                        "updated_at": "2026-01-01T00:00:00+00:00",
                        "items": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            history.write_text(
                json.dumps(
                    {
                        "gallery_id": "package_graph",
                        "run_id": "run1",
                        "updated_at": "2026-01-01T00:00:01+00:00",
                        "case_id": "case1",
                        "note": {
                            "case_id": "case1",
                            "issue_text": "pad should touch outline",
                            "category": "algorithm_error",
                            "status": "need_check",
                            "updated_at": "2026-01-01T00:00:01+00:00",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            restore_notes_from_history(notes, history)

            payload = json.loads(notes.read_text(encoding="utf-8"))
            self.assertIn("case1", payload["items"])
            self.assertEqual(payload["items"]["case1"]["issue_text"], "pad should touch outline")


if __name__ == "__main__":
    unittest.main()
