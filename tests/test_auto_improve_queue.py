from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from real_image_process.FPK_PJ_fullflow.auto_improve.queue import build_auto_improve_queue


class AutoImproveQueueTests(unittest.TestCase):
    def test_build_auto_improve_queue_orders_low_score_cases_and_links_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            alignment_dir = run_dir / "outputs" / "eval" / "gt_alignment"
            part_dir = run_dir / "outputs" / "multiview" / "parts" / "PART_A"
            part_dir.mkdir(parents=True)
            image_path = run_dir / "assets" / "PART_A.png"
            image_path.parent.mkdir(parents=True)
            image_path.write_text("", encoding="utf-8")
            write_json(
                part_dir / "evidence.json",
                {
                    "evidence_refs": [
                        {
                            "raw_view": "bottom",
                            "canonical_view": "bottom",
                            "image_path": str(image_path),
                            "annotation_path": "anno.json",
                            "graph_path": "graph.json",
                        }
                    ]
                },
            )
            write_json(
                alignment_dir / "summary.json",
                {
                    "parts": [
                        {
                            "part_number": "PART_B",
                            "status": "aligned",
                            "alignment_scores": {"overall_score": 0.9},
                            "reasons": [],
                            "stage_hints": [],
                            "error_sources": [],
                        },
                        {
                            "part_number": "PART_A",
                            "status": "mismatch",
                            "alignment_scores": {"overall_score": 0.2},
                            "reasons": ["land_count_mismatch"],
                            "stage_hints": ["package_graph_land_reconstruction_count_mismatch"],
                            "error_sources": ["package_graph_reconstruction"],
                            "objective_error_sources": ["model_prediction", "package_graph_reconstruction"],
                            "scan_result_path": "ScanResultFormat.txt",
                            "unified_multiview_layers_path": str(part_dir / "unified_multiview_layers.json"),
                            "alignment_path": "alignment.json",
                        },
                    ]
                },
            )

            summary = build_auto_improve_queue(run_dir=run_dir, output_root=run_dir / "outputs" / "auto_improve")

            self.assertEqual(summary["queued_cases"], 2)
            self.assertEqual(summary["risk_counts"], {"high": 1, "medium": 0, "low": 1})
            self.assertEqual(summary["lowest_score_cases"][0]["part_number"], "PART_A")
            self.assertEqual(
                summary["lowest_score_cases"][0]["objective_error_sources"],
                ["model_prediction", "package_graph_reconstruction"],
            )
            rows = read_jsonl(Path(summary["reviewed_cases_path"]))
            self.assertEqual(rows[0]["part_number"], "PART_A")
            self.assertEqual(rows[0]["suggested_action"], "inspect_package_graph_reconstruction")
            self.assertEqual(rows[0]["objective_error_sources"], ["model_prediction", "package_graph_reconstruction"])
            self.assertEqual(rows[0]["source_images"][0]["raw_view"], "bottom")
            self.assertEqual(rows[0]["paths"]["conflicts"], str(part_dir / "conflicts.json"))
            history = read_jsonl(Path(summary["score_history_path"]))
            self.assertEqual(history[0]["overall_score"], 0.2)
            self.assertEqual(history[0]["objective_error_sources"], ["model_prediction", "package_graph_reconstruction"])


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
