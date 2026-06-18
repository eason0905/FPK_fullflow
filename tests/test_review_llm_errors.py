from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from real_image_process.FPK_PJ_fullflow.review.builders.llm_errors import (
    build_llm_error_review,
    iter_llm_error_cases,
    resolve_source_info,
)
from real_image_process.FPK_PJ_fullflow.review.cli import default_llm_output_root


class LlmErrorReviewTests(unittest.TestCase):
    def test_iter_llm_error_cases_keeps_only_incorrect_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "image.png"
            image.write_bytes(b"fake")
            pred_dir = root / "eval" / "task5_dim_start_anchor"
            pred_dir.mkdir(parents=True)
            pred_path = pred_dir / "predictions.jsonl"
            rows = [
                {
                    "index": 0,
                    "dataset": "task5_dim_start_anchor",
                    "image_path": str(image),
                    "gold_refs": ["center", "center"],
                    "pred_refs": ["bottom_edge", "top_edge"],
                    "clean_exact_match": False,
                    "prompt": "prompt",
                },
                {
                    "index": 1,
                    "dataset": "task5_dim_start_anchor",
                    "image_path": str(image),
                    "gold_refs": ["center", "center"],
                    "pred_refs": ["center", "center"],
                    "clean_exact_match": True,
                },
            ]
            pred_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            cases = list(iter_llm_error_cases(pred_path, output_root=root / "review"))

            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].case_id, "task5_dim_start_anchor:0")
            self.assertEqual(cases[0].reason, "wrong_anchor")
            self.assertEqual(cases[0].expected, "center\ncenter")
            self.assertEqual(cases[0].predicted, "bottom_edge\ntop_edge")

    def test_build_llm_error_review_writes_per_task_gallery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_dir = root / "runs" / "r1" / "outputs" / "eval" / "eval1"
            image = root / "image.png"
            image.write_bytes(b"fake")
            task_dir = eval_dir / "task4_dim_target"
            task_dir.mkdir(parents=True)
            (task_dir / "predictions.jsonl").write_text(
                json.dumps(
                    {
                        "index": 2,
                        "dataset": "task4_dim_target",
                        "image_path": str(image),
                        "gold_refs": ["1", "2"],
                        "pred_refs": ["1"],
                        "clean_exact_match": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            output_root = root / "review"
            result = build_llm_error_review(eval_dir, output_root, tasks=("task4_dim_target",))

            self.assertEqual(result["total_cases"], 1)
            self.assertTrue((output_root / "index.html").exists())
            self.assertTrue((output_root / "task4_dim_target" / "index.html").exists())
            self.assertTrue((output_root / "task4_dim_target" / "cases.jsonl").exists())
            self.assertTrue((output_root / "assets" / "images" / "task4_dim_target").is_dir())

    def test_resolve_source_info_maps_overlay_to_part_number(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotation_dir = root / "assets" / "datasets" / "dataset_full_v4" / "PART123" / "extract_image"
            annotation_dir.mkdir(parents=True)
            annotation_path = annotation_dir / "sample_Bottom_0.json"
            annotation_path.write_text("{}", encoding="utf-8")
            overlay = (
                root
                / "assets"
                / "datasets"
                / "dataset_json"
                / "v4"
                / "task345_overlay_images"
                / "sample_Bottom_0__dim006.png"
            )
            overlay.parent.mkdir(parents=True)
            overlay.write_bytes(b"fake")

            source_info = resolve_source_info(overlay)

            self.assertEqual(source_info["part_number"], "PART123")
            self.assertEqual(source_info["annotation_path"], str(annotation_path.resolve()))
            self.assertEqual(source_info["source_image_stem"], "sample_Bottom_0")

    def test_default_llm_output_root_uses_run_outputs_review(self) -> None:
        eval_dir = Path("/tmp/run/outputs/eval/20260527_010110_nothinking")

        self.assertEqual(
            default_llm_output_root(eval_dir),
            Path("/tmp/run/outputs/review/llm_errors"),
        )


if __name__ == "__main__":
    unittest.main()
