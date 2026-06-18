from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from real_image_process.FPK_PJ_fullflow.fullflow.run_context import (
    FullflowConfig,
    RunContext,
    run_stage_command,
    shell_quote,
)
from real_image_process.FPK_PJ_fullflow.fullflow.stages.auto_improve import run_auto_improve
from real_image_process.FPK_PJ_fullflow.fullflow.stages.gt_alignment import build_gt_alignment_command
from real_image_process.FPK_PJ_fullflow.fullflow.stages.llm_review import build_llm_review_command
from real_image_process.FPK_PJ_fullflow.fullflow.stages.make_gallery import build_all_view_gallery_command
from real_image_process.FPK_PJ_fullflow.fullflow.stages.make_gallery import build_gallery_command
from real_image_process.FPK_PJ_fullflow.fullflow.stages.make_gallery import (
    build_package_graph_all_views_review_command,
)
from real_image_process.FPK_PJ_fullflow.fullflow.stages.multiview_integrate import build_multiview_integrate_command
from real_image_process.FPK_PJ_fullflow.fullflow.stages.predict_kie import build_predict_kie_command
from real_image_process.FPK_PJ_fullflow.fullflow.stages.predict_kie import ensure_prediction_jsonl_artifact
from real_image_process.FPK_PJ_fullflow.fullflow.stages.predict_kie import run_predict_kie
from real_image_process.FPK_PJ_fullflow.fullflow.stages.reconstruct_graph import build_reconstruct_command
from real_image_process.FPK_PJ_fullflow.fullflow.stages.score_merge_gt import build_score_merge_gt_command
from real_image_process.FPK_PJ_fullflow.fullflow.stages.yolo_review import build_yolo_review_command


class FullflowWrapperTests(unittest.TestCase):
    def test_shell_quote_preserves_safe_paths_and_quotes_spaces(self) -> None:
        self.assertEqual(shell_quote("abc/DEF-123"), "abc/DEF-123")
        self.assertEqual(shell_quote("path with space"), "'path with space'")
        self.assertEqual(shell_quote("a'b"), "'a'\"'\"'b'")

    def test_run_stage_command_dry_run_writes_status_and_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context = RunContext(run_id="unit", config=FullflowConfig(), root=Path(tmp))
            context.ensure_dirs()

            result = run_stage_command(context, "toy", ["python", "--version"], dry_run=True)

            self.assertEqual(result["status"], "dry_run")
            self.assertIn("toy", json.loads(context.status_path.read_text(encoding="utf-8")))
            self.assertIn("python --version", context.commands_path.read_text(encoding="utf-8"))

    def test_stage_commands_use_asset_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = FullflowConfig(
                asset_dataset_root=root / "assets" / "dataset",
                asset_model_path=root / "assets" / "model",
                asset_adapter_path=root / "assets" / "adapter",
                known_issues_path=root / "manifests" / "known.jsonl",
                include_views=("top", "bottom", "land"),
                exclude_value_sources=("table_lookup_missing",),
                yolo_review={
                    "enabled": True,
                    "model_path": str(root / "assets" / "yolo" / "best.pt"),
                    "data_yaml": str(root / "assets" / "yolo" / "dataset.yaml"),
                    "split": "val",
                    "conf": 0.25,
                    "iou": 0.5,
                    "imgsz": 1280,
                    "device": "0",
                    "max_images": 2,
                },
            )
            context = RunContext(run_id="unit", config=config, root=root)
            context.ensure_dirs()
            config.known_issues_path.parent.mkdir(parents=True, exist_ok=True)
            config.known_issues_path.write_text("", encoding="utf-8")

            predict_cmd = build_predict_kie_command(context, context.outputs_dir / "predictions", list_only=True)
            recon_cmd = build_reconstruct_command(context)
            gallery_cmd = build_gallery_command(context)
            all_view_gallery_cmd = build_all_view_gallery_command(context)
            all_view_review_cmd = build_package_graph_all_views_review_command(context)
            multiview_cmd = build_multiview_integrate_command(context)
            gt_alignment_cmd = build_gt_alignment_command(context)
            score_cmd = build_score_merge_gt_command(context)
            llm_review_cmd = build_llm_review_command(context, eval_dir=context.outputs_dir / "eval" / "unit_eval")
            yolo_review_cmd = build_yolo_review_command(context)

            self.assertIn(str(config.asset_dataset_root), predict_cmd)
            self.assertIn(str(config.asset_model_path), predict_cmd)
            self.assertIn(str(config.asset_adapter_path), predict_cmd)
            self.assertIn(str(config.asset_dataset_root), recon_cmd)
            self.assertIn(str(config.asset_dataset_root), multiview_cmd)
            self.assertIn(str(context.outputs_dir / "reconstruction" / context.run_id), multiview_cmd)
            self.assertIn(str(context.outputs_dir / "multiview"), multiview_cmd)
            self.assertIn(str(config.asset_dataset_root), gt_alignment_cmd)
            self.assertIn(str(context.outputs_dir / "multiview"), gt_alignment_cmd)
            self.assertIn(str(context.outputs_dir / "eval" / "gt_alignment"), gt_alignment_cmd)
            self.assertIn("real_image_process.FPK_PJ_fullflow.scoring.merge_gt_metrics", score_cmd)
            self.assertIn(str(context.outputs_dir / "review" / "package_graph_overlay_gallery"), score_cmd)
            self.assertIn(str(config.asset_dataset_root), score_cmd)
            self.assertIn(str(context.outputs_dir / "review" / "merge_gt_score_gallery"), score_cmd)
            self.assertIn(str(context.outputs_dir / "reconstruction" / context.run_id / "graphs"), score_cmd)
            self.assertIn("llm-errors", llm_review_cmd)
            self.assertIn(str(context.outputs_dir / "eval" / "unit_eval"), llm_review_cmd)
            self.assertIn(str(context.outputs_dir / "review" / "llm_errors"), llm_review_cmd)
            self.assertIn("yolo-errors", yolo_review_cmd)
            self.assertIn(str(root / "assets" / "yolo" / "best.pt"), yolo_review_cmd)
            self.assertIn(str(root / "assets" / "yolo" / "dataset.yaml"), yolo_review_cmd)
            self.assertIn("--max-images", yolo_review_cmd)
            self.assertIn("--include-view", gallery_cmd)
            self.assertIn("top", gallery_cmd)
            self.assertIn("--exclude-value-source", gallery_cmd)
            self.assertIn("table_lookup_missing", gallery_cmd)
            self.assertIn("--exclude-known-issues", gallery_cmd)
            self.assertNotIn("--include-view", all_view_gallery_cmd)
            self.assertIn("package_graph_all_views_source", all_view_gallery_cmd)
            self.assertIn(
                str(context.outputs_dir / "review" / "package_graph_all_views_source" / "risk_report.jsonl"),
                all_view_review_cmd,
            )
            self.assertIn(str(context.outputs_dir / "review" / "package_graph_all_views"), all_view_review_cmd)

    def test_predict_kie_dry_run_declares_predictions_jsonl_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = RunContext(run_id="unit", config=FullflowConfig(), root=root)
            context.ensure_dirs()

            result = run_predict_kie(context, dry_run=True)

            self.assertEqual(result["status"], "dry_run")
            self.assertEqual(
                result["expected_outputs"]["predictions"],
                str(context.outputs_dir / "predictions" / "predictions.jsonl"),
            )

    def test_auto_improve_dry_run_declares_queue_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context = RunContext(run_id="unit", config=FullflowConfig(), root=Path(tmp))
            context.ensure_dirs()

            result = run_auto_improve(context, dry_run=True)

            self.assertEqual(result["status"], "dry_run")
            self.assertEqual(result["iteration_summary_path"], str(context.outputs_dir / "auto_improve" / "iteration_summary.json"))
            self.assertEqual(result["reviewed_cases_path"], str(context.outputs_dir / "auto_improve" / "reviewed_cases.jsonl"))
            self.assertEqual(result["score_history_path"], str(context.outputs_dir / "auto_improve" / "score_history.jsonl"))

    def test_ensure_prediction_jsonl_artifact_creates_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "outputs" / "predictions" / "predictions.jsonl"

            ensure_prediction_jsonl_artifact(path)

            self.assertTrue(path.exists())
            self.assertEqual(path.read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
