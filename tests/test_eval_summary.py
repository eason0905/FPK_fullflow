from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from real_image_process.FPK_PJ_fullflow.fullflow.run_context import FullflowConfig, RunContext
from real_image_process.FPK_PJ_fullflow.fullflow.stages.eval_summary import build_eval_summary, run_eval_summary


class EvalSummaryTests(unittest.TestCase):
    def test_build_eval_summary_aggregates_stage_outputs_and_error_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = FullflowConfig(
                asset_dataset_root=root / "assets" / "dataset_full_v5",
                known_issues_path=root / "manifests" / "known.jsonl",
            )
            context = RunContext(run_id="run1", config=config, root=root)
            context.ensure_dirs()
            config.known_issues_path.parent.mkdir(parents=True, exist_ok=True)
            config.known_issues_path.write_text('{"part_number":"P"}\n', encoding="utf-8")
            write_json(context.outputs_dir / "predictions" / "run_summary.json", {
                "file_count": 3,
                "num_group_count": 5,
                "write_enabled": False,
                "valid_predictions": 4,
                "written_predictions": 0,
                "failed_predictions": 1,
                "skipped_predictions": 0,
                "predictions_path": "predictions.jsonl",
                "views": {"bottom": 2, "land": 1},
            })
            write_json(context.outputs_dir / "eval" / "20260101_nothinking" / "overall_summary.json", {
                "overall_accuracy": 0.9,
                "datasets": [
                    {"dataset": "task1_view_classification", "total": 10, "ref_exact_match": 9, "accuracy": 0.9}
                ],
            })
            touch(context.outputs_dir / "review" / "llm_errors" / "index.html")
            write_jsonl(context.outputs_dir / "review" / "llm_errors" / "all" / "cases.jsonl", [{}, {}, {}])
            write_jsonl(
                context.outputs_dir / "review" / "llm_errors" / "task1_view_classification" / "cases.jsonl",
                [{}, {}],
            )
            write_jsonl(
                context.outputs_dir / "review" / "llm_errors" / "task4_dim_target" / "cases.jsonl",
                [{}],
            )
            write_json(context.outputs_dir / "eval" / "gt_alignment" / "summary.json", {
                "summary_path": "gt_summary",
                "mismatches_path": "mismatches",
                "total_parts": 2,
                "aligned_parts": 1,
                "mismatch_parts": 1,
                "missing_gt_parts": 0,
                "missing_canonical_parts": 0,
                "error_source_counts": {
                    "data_coverage": 3,
                    "multiview_integration": 1,
                    "package_graph_reconstruction": 2,
                },
                "objective_error_source_counts": {
                    "gt_annotation_issue": 3,
                    "model_prediction": 2,
                    "multiview_alignment": 1,
                    "package_graph_reconstruction": 2,
                },
                "risk_counts": {"high": 1, "medium": 1, "low": 0},
                "mapping_counts": {"scan_land_to_package_pad_missing_land_view": 3},
                "mismatch_check_counts": {"land_count": 2},
                "count_delta_histograms": {"land_count": {"1": 2}},
                "stage_hint_reason_counts": {
                    "package_graph_land_reconstruction_count_mismatch|land_count_mismatch": 2
                },
            })
            write_json(context.outputs_dir / "multiview" / "summary.json", {
                "total_parts": 2,
                "part_outputs": 2,
                "canonical_parts": 1,
                "graph_based_parts": 1,
                "failure_reason_parts": 1,
                "missing_graph_parts": 1,
                "status_counts": {"canonical": 1, "missing_graphs": 1},
                "dimension_value_source_counts": {"table_lookup": 2, "text_parser": 5},
                "dimension_role_counts": {"pad_size": 3, "pad_spacing": 4},
                "dimension_canonical_view_counts": {"bottom": 4, "land": 3},
                "evidence_type_counts": {"package_graph": 3, "scan_result_format": 2, "table_lookup_files": 1},
            })
            table_missing = context.outputs_dir / "diagnosis" / "table_lookup_missing" / "table_lookup_missing.jsonl"
            table_missing.parent.mkdir(parents=True)
            table_missing.write_text("{}\n{}\n", encoding="utf-8")
            write_json(context.outputs_dir / "review" / "yolo_errors" / "summary.json", {
                "index_path": "yolo_index",
                "total_images": 87,
                "error_cases": 31,
                "model_path": "best.pt",
                "data_yaml": "dataset.yaml",
                "split": "val",
            })

            summary = build_eval_summary(context)

            self.assertEqual(summary["stage_summaries"]["llm_eval"]["overall_accuracy"], 0.9)
            self.assertTrue(summary["stage_summaries"]["llm_review"]["available"])
            self.assertEqual(summary["stage_summaries"]["llm_review"]["total_items"], 3)
            self.assertEqual(
                summary["stage_summaries"]["llm_review"]["task_case_counts"],
                {"task1_view_classification": 2, "task4_dim_target": 1},
            )
            self.assertEqual(summary["stage_summaries"]["predictions"]["run_summary"]["target_count"], 5)
            self.assertEqual(summary["stage_summaries"]["predictions"]["run_summary"]["success_count"], 4)
            self.assertEqual(summary["stage_summaries"]["predictions"]["run_summary"]["failure_count"], 1)
            self.assertEqual(summary["stage_summaries"]["predictions"]["run_summary"]["file_count"], 3)
            self.assertEqual(summary["stage_summaries"]["predictions"]["run_summary"]["valid_predictions"], 4)
            self.assertEqual(summary["stage_summaries"]["llm_eval"]["task_metrics"]["task1_view_classification"]["correct"], 9)
            self.assertEqual(summary["stage_summaries"]["gt_alignment"]["mismatch_parts"], 1)
            self.assertEqual(
                summary["objective_error_source_keys"],
                [
                    "model_prediction",
                    "table_lookup",
                    "package_graph_reconstruction",
                    "multiview_alignment",
                    "scan_result_parsing",
                    "gt_annotation_issue",
                ],
            )
            self.assertEqual(
                summary["objective_error_source_counts"],
                {
                    "gt_annotation_issue": 3,
                    "model_prediction": 2,
                    "multiview_alignment": 1,
                    "package_graph_reconstruction": 2,
                },
            )
            self.assertEqual(summary["alignment_risk_counts"], {"high": 1, "medium": 1, "low": 0})
            self.assertEqual(summary["error_source_overview"]["multiview_integration"]["count"], 1)
            self.assertEqual(summary["error_source_overview"]["package_graph_reconstruction"]["count"], 2)
            self.assertEqual(summary["error_source_overview"]["data_coverage"]["count"], 3)
            self.assertEqual(summary["error_source_overview"]["table_lookup"]["missing_count"], 2)
            self.assertEqual(summary["error_source_overview"]["annotation_gt_mismatch"]["known_issue_count"], 1)
            self.assertEqual(summary["error_source_overview"]["model_prediction"]["llm_review"]["total_items"], 3)
            self.assertEqual(summary["error_source_overview"]["model_prediction"]["yolo_review"]["total_items"], 87)
            self.assertEqual(summary["error_source_overview"]["model_prediction"]["yolo_review"]["error_cases"], 31)
            self.assertEqual(
                summary["error_source_overview"]["model_prediction"]["yolo_review"]["summary_path"],
                str(context.outputs_dir / "review" / "yolo_errors" / "summary.json"),
            )
            self.assertEqual(summary["stage_summaries"]["multiview"]["part_outputs"], 2)
            self.assertEqual(summary["stage_summaries"]["multiview"]["failure_reason_parts"], 1)
            self.assertEqual(summary["stage_summaries"]["multiview"]["status_counts"]["missing_graphs"], 1)
            self.assertEqual(
                summary["stage_summaries"]["multiview"]["dimension_value_source_counts"],
                {"table_lookup": 2, "text_parser": 5},
            )
            self.assertEqual(
                summary["stage_summaries"]["multiview"]["evidence_type_counts"],
                {"package_graph": 3, "scan_result_format": 2, "table_lookup_files": 1},
            )
            self.assertEqual(
                summary["stage_summaries"]["gt_alignment"]["mapping_counts"],
                {"scan_land_to_package_pad_missing_land_view": 3},
            )
            self.assertEqual(summary["stage_summaries"]["gt_alignment"]["risk_counts"], {"high": 1, "medium": 1, "low": 0})
            self.assertEqual(summary["stage_summaries"]["gt_alignment"]["mismatch_check_counts"], {"land_count": 2})

    def test_run_eval_summary_writes_summary_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context = RunContext(run_id="run1", config=FullflowConfig(), root=Path(tmp))
            context.ensure_dirs()

            result = run_eval_summary(context)

            self.assertEqual(result["status"], "success")
            self.assertTrue(Path(result["summary_path"]).exists())
            self.assertTrue(Path(result["stable_summary_path"]).exists())
            self.assertEqual(
                json.loads(Path(result["summary_path"]).read_text(encoding="utf-8")),
                json.loads(Path(result["stable_summary_path"]).read_text(encoding="utf-8")),
            )

    def test_build_eval_summary_reports_output_completeness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context = RunContext(run_id="run1", config=FullflowConfig(), root=Path(tmp))
            context.ensure_dirs()
            outputs = context.outputs_dir

            write_json(outputs / "predictions" / "run_summary.json", {})
            write_json(outputs / "predictions" / "target_summary.json", {})
            touch(outputs / "predictions" / "predictions.jsonl")
            write_json(outputs / "reconstruction" / "run1" / "summary.json", {})
            (outputs / "reconstruction" / "run1" / "graphs").mkdir(parents=True)
            write_json(outputs / "multiview" / "summary.json", {})
            part_dir = outputs / "multiview" / "parts" / "PART"
            write_json(part_dir / "unified_multiview_layers.json", {})
            write_json(part_dir / "evidence.json", {})
            write_json(part_dir / "conflicts.json", [])
            touch(part_dir / "unified_multiview_layers.svg")
            touch(part_dir / "gt_reference.svg")
            touch(part_dir / "aligned_result.svg")
            touch(part_dir / "comparison.svg")
            write_json(outputs / "eval" / "gt_alignment" / "summary.json", {})
            touch(outputs / "eval" / "gt_alignment" / "mismatches.jsonl")
            write_json(outputs / "auto_improve" / "iteration_summary.json", {"queued_cases": 1})
            touch(outputs / "auto_improve" / "reviewed_cases.jsonl")
            touch(outputs / "auto_improve" / "score_history.jsonl")
            for path in (
                outputs / "review" / "llm_errors" / "index.html",
                outputs / "review" / "yolo_errors" / "index.html",
                outputs / "review" / "package_graph" / "index.html",
                outputs / "review" / "package_graph_all_views" / "index.html",
                outputs / "review" / "multiview" / "index.html",
                outputs / "review" / "gt_alignment" / "index.html",
                outputs / "review" / "final_comparison" / "index.html",
            ):
                touch(path)

            summary = build_eval_summary(context)
            completeness = summary["output_completeness"]

            self.assertTrue(completeness["all_required_present"])
            self.assertEqual(completeness["missing_required_artifacts"], [])
            self.assertIn("package_graph_all_views_review", summary["artifacts"])
            self.assertEqual(completeness["multiview_part_files"]["total_part_dirs"], 1)
            self.assertEqual(completeness["multiview_part_files"]["complete_part_count"], 1)
            self.assertEqual(completeness["multiview_part_files"]["incomplete_part_count"], 0)
            self.assertEqual(completeness["multiview_evidence"]["missing_scan_result_evidence_count"], 1)
            self.assertEqual(completeness["multiview_evidence"]["missing_package_graph_evidence_count"], 1)

    def test_build_eval_summary_reports_multiview_evidence_completeness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context = RunContext(run_id="run1", config=FullflowConfig(), root=Path(tmp))
            context.ensure_dirs()
            outputs = context.outputs_dir

            write_json(outputs / "predictions" / "run_summary.json", {})
            write_json(outputs / "predictions" / "target_summary.json", {})
            touch(outputs / "predictions" / "predictions.jsonl")
            write_json(outputs / "reconstruction" / "run1" / "summary.json", {})
            (outputs / "reconstruction" / "run1" / "graphs").mkdir(parents=True)
            write_json(outputs / "multiview" / "summary.json", {})
            ok_part = outputs / "multiview" / "parts" / "OK"
            write_json(
                ok_part / "unified_multiview_layers.json",
                {
                    "status": "canonical",
                    "evidence_refs": [
                        {"evidence_type": "scan_result_format"},
                        {"evidence_type": "package_graph"},
                    ],
                },
            )
            write_json(ok_part / "evidence.json", {})
            write_json(ok_part / "conflicts.json", [])
            touch(ok_part / "unified_multiview_layers.svg")
            touch(ok_part / "gt_reference.svg")
            touch(ok_part / "aligned_result.svg")
            touch(ok_part / "comparison.svg")
            missing_graph_part = outputs / "multiview" / "parts" / "MISSING_GRAPH"
            write_json(
                missing_graph_part / "unified_multiview_layers.json",
                {
                    "status": "missing_graphs",
                    "evidence_refs": [{"evidence_type": "scan_result_format"}],
                },
            )
            write_json(missing_graph_part / "evidence.json", {})
            write_json(missing_graph_part / "conflicts.json", [])
            touch(missing_graph_part / "unified_multiview_layers.svg")
            touch(missing_graph_part / "gt_reference.svg")
            touch(missing_graph_part / "aligned_result.svg")
            touch(missing_graph_part / "comparison.svg")
            write_json(outputs / "eval" / "gt_alignment" / "summary.json", {})
            touch(outputs / "eval" / "gt_alignment" / "mismatches.jsonl")
            write_json(outputs / "auto_improve" / "iteration_summary.json", {"queued_cases": 1})
            touch(outputs / "auto_improve" / "reviewed_cases.jsonl")
            touch(outputs / "auto_improve" / "score_history.jsonl")
            for path in (
                outputs / "review" / "llm_errors" / "index.html",
                outputs / "review" / "yolo_errors" / "index.html",
                outputs / "review" / "package_graph" / "index.html",
                outputs / "review" / "package_graph_all_views" / "index.html",
                outputs / "review" / "multiview" / "index.html",
                outputs / "review" / "gt_alignment" / "index.html",
                outputs / "review" / "final_comparison" / "index.html",
            ):
                touch(path)

            completeness = build_eval_summary(context)["output_completeness"]["multiview_evidence"]

            self.assertEqual(completeness["total_part_dirs"], 2)
            self.assertEqual(completeness["scan_result_evidence_count"], 2)
            self.assertEqual(completeness["missing_scan_result_evidence_count"], 0)
            self.assertEqual(completeness["package_graph_evidence_expected_count"], 1)
            self.assertEqual(completeness["package_graph_evidence_count"], 1)
            self.assertEqual(completeness["missing_package_graph_evidence_count"], 0)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
