from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .adapters.package_graph import build_package_graph_review
from .adapters.gt_alignment import build_final_comparison_review, build_gt_alignment_review
from .adapters.multiview import build_multiview_review
from .builders.llm_errors import DEFAULT_TASKS, build_llm_error_review
from .builders.yolo_errors import build_yolo_error_review


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FPK_PJ fullflow review galleries.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    llm = subparsers.add_parser("llm-errors", help="Build per-task galleries for LLM eval errors.")
    llm.add_argument("--eval-dir", type=Path, required=True, help="Eval run dir containing task*/predictions.jsonl.")
    llm.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root. Defaults to <run>/outputs/review/llm_errors when eval-dir is under outputs/eval.",
    )
    llm.add_argument(
        "--task",
        action="append",
        default=[],
        help="Task name to include. Can be repeated. Defaults to task1..task5.",
    )
    llm.add_argument("--no-copy-images", action="store_true", help="Reference source images instead of copying them.")

    yolo = subparsers.add_parser("yolo-errors", help="Build YOLO detection error reviewer gallery.")
    yolo.add_argument("--model", type=Path, required=True, help="Path to YOLO model weights.")
    yolo.add_argument("--data", type=Path, required=True, help="Path to YOLO dataset yaml.")
    yolo.add_argument("--output-root", type=Path, required=True, help="Output root for review gallery.")
    yolo.add_argument("--split", type=str, default="val", choices=["train", "val"], help="YOLO dataset split.")
    yolo.add_argument("--conf", type=float, default=0.25, help="Prediction confidence threshold.")
    yolo.add_argument("--iou", type=float, default=0.5, help="IoU threshold for GT/pred matching.")
    yolo.add_argument("--imgsz", type=int, default=1280, help="Prediction image size.")
    yolo.add_argument("--device", type=str, default="0", help="CUDA device id or cpu.")
    yolo.add_argument("--max-images", type=int, default=0, help="Optional max images to inspect.")

    package_graph = subparsers.add_parser(
        "package-graph",
        help="Build package graph review pages with persisted issue notes.",
    )
    package_graph.add_argument("--risk-report", type=Path, required=True, help="risk_report.jsonl from graph gallery.")
    package_graph.add_argument("--output-root", type=Path, required=True, help="Output root for package graph review.")
    package_graph.add_argument("--fullflow-root", type=Path, default=None, help="FPK_PJ_fullflow root.")
    package_graph.add_argument("--run-id", type=str, default=None, help="Run id stored in notes.json.")
    package_graph.add_argument(
        "--split-by",
        choices=["view", "count"],
        default="view",
        help="How to split package graph pages. Defaults to view.",
    )
    package_graph.add_argument("--pages", type=int, default=5, help="Number of pages when --split-by count.")

    multiview = subparsers.add_parser("multiview", help="Build unified multiview layers reviewer gallery.")
    multiview.add_argument("--multiview-root", type=Path, required=True, help="outputs/multiview directory.")
    multiview.add_argument("--output-root", type=Path, required=True, help="Output root for multiview review.")
    multiview.add_argument("--fullflow-root", type=Path, default=None, help="FPK_PJ_fullflow root.")
    multiview.add_argument("--run-id", type=str, default=None, help="Run id stored in notes.json.")

    gt_alignment = subparsers.add_parser("gt-alignment", help="Build ScanResultFormat alignment reviewer gallery.")
    gt_alignment.add_argument("--alignment-root", type=Path, required=True, help="outputs/eval/gt_alignment directory.")
    gt_alignment.add_argument("--output-root", type=Path, required=True, help="Output root for GT alignment review.")
    gt_alignment.add_argument("--fullflow-root", type=Path, default=None, help="FPK_PJ_fullflow root.")
    gt_alignment.add_argument("--run-id", type=str, default=None, help="Run id stored in notes.json.")

    final_comparison = subparsers.add_parser(
        "final-comparison",
        help="Build all-part ScanResultFormat/final graph comparison reviewer gallery.",
    )
    final_comparison.add_argument("--alignment-root", type=Path, required=True, help="outputs/eval/gt_alignment directory.")
    final_comparison.add_argument("--output-root", type=Path, required=True, help="Output root for final comparison review.")
    final_comparison.add_argument("--fullflow-root", type=Path, default=None, help="FPK_PJ_fullflow root.")
    final_comparison.add_argument("--run-id", type=str, default=None, help="Run id stored in notes.json.")
    return parser.parse_args()


def default_llm_output_root(eval_dir: Path) -> Path:
    resolved = eval_dir.resolve()
    # Expected shape: runs/<run_id>/outputs/eval/<eval_run_id>
    if resolved.parent.name == "eval" and resolved.parent.parent.name == "outputs":
        return resolved.parent.parent / "review" / "llm_errors"
    return resolved.parent / "review_llm_errors"


def main() -> None:
    args = parse_args()
    if args.command == "llm-errors":
        eval_dir = args.eval_dir.resolve()
        output_root = args.output_root.resolve() if args.output_root else default_llm_output_root(eval_dir)
        result: dict[str, Any] = build_llm_error_review(
            eval_dir,
            output_root,
            tasks=tuple(args.task) if args.task else DEFAULT_TASKS,
            copy_images=not args.no_copy_images,
        )
    elif args.command == "yolo-errors":
        result = build_yolo_error_review(
            model_path=args.model.resolve(),
            data_yaml=args.data.resolve(),
            output_root=args.output_root.resolve(),
            split=args.split,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            max_images=args.max_images,
        )
    elif args.command == "package-graph":
        result = build_package_graph_review(
            risk_report_path=args.risk_report.resolve(),
            output_root=args.output_root.resolve(),
            fullflow_root=args.fullflow_root.resolve() if args.fullflow_root else None,
            run_id=args.run_id,
            page_count=args.pages,
            split_by=args.split_by,
        )
    elif args.command == "multiview":
        result = build_multiview_review(
            multiview_root=args.multiview_root.resolve(),
            output_root=args.output_root.resolve(),
            fullflow_root=args.fullflow_root.resolve() if args.fullflow_root else None,
            run_id=args.run_id,
        )
    elif args.command == "gt-alignment":
        result = build_gt_alignment_review(
            alignment_root=args.alignment_root.resolve(),
            output_root=args.output_root.resolve(),
            fullflow_root=args.fullflow_root.resolve() if args.fullflow_root else None,
            run_id=args.run_id,
        )
    elif args.command == "final-comparison":
        result = build_final_comparison_review(
            alignment_root=args.alignment_root.resolve(),
            output_root=args.output_root.resolve(),
            fullflow_root=args.fullflow_root.resolve() if args.fullflow_root else None,
            run_id=args.run_id,
        )
    else:  # pragma: no cover
        raise ValueError(f"Unknown command: {args.command}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
