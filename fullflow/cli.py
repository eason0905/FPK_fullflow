from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import paths
from .run_context import (
    FullflowConfig,
    RunContext,
    copy_manifest_inputs,
    ensure_snapshot,
    load_config,
    write_json,
)
from .stages.check_inputs import check_inputs
from .stages.auto_improve import run_auto_improve
from .stages.diagnose import run_table_lookup_diagnosis, run_table_lookup_gallery
from .stages.eval_model import run_eval
from .stages.eval_summary import run_eval_summary
from .stages.gt_alignment import run_gt_alignment
from .stages.llm_review import run_llm_review
from .stages.make_gallery import run_gallery
from .stages.multiview_integrate import run_multiview
from .stages.package_graph_overlay import run_package_graph_overlay
from .stages.predict_kie import run_predict_kie
from .stages.reconstruct_graph import run_reconstruct
from .stages.score_merge_gt import run_score_merge_gt
from .stages.yolo_review import run_yolo_review


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FPK_PJ fullflow wrappers.")
    parser.add_argument("--config", type=Path, default=paths.DEFAULT_CONFIG_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init-run", help="Create a run directory and optionally copy frozen assets.")
    init.add_argument("--run-id", default=None)
    init.add_argument("--copy-assets", action="store_true")
    init.add_argument("--refresh-assets", action="store_true")
    init.add_argument("--dry-run", action="store_true")

    check = subparsers.add_parser("check-inputs", help="Check dataset/model/adapter/manifests.")
    check.add_argument("--run-id", default=None)
    check.add_argument("--source", action="store_true", help="Check source paths instead of asset snapshot paths.")

    pred = subparsers.add_parser("predict-kie", help="Run task3/4/5 KIE linking prediction wrapper.")
    add_run_options(pred)
    pred.add_argument("--write", action="store_true", help="Write predict_kie_linking into asset annotations.")
    pred.add_argument("--overwrite-predictions", action="store_true")
    pred.add_argument("--list-only", action="store_true")
    pred.add_argument("--limit-files", type=int, default=0)
    pred.add_argument("--limit-num-groups", type=int, default=0)
    pred.add_argument("--dry-run", action="store_true")

    recon = subparsers.add_parser("reconstruct", help="Run reconstruction and visualization.")
    add_run_options(recon)
    recon.add_argument("--limit", type=int, default=0)
    recon.add_argument("--layout", choices=["overlay", "split_vertical"], default="split_vertical")
    recon.add_argument("--dry-run", action="store_true")

    gallery = subparsers.add_parser("gallery", help="Build filtered top/bottom/land review gallery.")
    add_run_options(gallery)
    gallery.add_argument("--max-items", type=int, default=0)
    gallery.add_argument("--limit", type=int, default=0)
    gallery.add_argument("--dry-run", action="store_true")

    multiview = subparsers.add_parser("multiview", help="Build per-part multiview canonical 2D graphs and reviewer.")
    add_run_options(multiview)
    multiview.add_argument("--limit", type=int, default=0)
    multiview.add_argument("--dry-run", action="store_true")

    gt_alignment = subparsers.add_parser("gt-alignment", help="Evaluate multiview graphs against ScanResultFormat GT.")
    add_run_options(gt_alignment)
    gt_alignment.add_argument("--limit", type=int, default=0)
    gt_alignment.add_argument("--dry-run", action="store_true")

    package_graph_overlay = subparsers.add_parser(
        "package-graph-overlay",
        help="Build the package graph multiview overlay/alignment/merge gallery.",
    )
    add_run_options(package_graph_overlay)
    package_graph_overlay.add_argument("--limit", type=int, default=0)
    package_graph_overlay.add_argument("--dry-run", action="store_true")

    score_merge_gt = subparsers.add_parser(
        "score-merge-gt",
        help="Score merge-stage geometry against ScanResultFormat GT and build low-score gallery.",
    )
    add_run_options(score_merge_gt)
    score_merge_gt.add_argument("--dry-run", action="store_true")

    diag = subparsers.add_parser("diagnose", help="Build table_lookup_missing gallery and reason report.")
    add_run_options(diag)
    diag.add_argument("--limit", type=int, default=0)
    diag.add_argument("--exclude-no-table", action="store_true")
    diag.add_argument("--dry-run", action="store_true")

    eval_parser = subparsers.add_parser("eval", help="Run existing Qwen eval wrapper on the configured val split.")
    add_run_options(eval_parser)
    eval_parser.add_argument("--limit", type=int, default=0)
    eval_parser.add_argument("--enable-thinking", action="store_true")
    eval_parser.add_argument("--dry-run", action="store_true")

    eval_summary = subparsers.add_parser("eval-summary", help="Summarize fullflow eval and review diagnostics.")
    add_run_options(eval_summary)
    eval_summary.add_argument("--dry-run", action="store_true")

    llm_review = subparsers.add_parser("llm-review", help="Build LLM eval error reviewer gallery.")
    add_run_options(llm_review)
    llm_review.add_argument("--dry-run", action="store_true")

    yolo_review = subparsers.add_parser("yolo-review", help="Build YOLO detection error reviewer gallery.")
    add_run_options(yolo_review)
    yolo_review.add_argument("--dry-run", action="store_true")

    auto_improve = subparsers.add_parser("auto-improve", help="Build low-score/high-risk case queue for iterative fixes.")
    add_run_options(auto_improve)
    auto_improve.add_argument("--limit", type=int, default=0)
    auto_improve.add_argument("--dry-run", action="store_true")

    review = subparsers.add_parser("run-review", help="Run check-inputs, reconstruction, gallery, and optional diagnosis.")
    add_run_options(review)
    review.add_argument("--limit", type=int, default=0)
    review.add_argument("--max-items", type=int, default=0)
    review.add_argument("--layout", choices=["overlay", "split_vertical"], default="split_vertical")
    review.add_argument("--diagnose", action="store_true")
    review.add_argument("--dry-run", action="store_true")

    all_parser = subparsers.add_parser("run-all", help="Run the full wrapper sequence without training.")
    add_run_options(all_parser)
    all_parser.add_argument("--copy-assets", action="store_true")
    all_parser.add_argument("--limit", type=int, default=0)
    all_parser.add_argument("--max-items", type=int, default=0)
    all_parser.add_argument("--write-predict-kie", action="store_true")
    all_parser.add_argument("--diagnose", action="store_true")
    all_parser.add_argument("--enable-thinking", action="store_true")
    all_parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", default=None, help="Existing run id. If omitted, create a new run.")


def context_from_args(args: argparse.Namespace) -> RunContext:
    if args.run_id:
        run_config = paths.RUNS_ROOT / args.run_id / "run_config.json"
        if run_config.exists():
            context = RunContext.load(args.run_id)
        else:
            context = RunContext.create(load_config(args.config), run_id=args.run_id)
    else:
        context = RunContext.create(load_config(args.config))
    copy_manifest_inputs(context.config)
    return context


def print_result(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def init_run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    context = RunContext.create(config, run_id=args.run_id)
    manifest_result = copy_manifest_inputs(config)
    snapshot_result = None
    if args.copy_assets or args.refresh_assets or args.dry_run:
        snapshot_result = ensure_snapshot(config, refresh=args.refresh_assets, dry_run=args.dry_run)
    payload = {
        "run_id": context.run_id,
        "run_dir": str(context.run_dir),
        "run_config": str(context.config_path),
        "manifest_copy": manifest_result,
        "snapshot": snapshot_result,
        "dry_run": args.dry_run,
    }
    context.update_status("init_run", {"status": "success", **payload})
    return payload


def run_check_inputs(args: argparse.Namespace) -> dict[str, Any]:
    if args.run_id:
        context = RunContext.load(args.run_id)
        config = context.config
        output_path = context.outputs_dir / "check_inputs" / "summary.json"
    else:
        config = load_config(args.config)
        output_path = None
    copy_manifest_inputs(config)
    summary = check_inputs(config, output_path=output_path, require_assets=not args.source)
    if args.run_id:
        context.update_status("check_inputs", {"status": "success", "output": str(output_path), **summary})
    return summary


def run_review(args: argparse.Namespace) -> dict[str, Any]:
    context = context_from_args(args)
    check_path = context.outputs_dir / "check_inputs" / "summary.json"
    check_summary = check_inputs(context.config, output_path=check_path, require_assets=not args.dry_run)
    context.update_status("check_inputs", {"status": "success", "output": str(check_path), **check_summary})
    reconstruction = run_reconstruct(context, limit=args.limit, layout=args.layout, dry_run=args.dry_run)
    multiview = run_multiview(context, limit=args.limit, dry_run=args.dry_run)
    package_graph_overlay = run_package_graph_overlay(context, limit=args.limit, dry_run=args.dry_run)
    score_merge_gt = run_score_merge_gt(context, dry_run=args.dry_run)
    gallery = run_gallery(context, max_items=args.max_items, limit=0, dry_run=args.dry_run)
    diagnosis = None
    if args.diagnose:
        table_gallery = run_table_lookup_gallery(
            context,
            include_no_table=True,
            limit=0,
            dry_run=args.dry_run,
        )
        table_reasons = run_table_lookup_diagnosis(context, dry_run=args.dry_run)
        diagnosis = {"table_lookup_gallery": table_gallery, "table_lookup_reasons": table_reasons}
    eval_summary = run_eval_summary(context, dry_run=args.dry_run)
    return {
        "run_id": context.run_id,
        "check_inputs": check_summary,
        "reconstruction": reconstruction,
        "multiview": multiview,
        "package_graph_overlay": package_graph_overlay,
        "score_merge_gt": score_merge_gt,
        "eval_summary": eval_summary,
        "gallery": gallery,
        "diagnosis": diagnosis,
    }


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    context = context_from_args(args)
    if args.copy_assets:
        copy_manifest_inputs(context.config)
        snapshot = ensure_snapshot(context.config, dry_run=args.dry_run)
        context.update_status("snapshot", {"status": "success", "result": snapshot, "dry_run": args.dry_run})
    check_path = context.outputs_dir / "check_inputs" / "summary.json"
    check_summary = check_inputs(context.config, output_path=check_path, require_assets=not args.dry_run)
    context.update_status("check_inputs", {"status": "success", "output": str(check_path), **check_summary})
    pred = run_predict_kie(
        context,
        write=args.write_predict_kie,
        limit_files=args.limit,
        dry_run=args.dry_run,
    )
    reconstruction = run_reconstruct(context, limit=args.limit, dry_run=args.dry_run)
    eval_result = run_eval(context, limit=args.limit, enable_thinking=args.enable_thinking, dry_run=args.dry_run)
    llm_review = run_llm_review(context, dry_run=args.dry_run)
    yolo_review = run_yolo_review(context, dry_run=args.dry_run)
    multiview = run_multiview(context, limit=args.limit, dry_run=args.dry_run)
    package_graph_overlay = run_package_graph_overlay(context, limit=args.limit, dry_run=args.dry_run)
    score_merge_gt = run_score_merge_gt(context, dry_run=args.dry_run)
    gallery = run_gallery(context, max_items=args.max_items, dry_run=args.dry_run)
    diagnosis = None
    if args.diagnose:
        table_gallery = run_table_lookup_gallery(context, dry_run=args.dry_run)
        table_reasons = run_table_lookup_diagnosis(context, dry_run=args.dry_run)
        diagnosis = {"table_lookup_gallery": table_gallery, "table_lookup_reasons": table_reasons}
    eval_summary = run_eval_summary(context, dry_run=args.dry_run)
    return {
        "run_id": context.run_id,
        "check_inputs": check_summary,
        "predict_kie": pred,
        "reconstruction": reconstruction,
        "eval": eval_result,
        "llm_review": llm_review,
        "yolo_review": yolo_review,
        "multiview": multiview,
        "package_graph_overlay": package_graph_overlay,
        "score_merge_gt": score_merge_gt,
        "eval_summary": eval_summary,
        "gallery": gallery,
        "diagnosis": diagnosis,
    }


def main() -> None:
    args = parse_args()
    if args.command == "init-run":
        result = init_run(args)
    elif args.command == "check-inputs":
        result = run_check_inputs(args)
    elif args.command == "predict-kie":
        result = run_predict_kie(
            context_from_args(args),
            write=args.write,
            overwrite_predictions=args.overwrite_predictions,
            list_only=args.list_only,
            limit_files=args.limit_files,
            limit_num_groups=args.limit_num_groups,
            dry_run=args.dry_run,
        )
    elif args.command == "reconstruct":
        result = run_reconstruct(context_from_args(args), limit=args.limit, layout=args.layout, dry_run=args.dry_run)
    elif args.command == "gallery":
        result = run_gallery(context_from_args(args), max_items=args.max_items, limit=args.limit, dry_run=args.dry_run)
    elif args.command == "multiview":
        result = run_multiview(context_from_args(args), limit=args.limit, dry_run=args.dry_run)
    elif args.command == "gt-alignment":
        result = run_gt_alignment(context_from_args(args), limit=args.limit, dry_run=args.dry_run)
    elif args.command == "package-graph-overlay":
        result = run_package_graph_overlay(context_from_args(args), limit=args.limit, dry_run=args.dry_run)
    elif args.command == "score-merge-gt":
        result = run_score_merge_gt(context_from_args(args), dry_run=args.dry_run)
    elif args.command == "diagnose":
        context = context_from_args(args)
        first = run_table_lookup_gallery(
            context,
            include_no_table=not args.exclude_no_table,
            limit=args.limit,
            dry_run=args.dry_run,
        )
        second = run_table_lookup_diagnosis(context, dry_run=args.dry_run)
        result = {"table_lookup_gallery": first, "table_lookup_reasons": second}
    elif args.command == "eval":
        result = run_eval(
            context_from_args(args),
            limit=args.limit,
            enable_thinking=args.enable_thinking,
            dry_run=args.dry_run,
        )
    elif args.command == "eval-summary":
        result = run_eval_summary(context_from_args(args), dry_run=args.dry_run)
    elif args.command == "llm-review":
        result = run_llm_review(context_from_args(args), dry_run=args.dry_run)
    elif args.command == "yolo-review":
        result = run_yolo_review(context_from_args(args), dry_run=args.dry_run)
    elif args.command == "auto-improve":
        result = run_auto_improve(context_from_args(args), limit=args.limit, dry_run=args.dry_run)
    elif args.command == "run-review":
        result = run_review(args)
    elif args.command == "run-all":
        result = run_all(args)
    else:  # pragma: no cover
        raise ValueError(f"Unknown command: {args.command}")
    print_result(result)


if __name__ == "__main__":
    main()
