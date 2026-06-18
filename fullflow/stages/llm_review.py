from __future__ import annotations

from pathlib import Path
from typing import Any

from ..run_context import RunContext, run_stage_command


def llm_review_dir(context: RunContext) -> Path:
    return context.outputs_dir / "review" / "llm_errors"


def latest_eval_dir(context: RunContext) -> Path | None:
    eval_root = context.outputs_dir / "eval"
    if not eval_root.exists():
        return None
    candidates = [
        path
        for path in eval_root.iterdir()
        if path.is_dir() and path.name != "gt_alignment" and (path / "overall_summary.json").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_llm_review_command(context: RunContext, *, eval_dir: Path | None = None) -> list[str]:
    config = context.config.llm_review
    resolved_eval_dir = eval_dir or latest_eval_dir(context) or (context.outputs_dir / "eval")
    command = [
        str(context.config.python),
        "-m",
        "real_image_process.FPK_PJ_fullflow.review.cli",
        "llm-errors",
        "--eval-dir",
        str(resolved_eval_dir),
        "--output-root",
        str(llm_review_dir(context)),
    ]
    tasks = config.get("tasks") or []
    for task in tasks:
        command.extend(["--task", str(task)])
    if bool(config.get("copy_images", True)) is False:
        command.append("--no-copy-images")
    return command


def run_llm_review(
    context: RunContext,
    *,
    eval_dir: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    if not bool(context.config.llm_review.get("enabled", True)):
        payload = {"status": "skipped", "reason": "llm_review.enabled is false"}
        context.update_status("llm_review", payload)
        return payload

    resolved_eval_dir = eval_dir or latest_eval_dir(context)
    if resolved_eval_dir is None and not dry_run:
        raise FileNotFoundError(f"No eval run found under {context.outputs_dir / 'eval'}")

    expected = {
        "gallery_dir": str(llm_review_dir(context)),
        "index": str(llm_review_dir(context) / "index.html"),
        "all_cases": str(llm_review_dir(context) / "all" / "cases.jsonl"),
    }
    return run_stage_command(
        context,
        "llm_review",
        build_llm_review_command(context, eval_dir=resolved_eval_dir),
        expected_outputs=expected,
        dry_run=dry_run,
    )
