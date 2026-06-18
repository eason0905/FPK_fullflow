from __future__ import annotations

from pathlib import Path

from real_image_process.FPK_PJ_fullflow.auto_improve.queue import build_auto_improve_queue

from ..run_context import RunContext


def auto_improve_run_dir(context: RunContext) -> Path:
    return context.outputs_dir / "auto_improve"


def run_auto_improve(context: RunContext, *, limit: int = 0, dry_run: bool = False) -> dict:
    output_root = auto_improve_run_dir(context)
    if dry_run:
        payload = {
            "stage": "auto_improve_queue",
            "status": "dry_run",
            "output_root": str(output_root),
            "iteration_summary_path": str(output_root / "iteration_summary.json"),
            "reviewed_cases_path": str(output_root / "reviewed_cases.jsonl"),
            "score_history_path": str(output_root / "score_history.jsonl"),
        }
        context.update_status("auto_improve", payload)
        return payload

    payload = build_auto_improve_queue(
        run_dir=context.run_dir,
        output_root=output_root,
        limit=limit,
    )
    context.update_status("auto_improve", payload)
    return payload

