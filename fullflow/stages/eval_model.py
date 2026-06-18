from __future__ import annotations

from pathlib import Path

from .. import paths
from ..run_context import RunContext, run_stage_command


def build_eval_command(context: RunContext, *, limit: int = 0, enable_thinking: bool = False) -> list[str]:
    config = context.config
    command = [
        str(config.python),
        str(paths.REAL_IMAGE_ROOT / "dataset" / "scripts" / "evaluate_llm" / "run_qwen_eval.py"),
        "--model-path",
        str(config.asset_model_path),
        "--adapter-path",
        str(config.asset_adapter_path),
        "--dataset-dir",
        str(config.eval_dataset_dir),
        "--output-root",
        str(context.outputs_dir / "eval"),
        "--dtype",
        "auto",
        "--device-map",
        "auto",
    ]
    if limit > 0:
        command.extend(["--limit", str(limit)])
    if enable_thinking:
        command.append("--enable-thinking")
    return command


def run_eval(context: RunContext, *, limit: int = 0, enable_thinking: bool = False, dry_run: bool = False) -> dict:
    command = build_eval_command(context, limit=limit, enable_thinking=enable_thinking)
    expected = {
        "output_root": str(context.outputs_dir / "eval"),
        "overall_summary": "created under output_root/<timestamp>_nothinking/overall_summary.json",
    }
    return run_stage_command(context, "eval", command, expected_outputs=expected, dry_run=dry_run)

