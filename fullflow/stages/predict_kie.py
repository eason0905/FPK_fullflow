from __future__ import annotations

from pathlib import Path

from .. import paths
from ..run_context import RunContext, run_stage_command


def build_predict_kie_command(
    context: RunContext,
    output_dir: Path,
    *,
    write: bool = False,
    overwrite_predictions: bool = False,
    list_only: bool = False,
    limit_files: int = 0,
    limit_num_groups: int = 0,
) -> list[str]:
    config = context.config
    command = [
        str(config.python),
        str(paths.REAL_IMAGE_ROOT / "dataset" / "scripts" / "predict_kie_linking_task345.py"),
        "--dataset-root",
        str(config.asset_dataset_root),
        "--output-dir",
        str(output_dir),
        "--model-path",
        str(config.asset_model_path),
        "--adapter-path",
        str(config.asset_adapter_path),
        "--views",
        ",".join(config.predict_views),
    ]
    if write:
        command.append("--write")
    if overwrite_predictions:
        command.append("--overwrite-predictions")
    if list_only:
        command.append("--list-only")
    if limit_files > 0:
        command.extend(["--limit-files", str(limit_files)])
    if limit_num_groups > 0:
        command.extend(["--limit-num-groups", str(limit_num_groups)])
    return command


def run_predict_kie(
    context: RunContext,
    *,
    write: bool = False,
    overwrite_predictions: bool = False,
    list_only: bool = False,
    limit_files: int = 0,
    limit_num_groups: int = 0,
    dry_run: bool = False,
) -> dict:
    output_dir = context.outputs_dir / "predictions"
    predictions_path = output_dir / "predictions.jsonl"
    command = build_predict_kie_command(
        context,
        output_dir,
        write=write,
        overwrite_predictions=overwrite_predictions,
        list_only=list_only,
        limit_files=limit_files,
        limit_num_groups=limit_num_groups,
    )
    expected = {
        "output_dir": str(output_dir),
        "target_summary": str(output_dir / "target_summary.json"),
    }
    if not list_only:
        expected["run_summary"] = str(output_dir / "run_summary.json")
        expected["predictions"] = str(predictions_path)
    if not dry_run and not list_only:
        ensure_prediction_jsonl_artifact(predictions_path)
    return run_stage_command(context, "predict_kie", command, expected_outputs=expected, dry_run=dry_run)


def ensure_prediction_jsonl_artifact(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")
