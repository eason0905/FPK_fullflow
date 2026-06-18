from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from . import paths


def default_multiview_config() -> dict[str, Any]:
    return {
        "enabled": True,
        "input_mode": "gt",
        "group_by": "part_number",
        "primary_package_pad_view": "bottom",
        "primary_land_pad_view": "land",
        "lateral_views": ["side", "front"],
        "conflict_abs_tol": 0.05,
        "conflict_rel_tol": 0.05,
        "ignore_lateral_height": True,
        "preserve_raw_view": True,
    }


def default_gt_alignment_config() -> dict[str, Any]:
    return {
        "enabled": True,
        "count_tolerance": 0,
        "bbox_rel_tol": 0.15,
        "bbox_abs_tol": 0.1,
    }


def default_llm_review_config() -> dict[str, Any]:
    return {
        "enabled": True,
        "copy_images": True,
        "tasks": [],
    }


def default_yolo_review_config() -> dict[str, Any]:
    return {
        "enabled": False,
        "model_path": "",
        "data_yaml": "",
        "split": "val",
        "conf": 0.25,
        "iou": 0.5,
        "imgsz": 1280,
        "device": "0",
        "max_images": 0,
    }


def default_scoring_config() -> dict[str, Any]:
    return {
        "enabled": True,
        "weights": {
            "iou_ic": 0.25,
            "pin_count": 0.25,
            "d_pin": 0.25,
            "iou_pin": 0.25,
        },
    }


def now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_path(value: str | Path | None) -> Path | None:
    if value is None or str(value) == "":
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (paths.PROJECT_ROOT / path).resolve()


def _path_to_str(path: Path | None) -> str | None:
    return str(path) if path is not None else None


@dataclass(frozen=True)
class FullflowConfig:
    source_dataset_root: Path = paths.DEFAULT_SOURCE_DATASET_ROOT
    asset_dataset_root: Path = paths.DEFAULT_ASSET_DATASET_ROOT
    source_model_path: Path | None = paths.DEFAULT_SOURCE_MODEL_PATH
    asset_model_path: Path | None = paths.DEFAULT_ASSET_MODEL_PATH
    source_adapter_path: Path | None = paths.DEFAULT_SOURCE_ADAPTER_PATH
    asset_adapter_path: Path | None = paths.DEFAULT_ASSET_ADAPTER_PATH
    eval_dataset_dir: Path = paths.DEFAULT_EVAL_DATASET_DIR
    known_issues_source: Path | None = paths.DEFAULT_KNOWN_ISSUES_SOURCE
    known_issues_path: Path = paths.DEFAULT_KNOWN_ISSUES_PATH
    table_review_log_source: Path | None = paths.DEFAULT_TABLE_REVIEW_LOG_SOURCE
    table_review_log_path: Path = paths.DEFAULT_TABLE_REVIEW_LOG_PATH
    include_views: tuple[str, ...] = ("top", "bottom", "land")
    predict_views: tuple[str, ...] = ("bottom", "land", "land_detail", "front", "side", "lead")
    exclude_value_sources: tuple[str, ...] = ("table_lookup_missing",)
    multiview: dict[str, Any] = field(default_factory=default_multiview_config)
    gt_alignment: dict[str, Any] = field(default_factory=default_gt_alignment_config)
    llm_review: dict[str, Any] = field(default_factory=default_llm_review_config)
    yolo_review: dict[str, Any] = field(default_factory=default_yolo_review_config)
    scoring: dict[str, Any] = field(default_factory=default_scoring_config)
    python: Path = Path(sys.executable)

    @classmethod
    def from_json(cls, path: Path) -> "FullflowConfig":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FullflowConfig":
        return cls(
            source_dataset_root=_resolve_path(payload.get("source_dataset_root")) or paths.DEFAULT_SOURCE_DATASET_ROOT,
            asset_dataset_root=_resolve_path(payload.get("asset_dataset_root")) or paths.DEFAULT_ASSET_DATASET_ROOT,
            source_model_path=_resolve_path(payload.get("source_model_path")),
            asset_model_path=_resolve_path(payload.get("asset_model_path")),
            source_adapter_path=_resolve_path(payload.get("source_adapter_path")),
            asset_adapter_path=_resolve_path(payload.get("asset_adapter_path")),
            eval_dataset_dir=_resolve_path(payload.get("eval_dataset_dir")) or paths.DEFAULT_EVAL_DATASET_DIR,
            known_issues_source=_resolve_path(payload.get("known_issues_source")),
            known_issues_path=_resolve_path(payload.get("known_issues_path")) or paths.DEFAULT_KNOWN_ISSUES_PATH,
            table_review_log_source=_resolve_path(payload.get("table_review_log_source")),
            table_review_log_path=_resolve_path(payload.get("table_review_log_path")) or paths.DEFAULT_TABLE_REVIEW_LOG_PATH,
            include_views=tuple(payload.get("include_views") or ("top", "bottom", "land")),
            predict_views=tuple(payload.get("predict_views") or ("bottom", "land")),
            exclude_value_sources=tuple(payload.get("exclude_value_sources") or ("table_lookup_missing",)),
            multiview=dict(payload.get("multiview") or default_multiview_config()),
            gt_alignment=dict(payload.get("gt_alignment") or default_gt_alignment_config()),
            llm_review=dict(payload.get("llm_review") or default_llm_review_config()),
            yolo_review=dict(payload.get("yolo_review") or default_yolo_review_config()),
            scoring=dict(payload.get("scoring") or default_scoring_config()),
            python=_resolve_path(payload.get("python")) or Path(sys.executable),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_dataset_root": str(self.source_dataset_root),
            "asset_dataset_root": str(self.asset_dataset_root),
            "source_model_path": _path_to_str(self.source_model_path),
            "asset_model_path": _path_to_str(self.asset_model_path),
            "source_adapter_path": _path_to_str(self.source_adapter_path),
            "asset_adapter_path": _path_to_str(self.asset_adapter_path),
            "eval_dataset_dir": str(self.eval_dataset_dir),
            "known_issues_source": _path_to_str(self.known_issues_source),
            "known_issues_path": str(self.known_issues_path),
            "table_review_log_source": _path_to_str(self.table_review_log_source),
            "table_review_log_path": str(self.table_review_log_path),
            "include_views": list(self.include_views),
            "predict_views": list(self.predict_views),
            "exclude_value_sources": list(self.exclude_value_sources),
            "multiview": self.multiview,
            "gt_alignment": self.gt_alignment,
            "llm_review": self.llm_review,
            "yolo_review": self.yolo_review,
            "scoring": self.scoring,
            "python": str(self.python),
        }


def load_config(config_path: Path | None = None) -> FullflowConfig:
    path = config_path or paths.DEFAULT_CONFIG_PATH
    if path.exists():
        return FullflowConfig.from_json(path)
    return FullflowConfig()


@dataclass
class RunContext:
    run_id: str
    config: FullflowConfig
    root: Path = paths.FULLFLOW_ROOT
    run_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    outputs_dir: Path = field(init=False)
    status_path: Path = field(init=False)
    commands_path: Path = field(init=False)
    config_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.run_dir = self.root / "runs" / self.run_id
        self.logs_dir = self.run_dir / "logs"
        self.outputs_dir = self.run_dir / "outputs"
        self.status_path = self.run_dir / "stage_status.json"
        self.commands_path = self.run_dir / "commands.sh"
        self.config_path = self.run_dir / "run_config.json"

    def ensure_dirs(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(self.config.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if not self.status_path.exists():
            self.status_path.write_text("{}\n", encoding="utf-8")
        if not self.commands_path.exists():
            self.commands_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n", encoding="utf-8")

    @classmethod
    def create(cls, config: FullflowConfig, run_id: str | None = None) -> "RunContext":
        context = cls(run_id=run_id or now_run_id(), config=config)
        context.ensure_dirs()
        return context

    @classmethod
    def load(cls, run_id: str) -> "RunContext":
        run_dir = paths.RUNS_ROOT / run_id
        config_path = run_dir / "run_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"run_config.json not found: {config_path}")
        context = cls(run_id=run_id, config=FullflowConfig.from_json(config_path))
        context.ensure_dirs()
        return context

    def read_status(self) -> dict[str, Any]:
        if not self.status_path.exists():
            return {}
        return json.loads(self.status_path.read_text(encoding="utf-8") or "{}")

    def update_status(self, stage: str, payload: dict[str, Any]) -> None:
        status = self.read_status()
        status[stage] = payload
        self.status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def copy_manifest_inputs(config: FullflowConfig) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name, source, target in (
        ("known_issues", config.known_issues_source, config.known_issues_path),
        ("table_review_log", config.table_review_log_source, config.table_review_log_path),
    ):
        if source is None or not source.exists():
            results[name] = {"status": "missing_source", "source": _path_to_str(source), "target": str(target)}
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        results[name] = {"status": "copied", "source": str(source), "target": str(target)}
    return results


def ensure_snapshot(config: FullflowConfig, refresh: bool = False, dry_run: bool = False) -> dict[str, Any]:
    items = {
        "dataset": (config.source_dataset_root, config.asset_dataset_root),
        "model": (config.source_model_path, config.asset_model_path),
        "adapter": (config.source_adapter_path, config.asset_adapter_path),
    }
    results: dict[str, Any] = {}
    for name, (source, target) in items.items():
        if source is None or target is None:
            results[name] = {"status": "skipped", "source": _path_to_str(source), "target": _path_to_str(target)}
            continue
        results[name] = copy_snapshot_item(name, source, target, refresh=refresh, dry_run=dry_run)
    return results


def copy_snapshot_item(name: str, source: Path, target: Path, refresh: bool = False, dry_run: bool = False) -> dict[str, Any]:
    source = source.resolve()
    target = target.resolve()
    if not source.exists():
        return {"status": "missing_source", "source": str(source), "target": str(target)}
    if target.exists() and not refresh:
        return {"status": "exists", "source": str(source), "target": str(target), **path_stats(target)}
    if dry_run:
        action = "refresh" if target.exists() and refresh else "copy"
        return {"status": f"dry_run_{action}", "source": str(source), "target": str(target), **path_stats(source)}
    if target.exists():
        _assert_inside_assets(target)
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)
    return {"status": "copied", "source": str(source), "target": str(target), **path_stats(target)}


def _assert_inside_assets(target: Path) -> None:
    assets = paths.ASSET_ROOT.resolve()
    if assets not in target.resolve().parents and target.resolve() != assets:
        raise ValueError(f"Refuse to refresh path outside assets: {target}")


def path_stats(path: Path) -> dict[str, int]:
    if not path.exists():
        return {"file_count": 0, "total_bytes": 0}
    if path.is_file():
        return {"file_count": 1, "total_bytes": path.stat().st_size}
    file_count = 0
    total_bytes = 0
    for child in path.rglob("*"):
        if child.is_file():
            file_count += 1
            total_bytes += child.stat().st_size
    return {"file_count": file_count, "total_bytes": total_bytes}


def append_command(commands_path: Path, command: Iterable[str]) -> None:
    line = " ".join(shell_quote(part) for part in command)
    with commands_path.open("a", encoding="utf-8") as file:
        file.write(line + "\n")


def shell_quote(value: str) -> str:
    if value == "":
        return "''"
    safe = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_@%+=:,./-"
    if all(char in safe for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def run_stage_command(
    context: RunContext,
    stage: str,
    command: list[str],
    expected_outputs: dict[str, str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    context.ensure_dirs()
    append_command(context.commands_path, command)
    stdout_path = context.logs_dir / f"{stage}.stdout.log"
    stderr_path = context.logs_dir / f"{stage}.stderr.log"
    metadata: dict[str, Any] = {
        "stage": stage,
        "command": command,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "expected_outputs": expected_outputs or {},
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    if dry_run:
        metadata.update({"status": "dry_run", "returncode": None, "ended_at": datetime.now().isoformat(timespec="seconds")})
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        context.update_status(stage, metadata)
        return metadata

    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=paths.PROJECT_ROOT, text=True, stdout=stdout, stderr=stderr)
    metadata.update(
        {
            "status": "success" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "ended_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    missing_outputs = missing_expected_outputs(expected_outputs or {})
    if completed.returncode == 0 and missing_outputs:
        metadata["status"] = "failed_missing_outputs"
        metadata["missing_outputs"] = missing_outputs
    context.update_status(stage, metadata)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)
    if missing_outputs:
        raise FileNotFoundError(f"Stage {stage} did not create expected outputs: {missing_outputs}")
    return metadata


def missing_expected_outputs(expected_outputs: dict[str, str]) -> list[str]:
    missing: list[str] = []
    for label, raw_path in expected_outputs.items():
        path = Path(str(raw_path))
        if not path.is_absolute():
            continue
        if not path.exists():
            missing.append(f"{label}: {path}")
    return missing


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def summarize_dataset(root: Path) -> dict[str, Any]:
    part_dirs = [path for path in root.iterdir() if path.is_dir()] if root.exists() else []
    annotation_files = list(root.glob("*/*/*.json")) if root.exists() else []
    table_dirs = list(root.glob("*/table")) if root.exists() else []
    scan_result_files = list(root.glob("*/ScanResultFormat.txt")) if root.exists() else []
    return {
        "root": str(root),
        "exists": root.exists(),
        "part_count": len(part_dirs),
        "annotation_json_count": len(annotation_files),
        "table_dir_count": len(table_dirs),
        "scan_result_format_count": len(scan_result_files),
    }


def fail_if_missing(paths_to_check: Iterable[tuple[str, Path | None]]) -> list[str]:
    missing = []
    for label, path in paths_to_check:
        if path is None or not path.exists():
            missing.append(f"{label}: {path}")
    return missing
