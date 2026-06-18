from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_ROOT = SCRIPT_DIR.parent
FULLFLOW_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_DATA = FULLFLOW_ROOT / "assets" / "datasets" / "yolo" / "v4_seed42" / "dataset.yaml"
DEFAULT_MODEL = FULLFLOW_ROOT / "assets" / "models" / "yolo" / "base" / "best.pt"
DEFAULT_PROJECT = TRAIN_ROOT / "runs"
DEFAULT_EXPORT_DIR = FULLFLOW_ROOT / "assets" / "models" / "yolo" / "v4"


def _model_source(value: str) -> str:
    path = Path(value)
    if path.exists():
        return str(path.resolve())
    return value


def _copy_if_exists(src: Path, dst: Path) -> str | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLO detector for IC package objects.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to YOLO dataset yaml.")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Ultralytics model checkpoint.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", type=str, default="0", help="CUDA device id or cpu.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT, help="Output project directory.")
    parser.add_argument("--name", type=str, default="v4_detector", help="Run name.")
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR, help="Directory to copy final weights/metrics.")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing an existing Ultralytics run directory.")
    parser.add_argument("--cache", action="store_true", help="Enable Ultralytics dataset cache.")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "ultralytics is not installed. Run `python3 -m pip install --user ultralytics` first."
        ) from exc

    model = YOLO(_model_source(args.model))
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project.resolve()),
        name=args.name,
        patience=args.patience,
        exist_ok=args.exist_ok,
        cache=args.cache,
    )
    run_dir = Path(model.trainer.save_dir).resolve()
    export_dir = args.export_dir.resolve()
    copied = {
        "best_pt": _copy_if_exists(run_dir / "weights" / "best.pt", export_dir / "best.pt"),
        "last_pt": _copy_if_exists(run_dir / "weights" / "last.pt", export_dir / "last.pt"),
        "results_csv": _copy_if_exists(run_dir / "results.csv", export_dir / "results.csv"),
        "args_yaml": _copy_if_exists(run_dir / "args.yaml", export_dir / "args.yaml"),
    }
    summary = {
        "data": str(args.data.resolve()),
        "model": _model_source(args.model),
        "run_dir": str(run_dir),
        "export_dir": str(export_dir),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "copied": copied,
    }
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
