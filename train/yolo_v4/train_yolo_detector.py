from __future__ import annotations

import argparse
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR / "dataset.yaml"
DEFAULT_PROJECT = SCRIPT_DIR / "runs"
PROJECT_ROOT = SCRIPT_DIR.parents[4]
DEFAULT_MODEL = PROJECT_ROOT / "yolo26n.pt"


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
    parser.add_argument("--name", type=str, default="lead_outline_package_pad_numgroup_v4", help="Run name.")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "ultralytics is not installed. Run `python3 -m pip install --user ultralytics` first."
        ) from exc

    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project.resolve()),
        name=args.name,
    )


if __name__ == "__main__":
    main()
