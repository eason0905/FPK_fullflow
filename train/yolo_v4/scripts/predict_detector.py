from __future__ import annotations

import argparse
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_ROOT = SCRIPT_DIR.parent
FULLFLOW_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_MODEL = FULLFLOW_ROOT / "assets" / "models" / "yolo" / "v4" / "best.pt"
DEFAULT_PROJECT = TRAIN_ROOT / "runs" / "predictions"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO detection on one image or a folder.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to a YOLO .pt model.")
    parser.add_argument("--source", type=Path, required=True, help="Image path or directory.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size.")
    parser.add_argument("--device", type=str, default="0", help="CUDA device id or cpu.")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT, help="Prediction output directory.")
    parser.add_argument("--name", type=str, default="lead_outline_package_pad_numgroup", help="Run name.")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "ultralytics is not installed. Run `python3 -m pip install --user ultralytics` first."
        ) from exc

    model = YOLO(str(args.model.resolve()))
    results = model.predict(
        source=str(args.source.resolve()),
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        project=str(args.project.resolve()),
        name=args.name,
        save=True,
        save_txt=True,
        save_conf=True,
    )

    summary = []
    for result in results:
        boxes = []
        names = result.names
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.tolist()
            confs = result.boxes.conf.tolist()
            classes = result.boxes.cls.tolist()
            for box, conf, cls_id in zip(xyxy, confs, classes):
                boxes.append(
                    {
                        "label": names[int(cls_id)],
                        "confidence": round(float(conf), 6),
                        "xyxy": [round(float(v), 3) for v in box],
                    }
                )
        summary.append({"image": str(result.path), "detections": boxes})

    output_dir = args.project.resolve() / args.name
    (output_dir / "predictions.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "images": len(summary)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
