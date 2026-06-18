from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path
import unittest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "train"
    / "yolo_v4"
    / "scripts"
    / "build_dataset.py"
)
LEGACY_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "train"
    / "yolo_v4"
    / "build_yolo_dataset.py"
)


def _load_build_dataset_module():
    spec = importlib.util.spec_from_file_location("fullflow_yolo_build_dataset", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_legacy_build_dataset_module():
    spec = importlib.util.spec_from_file_location("fullflow_legacy_yolo_build_dataset", LEGACY_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {LEGACY_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class YoloV4DatasetTest(unittest.TestCase):
    def test_build_dataset_uses_part_prefixed_image_names(self) -> None:
        module = _load_build_dataset_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_root = root / "dataset_full_v4"
            output_root = root / "yolo"
            dataset_yaml = output_root / "dataset.yaml"
            for part in ["PART_A", "PART_B"]:
                extract_dir = input_root / part / "extract_image"
                extract_dir.mkdir(parents=True)
                (extract_dir / "same.png").write_bytes(b"fake image")
                record = {
                    "imagePath": "same.png",
                    "imageWidth": 100,
                    "imageHeight": 80,
                    "shapes": [
                        {
                            "label": "pad",
                            "points": [[10, 20], [30, 20], [30, 40], [10, 40]],
                        }
                    ],
                }
                (extract_dir / "same.json").write_text(
                    json.dumps(record, ensure_ascii=False),
                    encoding="utf-8",
                )

            summary = module.build_dataset(
                input_root=input_root,
                output_root=output_root,
                dataset_yaml=dataset_yaml,
                val_ratio=0.5,
                seed=42,
                copy_images=False,
            )

            image_names = sorted(path.name for path in (output_root / "images").glob("*/*"))
            label_names = sorted(path.name for path in (output_root / "labels").glob("*/*"))
            self.assertEqual(summary["written_images"], 2)
            self.assertEqual(summary["written_boxes"], 2)
            self.assertEqual(image_names, ["PART_A__same.png", "PART_B__same.png"])
            self.assertEqual(label_names, ["PART_A__same.txt", "PART_B__same.txt"])
            self.assertTrue(dataset_yaml.exists())

    def test_legacy_build_dataset_writes_yaml_in_output_root(self) -> None:
        module = _load_legacy_build_dataset_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_root = root / "dataset_full_v4"
            output_root = root / "yolo"
            extract_dir = input_root / "PART_A" / "extract_image"
            extract_dir.mkdir(parents=True)
            (extract_dir / "one.png").write_bytes(b"fake image")
            record = {
                "imagePath": "one.png",
                "imageWidth": 100,
                "imageHeight": 80,
                "shapes": [
                    {
                        "label": "pad",
                        "points": [[10, 20], [30, 20], [30, 40], [10, 40]],
                    }
                ],
            }
            (extract_dir / "one.json").write_text(
                json.dumps(record, ensure_ascii=False),
                encoding="utf-8",
            )

            summary = module.build_dataset(
                input_root=input_root,
                output_root=output_root,
                val_ratio=0.0,
                seed=42,
                copy_images=False,
            )

            dataset_yaml = output_root / "dataset.yaml"
            self.assertTrue(dataset_yaml.exists())
            self.assertEqual(summary["dataset_yaml"], str(dataset_yaml.resolve()))
            self.assertIn("legacy_dataset_yaml", summary)


if __name__ == "__main__":
    unittest.main()
