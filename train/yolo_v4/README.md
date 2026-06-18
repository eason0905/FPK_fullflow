# YOLO v4 Training

This folder is a copied, v4-specific training entry for the package object detector.

## Structure

- `scripts/`: copied YOLO helper scripts with fullflow-local defaults.
- `configs/v4.yaml`: human-readable run config and fixed paths.
- `runs/`: Ultralytics training/evaluation outputs.
- `../../assets/datasets/yolo/v4_seed42/`: generated YOLO dataset.
- `../../assets/models/yolo/base/best.pt`: original detector checkpoint.
- `../../assets/models/yolo/v4/`: exported v4 detector weights.

The base checkpoint is copied from:

```text
real_image_process/dataset/scripts/objectdetection/runs/lead_outline_package_pad_numgroup/weights/best.pt
```

## Build dataset

```bash
bash real_image_process/FPK_PJ_fullflow/train/yolo_v4/run_build_dataset.sh
```

Output:
- `real_image_process/FPK_PJ_fullflow/assets/datasets/yolo/v4_seed42/`
- `real_image_process/FPK_PJ_fullflow/assets/datasets/yolo/v4_seed42/dataset.yaml`
- `real_image_process/FPK_PJ_fullflow/assets/datasets/yolo/v4_seed42/summary.json`

## Train

```bash
bash real_image_process/FPK_PJ_fullflow/train/yolo_v4/run_train.sh
```

Extra Ultralytics arguments can be appended, for example:

```bash
bash real_image_process/FPK_PJ_fullflow/train/yolo_v4/run_train.sh --epochs 150 --batch 8 --imgsz 1280 --device 0
```

If the active Python does not have Ultralytics, pass the Python explicitly:

```bash
PYTHON=/home/114/pohua1010/miniconda3/envs/py311/bin/python bash real_image_process/FPK_PJ_fullflow/train/yolo_v4/run_train.sh
```

The best/last weights are copied to:

```text
real_image_process/FPK_PJ_fullflow/assets/models/yolo/v4/
```
