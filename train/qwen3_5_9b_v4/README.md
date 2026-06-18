# Qwen3.5-9B v4 Training

This folder is a copied, v4-specific LlamaFactory training entry for tasks 1-5.

Base model:

```text
/home/114/pohua1010/workspace/FPK_pj/models/Qwen3.5-9B
```

Config:

```text
configs/qwen3_5_9b_sft_split_v4_px2400_cutoff8192.yaml
```

## Prepare dataset

```bash
bash real_image_process/FPK_PJ_fullflow/train/qwen3_5_9b_v4/run_prepare_dataset.sh
```

This writes:
- `real_image_process/FPK_PJ_fullflow/assets/datasets/dataset_json/v4`
- `real_image_process/FPK_PJ_fullflow/assets/datasets/dataset_json/splits/real_v4_seed42`
- `real_image_process/FPK_PJ_fullflow/assets/datasets/dataset_json/splits/real_v4_seed42/dataset_info.json`

## Train

```bash
bash real_image_process/FPK_PJ_fullflow/train/qwen3_5_9b_v4/run_train.sh
```

Equivalent direct command:

```bash
PYTHONPATH=/home/114/pohua1010/workspace/FPK_pj/LlamaFactory/src \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
llamafactory-cli train /home/114/pohua1010/workspace/FPK_pj/real_image_process/FPK_PJ_fullflow/train/qwen3_5_9b_v4/configs/qwen3_5_9b_sft_split_v4_px2400_cutoff8192.yaml
```
