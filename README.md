# FPK_PJ Fullflow

這個資料夾是把現有 `dataset_full_v3`、已訓練 model/adapter、KIE prediction、package graph reconstruction、review gallery、table lookup diagnosis 串成單一可追蹤流程的薄 wrapper。

主流程不包含訓練、不重新抓 table、不使用 `FPK_anaylsis`。

## 一次建立獨立 snapshot

正式要讓 `FPK_PJ_fullflow/` 獨立運作時，先複製 frozen assets：

```bash
python -m real_image_process.FPK_PJ_fullflow.fullflow.cli init-run \
  --copy-assets \
  --run-id v3_fullflow_assets
```

會複製：

- `real_image_process/dataset/dataset_full_v3` -> `assets/datasets/dataset_full_v3`
- `models/Qwen3-VL-8B-Instruct` -> `assets/models/Qwen3-VL-8B-Instruct`
- v3 LoRA adapter -> `assets/adapters/real-task12345-sft-split-20260516-v3-px2400-cutoff8192`

如果只是確認會做什麼，不真的複製：

```bash
python -m real_image_process.FPK_PJ_fullflow.fullflow.cli init-run --copy-assets --dry-run
```

## 檢查輸入

檢查正式 snapshot：

```bash
python -m real_image_process.FPK_PJ_fullflow.fullflow.cli check-inputs
```

檢查來源路徑：

```bash
python -m real_image_process.FPK_PJ_fullflow.fullflow.cli check-inputs --source
```

## 一鍵產生 review gallery

```bash
python -m real_image_process.FPK_PJ_fullflow.fullflow.cli run-review \
  --run-id v3_review_001 \
  --max-items 0
```

預設會：

1. 檢查 `assets/` snapshot。
2. 跑 package graph reconstruction。
3. 用 `split_vertical` 畫圖，原圖與重建圖上下分開。
4. 建立只看 `top,bottom,land` 的 gallery。
5. 排除 known issues。
6. 排除 `table_lookup_missing`。

輸出會集中到：

```text
real_image_process/FPK_PJ_fullflow/runs/<run_id>/
  run_config.json
  stage_status.json
  commands.sh
  logs/
  outputs/
```

## KIE prediction

只列出需要預測的 target，不載入模型：

```bash
python -m real_image_process.FPK_PJ_fullflow.fullflow.cli predict-kie \
  --run-id v3_predict_probe \
  --list-only
```

真的推論但不寫回 annotation：

```bash
python -m real_image_process.FPK_PJ_fullflow.fullflow.cli predict-kie \
  --run-id v3_predict_001
```

要寫回 `predict_kie_linking` 必須明確加：

```bash
python -m real_image_process.FPK_PJ_fullflow.fullflow.cli predict-kie \
  --run-id v3_predict_write \
  --write
```

它只寫 `predict_kie_linking`，不覆蓋人工 `kie_linking`。

## Table lookup diagnosis

在 reconstruction 已跑完後：

```bash
python -m real_image_process.FPK_PJ_fullflow.fullflow.cli diagnose \
  --run-id <run_id>
```

會產生 table lookup missing gallery 與原因分類，仍然只讀 `dataset_full_v3` 內已整理好的 table。

## Dry run

確認完整命令但不執行重型 stage：

```bash
python -m real_image_process.FPK_PJ_fullflow.fullflow.cli run-review \
  --run-id dryrun_v3_review \
  --dry-run
```

每個 stage 的命令都會寫到：

```text
runs/<run_id>/commands.sh
```

