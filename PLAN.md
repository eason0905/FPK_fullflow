# FPK_PJ Fullflow 整合計劃書

## 目標

把目前散在 `dataset/`、`package_graph/`、`evaluate/`、`LlamaFactory/` 的可用成果，收斂成一個可追蹤、可重跑、可驗證的 `FPK_PJ_fullflow/` 工作區。

這版 fullflow 的定位是：

- 使用已整理好的 `dataset_full_v3`。
- 使用已完成的 table 資料，不重新從 `FPK_anaylsis` 抓表格。
- 使用已訓練好的 model / adapter，不再跑訓練。
- 已完成的功能盡量用 wrapper 或複製固定版本，不重寫。
- dataset / model / adapter 可以複製成 fullflow 內部 snapshot，讓 `FPK_PJ_fullflow/` 能獨立運作。
- 每個 stage 都保留明確 input/output，方便判斷錯誤來自方法、實作、模型預測或整合。

## 明確不在主流程內的項目

- 不跑 LlamaFactory train。
- 不建立新的 train dataset 作為主流程必要步驟。
- 不重新收集 table 圖片。
- 不重新補 table xlsx。
- 不使用 `FPK_anaylsis`。
- 不重新產生 dataset / model / adapter；只複製已完成版本作為 frozen snapshot。
- 不覆蓋人工 `kie_linking`，模型預測只寫 `predict_kie_linking`。

## 現況盤點

| 功能 | 目前位置 | 狀態 | Fullflow 策略 |
|---|---|---|---|
| dataset v3 | `real_image_process/dataset/dataset_full_v3` | 已整理 | 複製到 fullflow snapshot 作為主輸入 |
| table 資料 | `dataset_full_v3/<part>/table/` | 已整理 | 跟 dataset v3 一起複製，只讀取，不重新抓 |
| 已訓練 model / adapter | `saves/`, `LlamaFactory/saves/`, 或指定路徑 | 已完成 | 複製到 fullflow snapshot 並登記 manifest |
| LLM eval | `dataset/scripts/evaluate_llm/*.py` | 已有 | wrapper 呼叫 |
| KIE linking predict | `dataset/scripts/predict_kie_linking_task345.py` | 已有 | wrapper 呼叫 |
| num_group 文字解析 | `dataset/scripts/origin_dataset_preprocess/split_num_group_text.py` | 已有 | copy stable 或 wrapper import |
| package graph reconstruction | `package_graph/` | 已有 | wrapper 呼叫，後續 copy stable |
| review gallery | `package_graph/cli/make_review_gallery.py` | 已有 | wrapper 呼叫 |
| table lookup diagnosis | `package_graph/cli/diagnose_table_lookup_missing.py`, `make_table_lookup_missing_gallery.py` | 已有 | optional diagnosis，只讀 v3 |
| known data issue log | `package_graph/review_known_data_issues.jsonl` | 已有 | copy 到 manifests 管理 |

## 目標資料夾架構

```text
real_image_process/FPK_PJ_fullflow/
  PLAN.md
  README.md

  configs/
    dataset/
      dataset_full_v3.yaml
    model/
      trained_adapter.yaml
    eval/
      eval_v3_nothinking.yaml
    reconstruction/
      reconstruction_v3.yaml
    gallery/
      top_bottom_land_review.yaml

  fullflow/
    __init__.py
    cli.py
    paths.py
    run_context.py
    stages/
      check_inputs.py
      eval_model.py
      predict_kie.py
      reconstruct_graph.py
      make_gallery.py
      diagnose.py

  assets/
    datasets/
      dataset_full_v3/
    models/
      <base_model_name>/
    adapters/
      <trained_adapter_name>/

  copied/
    README.md
    split_num_group_text.py
    predict_kie_linking_task345.py
    package_graph/
      ...

  manifests/
    known_data_issues.jsonl
    table_lookup_review_log.jsonl
    dataset_versions.json
    model_versions.json

  runs/
    <run_id>/
      run_config.json
      stage_status.json
      commands.sh
      logs/
      outputs/
        eval/
        predictions/
        reconstruction/
        visualization/
        review/
        diagnosis/

  reports/
    latest_summary.md
```

### 設計說明

- `configs/`：只放可讀、可 diff 的設定，不放大檔。
- `fullflow/`：新的薄 wrapper 層，負責串接，不重寫演算法。
- `assets/`：放可獨立運作所需的 frozen dataset / model / adapter snapshot。
- `copied/`：需要固定版本的既有功能放這裡，避免原目錄後續修改導致 fullflow 不可重現。
- `manifests/`：人工判斷與資料版本紀錄集中放這裡。
- `runs/`：所有 fullflow 輸出集中在單一 run id 下面。
- fullflow 執行時以 `assets/` 內的 snapshot 為主；外部路徑只作為建立 snapshot 的來源。

## Fullflow Stage 定義

### Stage 0：Input Check / Version Register

目的：

- 登記本次 run 使用的 dataset、model / adapter、known issue log。
- 若尚未建立 snapshot，將已完成的 dataset/model/adapter 複製到 `assets/`。
- 檢查必要輸入是否存在。

輸入：

- `source_dataset_root`: 預設 `real_image_process/dataset/dataset_full_v3`
- `source_adapter_path` 或 `source_model_path`
- `asset_dataset_root`: 預設 `real_image_process/FPK_PJ_fullflow/assets/datasets/dataset_full_v3`
- `asset_adapter_path` 或 `asset_model_path`
- `known_data_issues.jsonl`
- optional: eval split / dataset json

輸出：

- `runs/<run_id>/run_config.json`
- `runs/<run_id>/stage_status.json`
- `manifests/dataset_versions.json`
- `manifests/model_versions.json`
- `assets/datasets/dataset_full_v3/`
- `assets/models/` 或 `assets/adapters/`

驗證：

- source dataset root 存在。
- asset dataset root 存在，且 part 數量與 source 對齊。
- part folder 底下有 `extract_image/info.json`。
- `dataset_full_v3/<part>/table/` 若存在，只做讀取檢查，不重新生成。
- source model / adapter path 存在。
- asset model / adapter path 存在。

### Stage 1：Model Eval（可選，但建議每次記錄）

目的：

- 使用已訓練好的 model / adapter 評估 task1-task5。
- 不做訓練，不改模型。

沿用：

- `dataset/scripts/evaluate_llm/validate_real_v1.py`
- `dataset/scripts/evaluate_llm/run_qwen_eval.py`

輸出：

- `runs/<run_id>/outputs/eval/task*/summary.json`
- `runs/<run_id>/outputs/eval/overall_summary.json`
- 原腳本實際輸出位置可記在 manifest。

驗證：

- `overall_summary.json` 存在。
- task1/task3 clean exact match。
- task2 OCR exact / IoU。
- task4 target IoU / label match。
- task5 anchor exact。
- failed samples 可回查。

### Stage 2：KIE Linking Predict

目的：

- 使用已訓練好的 model / adapter，對指定 view 的 num_group 做 task3/4/5 預測。
- 寫入 `predict_kie_linking`，不覆蓋人工 `kie_linking`。

沿用：

- `dataset/scripts/predict_kie_linking_task345.py`

預設 view：

- reconstruction 主流程：`top,bottom,land`
- optional 擴充：`land_detail,front,side,lead`

輸出：

- `runs/<run_id>/outputs/predictions/predict_kie_summary.json`
- failed / skipped list。
- 若做 writeback，要記錄被修改的 annotation list。

驗證：

- 已有人標 `kie_linking` 的資料不覆蓋。
- 只新增或更新 `predict_kie_linking`。
- 每個 view 的 processed / skipped / failed count。
- 空 linking、table lookup missing、多料號 table 的 skip reason 可追蹤。

### Stage 3：Package Graph Reconstruction

目的：

- 將 YOLO/標註物件、num_group、`kie_linking` 或 `predict_kie_linking`、table lookup 結合成 package graph。

沿用：

- `package_graph/cli/run_reconstruction_and_visualize.py`
- `package_graph/orchestration/reconstruction.py`
- `package_graph/stages/*.py`

預設設定：

- input: `real_image_process/FPK_PJ_fullflow/assets/datasets/dataset_full_v3`
- `--skip-empty-dimensions`
- visualization layout: `split_vertical`
- 第一版以 `top,bottom,land` 為算法驗證主範圍。

輸出：

- `runs/<run_id>/outputs/reconstruction/summary.json`
- `runs/<run_id>/outputs/reconstruction/graphs/`
- `runs/<run_id>/outputs/visualization/`
- 若實際仍寫在 `package_graph/outputs/`，fullflow run 內要記錄實際路徑。

驗證：

- processed / failed。
- constraint accepted rate。
- residual p95。
- table_lookup_missing count。
- by view 統計。

### Stage 4：Review Gallery

目的：

- 只看算法驗證目標範圍。
- 排除 known issue 與 table_lookup_missing。
- 按 risk score 排序人工檢查。

沿用：

- `package_graph/cli/make_review_gallery.py`

預設 gallery 篩選：

```bash
--include-view top,bottom,land
--exclude-known-issues real_image_process/FPK_PJ_fullflow/manifests/known_data_issues.jsonl
--exclude-value-source table_lookup_missing
```

輸出：

- `runs/<run_id>/outputs/review/index.html`
- `runs/<run_id>/outputs/review/risk_report.jsonl`
- `runs/<run_id>/outputs/review/summary.json`

驗證：

- high risk count。
- medium risk count。
- 每輪新增 known issue 必須有 reason。
- 每個 algorithm patch 必須有 toy/regression test。

### Stage 5：Diagnosis（可選）

目的：

- 只針對已整理好的 `dataset_full_v3` 做診斷。
- 不重新抓 table，不呼叫 `FPK_anaylsis`。

可用診斷：

- table lookup missing reason。
- known issue coverage。
- view distribution。
- failed reconstruction list。

沿用：

- `package_graph/cli/diagnose_table_lookup_missing.py`
- `package_graph/cli/make_table_lookup_missing_gallery.py`

輸出：

- `runs/<run_id>/outputs/diagnosis/*.json`
- `runs/<run_id>/outputs/diagnosis/*.csv`
- optional diagnosis gallery。

## 單一入口 CLI 設計

建議新增：

```bash
python -m real_image_process.FPK_PJ_fullflow.fullflow.cli <command> [args]
```

子命令：

```text
init-run
check-inputs
eval
predict-kie
reconstruct
gallery
diagnose
run-review
run-all
```

不提供主流程命令：

```text
train
audit-table-from-fpk-analysis
collect-table
fill-table-xlsx
```

第一版不需要重寫各 stage，只做 wrapper：

```python
subprocess.run([...existing_script..., "--input", config.asset_dataset_root])
```

每個 wrapper 必須寫：

- 執行 command。
- start/end time。
- return code。
- stdout/stderr log path。
- output path。
- summary path。

## Copy vs Wrapper 策略

### 第一階段：Wrapper 優先

適合 wrapper 的既有功能：

- eval。
- KIE predict。
- reconstruction。
- visualization。
- review gallery。
- diagnosis。

優點：

- 低風險。
- 不破壞現有工作目錄。
- 很快能串起 fullflow。

### 第二階段：Copy stable function

適合 copy 的功能：

- `split_num_group_text.py`
- `predict_kie_linking_task345.py`
- table lookup helper。
- reconstruction 已穩定版本。

目的：

- 固定 fullflow 版本。
- 讓 fullflow 不受原本開發版變動影響。

### 第三階段：抽純函式共用

等 fullflow 跑穩後再做：

- 把 duplicated parser / table lookup / metrics 抽成共用 library。
- 不在第一版做，避免重構風險。

## Run Metadata 規格

每次 fullflow run 產生：

```json
{
  "run_id": "20260524_v3_fullflow_001",
  "source_dataset_root": "real_image_process/dataset/dataset_full_v3",
  "asset_dataset_root": "real_image_process/FPK_PJ_fullflow/assets/datasets/dataset_full_v3",
  "source_model_path": "...",
  "asset_model_path": "real_image_process/FPK_PJ_fullflow/assets/models/...",
  "source_adapter_path": "...",
  "asset_adapter_path": "real_image_process/FPK_PJ_fullflow/assets/adapters/...",
  "use_existing_table": true,
  "use_asset_snapshot": true,
  "uses_fpk_analysis": false,
  "run_training": false,
  "stages": {
    "check_inputs": {"status": "success", "output": "..."},
    "eval": {"status": "success", "output": "..."},
    "predict_kie": {"status": "success", "output": "..."},
    "reconstruction": {"status": "success", "output": "..."},
    "gallery": {"status": "success", "output": "..."},
    "diagnosis": {"status": "skipped", "output": null}
  }
}
```

## 驗證門檻

### Stage-level gate

| Stage | 必須檢查 |
|---|---|
| check inputs | source 與 asset snapshot 的 dataset v3、model/adapter、known issue log 存在 |
| eval | `overall_summary.json` 存在 |
| predict KIE | 不覆蓋人工標註，只寫 `predict_kie_linking` |
| reconstruction | failed = 0 或列出 failed |
| gallery | high risk / medium risk 統計 |
| diagnosis | reason 分類可解釋 |

### Algorithm patch gate

每個後處理算法修正必須：

1. 先單張重現。
2. 找到 failing stage。
3. 加 toy/regression test。
4. 單張 probe 通過。
5. full run 通過。
6. gallery risk 不惡化。

## 第一版實作順序

### P0：建立 fullflow skeleton

- 建立 `README.md`。
- 建立 `configs/`、`fullflow/`、`manifests/`、`assets/`、`runs/`。
- 實作 `paths.py` 與 `run_context.py`。
- 實作 `cli.py init-run`。
- 實作 snapshot copy：把 `dataset_full_v3`、已訓練 model / adapter 複製到 `assets/`。

### P1：包 reconstruction/review

- `reconstruct` wrapper 呼叫 `run_reconstruction_and_visualize.py`。
- `gallery` wrapper 呼叫 `make_review_gallery.py`。
- 把 known issue log 複製到 `manifests/known_data_issues.jsonl`。
- 先達成一鍵重跑目前 v3 review gallery。

### P2：包 eval

- `eval` wrapper 呼叫既有 eval script。
- 明確指定已訓練好的 adapter/model。
- 不產生 train command。

### P3：包 predict-kie

- `predict-kie` wrapper。
- 明確分開 `GT kie_linking` 與 `predict_kie_linking` 模式。
- 預設不覆蓋 dataset；若要 writeback，必須顯式 flag。

### P4：包 diagnosis

- `diagnose` wrapper。
- 只讀 dataset v3 既有 table。
- 不呼叫 `FPK_anaylsis`。
- 不重新補 xlsx。

### P5：run-all

依序執行：

```text
init-run
check-inputs
eval
predict-kie
reconstruct
gallery
diagnose
```

長時間或會修改資料的任務預設需要顯式 flag：

```bash
--run-eval
--run-predict-writeback
```

沒有 `--run-train`，因為訓練不屬於此版 fullflow。

## 建議第一個可交付版本

第一個版本先不要碰資料修改，目標是：

```bash
python -m real_image_process.FPK_PJ_fullflow.fullflow.cli run-review \
  --source-dataset-root real_image_process/dataset/dataset_full_v3 \
  --asset-dataset-root real_image_process/FPK_PJ_fullflow/assets/datasets/dataset_full_v3 \
  --known-issues real_image_process/FPK_PJ_fullflow/manifests/known_data_issues.jsonl \
  --run-id 20260524_v3_review
```

它要完成：

1. input check。
2. reconstruction。
3. split_vertical visualization。
4. review gallery。
5. risk summary。
6. 產生可點開的 `index.html` 路徑。

第二個版本再接：

```bash
python -m real_image_process.FPK_PJ_fullflow.fullflow.cli run-all \
  --source-dataset-root real_image_process/dataset/dataset_full_v3 \
  --asset-dataset-root real_image_process/FPK_PJ_fullflow/assets/datasets/dataset_full_v3 \
  --source-adapter-path <trained_adapter_path> \
  --asset-adapter-path real_image_process/FPK_PJ_fullflow/assets/adapters/<trained_adapter_name> \
  --run-eval
```

## 目前不做的事

- 不重新訓練。
- 不重新 export train dataset 作為主流程必需品。
- 不重新抓 table。
- 不重新補 table xlsx。
- 不使用 `FPK_anaylsis`。
- 不重寫 package graph。
- 不合併所有 scripts 成單一巨型 script。
- 不把人工標註與模型預測混在同一欄位。
- 不自動修 multi-part table。
- 不依賴外部 dataset/model/adapter 作為正式封存版的唯一來源；正式版要使用 `assets/` 內的 snapshot。

## 風險與對策

| 風險 | 對策 |
|---|---|
| 原目錄還在變動，fullflow 結果不穩 | 第一版 wrapper，第二版 copy stable scripts |
| asset snapshot 太大 | 仍然複製到 `assets/`，但在 manifest 記錄來源、大小、mtime 與 checksum，避免不知道是哪一版 |
| dataset v3 table 仍可能有缺漏 | diagnosis 只讀現有 table，缺漏列成資料問題 |
| predict 寫回污染人工標註 | 只寫 `predict_kie_linking`，writeback 需顯式 flag |
| table_lookup_missing 仍影響 reconstruction | gallery 可排除 `table_lookup_missing`，另用 diagnosis 看 |
| risk score 把資料問題算成算法問題 | known issue manifest 版本化 |
| full run output 太散 | 所有 run metadata 收進 `runs/<run_id>` |

## 完成定義

第一階段完成時，應該能做到：

- 用一個 run id 重跑 v3 reconstruction + gallery。
- 不需要 training。
- 不需要 `FPK_anaylsis`。
- 不需要重新處理 table。
- `FPK_PJ_fullflow/assets/` 內有可獨立運作的 dataset/model/adapter snapshot。
- 所有輸出都能在 `FPK_PJ_fullflow/runs/<run_id>/` 找到索引。
- 優先把輸出寫進 `FPK_PJ_fullflow/runs/<run_id>/outputs/`；若沿用舊腳本輸出到 `package_graph/outputs/`，fullflow 需要同步複製或記錄 manifest。
- gallery 可直接給 URL。
- high/medium/low risk count 可在 summary 中讀到。
