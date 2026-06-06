# liq_dataflow Manual

## 1. 项目定位

`liq_dataflow` 是纯特征工程项目。它不下载远端数据，只读取 `liq_data_download` 或本地已有 CSV，生成清洗结果、清算特征、校验报告和可视化。

推荐运行位置是仓库根目录：

```bash
python -m liq_dataflow.feature_liq_dataflow
```

主配置文件：

```text
liq_dataflow/configs/feature_engineering.json
```

可选指定配置：

```bash
python -m liq_dataflow.feature_liq_dataflow --config liq_dataflow/configs/feature_engineering.json
```

旧入口 `python -m liq_dataflow.scripts.run_feature_engineering` 仅保留兼容；正式参数应写入配置文件，不应依赖命令行传入输入/输出路径。

## 2. 目录结构

```text
liq_dataflow/
├── configs/
│   └── feature_engineering.json
├── feature_liq_dataflow.py
├── feature_engineering/
├── visualizer/
├── docs/
├── data/
│   ├── clean/
│   ├── cache/
│   ├── features/
│   └── report/
└── logs/
```

## 3. 输入数据

默认输入由配置项 `input.source_csv` 指定：

```text
../liq_data_download/data/raw/hourly/BTC_price_lld.csv
```

如果该文件不存在，则回退到 `input.fallback_csv`：

```text
data/clean/clean_data.csv
```

输入 CSV 可以是两种形态之一。

### 3.1 raw hourly bundle

默认字段：

- `time`
- `price`
- `futures_long_liquidations`
- `futures_short_liquidations`

### 3.2 clean frame

默认字段：

- `time`
- `price`
- `fll_normal`
- `fsl_normal`

字段名可以通过 `columns` 配置映射，但进入项目内部后统一规范化为上述 canonical 字段。

## 4. 主要输出

最终重要输出文件为：

```text
liq_dataflow/data/features/features_liq_dataflow.csv
```

该文件合并了原先拆开的：

- `data/features/liq_dataflow_features.csv`
- `data/features/fhmv_liq_features.csv`

新文件只保留一个方向占比字段：

- `risk_priority_number`

历史模型层别名 `RPN` 不再作为最终输出字段单独写出。

### 4.1 最终交付特征

`features_liq_dataflow.csv` 中除 `time` 和 `price` 外，以下全部是本项目重要输出特征：

- `fll_cwt_kf`
- `fsl_cwt_kf`
- `diff_ls_cwt_kf`
- `total_ls_cwt_kf`
- `risk_priority_number`
- `bin_index`
- `dominance`
- `diff_dom_ls_cwt_kf`
- `z_logTotalP`
- `z_sdom`
- `z_fll_cwt_kf`
- `z_fsl_cwt_kf`

### 4.2 中间 CSV

- `data/clean/clean_data.csv`
- `data/features/features_rpn.csv`
- `data/features/features_rpn_bin_stage.csv`
- `data/features/features_bin_stats.csv`
- `data/features/features_kmeans_stats.csv`
- `data/features/features_rpn_bin.csv`
- `data/features/feature_store.csv`

`feature_store.csv` 是宽表调试与可视化输入，不是替代最终交付文件。

### 4.3 可视化

- `data/report/rpn_dominance-latest.png`
- `data/report/rpn_dominance-latest.html`
- `data/report/rpn_features-latest.png`
- `data/report/rpn_features-latest.html`
- `data/report/feature_overview.html`
- `data/report/feature_pages/*.html`
- `data/report/plotly.min.js`

### 4.4 校验与日志

- `data/features/output_validation_report.csv`
- `data/features/output_validation_report.md`
- `data/features/output_validation_report.json`
- `logs/latest.log`
- `logs/run_history.csv`
- `logs/runs/<run_id>/pipeline.log`
- `logs/runs/<run_id>/pipeline.jsonl`
- `logs/runs/<run_id>/run_summary.md`
- `logs/runs/<run_id>/run_summary.json`

## 5. 配置参数说明

所有正式运行参数均放在 `liq_dataflow/configs/feature_engineering.json`。

### 5.1 `project`

- `name`：项目名称，仅用于运行摘要与日志说明。
- `start_time`：特征工程起始时间；早于该时间的输入记录会被过滤。

### 5.2 `input`

- `source_csv`：主输入 CSV，相对路径以 `liq_dataflow/` 为基准。
- `fallback_csv`：主输入不存在时的回退 CSV。

### 5.3 `execution`

- `build_visualizations`：是否在 pipeline 中生成 HTML/PNG 可视化。

### 5.4 `columns`

- `time_col`：输入时间字段名。
- `price_col`：输入价格字段名。
- `raw_long_liquidations_col`：raw 输入中的多头清算字段名。
- `raw_short_liquidations_col`：raw 输入中的空头清算字段名。
- `clean_long_col`：clean 输入中的多头清算字段名。
- `clean_short_col`：clean 输入中的空头清算字段名。

### 5.5 `data`

- `clean_subdir`：清洗数据输出目录。
- `cache_subdir`：平滑缓存目录。
- `features_subdir`：特征中间产物目录。
- `final_features_subdir`：最终合并特征子目录；当前为空，表示直接输出到 `data/features/`。
- `report_subdir`：报告目录。
- `feature_pages_subdir`：通用特征页面子目录。
- `clean_filename`：清洗结果文件名。
- `canonical_filename`：canonical liquidation family 文件名。
- `bin_stage_filename`：RPN 分箱逐点结果文件名。
- `bin_stats_filename`：分箱统计文件名。
- `legacy_bin_stats_filename`：兼容历史命名的分箱统计文件名。
- `dominance_filename`：dominance 规则层文件名。
- `final_features_filename`：最终合并特征文件名，当前为 `features_liq_dataflow.csv`。
- `feature_store_filename`：宽表 feature store 文件名。
- `fll_cache_filename`：多头清算平滑缓存文件名。
- `fsl_cache_filename`：空头清算平滑缓存文件名。

### 5.6 `preprocess`

- `outlier_iqr_window_days`：输入清算序列的 trailing IQR 异常值压缩窗口天数。

### 5.7 `smoothing`

- `wavelet`：wavelet trend extraction 使用的小波名称。
- `level`：wavelet 分解层数。
- `threshold_method`：wavelet threshold 方法。
- `window_size_hours`：trailing smoothing 最大窗口长度，单位小时。
- `kalman_initial_state_mean`：Kalman 初始状态均值。
- `kalman_initial_state_covariance`：Kalman 初始状态协方差。
- `kalman_observation_covariance`：Kalman 观测噪声协方差。
- `kalman_transition_covariance`：Kalman 状态转移噪声协方差。

### 5.8 `binning`

- `n_bins`：`risk_priority_number` 的有序分箱数量。
- `min_history_bars`：开始分箱前需要的最小历史 bar 数。
- `refit_every_bars`：分位数边界重拟合间隔。
- `neutral_bin`：历史不足时使用的中性分箱。
- `extreme_window_bars`：极端 RPN 识别窗口。

### 5.9 `dominance`

- `rolling_window_bars`：dominance 阈值使用的 rolling quantile 窗口。
- `rolling_min_periods`：rolling quantile 最小有效样本数。
- `reverse_window_bars`：ceiling/bottom reversal 检测窗口。
- `reverse_diff_threshold`：reversal 事件需要满足的 diff 反向变化阈值。

### 5.10 `model_features`

- `z_window_bars`：`z_logTotalP`、`z_sdom`、`z_fll_cwt_kf`、`z_fsl_cwt_kf` 的 rolling median/MAD robust z-score 窗口。

### 5.11 `output_features`

- `final_feature_columns`：最终输出文件中除 `time`、`price` 外的特征列顺序。当前该列表就是项目重要输出特征清单。

### 5.12 `visualization`

- `enabled`：是否允许生成可视化；会和 `execution.build_visualizations` 同时生效。
- `specialized_duration_months`：专题图默认近期窗口月数。
- `specialized_max_points`：专题图最大绘图点数。
- `generic_max_points`：通用特征页面最大绘图点数。
- `overview_filename`：通用特征总览 HTML 文件名。
- `catalog_filename`：通用特征目录 CSV 文件名。
- `dominant_html_filename`：dominance 专题 HTML 文件名。
- `dominant_png_filename`：dominance 专题 PNG 文件名。
- `features_html_filename`：特征专题 HTML 文件名。
- `features_png_filename`：特征专题 PNG 文件名。

### 5.13 `validation`

- `enabled`：是否启用输出校验配置开关。
- `raise_on_error`：阻塞性校验失败时是否抛出异常。
- `numeric_tolerance`：数值恒等式与边界校验容忍度。
- `max_model_nan_ratio`：标准化特征允许的最大 NaN 比例。
- `report_csv_filename`：校验 CSV 报告文件名。
- `report_md_filename`：校验 Markdown 报告文件名。
- `report_json_filename`：校验 JSON 报告文件名。

## 6. 标准工作顺序

正式流程：

```bash
python -m liq_data_download.scripts.run_data_download
python -m liq_dataflow.feature_liq_dataflow
```

本地已有 CSV 时，修改：

```text
liq_dataflow/configs/feature_engineering.json -> input.source_csv
```

然后运行：

```bash
python -m liq_dataflow.feature_liq_dataflow
```

## 7. 常见问题

### 7.1 输入文件不存在

检查：

- `configs/feature_engineering.json` 中的 `input.source_csv`
- 是否已先运行 `liq_data_download`
- 或是否需要把本地 CSV 路径写入 `input.source_csv`

### 7.2 HTML 打不开或太慢

先检查 PNG：

- `data/report/rpn_dominance-latest.png`
- `data/report/rpn_features-latest.png`

PNG 是静态审阅基准图；HTML 支持 hover、拖拽缩放、scroll zoom、range slider 与窗口内 Y 轴自适应。

### 7.3 校验失败

优先查看：

- `data/features/output_validation_report.md`
- `logs/latest.log`
