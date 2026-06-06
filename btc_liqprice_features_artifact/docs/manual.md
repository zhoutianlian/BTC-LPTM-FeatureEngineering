# 操作手册

## 1. 项目用途

`btc_liqprice_features_artifact` 用于把 BTC 清算量与价格输入加工为 9 个最终特征，并在完整 pipeline 中自动生成“输出特征检验与可视化报告”。

最终输出特征固定为：

- `fll_spike_kama`
- `fsl_spike_kama`
- `fll_velocity_gaussian`
- `fll_acceleration_gaussian`
- `fsl_velocity_gaussian`
- `fsl_acceleration_gaussian`
- `trend_pressure`
- `kalman_slope`
- `vol_adaptive`

特征含义与计算逻辑见：

```text
btc_liqprice_features_artifact/docs/liqprice_feature_engineering.md
btc_liqprice_features_artifact/docs/btc_liqprice_features_artifact_feature_engineering.md
```

## 2. 标准执行方式

标准执行位置是仓库上层目录，入口与 `liq_pressure_hmm` 风格一致：

```bash
python -m btc_liqprice_features_artifact.feature_liqprice
```

指定配置文件：

```bash
python -m btc_liqprice_features_artifact.feature_liqprice \
  --config btc_liqprice_features_artifact/configs/feature_liqprice.json
```

业务参数不再放在命令行中。输入、输出、字段名、计算窗口、报告开关和检验阈值均由配置文件控制。

## 3. 单独生成诊断报告

如果特征文件已经存在，只重建报告：

```bash
python -m btc_liqprice_features_artifact.feature_diagnostics \
  --config btc_liqprice_features_artifact/configs/feature_liqprice.json
```

该命令读取 `paths.output` 作为特征文件；若 `report.price_context_enabled=true` 且 `paths.input` 存在，会从原始输入中读取价格上下文，用于相关性与未来收益诊断。

## 4. 兼容旧入口

旧 CLI 仍保留，但建议只用于临时兼容：

```bash
python -m btc_liqprice_features_artifact.cli_build_features \
  --input liq_dataflow/data/features/feature_store.csv \
  --output btc_liqprice_features_artifact/output/liqprice_features.csv
```

新项目标准入口是：

```bash
python -m btc_liqprice_features_artifact.feature_liqprice
```

## 5. 默认配置文件

默认配置文件：

```text
btc_liqprice_features_artifact/configs/feature_liqprice.json
```

所有相对路径均按仓库上层目录解析。

### `paths`

- `input`：原始输入 CSV/Parquet 路径。
- `output`：最终特征输出 CSV/Parquet 路径。
- `resolved_config_output`：运行后保存的解析配置路径。
- `run_manifest`：运行清单路径。
- `log_dir`：日志目录。

### `columns`

- `time_col`：输入时间字段。
- `price_col`：输入价格字段。
- `fll_col`：多头清算量平滑序列字段。
- `fsl_col`：空头清算量平滑序列字段。
- `output_time_col`：输出特征文件中的时间字段。
- `output_features`：输出特征名单和顺序。默认包含上述 9 个最终特征。
- `aliases`：输入字段别名映射，例如 `close -> price`、`long_liq_cwt_kf -> fll_cwt_kf`。

### `feature_config`

- `bar_minutes`：输入基础 bar 周期；为 `null` 时自动从时间戳推断。
- `decision_minutes`：输出决策 bar 周期；为 `null` 时等于基础 bar。
- `liq_spike_window_min`：KAMA spike 基线窗口，单位分钟。
- `liq_kama_fast_ema`：KAMA fast EMA 参数。
- `liq_kama_slow_ema`：KAMA slow EMA 参数。
- `liq_roc_gaussian_window_min`：清算速度/加速度的单边 Gaussian 平滑窗口，单位分钟。
- `trend_pressure_mom_window_min`：`trend_pressure` 动量窗口，单位分钟。
- `trend_pressure_vol_window_min`：`trend_pressure` 背景波动窗口，单位分钟。
- `price_kalman_process_noise`：Kalman level 过程噪声。
- `price_kalman_slope_process_noise`：Kalman slope 过程噪声。
- `price_kalman_measurement_noise`：Kalman 观测噪声。
- `price_vol_adaptive_short_window_min`：自适应波动短窗，单位分钟。
- `price_vol_adaptive_long_window_min`：自适应波动长窗，单位分钟。
- `price_vol_adaptive_min_weight`：短窗波动最低权重。
- `price_vol_adaptive_sigmoid_gamma`：短长波动权重 sigmoid 映射强度。

### `report`

- `enabled`：完整 pipeline 是否自动生成报告。
- `output_dir`：报告目录。默认 `btc_liqprice_features_artifact/reports/feature_diagnostics`。
- `feature_doc_path`：重要特征文档路径，用于抽取特征名称、类别与说明。
- `generate_html`：是否生成 HTML 页面。
- `generate_summary_json`：是否生成机器可读的 `summary.json`。
- `rolling_window_minutes`：rolling 统计图与缺失率滚动窗口，单位分钟。
- `rolling_quantiles`：rolling quantile 线，默认 `[0.05, 0.95]`。
- `min_valid_count`：每个特征最低有效样本数。
- `missing_warn_ratio`：缺失/NaN 警告阈值。
- `missing_fail_ratio`：缺失/NaN 失败阈值。
- `inf_fail_count`：达到该 inf 数量即判为 FAIL。
- `zscore_threshold`：z-score 异常阈值。
- `iqr_multiplier`：IQR 异常倍数。
- `extreme_quantile_low`：低端极端分位阈值。
- `extreme_quantile_high`：高端极端分位阈值。
- `max_outlier_examples`：详情页展示的异常时间点上限。
- `constant_rtol`：近似常数判断相对容忍度。
- `constant_atol`：近似常数判断绝对容忍度。
- `long_constant_min_bars`：长时间常数、全 0 或缺失区间的最小 bars。
- `time_gap_multiplier`：异常时间跳跃阈值，按中位时间间隔倍数计算。
- `cliff_mad_multiplier`：断崖式跳变的 MAD 阈值倍数。
- `explosion_mad_multiplier`：爆炸值的 MAD 阈值倍数。
- `high_corr_threshold`：高相关特征对阈值。
- `correlation_method`：相关性方法，支持 `pearson` 或 `spearman`。
- `future_return_periods`：用于诊断的未来收益 bar 数，例如 `[1, 3, 6]`。该项只用于报告诊断，不会进入特征输出。
- `price_context_enabled`：是否在报告中合并价格上下文。

## 6. 输入要求

默认输入至少包含：

- `time`
- `price`
- `fll_cwt_kf`
- `fsl_cwt_kf`

约束：

- `time` 必须可解析为时间戳。
- `price` 应为正价格序列。
- `fll_cwt_kf` 与 `fsl_cwt_kf` 必须具有非负清算量代理语义。
- 支持 CSV 与 Parquet。

## 7. 输出结构

完整 pipeline 输出：

```text
btc_liqprice_features_artifact/output/liqprice_features.csv
btc_liqprice_features_artifact/output/resolved_feature_config.json
btc_liqprice_features_artifact/output/run_manifest.json
btc_liqprice_features_artifact/logs/latest_feature_liqprice.log
btc_liqprice_features_artifact/reports/feature_diagnostics/
```

报告目录结构：

```text
btc_liqprice_features_artifact/reports/feature_diagnostics/
├── index.html
├── assets/
│   ├── css/theme.css
│   └── js/
│       ├── plotly.min.js
│       └── table.js
├── features/
│   ├── fll_spike_kama.html
│   ├── fsl_spike_kama.html
│   └── ...
└── summary.json
```

所有 HTML 文件可本地直接打开。

## 8. 报告页面含义

### `index.html`

总览页包含：

- 报告生成时间。
- 数据时间范围。
- 总样本数。
- 重要特征总数。
- 已输出特征数。
- 文档存在但实际缺失的特征数。
- PASS / WARN / FAIL 数量。
- 所有特征的可排序、可搜索汇总表。
- 特征/价格/收益相关性热力图。
- 高相关特征对。

### `features/<feature_name>.html`

每个特征详情页包含：

- 特征说明与类别。
- PASS / WARN / FAIL 状态。
- 异常摘要。
- 统计指标表。
- 完整历史时间序列图，支持 hover、拖拽缩放、range slider、1D/1W/1M/3M/6M/1Y/ALL。
- 分布图、KDE 近似密度、箱线图。
- rolling mean / std / min / max / quantile。
- 缺失值时间分布。
- 异常点标记。
- 与价格、当前收益、未来收益的关系图；若价格上下文不存在，则在报告中说明跳过。

## 9. PASS / WARN / FAIL 规则

- `FAIL`：特征缺失、数据为空、全 NaN、非数值、inf 达到阈值、有效样本不足、缺失率超过失败阈值。
- `WARN`：缺失率超过警告阈值、近似常数列、z-score/IQR/分位数异常点、爆炸值、断崖跳变、长时间全 0/缺失/常数、重复时间戳、时间不单调、异常时间跳跃。
- `PASS`：未触发 FAIL 或 WARN。

异常值不等于错误，只表示需要人工核对。清算特征天然厚尾，因此 spike、velocity、acceleration 经常会出现 WARN。

## 10. 常见排查

- `missing_feature`：检查 `columns.output_features` 和特征计算代码是否一致。
- `Missing required columns`：检查 `columns.time_col`、`columns.price_col`、`columns.fll_col`、`columns.fsl_col`。
- `infinite_values`：检查上游输入是否有 0 基线、负值或异常极端值。
- `time_gap_anomaly`：检查原始输入是否存在断档，或 `bar_minutes` / `decision_minutes` 是否配置错误。
- `constant_feature` / `long_zero_run`：检查上游清算量是否长时间为 0，或窗口参数是否过大。
- `price_available=false`：单独报告时确认 `paths.input` 存在，且包含价格字段。

## 11. 验证方法

完整执行后检查：

```bash
ls btc_liqprice_features_artifact/reports/feature_diagnostics/index.html
ls btc_liqprice_features_artifact/reports/feature_diagnostics/features/*.html
```

查看机器可读摘要：

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("btc_liqprice_features_artifact/reports/feature_diagnostics/summary.json")
data = json.loads(p.read_text())
print(data["overview"])
PY
```

在浏览器中打开：

```text
btc_liqprice_features_artifact/reports/feature_diagnostics/index.html
```

