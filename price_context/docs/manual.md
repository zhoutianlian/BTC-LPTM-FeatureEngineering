# price_context 使用手册

## 1. 项目简介

`price_context` 是一个基于 OHLC 价格数据的价格上下文特征工程项目。项目读取价格 bar，输出 past return、realized volatility、range、trend、vol of vol、jump proxy 和 price quality 特征。

核心原则：

- 所有特征默认只使用当前时点及历史数据。
- 输入、输出、字段映射、窗口、阈值、权重、诊断和命令行打印行为都放在配置文件内。
- 推荐从仓库根目录执行，与 `liq_pressure_hmm` 的 `python -m liq_pressure_hmm.feature_plie_HMM` 方式保持一致。

## 2. 环境要求

推荐 Python 3.10+。依赖见 `price_context/requirements.txt`：

```bash
pip install -r price_context/requirements.txt
```

核心依赖：

- `pandas`
- `numpy`
- `PyYAML`
- `pytest`
- `numba`：仅用于可选 rolling MAD 加速；默认配置使用 IQR robust scale。

## 3. 推荐运行方式

在仓库根目录执行：

```bash
python -m price_context.feature_price_context
```

指定配置文件：

```bash
python -m price_context.feature_price_context --config price_context/configs/feature_price_context.json
```

默认主配置文件：

```text
price_context/configs/feature_price_context.json
```

兼容配置文件：

```text
price_context/config.yaml
```

除 `--config` 用于选择配置文件外，运行参数不再通过命令行传入。例如是否打印字段列表，由 `execution.print_columns` 控制。

兼容旧入口仍可使用，但不推荐作为主执行方式：

```bash
python -m price_context.src.main --config price_context/configs/feature_price_context.json
```

## 4. 执行流程

主入口 `price_context/feature_price_context.py` 的执行顺序：

1. 读取 JSON/YAML 配置。
2. 根据 `paths.project_root` 或配置文件位置解析相对路径。
3. 读取 `input.file_path` 指向的 OHLC CSV。
4. 按 `input` 字段映射标准化为 `time/open/high/low/close`。
5. 按 `data` 参数排序、去重、校验价格和 OHLC 关系。
6. 按 `windows`、`quality`、`realized_vol`、`range`、`trend`、`jump` 计算特征。
7. 按 `output` 写出 CSV 和可选 zip。
8. 按 `report` 生成诊断报告。
9. 按 `execution` 打印运行摘要、校验报告和字段列表。

## 5. 路径解析规则

相对路径默认相对于项目根目录 `price_context/` 解析。

- 当配置文件位于 `price_context/configs/` 时，项目根目录自动推断为 `price_context/`。
- 当配置文件位于 `price_context/config.yaml` 时，项目根目录也是 `price_context/`。
- 如果设置 `paths.project_root`，则所有相对路径改为相对于该目录解析。

默认输入：

```text
../liq_data_download/data/raw/intraday/ohlc.csv
```

在默认项目根目录 `price_context/` 下解析后，实际指向仓库内：

```text
liq_data_download/data/raw/intraday/ohlc.csv
```

## 6. 输入文件格式

输入 CSV 必须包含配置中指定的时间列和 OHLC 列。默认字段如下：

| 字段 | 含义 | 要求 |
|---|---|---|
| `time` | bar 时间 | 可解析为 datetime |
| `open` | 开盘价 | 数值、正数 |
| `high` | 最高价 | 数值、正数，且满足 OHLC 关系 |
| `low` | 最低价 | 数值、正数，且满足 OHLC 关系 |
| `close` | 收盘价 | 数值、正数 |

如果源文件字段不同，只修改 `input.*_column`，不要改代码。

## 7. 配置参数完整说明

### 7.1 paths

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `paths.project_root` | `null` | 相对路径解析根目录。`null` 表示自动推断为 `price_context/`。 |

### 7.2 input

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `input.file_path` | `../liq_data_download/data/raw/intraday/ohlc.csv` | 输入 OHLC CSV 路径，可为绝对路径或相对项目根目录路径。 |
| `input.time_column` | `time` | 原始时间字段名。 |
| `input.open_column` | `open` | 原始开盘价字段名。 |
| `input.high_column` | `high` | 原始最高价字段名。 |
| `input.low_column` | `low` | 原始最低价字段名。 |
| `input.close_column` | `close` | 原始收盘价字段名。 |
| `input.datetime_format` | `null` | 时间解析格式。`null` 表示由 pandas 自动解析。 |
| `input.timezone` | `null` | 目标时区。非空时 naive 时间会 localize，带时区时间会 convert。 |

### 7.3 data

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `data.bar_minutes` | `10` | 输入 bar 间隔分钟数，用于窗口换算、缺口检测和固定 bar 计算。 |
| `data.sort_by_time` | `true` | 是否按时间升序排序。 |
| `data.drop_duplicate_time` | `true` | 是否删除重复时间戳。 |
| `data.duplicate_keep` | `last` | 重复时间戳保留方式，传给 pandas `drop_duplicates(keep=...)`。 |
| `data.validate_ohlc` | `true` | 是否校验 `high/low/open/close` 关系。 |
| `data.fail_on_invalid_price` | `true` | 发现非正价格时是否抛错。 |
| `data.fail_on_ohlc_inconsistency` | `true` | 发现 OHLC 关系不一致时是否抛错。 |

### 7.4 windows

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `windows.return_windows` | `["1h","3h","6h","12h","24h"]` | past return 输出窗口。 |
| `windows.core_windows` | `["1h","6h","24h"]` | realized vol、range、trend、jump 的核心窗口。 |
| `windows.vol_of_vol_windows` | `["6h","24h","48h"]` | vol of vol 输出窗口。 |
| `windows.quality_windows` | `["1h","6h","24h","48h"]` | price quality 输出窗口。 |

窗口字符串使用 pandas 时间窗口格式，例如 `20min`、`1h`、`7d`。默认 10m bar 下，`1h=6 bars`、`6h=36 bars`、`24h=144 bars`。

### 7.5 quality

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `quality.min_obs_ratio` | `0.8` | rolling 窗口有效观测比例下限。 |
| `quality.gap_tolerance_minutes` | `10` | 相邻时间差超过该值时标记 gap。 |
| `quality.outlier_method` | `rolling_robust` | 单 bar 异常收益检测方法。可选 `rolling_robust` 或 `none`。 |
| `quality.outlier_lookback` | `7d` | outlier robust sigma 的历史窗口。 |
| `quality.outlier_z_threshold` | `5.0` | 单 bar outlier z-score 阈值。 |
| `quality.robust_sigma_estimator` | `iqr` | robust sigma 估计器。可选 `iqr` 或 `mad`。 |
| `quality.winsorize_outliers` | `false` | 是否对异常收益 winsorize。默认只输出标记，不改收益。 |

### 7.6 realized_vol

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `realized_vol.output_zscore` | `true` | 是否输出 `realized_vol_*_z`。 |
| `realized_vol.zscore_method` | `train_robust` | z-score 标准化方法。可选 `train_robust` 或 `past_rolling_robust`。 |
| `realized_vol.zscore_clip` | `[-5,5]` | z-score 裁剪范围。 |
| `realized_vol.zscore_lookback` | `30d` | past rolling robust z-score 的历史窗口。 |
| `realized_vol.train_split.start_time` | `null` | 训练期起点。与 `end_time` 同时设置时使用训练期 median/IQR。 |
| `realized_vol.train_split.end_time` | `null` | 训练期终点。 |
| `realized_vol.fallback_when_train_missing` | `past_rolling_robust` | 训练期为空或未配置时的回退策略。当前支持 `past_rolling_robust`。 |

### 7.7 range

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `range.compression_method` | `past_percentile` | range compression 计算方法。当前支持 past-only percentile。 |
| `range.percentile_lookback` | `30d` | range width 历史 percentile 窗口。 |
| `range.output_range_to_vol` | `true` | 是否输出 `range_to_vol_*`。 |

### 7.8 trend

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `trend.squash_c` | `2.0` | trend SNR 和 t-stat squash 参数。 |
| `trend.near_zero_return_bps` | `1e-8` | 判定近似 0 收益的阈值。 |
| `trend.regression_time_unit` | `hour` | rolling regression 的时间单位。可选 `minute`、`hour`、`day`。 |
| `trend.strength_weights.trend_efficiency` | `0.35` | trend strength 中 efficiency 权重。 |
| `trend.strength_weights.trend_snr` | `0.35` | trend strength 中 SNR 权重。 |
| `trend.strength_weights.trend_slope_tstat` | `0.30` | trend strength 中 slope t-stat 权重。 |
| `trend.consistency_weights.bar_direction_align` | `0.40` | trend consistency 中 bar 方向一致性权重。 |
| `trend.consistency_weights.block_direction_align` | `0.40` | trend consistency 中 block 方向一致性权重。 |
| `trend.consistency_weights.trend_r2` | `0.20` | trend consistency 中 rolling regression R2 权重。 |
| `trend.block_windows.1h` | `20min` | 1h 趋势窗口内 block return 的 block 长度。 |
| `trend.block_windows.6h` | `1h` | 6h 趋势窗口内 block return 的 block 长度。 |
| `trend.block_windows.24h` | `1h` | 24h 趋势窗口内 block return 的 block 长度。 |

如果新增 `core_windows`，建议同步补充对应 `trend.block_windows.<label>`，否则代码会使用 `1h` 作为默认 block 长度。

### 7.9 jump

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `jump.robust_sigma_lookback` | `7d` | jump robust sigma 历史窗口。 |
| `jump.robust_sigma_estimator` | `iqr` | jump robust sigma 估计器。可选 `iqr` 或 `mad`。 |
| `jump.jump_z_threshold` | `5.0` | jump count 的 z-score 阈值。 |
| `jump.squash_c` | `2.0` | jump proxy squash 参数。 |
| `jump.output_bipower_ratio` | `true` | 是否输出 `jump_ratio_bv_*`。 |
| `jump.output_signed_max_jump` | `true` | 是否输出 `signed_max_jump_return_*_bps`。 |
| `jump.proxy_weights.max_jump_z` | `0.50` | jump proxy 中 max jump z 权重。 |
| `jump.proxy_weights.jump_ratio_bv` | `0.30` | jump proxy 中 bipower ratio 权重。 |
| `jump.proxy_weights.jump_count` | `0.20` | jump proxy 中 jump count 权重。 |

### 7.10 output

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `output.output_dir` | `output` | 输出目录，相对项目根目录解析。 |
| `output.feature_file` | `price_context_features.csv` | 输出特征 CSV 文件名。 |
| `output.include_extended_features` | `true` | 是否输出扩展诊断型特征。`false` 时只保留必需字段。 |
| `output.csv_writer` | `pandas` | CSV 写出方式。`pandas` 保持 pandas 默认浮点文本表示；`fast` 启用快速写出。 |
| `output.float_precision` | `10` | `csv_writer=fast` 时的 CSV 浮点数有效位数。 |
| `output.write_zip` | `true` | 是否额外生成 zip。 |
| `output.zip_file` | `price_context_features.csv.zip` | zip 文件名。 |

### 7.11 report

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `report.enabled` | `true` | 完整 pipeline 后是否生成诊断报告。 |
| `report.output_dir` | `reports/feature_diagnostics` | 诊断报告输出目录。 |
| `report.documentation_file` | `docs/price_context_feature_engineering.md` | 用于读取特征定义的 Markdown 文件。 |
| `report.generate_html` | `true` | 是否生成 HTML 页面。 |
| `report.generate_summary_json` | `true` | 是否生成 `summary.json`。 |
| `report.include_actual_output_features` | `true` | 是否把实际输出中的额外数值字段纳入诊断。 |
| `report.features_file` | `null` | 仅单独运行诊断时使用的特征 CSV 路径。`null` 表示使用 `output` 配置。 |
| `report.fail_on_error` | `true` | 诊断报告生成失败时是否让完整 pipeline 失败。 |
| `report.rolling_window_bars` | `144` | 诊断 rolling 统计窗口，默认 10m bar 下约 24h。 |
| `report.min_valid_count` | `30` | 诊断统计最小有效样本数。 |
| `report.warn_missing_ratio` | `0.20` | 缺失比例超过该值时 WARN。 |
| `report.fail_missing_ratio` | `0.95` | 缺失比例超过该值时 FAIL。 |
| `report.fail_inf_ratio` | `0.05` | inf 比例超过该值时 FAIL。 |
| `report.zscore_threshold` | `5.0` | 诊断 z-score outlier 阈值。 |
| `report.iqr_multiplier` | `1.5` | IQR outlier 倍数。 |
| `report.extreme_quantile_low` | `0.001` | 极端分位数下界。 |
| `report.extreme_quantile_high` | `0.999` | 极端分位数上界。 |
| `report.max_outlier_timestamps` | `50` | summary 中最多记录的异常时间点数量。 |
| `report.max_plot_outlier_points` | `500` | 单特征图中最多绘制的异常点数量。 |
| `report.max_missing_markers` | `500` | 单特征图中最多绘制的缺失标记数量。 |
| `report.relationship_max_points` | `10000` | feature vs price/return 关系散点图最大点数。 |
| `report.future_return_window` | `1h` | 诊断用 future return 窗口。仅用于报告，不进入特征输出。 |
| `report.correlation_scope` | `documented` | 特征相关性热力图范围。`documented` 表示只看文档列。 |
| `report.high_correlation_threshold` | `0.95` | 高相关特征对阈值。 |
| `report.max_high_correlation_pairs` | `100` | summary 中最多记录的高相关特征对数量。 |

### 7.12 execution

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `execution.log_level` | `INFO` | Python logging 级别。 |
| `execution.print_summary` | `true` | 是否打印生成行列数、输出文件和报告路径。 |
| `execution.print_validation_report` | `true` | 是否打印 validation report JSON。 |
| `execution.print_columns` | `false` | 是否打印输出字段列表。替代旧命令中的 `--print-columns`。 |
| `execution.mark_required_columns` | `true` | 打印字段列表时，是否用 `*` 标记必需字段。 |

## 8. 输出文件

默认输出：

```text
price_context/output/price_context_features.csv
price_context/output/price_context_features.csv.zip
```

输出行数与清洗后的时间序列对齐。输出字段由 `windows` 和 `output.include_extended_features` 控制。

## 9. 输出字段说明

### 9.1 基础字段

| 字段 | 含义 |
|---|---|
| `time` | 原始 bar 时间 |
| `price_feature_time` | 特征计算时间，默认等于 `time` |
| `price_feature_age_min` | 特征延迟分钟数，即时计算默认为 0 |

### 9.2 核心必需字段

必需字段包括：

- `past_return_1h_bps`、`past_return_3h_bps`、`past_return_6h_bps`、`past_return_12h_bps`、`past_return_24h_bps`
- `realized_vol_1h_bps`、`realized_vol_6h_bps`、`realized_vol_24h_bps`
- `realized_vol_1h_per_sqrt_hour_bps`、`realized_vol_6h_per_sqrt_hour_bps`、`realized_vol_24h_per_sqrt_hour_bps`
- `range_width_1h_bps`、`range_width_6h_bps`、`range_width_24h_bps`
- `range_compression_1h`、`range_compression_6h`、`range_compression_24h`
- `trend_strength_1h`、`trend_strength_6h`、`trend_strength_24h`
- `trend_consistency_1h`、`trend_consistency_6h`、`trend_consistency_24h`
- `trend_direction_1h`、`trend_direction_6h`、`trend_direction_24h`
- `vol_of_vol_6h`、`vol_of_vol_24h`、`vol_of_vol_48h`
- `jump_proxy_1h`、`jump_proxy_6h`、`jump_proxy_24h`
- `max_jump_z_1h`、`max_jump_z_6h`、`max_jump_z_24h`
- `jump_count_1h`、`jump_count_6h`、`jump_count_24h`
- `price_missing_ratio_1h`、`price_missing_ratio_6h`、`price_missing_ratio_24h`
- `price_gap_flag_1h`、`price_gap_flag_6h`、`price_gap_flag_24h`
- `price_outlier_flag_1h`、`price_outlier_flag_6h`、`price_outlier_flag_24h`

### 9.3 扩展字段

当 `output.include_extended_features=true` 时，还会输出：

- `realized_vol_*_z`
- `range_to_vol_*`
- `trend_efficiency_*`
- `trend_snr_*`
- `trend_slope_*`
- `trend_slope_tstat_*`
- `trend_r2_*`
- `bar_direction_align_*`
- `block_direction_align_*`
- `vol_of_vol_abs_*`
- `vol_of_vol_*_z`
- `jump_ratio_bv_*`
- `signed_max_jump_return_*_bps`
- `price_obs_count_*`
- `price_expected_count_*`

## 10. 特征诊断与可视化报告

完整 pipeline 默认生成：

```text
price_context/reports/feature_diagnostics/index.html
price_context/reports/feature_diagnostics/summary.json
```

单独基于已有特征 CSV 生成报告：

```bash
python -m price_context.src.diagnostics --config price_context/configs/feature_price_context.json
```

如果需要指定已有特征文件，优先修改：

```json
{
  "report": {
    "features_file": "output/price_context_features.csv"
  }
}
```

命令行 `--features` 仅作为临时覆盖入口。

## 11. 如何确认特征已经正确生成

查看输出行数和字段数：

```bash
wc -l price_context/output/price_context_features.csv
python - <<'PY'
import pandas as pd
f = "price_context/output/price_context_features.csv"
df = pd.read_csv(f, nrows=5)
print(df.head())
print(len(df.columns))
PY
```

运行测试：

```bash
pytest -q price_context/tests
```

当前测试覆盖：

- 输入字段缺失；
- time 解析与排序；
- 重复时间戳处理；
- OHLC 一致性校验；
- 输出字段存在性；
- 初期长窗口特征为空；
- 时间缺口触发 gap flag；
- 修改未来价格不会改变过去特征。

## 12. 常见错误与解决方法

### 12.1 找不到输入文件

错误示例：

```text
Input OHLC file not found
```

解决：修改 `input.file_path`，或设置 `paths.project_root` 后使用相对路径。

### 12.2 time 解析失败

如果时间格式特殊，设置：

```json
{
  "input": {
    "datetime_format": "%Y-%m-%d %H:%M:%S"
  }
}
```

### 12.3 OHLC 不一致

例如 `high < close` 或 `low > open`。默认会抛出错误。建议先修复数据源；不建议在特征工程中静默修正严重价格错误。

### 12.4 窗口初期大量 NaN

这是正常现象。长窗口特征需要足够历史数据。例如 24h return 需要找到 `t-24h` 的 close；30d percentile/z-score 也需要历史样本。项目不会用未来数据补初期窗口。

### 12.5 极端收益没有被删除

这是默认设计。BTC 或高波动资产中，极端收益可能是真实跳跃、新闻冲击或清算链条。默认输出 outlier/jump 标记，而不是删除。只有 `quality.winsorize_outliers=true` 时才会裁剪异常收益。
