# QD-MAR 操作手册

## 1. 项目简介

QD-MAR（Quantile-Calibrated Denoised Market Absorption Rate）用于在 PLIE-PIC 输出基础上构造 BTC 市场吸收率与主动交易力量反推特征。它严格区分：

- 当前时点可用的 PLIE 被动清算压力；
- 只能在 `event_time + horizon` 后成熟的 event-level absorption；
- 可实时输入 Agent 的 rolling matured absorption memory；
- 使用过去到当前窗口的 path-level / episode-level absorption。

核心原则：

```text
当前 PLIE 可实时使用
Absorption_{t,h} 只有在 event_time + h 后成熟
Agent 只能读取 rolling matured absorption memory
```

## 2. 环境安装

建议 Python 3.10+。在仓库上层目录执行：

```bash
pip install -r qd_mar_project/requirements.txt
```

依赖：`pandas`、`numpy`、`PyYAML`、`plotly`、`pytest`。

## 3. 执行方法

标准执行位置是仓库上层目录，入口与 `liq_pressure_hmm` 保持同类风格：

```bash
python -m qd_mar_project.feature_qd_mar
```

指定配置：

```bash
python -m qd_mar_project.feature_qd_mar --config qd_mar_project/configs/default.yaml
```

旧入口仍保留用于兼容：

```bash
cd qd_mar_project
python scripts/run_pipeline.py --config configs/default.yaml
```

是否生成 HTML、输出路径、字段白名单、阈值和窗口等运行参数均由 `configs/default.yaml` 控制。新入口只接受 `--config`，不把业务参数放在命令行。

## 4. 数据准备

默认主输入：

```text
plie_pic_project/outputs/predictions/plie_predictions_source.csv
```

默认可选价格上下文输入：

```text
price_context/output/price_context_features.csv.zip
```

`columns.required_input` 定义主输入必需字段：

```text
time, price, split,
ret_20m_bps, ret_30m_bps, ret_60m_bps,
plie_passive_20m_bps, plie_passive_30m_bps, plie_passive_60m_bps,
plie_passive_20m_bps_mag_raw, plie_passive_30m_bps_mag_raw, plie_passive_60m_bps_mag_raw,
plie_main_bps, plie_direction, plie_reliability, plie_intensity, plie_phase,
hmm_state, hmm_conf, liq_entropy, age_in_state_source, state_severity
```

`columns.optional_input` 定义可选审计字段：

```text
liq_feature_time, liq_feature_time_raw, liq_feature_age_min, plie_intensity_scaled
```

生产流式数据中，`ret_*` 不应在事件发生时提前给 Agent 使用。它们只能在价格到达 `event_time + horizon` 后进入 matured absorption table。

## 5. 配置总表

默认配置文件：

```text
qd_mar_project/configs/default.yaml
```

### `project`

- `name`：项目名称。
- `version`：配置 / 算法版本。
- `root_dir`：项目根目录，相对配置文件解析；默认 `..` 表示 `qd_mar_project/`。

### `run`

- `make_html`：是否生成 HTML 可视化。
- `drop_unmatured_tail`：是否丢弃 `available_time` 晚于当前最大观测时间的尾部未成熟行。
- `print_outputs`：入口执行完成后是否打印输出文件路径。

### `paths`

- `input_csv`：主输入 CSV。
- `price_context_csv`：可选价格上下文文件。
- `output_dir`：总输出目录。
- `features_dir`：特征 CSV 输出目录。
- `reports_dir`：报告输出目录。
- `html_dir`：HTML 输出目录。
- `state_dir`：状态文件目录。
- `calibration_state`：校准器持久化路径。
- `latest_summary`：旧版 latest summary 路径；新版输出以 `outputs.reports.latest_summary` 为准。

### `data`

- `time_col`：事件时间列。
- `price_col`：价格列。
- `split_col`：训练 / 验证 / 测试切分列。
- `timezone`：时间语义，默认 `UTC`。
- `expected_splits`：合法 split 顺序，用于无泄漏校验。
- `source_clock`：输入是否是 source-clock 对齐结果。
- `source_time_col`：PLIE 源特征时间列。
- `source_time_raw_col`：PLIE 原始源时间列。
- `source_age_col`：PLIE 源特征年龄列。

### `io`

- `timestamp_parse_columns`：读取 CSV 时额外转换为 UTC datetime 的列。
- `csv_date_format`：写 CSV 的时间格式。
- `csv_chunksize`：写 CSV 的 pandas chunksize。

### `columns`

- `required_input`：主输入必须存在的字段清单。
- `optional_input`：存在时会被读取、校验或输出的字段清单。

### `horizons`

每个 horizon 一组：

- `name`：horizon 名称，例如 `20m`、`30m`、`60m`。
- `minutes`：成熟延迟分钟数。
- `ret_col`：实际未来收益 bps 标签列。
- `plie_col`：有效 PLIE 被动压力列。
- `raw_mag_col`：raw q65 PLIE 强度列。
- `eff_mag_col`：有效 PLIE 强度列。
- `b_min_bps`：该 horizon 下进入 directional context 的最低 raw PLIE bps。

### `absorption`

- `reliability_min`：directional core 的最低 PLIE reliability。
- `snr_min`：directional core 的最低 raw PLIE SNR。
- `snr_low` / `snr_full`：质量权重的 SNR 线性映射区间。
- `eps`：数值稳定项。
- `lambda_sigma`：波动项进入 z-score scale 的权重。
- `neutral_tanh_scale`：neutral active force 的 tanh 缩放。
- `active_residual_winsor_sigma`：active residual 按 sigma winsorize 的倍数。
- `deadzone.b_raw_fraction`：吸收 / stall 死区中 raw PLIE 的比例。
- `deadzone.sigma_fraction`：吸收 / stall 死区中 past sigma 的比例。
- `sigma.method`：波动估计方法；当前实现为 `rolling_mad`。
- `sigma.window`：past-matured rolling MAD 窗口。
- `sigma.min_periods`：rolling MAD 最少样本。
- `sigma.shift_by_maturity`：波动估计只使用已成熟标签。
- `labels.u_amplification`：同向放大分位阈值。
- `labels.u_baseline`：基线传导分位阈值。
- `labels.u_normal_low`：正常响应下界。
- `labels.u_partial_low`：部分吸收下界。
- `labels.active_z_low` / `active_z_normal` / `active_z_strong`：neutral active move 分级阈值。
- `weak_reliability_min`：weak directional context 的最低 reliability。
- `weak_snr_min`：weak directional context 的最低 SNR。
- `weak_directional_labels.amplification_tr`：weak directional 放大候选 transmission ratio。
- `curve.horizon_percentile_columns`：把 horizon percentile 重命名为 `u20/u30/u60` 等曲线列。
- `curve.maturity_minutes`：curve 成熟延迟；`auto` 表示取最大 horizon minutes。
- `curve.label_thresholds.*`：multi-horizon curve label 的所有分位阈值。

### `calibration`

- `min_bucket_size`：经验 CDF 校准最小桶样本数。
- `train_split`：用于拟合校准器的 split 名称。
- `bucket_levels`：directional calibration 的逐级 fallback 分桶字段。
- `neutral_bucket_levels`：neutral / weak active calibration 的逐级 fallback 分桶字段。

### `memory`

- `main_horizon`：rolling memory 使用的主 event horizon。
- `ewm_spans`：生成 EWM memory 的 span 列表。
- `persistence_window`：amplification / absorption persistence 的 rolling 窗口。
- `takeover_window`：takeover / neutral context 统计窗口。
- `curve_mode_window`：curve label rolling mode 窗口。
- `directional_decay_halflife_hours`：directional-core freshness 半衰期。

### `visualization`

- `include_plotlyjs`：Plotly JS 引入方式。
- `max_points`：普通时间序列 HTML 最大绘图点数。
- `latest_rows`：预留的最新行展示参数。

### `agent_inputs`

Agent 输入字段白名单。默认使用 staleness-aware matured memory 和 path episode 字段，不包含 `ret_*`、`event_time`、`available_time` 等未来标签或成熟时间字段。

### `path_absorption`

- `enabled`：是否生成 path-level absorption。
- `main_horizon`：path baseline 使用的 horizon 配置。
- `windows_hours`：path-level 回看窗口。
- `source_clock_default_minutes`：无法推断 source-clock 间隔时的默认分钟数。
- `min_window_points`：每个 path window 至少包含的 source 点数。
- `price_return_bps_scale`：log return 转 bps 的倍数。
- `vol_window` / `vol_min_periods`：path return past-only sigma 参数。
- `lambda_sigma`：path scale 中 sigma 权重。
- `sigmoid_clip`：sigmoid / tanh 输入裁剪范围。
- `active_score_clip_upper`：active dominance 原始分数上限。
- `intensity_train_quantile` / `intensity_default_scale` / `intensity_max`：PLIE intensity 缺失时的 train-only 归一化参数。
- `min_direction_consistency` / `core_direction_consistency`：方向一致性阈值。
- `min_net_braw_bps` / `min_total_braw_bps`：raw PLIE 路径强度阈值。
- `min_nonzero_direction_share`：非零方向占比阈值。
- `weak_reliability_min` / `core_reliability_min`：path context reliability 阈值。
- `weak_snr_min` / `core_snr_min`：path context SNR 阈值。
- `label_thresholds.*`：path directional label transmission ratio 阈值。
- `score_weights.*`：cascade / pressure rejection 辅助分数权重。
- `quiet_score_weights.*`：neutral quiet score 权重。
- `chop_score_weights.*`：mixed chop score 权重。

### `price_context`

- `enabled`：是否加载价格上下文。
- `required`：文件缺失时是否报错。
- `csv_path`：价格上下文 CSV/ZIP 路径。
- `merge_tolerance_minutes`：backward-asof 合并最大容忍分钟数。
- `usecols`：价格上下文读取字段白名单。
- `window_mapping`：path window 到价格上下文窗口后缀的映射。
- `use_realized_vol` / `use_range_compression` / `use_trend_strength` / `use_trend_consistency` / `use_vol_of_vol` / `use_jump_proxy`：价格上下文证据开关。

### `path_context`

- `windows`：path context 输出窗口。
- `pressure_intensity_source`：优先使用的压力强度字段。
- `fallback_intensity_source`：强度字段缺失时的语义 fallback。
- `intensity_train_quantile` / `intensity_default_scale` / `intensity_max`：强度归一化参数。
- `mass_low_quantile` / `mass_high_quantile`：train-only 压力质量分位阈值。
- `direction_core_threshold` / `direction_weak_threshold` / `mixed_threshold`：path PLIE context 分类阈值。
- `min_valid_obs_share`：窗口内有效 PLIE 观测占比下限。
- `fallback_context`：非 core/weak/mixed 且有压力时的 fallback context。

### `path_label`

- `baseline_impact_mode`：baseline impact 计算方式。
- `amplification_tr` / `baseline_tr` / `partial_tr` / `near_zero_tr` / `strong_negative_tr`：directional path label 阈值。
- `cascade_score_min`：cascade transmission 的最低辅助分。
- `active_z_normal` / `active_z_strong` / `active_z_extreme`：neutral/mixed active dominance 强度阈值。
- `quiet_score_min` / `chop_score_min`：quiet / chop 标签阈值。
- `vol_low_quantile` / `vol_high_quantile` / `trend_low_quantile` / `trend_high_quantile` / `range_high_quantile` / `jump_high_quantile`：价格上下文 train-only 分位阈值。
- `use_train_split_thresholds`：是否优先使用 `split == train_split` 拟合阈值。
- `train_split_ratio`：无 split 或禁用 split 阈值时的前段训练比例。

### `path_quality`

- `enable_refactor`：启用 decoupled quality 语义。
- `output_legacy_path_quality`：是否保留兼容字段 `path_quality`。
- `data_quality_weights.*`：数据质量缺失、gap、price missing、PLIE missing、price outlier 权重。
- `signal_clarity_weights.*`：方向清晰度、label margin、quiet/chop 支撑、data quality 权重。
- `activity_level_weights.*`：pressure mass、realized vol、trend strength、jump proxy 权重。

### `state_mapping`

- `partial_absorption_boosts_rha`：partial absorption 是否增强 RHA proxy。
- `mixed_pressure_auto_amb`：mixed pressure 是否自动增强 AMB proxy。
- `low_activity_is_bad_quality`：低活动是否视作坏质量；默认 false。
- `quiet_no_pressure_supports_rc`：quiet no pressure 是否支持 RC proxy。

### `validation`

- `source_time_col`：source-clock alignment 检查列。
- `forbidden_agent_input_fragments`：Agent 输入字段中禁止出现的未来标签 / 成熟时间片段。

### `outputs`

- `features.*`：各 feature CSV 文件名。
- `reports.*`：各 report CSV/JSON 文件名。
- `html.*`：各 HTML 文件名，导航也从这里读取。
- `filters.event_horizon`：持久化 event table 的 horizon；`auto` 使用 `memory.main_horizon`。
- `filters.path_window_hours`：持久化 path_absorption 长表的窗口。
- `report_params.state_exit_windows`：state exit 诊断窗口。
- `report_params.path_scenario_main_window`：典型 path scenario 使用的窗口。
- `columns.base_context`：`base_context.csv` 输出字段白名单。
- `columns.memory_context`：`absorption_memory.csv` 固定上下文字段。
- `columns.memory_audit_contains`：额外保留包含这些片段的审计字段。
- `columns.path_absorption`：`path_absorption.csv` 输出字段白名单。
- `columns.absorption_event_matured`：`absorption_event_matured.csv` 输出字段白名单。
- `column_prefixes.base_context`：`base_context.csv` 额外保留的价格上下文字段前缀。

默认输出字段白名单如下：

- `outputs.columns.base_context`：`time, price, split, liq_feature_time_raw, liq_feature_time, liq_feature_age_min, hmm_state, hmm_conf, liq_entropy, age_in_state_source, state_severity, plie_direction, plie_main_bps, plie_reliability, plie_intensity, plie_phase, mar_sigma_past_20m_bps, mar_sigma_past_30m_bps, mar_sigma_past_60m_bps, state_severity_bucket, plie_phase_group, vol_regime`。
- `outputs.columns.memory_context`：`time, price, hmm_state, plie_main_bps, plie_reliability, plie_direction, plie_phase, split`。
- `outputs.columns.memory_audit_contains`：`available_time, last_directional_core`，表示额外保留字段名中包含这些片段的审计列。
- `outputs.columns.path_absorption`：`time, available_time, window_hours, split, price, hmm_state, path_pressure_name, path_return_bps, path_pressure_mass, path_net_pressure, path_directionality, path_dominant_direction, path_pressure_obs_count, path_pressure_missing_ratio, path_aligned_return_bps, path_baseline_impact_bps, path_signed_plie_effective_sum_bps, path_raw_plie_total_bps, path_net_braw_bps, path_direction_consistency, path_liq_neutrality_score, path_snr, path_active_z, path_active_dominance_score, path_active_dominance_price_score, path_aligned_response_bps, path_transmission_ratio, path_absorption_score_0_100, path_quality, path_data_quality, path_signal_clarity, path_activity_level, path_pressure_rejection_score, path_cascade_score, path_quiet_score, path_chop_score, realized_vol_used_bps, range_compression_used, trend_strength_used, trend_consistency_used, trend_direction_used, jump_proxy_used, price_missing_ratio_used, price_gap_flag_used, price_outlier_flag_used, path_context, path_label`。
- `outputs.columns.absorption_event_matured`：`event_time, available_time, horizon, horizon_minutes, split, hmm_state, state_severity, plie_phase, plie_direction, plie_reliability, signed_plie_effective_bps, plie_reference_raw_bps, actual_return_bps, aligned_actual_response_bps, snr_raw, volatility_sigma_past_bps, quality_weight, response_context, transmission_ratio_raw, absorption_raw, response_percentile, absorption_score_q_0_100, active_force_aligned_score, active_force_price_score, neutral_active_strength_score_0_100, directional_absorption_label, market_response_label`。

## 6. 输出文件说明

默认输出位于 `qd_mar_project/output/`：

- `output/features/base_context.csv`：核心 PLIE / HMM / sigma / price-context 审计字段。
- `output/features/absorption_event_matured.csv`：主 horizon 的成熟 event-level absorption。
- `output/features/absorption_curve.csv`：多 horizon percentile curve 与 curve label。
- `output/features/absorption_memory.csv`：可实时输入 Agent 的 rolling matured memory。
- `output/features/path_absorption.csv`：主 path window 的路径级长表。
- `output/features/path_absorption_multiscale.csv`：一行一个时间点的多窗口宽表。
- `output/reports/*.csv`：校准、标签占比、质量、staleness、path context 和 state evidence 报告。
- `output/html/index.html`：HTML 入口页。

## 7. 如何运行推理 / 新数据

离线训练、校准和推理在当前管线中统一执行。生产增量场景可使用 `qdmar.streaming.StreamingQDMarState`：

```python
from qdmar.config import Config
from qdmar.streaming import StreamingQDMarState

cfg = Config.from_yaml("configs/default.yaml")
state = StreamingQDMarState(cfg, cfg.path("paths", "calibration_state"))
state.append_plie_events(new_events)
latest_memory = state.recompute_matured_from_available_labels(current_time)
```

生产流程：

1. 新 PLIE source-clock 行进入 `plie_event_table`。
2. 新 10m price bar 进入 price grid。
3. 对 pending event 检查是否满足 `current_time >= event_time + h`。
4. 满足则计算 `actual_return_bps`，生成 matured absorption row。
5. 用 matured rows 更新 rolling memory。
6. Agent 读取 current PLIE + matured memory。

## 8. 如何运行测试

从项目目录执行：

```bash
cd qd_mar_project
pytest -q
```

从仓库上层执行：

```bash
PYTHONPATH=qd_mar_project pytest -q qd_mar_project/tests
```

测试覆盖：timestamp 单调性、source-clock alignment、split 顺序、horizon maturity lag、available_time、Agent 输入未来字段检查、path absorption 无未来价格泄漏。
