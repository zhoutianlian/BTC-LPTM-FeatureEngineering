# QD-MAR 代码架构设计文档

## 1. 项目目录结构

```text
qd_mar_project/
├── configs/
│   └── default.yaml
├── data/
│   └── input/
│       └── plie_predictions_source.csv
├── docs/
│   ├── manual.md
│   ├── code_design.md
│   ├── algorithm.md
│   └── abs_feature_engineering.md
├── output/
│   ├── features/
│   ├── reports/
│   └── html/
├── qdmar/
│   ├── __init__.py
│   ├── absorption.py
│   ├── calibration.py
│   ├── config.py
│   ├── evaluation.py
│   ├── io.py
│   ├── memory.py
│   ├── pipeline.py
│   ├── statistics.py
│   ├── streaming.py
│   ├── validation.py
│   └── visualization.py
├── scripts/
│   └── run_pipeline.py
├── state/
│   └── qdmar_calibration.pkl
├── tests/
│   └── test_no_future_leakage.py
├── README.md
├── requirements.txt
└── pyproject.toml
```

## 2. 每个模块的职责

### `qdmar/config.py`

- 读取 YAML 配置。
- 统一管理路径。
- 提供 `HorizonConfig`。
- 避免在算法模块中硬编码路径或参数。

### `qdmar/io.py`

- 读取 PLIE CSV。
- 显式 UTC timestamp 解析。
- CSV / JSON 输出。
- CSV 输出会统一 timestamp 格式与 float 精度。

### `qdmar/statistics.py`

- 计算 horizon maturity lag。
- 计算 past-matured rolling MAD sigma。
- 用 train-only quantile 生成 volatility regime。
- row-wise winsorization。

### `qdmar/calibration.py`

- `EmpiricalCDF`：经验分布函数。
- `BucketCDFCalibrator`：分层 fallback empirical CDF calibration。
- calibrator 的 fit 仅使用 train split。

### `qdmar/absorption.py`

核心算法模块。

- `prepare_base_context`：计算 sigma、state severity bucket、phase group、vol regime。
- `fit_calibrators`：训练 directional 和 neutral empirical CDF。
- `compute_absorption_events`：生成 event × horizon matured absorption 特征。
- `compute_absorption_curve`：生成 20m/30m/60m curve label。

### `qdmar/memory.py`

- 从 matured absorption rows 构造 rolling memory。
- 使用 `merge_asof(..., direction='backward')` 保证 `available_time <= current_time`。
- 输出 Agent 可用字段。

### `qdmar/evaluation.py`

- q65 coverage。
- response label proportions。
- response percentile 分布稳定性。
- future state exit 研究诊断。
- Agent memory missing / outlier 质量检查。
- latest summary。

### `qdmar/validation.py`

- timestamp monotonicity。
- source-clock alignment。
- split order。
- available_time maturity。
- Agent 输入未来字段检查。
- memory as-of 检查。

### `qdmar/visualization.py`

- 生成深色交互式 HTML 页面。
- 使用 Plotly，支持 hover、缩放、拖拽、legend toggle。

### `qdmar/streaming.py`

- 提供流式场景状态容器 `StreamingQDMarState`。
- 明确在线更新不重新拟合 calibration。
- 只基于成熟标签更新 memory。

### `qdmar/pipeline.py`

- 串联完整离线训练 / 评价 / 报告流程。

### `scripts/run_pipeline.py`

- CLI 入口。

### `tests/test_no_future_leakage.py`

- 未来函数测试。

## 3. 核心类和函数说明

### `Config.from_yaml(path)`

读取配置并提供：

```python
cfg.path("paths", "input_csv")
cfg.horizons
cfg.agent_inputs
```

### `prepare_base_context(df, cfg)`

输入原始 PLIE dataframe，输出带有：

```text
mar_sigma_past_20m_bps
mar_sigma_past_30m_bps
mar_sigma_past_60m_bps
state_severity_bucket
plie_phase_group
vol_regime
```

其中 `vol_regime` 使用 train-only sigma tercile。

### `fit_calibrators(base_df, cfg)`

训练两个 calibrator：

```text
directional: TR_raw empirical CDF
neutral: ActiveZ empirical CDF
```

训练只使用 `split == train`。

### `compute_absorption_events(base_df, cfg, calibrators)`

生成长表：

```text
event_time
available_time
horizon
response_context
transmission_ratio_raw
absorption_raw
response_percentile
absorption_score_q_0_100
active_force_aligned_score
active_force_price_score
market_response_label
```

### `build_memory_features(base_df, event_df, curve_df, cfg)`

生成：

```text
mar_abs_score_q_ewm_6_30m
mar_active_force_aligned_ewm_6_30m
mar_active_force_price_ewm_6_30m
mar_takeover_count_12_30m
mar_amplification_persistence_6_30m
mar_curve_label_last
```

## 4. 数据流向

```text
plie_predictions_source.csv
    ↓
load_plie_csv
    ↓
prepare_base_context
    ↓
fit train-only calibrators
    ↓
compute_absorption_events
    ↓
compute_absorption_curve
    ↓
build_memory_features
    ↓
evaluation + validation + HTML
```

## 5. 训练流程

当前训练指 empirical CDF calibration，不训练黑箱预测模型。

```text
train split directional_core rows
    -> fit response_percentile CDF for TR_raw
train split neutral/weak rows
    -> fit active_z CDF
```

分桶层级：

```text
[horizon, state_severity_bucket, plie_phase_group, vol_regime]
[horizon, state_severity_bucket, vol_regime]
[horizon, vol_regime]
[horizon]
global fallback
```

## 6. 推理流程

每个 event × horizon：

1. 读取 PLIE direction、raw magnitude、reliability、past sigma。
2. 判断 `response_context`。
3. 如果 directional core：
   - 计算 aligned response；
   - 计算 TR raw；
   - 查 empirical CDF 得到 percentile；
   - 输出 score / active force / label。
4. 如果 neutral / weak：
   - 计算 ActiveZ；
   - 输出 neutral active score / label。
5. 设置 `available_time = event_time + horizon`。

## 7. 流式更新流程

生产场景中，离线校准状态固定：

```text
state/qdmar_calibration.pkl
```

在线：

```text
append new PLIE source event
append new 10m price bar
if current_time >= event_time+h:
    compute actual return
    compute matured absorption row
update rolling memory by available_time <= current_time
Agent reads current PLIE + matured memory
```

## 8. 模型与特征状态保存机制

- Calibration state: `state/qdmar_calibration.pkl`
- Event features: `output/features/absorption_event_matured.csv`
- Curve features: `output/features/absorption_curve.csv`
- Agent memory: `output/features/absorption_memory.csv`
- Reports: `output/reports/*.csv`

## 9. 错误处理机制

- 输入 CSV 不存在：`FileNotFoundError`。
- 缺少 `time`：`ValueError`。
- timestamp 解析失败：validation 报告失败。
- split 顺序错误：validation 报告失败。
- Agent 输入包含未来字段：validation 报告失败。

## 10. 测试设计

`tests/test_no_future_leakage.py` 覆盖：

- 10m grid 下 horizon maturity lag 是否正确；
- past sigma 是否不会使用当前未成熟 label；
- timestamp 和 source-clock 检查；
- available_time 是否晚于 event_time；
- Agent 输入是否包含未来字段。

---

# v0.2 Optimization: Staleness-aware Agent Memory

本轮执行结果显示：directional-core absorption 的 q65 coverage 与 response percentile 稳定性符合预期，但 directional-core 只占约 30%。如果直接使用旧版 `mar_abs_score_q_ewm_6_30m`，在长时间 neutral / low-quality PLIE 区间内，旧 directional absorption 可能因 `ignore_na=True` 被长期保留。该行为不构成未来函数，但会降低 Agent 实时输入质量。

因此 v0.2 增加 staleness-aware memory：

```text
mar_directional_core_age_hours_30m
mar_directional_core_freshness_30m = exp(-ln(2) * age_hours / directional_decay_halflife_hours)
```

并将 Agent 主输入切换为：

```text
mar_abs_score_q_staleaware_ewm_6_30m
mar_active_force_aligned_staleaware_ewm_6_30m
mar_active_force_price_staleaware_ewm_6_30m
mar_directional_core_freshness_30m
mar_directional_quality_ewm_6_30m
mar_neutral_active_strength_evidence_ewm_6_30m
```

金融解释：

- 当最近有成熟的 directional-core absorption 时，Agent 获得强吸收 / 放大证据。
- 当最近没有高质量 directional PLIE 事件时，absorption score 自动衰减回 50，active force 自动衰减回 0。
- 这避免把陈旧的清算吸收证据误当成当前市场主动交易力量。

该优化没有改变 event-level absorption 算法，只改变 rolling memory 的在线可用性与证据新鲜度控制。

# V2 Code Changes

New module:

- `qdmar/path_absorption.py`: computes causal path-level / episode-level absorption features.

Updated modules:

- `qdmar/absorption.py`: separates `weak_directional_context`, `true_neutral_plie`, and `low_quality_plie`.
- `qdmar/memory.py`: merges path-level features into online Agent memory.
- `qdmar/evaluation.py`: adds path context, path label, and path quality reports.
- `qdmar/visualization.py`: adds `path_absorption_dashboard.html`.
- `configs/default.yaml`: adds `path_absorption` settings and new Agent input fields.

The path-level module is online-causal: `available_time == time`, because it uses the already-realized path from `t-W` to `t`.


## v3 module changes

- `qdmar/path_absorption.py` now classifies directional pressure, weak directional pressure, mixed pressure, neutral active dominance, and quiet no-pressure path contexts.
- `qdmar/memory.py` exports additional Agent fields for path active dominance.
- `qdmar/evaluation.py` writes `path_scenario_examples.csv`.
- `qdmar/visualization.py` adds `scenario_examples.html` and keeps `market_response_dashboard.html`.


---

## QD-MAR v4 update — PLIE-only path_context and response-only path_label

v4 simplifies path context taxonomy so `path_context` is determined only by cumulative PLIE / liquidation-pressure evidence, not by price response. This makes context a clean pre-response market plumbing statement and leaves all price-path behavior to `path_label`.

### Path context values

- `path_directional_core`: cumulative PLIE has a clear net direction and passes direction consistency, reliability, and SNR gates.
- `path_directional_weak`: cumulative PLIE has a clear net direction, but quality is weaker than core.
- `path_mixed_pressure`: gross PLIE/liquidation pressure is meaningful, but opposing directions offset and no clean net PLIE direction exists.
- `path_neutral_pressure`: little effective cumulative PLIE/liquidation-pressure evidence.

### Path label values

For `path_directional_core` and `path_directional_weak`, labels are PLIE-aligned response labels:

- `path_cascade_transmission`
- `path_baseline_transmission`
- `path_partial_absorption`
- `path_full_absorption_stall`
- `path_pressure_rejection`
- `path_reversal_takeover`

For `path_mixed_pressure`, labels describe whether price breaks out despite mixed liquidation pressure:

- `path_mixed_active_breakout_up`
- `path_mixed_active_breakout_down`
- `path_normal_mixed_active_breakout_up`
- `path_normal_mixed_active_breakout_down`
- `path_mixed_pressure_chop`

For `path_neutral_pressure`, labels describe active-dominance price movement under low/neutral PLIE pressure:

- `path_active_dominance_up`
- `path_active_dominance_down`
- `path_normal_active_dominance_up`
- `path_normal_active_dominance_down`
- `path_quiet_no_pressure`

The numeric columns `path_active_z`, `path_active_dominance_score`, and `path_active_dominance_price_score` preserve movement intensity, so the categorical label space does not need separate sparse `extreme_*` classes.

### Why this change

The previous v3 context mixed pressure quality and price response, producing many context categories such as active-context and normal-move-context. v4 reduces context to four PLIE-only buckets, which makes the context-label matrix easier to audit and improves sample size per combination. Price response remains fully represented in `path_label` and numeric scores.



---

## v5 可视化与持久化设计更新

### Path dashboard full-resolution rendering

`qdmar.visualization.build_path_absorption_dashboard()` 在 v5 中不再使用：

```python
pmain.iloc[::p_step]
memory_df.iloc[::m_step]
```

而是：

```python
pmain_plot = pmain.copy()
memory_plot = memory_df.copy()
```

设计原因：

- `path_context` 与 `path_label` 是离散状态审查字段；
- 降采样会隐藏小时级标签切换；
- 用户需要在任意局部时间段内检查每个 source-clock 更新点；
- 页面已有 UTC 时间选择器，局部放大后可以详细观察。

其他页面仍可按需要降采样，例如 feature statistics 中的全历史多变量图。

### Path absorption CSV persistence

`qdmar.pipeline.run_pipeline()` 现在保存完整 24h 主 path series：

```python
path_to_save = path_df[path_df["window_hours"].eq(24)].copy()
```

而不是只保存最近 2000 行。这样 `output/features/path_absorption.csv` 可作为完整路径状态审查表。



## Multiscale path absorption persistence

`qdmar.path_absorption.compute_path_absorption` still returns the long path table. `qdmar.path_absorption.build_path_absorption_multiscale` pivots the long table into a wide online table with one row per timestamp and one column per metric/window. `qdmar.pipeline.run_pipeline` saves this table as `output/features/path_absorption_multiscale.csv`.

This design keeps diagnostics and visualization flexible through the long table while giving downstream modeling a compact, directly consumable multiscale feature table.


# QD-MAR v7 Absorption Optimization
# QD-MAR v7 Absorption Optimization Change Log

## 1. Why this change was made

Actual downstream state-model checks indicated three related risks in the previous path absorption layer:

1. The legacy `path_quality` could become low when PLIE directionality or activity was low, even when data were perfectly reliable. This is harmful for RC because quiet, low-pressure, low-volatility markets are often valid consolidation evidence, not bad-quality evidence.
2. `path_context`, `path_label`, and `path_quality` were not sufficiently separated for downstream six-state evidence mapping.
3. Price context was not used to distinguish neutral-pressure quiet markets from neutral-pressure active-dominance moves, or mixed-pressure chop from mixed-pressure active breakouts.

## 2. Adopted design

### 2.1 `path_context`

`path_context` now only describes cumulative PLIE / liquidation pressure. It does not use price.

For each window `W`:

```text
p_i = plie_direction_i * plie_reliability_i * I_i
PressureMass_W = sum(abs(p_i))
NetPressure_W = sum(p_i)
Directionality_W = abs(NetPressure_W) / (PressureMass_W + eps)
DominantDirection_W = sign(NetPressure_W)
```

where `I_i` is configured by `path_context.pressure_intensity_source`. If `plie_intensity_scaled` is missing, the code derives a train-only quantile-scaled `plie_intensity`; if that is unavailable it falls back to `abs(plie_main_bps)`.

Context labels:

- `path_directional_core`: high pressure mass and high directionality.
- `path_directional_weak`: meaningful net pressure, but weaker than core.
- `path_mixed_pressure`: meaningful pressure mass with poor net direction.
- `path_neutral_pressure`: low pressure mass or no effective PLIE pressure.

### 2.2 `path_label`

`path_label` describes how the already-realized price path `[T-W, T]` responds to the context.

For directional context:

```text
R_W = 10000 * log(P_T / P_{T-W})
AlignedReturn_W = DominantDirection_W * R_W
BaselineImpact_W = sum(plie_main_bps_i)
TransmissionRatio_W = AlignedReturn_W / (abs(BaselineImpact_W) + eps)
```

Labels:

- `path_cascade_transmission`
- `path_baseline_transmission`
- `path_partial_absorption`
- `path_full_absorption_stall`
- `path_pressure_rejection`
- `path_reversal_takeover`

`path_partial_absorption` is no longer treated as strong RHA evidence. It is orderly-trend / transition evidence.

For neutral context, labels are active-dominance or quiet labels. For mixed context, labels are active-breakout or chop labels.

### 2.3 `path_quality` refactor

The old single field is split into:

- `path_data_quality`: data reliability only; missingness, gaps, outliers, window maturity.
- `path_signal_clarity`: semantic clarity of context and label.
- `path_activity_level`: market activity; pressure mass, realized volatility, trend strength, jump proxy.

Low activity is not bad quality. Low activity with good data quality and quiet price context can support RC.

The legacy `path_quality` remains for compatibility and now equals approximately:

```text
path_quality = path_data_quality * path_signal_clarity
```

It should not be used as the only quality/evidence field.

## 3. Price context integration

The optional `price_context_features.csv.zip` is loaded with selected columns and merged by backward as-of on `time`. This preserves causality. Price context is used as supplementary evidence for labels and state evidence:

- RC: low realized volatility, low trend strength, high range compression, low jump proxy.
- VT: neutral/mixed PLIE with active dominance or active breakout.
- HPEM: directional PLIE with cascade plus realized volatility or jump support.
- RHA: directional pressure rejection / reversal takeover / stall.
- AMB: signal conflict, data-quality issue, cross-window conflict.

## 4. Audit-only six-state evidence

The project still does not train or output a production six-state model. v7 adds audit-only evidence fields such as:

- `e_rc_quiet_pressure`
- `e_rha_pressure_rejection`
- `e_vt_active_dominance`
- `e_hpem_cascade_transmission`
- `e_amb_signal_conflict`
- `e_st_partial_absorption`

and proxy scores:

- `score_rc_proxy`
- `score_rha_proxy`
- `score_vt_proxy`
- `score_hpem_proxy`
- `score_amb_proxy`
- `score_st_proxy`

These are for diagnosis and downstream model design, not final trading states.

## 5. Expected improvements

- RC should no longer be suppressed simply because activity is low.
- RHA should be driven mainly by rejection/takeover/stall, not partial absorption.
- Mixed pressure with clear active breakout can support VT rather than automatically AMB.
- AMB is reserved for signal conflict, label-margin weakness, cross-window conflict, or data-quality problems.
- The output contains enough intermediate fields for manual review and rollback.

## 6. Potential side effects

- Context proportions change because pressure mass is now defined by `d * reliability * scaled_intensity`.
- `path_mixed_pressure` becomes stricter; some previous mixed windows become neutral.
- State proxy fields are not final model states and should not be traded directly.
- Price context errors can affect labels; data-quality fields are included to audit this risk.

## 7. Rollback

Set `price_context.enabled: false` and restore the previous `qdmar/path_absorption.py` from v6. The v7 configuration preserves old output names where possible, and `path_quality` remains as a compatibility field.
