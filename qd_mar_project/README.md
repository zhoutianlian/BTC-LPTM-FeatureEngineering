# QD-MAR Absorption Project

QD-MAR = Quantile-Calibrated Denoised Market Absorption Rate。

当前版本：v5 full-resolution path dashboard。`path_absorption_dashboard.html` 不再降采样，完整显示每一个 source-clock path absorption 更新点。

本项目将 BTC PLIE-PIC 输出落地为可运行的市场吸收率 / 主动交易力量反推特征工程系统。它严格区分：

- 当前时点可用的 PLIE 被动清算压力；
- 只能在 `event_time + horizon` 后成熟的 event-level absorption；
- 可实时输入 Agent 的 rolling matured absorption memory。

## Quick start

```bash
pip install -r qd_mar_project/requirements.txt
python -m qd_mar_project.feature_qd_mar
PYTHONPATH=qd_mar_project pytest -q qd_mar_project/tests
```

主要输出：

- `output/features/absorption_event_matured.csv`
- `output/features/absorption_curve.csv`
- `output/features/absorption_memory.csv`
- `output/reports/*.csv`
- `output/html/index.html`

完整说明见 `docs/manual.md`。

## v0.2 Optimization

本版将 Agent 默认输入切换为 staleness-aware memory，避免旧 directional absorption 信号在长时间 neutral / low-quality PLIE 区间内被 EWM 保留。新增：

- `quality_assessment.html`
- `context_distribution.csv`
- `directional_label_proportions.csv`
- `directional_quality_summary.csv`
- `staleness_diagnostics.csv`
- `agent_feature_correlation.csv`

所有时间序列图均添加 Plotly range selector / range slider，并将 legend 放在图表下方，同时在图表卡片下方给出 legend 解释。


## v3 path active dominance

This version distinguishes directional path absorption/rejection from neutral or mixed liquidation-pressure active-dominance moves. See `output/html/path_absorption_dashboard.html` and `output/html/scenario_examples.html`.


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



## New multiscale path absorption output

The pipeline writes `output/features/path_absorption_multiscale.csv`, a wide one-row-per-time table exposing 6h/12h/24h/48h path context, label, scores, quality, clarity, and activity fields for downstream state modeling.


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
