# PLIE-PIC 操作手册

## 1. 项目定位

PLIE-PIC 是 Source-clock Mechanism-constrained Quantile Passive Liquidation Impact Curve。它消费上游 HMM 清算状态，不重新训练 HMM；目标是估计 BTC 期货清算 forced flow 对价格造成的被动冲击基线，而不是普通收益预测。

主输出包括：

| 字段 | 含义 |
|---|---|
| `plie_passive_<h>m_bps` | 配置中每个 horizon 的 signed passive PLIE，单位 bps |
| `plie_main_bps` | 主 horizon 输出，由 `features.main_horizon_min` 指定 |
| `plie_direction` | 被动清算压力方向，`1` 向上，`-1` 向下，`0` 中性 |
| `plie_reliability` | 当前 PLIE 基线可信度 |
| `plie_phase` | 当前 PLIE 解释阶段 |

后续吸收率或主动力量分析可以使用：

```text
actual price move - PLIE passive baseline
```

## 2. 推荐运行方式

在仓库根目录运行，也就是 `plie_pic_project` 的上层目录：

```bash
python -m plie_pic_project.feature_plie_pic
```

不带命令时，入口读取 `plie_pic_project/config/config.yaml` 中的 `runtime.command`，默认执行 `all`。

常用命令：

```bash
python -m plie_pic_project.feature_plie_pic train
python -m plie_pic_project.feature_plie_pic infer
python -m plie_pic_project.feature_plie_pic monitor
python -m plie_pic_project.feature_plie_pic report
python -m plie_pic_project.feature_plie_pic validate
python -m plie_pic_project.feature_plie_pic scheduled-retrain
```

指定配置文件：

```bash
python -m plie_pic_project.feature_plie_pic all --config plie_pic_project/config/config.yaml
```

所有输入、输出、字段、计算、评估、报告和重训参数都应写在配置文件中。CLI 只负责选择要运行的 command 和配置文件路径。

## 3. 输入文件

配置项：

```yaml
paths:
  input_csv: data/input/hmm_state.csv.zip
```

支持：

| 格式 | 示例 |
|---|---|
| 普通 CSV | `data/input/hmm_state.csv` |
| ZIP 压缩 CSV | `data/input/hmm_state.csv.zip` 或 `data/input/hmm_state.zip` |

ZIP 中应包含一个 CSV 数据文件。所有相对路径都以 `plie_pic_project` 项目根目录为基准解析，不依赖当前 shell 所在目录。

输入表必须包含 `schema` 中配置的字段。默认字段包括 10m 价格、清算金额、source-clock 时间、HMM hard state、HMM posterior、entropy、confidence 和 state age。

## 4. 配置文件总览

主配置文件：

```text
plie_pic_project/config/config.yaml
```

### 4.1 `project`

| 参数 | 说明 |
|---|---|
| `project.name` | 项目名称，仅用于识别和文档说明 |
| `project.version` | 配置/项目版本 |
| `project.description` | 项目描述 |

### 4.2 `runtime`

| 参数 | 说明 |
|---|---|
| `runtime.command` | 不带命令运行 `python -m plie_pic_project.feature_plie_pic` 时执行的命令，默认 `all` |
| `runtime.generate_html` | `all`、`monitor`、`scheduled-retrain` 是否生成 HTML 报告 |
| `runtime.force_retrain` | `scheduled-retrain` 是否强制重训；需要强制时在配置中设为 `true` |

### 4.3 `paths`

| 参数 | 说明 |
|---|---|
| `paths.input_csv` | 输入 HMM state 表，支持 `.csv`、`.zip`、`.csv.zip` |
| `paths.output_dir` | 总输出目录 |
| `paths.feature_dir` | source-clock 特征输出目录 |
| `paths.prediction_dir` | 预测输出目录 |
| `paths.evaluation_dir` | 评估表输出目录 |
| `paths.check_dir` | 未来函数和数据检查输出目录 |
| `paths.model_dir` | 模型 artifact 目录 |
| `paths.report_html_dir` | HTML 报告目录 |
| `paths.log_dir` | 日志目录 |
| `paths.train_log` | 训练日志文件 |
| `paths.model_artifact` | 训练后的 joblib 模型文件 |
| `paths.model_summary` | 模型摘要 JSON |
| `paths.source_features` | source-clock 特征和 label CSV |
| `paths.train_source_predictions` | 训练/全量 source-clock 预测 CSV |
| `paths.train_bar_predictions` | 广播到 10m bar 的训练/全量预测 CSV |
| `paths.latest_source_predictions` | 批量推理的最新 source-clock 预测 CSV |
| `paths.latest_bar_predictions` | 批量推理的最新 10m 预测 CSV |
| `paths.latest_agent_payload` | 最新 Agent payload JSON |
| `paths.latest_summary` | 最新训练/监控摘要 JSON |
| `paths.scheduled_retrain_decision` | 定期重训决策 JSON |
| `paths.pre_feature_checks` | 特征生成前检查 JSON |
| `paths.feature_checks` | 特征生成后检查 JSON |
| `paths.post_training_checks` | 训练后检查 JSON |
| `paths.manual_validation_checks` | 手动 `validate` 输出 JSON |

### 4.4 `schema`

| 参数 | 说明 |
|---|---|
| `schema.time_col` | 价格 bar 时间字段 |
| `schema.price_col` | 价格字段 |
| `schema.liq_time_col` | 清算特征可用时间/source-clock 字段 |
| `schema.liq_time_raw_col` | 上游原始清算 source 时间字段，可选但默认保留 |
| `schema.liq_age_col` | 当前 bar 距离清算 source 的分钟数 |
| `schema.long_liq_col` | 多头清算金额字段 |
| `schema.short_liq_col` | 空头清算金额字段 |
| `schema.total_liq_col` | 总清算金额字段 |
| `schema.hmm_state_col` | HMM hard state 字段 |
| `schema.hmm_conf_col` | HMM hard state 置信度字段 |
| `schema.hmm_state_conf_col` | HMM state confidence 字段 |
| `schema.entropy_col` | HMM posterior entropy 字段 |
| `schema.state_age_source_col` | source-clock 上当前状态持续长度字段 |
| `schema.posterior_cols` | HMM posterior 概率字段列表，顺序必须与 `features.posterior_severity_weights` 对齐 |

### 4.5 `features`

| 参数 | 说明 |
|---|---|
| `features.eps` | 数值稳定用极小值，避免除零或 log 异常 |
| `features.source_age_zero_value` | source-clock 行筛选值，默认只取 `liq_feature_age_min == 0` |
| `features.source_valid_states` | 允许进入 source-clock 的 HMM state |
| `features.strong_states` | 强清算压力状态，用于 strong entry 和机制切片 |
| `features.robust_window_source` | source-clock rolling median/MAD 窗口长度 |
| `features.robust_min_periods` | rolling robust z-score 的最小样本数 |
| `features.posterior_severity_weights` | HMM posterior severity 权重，默认 state 1/2 为上行压力、4/5 为下行压力 |
| `features.posterior_severity_scale` | posterior severity 坐标缩放因子 |
| `features.hmm_posterior_weight_lambda` | HMM severity coordinate 与原始清算 imbalance 的融合权重 |
| `features.direction_deadzone` | 方向坐标绝对值低于该阈值时置为中性 |
| `features.horizons_min` | 输出和训练 label 的 horizon 分钟列表 |
| `features.main_horizon_min` | `plie_main_bps` 使用的主 horizon；若不在 `horizons_min` 中则使用第一个 horizon |
| `features.quantile` | 被动冲击分位数目标，例如 `0.65` 表示 q65 |
| `features.freshness_no_decay_min` | source 特征在多少分钟内不做 freshness 衰减 |
| `features.freshness_decay_min` | 超过 no-decay 后的指数衰减时间尺度 |
| `features.entropy_max` | HMM entropy 归一化上限，5 状态默认 `ln(5)` |
| `features.transition_severity_map` | HMM 转移类型到强度的映射，key 为 `"prev->cur"` |

`features.phase`：

| 参数 | 说明 |
|---|---|
| `features.phase.accel_window_source` | 判断 accelerating phase 的 source-clock rolling 窗口 |
| `features.phase.accel_min_periods` | accelerating rolling median 的最小样本数 |
| `features.phase.mature_state_age_source` | state age 达到多少 source rows 后标记为 mature |
| `features.phase.labels.neutral` | 中性阶段标签 |
| `features.phase.labels.early_strong_entry` | 刚进入强状态阶段标签 |
| `features.phase.labels.accelerating` | 清算压力加速阶段标签 |
| `features.phase.labels.mature` | 状态成熟阶段标签 |
| `features.phase.labels.normal` | 默认阶段标签 |

默认 `transition_severity_map` 中，`2->1`、`4->5` 表示同向增强，`1->5`、`5->1` 表示 abrupt reversal，因此 severity 为负。

### 4.6 `model`

| 参数 | 说明 |
|---|---|
| `model.model_type` | 模型类型说明，默认 constrained quantile impact curve |
| `model.feature_names` | 进入模型的特征列；不能包含 `ret_*`、future、label、actual、residual 等未来信息 |
| `model.l2_alpha` | L2 正则强度 |
| `model.max_iter` | 优化最大迭代次数 |
| `model.optimizer` | 优化器说明 |
| `model.multiply_by_reliability` | 预测 magnitude 是否乘以 `plie_reliability` |
| `model.use_sample_weight` | 训练时是否使用样本权重 |
| `model.sample_weight_floor` | 样本权重下限 |
| `model.min_train_rows` | 最小训练样本数，不足则拒绝训练 |

### 4.7 `split`

| 参数 | 说明 |
|---|---|
| `split.train_ratio` | 时间顺序训练集占比 |
| `split.validation_ratio` | 时间顺序验证集占比 |
| `split.test_ratio` | 时间顺序测试集占比 |
| `split.embargo_minutes` | 预留隔离参数；当前用于配置说明，切分仍保持严格时间顺序 |

### 4.8 `evaluation`

| 参数 | 说明 |
|---|---|
| `evaluation.split_names` | 评估时使用的 split 名称 |
| `evaluation.all_subset_name` | 全样本汇总名称 |
| `evaluation.table_files` | 各评估表的输出文件名映射 |
| `evaluation.subset_quantiles.top20` | 高 PLIE/high reliability 等 top20 子集阈值 |
| `evaluation.subset_quantiles.top10` | 高 PLIE/high accel 等 top10 子集阈值 |
| `evaluation.decile_variables` | 生成 decile/monotonicity 诊断的变量 |
| `evaluation.decile_bins` | decile 分桶数量 |
| `evaluation.top_transition_count` | by-transition 评估保留的最高频 transition 数 |
| `evaluation.output_nan_or_inf_ratio_max` | 输出 sanity check 中允许的 NaN/Inf 最大比例 |
| `evaluation.state_direction_pass_ratio` | state 1/5 方向逻辑检查通过比例 |

默认 `evaluation.table_files` 包含：

| 表名 | 默认文件 |
|---|---|
| `overall_metrics` | `overall_metrics.csv` |
| `quantile_calibration_metrics` | `quantile_calibration_metrics.csv` |
| `by_state_metrics` | `by_state_metrics.csv` |
| `by_transition_metrics` | `by_transition_metrics.csv` |
| `decile_metrics` | `decile_metrics.csv` |
| `monotonicity_metrics` | `monotonicity_metrics.csv` |
| `conditional_subset_metrics` | `conditional_subset_metrics.csv` |
| `rolling_latest_monitoring` | `rolling_latest_monitoring.csv` |
| `retrain_monitoring` | `retrain_monitoring.csv` |
| `output_checks` | `output_checks.csv` |
| `model_coefficients` | `model_coefficients.csv` |
| `walk_forward` | `walk_forward_metrics.csv` |

### 4.9 `walk_forward`

| 参数 | 说明 |
|---|---|
| `walk_forward.enabled` | 是否执行 walk-forward 验证 |
| `walk_forward.train_months` | 每个 fold 的训练窗口月数 |
| `walk_forward.validation_months` | 每个 fold 的验证窗口月数 |
| `walk_forward.step_months` | fold 滚动步长月数 |
| `walk_forward.max_folds` | 最多 fold 数 |

### 4.10 `streaming`

| 参数 | 说明 |
|---|---|
| `streaming.price_frequency_min` | 价格更新频率，用于运行约定 |
| `streaming.liq_frequency_min` | 清算/HMM source 更新频率，用于运行约定 |
| `streaming.max_liq_feature_age_min` | asof merge 最大允许 source age |
| `streaming.output_store` | 在线推理 payload 累积输出 CSV |
| `streaming.require_hmm_columns` | 在线清算更新是否强制要求 HMM state/posterior 字段 |

### 4.11 `monitoring`

| 参数 | 说明 |
|---|---|
| `monitoring.enabled` | 是否启用监控配置 |
| `monitoring.rolling_windows_days` | rolling latest monitoring 的窗口天数 |
| `monitoring.primary_window_days` | 重训判断优先使用的 rolling window |
| `monitoring.coverage_error_watch` | coverage 误差超过该值进入 watch |
| `monitoring.coverage_error_retrain` | coverage 误差超过该值建议 retrain |
| `monitoring.improvement_vs_null_watch_pct` | 相对 null 改善低于该值进入 watch |
| `monitoring.improvement_vs_null_retrain_pct` | 相对 null 改善低于该值建议 retrain |
| `monitoring.note` | 运行说明 |

### 4.12 `retraining`

| 参数 | 说明 |
|---|---|
| `retraining.cadence` | 重训节奏说明，默认 monthly |
| `retraining.min_days_between_full_retrains` | 两次 full retrain 最小间隔天数 |
| `retraining.run_walk_forward_on_retrain` | 重训时是否保留 walk-forward 运行约定 |
| `retraining.trigger_on_retrain_monitoring_status` | 哪个监控状态触发提前重训 |
| `retraining.default_command` | 定期任务推荐命令文本 |
| `retraining.forced_command` | 强制重训推荐操作文本 |
| `retraining.note` | 重训说明 |

### 4.13 `reports`

| 参数 | 说明 |
|---|---|
| `reports.max_points` | HTML 时间序列最大采样点数，仅影响报告大小和渲染速度 |
| `reports.state_background_max_segments` | HMM 背景色最多使用的状态段采样数 |
| `reports.rolling_window_report` | feature statistics 中 rolling mean/std 窗口 |
| `reports.evaluation_rolling_window` | model evaluation 中 PLIE rolling mean 窗口 |
| `reports.table_preview_rows` | HTML 表格最多展示行数 |
| `reports.histogram_bins` | feature histogram bins |
| `reports.state_duration_bins` | state duration histogram bins |
| `reports.theme` | 报告主题说明 |
| `reports.pages.index` | 首页 HTML 文件名 |
| `reports.pages.plie_price` | PLIE x Price 页面文件名 |
| `reports.pages.hmm_state` | HMM State 页面文件名 |
| `reports.pages.feature_statistics` | Feature Statistics 页面文件名 |
| `reports.pages.model_evaluation` | Model Evaluation 页面文件名 |
| `reports.feature_statistics_variables` | feature statistics 页面展示的变量列表 |

### 4.14 `outputs`

| 参数 | 说明 |
|---|---|
| `outputs.latest_summary_fields` | `latest_summary.json` 中保留的字段 |
| `outputs.agent_input_columns` | Agent payload 基础字段；代码会按 `features.horizons_min` 自动追加 `plie_passive_<h>m_bps` 并保留 `plie_main_bps` |
| `outputs.bar_output_base_columns` | 10m 输出 CSV 的基础字段；代码会按 horizons 自动追加 PLIE horizon 字段和 `plie_main_bps` |

实时 Agent 输入不要包含 `ret_*`、`plie_aligned_ret_*`、`plie_residual_*`、`plie_absorption_*`，这些字段需要未来 horizon 成熟后才能计算。

### 4.15 `storage`

| 参数 | 说明 |
|---|---|
| `storage.max_10m_output_rows` | 10m broadcast CSV 最多保存行数；source-clock 历史不裁剪 |
| `storage.note` | 存储说明 |

## 5. 命令语义

| command | 作用 |
|---|---|
| `all` | 执行训练、评估、检查、walk-forward，并按配置生成 HTML |
| `train` | 只执行训练和评估产物生成 |
| `infer` | 使用 `paths.model_artifact` 和 `paths.input_csv` 做批量推理 |
| `monitor` | 基于 `paths.train_source_predictions` 刷新评估表和 rolling monitoring，不重训 |
| `report` | 基于已有预测和评估表重新生成 HTML |
| `validate` | 基于 `paths.input_csv` 做手动未来函数/数据检查 |
| `scheduled-retrain` | 按 `retraining` 与 `monitoring` 配置决定是否 full retrain |

强制重训方式：

```yaml
runtime:
  force_retrain: true
```

然后运行：

```bash
python -m plie_pic_project.feature_plie_pic scheduled-retrain
```

如果只想刷新监控表但不生成 HTML：

```yaml
runtime:
  generate_html: false
```

然后运行：

```bash
python -m plie_pic_project.feature_plie_pic monitor
```

## 6. 输出检查顺序

第一次运行建议检查：

1. `paths.post_training_checks` 对应 JSON 中 critical checks 是否通过。
2. `evaluation.table_files.quantile_calibration_metrics` 对应 CSV 中 coverage 是否接近 `features.quantile`。
3. `evaluation.table_files.retrain_monitoring` 对应 CSV 状态是否为 `ok` 或 `watch`。
4. `reports.pages.index` 对应 HTML 是否可打开。

## 7. Python API

```python
from plie_pic.config import load_config
from plie_pic.streaming import OnlinePLIEEngine

cfg = load_config("config/config.yaml")
engine = OnlinePLIEEngine(cfg)

payload = engine.update_price_data(new_price_df)
payload = engine.update_liquidation_state_data(new_liq_hmm_df)
```

`new_liq_hmm_df` 必须包含 `schema.hmm_state_col` 和 `schema.posterior_cols` 中的 HMM 字段，除非将 `streaming.require_hmm_columns` 设为 `false`。

## 8. 测试

在仓库根目录运行：

```bash
PYTHONPATH=plie_pic_project/src pytest -q plie_pic_project/tests
```

