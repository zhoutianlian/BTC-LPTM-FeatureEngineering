# PLIE-PIC 代码架构设计文档

## 1. 项目目录结构

```text
plie_pic_project/
  config/config.yaml
  data/input/hmm_state.csv.zip
  src/plie_pic/
    config.py
    io.py
    utils.py
    validation.py
    features.py
    model.py
    train.py
    inference.py
    streaming.py
    evaluation.py
    visualization.py
    cli.py
  tests/
    test_no_future_leakage.py
    test_features.py
  docs/
    manual.md
    code_design.md
    algorithm.md
    plie_feature_engineering.md
  outputs/
    features/
    predictions/
    evaluation/
    checks/
  reports/html/
  models/
  logs/
  run_pipeline.py
```

## 2. 模块职责

| 模块 | 职责 |
|---|---|
| `config.py` | 读取 YAML 配置，统一路径管理 |
| `io.py` | CSV/JSON/model artifact 读写 |
| `utils.py` | UTC 转换、past-only robust z-score、pinball loss、采样等工具 |
| `validation.py` | 数据质量与未来函数检查 |
| `features.py` | source-clock 折叠、PLIE 特征、label、10m 广播 |
| `model.py` | 机制约束 quantile passive impact curve 模型 |
| `train.py` | 离线训练、walk-forward、保存模型和输出 |
| `inference.py` | 批量推理与 latest Agent payload |
| `streaming.py` | 流式数据增量更新与在线推理 |
| `evaluation.py` | 评价指标、by-state/by-transition/decile/output sanity checks |
| `visualization.py` | 生成交互式 HTML 报告 |
| `cli.py` | 命令行入口 |

## 3. 核心类和函数

### `ProjectConfig`

读取配置并把所有相对路径解析为项目根目录下的绝对路径。

### `NoFutureLeakageChecker`

检查：

- 时间戳单调性
- 清算 source time 不晚于价格 bar time
- `liq_feature_age_min` 与时间差一致
- HMM posterior 非负且和为 1
- source-clock 去重
- train/validation/test 时间切分
- model feature 不包含 future label
- Agent input 不包含 realized outcome

### `build_source_clock_frame`

把 10m forward-filled frame 折叠为唯一 source-clock liquidation snapshot。

### `build_plie_features`

在 source clock 上计算：

- `plie_raw_signed_up`
- `plie_hmm_severity_coord`
- `plie_fused_pressure_coord`
- `plie_direction`
- `plie_force_up`
- `plie_intensity`
- `plie_accel_pos`
- `plie_transition_type`
- `plie_transition_severity`
- `plie_strong_entry`
- `plie_reliability`
- 多 horizon label

### `QuantileImpactCurveModel`

低自由度 quantile 模型。系数非负，保证 PLIE 随 intensity、acceleration、strong entry 等机制变量增加而增加或被 transition severity 合理调节。

### `MultiHorizonPLIEPICModel`

为 20m、30m、60m 各训练一个 constrained quantile impact curve。

## 4. 数据流向

```text
hmm_state.csv.zip
  -> read_input_frame
  -> NoFutureLeakageChecker pre-check
  -> build_source_clock_frame
  -> build_plie_features
  -> assign_chronological_split
  -> train MultiHorizonPLIEPICModel
  -> predict source-clock PLIE
  -> broadcast_source_predictions_to_10m
  -> evaluation / reports / agent payload
```

## 5. 训练流程

1. 读取输入。
2. 做 pre-feature leakage checks。
3. 折叠 source clock。
4. 构建 PLIE 特征和 label。
5. 按时间顺序切分 train/validation/test。
6. 在 train split 上训练模型。
7. 保存模型。
8. 对全样本生成 out-of-sample 标识下的预测。
9. 生成评价指标、walk-forward、HTML 报告。

## 6. 推理流程

1. 读取保存的 `plie_pic_model.joblib`。
2. 根据新输入重建 source-clock 特征。
3. 调用模型生成 source-clock PLIE。
4. 广播到 10m 执行网格。
5. 输出 latest Agent payload。

## 7. 流式更新流程

```text
new_price_data
  -> update_price_data
  -> merge_asof latest liquidation source
  -> if enough data, infer latest Agent payload

new_liquidation_state_data
  -> update_liquidation_state_data
  -> validate HMM columns exist
  -> update source store
  -> infer latest Agent payload
```

本项目不在流式层中重新训练 HMM。HMM 状态来自上游系统。

## 8. 模型与特征状态保存机制

- 模型：`models/plie_pic_model.joblib`
- 模型摘要：`models/plie_pic_model_summary.json`
- 训练系数：`outputs/evaluation/model_coefficients.csv`
- source 特征：`outputs/features/plie_source_features.csv`
- 10m 输出：`outputs/predictions/plie_predictions_10m.csv`，默认保存最近 60,000 行；完整 source-clock 历史始终保存。

## 9. 错误处理机制

- 缺失关键列：训练前中止。
- source-clock 行为空：中止。
- HMM posterior 不合法：critical check 失败。
- 流式 HMM 列缺失：拒绝推理。
- 模型未训练：推理报错。

## 10. 测试设计

`tests/test_no_future_leakage.py` 覆盖：

- 时间戳和 as-of 对齐
- source-clock 去重
- rolling past-only 边界
- 时间序列切分
- model feature / Agent input 排除未来 label

`tests/test_features.py` 覆盖：

- state 1/5 与 PLIE direction 语义一致
- 多 horizon label 存在


## 11. 本次优化后的评价与可视化变更

`evaluation.py` 新增：

- `quantile_calibration_metrics`：q65 覆盖率、pinball loss、常数 q65 baseline、zero baseline 对比。
- `conditional_subset_metrics`：all、state 1/5、strong entry、PLIE top decile、acceleration top decile、reliability top quintile 的分层表现。
- `monotonicity_metrics`：检查 PLIE magnitude 与 intensity / PLIE decile 的机制单调性。
- `retrain_monitoring`：根据覆盖率与 pinball baseline improvement 给出重训监控状态。

`visualization.py` 调整：

- `plie_price.html` 将 actual 30m return 与 `actual - PLIE residual` 拆成两个独立子图，避免含义混淆与视觉覆盖。
- `hmm_state.html` 第一张主图改为 PLIE 折线，并用 HMM state 作为背景色填充。
- `model_evaluation.html` 增加 q65 coverage 图与新增评价表。

---

## 12. 本版本新增模块：rolling monitoring 与 scheduled retrain

### 12.1 `src/plie_pic/scheduler.py`

新增职责：

- `run_monitoring(cfg, generate_html=True)`：基于既有 `plie_predictions_source.csv` 刷新评价表、rolling latest monitoring 与 HTML 报告，不训练模型。
- `monthly_retrain_if_due(cfg, force=False, generate_html=True)`：检查是否满足月度重训或漂移触发条件，满足时运行 full training pipeline。
- `should_retrain(cfg, force=False)`：返回重训决策对象，包括是否 due、原因、模型文件年龄与 retrain_monitoring 状态。

### 12.2 CLI 新增命令

| 命令 | 作用 |
|---|---|
| `python run_pipeline.py monitor --config config/config.yaml` | 刷新 rolling latest monitoring 与报告，不重训 |
| `python run_pipeline.py monitor --config config/config.yaml --no-report` | 只刷新评价 CSV，不生成 HTML |
| `python run_pipeline.py scheduled-retrain --config config/config.yaml` | 若达到月度周期或 drift trigger，则执行 full retrain |
| `python run_pipeline.py scheduled-retrain --config config/config.yaml --force` | 强制 full retrain |

### 12.3 新增评价输出

| 文件 | 说明 |
|---|---|
| `outputs/evaluation/rolling_latest_monitoring.csv` | 最近 7/14/30/60/90 天 rolling health metrics |
| `outputs/evaluation/scheduled_retrain_decision.json` | scheduled retrain 的执行决策 |

### 12.4 设计边界

- `monitor` 使用已成熟标签，只做后验监控，不作为实时 Agent 输入。
- `scheduled-retrain` 不是后台常驻服务；生产中可由 cron、Airflow、Prefect 或其他调度系统定期调用。
- 10m price update 与 1h liquidation update 仍然只触发在线推理，不触发 full retrain。

---

## 13. 可视化更新

`visualization.py` 新增：

- `_range_selector()`：统一提供 `1D / 1W / 1M / 3M / 6M / 1Y / ALL` 时间按钮。
- `_apply_time_controls()`：给时间序列图添加 range selector 与 range slider。
- `_add_state_legend()`：为 HMM state 背景填充图添加状态颜色图例。

`hmm_state.html` 的主图被拆分为：

1. `PLIE main line with HMM state background`；
2. `Price with HMM state background`；
3. `HMM hard state and confidence`。

PLIE 分布 by state 被放入单独图表，避免 categorical x-axis 与时间 range selector 冲突。
