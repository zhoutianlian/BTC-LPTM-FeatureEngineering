# PLIE-PIC 算法设计文档

## 1. 算法总体目标

PLIE-PIC，全称 **Source-clock Mechanism-constrained Quantile Passive Liquidation Impact Curve**，用于估计：

> 给定当前 BTC 期货清算量、清算方向、HMM 清算压力状态、状态切换与压力增强，如果市场没有出现显著反向主动吸收，则这批 liquidation forced flow 大致应造成的被动价格冲击。

它不是普通 BTC return predictor，也不是直接交易信号。PLIE-PIC 的输出是 **passive liquidation-implied impact baseline**。后续吸收率模块可以用：

```text
actual price move - PLIE passive baseline
```

来反推主动交易力量、吸收、反向接管或同向放大。

本项目的核心设计原则是：

1. 先估计清算 forced flow 本身应该施加给价格的机械压力。
2. 不把主动交易、吸收、价格动量、宏观冲击提前学进 PLIE。
3. 用实际价格相对 PLIE 的偏离解释主动交易力量。
4. 所有特征必须在 source clock 上严格因果生成。
5. 在线 10m / 1h 更新只做推理与监控；full retrain 默认按月执行。

---

## 2. 金融机制与算法设计过程

BTC 期货清算链条可以抽象为：

```text
价格变化
  -> 保证金缓冲被压缩
  -> 脆弱仓位触发清算
  -> 清算转化为 forced flow
  -> forced flow 冲击市场承接
  -> 价格顺向传导 / 被吸收 / 被主动反向接管
  -> 下一轮脆弱性分布被重塑
```

PLIE-PIC 只估计中间的：

```text
清算 forced flow -> passive price pressure
```

而不是估计最终价格。最终价格还受到主动交易力量、现货流、ETF 流、宏观风险偏好、盘口深度、做市库存、期权对冲等多种力量共同影响。

因此，若当前清算压力方向为向上，PLIE 给出 `+8 bps`，但实际 30m 只上涨 `+1 bps`，这不是 PLIE 的失败，而是后续 absorption / residual 模块的研究对象：市场中可能存在主动卖压或上方供给吸收了这批强制买回流。

### 2.1 为什么不直接预测 realized return

若直接训练模型拟合：

\[
r_{t,h}=\log(P_{t+h}/P_t)
\]

模型会自然学习到：

- 清算压力；
- 主动交易；
- 价格趋势；
- 吸收；
- 波动率环境；
- 偶然相关性。

这样模型可能变成普通价格预测器，PLIE residual 就不再能干净地解释主动交易力量。因此 PLIE-PIC 采用受约束的 q65 quantile passive impact curve，而不是无约束均值回归。

### 2.2 为什么使用 source clock

价格数据是 10m bar，清算量特征通常每小时更新一次。如果把同一个 hourly liquidation snapshot forward-fill 到 6 根 10m bar 后再计算 rolling / diff / transition，会把同一个清算观测重复计算，制造伪持续性和伪 acceleration。

因此，本项目所有 liquidation-derived 特征都先折叠到 source clock：

```text
liq_feature_age_min == 0
```

在 source clock 上计算压力、状态切换、压力增强、duration 和模型输入；最后再把 PLIE 输出广播回 10m execution grid。

---

## 3. 输入数据

默认输入：

```text
data/input/hmm_state.csv.zip
```

输入表是已经完成 10m 价格、小时清算特征、HMM filtered posterior/source-clock 对齐后的数据。

关键字段：

| 字段 | 含义 |
|---|---|
| `time` | 10m price bar timestamp |
| `price` | 价格 |
| `fll_cwt_kf` | 多头清算，被迫卖出，向下压力 |
| `fsl_cwt_kf` | 空头清算，被迫买回，向上压力 |
| `total_ls_cwt_kf` | 有效总清算强度 |
| `liq_feature_time` | 清算特征因果可得时间 |
| `liq_feature_age_min` | 当前价格 bar 使用的清算特征年龄 |
| `hmm_state` | 上游 HMM hard state |
| `p_state_1` ... `p_state_5` | 上游 HMM filtered posterior |
| `hmm_conf` | hard state 置信度 |
| `liq_entropy` | HMM posterior entropy |
| `age_in_state_source` | source-clock 状态持续期 |

本项目不重新训练 HMM。HMM state 与 posterior 必须来自上游 filtered inference，而不是未来 smoothed posterior。

---

## 4. 数据预处理

### 4.1 时间标准化

所有时间字段使用：

```python
pd.to_datetime(..., utc=True)
```

并按 `time` / `liq_feature_time` 排序。重复 `liq_feature_time` 默认保留最后一条。

### 4.2 Source-clock 折叠

训练和 source-level inference 只使用：

```text
liq_feature_age_min == features.source_age_zero_value
```

并要求：

```text
hmm_state in {1,2,3,4,5}
```

### 4.3 标签生成

标签只在 source-clock 特征完全生成之后创建。对 horizon `h`：

\[
r_{t,h}=10000\cdot\log(P_{t+h}/P_t)
\]

它只用于训练、评价和后验诊断，不能进入实时 Agent 输入。

---

## 5. PLIE 方向定义

方向约定：

```text
向上压力为正
```

令：

\[
L_t=fll\_cwt\_kf
\]

表示多头清算，被迫卖出，向下压力。

\[
S_t=fsl\_cwt\_kf
\]

表示空头清算，被迫买回，向上压力。

\[
T_t=L_t+S_t
\]

原始清算方向：

\[
u_t=\frac{S_t-L_t}{T_t+\varepsilon}
\]

HMM severity coordinate：

\[
q_t=\frac{2p_{1,t}+p_{2,t}-p_{4,t}-2p_{5,t}}{2}
\]

其中 state 1 / 5 是强压力状态，权重为 2；state 2 / 4 是轻度压力状态，权重为 1；state 3 中性，不进入方向坐标。

融合方向坐标：

\[
c_t=\lambda q_t+(1-\lambda)u_t
\]

默认：

```yaml
features.hmm_posterior_weight_lambda: 0.65
```

方向：

\[
d_t=sign(c_t)
\]

若：

\[
|c_t| < direction\_deadzone
\]

则 `plie_direction = 0`，PLIE 输出接近中性。

---

## 6. PLIE 强度、压力增强与 transition

### 6.1 清算强度 robust z-score

总清算强度先做 log 压缩：

\[
logT_t=\log(1+T_t)
\]

在 source clock 上使用 past-only rolling median / MAD：

\[
z_t=\frac{logT_t - median_t(logT)}{1.4826\cdot MAD_t(logT)+\varepsilon}
\]

使用 softplus 生成非负、连续强度：

\[
m_t=\log(1+e^{z_t})
\]

### 6.2 Signed force 与 intensity

\[
F_t=c_t m_t
\]

\[
I_t=|c_t|m_t
\]

代码字段：

| 字段 | 含义 |
|---|---|
| `plie_force_up` | signed force，向上为正 |
| `plie_intensity` | 非负清算压力强度 |

### 6.3 当前方向上的压力增强

\[
a_t^+=\max(0,d_t(F_t-F_{t-1}))
\]

若当前方向上的清算压力增强，则 `plie_accel_pos > 0`。这表示 forced flow 不只是大，而且正在变得更大。

### 6.4 Strong entry

\[
E_t=1[hmm\_state_t\in\{1,5\},\ hmm\_state_t\ne hmm\_state_{t-1}]
\]

`plie_strong_entry=1` 表示刚进入强压力状态，通常是 liquidation cascade early stage 的更高价值切片。

### 6.5 Transition severity

| Transition | `plie_transition_severity` | 金融解释 |
|---|---:|---|
| `2->1` | 1.0 | 向上压力由轻度进入强势 |
| `4->5` | 1.0 | 向下压力由轻度进入强势 |
| `3->1` | 0.7 | 中性直接进入强向上压力 |
| `3->5` | 0.7 | 中性直接进入强向下压力 |
| `1->2` | 0.3 | 强向上压力降级 |
| `5->4` | 0.3 | 强向下压力降级 |
| `1->5` | -1.0 | 强方向翻转，可靠性应谨慎 |
| `5->1` | -1.0 | 强方向翻转，可靠性应谨慎 |
| other | 0.0 | 无特殊 transition boost |

---

## 7. Reliability 与 phase

`plie_reliability` 是因果可得的质量权重，不是未来预测准确率。它主要由：

- state 是否处于强压力 / 有方向压力；
- HMM confidence；
- posterior entropy；
- liquidation snapshot freshness；
- direction deadzone；

共同决定。

它的用途是降低 neutral、stale、uncertain context 下的 PLIE 输出权重。

`plie_phase` 是解释字段，帮助 Agent 和报告理解当前 PLIE 处于：

- neutral；
- normal；
- accelerating；
- strong_entry；
- stale；

等状态。

---

## 8. 模型输入特征

模型输入保持低自由度、机制约束：

| 字段 | 公式 / 含义 |
|---|---|
| `model_log1p_intensity` | `log1p(plie_intensity)` |
| `model_log1p_accel_pos` | `log1p(plie_accel_pos)` |
| `model_strong_entry` | 是否刚进入 state 1/5 |
| `model_transition_severity` | transition boost / caution score |
| `model_strong_state` | 当前是否为 state 1/5 |

这些特征全部在 source clock 上计算，不使用未来价格、未来状态或未来清算量。

---

## 9. 模型训练逻辑

对每个 horizon 独立训练一条 q65 constrained quantile impact curve。

horizon：

```yaml
features.horizons_min: [20, 30, 60]
```

aligned label：

\[
y_{t,h}^{obs}=d_t\cdot10000\log(P_{t+h}/P_t)
\]

PLIE magnitude 目标：

\[
\hat y_{t,h}^{PLIE}=Q_{0.65}(y_{t,h}^{obs}\mid I_t,a_t^+,E_t,TR_t)
\]

模型形式：

\[
\hat y_{t,h}^{PLIE}
=\beta_{0,h}
+\beta_{1,h}\log(1+I_t)
+\beta_{2,h}\log(1+a_t^+)
+\beta_{3,h}E_t
+\beta_{4,h}TR_t
+\beta_{5,h}StrongState_t
\]

约束：

\[
\beta_{1,h},...,\beta_{5,h}\ge0
\]

最终 magnitude 裁剪为非负：

\[
\hat y_{t,h}^{PLIE}=\max(0, model(x_t))
\]

signed PLIE：

\[
\hat r_{t,h}^{PLIE}=d_t\cdot\hat y_{t,h}^{PLIE}\cdot R_t
\]

其中 `R_t` 是 `plie_reliability`。

工程实现使用平滑 pinball loss 近似 q65 pinball loss，以提高大样本训练收敛稳定性。这不改变 quantile PLIE-PIC 的目标。

---

## 10. 离线训练、在线推理与周期性重训

### 10.1 离线训练

离线训练运行：

```bash
python run_pipeline.py train --config config/config.yaml
```

包括：

1. 读取完整历史输入；
2. source-clock 特征构建；
3. chronological train / validation / test split；
4. constrained q65 model training；
5. source predictions；
6. 10m broadcast predictions；
7. evaluation tables；
8. walk-forward validation；
9. no-future leakage checks。

### 10.2 在线推理

在线推理不重训模型。

价格 10m 更新：

```text
new_price_data
  -> append price
  -> backward merge latest available liquidation/HMM source snapshot
  -> broadcast latest PLIE if available
  -> update latest Agent payload
```

清算/HMM 1h 更新：

```text
new_liquidation_state_data
  -> validate HMM state/posterior columns
  -> append source snapshot
  -> compute source-clock PLIE features
  -> infer PLIE using existing model
  -> broadcast to 10m grid
  -> update latest Agent payload
```

### 10.3 月度重训

不需要每 10m / 1h 重新训练模型。默认 full retrain cadence：

```yaml
retraining.cadence: monthly
retraining.min_days_between_full_retrains: 25
```

推荐执行：

```bash
python run_pipeline.py scheduled-retrain --config config/config.yaml
```

强制重训：

```bash
python run_pipeline.py scheduled-retrain --config config/config.yaml --force
```

触发重训条件：

- 模型 artifact 缺失；
- 距离上次模型更新时间超过 `min_days_between_full_retrains`；
- `outputs/evaluation/retrain_monitoring.csv` 的 `status == retrain_now`。

---

## 11. 评价体系

PLIE-PIC 是 passive impact baseline，不是普通收益预测器。因此评价体系分为六层。

### 11.1 因果与数据质量

必须通过：

- 时间戳单调性；
- source-clock dedup；
- as-of backward alignment；
- model features 不含 `ret_*` / `plie_residual_*` / `plie_absorption_*`；
- Agent inputs 不含未来标签；
- HMM posterior 概率合法。

输出：

```text
outputs/checks/post_training_leakage_checks.json
```

### 11.2 Quantile calibration

核心指标：

\[
Coverage_h=P(y_{t,h}^{aligned}\le |PLIE_{t,h}|)
\]

目标接近：

```text
features.quantile = 0.65
```

解释：若模型是 q65 passive baseline，则实际 aligned response 小于等于 PLIE magnitude 的比例应接近 65%。

文件：

```text
outputs/evaluation/quantile_calibration_metrics.csv
```

### 11.3 Pinball baseline improvement

模型 q65 pinball loss 应优于：

1. 训练集常数 q65 baseline；
2. zero baseline。

若 `improvement_vs_null_pct < 0`，说明模型不如简单常数分位数，需检查或重训。

### 11.4 Mechanism monotonicity

PLIE 本身应满足机制形状：

```text
plie_intensity decile ↑ -> PLIE magnitude ↑
plie_abs_main_bps decile ↑ -> PLIE magnitude ↑
```

实际收益不要求严格单调，因为实际收益包含主动交易和吸收。

文件：

```text
outputs/evaluation/monotonicity_metrics.csv
```

### 11.5 Conditional subsets

重点切片：

- `state_1_5`；
- `strong_entry`；
- `plie_abs_top20`；
- `plie_abs_top10`；
- `accel_top10`；
- `reliability_top20`。

文件：

```text
outputs/evaluation/conditional_subset_metrics.csv
```

### 11.6 Rolling latest monitoring

新增 rolling latest monitoring，用于生产环境持续观察最近窗口是否漂移。

配置：

```yaml
monitoring.rolling_windows_days: [7, 14, 30, 60, 90]
monitoring.primary_window_days: 30
```

输出：

```text
outputs/evaluation/rolling_latest_monitoring.csv
```

该文件按最近 7/14/30/60/90 天、不同 subset、不同 horizon 输出：

- `coverage_actual_le_plie`；
- `coverage_error`；
- `pinball_q65`；
- `improvement_vs_null_pct`；
- `mean_aligned_actual_bps`；
- `transmission_rate`；
- `mean_absorption`；
- `spearman_abs_plie_vs_aligned`。

它只在标签成熟后作为监控使用，不进入实时 Agent。

### 11.7 Retrain monitoring

输出：

```text
outputs/evaluation/retrain_monitoring.csv
```

状态：

| 状态 | 含义 |
|---|---|
| `ok` | 当前校准与基线提升可接受，无需立即重训 |
| `watch` | 可上线但偏弱，应优先安排下次月度重训 |
| `retrain_now` | calibration 或 pinball baseline 明显漂移，应重训或校准 |

默认阈值：

| 条件 | watch | retrain_now |
|---|---:|---:|
| abs coverage error | > 0.05 | > 0.08 |
| improvement_vs_null_pct | < 0.5% | < -1.0% |

---

## 12. 输出变量

核心输出：

| 字段 | 含义 |
|---|---|
| `plie_direction` | +1 向上压力，-1 向下压力，0 中性 |
| `plie_force_up` | signed liquidation force |
| `plie_intensity` | 非负清算压力强度 |
| `plie_accel_pos` | 当前方向上的压力增强 |
| `plie_strong_entry` | 是否刚进入 state 1/5 |
| `plie_transition_type` | transition string，例如 `2->1` |
| `plie_transition_severity` | transition boost / caution score |
| `plie_reliability` | 因果 reliability |
| `plie_phase` | 解释型 phase |
| `plie_passive_20m_bps` | 20m signed passive PLIE |
| `plie_passive_30m_bps` | 30m signed passive PLIE |
| `plie_passive_60m_bps` | 60m signed passive PLIE |
| `plie_main_bps` | 主 PLIE，默认 30m |

后验诊断字段：

| 字段 | 用途 | 是否 Agent 输入 |
|---|---|---|
| `ret_*` | realized future return label | 否 |
| `plie_aligned_ret_*` | aligned realized response | 否 |
| `plie_residual_*` | actual - signed PLIE | 否 |
| `plie_absorption_*` | 后验吸收率诊断 | 否 |

---

## 13. Agent 输入变量

Agent 可使用：

- 当前 PLIE 输出；
- HMM state/conf/entropy；
- state age；
- transition / phase / reliability；
- passive baseline。

Agent 不可使用：

- `ret_*`；
- `plie_aligned_ret_*`；
- `plie_residual_*`；
- `plie_absorption_*`。

这些字段是未来 horizon 成熟后才能计算的标签或后验诊断变量。

---

## 14. 未来函数防控机制

1. 所有 liquidation rolling/diff/transition 在 source clock 上计算。
2. rolling median/MAD 使用当前及过去样本，不使用未来样本。
3. label 在特征生成后创建，不反向进入特征。
4. train/validation/test 按时间切分，不随机打乱。
5. 标准化/拟合统计量只在训练窗口内学习。
6. HMM posterior 必须是 upstream filtered posterior。
7. 10m broadcast 只使用已可得的 `liq_feature_time`。
8. `tests/test_no_future_leakage.py` 覆盖关键检查。

---

## 15. 可视化设计

HTML 页面：

| 页面 | 说明 |
|---|---|
| `index.html` | 项目概览与最新状态 |
| `plie_price.html` | PLIE、价格、实际收益、residual、absorption |
| `hmm_state.html` | HMM state 背景填充到 PLIE 折线图、价格图、状态统计 |
| `feature_statistics.html` | Agent 关键变量统计 |
| `model_evaluation.html` | 量化评价、rolling monitoring、重训状态 |

时间序列图提供：

```text
1D / 1W / 1M / 3M / 6M / 1Y / ALL
```

快速区间选择和 range slider。

---

## 16. 算法局限性

- PLIE 估计的是被动清算冲击基线，不保证实际价格跟随。
- 若 ETF/现货大额流、宏观冲击、期权对冲主导市场，PLIE residual 可能很大。
- 若交易所清算口径变化，模型需要重训并重新审查。
- 极端跳价行情中，价格可能先于清算记录出现，PLIE 更像 echo 而不是 first signal。
- HMM state 质量依赖上游模型，本项目不修正 HMM 语义漂移。
- 为保持 residual 可解释性，PLIE 不应被过度复杂化为全因子收益模型。
