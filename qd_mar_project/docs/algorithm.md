# QD-MAR v5 算法设计文档

> QD-MAR = Quantile-Calibrated Denoised Market Absorption Rate  
> v5 版本重点：保留 event-level QD-MAR 与 staleness-aware memory；保留 v4 的 PLIE-only path context；`path_absorption_dashboard.html` 不再降采样，完整展示每一个 source-clock path 更新点。

---

## 1. 算法总体目标

本项目基于上游 PLIE-PIC 输出构造 BTC 期货清算压力的市场响应特征体系。PLIE-PIC 的职责是估计清算 forced flow 理论上应当给价格施加的被动方向性压力；QD-MAR 的职责是判断真实市场对这股压力的响应：

```text
实际市场是服从 PLIE、放大 PLIE、部分吸收 PLIE、完全吸收 PLIE，
还是反向接管 PLIE？
如果 PLIE 本身中性或混合，价格是否由主动交易力量主导？
```

本项目不把 `actual - PLIE` 机械解释为纯主动交易力量。它只把该残差作为市场响应证据，因为其中还混有：

- PLIE 估计误差；
- 价格本身的高噪声；
- 外生冲击和宏观/现货/ETF 流；
- 流动性真空；
- 数据延迟；
- 清算统计口径噪声；
- HMM regime 不确定性。

因此，QD-MAR 使用四层结构降低噪声：

```text
1. Context gate：先判断 PLIE 是否有方向、是否可靠。
2. Volatility-denoising：用 past-only robust sigma 降低价格噪声影响。
3. Quantile calibration：用 train-only empirical CDF 把 actual response 转成可比 percentile。
4. Matured memory / path features：只把成熟短期裁决和已发生路径证据输入 Agent。
```

---

## 2. 金融逻辑

### 2.1 PLIE 与 QD-MAR 的角色分工

`plie_main_bps` 默认等于 `plie_passive_30m_bps`，表示当前清算 forced flow 在 30m horizon 上对价格施加的被动冲击基线。它不是最终价格预测器，而是后续吸收率与主动交易 residual 的比较基准。

QD-MAR 不预测“价格会不会涨跌”，而是回答：

```text
给定当前清算压力，市场实际响应是否符合清算压力？
```

### 2.2 方向约定

- `plie_direction = +1`：空头清算占优，空头被迫买回，形成向上压力。
- `plie_direction = -1`：多头清算占优，多头被迫卖出，形成向下压力。
- `plie_direction = 0`：清算方向不足或多空接近平衡。

任何吸收率计算都必须遵守这个方向约定。

### 2.3 为什么必须区分 event-level 与 path-level

event-level absorption 适合判断：

```text
某个 PLIE event 之后 20m/30m/60m 的短期价格裁决。
```

但有些真正重要的市场状态是路径级的：

```text
过去 6h/12h/24h/48h 内清算压力持续向下，
价格却不跌甚至上涨。
```

这种场景未必每个 30m 都产生显著 `reversal_takeover`，但从路径看是非常明确的强吸收 / 压力拒绝。因此 v2 以后增加 path-level episode absorption，v4/v5 将其整理为 PLIE-only context + price-response label。

---

## 3. 输入数据

默认输入：

```text
data/input/plie_predictions_source.csv
```

关键字段：

| 字段 | 含义 |
|---|---|
| `time` | source-clock 时间戳 |
| `price` | 当前价格 |
| `split` | train / validation / test |
| `plie_passive_20m_bps` | signed effective 20m PLIE |
| `plie_passive_30m_bps` | signed effective 30m PLIE |
| `plie_passive_60m_bps` | signed effective 60m PLIE |
| `plie_passive_*_bps_mag_raw` | 未乘 reliability 的 raw q65 PLIE magnitude |
| `plie_main_bps` | 主 30m PLIE，等于 `plie_passive_30m_bps` |
| `plie_direction` | 清算压力方向 |
| `plie_reliability` | PLIE 解释可信度 |
| `plie_intensity` | 清算压力强度 |
| `plie_phase` | neutral / normal / accelerating / mature / early_strong_entry |
| `hmm_state` | 上游 HMM hard state |
| `liq_entropy` | HMM posterior 熵 |
| `age_in_state_source` | source-clock hard state 年龄 |
| `ret_20m_bps` / `ret_30m_bps` / `ret_60m_bps` | 离线训练/评价用未来收益 label |

注意：`ret_*` 在 event time 不可用于实时 Agent，只能在 `event_time + horizon` 后作为 matured label 使用。

---

## 4. 严格时间因果规则

### 4.1 Event-level absorption maturity

对于 event time \(t\) 与 horizon \(h\)：

\[
R_{t,h}=10000\log(P_{t+h}/P_t)
\]

因此：

```text
available_time = event_time + horizon
```

实时 Agent 在时间 \(T\) 只能使用：

```text
available_time <= T
```

的 event-level absorption。

### 4.2 Path-level absorption availability

Path-level 特征在当前时间 \(T\) 只使用过去窗口：

\[
[T-W,T]
\]

因此：

```text
available_time = T
```

它不是未来标签。它是已发生路径的状态描述，可实时输入 Agent。

### 4.3 Calibration / sigma no-leakage

- empirical CDF calibrator 只在 `train` split 拟合；
- rolling MAD sigma 使用 shift，将当前 path return 或当前 event label 移出估计窗口；
- train / validation / test 按时间顺序，不随机打乱；
- HMM posterior 必须来自 filtered inference，不使用未来 smoothed posterior。

---

## 5. Event-level QD-MAR

### 5.1 核心变量

对每个 horizon \(h \in \{20m,30m,60m\}\)：

\[
R_{t,h}=ret_{h,bps}
\]

\[
d_t=plie\_direction_t
\]

\[
B^{raw}_{t,h}=plie\_passive\_h\_bps\_mag\_raw
\]

\[
P^{eff}_{t,h}=plie\_passive\_h\_bps
\]

\[
Y_{t,h}=d_t R_{t,h}
\]

其中 \(Y\) 是实际收益沿 PLIE 方向对齐后的 actual response。

### 5.2 past-only volatility scale

代码中先构造：

```text
mar_sigma_past_20m_bps
mar_sigma_past_30m_bps
mar_sigma_past_60m_bps
```

使用 robust rolling MAD，且 shift 出当前观测，防止当前大波动降低自身标准化分数。

### 5.3 Directional context gate

`directional_core` 条件：

```text
plie_direction != 0
plie_reliability >= absorption.reliability_min
raw PLIE SNR >= absorption.snr_min
Braw >= horizon.b_min_bps
```

其中：

\[
SNR^{raw}_{t,h}=\frac{B^{raw}_{t,h}}{\sigma^{past}_{t,h}+\epsilon}
\]

其他 context：

| `response_context` | 含义 |
|---|---|
| `directional_core` | PLIE 有方向、有可靠性、有 SNR，可计算正式 directional absorption |
| `weak_directional_context` | PLIE 有方向，但不足以作为强 directional evidence |
| `true_neutral_plie` | `plie_direction == 0` |
| `low_quality_plie` | PLIE 有方向或幅度，但质量不足，转入低质量主动响应分支 |

### 5.4 Raw transmission ratio

Directional core 下：

\[
TR^{raw}_{t,h}=\frac{Y_{t,h}}{B^{raw}_{t,h}+\epsilon}
\]

\[
A^{raw}_{t,h}=1-TR^{raw}_{t,h}
\]

这些 raw ratio 主要用于研究与诊断，不建议直接输入 Agent，因为 BTC 短期价格噪声远大于典型 PLIE magnitude。

### 5.5 Volatility-denoised scores

组合尺度：

\[
S_{t,h}=\sqrt{(B^{raw}_{t,h})^2+(\lambda_\sigma\sigma^{past}_{t,h})^2+\epsilon}
\]

同向放大：

\[
Z^{amp}_{t,h}=\frac{Y_{t,h}-B^{raw}_{t,h}}{S_{t,h}}
\]

吸收：

\[
Z^{abs}_{t,h}=\frac{B^{raw}_{t,h}-Y_{t,h}}{S_{t,h}}
\]

反向接管：

\[
Z^{takeover}_{t,h}=\frac{-Y_{t,h}}{S_{t,h}}
\]

### 5.6 Train-only quantile calibration

对 `directional_core` 样本，使用 train split 中的 empirical CDF：

\[
U_{t,h}=F^{train}_{bucket}(TR^{raw}_{t,h})
\]

若细 bucket 样本不足，逐级 fallback：

```text
(horizon, state_severity_bucket, plie_phase_group, vol_regime)
(horizon, state_severity_bucket, vol_regime)
(horizon, vol_regime)
(horizon)
```

主吸收分数：

\[
absorption\_score^q_{t,h}=100(1-U_{t,h})
\]

解释：

| 数值 | 含义 |
|---:|---|
| 高 \(U\) / 低 score | 同向传导或放大 |
| 中间 | 普通响应 / 噪声 |
| 低 \(U\) / 高 score | 吸收或反向接管 |

### 5.7 Event-level labels

Directional core 标签：

| label | 触发逻辑 | 金融含义 |
|---|---|---|
| `passive_amplification` | \(U \ge 0.85\) 且同向超基线 | 清算压力被同向放大 |
| `baseline_transmission` | \(0.65 \le U < 0.85\) | 接近 PLIE q65 传导 |
| `normal_response` | \(0.35 \le U < 0.65\) | 正常响应，不足以判定强吸收/放大 |
| `partial_absorption` | \(0.15 \le U < 0.35\) | 价格顺向但弱于基线 |
| `full_absorption_stall` | 低 \(U\) 且 \(Y\) 接近 0 | 压力存在但价格不推进 |
| `reversal_takeover` | \(U < 0.15\) 且 \(Y < 0\) | 实际价格反向于 PLIE |

Weak directional candidate 标签：

```text
weak_directional_amplification_candidate
weak_directional_transmission_candidate
weak_directional_stall_candidate
weak_directional_rejection_candidate
weak_directional_uncertain
```

Neutral / low-quality 标签：

```text
neutral_low_active_move
neutral_normal_active_move
neutral_strong_active_move
neutral_extreme_active_move

low_quality_low_active_move
low_quality_normal_active_move
low_quality_strong_active_move
low_quality_extreme_active_move
```

---

## 6. Absorption curve

Event-level 产生 20m/30m/60m 三个成熟响应。曲线字段：

```text
u20, u30, u60
mar_curve_label
```

用于判断：

| curve label | 含义 |
|---|---|
| `persistent_cascade_transmission` | 多 horizon 都高传导 |
| `fast_to_sustained_transmission` | 20m 传导且 30/60m 维持 |
| `delayed_transmission` | 初期弱，60m 后传导 |
| `delayed_absorption` | 初期传导，后续被吸收 |
| `fast_absorption_or_takeover` | 短期即被吸收/接管 |
| `persistent_reversal_takeover` | 多 horizon 都反向接管 |
| `mixed_or_noise` | 形态不稳定 |

---

## 7. Staleness-aware rolling memory

由于 `directional_core` 并不是每个时点都有，直接 EWM 会产生 stale signal。代码新增：

\[
freshness=\exp\left(-\ln(2)\cdot\frac{age\_hours}{halflife}\right)
\]

其中 halflife 默认 24h。

Stale-aware score：

\[
mar\_abs\_score\_staleaware
=
50+freshness\cdot(raw\_ewm-50)
\]

Stale-aware active force：

\[
mar\_active\_force\_staleaware
=
freshness\cdot raw\_active\_force\_ewm
\]

这样当很久没有新的 directional-core event 时，Agent 不会继续相信陈旧的 absorption signal。

---

## 8. Path-level / episode-level QD-MAR

### 8.1 目的

Path absorption 用于识别 event-level 不敏感的持续路径机制：

```text
持续清算压力穿透；
持续清算压力被拒绝；
慢速强吸收；
路径级反向接管；
混合清算压力下的主动突破；
清算压力中性但价格主动大幅运动。
```

### 8.2 窗口与更新频率

配置窗口：

```text
6h / 12h / 24h / 48h
```

`window_hours` 是回看窗口长度，不是更新频率。

当前输入为小时级 source-clock，因此 path feature 每个 source-clock 行都重新计算一次。v5 的 `path_absorption_dashboard.html` 不再降采样，完整显示每小时更新点。

### 8.3 Path variables

窗口 \(W\) 内：

\[
R^{path}_{T,W}=10000\log(P_T/P_{T-W})
\]

\[
C^{raw}_{T,W}=\sum_{i\in[T-W,T]}d_iB^{raw}_{i,30m}
\]

\[
G^{raw}_{T,W}=\sum_{i\in[T-W,T]}|d_iB^{raw}_{i,30m}|
\]

\[
C^{eff}_{T,W}=\sum_{i\in[T-W,T]}plie\_passive\_30m\_bps_i
\]

路径方向：

\[
d^{path}_{T,W}=sign(C^{eff}_{T,W})
\]

如果 effective sum 为 0，fallback 到 raw signed sum。

净方向强度：

\[
B^{path}_{T,W}=|C^{raw}_{T,W}|
\]

方向一致性：

\[
Consistency_{T,W}=\frac{|C^{raw}_{T,W}|}{G^{raw}_{T,W}+\epsilon}
\]

路径 aligned response：

\[
Y^{path}_{T,W}=d^{path}_{T,W}R^{path}_{T,W}
\]

传导比例：

\[
TR^{path}_{T,W}=\frac{Y^{path}_{T,W}}{B^{path}_{T,W}+\epsilon}
\]

路径波动：

\[
\sigma^{path,past}_{T,W}
\]

由 shifted rolling MAD 估计。

路径 divergence：

\[
D^{path}_{T,W}
=
-\frac{Y^{path}_{T,W}}
{\sqrt{(B^{path}_{T,W})^2+(\lambda\sigma^{path,past}_{T,W})^2+\epsilon}}
\]

路径 absorption score：

\[
A^{path}_{T,W}=100\cdot sigmoid(D^{path}_{T,W})
\]

### 8.4 PLIE-only path context

v4/v5 把 `path_context` 明确设计成只看 PLIE / liquidation pressure，不看价格：

| path_context | 触发逻辑 | 含义 |
|---|---|---|
| `path_directional_core` | 有净方向，方向一致性、reliability、SNR 达标 | 高质量方向性累计清算压力 |
| `path_directional_weak` | 有净方向，但质量弱于 core | 有方向但证据较弱 |
| `path_mixed_pressure` | gross PLIE 活动不低，但多空方向抵消 | 混合清算压力 / 双边冲突 |
| `path_neutral_pressure` | PLIE 压力很低或没有有效净压力 | 清算压力中性或不足 |

这样：

```text
path_context = PLIE 环境
path_label   = 价格响应
```

### 8.5 Path labels

Directional context 下：

| path_label | 含义 |
|---|---|
| `path_cascade_transmission` | 价格顺累计 PLIE 方向走得很远，路径级级联/穿透 |
| `path_baseline_transmission` | 价格顺 PLIE 方向，接近正常传导 |
| `path_partial_absorption` | 顺 PLIE 但弱于累计压力 |
| `path_full_absorption_stall` | 有压力但路径接近停滞 |
| `path_pressure_rejection` | 价格反向于累计 PLIE，压力被拒绝 |
| `path_reversal_takeover` | 价格强烈反向，路径级主导权切换 |

Mixed context 下：

| path_label | 含义 |
|---|---|
| `path_mixed_active_breakout_up` | 混合清算压力下强主动上行突破 |
| `path_mixed_active_breakout_down` | 混合清算压力下强主动下行突破 |
| `path_normal_mixed_active_breakout_up` | 混合压力下中等主动上行 |
| `path_normal_mixed_active_breakout_down` | 混合压力下中等主动下行 |
| `path_mixed_pressure_chop` | 混合压力且价格无清晰突破 |

Neutral context 下：

| path_label | 含义 |
|---|---|
| `path_active_dominance_up` | 清算压力中性但价格强主动上行 |
| `path_active_dominance_down` | 清算压力中性但价格强主动下行 |
| `path_normal_active_dominance_up` | 清算压力中性，价格中等主动上行 |
| `path_normal_active_dominance_down` | 清算压力中性，价格中等主动下行 |
| `path_quiet_no_pressure` | 清算压力弱，价格也安静 |

### 8.6 Active dominance in neutral/mixed context

非 directional context 下，不能谈 directional absorption。代码改为计算：

\[
ActiveZ_{T,W}=\frac{|R^{path}_{T,W}|}{\sigma^{path,past}_{T,W}+\epsilon}
\]

\[
ActiveDominanceScore_{T,W}
=
\tanh\left(
clip\left(\frac{ActiveZ-0.75}{3.0-0.75},0,2\right)
\right)
\]

价格方向版本：

\[
ActiveDominancePriceScore=sign(R^{path})\cdot ActiveDominanceScore
\]

---

## 9. Path absorption 与 market response 的关系

| 层级 | 时间尺度 | 可用性 | 作用 |
|---|---|---|---|
| Current PLIE | 当前 source-clock | 实时 | 当前 forced-flow 压力源 |
| Market response | 20/30/60m event | event+h 后成熟 | 最近短期清算冲击裁决 |
| Absorption curve | 20/30/60m 组合 | event+60m 后成熟 | 冲击形态 |
| Path absorption | 6/12/24/48h path | 当前可用 | 路径级背景状态 |

组合解释：

| Path label | Market response | 综合解释 |
|---|---|---|
| `path_cascade_transmission` | `passive_amplification` | 路径与短期都在穿透，HPEM 强证据 |
| `path_cascade_transmission` | `partial_absorption` | 路径还在传导，但短期开始被接住 |
| `path_pressure_rejection` | `partial_absorption` / `reversal_takeover` | 路径和短期都吸收，RHA 增强 |
| `path_pressure_rejection` | `passive_amplification` | 路径拒绝但短期重试穿透，拉锯/AMB |
| `path_reversal_takeover` | `reversal_takeover` | 路径和短期都反向，主导权切换强证据 |
| `path_mixed_pressure_chop` | `normal_response` / `weak_directional_uncertain` | 混合压力且短期不清晰，AMB/no-trade |
| `path_active_dominance_up/down` | `neutral_strong_active_move` / `neutral_extreme_active_move` | 非清算主导的主动行情，偏 VT / 外生冲击 |
| `path_quiet_no_pressure` | `neutral_low_active_move` | 清算弱、价格也弱，RC / low activity |

---

## 10. Agent 输入

推荐默认 Agent 输入来自 `absorption_memory.csv`：

### Event-level staleness-aware memory

```text
mar_abs_score_q_staleaware_ewm_6_30m
mar_active_force_aligned_staleaware_ewm_6_30m
mar_active_force_price_staleaware_ewm_6_30m
mar_directional_core_freshness_30m
mar_directional_quality_ewm_6_30m
mar_takeover_count_12_30m
mar_amplification_persistence_6_30m
mar_absorption_persistence_6_30m
```

### Curve memory

```text
mar_curve_label_last
mar_curve_label_mode_6
```

### Neutral event memory

```text
mar_neutral_active_strength_evidence_ewm_6_30m
mar_neutral_context_persistence_12_30m
```

### Path memory

```text
mar_episode_abs_score_12h
mar_episode_abs_score_24h
mar_episode_abs_score_48h
mar_episode_pressure_rejection_score_12h
mar_episode_pressure_rejection_score_24h
mar_episode_pressure_rejection_score_48h
mar_episode_active_force_aligned_24h
mar_episode_active_force_price_24h
mar_episode_active_z_24h
mar_episode_active_dominance_score_24h
mar_episode_active_dominance_price_score_24h
mar_episode_liq_neutrality_score_24h
mar_episode_quality_24h
mar_episode_context_24h
mar_episode_label_24h
```

---

## 11. 评价体系

### 11.1 No-leakage validation

输出：

```text
output/reports/validation_results.csv
```

检查：

- timestamp monotonicity；
- split order；
- available_time；
- memory as-of；
- agent input 中是否含未来字段名；
- path available_time 是否等于当前 time。

### 11.2 PLIE q65 consistency

在 directional-core 样本中检查：

\[
P(Y\le B^{raw})\approx 0.65
\]

### 11.3 Label distribution / combo counts

输出：

```text
path_context_distribution.csv
path_label_proportions.csv
path_context_label_combo_counts.csv
```

尤其 `path_context_label_combo_counts.csv` 用于确认每种合法 context-label 组合是否有足够样本量。

### 11.4 Financial consistency

检查：

- `reversal_takeover` 是否更容易对应 HMM state exit；
- `passive_amplification` 是否更偏 continuation；
- `path_pressure_rejection` 是否更支持 RHA；
- `path_cascade_transmission` 是否更支持 HPEM；
- `path_active_dominance_*` 是否更多出现在 PLIE neutral / mixed context。

---

## 12. HTML 可视化

v5 输出：

```text
index.html
quality_assessment.html
market_response_dashboard.html
calibration.html
curve_dashboard.html
path_absorption_dashboard.html
scenario_examples.html
rolling_monitoring.html
feature_statistics.html
extreme_events.html
```

其中 `path_absorption_dashboard.html` 特别说明：

```text
不降采样；
完整显示每一个 source-clock path absorption 更新点；
当前正式输入数据为小时级 source-clock，因此 path_context/path_label 是逐小时更新；
6h/12h/24h/48h 是回看窗口，不是更新频率。
```

---

## 13. 算法局限性

1. PLIE residual 不是纯主动交易力。
2. Event-level absorption 对慢速路径吸收不敏感，因此必须结合 path absorption。
3. Path absorption 可能滞后于突发新闻或外生冲击。
4. Neutral / mixed active dominance 不能解释主动流来源，只能说明清算压力不是主要解释因子。
5. 如果上游 liquidation 数据口径变化，PLIE q65 calibration 与 HMM state 语义都需要重审。
6. HTML 全量 path 点展示会增大文件体积，但这是 path label 逐小时审查所必需的。



## Addendum: Multiscale path absorption wide output

The pipeline now persists `output/features/path_absorption_multiscale.csv` in addition to the long-format `path_absorption.csv`.

### Purpose

`path_absorption.csv` is a long audit table with one row per `(time, window_hours)` pair. It is ideal for detailed diagnostics. `path_absorption_multiscale.csv` is a wide, model-ready table with one row per `time` and simultaneous 6h/12h/24h/48h path evidence columns. This file is designed for downstream state engines such as HSMM / IO-HSMM / Agent memory, where each online timestamp should expose all path scales in one row.

### Causality

For each timestamp `T`, every `*_6h`, `*_12h`, `*_24h`, and `*_48h` field is copied from a path feature computed only over `[T-W, T]`. Therefore `available_time = time`. No future price, future PLIE, future HMM state, or future market-response label is used.

### Output fields

The multiscale table contains `time`, `available_time`, and the following fields for each window suffix `6h`, `12h`, `24h`, and `48h`:

- `path_context_*`: PLIE-only pressure context.
- `path_label_*`: price-path response under that context.
- `path_absorption_score_*`: 0-100 path absorption / rejection score.
- `path_pressure_rejection_score_*`: bounded directional rejection evidence.
- `path_active_dominance_score_*`: bounded neutral/mixed active-dominance evidence.
- `path_transmission_ratio_*`: path-aligned response divided by cumulative raw PLIE net pressure.
- `path_direction_consistency_*`: cumulative net raw PLIE pressure divided by cumulative gross raw PLIE pressure.
- `path_cascade_score_*`: bounded directional cascade/transmission evidence.
- `path_data_quality_*`: causal maturity/availability indicator for that path window.
- `path_signal_clarity_*`: PLIE-context clarity independent of price response.
- `path_activity_level_*`: price-path activity z-score used by neutral/mixed active-dominance logic.


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
