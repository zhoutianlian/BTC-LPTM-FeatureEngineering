# QD-MAR v5 特征工程文档

本文面向没有项目背景的读者，解释 QD-MAR 的核心输出特征、金融含义、计算方法、时间可得性、Agent 应用方式，以及 `path_absorption` 与 `market_response` 的组合关系。

---

## 0. 总览

QD-MAR 的特征分为四层：

| 层级 | 输出位置 | 是否实时输入 Agent | 金融问题 |
|---|---|---:|---|
| PLIE current pressure | `base_context.csv` / 原始输入 | 是 | 当前清算 forced flow 给价格施加什么方向压力？ |
| Event-level market response | `absorption_event_matured.csv` | 只在成熟后通过 memory 输入 | 某次 PLIE event 后 20/30/60m 市场如何裁决？ |
| Rolling matured memory | `absorption_memory.csv` | 是 | 最近成熟的 market response 是否持续吸收、放大、接管？ |
| Path-level absorption | `path_absorption.csv` / memory episode fields | 是 | 过去 6/12/24/48h 的累计清算压力是否被路径服从、吸收、拒绝或主动交易覆盖？ |

核心原则：

```text
PLIE 是压力源；
market_response 是短期成熟裁决；
path_absorption 是路径级状态背景；
Agent 应融合三者，而不是只看单个 label。
```

---

# Part I. PLIE 与基础上下文特征

## 1) `plie_main_bps`

### Financial meaning

主 PLIE 输出，默认等于 `plie_passive_30m_bps`。它表示当前清算 forced flow 在 30m horizon 上理论上给价格施加的 signed passive impact，单位 bps。

它不是最终价格预测值。它是后续 market response 与 absorption 的基准。

### Application

- 当前清算压力主证据；
- Agent 判断当前 forced-flow 是否在向上或向下施压；
- 与 matured actual return 比较，生成 event-level absorption；
- 与 path window 累计，生成 path-level pressure context。

### Calculation

\[
plie\_main\_bps = plie\_passive\_30m\_bps
\]

\[
plie\_passive\_30m\_bps
=
plie\_direction_t\cdot \hat y^{PLIE}_{t,30m}\cdot plie\_reliability_t
\]

其中 `plie_passive_30m_bps_mag_raw` 是未乘 reliability 的 raw q65 reference magnitude。

### Time alignment

source-clock 当前行可得。不是未来函数。

---

## 2) `plie_direction`

### Financial meaning

清算压力方向：

| 值 | 含义 |
|---:|---|
| +1 | 空头清算占优，forced buying，向上压力 |
| -1 | 多头清算占优，forced selling，向下压力 |
| 0 | 多空接近平衡或方向不足 |

### Application

- 将实际收益转成 aligned response；
- 判断 event-level absorption 是否可以做 directional 分支；
- 构造 path-level signed raw PLIE sum。

### Calculation

来自上游 PLIE / HMM 方向融合。QD-MAR 不重新估计方向，只使用输入字段。

---

## 3) `plie_reliability`

### Financial meaning

PLIE 当前作为 liquidation-driven baseline 的可信度。低 reliability 表示当前可能不适合把价格变化解释为清算压力传导或吸收。

### Application

- event-level directional gate；
- path-level quality；
- Agent 权重控制；
- 不应把 low reliability 的小 denominator 当成主动放大。

### Calculation

由上游 PLIE 输出，通常融合状态方向强度、entropy、新鲜度等信息。QD-MAR 直接使用。

---

## 4) `plie_passive_*_bps_mag_raw`

### Financial meaning

没有乘 reliability 的 raw q65 passive impact magnitude。它更适合作为 absorption denominator / reference，因为它保留 PLIE q65 calibration 语义。

### Application

- event-level \(B^{raw}_{t,h}\)；
- path-level `path_raw_plie_total_bps` 与 `path_net_braw_bps`；
- q65 coverage 检查。

### Calculation

由上游 PLIE impact curve 输出，QD-MAR 直接使用：

\[
B^{raw}_{t,h}=plie\_passive\_h\_bps\_mag\_raw
\]

---

# Part II. Event-level Market Response

Event-level 特征是一行 event × horizon。它们只有在：

```text
available_time = event_time + horizon
```

之后才成熟。

---

## 5) `response_context`

### Financial meaning

描述当前 event × horizon 是否适合计算 directional absorption。

### Application

- 决定 market response label 的解释框架；
- Agent 不应把 neutral / low-quality context 的 label 当作 directional absorption。

### Calculation

先计算：

\[
SNR^{raw}_{t,h}=\frac{B^{raw}_{t,h}}{\sigma^{past}_{t,h}+\epsilon}
\]

然后分类：

| context | 条件 | 含义 |
|---|---|---|
| `directional_core` | `plie_direction != 0` 且 reliability、SNR、Braw 达标 | 可以正式计算 directional absorption |
| `weak_directional_context` | 有方向但质量较弱 | 只能作为候选证据 |
| `true_neutral_plie` | `plie_direction == 0` | 清算方向中性 |
| `low_quality_plie` | PLIE 有方向/幅度但质量不足 | 不能强行解释 absorption |

### No-leakage

只使用 event time 可得的 PLIE、past-only sigma，不使用未来收益。

---

## 6) `aligned_actual_response_bps`

### Financial meaning

实际价格收益沿 PLIE 方向对齐后的响应。

\[
Y_{t,h}=d_t R_{t,h}
\]

| \(Y\) | 含义 |
|---:|---|
| \(>0\) | 价格顺 PLIE 方向 |
| \(\approx0\) | PLIE 没明显穿透 |
| \(<0\) | 价格反向于 PLIE |

### Application

- 计算 `transmission_ratio_raw`；
- 判断 `reversal_takeover`；
- 可视化 PLIE 与 actual response。

### Calculation

\[
R_{t,h}=10000\log(P_{t+h}/P_t)
\]

\[
Y_{t,h}=plie\_direction_t\cdot R_{t,h}
\]

### No-leakage

需要 \(P_{t+h}\)，因此只在 `available_time` 后生成。

---

## 7) `transmission_ratio_raw`

### Financial meaning

实际 aligned response 相对 raw PLIE reference 的传导比例。

\[
TR^{raw}_{t,h}=\frac{Y_{t,h}}{B^{raw}_{t,h}+\epsilon}
\]

### Interpretation

| TR | 含义 |
|---:|---|
| > 1 | 实际顺向超过 PLIE，可能同向放大 |
| 0~1 | 顺向但弱于 PLIE，部分吸收 |
| 约 0 | 压力未穿透 |
| < 0 | 价格反向，压力被拒绝 |

### Application

- research / diagnostic；
- response percentile calibration 的输入；
- 不建议直接输入 Agent。

---

## 8) `response_percentile`

### Financial meaning

在 train-only 可比 context 中，当前 `TR_raw` 位于历史分布的百分位。

\[
U_{t,h}=F^{train}_{bucket}(TR^{raw}_{t,h})
\]

### Interpretation

| U | 金融含义 |
|---:|---|
| 高 | 实际响应偏强传导 / 放大 |
| 中 | 普通响应 |
| 低 | 实际响应偏吸收 / 拒绝 / 反向接管 |

### Application

- 去噪 market response 主指标；
- 生成 `absorption_score_q_0_100`；
- 生成 market response label。

### No-leakage

CDF 只用 train split 拟合。validation/test 只 transform。

---

## 9) `absorption_score_q_0_100`

### Financial meaning

分位校准吸收率分数。

\[
absorption\_score^q=100(1-U)
\]

### Interpretation

| score | 含义 |
|---:|---|
| 0~20 | 同向放大 |
| 20~40 | 偏传导 |
| 40~60 | 普通响应 |
| 60~80 | 偏吸收 |
| 80~100 | 高吸收 / 反向接管 |

### Application

- rolling matured memory 的核心原料；
- 可视化；
- Agent 不直接使用单点值，而使用 EWM / stale-aware 版本。

---

## 10) `market_response_label`

### Financial meaning

短期 matured market response 分类。

### Valid combinations with `response_context`

#### `directional_core`

| label | 含义 |
|---|---|
| `passive_amplification` | PLIE 被同向放大 |
| `baseline_transmission` | 接近 q65 基线传导 |
| `normal_response` | 普通响应 / 噪声区 |
| `partial_absorption` | 部分吸收 |
| `full_absorption_stall` | 压力存在但价格停滞 |
| `reversal_takeover` | 价格反向，短期接管 |

#### `weak_directional_context`

| label | 含义 |
|---|---|
| `weak_directional_amplification_candidate` | 弱置信同向放大候选 |
| `weak_directional_transmission_candidate` | 弱置信传导候选 |
| `weak_directional_stall_candidate` | 弱置信停滞候选 |
| `weak_directional_rejection_candidate` | 弱置信拒绝候选 |
| `weak_directional_uncertain` | 信息不足 |

#### `true_neutral_plie`

| label | 含义 |
|---|---|
| `neutral_low_active_move` | 清算中性，价格低波动 |
| `neutral_normal_active_move` | 普通主动运动 |
| `neutral_strong_active_move` | 强主动运动 |
| `neutral_extreme_active_move` | 极端主动运动 |

#### `low_quality_plie`

| label | 含义 |
|---|---|
| `low_quality_low_active_move` | 低质量 PLIE，价格低波动 |
| `low_quality_normal_active_move` | 普通价格运动 |
| `low_quality_strong_active_move` | 强主动运动 |
| `low_quality_extreme_active_move` | 极端主动运动 |

---

## 11) `active_force_aligned_score` / `active_force_price_score`

### Financial meaning

低噪声主动交易力量 proxy。

\[
active\_force^{aligned}=quality\_weight\cdot(2U-1)
\]

- 正值：主动/实际响应与 PLIE 同向；
- 负值：主动/实际响应反向于 PLIE。

价格方向版本：

\[
active\_force^{price}=plie\_direction\cdot active\_force^{aligned}
\]

### Application

- 研究和 rolling memory；
- 不建议直接使用单点，推荐 stale-aware EWM。

---

# Part III. Rolling Matured Memory

## 12) `mar_abs_score_q_staleaware_ewm_6_30m`

### Financial meaning

30m event-level absorption score 的 staleness-aware EWM。它表示近期成熟清算冲击是否持续被吸收或放大。

### Application

Agent 默认输入。比单点 event-level absorption 更稳健。

### Calculation

先对成熟 `absorption_score_q_0_100` 做 EWM，再用 directional freshness 衰减：

\[
score^{staleaware}=50+freshness(raw\_ewm-50)
\]

其中：

\[
freshness=\exp(-\ln(2)\cdot age\_hours/24)
\]

### Interpretation

| 值 | 含义 |
|---:|---|
| > 50 | 近期偏吸收 |
| ≈ 50 | 中性 / 无新鲜证据 |
| < 50 | 近期偏传导 / 放大 |

---

## 13) `mar_active_force_price_staleaware_ewm_6_30m`

### Financial meaning

近期成熟 event-level 主动力量 proxy，转成价格方向坐标。

### Calculation

\[
active\_force^{staleaware}=freshness\cdot raw\_active\_force\_ewm
\]

### Interpretation

| 值 | 含义 |
|---:|---|
| > 0 | 近期主动价格力量偏向上 |
| < 0 | 近期主动价格力量偏向下 |
| ≈ 0 | 无新鲜 directional evidence |

---

## 14) `mar_directional_core_freshness_30m`

### Financial meaning

最近一次 directional-core matured event 的新鲜度。

### Application

Agent 证据权重控制。避免使用陈旧 absorption 证据。

---

## 15) `mar_amplification_persistence_6_30m` / `mar_takeover_count_12_30m`

### Financial meaning

- `mar_amplification_persistence_6_30m`：最近 6 个成熟事件中同向放大占比；
- `mar_takeover_count_12_30m`：最近 12 个成熟事件中反向接管次数。

### Application

- HPEM：amplification persistence 增强；
- RHA：takeover count 增强。

---

# Part IV. Path-level Absorption

## 16) `path_context`

### Financial meaning

只描述 PLIE / liquidation pressure 本身，不看价格。

### Values

| path_context | 金融含义 | 可否谈 directional absorption |
|---|---|---|
| `path_directional_core` | 累计 PLIE 有高质量方向 | 是 |
| `path_directional_weak` | 累计 PLIE 有方向但证据较弱 | 弱证据 |
| `path_mixed_pressure` | gross PLIE 不低但多空抵消 | 否，只看主动突破 |
| `path_neutral_pressure` | PLIE 很弱或方向不足 | 否，只看 active dominance / quiet |

### Calculation

窗口 \(W\) 内：

\[
C^{raw}=\sum d_iB^{raw}_i
\]

\[
G^{raw}=\sum |d_iB^{raw}_i|
\]

\[
Consistency=\frac{|C^{raw}|}{G^{raw}+\epsilon}
\]

还计算：

- `path_reliability_mean`
- `path_snr`
- `path_nonzero_direction_share`

分类逻辑：

```text
if raw_total_sum 小 or nonzero_share 不足:
    path_neutral_pressure
elif 有 gross pressure 但 direction consistency 不足:
    path_mixed_pressure
elif 有净方向但 quality 不达 core:
    path_directional_weak
else:
    path_directional_core
```

### Time alignment

`available_time = time`。只使用过去窗口。

---

## 17) `path_label`

### Financial meaning

描述价格路径在 `path_context` 下的响应。

### Directional context labels

适用于：

```text
path_directional_core
path_directional_weak
```

| label | 金融含义 | 典型 Agent 解释 |
|---|---|---|
| `path_cascade_transmission` | 路径顺累计 PLIE 方向大幅推进 | HPEM / 级联延续 |
| `path_baseline_transmission` | 路径正常服从 PLIE | ST / 有序传导 |
| `path_partial_absorption` | 顺向但弱于压力 | 趋势动能减弱 |
| `path_full_absorption_stall` | 有压力但路径停滞 | 库存承接 / 燃料衰竭 |
| `path_pressure_rejection` | 路径反向于累计 PLIE | 高吸收 / RHA 候选 |
| `path_reversal_takeover` | 路径强烈反向 | 主导权切换强证据 |

### Mixed pressure labels

适用于：

```text
path_mixed_pressure
```

| label | 金融含义 |
|---|---|
| `path_mixed_active_breakout_up` | 混合清算压力下强主动上行突破 |
| `path_mixed_active_breakout_down` | 混合清算压力下强主动下行突破 |
| `path_normal_mixed_active_breakout_up` | 中等主动上行 |
| `path_normal_mixed_active_breakout_down` | 中等主动下行 |
| `path_mixed_pressure_chop` | 混合压力且无清晰突破 |

### Neutral pressure labels

适用于：

```text
path_neutral_pressure
```

| label | 金融含义 |
|---|---|
| `path_active_dominance_up` | 清算压力中性，但价格强主动上行 |
| `path_active_dominance_down` | 清算压力中性，但价格强主动下行 |
| `path_normal_active_dominance_up` | 中等主动上行 |
| `path_normal_active_dominance_down` | 中等主动下行 |
| `path_quiet_no_pressure` | 清算弱且价格安静 |

---

## 18) `path_return_bps`

### Financial meaning

窗口内已发生价格路径收益。

\[
R^{path}_{T,W}=10000\log(P_T/P_{T-W})
\]

### Application

- 价格路径是否顺 PLIE；
- neutral/mixed context 下 active dominance 判断。

---

## 19) `path_signed_plie_effective_sum_bps`

### Financial meaning

窗口内 reliability-weighted signed effective PLIE 累计。

\[
C^{eff}_{T,W}=\sum plie\_passive\_30m\_bps
\]

### Application

- 判断累计清算压力方向；
- 可视化压力路径。

---

## 20) `path_net_braw_bps` / `path_raw_plie_total_bps`

### Financial meaning

- `path_net_braw_bps`：窗口内 raw PLIE 净方向压力；
- `path_raw_plie_total_bps`：窗口内 raw PLIE gross 活动量。

\[
path\_net\_braw=|\sum d_iB^{raw}_i|
\]

\[
path\_raw\_plie\_total=\sum |d_iB^{raw}_i|
\]

### Application

- 区分 directional / mixed / neutral pressure；
- 分母用于 path transmission ratio。

---

## 21) `path_direction_consistency`

### Financial meaning

窗口内清算方向是否一致。

\[
Consistency=\frac{path\_net\_braw}{path\_raw\_plie\_total+\epsilon}
\]

| 值 | 含义 |
|---:|---|
| 接近 1 | 单边方向很一致 |
| 接近 0 | 多空方向互相抵消 |

---

## 22) `path_liq_neutrality_score`

### Financial meaning

清算压力中性程度。高值表示净方向弱或 gross activity 很低。

### Application

- Agent 判断当前是否不应从清算压力解释价格；
- neutral active dominance 的辅助权重。

---

## 23) `path_active_z`

### Financial meaning

路径收益相对 past-only path volatility 的标准化幅度。

\[
ActiveZ=\frac{|R^{path}|}{\sigma^{path,past}+\epsilon}
\]

### Application

- neutral/mixed context 下识别主动行情；
- 数值保留强弱程度，categorical label 保持简洁。

---

## 24) `path_active_dominance_score`

### Financial meaning

在 neutral/mixed pressure 下，主动价格路径的强度分数。

\[
score=\tanh\left(clip\left(\frac{ActiveZ-0.75}{3.0-0.75},0,2\right)\right)
\]

### Application

- `path_active_dominance_up/down` 的连续强度；
- Agent 识别 VT / macro / spot-flow / ETF-flow 主导行情。

---

## 25) `path_active_dominance_price_score`

### Financial meaning

主动主导分数转成价格方向坐标。

\[
price\_score=sign(R^{path})\cdot path\_active\_dominance\_score
\]

| 值 | 含义 |
|---:|---|
| >0 | 主动上行 |
| <0 | 主动下行 |
| ≈0 | 没有明显主动路径 |

---

## 26) `path_aligned_response_bps`

### Financial meaning

路径收益沿累计 PLIE 方向对齐：

\[
Y^{path}=d^{path}R^{path}
\]

### Interpretation

| 值 | 含义 |
|---:|---|
| >0 | 路径顺 PLIE |
| ≈0 | 压力未明显穿透 |
| <0 | 路径反向于 PLIE |

---

## 27) `path_transmission_ratio`

### Financial meaning

路径级传导比例：

\[
TR^{path}=\frac{Y^{path}}{path\_net\_braw+\epsilon}
\]

### Application

生成 directional path labels。

---

## 28) `path_absorption_score_0_100`

### Financial meaning

路径级吸收 / 压力拒绝分数。

\[
D^{path}
=
-\frac{Y^{path}}
{\sqrt{(path\_net\_braw)^2+(\lambda\sigma^{path})^2+\epsilon}}
\]

\[
score=100\cdot sigmoid(D^{path})
\]

| 值 | 含义 |
|---:|---|
| >50 | 偏吸收 / 压力拒绝 |
| ≈50 | 中性 |
| <50 | 偏传导 / 级联 |

---

## 29) `path_pressure_rejection_score`

### Financial meaning

路径级压力拒绝强度，范围 0~1。

\[
rejection=path\_quality\cdot max(0,\tanh(D^{path}))
\]

### Application

- RHA / reversal evidence；
- 对持续向下压力不跌反升、持续向上压力涨不动回落的场景很敏感。

---

## 30) `path_cascade_score`

### Financial meaning

路径级压力穿透/级联强度，范围 0~1。

\[
cascade=path\_quality\cdot max(0,\tanh(transmission\_score))
\]

### Application

- HPEM / squeeze / flush continuation evidence。

---

## 31) `path_quality`

### Financial meaning

路径级 directional evidence 的质量权重。

\[
quality=reliability\_mean\cdot direction\_consistency\cdot snr\_weight
\]

### Application

- 降低低质量 path label 对 Agent 的影响；
- 区分 core/weak directional context。

---

# Part V. Path memory Agent fields

## 32) `mar_episode_abs_score_24h`

### Financial meaning

24h 路径级吸收分数，是路径级状态底盘。

### Application

Agent 主输入。高值偏 RHA / pressure rejection，低值偏 HPEM / transmission。

---

## 33) `mar_episode_pressure_rejection_score_24h`

### Financial meaning

24h 路径压力拒绝强度。

### Application

识别：

```text
持续下行清算压力被价格拒绝；
持续上行清算压力被价格拒绝。
```

---

## 34) `mar_episode_active_dominance_price_score_24h`

### Financial meaning

24h neutral/mixed context 下主动价格方向分数。

### Application

识别非清算主导行情：

- 正：主动上行；
- 负：主动下行。

---

## 35) `mar_episode_context_24h` / `mar_episode_label_24h`

### Financial meaning

24h path context 与 path label 的 Agent 友好版本。

### Application

Agent 读取：

```text
context 判断清算压力结构；
label 判断价格路径响应。
```

---

# Part VI. Path absorption 与 market response 的组合关系

## 36) 为什么要组合

- `market_response` 灵敏，但只看 20/30/60m 成熟短期裁决；
- `path_absorption` 稳定，但可能对突发短期变化反应慢；
- 两者结合才能区分“趋势延续、短期衰竭、路径拒绝、主动交易主导、混合噪声”。

## 37) 主要组合

| Path context / label | Market response | 综合含义 |
|---|---|---|
| `path_directional_core + path_cascade_transmission` | `passive_amplification` | 路径和短期都穿透，HPEM 强证据 |
| `path_directional_core + path_cascade_transmission` | `partial_absorption` | 路径仍传导，但短期开始钝化 |
| `path_directional_core + path_pressure_rejection` | `partial_absorption` | 路径和短期都吸收，RHA 增强 |
| `path_directional_core + path_pressure_rejection` | `passive_amplification` | 路径拒绝但短期重试穿透，拉锯/AMB |
| `path_directional_core + path_reversal_takeover` | `reversal_takeover` | 路径与短期都反向，主导权切换强证据 |
| `path_mixed_pressure + path_mixed_pressure_chop` | `normal_response` / `weak_directional_uncertain` | 混合压力且短期不清晰，AMB/no-trade |
| `path_mixed_pressure + path_mixed_active_breakout_up/down` | `neutral_strong_active_move` | 混合清算压力下主动突破 |
| `path_neutral_pressure + path_active_dominance_up/down` | `neutral_strong_active_move` / `low_quality_strong_active_move` | 非清算主导行情，偏 VT / 外生冲击 |
| `path_neutral_pressure + path_quiet_no_pressure` | `neutral_low_active_move` | 清算弱、价格弱，RC / low activity |
| `path_neutral_pressure + path_quiet_no_pressure` | `passive_amplification` | 局部新清算冲击出现，可能是新 regime 起点 |

## 38) 实盘使用建议

```text
慢变量：path_context + path_label
快变量：market_response matured memory
触发器：current PLIE / plie_accel_pos / strong_entry
保护层：reliability / entropy / staleness / mixed context
```

---

# Part VII. 可视化与审查

## 39) `path_absorption_dashboard.html`

v5 起，主图不降采样，完整显示每个 source-clock path update。当前正式输入是小时级，因此图上 path_context/path_label 是逐小时更新。

图中最重要行：

1. Price & cumulative effective PLIE；
2. Path return vs aligned pressure；
3. Path absorption / rejection / active dominance；
4. Path quality / neutrality；
5. Path context / label；
6. Memory episode Agent fields。

## 40) `market_response_dashboard.html`

展示短期 matured market response：

- price；
- PLIE main；
- aligned actual response；
- Braw q65 reference；
- response percentile；
- absorption score；
- market response label。

## 41) `scenario_examples.html`

用于人工核对每个 path pressure/context/label 组合的典型场景。

## 42) `path_context_label_combo_counts.csv`

用于检查每个合法 context-label 组合的样本数和占比。若某组合样本过少，应考虑合并 label 或调整 threshold。

---

# Part VIII. 失效条件与注意事项

1. `actual - PLIE` 不是纯主动交易力量。
2. PLIE 低 reliability 时，不能把小 denominator 的 ratio 当强信号。
3. Neutral / mixed context 中不能谈 directional absorption，只能谈 active dominance 或 breakout。
4. Path-level 特征是已发生路径，不是未来预测。
5. Event-level feature 必须成熟后才可进入 Agent。
6. path dashboard 全量显示会增加 HTML 大小，但这是逐小时标签审查的必要代价。
7. 如果交易所清算统计口径变化，PLIE q65、HMM state、path context 都需重审。



## Multiscale Path Absorption Wide Table

### Financial meaning

`path_absorption_multiscale.csv` aggregates the path absorption family across 6h, 12h, 24h, and 48h in one row per timestamp. It gives a state model both fast episode context and slower structural context at the same online time.

- 6h: fast path pressure / response detection.
- 12h: half-day confirmation.
- 24h: primary episode context.
- 48h: slower structural pressure rejection or cascade persistence.

### Application

This table is the preferred input for downstream six-state modeling because it exposes all path scales together without requiring the model pipeline to pivot the long path table.

### Calculation

The long path table is first computed causally for each `(time, window_hours)` pair. The multiscale table then pivots these values into columns:

```text
path_context_6h, path_context_12h, path_context_24h, path_context_48h
path_label_6h, path_label_12h, path_label_24h, path_label_48h
...
```

No new look-ahead calculation is introduced by the pivot. Each value at time `T` uses only data in `[T-W, T]`, and `available_time = T`.

### Additional multiscale quality fields

- `path_data_quality_*`: equals 1 when the path window has enough causal observations and required rolling statistics are available; otherwise 0.
- `path_signal_clarity_*`: PLIE-context clarity independent of price response. Directional contexts use reliability, direction consistency, and SNR; mixed contexts use gross PLIE activity and low net consistency; neutral contexts use low gross PLIE activity.
- `path_activity_level_*`: absolute path return scaled by past-only path-return volatility. It captures price activity in neutral or mixed pressure environments.

### Relationship with path context and path label

`path_context_*` answers: what is the PLIE / liquidation pressure structure over the window?

`path_label_*` answers: how did price respond to that pressure structure?

A downstream model should not interpret active-dominance labels as absorption. Absorption/rejection labels are only meaningful when `path_context_*` is `path_directional_core` or `path_directional_weak`.


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
