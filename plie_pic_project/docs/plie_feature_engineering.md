# PLIE-PIC 特征工程文档

本文覆盖所有重要输出特征，尤其是输入给 Agent 的变量。

## 1) PLIE / `plie_main_bps`

### Financial meaning

`plie_main_bps` 是主 PLIE 输出，默认等于 `plie_passive_30m_bps`。它表示当前清算 forced flow 在 30m horizon 上对价格施加的被动冲击基线，单位 bps。

它不是最终价格预测值，而是后续吸收率与主动交易 residual 的基准。

### Application

- Agent 的主 liquidation-driven impact evidence。
- 与实际 30m price move 比较，构造 absorption / active residual。
- 当 `plie_reliability` 低时，应降低解释权重。

### Calculation

输入数据：

- `plie_direction`
- model-predicted non-negative 30m impact magnitude
- `plie_reliability`

公式：

\[
plie\_main\_bps = plie\_passive\_30m\_bps
\]

\[
plie\_passive\_30m\_bps = d_t \cdot \hat y_{t,30m}^{PLIE} \cdot R_t
\]

时间对齐：source-clock 生成，按 `liq_feature_time` 广播到 10m bar。

未来函数防控：模型输入不包含 `ret_30m_bps`；未来价格只用于训练 label 和评价。

## 2) `plie_direction`

### Financial meaning

清算压力方向。`+1` 表示空头清算占优，强制买回，对价格形成向上压力；`-1` 表示多头清算占优，被迫卖出，对价格形成向下压力；`0` 表示方向不足。

### Application

- 将非负 passive impact magnitude 转换为 signed PLIE。
- 将实际收益转成 liquidation-aligned response。
- Agent 判断当前清算压力方向。

### Calculation

输入：

- `fll_cwt_kf = L_t`
- `fsl_cwt_kf = S_t`
- `total_ls_cwt_kf = T_t`
- `p_state_1` ... `p_state_5`

原始方向：

\[
u_t = \frac{S_t-L_t}{T_t+\varepsilon}
\]

HMM severity：

\[
q_t = \frac{2p_1+p_2-p_4-2p_5}{2}
\]

融合：

\[
c_t = \lambda q_t + (1-\lambda)u_t
\]

方向：

\[
d_t = sign(c_t)
\]

若 \(|c_t| < direction\_deadzone\)，则 `d_t=0`。

时间对齐：source-clock 当前行计算，不使用未来。

## 3) `plie_force_up`

### Financial meaning

`signed liquidation force`，向上为正。它同时包含方向与总清算强度，是 PLIE 的 signed pressure coordinate。

### Application

- 解释当前 forced flow 的方向与压力。
- 计算 `plie_accel_pos`。
- 可视化清算压力状态。

### Calculation

总清算 log 强度：

\[
logT_t=\log(1+T_t)
\]

过去窗口 robust z-score：

\[
z_t = \frac{logT_t-median_t}{1.4826 MAD_t+\varepsilon}
\]

非负强度：

\[
m_t = softplus(z_t)
\]

signed force：

\[
F_t=c_t m_t
\]

rolling 只在 source-clock 上计算，窗口默认 24 个 source snapshot。

## 4) `plie_intensity`

### Financial meaning

当前清算被动压力强度，不带方向。它衡量清算 forced flow 本身大不大。

### Application

- 模型核心输入。
- 分桶/单调性检查。
- Agent 判断当前清算压力是否值得解释。

### Calculation

\[
I_t=|c_t|m_t
\]

其中 `c_t` 是融合方向坐标，`m_t` 是 source-clock past-only robust total liquidation intensity。

## 5) `plie_accel_pos`

### Financial meaning

当前方向上的清算压力增强。它回答：当前 forced flow 是否正在沿同一方向变强。

### Application

- 模型输入。
- 辅助识别 pressure acceleration / regime early cascade。
- 可视化与 decile monotonicity 检查。

### Calculation

\[
a_t^+ = \max(0, d_t(F_t-F_{t-1}))
\]

其中 `F_{t-1}` 是上一个 source-clock snapshot 的 signed force。

时间对齐：只使用当前和上一 source-clock 行。

## 6) `plie_strong_entry`

### Financial meaning

是否刚进入强清算压力状态。state 1 是空头清算强势占优，向上压力强；state 5 是多头清算强势占优，向下压力强。

### Application

- 模型输入。
- Agent 识别清算压力刚从背景状态进入强状态。
- event window 分析。

### Calculation

\[
E_t = 1[hmm\_state_t\in\{1,5\} \land hmm\_state_t\ne hmm\_state_{t-1}]
\]

只使用当前和上一 source-clock hard state。

## 7) `plie_transition_type`

### Financial meaning

HMM 状态切换类型，例如 `2->1`, `4->5`, `1->2`。它描述 liquidation regime 的边际变化。

### Application

- Agent 解释 transition。
- 评价 by-transition 表现。
- 输入 transition severity 映射。

### Calculation

```text
prev_hmm_state + "->" + current_hmm_state
```

只使用上一 source-clock state。

## 8) `plie_transition_severity`

### Financial meaning

机制编码的 transition 强度。增强进入强状态时为正，突发方向翻转时为负。

### Application

- 模型输入。
- Agent 判断当前 transition 是否支持 PLIE boost。

### Calculation

默认映射：

| Transition | Severity |
|---|---:|
| `2->1`, `4->5` | 1.0 |
| `3->1`, `3->5` | 0.7 |
| `1->2`, `5->4` | 0.3 |
| `1->5`, `5->1` | -1.0 |
| other | 0.0 |

## 9) `plie_reliability`

### Financial meaning

PLIE baseline 的因果可信度。它不是吸收率，不使用未来价格。它反映当前是否真的处在 liquidation-driven context，以及数据是否新鲜、状态是否清晰。

### Application

- 对 raw PLIE magnitude 做降权。
- Agent 判断是否应重视 PLIE 输出。
- 输出异常检测。

### Calculation

\[
R_t=R^{state}_t R^{entropy}_t R^{fresh}_t
\]

其中：

\[
R^{state}_t=clip(|c_t|,0,1)
\]

\[
R^{entropy}_t=clip(1-\frac{liq\_entropy_t}{\log 5},0,1)
\]

\[
R^{fresh}_t=\exp(-\frac{\max(0,age_t-no\_decay)}{decay})
\]

source-clock 上 `age_t=0`；广播到 10m bar 后随 `liq_feature_age_min` 增加而下降。

## 10) `plie_phase`

### Financial meaning

PLIE 当前阶段标签：`neutral`, `early_strong_entry`, `accelerating`, `mature`, `normal`。

### Application

- Agent readable explanation。
- 可视化筛选。
- 不是模型核心输入。

### Calculation

规则优先级：

1. `plie_direction == 0` -> `neutral`
2. `plie_strong_entry == 1` -> `early_strong_entry`
3. `plie_accel_pos` 高于近期中位数 -> `accelerating`
4. `age_in_state_source >= 24` -> `mature`
5. 其他 -> `normal`

仅使用当前与过去 source-clock 信息。

## 11) `plie_passive_20m_bps`

### Financial meaning

20m 被动清算冲击基线，偏 immediate forced-flow impulse。

### Application

- 快速冲击诊断。
- 与 30m/60m 构成 impact curve。

### Calculation

由 20m quantile impact curve 输出 magnitude，再乘方向和 reliability。

训练 label：

\[
y_{t,20}^{obs}=d_t\cdot10000\log(P_{t+20m}/P_t)
\]

label 只用于训练/评价。

## 12) `plie_passive_30m_bps`

### Financial meaning

30m 被动清算冲击基线，是主 PLIE。它兼顾 immediate impulse 与短期传导。

### Application

- `plie_main_bps`
- 后续 absorption/residual 主比较对象。

### Calculation

同 20m，只是 horizon 为 30m。

## 13) `plie_passive_60m_bps`

### Financial meaning

60m 被动清算冲击基线，更接近 cascade / sustained pressure，但也更可能混入非清算力量。

### Application

- 辅助判断清算压力是否有延续性。
- 不建议单独作为唯一 PLIE 主值。

### Calculation

同 20m，只是 horizon 为 60m。

## 14) `hmm_state`

### Financial meaning

HMM hard liquidation pressure state。state 1 表示空头清算强势占优，state 5 表示多头清算强势占优。

### Application

- PLIE direction/transition/entry 上游。
- HMM state 背景可视化。
- Agent 状态语境。

### Calculation

由上游 HMM filtered inference 提供。本项目不重新训练 HMM。

## 15) `hmm_conf`

### Financial meaning

当前 hard state 的后验置信度。

### Application

- 状态质量诊断。
- 可视化。
- 未来可以接入更复杂 reliability，但当前核心 reliability 使用 entropy 与方向强度。

### Calculation

由上游提供；若无 `hmm_conf`，上游文档建议可由 `max(p_state_k)` 计算，但本项目默认输入已有该字段。

## 16) `liq_entropy`

### Financial meaning

HMM posterior 熵。高熵表示状态不清晰，低熵表示状态集中。

### Application

- `plie_reliability` 输入。
- AMB / no-trade 降权证据。
- 可视化。

### Calculation

\[
liq\_entropy_t=-\sum_k p_{t,k}\log(p_{t,k}+\varepsilon)
\]

由上游输入，本项目用于 reliability。

## 17) `age_in_state_source`

### Financial meaning

当前 HMM hard state 在 liquidation source clock 上持续多久。它表示 regime maturity / exhaustion context。

### Application

- `plie_phase`。
- HMM duration diagnostics。
- Agent 判断 early / mature 语境。

### Calculation

由上游 HMM 输出。本项目不重新计算 HMM duration，但在 source-clock feature frame 中使用。

## 18) `ret_*`, `plie_residual_*`, `plie_absorption_*`

### Financial meaning

这些是训练和后验诊断变量，不是实时 Agent 输入。

- `ret_*`: 实际未来收益 label。
- `plie_residual_*`: actual return - signed PLIE。
- `plie_absorption_*`: 价格实际响应相对 PLIE 的不足/超额。

### Application

- 评价 PLIE 与价格变化关系。
- 后续吸收率研究。
- 不可用于 live Agent 输入。

### Calculation

\[
ret_{t,h}=10000\log(P_{t+h}/P_t)
\]

\[
residual_{t,h}=ret_{t,h}-\widehat r^{PLIE}_{t,h}
\]

\[
absorption_{t,h}=1-\frac{d_t ret_{t,h}}{|\widehat r^{PLIE}_{t,h}|+\eta}
\]

这些字段显式包含未来价格，因此只能用于训练/评价/研究报告。
