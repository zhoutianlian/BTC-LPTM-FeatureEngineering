# 4.4 HMM 清算压力状态概率与派生特征

> 适用代码版本：`liq_pressure_hmm_v3_background_fixed` 及之后版本。  
> 核心模块：`liq_pressure_hmm/feature_plie_HMM.py`、`liq_pressure_hmm/alignment.py`、`liq_pressure_hmm/state_summary.py`、`liq_pressure_hmm/vis_HMM.py`、`liq_pressure_hmm/diagnostics.py`。  
> 本节文档只描述 HMM liquidation regime probability 及其直接派生字段，不把这些字段解释为独立价格预测信号。它们的核心用途是：描述 BTC 期货清算压力状态、方向、强度、置信度与状态持续性，为下游 Agent 提供状态语境。

---

## 4.4.0 数据、方向与时间可得性约定

### 基础符号

在时点 \(t\)，定义：

- \(L_t\)：多头清算代理变量，对应代码字段 `fll_cwt_kf`。多头被清算时需要被迫卖出，因此代表 **forced selling / downward liquidation pressure**。
- \(S_t\)：空头清算代理变量，对应代码字段 `fsl_cwt_kf`。空头被清算时需要被迫买回，因此代表 **forced buying / upward liquidation pressure**。
- \(\varepsilon\)：数值稳定项，代码中使用很小的正数，避免除零或 \(\log(0)\)。

由此定义清算方向与强度描述量：

$$
\text{short\_dom}_t = S_t - L_t
$$

$$
\text{long\_dom}_t = L_t - S_t = -\text{short\_dom}_t
$$

$$
\text{Total}_t = L_t + S_t
$$

$$
\text{RPN\_short}_t = \frac{S_t}{L_t + S_t + \varepsilon}
$$

解释：

- \(\text{short\_dom}_t > 0\)：空头清算更强，强制买回占优，清算量对价格形成向上压力。
- \(\text{short\_dom}_t < 0\)：多头清算更强，被迫卖出占优，清算量对价格形成向下压力。
- \(\text{short\_dom}_t \approx 0\)：多空清算相对均衡，清算量本身没有明显单边方向压力。

### 因果对齐与 source clock

当前代码不再把小时级 liquidation 特征简单复制到每个 10 分钟 bar 后直接训练 HMM，而是显式区分：

- **价格目标时钟**：10 分钟价格 bar，字段 `time`。
- **清算特征可得时钟**：liquidation source clock，字段 `liq_feature_time`。

若上游小时 liquidation 特征是左标签，例如 `02:00` 表示 `[02:00, 03:00)` 这一小时内计算出的信息，那么该信息不能在 `02:00` 就被 10m bar 使用。当前默认配置使用：

```json
"feature_timestamp_offset": "50min",
"merge_tolerance": "70min"
```

其含义是：

- `liq_feature_time_raw`：上游原始 liquidation 特征时间戳。
- `liq_feature_time = liq_feature_time_raw + 50min`：该 liquidation 特征在 10m 收盘网格上最早可用的时间。
- `liq_feature_age_min = time - liq_feature_time`：当前 10m bar 使用的 liquidation 特征已经过去多少分钟。
- `merge_tolerance = 70min`：防止过旧的 liquidation 特征被无限期 forward-fill。

HMM 训练与推断默认使用 `use_source_clock = true`：

1. 先把 10m merged dataframe 折叠到唯一的 `liq_feature_time`。
2. 在 source clock 上构造 HMM 输入并进行训练/filtered inference。
3. 再把 source-clock posterior 与 hard state 因果广播回 10m bar。

这样做的目的不是改变金融含义，而是避免同一小时清算观测在 10m 网格上重复输入 6 次，从而人为抬高 `hmm_conf`、压低 `liq_entropy`、并夸大状态持续性。

---

## 13–17) `p_state_1` … `p_state_5`

### Financial meaning / 金融含义

`p_state_1` … `p_state_5` 是 HMM 在当前 10m bar 上输出的五类清算压力状态后验概率。它们不是跳跃式硬标签，而是连续的 soft regime representation，用来描述市场当前更接近哪一种清算压力结构。

每个概率满足：

$$
0 \le p_{t,k} \le 1, \qquad k = 1,2,3,4,5
$$

并且：

$$
\sum_{k=1}^{5} p_{t,k} = 1
$$

其中，\(p_{t,k}\) 对应代码字段 `p_state_k`。

### 稳定 state 语义

当前代码使用固定五状态语义，并通过训练后 relabeling 保证每次重训后的状态编号含义尽量稳定。

| State | 字段 | 中文含义 | English meaning | 清算侧 | 价格压力 |
|---:|---|---|---|---|---|
| 1 | `p_state_1` | 空头清算强势占优 | Short-liquidation strong dominance | 空头被迫买回 | 清算量向上压力强势 |
| 2 | `p_state_2` | 空头清算轻度占优 | Short-liquidation mild dominance | 空头被迫买回 | 清算量向上压力轻度 |
| 3 | `p_state_3` | 空头/多头清算均衡 | Balanced long/short liquidations | 多空相对均衡 | 清算量对价格没有明显压力 |
| 4 | `p_state_4` | 多头清算轻度占优 | Long-liquidation mild dominance | 多头被迫卖出 | 清算量向下压力轻度 |
| 5 | `p_state_5` | 多头清算强势占优 | Long-liquidation strong dominance | 多头被迫卖出 | 清算量向下压力强势 |

注意：空头清算对应强制买回，因此是向上压力；多头清算对应被迫卖出，因此是向下压力。不要把“空头清算”理解成价格向下，也不要把“多头清算”理解成价格向上。

### Application / 应用方式

这些概率适合用作下游 Agent 的状态证据，而不是直接作为单独交易信号。推荐聚合为：

**空头清算 / 向上压力概率：**

$$
\text{p\_short\_liq}_t = \text{p\_up\_pressure}_t = p_{t,1} + p_{t,2}
$$

**中性 / 均衡概率：**

$$
\text{p\_neutral}_t = p_{t,3}
$$

**多头清算 / 向下压力概率：**

$$
\text{p\_long\_liq}_t = \text{p\_down\_pressure}_t = p_{t,4} + p_{t,5}
$$

为什么使用 posterior probability 而不是只使用 `hmm_state`：

- posterior 可以表达边界状态的不确定性。
- posterior 可以被平滑、加权、门控，也可以进入下游 evidence fusion。
- posterior 可以避免 hard state 在 regime 边界附近频繁翻转造成的解释噪声。

### Calculation / 计算逻辑

#### 1. HMM 输入特征

代码使用清算方向、相对占比和强度类变量构造 HMM 输入，例如：

$$
\text{short\_dom}_t = S_t - L_t
$$

$$
\text{Total}_t = L_t + S_t
$$

$$
\text{RPN\_short}_t = \frac{S_t}{L_t + S_t + \varepsilon}
$$

这些变量经过 past-only rolling rank / Gaussianization 处理后进入 Gaussian HMM。Gaussianization 的目标是降低重尾和尺度差异对 Gaussian emission 的破坏，不改变其金融方向含义。

#### 2. HMM filtered posterior

HMM 在 source clock 上进行 filtered posterior inference：

$$
\boldsymbol{p}_\tau
= \left(p_{\tau,1}, p_{\tau,2}, p_{\tau,3}, p_{\tau,4}, p_{\tau,5}\right)
= P(z_\tau = k \mid x_{1:\tau}), \qquad k=1,\ldots,5
$$

其中：

- \(\tau\) 表示 liquidation source clock。
- \(x_{1:\tau}\) 表示截至 \(\tau\) 已经因果可得的 HMM 输入序列。
- 使用 filtered posterior，而不是 smoothed posterior，因此不使用未来信息。

随后，source-clock posterior 通过 `liq_feature_time` 因果广播到 10m bar。

#### 3. Sticky transition prior

HMM 使用 sticky transition prior，使状态具备合理持续性。当前默认配置包括：

```json
"p_stay": 0.985,
"transmat_diag_conc": 25.0,
"transmat_offdiag_conc": 1.0
```

金融含义：清算压力状态通常不是完全独立的单 bar 噪声，而会在数小时到数日内延续。但过强的 sticky 会降低状态切换敏感度，因此必须结合 `transition_duration.html` 和 `state_distribution.html` 做诊断。

#### 4. Stable semantic relabeling

HMM 原始状态编号没有天然金融含义，因此训练后必须重排。当前代码以：

$$
D_j = \operatorname{median}\left(S_t - L_t \mid z_t^{\text{raw}} = j\right)
$$

作为每个原始状态 \(j\) 的清算方向锚。

首先选择中性锚：

$$
 j_0 = \arg\min_j |D_j|
$$

该状态被映射为：

$$
 j_0 \rightarrow \text{state } 3
$$

其余状态按照 \(D_j\) 从高到低排列：

- 最大的 \(D_j\) → `state 1`：空头清算强势占优，向上压力强势。
- 第二大的 \(D_j\) → `state 2`：空头清算轻度占优，向上压力轻度。
- 剩余两个负向或较弱状态中，较接近 0 的 \(D_j\) → `state 4`：多头清算轻度占优，向下压力轻度。
- 最小的 \(D_j\) → `state 5`：多头清算强势占优，向下压力强势。

因此，语义排序是：

$$
\text{state 1} \; \rightarrow \; \text{state 2} \; \rightarrow \; \text{state 3} \; \rightarrow \; \text{state 4} \; \rightarrow \; \text{state 5}
$$

对应的清算压力方向从强向上逐渐过渡到强向下。

重要说明：`state 3` 是**相对中性 / 最接近平衡**，不要求 \(D_{j_0}=0\)。如果 BTC 长期清算基线存在结构性偏斜，例如长期 \(L_t > S_t\)，那么中性状态的 `short_dom` 中位数也可能略偏负或略偏正。

#### 5. 状态数量分布

HMM 不强制每个 state 样本数量相等。某个状态，例如 `state 4`，数量高于 20% 不一定是错误。它可能表示市场更长时间处在“多头清算轻度占优 / 向下压力轻度”的背景状态。

如果希望状态更均匀，优先考虑：

1. 检查 HMM 输入特征是否尺度过于偏向某一维。
2. 检查 `state_feature_boxplots.html` 中 \(S-L\) 是否按 state 单调。
3. 检查 `transition_duration.html` 是否存在状态过黏。
4. 调整初始化、`p_stay`、训练窗口或输入变换，而不是简单对 hard state 做等频分箱。

---

## 18) `dir_expect`

### Financial meaning / 金融含义

`dir_expect` 是从五状态 posterior 压缩得到的 signed liquidation pressure coordinate，用来表达当前清算量对价格的机械方向压力：

- `dir_expect > 0`：空头清算占优，强制买回压力更强，清算量对价格形成向上压力。
- `dir_expect < 0`：多头清算占优，被迫卖出压力更强，清算量对价格形成向下压力。
- `dir_expect ≈ 0`：多空清算相对均衡，或 posterior 在向上/向下压力之间相互抵消。

它描述的是 **清算压力方向**，不是最终价格方向。最终价格还取决于对手盘吸收、主动交易流、订单簿深度、波动环境和后续清算链条。

### Application / 应用方式

`dir_expect` 适合用于：

- 下游 Agent 的 directional pressure evidence。
- 与价格响应特征组合，区分“压力穿透”与“压力被吸收”。
- 与 `hmm_conf` / `liq_entropy` 组合，控制状态信号的可信度。
- 与 `age_in_state_source` 组合，识别早段压力延续与后段压力衰竭。

典型解释：

- `dir_expect > 0` 且价格跟随上涨：空头清算压力被传导。
- `dir_expect > 0` 但价格涨不动或回落：强制买回被卖方吸收，上方供给可能较强。
- `dir_expect < 0` 且价格跟随下跌：多头清算压力被传导。
- `dir_expect < 0` 但价格跌不动或反弹：强制卖出被买方吸收，可能出现高吸收或反转语境。

### Calculation / 计算逻辑

先定义：

$$
\text{p\_up\_pressure}_t = p_{t,1} + p_{t,2}
$$

$$
\text{p\_down\_pressure}_t = p_{t,4} + p_{t,5}
$$

则：

$$
\text{dir\_expect}_t
= \text{p\_up\_pressure}_t - \text{p\_down\_pressure}_t
$$

等价于：

$$
\text{dir\_expect}_t
= (p_{t,1} + p_{t,2}) - (p_{t,4} + p_{t,5})
$$

由于 posterior 概率和为 1，因此：

$$
-1 \le \text{dir\_expect}_t \le 1
$$

边界含义：

- \(\text{dir\_expect}_t = 1\)：全部 posterior mass 位于空头清算 / 向上压力状态。
- \(\text{dir\_expect}_t = -1\)：全部 posterior mass 位于多头清算 / 向下压力状态。
- \(\text{dir\_expect}_t = 0\)：中性状态占优，或向上/向下压力概率相互抵消。

### Backward compatibility / 兼容字段

旧代码中可能仍然使用：

$$
\text{p\_bull}_t = \text{p\_up\_pressure}_t
$$

$$
\text{p\_bear}_t = \text{p\_down\_pressure}_t
$$

但新版本推荐使用：

- `p_short_liq`
- `p_up_pressure`
- `p_long_liq`
- `p_down_pressure`

原因是 `bull/bear` 容易混淆“清算侧”和“价格方向”。

---

## 19) `liq_entropy`

### Financial meaning / 金融含义

`liq_entropy` 衡量 HMM posterior 在五个清算压力状态上的分散程度。

- 高 `liq_entropy`：posterior 分散，状态边界模糊，当前 liquidation regime 不够清晰。
- 低 `liq_entropy`：posterior 集中，模型对当前状态有更强单一归属。

它是状态不确定性的度量，适合用作 AMB / no-trade / 降权逻辑的输入之一。

### Application / 应用方式

`liq_entropy` 可以用于：

- 状态置信度过滤。
- 降低 regime 边界附近的下游决策权重。
- 标记 posterior 质量不佳的时间段。
- 与 `hmm_conf` 搭配检查模型是否过度自信。

特别注意：在 source-clock HMM 与 sticky transition 存在时，`liq_entropy` 可能整体偏低。这不一定是错误，但必须通过 `regime_dashboard.html`、`transition_duration.html` 与样本外稳定性检查确认是否合理。

### Calculation / 计算逻辑

后验熵定义为：

$$
\text{liq\_entropy}_t
= -\sum_{k=1}^{5} p_{t,k}\log(p_{t,k}+\varepsilon)
$$

其中：

- \(p_{t,k}\)：当前 bar 属于 state \(k\) 的 posterior probability。
- \(\log(\cdot)\)：自然对数。
- \(\varepsilon\)：数值稳定项，避免 \(\log(0)\)。

单位为 nats。

理论范围：

$$
0 \le \text{liq\_entropy}_t \le \log(5)
$$

其中：

- \(0\)：posterior 完全集中在单一状态。
- \(\log(5)\)：posterior 在五个状态上完全均匀。

### Alias / 兼容别名

代码同时输出：

```text
hmm_entropy = liq_entropy
```

`hmm_entropy` 是 legacy alias，推荐新逻辑统一使用 `liq_entropy`。

---

## 20) `hmm_conf`

### Financial meaning / 金融含义

`hmm_conf` 是当前 hard state 的状态归属置信度，用来衡量模型是否清楚地把当前市场归入某一个 liquidation regime。

- 高 `hmm_conf`：当前状态归属清晰，regime evidence 较强。
- 低 `hmm_conf`：posterior 混合，可能处于状态切换、清算压力冲突或样本信息不足阶段。

### Application / 应用方式

`hmm_conf` 可用于：

- gating：低置信度时降低 regime-dependent rule 的权重。
- weighting：将状态信号强度乘以置信度。
- diagnostics：检查状态边界、模型过度自信和异常不确定区间。
- Agent memory：作为 AMB 或 transition warning 的证据之一。

### Calculation / 计算逻辑

若存在 hard state：

$$
\hat{k}_t = \text{hmm\_state}_t
$$

并且存在对应 posterior，则：

$$
\text{hmm\_conf}_t = p_{t,\hat{k}_t}
$$

代码中若已经有 `hmm_state_conf`，则优先使用：

$$
\text{hmm\_conf}_t = \text{hmm\_state\_conf}_t
$$

若没有 `hmm_state_conf`，但有 `p_state_1` … `p_state_5`，则 fallback 为：

$$
\text{hmm\_conf}_t = \max_{k \in \{1,2,3,4,5\}} p_{t,k}
$$

代码还输出：

$$
\text{hmm\_maxp}_t = \max_{k \in \{1,2,3,4,5\}} p_{t,k}
$$

在 filtered posterior 模式下，`hmm_conf` 通常等于 hard state 对应的 posterior；若 hard state 经过 debounce 后不等于 posterior 最大状态，则二者可能略有差异。

---

## 21) `age_in_state` 与 `age_in_state_source`

### Financial meaning / 金融含义

`age_in_state` 和 `age_in_state_source` 描述当前 hard liquidation regime 已经持续多久。它们是 regime maturity / exhaustion proxy。

- 低 age：状态刚进入，清算压力可能仍处在早段传导阶段。
- 高 age：状态已经持续较久，可能接近压力衰竭、吸收增强或状态切换阶段。

对 BTC 期货清算压力而言，状态年龄不应被机械解释为趋势一定延续。它更适合与价格响应、吸收率、波动率和 posterior 置信度一起使用。

### Difference / 两个年龄字段的区别

#### `age_in_state`

`age_in_state` 在输出 10m bar 网格上计算，表示当前 hard state 已经连续持续了多少个 10m bar。

#### `age_in_state_source`

`age_in_state_source` 在 liquidation source clock 上计算，表示当前 hard state 已经连续持续了多少个 liquidation source update。

当 liquidation 数据是小时级、价格数据是 10m 时，`age_in_state_source` 通常更接近真实清算信息更新节奏；`age_in_state` 则更适合做 10m 执行层或展示层的状态持续计数。

### Calculation / 计算逻辑

当前代码使用 0-based episode age。若 hard state 在当前 bar 刚刚切换，则 age 为 0。

对 10m bar 网格：

$$
\text{age\_in\_state}_t =
\begin{cases}
0, & \text{if } \text{hmm\_state}_t \ne \text{hmm\_state}_{t-1} \\
\text{age\_in\_state}_{t-1}+1, & \text{if } \text{hmm\_state}_t = \text{hmm\_state}_{t-1}
\end{cases}
$$

对 source clock：

$$
\text{age\_in\_state\_source}_\tau =
\begin{cases}
0, & \text{if } \text{hmm\_state}_\tau \ne \text{hmm\_state}_{\tau-1} \\
\text{age\_in\_state\_source}_{\tau-1}+1, & \text{if } \text{hmm\_state}_\tau = \text{hmm\_state}_{\tau-1}
\end{cases}
$$

其中 \(\tau\) 表示 `liq_feature_time`。

### Application / 应用方式

推荐用途：

- 早段 continuation vs 后段 exhaustion 的条件变量。
- 与 `dir_expect` 联合判断同向压力是否仍在延续。
- 与价格响应联合判断是否出现“清算还在发生，但价格不再服从”的高吸收/反转语境。
- 与 `transition_duration.html` 检查 sticky HMM 是否过黏或过碎。

---

## 22) State metadata columns / 状态解释字段

当前代码会根据 `hmm_state` 输出一组可审计 metadata，便于下游 Agent、日志和可视化直接解释状态。

| 字段 | 含义 |
|---|---|
| `state_name` | 程序内部英文短名，例如 `short_liq_strong` |
| `state_name_en` | 英文可读名称，例如 `Short-liquidation strong dominance` |
| `state_name_cn` | 中文状态名，例如 `空头清算强势占优` |
| `state_liq_side` | 主导清算侧：`short_liquidation`、`long_liquidation`、`balanced` |
| `state_liq_side_cn` | 中文主导清算侧 |
| `state_pressure` | 价格压力方向：`up`、`down`、`neutral` |
| `state_pressure_cn` | 中文价格压力描述 |
| `state_pressure_direction` | 数值方向：向上为 `+1`，中性为 `0`，向下为 `-1` |
| `state_severity` | 强度等级：强势为 `2`，轻度为 `1`，中性为 `0` |

状态映射为：

| `hmm_state` | `state_name` | `state_name_cn` | `state_pressure_direction` | `state_severity` |
|---:|---|---|---:|---:|
| 1 | `short_liq_strong` | 空头清算强势占优 | +1 | 2 |
| 2 | `short_liq_mild` | 空头清算轻度占优 | +1 | 1 |
| 3 | `balanced_liq` | 空头/多头清算均衡 | 0 | 0 |
| 4 | `long_liq_mild` | 多头清算轻度占优 | -1 | 1 |
| 5 | `long_liq_strong` | 多头清算强势占优 | -1 | 2 |

---

## 23) 可视化与诊断输出

当前代码会生成一组 HTML 诊断文件，用于检查 HMM state 的金融语义、分布稳定性、状态持续性和清算事件响应。

### `output/index.html`

总目录页，链接到主图和 diagnostics 下的所有 HTML。建议每次重训后从这里开始检查。

### `output/hmm_price_states.html`

价格主图，带 5 个 state 背景色带。当前实现使用 compact categorical heatmap background，而不是大量 Plotly rectangle shapes，因此能够保留背景色，同时避免 HTML 生成过慢。

鼠标 hover 显示：

- timestamp
- price
- state id
- `state_name_cn`
- `state_name_en`
- `state_pressure_cn`
- `dir_expect`
- `hmm_conf`

### `output/diagnostics/regime_dashboard.html`

综合状态诊断面板，检查：

- 价格与 state 背景是否一致。
- `dir_expect` 是否和清算压力方向一致。
- `p_up_pressure` / `p_down_pressure` / `p_neutral` 是否合理切换。
- `hmm_conf` 与 `liq_entropy` 是否在状态边界附近出现合理变化。
- `age_in_state_source` 与 `liq_feature_age_min` 是否暴露 stale fill 或状态过黏问题。

### `output/diagnostics/state_distribution.html`

展示每个 state 的数量和占比。用于检查：

- 是否有 state collapse。
- 是否某个 state 成为 catch-all。
- state 4 或其他 mild state 是否占比过高。
- 状态分布是否随重训窗口发生明显漂移。

注意：状态不需要强制等频。数量不均衡只有在破坏语义、导致某些状态无法被识别，或明显影响下游 Agent 时才是问题。

### `output/diagnostics/state_feature_boxplots.html`

按 state 展示清算方向、强度和相关特征的分布。核心检查问题：

- `short_dom = S-L` 是否从 state 1 到 state 5 大体单调下降。
- state 3 是否最接近平衡。
- state 1/5 是否确实比 state 2/4 更极端。
- 是否存在某个 state 的金融语义不稳定。

### `output/diagnostics/transition_duration.html`

展示状态转移和持续期。用于判断：

- HMM 是否过黏。
- state 是否过碎。
- debounce 与 sticky prior 是否过强。
- `age_in_state_source` 的解释是否可信。

### `output/diagnostics/event_windows.html`

针对极端清算事件进行事件窗口分析。用于检查：

- 大额空头清算后，价格是否更容易表现为向上压力传导或被吸收。
- 大额多头清算后，价格是否更容易表现为向下压力传导或被吸收。
- 不同 state 下，事件后的路径是否符合清算压力机制。

### 可视化性能参数

当前配置：

```json
"price_plot_max_points": 60000,
"dashboard_max_points": 30000,
"state_boxplot_max_points_per_state": 6000,
"max_shape_segments": 1200,
"state_background_max_points": 8000,
"state_background_opacity": 0.20
```

说明：

- 这些参数只影响 HTML 绘图速度和文件大小，不影响模型训练、推断或 CSV 输出。
- `state_background_max_points` 控制背景 heatmap 的显示采样点数。
- `state_background_opacity` 控制 state 背景透明度。
- 若背景不够细，可提高 `state_background_max_points`。
- 若浏览器打开仍慢，可降低 `state_background_max_points` 或 `dashboard_max_points`。

---

## 24) 数据质量、失效条件与验证要求

### 必须检查的数据质量项

1. `liq_feature_age_min` 是否存在异常长尾。
2. `liq_feature_time_raw` 到 `liq_feature_time` 的 offset 是否符合上游数据时间戳语义。
3. `p_state_1` … `p_state_5` 是否非负且每行和接近 1。
4. `hmm_state` 是否只在 1–5 之间。
5. `state_semantic_summary.csv` 中 `short_dom_med` 是否支持当前 state 语义。
6. `state_distribution.html` 是否存在极端 state collapse。
7. `transition_duration.html` 是否显示状态过黏或过碎。

### 可能失效的市场条件

这些特征的金融机制依赖于“清算量能够代表已实现杠杆脆弱性”。在以下条件下，其解释力可能下降：

- 交易所清算统计口径发生变化。
- 上游 liquidation 数据延迟、缺失或聚合方式改变。
- 市场主要杠杆迁移到未覆盖交易所或产品。
- ETF、现货大额流、期权 dealer hedge、宏观冲击成为主导，清算只是被动结果。
- 极端流动性枯竭阶段，价格跳跃先于清算数据被记录，导致清算特征滞后。
- 稳定币流动性或交易所信用风险导致清算机制与正常市场状态不同。

### 必做验证

每次重训或修改 HMM 输入后，至少检查：

1. `state_semantic_summary.csv`：确认 state 语义是否与 `S-L` 中位数一致。
2. `state_distribution.html`：确认没有严重 state collapse。
3. `state_feature_boxplots.html`：确认 state 1 → state 5 的方向排序合理。
4. `regime_dashboard.html`：确认 posterior、`dir_expect`、`hmm_conf`、`liq_entropy` 的时间行为合理。
5. `event_windows.html`：确认极端空头/多头清算事件后的路径与金融机制大体一致。
6. out-of-sample / walk-forward：确认重训后 state 语义不频繁漂移。

---

## 25) 输出字段汇总

| 字段 | 类型 | 含义 | 推荐用途 |
|---|---|---|---|
| `p_state_1` | probability | 空头清算强势占优后验概率 | 强向上清算压力证据 |
| `p_state_2` | probability | 空头清算轻度占优后验概率 | 轻向上清算压力证据 |
| `p_state_3` | probability | 多空清算均衡后验概率 | 中性 / AMB 证据 |
| `p_state_4` | probability | 多头清算轻度占优后验概率 | 轻向下清算压力证据 |
| `p_state_5` | probability | 多头清算强势占优后验概率 | 强向下清算压力证据 |
| `hmm_state` | integer | debounced hard state | 状态标签、可视化、Agent memory |
| `hmm_state_conf` | probability | hard state 对应置信度 | 状态质量控制 |
| `p_short_liq` | probability | `p_state_1 + p_state_2` | 空头清算占优概率 |
| `p_up_pressure` | probability | `p_state_1 + p_state_2` | 向上清算压力概率 |
| `p_neutral` | probability | `p_state_3` | 中性状态概率 |
| `p_long_liq` | probability | `p_state_4 + p_state_5` | 多头清算占优概率 |
| `p_down_pressure` | probability | `p_state_4 + p_state_5` | 向下清算压力概率 |
| `dir_expect` | [-1, 1] | 向上压力概率减向下压力概率 | 清算压力方向坐标 |
| `liq_entropy` | nats | posterior entropy | 不确定性 / AMB / gating |
| `hmm_entropy` | nats | `liq_entropy` legacy alias | 兼容旧字段 |
| `hmm_conf` | probability | hard state 置信度 | 置信度过滤 |
| `hmm_maxp` | probability | posterior 最大概率 | posterior 集中度诊断 |
| `age_in_state` | bar count | 10m 网格 hard state 连续年龄 | 执行层状态持续性 |
| `age_in_state_source` | source count | source clock hard state 连续年龄 | 清算信息节奏下的状态成熟度 |
| `liq_feature_time_raw` | timestamp | 原始 liquidation 特征时间戳 | 时间戳审计 |
| `liq_feature_time` | timestamp | 因果可得时间戳 | source-clock 对齐 |
| `liq_feature_age_min` | minutes | 当前 10m bar 使用的 liquidation 特征年龄 | stale fill 诊断 |
| `state_name_cn` | string | 中文状态名 | 日志、可视化、Agent 解释 |
| `state_name_en` | string | 英文状态名 | 日志、可视化、Agent 解释 |
| `state_pressure_cn` | string | 中文价格压力描述 | hover、报告、规则解释 |
| `state_pressure_direction` | integer | +1 / 0 / -1 | 规则化方向输入 |
| `state_severity` | integer | 2 / 1 / 0 | 强弱等级输入 |

---

## 26) 最小审查清单

上线或重训后，不建议只看 `hmm_state.csv`。建议按顺序检查：

1. 打开 `output/index.html`。
2. 看 `hmm_price_states.html`：确认 5 色背景存在，hover 能显示 price 与 state。
3. 看 `state_distribution.html`：确认没有明显 state collapse。
4. 看 `state_feature_boxplots.html`：确认 state 1 到 state 5 的 `S-L` 方向排序合理。
5. 看 `transition_duration.html`：确认状态持续期不过黏也不过碎。
6. 看 `regime_dashboard.html`：确认 `dir_expect`、`hmm_conf`、`liq_entropy`、`age_in_state_source` 行为符合机制。
7. 看 `event_windows.html`：确认极端清算事件后的价格响应大体符合“清算压力传导 / 吸收”逻辑。

