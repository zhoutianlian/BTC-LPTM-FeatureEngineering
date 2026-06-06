# 特征工程文档

本文档仅描述当前项目最终输出并用于模型观测层的 9 个特征。项目不会再输出其他中间特征。

## 4.2 Side-specific liquidation bursts (cascade texture)

Liquidation Magnitude 用于刻画清算压力的量级，而 Spike Ratio 用于刻画当前清算活动相对于过去自适应基线是否异常。它们的核心作用是确认市场是否处于明显的清算驱动环境。

Spike Ratio 可用于 HPEM / RHA 识别：当市场确实由某一侧清算驱动时，对应侧的 Spike 通常应显著抬升。

Spike 的解释强依赖语境：RC 边界附近的小幅抬升，与 HPEM 中持续、极端、同向的异常爆发，在金融学含义上并不相同。

### 1) fll_spike_kama

#### Financial meaning

- `fll_spike_kama` 用于刻画期货多头清算强度相对于过去自适应基线是否异常。
- 它回答的问题是：当前多头清算是否显著高于当前 regime 下的常态水平。
- 它反映的是被迫卖出压力的突发性，而不是稳态多头清算流本身的高低。
- 该序列通常重尾、偏态、存在显著自相关，并可能在极端踩踏中出现异常尖峰。

#### Application

- `fll_spike_kama > 0` 表示当前多头清算高于过去自适应基线。
- `fll_spike_kama >> 0` 表示当前多头清算异常强，可能对应下行方向的级联释放、恐慌踩踏或局部高潮时刻。
- 它适合与 `fll_velocity_gaussian` 联合解释：高 spike 且速度为正，更像清算扩散；高 spike 但速度转负，更像高潮后钝化。
- 该特征的建模价值在于补充单纯量级特征对突发事件敏感性不足的问题。

#### Calculation

1. Inputs and smoothing

设 `FLL_t` 为时点 `t` 的多头清算量平滑序列。由于上游 `fll_cwt_kf` 已经是非负、多头侧、因果平滑后的清算量代理，因此本项目直接取：

$$
\widetilde{FLL}_t = fll\_cwt\_kf(t)
$$

2. Adaptive baseline via causal KAMA

对 `\widetilde{FLL}_t` 计算严格因果的 Kaufman Adaptive Moving Average：

$$
ER_t = \frac{|\widetilde{FLL}_t - \widetilde{FLL}_{t-N}|}{\sum_{i=t-N+1}^{t}|\widetilde{FLL}_i - \widetilde{FLL}_{i-1}| + \varepsilon}
$$

$$
SC_t = \left(ER_t\left(\frac{2}{FastN+1} - \frac{2}{SlowN+1}\right) + \frac{2}{SlowN+1}\right)^2
$$

$$
KAMA_t = KAMA_{t-1} + SC_t\big(\widetilde{FLL}_t - KAMA_{t-1}\big)
$$

为避免当前值污染当前基线，实际实现使用滞后一期基线：

$$
B_t = KAMA_{t-1}
$$

3. Spike ratio definition

当前生产实现采用对数比值形式，以压缩重尾并保持原特征名不变：

$$
fll\_spike\_kama_t = \log\left(\frac{\widetilde{FLL}_t + \varepsilon}{B_t + \varepsilon}\right)
$$

4. Signal interpretation and thresholding

- `fll_spike_kama > 0`：当前多头清算高于过去自适应基线。
- `fll_spike_kama \gg 0`：当前多头清算显著异常，可视为当前 bar 相对于前期 regime 的清算峰值。
- `fll_spike_kama < 0`：当前多头清算低于过去自适应基线。

窗口 `N`、`FastN`、`SlowN` 均为可调超参数，应结合 bar 周期与期望响应速度设定。

### 2) fsl_spike_kama

#### Financial meaning

- `fsl_spike_kama` 用于刻画期货空头清算强度相对于过去自适应基线是否异常。
- 它回答的问题是：当前空头清算是否显著高于当前 regime 下的常态水平。
- 它反映的是被迫买回压力的突发性，而不是稳态空头清算流本身的高低。
- 在 short squeeze 环境下，该特征可能出现极端尖峰与厚尾。

#### Application

- `fsl_spike_kama > 0` 表示当前空头清算高于过去自适应基线。
- `fsl_spike_kama >> 0` 表示当前空头清算异常强，可能对应上行方向的 squeeze 扩散或高潮释放。
- 它适合与 `fsl_velocity_gaussian` 联合解释：高 spike 且速度为正，更像 squeeze 放大；高 spike 但速度转负，更像 squeeze 尾声。
- 该特征用于确认上涨过程中是否掺入较强的被动回补成分。

#### Calculation

1. Inputs and smoothing

$$
\widetilde{FSL}_t = fsl\_cwt\_kf(t)
$$

2. Adaptive baseline via causal KAMA

$$
ER_t = \frac{|\widetilde{FSL}_t - \widetilde{FSL}_{t-N}|}{\sum_{i=t-N+1}^{t}|\widetilde{FSL}_i - \widetilde{FSL}_{i-1}| + \varepsilon}
$$

$$
SC_t = \left(ER_t\left(\frac{2}{FastN+1} - \frac{2}{SlowN+1}\right) + \frac{2}{SlowN+1}\right)^2
$$

$$
KAMA_t = KAMA_{t-1} + SC_t\big(\widetilde{FSL}_t - KAMA_{t-1}\big)
$$

实际实现同样使用滞后一期基线：

$$
B_t = KAMA_{t-1}
$$

3. Spike ratio definition

$$
fsl\_spike\_kama_t = \log\left(\frac{\widetilde{FSL}_t + \varepsilon}{B_t + \varepsilon}\right)
$$

4. Signal interpretation and thresholding

- `fsl_spike_kama > 0`：当前空头清算高于过去自适应基线。
- `fsl_spike_kama \gg 0`：当前空头清算显著异常，可视为当前 bar 相对于前期 regime 的 squeeze 峰值。
- `fsl_spike_kama < 0`：当前空头清算低于过去自适应基线。

## 4.3 Side-specific liquidation dynamics (Gaussian RoC / RoC²)

RoC 用于刻画清算量级变化的速度，RoC² 用于刻画该速度本身的变化，也就是加速度。它们的核心作用是帮助识别清算压力是在持续增强、开始钝化，还是已经进入拐点阶段。

RoC 与 RoC² 可用于 HPEM 识别：若单侧清算量级处于高位，同时速度与加速度均为正，说明清算级联可能正在加速扩散。

这些导数类特征对噪声更敏感，因此实现时必须严格保持因果性，并通过对数压缩与单边 Gaussian 平滑降低重尾与尖峰影响。

### 3) fll_velocity_gaussian

#### Financial meaning

- `fll_velocity_gaussian` 衡量多头清算强度变化的速度，即多头清算压力是在变强还是变弱。
- 它回答的问题是：多头清算正在加速释放，还是正在放缓。
- 正值表示多头清算压力相较上一时点继续增强；负值表示其正在减弱。
- 它不是量级本身，而是量级的变化速度，因此更适合作为级联强化或衰竭的动态确认变量。

#### Application

- `fll_velocity_gaussian > 0`：多头清算强度正在上升，可能对应下行方向压力继续释放。
- `fll_velocity_gaussian < 0`：多头清算强度正在下降，可能意味着踩踏卖压在钝化。
- 它适合与 `fll_spike_kama` 联合使用：高 spike 且 velocity 为正，更像级联扩散；高 spike 但 velocity 为负，更像高潮后钝化。

#### Calculation

1. Input transform

本项目以非负多头清算量 `fll_cwt_kf` 作为输入，并先做对数压缩：

$$
X_t = \log\big(1 + fll\_cwt\_kf(t)\big)
$$

2. Causal Gaussian smoothing

使用严格单边、只依赖历史样本的 Gaussian 核：

$$
\widetilde{X}_t = \sum_{i=0}^{L-1} w_i X_{t-i}, \qquad
w_i \propto \exp\left(-\frac{i^2}{2\sigma^2}\right)
$$

其中 `L` 为平滑窗口，对应代码中的 `liq_roc_gaussian_window_min` 换算后的 bars。

3. Velocity definition

生产实现使用 backward difference，并统一换算为每小时单位：

$$
fll\_velocity\_gaussian_t = \frac{\widetilde{X}_t - \widetilde{X}_{t-1}}{\Delta t_{hour}}
$$

### 4) fll_acceleration_gaussian

#### Financial meaning

- `fll_acceleration_gaussian` 衡量多头清算速度本身的变化，即多头清算动量是在继续增强，还是已经开始失速。
- 相比 velocity，它更敏感、更不稳定，也更容易受单次极端事件和滤波形状影响。
- 因此它具有金融学意义，但在建模中应视为次级辅助特征，而不是优先级最高的核心特征。

#### Application

- `fll_acceleration_gaussian > 0`：多头清算速度仍在继续上升，说明下行踩踏可能进一步加速。
- `fll_acceleration_gaussian < 0`：多头清算仍可能存在，但其增长速度已经开始放缓，提示踩踏压力可能进入峰值后阶段。
- 在模型中建议降低其解释权重，主要将其作为 velocity 的补充确认变量。

#### Calculation

在 `fll_velocity_gaussian` 的基础上继续做 backward difference，并统一换算为每小时平方单位：

$$
fll\_acceleration\_gaussian_t = \frac{fll\_velocity\_gaussian_t - fll\_velocity\_gaussian_{t-1}}{\Delta t_{hour}}
$$

该定义保持了严格因果性，同时避免了中心差分带来的未来信息泄漏。

### 5) fsl_velocity_gaussian

#### Financial meaning

- `fsl_velocity_gaussian` 衡量空头清算强度变化的速度，即空头回补压力是在增强还是在减弱。
- 正值表示空头清算压力相较上一时点继续增强；负值表示其正在减弱。
- 该特征帮助识别 short squeeze 是否正在持续放大，或者已经进入钝化阶段。

#### Application

- `fsl_velocity_gaussian > 0`：空头清算强度正在上升，可能对应上行方向被动买回压力继续扩散。
- `fsl_velocity_gaussian < 0`：空头清算强度正在下降，可能意味着 squeeze 动能开始衰减。
- 它适合与 `fsl_spike_kama` 结合解释：高 spike 且 velocity 为正，更像 squeeze 扩散；高 spike 但 velocity 为负，更像 squeeze 尾声。

#### Calculation

1. Input transform

$$
X_t = \log\big(1 + fsl\_cwt\_kf(t)\big)
$$

2. Causal Gaussian smoothing

$$
\widetilde{X}_t = \sum_{i=0}^{L-1} w_i X_{t-i}, \qquad
w_i \propto \exp\left(-\frac{i^2}{2\sigma^2}\right)
$$

3. Velocity definition

$$
fsl\_velocity\_gaussian_t = \frac{\widetilde{X}_t - \widetilde{X}_{t-1}}{\Delta t_{hour}}
$$

### 6) fsl_acceleration_gaussian

#### Financial meaning

- `fsl_acceleration_gaussian` 衡量空头清算速度本身的变化，即 short squeeze 动量是在继续增强，还是开始失速。
- 与多头侧加速度相同，它更容易受噪声与单次尖峰影响，因此在模型中的优先级应低于 level、spike 与 velocity。

#### Application

- `fsl_acceleration_gaussian > 0`：空头清算速度仍在继续上升，说明 squeeze 可能进一步放大。
- `fsl_acceleration_gaussian < 0`：空头清算仍可能维持高位，但其增长速度已经开始放缓，提示上行动能可能进入高潮后阶段。
- 在模型中建议降低其解释权重，主要将其作为 velocity 的补充确认变量。

#### Calculation

$$
fsl\_acceleration\_gaussian_t = \frac{fsl\_velocity\_gaussian_t - fsl\_velocity\_gaussian_{t-1}}{\Delta t_{hour}}
$$

## 4.7 Price-action trend + volatility context

Price 提供价格水平背景，Trend 提供近期方向性，而 Volatility 提供风险与市场扰动背景。它们用于补充清算侧特征，使模型能够同时观察清算压力、价格方向与风险状态。

它们可用于 HPEM 识别：在 HPEM 中，价格通常快速移动，趋势强且与主导清算方向一致。

它们也可用于 RC 识别：RC 往往表现为趋势不清晰，而背景波动较高或仍在扰动中。

### 7) trend_pressure

#### Financial meaning

- `trend_pressure` 用于衡量近期方向性位移相对于背景波动的强弱，是一个无量纲的波动率标准化趋势强度指标。
- 它回答的问题是：市场最近是否在做一个相对于当前噪声水平而言足够显著的方向性运动。
- 该特征本质上是趋势位移的标准化版本，而不是简单动量。

#### Application

- `trend_pressure > 0`：近期上行趋势占优。
- `trend_pressure < 0`：近期下行趋势占优。
- `|trend_pressure|` 越大，说明趋势相对于背景波动越显著。
- 它适合用作 regime/context 特征，用来区分趋势环境与震荡环境。

#### Calculation

设 `logp_t = \log(price_t)`，`r_t = logp_t - logp_{t-1}`。

生产实现定义为：

$$
trend\_pressure_t = \frac{logp_t - logp_{t-k_{mom}}}{\sqrt{k_{mom}}\cdot \hat{\sigma}_t^{(k_{vol})} + \varepsilon}
$$

其中：

- `k_mom` 为趋势窗口，对应 `trend_pressure_mom_window_min`
- `\hat{\sigma}_t^{(k_vol)}` 为背景 realized volatility，对应 `trend_pressure_vol_window_min`

该定义更接近将最近 `k` 个 bar 的方向性位移按背景波动换算成若干 sigma 的趋势运动。

### 8) kalman_slope

#### Financial meaning

- `kalman_slope` 是对近期价格潜在漂移的平滑估计，用于描述当前价格变化趋势。
- 与固定窗口动量相比，它更平滑、更鲁棒，对单个噪声 bar 不那么敏感。
- 它不是原始价格斜率，而是 log-price 空间下的局部线性趋势斜率，因此天然具有尺度可比性。

#### Application

- 正值表示平滑后的潜在价格漂移向上。
- 负值表示平滑后的潜在价格漂移向下。
- 它适合与 `trend_pressure` 联合使用：前者强调固定窗口的趋势强弱，后者强调潜在漂移的连续估计。

#### Calculation

对 `logp_t = \log(price_t)` 建立一侧 local linear trend Kalman 模型：

$$
\begin{bmatrix}
\ell_t \\
\beta_t
\end{bmatrix}
=
\begin{bmatrix}
1 & 1 \\
0 & 1
\end{bmatrix}
\begin{bmatrix}
\ell_{t-1} \\
\beta_{t-1}
\end{bmatrix}
+
\eta_t,
\qquad
logp_t = \ell_t + \epsilon_t
$$

其中 `\beta_t` 即局部趋势斜率。最终定义：

$$
kalman\_slope_t = \hat{\beta}_t
$$

生产实现再按 bar 长度统一换算为每小时尺度，使不同决策周期下数值更可比。

### 9) vol_adaptive

#### Financial meaning

- `vol_adaptive` 是一个严格因果、能根据当前市场状态在短窗与长窗之间自动调节的自适应 realized volatility。
- 它回答的问题是：市场当前到底有多 turbulent，而且这种 turbulent 是短期爆发还是长期背景风险的一部分。
- 与单一窗口 volatility 相比，它更适合 BTC 这种具有强波动聚集与 regime switching 特征的市场。

#### Application

- 低值通常对应较平静、较稳定的环境。
- 高值通常对应更强的风险扰动、波动扩张或压力传导阶段。
- 它适合作为状态模型中的风险背景变量，用来补充清算与趋势信息。

#### Calculation

1. 计算 log return：

$$
r_t = \log\left(\frac{P_t}{P_{t-1}}\right)
$$

2. 计算两个严格因果 realized volatility：

- `short_vol_t`：短窗 realized volatility，对应 `price_vol_adaptive_short_window_min`
- `long_vol_t`：长窗 realized volatility，对应 `price_vol_adaptive_long_window_min`

3. 计算短长波动比的对数形式：

$$
u_t = \log\left(\frac{short\_vol_t + \varepsilon}{long\_vol_t + \varepsilon}\right)
$$

4. 使用 sigmoid 将其映射为自适应权重：

$$
w_t = w_{min} + (1 - w_{min})\cdot \sigma(\gamma \nu_t)
$$

其中 `\sigma(\cdot)` 为 sigmoid 函数，`w_min` 与 `\gamma` 分别对应 `price_vol_adaptive_min_weight` 与 `price_vol_adaptive_sigmoid_gamma`。

5. 形成最终自适应波动：

$$
vol\_adaptive_t = w_t \cdot short\_vol_t + (1 - w_t) \cdot long\_vol_t
$$

当短窗波动显著高于长窗波动时，权重向短窗移动；当市场较平稳时，权重更多保留在长窗。

## 5. 输出特征检验与可视化报告

完整 pipeline 会对本文档列出的全部 9 个最终输出特征生成自动化诊断报告。报告只用于研究校验和问题定位，不改变特征计算结果。

默认报告目录：

```text
btc_liqprice_features_artifact/reports/feature_diagnostics/
```

报告包含：

- `index.html`：总览页，展示样本区间、特征数量、缺失特征数量、PASS/WARN/FAIL 统计、可搜索排序的特征汇总表、相关性热力图和高相关特征对。
- `features/<feature_name>.html`：单特征详情页，展示特征说明、类别、检验状态、异常摘要、统计表、全历史交互式时间序列、分布图、箱线图、rolling 统计、缺失值分布和价格/收益关系图。
- `summary.json`：机器可读的检验结果。

检验范围：

- 基础完整性：字段是否存在、dtype、全空、NaN/inf、有效样本数、重复时间戳、时间单调性、时间间隔异常。
- 数值分布：count、mean、std、min、max、median、1/5/25/75/95/99 分位数、skew、kurtosis、zero/positive/negative ratio、unique count、constant flag。
- 异常值：z-score、IQR、分位数尾部、爆炸值、断崖跳变和对应时间点。
- 时间序列合理性：rolling mean/std/min/max/quantile、长时间全 0、长时间缺失、长时间近似常数。
- 特征关系：若价格上下文存在，报告会计算特征与价格、当前收益、未来收益的相关性，并生成特征相关性矩阵。相关性仅用于诊断，不代表确定预测能力。
- 未来函数风险：报告会对 rolling、KAMA、Gaussian、Kalman 等特征给出自动可见的人工确认提示。无法从输出数据完全证明的项会标记为需要人工确认。

状态规则：

- `FAIL`：特征缺失、数据为空、全 NaN、非数值、inf 达到阈值、有效样本不足或缺失率超过失败阈值。
- `WARN`：缺失率超过警告阈值、近似常数列、异常点、爆炸值、断崖跳变、长时间全 0/缺失/常数或时间索引异常。
- `PASS`：未触发 FAIL 或 WARN。

执行入口：

```bash
python -m btc_liqprice_features_artifact.feature_liqprice
```

单独重建报告：

```bash
python -m btc_liqprice_features_artifact.feature_diagnostics \
  --config btc_liqprice_features_artifact/configs/feature_liqprice.json
```
