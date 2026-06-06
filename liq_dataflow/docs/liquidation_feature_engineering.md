# Feature Design (Feature Engineering Project)

## 输入数据契约

当前 `liq_dataflow` 是纯特征工程项目，不再直接访问 Glassnode。
标准输入是一个 CSV，路径默认由 `configs/feature_engineering.json` 中的 `input.source_csv` 指定，默认指向同级下载项目：

- `../liq_data_download/data/raw/hourly/BTC_price_lld.csv`

如果该文件不存在，则会回退到本项目内的样例输入：

- `data/clean/clean_data.csv`

输入 CSV 可以是两种形态之一：

1. **raw hourly bundle**
   - `time`
   - `price`
   - `futures_long_liquidations`
   - `futures_short_liquidations`

2. **clean frame**
   - `time`
   - `price`
   - `fll_normal`
   - `fsl_normal`

最终合并输出文件由 `data.final_features_subdir` 和 `data.final_features_filename` 控制，当前为：

- `data/features/features_liq_dataflow.csv`

该文件中除 `time` 与 `price` 外，所有列均是当前项目的重要输出特征。

下面的特征定义均严格对应当前代码实现。

# 【特征工程文档】

## 4.1 Liquidation stress + direction (normalized)

本节描述当前 `liq_dataflow` 中与 liquidation stress / direction 直接相关、并最终用于交付的标准化特征族。当前代码严格采用 **side-first canonical definition**：

1. 先对两条边际清算序列做平滑与去噪：
   - `fll_normal -> fll_cwt_kf`
   - `fsl_normal -> fsl_cwt_kf`
2. 再由两条有效边际序列统一派生：
   - `total_ls_cwt_kf = fll_cwt_kf + fsl_cwt_kf`
   - `diff_ls_cwt_kf = fll_cwt_kf - fsl_cwt_kf`
   - `diff_dom_ls_cwt_kf = diff_ls_cwt_kf / (total_ls_cwt_kf + eps)`
   - `risk_priority_number = fll_cwt_kf / (total_ls_cwt_kf + eps)`
3. 不再独立 detrend `diff / total / sdom / RPN`，从而保证代数关系、金融语义和输出校验的一致性。
4. magnitude 类有效序列施加显式非负约束：
   - `fll_cwt_kf >= 0`
   - `fsl_cwt_kf >= 0`
   - `total_ls_cwt_kf >= 0`
5. 标准化特征统一由 `feature_engineering/model_features.py` 生成，当前 robust z-score 采用 rolling median / MAD，窗口由配置项 `model_features.z_window_bars` 控制，并使用：

\[
robust\_z_t = \frac{x_t - median_t}{1.4826 \cdot MAD_t + \varepsilon}
\]

### Canonical definition

#### Raw / Event layer

\[
L_t^{raw}:=fll\_normal, \qquad S_t^{raw}:=fsl\_normal
\]

\[
T_t^{raw}:=L_t^{raw}+S_t^{raw}, \qquad N_t^{raw}:=L_t^{raw}-S_t^{raw}
\]

\[
RPN_t^{raw}=
\begin{cases}
\dfrac{L_t^{raw}}{T_t^{raw}+\varepsilon}, & T_t^{raw}>0 \\
0.5, & T_t^{raw}=0
\end{cases}
\]

\[
sdom_t^{raw}=
\begin{cases}
\dfrac{N_t^{raw}}{T_t^{raw}+\varepsilon}, & T_t^{raw}>0 \\
0, & T_t^{raw}=0
\end{cases}
\]

并显式保留：

\[
liq\_active\_raw = \mathbf{1}(T_t^{raw}>0)
\]

#### Effective / Regime layer

先对 side series 做趋势提取与平滑：

\[
L_t^{eff}:=fll\_cwt\_kf, \qquad S_t^{eff}:=fsl\_cwt\_kf
\]

再统一派生 family：

\[
T_t^{eff}=L_t^{eff}+S_t^{eff}, \qquad N_t^{eff}=L_t^{eff}-S_t^{eff}
\]

\[
sdom_t=\dfrac{N_t^{eff}}{T_t^{eff}+\varepsilon}, \qquad RPN_t=\dfrac{L_t^{eff}}{T_t^{eff}+\varepsilon}
\]

因此当前代码中必须始终满足：

\[
RPN_t = \frac{1+sdom_t}{2}
\]

有效边际序列的生成管线为：

- preprocessing：异常值压缩与 `fll_normal / fsl_normal` 构建
- smoothing：trailing-window wavelet approximation trend extraction
- smoothing：1D Kalman smoothing
- projection：对有效 magnitude 序列做 non-negative projection
- caching：
  - `data/cache/fll_cwt.csv`
  - `data/cache/fsl_cwt.csv`

最终交付特征列为：

- `fll_cwt_kf`
- `fsl_cwt_kf`
- `diff_ls_cwt_kf`
- `total_ls_cwt_kf`
- `risk_priority_number`
- `bin_index`
- `dominance`
- `diff_dom_ls_cwt_kf`
- `z_logTotalP`
- `z_sdom`
- `z_fll_cwt_kf`
- `z_fsl_cwt_kf`

以下 12 个特征属于当前重要输出特征集。

---

### 1) fll_cwt_kf

#### Financial meaning — Effective futures long-liquidation intensity

`fll_cwt_kf` 是多头清算侧的有效强度序列。它不是 raw liquidation spike 本身，而是对 `fll_normal` 做过去窗口内的趋势提取、平滑与非负投影后的 regime-level pressure leg。

金融语义上，它表达的是：

- 多头仓位被强制平仓带来的 forced sell pressure
- downside deleveraging / long flush 环境中的侧向压力
- 计算 `total_ls_cwt_kf`、`diff_ls_cwt_kf`、`risk_priority_number` 与 `z_fll_cwt_kf` 的基础输入

#### Application

`fll_cwt_kf` 适合用作 canonical source feature，而不是最终标准化模型变量。解释时应关注其与 `fsl_cwt_kf` 的相对关系：

- `fll_cwt_kf` 高且显著高于 `fsl_cwt_kf`：多头清算压力占优
- `fll_cwt_kf` 与 `fsl_cwt_kf` 同时高：总强平压力高，但方向需要看比例
- `fll_cwt_kf` 低：多头清算不是当前主要压力腿

#### Calculation

1. Raw input:

\[
futures\_long\_liquidations \rightarrow fll\_normal
\]

2. Effective side extraction:

- trailing-window wavelet approximation trend extraction
- Kalman smoothing
- non-negative projection
- cache: `data/cache/fll_cwt.csv`

3. Output:

\[
fll\_cwt\_kf = \max(0, Kalman(WaveletTrend(fll\_normal)))
\]

---

### 2) fsl_cwt_kf

#### Financial meaning — Effective futures short-liquidation intensity

`fsl_cwt_kf` 是空头清算侧的有效强度序列。它表达空头被强制平仓带来的 forced buy pressure，是 upside squeeze pressure 的基础侧向变量。

金融语义上，它表达的是：

- 空头仓位被强制买回的强度
- short squeeze / upside burst 环境中的侧向压力
- 计算 `total_ls_cwt_kf`、`diff_ls_cwt_kf`、`risk_priority_number` 与 `z_fsl_cwt_kf` 的基础输入

#### Application

`fsl_cwt_kf` 应与 `fll_cwt_kf` 联合解释：

- `fsl_cwt_kf` 高且显著高于 `fll_cwt_kf`：空头清算压力占优
- `fsl_cwt_kf` 与 `fll_cwt_kf` 同时高：清算环境强，但未必单边
- `fsl_cwt_kf` 低：空头清算不是当前主要压力腿

#### Calculation

1. Raw input:

\[
futures\_short\_liquidations \rightarrow fsl\_normal
\]

2. Effective side extraction:

- trailing-window wavelet approximation trend extraction
- Kalman smoothing
- non-negative projection
- cache: `data/cache/fsl_cwt.csv`

3. Output:

\[
fsl\_cwt\_kf = \max(0, Kalman(WaveletTrend(fsl\_normal)))
\]

---

### 3) diff_ls_cwt_kf

#### Financial meaning — Effective net liquidation delta

`diff_ls_cwt_kf` 是 effective liquidation family 中的净方向差：

\[
diff\_ls\_cwt\_kf = fll\_cwt\_kf - fsl\_cwt\_kf
\]

它保留了 magnitude scale，因此可以同时表达方向和绝对差距。

解释上：

- `diff_ls_cwt_kf > 0`：多头清算侧更强，偏 forced sell pressure
- `diff_ls_cwt_kf < 0`：空头清算侧更强，偏 forced buy pressure
- `diff_ls_cwt_kf ≈ 0`：两侧清算压力接近平衡

#### Application

`diff_ls_cwt_kf` 是 dominance 规则层的重要输入。相比 `risk_priority_number`，它没有除以 total，因此不会丢掉绝对差距信息。

Practical guidance:

- 与 `total_ls_cwt_kf` 联合看，可以区分“高压力下的方向差”与“低压力下的小分母偏差”。
- 与 rolling quantile thresholds 联合看，用于判断 dominance 是否足够强。

#### Calculation

\[
diff\_ls\_cwt\_kf = fll\_cwt\_kf - fsl\_cwt\_kf
\]

当前实现位置：

- `feature_engineering/pipeline.py::build_canonical_liquidation_family`

---

### 4) total_ls_cwt_kf

#### Financial meaning — Effective total liquidation pressure

`total_ls_cwt_kf` 是两侧有效清算强度之和：

\[
total\_ls\_cwt\_kf = fll\_cwt\_kf + fsl\_cwt\_kf
\]

它是不带方向的清算压力总量，表示当前市场中强制去杠杆流本身有多强。

#### Application

`total_ls_cwt_kf` 是解释所有方向比例类特征的前提：

- total 高：方向特征更可能有真实市场意义
- total 低：极端比例可能来自小分母噪声
- total 持续高：可能处于 liquidation stress regime

它也是 `z_logTotalP` 的直接来源。

#### Calculation

\[
total\_ls\_cwt\_kf = fll\_cwt\_kf + fsl\_cwt\_kf
\]

当前实现要求：

- `fll_cwt_kf >= 0`
- `fsl_cwt_kf >= 0`
- `total_ls_cwt_kf >= 0`

---

### 5) diff_dom_ls_cwt_kf

#### Financial meaning — Scale-free liquidation direction dominance

`diff_dom_ls_cwt_kf` 是 effective net liquidation delta 的 scale-free 版本：

\[
diff\_dom\_ls\_cwt\_kf = \frac{diff\_ls\_cwt\_kf}{total\_ls\_cwt\_kf + \varepsilon}
\]

它把方向偏置压缩到稳定范围，用于跨清算强度 regime 比较。

#### Application

解释上：

- `diff_dom_ls_cwt_kf > 0`：多头清算占优
- `diff_dom_ls_cwt_kf < 0`：空头清算占优
- `diff_dom_ls_cwt_kf = 0`：方向中性

它与 `risk_priority_number` 是同一方向信息的两种参数化：

\[
diff\_dom\_ls\_cwt\_kf = 2 \cdot risk\_priority\_number - 1
\]

因此不应将它们当作两份独立 alpha。

#### Calculation

\[
diff\_dom\_ls\_cwt\_kf =
\begin{cases}
\dfrac{diff\_ls\_cwt\_kf}{total\_ls\_cwt\_kf+\varepsilon}, & total>0 \\
0, & total=0
\end{cases}
\]

当前实现位置：

- `feature_engineering/preprocess.py::safe_sdom`
- `feature_engineering/pipeline.py::build_canonical_liquidation_family`

---

### 6) z_logTotalP

#### Financial meaning — Total Liquidation Magnitude

`z_logTotalP` 衡量当前总 liquidation pressure 相对近期基线是否异常。它把多头清算和空头清算聚合成一个**非方向性压力强度变量**，表达的是“当前有多强的强平压力正在发生”，而不是“哪一边主导方向”。

与主动交易流不同，清算流是非自主、由杠杆与保证金机制触发、执行节奏急迫的成交流，因此它更接近：

- 市场脆弱性被真实执行后的结果层记录
- 去杠杆强度与 stress regime 的代理变量
- squeeze / cascade / churn 环境下的重要温度计

Key properties:

- 代理 market stress 与 deleveraging intensity。
- 分布通常重尾、非高斯，并容易时间聚集。
- magnitude 自身不携带方向，需要结合方向偏置特征解释。
- 可作为后续 Abs / transmission 家族的条件变量。

#### Application

将 `z_logTotalP` 理解为：

“当前总清算流相对近期正常状态，是不是已经进入异常高压区间。”

解释上：

- **低值**：清算并不是当前价格行为的主导力量。
- **中高值**：强平流已经大到足以显著影响短期路径。
- **高值**：市场进入高压力、高脆弱、可能级联的 regime。

Practical guidance:

- 必须和 `z_sdom`、`risk_priority_number` 联合解释。
- `z_logTotalP` 高 + `|z_sdom|` 高，更接近“显著强平 + 明显方向主导”。
- `z_logTotalP` 低时，即使方向比值极端，也要降低解释权重。

#### Calculation

`z_logTotalP` 是对 **log 压缩后的有效总清算强度** 做 rolling robust z-score。

1. Construct total liquidation pressure

- Raw-space：`total_ls_normal = fll_normal + fsl_normal`
- Effective-space：`TotalP_eff := total_ls_cwt_kf`
- 当前代码中：`total_ls_cwt_kf = fll_cwt_kf + fsl_cwt_kf`

2. Log transform

\[
logTotalP_t = \log(1 + TotalP_t^{eff})
\]

3. Robust rolling z-score

在窗口 `W=24` 的 rolling window 内：

\[
med_t = rolling\_median(logTotalP, W)
\]
\[
mad_t = rolling\_MAD(logTotalP, W)
\]
\[
z\_logTotalP_t = \frac{logTotalP_t - med_t}{1.4826 \cdot mad_t + \varepsilon}
\]

代码位置：

- `feature_engineering/model_features.py::build_liquidation_model_features`

---

### 7) z_sdom

#### Financial meaning — Net Liquidation Delta (normalized)

`z_sdom` 衡量当前 liquidation direction dominance 相对近期历史是否异常。它不是总量指标，而是**方向偏置指标**，回答的是：

- 当前是多头清算更占主导，还是空头清算更占主导
- 这种主导程度相对近期是否异常

定义上：

\[
sdom_t = \frac{fll\_cwt\_kf - fsl\_cwt\_kf}{fll\_cwt\_kf + fsl\_cwt\_kf + \varepsilon}
\]

因此：

- `sdom > 0`：多头清算占优，偏向强制卖出压力
- `sdom < 0`：空头清算占优，偏向强制买回压力
- `sdom ≈ 0`：方向接近平衡

Key properties:

- 这是 scale-free 的方向偏置变量，更适合跨 regime 比较。
- 它直接来自 canonical effective family，不再独立平滑。
- 它与 `risk_priority_number` 是严格仿射等价，而不是松散相关。

#### Application

将 `z_sdom` 理解为：

“当前哪一侧的强平更占主导，以及这种主导相对近期是否不寻常。”

典型用途：

- 识别 short squeeze 与 long flush 的方向背景
- 为 RC / VT / RHA / liquidation-driven market 识别提供方向输入
- 与 `z_logTotalP` 联合判断当前方向偏置是否具有真实交易意义

Practical considerations:

- 单独看 `z_sdom` 不稳，因为低总清算时小分母会放大噪音。
- 应结合 `z_logTotalP`：
  - 高 `|z_sdom|` + 低 `z_logTotalP`：更可能是小分母噪声
  - 高 `|z_sdom|` + 高 `z_logTotalP`：更可能是真正的方向主导

#### Calculation

1. Define canonical effective dominance

\[
diff\_ls\_cwt\_kf = fll\_cwt\_kf - fsl\_cwt\_kf
\]
\[
total\_ls\_cwt\_kf = fll\_cwt\_kf + fsl\_cwt\_kf
\]
\[
sdom_t = \frac{diff\_ls\_cwt\_kf}{total\_ls\_cwt\_kf + \varepsilon}
\]

2. Robust z-score

\[
z\_sdom_t = \frac{sdom_t - rolling\_median(sdom, W)}{1.4826 \cdot rolling\_MAD(sdom, W) + \varepsilon}
\]

当前实现参数：

- `W = 24`
- 别名字段：`diff_dom_ls_cwt_kf`

---

### 8) risk_priority_number

#### Financial meaning — Risk Priority Number / Liquidation Dominance Ratio

`risk_priority_number` 是多头清算占总清算的比例，用于表达 effective liquidation family 内部的方向占比关系。它把“净方向”改写成了更直观的比例形式。历史模型层别名为 `RPN`，当前最终合并输出只保留 `risk_priority_number`：

\[
risk\_priority\_number = \frac{Long\ Liqs}{Long\ Liqs + Short\ Liqs}
\]

当前实现中：

- `risk_priority_number` 高：多头清算占比高，下行去杠杆压力更强
- `risk_priority_number` 低：空头清算占比高，上行 squeeze 压力更强
- `risk_priority_number = 0.5`：方向中性 / 无有效方向信息

Boundary for Total Liq = 0:

- 若 `fll_cwt_kf + fsl_cwt_kf = 0`，代码中定义 `risk_priority_number = 0.5`

Key properties:

- 值域稳定在 `[0, 1]`
- 与 `sdom` 同源，但解释上更直观
- 适合规则层、解释层和 regime thermometer，而不是被当成全新信息源

#### Application

`risk_priority_number` 当前主要用于三类目的：

1. 作为 dominance regime 的连续比例主轴
2. 与 `bin_index` 共同构造 dominance 状态规则
3. 在中性区附近辅助判断当前市场是否正在偏离平衡

Practical guidance:

- 不要把 `risk_priority_number` 与 `z_sdom` 当成两份独立 alpha，它们本质上是同一方向信息的两种参数化。
- `risk_priority_number` 更适合解释和规则层；`z_sdom` 更适合进入 standardized model layer。

#### Calculation

- Raw-space：

\[
RPN_t^{raw}=
\begin{cases}
\dfrac{fll\_normal}{fll\_normal+fsl\_normal+\varepsilon}, & total>0 \\
0.5, & total=0
\end{cases}
\]

- Effective-space：

\[
RPN_t=
\begin{cases}
\dfrac{fll\_cwt\_kf}{fll\_cwt\_kf+fsl\_cwt\_kf+\varepsilon}, & total>0 \\
0.5, & total=0
\end{cases}
\]

并满足：

\[
RPN_t = \frac{1+sdom_t}{2}
\]

当前实现位置：

- `feature_engineering/preprocess.py`
- `feature_engineering/pipeline.py`
- `feature_engineering/model_features.py`

---

### 9) bin_index

#### Financial meaning — Ordinal bin index of risk_priority_number

`bin_index` 是 `risk_priority_number` 的有序离散化结果。它不是新的金融原语，而是把连续的 `risk_priority_number` 压缩成 **0..8 的 ordinal regime label**，用于提高解释性与规则层稳定性。

Interpretation is ordered, not metric：

- `0` 表示最低 long-liquidation dominance（最偏 short-liquidation dominance）
- `8` 表示最高 long-liquidation dominance
- 中间 bins 表示不同程度的平衡区或过渡区

当前实现不再使用 KMeans，而是使用 **point-in-time expanding quantile binning**。

#### Application

`bin_index` 主要用于：

- 提供离散 regime label
- 为 dominance state 提供 regime gating
- 降低对连续 `risk_priority_number` 微小波动的敏感度

Practical guidance:

- 它是 rank-order label，不应解释为等距距离。
- 由于是 past-only quantile binning，不存在 full-sample KMeans 带来的未来泄漏。

#### Calculation

定义：

\[
bin\_index_t \in \{0,1,\dots,8\}
\]

构造方式：

1. 对于时点 `t`，只使用 `t` 之前的历史 `risk_priority_number` 样本形成 expanding history
2. 当历史样本长度达到 `min_history_bars = 72` 之后，按 `n_bins = 9` 计算分位数边界
3. 每隔 `refit_every_bars = 24` 根 bar 重新拟合一次边界
4. 当前值按 `np.searchsorted(bounds, value, side="right")` 映射到 `0..8`
5. 历史不足时，使用中性档 `neutral_bin = 4`

当前实现位置：

- `feature_engineering/binning.py::point_in_time_quantile_binning`

---

### 10) z_fll_cwt_kf

#### Financial meaning — Effective futures long-liquidation intensity (standardized)

`z_fll_cwt_kf` 衡量当前**有效多头清算强度**相对近期是否异常。它描述的是强制卖出这一侧的压力腿是否进入异常状态。

多头清算通常对应：

- 持续下跌中的 trend-driven deleveraging
- 上涨趋势中的 sharp flush-out event
- downside cascade 环境中的 forced unwind flow

Key properties:

- 它是 side-specific intensity leg
- 更接近 downside deleveraging pressure 的代理变量
- 经平滑后更适合 regime inference，而非逐点事件精确定位

#### Application

典型理解方式：

- `z_fll_cwt_kf` 高：多头清算腿异常强，可能处于 downside deleveraging 区间
- `z_fll_cwt_kf` 中高且持续：去杠杆压力有持续性
- 价格最终是否继续下行，仍需结合吸收能力与 price response 判断

#### Calculation

1. Raw data source

- `futures_long_liquidations -> fll_normal`

2. Processing pipeline

- outlier compression
- trailing-window wavelet approximation trend extraction
- Kalman smoothing
- non-negative projection
- output: `fll_cwt_kf`

3. Standardization

\[
z\_fll\_cwt\_kf = robust\_z(\log(1 + fll\_cwt\_kf))
\]

当前实现参数：

- rolling window：`24`
- 代码位置：`feature_engineering/model_features.py`

---

### 11) z_fsl_cwt_kf

#### Financial meaning — Effective futures short-liquidation intensity (standardized)

`z_fsl_cwt_kf` 衡量当前**有效空头清算强度**相对近期是否异常。它描述的是强制买回这一侧的压力腿是否进入异常状态。

空头清算通常对应：

- 持续上涨中的 short squeeze continuation
- bear-market rally 中的急促逼空
- upside burst 环境中的 forced buy flow

Key properties:

- 它是 side-specific intensity leg
- 更接近 upside squeeze pressure 的代理变量
- 经平滑后更适合 regime inference，而不是逐笔事件检测

#### Application

典型理解方式：

- `z_fsl_cwt_kf` 高：空头清算腿异常强，可能存在 squeeze-driven demand
- 若价格被有效吸收，则即使 `z_fsl_cwt_kf` 高，也未必出现持续上行
- 对 short risk control 尤其重要

#### Calculation

1. Raw data source

- `futures_short_liquidations -> fsl_normal`

2. Processing pipeline

- outlier compression
- trailing-window wavelet approximation trend extraction
- Kalman smoothing
- non-negative projection
- output: `fsl_cwt_kf`

3. Standardization

\[
z\_fsl\_cwt\_kf = robust\_z(\log(1 + fsl\_cwt\_kf))
\]

当前实现参数：

- rolling window：`24`
- 代码位置：`feature_engineering/model_features.py`

---

## 4.2 Dominance state (rule-derived)

`dominance` 建立在 canonical family 之上，是当前最终合并输出中的规则状态特征。它用来表达当前清算主导状态是否已经强到足以被规则层正式判定。

### 12) dominance

#### Financial meaning — Rule-derived liquidation dominance state

`dominance` 用离散状态表达当前是否处于：

- `1`：**FSL dominant / upward pressure**
- `-1`：**FLL dominant / downward pressure**
- `0`：neutral / congestion / no confirmed dominance

它和 `diff_dom_ls_cwt_kf`、`risk_priority_number` 的区别在于：

- `diff_dom_ls_cwt_kf` / `risk_priority_number` 表达的是连续比例偏置
- `dominance` 表达的是“当前偏置是否已经强到足以被判定为状态”

因此它更像一个 **state flag**，而不是一个连续强度变量。

Key properties:

- 离散、可解释、适合规则层 gating
- 与 `bin_index`、`risk_priority_number`、`diff_ls_cwt_kf` 联合决定
- 允许存在大量 `0`，表示当前没有充分证据确认单边主导

#### Application

`dominance` 的当前主要用途：

1. 作为自定义专题图中的填充状态
2. 作为 `hit_ceiling_bottom` / `reverse_ceiling_bottom` 的上游语境
3. 作为 feature portal 中的重要审阅特征
4. 可用于过滤低质量方向信号，只在 dominance 成立时解释相应方向

解释上：

- `dominance = 1`：当前更偏 FSL dominant，对应 upward pressure regime
- `dominance = -1`：当前更偏 FLL dominant，对应 downward pressure regime
- `dominance = 0`：当前仍在平衡区 / 证据不足 / 不应强行解释单边主导

#### Calculation

当前代码完全由 `feature_engineering/dominance.py::build_dominance_features` 实现。

1. 先构造 rolling thresholds

对 `diff_ls_cwt_kf` 计算过去一年的 rolling quantile：

- `thr_diff_pos = rolling Q80(diff)`
- `thr_diff_neg = rolling Q20(diff)`
- `thr_diff_pos_base = rolling Q60(diff)`
- `thr_diff_neg_base = rolling Q40(diff)`

参数：

- `rolling_window_bars = 365 * 24`
- `rolling_min_periods = 30 * 24`

2. Regime + threshold gating

bear 状态（下行压力 / FLL dominant）定义为：

\[
bear_t = \Big[(bin_t \ge 5) \land (diff_t \ge Q80_t)\Big]
\ \lor \ 
\Big[(bin_t = 4) \land (risk\_priority\_number_t \ge 0.52) \land (diff_t \ge Q60_t)\Big]
\]

bull 状态（上行压力 / FSL dominant）定义为：

\[
bull_t = \Big[(bin_t \le 3) \land (diff_t \le Q20_t)\Big]
\ \lor \ 
\Big[(bin_t = 4) \land (risk\_priority\_number_t \le 0.48) \land (diff_t \le Q40_t)\Big]
\]

最终状态：

\[
dominance_t =
\begin{cases}
1, & bull_t \\
-1, & bear_t \\
0, & otherwise
\end{cases}
\]

3. Associated rule-state columns

在 `dominance` 基础上，代码还派生：

- `dominance_duration`
- `dominance_duration_total`
- `dominance_last`
- `dominance_prev`
- `dominance_class`
- `is_keep`
- `is_strengthen`
- `dominance_time`
- `hit_ceiling_bottom`
- `reverse_ceiling_bottom`

这些列属于 dominance 规则层的辅助变量，不属于最终合并输出的核心交付字段，但属于当前自定义可视化与状态解释的重要上下文。

#### Notes aligned with current code

- `dominance` 是 ternary state，当前输出校验要求其取值严格属于 `{-1, 0, 1}`。
- 在项目的自定义作图中，颜色约定为：
  - `dominance = -1`：FLL Dominant，粉红填充
  - `dominance = 1`：FSL Dominant，浅绿填充
- `dominance` 是规则状态，不应与 `z_sdom` 当作同一层级的独立连续信息源。
