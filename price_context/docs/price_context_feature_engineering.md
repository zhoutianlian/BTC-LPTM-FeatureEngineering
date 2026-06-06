# price_context 特征工程文档

## 1. 总体定位

`price_context` 只使用 OHLC 价格数据，刻画“价格路径本身已经表现出来的市场上下文”。它不尝试从价格中伪造清算压力、PLIE、path_context 或 market_response，而是为六状态识别提供价格侧证据：市场是否在压缩、是否趋势化、趋势是否有序、波动是否稳定、是否存在跳跃冲击，以及当前特征是否受到数据质量影响。

核心因果链定位：

```text
上游宏观/流动性/资金流/清算/新闻冲击
→ 订单流与流动性变化
→ OHLC 价格路径、波动、区间、跳跃
→ price_context 特征
→ 六状态识别或市场状态分析
```

本项目的特征主要位于“可观测市场结果”层，不是上游原因本身。因此使用时应与 PLIE、path_context、资金费率、basis、OI、ETF flow 等上游特征联合判断。

## 2. 基础计算原则

### 2.1 收益率单位

对相邻 bar：

```text
r_i = 10000 * log(close_i / close_{i-1})
```

单位为 bps。使用对数收益可以保证路径收益可加，且在高波动资产中比简单收益更稳定。

### 2.2 时间窗口

默认 10m bar：

| 窗口 | bars |
|---|---:|
| 1h | 6 |
| 3h | 18 |
| 6h | 36 |
| 12h | 72 |
| 24h | 144 |
| 48h | 288 |

核心 rolling 计算优先使用时间窗口，例如 `rolling("6h")`。这比固定行数更适合处理不规则时间戳。对于 block consistency 和部分 bipower 计算，会使用由 `bar_minutes` 推导的 bar 数；若数据有缺口，质量字段会显式标记。

### 2.3 防未来函数

本项目避免以下泄漏：

- 当前特征不使用未来 OHLC；
- z-score 不使用全样本统计量；
- percentile 使用当前及历史窗口的 rank，并调整为 past-only percentile；
- robust sigma 默认使用历史 rolling IQR 或可选 rolling MAD；
- train split 只有在配置了明确起止时间时才用于 scaler，否则回退到 past-only rolling scaler。

## 3. Past return

### 3.1 计算

```text
past_return_W_bps(t) = 10000 * log(close_t / close_{t-W})
```

输出字段：

- `past_return_1h_bps`
- `past_return_3h_bps`
- `past_return_6h_bps`
- `past_return_12h_bps`
- `past_return_24h_bps`

`close_{t-W}` 使用 `merge_asof` 查找 `t-W` 或其之前、且在一个 bar 容忍范围内的历史 close，不使用未来 anchor。

### 3.2 金融学含义

past return 衡量窗口内价格是否发生了净位移。它回答：

- 市场在这个窗口内是否已经明显上行或下行？
- 当前状态是否具有方向性结果？
- 短期和中期方向是否一致？

它不是趋势强度本身。一次跳跃也可能造成大收益，但路径并不一定有序，因此 past return 必须与 trend efficiency、trend consistency 和 jump proxy 联合使用。

### 3.3 应用

在六状态中：

- RC：past return 通常较小；
- ST：past return 有方向，但波动不过度极端；
- VT：past return 较大，且常伴随较高 vol；
- HPEM：past return 可能较大，但需要 PLIE/cascade 同向确认；
- RHA：可能出现与压力方向相反的 past return；
- AMB：多窗口 past return 方向冲突时提供 ambiguity 证据。

## 4. Realized volatility

### 4.1 计算

窗口总 realized volatility：

```text
RV_W(t) = sqrt(sum(r_i^2)), i in [t-W, t]
```

标准化到每 sqrt hour：

```text
RV_per_sqrt_hour_W(t) = sqrt(sum(r_i^2) / W_hour)
```

输出字段：

- `realized_vol_1h_bps`
- `realized_vol_6h_bps`
- `realized_vol_24h_bps`
- `realized_vol_1h_per_sqrt_hour_bps`
- `realized_vol_6h_per_sqrt_hour_bps`
- `realized_vol_24h_per_sqrt_hour_bps`
- `realized_vol_1h_z`
- `realized_vol_6h_z`
- `realized_vol_24h_z`

z-score：

```text
z(x) = (x - median_train_or_past(x)) / (IQR_train_or_past(x) + epsilon)
```

并 clip 到 `[-5, 5]`。

### 4.2 金融学含义

realized vol 衡量价格路径实际释放的风险强度。它回答：

- 当前市场是否足够剧烈？
- 剧烈程度是短期冲击，还是 24h 背景已经高波？
- 市场处于安静整理、正常趋势、高波趋势还是极端运动？

vol 是风险释放结果，不直接说明上游原因。高 vol 可能来自主动趋势资金，也可能来自清算、新闻、流动性枯竭或跳跃。

### 4.3 六状态应用

- RC：1h/6h/24h realized vol 偏低；
- ST：中等 vol，趋势有序；
- VT：vol 高，尤其 6h 高；
- HPEM：vol 高且常伴随 jump/cascade；
- RHA：可以中高，但高 vol 不能单独定义 RHA；
- AMB：高 vol 且证据冲突时提高 AMB 概率。

## 5. Range width

### 5.1 计算

```text
range_width_W_bps = 10000 * log(max(high in [t-W,t]) / min(low in [t-W,t]))
```

输出字段：

- `range_width_1h_bps`
- `range_width_6h_bps`
- `range_width_24h_bps`

### 5.2 金融学含义

range width 衡量窗口内价格覆盖的高低区间。它回答：

- 市场是否已经展开足够空间？
- 价格是否仍被压在狭窄区间？
- 当前波动是区间展开，还是路径内部来回扫？

range width 与 realized vol 不同：vol 是路径能量，range 是空间覆盖。高频来回扫可能 RV 高但 range 不一定显著；单边趋势可能 range 与 RV 同时高。

### 5.3 应用

- RC：range width 通常低；
- VT/HPEM：range width 常扩大；
- ST：range width 适中到较高，但通常伴随 trend consistency；
- AMB：range 与 vol/趋势证据不一致时有解释价值。

## 6. Range compression

### 6.1 计算

先计算 `range_width_W_bps`，再计算 past-only percentile：

```text
RangePct_W(t) = PastPercentile_L(range_width_W(t))
RangeCompression_W(t) = 1 - RangePct_W(t)
```

输出字段：

- `range_compression_1h`
- `range_compression_6h`
- `range_compression_24h`

### 6.2 金融学含义

range compression 衡量当前区间相对历史是否窄。它回答：

- 市场是否在等待、整理、蓄势？
- 价格是否处于低交易价值区？
- 趋势是否尚未展开？

低 realized vol 不等于 RC。缓慢爬升可能 vol 低但趋势清晰；真正的 RC 更应表现为 range compression 高、trend strength 低、conflict 低。

### 6.3 应用

- RC：核心证据，高 compression；
- ST：compression 不应极高；
- VT/HPEM：compression 通常下降；
- RHA：吸收后 compression 可能逐步升高；
- AMB：压缩但方向/路径证据冲突时需谨慎。

## 7. Range to vol

### 7.1 计算

```text
range_to_vol_W = range_width_W_bps / (realized_vol_W_bps + epsilon)
```

输出字段：

- `range_to_vol_1h`
- `range_to_vol_6h`
- `range_to_vol_24h`

### 7.2 金融学含义

range_to_vol 比较“空间展开”与“路径消耗”。它回答：

- 同样的波动能量是否带来了有效位移？
- 市场是单边推进，还是来回扫单？
- 当前 range 扩大是否主要由趋势造成？

### 7.3 应用

- 单边趋势：range_to_vol 通常较高，trend efficiency 也较高；
- 高频震荡：RV 高但 range_to_vol 可能较低；
- HPEM 初段：jump 可能拉大 range_to_vol，但 consistency 未必高。

## 8. Trend efficiency

### 8.1 计算

```text
R_W(t) = 10000 * log(close_t / close_{t-W})
PathLength_W(t) = sum(abs(r_i))
TrendEfficiency_W(t) = abs(R_W(t)) / (PathLength_W(t) + epsilon)
```

输出字段：

- `trend_efficiency_1h`
- `trend_efficiency_6h`
- `trend_efficiency_24h`

### 8.2 金融学含义

trend efficiency 衡量窗口总位移相对于路径总消耗的效率。它回答：

- 价格路径是否单边？
- 收益是有效推进，还是来回震荡后的净结果？
- 市场是否从整理进入方向性状态？

### 8.3 应用

- RC/AMB：通常低；
- ST：较高；
- VT：较高但可能伴随噪声；
- HPEM：可能高，但需要 jump/PLIE/cascade 进一步区分。

## 9. Trend SNR

### 9.1 计算

```text
TrendSNR_W(t) = abs(R_W(t)) / (RV_W(t) + epsilon)
```

输出字段：

- `trend_snr_1h`
- `trend_snr_6h`
- `trend_snr_24h`

### 9.2 金融学含义

Trend SNR 衡量净收益相对于窗口波动是否显著。它回答：

- 当前方向是否足够清晰？
- 价格位移是否只是高波环境中的随机结果？
- 高收益是否被同等高的路径噪声抵消？

### 9.3 应用

SNR 高更支持 ST/VT；SNR 低而 RV 高更偏震荡、冲击或 AMB。

## 10. Trend slope / t-stat / R²

### 10.1 计算

对窗口内 log close 做 OLS：

```text
log(P_i) = alpha + beta * tau_i + error_i
```

其中 `tau_i` 为距离样本起点的小时数。输出：

- `trend_slope_1h`、`trend_slope_6h`、`trend_slope_24h`
- `trend_slope_tstat_1h`、`trend_slope_tstat_6h`、`trend_slope_tstat_24h`
- `trend_r2_1h`、`trend_r2_6h`、`trend_r2_24h`

### 10.2 金融学含义

slope 表示方向性斜率，t-stat 衡量斜率相对于残差噪声是否显著，R² 衡量价格路径是否接近线性趋势。它回答：

- 趋势是否持续、有序？
- 趋势是否由少数点造成？
- 价格路径是否有稳定方向结构？

### 10.3 应用

- ST：t-stat 与 R² 通常较高；
- VT：t-stat 可能高，但 R² 受高波噪声影响；
- HPEM：跳跃型 HPEM 可能 t-stat 高但 R² 未必稳定；
- RC/AMB：通常较低或与其他特征冲突。

## 11. Trend strength

### 11.1 计算

```text
squash(x) = 1 - exp(-x / c)

trend_strength_W =
    0.35 * clip(trend_efficiency_W, 0, 1)
  + 0.35 * squash(abs(trend_snr_W))
  + 0.30 * squash(abs(trend_slope_tstat_W))
```

默认 `c=2.0`。输出：

- `trend_strength_1h`
- `trend_strength_6h`
- `trend_strength_24h`

### 11.2 金融学含义

trend strength 衡量价格是否真的形成方向性路径，而不是单纯收益大。它结合：

- 路径效率；
- 收益/波动信噪比；
- 线性趋势统计显著性。

它回答：

- 市场是否已经从 RC/AMB 中脱离？
- 当前价格是否在有效位移？
- 趋势证据是否足够强？

### 11.3 应用

- RC：低；
- ST：中高，且 consistency 高、vol 不极端；
- VT：高，且 vol 高；
- HPEM：高，但需要 PLIE/cascade/jump 同向；
- AMB：strength 高但其他证据冲突时可能升高。

## 12. Trend direction

### 12.1 计算

```text
trend_direction_W = sign(past_return_W_bps)
```

收益绝对值小于 `near_zero_return_bps` 时记为 0。

输出字段：

- `trend_direction_1h`
- `trend_direction_6h`
- `trend_direction_24h`

### 12.2 金融学含义

trend direction 给出价格侧方向，但不说明趋势来源。方向可能来自主动买卖、清算、空头回补、宏观消息或流动性真空。

### 12.3 应用

与 PLIE/path direction 对比可以判断：

- 同向 cascade：更支持 HPEM；
- 反向接管：可能支持 RHA；
- 方向冲突：可能提高 AMB。

## 13. Bar direction align

### 13.1 计算

令：

```text
s_W = sign(R_W)
BarAlign_W = mean(1[sign(r_i) = s_W])
```

输出字段：

- `bar_direction_align_1h`
- `bar_direction_align_6h`
- `bar_direction_align_24h`

如果窗口总收益接近 0，则该指标为 missing。

### 13.2 金融学含义

bar direction align 衡量单根 bar 是否与窗口总方向一致。它回答：

- 趋势是否由多数 bar 连续推动？
- 是否只是少数极端 bar 改变了窗口收益？
- 价格路径是否有序？

### 13.3 应用

高 bar align 更支持 ST；低 bar align 但 jump 高，可能是 HPEM 初段、新闻冲击或 AMB。

## 14. Block direction align

### 14.1 计算

把窗口拆成子窗口：

- 1h 默认拆成 3 个 20m block；
- 6h 默认拆成 6 个 1h block；
- 24h 默认拆成 24 个 1h block。

```text
BlockAlign_W = mean(1[sign(R_block_j) = sign(R_W)])
```

输出字段：

- `block_direction_align_1h`
- `block_direction_align_6h`
- `block_direction_align_24h`

### 14.2 金融学含义

block align 衡量更粗粒度的趋势连续性。它比 bar align 更稳健，能减少单根 bar 噪声。

### 14.3 应用

- ST：block align 往往较高；
- VT：中高，但可能受波动噪声扰动；
- HPEM：若连续清算传导，block align 可高；若单点穿透，则不一定高；
- AMB：多 block 方向冲突时更可能提高 ambiguity。

## 15. Trend consistency

### 15.1 计算

```text
trend_consistency_W =
    0.40 * bar_direction_align_W
  + 0.40 * block_direction_align_W
  + 0.20 * trend_r2_W
```

输出字段：

- `trend_consistency_1h`
- `trend_consistency_6h`
- `trend_consistency_24h`

### 15.2 金融学含义

trend strength 说明“走得够不够远”，trend consistency 说明“走得是否有序”。它回答：

- 趋势是否连续？
- 当前强收益是否由跳跃或扫单造成？
- 市场是否处于有序趋势还是高波混沌？

### 15.3 应用

| 组合 | 解释 |
|---|---|
| strength 高、consistency 高 | 有序趋势，偏 ST/VT/HPEM，需看上游压力 |
| strength 高、consistency 低 | 跳跃、扫单、新闻冲击、AMB 或 HPEM 初段 |
| strength 低、consistency 低 | 震荡、无方向，偏 RC/AMB |
| strength 低、consistency 高 | 缓慢爬行，可能是低波 ST |

## 16. Vol of vol

### 16.1 计算

先计算滚动 1h realized vol：

```text
RV_1h(u)
```

再在窗口 W 内计算：

```text
vol_of_vol_W = std(RV_1h(u)) / (mean(RV_1h(u)) + epsilon)
vol_of_vol_abs_W = std(RV_1h(u))
```

输出字段：

- `vol_of_vol_6h`
- `vol_of_vol_24h`
- `vol_of_vol_48h`
- `vol_of_vol_abs_6h`
- `vol_of_vol_abs_24h`
- `vol_of_vol_abs_48h`
- `vol_of_vol_6h_z`
- `vol_of_vol_24h_z`
- `vol_of_vol_48h_z`

### 16.2 金融学含义

vol_of_vol 衡量波动率本身是否稳定。它回答：

- 市场是否从安静突然切到剧烈？
- 是否处在 regime transition？
- 流动性是否时好时坏？
- 冲击后风险释放是否不稳定？

BTC 中很多状态切换不是单纯 vol 高，而是 vol 结构突然变得不稳定。

### 16.3 应用

- RC：低；
- ST：低到中等；
- VT：中高；
- HPEM：高，尤其伴随 jump/cascade；
- RHA：中高，常见于级联后吸收阶段；
- AMB：高且证据冲突时有用，但不能单独定义 AMB。

## 17. Robust jump z-score

### 17.1 计算

默认生产配置使用 past-only rolling IQR robust scale：

```text
sigma_robust ≈ IQR_past / 1.349
jump_z_i = abs(r_i) / (sigma_robust_i + epsilon)
```

代码也支持把 `robust_sigma_estimator` 设为 `mad`，使用：

```text
sigma_robust = 1.4826 * median(abs(r_i - median(r)))
```

但精确 rolling MAD 在大样本 7d 窗口下计算较慢，因此默认采用 IQR robust scale 作为工程化近似。两者都只使用过去窗口，不使用当前之后的数据。

输出字段：

- `max_jump_z_1h`
- `max_jump_z_6h`
- `max_jump_z_24h`

### 17.2 金融学含义

jump z-score 衡量单根收益相对于过去正常波动尺度是否异常。它回答：

- 是否出现离散冲击？
- 当前高波是平滑趋势还是突然扫穿？
- 是否可能有新闻、强制去杠杆、流动性真空？

### 17.3 应用

jump z 高不自动等于坏数据；在 BTC 中真实跳跃有重要状态信息。默认保留数据并打标。

## 18. Jump count

### 18.1 计算

```text
jump_count_W = count(jump_z_i > theta), i in [t-W,t]
```

默认 `theta=5.0`。输出字段：

- `jump_count_1h`
- `jump_count_6h`
- `jump_count_24h`

### 18.2 金融学含义

jump count 衡量窗口内离散冲击的频次。一个 max jump 可能只是单点新闻，而多个 jump 可能代表持续扫单、流动性破裂或清算链条。

### 18.3 应用

- HPEM：jump count 常升高，并需要上游 PLIE/cascade 同向；
- VT：jump count 可中高，但压力解释力可能较低；
- AMB：jump 与路径/压力证据矛盾时更重要。

## 19. Bipower variation jump ratio

### 19.1 计算

```text
RV_W = sum(r_i^2)
BV_W = (pi / 2) * sum(abs(r_i) * abs(r_{i-1}))
JumpVar_W = max(RV_W - BV_W, 0)
JumpRatio_W = JumpVar_W / (RV_W + epsilon)
```

输出字段：

- `jump_ratio_bv_1h`
- `jump_ratio_bv_6h`
- `jump_ratio_bv_24h`

1h 样本较少，`jump_ratio_bv_1h` 不如 6h/24h 稳定，当前作为扩展字段输出。

### 19.2 金融学含义

bipower variation 尝试把连续波动与跳跃波动分开。它回答：

- 窗口 RV 中有多少可能来自跳跃？
- 高波动是连续趋势，还是由离散冲击主导？

### 19.3 应用

- HPEM：JumpRatio 高且上游清算压力同向时更有解释力；
- VT：JumpRatio 高但 PLIE neutral 时可能是主动资金或新闻驱动；
- ST：通常不应长期依赖高 JumpRatio。

## 20. Jump proxy

### 20.1 计算

```text
jump_proxy_W =
    0.50 * squash(max_jump_z_W)
  + 0.30 * clip(jump_ratio_bv_W, 0, 1)
  + 0.20 * squash(jump_count_W)
```

输出字段：

- `jump_proxy_1h`
- `jump_proxy_6h`
- `jump_proxy_24h`

### 20.2 金融学含义

jump proxy 是离散冲击综合评分。它回答：

- 市场是否被突然扫穿？
- 高波动是否由跳跃主导？
- 当前是否可能处于状态切换触发点？

### 20.3 应用

- RC：低；
- ST：低到中等；
- VT：中高但压力可能 neutral；
- HPEM：高且 PLIE/cascade 同向；
- RHA：高 jump 后若反向接管，需结合 response；
- AMB：jump 与 path/response/PLIE 冲突时增加 ambiguity。

## 21. Price missing ratio

### 21.1 计算

```text
price_obs_count_W = count(valid OHLC observations in W)
price_expected_count_W = W_minutes / bar_minutes
price_missing_ratio_W = 1 - price_obs_count_W / price_expected_count_W
```

输出字段：

- `price_missing_ratio_1h`
- `price_missing_ratio_6h`
- `price_missing_ratio_24h`
- 扩展：`price_missing_ratio_48h`、`price_missing_ratio_3h`、`price_missing_ratio_12h`

### 21.2 金融学含义

missing ratio 是特征可信度约束，不是交易信号。它回答：

- 当前窗口内 OHLC 数据是否足够完整？
- 某个 NaN 特征是否因历史不足或缺失导致？
- 当前状态判断是否应降低置信度？

### 21.3 应用

若 missing ratio 高，应避免把状态变化误判为市场机制变化。数据缺失可能让 realized vol、range 或 trend 被低估。

## 22. Price gap flag

### 22.1 计算

如果窗口内存在时间间隔超过 `gap_tolerance_minutes`，或者窗口观测数少于 expected count，则：

```text
price_gap_flag_W = 1
```

输出字段：

- `price_gap_flag_1h`
- `price_gap_flag_6h`
- `price_gap_flag_24h`

### 22.2 金融学含义

gap flag 表示数据时间结构不连续。对于 BTC 这类 24/7 市场，时间缺口尤其需要关注，因为它可能来自交易所中断、数据源问题或采集延迟。

### 22.3 应用

gap flag 为 1 时，趋势、vol、jump 的解释力要降权。否则可能把数据缺口误认为价格跳跃或流动性冲击。

## 23. Price outlier flag

### 23.1 计算

单根 return 的 robust z 超过阈值后，窗口 outlier flag 为 1：

```text
price_outlier_flag_W = max(single_bar_outlier_flag_i), i in [t-W,t]
```

输出字段：

- `price_outlier_flag_1h`
- `price_outlier_flag_6h`
- `price_outlier_flag_24h`

### 23.2 金融学含义

outlier flag 标记极端收益，但不直接判定为坏数据。它回答：

- 当前窗口内是否有异常大的单根价格变化？
- 是否需要结合 jump proxy 判断离散冲击？
- 是否需要人工检查数据源？

### 23.3 应用

真实跳跃是状态识别的重要信息，尤其对 HPEM、VT、RHA 和 AMB。默认不删除极端收益，而是保留并标记。

## 24. 当前项目不可计算的外部上游特征

以下特征不属于纯 OHLC price context，本项目不会编造：

- `PLIE`
- `liquidation pressure`
- `path_context`
- `path_label`
- `path_absorption_score`
- `path_pressure_rejection_score`
- `path_active_dominance_score`
- `path_transmission_ratio`
- `path_cascade_score`
- `market_response`

原因：这些字段对应清算压力、参与者行为、压力传播、吸收/拒绝和主动资金接管等上游机制，不能从 OHLC 中可靠反推出。OHLC 只能看到价格结果，不能唯一识别原因。

## 25. 如何接入外部上游模块

后续可以新增 `path_context` 或 `plie_context` 模块，输入字段建议包括：

- `plie_direction`
- `plie_reliability`
- `plie_intensity`
- `liquidation_pressure_bps`
- `path_label_W`
- `path_absorption_score_W`
- `path_cascade_score_W`
- `market_response_score_W`

接入方式：

1. 确保上游字段有明确发布时间和可得时间；
2. 以 `time` 做 as-of join，只能连接当前及过去可得值；
3. 输出外部机制特征，不覆盖 price_context 字段；
4. 在状态模型中使用 interaction，例如：
   - `jump_proxy_6h * path_cascade_score_6h`
   - `trend_direction_6h != plie_direction_6h`
   - `range_compression_6h * path_neutral_pressure_flag`

这样可以把价格结果与上游机制分开，避免把价格变化误当成原因本身。

## 26. 输出特征检验与可视化报告

项目新增自动化诊断模块，用于快速检查本文件中列出的所有 price_context 输出特征，并默认把实际 CSV 中额外输出的数值字段一并纳入报告。报告不会改变任何既有特征计算结果，只读取特征宽表和可选 OHLC close 价格，生成本地 HTML。

默认输出目录：

```text
reports/feature_diagnostics/
├── index.html
├── summary.json
├── assets/
│   ├── css/report.css
│   ├── js/plotly.min.js
│   ├── js/table.js
│   └── js/feature_charts.js
└── features/
    ├── past_return_1h_bps.html
    └── ...
```

### 26.1 自动生成方式

完整 pipeline 执行后会自动生成报告：

```bash
python -m price_context.feature_price_context
```

如果只想基于已有特征 CSV 重新生成报告：

```bash
python -m price_context.src.diagnostics --config price_context/configs/feature_price_context.json
```

配置项位于 `price_context/configs/feature_price_context.json` 的 `report` section：

```yaml
report:
  enabled: true
  output_dir: "reports/feature_diagnostics"
  documentation_file: "docs/price_context_feature_engineering.md"
  generate_html: true
  generate_summary_json: true
  include_actual_output_features: true
  rolling_window_bars: 144
```

### 26.2 总览页含义

`index.html` 汇总：

- 报告生成时间、数据时间范围、样本数；
- 文档特征数量、实际存在数量、缺失数量；
- PASS / WARN / FAIL 特征数量；
- 所有特征统计表，支持浏览器内搜索和点击表头排序；
- documented feature 相关性热力图；
- 高相关特征对；
- 防未来函数相关的自动检查和需要人工确认项。

当前实现会从本文件自动识别“输出字段”列表。本文件中列出的 86 个输出特征均被视为重要特征；若实际输出缺失，会在总览页和 `summary.json` 中标记为 FAIL。

### 26.3 单特征页含义

`features/<feature_name>.html` 包含：

- 特征定义、类别和关键问题摘要；
- count、mean、std、min、max、median、1%/5%/25%/75%/95%/99% 分位数、skew、kurtosis、zero/positive/negative ratio、unique count、constant flag；
- 全历史时间序列图，支持 hover、拖拽缩放、range slider 和 1D/1W/1M/3M/6M/1Y/ALL 时间按钮；
- histogram、近似密度曲线、均值/中位数/极端分位线；
- 箱线图；
- rolling mean、rolling std、rolling q05/q95；
- 缺失值时间分布和 rolling missing ratio；
- z-score outlier 和断崖式跳变标记；
- 如果原始 OHLC 可加载，则展示 feature vs close、feature vs short return、feature vs future 1h return 的诊断散点图。

为控制全量历史报告体积，单特征页采用客户端渲染：HTML 中只保存该特征一份全历史数组，浏览器本地计算分布和 rolling 图。时间序列仍展示全部历史样本；关系散点图按配置抽样，仅用于诊断展示。

### 26.4 PASS / WARN / FAIL 规则

FAIL 通常表示报告无法可靠使用该特征：

- 文档要求的输出字段缺失；
- 列存在但没有有限数值；
- 有效样本数低于 `report.min_valid_count`；
- 缺失比例达到 `report.fail_missing_ratio`；
- inf 比例达到 `report.fail_inf_ratio`；
- 输出时间戳重复或非单调。

WARN 表示需要人工检查但不一定是错误：

- 缺失比例超过 `report.warn_missing_ratio`；
- 出现 NaN / inf；
- 近似常数列；
- z-score 异常值、IQR 异常值或分位数极端值较多；
- 出现大幅单步跳变；
- 长时间不变、长时间全 0 或长时间全空；
- 时间轴存在异常间隔；
- bounded feature 超出预期范围，例如 compression、ratio、proxy、align、r2 超出 `[0, 1]`；
- direction 或 flag 字段出现非预期枚举值。

PASS 表示未触发当前规则，不代表特征一定有预测能力。

### 26.5 相关性与未来函数说明

如果原始 OHLC close 可以按 `time` 对齐，报告会计算：

- 特征与 close 的相关性；
- 特征与 1 bar return 的相关性；
- 特征与 future 1h return 的相关性；
- 特征之间的相关性矩阵和高相关特征对。

这些相关性只用于诊断特征形态、冗余和潜在异常，不能解释为确定预测能力。

CSV 级报告可以自动检查：

- `price_feature_time <= time`；
- `price_feature_age_min >= 0`；
- 是否存在名称包含 future / fwd / target / label 的疑似标签列。

rolling 是否 past-only、shift 方向是否正确、as-of join 是否只向过去查找，仍需结合源码和测试人工确认。项目已有 `tests/test_no_future_leakage.py` 覆盖“修改未来价格不改变过去特征”的核心回归测试。

### 26.6 常见排查方式

- 文档特征缺失：检查 `_ordered_columns`、`include_extended_features` 和对应 compute 模块是否输出该列。
- 大量初期 NaN：长窗口、30d rolling percentile/z-score 或 past return anchor 历史不足通常会导致初期为空，先查看缺失是否集中在样本开头。
- gap flag 或 missing ratio 异常：检查原始 OHLC 是否缺 bar、重复时间戳、时间频率不稳定。
- z-score outlier 很多：先判断是否真实 BTC 跳跃行情，再检查 close 是否有坏点；不要默认删除极端收益。
- 常数列：如果是 `price_feature_age_min` 或某些 flag，可能表示数据质量稳定；如果是核心强度、波动或收益特征，应检查上游输入和窗口 mask。
- bounded feature 越界：优先检查计算模块是否遗漏 clip 或 denominator 是否接近 0。
