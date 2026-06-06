# btc_liqprice_features_artifact Feature Engineering

本文件是 `liqprice_feature_engineering.md` 的工程索引版，用于明确当前项目最终输出特征、类别、含义和报告检查范围。生产配置默认仍读取 `liqprice_feature_engineering.md`，两份文档中的重要特征清单应保持一致。

## 4.2 Side-specific liquidation bursts (cascade texture)

### 1) fll_spike_kama

#### Financial meaning

- `fll_spike_kama` 刻画多头清算强度相对过去自适应 KAMA 基线是否异常。
- 正值表示当前多头清算高于过去基线，极高正值可能对应下行清算级联或恐慌踩踏。

#### Calculation

以非负 `fll_cwt_kf` 为输入，计算严格因果 KAMA，并使用滞后一期 KAMA 作为基线：

```text
fll_spike_kama = log((fll_cwt_kf + eps) / (KAMA.shift(1) + eps))
```

### 2) fsl_spike_kama

#### Financial meaning

- `fsl_spike_kama` 刻画空头清算强度相对过去自适应 KAMA 基线是否异常。
- 正值表示当前空头清算高于过去基线，极高正值可能对应 short squeeze 或上行被动回补。

#### Calculation

以非负 `fsl_cwt_kf` 为输入，计算严格因果 KAMA，并使用滞后一期 KAMA 作为基线：

```text
fsl_spike_kama = log((fsl_cwt_kf + eps) / (KAMA.shift(1) + eps))
```

## 4.3 Side-specific liquidation dynamics (Gaussian RoC / RoC²)

### 3) fll_velocity_gaussian

#### Financial meaning

- `fll_velocity_gaussian` 衡量多头清算压力变化速度。
- 正值表示多头清算压力继续增强，负值表示压力减弱。

#### Calculation

对 `log1p(fll_cwt_kf)` 做单边 Gaussian 平滑，再做 backward difference，并换算为每小时单位。

### 4) fll_acceleration_gaussian

#### Financial meaning

- `fll_acceleration_gaussian` 衡量多头清算速度本身的变化。
- 正值表示多头清算速度继续上升，负值表示增长速度放缓。

#### Calculation

对 `fll_velocity_gaussian` 做 backward difference，并换算为每小时平方单位。

### 5) fsl_velocity_gaussian

#### Financial meaning

- `fsl_velocity_gaussian` 衡量空头清算压力变化速度。
- 正值表示空头清算压力继续增强，负值表示压力减弱。

#### Calculation

对 `log1p(fsl_cwt_kf)` 做单边 Gaussian 平滑，再做 backward difference，并换算为每小时单位。

### 6) fsl_acceleration_gaussian

#### Financial meaning

- `fsl_acceleration_gaussian` 衡量空头清算速度本身的变化。
- 正值表示 squeeze 动量继续增强，负值表示增长速度放缓。

#### Calculation

对 `fsl_velocity_gaussian` 做 backward difference，并换算为每小时平方单位。

## 4.7 Price-action trend + volatility context

### 7) trend_pressure

#### Financial meaning

- `trend_pressure` 衡量近期方向性位移相对于背景波动的强弱。
- 正值表示近期上行趋势占优，负值表示近期下行趋势占优，绝对值越大说明趋势越显著。

#### Calculation

```text
trend_pressure = (logp_t - logp_{t-k_mom}) / (sqrt(k_mom) * rolling_std(return, k_vol) + eps)
```

### 8) kalman_slope

#### Financial meaning

- `kalman_slope` 是 log-price 空间下局部线性趋势斜率的 forward filter 估计。
- 正值表示潜在价格漂移向上，负值表示潜在价格漂移向下。

#### Calculation

使用一侧 local linear trend Kalman filter，输出 slope 状态，并按 bar 长度换算为每小时尺度。

### 9) vol_adaptive

#### Financial meaning

- `vol_adaptive` 是严格因果的自适应 realized volatility。
- 高值表示市场风险扰动或波动扩张，低值表示相对平静环境。

#### Calculation

计算短窗与长窗 trailing realized volatility，用短长波动比经过 sigmoid 得到短窗权重：

```text
vol_adaptive = weight * short_vol + (1 - weight) * long_vol
```

## 5. Diagnostics Coverage

以上 9 个特征全部视为重要输出特征。诊断报告必须覆盖这些特征，即使某个特征在实际输出中缺失，也要在 `index.html` 与 `summary.json` 中标记为 `FAIL`。

默认报告目录：

```text
btc_liqprice_features_artifact/reports/feature_diagnostics/
```

报告生成入口：

```bash
python -m btc_liqprice_features_artifact.feature_liqprice
```

单独重建报告：

```bash
python -m btc_liqprice_features_artifact.feature_diagnostics \
  --config btc_liqprice_features_artifact/configs/feature_liqprice.json
```

