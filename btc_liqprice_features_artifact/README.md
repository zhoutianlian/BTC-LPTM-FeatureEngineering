# btc_liqprice_features_artifact

BTC 清算-价格特征工程模块。项目只输出 9 个最终特征，并在完整 pipeline 中自动生成输出特征检验与可视化报告。

## 标准运行

在仓库上层目录执行：

```bash
python -m btc_liqprice_features_artifact.feature_liqprice
```

默认配置：

```text
btc_liqprice_features_artifact/configs/feature_liqprice.json
```

输入、输出、字段名、计算参数、报告目录和检验阈值均在配置文件中维护。

## 输出特征

- `fll_spike_kama`
- `fsl_spike_kama`
- `fll_velocity_gaussian`
- `fll_acceleration_gaussian`
- `fsl_velocity_gaussian`
- `fsl_acceleration_gaussian`
- `trend_pressure`
- `kalman_slope`
- `vol_adaptive`

## 报告

默认报告目录：

```text
btc_liqprice_features_artifact/reports/feature_diagnostics/
```

主要文件：

- `index.html`：总览页，包含检验状态、汇总表、相关性热力图和详情页链接。
- `features/<feature_name>.html`：单特征详情页，包含完整历史时间序列、异常标记、分布、箱线图、rolling 统计、缺失值时间分布和价格/收益关系图。
- `summary.json`：机器可读检验结果。

单独重建报告：

```bash
python -m btc_liqprice_features_artifact.feature_diagnostics \
  --config btc_liqprice_features_artifact/configs/feature_liqprice.json
```

更多参数说明见 `docs/manual.md`。

