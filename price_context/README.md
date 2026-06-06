# price_context

`price_context` 是一个基于 OHLC 价格数据的价格上下文特征工程项目。它读取配置文件指定的 OHLC CSV，计算 past return、realized volatility、range compression、trend strength、trend consistency、vol of vol、jump proxy 以及价格数据质量特征，并输出 `price_context/output/price_context_features.csv`。

核心原则：所有特征只使用当前时点及历史数据；缺失、时间缺口和异常收益以质量字段显式标记，不默认静默填充或删除。

## 快速运行

```bash
pip install -r price_context/requirements.txt
python -m price_context.feature_price_context
```

完整 pipeline 默认还会生成特征检验与交互式 HTML 可视化报告：

```text
price_context/reports/feature_diagnostics/index.html
```

如果只想基于已有特征 CSV 重新生成报告：

```bash
python -m price_context.src.diagnostics --config price_context/configs/feature_price_context.json
```

默认主配置文件为 `price_context/configs/feature_price_context.json`。输入路径、输出路径、字段映射、窗口、阈值、权重、诊断参数和运行打印行为都在配置文件中维护。

详细说明见：

- `docs/manual.md`
- `docs/price_context_feature_engineering.md`
- `docs/code_design.md`
