# 代码设计文档

## 1. 设计目标

`btc_liqprice_features_artifact` 只负责 BTC 清算-价格特征工程与输出特征诊断，不负责状态建模或交易决策。

当前设计目标：

- 从仓库上层目录用 `python -m btc_liqprice_features_artifact.feature_liqprice` 执行。
- 所有业务参数集中在 `configs/feature_liqprice.json`。
- 特征计算逻辑保持稳定，只输出 9 个最终特征。
- 完整 pipeline 自动生成独立诊断报告目录。
- 报告模块独立于特征计算模块，便于单独复用。

## 2. 目录结构

```text
btc_liqprice_features_artifact/
├── __init__.py
├── configs/
│   └── feature_liqprice.json
├── feature_liqprice.py
├── feature_diagnostics.py
├── cli_build_features.py
├── cli_visualize.py
├── config.py
├── execution_logger.py
├── features.py
├── visualization.py
├── diagnostics/
│   ├── __init__.py
│   ├── feature_checks.py
│   ├── feature_stats.py
│   ├── feature_plots.py
│   ├── report_builder.py
│   └── html_templates.py
├── docs/
├── output/
├── reports/
└── logs/
```

## 3. 主入口

### `feature_liqprice.py`

配置驱动主入口，职责：

1. 读取 `PipelineConfig`。
2. 按仓库上层目录解析输入、输出、报告、日志路径。
3. 加载原始输入并标准化字段。
4. 计算 9 个最终特征。
5. 保存特征文件、解析配置、运行清单。
6. 若 `report.enabled=true`，自动调用 diagnostics 生成 HTML 与 `summary.json`。

### `feature_diagnostics.py`

单独报告入口，职责：

1. 读取同一份配置。
2. 读取 `paths.output` 中已有特征文件。
3. 可选读取 `paths.input` 中的价格上下文。
4. 生成同样结构的诊断报告。

### 旧 CLI

`cli_build_features.py` 和 `cli_visualize.py` 保留兼容，内部已改用新的 diagnostics 报告生成器。新流程优先使用 `feature_liqprice.py` 与 `feature_diagnostics.py`。

## 4. 配置设计

`config.py` 包含：

- `FeatureConfig`：特征计算参数。
- `PathsConfig`：输入、输出、日志、运行清单路径。
- `ColumnsConfig`：输入字段、输出字段和别名。
- `ReportConfig`：报告目录、HTML/JSON 开关、检验阈值、rolling 与相关性参数。
- `PipelineConfig`：上述配置的组合。

`load_config()` 保持旧版 `FeatureConfig` 兼容；`load_pipeline_config()` 用于新入口。

## 5. 特征计算层

`features.py` 仍是唯一特征计算核心。关键函数：

- `canonicalize_input_columns()`：按 `ColumnsConfig` 和 aliases 标准化字段。
- `validate_and_sort()`：时间解析、排序、去重。
- `prepare_decision_frame()`：推断 bar 周期并按决策周期重采样。
- `compute_features()`：计算 9 个最终特征。
- `save_feature_frame()`：保存 CSV/Parquet。

本次新增报告能力没有改变既有 9 个特征的计算公式。

## 6. 诊断报告层

### `feature_stats.py`

负责数值统计：

- count、mean、std、min、max、median。
- p01、p05、p25、p75、p95、p99。
- skew、kurtosis。
- zero / positive / negative ratio。
- unique count 和 constant flag。
- NaN、inf、valid count 等完整性指标。

### `feature_checks.py`

负责有效性检查：

- 从特征工程文档解析重要特征名称、类别、说明和计算段落。
- 检查缺失特征、非数值列、全空列、缺失比例、inf。
- 检查重复时间戳、时间单调性和异常时间跳跃。
- 检查 z-score、IQR、分位数尾部、爆炸值和断崖跳变。
- 检查长时间缺失、全 0、近似常数区间。
- 给出 PASS / WARN / FAIL。
- 对未来函数风险输出自动可见的人工确认项。

### `feature_plots.py`

负责 Plotly 图表：

- 全历史时间序列图，带 hover、缩放、range slider、1D/1W/1M/3M/6M/1Y/ALL。
- 异常点标记。
- histogram、KDE 近似密度、均值/中位数/分位数标记、箱线图。
- rolling mean/std/min/max/quantile。
- 缺失值时间分布。
- feature vs price / return / future return。
- 相关性热力图。

### `html_templates.py`

负责统一金融科技深色主题、表格搜索排序 JS、状态徽标和基础 HTML 组件。

### `report_builder.py`

负责报告编排：

1. 规范化特征数据时间列。
2. 合并价格上下文并计算当前收益、未来收益诊断列。
3. 对文档中的每个重要特征执行检查。
4. 生成 `summary.json`。
5. 生成 `index.html` 和每个 `features/<feature>.html`。

## 7. 因果性约束

特征计算仍遵循：

- KAMA spike 使用滞后一期基线。
- Gaussian dynamics 使用单边历史核。
- velocity / acceleration 使用 backward difference。
- Kalman 使用 forward filter。
- trend / volatility 使用 trailing window。

报告中的未来收益只用于离线诊断，不写入输出特征文件，也不应被当作可交易时点可用信息。

