# Code Design

## 1. 架构原则

当前代码按“两项目解耦”设计：

- `liq_data_download`：只负责下载原始数据并保存 CSV。
- `liq_dataflow`：只负责读取 CSV，做特征工程、输出校验和可视化。

这样新增下载数据时，只改下载项目的配置；特征逻辑不需要跟着改。

---

## 2. liq_dataflow 核心模块

### 2.0 `feature_liq_dataflow.py`

顶层执行入口，负责读取配置并启动 pipeline。正式运行方式：

```bash
python -m liq_dataflow.feature_liq_dataflow
```

### 2.1 `feature_engineering/config.py`

负责：

- 项目配置 dataclass
- `configs/feature_engineering.json` 读取
- 输入 CSV 配置
- 输出路径定义
- 字段映射、最终交付字段、计算参数配置
- `ProjectPaths` 路径管理

### 2.2 `feature_engineering/data_source.py`

负责：

- 从 CSV 读取输入数据
- 按优先级尝试 `source_csv` / `fallback_csv`
- 记录输入文件的行数、时间范围和缺失 bar 情况

### 2.3 `feature_engineering/preprocess.py`

负责：

- raw hourly bundle -> clean frame
- outlier smoothing
- safe `risk_priority_number` / safe sdom
- clean frame 统一格式化

### 2.4 `feature_engineering/transforms.py`

负责：

- trailing wavelet trend
- Kalman smoothing
- cache
- delta / corr_case 等基础变换

### 2.5 `feature_engineering/binning.py`

负责：

- past-only binning
- extreme bin 标记
- bin 统计

### 2.6 `feature_engineering/dominance.py`

负责：

- dominance 状态构造
- threshold / event / reverse 规则

### 2.7 `feature_engineering/model_features.py`

负责：

- `z_logTotalP`
- `z_sdom`
- `z_fll_cwt_kf`
- `z_fsl_cwt_kf`

历史模型层别名 `RPN` 已收敛为最终输出字段 `risk_priority_number`。

### 2.8 `feature_engineering/validation.py`

负责：

- 非负性
- 代数一致性
- 范围检查
- CSV / HTML / PNG 输出检查

### 2.9 `feature_engineering/pipeline.py`

负责 orchestrate：

1. 读 CSV
2. preprocess
3. canonical family
4. binning
5. dominance
6. merged final features
7. feature store
8. visualization
9. validation

---

## 3. visualizer 层

### 3.1 `visualizer/custom_style_matplotlib.py`

生成与原始风格一致的静态 PNG 专题图。

### 3.2 `visualizer/specialized_dashboard.py`

生成两个专题 HTML：

- `data/report/rpn_dominance-latest.html`
- `data/report/rpn_features-latest.html`

专题 HTML 与对应 PNG 共用上下两张主图结构、标题时间戳、坐标轴含义和变量配色；PNG 保留 3M 静态审阅窗口，HTML 默认展示全量历史，并提供时间范围选择、range slider、拖拽缩放和窗口内 Y 轴自适应。

### 3.3 `visualizer/generic_feature_dashboard.py`

针对最终合并输出中的重要交付特征生成统计学页面和总览入口页，输出到 `data/report/feature_overview.html` 与 `data/report/feature_pages/*.html`。

---

## 4. 输出契约

### 4.1 输入契约

支持两类输入 CSV：

- raw hourly bundle
- clean frame

### 4.2 交付契约

核心交付 CSV：

- `data/features/features_liq_dataflow.csv`

调试与可视化宽表：

- `data/features/feature_store.csv`

可视化报告：

- `data/report/rpn_dominance-latest.{html,png}`
- `data/report/rpn_features-latest.{html,png}`
- `data/report/feature_overview.html`
- `data/report/feature_pages/*.html`

---

## 5. 日志设计

每次运行单独落盘到：

- `logs/runs/<run_id>/pipeline.log`
- `logs/runs/<run_id>/pipeline.jsonl`
- `logs/runs/<run_id>/run_summary.*`

全局索引：

- `logs/latest.log`
- `logs/run_history.csv`
