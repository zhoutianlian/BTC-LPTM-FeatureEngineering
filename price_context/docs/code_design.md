# price_context 代码设计文档

## 1. 设计目标

`price_context` 的代码目标是把 OHLC price context 设计工程化为可执行、可维护、可扩展的 Python 项目。核心要求：

1. 所有关键参数集中在 `configs/feature_price_context.json`；
2. 输入数据先校验，再计算；
3. 特征计算模块化；
4. rolling/window 逻辑避免未来函数；
5. 缺失、时间缺口、异常收益不静默修复，而是输出质量字段；
6. 输出字段顺序尽量贴近需求文档；
7. 提供测试覆盖 schema、形状、gap、无未来泄漏。

## 2. 项目目录结构

```text
price_context/
  __init__.py
  feature_price_context.py
  config.yaml
  configs/
    feature_price_context.json
  requirements.txt
  README.md
  docs/
    manual.md
    price_context_feature_engineering.md
    code_design.md

  src/
    __init__.py
    main.py
    config.py
    data_loader.py
    validation.py
    preprocessing.py
    returns.py
    realized_vol.py
    range_features.py
    trend_features.py
    vol_of_vol.py
    jump_features.py
    quality_features.py
    feature_pipeline.py
    utils.py

  output/
    price_context_features.csv

  reports/
    feature_diagnostics/
      index.html
      summary.json
      assets/
      features/

  tests/
    conftest.py
    test_no_future_leakage.py
    test_schema_validation.py
    test_feature_shapes.py
```

## 3. 主执行流程

入口：`feature_price_context.py`

```text
feature_price_context.py
→ load_config(configs/feature_price_context.json)
→ run_pipeline(cfg)
   → load_ohlc_csv(cfg)
   → validate_and_prepare_raw(raw, cfg)
   → add_base_price_columns(prepared, cfg)
   → compute_quality_features(...)
   → compute_past_returns(...)
   → compute_realized_vol(...)
   → compute_range_features(...)
   → compute_trend_features(...)
   → compute_vol_of_vol(...)
   → compute_jump_features(...)
   → concat + reorder columns
   → assert required output columns
   → write_features_csv(...)
   → generate_feature_diagnostics_report(...)
```

运行命令：

```bash
python -m price_context.feature_price_context
```

## 4. 配置文件如何驱动代码

`configs/feature_price_context.json` 决定：

- 输入路径与字段名映射；
- bar 频率；
- 是否排序、是否去重、OHLC 校验策略；
- return/core/vol-of-vol/quality 窗口；
- min_obs_ratio；
- outlier/jump robust sigma lookback 与阈值；
- realized vol z-score 方法与 clip；
- range compression percentile lookback；
- trend strength/consistency 权重；
- jump proxy 权重；
- 输出目录、文件名、是否包含扩展字段、是否写 zip。
- 特征诊断报告是否启用、报告目录、HTML/JSON 输出、rolling 诊断窗口、异常阈值和相关性阈值。
- 运行日志级别、是否打印校验报告、是否打印字段列表。

核心逻辑不硬编码窗口和权重。新增窗口时，优先修改配置而不是改函数。

## 5. 模块职责

### 5.1 `config.py`

职责：

- 读取 YAML；
- 校验必需 section 和关键参数；
- 解析相对路径到项目根目录。

核心函数：

- `load_config(config_path)`
- `validate_config(cfg)`
- `resolve_path(path_value, project_root)`

### 5.2 `data_loader.py`

职责：

- 根据配置读取 `ohlc.csv`；
- 不在读取阶段静默修正 schema。

核心函数：

- `load_ohlc_csv(cfg)`

### 5.3 `validation.py`

职责：

- 检查必需字段；
- 解析 `time`；
- 标准化字段名为 `time/open/high/low/close`；
- 数值化 OHLC；
- 按时间排序；
- 处理重复时间戳；
- 校验正价格和 OHLC 基本关系；
- 统计时间缺口；
- 输出 `ValidationReport`。

核心函数：

- `validate_required_columns(df, cfg)`
- `parse_and_validate_time(df, cfg)`
- `standardize_columns(df, cfg)`
- `sort_and_deduplicate(df, cfg)`
- `check_price_values(df, cfg)`
- `validate_and_prepare_raw(df, cfg)`
- `assert_required_output_columns(df, required_columns)`

异常策略：

- 缺少字段：直接 raise；
- time 无法解析：直接 raise；
- 非正价格：默认 raise；
- OHLC 不一致：默认 raise；
- 重复时间戳：默认 warning 后保留最后一条；
- 时间缺口：warning，并在质量特征中标记。

### 5.4 `preprocessing.py`

职责：

- 添加 `log_close`；
- 添加基础相邻 bar 对数收益 `ret_bps`；
- 计算单根 outlier z 与 `single_bar_outlier_flag`；
- 可选 winsorize，但默认关闭。

核心函数：

- `add_base_price_columns(df, cfg)`

未来函数控制：

- `ret_bps` 只用当前和前一根 close；
- outlier scale 使用 past-only rolling robust scale，默认 IQR，支持可选 MAD。

### 5.5 `quality_features.py`

职责：

计算每个窗口的数据质量字段：

- `price_obs_count_W`
- `price_expected_count_W`
- `price_missing_ratio_W`
- `price_gap_flag_W`
- `price_outlier_flag_W`
- 内部字段：`price_valid_window_W`

核心函数：

- `compute_quality_features(df, cfg, windows)`
- `get_valid_window_mask(quality, window, cfg)`

质量字段不仅用于输出，也用于屏蔽核心 price context 特征。如果窗口有效观测不足，核心特征置为 missing。

### 5.6 `returns.py`

职责：

计算过去窗口净收益：

```text
past_return_W_bps = 10000 * log(close_t / close_{t-W})
```

核心函数：

- `compute_past_returns(df, quality, cfg, windows)`

实现细节：

- 使用 `merge_asof` 查找 `t-W` 或一个 bar 容忍范围内的历史 close；
- 查找方向为 backward，绝不使用未来 anchor；
- 结合质量 mask 输出 missing。

### 5.7 `realized_vol.py`

职责：

计算窗口 realized volatility：

- `realized_vol_W_bps`
- `realized_vol_W_per_sqrt_hour_bps`
- `realized_vol_W_z`

核心函数：

- `compute_realized_vol(df, quality, cfg, windows)`

z-score 策略：

- 如果配置了 train split，则使用 train median/IQR；
- 如果没有 train split，则使用 past-only rolling median/IQR；
- clip 到配置范围。

### 5.8 `range_features.py`

职责：

计算：

- `range_width_W_bps`
- `range_compression_W`
- `range_to_vol_W`

核心函数：

- `compute_range_features(df, quality, realized_vol, cfg, windows)`

实现细节：

- range width 使用窗口内 max high / min low；
- compression 使用 rolling rank 转换为 past-only percentile；
- range_to_vol 使用同窗口 realized vol。

### 5.9 `trend_features.py`

职责：

计算趋势强度与趋势一致性相关字段：

- `trend_efficiency_W`
- `trend_snr_W`
- `trend_slope_W`
- `trend_slope_tstat_W`
- `trend_r2_W`
- `trend_strength_W`
- `trend_direction_W`
- `bar_direction_align_W`
- `block_direction_align_W`
- `trend_consistency_W`

核心函数：

- `compute_trend_features(df, quality, past_returns, realized_vol, cfg, windows)`

实现细节：

- OLS 使用 rolling sum 公式向量化实现，不逐窗口 Python 回归；
- `trend_slope` 为 log-price 对小时的斜率；
- `trend_slope_tstat` 与 `trend_r2` 来自同一 rolling regression；
- block alignment 默认 1h 用 20m block，6h/24h 用 1h block；
- block 计算只使用当前及过去 block return。

### 5.10 `vol_of_vol.py`

职责：

先使用 `realized_vol_1h_bps`，再计算：

- `vol_of_vol_W`
- `vol_of_vol_abs_W`
- `vol_of_vol_W_z`

核心函数：

- `compute_vol_of_vol(realized_vol, quality, cfg, windows)`

实现细节：

- `vol_of_vol_W = std(RV_1h) / mean(RV_1h)`；
- `vol_of_vol_abs_W = std(RV_1h)`；
- 结合质量 mask 和最小观测比例。

### 5.11 `jump_features.py`

职责：

计算 jump proxy 相关字段：

- `max_jump_z_W`
- `jump_count_W`
- `jump_ratio_bv_W`
- `signed_max_jump_return_W_bps`
- `jump_proxy_W`

核心函数：

- `compute_jump_features(df, quality, cfg, windows)`

实现细节：

- jump z 的 sigma 默认复用 preprocessing 中的 past-only robust scale；
- IQR robust scale 为生产默认，MAD 为可选；
- jump_count 使用配置阈值；
- bipower variation ratio 使用当前及历史 returns；
- signed max jump 使用 rolling apply 找窗口内绝对收益最大的 signed return。

### 5.12 `feature_pipeline.py`

职责：

- 组织全流程；
- 定义最低必需字段清单；
- 拼接各模块结果；
- 字段排序；
- 校验输出字段；
- 写出 CSV。

核心函数：

- `build_features_from_dataframe(df_raw, cfg)`
- `run_pipeline(cfg)`
- `write_features_csv(features, output_path, cfg)`
- `write_features_csv_fast(features, output_path, float_precision, chunk_size)`

默认使用 pandas `to_csv(index=False)`，以保持历史输出的浮点文本表示。`write_features_csv_fast` 是可选的宽表输出优化 writer，仅在配置 `output.csv_writer=fast` 时启用；它按 chunk 将 datetime 和 numeric 字段转换为字符串，避免 pandas `to_csv(float_format=...)` 在宽表上逐 cell 变慢。

### 5.13 `utils.py`

职责：

提供通用工具：

- 窗口解析；
- 窗口转 bars；
- squash；
- past rolling percentile；
- robust z-score；
- past rolling robust sigma；
- signed max abs；
- index 工具。

### 5.14 `diagnostics/`

职责：

- 从 `docs/price_context_feature_engineering.md` 自动抽取重要输出特征；
- 对所有重要特征和实际输出扩展字段做完整性、分布、异常值、时间序列稳定性、相关性和未来函数风险诊断；
- 生成 `reports/feature_diagnostics/index.html`、`summary.json` 和每个特征的详情页；
- 提供独立 CLI：`python -m price_context.src.diagnostics --config price_context/configs/feature_price_context.json`。

模块划分：

- `metadata.py`：解析文档输出字段，构建特征 catalog；
- `feature_stats.py`：统计指标、异常值、时间轴和 rolling 窗口工具；
- `feature_checks.py`：PASS / WARN / FAIL 规则；
- `feature_plots.py`：总览页相关性 heatmap 和 Plotly 资源；
- `html_templates.py`：深色金融科技主题 HTML/CSS/JS；
- `report_builder.py`：报告编排、summary JSON、单特征页输出、价格/收益关系诊断；
- `__main__.py`：单独生成报告的命令入口。

单特征页采用客户端 Plotly 渲染。HTML 中只写入该特征的一份全历史数组，由 `assets/js/feature_charts.js` 在浏览器中计算 histogram、box、rolling mean/std/q05/q95、缺失分布和异常点，以避免全量历史数据在多个图里重复嵌入。

## 6. 输入输出关系

输入：

```text
ohlc.csv
  time, open, high, low, close
```

中间字段：

```text
log_close
ret_bps
ret_robust_sigma_bps
ret_outlier_z
single_bar_outlier_flag
price_valid_window_W
```

输出：

```text
output/price_context_features.csv
reports/feature_diagnostics/index.html
reports/feature_diagnostics/summary.json
reports/feature_diagnostics/features/<feature_name>.html
```

输出包含 50 个最低必需字段和推荐扩展字段，当前默认共 110 个字段。

## 7. 数据校验与异常处理策略

### 7.1 不静默修复严重问题

以下问题默认抛出错误：

- 缺少必需列；
- time 无法解析；
- 非正价格；
- OHLC 不一致。

### 7.2 可解释保留的问题

以下问题保留并输出质量信息：

- 初期窗口历史不足；
- 时间缺口；
- 缺失 OHLC；
- 极端收益。

### 7.3 特征置空规则

对于核心窗口，如果：

```text
price_obs_count_W < min_obs_ratio * price_expected_count_W
```

则该窗口对应核心 price context 特征置为 missing。

## 8. 防未来函数设计

### 8.1 rolling 特征

所有 rolling 特征均为右闭窗口，只使用当前及历史数据。

### 8.2 past return anchor

`close_{t-W}` 使用 backward as-of 查找，不允许使用 `t-W` 之后的价格作为历史 anchor。

### 8.3 z-score / percentile

- train split 明确配置时，使用 train 统计量；
- 未配置 train split 时，回退到 past-only rolling robust scaler；
- compression percentile 使用 rolling rank，不使用未来样本。

### 8.4 测试覆盖

`tests/test_no_future_leakage.py` 会修改未来价格，确认 cutoff 之前的特征不变。

诊断报告还会在 CSV 层检查 `price_feature_time <= time`、`price_feature_age_min >= 0`，并标记疑似 future / target / label 字段。rolling 和 shift 方向无法仅从 CSV 证明，因此报告中标记为需要结合源码与测试人工确认。

## 9. 测试设计

### 9.1 `test_schema_validation.py`

覆盖：

- 缺少字段 raise；
- OHLC 不一致 raise；
- 重复时间戳可按配置去重。

### 9.2 `test_feature_shapes.py`

覆盖：

- 输出行数与输入一致；
- 必需字段存在；
- 初期长窗口 return 为 NaN；
- 之后长窗口 return 可用；
- 删除一个 bar 后 gap flag 触发。

### 9.3 `test_no_future_leakage.py`

覆盖：

- 修改未来 OHLC，不改变过去已计算特征。

运行：

```bash
pytest -q
```

## 10. 如何扩展新特征

推荐步骤：

1. 在 `configs/feature_price_context.json` 中新增参数；
2. 在 `src/` 下新增模块，例如 `liquidity_features.py`；
3. 只接收标准化后的 DataFrame 和配置；
4. 输出 DataFrame，index 与输入一致；
5. 在 `feature_pipeline.py` 中接入；
6. 在 `_ordered_columns` 中补充字段顺序；
7. 在测试中新增 schema、shape、no future leakage case；
8. 在文档中写清金融机制、公式、失效条件。

## 11. 如何接入未来 path_context / PLIE 上游特征

`path_context` 与 `PLIE` 不应在 price_context 内伪造。建议新增独立模块：

```text
src/plie_context.py
src/path_context.py
```

接入原则：

1. 上游字段必须带有 `time` 和真实可得时间；
2. 使用 `merge_asof(direction="backward")` 对齐到 OHLC bar；
3. 不使用未来 source-clock observation；
4. 不覆盖 price_context 字段；
5. 与 price_context 通过 interaction 或状态模型融合。

示例 interaction：

```text
hpe_trigger = jump_proxy_6h * path_cascade_score_6h
rha_trigger = 1[trend_direction_6h != plie_direction_6h] * path_absorption_score_6h
rc_trigger = range_compression_6h * (1 - trend_strength_6h) * path_neutral_pressure_flag_6h
```

## 12. 当前实现的最小合理假设

1. `time` 表示已完成 bar 的时间戳；rolling 窗口使用当前 bar 及历史完成 bar。
2. 默认 `bar_minutes=10`，block consistency 用该频率推导 block bars。
3. 大样本默认 jump/outlier robust sigma 使用 IQR robust scale；如需严格 MAD，可将 `robust_sigma_estimator` 改为 `mad`，但运行会更慢。
4. 无明确 train split 时，z-score 使用 past-only rolling robust scaler。
5. 1h bipower jump ratio 样本较少，作为扩展字段输出，但解释上更信任 6h/24h。

## 13. 运行结果摘要

对上传的 `ohlc.csv` 当前运行结果：

- 输入行数：275,526；
- 输出行数：275,526；
- 输出字段数：110；
- 数据校验：无缺失、无重复、无 OHLC 不一致、无 10m 时间缺口；
- 测试结果：`7 passed`。
