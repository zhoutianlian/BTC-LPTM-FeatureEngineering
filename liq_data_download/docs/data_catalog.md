# Raw Data Catalog

## 1. 项目目标

本项目当前下载三类 BTC 原始数据：

1. 小时级 BTC 价格 + 多空期货清算量组合数据集
2. 10 分钟级 BTC 价格数据集
3. 10 分钟级 BTC OHLC 数据集

这些数据的目的不是直接做交易信号，而是为后续特征工程、状态识别、事件对齐和回测提供稳定的数据底座。

---

## 2. 数据集一：Hourly liquidation bundle

**输出文件**

`data/raw/hourly/BTC_price_lld.csv`

**配置名**

`hourly_liquidation_bundle`

### 2.1 字段

- `time`
- `price`
- `futures_long_liquidations`
- `futures_short_liquidations`

### 2.2 Glassnode endpoints

- `https://api.glassnode.com/v1/metrics/market/price_usd_close`
- `https://api.glassnode.com/v1/metrics/derivatives/futures_liquidated_volume_long_sum`
- `https://api.glassnode.com/v1/metrics/derivatives/futures_liquidated_volume_short_sum`

### 2.3 频率与标的

- Asset: `BTC`
- Interval: `1h`

### 2.4 金融学含义

#### price
BTC 的美元收盘价是后续特征工程的价格锚点。它不是订单流细节，而是市场最终成交与承接结果的压缩表达。对 liquidation 研究来说，价格是“市场裁决层”。

#### futures_long_liquidations
表示多头期货仓位被强制平仓的总量。其金融学含义是：

- 被迫卖出压力
- 去杠杆链条中的 downside stress
- 多头脆弱性真实落地后的成交结果

#### futures_short_liquidations
表示空头期货仓位被强制平仓的总量。其金融学含义是：

- 被迫买入压力
- short squeeze 的机械成交来源
- 空头脆弱性真实落地后的成交结果

### 2.5 为什么要把这三列放在同一个小时级 CSV

因为后续 liquidation 特征工程需要同时观察：

- 同一时刻的价格
- long-side forced selling
- short-side forced buying

只有放在同一时间栅格上，才能稳定构造：

- total pressure
- net pressure / sdom
- RPN
- dominance

---

## 3. 数据集二：10-minute BTC price

**输出文件**

`data/raw/intraday/BTC_price_10m.csv`

**配置名**

`btc_price_10m`

### 3.1 字段

- `time`
- `price`

### 3.2 Glassnode endpoint

- `https://api.glassnode.com/v1/metrics/market/price_usd_close`

### 3.3 频率与标的

- Asset: `BTC`
- Interval: `10m`
- 起始时间：`2021-02-01T00:00:00Z`

### 3.4 金融学含义

10 分钟价格序列不是为了替代小时级主建模，而是为了补充：

- 事件对齐
- liquidation 之后的短时价格反应观察
- intraday / execution / micro-timing 检查
- 后续 absorption 研究所需的更细粒度价格路径

也就是说，10m 数据更偏向“事件窗口”和“短时反应诊断”，而 1h liquidation bundle 更偏向“主状态建模”。

## 4. 数据集三：10-minute BTC OHLC

**输出文件**

`data/raw/intraday/ohlc.csv`

**配置名**

`btc_ohlc_10m`

### 4.1 字段

- `time`
- `open`
- `high`
- `low`
- `close`

### 4.2 Glassnode endpoint

- `https://api.glassnode.com/v1/metrics/market/price_usd_ohlc`

### 4.3 频率与标的

- Asset: `BTC`
- Interval: `10m`
- 起始时间：`2021-02-01T00:00:00Z`

### 4.4 字段映射

Glassnode OHLC endpoint 的 `v` 字段是对象。下载配置中使用 `value_fields` 将对象字段映射为 CSV 字段：

- `v.o` -> `open`
- `v.h` -> `high`
- `v.l` -> `low`
- `v.c` -> `close`

### 4.5 金融学含义

10 分钟 OHLC 比单一 close 价格保留了更多 bar 内路径信息，可用于：

- 事件窗口内的高低点冲击观察
- liquidation 后的短时波动与回撤测量
- 后续构造 intraday range、candle body、wick 等价格形态特征

---

## 5. 配置说明

所有下载对象都通过 `config.yaml` 中的 `datasets:` 配置定义。

每个 dataset 至少包含：

- `asset`
- `interval`
- `output_csv`
- `description`
- `metrics`

每个 metric 至少包含：

- `endpoint`
- `value_name`
- `description`

这使得你后续要增加数据时，只需要修改 YAML，而不是改下载代码。

对于单值 endpoint，`value_name` 会成为 CSV 字段名。对于 `v` 是对象的 endpoint，可以使用 `value_fields` 将源字段映射为多个输出字段。

---

## 6. 数据质量检查

下载完成后，项目会输出：

- 行数
- 起始时间
- 最新时间
- 缺失 bar 数量
- 缺失比例
- stale_hours

相关文件：

- `data/manifests/download_catalog.csv`
- `data/manifests/<dataset_name>_metric_health.csv`
- `logs/latest.log`

这些字段的作用是判断：

- 数据是否下载到了预期最新时间
- 时间轴是否连续
- 是否存在大面积缺失
- 数据是否已经陈旧
