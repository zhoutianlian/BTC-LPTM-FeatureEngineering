# liq_data_download Manual

## 1. 项目定位

`liq_data_download` 是独立的原始数据下载项目。
它只负责三件事：

1. 从 Glassnode 下载原始时间序列。
2. 将每个数据集保存为独立 CSV。
3. 生成下载日志、运行历史和数据目录清单。

它不做特征工程、不做状态推断、不做可视化特征生成。

---

## 2. 目录结构

```text
liq_data_download/
├── config.py
├── config.yaml
├── glassnode.yaml.example
├── glassnode_client.py
├── logging_utils.py
├── pipeline.py
├── scripts/
│   └── run_data_download.py
├── docs/
│   ├── manual.md
│   └── data_catalog.md
├── data/
│   ├── raw/
│   │   ├── hourly/
│   │   └── intraday/
│   └── manifests/
└── logs/
```

---

## 3. 凭证配置

优先方式：环境变量。

```bash
export GLASSNODE_API_KEY="你的_api_key"
```

或者在项目根目录创建 `glassnode.yaml`：

```yaml
glassnode:
  api_key: "你的_api_key"
  base_url: "https://api.glassnode.com/v1"
  requests_per_minute: 10
```

建议保留 `glassnode.yaml.example` 作为模板，不要把真实密钥提交到版本库。

---

## 4. 运行方式

### 4.1 正常下载

```bash
python -m liq_data_download.scripts.run_data_download
```

### 4.2 只下载指定数据集

```bash
python -m liq_data_download.scripts.run_data_download --only hourly_liquidation_bundle
```

```bash
python -m liq_data_download.scripts.run_data_download --only btc_price_10m
```

```bash
python -m liq_data_download.scripts.run_data_download --only btc_ohlc_10m
```

### 4.3 dry-run

用于检查配置、输出目录和日志，不触发远端请求。

```bash
python -m liq_data_download.scripts.run_data_download --dry-run
```

---

## 5. 输出文件

### 5.1 原始数据 CSV

- `data/raw/hourly/BTC_price_lld.csv`
- `data/raw/intraday/BTC_price_10m.csv`
- `data/raw/intraday/ohlc.csv`

### 5.2 manifest / catalog

- `data/manifests/download_catalog.csv`
- `data/manifests/download_catalog.md`
- `data/manifests/download_catalog.json`
- `data/manifests/<dataset_name>_metric_health.csv`

### 5.3 日志

- `logs/latest.log`
- `logs/run_history.csv`
- `logs/runs/<run_id>/pipeline.log`
- `logs/runs/<run_id>/pipeline.jsonl`
- `logs/runs/<run_id>/run_summary.md`
- `logs/runs/<run_id>/run_summary.json`

---

## 6. 如何添加新的下载数据

本项目支持通过 `config.yaml` 直接扩展数据集，而不改 Python 下载逻辑。

你只需要在 `datasets:` 下新增一个 dataset：

```yaml
datasets:
  my_new_dataset:
    enabled: true
    asset: "BTC"
    interval: "1h"
    start_time: "2024-01-01T00:00:00Z"
    output_csv: "data/raw/hourly/my_new_dataset.csv"
    description: "example"
    join: "inner"
    metrics:
      - endpoint: "metrics/market/price_usd_close"
        value_name: "price"
        description: "BTC price"
```

如果你要组合多个字段，只需要继续往 `metrics:` 里添加 endpoint。

如果 Glassnode endpoint 的 `v` 字段是对象，可以用 `value_fields` 把对象字段展开成 CSV 列。例如 10 分钟 OHLC：

```yaml
datasets:
  btc_ohlc_10m:
    enabled: true
    asset: "BTC"
    interval: "10m"
    start_time: "2021-02-01T00:00:00Z"
    output_csv: "data/raw/intraday/ohlc.csv"
    description: "10-minute BTC USD OHLC candles."
    join: "inner"
    metrics:
      - endpoint: "metrics/market/price_usd_ohlc"
        value_name: "ohlc"
        description: "BTC composite USD OHLC candles at 10-minute resolution."
        value_fields:
          o: "open"
          h: "high"
          l: "low"
          c: "close"
```

该配置会输出字段：

- `time`
- `open`
- `high`
- `low`
- `close`

---

## 7. 与特征工程项目的衔接

标准流程：

1. 先运行 `liq_data_download` 下载并落盘 CSV。
2. 再运行 `liq_dataflow`，让特征工程项目读取这些 CSV。

这样下载逻辑与特征逻辑完全解耦。

---

## 8. 常见问题

### 8.1 提示没有 API key

检查：

- 是否已设置 `GLASSNODE_API_KEY`
- 或者项目根目录是否存在 `glassnode.yaml`

### 8.2 10m 数据没有生成

检查：

- `btc_price_10m` 是否在 `config.yaml` 中启用
- `lookback_days` 是否设置合理
- 日志中是否有 401/429/400 响应

### 8.3 下载成功但时间有缺口

查看：

- `data/manifests/download_catalog.csv`
- `data/manifests/<dataset_name>_metric_health.csv`
- `logs/latest.log`

这里会记录每个 metric 的：

- `latest_time`
- `missing_bars`
- `missing_ratio_pct`
- `stale_hours`
