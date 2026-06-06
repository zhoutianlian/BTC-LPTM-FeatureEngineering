# Download Catalog

## hourly_liquidation_bundle
- **dataset**: hourly_liquidation_bundle
- **asset**: BTC
- **interval**: 1h
- **output_csv**: /Users/zhoutianlian/Documents/MyGitHub/WTRCodeHub/BTC-LPTM-FeatureEngineering/liq_data_download/data/raw/hourly/BTC_price_lld.csv
- **description**: Hourly BTC USD close price plus long/short futures liquidation volumes.
- **join**: inner
- **rows**: 46227
- **start_time**: 2021-02-01 00:00:00
- **latest_time**: 2026-05-12 02:00:00
- **missing_bars**: 0
- **missing_ratio_pct**: 0.0
- **stale_hours**: 1.779
- **metrics**: price, futures_long_liquidations, futures_short_liquidations
- **endpoints**: metrics/market/price_usd_close | metrics/derivatives/futures_liquidated_volume_long_sum | metrics/derivatives/futures_liquidated_volume_short_sum
- **status**: OK

## btc_price_10m
- **dataset**: btc_price_10m
- **asset**: BTC
- **interval**: 10m
- **output_csv**: /Users/zhoutianlian/Documents/MyGitHub/WTRCodeHub/BTC-LPTM-FeatureEngineering/liq_data_download/data/raw/intraday/BTC_price_10m.csv
- **description**: 10-minute BTC USD close price.
- **join**: inner
- **rows**: 277366
- **start_time**: 2021-02-01 00:00:00
- **latest_time**: 2026-05-12 03:30:00
- **missing_bars**: 0
- **missing_ratio_pct**: 0.0
- **stale_hours**: 0.283
- **metrics**: price
- **endpoints**: metrics/market/price_usd_close
- **status**: OK

## btc_ohlc_10m
- **dataset**: btc_ohlc_10m
- **asset**: BTC
- **interval**: 10m
- **output_csv**: /Users/zhoutianlian/Documents/MyGitHub/WTRCodeHub/BTC-LPTM-FeatureEngineering/liq_data_download/data/raw/intraday/ohlc.csv
- **description**: 10-minute BTC USD OHLC candles.
- **join**: inner
- **rows**: 277366
- **start_time**: 2021-02-01 00:00:00
- **latest_time**: 2026-05-12 03:30:00
- **missing_bars**: 0
- **missing_ratio_pct**: 0.0
- **stale_hours**: 0.287
- **metrics**: open, high, low, close
- **endpoints**: metrics/market/price_usd_ohlc
- **status**: OK

