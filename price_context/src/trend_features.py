from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .quality_features import get_valid_window_mask
from .utils import EPSILON, parse_window_to_timedelta, squash, window_to_bars, window_to_label


def _rolling_regression_features(df: pd.DataFrame, quality: pd.DataFrame, cfg: dict[str, Any], window: str) -> pd.DataFrame:
    """Rolling OLS of log(close) on elapsed hours; vectorized rolling-sum implementation."""
    label = window_to_label(window)
    out = pd.DataFrame(index=df.index)
    y = df["log_close"].astype("float64")
    unit = cfg.get("trend", {}).get("regression_time_unit", "hour")
    unit_seconds = {"minute": 60.0, "hour": 3600.0, "day": 86400.0}
    if unit not in unit_seconds:
        raise ValueError("trend.regression_time_unit must be one of: minute, hour, day.")
    x = ((df["time"] - df["time"].iloc[0]).dt.total_seconds() / unit_seconds[unit]).astype("float64")
    x = pd.Series(x.to_numpy(), index=df.index)
    valid = y.notna() & x.notna()
    xv = x.where(valid)
    yv = y.where(valid)

    roll_valid = valid.astype("float64").rolling(window, min_periods=1)
    n = roll_valid.sum()
    sx = xv.rolling(window, min_periods=1).sum()
    sy = yv.rolling(window, min_periods=1).sum()
    sx2 = (xv * xv).rolling(window, min_periods=1).sum()
    sy2 = (yv * yv).rolling(window, min_periods=1).sum()
    sxy = (xv * yv).rolling(window, min_periods=1).sum()

    sxx = sx2 - (sx * sx) / n
    syy = sy2 - (sy * sy) / n
    sxy_c = sxy - (sx * sy) / n
    beta = sxy_c / (sxx + EPSILON)
    r2 = (sxy_c * sxy_c) / ((sxx * syy) + EPSILON)
    r2 = r2.clip(0.0, 1.0)
    sse = (syy - beta * sxy_c).clip(lower=0.0)
    se_beta = np.sqrt((sse / (n - 2.0).clip(lower=1.0)) / (sxx + EPSILON))
    tstat = beta / (se_beta + EPSILON)

    enough = (n >= 3) & get_valid_window_mask(quality, window, cfg)
    out[f"trend_slope_{label}"] = beta.where(enough)
    out[f"trend_slope_tstat_{label}"] = tstat.where(enough).replace([np.inf, -np.inf], np.nan)
    out[f"trend_r2_{label}"] = r2.where(enough)
    return out


def _bar_direction_align(
    df: pd.DataFrame,
    past_returns: pd.DataFrame,
    quality: pd.DataFrame,
    cfg: dict[str, Any],
    window: str,
) -> pd.Series:
    label = window_to_label(window)
    ret = df["ret_bps"]
    valid_ret = ret.notna()
    pos = ((ret > 0) & valid_ret).astype("float64")
    neg = ((ret < 0) & valid_ret).astype("float64")
    cnt = valid_ret.astype("float64").rolling(window, min_periods=1).sum()
    pos_cnt = pos.rolling(window, min_periods=1).sum()
    neg_cnt = neg.rolling(window, min_periods=1).sum()
    total_ret = past_returns[f"past_return_{label}_bps"]
    sign = np.sign(total_ret)
    near_zero = float(cfg["trend"].get("near_zero_return_bps", 1e-8))
    align = pd.Series(np.nan, index=df.index, dtype="float64")
    align.loc[(sign > 0) & (total_ret.abs() > near_zero)] = (pos_cnt / (cnt + EPSILON)).loc[(sign > 0) & (total_ret.abs() > near_zero)]
    align.loc[(sign < 0) & (total_ret.abs() > near_zero)] = (neg_cnt / (cnt + EPSILON)).loc[(sign < 0) & (total_ret.abs() > near_zero)]
    return align.where(get_valid_window_mask(quality, window, cfg)).clip(0.0, 1.0)


def _block_return_fixed_bars(close: pd.Series, block_window: str, bar_minutes: int) -> pd.Series:
    bars = window_to_bars(block_window, bar_minutes)
    return 10000.0 * np.log(close / close.shift(bars))


def _block_direction_align(
    df: pd.DataFrame,
    past_returns: pd.DataFrame,
    quality: pd.DataFrame,
    cfg: dict[str, Any],
    window: str,
) -> pd.Series:
    label = window_to_label(window)
    bar_minutes = int(cfg["data"].get("bar_minutes", 10))
    block_window = cfg["trend"].get("block_windows", {}).get(label, "1h")
    block_bars = window_to_bars(block_window, bar_minutes)
    total_bars = window_to_bars(window, bar_minutes)
    k = max(1, total_bars // block_bars)
    block_ret = _block_return_fixed_bars(df["close"], block_window, bar_minutes)
    total_ret = past_returns[f"past_return_{label}_bps"]
    total_sign = np.sign(total_ret)
    near_zero = float(cfg["trend"].get("near_zero_return_bps", 1e-8))
    aligned_sum = pd.Series(0.0, index=df.index)
    valid_sum = pd.Series(0.0, index=df.index)
    for j in range(k):
        br = block_ret.shift(j * block_bars)
        valid = br.notna() & (br.abs() > near_zero)
        aligned = valid & (np.sign(br) == total_sign) & (total_ret.abs() > near_zero)
        aligned_sum = aligned_sum + aligned.astype("float64")
        valid_sum = valid_sum + valid.astype("float64")
    align = aligned_sum / (valid_sum + EPSILON)
    align = align.where((valid_sum > 0) & (total_ret.abs() > near_zero))
    return align.where(get_valid_window_mask(quality, window, cfg)).clip(0.0, 1.0)


def compute_trend_features(
    df: pd.DataFrame,
    quality: pd.DataFrame,
    past_returns: pd.DataFrame,
    realized_vol: pd.DataFrame,
    cfg: dict[str, Any],
    windows: Iterable[str],
) -> pd.DataFrame:
    """Compute trend efficiency, SNR, slope/t-stat/R², strength, direction, and consistency."""
    out = pd.DataFrame(index=df.index)
    trend_cfg = cfg.get("trend", {})
    sw = trend_cfg.get("strength_weights", {})
    cw = trend_cfg.get("consistency_weights", {})
    c = float(trend_cfg.get("squash_c", 2.0))
    near_zero = float(trend_cfg.get("near_zero_return_bps", 1e-8))

    for w in windows:
        label = window_to_label(w)
        total_ret = past_returns[f"past_return_{label}_bps"]
        path_length = df["ret_bps"].abs().rolling(w, min_periods=1).sum().where(get_valid_window_mask(quality, w, cfg))
        efficiency = (total_ret.abs() / (path_length + EPSILON)).clip(0.0, 1.0)
        rv = realized_vol[f"realized_vol_{label}_bps"]
        snr = total_ret.abs() / (rv + EPSILON)
        reg = _rolling_regression_features(df, quality, cfg, w)
        out = pd.concat([out, reg], axis=1)
        out[f"trend_efficiency_{label}"] = efficiency
        out[f"trend_snr_{label}"] = snr.where(get_valid_window_mask(quality, w, cfg))

        direction = pd.Series(np.sign(total_ret), index=df.index, dtype="float64")
        direction = direction.mask(total_ret.abs() <= near_zero, 0.0).where(total_ret.notna())
        out[f"trend_direction_{label}"] = direction.where(get_valid_window_mask(quality, w, cfg))

        tstat_abs = out[f"trend_slope_tstat_{label}"].abs()
        strength = (
            float(sw.get("trend_efficiency", 0.35)) * efficiency.clip(0.0, 1.0)
            + float(sw.get("trend_snr", 0.35)) * pd.Series(squash(snr.abs(), c=c), index=df.index)
            + float(sw.get("trend_slope_tstat", 0.30)) * pd.Series(squash(tstat_abs, c=c), index=df.index)
        )
        out[f"trend_strength_{label}"] = strength.where(get_valid_window_mask(quality, w, cfg)).clip(0.0, 1.0)

        bar_align = _bar_direction_align(df, past_returns, quality, cfg, w)
        block_align = _block_direction_align(df, past_returns, quality, cfg, w)
        out[f"bar_direction_align_{label}"] = bar_align
        out[f"block_direction_align_{label}"] = block_align
        consistency = (
            float(cw.get("bar_direction_align", 0.40)) * bar_align
            + float(cw.get("block_direction_align", 0.40)) * block_align
            + float(cw.get("trend_r2", 0.20)) * out[f"trend_r2_{label}"]
        )
        out[f"trend_consistency_{label}"] = consistency.where(get_valid_window_mask(quality, w, cfg)).clip(0.0, 1.0)
    return out
