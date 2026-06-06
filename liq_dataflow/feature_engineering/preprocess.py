from __future__ import annotations

import numpy as np
import pandas as pd

from liq_dataflow.feature_engineering.config import ColumnConfig


EPS = 1e-12


RAW_REQUIRED_COLUMNS = {"time", "price", "futures_long_liquidations", "futures_short_liquidations"}
CLEAN_REQUIRED_COLUMNS = {"time", "price", "fll_normal", "fsl_normal"}


def normalize_input_columns(df: pd.DataFrame, *, columns: ColumnConfig | None = None) -> pd.DataFrame:
    cfg = columns or ColumnConfig()
    rename_map = {
        cfg.time_col: "time",
        cfg.price_col: "price",
        cfg.raw_long_liquidations_col: "futures_long_liquidations",
        cfg.raw_short_liquidations_col: "futures_short_liquidations",
        cfg.clean_long_col: "fll_normal",
        cfg.clean_short_col: "fsl_normal",
    }
    out = df.copy()
    for src, dst in rename_map.items():
        if src != dst and src in out.columns and dst not in out.columns:
            out = out.rename(columns={src: dst})
    return out


def safe_rpn(long_side: pd.Series, short_side: pd.Series) -> pd.Series:
    total = pd.to_numeric(long_side, errors="coerce").astype(float) + pd.to_numeric(short_side, errors="coerce").astype(float)
    long_side = pd.to_numeric(long_side, errors="coerce").astype(float)
    return pd.Series(np.where(total > EPS, long_side / (total + EPS), 0.5), index=long_side.index, dtype=float)


def safe_sdom(diff: pd.Series, total: pd.Series) -> pd.Series:
    diff = pd.to_numeric(diff, errors="coerce").astype(float)
    total = pd.to_numeric(total, errors="coerce").astype(float)
    return pd.Series(np.where(total > EPS, diff / (total + EPS), 0.0), index=total.index, dtype=float)


def smooth_outlier(df: pd.DataFrame, col: str, col_new: str, *, window_days: int = 180, time_col: str = "time") -> pd.DataFrame:
    if window_days <= 0:
        raise ValueError(f"window_days must be positive, got {window_days}")
    if time_col not in df.columns:
        raise ValueError(f"'{time_col}' column is required for trailing-window outlier smoothing")

    out = df.copy()
    ordered = out.sort_values(time_col).copy()
    times = pd.to_datetime(ordered[time_col], errors="coerce")
    values = pd.to_numeric(ordered[col], errors="coerce")
    series = pd.Series(values.to_numpy(dtype=float), index=times)

    window = f"{int(window_days)}D"
    rolling = series.rolling(window=window, min_periods=1)
    q1 = rolling.quantile(0.25)
    q3 = rolling.quantile(0.75)
    iqr = q3 - q1
    vmax = rolling.max()

    upper_bound = q3 + 3 * iqr
    cap = q3 + 4 * iqr
    compressed = series.copy()
    mask = series > upper_bound
    denom = (vmax - q3).where(mask & ((vmax - q3) > 0))
    compressed.loc[mask] = q3.loc[mask] + 4 * iqr.loc[mask] * (series.loc[mask] - q3.loc[mask]) / denom.loc[mask]
    compressed.loc[mask] = compressed.loc[mask].fillna(cap.loc[mask])

    ordered[col_new] = compressed.to_numpy()
    ordered[col_new] = pd.Series(ordered[col_new].to_numpy()).interpolate(method="linear").to_numpy()
    out[col_new] = ordered[col_new].reindex(out.index)
    return out


def finalize_clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    out = out.dropna(subset=["time"]).sort_values("time").drop_duplicates(subset=["time"], keep="last")
    out["fll_normal"] = pd.to_numeric(out["fll_normal"], errors="coerce").astype(float).clip(lower=0.0)
    out["fsl_normal"] = pd.to_numeric(out["fsl_normal"], errors="coerce").astype(float).clip(lower=0.0)
    out["total_ls_normal"] = out["fll_normal"] + out["fsl_normal"]
    out["diff_ls_normal"] = out["fll_normal"] - out["fsl_normal"]
    out["liq_active_raw"] = (out["total_ls_normal"] > 0).astype(int)
    out["lld_normal"] = safe_rpn(out["fll_normal"], out["fsl_normal"])
    out["diff_dom_ls_normal"] = safe_sdom(out["diff_ls_normal"], out["total_ls_normal"])
    keep = [
        "time",
        "price",
        "fll_normal",
        "fsl_normal",
        "total_ls_normal",
        "diff_ls_normal",
        "lld_normal",
        "diff_dom_ls_normal",
        "liq_active_raw",
    ]
    return out[keep].copy()


def preprocess_liquidation_data(
    df: pd.DataFrame,
    *,
    start_time: str,
    outlier_iqr_window_days: int,
    columns: ColumnConfig | None = None,
) -> pd.DataFrame:
    data = normalize_input_columns(df, columns=columns)
    data["time"] = pd.to_datetime(data["time"], errors="coerce")
    data = data.dropna(subset=["time"]).sort_values("time").drop_duplicates(subset=["time"], keep="last")
    data = data[data["time"] >= pd.to_datetime(start_time)].copy()

    columns = set(data.columns)
    if RAW_REQUIRED_COLUMNS.issubset(columns):
        data = smooth_outlier(
            data,
            "futures_long_liquidations",
            "fll_normal",
            window_days=outlier_iqr_window_days,
        )
        data = smooth_outlier(
            data,
            "futures_short_liquidations",
            "fsl_normal",
            window_days=outlier_iqr_window_days,
        )
        return finalize_clean_frame(data)

    if CLEAN_REQUIRED_COLUMNS.issubset(columns):
        return finalize_clean_frame(data)

    raise ValueError(
        "Input dataframe must be either raw hourly liquidation data with columns "
        "{time, price, futures_long_liquidations, futures_short_liquidations} or a preprocessed clean frame "
        "with {time, price, fll_normal, fsl_normal}."
    )
