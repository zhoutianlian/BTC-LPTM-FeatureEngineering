from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class AlignmentConfig:
    time_col: str = "time"
    price_col: str = "price"
    feature_timestamp_offset: Optional[str] = "50min"
    merge_tolerance: Optional[str] = "70min"
    source_time_col: str = "liq_feature_time"
    source_age_col: str = "liq_feature_age_min"
    keep_raw_source_time: bool = True
    raw_source_time_col: str = "liq_feature_time_raw"


def _as_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True)


def merge_features_asof(
    df_10m: pd.DataFrame,
    df_other: pd.DataFrame,
    *,
    cfg: AlignmentConfig,
) -> pd.DataFrame:
    """Causal as-of merge with explicit feature availability timestamp.

    Typical use-case here is hourly liquidation features merged onto a 10-minute
    price grid. If upstream feature rows are left-labeled (e.g. 02:00 row means
    information from [02:00,03:00)), then the feature only becomes tradable at
    02:50 on a 10m close grid. ``feature_timestamp_offset='50min'`` implements
    that rule directly.
    """
    left = df_10m[[cfg.time_col, cfg.price_col]].copy()
    right = df_other.copy()

    left[cfg.time_col] = _as_utc(left[cfg.time_col])
    right[cfg.time_col] = _as_utc(right[cfg.time_col])

    left = left.sort_values(cfg.time_col).drop_duplicates(subset=cfg.time_col, keep="last")
    right = right.sort_values(cfg.time_col).drop_duplicates(subset=cfg.time_col, keep="last")

    if cfg.price_col in right.columns:
        right = right.drop(columns=[cfg.price_col])

    if cfg.keep_raw_source_time:
        right[cfg.raw_source_time_col] = right[cfg.time_col]

    if cfg.feature_timestamp_offset:
        right[cfg.source_time_col] = right[cfg.time_col] + pd.Timedelta(cfg.feature_timestamp_offset)
    else:
        right[cfg.source_time_col] = right[cfg.time_col]

    tolerance = pd.Timedelta(cfg.merge_tolerance) if cfg.merge_tolerance else None

    merged = pd.merge_asof(
        left,
        right.drop(columns=[cfg.time_col]),
        left_on=cfg.time_col,
        right_on=cfg.source_time_col,
        direction="backward",
        tolerance=tolerance,
    )

    if cfg.source_time_col in merged.columns:
        age_min = (merged[cfg.time_col] - merged[cfg.source_time_col]).dt.total_seconds() / 60.0
        merged[cfg.source_age_col] = age_min
    else:
        merged[cfg.source_age_col] = pd.NA

    return merged


def collapse_to_source_clock(
    df: pd.DataFrame,
    *,
    time_col: str = "time",
    source_time_col: str = "liq_feature_time",
) -> pd.DataFrame:
    """Collapse target-bar dataframe to unique source feature updates.

    Keeps the last target-bar row associated with each source feature timestamp.
    The returned index is the feature availability timestamp (source clock).
    """
    out = df.copy()
    if source_time_col not in out.columns:
        raise ValueError(f"Missing source_time_col='{source_time_col}'")

    out[source_time_col] = pd.to_datetime(out[source_time_col], utc=False)
    if time_col in out.columns:
        out[time_col] = pd.to_datetime(out[time_col], utc=False)
        sort_cols = [source_time_col, time_col]
    else:
        sort_cols = [source_time_col]

    out = out.dropna(subset=[source_time_col]).sort_values(sort_cols)
    out = out.drop_duplicates(subset=[source_time_col], keep="last")
    out = out.set_index(source_time_col, drop=False).sort_index()
    return out
