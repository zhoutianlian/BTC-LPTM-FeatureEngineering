"""Price-context loading and causal alignment utilities.

The price context file is optional.  When present, it provides past-only price
state features such as realized volatility, trend strength, range compression,
and jump proxies.  These features are used only as *response/context
interpreters* for QD-MAR path labels and state evidence; they do not override
PLIE pressure context.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import zipfile
import warnings

import pandas as pd

from .config import Config


DEFAULT_PRICE_CONTEXT_COLUMNS = [
    "time",
    "price_feature_time",
    "price_feature_age_min",
    "realized_vol_1h_bps",
    "realized_vol_6h_bps",
    "realized_vol_24h_bps",
    "realized_vol_1h_per_sqrt_hour_bps",
    "realized_vol_6h_per_sqrt_hour_bps",
    "realized_vol_24h_per_sqrt_hour_bps",
    "range_compression_1h",
    "range_compression_6h",
    "range_compression_24h",
    "trend_strength_1h",
    "trend_strength_6h",
    "trend_strength_24h",
    "trend_consistency_1h",
    "trend_consistency_6h",
    "trend_consistency_24h",
    "trend_direction_1h",
    "trend_direction_6h",
    "trend_direction_24h",
    "vol_of_vol_6h",
    "vol_of_vol_24h",
    "vol_of_vol_48h",
    "jump_proxy_1h",
    "jump_proxy_6h",
    "jump_proxy_24h",
    "price_missing_ratio_1h",
    "price_missing_ratio_6h",
    "price_missing_ratio_24h",
    "price_gap_flag_1h",
    "price_gap_flag_6h",
    "price_gap_flag_24h",
    "price_outlier_flag_1h",
    "price_outlier_flag_6h",
    "price_outlier_flag_24h",
]


def _read_csv_or_zip(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    """Read a CSV path or a ZIP containing one CSV.

    Pandas' automatic ZIP reader can be slow or fail when the archive contains
    macOS metadata files.  This helper explicitly selects the first non-metadata
    CSV entry.
    """
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            csv_names = [n for n in z.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX/") and not Path(n).name.startswith("._")]
            if not csv_names:
                raise FileNotFoundError(f"No CSV file found inside price context ZIP: {path}")
            with z.open(csv_names[0]) as f:
                return pd.read_csv(f, usecols=(lambda c: c in set(usecols)) if usecols else None)
    return pd.read_csv(path, usecols=(lambda c: c in set(usecols)) if usecols else None)


def load_price_context(cfg: Config) -> pd.DataFrame | None:
    """Load optional price-context dataframe.

    Returns ``None`` when disabled or when the configured file is absent and
    ``required`` is false.
    """
    pcfg = cfg.get("price_context", default={}) or {}
    if not bool(pcfg.get("enabled", False)):
        return None
    rel = pcfg.get("csv_path") or cfg.get("paths", "price_context_csv")
    if not rel:
        if bool(pcfg.get("required", False)):
            raise ValueError("price_context.enabled=true but no price_context.csv_path was configured")
        return None
    path = Path(rel)
    if not path.is_absolute():
        path = cfg.root_dir / path
    if not path.exists():
        if bool(pcfg.get("required", False)):
            raise FileNotFoundError(f"Price context file not found: {path}")
        warnings.warn(f"Price context file not found; continuing without price context: {path}", RuntimeWarning)
        return None

    cols = list(dict.fromkeys(pcfg.get("usecols", DEFAULT_PRICE_CONTEXT_COLUMNS)))
    df = _read_csv_or_zip(path, cols)
    if "time" not in df.columns:
        raise ValueError("Price context file must contain a 'time' column")
    for c in ["time", "price_feature_time"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    gated_prefixes = {
        "use_realized_vol": ["realized_vol_"],
        "use_range_compression": ["range_compression_"],
        "use_trend_strength": ["trend_strength_"],
        "use_trend_consistency": ["trend_consistency_", "trend_direction_"],
        "use_vol_of_vol": ["vol_of_vol_"],
        "use_jump_proxy": ["jump_proxy_"],
    }
    drop_cols: list[str] = []
    for flag, prefixes in gated_prefixes.items():
        if bool(pcfg.get(flag, True)):
            continue
        drop_cols.extend([c for c in df.columns if any(c.startswith(prefix) for prefix in prefixes)])
    if drop_cols:
        df = df.drop(columns=sorted(set(drop_cols)), errors="ignore")
    df = df.dropna(subset=["time"]).sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
    return df


def merge_price_context_asof(base_df: pd.DataFrame, price_context_df: pd.DataFrame | None, cfg: Config) -> pd.DataFrame:
    """Causally attach price context to source-clock PLIE rows.

    The merge is backward-asof on timestamp.  This permits a 10m price-context
    table to support source-clock PLIE rows without looking forward.
    """
    if price_context_df is None or price_context_df.empty:
        return base_df

    time_col = cfg.get("data", "time_col", default="time")
    pcfg = cfg.get("price_context", default={}) or {}
    tolerance_min = pcfg.get("merge_tolerance_minutes", None)

    left = base_df.copy()
    left[time_col] = pd.to_datetime(left[time_col], utc=True, errors="coerce")
    right = price_context_df.copy()
    right["time"] = pd.to_datetime(right["time"], utc=True, errors="coerce")

    tol = None
    if tolerance_min is not None:
        tol = pd.Timedelta(minutes=float(tolerance_min))

    merged = pd.merge_asof(
        left.sort_values(time_col),
        right.sort_values("time"),
        left_on=time_col,
        right_on="time",
        direction="backward",
        tolerance=tol,
        suffixes=("", "_price_context"),
    )
    # Drop the duplicate right time column if pandas generated it.
    for c in ["time_price_context"]:
        if c in merged.columns:
            merged = merged.drop(columns=[c])
    return merged.sort_values(time_col).reset_index(drop=True)
