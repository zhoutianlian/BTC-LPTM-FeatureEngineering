from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


STAT_KEYS = [
    'count',
    'mean',
    'std',
    'min',
    'max',
    'median',
    'p01',
    'p05',
    'p25',
    'p75',
    'p95',
    'p99',
    'skew',
    'kurtosis',
    'zero_ratio',
    'positive_ratio',
    'negative_ratio',
    'unique_count',
    'constant_flag',
]


def as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce').astype(float)


def finite_series(series: pd.Series) -> pd.Series:
    x = as_numeric(series)
    return x[np.isfinite(x)]


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def compute_numeric_stats(series: pd.Series, *, constant_rtol: float = 1e-10, constant_atol: float = 1e-12) -> Dict[str, Any]:
    x = as_numeric(series)
    finite = x[np.isfinite(x)]
    n = int(len(finite))
    if n == 0:
        return {
            'count': 0,
            'mean': None,
            'std': None,
            'min': None,
            'max': None,
            'median': None,
            'p01': None,
            'p05': None,
            'p25': None,
            'p75': None,
            'p95': None,
            'p99': None,
            'skew': None,
            'kurtosis': None,
            'zero_ratio': None,
            'positive_ratio': None,
            'negative_ratio': None,
            'unique_count': 0,
            'constant_flag': True,
        }

    mean = _safe_float(finite.mean())
    std = _safe_float(finite.std(ddof=1)) if n > 1 else 0.0
    min_v = _safe_float(finite.min())
    max_v = _safe_float(finite.max())
    median = _safe_float(finite.median())
    unique_count = int(finite.nunique(dropna=True))
    scale = max(abs(mean or 0.0), abs(median or 0.0), 1.0)
    constant_flag = bool(unique_count <= 1 or (std is not None and std <= constant_atol + constant_rtol * scale))

    quantiles = finite.quantile([0.01, 0.05, 0.25, 0.75, 0.95, 0.99])
    return {
        'count': n,
        'mean': mean,
        'std': std,
        'min': min_v,
        'max': max_v,
        'median': median,
        'p01': _safe_float(quantiles.loc[0.01]),
        'p05': _safe_float(quantiles.loc[0.05]),
        'p25': _safe_float(quantiles.loc[0.25]),
        'p75': _safe_float(quantiles.loc[0.75]),
        'p95': _safe_float(quantiles.loc[0.95]),
        'p99': _safe_float(quantiles.loc[0.99]),
        'skew': _safe_float(finite.skew()) if n > 2 else None,
        'kurtosis': _safe_float(finite.kurtosis()) if n > 3 else None,
        'zero_ratio': float(np.isclose(finite.to_numpy(dtype=float), 0.0, rtol=constant_rtol, atol=constant_atol).mean()),
        'positive_ratio': float((finite > 0.0).mean()),
        'negative_ratio': float((finite < 0.0).mean()),
        'unique_count': unique_count,
        'constant_flag': constant_flag,
    }


def compute_missing_profile(series: pd.Series) -> Dict[str, Any]:
    numeric = as_numeric(series)
    total = int(len(series))
    raw_na = series.isna()
    numeric_na = numeric.isna()
    pos_inf = numeric == np.inf
    neg_inf = numeric == -np.inf
    inf = pos_inf | neg_inf
    invalid = numeric_na | inf
    return {
        'total_count': total,
        'missing_count': int(raw_na.sum()),
        'missing_ratio': float(raw_na.mean()) if total else 1.0,
        'nan_count': int(numeric_na.sum()),
        'nan_ratio': float(numeric_na.mean()) if total else 1.0,
        'pos_inf_count': int(pos_inf.sum()),
        'neg_inf_count': int(neg_inf.sum()),
        'inf_count': int(inf.sum()),
        'inf_ratio': float(inf.mean()) if total else 0.0,
        'valid_count': int((~invalid).sum()),
        'valid_ratio': float((~invalid).mean()) if total else 0.0,
    }

