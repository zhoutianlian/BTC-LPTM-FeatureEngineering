from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from liq_dataflow.feature_engineering.config import SmoothingConfig

try:
    import pywt
except ImportError:  # pragma: no cover
    pywt = None

try:
    from pykalman import KalmanFilter
except ImportError:  # pragma: no cover
    KalmanFilter = None


@dataclass(frozen=True)
class CacheStore:
    path: Path

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=["time"])
        df = pd.read_csv(self.path)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df = df.dropna(subset=["time"]).sort_values("time").drop_duplicates(subset=["time"], keep="last")
        return df

    def latest_time(self) -> pd.Timestamp | None:
        df = self.load()
        if df.empty or "time" not in df.columns:
            return None
        return pd.to_datetime(df["time"], errors="coerce").max()

    def save_incremental(self, new_rows: pd.DataFrame) -> pd.DataFrame:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing = self.load()
        if new_rows is None or new_rows.empty:
            merged = existing
        else:
            incoming = new_rows.copy()
            incoming["time"] = pd.to_datetime(incoming["time"], errors="coerce")
            merged = pd.concat([existing, incoming], ignore_index=True)
            merged = merged.dropna(subset=["time"]).sort_values("time").drop_duplicates(subset=["time"], keep="last")
        merged.to_csv(self.path, index=False)
        return merged


def _apply_lower_clip(series: pd.Series, clip_lower: float | None) -> pd.Series:
    if clip_lower is None:
        return series
    s = pd.to_numeric(series, errors="coerce").astype(float)
    return s.clip(lower=float(clip_lower))


def wavelet_approximation_trend(series: pd.Series, *, wavelet: str, level: int, clip_lower: float | None = None) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    s = _apply_lower_clip(s, clip_lower)
    if s.isna().all() or s.empty:
        return pd.Series(np.nan, index=s.index, dtype=float)

    values = s.to_numpy(dtype=float)
    if pywt is None:
        span = max(int(2 ** max(1, int(level))), 8)
        out = s.ewm(span=min(span, max(len(s), 1)), adjust=False).mean()
        return _apply_lower_clip(out, clip_lower)

    wavelet_obj = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(len(values), wavelet_obj.dec_len)
    use_level = max(1, min(int(level), int(max_level) if max_level > 0 else 1))
    coeffs = pywt.wavedec(values, wavelet_obj, level=use_level)
    approx_only = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    trend = pywt.waverec(approx_only, wavelet_obj)[: len(values)]
    out = pd.Series(trend, index=s.index, dtype=float)
    return _apply_lower_clip(out, clip_lower)


def kalman_smooth(series: pd.Series, *, cfg: SmoothingConfig, clip_lower: float | None = None) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    s = _apply_lower_clip(s, clip_lower)
    if s.empty:
        return pd.Series(np.nan, index=s.index, dtype=float)

    if KalmanFilter is None:
        out = s.ewm(span=min(12, max(len(s), 1)), adjust=False).mean()
        return _apply_lower_clip(out, clip_lower)

    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=cfg.kalman_initial_state_mean,
        initial_state_covariance=cfg.kalman_initial_state_covariance,
        observation_covariance=cfg.kalman_observation_covariance,
        transition_covariance=cfg.kalman_transition_covariance,
    )
    filtered_state_means, _ = kf.filter(s.to_numpy(dtype=float))
    out = pd.Series(filtered_state_means.reshape(-1), index=s.index, dtype=float)
    return _apply_lower_clip(out, clip_lower)


class TrailingWaveletKalmanDetrender:
    """Causal smoothing using a trailing window wavelet trend followed by Kalman smoothing."""

    def __init__(self, *, cfg: SmoothingConfig, cache: CacheStore) -> None:
        self.cfg = cfg
        self.cache = cache

    def transform(
        self,
        df: pd.DataFrame,
        *,
        input_col: str,
        output_col: str,
        clip_lower: float | None = None,
    ) -> pd.DataFrame:
        ordered = df.copy()
        ordered["time"] = pd.to_datetime(ordered["time"], errors="coerce")
        ordered = ordered.dropna(subset=["time"]).sort_values("time").drop_duplicates(subset=["time"], keep="last")
        ordered[input_col] = pd.to_numeric(ordered[input_col], errors="coerce").astype(float)
        ordered[input_col] = _apply_lower_clip(ordered[input_col], clip_lower)

        latest = self.cache.latest_time()
        cwt_col = f"{input_col}_cwt"
        records: list[dict[str, float | pd.Timestamp]] = []
        window = timedelta(hours=int(self.cfg.window_size_hours))

        for _, row in ordered.iterrows():
            t = row["time"]
            if latest is not None and t <= latest:
                continue
            hist = ordered[(ordered["time"] < t) & (ordered["time"] >= t - window)]
            if hist.empty:
                continue
            trend = wavelet_approximation_trend(
                hist[input_col],
                wavelet=self.cfg.wavelet,
                level=self.cfg.level,
                clip_lower=clip_lower,
            )
            trend_value = float(trend.iloc[-1])
            if clip_lower is not None:
                trend_value = max(trend_value, float(clip_lower))
            records.append({"time": t, cwt_col: trend_value})

        hist_df = self.cache.save_incremental(pd.DataFrame(records))
        if hist_df.empty or cwt_col not in hist_df.columns:
            return pd.DataFrame(columns=["time", output_col])

        hist_df[cwt_col] = pd.to_numeric(hist_df[cwt_col], errors="coerce").astype(float)
        hist_df[cwt_col] = _apply_lower_clip(hist_df[cwt_col], clip_lower)
        hist_df = hist_df.dropna(subset=["time"]).sort_values("time").drop_duplicates(subset=["time"], keep="last")
        # persist sanitized cache so historical negative wavelet values do not survive subsequent runs
        hist_df.to_csv(self.cache.path, index=False)

        smoothed = kalman_smooth(hist_df[cwt_col], cfg=self.cfg, clip_lower=clip_lower)
        out = hist_df[["time"]].copy()
        out[output_col] = _apply_lower_clip(smoothed, clip_lower).to_numpy(dtype=float)
        return out


def ema(series: pd.Series, *, span: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").ewm(span=span, adjust=False).mean()


def classify_delta_corr(df: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    out = df.copy()
    out["delta_fll"] = pd.to_numeric(out[col1], errors="coerce").diff()
    out["delta_fsl"] = pd.to_numeric(out[col2], errors="coerce").diff()

    conditions = [
        (out["delta_fll"] > 0) & (out["delta_fsl"] > 0) & (out[col1] > out[col2]),
        (out["delta_fll"] > 0) & (out["delta_fsl"] > 0) & (out[col1] <= out[col2]),
        (out["delta_fll"] < 0) & (out["delta_fsl"] < 0) & (out[col1] > out[col2]),
        (out["delta_fll"] < 0) & (out["delta_fsl"] < 0) & (out[col1] <= out[col2]),
        (out["delta_fll"] > 0) & (out["delta_fsl"] < 0) & (out[col1] > out[col2]),
        (out["delta_fll"] > 0) & (out["delta_fsl"] < 0) & (out[col1] <= out[col2]),
        (out["delta_fll"] < 0) & (out["delta_fsl"] > 0) & (out[col1] > out[col2]),
        (out["delta_fll"] < 0) & (out["delta_fsl"] > 0) & (out[col1] <= out[col2]),
    ]
    labels = ["LUSUL", "LUSUS", "LDSDL", "LDSDS", "LUSDL", "LUSDS", "LDSUL", "LDSUS"]
    out["corr_case"] = np.select(conditions, labels, default="None")
    return out
