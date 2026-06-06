from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .config import ColumnsConfig, FeatureConfig
except ImportError:  # pragma: no cover - kept for direct script execution
    from config import ColumnsConfig, FeatureConfig

EPS = 1e-12

OUTPUT_FEATURE_COLUMNS = [
    'fll_spike_kama',
    'fsl_spike_kama',
    'fll_velocity_gaussian',
    'fll_acceleration_gaussian',
    'fsl_velocity_gaussian',
    'fsl_acceleration_gaussian',
    'trend_pressure',
    'kalman_slope',
    'vol_adaptive',
]

REQUIRED_INPUT_COLUMNS = ['time', 'price', 'fll_cwt_kf', 'fsl_cwt_kf']



def infer_bar_minutes(time_index: pd.DatetimeIndex) -> int:
    if len(time_index) < 3:
        raise ValueError('Need at least 3 timestamps to infer bar size.')
    diffs = time_index.to_series().diff().dropna().dt.total_seconds().to_numpy(dtype=float)
    sec = float(np.median(diffs))
    minutes = int(round(sec / 60.0))
    if minutes <= 0:
        raise ValueError('Could not infer positive bar size.')
    return minutes



def validate_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    if 'time' not in df.columns:
        raise ValueError("Input data must contain a 'time' column.")
    out = df.copy()
    out['time'] = pd.to_datetime(out['time'], utc=True, errors='coerce')
    out = out.dropna(subset=['time']).sort_values('time')
    out = out.drop_duplicates(subset=['time'], keep='last')
    out = out.set_index('time')
    out = out[~out.index.duplicated(keep='last')]
    return out



def canonicalize_input_columns(df: pd.DataFrame, columns: Optional[ColumnsConfig] = None) -> pd.DataFrame:
    out = df.copy()
    rename: Dict[str, str] = {}
    col_cfg = columns or ColumnsConfig()
    explicit = {
        col_cfg.time_col: 'time',
        col_cfg.price_col: 'price',
        col_cfg.fll_col: 'fll_cwt_kf',
        col_cfg.fsl_col: 'fsl_cwt_kf',
    }
    for src, dst in explicit.items():
        if src != dst and src in out.columns and dst not in out.columns:
            rename[src] = dst

    alias = dict(col_cfg.aliases)
    for src, dst in alias.items():
        if dst not in out.columns and src in out.columns:
            rename[src] = dst
    return out.rename(columns=rename) if rename else out



def _ensure_nonnegative_magnitude(series: pd.Series, name: str, tol: float = 1e-10) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce').astype(float)
    neg_mask = s < -tol
    if bool(neg_mask.any()):
        min_val = float(s[neg_mask].min())
        raise ValueError(f'{name} contains materially negative values (min={min_val:.6g}).')
    s = s.copy()
    s[(s < 0.0) & ~neg_mask] = 0.0
    return s.clip(lower=0.0)



def resample_to_decision_bars(df: pd.DataFrame, bar_minutes: int, decision_minutes: int) -> pd.DataFrame:
    if decision_minutes == bar_minutes:
        return df.copy()
    if decision_minutes % bar_minutes != 0:
        raise ValueError('decision_minutes must be a multiple of bar_minutes.')

    rule = f'{decision_minutes}min'
    out = pd.DataFrame(index=df.resample(rule).last().index)
    for c in ['price', 'fll_cwt_kf', 'fsl_cwt_kf']:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors='coerce').resample(rule).last()
    out = out.dropna(subset=['price']).copy()
    return out



def _causal_gaussian_kernel_1d(window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError('window must be positive.')
    if window == 1:
        return np.array([1.0], dtype=float)
    offsets = np.arange(window, dtype=float)
    sigma = max(float(window) / 3.0, 1.0)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    total = float(kernel.sum())
    return kernel / total if total > 0 else np.array([1.0], dtype=float)



def gaussian_smooth_1d(series: pd.Series, window: int) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce').astype(float)
    x = s.to_numpy(dtype=float)
    n = int(x.size)
    if n == 0 or window <= 1 or n == 1:
        return s.copy()

    kernel = _causal_gaussian_kernel_1d(window)
    mask = np.isfinite(x)
    x0 = np.where(mask, x, 0.0)
    m0 = mask.astype(float)

    num = np.convolve(x0, kernel, mode='full')[:n]
    den = np.convolve(m0, kernel, mode='full')[:n]
    out = num / (den + EPS)
    out[den < EPS] = np.nan
    return pd.Series(out, index=s.index, dtype=float)



def calculate_kama(series: pd.Series, window: int, fast_ema: int = 2, slow_ema: int = 30) -> pd.Series:
    if window <= 0:
        raise ValueError('window must be positive.')
    s = pd.to_numeric(series, errors='coerce').astype(float)
    if len(s) == 0:
        return s.copy()

    direction = (s - s.shift(window)).abs()
    volatility = s.diff().abs().rolling(window=window, min_periods=1).sum()
    er = direction / (volatility + EPS)
    fast_sc = 2.0 / (fast_ema + 1.0)
    slow_sc = 2.0 / (slow_ema + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    kama = pd.Series(index=s.index, dtype=float)
    kama.iloc[0] = s.iloc[0]
    for i in range(1, len(s)):
        prev = kama.iloc[i - 1]
        cur = s.iloc[i]
        alpha = sc.iloc[i]
        if pd.notna(cur) and pd.notna(alpha) and np.isfinite(alpha):
            kama.iloc[i] = prev + alpha * (cur - prev)
        else:
            kama.iloc[i] = prev
    return kama



def calculate_spike_ratio_kama(series: pd.Series, window: int, fast_ema: int = 2, slow_ema: int = 30) -> pd.Series:
    s = _ensure_nonnegative_magnitude(series, 'liquidation_series')
    baseline = calculate_kama(s, window=window, fast_ema=fast_ema, slow_ema=slow_ema).shift(1)
    feature = np.log((s + EPS) / (baseline + EPS))
    return feature.replace([np.inf, -np.inf], np.nan).astype(float)



def calculate_gaussian_roc_dynamics(series: pd.Series, window: int, bar_minutes: int) -> Tuple[pd.Series, pd.Series]:
    s = _ensure_nonnegative_magnitude(series, 'liquidation_series')
    x = np.log1p(s)
    smoothed = gaussian_smooth_1d(x, window=window)
    scale_per_hour = 60.0 / max(int(bar_minutes), 1)
    velocity = smoothed.diff() * scale_per_hour
    acceleration = velocity.diff() * scale_per_hour
    return velocity.astype(float), acceleration.astype(float)



def kalman_filter_slope(
    log_price: pd.Series,
    process_noise: float = 1e-6,
    slope_process_noise: float = 1e-7,
    measurement_noise: float = 1e-4,
) -> pd.Series:
    y = pd.to_numeric(log_price, errors='coerce').astype(float)
    if len(y) == 0:
        return y.copy()

    x = np.array([[y.iloc[0]], [0.0]], dtype=float)
    P = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    F = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=float)
    H = np.array([[1.0, 0.0]], dtype=float)
    Q = np.array([[float(process_noise), 0.0], [0.0, float(slope_process_noise)]], dtype=float)
    R = np.array([[float(measurement_noise)]], dtype=float)

    slopes = np.zeros(len(y), dtype=float)
    for i, yi in enumerate(y.to_numpy(dtype=float)):
        x = F @ x
        P = F @ P @ F.T + Q
        if not np.isfinite(yi):
            slopes[i] = x[1, 0]
            continue
        innov = yi - (H @ x)[0, 0]
        S = (H @ P @ H.T + R)[0, 0]
        K = (P @ H.T) / (S + EPS)
        x = x + K * innov
        P = (np.eye(2) - K @ H) @ P
        slopes[i] = x[1, 0]
    return pd.Series(slopes, index=y.index, dtype=float)



def _rolling_std(s: pd.Series, window_bars: int) -> pd.Series:
    min_periods = min(window_bars, max(3, window_bars // 3))
    return s.rolling(window_bars, min_periods=min_periods).std()



def compute_features(df: pd.DataFrame, bar_minutes: int, cfg: FeatureConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    price = pd.to_numeric(df['price'], errors='coerce').astype(float)
    logp = np.log(price.clip(lower=EPS))
    r_bar = logp.diff()

    fll = _ensure_nonnegative_magnitude(df['fll_cwt_kf'], 'fll_cwt_kf')
    fsl = _ensure_nonnegative_magnitude(df['fsl_cwt_kf'], 'fsl_cwt_kf')

    spike_window = max(int(round(cfg.liq_spike_window_min / bar_minutes)), 1)
    out['fll_spike_kama'] = calculate_spike_ratio_kama(
        fll,
        window=spike_window,
        fast_ema=cfg.liq_kama_fast_ema,
        slow_ema=cfg.liq_kama_slow_ema,
    )
    out['fsl_spike_kama'] = calculate_spike_ratio_kama(
        fsl,
        window=spike_window,
        fast_ema=cfg.liq_kama_fast_ema,
        slow_ema=cfg.liq_kama_slow_ema,
    )

    gaussian_window = max(int(round(cfg.liq_roc_gaussian_window_min / bar_minutes)), 1)
    fll_v, fll_a = calculate_gaussian_roc_dynamics(fll, window=gaussian_window, bar_minutes=bar_minutes)
    fsl_v, fsl_a = calculate_gaussian_roc_dynamics(fsl, window=gaussian_window, bar_minutes=bar_minutes)
    out['fll_velocity_gaussian'] = fll_v
    out['fll_acceleration_gaussian'] = fll_a
    out['fsl_velocity_gaussian'] = fsl_v
    out['fsl_acceleration_gaussian'] = fsl_a

    mom_k = max(int(round(cfg.trend_pressure_mom_window_min / bar_minutes)), 1)
    vol_k = max(int(round(cfg.trend_pressure_vol_window_min / bar_minutes)), 1)
    momentum = logp - logp.shift(mom_k)
    realized_vol = _rolling_std(r_bar, vol_k)
    out['trend_pressure'] = momentum / (np.sqrt(float(mom_k)) * (realized_vol + EPS))

    kalman_slope = kalman_filter_slope(
        logp,
        process_noise=float(cfg.price_kalman_process_noise),
        slope_process_noise=float(cfg.price_kalman_slope_process_noise),
        measurement_noise=float(cfg.price_kalman_measurement_noise),
    )
    out['kalman_slope'] = kalman_slope * (60.0 / max(int(bar_minutes), 1))

    short_k = max(int(round(cfg.price_vol_adaptive_short_window_min / bar_minutes)), 1)
    long_k = max(int(round(cfg.price_vol_adaptive_long_window_min / bar_minutes)), 1)
    short_vol = _rolling_std(r_bar, short_k)
    long_vol = _rolling_std(r_bar, long_k)
    log_ratio = np.log((short_vol + EPS) / (long_vol + EPS))
    gamma = float(cfg.price_vol_adaptive_sigmoid_gamma)
    sig = 1.0 / (1.0 + np.exp(-np.clip(gamma * log_ratio, -40.0, 40.0)))
    w_min = float(cfg.price_vol_adaptive_min_weight)
    weight = w_min + (1.0 - w_min) * sig
    out['vol_adaptive'] = weight * short_vol + (1.0 - weight) * long_vol

    return out[OUTPUT_FEATURE_COLUMNS].astype(float)



def prepare_decision_frame(
    raw_df: pd.DataFrame,
    cfg: FeatureConfig,
    columns: Optional[ColumnsConfig] = None,
) -> tuple[pd.DataFrame, int]:
    df = validate_and_sort(canonicalize_input_columns(raw_df, columns=columns))
    required = {'price', 'fll_cwt_kf', 'fsl_cwt_kf'}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')

    base_bar_minutes = cfg.bar_minutes or infer_bar_minutes(df.index)
    decision_minutes = cfg.decision_minutes or base_bar_minutes
    df_decision = resample_to_decision_bars(df, bar_minutes=base_bar_minutes, decision_minutes=decision_minutes)
    return df_decision, decision_minutes



def build_feature_matrix(
    raw_df: pd.DataFrame,
    cfg: FeatureConfig,
    columns: Optional[ColumnsConfig] = None,
) -> tuple[pd.DataFrame, int]:
    df_decision, decision_minutes = prepare_decision_frame(raw_df, cfg=cfg, columns=columns)
    feat = compute_features(df_decision, bar_minutes=decision_minutes, cfg=cfg)
    return feat, decision_minutes



def load_dataframe(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == '.parquet':
        return pd.read_parquet(p)
    return pd.read_csv(p)



def build_features_from_path(
    input_path: str | Path,
    cfg: FeatureConfig,
    columns: Optional[ColumnsConfig] = None,
) -> tuple[pd.DataFrame, int]:
    raw_df = load_dataframe(input_path)
    return build_feature_matrix(raw_df, cfg, columns=columns)



def save_feature_frame(df: pd.DataFrame, output_path: str | Path, time_col: str = 'time') -> str:
    out = df.reset_index().rename(columns={'index': time_col})
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == '.parquet':
        out.to_parquet(p, index=False)
    else:
        out.to_csv(p, index=False)
    return str(p)



def feature_matrix_profile(df: pd.DataFrame) -> dict:
    profile = {
        'rows': int(len(df)),
        'columns': list(df.columns),
        'nan_ratio': {c: float(pd.to_numeric(df[c], errors='coerce').isna().mean()) for c in df.columns},
    }
    if len(df) > 0:
        profile['start_time'] = str(pd.Timestamp(df.index.min()).isoformat())
        profile['end_time'] = str(pd.Timestamp(df.index.max()).isoformat())
    else:
        profile['start_time'] = None
        profile['end_time'] = None
    return profile
