from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

try:
    from ..config import ReportConfig
except ImportError:  # pragma: no cover
    from config import ReportConfig

from .feature_stats import as_numeric, compute_missing_profile, compute_numeric_stats


@dataclass
class FeatureMetadata:
    name: str
    category: str
    description: str
    calculation: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def extract_feature_metadata(doc_path: str | Path, fallback_features: Iterable[str]) -> Dict[str, FeatureMetadata]:
    path = Path(doc_path)
    if not path.exists():
        return {
            f: FeatureMetadata(name=f, category='undocumented', description='Feature documentation not found.', calculation='')
            for f in fallback_features
        }

    lines = path.read_text(encoding='utf-8').splitlines()
    category = 'uncategorized'
    items: Dict[str, FeatureMetadata] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('## '):
            category = re.sub(r'^#+\s*', '', line).strip()
        match = re.match(r'^###\s+\d+\)\s+`?([A-Za-z0-9_]+)`?\s*$', line.strip())
        if match:
            name = match.group(1)
            block: List[str] = []
            i += 1
            while i < len(lines) and not re.match(r'^###\s+\d+\)\s+`?[A-Za-z0-9_]+`?\s*$', lines[i].strip()):
                block.append(lines[i])
                i += 1
            description = _extract_section_text(block, 'Financial meaning')
            calculation = _extract_section_text(block, 'Calculation')
            items[name] = FeatureMetadata(
                name=name,
                category=category,
                description=description or _first_nonempty_lines(block, limit=3),
                calculation=calculation,
            )
            continue
        i += 1

    for f in fallback_features:
        items.setdefault(
            f,
            FeatureMetadata(name=f, category='undocumented', description='Feature is expected by code but not found in docs.', calculation=''),
        )
    return items


def _extract_section_text(block: List[str], section_title: str) -> str:
    start = None
    for i, line in enumerate(block):
        if line.strip().lower() == f'#### {section_title}'.lower():
            start = i + 1
            break
    if start is None:
        return ''
    out: List[str] = []
    for line in block[start:]:
        if line.startswith('#### '):
            break
        if line.strip():
            out.append(line.strip())
    return '\n'.join(out).strip()


def _first_nonempty_lines(block: List[str], limit: int) -> str:
    out = [line.strip() for line in block if line.strip() and not line.startswith('#')]
    return '\n'.join(out[:limit])


def _max_true_run(mask: pd.Series) -> int:
    max_run = 0
    cur = 0
    for value in mask.fillna(False).astype(bool).to_numpy():
        if value:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return int(max_run)


def time_index_profile(df: pd.DataFrame, *, time_col: str = 'time', gap_multiplier: float = 3.0) -> Dict[str, Any]:
    if time_col not in df.columns:
        return {
            'time_col': time_col,
            'has_time_col': False,
            'duplicate_timestamp_count': None,
            'is_monotonic_increasing': None,
            'median_interval_minutes': None,
            'time_gap_anomaly_count': None,
            'time_gap_examples': [],
        }
    t = pd.to_datetime(df[time_col], utc=True, errors='coerce')
    valid = t.dropna()
    duplicate_count = int(valid.duplicated(keep=False).sum())
    monotonic = bool(valid.is_monotonic_increasing)
    diffs = valid.diff().dropna()
    if diffs.empty:
        median_minutes = None
        anomaly_count = 0
        examples: List[Dict[str, Any]] = []
    else:
        seconds = diffs.dt.total_seconds()
        median_sec = float(seconds.median())
        median_minutes = median_sec / 60.0 if np.isfinite(median_sec) else None
        threshold = median_sec * float(gap_multiplier) if median_sec > 0 else np.inf
        gap_mask = seconds > threshold
        anomaly_count = int(gap_mask.sum())
        examples = []
        if anomaly_count:
            gap_df = pd.DataFrame({'time': valid.loc[gap_mask.index], 'gap_minutes': seconds / 60.0})
            for _, row in gap_df.loc[gap_mask].head(20).iterrows():
                examples.append({'time': row['time'].isoformat(), 'gap_minutes': float(row['gap_minutes'])})
    return {
        'time_col': time_col,
        'has_time_col': True,
        'start_time': valid.min().isoformat() if len(valid) else None,
        'end_time': valid.max().isoformat() if len(valid) else None,
        'duplicate_timestamp_count': duplicate_count,
        'is_monotonic_increasing': monotonic,
        'median_interval_minutes': median_minutes,
        'time_gap_anomaly_count': anomaly_count,
        'time_gap_examples': examples,
    }


def outlier_diagnostics(
    series: pd.Series,
    times: pd.Series,
    cfg: ReportConfig,
) -> Dict[str, Any]:
    x = as_numeric(series)
    finite_mask = np.isfinite(x)
    finite = x[finite_mask]
    if len(finite) == 0:
        return {
            'zscore_outlier_count': 0,
            'iqr_outlier_count': 0,
            'extreme_quantile_count': 0,
            'explosion_value_count': 0,
            'cliff_jump_count': 0,
            'examples': [],
        }

    mean = float(finite.mean())
    std = float(finite.std(ddof=1)) if len(finite) > 1 else 0.0
    z = pd.Series(np.nan, index=x.index, dtype=float)
    if std > 0:
        z.loc[finite.index] = (finite - mean) / std
    z_mask = z.abs() > float(cfg.zscore_threshold)

    q1 = float(finite.quantile(0.25))
    q3 = float(finite.quantile(0.75))
    iqr = q3 - q1
    if iqr > 0:
        iqr_low = q1 - float(cfg.iqr_multiplier) * iqr
        iqr_high = q3 + float(cfg.iqr_multiplier) * iqr
        iqr_mask = (x < iqr_low) | (x > iqr_high)
    else:
        iqr_low = q1
        iqr_high = q3
        iqr_mask = pd.Series(False, index=x.index)

    q_low = float(finite.quantile(float(cfg.extreme_quantile_low)))
    q_high = float(finite.quantile(float(cfg.extreme_quantile_high)))
    extreme_mask = (x < q_low) | (x > q_high)

    med = float(finite.median())
    mad = float(np.nanmedian(np.abs(finite.to_numpy(dtype=float) - med)))
    explosion_threshold = abs(med) + float(cfg.explosion_mad_multiplier) * max(mad, 1e-12)
    explosion_mask = x.abs() > explosion_threshold

    diff = x.diff().abs()
    diff_finite = diff[np.isfinite(diff)]
    if len(diff_finite):
        diff_med = float(diff_finite.median())
        diff_mad = float(np.nanmedian(np.abs(diff_finite.to_numpy(dtype=float) - diff_med)))
        cliff_threshold = diff_med + float(cfg.cliff_mad_multiplier) * max(diff_mad, 1e-12)
        cliff_mask = diff > cliff_threshold
    else:
        cliff_threshold = None
        cliff_mask = pd.Series(False, index=x.index)

    combined = (z_mask | iqr_mask | extreme_mask | explosion_mask | cliff_mask).fillna(False)
    score = z.abs().fillna(0.0)
    if float(score.max() or 0.0) == 0.0:
        scale = max(float(finite.abs().median()), 1e-12)
        score = x.abs() / scale
    examples = []
    top_index = score[combined].sort_values(ascending=False).head(int(cfg.max_outlier_examples)).index
    time_values = pd.to_datetime(times, utc=True, errors='coerce')
    for idx in top_index:
        ts = time_values.iloc[idx] if isinstance(idx, int) and idx < len(time_values) else time_values.loc[idx]
        examples.append({
            'time': ts.isoformat() if pd.notna(ts) else None,
            'value': _safe_json_float(x.loc[idx]),
            'zscore': _safe_json_float(z.loc[idx]),
            'reason': ','.join(_outlier_reasons(idx, z_mask, iqr_mask, extreme_mask, explosion_mask, cliff_mask)),
        })

    return {
        'zscore_outlier_count': int(z_mask.fillna(False).sum()),
        'iqr_outlier_count': int(iqr_mask.fillna(False).sum()),
        'iqr_low': _safe_json_float(iqr_low),
        'iqr_high': _safe_json_float(iqr_high),
        'extreme_quantile_count': int(extreme_mask.fillna(False).sum()),
        'extreme_low': _safe_json_float(q_low),
        'extreme_high': _safe_json_float(q_high),
        'explosion_value_count': int(explosion_mask.fillna(False).sum()),
        'explosion_threshold': _safe_json_float(explosion_threshold),
        'cliff_jump_count': int(cliff_mask.fillna(False).sum()),
        'cliff_threshold': _safe_json_float(cliff_threshold),
        'examples': examples,
    }


def _safe_json_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _outlier_reasons(idx: Any, *masks: pd.Series) -> List[str]:
    names = ['zscore', 'iqr', 'tail', 'explosion', 'cliff']
    reasons = []
    for name, mask in zip(names, masks):
        try:
            if bool(mask.loc[idx]):
                reasons.append(name)
        except Exception:
            pass
    return reasons


def time_series_diagnostics(series: pd.Series, cfg: ReportConfig) -> Dict[str, Any]:
    x = as_numeric(series)
    valid = np.isfinite(x)
    missing = ~valid
    zero = pd.Series(np.isclose(x.fillna(np.nan), 0.0, rtol=float(cfg.constant_rtol), atol=float(cfg.constant_atol)), index=x.index)
    near_same = x.diff().abs() <= float(cfg.constant_atol) + float(cfg.constant_rtol) * x.abs().shift(1).fillna(1.0).clip(lower=1.0)
    return {
        'max_missing_run': _max_true_run(missing),
        'max_zero_run': _max_true_run(zero & valid),
        'max_near_constant_run': _max_true_run(near_same.fillna(False) & valid),
    }


def leakage_diagnostics(feature_name: str) -> Dict[str, Any]:
    if 'spike_kama' in feature_name:
        note = 'KAMA baseline is expected to be shifted by one bar; confirm upstream source timestamps.'
        status = 'NEEDS_MANUAL_CONFIRM'
    elif 'gaussian' in feature_name:
        note = 'Gaussian dynamics are expected to use one-sided smoothing and backward differences.'
        status = 'NEEDS_MANUAL_CONFIRM'
    elif feature_name in {'trend_pressure', 'vol_adaptive'}:
        note = 'Rolling windows are expected to be trailing windows; confirm no centered rolling upstream.'
        status = 'NEEDS_MANUAL_CONFIRM'
    elif feature_name == 'kalman_slope':
        note = 'Kalman slope is expected to use forward filtering only, not smoothing.'
        status = 'NEEDS_MANUAL_CONFIRM'
    else:
        note = 'No automatic leakage rule is available for this feature.'
        status = 'NEEDS_MANUAL_CONFIRM'
    return {'status': status, 'note': note}


def diagnose_feature(
    df: pd.DataFrame,
    feature: str,
    metadata: FeatureMetadata,
    time_profile: Dict[str, Any],
    cfg: ReportConfig,
    *,
    time_col: str = 'time',
) -> Dict[str, Any]:
    issues: List[Dict[str, str]] = []
    if feature not in df.columns:
        return {
            'feature_name': feature,
            'category': metadata.category,
            'description': metadata.description,
            'calculation': metadata.calculation,
            'exists': False,
            'dtype': None,
            'status': 'FAIL',
            'issues': [{'severity': 'FAIL', 'code': 'missing_feature', 'message': 'Feature is documented but missing from output data.'}],
            'stats': {},
            'missing': {},
            'outliers': {},
            'time_series': {},
            'time_index': time_profile,
            'leakage': leakage_diagnostics(feature),
        }

    series = df[feature]
    dtype = str(series.dtype)
    missing = compute_missing_profile(series)
    stats = compute_numeric_stats(series, constant_rtol=float(cfg.constant_rtol), constant_atol=float(cfg.constant_atol))
    outliers = outlier_diagnostics(series, df[time_col], cfg)
    ts_diag = time_series_diagnostics(series, cfg)

    non_null = int(series.notna().sum())
    numeric_convertible = int(missing['valid_count']) + int(missing['inf_count'])
    dtype_reasonable = bool(non_null == 0 or numeric_convertible / max(non_null, 1) >= 0.95)

    if not dtype_reasonable:
        issues.append({'severity': 'FAIL', 'code': 'non_numeric_dtype', 'message': f'Column dtype {dtype} is not reliably numeric.'})
    if int(missing['total_count']) == 0:
        issues.append({'severity': 'FAIL', 'code': 'empty_data', 'message': 'Feature output has no rows.'})
    if int(missing['nan_count']) == int(missing['total_count']):
        issues.append({'severity': 'FAIL', 'code': 'all_nan', 'message': 'Feature is entirely NaN after numeric coercion.'})
    if int(missing['inf_count']) >= int(cfg.inf_fail_count):
        issues.append({'severity': 'FAIL', 'code': 'infinite_values', 'message': 'Feature contains inf or -inf values.'})
    if int(missing['valid_count']) < int(cfg.min_valid_count):
        issues.append({'severity': 'FAIL', 'code': 'insufficient_valid_samples', 'message': 'Valid sample count is below configured minimum.'})
    if float(missing['nan_ratio']) >= float(cfg.missing_fail_ratio):
        issues.append({'severity': 'FAIL', 'code': 'high_missing_ratio', 'message': 'Missing/NaN ratio exceeds fail threshold.'})
    elif float(missing['nan_ratio']) > float(cfg.missing_warn_ratio):
        issues.append({'severity': 'WARN', 'code': 'missing_ratio', 'message': 'Missing/NaN ratio exceeds warning threshold.'})
    if bool(stats.get('constant_flag')):
        issues.append({'severity': 'WARN', 'code': 'constant_feature', 'message': 'Feature is approximately constant.'})
    if int(outliers['zscore_outlier_count']) > 0 or int(outliers['iqr_outlier_count']) > 0:
        issues.append({'severity': 'WARN', 'code': 'outliers_detected', 'message': 'Outliers were detected by z-score or IQR diagnostics.'})
    if int(outliers.get('explosion_value_count') or 0) > 0:
        issues.append({'severity': 'WARN', 'code': 'explosion_values', 'message': 'Potential explosion values detected by robust MAD threshold.'})
    if int(outliers.get('cliff_jump_count') or 0) > 0:
        issues.append({'severity': 'WARN', 'code': 'cliff_jumps', 'message': 'Large one-bar jumps detected.'})
    if int(ts_diag['max_near_constant_run']) >= int(cfg.long_constant_min_bars):
        issues.append({'severity': 'WARN', 'code': 'long_constant_run', 'message': 'Long near-constant run detected.'})
    if int(ts_diag['max_zero_run']) >= int(cfg.long_constant_min_bars):
        issues.append({'severity': 'WARN', 'code': 'long_zero_run', 'message': 'Long zero run detected.'})
    if int(ts_diag['max_missing_run']) >= int(cfg.long_constant_min_bars):
        issues.append({'severity': 'WARN', 'code': 'long_missing_run', 'message': 'Long missing run detected.'})

    if time_profile.get('duplicate_timestamp_count'):
        issues.append({'severity': 'WARN', 'code': 'duplicate_timestamps', 'message': 'Duplicate timestamps exist in report data.'})
    if time_profile.get('is_monotonic_increasing') is False:
        issues.append({'severity': 'WARN', 'code': 'time_not_monotonic', 'message': 'Time index is not monotonic increasing.'})
    if time_profile.get('time_gap_anomaly_count'):
        issues.append({'severity': 'WARN', 'code': 'time_gap_anomaly', 'message': 'Abnormal time interval jumps were detected.'})

    status = 'PASS'
    if any(issue['severity'] == 'FAIL' for issue in issues):
        status = 'FAIL'
    elif any(issue['severity'] == 'WARN' for issue in issues):
        status = 'WARN'

    return {
        'feature_name': feature,
        'category': metadata.category,
        'description': metadata.description,
        'calculation': metadata.calculation,
        'exists': True,
        'dtype': dtype,
        'dtype_reasonable': dtype_reasonable,
        'status': status,
        'issues': issues,
        'stats': stats,
        'missing': missing,
        'outliers': outliers,
        'time_series': ts_diag,
        'time_index': time_profile,
        'leakage': leakage_diagnostics(feature),
    }

