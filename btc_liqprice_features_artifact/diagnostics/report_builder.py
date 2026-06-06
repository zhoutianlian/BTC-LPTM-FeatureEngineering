from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs

try:
    from ..config import ReportConfig
    from ..features import OUTPUT_FEATURE_COLUMNS
except ImportError:  # pragma: no cover
    from config import ReportConfig
    from features import OUTPUT_FEATURE_COLUMNS

from .feature_checks import diagnose_feature, extract_feature_metadata, time_index_profile
from .feature_plots import (
    figure_html,
    make_correlation_heatmap,
    make_distribution_figure,
    make_missing_figure,
    make_relationship_figure,
    make_rolling_figure,
    make_time_series_figure,
)
from .html_templates import format_value, issues_html, metric_cards, status_badge, write_assets


def generate_feature_diagnostics_report(
    feature_df: pd.DataFrame,
    *,
    output_dir: str | Path,
    cfg: Optional[ReportConfig] = None,
    feature_doc_path: str | Path | None = None,
    price_context_df: Optional[pd.DataFrame] = None,
    time_col: str = 'time',
    logger: Any = None,
) -> Dict[str, Any]:
    report_cfg = cfg or ReportConfig()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features_dir = out_dir / 'features'
    assets_dir = out_dir / 'assets'
    if report_cfg.generate_html:
        features_dir.mkdir(parents=True, exist_ok=True)
        write_assets(assets_dir, get_plotlyjs())

    df = _prepare_report_frame(feature_df, price_context_df=price_context_df, time_col=time_col, cfg=report_cfg)
    doc_path = feature_doc_path or report_cfg.feature_doc_path
    metadata = extract_feature_metadata(doc_path, OUTPUT_FEATURE_COLUMNS)
    important_features = list(metadata.keys())
    time_profile = time_index_profile(df, time_col='time', gap_multiplier=float(report_cfg.time_gap_multiplier))

    diagnostics: Dict[str, Dict[str, Any]] = {}
    for feature in important_features:
        diagnostics[feature] = diagnose_feature(df, feature, metadata[feature], time_profile, report_cfg, time_col='time')

    corr_df, high_corr_pairs, relationship_notes = _relationship_diagnostics(df, important_features, report_cfg)
    for feature, rel in relationship_notes.items():
        if feature in diagnostics:
            diagnostics[feature]['relationships'] = rel

    overview = _build_overview(df, diagnostics, important_features, corr_df, high_corr_pairs)
    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        'project': 'btc_liqprice_features_artifact Feature Diagnostics',
        'generated_at': generated_at,
        'output_dir': str(out_dir),
        'feature_doc_path': str(doc_path),
        'overview': overview,
        'time_profile': time_profile,
        'important_features': important_features,
        'features': diagnostics,
        'correlation_matrix': corr_df.to_dict() if not corr_df.empty else {},
        'high_correlation_pairs': high_corr_pairs,
    }

    created_files: List[str] = []
    if report_cfg.generate_summary_json:
        summary_path = out_dir / 'summary.json'
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding='utf-8')
        created_files.append(str(summary_path))

    if report_cfg.generate_html:
        for feature in important_features:
            page = _write_feature_page(
                df,
                diagnostics[feature],
                out_dir=features_dir,
                cfg=report_cfg,
            )
            diagnostics[feature]['detail_link'] = f'features/{page.name}'
            created_files.append(str(page))
        index_path = _write_index_page(
            out_dir,
            diagnostics,
            overview,
            generated_at,
            time_profile,
            corr_df,
            high_corr_pairs,
        )
        created_files.append(str(index_path))

    if logger is not None:
        logger.info('feature_diagnostics_report_dir=%s', out_dir.resolve())
        logger.info('feature_diagnostics_feature_count=%s', len(important_features))
        logger.info('feature_diagnostics_fail_count=%s', overview['fail_feature_count'])
        logger.info('feature_diagnostics_warn_count=%s', overview['warn_feature_count'])

    return {'output_dir': str(out_dir), 'created_files': created_files, 'summary': payload}


def _prepare_report_frame(
    feature_df: pd.DataFrame,
    *,
    price_context_df: Optional[pd.DataFrame],
    time_col: str,
    cfg: ReportConfig,
) -> pd.DataFrame:
    df = _ensure_time_column(feature_df, time_col=time_col)
    df = df.sort_values('time').reset_index(drop=True)

    if price_context_df is not None and bool(cfg.price_context_enabled):
        ctx = _ensure_time_column(price_context_df, time_col=time_col)
        if 'price' in ctx.columns:
            ctx = ctx[['time', 'price']].copy().sort_values('time')
            if 'price' not in df.columns:
                if len(ctx):
                    tolerance = _infer_merge_tolerance(df['time'])
                    df = pd.merge_asof(df.sort_values('time'), ctx, on='time', direction='backward', tolerance=tolerance)
                else:
                    df['price'] = np.nan
    if 'price' in df.columns:
        price = pd.to_numeric(df['price'], errors='coerce').astype(float)
        logp = np.log(price.clip(lower=1e-12))
        df['return_1bar'] = logp.diff()
        for period in cfg.future_return_periods:
            p = int(period)
            if p > 0:
                df[f'future_return_{p}bar'] = logp.shift(-p) - logp
    return df


def _ensure_time_column(df: pd.DataFrame, *, time_col: str = 'time') -> pd.DataFrame:
    out = df.copy()
    if time_col in out.columns:
        out = out.rename(columns={time_col: 'time'}) if time_col != 'time' else out
    elif isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index().rename(columns={'index': 'time'})
    else:
        raise ValueError(f"Report data must contain a '{time_col}' column or have a DatetimeIndex.")
    out['time'] = pd.to_datetime(out['time'], utc=True, errors='coerce')
    out = out.dropna(subset=['time'])
    return out


def _infer_merge_tolerance(time_values: pd.Series) -> pd.Timedelta:
    t = pd.to_datetime(time_values, utc=True, errors='coerce').dropna()
    diffs = t.diff().dropna()
    if diffs.empty:
        return pd.Timedelta('1D')
    return pd.Timedelta(seconds=max(float(diffs.dt.total_seconds().median()) * 1.5, 1.0))


def _relationship_diagnostics(
    df: pd.DataFrame,
    features: List[str],
    cfg: ReportConfig,
) -> tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    existing_features = [f for f in features if f in df.columns]
    rel_cols = existing_features.copy()
    relationship_targets = []
    if 'price' in df.columns:
        relationship_targets.append('price')
    if 'return_1bar' in df.columns:
        relationship_targets.append('return_1bar')
    future_cols = [c for c in df.columns if c.startswith('future_return_')]
    relationship_targets.extend(future_cols)
    rel_cols.extend(relationship_targets)
    rel_cols = list(dict.fromkeys(rel_cols))
    if not rel_cols:
        return pd.DataFrame(), [], {
            f: {'price_available': False, 'skip_reason': 'No price or return context is available.'}
            for f in features
        }

    numeric = df[rel_cols].apply(pd.to_numeric, errors='coerce')
    corr = numeric.corr(method=cfg.correlation_method, min_periods=max(3, int(cfg.min_valid_count // 3)))
    high_pairs: List[Dict[str, Any]] = []
    for i, left in enumerate(existing_features):
        for right in existing_features[i + 1:]:
            value = corr.loc[left, right] if left in corr.index and right in corr.columns else np.nan
            if np.isfinite(value) and abs(float(value)) >= float(cfg.high_corr_threshold):
                high_pairs.append({'feature_a': left, 'feature_b': right, 'correlation': float(value)})

    notes: Dict[str, Dict[str, Any]] = {}
    for feature in features:
        rel: Dict[str, Any] = {'price_available': 'price' in df.columns}
        if feature not in corr.index:
            rel['skip_reason'] = 'Feature is missing or not numeric enough for correlation.'
        else:
            for target in relationship_targets:
                rel[f'corr_{target}'] = _json_float(corr.loc[feature, target]) if target in corr.columns else None
            if not relationship_targets:
                rel['skip_reason'] = 'No price, current return, or future return columns are available.'
        notes[feature] = rel
    return corr, high_pairs, notes


def _build_overview(
    df: pd.DataFrame,
    diagnostics: Dict[str, Dict[str, Any]],
    important_features: List[str],
    corr_df: pd.DataFrame,
    high_corr_pairs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    statuses = [diagnostics[f]['status'] for f in important_features]
    existing = [f for f in important_features if diagnostics[f].get('exists')]
    missing = [f for f in important_features if not diagnostics[f].get('exists')]
    return {
        'sample_count': int(len(df)),
        'data_start_time': df['time'].min().isoformat() if len(df) else None,
        'data_end_time': df['time'].max().isoformat() if len(df) else None,
        'feature_count': int(len(important_features)),
        'existing_feature_count': int(len(existing)),
        'missing_feature_count': int(len(missing)),
        'fail_feature_count': int(sum(s == 'FAIL' for s in statuses)),
        'warn_feature_count': int(sum(s == 'WARN' for s in statuses)),
        'pass_feature_count': int(sum(s == 'PASS' for s in statuses)),
        'missing_features': missing,
        'high_correlation_pair_count': int(len(high_corr_pairs)),
        'correlation_columns': list(corr_df.columns) if not corr_df.empty else [],
    }


def _write_index_page(
    out_dir: Path,
    diagnostics: Dict[str, Dict[str, Any]],
    overview: Dict[str, Any],
    generated_at: str,
    time_profile: Dict[str, Any],
    corr_df: pd.DataFrame,
    high_corr_pairs: List[Dict[str, Any]],
) -> Path:
    rows_html = []
    headers = [
        'feature_name',
        'category',
        'exists',
        'dtype',
        'valid_count',
        'missing_ratio',
        'inf_count',
        'mean',
        'std',
        'min',
        'max',
        'p01',
        'p50',
        'p99',
        'skew',
        'kurtosis',
        'outlier_count',
        'constant_flag',
        'status',
        'detail_link',
    ]
    for feature, diag in diagnostics.items():
        stats = diag.get('stats', {})
        missing = diag.get('missing', {})
        outliers = diag.get('outliers', {})
        status = diag.get('status', 'WARN')
        link = diag.get('detail_link') or f'features/{_feature_filename(feature)}'
        cells = [
            feature,
            diag.get('category'),
            diag.get('exists'),
            diag.get('dtype'),
            missing.get('valid_count'),
            missing.get('nan_ratio'),
            missing.get('inf_count'),
            stats.get('mean'),
            stats.get('std'),
            stats.get('min'),
            stats.get('max'),
            stats.get('p01'),
            stats.get('median'),
            stats.get('p99'),
            stats.get('skew'),
            stats.get('kurtosis'),
            max(
                int(outliers.get('zscore_outlier_count') or 0),
                int(outliers.get('iqr_outlier_count') or 0),
                int(outliers.get('extreme_quantile_count') or 0),
            ),
            stats.get('constant_flag'),
        ]
        row_class = f'row-{status.lower()}'
        cell_html = ''.join(f'<td>{html.escape(format_value(v))}</td>' for v in cells)
        cell_html += f'<td>{status_badge(status)}</td>'
        cell_html += f'<td><a href="{html.escape(link)}">detail</a></td>'
        rows_html.append(f'<tr class="{row_class}">{cell_html}</tr>')

    cards = metric_cards([
        ('Generated At', generated_at),
        ('Data Range', f"{overview.get('data_start_time')} to {overview.get('data_end_time')}"),
        ('Total Samples', overview.get('sample_count')),
        ('Important Features', overview.get('feature_count')),
        ('Existing Features', overview.get('existing_feature_count')),
        ('Missing Features', overview.get('missing_feature_count')),
        ('FAIL Features', overview.get('fail_feature_count')),
        ('WARN Features', overview.get('warn_feature_count')),
        ('PASS Features', overview.get('pass_feature_count')),
        ('Median Bar Minutes', time_profile.get('median_interval_minutes')),
        ('Duplicate Timestamps', time_profile.get('duplicate_timestamp_count')),
        ('Gap Anomalies', time_profile.get('time_gap_anomaly_count')),
    ])

    heatmap_html = ''
    heatmap_fig = make_correlation_heatmap(corr_df, title='Feature / price / return correlation heatmap')
    if heatmap_fig is not None:
        heatmap_html = f'<div class="plot">{figure_html(heatmap_fig)}</div>'

    high_corr_rows = [
        [item['feature_a'], item['feature_b'], item['correlation']]
        for item in high_corr_pairs
    ]
    high_corr_table = _table_html(['feature_a', 'feature_b', 'correlation'], high_corr_rows, table_id='high-corr-table')
    html_text = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>btc_liqprice_features_artifact Feature Diagnostics</title>
  <link rel="stylesheet" href="assets/css/theme.css" />
</head>
<body>
  <div class="page">
    <div class="topbar">
      <div>
        <div class="title">btc_liqprice_features_artifact Feature Diagnostics</div>
        <div class="subtitle">Automated validity checks and interactive statistical charts for documented BTC liquidation-price output features.</div>
      </div>
    </div>
    <div class="grid metric-grid">{cards}</div>
    <div class="panel">
      <div class="table-toolbar">
        <div class="panel-title">Feature Summary</div>
        <input class="search" data-table-search="#summary-table" placeholder="Search feature, category, status..." />
      </div>
      <div class="table-wrap">
        <table id="summary-table" data-sortable="true">
          <thead><tr>{''.join(f'<th>{html.escape(h)}</th>' for h in headers)}</tr></thead>
          <tbody>{''.join(rows_html)}</tbody>
        </table>
      </div>
    </div>
    <div class="panel">
      <div class="panel-title">Correlation Diagnostics</div>
      <div class="note">Correlations are diagnostic only and should not be read as confirmed predictive power.</div>
      {heatmap_html}
      <div class="panel-title">High Correlation Feature Pairs</div>
      {high_corr_table}
    </div>
  </div>
  <script src="assets/js/plotly.min.js"></script>
  <script src="assets/js/table.js"></script>
</body>
</html>"""
    index_path = out_dir / 'index.html'
    index_path.write_text(html_text, encoding='utf-8')
    return index_path


def _write_feature_page(df: pd.DataFrame, diagnostics: Dict[str, Any], *, out_dir: Path, cfg: ReportConfig) -> Path:
    feature = diagnostics['feature_name']
    out_path = out_dir / _feature_filename(feature)
    status = diagnostics.get('status', 'WARN')
    missing = diagnostics.get('missing', {})
    stats = diagnostics.get('stats', {})
    outliers = diagnostics.get('outliers', {})
    ts_diag = diagnostics.get('time_series', {})
    relationship = diagnostics.get('relationships', {})

    cards = metric_cards([
        ('Status', status),
        ('Category', diagnostics.get('category')),
        ('Exists', diagnostics.get('exists')),
        ('Valid Count', missing.get('valid_count')),
        ('Missing Ratio', format_value(missing.get('nan_ratio'))),
        ('Inf Count', missing.get('inf_count')),
        ('Z Outliers', outliers.get('zscore_outlier_count')),
        ('IQR Outliers', outliers.get('iqr_outlier_count')),
        ('Cliff Jumps', outliers.get('cliff_jump_count')),
        ('Max Missing Run', ts_diag.get('max_missing_run')),
        ('Max Zero Run', ts_diag.get('max_zero_run')),
        ('Constant Flag', stats.get('constant_flag')),
    ])

    stat_rows = [[k, stats.get(k)] for k in [
        'count', 'mean', 'std', 'min', 'max', 'median', 'p01', 'p05', 'p25', 'p75', 'p95', 'p99',
        'skew', 'kurtosis', 'zero_ratio', 'positive_ratio', 'negative_ratio', 'unique_count', 'constant_flag',
    ]]
    missing_rows = [[k, v] for k, v in missing.items()]
    outlier_rows = [[k, v] for k, v in outliers.items() if k != 'examples']
    rel_rows = [[k, v] for k, v in relationship.items()]
    examples = outliers.get('examples') or []
    example_rows = [[item.get('time'), item.get('value'), item.get('zscore'), item.get('reason')] for item in examples]

    plot_html = ''
    if diagnostics.get('exists') and feature in df.columns:
        future_col = _first_future_return_col(df)
        figures = [
            make_time_series_figure(df, feature, diagnostics),
            make_distribution_figure(df, feature, diagnostics),
            make_rolling_figure(df, feature, cfg),
            make_missing_figure(df, feature, cfg),
        ]
        rel_fig = make_relationship_figure(df, feature, future_return_col=future_col)
        if rel_fig is not None:
            figures.append(rel_fig)
        plot_html = ''.join(f'<div class="plot">{figure_html(fig)}</div>' for fig in figures)
    else:
        plot_html = '<div class="panel note">This feature is missing from output data, so no charts were generated.</div>'

    html_text = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(feature)} diagnostics</title>
  <link rel="stylesheet" href="../assets/css/theme.css" />
</head>
<body>
  <div class="page">
    <div class="topbar">
      <div>
        <div class="title">{html.escape(feature)} {status_badge(status)}</div>
        <div class="subtitle">{html.escape(str(diagnostics.get('category') or ''))}</div>
      </div>
      <a class="nav-link" href="../index.html">Back to overview</a>
    </div>
    <div class="grid metric-grid">{cards}</div>
    <div class="two-col">
      <div class="panel">
        <div class="panel-title">Feature Definition</div>
        <div class="desc-block">{html.escape(str(diagnostics.get('description') or 'No description extracted from docs.'))}</div>
      </div>
      <div class="panel">
        <div class="panel-title">Leakage Check</div>
        <div class="note"><strong>{html.escape(str(diagnostics.get('leakage', {}).get('status', '')))}</strong>: {html.escape(str(diagnostics.get('leakage', {}).get('note', '')))}</div>
      </div>
    </div>
    <div class="panel">
      <div class="panel-title">Key Issues</div>
      {issues_html(diagnostics.get('issues', []))}
    </div>
    {plot_html}
    <div class="two-col">
      <div class="panel">
        <div class="panel-title">Numeric Statistics</div>
        {_table_html(['metric', 'value'], stat_rows)}
      </div>
      <div class="panel">
        <div class="panel-title">Completeness</div>
        {_table_html(['metric', 'value'], missing_rows)}
      </div>
    </div>
    <div class="two-col">
      <div class="panel">
        <div class="panel-title">Outlier Diagnostics</div>
        {_table_html(['metric', 'value'], outlier_rows)}
      </div>
      <div class="panel">
        <div class="panel-title">Relationship Diagnostics</div>
        {_table_html(['metric', 'value'], rel_rows)}
      </div>
    </div>
    <div class="panel">
      <div class="panel-title">Flagged Extreme Time Points</div>
      {_table_html(['time', 'value', 'zscore', 'reason'], example_rows)}
    </div>
  </div>
  <script src="../assets/js/plotly.min.js"></script>
  <script src="../assets/js/table.js"></script>
</body>
</html>"""
    out_path.write_text(html_text, encoding='utf-8')
    return out_path


def _first_future_return_col(df: pd.DataFrame) -> Optional[str]:
    cols = [c for c in df.columns if c.startswith('future_return_')]
    if not cols:
        return None
    def key(col: str) -> int:
        match = re.search(r'(\d+)bar', col)
        return int(match.group(1)) if match else 999999
    return sorted(cols, key=key)[0]


def _feature_filename(feature: str) -> str:
    safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', feature).strip('_')
    return f'{safe or "feature"}.html'


def _table_html(headers: List[str], rows: List[List[Any]], *, table_id: str | None = None) -> str:
    table_attr = f' id="{html.escape(table_id)}"' if table_id else ''
    head = ''.join(f'<th>{html.escape(str(h))}</th>' for h in headers)
    body = []
    for row in rows:
        body.append('<tr>' + ''.join(f'<td>{html.escape(format_value(v))}</td>' for v in row) + '</tr>')
    return f'<div class="table-wrap"><table{table_attr} data-sortable="true"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _json_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        out = float(value)
        return out if np.isfinite(out) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if value is pd.NaT:
        return None
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')

