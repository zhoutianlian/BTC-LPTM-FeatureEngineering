from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from ..config import ReportConfig
except ImportError:  # pragma: no cover
    from config import ReportConfig


COLORS = {
    'cyan': '#00D1FF',
    'green': '#39FF88',
    'orange': '#FFB000',
    'purple': '#B86BFF',
    'pink': '#FF4D8D',
    'red': '#FF3B5C',
    'blue': '#4C8DFF',
    'muted': '#8DA2C0',
    'grid': '#24324A',
    'panel': '#0B1020',
    'paper': '#050816',
    'text': '#DCE7FF',
}


PLOT_CONFIG = {
    'displaylogo': False,
    'responsive': True,
    'scrollZoom': True,
}


def figure_html(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, config=PLOT_CONFIG)


def _trace_cls(n: int):
    return go.Scattergl if n > 5000 else go.Scatter


def _base_layout(fig: go.Figure, title: str, height: int) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.01, font=dict(size=20, color=COLORS['text'])),
        template='plotly_dark',
        height=height,
        paper_bgcolor=COLORS['paper'],
        plot_bgcolor=COLORS['panel'],
        font=dict(color=COLORS['text'], family='Inter, Arial, Helvetica, sans-serif', size=12),
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0,
            bgcolor='rgba(5,8,22,0.72)',
            bordercolor='#20304A',
            borderwidth=1,
        ),
        margin=dict(l=58, r=32, t=76, b=42),
    )
    fig.update_xaxes(showgrid=True, gridcolor=COLORS['grid'], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=COLORS['grid'], zeroline=False, autorange=True)
    return fig


def _range_selector() -> Dict[str, Any]:
    return {
        'buttons': [
            dict(count=1, label='1D', step='day', stepmode='backward'),
            dict(count=7, label='1W', step='day', stepmode='backward'),
            dict(count=1, label='1M', step='month', stepmode='backward'),
            dict(count=3, label='3M', step='month', stepmode='backward'),
            dict(count=6, label='6M', step='month', stepmode='backward'),
            dict(count=1, label='1Y', step='year', stepmode='backward'),
            dict(step='all', label='ALL'),
        ],
        'bgcolor': '#0F172A',
        'activecolor': '#1D4ED8',
        'font': dict(color=COLORS['text']),
    }


def make_time_series_figure(df: pd.DataFrame, feature: str, diagnostics: Dict[str, Any]) -> go.Figure:
    x = pd.to_datetime(df['time'], utc=True, errors='coerce')
    y = pd.to_numeric(df[feature], errors='coerce').astype(float)
    trace = _trace_cls(len(df))
    fig = go.Figure()
    fig.add_trace(
        trace(
            x=x,
            y=y,
            mode='lines',
            name=feature,
            line=dict(color=COLORS['cyan'], width=1.25),
            connectgaps=False,
            hovertemplate='%{x}<br>value=%{y:.8g}<extra></extra>',
        )
    )

    examples = diagnostics.get('outliers', {}).get('examples', [])
    if examples:
        times = [item.get('time') for item in examples]
        values = [item.get('value') for item in examples]
        reasons = [item.get('reason') for item in examples]
        fig.add_trace(
            go.Scatter(
                x=times,
                y=values,
                mode='markers',
                name='flagged_outliers',
                marker=dict(color=COLORS['orange'], size=8, symbol='diamond', line=dict(color='#FFFFFF', width=0.5)),
                text=reasons,
                hovertemplate='%{x}<br>value=%{y:.8g}<br>reason=%{text}<extra></extra>',
            )
        )

    missing_mask = y.isna() | np.isinf(y)
    if bool(missing_mask.any()):
        finite = y[np.isfinite(y)]
        marker_y = float(finite.min()) if len(finite) else 0.0
        fig.add_trace(
            go.Scatter(
                x=x[missing_mask],
                y=np.full(int(missing_mask.sum()), marker_y),
                mode='markers',
                name='missing_or_inf',
                marker=dict(color=COLORS['red'], size=6, symbol='x'),
                hovertemplate='%{x}<br>missing/inf<extra></extra>',
            )
        )

    fig.update_xaxes(rangeselector=_range_selector(), rangeslider=dict(visible=True), type='date')
    return _base_layout(fig, f'{feature} time series with anomaly markers', 620)


def _kde_curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = values[np.isfinite(values)]
    if values.size < 5:
        return np.array([]), np.array([])
    if values.size > 5000:
        idx = np.linspace(0, values.size - 1, 5000).astype(int)
        values = np.sort(values)[idx]
    std = float(values.std(ddof=1))
    if std <= 0:
        return np.array([]), np.array([])
    bw = 1.06 * std * (values.size ** (-1 / 5))
    if bw <= 0:
        return np.array([]), np.array([])
    low, high = np.nanquantile(values, [0.005, 0.995])
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        low, high = float(values.min()), float(values.max())
    grid = np.linspace(low, high, 220)
    z = (grid[:, None] - values[None, :]) / bw
    density = np.exp(-0.5 * z * z).mean(axis=1) / (bw * np.sqrt(2.0 * np.pi))
    return grid, density


def make_distribution_figure(df: pd.DataFrame, feature: str, diagnostics: Dict[str, Any]) -> go.Figure:
    y = pd.to_numeric(df[feature], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Histogram and density', 'Box plot'), column_widths=[0.68, 0.32])
    fig.add_trace(
        go.Histogram(
            x=y,
            name='histogram',
            nbinsx=120,
            histnorm='probability density',
            marker=dict(color=COLORS['purple'], line=dict(color='#D8B4FE', width=0.4)),
            opacity=0.82,
        ),
        row=1,
        col=1,
    )
    grid, density = _kde_curve(y.to_numpy(dtype=float))
    if grid.size:
        fig.add_trace(go.Scatter(x=grid, y=density, mode='lines', name='kde_approx', line=dict(color=COLORS['green'], width=2)), row=1, col=1)

    stats = diagnostics.get('stats', {})
    markers = [
        ('mean', stats.get('mean'), COLORS['orange']),
        ('median', stats.get('median'), COLORS['cyan']),
        ('p01', stats.get('p01'), COLORS['pink']),
        ('p99', stats.get('p99'), COLORS['pink']),
    ]
    for name, value, color in markers:
        if value is not None and np.isfinite(float(value)):
            fig.add_vline(x=float(value), line_color=color, line_dash='dot', opacity=0.9, row=1, col=1)
            fig.add_annotation(x=float(value), y=1.02, yref='paper', text=name, showarrow=False, font=dict(color=color, size=11), row=1, col=1)

    fig.add_trace(
        go.Box(
            y=y,
            name=feature,
            boxpoints='suspectedoutliers',
            marker=dict(color=COLORS['green']),
            line=dict(color=COLORS['green']),
            fillcolor='rgba(57,255,136,0.15)',
        ),
        row=1,
        col=2,
    )
    return _base_layout(fig, f'{feature} distribution diagnostics', 520)


def _rolling_window_bars(df: pd.DataFrame, window_minutes: int) -> int:
    t = pd.to_datetime(df['time'], utc=True, errors='coerce')
    diffs = t.diff().dropna().dt.total_seconds()
    if diffs.empty:
        return max(int(window_minutes), 3)
    median_min = max(float(diffs.median()) / 60.0, 1.0)
    return max(int(round(window_minutes / median_min)), 3)


def make_rolling_figure(df: pd.DataFrame, feature: str, cfg: ReportConfig) -> go.Figure:
    x = pd.to_datetime(df['time'], utc=True, errors='coerce')
    y = pd.to_numeric(df[feature], errors='coerce').astype(float)
    k = _rolling_window_bars(df, int(cfg.rolling_window_minutes))
    min_periods = max(3, min(k, k // 4))
    roll_mean = y.rolling(k, min_periods=min_periods).mean()
    roll_std = y.rolling(k, min_periods=min_periods).std()
    roll_min = y.rolling(k, min_periods=min_periods).min()
    roll_max = y.rolling(k, min_periods=min_periods).max()
    q_low, q_high = [float(q) for q in cfg.rolling_quantiles[:2]]
    roll_q_low = y.rolling(k, min_periods=min_periods).quantile(q_low)
    roll_q_high = y.rolling(k, min_periods=min_periods).quantile(q_high)

    trace = _trace_cls(len(df))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.08)
    fig.add_trace(trace(x=x, y=y, mode='lines', name='raw', line=dict(color=COLORS['cyan'], width=0.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=roll_mean, mode='lines', name=f'rolling_mean_{k}', line=dict(color=COLORS['orange'], width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=roll_q_low, mode='lines', name=f'rolling_q{q_low:g}', line=dict(color=COLORS['pink'], width=1.0, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=roll_q_high, mode='lines', name=f'rolling_q{q_high:g}', line=dict(color=COLORS['green'], width=1.0, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=roll_min, mode='lines', name='rolling_min', line=dict(color=COLORS['muted'], width=0.8, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=roll_max, mode='lines', name='rolling_max', line=dict(color=COLORS['muted'], width=0.8, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=roll_std, mode='lines', name=f'rolling_std_{k}', line=dict(color=COLORS['purple'], width=1.5)), row=2, col=1)
    fig.update_xaxes(rangeselector=_range_selector(), rangeslider=dict(visible=True), type='date', row=2, col=1)
    return _base_layout(fig, f'{feature} rolling stability diagnostics | window={k} bars', 680)


def make_missing_figure(df: pd.DataFrame, feature: str, cfg: ReportConfig) -> go.Figure:
    x = pd.to_datetime(df['time'], utc=True, errors='coerce')
    y = pd.to_numeric(df[feature], errors='coerce')
    missing = (y.isna() | np.isinf(y)).astype(int)
    k = _rolling_window_bars(df, int(cfg.rolling_window_minutes))
    miss_roll = missing.rolling(k, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=missing, name='missing_or_inf_flag', marker=dict(color=COLORS['red']), opacity=0.55))
    fig.add_trace(go.Scatter(x=x, y=miss_roll, mode='lines', name=f'rolling_missing_ratio_{k}', line=dict(color=COLORS['orange'], width=1.6)))
    fig.update_xaxes(rangeselector=_range_selector(), rangeslider=dict(visible=True), type='date')
    fig.update_yaxes(range=[-0.02, 1.05])
    return _base_layout(fig, f'{feature} missing value timeline', 430)


def make_relationship_figure(df: pd.DataFrame, feature: str, future_return_col: Optional[str] = None) -> Optional[go.Figure]:
    if 'price' not in df.columns:
        return None
    cols = [feature, 'price', 'return_1bar']
    if future_return_col and future_return_col in df.columns:
        cols.append(future_return_col)
    if any(c not in df.columns for c in cols):
        return None

    x = pd.to_datetime(df['time'], utc=True, errors='coerce')
    y = pd.to_numeric(df[feature], errors='coerce').astype(float)
    price = pd.to_numeric(df['price'], errors='coerce').astype(float)
    ret = pd.to_numeric(df['return_1bar'], errors='coerce').astype(float)
    future = pd.to_numeric(df[future_return_col], errors='coerce').astype(float) if future_return_col else None
    trace = _trace_cls(len(df))
    rows = 3 if future is not None else 2
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=False,
        specs=[[{'secondary_y': True}], [{}]] + ([[{}]] if future is not None else []),
        row_heights=[0.46, 0.27, 0.27] if future is not None else [0.56, 0.44],
        vertical_spacing=0.10,
        subplot_titles=('Feature vs price over time', 'Feature vs current return', 'Feature vs future return' if future is not None else ''),
    )
    fig.add_trace(trace(x=x, y=y, mode='lines', name=feature, line=dict(color=COLORS['cyan'], width=1.0)), row=1, col=1, secondary_y=False)
    fig.add_trace(trace(x=x, y=price, mode='lines', name='price', line=dict(color=COLORS['orange'], width=1.0)), row=1, col=1, secondary_y=True)
    fig.add_trace(
        go.Scattergl(x=y, y=ret, mode='markers', name='feature_vs_return_1bar', marker=dict(color=COLORS['green'], size=4, opacity=0.36)),
        row=2,
        col=1,
    )
    if future is not None:
        fig.add_trace(
            go.Scattergl(x=y, y=future, mode='markers', name=f'feature_vs_{future_return_col}', marker=dict(color=COLORS['pink'], size=4, opacity=0.36)),
            row=3,
            col=1,
        )
    return _base_layout(fig, f'{feature} relationship diagnostics', 760 if future is not None else 620)


def make_correlation_heatmap(corr: pd.DataFrame, title: str = 'Correlation heatmap') -> Optional[go.Figure]:
    if corr.empty:
        return None
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.to_numpy(dtype=float),
            x=list(corr.columns),
            y=list(corr.index),
            colorscale=[
                [0.0, '#FF3B5C'],
                [0.5, '#101827'],
                [1.0, '#39FF88'],
            ],
            zmin=-1,
            zmax=1,
            colorbar=dict(title='corr'),
            hovertemplate='x=%{x}<br>y=%{y}<br>corr=%{z:.4f}<extra></extra>',
        )
    )
    return _base_layout(fig, title, max(540, 35 * len(corr.index) + 220))
