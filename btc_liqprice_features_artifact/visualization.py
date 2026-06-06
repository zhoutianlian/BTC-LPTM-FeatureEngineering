from __future__ import annotations

import html
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

try:
    from .features import OUTPUT_FEATURE_COLUMNS
except ImportError:  # pragma: no cover - kept for direct script execution
    from features import OUTPUT_FEATURE_COLUMNS

DARK_BG = '#050816'
PANEL_BG = '#0B1020'
GRID = '#24324A'
TEXT = '#DCE7FF'
SUBTEXT = '#8DA2C0'
COLORS = {
    'series': '#00D1FF',
    'series_alt': '#7DF9FF',
    'mean': '#FFB703',
    'median': '#00F5D4',
    'std': '#C77DFF',
    'q05': '#F72585',
    'q95': '#4CC9F0',
    'hist': '#9D4EDD',
    'box': '#80FF72',
    'outlier_high': '#FFD166',
    'outlier_low': '#EF476F',
    'zero': '#94A3B8',
}



def load_feature_frame(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == '.parquet':
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    if 'time' not in df.columns:
        raise ValueError("Feature file must contain a 'time' column.")
    df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
    df = df.dropna(subset=['time']).sort_values('time').reset_index(drop=True)
    return df



def _infer_bar_minutes(df: pd.DataFrame) -> int:
    if 'time' not in df.columns or len(df) < 3:
        return 1
    diffs = pd.to_datetime(df['time'], utc=True, errors='coerce').diff().dropna().dt.total_seconds().to_numpy(dtype=float)
    if len(diffs) == 0:
        return 1
    return max(int(round(float(np.median(diffs)) / 60.0)), 1)



def summarize_series(s: pd.Series) -> Dict[str, float]:
    x = pd.to_numeric(s, errors='coerce').astype(float)
    finite = np.isfinite(x)
    x_fin = x[finite]
    if len(x_fin) == 0:
        return {
            'count': 0,
            'non_na_ratio': float(x.notna().mean()),
            'finite_ratio': 0.0,
            'zero_ratio': np.nan,
            'mean': np.nan,
            'std': np.nan,
            'median': np.nan,
            'skew': np.nan,
            'kurtosis': np.nan,
            'p005': np.nan,
            'p01': np.nan,
            'p05': np.nan,
            'p95': np.nan,
            'p99': np.nan,
            'p995': np.nan,
            'min': np.nan,
            'max': np.nan,
        }
    return {
        'count': int(len(x_fin)),
        'non_na_ratio': float(x.notna().mean()),
        'finite_ratio': float(finite.mean()),
        'zero_ratio': float(np.isclose(x_fin, 0.0).mean()),
        'mean': float(x_fin.mean()),
        'std': float(x_fin.std(ddof=1)) if len(x_fin) > 1 else 0.0,
        'median': float(x_fin.median()),
        'skew': float(x_fin.skew()) if len(x_fin) > 2 else np.nan,
        'kurtosis': float(x_fin.kurtosis()) if len(x_fin) > 3 else np.nan,
        'p005': float(x_fin.quantile(0.005)),
        'p01': float(x_fin.quantile(0.01)),
        'p05': float(x_fin.quantile(0.05)),
        'p95': float(x_fin.quantile(0.95)),
        'p99': float(x_fin.quantile(0.99)),
        'p995': float(x_fin.quantile(0.995)),
        'min': float(x_fin.min()),
        'max': float(x_fin.max()),
    }



def summarize_feature_frame(df: pd.DataFrame, feature_columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    cols = [c for c in (feature_columns or OUTPUT_FEATURE_COLUMNS) if c in df.columns]
    rows = []
    for c in cols:
        row = {'feature': c}
        row.update(summarize_series(df[c]))
        rows.append(row)
    return pd.DataFrame(rows)



def _rolling_window_bars(df: pd.DataFrame, window_minutes: int = 24 * 60) -> int:
    return max(int(round(window_minutes / _infer_bar_minutes(df))), 3)



def _extreme_points(df: pd.DataFrame, feature: str, top_n: int = 20) -> pd.DataFrame:
    tmp = df[['time', feature]].copy()
    tmp[feature] = pd.to_numeric(tmp[feature], errors='coerce').astype(float)
    tmp = tmp.dropna(subset=[feature]).copy()
    if tmp.empty:
        return tmp
    tmp['abs_value'] = tmp[feature].abs()
    tmp = tmp.sort_values('abs_value', ascending=False).head(top_n)
    tmp['time'] = pd.to_datetime(tmp['time'], utc=True, errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S UTC')
    return tmp[['time', feature, 'abs_value']]



def _robust_bounds(y: pd.Series, stats: Dict[str, float]) -> tuple[float, float]:
    finite = pd.to_numeric(y, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return -1.0, 1.0
    low = float(stats['p005'])
    high = float(stats['p995'])
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        low = float(finite.min())
        high = float(finite.max())
    span = high - low
    if span <= 0:
        center = float(finite.median())
        pad = max(abs(center) * 0.1, 1e-6)
        return center - pad, center + pad
    pad = span * 0.08
    return low - pad, high + pad



def _summary_cards(stats: Dict[str, float], robust_low: float, robust_high: float, outlier_low_count: int, outlier_high_count: int) -> str:
    items = [
        ('样本数', f"{stats['count']}"),
        ('均值', f"{stats['mean']:.6f}"),
        ('标准差', f"{stats['std']:.6f}"),
        ('中位数', f"{stats['median']:.6f}"),
        ('P01 / P99', f"{stats['p01']:.6f} / {stats['p99']:.6f}"),
        ('稳健显示区间', f'{robust_low:.6f} ~ {robust_high:.6f}'),
        ('低端超界点数', str(outlier_low_count)),
        ('高端超界点数', str(outlier_high_count)),
    ]
    cards = []
    for title, value in items:
        cards.append(
            '<div class="metric-card">'
            f'<div class="metric-title">{html.escape(title)}</div>'
            f'<div class="metric-value">{html.escape(value)}</div>'
            '</div>'
        )
    return ''.join(cards)



def _page_filename(feature: str) -> str:
    return f'{feature}.html'



def make_feature_figure(df: pd.DataFrame, feature: str, rolling_window_minutes: int = 24 * 60) -> tuple[go.Figure, Dict[str, float], Dict[str, int]]:
    if feature not in df.columns:
        raise KeyError(feature)

    x = pd.to_datetime(df['time'], utc=True, errors='coerce')
    y = pd.to_numeric(df[feature], errors='coerce').astype(float)
    stats = summarize_series(y)

    roll_k = _rolling_window_bars(df, rolling_window_minutes)
    min_periods = max(5, roll_k // 4)
    roll_mean = y.rolling(roll_k, min_periods=min_periods).mean()
    roll_median = y.rolling(roll_k, min_periods=min_periods).median()
    roll_std = y.rolling(roll_k, min_periods=min_periods).std()
    q05 = y.rolling(roll_k, min_periods=min_periods).quantile(0.05)
    q95 = y.rolling(roll_k, min_periods=min_periods).quantile(0.95)

    robust_low, robust_high = _robust_bounds(y, stats)
    clipped = y.clip(lower=robust_low, upper=robust_high)
    high_mask = y > robust_high
    low_mask = y < robust_low
    outlier_counts = {
        'high': int(high_mask.fillna(False).sum()),
        'low': int(low_mask.fillna(False).sum()),
    }

    extremes = _extreme_points(df, feature, top_n=20)

    fig = make_subplots(
        rows=5,
        cols=2,
        shared_xaxes=True,
        specs=[
            [{'colspan': 2}, None],
            [{'colspan': 2}, None],
            [{'colspan': 2}, None],
            [{}, {}],
            [{'colspan': 2, 'type': 'table'}, None],
        ],
        row_heights=[0.26, 0.18, 0.22, 0.16, 0.18],
        subplot_titles=(
            f'{feature} 主图（稳健缩放，便于查看主体结构）',
            f'{feature} 原始全尺度时间序列',
            f'{feature} 滚动统计（窗口={roll_k} bars）',
            f'{feature} 分布直方图',
            f'{feature} 箱线图',
            f'{feature} 极值定位',
        ),
        vertical_spacing=0.06,
        horizontal_spacing=0.08,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=clipped,
            mode='lines',
            name=f'{feature}_robust',
            line=dict(color=COLORS['series'], width=1.2),
            hovertemplate='%{x}<br>显示值=%{y:.6f}<extra></extra>',
        ),
        row=1,
        col=1,
    )
    if outlier_counts['high'] > 0:
        fig.add_trace(
            go.Scatter(
                x=x[high_mask.fillna(False)],
                y=np.full(outlier_counts['high'], robust_high),
                mode='markers',
                name='high_outlier_clipped',
                marker=dict(color=COLORS['outlier_high'], size=5, symbol='triangle-up'),
                customdata=np.stack([y[high_mask.fillna(False)].to_numpy(dtype=float)], axis=-1),
                hovertemplate='%{x}<br>真实值=%{customdata[0]:.6f}<br>显示边界=robust_high<extra></extra>',
            ),
            row=1,
            col=1,
        )
    if outlier_counts['low'] > 0:
        fig.add_trace(
            go.Scatter(
                x=x[low_mask.fillna(False)],
                y=np.full(outlier_counts['low'], robust_low),
                mode='markers',
                name='low_outlier_clipped',
                marker=dict(color=COLORS['outlier_low'], size=5, symbol='triangle-down'),
                customdata=np.stack([y[low_mask.fillna(False)].to_numpy(dtype=float)], axis=-1),
                hovertemplate='%{x}<br>真实值=%{customdata[0]:.6f}<br>显示边界=robust_low<extra></extra>',
            ),
            row=1,
            col=1,
        )
    if np.isfinite(stats['median']):
        fig.add_hline(y=stats['median'], line_color=COLORS['median'], line_dash='dot', opacity=0.7, row=1, col=1)
    if np.isfinite(stats['min']) and stats['min'] <= 0.0 <= stats['max']:
        fig.add_hline(y=0.0, line_color=COLORS['zero'], line_dash='dash', opacity=0.5, row=1, col=1)
    fig.update_yaxes(range=[robust_low, robust_high], row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name=feature,
            line=dict(color=COLORS['series_alt'], width=1.0),
            hovertemplate='%{x}<br>%{y:.6f}<extra></extra>',
        ),
        row=2,
        col=1,
    )
    if np.isfinite(stats['min']) and stats['min'] <= 0.0 <= stats['max']:
        fig.add_hline(y=0.0, line_color=COLORS['zero'], line_dash='dash', opacity=0.4, row=2, col=1)

    fig.add_trace(go.Scatter(x=x, y=roll_mean, mode='lines', name='rolling_mean', line=dict(color=COLORS['mean'], width=1.3)), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=roll_median, mode='lines', name='rolling_median', line=dict(color=COLORS['median'], width=1.1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=roll_std, mode='lines', name='rolling_std', line=dict(color=COLORS['std'], width=1.1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=q05, mode='lines', name='rolling_q05', line=dict(color=COLORS['q05'], width=1.0, dash='dot')), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=q95, mode='lines', name='rolling_q95', line=dict(color=COLORS['q95'], width=1.0, dash='dot')), row=3, col=1)

    hist_x = y.replace([np.inf, -np.inf], np.nan).dropna()
    fig.add_trace(
        go.Histogram(
            x=hist_x,
            name='histogram',
            nbinsx=100,
            marker=dict(color=COLORS['hist'], line=dict(color='#C084FC', width=0.5)),
            opacity=0.85,
            hovertemplate='bin=%{x}<br>count=%{y}<extra></extra>',
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Box(
            y=hist_x,
            name='boxplot',
            boxpoints='suspectedoutliers',
            marker=dict(color=COLORS['box']),
            line=dict(color=COLORS['box']),
            fillcolor='rgba(128,255,114,0.15)',
            hovertemplate='%{y:.6f}<extra></extra>',
        ),
        row=4,
        col=2,
    )

    if extremes.empty:
        table_values = [['暂无数据'], [''], ['']]
    else:
        table_values = [
            extremes['time'].tolist(),
            np.round(extremes[feature], 6).tolist(),
            np.round(extremes['abs_value'], 6).tolist(),
        ]
    fig.add_trace(
        go.Table(
            header=dict(
                values=['time', feature, 'abs_value'],
                fill_color='#10182B',
                line_color='#20304A',
                font=dict(color=TEXT, size=12),
                align='left',
            ),
            cells=dict(
                values=table_values,
                fill_color='#0B1020',
                line_color='#18253A',
                font=dict(color=TEXT, size=11),
                align='left',
                height=28,
            ),
        ),
        row=5,
        col=1,
    )

    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=1, label='1D', step='day', stepmode='backward'),
                dict(count=3, label='3D', step='day', stepmode='backward'),
                dict(count=7, label='7D', step='day', stepmode='backward'),
                dict(count=30, label='30D', step='day', stepmode='backward'),
                dict(count=90, label='90D', step='day', stepmode='backward'),
                dict(step='all', label='ALL'),
            ],
            bgcolor='#0F172A',
            activecolor='#1D4ED8',
            font=dict(color=TEXT),
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(rangeslider_visible=True, row=2, col=1)

    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)

    title_text = (
        f'{feature} 统计学诊断 | '
        f'count={stats["count"]}, mean={stats["mean"]:.6f}, std={stats["std"]:.6f}, '
        f'median={stats["median"]:.6f}, p01={stats["p01"]:.6f}, p99={stats["p99"]:.6f}'
    )
    fig.update_layout(
        title=dict(text=title_text, x=0.5, font=dict(size=28, color=TEXT)),
        hovermode='x unified',
        height=1900,
        template='plotly_dark',
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT, family='Inter, Arial, Helvetica, sans-serif', size=12),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0.0,
            bgcolor='rgba(5,8,22,0.65)',
            bordercolor='#20304A',
            borderwidth=1,
            font=dict(color=TEXT),
        ),
        margin=dict(l=70, r=40, t=120, b=60),
        bargap=0.05,
    )
    return fig, stats, outlier_counts



def _ensure_plotly_bundle(out_dir: Path) -> Path:
    bundle_path = out_dir / 'plotly.min.js'
    if not bundle_path.exists():
        bundle_path.write_text(get_plotlyjs(), encoding='utf-8')
    return bundle_path


def write_feature_page(df: pd.DataFrame, feature: str, out_dir: str | Path, rolling_window_minutes: int = 24 * 60) -> str:
    fig, stats, outlier_counts = make_feature_figure(df, feature, rolling_window_minutes=rolling_window_minutes)
    robust_low, robust_high = _robust_bounds(pd.to_numeric(df[feature], errors='coerce').astype(float), stats)
    cards = _summary_cards(stats, robust_low, robust_high, outlier_counts['low'], outlier_counts['high'])
    out_path = Path(out_dir) / _page_filename(feature)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path = _ensure_plotly_bundle(out_path.parent)
    plot_html = fig.to_html(full_html=False, include_plotlyjs=False)
    page_html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(feature)} 统计学诊断</title>
  <style>
    body {{ margin: 0; background: {DARK_BG}; color: {TEXT}; font-family: Inter, Arial, Helvetica, sans-serif; }}
    .page {{ max-width: 1800px; margin: 0 auto; padding: 20px 24px 28px 24px; }}
    .topbar {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }}
    .title {{ font-size: 28px; font-weight: 700; letter-spacing: 0.3px; }}
    .back-link {{ color: #8BCBFF; text-decoration: none; font-size: 14px; }}
    .desc {{ color: {SUBTEXT}; margin: 10px 0 18px 0; line-height: 1.6; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 18px; }}
    .metric-card {{ background: linear-gradient(180deg, #0E1629 0%, #09101F 100%); border: 1px solid #1D2A42; border-radius: 14px; padding: 14px 16px; box-shadow: 0 0 0 1px rgba(0,209,255,0.02) inset, 0 12px 32px rgba(0,0,0,0.25); }}
    .metric-title {{ color: {SUBTEXT}; font-size: 12px; margin-bottom: 8px; letter-spacing: 0.2px; }}
    .metric-value {{ color: {TEXT}; font-size: 18px; font-weight: 700; }}
    .note {{ margin-top: 12px; color: {SUBTEXT}; line-height: 1.7; }}
    @media (max-width: 1200px) {{ .metric-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
    @media (max-width: 720px) {{ .metric-grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="page">
    <div class="topbar">
      <div class="title">{html.escape(feature)} 交互式统计学诊断</div>
      <a class="back-link" href="index.html">返回总览页</a>
    </div>
    <div class="desc">本页提供稳健缩放主图、原始全尺度时间序列、滚动统计、分布图与极值定位表。主图默认按稳健分位区间显示主体结构，避免少数极端值压扁大部分样本；原始全尺度图保留全部真实振幅，便于识别初始化异常、脉冲尖峰与结构性断层。</div>
    <div class="metric-grid">{cards}</div>
    <script src="{html.escape(bundle_path.name)}"></script>
    {plot_html}
    <div class="note">判读建议：先看主图确认主体波动结构，再看原始全尺度图确认极端值是否集中在少数时点；随后结合滚动统计检查分布漂移与波动聚集，最后用极值定位表回到具体时间段排查数据或算法问题。</div>
  </div>
</body>
</html>'''
    out_path.write_text(page_html, encoding='utf-8')
    return str(out_path)



def write_overview_page(df: pd.DataFrame, out_dir: str | Path, feature_columns: Optional[Sequence[str]] = None) -> str:
    out_path = Path(out_dir) / 'index.html'
    stats_df = summarize_feature_frame(df, feature_columns=feature_columns)

    rows = []
    for _, row in stats_df.iterrows():
        feature = str(row['feature'])
        link = html.escape(_page_filename(feature))
        rows.append(
            '<tr>'
            f'<td><a href="{link}">{html.escape(feature)}</a></td>'
            f'<td>{int(row["count"])}</td>'
            f'<td>{row["mean"]:.6f}</td>'
            f'<td>{row["std"]:.6f}</td>'
            f'<td>{row["median"]:.6f}</td>'
            f'<td>{row["p01"]:.6f}</td>'
            f'<td>{row["p99"]:.6f}</td>'
            f'<td>{row["min"]:.6f}</td>'
            f'<td>{row["max"]:.6f}</td>'
            '</tr>'
        )

    html_text = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>BTC 清算—价格特征总览</title>
  <style>
    body {{ margin: 0; background: {DARK_BG}; color: {TEXT}; font-family: Inter, Arial, Helvetica, sans-serif; }}
    .page {{ max-width: 1600px; margin: 0 auto; padding: 28px 24px 36px 24px; }}
    .hero {{ display: flex; justify-content: space-between; align-items: end; gap: 24px; margin-bottom: 20px; }}
    .title {{ font-size: 34px; font-weight: 800; letter-spacing: 0.4px; }}
    .subtitle {{ color: {SUBTEXT}; line-height: 1.7; max-width: 980px; margin-top: 8px; }}
    .tag {{ display: inline-block; margin-top: 10px; padding: 6px 12px; border-radius: 999px; background: rgba(0,209,255,0.12); color: #8BD9FF; border: 1px solid rgba(0,209,255,0.18); font-size: 12px; }}
    .table-wrap {{ background: linear-gradient(180deg, #0E1629 0%, #09101F 100%); border: 1px solid #1D2A42; border-radius: 16px; padding: 12px; overflow-x: auto; box-shadow: 0 12px 32px rgba(0,0,0,0.25); }}
    table {{ border-collapse: collapse; width: 100%; min-width: 980px; }}
    th, td {{ padding: 12px 10px; border-bottom: 1px solid #18253A; text-align: right; white-space: nowrap; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ color: #BFD2F4; font-size: 12px; text-transform: uppercase; letter-spacing: 0.4px; }}
    tr:hover td {{ background: rgba(0,209,255,0.04); }}
    a {{ color: #8BCBFF; text-decoration: none; font-weight: 600; }}
    .note {{ margin-top: 16px; color: {SUBTEXT}; line-height: 1.7; }}
  </style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <div>
        <div class="title">BTC 清算—价格特征交互式总览</div>
        <div class="subtitle">点击特征名进入单独的交互式统计诊断页面。每个页面都提供稳健缩放主图、原始全尺度时间序列、滚动统计、分布图、箱线图与极值定位表，并支持鼠标悬停、时间段切换与范围滑块。</div>
        <div class="tag">黑底亮色 · 可交互 · 面向问题定位</div>
      </div>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>feature</th>
            <th>count</th>
            <th>mean</th>
            <th>std</th>
            <th>median</th>
            <th>p01</th>
            <th>p99</th>
            <th>min</th>
            <th>max</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </div>
    <div class="note">建议的检查顺序：先看 spike 是否在关键事件期显著抬升；再看 velocity 是否与事件放大阶段同向；随后核对 acceleration 是否主要反映形态拐点而非噪声；最后看 trend_pressure、kalman_slope、vol_adaptive 是否给出一致的价格语境与风险背景。</div>
  </div>
</body>
</html>'''
    out_path.write_text(html_text, encoding='utf-8')
    return str(out_path)



def write_feature_report_pages(
    df: pd.DataFrame,
    out_dir: str | Path,
    *,
    feature_columns: Optional[Sequence[str]] = None,
    rolling_window_minutes: int = 24 * 60,
) -> List[str]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cols = [c for c in (feature_columns or OUTPUT_FEATURE_COLUMNS) if c in df.columns]
    created: List[str] = []
    for feature in cols:
        created.append(write_feature_page(df, feature, out_dir=out_path, rolling_window_minutes=rolling_window_minutes))

    summary_csv = out_path / 'feature_summary.csv'
    summarize_feature_frame(df, feature_columns=cols).to_csv(summary_csv, index=False)
    created.append(str(summary_csv))
    created.append(write_overview_page(df, out_dir=out_path, feature_columns=cols))
    return created
