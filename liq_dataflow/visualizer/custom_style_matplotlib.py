from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


BIN_COLORS = {
    0: '#cc0000',
    1: '#e06666',
    2: '#f6b26b',
    3: '#ffd966',
    4: '#b6d7a8',
    5: '#6aa84f',
    6: '#45818e',
    7: '#3d85c6',
    8: '#a64d79',
}

CASE_COLORS = {
    'None': '#38761d',
    'LUSUL': '#38761d',
    'LUSUS': '#cc0000',
    'LDSDL': '#6aa84f',
    'LDSDS': '#e06666',
    'LUSDL': '#93c47d',
    'LUSDS': '#ea9999',
    'LDSUL': '#93c47d',
    'LDSUS': '#ea9999',
}

PRICE_COLOR = '#f1c232'
RPN_AXIS_COLOR = '#3d85c6'
FLL_COLOR = '#f44336'
FSL_COLOR = '#8fce00'
FLL_FILL = '#ea9999'  # FLL dominant / downward pressure
FSL_FILL = '#b6d7a8'  # FSL dominant / upward pressure
EVENT_BUY = '#38761d'
EVENT_SELL = '#cc0000'


def set_time_param(duration_months: int | None = None) -> dict[str, object]:
    params: dict[str, object] = {
        'n_days': None,
        'locator_type': 'monthly',
        'interval': 1,
    }
    if duration_months == 1:
        params['n_days'] = 30
        params['locator_type'] = 'daily'
        params['interval'] = 1
    elif duration_months == 3:
        params['n_days'] = 90
        params['locator_type'] = 'daily'
        params['interval'] = 3
    elif duration_months == 12:
        params['n_days'] = 365
        params['locator_type'] = 'monthly'
        params['interval'] = 1
    return params


def filter_recent(df: pd.DataFrame, duration_months: int | None = None) -> pd.DataFrame:
    out = df.copy()
    out['time'] = pd.to_datetime(out['time'], errors='coerce')
    out = out.dropna(subset=['time']).sort_values('time').reset_index(drop=True)
    if out.empty or duration_months is None:
        return out
    cutoff = out['time'].max() - pd.DateOffset(months=duration_months)
    return out[out['time'] >= cutoff].reset_index(drop=True)


def _setup_style() -> None:
    plt.close('all')
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.figsize'] = (28, 16)
    plt.rcParams['font.size'] = 12


def _configure_time_axis(ax, locator_type: str, interval: int) -> None:
    if locator_type == 'daily':
        locator = mdates.DayLocator(interval=interval)
        formatter = mdates.DateFormatter('%Y-%m-%d')
    else:
        locator = mdates.MonthLocator(interval=interval)
        formatter = mdates.DateFormatter('%Y-%m')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def _add_price_fill(ax, df: pd.DataFrame) -> None:
    x = df['time'].to_numpy()
    y = pd.to_numeric(df['price'], errors='coerce').to_numpy(dtype=float)
    dom = pd.to_numeric(df['dominance'], errors='coerce').fillna(0).astype(int).to_numpy()
    for i in range(1, len(df)):
        if dom[i] == 1:
            ax.fill_between(x[i - 1:i + 1], y[i - 1:i + 1], color=FSL_FILL, alpha=0.10)
        elif dom[i] == -1:
            ax.fill_between(x[i - 1:i + 1], y[i - 1:i + 1], color=FLL_FILL, alpha=0.10)


def _add_rpn_fill(ax, df: pd.DataFrame) -> None:
    x = df['time'].to_numpy()
    y = pd.to_numeric(df['risk_priority_number'], errors='coerce').to_numpy(dtype=float)
    dom = pd.to_numeric(df['dominance'], errors='coerce').fillna(0).astype(int).to_numpy()
    ax.fill_between(x, y, where=(dom == 1), color=FSL_FILL, alpha=0.30, interpolate=True)
    ax.fill_between(x, y, where=(dom == -1), color=FLL_FILL, alpha=0.30, interpolate=True)


def _colored_line_segments(ax, x: np.ndarray, y: np.ndarray, color_values: np.ndarray) -> None:
    if len(x) < 2:
        return
    xn = mdates.date2num(pd.to_datetime(x).to_pydatetime())
    points = np.array([xn, y], dtype=float).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = [BIN_COLORS.get(int(v), '#999999') for v in color_values[:-1]]
    lc = LineCollection(segments, colors=colors, linewidths=1.3, linestyles=':')
    ax.add_collection(lc)
    ax.scatter(pd.to_datetime(x), y, c=[BIN_COLORS.get(int(v), '#999999') for v in color_values], s=7, alpha=0.85)


def _render_dominance_figure(df: pd.DataFrame, *, duration_months: int) -> tuple[plt.Figure, pd.Timestamp]:
    _setup_style()
    params = set_time_param(duration_months)
    data = filter_recent(df, duration_months)
    if data.empty:
        raise ValueError('No data available for dominance figure.')
    current_time = pd.to_datetime(data.iloc[-1]['time'])

    fig, axes = plt.subplots(2, 1, figsize=(28, 16), sharex=True)

    # Top: Price + RPN colored by bin + dominance fill
    ax1 = axes[0]
    ax1.set_xlabel('Time', color='#5b5b5b', fontsize=16, fontweight='bold')
    ax1.tick_params(axis='x', labelcolor='#5b5b5b', rotation=45)
    ax1.plot(data['time'], data['price'], color=PRICE_COLOR, label='Price', linewidth=1.2)
    ax1.set_ylabel('Price', color=PRICE_COLOR, fontsize=16, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=PRICE_COLOR)

    ax2 = ax1.twinx()
    _add_rpn_fill(ax2, data)
    _colored_line_segments(
        ax2,
        data['time'].to_numpy(),
        pd.to_numeric(data['risk_priority_number'], errors='coerce').to_numpy(dtype=float),
        pd.to_numeric(data['bin_index'], errors='coerce').fillna(4).astype(int).to_numpy(),
    )
    ax2.set_ylabel('risk_priority_number', color=RPN_AXIS_COLOR, fontsize=16, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=RPN_AXIS_COLOR)
    ax1.set_title(f'BTC Price Colored by Risk Priority Number Bins With Dominance - {current_time.strftime("%Y-%m-%d %H:%M")}', fontsize=24, fontweight='bold')
    ax1.legend(loc='upper left')
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=f'bin{i}', markerfacecolor=BIN_COLORS[i], markersize=8)
        for i in sorted(BIN_COLORS)
    ]
    legend_elements.append(Patch(facecolor=FLL_FILL, alpha=0.30, label='FLL Dominant'))
    legend_elements.append(Patch(facecolor=FSL_FILL, alpha=0.30, label='FSL Dominant'))
    ax2.legend(handles=legend_elements, title='Risk Priority Number Bins', bbox_to_anchor=(1.12, 1), loc='upper left', framealpha=0.9)

    # Bottom: Price + diff, colored by sign and filled by dominance
    bx1 = axes[1]
    bx1.set_xlabel('Time', color='#5b5b5b', fontsize=16, fontweight='bold')
    bx1.tick_params(axis='x', labelcolor='#5b5b5b', rotation=45)
    bx1.plot(data['time'], data['price'], color=PRICE_COLOR, label='Price', alpha=0.6, linewidth=1.0)
    _add_price_fill(bx1, data)
    bx1.set_ylabel('Price', color=PRICE_COLOR, fontsize=16, fontweight='bold')
    bx1.tick_params(axis='y', labelcolor=PRICE_COLOR)
    bx1.set_ylim(data['price'].min() - 500, data['price'].max() + 500)

    bx2 = bx1.twinx()
    diff = pd.to_numeric(data['diff_ls_cwt_kf'], errors='coerce')
    pos = diff >= 0
    neg = diff < 0
    bx2.scatter(data.loc[pos, 'time'], diff[pos], color=FLL_COLOR, label='FLL Dominant', alpha=0.5, s=4)
    bx2.scatter(data.loc[neg, 'time'], diff[neg], color=FSL_COLOR, label='FSL Dominant', alpha=0.5, s=4)
    bx2.set_ylabel('diff_ls_cwt_kf', color=RPN_AXIS_COLOR, fontsize=16, fontweight='bold')
    bx2.tick_params(axis='y', labelcolor=RPN_AXIS_COLOR)
    bx1.set_title(f'Price & FLL FSL Diff - {current_time.strftime("%Y-%m-%d %H:%M")}', fontsize=24, fontweight='bold')
    bx1.legend(loc='upper left')
    bx2.legend(loc='upper right')

    _configure_time_axis(bx1, params['locator_type'], int(params['interval']))
    _configure_time_axis(ax1, params['locator_type'], int(params['interval']))
    fig.tight_layout()
    return fig, current_time


def _render_feature_figure(df: pd.DataFrame, *, duration_months: int) -> tuple[plt.Figure, pd.Timestamp]:
    _setup_style()
    params = set_time_param(duration_months)
    data = filter_recent(df, duration_months)
    if data.empty:
        raise ValueError('No data available for feature figure.')
    current_time = pd.to_datetime(data.iloc[-1]['time'])

    fig, axes = plt.subplots(2, 1, figsize=(28, 16), sharex=True)

    # Top: Price + FLL/FSL + corr case fill
    ax1 = axes[0]
    ax1.set_xlabel('Time', color='#5b5b5b', fontsize=16, fontweight='bold')
    ax1.tick_params(axis='x', labelcolor='#5b5b5b', rotation=45)
    ax1.plot(data['time'], data['price'], color=PRICE_COLOR, label='BTC Price', linewidth=1.0)
    ax1.set_ylabel('Price', color=PRICE_COLOR, fontsize=16, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=PRICE_COLOR)

    ax2 = ax1.twinx()
    fll = pd.to_numeric(data['fll_cwt_kf'], errors='coerce')
    fsl = pd.to_numeric(data['fsl_cwt_kf'], errors='coerce')
    ax2.plot(data['time'], fll, color=FSL_COLOR, label='fll_cwt_kf', alpha=0.7)
    ax2.plot(data['time'], fsl, color=FLL_COLOR, label='fsl_cwt_kf', alpha=0.7)
    if 'fll_rolling_high' in data.columns:
        ax2.plot(data['time'], data['fll_rolling_high'], color=FSL_COLOR, label='fll_rolling_high', linewidth=0.8, alpha=0.7)
    if 'fsl_rolling_high' in data.columns:
        ax2.plot(data['time'], data['fsl_rolling_high'], color=FLL_COLOR, label='fsl_rolling_high', linewidth=0.8, alpha=0.7)
    fsl_med = fsl.rolling(window=365 * 24, min_periods=1).quantile(0.5)
    fsl_low = fsl.rolling(window=365 * 24, min_periods=1).quantile(0.25)
    ax2.plot(data['time'], fsl_med, color='#ea9999', label='fsl_rolling_median', linewidth=0.8, alpha=0.7)
    ax2.plot(data['time'], fsl_low, color='#f4cccc', label='fsl_rolling_low', linewidth=0.8, alpha=0.7)
    for case, color in CASE_COLORS.items():
        mask = data['corr_case'].astype(str) == case
        if mask.any():
            ax2.fill_between(data['time'], fll, fsl, where=mask, color=color, alpha=0.30)
    ax2.set_ylabel('fll_cwt_kf', color=FSL_COLOR, fontsize=16, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=FSL_COLOR)
    ax1.set_title(f'BTC Price Colored by Risk Priority Number Bins - {current_time.strftime("%Y-%m-%d %H:%M")}', fontsize=24, fontweight='bold')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Bottom: Price + diff + thresholds + event lines
    bx1 = axes[1]
    bx1.set_xlabel('Time', color='#5b5b5b', fontsize=16, fontweight='bold')
    bx1.tick_params(axis='x', labelcolor='#5b5b5b', rotation=45)
    bx1.plot(data['time'], data['price'], color=PRICE_COLOR, label='Price', alpha=0.6, linewidth=1.0)
    _add_price_fill(bx1, data)
    for time in data.loc[data['hit_ceiling_bottom'] == 1, 'time']:
        bx1.axvline(x=time, color=EVENT_BUY, linewidth=0.8, alpha=0.6)
    for time in data.loc[data['hit_ceiling_bottom'] == -1, 'time']:
        bx1.axvline(x=time, color=EVENT_SELL, linewidth=0.8, alpha=0.6)
    for time in data.loc[data['reverse_ceiling_bottom'] == 1, 'time']:
        bx1.axvline(x=time, color=EVENT_SELL, linestyle='--', linewidth=0.8, alpha=0.6)
    for time in data.loc[data['reverse_ceiling_bottom'] == -1, 'time']:
        bx1.axvline(x=time, color=EVENT_BUY, linestyle='--', linewidth=0.8, alpha=0.6)
    bx1.set_ylabel('Price', color=PRICE_COLOR, fontsize=16, fontweight='bold')
    bx1.tick_params(axis='y', labelcolor=PRICE_COLOR)
    bx1.set_ylim(data['price'].min() - 500, data['price'].max() + 500)

    bx2 = bx1.twinx()
    diff = pd.to_numeric(data['diff_ls_cwt_kf'], errors='coerce')
    pos = diff >= 0
    neg = diff < 0
    bx2.scatter(data.loc[pos, 'time'], diff[pos], color=FLL_COLOR, label='FLL Dominant', alpha=0.5, s=4)
    bx2.scatter(data.loc[neg, 'time'], diff[neg], color=FSL_COLOR, label='FSL Dominant', alpha=0.5, s=4)
    if 'thr_diff_pos' in data.columns:
        bx2.plot(data['time'], data['thr_diff_pos'], color=FLL_COLOR, label='thr_diff_pos', alpha=0.7)
    if 'thr_diff_neg' in data.columns:
        bx2.plot(data['time'], data['thr_diff_neg'], color=FSL_COLOR, label='thr_diff_neg', alpha=0.7)
    bx2.set_ylabel('diff_ls_cwt_kf', color=RPN_AXIS_COLOR, fontsize=16, fontweight='bold')
    bx2.tick_params(axis='y', labelcolor=RPN_AXIS_COLOR)
    bx1.set_title(f'Price & FLL FSL Diff - {current_time.strftime("%Y-%m-%d %H:%M")}', fontsize=24, fontweight='bold')
    bx1.legend(loc='upper left')
    bx2.legend(loc='upper right')

    _configure_time_axis(bx1, params['locator_type'], int(params['interval']))
    _configure_time_axis(ax1, params['locator_type'], int(params['interval']))
    fig.tight_layout()
    return fig, current_time


def render_rpn_dominance_png(df: pd.DataFrame, output_path: Path, *, duration_months: int = 3) -> tuple[Path, pd.Timestamp]:
    fig, current_time = _render_dominance_figure(df, duration_months=duration_months)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    return output_path, current_time


def render_rpn_features_png(df: pd.DataFrame, output_path: Path, *, duration_months: int = 3) -> tuple[Path, pd.Timestamp]:
    fig, current_time = _render_feature_figure(df, duration_months=duration_months)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    return output_path, current_time
