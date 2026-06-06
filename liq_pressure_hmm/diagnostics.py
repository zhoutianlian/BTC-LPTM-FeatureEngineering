from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from .state_summary import STATE_ROLE_MAP

STATE_COLORS = {
    1: "#0b8f3a",  # short-liq strong / up-pressure strong
    2: "#66c56c",  # short-liq mild / up-pressure mild
    3: "#d6b24c",  # balanced
    4: "#f28e2b",  # long-liq mild / down-pressure mild
    5: "#d62728",  # long-liq strong / down-pressure strong
}


def _write_html(fig: go.Figure, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(
        fig,
        file=str(output_path),
        full_html=True,
        include_plotlyjs="cdn",
        config={"responsive": True, "scrollZoom": True, "displaylogo": False},
    )
    return output_path


def _coerce_time_index(df: pd.DataFrame, time_col: str | None) -> pd.DataFrame:
    out = df.copy()
    if time_col is not None:
        out[time_col] = pd.to_datetime(out[time_col])
        out = out.set_index(time_col)
    elif not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    return out.sort_index()


def _thin_time_series(df: pd.DataFrame, max_points: Optional[int]) -> pd.DataFrame:
    """Uniformly thin long time series for Plotly HTML speed.

    Plotly HTML generation time grows quickly when hundreds of thousands of
    points and many shapes are written.  The diagnostic plots are visual checks,
    not model inputs, so uniform thinning is acceptable and explicitly
    controlled by config.
    """
    if max_points is None or max_points <= 0 or len(df) <= max_points:
        return df
    loc = np.linspace(0, len(df) - 1, int(max_points)).round().astype(int)
    loc = np.unique(loc)
    return df.iloc[loc]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _infer_time_step(index: pd.Index) -> pd.Timedelta:
    if len(index) < 2:
        return pd.Timedelta(minutes=10)
    idx = pd.to_datetime(index)
    diffs = pd.Series(idx[1:] - idx[:-1]).dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return pd.Timedelta(minutes=10)
    return pd.to_timedelta(diffs.median())


def _exact_state_segments(df: pd.DataFrame, state_col: str) -> List[Tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Exact consecutive state intervals as half-open visual bands."""
    if df.empty:
        return []
    idx = pd.to_datetime(df.index)
    states = pd.to_numeric(df[state_col], errors="coerce").astype("Int64").to_numpy()
    valid = pd.notna(states)
    if not valid.all():
        return _exact_state_segments(df.loc[valid].copy(), state_col)
    states = states.astype(int)
    if len(states) == 0:
        return []
    starts = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1]
    step = _infer_time_step(idx)
    segments: List[Tuple[pd.Timestamp, pd.Timestamp, int]] = []
    for i, start_pos in enumerate(starts):
        end_pos = int(starts[i + 1]) if i + 1 < len(starts) else len(states)
        x0 = pd.Timestamp(idx[start_pos])
        x1 = pd.Timestamp(idx[end_pos]) if end_pos < len(idx) else pd.Timestamp(idx[-1]) + step
        if x1 <= x0:
            x1 = x0 + step
        segments.append((x0, x1, int(states[start_pos])))
    return segments


def _state_segments(
    df: pd.DataFrame,
    state_col: str,
    max_shape_segments: Optional[int] = None,
) -> Tuple[List[Tuple[pd.Timestamp, pd.Timestamp, int]], bool]:
    """Return state background bands, compressing when exact bands are too many.

    ``compressed=True`` means the state path was visually coarsened into modal
    time bins.  This keeps background colors available even for long samples,
    while bounding expensive Plotly layout-shape count.
    """
    exact = _exact_state_segments(df, state_col)
    if max_shape_segments is None or max_shape_segments <= 0 or len(exact) <= int(max_shape_segments):
        return exact, False

    max_shape_segments = int(max_shape_segments)
    n = len(df)
    if n == 0:
        return [], False
    idx = pd.to_datetime(df.index)
    states = pd.to_numeric(df[state_col], errors="coerce").astype("Int64").to_numpy()
    valid = pd.notna(states)
    if not valid.all():
        return _state_segments(df.loc[valid].copy(), state_col, max_shape_segments)
    states = states.astype(int)
    step = _infer_time_step(idx)
    n_bins = min(max_shape_segments, n)
    edges = np.linspace(0, n, n_bins + 1).round().astype(int)
    edges[0] = 0
    edges[-1] = n
    edges = np.unique(edges)

    raw: List[Tuple[pd.Timestamp, pd.Timestamp, int]] = []
    for start_pos, end_pos in zip(edges[:-1], edges[1:]):
        start_pos = int(start_pos)
        end_pos = int(end_pos)
        if end_pos <= start_pos:
            continue
        vals, counts = np.unique(states[start_pos:end_pos], return_counts=True)
        st = int(vals[np.argmax(counts)])
        x0 = pd.Timestamp(idx[start_pos])
        x1 = pd.Timestamp(idx[end_pos]) if end_pos < n else pd.Timestamp(idx[-1]) + step
        if x1 <= x0:
            x1 = x0 + step
        raw.append((x0, x1, st))

    merged: List[Tuple[pd.Timestamp, pd.Timestamp, int]] = []
    for x0, x1, st in raw:
        if merged and merged[-1][2] == st:
            old_x0, _, _ = merged[-1]
            merged[-1] = (old_x0, x1, st)
        else:
            merged.append((x0, x1, st))
    return merged, True


def _state_band_shapes(
    segments: List[Tuple[pd.Timestamp, pd.Timestamp, int]],
    *,
    alpha: float,
    xref: str = "x",
    yref: str = "y domain",
) -> List[Dict[str, Any]]:
    return [
        {
            "type": "rect",
            "xref": xref,
            "yref": yref,
            "x0": start,
            "x1": end,
            "y0": 0,
            "y1": 1,
            "fillcolor": _hex_to_rgba(STATE_COLORS.get(st, "#888888"), alpha),
            "line": {"width": 0},
            "layer": "below",
        }
        for start, end, st in segments
    ]

def _state_colorscale(state_color: Dict[int, str]) -> List[List[float | str]]:
    """Build a discrete Plotly colorscale for state values 1..5."""
    boundaries = {
        1: (0.00, 0.125),
        2: (0.125, 0.375),
        3: (0.375, 0.625),
        4: (0.625, 0.875),
        5: (0.875, 1.00),
    }
    scale: List[List[float | str]] = []
    for st in [1, 2, 3, 4, 5]:
        c = state_color.get(st, STATE_COLORS.get(st, "#999999"))
        lo, hi = boundaries[st]
        scale.append([lo, c])
        scale.append([hi, c])
    return scale


def _add_state_heatmap_background(
    fig: go.Figure,
    dfp: pd.DataFrame,
    *,
    state_col: str,
    y_min: float,
    y_max: float,
    opacity: float,
    max_background_points: int,
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Fast 5-state background layer for dashboard row 1.

    This uses a compact categorical heatmap instead of thousands of layout
    rectangles. It preserves the five-state background colors while keeping HTML
    generation fast. Downsampling is visualization-only.
    """
    if state_col not in dfp.columns or dfp.empty:
        return
    y_min = float(y_min)
    y_max = float(y_max)
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return
    if y_min == y_max:
        pad = abs(y_min) * 0.001 + 1.0
        y_min -= pad
        y_max += pad

    bg = dfp[[state_col]].dropna().copy()
    if bg.empty:
        return
    bg[state_col] = pd.to_numeric(bg[state_col], errors="coerce")
    bg = bg.dropna(subset=[state_col])
    bg[state_col] = bg[state_col].astype(int).clip(1, 5)
    bg = _thin_time_series(bg, max_background_points)
    if bg.empty:
        return

    z = np.vstack([bg[state_col].to_numpy(dtype=float), bg[state_col].to_numpy(dtype=float)])
    trace = go.Heatmap(
        x=bg.index,
        y=[y_min, y_max],
        z=z,
        zmin=1,
        zmax=5,
        colorscale=_state_colorscale(STATE_COLORS),
        showscale=False,
        opacity=opacity,
        hoverinfo="skip",
        name="State background",
    )
    if row is None or col is None:
        fig.add_trace(trace)
    else:
        fig.add_trace(trace, row=row, col=col)


def _custom_state_data(dfp: pd.DataFrame, state_col: str) -> np.ndarray:
    state = pd.to_numeric(dfp[state_col], errors="coerce").astype("Int64")
    name_cn = [STATE_ROLE_MAP.get(int(v), {}).get("name_cn", "") if pd.notna(v) else "" for v in state]
    pressure_cn = [STATE_ROLE_MAP.get(int(v), {}).get("pressure_cn", "") if pd.notna(v) else "" for v in state]
    return np.column_stack([state.astype(str).to_numpy(), np.array(name_cn), np.array(pressure_cn)])


def state_distribution(df: pd.DataFrame, *, state_col: str = "hmm_state") -> pd.DataFrame:
    s = pd.to_numeric(df[state_col], errors="coerce").dropna().astype(int)
    counts = s.value_counts().reindex(range(1, 6), fill_value=0).sort_index()
    total = float(counts.sum())
    out = pd.DataFrame({
        "state": counts.index.astype(int),
        "count": counts.values.astype(int),
        "share": counts.values / total if total > 0 else np.nan,
    })
    out["state_name_en"] = out["state"].map(lambda x: STATE_ROLE_MAP.get(int(x), {}).get("name_en"))
    out["state_name_cn"] = out["state"].map(lambda x: STATE_ROLE_MAP.get(int(x), {}).get("name_cn"))
    out["pressure_cn"] = out["state"].map(lambda x: STATE_ROLE_MAP.get(int(x), {}).get("pressure_cn"))
    return out


def plot_state_distribution_html(
    df: pd.DataFrame,
    *,
    output_path: str | Path,
    state_col: str = "hmm_state",
    title: str = "State count distribution",
) -> Path:
    dist = state_distribution(df, state_col=state_col)
    labels = [f"S{r.state}<br>{r.state_name_cn}" for r in dist.itertuples()]
    hover = [
        f"State {r.state}<br>{r.state_name_en}<br>{r.pressure_cn}<br>count={r.count:,}<br>share={r.share:.2%}"
        for r in dist.itertuples()
    ]

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.62, 0.38],
        specs=[[{"type": "xy"}, {"type": "table"}]],
        subplot_titles=("Counts / shares", "State semantics"),
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=dist["count"],
            marker_color=[STATE_COLORS.get(int(s), "#999") for s in dist["state"]],
            customdata=np.column_stack([dist["share"], dist["state_name_en"], dist["pressure_cn"]]),
            hovertemplate=(
                "%{x}<br>count=%{y:,}<br>share=%{customdata[0]:.2%}"
                "<br>%{customdata[1]}<br>%{customdata[2]}<extra></extra>"
            ),
            text=[f"{v:.1%}" for v in dist["share"]],
            textposition="outside",
            name="state count",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Table(
            header={"values": ["State", "中文含义", "English", "价格压力", "Count", "Share"], "fill_color": "#1f2937", "font": {"color": "white"}},
            cells={
                "values": [
                    [f"S{s}" for s in dist["state"]],
                    dist["state_name_cn"],
                    dist["state_name_en"],
                    dist["pressure_cn"],
                    [f"{int(c):,}" for c in dist["count"]],
                    [f"{x:.2%}" for x in dist["share"]],
                ],
                "fill_color": "#111827",
                "font": {"color": "#e5e7eb"},
                "align": "left",
            },
        ),
        row=1,
        col=2,
    )
    fig.update_layout(template="plotly_dark", title=title, height=560, margin={"l": 70, "r": 30, "t": 80, "b": 60})
    fig.update_yaxes(title_text="bar count", row=1, col=1)
    return _write_html(fig, Path(output_path))


def plot_regime_diagnostic_dashboard_html(
    df: pd.DataFrame,
    *,
    output_path: str | Path,
    time_col: str | None = None,
    price_col: str = "price",
    state_col: str = "hmm_state",
    title: str = "Liquidation HMM diagnostic dashboard",
    max_points: int = 30000,
    max_shape_segments: int = 1200,
    background_max_points: int = 8000,
    background_opacity: float = 0.18,
) -> Path:
    dfp = _coerce_time_index(df, time_col)
    req_cols = [price_col, state_col]
    dfp = dfp.dropna(subset=req_cols)
    if dfp.empty:
        raise ValueError("No valid rows for dashboard plot.")
    dfp = _thin_time_series(dfp, max_points=max_points)

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.35, 0.18, 0.16, 0.15, 0.16],
        subplot_titles=(
            "Price with regime hover",
            "Regime probabilities",
            "Direction expectation",
            "Regime confidence / entropy",
            "Regime age / feature staleness",
        ),
    )

    custom = _custom_state_data(dfp, state_col)
    price_series = pd.to_numeric(dfp[price_col], errors="coerce")
    _add_state_heatmap_background(
        fig,
        dfp,
        state_col=state_col,
        y_min=float(price_series.min()),
        y_max=float(price_series.max()),
        opacity=background_opacity,
        max_background_points=background_max_points,
        row=1,
        col=1,
    )
    Trace = go.Scattergl if len(dfp) >= 15000 else go.Scatter
    fig.add_trace(
        Trace(
            x=dfp.index,
            y=pd.to_numeric(dfp[price_col], errors="coerce"),
            mode="lines",
            name="Price",
            line={"color": "#e8eaed", "width": 1.2},
            customdata=custom,
            hovertemplate=(
                "%{x|%Y-%m-%d %H:%M}<br>Price=%{y:.2f}"
                "<br>State=%{customdata[0]} / %{customdata[1]}"
                "<br>%{customdata[2]}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    # The state background is drawn as a compact heatmap trace before the price
    # trace.  `max_shape_segments` is kept as a backwards-compatible config key,
    # but the dashboard no longer drops the background when state transitions are
    # frequent.

    # Row 2: probabilities
    for col, name, color in [
        ("p_up_pressure", "p_up_pressure / p_short_liq", "#66c56c"),
        ("p_neutral", "p_neutral", "#d6b24c"),
        ("p_down_pressure", "p_down_pressure / p_long_liq", "#d62728"),
    ]:
        if col in dfp.columns:
            fig.add_trace(
                go.Scattergl(x=dfp.index, y=dfp[col], mode="lines", name=name, line={"color": color, "width": 1.0}),
                row=2,
                col=1,
            )

    # Row 3: dir_expect
    if "dir_expect" in dfp.columns:
        fig.add_trace(
            go.Scattergl(x=dfp.index, y=dfp["dir_expect"], mode="lines", name="dir_expect", line={"color": "#4cc9f0", "width": 1.0}),
            row=3,
            col=1,
        )
        fig.add_hline(y=0.0, line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.4)", row=3, col=1)

    # Row 4: confidence / entropy
    if "hmm_conf" in dfp.columns:
        fig.add_trace(go.Scattergl(x=dfp.index, y=dfp["hmm_conf"], mode="lines", name="hmm_conf", line={"color": "#90e0ef", "width": 1.0}), row=4, col=1)
    if "liq_entropy" in dfp.columns:
        fig.add_trace(go.Scattergl(x=dfp.index, y=dfp["liq_entropy"], mode="lines", name="liq_entropy", line={"color": "#f72585", "width": 1.0}), row=4, col=1)

    # Row 5: age / staleness
    if "age_in_state" in dfp.columns:
        fig.add_trace(go.Scattergl(x=dfp.index, y=dfp["age_in_state"], mode="lines", name="age_in_state_10m", line={"color": "#ffd166", "width": 1.0}), row=5, col=1)
    if "age_in_state_source" in dfp.columns:
        fig.add_trace(go.Scattergl(x=dfp.index, y=dfp["age_in_state_source"], mode="lines", name="age_in_state_source", line={"color": "#06d6a0", "width": 1.0}), row=5, col=1)
    if "liq_feature_age_min" in dfp.columns:
        fig.add_trace(go.Scattergl(x=dfp.index, y=dfp["liq_feature_age_min"], mode="lines", name="liq_feature_age_min", line={"color": "#ef476f", "width": 1.0}), row=5, col=1)

    fig.update_layout(
        template="plotly_dark",
        title=title,
        height=1250,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 70, "r": 30, "t": 80, "b": 50},
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    return _write_html(fig, Path(output_path))


def plot_state_feature_boxplots_html(
    df: pd.DataFrame,
    *,
    output_path: str | Path,
    time_col: str | None = None,
    state_col: str = "hmm_state",
    max_points_per_state: int = 6000,
) -> Path:
    dfp = _coerce_time_index(df, time_col)
    cols = [c for c in [state_col, "fll_cwt_kf", "fsl_cwt_kf", "risk_priority_number", "diff_ls_cwt_kf", "total_ls_cwt_kf"] if c in dfp.columns]
    if state_col not in cols:
        raise ValueError("State column missing for boxplot diagnostics.")
    dfp = dfp[cols].dropna(subset=[state_col]).copy()

    # Deterministic per-state sampling keeps each state visible and avoids huge HTML arrays.
    sampled = []
    for st, grp in dfp.groupby(dfp[state_col].astype(int)):
        if len(grp) > max_points_per_state:
            sampled.append(grp.iloc[np.linspace(0, len(grp) - 1, max_points_per_state).round().astype(int)])
        else:
            sampled.append(grp)
    dfp = pd.concat(sampled, axis=0).sort_index() if sampled else dfp

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Long-liquidation proxy by state (FLL ≈ forced selling)",
        "Short-liquidation proxy by state (FSL ≈ forced buying)",
        "Directional imbalance L-S by state",
        "Total liquidation intensity by state",
    ))

    metric_map = [
        ("fll_cwt_kf", 1, 1),
        ("fsl_cwt_kf", 1, 2),
        ("diff_ls_cwt_kf", 2, 1),
        ("total_ls_cwt_kf", 2, 2),
    ]
    for metric, row, col in metric_map:
        if metric not in dfp.columns:
            continue
        for st in sorted(dfp[state_col].dropna().astype(int).unique()):
            sub = dfp.loc[dfp[state_col].astype(int) == st, metric]
            fig.add_trace(
                go.Box(
                    y=sub,
                    name=f"S{st}-{STATE_ROLE_MAP.get(st, {}).get('name_cn', st)}",
                    marker_color=STATE_COLORS.get(st, None),
                    boxmean=True,
                    boxpoints=False,
                    showlegend=(row == 1 and col == 1),
                    hovertemplate=f"S{st} {STATE_ROLE_MAP.get(st, {}).get('name_cn', '')}<br>{metric}=%{{y:.4f}}<extra></extra>",
                ),
                row=row,
                col=col,
            )

    fig.update_layout(template="plotly_dark", title="State feature distribution diagnostics", height=900)
    return _write_html(fig, Path(output_path))


def _episode_durations(df: pd.DataFrame, state_col: str) -> pd.DataFrame:
    s = pd.to_numeric(df[state_col], errors="coerce").dropna().astype(int)
    episode_id = (s != s.shift(1)).cumsum()
    out = pd.DataFrame({"state": s, "episode_id": episode_id}, index=s.index)
    dur = out.groupby(["episode_id", "state"]).size().reset_index(name="duration_bars")
    return dur


def plot_transition_duration_html(
    df: pd.DataFrame,
    *,
    output_path: str | Path,
    time_col: str | None = None,
    state_col: str = "hmm_state",
) -> Path:
    dfp = _coerce_time_index(df, time_col)
    s = pd.to_numeric(dfp[state_col], errors="coerce").dropna().astype(int)
    if s.empty:
        raise ValueError("No state data for transition diagnostics.")
    trans = pd.crosstab(s.shift(1), s, normalize="index").reindex(index=range(1, 6), columns=range(1, 6), fill_value=0.0)
    dur = _episode_durations(dfp, state_col)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Transition matrix (row-normalized)", "Episode durations by state"), specs=[[{"type": "heatmap"}, {"type": "xy"}]])
    fig.add_trace(go.Heatmap(z=trans.values, x=[f"to {c}" for c in trans.columns], y=[f"from {i}" for i in trans.index], colorscale="Viridis"), row=1, col=1)
    for st in sorted(dur["state"].unique()):
        fig.add_trace(go.Histogram(x=dur.loc[dur["state"] == st, "duration_bars"], name=f"S{st}-{STATE_ROLE_MAP.get(int(st), {}).get('name_cn', '')}", opacity=0.65, marker_color=STATE_COLORS.get(int(st), None)), row=1, col=2)
    fig.update_layout(template="plotly_dark", title="Transition / duration diagnostics", height=600, barmode="overlay")
    fig.update_xaxes(title_text="bars", row=1, col=2)
    return _write_html(fig, Path(output_path))


def _select_event_times(series: pd.Series, quantile: float, cooldown_bars: int) -> pd.Index:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return pd.Index([])
    thr = float(s.quantile(quantile))
    events = s[s >= thr]
    selected: List[pd.Timestamp] = []
    last_idx: Optional[int] = None
    positions = {ts: i for i, ts in enumerate(s.index)}
    for ts in events.sort_index().index:
        pos = positions[ts]
        if last_idx is None or pos - last_idx > cooldown_bars:
            selected.append(ts)
            last_idx = pos
    return pd.Index(selected)


def _event_window_curve(price: pd.Series, event_times: pd.Index, back: int, fwd: int) -> pd.Series:
    price = pd.to_numeric(price, errors="coerce").astype(float)
    rets = np.log(price).diff()
    paths = []
    idx = list(price.index)
    pos_map = {ts: i for i, ts in enumerate(idx)}
    for ts in event_times:
        if ts not in pos_map:
            continue
        i = pos_map[ts]
        if i - back < 0 or i + fwd >= len(price):
            continue
        window = rets.iloc[i - back + 1 : i + fwd + 1].to_numpy(dtype=float)
        cum = np.concatenate([[0.0], np.nancumsum(window)])
        if len(cum) == back + fwd + 1:
            paths.append(cum)
    if not paths:
        return pd.Series(dtype=float)
    med = np.nanmedian(np.vstack(paths), axis=0)
    x = np.arange(-back, fwd + 1)
    return pd.Series(med, index=x)


def plot_event_windows_html(
    df: pd.DataFrame,
    *,
    output_path: str | Path,
    time_col: str | None = None,
    price_col: str = "price",
    long_liq_col: str = "fll_cwt_kf",
    short_liq_col: str = "fsl_cwt_kf",
    quantile: float = 0.99,
    back: int = 12,
    fwd: int = 24,
    cooldown_bars: int = 12,
    source_time_col: str = "liq_feature_time",
) -> Path:
    dfp = _coerce_time_index(df, time_col)
    if price_col not in dfp.columns:
        raise ValueError("Price column missing for event study plot.")

    event_df = dfp
    if source_time_col in event_df.columns:
        event_df = event_df.copy()
        event_df[source_time_col] = pd.to_datetime(event_df[source_time_col], errors="coerce")
        event_df = event_df.dropna(subset=[source_time_col]).sort_values(source_time_col)
        event_df = event_df.drop_duplicates(subset=[source_time_col], keep="last").set_index(source_time_col, drop=False).sort_index()

    long_events = _select_event_times(event_df[long_liq_col], quantile=quantile, cooldown_bars=cooldown_bars) if long_liq_col in event_df.columns else pd.Index([])
    short_events = _select_event_times(event_df[short_liq_col], quantile=quantile, cooldown_bars=cooldown_bars) if short_liq_col in event_df.columns else pd.Index([])

    long_curve = _event_window_curve(dfp[price_col], long_events, back, fwd)
    short_curve = _event_window_curve(dfp[price_col], short_events, back, fwd)

    fig = go.Figure()
    if not long_curve.empty:
        fig.add_trace(go.Scatter(x=long_curve.index, y=long_curve.values, mode="lines", name=f"Long-liquidation shocks / down pressure (q>={quantile:.2f}, n={len(long_events)})", line={"color": "#d62728", "width": 2.0}))
    if not short_curve.empty:
        fig.add_trace(go.Scatter(x=short_curve.index, y=short_curve.values, mode="lines", name=f"Short-liquidation shocks / up pressure (q>={quantile:.2f}, n={len(short_events)})", line={"color": "#0b8f3a", "width": 2.0}))
    fig.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.4)")
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.4)")
    fig.update_layout(template="plotly_dark", title="Median event-window cumulative log return", xaxis_title="bars relative to event", yaxis_title="cumulative log return", height=550)
    return _write_html(fig, Path(output_path))


def summarize_state_semantics(df: pd.DataFrame, *, state_col: str = "hmm_state") -> pd.DataFrame:
    preferred_cols = [
        state_col,
        "fll_cwt_kf",
        "fsl_cwt_kf",
        "total_ls_cwt_kf",
        "diff_ls_cwt_kf",
        "p_short_liq",
        "p_up_pressure",
        "p_neutral",
        "p_long_liq",
        "p_down_pressure",
        "dir_expect",
        "hmm_conf",
        "liq_entropy",
    ]
    cols = [c for c in preferred_cols if c in df.columns]
    tmp = df[cols].dropna(subset=[state_col]).copy()
    if "total_ls_cwt_kf" not in tmp.columns and {"fll_cwt_kf", "fsl_cwt_kf"}.issubset(tmp.columns):
        tmp["total_ls_cwt_kf"] = tmp["fll_cwt_kf"] + tmp["fsl_cwt_kf"]
    if "diff_ls_cwt_kf" not in tmp.columns and {"fll_cwt_kf", "fsl_cwt_kf"}.issubset(tmp.columns):
        tmp["diff_ls_cwt_kf"] = tmp["fll_cwt_kf"] - tmp["fsl_cwt_kf"]
    if {"fll_cwt_kf", "fsl_cwt_kf"}.issubset(tmp.columns):
        tmp["short_dom"] = tmp["fsl_cwt_kf"] - tmp["fll_cwt_kf"]
        tmp["long_dom"] = tmp["fll_cwt_kf"] - tmp["fsl_cwt_kf"]
    summary = tmp.groupby(state_col).agg(["count", "median", "mean"])
    summary.columns = ["_".join(col).strip() for col in summary.columns.to_flat_index()]
    summary = summary.reset_index().rename(columns={state_col: "state"})

    dist = state_distribution(df, state_col=state_col)[["state", "count", "share", "state_name_en", "state_name_cn", "pressure_cn"]]
    summary = summary.merge(dist, on="state", how="left", suffixes=("", "_dist"))
    summary = summary.rename(columns={"count": "state_count", "share": "state_share"})
    return summary


def _rel_link(base: Path, target: Path) -> str:
    try:
        return escape(str(Path(target).resolve().relative_to(base.resolve())))
    except Exception:
        return escape(str(Path(target)))


def write_report_index_html(
    *,
    output_path: str | Path,
    title: str,
    links: Dict[str, Path],
    notes: Optional[List[str]] = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    notes = notes or []
    rows = []
    for name, path in links.items():
        if path is None:
            continue
        rows.append(f"<li><a href='{_rel_link(output_path.parent, path)}'>{escape(name)}</a><span>{escape(str(path.name))}</span></li>")
    note_html = "".join(f"<p>{escape(n)}</p>" for n in notes)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>{escape(title)}</title>
<style>
body {{ margin:0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background:#0b0f1a; color:#e5e7eb; }}
main {{ max-width: 1080px; margin: 48px auto; padding: 0 24px; }}
h1 {{ font-size: 28px; margin-bottom: 8px; }}
p {{ color:#a7b0c0; line-height:1.65; }}
ul {{ list-style: none; padding: 0; display:grid; gap:12px; }}
li {{ background:#111827; border:1px solid rgba(255,255,255,.08); border-radius:14px; padding:16px 18px; display:flex; justify-content:space-between; gap:16px; align-items:center; }}
a {{ color:#7dd3fc; font-size:17px; text-decoration:none; font-weight:650; }}
a:hover {{ text-decoration:underline; }}
span {{ color:#9ca3af; font-size:13px; }}
.card {{ margin-top:24px; padding:18px; background:#111827; border-radius:14px; border:1px solid rgba(255,255,255,.08); }}
</style>
</head>
<body><main>
<h1>{escape(title)}</h1>
<div class="card">{note_html}</div>
<ul>{''.join(rows)}</ul>
</main></body></html>"""
    output_path.write_text(html, encoding="utf-8")
    return output_path


def generate_diagnostic_report(
    df: pd.DataFrame,
    *,
    out_dir: str | Path,
    time_col: str | None = None,
    event_quantile: float = 0.99,
    dashboard_max_points: int = 30000,
    state_boxplot_max_points_per_state: int = 6000,
    max_shape_segments: int = 1200,
    background_max_points: int = 8000,
    background_opacity: float = 0.18,
    write_index: bool = True,
) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {
        "dashboard": plot_regime_diagnostic_dashboard_html(
            df,
            output_path=out_dir / "regime_dashboard.html",
            time_col=time_col,
            max_points=dashboard_max_points,
            max_shape_segments=max_shape_segments,
            background_max_points=background_max_points,
            background_opacity=background_opacity,
        ),
        "state_distribution": plot_state_distribution_html(df, output_path=out_dir / "state_distribution.html"),
        "state_boxplots": plot_state_feature_boxplots_html(
            df,
            output_path=out_dir / "state_feature_boxplots.html",
            time_col=time_col,
            max_points_per_state=state_boxplot_max_points_per_state,
        ),
        "transition_duration": plot_transition_duration_html(df, output_path=out_dir / "transition_duration.html", time_col=time_col),
        "event_windows": plot_event_windows_html(df, output_path=out_dir / "event_windows.html", time_col=time_col, quantile=event_quantile),
    }
    summary = summarize_state_semantics(df)
    summary_path = out_dir / "state_semantic_summary.csv"
    summary.to_csv(summary_path, index=False)
    paths["state_semantic_summary"] = summary_path

    if write_index:
        paths["diagnostic_index"] = write_report_index_html(
            output_path=out_dir / "index.html",
            title="BTC liquidation HMM diagnostics",
            links={
                "1. Regime dashboard": paths["dashboard"],
                "2. State count distribution": paths["state_distribution"],
                "3. State feature boxplots": paths["state_boxplots"],
                "4. Transition / duration diagnostics": paths["transition_duration"],
                "5. Liquidation event windows": paths["event_windows"],
                "6. State semantic summary CSV": paths["state_semantic_summary"],
            },
            notes=[
                "State semantics: S1/S2 = short-liquidation dominance and upward liquidation pressure; S4/S5 = long-liquidation dominance and downward liquidation pressure; S3 = balanced.",
                "Large HTML plots are downsampled only for visualization speed. Model outputs and CSV files are not downsampled.",
            ],
        )
    return paths
