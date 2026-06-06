from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.colors import qualitative
from plotly.subplots import make_subplots

from liq_pressure_hmm.state_summary import STATE_ROLE_MAP


def _rangeselector_buttons() -> list[dict[str, Any]]:
    return [
        {"count": 24, "label": "1D", "step": "hour", "stepmode": "backward"},
        {"count": 7, "label": "1W", "step": "day", "stepmode": "backward"},
        {"count": 1, "label": "1M", "step": "month", "stepmode": "backward"},
        {"count": 3, "label": "3M", "step": "month", "stepmode": "backward"},
        {"count": 6, "label": "6M", "step": "month", "stepmode": "backward"},
        {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
        {"step": "all", "label": "ALL"},
    ]


def _full_viewport_html(html: str, *, bg: str = "#0b0f1a") -> str:
    """Make Plotly's standalone HTML fill the browser viewport."""
    css = f"""
<style>
html, body {{
  width: 100%;
  height: 100%;
  margin: 0;
  padding: 0;
  overflow: hidden;
  background: {bg};
}}
body > div {{
  width: 100vw;
  height: 100vh;
}}
.plotly-graph-div {{
  width: 100vw !important;
  height: 100vh !important;
}}
</style>"""
    if "</head>" in html:
        html = html.replace("</head>", f"{css}</head>", 1)
    return html


def _write_html(fig: go.Figure, output_path: Path, *, post_script: str | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = pio.to_html(
        fig,
        full_html=True,
        include_plotlyjs="cdn",
        config={"responsive": True, "scrollZoom": True, "displaylogo": False},
        post_script=post_script,
    )
    output_path.write_text(_full_viewport_html(html), encoding="utf-8")


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


@dataclass(frozen=True)
class HMMVisStyle:
    template: str = "plotly_dark"
    bg: str = "#0b0f1a"
    grid: str = "rgba(255,255,255,0.08)"
    price_color: str = "#f1c232"
    band_alpha: float = 0.20
    background_max_points: int = 8000


DEFAULT_STATE_COLORS: dict[int, str] = {
    1: "#0b8f3a",  # short-liq strong / up-pressure strong
    2: "#66c56c",  # short-liq mild / up-pressure mild
    3: "#d6b24c",  # balanced
    4: "#f28e2b",  # long-liq mild / down-pressure mild
    5: "#d62728",  # long-liq strong / down-pressure strong
}


def _coerce_time_index(df: pd.DataFrame, time_col: str | None) -> pd.DataFrame:
    dfp = df.copy()
    if time_col is not None:
        dfp[time_col] = pd.to_datetime(dfp[time_col])
        dfp = dfp.set_index(time_col)
    else:
        if not isinstance(dfp.index, pd.DatetimeIndex):
            dfp.index = pd.to_datetime(dfp.index)
    return dfp.sort_index()


def _thin_time_series(df: pd.DataFrame, max_points: Optional[int]) -> pd.DataFrame:
    if max_points is None or max_points <= 0 or len(df) <= max_points:
        return df
    loc = np.linspace(0, len(df) - 1, int(max_points)).round().astype(int)
    loc = np.unique(loc)
    return df.iloc[loc]


def _price_axis_range(series: pd.Series, padding_frac: float = 0.07) -> tuple[float, float] | None:
    """Return a padded visual y-axis range from valid price observations only."""
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if values.size == 0:
        return None

    positive_values = values[values > 0]
    if positive_values.size >= max(3, int(values.size * 0.8)):
        values = positive_values
    if values.size == 0:
        return None

    y_min = float(np.nanmin(values))
    y_max = float(np.nanmax(values))
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return None

    span = y_max - y_min
    if span <= 0:
        pad = max(abs(y_min) * 0.005, 1.0)
    else:
        pad = max(span * padding_frac, abs(y_max) * 1e-6)

    lower = y_min - pad
    upper = y_max + pad
    if y_min > 0 and lower <= 0:
        lower = max(y_min * (1.0 - padding_frac), np.nextafter(0.0, 1.0))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        return None
    return lower, upper


def _visible_price_axis_autorange_script(padding_frac: float = 0.07) -> str:
    """Keep the price y-axis fitted to the current visible x-window."""
    return f"""
(function() {{
  var gd = document.getElementById('{{plot_id}}');
  if (!gd || !gd.data) return;
  var priceIndex = gd.data.findIndex(function(trace) {{ return trace.name === 'Price'; }});
  if (priceIndex < 0) return;
  var trace = gd.data[priceIndex];
  var xs = (trace.x || []).map(function(x) {{ return new Date(x).getTime(); }});
  var ys = (trace.y || []).map(function(y) {{ return Number(y); }});
  var padding = {float(padding_frac):.8f};
  var pending = null;

  function rangeFromValues(x0, x1) {{
    var lo = Infinity;
    var hi = -Infinity;
    var n = 0;
    for (var i = 0; i < ys.length; i++) {{
      var x = xs[i];
      var y = ys[i];
      if (!Number.isFinite(x) || !Number.isFinite(y) || y <= 0) continue;
      if (x0 !== null && x < x0) continue;
      if (x1 !== null && x > x1) continue;
      if (y < lo) lo = y;
      if (y > hi) hi = y;
      n++;
    }}
    if (n === 0 && (x0 !== null || x1 !== null)) return rangeFromValues(null, null);
    if (n === 0) return null;
    var span = hi - lo;
    var pad = span > 0 ? Math.max(span * padding, Math.abs(hi) * 1e-6) : Math.max(Math.abs(lo) * 0.005, 1.0);
    var lower = lo - pad;
    var upper = hi + pad;
    if (lo > 0 && lower <= 0) lower = Math.max(lo * (1.0 - padding), Number.MIN_VALUE);
    if (!Number.isFinite(lower) || !Number.isFinite(upper) || lower >= upper) return null;
    return [lower, upper];
  }}

  function currentXRange() {{
    var axis = (gd._fullLayout && gd._fullLayout.xaxis) || (gd.layout && gd.layout.xaxis) || {{}};
    var xr = axis.range;
    if (!xr || xr.length < 2) return [null, null];
    return [new Date(xr[0]).getTime(), new Date(xr[1]).getTime()];
  }}

  function refitYAxis() {{
    var xr = currentXRange();
    var yr = rangeFromValues(xr[0], xr[1]);
    if (!yr) return;
    Plotly.relayout(gd, {{'yaxis.range': yr, 'yaxis.autorange': false}});
  }}

  function scheduleRefit() {{
    if (pending !== null) clearTimeout(pending);
    pending = setTimeout(refitYAxis, 60);
  }}

  gd.on('plotly_relayout', function(evt) {{
    evt = evt || {{}};
    var hasXChange = evt['xaxis.range'] || evt['xaxis.range[0]'] || evt['xaxis.range[1]'] || evt['xaxis.autorange'];
    var hasYOnlyChange = !hasXChange && (evt['yaxis.range'] || evt['yaxis.range[0]'] || evt['yaxis.range[1]'] || evt['yaxis.autorange']);
    if (hasYOnlyChange) return;
    if (hasXChange) scheduleRefit();
  }});
  requestAnimationFrame(refitYAxis);
}})();
"""


def _viewport_height_script(min_height: int = 720) -> str:
    """Resize the chart to the current viewport height."""
    return f"""
(function() {{
  var gd = document.getElementById('{{plot_id}}');
  if (!gd) return;
  var minHeight = {int(min_height)};
  var pending = null;

  function resizeToViewport() {{
    var h = Math.max(minHeight, window.innerHeight || document.documentElement.clientHeight || minHeight);
    gd.style.height = h + 'px';
    Plotly.relayout(gd, {{height: h}});
    Plotly.Plots.resize(gd);
  }}

  function scheduleResize() {{
    if (pending !== null) cancelAnimationFrame(pending);
    pending = requestAnimationFrame(function() {{
      pending = null;
      resizeToViewport();
    }});
  }}

  window.addEventListener('resize', scheduleResize);
  requestAnimationFrame(resizeToViewport);
}})();
"""


def _infer_time_step(index: pd.Index) -> pd.Timedelta:
    """Infer one visual bar width for state bands.

    The previous implementation used the last timestamp of each state segment as
    x1.  Single-bar segments therefore had x0 == x1 and became invisible.  This
    helper lets us extend the final segment and one-bar segments by one inferred
    bar step without touching model outputs.
    """
    if len(index) < 2:
        return pd.Timedelta(minutes=10)
    idx = pd.to_datetime(index)
    diffs = pd.Series(idx[1:] - idx[:-1]).dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return pd.Timedelta(minutes=10)
    return pd.to_timedelta(diffs.median())


def _exact_state_segments(df: pd.DataFrame, state_col: str) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Return exact consecutive state intervals as half-open visual bands.

    x1 is the next segment's first timestamp, not the previous segment's last
    timestamp.  This avoids invisible one-bar state bands and makes the colored
    background cover the full time axis more naturally.
    """
    if df.empty:
        return []
    idx = pd.to_datetime(df.index)
    states = pd.to_numeric(df[state_col], errors="coerce").astype("Int64").to_numpy()
    valid = pd.notna(states)
    if not valid.all():
        tmp = df.loc[valid].copy()
        return _exact_state_segments(tmp, state_col)
    states = states.astype(int)
    if len(states) == 0:
        return []
    starts = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1]
    step = _infer_time_step(idx)
    segments: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    for i, start_pos in enumerate(starts):
        end_pos = int(starts[i + 1]) if i + 1 < len(starts) else len(states)
        x0 = pd.Timestamp(idx[start_pos])
        x1 = pd.Timestamp(idx[end_pos]) if end_pos < len(idx) else pd.Timestamp(idx[-1]) + step
        if x1 <= x0:
            x1 = x0 + step
        segments.append((x0, x1, int(states[start_pos])))
    return segments


def _compressed_state_segments(
    df: pd.DataFrame,
    state_col: str,
    max_shape_segments: int,
) -> tuple[list[tuple[pd.Timestamp, pd.Timestamp, int]], bool]:
    """Build state background bands with bounded shape count.

    If exact state switching creates too many bands, we partition the visible
    sample into at most ``max_shape_segments`` contiguous bins and use the modal
    state in each bin.  This preserves the visual background map without writing
    tens of thousands of Plotly layout shapes, which was the source of very slow
    HTML generation.

    Returns
    -------
    segments, compressed
        ``compressed`` is True when the exact state path was visually coarsened.
    """
    max_shape_segments = int(max_shape_segments or 0)
    exact = _exact_state_segments(df, state_col)
    if max_shape_segments <= 0:
        return [], bool(exact)
    if len(exact) <= max_shape_segments:
        return exact, False

    n = len(df)
    if n == 0:
        return [], False
    idx = pd.to_datetime(df.index)
    states = pd.to_numeric(df[state_col], errors="coerce").astype("Int64").to_numpy()
    valid = pd.notna(states)
    if not valid.all():
        tmp = df.loc[valid].copy()
        return _compressed_state_segments(tmp, state_col, max_shape_segments)
    states = states.astype(int)
    step = _infer_time_step(idx)

    n_bins = min(max_shape_segments, n)
    edges = np.linspace(0, n, n_bins + 1).round().astype(int)
    edges[0] = 0
    edges[-1] = n
    edges = np.unique(edges)
    if len(edges) < 2:
        return exact[:max_shape_segments], True

    raw: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
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

    # Merge adjacent visual bins with the same modal state; this further reduces
    # shape count while preserving state color continuity.
    merged: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    for x0, x1, st in raw:
        if merged and merged[-1][2] == st:
            prev_x0, _, _ = merged[-1]
            merged[-1] = (prev_x0, x1, st)
        else:
            merged.append((x0, x1, st))
    return merged, True


def _state_band_shapes(
    segments: list[tuple[pd.Timestamp, pd.Timestamp, int]],
    state_color: dict[int, str],
    band_alpha: float,
    *,
    xref: str = "x",
    yref: str = "paper",
) -> list[dict[str, Any]]:
    return [
        {
            "type": "rect",
            "xref": xref,
            "yref": yref,
            "x0": start,
            "x1": end,
            "y0": 0,
            "y1": 1,
            "fillcolor": _hex_to_rgba(state_color.get(state, "#888888"), band_alpha),
            "line": {"width": 0},
            "layer": "below",
        }
        for start, end, state in segments
    ]

def _state_colorscale(state_color: dict[int, str]) -> list[list[float | str]]:
    """Build a discrete Plotly colorscale for state values 1..5."""
    boundaries = {
        1: (0.00, 0.125),
        2: (0.125, 0.375),
        3: (0.375, 0.625),
        4: (0.625, 0.875),
        5: (0.875, 1.00),
    }
    scale: list[list[float | str]] = []
    for st in [1, 2, 3, 4, 5]:
        c = state_color.get(st, DEFAULT_STATE_COLORS.get(st, "#999999"))
        lo, hi = boundaries[st]
        scale.append([lo, c])
        scale.append([hi, c])
    return scale


def _add_state_heatmap_background(
    fig: go.Figure,
    dfp: pd.DataFrame,
    *,
    state_col: str,
    state_color: dict[int, str],
    y_min: float,
    y_max: float,
    opacity: float,
    max_background_points: int,
    row: int | None = None,
    col: int | None = None,
    secondary_y: bool | None = None,
) -> None:
    """Add a fast 5-state background layer.

    This uses one categorical heatmap trace instead of thousands of layout
    rectangles. It keeps the full 5-color state background visible while making
    HTML generation much faster. The background is downsampled only for display.
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
        colorscale=_state_colorscale(state_color),
        showscale=False,
        opacity=opacity,
        hoverinfo="skip",
        name="State background",
    )
    if row is None or col is None:
        fig.add_trace(trace)
    elif secondary_y is None:
        fig.add_trace(trace, row=row, col=col)
    else:
        fig.add_trace(trace, row=row, col=col, secondary_y=secondary_y)


def _state_customdata(dfp: pd.DataFrame, state_col: str) -> np.ndarray:
    state = pd.to_numeric(dfp[state_col], errors="coerce").astype("Int64")
    name_cn = [STATE_ROLE_MAP.get(int(v), {}).get("name_cn", "") if pd.notna(v) else "" for v in state]
    name_en = [STATE_ROLE_MAP.get(int(v), {}).get("name_en", "") if pd.notna(v) else "" for v in state]
    pressure_cn = [STATE_ROLE_MAP.get(int(v), {}).get("pressure_cn", "") if pd.notna(v) else "" for v in state]
    dir_expect = pd.to_numeric(dfp.get("dir_expect", pd.Series(np.nan, index=dfp.index)), errors="coerce").to_numpy()
    conf = pd.to_numeric(dfp.get("hmm_conf", pd.Series(np.nan, index=dfp.index)), errors="coerce").to_numpy()
    return np.column_stack([state.astype(str).to_numpy(), np.array(name_cn), np.array(name_en), np.array(pressure_cn), dir_expect, conf])


def _add_state_legend(fig: go.Figure, unique_states: list[int], state_color: dict[int, str], *, secondary_y: bool | None = None) -> None:
    for s in unique_states:
        trace = go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={"symbol": "square", "size": 10, "color": state_color[s]},
            name=f"State {s} - {STATE_ROLE_MAP.get(s, {}).get('name_cn', '')}",
        )
        if secondary_y is None:
            fig.add_trace(trace)
        else:
            fig.add_trace(trace, secondary_y=secondary_y)


def _apply_state_bands(fig: go.Figure, dfp: pd.DataFrame, state_col: str, state_color: dict[int, str], band_alpha: float, max_shape_segments: int) -> None:
    segments, compressed = _compressed_state_segments(dfp, state_col=state_col, max_shape_segments=max_shape_segments)
    if not segments:
        return
    existing = list(fig.layout.shapes) if fig.layout.shapes else []
    fig.update_layout(shapes=existing + _state_band_shapes(segments, state_color, band_alpha, yref="paper"))
    if compressed:
        fig.add_annotation(
            x=1.0,
            y=1.065,
            xref="paper",
            yref="paper",
            xanchor="right",
            showarrow=False,
            text=f"state background compressed to ≤{int(max_shape_segments)} bands for speed",
            font={"size": 11, "color": "rgba(232,234,237,0.70)"},
        )


def plot_price_with_hmm_states_html(
    df: pd.DataFrame,
    *,
    output_path: str | Path = "hmm_price_states.html",
    time_col: str | None = None,
    price_col: str = "price",
    state_col: str = "hmm_state",
    title: str | None = None,
    style: HMMVisStyle = HMMVisStyle(),
    max_points: Optional[int] = 60000,
    max_shape_segments: int = 1500,
) -> Path:
    dfp = _coerce_time_index(df, time_col=time_col)
    dfp = dfp.dropna(subset=[price_col, state_col])
    if dfp.empty:
        raise ValueError("Empty dataframe after dropping NaNs for price/state.")

    dfp[state_col] = pd.to_numeric(dfp[state_col], errors="coerce").astype("Int64")
    dfp = dfp.dropna(subset=[state_col])
    dfp[state_col] = dfp[state_col].astype(int)
    dfp = _thin_time_series(dfp, max_points=max_points)

    unique_states = sorted(dfp[state_col].unique().tolist())
    palette = qualitative.Dark24 if len(unique_states) <= len(qualitative.Dark24) else qualitative.Alphabet
    state_color = {s: DEFAULT_STATE_COLORS.get(s, palette[i % len(palette)]) for i, s in enumerate(unique_states)}

    if title is None:
        end_time = pd.to_datetime(dfp.index.max())
        title = f"HMM Regimes on Price — {end_time.strftime('%Y-%m-%d %H:%M')}"

    fig = go.Figure()

    y_series = pd.to_numeric(dfp[price_col], errors="coerce")
    full_price_range = _price_axis_range(y_series)
    if full_price_range is None:
        y_min = float(y_series.min())
        y_max = float(y_series.max())
    else:
        y_min, y_max = full_price_range

    initial_x_end = pd.Timestamp(dfp.index.max())
    initial_x_start = initial_x_end - pd.DateOffset(months=3)
    initial_y_range = _price_axis_range(y_series.loc[dfp.index >= initial_x_start]) or full_price_range

    _add_state_heatmap_background(
        fig,
        dfp,
        state_col=state_col,
        state_color=state_color,
        y_min=y_min,
        y_max=y_max,
        opacity=style.band_alpha,
        max_background_points=style.background_max_points,
    )

    PriceTrace = go.Scattergl if len(dfp) >= 15000 else go.Scatter
    custom = _state_customdata(dfp, state_col)
    fig.add_trace(
        PriceTrace(
            x=dfp.index,
            y=pd.to_numeric(dfp[price_col], errors="coerce"),
            mode="lines",
            name="Price",
            line={"color": style.price_color, "width": 2.0},
            customdata=custom,
            hovertemplate=(
                "%{x|%Y-%m-%d %H:%M}<br>Price=%{y:.2f}"
                "<br>State=%{customdata[0]} / %{customdata[1]}"
                "<br>%{customdata[2]}"
                "<br>%{customdata[3]}"
                "<br>dir_expect=%{customdata[4]:.4f}"
                "<br>hmm_conf=%{customdata[5]:.4f}<extra></extra>"
            ),
        )
    )

    # State background is rendered as a compact heatmap trace above. This keeps
    # the 5-color regime background visible without writing thousands of shapes.
    _add_state_legend(fig, unique_states, state_color)

    fig.update_layout(
        template=style.template,
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "y": 0.992,
            "yanchor": "top",
            "font": {"size": 24},
        },
        hovermode="x unified",
        height=980,
        autosize=True,
        margin={"l": 86, "r": 300, "t": 86, "b": 82},
        paper_bgcolor=style.bg,
        plot_bgcolor=style.bg,
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 0.965,
            "xanchor": "left",
            "x": 1.012,
            "bgcolor": "rgba(11,15,26,0.72)",
            "bordercolor": "rgba(255,255,255,0.10)",
            "borderwidth": 1,
            "font": {"size": 13},
            "itemsizing": "constant",
        },
        font={"family": "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Arial", "color": "#e8eaed", "size": 14},
    )

    fig.update_xaxes(
        rangeselector={
            "buttons": _rangeselector_buttons(),
            "x": 0.0,
            "xanchor": "left",
            "y": 1.018,
            "yanchor": "bottom",
            "font": {"size": 12, "color": "#e8eaed"},
            "bgcolor": "rgba(255,255,255,0.06)",
            "activecolor": "rgba(0,229,255,0.18)",
            "bordercolor": "rgba(255,255,255,0.12)",
            "borderwidth": 1,
        },
        rangeslider={
            "visible": True,
            "thickness": 0.045,
            "bgcolor": "rgba(255,255,255,0.03)",
            "bordercolor": "rgba(255,255,255,0.10)",
            "borderwidth": 1,
        },
        range=[initial_x_start, initial_x_end],
        type="date",
        showgrid=True,
        gridcolor=style.grid,
        zeroline=False,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="rgba(255,255,255,0.35)",
        spikethickness=1,
        tickfont={"size": 14},
    )
    yaxis_kwargs: dict[str, Any] = {
        "showgrid": True,
        "gridcolor": style.grid,
        "zeroline": False,
        "title_text": "Price",
        "title_standoff": 14,
        "automargin": True,
        "fixedrange": False,
        "rangemode": "normal",
        "tickfont": {"size": 14},
    }
    if initial_y_range is not None:
        yaxis_kwargs.update({"range": [initial_y_range[0], initial_y_range[1]], "autorange": False})
    fig.update_yaxes(**yaxis_kwargs)

    out_path = Path(output_path)
    _write_html(
        fig,
        out_path,
        post_script=_viewport_height_script() + _visible_price_axis_autorange_script(),
    )
    return out_path


def plot_price_and_exp_ret_with_hmm_states_html(
    df: pd.DataFrame,
    *,
    output_path: str | Path = "hmm_price_exp_ret_30m.html",
    time_col: str | None = None,
    price_col: str = "price",
    exp_ret_col: str = "exp_ret_30m",
    state_col: str = "hmm_state",
    title: str | None = None,
    style: HMMVisStyle = HMMVisStyle(),
    max_points: Optional[int] = 60000,
    max_shape_segments: int = 1500,
) -> Path:
    dfp = _coerce_time_index(df, time_col=time_col)
    dfp = dfp.dropna(subset=[price_col, exp_ret_col, state_col])
    if dfp.empty:
        raise ValueError("Empty dataframe after dropping NaNs for price/exp_ret/state.")

    dfp[state_col] = pd.to_numeric(dfp[state_col], errors="coerce").astype("Int64")
    dfp = dfp.dropna(subset=[state_col])
    dfp[state_col] = dfp[state_col].astype(int)
    dfp = _thin_time_series(dfp, max_points=max_points)

    unique_states = sorted(dfp[state_col].unique().tolist())
    palette = qualitative.Dark24 if len(unique_states) <= len(qualitative.Dark24) else qualitative.Alphabet
    state_color = {s: DEFAULT_STATE_COLORS.get(s, palette[i % len(palette)]) for i, s in enumerate(unique_states)}

    if title is None:
        end_time = pd.to_datetime(dfp.index.max())
        title = f"HMM Regimes — Price & Exp. Return — {end_time.strftime('%Y-%m-%d %H:%M')}"

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    y_series = pd.to_numeric(dfp[price_col], errors="coerce")
    _add_state_heatmap_background(
        fig,
        dfp,
        state_col=state_col,
        state_color=state_color,
        y_min=float(y_series.min()),
        y_max=float(y_series.max()),
        opacity=style.band_alpha,
        max_background_points=style.background_max_points,
        row=1,
        col=1,
        secondary_y=False,
    )

    Trace = go.Scattergl if len(dfp) >= 15000 else go.Scatter
    custom = _state_customdata(dfp, state_col)
    fig.add_trace(
        Trace(
            x=dfp.index,
            y=pd.to_numeric(dfp[price_col], errors="coerce"),
            mode="lines",
            name="Price",
            line={"color": style.price_color, "width": 1.25},
            customdata=custom,
            hovertemplate=(
                "%{x|%Y-%m-%d %H:%M}<br>Price=%{y:.2f}"
                "<br>State=%{customdata[0]} / %{customdata[1]}"
                "<br>%{customdata[2]}<br>%{customdata[3]}<extra></extra>"
            ),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        Trace(
            x=dfp.index,
            y=pd.to_numeric(dfp[exp_ret_col], errors="coerce"),
            mode="lines",
            name="Exp. Return (30m)",
            line={"color": "#00e5ff", "width": 1.1, "dash": "dot"},
        ),
        secondary_y=True,
    )

    # State background is rendered as a compact heatmap trace above.
    _add_state_legend(fig, unique_states, state_color, secondary_y=False)

    fig.update_layout(
        template=style.template,
        title=title,
        hovermode="x unified",
        height=720,
        margin={"l": 70, "r": 190, "t": 120, "b": 60},
        paper_bgcolor=style.bg,
        plot_bgcolor=style.bg,
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 1.0,
            "xanchor": "left",
            "x": 1.02,
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"size": 11},
            "itemsizing": "constant",
        },
        font={"family": "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Arial", "color": "#e8eaed"},
    )

    fig.update_xaxes(
        rangeselector={
            "buttons": _rangeselector_buttons(),
            "x": 0.0,
            "xanchor": "left",
            "y": 1.18,
            "yanchor": "top",
            "font": {"size": 11, "color": "#e8eaed"},
            "bgcolor": "rgba(255,255,255,0.06)",
            "activecolor": "rgba(0,229,255,0.18)",
            "bordercolor": "rgba(255,255,255,0.12)",
            "borderwidth": 1,
        },
        rangeslider={
            "visible": True,
            "thickness": 0.06,
            "bgcolor": "rgba(255,255,255,0.03)",
            "bordercolor": "rgba(255,255,255,0.10)",
            "borderwidth": 1,
        },
        type="date",
        showgrid=True,
        gridcolor=style.grid,
        zeroline=False,
    )
    fig.update_yaxes(showgrid=True, gridcolor=style.grid, zeroline=False, secondary_y=False, title_text="Price")
    fig.update_yaxes(showgrid=False, zeroline=False, secondary_y=True, title_text="Exp. Return (30m)")

    out_path = Path(output_path)
    _write_html(fig, out_path)
    return out_path
