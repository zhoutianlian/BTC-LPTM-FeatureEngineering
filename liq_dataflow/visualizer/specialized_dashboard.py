from __future__ import annotations

import html
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from liq_dataflow.visualizer.custom_style_matplotlib import (
    BIN_COLORS,
    EVENT_BUY,
    EVENT_SELL,
    render_rpn_dominance_png,
    render_rpn_features_png,
)
from liq_dataflow.visualizer.html_assets import FONT, ensure_plotly_bundle


PAGE_BG = "#050b14"
PLOT_BG = "#0b1524"
PANEL_BG = "#08111f"
TEXT = "#e5edf7"
MUTED = "#94a3b8"
GRID = "rgba(148,163,184,0.18)"
AXIS_LINE = "rgba(148,163,184,0.40)"
PRICE_LINE = "#f4c542"
RPN_LINE = "#60a5fa"
FLL_LINE = "#ff6b5f"
FSL_LINE = "#a3e635"
FLL_BAND = "#ff5f6d"
FSL_BAND = "#38d996"


def _rgba(value: str, alpha: float) -> str:
    value = value.lstrip("#")
    if len(value) != 6:
        return value
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _clean_numeric(values: pd.Series | Iterable[object]) -> pd.Series:
    return pd.to_numeric(pd.Series(values), errors="coerce").astype(float)


def _masked(values: pd.Series, mask: pd.Series) -> pd.Series:
    out = pd.to_numeric(values, errors="coerce").astype(float).copy()
    out.loc[~mask.fillna(False).astype(bool)] = np.nan
    return out


def _base_layout(height: int) -> dict:
    return dict(
        template="none",
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font={"family": FONT, "size": 12, "color": TEXT},
        margin={"l": 78, "r": 245, "t": 48, "b": 104},
        height=height,
        hovermode="x unified",
        hoverlabel={"bgcolor": "rgba(8,17,31,0.96)", "bordercolor": "rgba(148,163,184,0.35)", "font_size": 12, "font_family": FONT, "font_color": TEXT},
        legend={
            "orientation": "v",
            "x": 1.015,
            "xanchor": "left",
            "y": 1.0,
            "yanchor": "top",
            "bgcolor": "rgba(8,17,31,0.78)",
            "bordercolor": "rgba(148,163,184,0.22)",
            "borderwidth": 1,
            "font": {"size": 11, "color": TEXT},
            "tracegroupgap": 6,
            "itemclick": "toggleothers",
            "itemdoubleclick": "toggle",
        },
        uirevision="liq_specialized_full_history_v2",
    )


def _style_axes(fig: go.Figure) -> None:
    fig.update_xaxes(type="date", showgrid=True, gridcolor=GRID, linecolor=AXIS_LINE, showline=True, zeroline=False, tickfont={"color": MUTED}, title_font={"color": MUTED})
    fig.update_yaxes(showgrid=True, gridcolor=GRID, linecolor=AXIS_LINE, showline=True, zeroline=False, tickfont={"color": MUTED}, title_font={"color": MUTED})
    fig.update_annotations(font={"size": 20, "color": TEXT})


def _report_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["time"] = pd.to_datetime(data["time"], errors="coerce")
    return data.dropna(subset=["time"]).sort_values("time").drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)


def _axis_range(values: pd.Series, *, pad: float = 0.05, min_pad: float = 0.0, floor: float | None = None, positive_floor: bool = False) -> list[float] | None:
    vals = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if vals.empty:
        return None
    lo = float(vals.min())
    hi = float(vals.max())
    span = max(hi - lo, min_pad)
    out_lo = lo - span * pad
    out_hi = hi + span * pad
    if floor is not None:
        out_lo = min(out_lo, floor)
    if positive_floor and lo >= 0:
        out_lo = max(out_lo, lo * 0.70)
    return [out_lo, out_hi]


def _price_range(price: pd.Series) -> list[float] | None:
    return _axis_range(price, pad=0.04, min_pad=1000.0)


def _diff_range(*series: pd.Series) -> list[float] | None:
    vals = pd.concat([pd.to_numeric(s, errors="coerce") for s in series], ignore_index=True).dropna().astype(float)
    if vals.empty:
        return None
    lo = min(float(vals.min()), 0.0)
    hi = max(float(vals.max()), 0.0)
    span = max(hi - lo, 0.5)
    return [lo - span * 0.08, hi + span * 0.08]


def _time_range(data: pd.DataFrame) -> list[pd.Timestamp]:
    t = pd.to_datetime(data["time"], errors="coerce")
    return [pd.to_datetime(t.min()), pd.to_datetime(t.max())]


def _range_selector() -> dict:
    return {
        "buttons": [
            {"count": 1, "label": "1M", "step": "month", "stepmode": "backward"},
            {"count": 3, "label": "3M", "step": "month", "stepmode": "backward"},
            {"count": 6, "label": "6M", "step": "month", "stepmode": "backward"},
            {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
            {"step": "all", "label": "All"},
        ],
        "bgcolor": "rgba(15,23,42,0.86)",
        "activecolor": "rgba(96,165,250,0.36)",
        "bordercolor": "rgba(148,163,184,0.25)",
        "font": {"color": TEXT, "size": 12},
        "x": 0.0,
        "y": 1.11,
    }


def _add_between_fill(fig: go.Figure, *, x: pd.Series, upper: pd.Series, lower: pd.Series, color: str, name: str, row: int, showlegend: bool = False, opacity: float = 0.12) -> None:
    fig.add_trace(go.Scatter(x=x, y=upper, mode="lines", line={"width": 0.01, "color": _rgba(color, 0.01)}, hoverinfo="skip", showlegend=False, connectgaps=False), row=row, col=1, secondary_y=True)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=lower,
            mode="lines",
            line={"width": 0.01, "color": _rgba(color, 0.01)},
            fill="tonexty",
            fillcolor=_rgba(color, opacity),
            name=name,
            legendgroup=name,
            showlegend=showlegend,
            connectgaps=False,
            hoverinfo="skip",
        ),
        row=row,
        col=1,
        secondary_y=True,
    )


def _add_segmented_band_fill(fig: go.Figure, *, x: pd.Series, upper: pd.Series, lower: pd.Series, mask: pd.Series, color: str, name: str, row: int, opacity: float) -> None:
    x_values = pd.Series(x).reset_index(drop=True)
    upper_values = pd.to_numeric(pd.Series(upper), errors="coerce").reset_index(drop=True)
    lower_values = pd.to_numeric(pd.Series(lower), errors="coerce").reset_index(drop=True)
    active = pd.Series(mask).fillna(False).astype(bool).reset_index(drop=True) & upper_values.notna() & lower_values.notna() & x_values.notna()

    xs: list[object] = []
    ys: list[object] = []
    start: int | None = None

    def append_segment(end: int) -> None:
        if start is None or end - start < 2:
            return
        seg_x = x_values.iloc[start:end].tolist()
        seg_upper = upper_values.iloc[start:end].tolist()
        seg_lower = lower_values.iloc[start:end].tolist()
        xs.extend(seg_x)
        ys.extend(seg_upper)
        xs.extend(reversed(seg_x))
        ys.extend(reversed(seg_lower))
        xs.append(None)
        ys.append(None)

    for idx, is_active in enumerate(active.tolist()):
        if is_active and start is None:
            start = idx
        elif not is_active and start is not None:
            append_segment(idx)
            start = None
    if start is not None:
        append_segment(len(active))

    if not xs:
        return

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            fill="toself",
            fillcolor=_rgba(color, opacity),
            line={"width": 0.01, "color": _rgba(color, 0.02)},
            name=name,
            legendgroup=name,
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=1,
        secondary_y=True,
    )


def _dominance_segments(data: pd.DataFrame) -> list[tuple[int, pd.Timestamp, pd.Timestamp]]:
    if data.empty or "dominance" not in data.columns:
        return []
    t = pd.to_datetime(data["time"], errors="coerce")
    dom = pd.to_numeric(data["dominance"], errors="coerce").fillna(0).astype(int)
    segments: list[tuple[int, pd.Timestamp, pd.Timestamp]] = []
    start_idx = 0
    current = int(dom.iloc[0])
    for idx in range(1, len(data)):
        value = int(dom.iloc[idx])
        if value == current:
            continue
        if current != 0:
            segments.append((current, pd.to_datetime(t.iloc[start_idx]), pd.to_datetime(t.iloc[idx])))
        start_idx = idx
        current = value
    if current != 0:
        segments.append((current, pd.to_datetime(t.iloc[start_idx]), pd.to_datetime(t.iloc[-1])))
    return segments


def _add_dominance_background(fig: go.Figure, data: pd.DataFrame, *, rows: Iterable[int], opacity: float = 0.14) -> None:
    colors = {-1: FLL_BAND, 1: FSL_BAND}
    refs = {
        1: ("x", "y domain"),
        2: ("x2", "y3 domain"),
    }
    shapes = list(fig.layout.shapes or [])
    for value, x0, x1 in _dominance_segments(data):
        if x1 <= x0:
            continue
        for row in rows:
            xref, yref = refs[row]
            shapes.append(
                {
                    "type": "rect",
                    "xref": xref,
                    "yref": yref,
                    "x0": x0,
                    "x1": x1,
                    "y0": 0,
                    "y1": 1,
                    "fillcolor": _rgba(colors.get(value, "#64748b"), opacity),
                    "line": {"width": 0},
                    "layer": "below",
                }
            )
    fig.update_layout(shapes=shapes)


def _add_dominance_legend(fig: go.Figure, *, row: int) -> None:
    for name, color in [("FLL Dominant", FLL_BAND), ("FSL Dominant", FSL_BAND)]:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line={"color": _rgba(color, 0.45), "width": 10},
                name=name,
                legendgroup=name,
                hoverinfo="skip",
            ),
            row=row,
            col=1,
            secondary_y=False,
        )


def _add_rpn_bin_layer(fig: go.Figure, *, x: pd.Series, rpn: pd.Series, bins: pd.Series, row: int) -> None:
    bin_values = pd.to_numeric(bins, errors="coerce").fillna(4).astype(int)
    for bin_id in sorted(BIN_COLORS):
        mask = bin_values.eq(bin_id)
        y = _masked(rpn, mask)
        if y.notna().sum() == 0:
            continue
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                mode="lines+markers",
                name=f"bin{bin_id}",
                legendgroup=f"bin{bin_id}",
                line={"color": BIN_COLORS[bin_id], "width": 1.2, "dash": "dot"},
                marker={"color": BIN_COLORS[bin_id], "size": 2.6, "opacity": 0.82},
                connectgaps=False,
                hovertemplate=f"%{{x}}<br>RPN=%{{y:.4f}}<br>bin={bin_id}<extra></extra>",
            ),
            row=row,
            col=1,
            secondary_y=True,
        )


def _add_diff_scatter(fig: go.Figure, *, x: pd.Series, diff: pd.Series, row: int, showlegend: bool = True, use_gl: bool = True) -> None:
    trace_cls = go.Scattergl if use_gl else go.Scatter
    pos = diff.ge(0)
    neg = diff.lt(0)
    if pos.any():
        fig.add_trace(
            trace_cls(
                x=x[pos],
                y=diff[pos],
                mode="markers",
                name="FLL Dominant",
                legendgroup="FLL Dominant",
                showlegend=showlegend,
                marker={"color": FLL_LINE, "size": 2.7, "opacity": 0.68},
                hovertemplate="%{x}<br>diff=%{y:.4f}<extra>FLL Dominant</extra>",
            ),
            row=row,
            col=1,
            secondary_y=True,
        )
    if neg.any():
        fig.add_trace(
            trace_cls(
                x=x[neg],
                y=diff[neg],
                mode="markers",
                name="FSL Dominant",
                legendgroup="FSL Dominant",
                showlegend=showlegend,
                marker={"color": FSL_LINE, "size": 2.7, "opacity": 0.68},
                hovertemplate="%{x}<br>diff=%{y:.4f}<extra>FSL Dominant</extra>",
            ),
            row=row,
            col=1,
            secondary_y=True,
        )


def _add_event_lines(fig: go.Figure, *, data: pd.DataFrame, row: int) -> None:
    refs = {
        1: ("x", "y domain"),
        2: ("x2", "y3 domain"),
    }
    xref, yref = refs[row]
    shapes = list(fig.layout.shapes or [])
    if "hit_ceiling_bottom" in data.columns:
        hits = pd.to_numeric(data["hit_ceiling_bottom"], errors="coerce").fillna(0)
        for value, color in [(1, EVENT_BUY), (-1, EVENT_SELL)]:
            for time in data.loc[hits.eq(value), "time"]:
                shapes.append(
                    {
                        "type": "line",
                        "xref": xref,
                        "yref": yref,
                        "x0": time,
                        "x1": time,
                        "y0": 0,
                        "y1": 1,
                        "line": {"color": color, "width": 0.8},
                        "opacity": 0.55,
                        "layer": "below",
                    }
                )
    if "reverse_ceiling_bottom" in data.columns:
        reverses = pd.to_numeric(data["reverse_ceiling_bottom"], errors="coerce").fillna(0)
        for value, color in [(1, EVENT_SELL), (-1, EVENT_BUY)]:
            for time in data.loc[reverses.eq(value), "time"]:
                shapes.append(
                    {
                        "type": "line",
                        "xref": xref,
                        "yref": yref,
                        "x0": time,
                        "x1": time,
                        "y0": 0,
                        "y1": 1,
                        "line": {"color": color, "dash": "dash", "width": 0.8},
                        "opacity": 0.55,
                        "layer": "below",
                    }
                )
    fig.update_layout(shapes=shapes)


def _add_diff_sign_band(fig: go.Figure, *, x: pd.Series, fll: pd.Series, fsl: pd.Series, row: int) -> None:
    diff = fll - fsl
    _add_segmented_band_fill(fig, x=x, upper=fll, lower=fsl, mask=diff.ge(0), color=FLL_BAND, name="FLL > FSL Band", row=row, opacity=0.16)
    _add_segmented_band_fill(fig, x=x, upper=fsl, lower=fll, mask=diff.lt(0), color=FSL_BAND, name="FSL > FLL Band", row=row, opacity=0.16)


def _dominance_figure(df: pd.DataFrame, *, default_view_months: int, max_points: int) -> tuple[go.Figure, pd.Timestamp, dict[str, int]]:
    data = _report_data(df)
    if data.empty:
        raise ValueError("No data available for dominance dashboard.")

    t = pd.to_datetime(data["time"], errors="coerce")
    current_time = pd.to_datetime(t.max())
    stamp = current_time.strftime("%Y-%m-%d %H:%M")

    price = _clean_numeric(data["price"])
    rpn = _clean_numeric(data["risk_priority_number"]).clip(lower=0.0, upper=1.0)
    bins = _clean_numeric(data.get("bin_index", pd.Series(index=data.index))).fillna(4).astype(int)
    diff = _clean_numeric(data["diff_ls_cwt_kf"])
    dom = _clean_numeric(data.get("dominance", pd.Series(index=data.index))).fillna(0).astype(int)
    thr_pos = _clean_numeric(data.get("thr_diff_pos", pd.Series(index=data.index)))
    thr_neg = _clean_numeric(data.get("thr_diff_neg", pd.Series(index=data.index)))
    hit_vals = _clean_numeric(data.get("hit_ceiling_bottom", pd.Series(index=data.index))).fillna(0)
    rev_vals = _clean_numeric(data.get("reverse_ceiling_bottom", pd.Series(index=data.index))).fillna(0)

    event_counts = {
        "hit_ceiling_bottom": int(hit_vals.ne(0).sum()),
        "reverse_ceiling_bottom": int(rev_vals.ne(0).sum()),
    }

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.075,
        row_heights=[0.50, 0.50],
        subplot_titles=(
            f"BTC Price Colored by Risk Priority Number Bins With Dominance - {stamp}",
            f"Price & FLL FSL Diff - {stamp}",
        ),
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
    )

    _add_dominance_background(fig, data, rows=[1, 2], opacity=0.18)
    fig.add_trace(go.Scattergl(x=t, y=price, mode="lines", name="Price", line={"color": PRICE_LINE, "width": 1.35}, hovertemplate="%{x}<br>Price=%{y:,.2f}<extra></extra>"), row=1, col=1, secondary_y=False)
    _add_dominance_legend(fig, row=1)
    _add_rpn_bin_layer(fig, x=t, rpn=rpn, bins=bins, row=1)

    fig.add_trace(go.Scattergl(x=t, y=price, mode="lines", name="Price", legendgroup="Price", line={"color": PRICE_LINE, "width": 1.0}, opacity=0.72, showlegend=False, hovertemplate="%{x}<br>Price=%{y:,.2f}<extra></extra>"), row=2, col=1, secondary_y=False)
    _add_diff_scatter(fig, x=t, diff=diff, row=2, showlegend=False)
    if thr_pos.notna().sum() > 0:
        fig.add_trace(go.Scattergl(x=t, y=thr_pos, mode="lines", name="thr_diff_pos", line={"color": FLL_LINE, "width": 1.0, "dash": "dot"}, opacity=0.72, hovertemplate="%{x}<br>thr_diff_pos=%{y:.4f}<extra></extra>"), row=2, col=1, secondary_y=True)
    if thr_neg.notna().sum() > 0:
        fig.add_trace(go.Scattergl(x=t, y=thr_neg, mode="lines", name="thr_diff_neg", line={"color": FSL_LINE, "width": 1.0, "dash": "dot"}, opacity=0.72, hovertemplate="%{x}<br>thr_diff_neg=%{y:.4f}<extra></extra>"), row=2, col=1, secondary_y=True)

    fig.update_layout(**_base_layout(height=1500))
    _style_axes(fig)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(rangeselector=_range_selector(), range=_time_range(data), row=1, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Time", tickformat="%Y-%m", tickangle=-35, range=_time_range(data), rangeslider={"visible": True, "thickness": 0.055, "bgcolor": "rgba(15,23,42,0.78)", "bordercolor": "rgba(148,163,184,0.20)", "borderwidth": 1}, row=2, col=1)
    fig.update_yaxes(title_text="Price", title_font={"color": PRICE_LINE}, tickfont={"color": PRICE_LINE}, row=1, col=1, secondary_y=False)
    fig.update_yaxes(range=_price_range(price), row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="risk_priority_number", title_font={"color": RPN_LINE}, tickfont={"color": RPN_LINE}, row=1, col=1, secondary_y=True, range=_axis_range(rpn, pad=0.05, min_pad=0.10, positive_floor=True))
    fig.update_yaxes(title_text="Price", title_font={"color": PRICE_LINE}, tickfont={"color": PRICE_LINE}, row=2, col=1, secondary_y=False)
    fig.update_yaxes(range=_price_range(price), row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="diff_ls_cwt_kf", title_font={"color": RPN_LINE}, tickfont={"color": RPN_LINE}, range=_diff_range(diff, thr_pos, thr_neg), row=2, col=1, secondary_y=True)
    return fig, current_time, event_counts

def _feature_figure(df: pd.DataFrame, *, default_view_months: int, max_points: int) -> tuple[go.Figure, pd.Timestamp, dict[str, int]]:
    data = _report_data(df)
    if data.empty:
        raise ValueError("No data available for feature stack dashboard.")

    t = pd.to_datetime(data["time"], errors="coerce")
    current_time = pd.to_datetime(t.max())
    stamp = current_time.strftime("%Y-%m-%d %H:%M")

    price = _clean_numeric(data["price"])
    fll = _clean_numeric(data["fll_cwt_kf"])
    fsl = _clean_numeric(data["fsl_cwt_kf"])
    fll_high = _clean_numeric(data.get("fll_rolling_high", pd.Series(index=data.index)))
    fsl_high = _clean_numeric(data.get("fsl_rolling_high", pd.Series(index=data.index)))
    fsl_med = _clean_numeric(data.get("fsl_rolling_median", pd.Series(index=data.index)))
    if fsl_med.isna().all():
        fsl_med = fsl.rolling(window=min(365 * 24, max(24, len(fsl))), min_periods=1).quantile(0.5)
    fsl_low = _clean_numeric(data.get("fsl_rolling_low", pd.Series(index=data.index)))
    if fsl_low.isna().all():
        fsl_low = fsl.rolling(window=min(365 * 24, max(24, len(fsl))), min_periods=1).quantile(0.25)
    diff = _clean_numeric(data["diff_ls_cwt_kf"])
    thr_pos = _clean_numeric(data.get("thr_diff_pos", pd.Series(index=data.index)))
    thr_neg = _clean_numeric(data.get("thr_diff_neg", pd.Series(index=data.index)))
    cases = data.get("corr_case", pd.Series(index=data.index, dtype="object")).fillna("None").astype(str)
    hit_vals = _clean_numeric(data.get("hit_ceiling_bottom", pd.Series(index=data.index))).fillna(0)
    rev_vals = _clean_numeric(data.get("reverse_ceiling_bottom", pd.Series(index=data.index))).fillna(0)

    event_counts = {"hit_ceiling_bottom": int(hit_vals.ne(0).sum()), "reverse_ceiling_bottom": int(rev_vals.ne(0).sum())}

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.075,
        row_heights=[0.50, 0.50],
        subplot_titles=(
            f"BTC Price Colored by Risk Priority Number Bins - {stamp}",
            f"Price & FLL FSL Diff - {stamp}",
        ),
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
    )

    _add_dominance_background(fig, data, rows=[2], opacity=0.15)
    fig.add_trace(go.Scatter(x=t, y=price, mode="lines", name="BTC Price", line={"color": PRICE_LINE, "width": 1.2}, hovertemplate="%{x}<br>Price=%{y:,.2f}<extra></extra>"), row=1, col=1, secondary_y=False)
    _add_diff_sign_band(fig, x=t, fll=fll, fsl=fsl, row=1)
    fig.add_trace(go.Scatter(x=t, y=fll, mode="lines", name="fll_cwt_kf", line={"color": FSL_LINE, "width": 1.35}, opacity=0.82, hovertemplate="%{x}<br>fll_cwt_kf=%{y:.4f}<extra></extra>"), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=t, y=fsl, mode="lines", name="fsl_cwt_kf", line={"color": FLL_LINE, "width": 1.35}, opacity=0.82, hovertemplate="%{x}<br>fsl_cwt_kf=%{y:.4f}<extra></extra>"), row=1, col=1, secondary_y=True)
    if fll_high.notna().sum() > 0:
        fig.add_trace(go.Scatter(x=t, y=fll_high, mode="lines", name="fll_rolling_high", line={"color": FSL_LINE, "width": 0.8}, opacity=0.58, hovertemplate="%{x}<br>fll_rolling_high=%{y:.4f}<extra></extra>"), row=1, col=1, secondary_y=True)
    if fsl_high.notna().sum() > 0:
        fig.add_trace(go.Scatter(x=t, y=fsl_high, mode="lines", name="fsl_rolling_high", line={"color": FLL_LINE, "width": 0.8}, opacity=0.58, hovertemplate="%{x}<br>fsl_rolling_high=%{y:.4f}<extra></extra>"), row=1, col=1, secondary_y=True)
    if fsl_med.notna().sum() > 0:
        fig.add_trace(go.Scatter(x=t, y=fsl_med, mode="lines", name="fsl_rolling_median", line={"color": "#fca5a5", "width": 0.8}, opacity=0.52, hovertemplate="%{x}<br>fsl_rolling_median=%{y:.4f}<extra></extra>"), row=1, col=1, secondary_y=True)
    if fsl_low.notna().sum() > 0:
        fig.add_trace(go.Scatter(x=t, y=fsl_low, mode="lines", name="fsl_rolling_low", line={"color": "#fecaca", "width": 0.8}, opacity=0.45, hovertemplate="%{x}<br>fsl_rolling_low=%{y:.4f}<extra></extra>"), row=1, col=1, secondary_y=True)

    _add_dominance_legend(fig, row=2)
    _add_event_lines(fig, data=data, row=2)
    fig.add_trace(go.Scatter(x=t, y=price, mode="lines", name="Price", line={"color": PRICE_LINE, "width": 1.0}, opacity=0.72, showlegend=False, hovertemplate="%{x}<br>Price=%{y:,.2f}<extra></extra>"), row=2, col=1, secondary_y=False)
    _add_diff_scatter(fig, x=t, diff=diff, row=2, use_gl=False)
    if thr_pos.notna().sum() > 0:
        fig.add_trace(go.Scatter(x=t, y=thr_pos, mode="lines", name="thr_diff_pos", line={"color": FLL_LINE, "width": 1.0}, opacity=0.72, hovertemplate="%{x}<br>thr_diff_pos=%{y:.4f}<extra></extra>"), row=2, col=1, secondary_y=True)
    if thr_neg.notna().sum() > 0:
        fig.add_trace(go.Scatter(x=t, y=thr_neg, mode="lines", name="thr_diff_neg", line={"color": FSL_LINE, "width": 1.0}, opacity=0.72, hovertemplate="%{x}<br>thr_diff_neg=%{y:.4f}<extra></extra>"), row=2, col=1, secondary_y=True)

    fig.update_layout(**_base_layout(height=1500))
    _style_axes(fig)
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(rangeselector=_range_selector(), range=_time_range(data), row=1, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Time", tickformat="%Y-%m", tickangle=-35, range=_time_range(data), rangeslider={"visible": True, "thickness": 0.055, "bgcolor": "rgba(15,23,42,0.78)", "bordercolor": "rgba(148,163,184,0.20)", "borderwidth": 1}, row=2, col=1)
    fig.update_yaxes(title_text="Price", title_font={"color": PRICE_LINE}, tickfont={"color": PRICE_LINE}, row=1, col=1, secondary_y=False)
    fig.update_yaxes(range=_price_range(price), row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="fll_cwt_kf", title_font={"color": FSL_LINE}, tickfont={"color": FSL_LINE}, range=_axis_range(pd.concat([fll, fsl, fll_high, fsl_high, fsl_med, fsl_low], ignore_index=True), pad=0.06, min_pad=1.0, positive_floor=True), row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Price", title_font={"color": PRICE_LINE}, tickfont={"color": PRICE_LINE}, row=2, col=1, secondary_y=False)
    fig.update_yaxes(range=_price_range(price), row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="diff_ls_cwt_kf", title_font={"color": RPN_LINE}, tickfont={"color": RPN_LINE}, range=_diff_range(diff, thr_pos, thr_neg), row=2, col=1, secondary_y=True)
    return fig, current_time, event_counts


def _html_page(*, title: str, plot_html: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <script src="plotly.min.js"></script>
  <style>
    :root {{ color-scheme: dark; }}
    html, body {{ margin: 0; padding: 0; background: {PAGE_BG}; }}
    body {{ font-family: {FONT}; color: {TEXT}; }}
    .report-figure {{
      width: min(2240px, calc(100vw - 20px));
      margin: 0 auto;
      padding: 10px 10px 16px;
      box-sizing: border-box;
      background:
        radial-gradient(circle at 12% 0%, rgba(96,165,250,0.10), transparent 28%),
        linear-gradient(180deg, {PANEL_BG}, {PAGE_BG});
    }}
    .plotly-graph-div {{ width: 100% !important; }}
  </style>
</head>
<body>
  <main class="report-figure">{plot_html}</main>
</body>
</html>
"""


def _autorange_post_script() -> str:
    return r"""
(function() {
  const gd = document.getElementById('{plot_id}');
  if (!gd) return;

  const axisByTraceId = {y: 'yaxis', y2: 'yaxis2', y3: 'yaxis3', y4: 'yaxis4'};
  const rules = {
    yaxis: {pad: 0.045, minPad: 1000},
    yaxis2: {pad: 0.060, minPad: 0.10, positiveFloor: true},
    yaxis3: {pad: 0.045, minPad: 1000},
    yaxis4: {pad: 0.090, minPad: 0.50, includeZero: true}
  };
  let updating = false;

  function toMs(value) {
    const out = Date.parse(value);
    return Number.isFinite(out) ? out : null;
  }

  function fullXRange() {
    let lo = Infinity;
    let hi = -Infinity;
    (gd.data || []).forEach((trace) => {
      (trace.x || []).forEach((value) => {
        const ms = toMs(value);
        if (ms === null) return;
        lo = Math.min(lo, ms);
        hi = Math.max(hi, ms);
      });
    });
    return Number.isFinite(lo) && Number.isFinite(hi) ? [lo, hi] : null;
  }

  function parseRange(range) {
    if (range && range.length === 2) {
      const lo = toMs(range[0]);
      const hi = toMs(range[1]);
      if (lo !== null && hi !== null) return [lo, hi];
    }
    return null;
  }

  function eventXRange(eventData) {
    if (!eventData) return null;
    const pairs = [
      ['xaxis.range[0]', 'xaxis.range[1]'],
      ['xaxis2.range[0]', 'xaxis2.range[1]']
    ];
    for (const [loKey, hiKey] of pairs) {
      if (Object.prototype.hasOwnProperty.call(eventData, loKey) && Object.prototype.hasOwnProperty.call(eventData, hiKey)) {
        const parsed = parseRange([eventData[loKey], eventData[hiKey]]);
        if (parsed) return parsed;
      }
    }
    const packed = parseRange(eventData['xaxis.range']) || parseRange(eventData['xaxis2.range']);
    if (packed) return packed;
    if (eventData['xaxis.autorange'] || eventData['xaxis2.autorange']) return fullXRange();
    return null;
  }

  function currentXRange(eventData) {
    const fromEvent = eventXRange(eventData);
    if (fromEvent) return fromEvent;
    const layout = gd.layout || {};
    const candidates = [
      layout.xaxis && layout.xaxis.range,
      layout.xaxis2 && layout.xaxis2.range
    ];
    for (const range of candidates) {
      const parsed = parseRange(range);
      if (parsed) return parsed;
    }
    return fullXRange();
  }

  function padded(values, rule) {
    const nums = values.filter((value) => Number.isFinite(value));
    if (!nums.length) return null;
    let lo = Math.min.apply(null, nums);
    let hi = Math.max.apply(null, nums);
    if (rule.includeZero) {
      lo = Math.min(lo, 0);
      hi = Math.max(hi, 0);
    }
    const span = Math.max(hi - lo, rule.minPad || 1);
    let outLo = lo - span * (rule.pad || 0.05);
    if (rule.positiveFloor && lo >= 0) outLo = Math.max(outLo, lo * 0.70);
    return [outLo, hi + span * (rule.pad || 0.05)];
  }

  function hasXRelayout(eventData) {
    if (!eventData) return true;
    return Object.keys(eventData).some((key) => key.startsWith('xaxis') || key === 'autosize');
  }

  function updateYRanges(eventData) {
    if (updating || !hasXRelayout(eventData)) return;
    const xr = currentXRange(eventData);
    if (!xr) return;
    const [xmin, xmax] = xr;
    const buckets = {yaxis: [], yaxis2: [], yaxis3: [], yaxis4: []};

    (gd.data || []).forEach((trace) => {
      if (trace.visible === false || trace.visible === 'legendonly') return;
      const axisName = axisByTraceId[trace.yaxis || 'y'];
      if (!axisName || !buckets[axisName]) return;
      const xs = trace.x || [];
      const ys = trace.y || [];
      for (let i = 0; i < Math.min(xs.length, ys.length); i += 1) {
        const ms = toMs(xs[i]);
        if (ms === null || ms < xmin || ms > xmax) continue;
        const y = Number(ys[i]);
        if (Number.isFinite(y)) buckets[axisName].push(y);
      }
    });

    const update = {};
    Object.keys(buckets).forEach((axisName) => {
      const range = padded(buckets[axisName], rules[axisName] || {});
      if (range) update[axisName + '.range'] = range;
    });
    if (!Object.keys(update).length) return;
    updating = true;
    Plotly.relayout(gd, update).finally(() => { updating = false; });
  }

  gd.on('plotly_relayout', updateYRanges);
  gd.on('plotly_restyle', () => updateYRanges({autosize: true}));
})();
"""


def _write_plot_html(*, output_path: Path, fig: go.Figure, title: str) -> None:
    ensure_plotly_bundle(output_path.parent)
    height = int(fig.layout.height or 1500)
    plot_html = pio.to_html(
        fig,
        include_plotlyjs=False,
        full_html=False,
        validate=False,
        post_script=_autorange_post_script(),
        config={"responsive": True, "displaylogo": False, "scrollZoom": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
        default_width="100%",
        default_height=f"{height}px",
    )
    output_path.write_text(_html_page(title=title, plot_html=plot_html), encoding="utf-8")


def generate_specialized_dashboards(
    df: pd.DataFrame,
    *,
    output_dir: Path,
    dominant_html_name: str,
    dominant_png_name: str,
    features_html_name: str,
    features_png_name: str,
    duration_months: int = 3,
    max_points: int = 50000,
) -> None:
    """Generate the two specialized dashboards.

    PNG keeps the compact review window. HTML keeps the same core subplot grammar
    but includes full history with range controls and adaptive y-axis behavior.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for pattern in ["rpn_dominance-*.html", "rpn_features-*.html", "rpn_dominance-*.png", "rpn_features-*.png"]:
        for old in output_dir.glob(pattern):
            if old.is_file():
                old.unlink()

    dominant_png_path, current_time = render_rpn_dominance_png(df, output_dir / dominant_png_name, duration_months=duration_months)
    features_png_path, current_time_2 = render_rpn_features_png(df, output_dir / features_png_name, duration_months=duration_months)
    current_time = max(current_time, current_time_2)
    stamp = pd.to_datetime(current_time).strftime("%Y-%m-%d-%H")

    timestamped_dominant = output_dir / f"rpn_dominance-3M-{stamp}.png"
    timestamped_features = output_dir / f"rpn_features-3M-{stamp}.png"
    timestamped_dominant.write_bytes(dominant_png_path.read_bytes())
    timestamped_features.write_bytes(features_png_path.read_bytes())

    dom_fig, _, _ = _dominance_figure(df, default_view_months=duration_months, max_points=max_points)
    feat_fig, _, _ = _feature_figure(df, default_view_months=duration_months, max_points=max_points)

    _write_plot_html(output_path=output_dir / dominant_html_name, fig=dom_fig, title="BTC Price Colored by Risk Priority Number Bins With Dominance")
    _write_plot_html(output_path=output_dir / features_html_name, fig=feat_fig, title="BTC Price Colored by Risk Priority Number Bins")
