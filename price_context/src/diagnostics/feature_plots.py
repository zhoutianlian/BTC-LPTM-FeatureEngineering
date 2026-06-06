from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from .feature_stats import to_numeric_series


PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def write_plotly_asset(report_dir) -> None:
    js_dir = report_dir / "assets" / "js"
    js_dir.mkdir(parents=True, exist_ok=True)
    (js_dir / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")


def figure_to_html(fig: go.Figure) -> str:
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config=PLOT_CONFIG)


def make_time_series_figure(
    df: pd.DataFrame,
    feature_name: str,
    result: dict[str, Any],
    cfg: dict[str, Any],
) -> go.Figure:
    time = pd.to_datetime(df["time"], errors="coerce")
    y = to_numeric_series(df[feature_name])
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            **_time_axis_kwargs(time),
            y=y,
            mode="lines",
            name=feature_name,
            line={"color": "#18E0FF", "width": 1.2},
            connectgaps=False,
            hovertemplate="time=%{x}<br>value=%{y:.6g}<extra></extra>",
        )
    )

    outlier_points = _select_outlier_marker_points(df, feature_name, cfg)
    if not outlier_points.empty:
        fig.add_trace(
            go.Scattergl(
                x=outlier_points["time"],
                y=outlier_points["value"],
                mode="markers",
                name="marked outliers",
                marker={"color": "#FF5A7A", "size": 6, "symbol": "x"},
                hovertemplate="time=%{x}<br>value=%{y:.6g}<extra>outlier</extra>",
            )
        )

    missing_points = _select_missing_points(time, y, cfg)
    if not missing_points.empty:
        baseline = _missing_baseline(y)
        fig.add_trace(
            go.Scattergl(
                x=missing_points,
                y=np.full(len(missing_points), baseline),
                mode="markers",
                name="missing samples",
                marker={"color": "#F6C945", "size": 4, "symbol": "line-ns-open"},
                hovertemplate="time=%{x}<extra>missing</extra>",
            )
        )

    _apply_fintech_layout(fig, f"{feature_name} time series", yaxis_title=feature_name)
    fig.update_layout(
        xaxis={
            "rangeslider": {"visible": True, "bgcolor": "#08111F", "bordercolor": "#1B2A3D"},
            "rangeselector": {
                "bgcolor": "#101C2C",
                "activecolor": "#18E0FF",
                "font": {"color": "#E7F6FF"},
                "buttons": [
                    {"count": 1, "label": "1D", "step": "day", "stepmode": "backward"},
                    {"count": 7, "label": "1W", "step": "day", "stepmode": "backward"},
                    {"count": 1, "label": "1M", "step": "month", "stepmode": "backward"},
                    {"count": 3, "label": "3M", "step": "month", "stepmode": "backward"},
                    {"count": 6, "label": "6M", "step": "month", "stepmode": "backward"},
                    {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
                    {"label": "ALL", "step": "all"},
                ],
            },
        }
    )
    fig.update_yaxes(fixedrange=False, rangemode="normal")
    return fig


def make_distribution_figure(df: pd.DataFrame, feature_name: str) -> go.Figure:
    values = to_numeric_series(df[feature_name])
    values = values[np.isfinite(values)]
    fig = go.Figure()
    if values.empty:
        _apply_fintech_layout(fig, f"{feature_name} distribution", yaxis_title="count")
        return fig

    density = _histogram_density_curve(values)
    if density is not None:
        fig.add_trace(
            go.Bar(
                x=density["x"],
                y=density["raw"],
                name="histogram",
                marker={"color": "rgba(24, 224, 255, 0.62)", "line": {"color": "#18E0FF", "width": 0.5}},
                hovertemplate="value=%{x:.6g}<br>density=%{y:.6g}<extra></extra>",
            )
        )
    else:
        fig.add_trace(
            go.Bar(
                x=[float(values.iloc[0])],
                y=[1.0],
                name="histogram",
                marker={"color": "rgba(24, 224, 255, 0.62)", "line": {"color": "#18E0FF", "width": 0.5}},
            )
        )
    if density is not None:
        fig.add_trace(
            go.Scatter(
                x=density["x"],
                y=density["y"],
                mode="lines",
                name="smoothed density",
                line={"color": "#A46CFF", "width": 2},
                hovertemplate="value=%{x:.6g}<br>density=%{y:.6g}<extra></extra>",
            )
        )

    stats = {
        "mean": values.mean(),
        "median": values.median(),
        "p01": values.quantile(0.01),
        "p99": values.quantile(0.99),
    }
    colors = {"mean": "#00FF88", "median": "#F6C945", "p01": "#FF7A1A", "p99": "#FF7A1A"}
    for name, x in stats.items():
        fig.add_vline(
            x=float(x),
            line={"color": colors[name], "dash": "dash", "width": 1.4},
            annotation_text=name,
            annotation_font_color=colors[name],
        )

    _apply_fintech_layout(fig, f"{feature_name} distribution", xaxis_title=feature_name, yaxis_title="density")
    return fig


def make_box_figure(df: pd.DataFrame, feature_name: str) -> go.Figure:
    values = to_numeric_series(df[feature_name])
    values = values[np.isfinite(values)]
    fig = go.Figure()
    if not values.empty:
        quantiles = values.quantile([0.25, 0.5, 0.75])
        fig.add_trace(
            go.Box(
                q1=[float(quantiles.loc[0.25])],
                median=[float(quantiles.loc[0.5])],
                q3=[float(quantiles.loc[0.75])],
                lowerfence=[float(values.min())],
                upperfence=[float(values.max())],
                name=feature_name,
                boxpoints=False,
                fillcolor="rgba(24, 224, 255, 0.32)",
                line={"color": "#18E0FF"},
                marker={"color": "#18E0FF"},
                hovertemplate="value=%{y:.6g}<extra></extra>",
            )
        )
    _apply_fintech_layout(fig, f"{feature_name} box plot", yaxis_title=feature_name)
    return fig


def make_rolling_figure(df: pd.DataFrame, feature_name: str, window: int) -> go.Figure:
    time = pd.to_datetime(df["time"], errors="coerce")
    values = to_numeric_series(df[feature_name])
    roll = values.rolling(window, min_periods=max(2, min(window, 10)))
    rolling_mean = roll.mean()
    rolling_std = roll.std()
    rolling_q05 = roll.quantile(0.05)
    rolling_q95 = roll.quantile(0.95)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.68, 0.32],
        subplot_titles=(f"value / rolling mean / rolling quantiles ({window} bars)", "rolling std"),
    )
    fig.add_trace(
        go.Scattergl(**_time_axis_kwargs(time), y=values, mode="lines", name="raw", line={"color": "rgba(24,224,255,0.50)", "width": 1}),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(**_time_axis_kwargs(time), y=rolling_mean, mode="lines", name="rolling mean", line={"color": "#00FF88", "width": 1.6}),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(**_time_axis_kwargs(time), y=rolling_q05, mode="lines", name="rolling q05", line={"color": "#7A8CA5", "width": 1, "dash": "dot"}),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(**_time_axis_kwargs(time), y=rolling_q95, mode="lines", name="rolling q95", line={"color": "#7A8CA5", "width": 1, "dash": "dot"}),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(**_time_axis_kwargs(time), y=rolling_std, mode="lines", name="rolling std", line={"color": "#FF7A1A", "width": 1.4}),
        row=2,
        col=1,
    )
    _apply_fintech_layout(fig, f"{feature_name} rolling diagnostics")
    fig.update_layout(xaxis2={"rangeslider": {"visible": True, "bgcolor": "#08111F", "bordercolor": "#1B2A3D"}})
    return fig


def make_missing_figure(df: pd.DataFrame, feature_name: str, window: int) -> go.Figure:
    time = pd.to_datetime(df["time"], errors="coerce")
    missing = to_numeric_series(df[feature_name]).isna().astype("float64")
    rolling_missing = missing.rolling(window, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            **_time_axis_kwargs(time),
            y=missing,
            mode="markers",
            name="missing indicator",
            marker={"color": "#F6C945", "size": 3},
            hovertemplate="time=%{x}<br>missing=%{y:.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scattergl(
            **_time_axis_kwargs(time),
            y=rolling_missing,
            mode="lines",
            name=f"rolling missing ratio ({window} bars)",
            line={"color": "#FF7A1A", "width": 1.5},
            hovertemplate="time=%{x}<br>ratio=%{y:.4f}<extra></extra>",
        )
    )
    _apply_fintech_layout(fig, f"{feature_name} missing value timeline", yaxis_title="missing / ratio")
    fig.update_yaxes(range=[-0.05, 1.05])
    return fig


def make_relationship_figure(
    df: pd.DataFrame,
    feature_name: str,
    reference: pd.DataFrame | None,
    correlations: dict[str, float | None],
    cfg: dict[str, Any],
) -> go.Figure | None:
    if reference is None or reference.empty:
        return None
    x = to_numeric_series(df[feature_name])
    panels = [
        ("close", "feature vs price", "#18E0FF"),
        ("return_1bar_bps", "feature vs short return", "#00FF88"),
        ("future_return_1h_bps", "feature vs future 1h return", "#FF7A1A"),
    ]
    available = [(col, title, color) for col, title, color in panels if col in reference.columns]
    if not available:
        return None

    fig = make_subplots(rows=1, cols=len(available), subplot_titles=[title for _, title, _ in available])
    for idx, (col, title, color) in enumerate(available, start=1):
        y = to_numeric_series(reference[col])
        sample = _sample_pair(x, y, int(cfg.get("report", {}).get("relationship_max_points", 10000)))
        corr = correlations.get(col)
        name = f"{col} corr={corr:.4f}" if corr is not None else col
        fig.add_trace(
            go.Scattergl(
                x=sample["x"],
                y=sample["y"],
                mode="markers",
                name=name,
                marker={"color": color, "size": 4, "opacity": 0.38},
                hovertemplate=f"{feature_name}=%{{x:.6g}}<br>{col}=%{{y:.6g}}<extra></extra>",
            ),
            row=1,
            col=idx,
        )
        fig.update_xaxes(title_text=feature_name, row=1, col=idx)
        fig.update_yaxes(title_text=col, row=1, col=idx)
    _apply_fintech_layout(fig, f"{feature_name} relationship diagnostics")
    return fig


def make_correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not corr.empty:
        fig.add_trace(
            go.Heatmap(
                z=corr.to_numpy(),
                x=list(corr.columns),
                y=list(corr.index),
                colorscale=[
                    [0.0, "#0B1020"],
                    [0.25, "#284A7E"],
                    [0.5, "#101C2C"],
                    [0.75, "#FF7A1A"],
                    [1.0, "#F6C945"],
                ],
                zmin=-1,
                zmax=1,
                colorbar={"title": "corr"},
                hovertemplate="x=%{x}<br>y=%{y}<br>corr=%{z:.4f}<extra></extra>",
            )
        )
    _apply_fintech_layout(fig, "Feature correlation heatmap", xaxis_title="feature", yaxis_title="feature")
    fig.update_layout(height=max(620, min(1500, 24 * max(len(corr), 20))))
    return fig


def _apply_fintech_layout(fig: go.Figure, title: str, xaxis_title: str | None = None, yaxis_title: str | None = None) -> None:
    fig.update_layout(
        template="plotly_dark",
        title={"text": title, "font": {"color": "#E7F6FF", "size": 17}},
        paper_bgcolor="#07111F",
        plot_bgcolor="#08111F",
        font={"color": "#D8E7F5", "family": "Inter, Segoe UI, Arial, sans-serif"},
        legend={
            "bgcolor": "rgba(7,17,31,0.76)",
            "bordercolor": "rgba(130,180,255,0.22)",
            "borderwidth": 1,
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        margin={"l": 58, "r": 32, "t": 76, "b": 52},
        hovermode="closest",
    )
    fig.update_xaxes(
        title=xaxis_title,
        gridcolor="rgba(170, 210, 255, 0.10)",
        zerolinecolor="rgba(170, 210, 255, 0.16)",
        showline=True,
        linecolor="rgba(170, 210, 255, 0.18)",
    )
    fig.update_yaxes(
        title=yaxis_title,
        gridcolor="rgba(170, 210, 255, 0.10)",
        zerolinecolor="rgba(170, 210, 255, 0.16)",
        showline=True,
        linecolor="rgba(170, 210, 255, 0.18)",
        rangemode="normal",
    )


def _select_outlier_marker_points(df: pd.DataFrame, feature_name: str, cfg: dict[str, Any]) -> pd.DataFrame:
    values = to_numeric_series(df[feature_name])
    finite = values[np.isfinite(values)]
    if finite.empty or len(finite) < 3:
        return pd.DataFrame(columns=["time", "value"])
    z_threshold = float(cfg.get("report", {}).get("zscore_threshold", 5.0))
    std = finite.std(ddof=1)
    if std <= 0 or not np.isfinite(std):
        return pd.DataFrame(columns=["time", "value"])
    z = ((values - finite.mean()) / std).abs()
    mask = z > z_threshold
    selected = pd.DataFrame({"time": pd.to_datetime(df["time"], errors="coerce"), "value": values, "z": z})
    selected = selected[mask & selected["value"].notna()].sort_values("z", ascending=False)
    max_points = int(cfg.get("report", {}).get("max_plot_outlier_points", 500))
    return selected.head(max_points)


def _select_missing_points(time: pd.Series, values: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    missing = time[values.isna()]
    max_points = int(cfg.get("report", {}).get("max_missing_markers", 500))
    if len(missing) <= max_points:
        return missing
    indices = np.linspace(0, len(missing) - 1, max_points).astype(int)
    return missing.iloc[indices]


def _missing_baseline(values: pd.Series) -> float:
    finite = values[np.isfinite(values)]
    if finite.empty:
        return 0.0
    y_min = float(finite.min())
    y_max = float(finite.max())
    if y_min == y_max:
        return y_min
    return y_min - 0.03 * (y_max - y_min)


def _histogram_density_curve(values: pd.Series) -> dict[str, np.ndarray] | None:
    if len(values) < 10 or values.nunique() <= 1:
        return None
    counts, edges = np.histogram(values.to_numpy(dtype="float64"), bins=140, density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    kernel = np.array([1, 2, 3, 4, 3, 2, 1], dtype="float64")
    kernel = kernel / kernel.sum()
    smooth = np.convolve(counts, kernel, mode="same")
    return {"x": centers, "y": smooth, "raw": counts}


def _time_axis_kwargs(time: pd.Series) -> dict[str, Any]:
    valid = pd.to_datetime(time, errors="coerce")
    if len(valid) < 3 or valid.isna().any():
        return {"x": valid}
    diffs = valid.diff().dropna()
    if diffs.empty:
        return {"x": valid}
    first = diffs.iloc[0]
    if not (diffs == first).all():
        return {"x": valid}
    dx_ms = int(first.total_seconds() * 1000)
    if dx_ms <= 0:
        return {"x": valid}
    return {"x0": valid.iloc[0].isoformat(), "dx": dx_ms}


def _sample_pair(x: pd.Series, y: pd.Series, max_points: int) -> pd.DataFrame:
    pair = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) <= max_points:
        return pair
    indices = np.linspace(0, len(pair) - 1, max_points).astype(int)
    return pair.iloc[indices]
