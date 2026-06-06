from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from liq_dataflow.feature_engineering.metadata import DELIVERED_FEATURES, FeatureDescriptor, get_feature_descriptor
from liq_dataflow.visualizer.html_assets import (
    ACCENT,
    BORDER,
    FONT,
    GRID,
    PAGE_BG,
    TEXT,
    ensure_plotly_bundle,
    html_template,
)


NUMERIC_CATEGORICAL_LIKE = {"bin_index", "dominance"}


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-zA-Z0-9_-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "feature"


def _downsample(df: pd.DataFrame, max_points: int, *, preserve_cols: Iterable[str] | None = None) -> pd.DataFrame:
    if max_points <= 0 or len(df) <= max_points:
        return df.copy()

    preserve_cols = [c for c in (preserve_cols or []) if c in df.columns]
    bucket_count = max(50, max_points // max(2, len(preserve_cols) + 1))
    edges = np.linspace(0, len(df), bucket_count + 1).round().astype(int)

    keep_idx: set[int] = {0, len(df) - 1}
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            continue
        chunk = df.iloc[start:end]
        keep_idx.add(start)
        keep_idx.add(end - 1)
        for col in preserve_cols:
            vals = pd.to_numeric(chunk[col], errors="coerce")
            if vals.notna().any():
                keep_idx.add(int(vals.idxmin()))
                keep_idx.add(int(vals.idxmax()))

    idx = sorted(i for i in keep_idx if 0 <= i < len(df))
    out = df.iloc[idx].copy()
    return out.sort_values("time").drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)


def _base_layout(title: str, *, height: int) -> dict:
    return dict(
        template="plotly_dark",
        title={"text": title, "x": 0.01, "xanchor": "left"},
        height=height,
        paper_bgcolor=PAGE_BG,
        plot_bgcolor=PAGE_BG,
        font={"family": FONT, "size": 14, "color": TEXT},
        hovermode="x unified",
        margin={"l": 78, "r": 70, "t": 66, "b": 64},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "bgcolor": "rgba(18,22,28,0.78)",
        },
    )


def _apply_axes(fig: go.Figure) -> None:
    fig.update_xaxes(showgrid=True, gridcolor=GRID, linecolor=BORDER, zeroline=False, showline=True)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, linecolor=BORDER, zeroline=False, showline=True)


def _range_selector() -> dict:
    return {
        "buttons": [
            {"count": 1, "label": "1M", "step": "month", "stepmode": "backward"},
            {"count": 3, "label": "3M", "step": "month", "stepmode": "backward"},
            {"count": 6, "label": "6M", "step": "month", "stepmode": "backward"},
            {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
            {"step": "all", "label": "All"},
        ]
    }


def _format_stat(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "-"
        abs_v = abs(float(value))
        if abs_v >= 1000:
            return f"{value:,.2f}"
        if abs_v >= 1:
            return f"{value:.4f}"
        return f"{value:.6f}"
    return str(value)


def _stats_table_html(stats: dict[str, object], *, title: str) -> str:
    rows = [f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(_format_stat(v))}</td></tr>" for k, v in stats.items()]
    return (
        f"<div class='stats-card'><h3>{html.escape(title)}</h3><table class='stats-table'>"
        + "".join(rows)
        + "</table></div>"
    )


def _numeric_stats(series: pd.Series) -> dict[str, object]:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    valid = s.dropna()
    if valid.empty:
        return {
            "count": 0,
            "missing_pct": 100.0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "q05": np.nan,
            "q95": np.nan,
            "max": np.nan,
            "latest": np.nan,
        }
    return {
        "count": int(valid.shape[0]),
        "missing_pct": float(s.isna().mean() * 100.0),
        "mean": float(valid.mean()),
        "median": float(valid.median()),
        "std": float(valid.std()),
        "min": float(valid.min()),
        "q05": float(valid.quantile(0.05)),
        "q95": float(valid.quantile(0.95)),
        "max": float(valid.max()),
        "latest": float(valid.iloc[-1]),
    }


def _categorical_stats(series: pd.Series) -> dict[str, object]:
    s = series.astype("object")
    valid = s.dropna().astype(str)
    return {
        "count": int(valid.shape[0]),
        "missing_pct": float(s.isna().mean() * 100.0),
        "nunique": int(valid.nunique()),
        "mode": None if valid.empty else valid.mode().iloc[0],
        "latest": None if valid.empty else valid.iloc[-1],
    }


def _descriptor_html(descriptor: FeatureDescriptor) -> str:
    blocks = []
    for label, value in (
        ("来源", descriptor.source),
        ("处理方法", descriptor.processing),
        ("金融学含义", descriptor.meaning),
        ("备注", descriptor.notes),
    ):
        if value:
            blocks.append(f"<p class='small'><strong>{html.escape(label)}：</strong>{html.escape(value)}</p>")
    return "".join(blocks)


def _is_categorical(feature: str, descriptor: FeatureDescriptor) -> bool:
    return descriptor.kind == "categorical" or feature in NUMERIC_CATEGORICAL_LIKE


def _make_numeric_figures(df: pd.DataFrame, feature: str, descriptor: FeatureDescriptor) -> tuple[go.Figure, go.Figure, dict[str, object]]:
    stats = _numeric_stats(df[feature])
    fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
    fig_ts.add_trace(
        go.Scattergl(x=df["time"], y=df["price"], name="BTC Price", line={"width": 1.1, "color": "#f1c232"}),
        secondary_y=False,
    )
    fig_ts.add_trace(
        go.Scattergl(x=df["time"], y=df[feature], name=feature, line={"width": 1.2, "color": ACCENT}),
        secondary_y=True,
    )
    fig_ts.update_layout(**_base_layout(f"{descriptor.title}｜时间序列", height=760))
    fig_ts.update_xaxes(rangeslider={"visible": True}, rangeselector=_range_selector())
    fig_ts.update_yaxes(title_text="BTC Price", secondary_y=False)
    fig_ts.update_yaxes(title_text=feature, secondary_y=True)
    fig_ts.update_layout(uirevision="static")
    _apply_axes(fig_ts)

    vals = pd.to_numeric(df[feature], errors="coerce").astype(float)
    fig_dist = make_subplots(rows=1, cols=2, subplot_titles=("分布直方图", "箱线图"))
    fig_dist.add_trace(go.Histogram(x=vals, nbinsx=80, name=feature, marker={"color": ACCENT}, opacity=0.85), row=1, col=1)
    fig_dist.add_trace(
        go.Box(y=vals, name=feature, boxmean="sd", marker={"color": "#6aa84f"}, line={"color": "#6aa84f"}),
        row=1,
        col=2,
    )
    fig_dist.update_layout(**_base_layout(f"{descriptor.title}｜统计分布", height=560))
    _apply_axes(fig_dist)
    return fig_ts, fig_dist, stats


def _make_categorical_figures(df: pd.DataFrame, feature: str, descriptor: FeatureDescriptor) -> tuple[go.Figure, go.Figure, dict[str, object]]:
    stats = _categorical_stats(df[feature])
    bins = pd.to_numeric(df[feature], errors="coerce").ffill()

    fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
    fig_ts.add_trace(
        go.Scattergl(x=df["time"], y=df["price"], name="BTC Price", line={"width": 1.1, "color": "#f1c232"}),
        secondary_y=False,
    )
    fig_ts.add_trace(
        go.Scattergl(x=df["time"], y=bins, mode="markers+lines", name=feature, marker={"size": 5, "color": ACCENT}),
        secondary_y=True,
    )
    fig_ts.update_layout(**_base_layout(f"{descriptor.title}｜状态时间序列", height=760))
    fig_ts.update_xaxes(rangeslider={"visible": True}, rangeselector=_range_selector())
    fig_ts.update_yaxes(title_text="BTC Price", secondary_y=False)
    fig_ts.update_yaxes(title_text=feature, secondary_y=True)
    fig_ts.update_layout(uirevision="static")
    _apply_axes(fig_ts)

    counts = bins.astype("Int64").astype(str).value_counts().sort_index()
    fig_dist = make_subplots(rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "domain"}]], subplot_titles=("频数统计", "占比结构"))
    fig_dist.add_trace(go.Bar(x=list(counts.index), y=list(counts.values), marker={"color": ACCENT}, name="count"), row=1, col=1)
    fig_dist.add_trace(go.Pie(labels=list(counts.index), values=list(counts.values), hole=0.4, sort=False, name="share"), row=1, col=2)
    fig_dist.update_layout(**_base_layout(f"{descriptor.title}｜统计分布", height=560))
    _apply_axes(fig_dist)
    return fig_ts, fig_dist, stats


def generate_feature_portal(
    df: pd.DataFrame,
    *,
    output_dir: Path,
    overview_filename: str,
    catalog_filename: str,
    pages_dir_name: str = "feature_pages",
    max_points: int = 8000,
    features: Iterable[str] | None = None,
    specialized_links: dict[str, str] | None = None,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = output_dir / pages_dir_name
    if pages_dir.exists():
        for old in pages_dir.glob("*.html"):
            old.unlink()
    pages_dir.mkdir(parents=True, exist_ok=True)
    ensure_plotly_bundle(output_dir)

    data = df.copy()
    data["time"] = pd.to_datetime(data["time"], errors="coerce")
    data = data.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    features = list(features or DELIVERED_FEATURES)
    preserve_cols = ["price", *features]
    plot_df = _downsample(data, max_points=max_points, preserve_cols=preserve_cols)

    rows = []
    for feature in features:
        if feature not in plot_df.columns:
            continue
        descriptor = get_feature_descriptor(feature)
        feature_df = plot_df[["time", "price", feature]].copy()
        if _is_categorical(feature, descriptor):
            fig_ts, fig_dist, stats = _make_categorical_figures(feature_df, feature, descriptor)
            stats_html = _stats_table_html(stats, title="状态摘要")
        else:
            fig_ts, fig_dist, stats = _make_numeric_figures(feature_df, feature, descriptor)
            stats_html = _stats_table_html(stats, title="数值摘要")

        fig_ts_html = pio.to_html(
            fig_ts,
            include_plotlyjs=False,
            full_html=False,
            config={"responsive": True, "displaylogo": False, "scrollZoom": True},
        )
        fig_dist_html = pio.to_html(
            fig_dist,
            include_plotlyjs=False,
            full_html=False,
            config={"responsive": True, "displaylogo": False, "scrollZoom": True},
        )

        page_name = f"{_slugify(feature)}.html"
        body = (
            "<div class='topbar'><div><h1 style='margin:0 0 6px'>{title}</h1><span class='chip'>{category}</span></div>"
            "<a class='back' href='../{overview}'>返回总览</a></div>"
            "<div class='panel'><p class='small'>本页面用于检查交付特征的时间行为、分布结构和近期最新值。"
            "支持鼠标 hover、拖拽缩放、scroll zoom 与范围选择。页面宽度已放宽，并使用保极值采样，以避免长样本压缩后丢失尖峰。</p>{note}</div>"
            "<div class='panel figure-wrap'>{ts}</div>"
            "<div class='panel figure-wrap'>{dist}</div>"
            "<div class='stats-grid'>{stats}</div>"
        ).format(
            title=html.escape(descriptor.title),
            category=html.escape(descriptor.category),
            overview=html.escape(overview_filename),
            note=_descriptor_html(descriptor),
            ts=fig_ts_html,
            dist=fig_dist_html,
            stats=stats_html,
        )
        page_html = html_template(title=descriptor.title, body=body, plotly_src="../plotly.min.js", page_class="wide")
        (pages_dir / page_name).write_text(page_html, encoding="utf-8")

        row = {
            "feature": feature,
            "title": descriptor.title,
            "category": descriptor.category,
            "kind": descriptor.kind,
            "page_path": f"{pages_dir_name}/{page_name}",
            "source": descriptor.source,
            "processing": descriptor.processing,
            "meaning": descriptor.meaning,
            "notes": descriptor.notes,
        }
        row.update(stats)
        rows.append(row)

    catalog = pd.DataFrame(rows)
    catalog.to_csv(output_dir / catalog_filename, index=False)

    sections = [
        "<div class='panel'><h1 style='margin:0 0 8px'>BTC Liquidation Feature Portal</h1>"
        f"<p class='small'>本页面展示当前最终交付审阅的 {len(features)} 个重要特征页面，以及 2 个自定义专题图表。"
        "通用页面专注于统计学审阅，自定义页面专注于 liquidation 结构与规则解释。</p></div>"
    ]
    if specialized_links:
        links_html = "".join(f"<a href='{html.escape(path)}'>{html.escape(name)}</a>" for name, path in specialized_links.items())
        sections.append(f"<div class='panel'><h2 style='margin:0 0 12px'>自定义可视化</h2><div class='link-list'>{links_html}</div></div>")

    cards = []
    for _, row in catalog.iterrows():
        cards.append(
            "<div class='card'>"
            f"<a class='link' href='{html.escape(row['page_path'])}'>{html.escape(str(row['title']))}</a>"
            f"<div class='small' style='margin-top:6px'>{html.escape(str(row['meaning']))}</div>"
            f"<div class='small' style='margin-top:8px'>字段名：{html.escape(str(row['feature']))}<br/>最新值：{html.escape(_format_stat(row.get('latest')))}</div>"
            "</div>"
        )
    sections.append(f"<div class='panel'><h2 style='margin:0 0 12px'>重要交付特征</h2><div class='cards'>{''.join(cards)}</div></div>")

    overview_html = html_template(title="BTC Liquidation Feature Portal", body="".join(sections), plotly_src=None, page_class="wide")
    (output_dir / overview_filename).write_text(overview_html, encoding="utf-8")
    return catalog
