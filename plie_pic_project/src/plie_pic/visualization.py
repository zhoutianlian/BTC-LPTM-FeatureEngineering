from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import ProjectConfig
from .evaluation import evaluation_table_path
from .features import get_main_horizon
from .io import read_json
from .utils import sample_frame, safe_corr, to_utc_series

STATE_COLORS = {
    1: "rgba(0, 210, 125, 0.34)",
    2: "rgba(84, 225, 165, 0.29)",
    3: "rgba(150, 150, 150, 0.24)",
    4: "rgba(255, 165, 64, 0.30)",
    5: "rgba(255, 70, 120, 0.34)",
}

STATE_LEGEND = {
    1: "State 1 - 空头清算强势占优 / Up strong",
    2: "State 2 - 空头清算轻度占优 / Up mild",
    3: "State 3 - 多空清算均衡 / Neutral",
    4: "State 4 - 多头清算轻度占优 / Down mild",
    5: "State 5 - 多头清算强势占优 / Down strong",
}


def _range_selector() -> dict[str, Any]:
    """Plotly time-range selector used by all time-series pages."""
    return {
        "buttons": [
            {"count": 1, "label": "1D", "step": "day", "stepmode": "backward"},
            {"count": 7, "label": "1W", "step": "day", "stepmode": "backward"},
            {"count": 1, "label": "1M", "step": "month", "stepmode": "backward"},
            {"count": 3, "label": "3M", "step": "month", "stepmode": "backward"},
            {"count": 6, "label": "6M", "step": "month", "stepmode": "backward"},
            {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
            {"label": "ALL", "step": "all"},
        ],
        "bgcolor": "rgba(20,28,43,0.95)",
        "activecolor": "rgba(76,201,240,0.35)",
        "font": {"color": "#dbe7ff"},
        "x": 0.0,
        "xanchor": "left",
        "y": 1.15,
        "yanchor": "top",
    }


def _apply_time_controls(fig: go.Figure, slider_row: int | None = None, range_selector: bool = True) -> None:
    """Attach 1D/1W/1M/3M/6M/1Y/ALL buttons and an optional range slider."""
    fig.update_layout(xaxis={"type": "date"})
    if range_selector:
        fig.update_layout(xaxis={"rangeselector": _range_selector(), "type": "date"})
    if slider_row is not None:
        fig.update_xaxes(rangeslider_visible=True, row=slider_row, col=1)


def _add_state_legend(fig: go.Figure, row: int = 1, col: int = 1) -> None:
    """Add HMM state color legend entries for background-fill charts."""
    for state, label in STATE_LEGEND.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"size": 10, "color": STATE_COLORS[state]},
                name=label,
                showlegend=True,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )


def _padded_numeric_range(values: pd.Series, padding_ratio: float = 0.07) -> list[float] | None:
    """Return a readable numeric axis range without forcing the axis to zero."""
    s = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return None
    low = float(s.min())
    high = float(s.max())
    if len(s) >= 100:
        q01 = float(s.quantile(0.01))
        q99 = float(s.quantile(0.99))
        central_span = q99 - q01
        full_span = high - low
        if central_span > 0 and full_span > central_span * 4:
            low, high = q01, q99
    span = high - low
    if not np.isfinite(span) or span <= 0:
        base = max(abs(high), 1.0)
        span = base * 0.02
        low -= span / 2
        high += span / 2
    padding = span * padding_ratio
    return [low - padding, high + padding]


def _state_hover_data(df: pd.DataFrame) -> list[str]:
    if "hmm_state" not in df.columns:
        return ["hmm_state: NA"] * len(df)
    states = pd.to_numeric(df["hmm_state"], errors="coerce")
    labels: list[str] = []
    for state in states:
        if pd.isna(state):
            labels.append("hmm_state: NA")
            continue
        state_int = int(state)
        labels.append(f"hmm_state: {STATE_LEGEND.get(state_int, f'State {state_int}')}")
    return labels


def _hover_template(value_name: str, value_format: str = ":,.2f", suffix: str = "") -> str:
    return (
        "%{x|%Y-%m-%d %H:%M UTC}<br>"
        "%{customdata}<br>"
        f"{value_name}: %{{y{value_format}}}{suffix}"
        "<extra></extra>"
    )


def _external_time_controls_script() -> str:
    """Create HTML-level time buttons so they cannot overlap Plotly titles."""
    return """
(function() {
  var gd = document.getElementById('{plot_id}');
  if (!gd || !gd.data || !gd.data.length) return;
  var controlId = gd.id + '-time-controls';
  if (document.getElementById(controlId)) return;

  var sourceTrace = gd.data.find(function(t) { return Array.isArray(t.x) && t.x.length; });
  if (!sourceTrace) return;
  var xMillis = sourceTrace.x
    .map(function(v) { return new Date(v).getTime(); })
    .filter(function(v) { return Number.isFinite(v); });
  if (!xMillis.length) return;
  var fullStart = Math.min.apply(null, xMillis);
  var fullEnd = Math.max.apply(null, xMillis);

  var wrap = document.createElement('div');
  wrap.id = controlId;
  wrap.style.cssText = [
    'display:flex',
    'gap:6px',
    'align-items:center',
    'margin:0 0 10px 98px',
    'height:30px',
    'position:relative',
    'z-index:2'
  ].join(';');

  var buttonSpecs = [
    {label:'1D', ms:24 * 60 * 60 * 1000},
    {label:'1W', ms:7 * 24 * 60 * 60 * 1000},
    {label:'1M', ms:30 * 24 * 60 * 60 * 1000},
    {label:'3M', ms:90 * 24 * 60 * 60 * 1000},
    {label:'6M', ms:182 * 24 * 60 * 60 * 1000},
    {label:'1Y', ms:365 * 24 * 60 * 60 * 1000},
    {label:'ALL', ms:null}
  ];
  var buttons = [];

  function axisKeys() {
    return Object.keys(gd.layout || {}).filter(function(k) {
      return /^xaxis[0-9]*$/.test(k);
    });
  }

  function setActive(label) {
    buttons.forEach(function(btn) {
      var active = btn.textContent === label;
      btn.style.background = active ? 'rgba(76,201,240,0.35)' : 'rgba(20,28,43,0.95)';
      btn.style.borderColor = active ? 'rgba(141,227,255,0.8)' : 'rgba(45,66,107,0.9)';
    });
  }

  function relayoutRange(label, ms) {
    var update = {};
    axisKeys().forEach(function(axisKey) {
      if (ms === null) {
        update[axisKey + '.autorange'] = true;
      } else {
        update[axisKey + '.range'] = [new Date(Math.max(fullStart, fullEnd - ms)), new Date(fullEnd)];
        update[axisKey + '.autorange'] = false;
      }
    });
    setActive(label);
    Plotly.relayout(gd, update);
  }

  buttonSpecs.forEach(function(spec) {
    var btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = spec.label;
    btn.style.cssText = [
      'height:24px',
      'min-width:34px',
      'padding:0 9px',
      'border-radius:5px',
      'border:1px solid rgba(45,66,107,0.9)',
      'background:rgba(20,28,43,0.95)',
      'color:#dbe7ff',
      'font:600 13px Inter, Segoe UI, Arial, sans-serif',
      'line-height:22px',
      'cursor:pointer'
    ].join(';');
    btn.addEventListener('click', function() { relayoutRange(spec.label, spec.ms); });
    buttons.push(btn);
    wrap.appendChild(btn);
  });

  gd.parentNode.insertBefore(wrap, gd);
  setActive('ALL');
})();
"""


def _dynamic_y_range_script(trace_name: str, yaxis_layout_key: str, padding_ratio: float = 0.08) -> str:
    """Browser-side y autoscale for range selector / range slider window changes."""
    return f"""
(function() {{
  var gd = document.getElementById('{{plot_id}}');
  if (!gd) return;
  var traceName = {json.dumps(trace_name)};
  var yaxisLayoutKey = {json.dumps(yaxis_layout_key)};
  var paddingRatio = {padding_ratio};
  var trace = (gd.data || []).find(function(t) {{ return t.name === traceName; }});
  if (!trace || !trace.x || !trace.y) return;

  function toMillis(v) {{
    var t = new Date(v).getTime();
    return Number.isFinite(t) ? t : null;
  }}

  var xs = trace.x.map(toMillis);
  var ys = trace.y.map(function(v) {{ return Number(v); }});
  var allFiniteY = ys.filter(function(v) {{ return Number.isFinite(v); }});

  function quantile(sorted, q) {{
    if (!sorted.length) return null;
    var pos = (sorted.length - 1) * q;
    var base = Math.floor(pos);
    var rest = pos - base;
    if (sorted[base + 1] !== undefined) {{
      return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
    }}
    return sorted[base];
  }}

  function rangeForVisible(x0, x1) {{
    var visible = [];
    for (var i = 0; i < ys.length; i++) {{
      if (!Number.isFinite(ys[i]) || xs[i] === null) continue;
      if (x0 !== null && xs[i] < x0) continue;
      if (x1 !== null && xs[i] > x1) continue;
      visible.push(ys[i]);
    }}
    if (visible.length < 2) visible = allFiniteY.slice();
    if (!visible.length) return null;
    visible.sort(function(a, b) {{ return a - b; }});
    var low = visible[0];
    var high = visible[visible.length - 1];
    if (visible.length >= 100) {{
      var q01 = quantile(visible, 0.01);
      var q99 = quantile(visible, 0.99);
      var centralSpan = q99 - q01;
      var fullSpan = high - low;
      if (centralSpan > 0 && fullSpan > centralSpan * 4) {{
        low = q01;
        high = q99;
      }}
    }}
    var span = high - low;
    if (!Number.isFinite(span) || span <= 0) {{
      var base = Math.max(Math.abs(high), 1);
      span = base * 0.02;
      low -= span / 2;
      high += span / 2;
    }}
    var pad = span * paddingRatio;
    return [low - pad, high + pad];
  }}

  function currentXRange() {{
    var layout = gd.layout || {{}};
    var axisKeys = Object.keys(layout).filter(function(k) {{
      return /^xaxis[0-9]*$/.test(k) && layout[k] && layout[k].range;
    }});
    var xa = layout.xaxis && layout.xaxis.range ? layout.xaxis : layout[axisKeys[0]] || {{}};
    var r = xa.range;
    if (!r || r.length < 2) return [null, null];
    return [toMillis(r[0]), toMillis(r[1])];
  }}

  var pending = null;
  function applyRange() {{
    pending = null;
    var xr = currentXRange();
    var yr = rangeForVisible(xr[0], xr[1]);
    if (!yr) return;
    var update = {{}};
    update[yaxisLayoutKey + ".range"] = yr;
    update[yaxisLayoutKey + ".autorange"] = false;
    Plotly.relayout(gd, update);
  }}

  function scheduleApply() {{
    if (pending !== null) window.clearTimeout(pending);
    pending = window.setTimeout(applyRange, 40);
  }}

  gd.on("plotly_relayout", function(ev) {{
    if (!ev) return;
    var changedX = Object.keys(ev).some(function(k) {{
      return /^xaxis[0-9]*\\.range/.test(k) || /^xaxis[0-9]*\\.autorange$/.test(k);
    }});
    if (changedX) scheduleApply();
  }});
  scheduleApply();
}})();
"""


def generate_reports(cfg: ProjectConfig) -> dict[str, Path]:
    """Generate all requested interactive HTML reports."""
    cfg.ensure_dirs()
    pred_path = cfg.path("paths", "train_source_predictions")
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing source predictions: {pred_path}. Run training first.")
    pred = pd.read_csv(pred_path)
    pred["time"] = to_utc_series(pred["time"])
    report_dir = cfg.path("paths", "report_html_dir")
    report_dir.mkdir(parents=True, exist_ok=True)

    pages = cfg.get("reports", "pages", default={}) or {}
    paths = {
        "index": _index_page(pred, cfg, report_dir / str(pages.get("index", "index.html"))),
        "plie_price": _plie_price_page(pred, cfg, report_dir / str(pages.get("plie_price", "plie_price.html"))),
        "hmm_state": _hmm_state_page(pred, cfg, report_dir / str(pages.get("hmm_state", "hmm_state.html"))),
        "feature_statistics": _feature_statistics_page(pred, cfg, report_dir / str(pages.get("feature_statistics", "feature_statistics.html"))),
        "model_evaluation": _model_evaluation_page(pred, cfg, report_dir / str(pages.get("model_evaluation", "model_evaluation.html"))),
    }
    return paths


def _html_shell(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<style>
body {{ background:#070b12; color:#dbe7ff; font-family: Inter, Segoe UI, Arial, sans-serif; margin:0; }}
a {{ color:#4cc9f0; text-decoration:none; }}
a:hover {{ text-decoration:underline; }}
.container {{ max-width: 1420px; margin: 0 auto; padding: 28px; }}
.card {{ background: linear-gradient(145deg,#0c1220,#111827); border:1px solid #1f2a44; border-radius:16px; padding:20px; margin:16px 0; box-shadow:0 0 24px rgba(76,201,240,.08); }}
.grid {{ display:grid; grid-template-columns: repeat(auto-fit,minmax(250px,1fr)); gap:16px; }}
.metric {{ font-size:28px; font-weight:700; color:#f5fbff; }}
.label {{ color:#8da2c8; font-size:13px; text-transform:uppercase; letter-spacing:.08em; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th, td {{ border-bottom:1px solid #1f2a44; padding:8px 10px; text-align:left; }}
th {{ color:#8de3ff; }}
.badge {{ display:inline-block; border:1px solid #2d426b; border-radius:999px; padding:4px 10px; margin:2px; color:#bfe8ff; background:#0a1424; }}
</style>
</head>
<body><div class="container">{body}</div></body></html>"""


def _write_html(path: Path, html: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def _latest_row(pred: pd.DataFrame) -> pd.Series:
    return pred.sort_values("time").iloc[-1]


def _index_page(pred: pd.DataFrame, cfg: ProjectConfig, path: Path) -> Path:
    latest = _latest_row(pred)
    latest_summary_path = cfg.path("paths", "latest_summary")
    latest_summary = read_json(latest_summary_path) if latest_summary_path.exists() else {}
    metrics_path = evaluation_table_path(cfg, "overall_metrics")
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    main_horizon = get_main_horizon(cfg)
    test_main = metrics.loc[(metrics.get("subset") == "test") & (metrics.get("horizon_min") == main_horizon)] if not metrics.empty else pd.DataFrame()
    test_metric = test_main.iloc[0].to_dict() if len(test_main) else {}
    body = f"""
<h1>PLIE-PIC Research Terminal</h1>
<div class="card">
  <p><b>PLIE-PIC</b> is a source-clock, mechanism-constrained, quantile passive impact curve. It estimates the approximate passive price pressure caused by BTC futures liquidation forced flow. It is not a generic BTC return predictor.</p>
  <p>Core pipeline: liquidation pressure state/change → passive PLIE baseline → compare with realized price move → absorption / active residual diagnostics.</p>
</div>
<div class="grid">
  <div class="card"><div class="label">Latest source time</div><div class="metric">{latest.get('time')}</div></div>
  <div class="card"><div class="label">Latest HMM state</div><div class="metric">{latest.get('hmm_state')}</div></div>
  <div class="card"><div class="label">Latest PLIE main</div><div class="metric">{_fmt(latest.get('plie_main_bps'))} bps</div></div>
  <div class="card"><div class="label">Latest reliability</div><div class="metric">{_fmt(latest.get('plie_reliability'))}</div></div>
</div>
<div class="card">
<h2>Navigation</h2>
<p>
<span class="badge"><a href="plie_price.html">PLIE × Price</a></span>
<span class="badge"><a href="hmm_state.html">HMM State</a></span>
<span class="badge"><a href="feature_statistics.html">Feature Statistics</a></span>
<span class="badge"><a href="model_evaluation.html">Model Evaluation</a></span>
</p>
</div>
<div class="card">
<h2>Latest inference summary</h2>
{_dict_table(latest_summary)}
</div>
<div class="card">
<h2>Test main-horizon core metric snapshot</h2>
{_dict_table(test_metric)}
</div>
"""
    return _write_html(path, _html_shell("PLIE-PIC Index", body))


def _plie_price_page(pred: pd.DataFrame, cfg: ProjectConfig, path: Path) -> Path:
    main_horizon = get_main_horizon(cfg)
    ret_col = f"ret_{main_horizon}m_bps"
    residual_col = f"plie_residual_{main_horizon}m_bps"
    absorption_col = f"plie_absorption_{main_horizon}m"
    max_points = int(cfg.get("reports", "max_points"))
    df = sample_frame(pred.sort_values("time"), max_points)
    hover_state = _state_hover_data(df)
    corr = safe_corr(df["plie_main_bps"], df.get(ret_col, pd.Series(index=df.index)), method="spearman")
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.028,
        subplot_titles=(
            "BTC price with HMM state background",
            "Signed PLIE passive baseline",
            f"Actual {main_horizon}m return only",
            f"Active residual: actual {main_horizon}m return - signed PLIE",
            "Absorption diagnostic",
        ),
    )
    _add_state_background(fig, df, cfg, row=1, col=1)
    _add_state_background(fig, df, cfg, row=2, col=1)
    _add_state_legend(fig, row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["price"],
            name="price",
            mode="lines",
            line={"width": 1.0},
            customdata=hover_state,
            hovertemplate=_hover_template("price"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["plie_main_bps"],
            name=f"PLIE main {main_horizon}m bps",
            mode="lines",
            line={"width": 0.85},
            customdata=hover_state,
            hovertemplate=_hover_template(f"PLIE main {main_horizon}m", suffix=" bps"),
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line_width=0.8, line_dash="dot", line_color="rgba(255,255,255,0.35)", row=2, col=1)

    if ret_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df[ret_col],
                name=f"actual {main_horizon}m return bps",
                mode="lines",
                line={"width": 0.75},
                customdata=hover_state,
                hovertemplate=_hover_template(f"actual {main_horizon}m return", suffix=" bps"),
            ),
            row=3,
            col=1,
        )
        fig.add_hline(y=0, line_width=0.8, line_dash="dot", line_color="rgba(255,255,255,0.35)", row=3, col=1)
    if residual_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df[residual_col],
                name="actual - PLIE residual",
                mode="lines",
                line={"width": 0.75},
                customdata=hover_state,
                hovertemplate=_hover_template("actual - PLIE residual", suffix=" bps"),
            ),
            row=4,
            col=1,
        )
        fig.add_hline(y=0, line_width=0.8, line_dash="dot", line_color="rgba(255,255,255,0.35)", row=4, col=1)
    if absorption_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df[absorption_col],
                name=f"absorption {main_horizon}m",
                mode="lines",
                line={"width": 0.8},
                customdata=hover_state,
                hovertemplate=_hover_template(f"absorption {main_horizon}m"),
            ),
            row=5,
            col=1,
        )
        fig.add_hline(y=1, line_width=0.8, line_dash="dot", line_color="rgba(255,255,255,0.35)", row=5, col=1)
        fig.add_hline(y=0, line_width=0.8, line_dash="dot", line_color="rgba(255,255,255,0.25)", row=5, col=1)

    price_range = _padded_numeric_range(df["price"])
    if price_range is not None:
        fig.update_yaxes(range=price_range, row=1, col=1)
    fig.update_layout(
        template="plotly_dark",
        height=1500,
        hovermode="x unified",
        title={
            "text": f"PLIE × Price | Spearman(PLIE, {main_horizon}m return) = {_fmt(corr)}",
            "x": 0.0,
            "xanchor": "left",
            "y": 0.985,
            "yanchor": "top",
        },
        legend={"orientation": "v", "x": 1.01, "y": 0.99},
        margin={"l": 72, "r": 360, "t": 105, "b": 64},
    )
    _apply_time_controls(fig, slider_row=5, range_selector=False)
    price_axis_script = _external_time_controls_script() + _dynamic_y_range_script("price", "yaxis")
    body = """
<h1>PLIE × Price</h1>
<div class='card'>
  <p><b>Reading guide:</b> PLIE is the signed passive liquidation-impact baseline. Actual return is the realized main-horizon price move. Residual is displayed in a separate panel because it has a different meaning: <code>actual return - signed PLIE</code>. A negative residual under positive PLIE indicates absorption / active sell pressure; a positive residual under negative PLIE indicates absorption / active buy support. Residual and absorption are post-horizon diagnostics and must not be used as live Agent inputs.</p>
</div>
""" + f"<div class='card'>{fig.to_html(full_html=False, include_plotlyjs='cdn', post_script=price_axis_script)}</div>"
    return _write_html(path, _html_shell("PLIE × Price", body))

def _hmm_state_page(pred: pd.DataFrame, cfg: ProjectConfig, path: Path) -> Path:
    main_horizon = get_main_horizon(cfg)
    df = sample_frame(pred.sort_values("time"), int(cfg.get("reports", "max_points")))
    hover_state = _state_hover_data(df)

    # Main time-series figure: HMM state is rendered as background fill behind
    # the PLIE line, matching a trading-terminal regime overlay style.
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "PLIE main line with HMM state background",
            "Price with HMM state background",
            "HMM hard state and confidence",
        ),
        vertical_spacing=0.045,
    )
    _add_state_background(fig, df, cfg, row=1, col=1)
    _add_state_background(fig, df, cfg, row=2, col=1)
    _add_state_legend(fig, row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["plie_main_bps"],
            name=f"PLIE main {main_horizon}m bps",
            mode="lines",
            line={"width": 0.85},
            customdata=hover_state,
            hovertemplate=_hover_template(f"PLIE main {main_horizon}m", suffix=" bps"),
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, line_width=0.8, line_dash="dot", line_color="rgba(255,255,255,0.35)", row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["price"],
            name="price",
            mode="lines",
            line={"width": 1.0},
            customdata=hover_state,
            hovertemplate=_hover_template("price"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["hmm_state"],
            name="hmm_state",
            mode="lines",
            line={"shape": "hv", "width": 0.85},
            customdata=hover_state,
            hovertemplate=_hover_template("hmm_state", ":.0f"),
        ),
        row=3,
        col=1,
    )
    if "hmm_conf" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df["hmm_conf"],
                name="hmm_conf",
                mode="lines",
                line={"width": 0.8},
                customdata=hover_state,
                hovertemplate=_hover_template("hmm_conf", ":.3f"),
                yaxis="y3",
            ),
            row=3,
            col=1,
        )
    fig.update_layout(
        template="plotly_dark",
        height=1180,
        hovermode="x unified",
        title={
            "text": "HMM regimes over PLIE and price",
            "x": 0.0,
            "xanchor": "left",
            "y": 0.985,
            "yanchor": "top",
        },
        legend={"orientation": "v", "x": 1.01, "y": 0.99},
        margin={"l": 72, "r": 360, "t": 105, "b": 64},
    )
    price_range = _padded_numeric_range(df["price"])
    if price_range is not None:
        fig.update_yaxes(range=price_range, row=2, col=1)
    _apply_time_controls(fig, slider_row=3, range_selector=False)
    price_axis_script = _external_time_controls_script() + _dynamic_y_range_script("price", "yaxis2")

    # Separate categorical distribution figure. Keeping it separate avoids mixing
    # categorical state bins with the time-range x-axis controls.
    box = go.Figure()
    if "hmm_state" in pred.columns and "plie_main_bps" in pred.columns:
        box.add_trace(go.Box(x=pred["hmm_state"].astype(str), y=pred["plie_main_bps"], name="PLIE by state", boxmean=True))
    box.update_layout(template="plotly_dark", title="PLIE distribution by HMM state", height=520, xaxis_title="HMM state", yaxis_title="PLIE main bps")

    trans = _transition_matrix(pred)
    heat = go.Figure(data=go.Heatmap(z=trans.values, x=trans.columns.astype(str), y=trans.index.astype(str), colorscale="Viridis"))
    heat.update_layout(template="plotly_dark", title="Source-clock HMM transition matrix", height=520)
    duration_fig = go.Figure()
    if "age_in_state_source" in pred.columns:
        duration_fig.add_trace(go.Histogram(x=pred["age_in_state_source"], nbinsx=int(cfg.get("reports", "state_duration_bins", default=80)), name="age_in_state_source"))
    duration_fig.update_layout(template="plotly_dark", title="State duration distribution", height=420)
    body = """
<h1>HMM State Diagnostics</h1>
<div class='card'>
  <p>The first panel uses HMM state as a background regime fill behind the PLIE line. Use the 1D / 1W / 1M / 3M / 6M / 1Y / ALL buttons to zoom into specific market periods without losing the full-history context.</p>
</div>
""" + f"<div class='card'>{fig.to_html(full_html=False, include_plotlyjs='cdn', post_script=price_axis_script)}</div><div class='card'>{box.to_html(full_html=False, include_plotlyjs=False)}</div><div class='card'>{heat.to_html(full_html=False, include_plotlyjs=False)}</div><div class='card'>{duration_fig.to_html(full_html=False, include_plotlyjs=False)}</div>"
    return _write_html(path, _html_shell("HMM State", body))

def _feature_statistics_page(pred: pd.DataFrame, cfg: ProjectConfig, path: Path) -> Path:
    variables = list(cfg.get("reports", "feature_statistics_variables", default=[])) or [
        "plie_force_up",
        "plie_intensity",
        "plie_accel_pos",
        "plie_reliability",
        "plie_main_bps",
        "plie_passive_20m_bps",
        "plie_passive_30m_bps",
        "plie_passive_60m_bps",
        "hmm_conf",
        "liq_entropy",
        "age_in_state_source",
    ]
    variables = [v for v in variables if v in pred.columns]
    df = sample_frame(pred.sort_values("time"), int(cfg.get("reports", "max_points")))
    hist_bins = int(cfg.get("reports", "histogram_bins", default=80))
    figs: list[str] = []
    stats_rows = []
    for var in variables:
        s = pd.to_numeric(pred[var], errors="coerce")
        stats_rows.append({
            "variable": var,
            "missing_ratio": float(s.isna().mean()),
            "finite_ratio": float(np.isfinite(s.dropna()).mean()) if len(s.dropna()) else 0.0,
            "mean": float(s.mean()) if s.notna().any() else np.nan,
            "std": float(s.std()) if s.notna().any() else np.nan,
            "p01": float(s.quantile(0.01)) if s.notna().any() else np.nan,
            "p50": float(s.quantile(0.50)) if s.notna().any() else np.nan,
            "p99": float(s.quantile(0.99)) if s.notna().any() else np.nan,
        })
        fig = make_subplots(rows=3, cols=2, subplot_titles=(f"{var} time series", f"{var} distribution", "Rolling mean", "Rolling std", "Relation with PLIE main", "By HMM state"))
        fig.add_trace(go.Scatter(x=df["time"], y=df[var], mode="lines", name=var), row=1, col=1)
        fig.add_trace(go.Histogram(x=df[var], nbinsx=hist_bins, name="hist"), row=1, col=2)
        roll = pd.to_numeric(df[var], errors="coerce").rolling(int(cfg.get("reports", "rolling_window_report")), min_periods=10)
        fig.add_trace(go.Scatter(x=df["time"], y=roll.mean(), mode="lines", name="rolling mean"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df["time"], y=roll.std(), mode="lines", name="rolling std"), row=2, col=2)
        fig.add_trace(go.Scatter(x=df[var], y=df["plie_main_bps"], mode="markers", name="vs PLIE", marker={"size": 3, "opacity": 0.45}), row=3, col=1)
        fig.add_trace(go.Box(x=df["hmm_state"].astype(str), y=df[var], name="by state"), row=3, col=2)
        fig.update_layout(template="plotly_dark", height=900, showlegend=False, title=f"Feature statistics: {var}")
        figs.append(f"<div class='card'>{fig.to_html(full_html=False, include_plotlyjs='cdn' if not figs else False)}</div>")
    body = f"<h1>Feature Statistics</h1><div class='card'><h2>Basic statistics</h2>{_df_table(pd.DataFrame(stats_rows))}</div>" + "\n".join(figs)
    return _write_html(path, _html_shell("Feature Statistics", body))


def _model_evaluation_page(pred: pd.DataFrame, cfg: ProjectConfig, path: Path) -> Path:
    table_names = [
        "retrain_monitoring",
        "rolling_latest_monitoring",
        "quantile_calibration_metrics",
        "overall_metrics",
        "conditional_subset_metrics",
        "monotonicity_metrics",
        "by_state_metrics",
        "by_transition_metrics",
        "decile_metrics",
        "model_coefficients",
        "walk_forward",
        "output_checks",
    ]
    tables = {}
    for name in table_names:
        p = evaluation_table_path(cfg, name)
        if p.exists():
            tables[name] = pd.read_csv(p)
    checks_path = cfg.path("paths", "post_training_checks")
    checks = read_json(checks_path) if checks_path.exists() else {}

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "PLIE main distribution by split",
            "PLIE main rolling mean",
            "Quantile coverage: P(aligned actual <= PLIE magnitude)",
        ),
        vertical_spacing=0.07,
    )
    if "split" in pred.columns:
        for split in [str(x) for x in cfg.get("evaluation", "split_names", default=["train", "validation", "test"])]:
            sub = pred.loc[pred["split"].eq(split)]
            fig.add_trace(go.Histogram(x=sub["plie_main_bps"], name=split, opacity=0.65), row=1, col=1)
    df = sample_frame(pred.sort_values("time"), int(cfg.get("reports", "max_points")))
    eval_roll = int(cfg.get("reports", "evaluation_rolling_window", default=240))
    fig.add_trace(go.Scatter(x=df["time"], y=df["plie_main_bps"].rolling(eval_roll, min_periods=20).mean(), name="rolling PLIE mean"), row=2, col=1)

    calib = tables.get("quantile_calibration_metrics", pd.DataFrame())
    if not calib.empty:
        splits = [str(x) for x in cfg.get("evaluation", "split_names", default=["train", "validation", "test"])]
        splits.append(str(cfg.get("evaluation", "all_subset_name", default="all")))
        for split in splits:
            sub = calib.loc[calib["split"].eq(split)]
            if sub.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub["horizon_min"].astype(str),
                    y=sub["coverage_actual_le_plie"],
                    name=f"coverage {split}",
                    mode="lines+markers",
                ),
                row=3,
                col=1,
            )
        target = float(calib["target_quantile"].dropna().iloc[0]) if calib["target_quantile"].notna().any() else 0.65
        fig.add_hline(y=target, line_width=1, line_dash="dot", line_color="rgba(255,255,255,0.5)", row=3, col=1)
    fig.update_layout(template="plotly_dark", height=980, barmode="overlay")

    body = """
<h1>Model Evaluation</h1>
<div class='card'>
  <p><b>Evaluation principle:</b> PLIE-PIC is a passive liquidation-impact quantile baseline, so the primary checks are quantile coverage, pinball loss against null baselines, conditional subset behavior, monotonicity of PLIE versus liquidation pressure, and leakage/output sanity. Low full-sample return IC is not by itself a failure.</p>
</div>
"""
    body += f"<div class='card'>{fig.to_html(full_html=False, include_plotlyjs='cdn')}</div>"
    body += f"<div class='card'><h2>Future leakage checks</h2>{_dict_table(checks)}</div>"
    preview_rows = int(cfg.get("reports", "table_preview_rows", default=300))
    for name in table_names:
        table = tables.get(name)
        if table is not None:
            body += f"<div class='card'><h2>{name}</h2>{_df_table(table.head(preview_rows))}</div>"
    return _write_html(path, _html_shell("Model Evaluation", body))

def _add_state_background(fig: go.Figure, df: pd.DataFrame, cfg: ProjectConfig, row: int, col: int) -> None:
    if "hmm_state" not in df.columns or df.empty:
        return
    d = df[["time", "hmm_state"]].dropna().copy()
    if d.empty:
        return
    d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
    d["hmm_state"] = pd.to_numeric(d["hmm_state"], errors="coerce")
    d = d.dropna().sort_values("time")
    max_segments = int(cfg.get("reports", "state_background_max_segments", default=1200))
    if max_segments > 0 and len(d) > max_segments:
        d = sample_frame(d, max_segments)
    if d.empty:
        return
    states = d["hmm_state"].astype(int).to_numpy()
    times = d["time"].tolist()
    deltas = d["time"].diff().dropna()
    median_delta = deltas[deltas > pd.Timedelta(0)].median() if not deltas.empty else pd.Timedelta(0)
    final_x1 = times[-1] + median_delta if pd.notna(median_delta) and median_delta > pd.Timedelta(0) else times[-1]
    subplot = fig.get_subplot(row, col)
    xref = _axis_ref(subplot.xaxis.plotly_name)
    yref = f"{_axis_ref(subplot.yaxis.plotly_name)} domain"
    shapes: list[dict[str, Any]] = []
    start_idx = 0
    for i in range(1, len(d)):
        if states[i] != states[start_idx]:
            shapes.append(_state_rect_shape(times[start_idx], times[i], states[start_idx], xref, yref))
            start_idx = i
    shapes.append(_state_rect_shape(times[start_idx], final_x1, states[start_idx], xref, yref))
    fig.update_layout(shapes=tuple(fig.layout.shapes) + tuple(shapes))


def _axis_ref(plotly_axis_name: str) -> str:
    return plotly_axis_name.replace("axis", "")


def _state_rect_shape(x0: Any, x1: Any, state: int, xref: str, yref: str) -> dict[str, Any]:
    return {
        "type": "rect",
        "xref": xref,
        "yref": yref,
        "x0": x0,
        "x1": x1,
        "y0": 0,
        "y1": 1,
        "fillcolor": STATE_COLORS.get(int(state), "rgba(80,80,80,0.12)"),
        "opacity": 1.0,
        "layer": "below",
        "line": {"width": 0},
    }


def _transition_matrix(pred: pd.DataFrame) -> pd.DataFrame:
    s = pd.to_numeric(pred["hmm_state"], errors="coerce").dropna().astype(int)
    prev = s.shift(1)
    mat = pd.crosstab(prev, s, normalize="index").fillna(0.0)
    for k in [1, 2, 3, 4, 5]:
        if k not in mat.index:
            mat.loc[k] = 0.0
        if k not in mat.columns:
            mat[k] = 0.0
    return mat.sort_index().sort_index(axis=1)


def _fmt(x: Any) -> str:
    try:
        if x is None or pd.isna(x):
            return "NA"
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def _dict_table(d: dict[str, Any]) -> str:
    if not d:
        return "<p>No data.</p>"
    rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in d.items())
    return f"<table>{rows}</table>"


def _df_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<p>No data.</p>"
    return df.to_html(index=False, escape=False, classes="dataframe")
