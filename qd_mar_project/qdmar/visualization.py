"""Interactive dark-theme HTML visualization for QD-MAR.

All time-series plots include a Plotly range selector / range slider so the
researcher can select any interval and expand it across the X-axis.  Legends are
placed below figures and accompanied by an explicit legend explanation block.
"""
from __future__ import annotations

from pathlib import Path
import html
import uuid
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import Config

DARK_BG = "#080b12"
PANEL_BG = "#0f1624"
GRID = "#263246"
CYAN = "#00d4ff"
MAGENTA = "#ff4fd8"
GREEN = "#22ff9a"
AMBER = "#ffd166"
RED = "#ff5c7a"
MUTED = "#8fb3ff"


def _range_selector() -> dict:
    return dict(
        buttons=[
            dict(count=1, label="1D", step="day", stepmode="backward"),
            dict(count=7, label="1W", step="day", stepmode="backward"),
            dict(count=1, label="1M", step="month", stepmode="backward"),
            dict(count=3, label="3M", step="month", stepmode="backward"),
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(step="all", label="All"),
        ],
        bgcolor="#111b2e",
        activecolor="#1e88ff",
        font=dict(color="#d7e3ff"),
    )


def _axis_order_key(layout_key: str) -> int:
    """Return Plotly x-axis order from layout key, e.g. xaxis -> 1, xaxis4 -> 4."""
    suffix = layout_key.replace("xaxis", "")
    return int(suffix) if suffix else 1


def _xaxis_layout_keys(fig: go.Figure) -> list[str]:
    """List x-axis layout keys in display order."""
    keys = [k for k in fig.layout if isinstance(k, str) and k.startswith("xaxis")]
    return sorted(keys, key=_axis_order_key)


def _fig_has_date_x(fig: go.Figure) -> bool:
    """Infer whether a figure is a time-series figure from its trace x values."""
    for trace in fig.data:
        x = getattr(trace, "x", None)
        if x is None:
            continue
        try:
            vals = list(x)
        except TypeError:
            continue
        for v in vals[:20]:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            # Numeric x values can be timestamps as int64; they are not treated as
            # user-facing date axes here.  Plotly date axes are generated from
            # datetime-like objects or ISO strings.
            if isinstance(v, (pd.Timestamp, np.datetime64)):
                return True
            if isinstance(v, str) and pd.notna(pd.to_datetime(v, utc=True, errors="coerce")):
                return True
            break
    return False


def _configure_time_axes(fig: go.Figure) -> None:
    """Configure one effective Plotly date-range control per time-series figure.

    The previous implementation attached a range selector / range slider to every
    subplot x-axis.  On shared-axis subplots this can create repeated controls
    that do not reliably drive the displayed range.  Here all x axes are forced
    to date axes; only the bottom axis receives Plotly's native range slider and
    range selector.  A separate custom date-control block in `_figure_card` then
    applies the selected range to every x-axis via Plotly.relayout, ensuring that
    the selected interval expands to occupy the full X-axis.
    """
    axis_keys = _xaxis_layout_keys(fig)
    if not axis_keys:
        return
    for key in axis_keys:
        ax = getattr(fig.layout, key)
        ax.update(type="date", rangeslider=dict(visible=False))
    bottom = axis_keys[-1]
    getattr(fig.layout, bottom).update(
        type="date",
        rangeslider=dict(visible=True, thickness=0.06, bgcolor="#0b1020", bordercolor="#263246", borderwidth=1),
        rangeselector=_range_selector(),
    )


def _apply_dark(fig: go.Figure, title: str = "", time_axis: bool = True) -> go.Figure:
    """Apply dark terminal style and optional time-axis controls."""
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(color="#d7e3ff"),
        hovermode="x unified",
        dragmode="zoom",
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="top",
            y=-0.24,
            xanchor="left",
            x=0,
            tracegroupgap=8,
        ),
        margin=dict(l=55, r=35, t=75, b=150),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zerolinecolor="#40506d")
    if time_axis:
        _configure_time_axes(fig)
    return fig


def _fig_html(fig: go.Figure, include_plotlyjs: str | bool = "cdn", div_id: str | None = None) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs, div_id=div_id, config={
        "displaylogo": False,
        "scrollZoom": True,
        "responsive": True,
        "modeBarButtonsToAdd": ["drawline", "drawrect", "eraseshape"],
    })


def _figure_time_bounds(fig: go.Figure) -> tuple[str, str] | None:
    """Return min/max datetime-local values for a time figure."""
    vals: list[pd.Timestamp] = []
    for trace in fig.data:
        x = getattr(trace, "x", None)
        if x is None:
            continue
        try:
            xs = pd.to_datetime(list(x), utc=True, errors="coerce")
        except Exception:
            continue
        xs = xs[pd.notna(xs)]
        if len(xs):
            vals.extend(list(xs))
    if not vals:
        return None
    mn = min(vals).tz_convert("UTC") if hasattr(min(vals), "tz_convert") else pd.Timestamp(min(vals), tz="UTC")
    mx = max(vals).tz_convert("UTC") if hasattr(max(vals), "tz_convert") else pd.Timestamp(max(vals), tz="UTC")
    def fmt(ts: pd.Timestamp) -> str:
        return ts.strftime("%Y-%m-%dT%H:%M")
    return fmt(mn), fmt(mx)


def _time_control_html(div_id: str, bounds: tuple[str, str] | None) -> str:
    """Custom time-range controller that reliably zooms all subplot x-axes."""
    start, end = bounds if bounds else ("", "")
    return f"""
    <div class='time-control' data-target='{html.escape(div_id)}'>
      <div class='time-row'>
        <span class='time-title'>时间轴选择 / UTC</span>
        <label>开始 <input type='datetime-local' class='time-start' value='{html.escape(start)}'></label>
        <label>结束 <input type='datetime-local' class='time-end' value='{html.escape(end)}'></label>
        <button type='button' class='time-apply'>放大所选区间</button>
        <button type='button' class='time-reset'>全历史</button>
      </div>
      <div class='quick-row'>
        <button type='button' data-range='1d'>1D</button>
        <button type='button' data-range='1w'>1W</button>
        <button type='button' data-range='1m'>1M</button>
        <button type='button' data-range='3m'>3M</button>
        <button type='button' data-range='1y'>1Y</button>
        <button type='button' data-range='all'>All</button>
        <span class='time-hint'>选择开始/结束并点击“放大所选区间”，该时间段会铺满整张图的 X 轴；底部 range slider 与鼠标框选也可继续使用。</span>
      </div>
    </div>
    <script>
    (function() {{
      const targetId = {div_id!r};
      const card = document.querySelector(".time-control[data-target='" + targetId + "']");
      if (!card) return;
      const startEl = card.querySelector('.time-start');
      const endEl = card.querySelector('.time-end');
      function getPlot() {{ return document.getElementById(targetId); }}
      function toUtc(v) {{
        if (!v) return null;
        return v.length === 16 ? v + ':00Z' : v + 'Z';
      }}
      function inputFromDate(d) {{
        const pad = (n) => String(n).padStart(2, '0');
        return d.getUTCFullYear() + '-' + pad(d.getUTCMonth()+1) + '-' + pad(d.getUTCDate()) + 'T' + pad(d.getUTCHours()) + ':' + pad(d.getUTCMinutes());
      }}
      function axisUpdate(startIso, endIso) {{
        const plot = getPlot();
        if (!plot || typeof Plotly === 'undefined') return;
        const update = {{}};
        const layout = plot._fullLayout ? plot._fullLayout : {{}};
        Object.keys(layout).forEach(function(k) {{
          if (/^xaxis[0-9]*$/.test(k)) {{
            update[k + '.range'] = [startIso, endIso];
            update[k + '.autorange'] = false;
          }}
        }});
        if (Object.keys(update).length === 0) {{
          update['xaxis.range'] = [startIso, endIso];
          update['xaxis.autorange'] = false;
        }}
        Plotly.relayout(plot, update);
      }}
      function applyRange() {{
        const s = toUtc(startEl.value);
        const e = toUtc(endEl.value);
        if (!s || !e) {{ alert('请同时选择开始和结束时间。'); return; }}
        if (new Date(s).getTime() >= new Date(e).getTime()) {{ alert('开始时间必须早于结束时间。'); return; }}
        axisUpdate(s, e);
      }}
      function resetRange() {{
        const plot = getPlot();
        if (!plot || typeof Plotly === 'undefined') return;
        const update = {{}};
        const layout = plot._fullLayout ? plot._fullLayout : {{}};
        Object.keys(layout).forEach(function(k) {{
          if (/^xaxis[0-9]*$/.test(k)) update[k + '.autorange'] = true;
        }});
        if (Object.keys(update).length === 0) update['xaxis.autorange'] = true;
        Plotly.relayout(plot, update);
      }}
      function quickRange(code) {{
        if (code === 'all') {{ resetRange(); return; }}
        const end = endEl.value ? new Date(toUtc(endEl.value)) : new Date(toUtc({end!r}));
        const start = new Date(end.getTime());
        if (code === '1d') start.setUTCDate(start.getUTCDate() - 1);
        if (code === '1w') start.setUTCDate(start.getUTCDate() - 7);
        if (code === '1m') start.setUTCMonth(start.getUTCMonth() - 1);
        if (code === '3m') start.setUTCMonth(start.getUTCMonth() - 3);
        if (code === '1y') start.setUTCFullYear(start.getUTCFullYear() - 1);
        startEl.value = inputFromDate(start);
        endEl.value = inputFromDate(end);
        applyRange();
      }}
      card.querySelector('.time-apply').addEventListener('click', applyRange);
      card.querySelector('.time-reset').addEventListener('click', resetRange);
      card.querySelectorAll('[data-range]').forEach(function(btn) {{
        btn.addEventListener('click', function() {{ quickRange(btn.getAttribute('data-range')); }});
      }});
    }})();
    </script>
    """


def _legend_note(items: dict[str, str] | None = None, note: str = "") -> str:
    if not items and not note:
        return ""
    rows = ""
    if items:
        rows = "".join(
            f"<div class='legend-item'><span class='legend-key'>{html.escape(k)}</span><span>{html.escape(v)}</span></div>"
            for k, v in items.items()
        )
    note_html = f"<p>{html.escape(note)}</p>" if note else ""
    return f"<div class='legend-note'><b>图例与读图说明</b>{note_html}<div class='legend-grid'>{rows}</div></div>"


def _figure_card(fig: go.Figure, title: str, explanation: str, legend_items: dict[str, str] | None, include_plotlyjs: str | bool) -> str:
    div_id = "fig_" + uuid.uuid4().hex
    is_time = _fig_has_date_x(fig)
    controls = _time_control_html(div_id, _figure_time_bounds(fig)) if is_time else ""
    note = (
        "时间序列图已提供独立 UTC 时间选择器：选择开始/结束后点击“放大所选区间”，该时间段会铺满整张图的 X 轴；也可使用底部 range slider、鼠标拖拽缩放和滚轮缩放。"
        if is_time else
        "本图不是时间序列图，因此不显示时间轴选择器；可通过 legend 点击控制显示/隐藏。"
    )
    return (
        f"<div class='card'><h2>{html.escape(title)}</h2>"
        f"<p class='explain'>{html.escape(explanation)}</p>"
        f"{controls}"
        f"{_fig_html(fig, include_plotlyjs, div_id=div_id)}"
        f"{_legend_note(legend_items, note)}"
        "</div>"
    )


HTML_DEFAULTS = {
    "index": "index.html",
    "quality_assessment": "quality_assessment.html",
    "feature_statistics": "feature_statistics.html",
    "market_response_dashboard": "market_response_dashboard.html",
    "calibration": "calibration.html",
    "curve_dashboard": "curve_dashboard.html",
    "path_absorption_dashboard": "path_absorption_dashboard.html",
    "scenario_examples": "scenario_examples.html",
    "rolling_monitoring": "rolling_monitoring.html",
    "extreme_events": "extreme_events.html",
}


def _html_name(cfg: Config, key: str) -> str:
    return cfg.html_filename(key, HTML_DEFAULTS[key])


def _html_path(html_dir: Path, cfg: Config, key: str) -> Path:
    return html_dir / _html_name(cfg, key)


def _nav_html(cfg: Config | None = None) -> str:
    def name(key: str) -> str:
        return _html_name(cfg, key) if cfg is not None else HTML_DEFAULTS[key]
    return f"""
    <div class='nav'>
      <a href='{html.escape(name("index"))}'>Index</a> ·
      <a href='{html.escape(name("quality_assessment"))}'>Quality Assessment</a> ·
      <a href='{html.escape(name("feature_statistics"))}'>Feature Statistics</a> ·
      <a href='{html.escape(name("market_response_dashboard"))}'>Market Response</a> ·
      <a href='{html.escape(name("calibration"))}'>Calibration</a> ·
      <a href='{html.escape(name("curve_dashboard"))}'>Absorption Curve</a> ·
      <a href='{html.escape(name("path_absorption_dashboard"))}'>Path Absorption</a> ·
      <a href='{html.escape(name("scenario_examples"))}'>Scenario Examples</a> ·
      <a href='{html.escape(name("rolling_monitoring"))}'>Rolling Monitoring</a> ·
      <a href='{html.escape(name("extreme_events"))}'>Extreme Events</a>
    </div>
    """


def _page(title: str, body: str, cfg: Config | None = None, nav: str = "") -> str:
    css = f"""
    <style>
      body {{ background:{DARK_BG}; color:#d7e3ff; font-family:Inter,Segoe UI,Arial,sans-serif; margin:0; }}
      .wrap {{ padding: 24px 28px; }}
      a {{ color:{CYAN}; text-decoration:none; }}
      a:hover {{ text-decoration:underline; }}
      .nav {{ background:#050711; padding:14px 28px; border-bottom:1px solid #202a3d; position:sticky; top:0; z-index:10; }}
      .card {{ background:{PANEL_BG}; border:1px solid #202a3d; border-radius:14px; padding:16px; margin:18px 0; box-shadow:0 0 22px rgba(0,212,255,.08); overflow:hidden; }}
      .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:14px; }}
      .metric {{ background:linear-gradient(135deg,#101a2c,#0b1020); border:1px solid #263246; border-radius:14px; padding:16px; }}
      .metric h3 {{ margin:0 0 8px; color:#8fb3ff; font-size:13px; letter-spacing:.04em; text-transform:uppercase; }}
      .metric .v {{ font-size:23px; font-weight:700; color:{CYAN}; word-break:break-word; }}
      .legend-note {{ margin-top:10px; padding:12px 14px; border:1px solid #263246; border-radius:10px; background:#0b1020; color:#c9d8ff; font-size:13px; }}
      .legend-note b {{ color:#ffffff; }}
      .legend-note p {{ margin:6px 0 8px; color:#9db7e8; }}
      .legend-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:8px 16px; }}
      .legend-item {{ display:flex; gap:8px; align-items:flex-start; }}
      .legend-key {{ min-width:130px; color:{CYAN}; font-weight:700; }}
      .time-control {{ margin:12px 0 10px; padding:12px 14px; border:1px solid #29405f; border-radius:12px; background:linear-gradient(135deg,#0b1322,#101b2d); box-shadow:0 0 16px rgba(0,212,255,.06); }}
      .time-row, .quick-row {{ display:flex; flex-wrap:wrap; align-items:center; gap:10px; }}
      .quick-row {{ margin-top:8px; }}
      .time-title {{ color:{CYAN}; font-weight:800; margin-right:4px; }}
      .time-control label {{ color:#c9d8ff; font-size:13px; }}
      .time-control input {{ background:#060a12; color:#d7e3ff; border:1px solid #2b3b58; border-radius:8px; padding:6px 8px; }}
      .time-control button {{ background:#17233a; color:#d7e3ff; border:1px solid #35517a; border-radius:8px; padding:6px 10px; cursor:pointer; }}
      .time-control button:hover {{ background:#1e88ff; color:white; }}
      .time-hint {{ color:#9db7e8; font-size:12px; }}
      .explain {{ color:#b9c9ee; }}
      table {{ border-collapse:collapse; width:100%; font-size:13px; }}
      th, td {{ border:1px solid #263246; padding:6px 8px; text-align:left; }}
      th {{ background:#111b2e; color:#9bc4ff; }}
      h1,h2,h3 {{ color:#ffffff; }}
      code {{ color:{GREEN}; }}
    </style>
    """
    default_nav = _nav_html(cfg)
    return f"<html><head><meta charset='utf-8'><title>{html.escape(title)}</title>{css}</head><body>{nav or default_nav}<div class='wrap'>{body}</div></body></html>"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _format_value(v) -> str:
    if isinstance(v, float):
        if np.isnan(v):
            return "NA"
        return f"{v:.4f}"
    return html.escape(str(v))


def build_index(html_dir: Path, summary: dict, cfg: Config) -> None:
    cards = "".join(
        f"<div class='metric'><h3>{html.escape(k)}</h3><div class='v'>{_format_value(v)}</div></div>"
        for k, v in summary.items() if k != "latest_agent_inputs"
    )
    agent_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{_format_value(v)}</td></tr>" for k, v in summary.get("latest_agent_inputs", {}).items()
    )
    page_links = {
        "quality_assessment": "输出质量、方向样本比例、staleness、Agent 冗余和金融一致性总览。",
        "feature_statistics": "Agent 输入与重要输出变量的统计、分布、关系图。",
        "market_response_dashboard": "PLIE、actual response、market response label 总览。",
        "calibration": "q65 coverage、response percentile、label proportions。",
        "curve_dashboard": "20m/30m/60m absorption curve。",
        "path_absorption_dashboard": "6h/12h/24h/48h 路径级压力拒绝、级联传导、混合压力与主动主导。",
        "scenario_examples": "每类 path context / label 的典型案例，便于人工机制核对。",
        "rolling_monitoring": "price + PLIE + matured MAR memory 终端式监控。",
        "extreme_events": "极端事件复盘。",
    }
    links_html = "".join(
        f"<li><a href='{html.escape(_html_name(cfg, key))}'>{html.escape(_html_name(cfg, key))}</a> — {html.escape(desc)}</li>"
        for key, desc in page_links.items()
    )
    body = f"""
    <h1>QD-MAR Market Absorption Dashboard</h1>
    <div class='card'>
      <p><b>QD-MAR</b> 是 Quantile-Calibrated Denoised Market Absorption Rate。v2 优化版同时包含 <b>event-level matured absorption</b>、<b>staleness-aware memory</b> 与 <b>path-level episode absorption</b>：后者用于识别持续清算压力被价格路径拒绝、吸收或反向接管的场景。</p>
      <p>核心原则：当前 PLIE 可实时使用；event absorption 只有在 <code>available_time = event_time + horizon</code> 后成熟；Agent 只能读取 rolling matured absorption memory。</p>
    </div>
    <div class='grid'>{cards}</div>
    <div class='card'><h2>Latest Agent Inputs</h2><table><tr><th>Feature</th><th>Value</th></tr>{agent_rows}</table></div>
    <div class='card'><h2>Pages</h2>
      <ul>
        {links_html}
      </ul>
    </div>
    """
    _write(_html_path(html_dir, cfg, "index"), _page("QD-MAR Index", body, cfg))


def build_market_response_dashboard(html_dir: Path, base_df: pd.DataFrame, event_df: pd.DataFrame, cfg: Config) -> None:
    main = event_df[event_df["horizon"].eq(cfg.get("memory", "main_horizon", default="30m"))].copy().sort_values("event_time")
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.035,
                        subplot_titles=("Price & PLIE main", "Aligned actual response vs raw PLIE reference", "Response percentile / absorption score", "Labels"))
    fig.add_trace(go.Scatter(x=base_df["time"], y=base_df["price"], name="price", line=dict(color=CYAN, width=1)), row=1, col=1)
    if "plie_main_bps" in base_df:
        fig.add_trace(go.Scatter(x=base_df["time"], y=base_df["plie_main_bps"], name="plie_main_bps", line=dict(color=AMBER, width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=main["event_time"], y=main["aligned_actual_response_bps"], name="Y aligned", line=dict(color=GREEN, width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=main["event_time"], y=main["plie_reference_raw_bps"], name="Braw q65 reference", line=dict(color=AMBER, width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=main["event_time"], y=main["response_percentile"], name="response_percentile U", line=dict(color=MAGENTA, width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=main["event_time"], y=main["absorption_score_q_0_100"], name="abs_score_q", line=dict(color=CYAN, width=1)), row=3, col=1)
    cats = pd.Categorical(main["market_response_label"])
    label_codes = cats.codes
    fig.add_trace(go.Scatter(x=main["event_time"], y=label_codes, name="market_response_label", mode="markers", marker=dict(size=4, color=label_codes, colorscale="Turbo"), text=main["market_response_label"]), row=4, col=1)
    fig.update_yaxes(title_text="label code", row=4, col=1)
    _apply_dark(fig, "QD-MAR Market Response Dashboard", time_axis=True)
    body = "<h1>Market Response Dashboard</h1>" + _figure_card(
        fig,
        "PLIE 与实际响应总览",
        "用于检查实际价格是否服从 PLIE、是否出现同向放大、吸收或反向接管。第四行使用 label code 显示标签序列，hover 可看到原始标签。",
        {
            "price": "BTC 价格。",
            "plie_main_bps": "30m reliability-weighted signed PLIE，被动清算压力。",
            "Y aligned": "actual return 沿 PLIE 方向对齐后的实际响应。",
            "Braw q65 reference": "raw q65 passive impact reference，用作吸收率校准分母。",
            "response_percentile U": "历史可比 context 下的实际传导分位；越高越偏同向传导/放大。",
            "abs_score_q": "100*(1-U)；越高越偏吸收/反向接管。",
        },
        cfg.get("visualization", "include_plotlyjs"),
    )
    _write(_html_path(html_dir, cfg, "market_response_dashboard"), _page("Market Response", body, cfg))


def build_calibration_page(html_dir: Path, coverage: pd.DataFrame, labels: pd.DataFrame, percentile: pd.DataFrame, cfg: Config) -> None:
    fig1 = go.Figure()
    core = coverage[coverage["response_context"].eq("directional_core")]
    for h, grp in core.groupby("horizon"):
        fig1.add_trace(go.Bar(x=grp["split"], y=grp["coverage"], name=h, text=grp["n"]))
    fig1.add_hline(y=0.65, line_dash="dash", line_color=AMBER, annotation_text="q65 target")
    _apply_dark(fig1, "Directional-core q65 coverage: P(Y <= Braw)", time_axis=False)

    fig2 = go.Figure()
    for (h, sp), grp in labels.groupby(["horizon", "split"]):
        fig2.add_trace(go.Bar(x=grp["market_response_label"], y=grp["proportion"], name=f"{h}-{sp}"))
    _apply_dark(fig2, "Market response label proportions", time_axis=False)

    table_html = coverage.round(4).to_html(index=False, classes="table")
    body = "<h1>Calibration Diagnostics</h1>"
    body += _figure_card(fig1, "Directional-core q65 覆盖率", "仅 directional_core 样本应围绕 q65 目标检查。Neutral PLIE 不应参与方向 q65 解释。", {"20m/30m/60m": "不同响应 horizon。柱上 text 为样本数量。"}, cfg.get("visualization","include_plotlyjs"))
    body += _figure_card(fig2, "Market response label proportions", "检查各 horizon / split 的标签占比是否严重漂移。", {"label proportions": "包含 directional 与 neutral 分支标签，需结合 context 分布解读。"}, False)
    body += f"<div class='card'><h2>Coverage table</h2>{table_html}</div>"
    _write(_html_path(html_dir, cfg, "calibration"), _page("Calibration", body, cfg))


def build_curve_dashboard(html_dir: Path, curve_df: pd.DataFrame, cfg: Config) -> None:
    fig_curve = go.Figure()
    for col, color in [("u20", CYAN), ("u30", AMBER), ("u60", MAGENTA)]:
        if col in curve_df:
            fig_curve.add_trace(go.Scatter(x=curve_df["event_time"], y=curve_df[col], name=col, line=dict(width=1, color=color)))
    _apply_dark(fig_curve, "Multi-Horizon Absorption Curve Percentiles", time_axis=True)

    counts = curve_df["mar_curve_label"].value_counts().reset_index()
    counts.columns = ["label", "n"]
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=counts["label"], y=counts["n"], name="curve labels", marker_color=CYAN))
    _apply_dark(fig_bar, "Curve label distribution", time_axis=False)

    body = "<h1>Absorption Curve Dashboard</h1>"
    body += _figure_card(fig_curve, "20m / 30m / 60m response percentile curve", "用于判断即时承接、主 horizon 响应和延迟传导/吸收之间的形态差异。", {"u20": "20m response percentile，偏即时承接。", "u30": "30m 主吸收率分位。", "u60": "60m 延迟吸收/级联分位。"}, cfg.get("visualization", "include_plotlyjs"))
    body += _figure_card(fig_bar, "Curve label distribution", "展示 multi-horizon curve 的形态标签占比。", {"curve labels": "persistent cascade、delayed absorption、mixed/noise 等形态。"}, False)
    _write(_html_path(html_dir, cfg, "curve_dashboard"), _page("Curve", body, cfg))


def build_rolling_monitoring(html_dir: Path, memory_df: pd.DataFrame, cfg: Config) -> None:
    main_h = cfg.get("memory", "main_horizon", default="30m")
    max_points = int(cfg.get("visualization", "max_points", default=8000))
    step = max(1, int(np.ceil(len(memory_df) / max_points)))
    memory_plot = memory_df.iloc[::step].copy()
    fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        subplot_titles=("Price", "PLIE main", "Staleness-aware absorption score", "Staleness-aware active force", "Directional freshness / quality", "Persistence / takeover"))
    fig.add_trace(go.Scatter(x=memory_plot["time"], y=memory_plot.get("price"), name="price", line=dict(color=CYAN, width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=memory_plot["time"], y=memory_plot.get("plie_main_bps"), name="plie_main_bps", line=dict(color=AMBER, width=1)), row=2, col=1)
    col_abs = f"mar_abs_score_q_staleaware_ewm_6_{main_h}"
    col_af = f"mar_active_force_price_staleaware_ewm_6_{main_h}"
    fig.add_trace(go.Scatter(x=memory_plot["time"], y=memory_plot.get(col_abs), name=col_abs, line=dict(color=MAGENTA, width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=memory_plot["time"], y=memory_plot.get(col_af), name=col_af, line=dict(color=GREEN, width=1)), row=4, col=1)
    fig.add_trace(go.Scatter(x=memory_plot["time"], y=memory_plot.get(f"mar_directional_core_freshness_{main_h}"), name="directional_core_freshness", line=dict(color=CYAN, width=1)), row=5, col=1)
    fig.add_trace(go.Scatter(x=memory_plot["time"], y=memory_plot.get(f"mar_directional_quality_ewm_6_{main_h}"), name="directional_quality_ewm_6", line=dict(color=AMBER, width=1)), row=5, col=1)
    fig.add_trace(go.Scatter(x=memory_plot["time"], y=memory_plot.get(f"mar_amplification_persistence_6_{main_h}"), name="amplification_persistence", line=dict(color=RED, width=1)), row=6, col=1)
    fig.add_trace(go.Scatter(x=memory_plot["time"], y=memory_plot.get(f"mar_takeover_count_12_{main_h}"), name="takeover_count", line=dict(color=CYAN, width=1)), row=6, col=1)
    _apply_dark(fig, "Rolling Matured Absorption Memory", time_axis=True)
    body = "<h1>Rolling Monitoring</h1>" + _figure_card(
        fig,
        "实时 Agent 记忆监控",
        "该图展示推荐 Agent 输入的 staleness-aware 版本。当 directional_core 事件长时间不出现时，吸收分数回到 50，active force 回到 0。",
        {
            "staleness-aware absorption": "50 为中性；高于 50 偏吸收，低于 50 偏传导/放大。",
            "active force price": "价格方向坐标下的主动力量 proxy。",
            "directional freshness": "最近 directional-core 成熟事件的新鲜度，1 为刚成熟，接近 0 为陈旧。",
            "directional quality": "近期 directional-core 证据权重。",
            "persistence / takeover": "放大持续性与反向接管计数。",
        },
        cfg.get("visualization", "include_plotlyjs"),
    )
    _write(_html_path(html_dir, cfg, "rolling_monitoring"), _page("Rolling Monitoring", body, cfg))


def build_feature_statistics(html_dir: Path, memory_df: pd.DataFrame, event_df: pd.DataFrame, cfg: Config) -> None:
    """Build all-agent-feature statistics page."""
    agent_cols = [c for c in cfg.agent_inputs if c in memory_df.columns]
    numeric_cols = [c for c in agent_cols if pd.api.types.is_numeric_dtype(memory_df[c])]
    # Plot a representative subset to keep the interactive page responsive;
    # the statistics table above still covers every configured Agent input.
    priority = [
        "mar_abs_score_q_staleaware_ewm_6_30m",
        "mar_active_force_price_staleaware_ewm_6_30m",
        "mar_directional_core_freshness_30m",
        "mar_directional_quality_ewm_6_30m",
        "mar_episode_abs_score_24h",
        "mar_episode_pressure_rejection_score_24h",
        "mar_episode_active_force_price_24h",
        "mar_episode_active_dominance_score_24h",
        "mar_episode_active_dominance_price_score_24h",
        "mar_episode_active_z_24h",
        "mar_episode_liq_neutrality_score_24h",
        "mar_episode_quality_24h",
        "mar_neutral_active_strength_evidence_ewm_6_30m",
        "mar_neutral_context_persistence_12_30m",
        "mar_amplification_persistence_6_30m",
        "mar_takeover_count_12_30m",
    ]
    numeric_plot_cols = [c for c in priority if c in numeric_cols] + [c for c in numeric_cols if c not in priority]
    numeric_plot_cols = numeric_plot_cols[:12]
    main_h = cfg.get("memory", "main_horizon", default="30m")
    main_event = event_df[event_df["horizon"].eq(main_h)].copy().sort_values("event_time")

    stats_rows = []
    for col in agent_cols:
        s = memory_df[col]
        is_num = pd.api.types.is_numeric_dtype(s)
        q1 = s.quantile(0.25) if is_num else np.nan
        q3 = s.quantile(0.75) if is_num else np.nan
        iqr = q3 - q1 if is_num else np.nan
        outlier_rate = (((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).mean() if is_num and np.isfinite(iqr) and iqr > 0 else np.nan)
        stats_rows.append({
            "feature": col, "dtype": str(s.dtype), "missing_rate": s.isna().mean(), "outlier_rate_iqr3": outlier_rate,
            "mean": s.mean() if is_num else np.nan, "std": s.std() if is_num else np.nan,
            "min": s.min() if is_num else np.nan, "p25": q1, "p50": s.median() if is_num else np.nan,
            "p75": q3, "max": s.max() if is_num else np.nan,
        })
    stats_table = pd.DataFrame(stats_rows).round(4).to_html(index=False)

    max_points = min(int(cfg.get("visualization", "max_points", default=8000)), 5000)
    step = max(1, int(np.ceil(len(memory_df) / max_points)))
    m_time = memory_df.iloc[::step].copy()

    fig_ts = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                           subplot_titles=("Agent input time series", "Rolling mean, 48 snapshots"))
    for col in numeric_plot_cols:
        fig_ts.add_trace(go.Scatter(x=m_time["time"], y=m_time[col], name=col, line=dict(width=1)), row=1, col=1)
        fig_ts.add_trace(go.Scatter(x=m_time["time"], y=m_time[col].rolling(48, min_periods=12).mean(), name=f"{col} rolling_mean", line=dict(width=1), visible="legendonly"), row=2, col=1)
    _apply_dark(fig_ts, "Agent Input Time Series and Rolling Statistics", time_axis=True)

    fig_dist = make_subplots(rows=2, cols=1, vertical_spacing=0.08,
                             subplot_titles=("Distributions", "Box by HMM state"))
    for col in numeric_plot_cols:
        fig_dist.add_trace(go.Histogram(x=memory_df[col], name=col, nbinsx=80, opacity=0.55), row=1, col=1)
        if "hmm_state" in memory_df:
            fig_dist.add_trace(go.Box(x=memory_df["hmm_state"].astype(str), y=memory_df[col], name=f"{col} by state", visible="legendonly"), row=2, col=1)
    fig_dist.update_layout(barmode="overlay")
    _apply_dark(fig_dist, "Agent Input Distribution and HMM-State Grouping", time_axis=False)

    m = m_time.copy()
    if not main_event.empty:
        left_rel = m[["time"] + numeric_cols + [c for c in ["plie_main_bps"] if c in m.columns]].copy()
        right_rel = main_event[["event_time", "actual_return_bps"]].copy()
        left_rel["time"] = pd.to_datetime(left_rel["time"], utc=True, errors="coerce")
        right_rel["event_time"] = pd.to_datetime(right_rel["event_time"], utc=True, errors="coerce")
        rel = pd.merge_asof(left_rel.sort_values("time"), right_rel.sort_values("event_time"), left_on="time", right_on="event_time", direction="backward")
    else:
        rel = m
        rel["actual_return_bps"] = np.nan

    fig_rel = make_subplots(rows=2, cols=1, vertical_spacing=0.08,
                            subplot_titles=("Feature vs PLIE main", "Feature vs actual 30m return"))
    relationship_cols = numeric_plot_cols[:8]
    for i, col in enumerate(relationship_cols):
        visible = True if i == 0 else "legendonly"
        if "plie_main_bps" in rel:
            fig_rel.add_trace(go.Scatter(x=rel["plie_main_bps"], y=rel[col], mode="markers", name=f"{col} vs PLIE", marker=dict(size=3, opacity=0.35), visible=visible), row=1, col=1)
        fig_rel.add_trace(go.Scatter(x=rel["actual_return_bps"], y=rel[col], mode="markers", name=f"{col} vs ret30", marker=dict(size=3, opacity=0.35), visible=visible), row=2, col=1)
    _apply_dark(fig_rel, "Agent Input Relationships", time_axis=False)

    body = f"""
    <h1>Feature Statistics</h1>
    <div class='card'><p>本页覆盖配置中的全部 Agent 输入变量。优化版推荐使用 staleness-aware directional memory，旧 raw EWM 字段仍保留在 CSV 中用于审计，但不再作为默认 Agent 输入。</p></div>
    <div class='card'><h2>Missing / Outlier / Basic Statistics</h2>{stats_table}</div>
    """
    body += _figure_card(fig_ts, "Agent 输入时间序列与滚动统计", "展示优先级最高的 Agent 输入变量；完整缺失率、异常值和基本统计仍在上方表格覆盖所有 Agent 输入。为保证全历史交互性能，本页对时间序列做等距下采样。", {"agent input": "默认 Agent 输入变量的代表性子集。", "rolling_mean": "48 个 source snapshot 的滚动均值。"}, cfg.get('visualization','include_plotlyjs'))
    body += _figure_card(fig_dist, "Agent 输入分布与 HMM state 分组", "用于检查缺失、异常、尾部和不同 HMM 状态下的分布差异。", {"histogram": "变量分布。", "box by state": "按 HMM state 分组的箱线图，默认 legendonly。"}, False)
    body += _figure_card(fig_rel, "Agent 输入与 PLIE / 实际收益关系", "检查 Agent 输入是否只是 PLIE 的重复参数化，以及是否过度贴近实际收益噪声。", {"vs PLIE": "横轴为 plie_main_bps。", "vs ret30": "横轴为成熟后的实际 30m return，仅用于诊断。"}, False)
    _write(_html_path(html_dir, cfg, "feature_statistics"), _page("Feature Statistics", body, cfg))


def build_quality_assessment(html_dir: Path, memory_df: pd.DataFrame, reports: dict, cfg: Config) -> None:
    contexts = reports.get("context_distribution", pd.DataFrame())
    q65 = reports.get("q65_coverage", pd.DataFrame())
    pct = reports.get("percentile_summary", pd.DataFrame())
    staleness = reports.get("staleness_diagnostics", pd.DataFrame())
    corr = reports.get("agent_feature_correlation", pd.DataFrame())
    exits = reports.get("state_exit_rates", pd.DataFrame())

    fig_ctx = go.Figure()
    if not contexts.empty:
        for ctx, grp in contexts[contexts["horizon"].eq(cfg.get("memory","main_horizon",default="30m"))].groupby("response_context"):
            fig_ctx.add_trace(go.Bar(x=grp["split"], y=grp["proportion"], name=ctx, text=grp["n"]))
    _apply_dark(fig_ctx, "30m response context distribution", time_axis=False)

    fig_q65 = go.Figure()
    if not q65.empty:
        core = q65[q65["response_context"].eq("directional_core")]
        for h, grp in core.groupby("horizon"):
            fig_q65.add_trace(go.Bar(x=grp["split"], y=grp["coverage"], name=h, text=grp["n"]))
        fig_q65.add_hline(y=0.65, line_dash="dash", line_color=AMBER)
    _apply_dark(fig_q65, "Directional-core q65 coverage", time_axis=False)

    fig_stale = go.Figure()
    if not staleness.empty:
        fig_stale.add_trace(go.Bar(x=staleness["metric"], y=staleness["value"], name="staleness metrics", marker_color=AMBER))
    _apply_dark(fig_stale, "Directional memory staleness diagnostics", time_axis=False)

    fig_corr = go.Figure()
    if not corr.empty:
        pivot = corr.pivot(index="feature_a", columns="feature_b", values="spearman_corr")
        fig_corr.add_trace(go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale="RdBu", zmin=-1, zmax=1, name="Spearman corr"))
    _apply_dark(fig_corr, "Agent input Spearman correlation", time_axis=False)

    fig_exit = go.Figure()
    if not exits.empty:
        main = exits[exits["window_hours"].eq(12)].copy()
        for ctx, grp in main.groupby("response_context"):
            fig_exit.add_trace(go.Bar(x=grp["market_response_label"], y=grp["exit_rate"], name=ctx, text=grp["n"]))
    _apply_dark(fig_exit, "12h HMM state-exit rate by context / label", time_axis=False)

    pct_table = pct.round(4).to_html(index=False) if not pct.empty else "<p>No percentile summary.</p>"
    body = "<h1>QD-MAR Output Quality Assessment</h1>"
    body += "<div class='card'><p>本页用于判断吸收率是否可作为 Agent 输入。核心判据：directional_core 内 q65 coverage 是否稳定、response percentile OOS 是否接近训练分布、directional 证据是否新鲜、Agent 输入是否存在过强冗余，以及标签与 HMM 状态退出是否符合金融机制。</p></div>"
    body += _figure_card(fig_ctx, "Context distribution", "directional_core 是真正可解释吸收率的样本；neutral/low-quality 不应强行解释为吸收。", {"directional_core": "PLIE 方向、reliability 与 SNR 均达标。", "low_quality/neutral": "进入主动行情或 AMB 语境，不计算 directional absorption。"}, cfg.get("visualization","include_plotlyjs"))
    body += _figure_card(fig_q65, "q65 coverage", "高质量 directional 样本中，P(Y <= Braw) 应接近 0.65。", {"coverage": "actual aligned response 不超过 raw PLIE q65 reference 的概率。"}, False)
    body += _figure_card(fig_stale, "Directional memory staleness", "检查旧 directional 信号是否可能长时间停留。优化版使用 freshness 对 Agent 输入衰减。", {"age_hours": "最近 directional-core 成熟事件距当前的小时数。", "freshness": "半衰期衰减后的证据新鲜度。"}, False)
    body += _figure_card(fig_corr, "Agent feature redundancy", "检查 Agent 输入之间的 Spearman 相关，避免过多同源特征重复进入。", {"heatmap": "红/蓝表示正/负相关；绝对值过高代表信息冗余。"}, False)
    body += _figure_card(fig_exit, "Financial consistency: state exit", "研究型诊断：reversal_takeover 通常应比 passive_amplification 更容易对应 HMM state exit。neutral 标签应按 context 解读。", {"exit_rate": "未来 12 个 source snapshots 内 hard HMM state 是否退出。"}, False)
    body += f"<div class='card'><h2>Response percentile summary</h2>{pct_table}</div>"
    _write(_html_path(html_dir, cfg, "quality_assessment"), _page("Quality Assessment", body, cfg))


def build_path_absorption_dashboard(html_dir: Path, base_df: pd.DataFrame, path_df: pd.DataFrame, memory_df: pd.DataFrame, reports: dict, cfg: Config) -> None:
    """Build path-level / episode-level absorption dashboard."""
    body = "<h1>Path-level Absorption Dashboard</h1>"
    if path_df is None or path_df.empty:
        body += "<div class='card'><p>No path-level absorption rows were generated.</p></div>"
        _write(_html_path(html_dir, cfg, "path_absorption_dashboard"), _page("Path Absorption", body, cfg))
        return

    main_window = 24 if 24 in set(path_df["window_hours"].dropna().astype(int)) else int(path_df["window_hours"].dropna().astype(int).iloc[0])
    pmain = path_df[path_df["window_hours"].eq(main_window)].copy().sort_values("time")
    # IMPORTANT: path absorption is a state-classification chart. It must show
    # every source-clock update so researchers can inspect each hourly
    # context/label transition. Do not downsample this page. Other heavy
    # diagnostic pages may still downsample for browser performance, but the
    # path dashboard is intentionally full-resolution.
    pmain_plot = pmain.copy()
    memory_plot = memory_df.copy()
    fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        subplot_titles=("Price & cumulative effective PLIE", "Path return vs aligned pressure", "Path absorption / rejection / active dominance", "Path quality / neutrality", "Path context / label", "Memory episode Agent fields"))
    fig.add_trace(go.Scatter(x=base_df["time"], y=base_df.get("price"), name="price", line=dict(color=CYAN, width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=pmain_plot["time"], y=pmain_plot["path_signed_plie_effective_sum_bps"], name=f"{main_window}h signed effective PLIE sum", line=dict(color=AMBER, width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=pmain_plot["time"], y=pmain_plot["path_return_bps"], name="path_return_bps", line=dict(color=CYAN, width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=pmain_plot["time"], y=pmain_plot["path_aligned_response_bps"], name="aligned_path_response", line=dict(color=GREEN, width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=pmain_plot["time"], y=pmain_plot["path_net_braw_bps"], name="net Braw path reference", line=dict(color=AMBER, width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=pmain_plot["time"], y=pmain_plot["path_absorption_score_0_100"], name="path_absorption_score", line=dict(color=MAGENTA, width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=pmain_plot["time"], y=100*pmain_plot["path_pressure_rejection_score"], name="100*pressure_rejection_score", line=dict(color=RED, width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=pmain_plot["time"], y=100*pmain_plot["path_cascade_score"], name="100*cascade_score", line=dict(color=GREEN, width=1)), row=3, col=1)
    if "path_active_dominance_score" in pmain_plot.columns:
        fig.add_trace(go.Scatter(x=pmain_plot["time"], y=100*pmain_plot["path_active_dominance_score"], name="100*active_dominance_score", line=dict(color=CYAN, width=1, dash="dot")), row=3, col=1)
    if "path_data_quality" in pmain_plot.columns:
        fig.add_trace(go.Scatter(x=pmain_plot["time"], y=pmain_plot["path_data_quality"], name="path_data_quality", line=dict(color=GREEN, width=1)), row=4, col=1)
    if "path_signal_clarity" in pmain_plot.columns:
        fig.add_trace(go.Scatter(x=pmain_plot["time"], y=pmain_plot["path_signal_clarity"], name="path_signal_clarity", line=dict(color=CYAN, width=1)), row=4, col=1)
    if "path_activity_level" in pmain_plot.columns:
        fig.add_trace(go.Scatter(x=pmain_plot["time"], y=pmain_plot["path_activity_level"], name="path_activity_level", line=dict(color=AMBER, width=1)), row=4, col=1)
    if "path_quality" in pmain_plot.columns:
        fig.add_trace(go.Scatter(x=pmain_plot["time"], y=pmain_plot["path_quality"], name="legacy_path_quality=data_quality×signal_clarity", line=dict(color=MUTED, width=1, dash="dot"), visible="legendonly"), row=4, col=1)
    fig.add_trace(go.Scatter(x=pmain_plot["time"], y=pmain_plot["path_direction_consistency"], name="directionality", line=dict(color=MAGENTA, width=1), visible="legendonly"), row=4, col=1)
    if "path_active_z" in pmain_plot.columns:
        fig.add_trace(go.Scatter(x=pmain_plot["time"], y=pmain_plot["path_active_z"], name="path_active_z", line=dict(color=RED, width=1), visible="legendonly"), row=4, col=1)
    label_codes = pd.Categorical(pmain_plot["path_label"]).codes
    ctx_codes = pd.Categorical(pmain_plot["path_context"]).codes
    fig.add_trace(go.Scatter(x=pmain_plot["time"], y=ctx_codes, name="path_context", mode="markers", marker=dict(size=4, color=ctx_codes, colorscale="Viridis"), text=pmain_plot["path_context"]), row=5, col=1)
    fig.add_trace(go.Scatter(x=pmain_plot["time"], y=label_codes, name="path_label", mode="markers", marker=dict(size=4, color=label_codes, colorscale="Turbo"), text=pmain_plot["path_label"]), row=5, col=1)
    for wh in [12,24,48]:
        col = f"mar_episode_abs_score_{wh}h"
        rej = f"mar_episode_pressure_rejection_score_{wh}h"
        if col in memory_df.columns:
            fig.add_trace(go.Scatter(x=memory_plot["time"], y=memory_plot[col], name=col, line=dict(width=1), visible=True if wh==24 else "legendonly"), row=6, col=1)
        if rej in memory_df.columns:
            fig.add_trace(go.Scatter(x=memory_plot["time"], y=100*memory_plot[rej], name=f"100*{rej}", line=dict(width=1, dash="dot"), visible=True if wh==24 else "legendonly"), row=6, col=1)
    _apply_dark(fig, f"Path-level QD-MAR ({main_window}h main view)", time_axis=True)

    labels = reports.get("path_label_proportions", pd.DataFrame())
    ctx = reports.get("path_context_distribution", pd.DataFrame())
    quality = reports.get("path_quality_summary", pd.DataFrame())
    fig_lab = go.Figure()
    if not labels.empty:
        for (wh, sp), grp in labels.groupby(["window_hours", "split"]):
            if int(wh) in [12,24,48]:
                fig_lab.add_trace(go.Bar(x=grp["path_label"], y=grp["proportion"], name=f"{int(wh)}h-{sp}", text=grp["n"]))
    _apply_dark(fig_lab, "Path label proportions", time_axis=False)

    tables = ""
    if not ctx.empty:
        tables += "<h2>Path context distribution</h2>" + ctx.round(4).to_html(index=False)
    if not quality.empty:
        tables += "<h2>Path quality summary</h2>" + quality.round(4).to_html(index=False)
    state_proxy = reports.get("state_proxy_distribution", pd.DataFrame())
    state_evidence = reports.get("state_evidence_summary", pd.DataFrame())
    examples = reports.get("typical_state_evidence_examples", pd.DataFrame())
    if state_proxy is not None and not state_proxy.empty:
        tables += "<h2>Audit-only state proxy distribution</h2>" + state_proxy.round(4).to_html(index=False)
    if state_evidence is not None and not state_evidence.empty:
        tables += "<h2>State evidence summary</h2>" + state_evidence.round(4).to_html(index=False)
    if examples is not None and not examples.empty:
        tables += "<h2>Typical mechanism examples</h2>" + examples.round(4).to_html(index=False)

    body += "<div class='card'><p>路径级吸收率使用当前与过去窗口内的价格和 PLIE 压力，不使用未来标签。它补足 20/30/60m event-level absorption 无法识别的持续压力拒绝、慢速吸收、路径级级联传导与反向接管。</p><p><b>显示说明：</b>本页主图不做降采样，完整显示每一个 source-clock path absorption 更新点。当前输入数据为小时级 source-clock，因此 path_context 与 path_label 会逐小时更新；6h/12h/24h/48h 表示回看窗口长度，不表示更新频率。</p></div>"
    body += _figure_card(fig, "路径级吸收 / 压力拒绝主图", "当累计 PLIE 方向明确而价格路径反向时，pressure_rejection_score 与 path_absorption_score 会升高；当价格顺着累计 PLIE 方向延续，cascade_score 会升高。", {
        "signed effective PLIE sum": "窗口内 reliability-weighted signed PLIE 累计。",
        "path_return_bps": "当前价格相对窗口起点的累计收益，只使用过去到当前。",
        "aligned_path_response": "沿累计 PLIE 方向对齐后的路径响应，负值代表压力被拒绝。",
        "path_absorption_score": "0-100，越高越偏路径级吸收/反向接管。",
        "pressure_rejection_score": "0-1，正向表示价格路径反向于累计清算压力。",
        "cascade_score": "0-1，正向表示价格路径顺着累计清算压力延续。",
        "path_data_quality": "只描述数据可靠性，不包含低活动惩罚。",
        "path_signal_clarity": "表示 PLIE context 与 label 语义是否清楚。",
        "path_activity_level": "表示市场活动强弱；低值可支持 RC，不代表坏质量。",
        "legacy_path_quality": "兼容字段，当前等于 data_quality × signal_clarity。",
        "active_dominance_score": "清算压力中性/混合时，价格路径大幅移动的主动主导强度。",
        "liq_neutrality_score": "清算压力越接近中性/抵消，该值越高。",
        "path_active_z": "路径收益相对 past-only 路径波动的标准化幅度。",
    }, cfg.get("visualization", "include_plotlyjs"))
    body += _figure_card(fig_lab, "路径标签占比", "检查 path-level label 在不同窗口与 split 下是否严重漂移。", {"path_label": "路径级响应标签。"}, False)
    body += f"<div class='card'>{tables}</div>"
    _write(_html_path(html_dir, cfg, "path_absorption_dashboard"), _page("Path Absorption", body, cfg))



def build_scenario_examples(html_dir: Path, reports: dict, cfg: Config) -> None:
    """Build representative path scenario examples page."""
    scenarios = reports.get("path_scenario_examples", pd.DataFrame())
    body = "<h1>Path Scenario Examples</h1>"
    body += """
    <div class='card'>
      <p>本页用于人工检查不同清算压力方向与价格路径反应的组合是否被正确分类。每行是 24h 主路径窗口中某个 <code>path_pressure_name × path_context × path_label</code> 组合的代表性样本。</p>
      <p><b>读法：</b>directional context 用 <code>path_transmission_ratio</code> 判断传导、吸收、拒绝；neutral / mixed context 用 <code>path_active_z</code> 与 <code>path_active_dominance_score</code> 判断主动主导。</p>
    </div>
    """
    if scenarios is None or scenarios.empty:
        body += "<div class='card'><p>No scenario examples were generated.</p></div>"
        _write(_html_path(html_dir, cfg, "scenario_examples"), _page("Scenario Examples", body, cfg))
        return

    cols = [c for c in [
        "time", "window_hours", "split", "hmm_state", "path_pressure_name",
        "path_context", "path_label", "path_return_bps",
        "path_signed_plie_effective_sum_bps", "path_raw_plie_total_bps",
        "path_net_braw_bps", "path_direction_consistency",
        "path_liq_neutrality_score", "path_snr", "path_active_z",
        "path_transmission_ratio", "path_absorption_score_0_100",
        "path_quality", "path_pressure_rejection_score", "path_cascade_score",
        "path_active_dominance_score", "path_active_dominance_price_score",
    ] if c in scenarios.columns]
    table = scenarios[cols].round(4).to_html(index=False)

    fig = go.Figure()
    if {"path_return_bps", "path_signed_plie_effective_sum_bps"}.issubset(scenarios.columns):
        fig.add_trace(go.Scatter(
            x=scenarios["path_signed_plie_effective_sum_bps"],
            y=scenarios["path_return_bps"],
            mode="markers+text",
            text=scenarios["path_label"],
            textposition="top center",
            name="scenario",
            marker=dict(size=10, color=scenarios.get("path_active_z", pd.Series(0, index=scenarios.index)), colorscale="Turbo", showscale=True),
            customdata=scenarios[["time", "path_context", "path_pressure_name"]].astype(str) if {"time","path_context","path_pressure_name"}.issubset(scenarios.columns) else None,
            hovertemplate="signed PLIE=%{x:.2f}<br>path return=%{y:.2f}<br>label=%{text}<extra></extra>",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="#40506d")
        fig.add_vline(x=0, line_dash="dash", line_color="#40506d")
    _apply_dark(fig, "Scenario examples: signed PLIE sum vs path return", time_axis=False)

    body += _figure_card(
        fig,
        "典型场景散点图",
        "横轴为窗口内 signed effective PLIE 累计，纵轴为路径收益。左上/右下通常是压力拒绝或反向接管；横轴接近 0 但纵轴很大是主动主导。",
        {
            "signed PLIE sum": "累计清算压力，正为向上，负为向下。",
            "path return": "窗口内已发生价格路径收益。",
            "color": "path_active_z，颜色越亮代表主动路径幅度越显著。",
        },
        cfg.get("visualization", "include_plotlyjs"),
    )
    body += f"<div class='card'><h2>Representative scenario table</h2>{table}</div>"
    _write(_html_path(html_dir, cfg, "scenario_examples"), _page("Scenario Examples", body, cfg))

def build_extreme_events(html_dir: Path, base_df: pd.DataFrame, event_df: pd.DataFrame, cfg: Config) -> None:
    main_h = cfg.get("memory", "main_horizon", default="30m")
    main = event_df[event_df["horizon"].eq(main_h)].copy()
    candidates = pd.concat([
        main.nlargest(10, "plie_reference_raw_bps"),
        main.nlargest(10, "z_amp"),
        main.nsmallest(10, "response_percentile"),
    ]).drop_duplicates("event_time").sort_values("event_time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=base_df["time"], y=base_df["price"], name="price", line=dict(color=CYAN, width=1)))
    if not candidates.empty:
        price_x = pd.to_datetime(base_df["time"]).astype("int64")
        event_x = pd.to_datetime(candidates["event_time"]).astype("int64")
        fig.add_trace(go.Scatter(
            x=candidates["event_time"],
            y=np.interp(event_x, price_x, base_df["price"]),
            mode="markers",
            marker=dict(size=9, color=RED, symbol="diamond"),
            name="extreme MAR events",
            text=candidates["market_response_label"],
        ))
    _apply_dark(fig, "Extreme QD-MAR Events", time_axis=True)
    table = candidates[["event_time", "market_response_label", "plie_reference_raw_bps", "actual_return_bps", "response_percentile", "z_amp", "z_takeover"]].round(4).to_html(index=False)
    body = "<h1>Extreme Events</h1>"
    body += _figure_card(fig, "极端事件时间线", "挑选 raw PLIE、同向放大 z_amp、低 response percentile 的极端事件，用于人工机制复盘。", {"extreme MAR events": "红色菱形为候选极端事件；hover 显示响应标签。"}, cfg.get('visualization','include_plotlyjs'))
    body += f"<div class='card'><h2>Selected events</h2>{table}</div>"
    _write(_html_path(html_dir, cfg, "extreme_events"), _page("Extreme Events", body, cfg))


def build_all_html(base_df: pd.DataFrame, event_df: pd.DataFrame, curve_df: pd.DataFrame, memory_df: pd.DataFrame, reports: dict, cfg: Config, path_df: pd.DataFrame | None = None) -> None:
    html_dir = cfg.path("paths", "html_dir")
    build_index(html_dir, reports["latest_summary"], cfg)
    build_quality_assessment(html_dir, memory_df, reports, cfg)
    build_market_response_dashboard(html_dir, base_df, event_df, cfg)
    build_calibration_page(html_dir, reports["q65_coverage"], reports["label_proportions"], reports["percentile_summary"], cfg)
    build_curve_dashboard(html_dir, curve_df, cfg)
    build_path_absorption_dashboard(html_dir, base_df, path_df if path_df is not None else pd.DataFrame(), memory_df, reports, cfg)
    build_scenario_examples(html_dir, reports, cfg)
    build_rolling_monitoring(html_dir, memory_df, cfg)
    build_feature_statistics(html_dir, memory_df, event_df, cfg)
    build_extreme_events(html_dir, base_df, event_df, cfg)
