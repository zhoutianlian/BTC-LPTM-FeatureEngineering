from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def write_static_assets(report_dir: Path) -> None:
    css_dir = report_dir / "assets" / "css"
    js_dir = report_dir / "assets" / "js"
    css_dir.mkdir(parents=True, exist_ok=True)
    js_dir.mkdir(parents=True, exist_ok=True)
    (css_dir / "report.css").write_text(REPORT_CSS, encoding="utf-8")
    (js_dir / "table.js").write_text(TABLE_JS, encoding="utf-8")
    (js_dir / "feature_charts.js").write_text(FEATURE_CHARTS_JS, encoding="utf-8")


def render_index_html(summary: dict[str, Any], rows: list[dict[str, Any]], heatmap_html: str, high_corr_rows: list[dict[str, Any]]) -> str:
    cards = [
        ("Generated", summary.get("generated_at")),
        ("Data Range", f"{summary.get('data_start') or 'n/a'} -> {summary.get('data_end') or 'n/a'}"),
        ("Rows", _fmt(summary.get("sample_count"))),
        ("Features", _fmt(summary.get("feature_total"))),
        ("Documented Exists", _fmt(summary.get("documented_existing_count"))),
        ("Documented Missing", _fmt(summary.get("documented_missing_count"))),
        ("FAIL", _fmt(summary.get("fail_count"))),
        ("WARN", _fmt(summary.get("warn_count"))),
    ]
    table = _summary_table(rows)
    corr_table = _high_corr_table(high_corr_rows)
    leak = _leakage_block(summary.get("leakage_checks", {}))
    relationship_note = html.escape(summary.get("relationship_diagnostics", {}).get("status_note", ""))
    body = f"""
    <header class="hero">
      <div>
        <p class="eyebrow">price_context</p>
        <h1>price_context Feature Diagnostics</h1>
      </div>
      <a class="ghost-link" href="summary.json">summary.json</a>
    </header>
    <section class="metric-grid">{''.join(_metric_card(k, v) for k, v in cards)}</section>
    <section class="panel">
      <div class="section-head">
        <h2>Feature Summary</h2>
        <input id="featureSearch" class="search" placeholder="Search features, category, status" />
      </div>
      {table}
    </section>
    <section class="panel">
      <div class="section-head"><h2>Correlation Diagnostics</h2></div>
      <p class="note">{relationship_note}</p>
      <div class="chart-wrap">{heatmap_html}</div>
      <h3>High Correlation Pairs</h3>
      {corr_table}
    </section>
    <section class="panel">
      <div class="section-head"><h2>Future Leakage Review</h2></div>
      {leak}
    </section>
    """
    return _page("price_context Feature Diagnostics", body, "assets/css/report.css", ["assets/js/plotly.min.js", "assets/js/table.js"])


def render_feature_html(result: dict[str, Any], plots: dict[str, str], relationship_note: str) -> str:
    status = result.get("status", "PASS")
    stats = result.get("stats", {})
    outliers = result.get("outliers", {})
    temporal = result.get("temporal", {})
    issues = result.get("issues", [])
    rows = [
        ("category", result.get("category")),
        ("dtype", result.get("dtype")),
        ("exists", result.get("exists")),
        ("valid_count", stats.get("count")),
        ("missing_ratio", _ratio(stats.get("missing_ratio"))),
        ("inf_count", stats.get("inf_count")),
        ("zscore_outliers", outliers.get("zscore_outlier_count")),
        ("iqr_outliers", outliers.get("iqr_outlier_count")),
        ("quantile_extremes", outliers.get("quantile_extreme_count")),
        ("cliff_jumps", outliers.get("cliff_jump_count")),
        ("max_constant_run", temporal.get("max_consecutive_constant")),
    ]
    body = f"""
    <header class="feature-header">
      <a class="ghost-link" href="../index.html">Back to overview</a>
      <div>
        <p class="eyebrow">{html.escape(str(result.get('category', '')))}</p>
        <h1>{html.escape(result.get('feature_name', 'feature'))}</h1>
      </div>
      {_status_badge(status)}
    </header>
    <section class="panel">
      <h2>Definition</h2>
      <p class="description">{html.escape(result.get('description') or 'No documentation description was found for this output column.')}</p>
      <div class="mini-grid">{''.join(_metric_card(k, _fmt(v)) for k, v in rows)}</div>
      {_issues_block(issues)}
    </section>
    <section class="panel">
      <h2>Statistics</h2>
      {_stats_table(stats)}
    </section>
    <section class="panel">
      <h2>Outlier Time Points</h2>
      {_outlier_table(outliers.get('outlier_points', []))}
    </section>
    {_plot_section('Time Series', plots.get('time_series'))}
    {_plot_section('Distribution', plots.get('distribution'))}
    {_plot_section('Box Plot', plots.get('box'))}
    {_plot_section('Rolling Statistics', plots.get('rolling'))}
    {_plot_section('Missing Timeline', plots.get('missing'))}
    <section class="panel">
      <h2>Relationship Diagnostics</h2>
      <p class="note">{html.escape(relationship_note)}</p>
      {plots.get('relationship') or ''}
    </section>
    """
    return _page(result.get("feature_name", "Feature Diagnostics"), body, "../assets/css/report.css", ["../assets/js/plotly.min.js"])


def render_feature_html_client(result: dict[str, Any], payload_json: str, relationship_note: str) -> str:
    status = result.get("status", "PASS")
    stats = result.get("stats", {})
    outliers = result.get("outliers", {})
    temporal = result.get("temporal", {})
    issues = result.get("issues", [])
    rows = [
        ("category", result.get("category")),
        ("dtype", result.get("dtype")),
        ("exists", result.get("exists")),
        ("valid_count", stats.get("count")),
        ("missing_ratio", _ratio(stats.get("missing_ratio"))),
        ("inf_count", stats.get("inf_count")),
        ("zscore_outliers", outliers.get("zscore_outlier_count")),
        ("iqr_outliers", outliers.get("iqr_outlier_count")),
        ("quantile_extremes", outliers.get("quantile_extreme_count")),
        ("cliff_jumps", outliers.get("cliff_jump_count")),
        ("max_constant_run", temporal.get("max_consecutive_constant")),
    ]
    chart_blocks = ""
    if result.get("exists") and result.get("is_numeric", False):
        chart_blocks = """
        <section class="panel"><h2>Time Series</h2><div id="chart-time-series" class="chart-root"></div></section>
        <section class="panel"><h2>Distribution</h2><div id="chart-distribution" class="chart-root"></div></section>
        <section class="panel"><h2>Box Plot</h2><div id="chart-box" class="chart-root compact"></div></section>
        <section class="panel"><h2>Rolling Statistics</h2><div id="chart-rolling" class="chart-root tall"></div></section>
        <section class="panel"><h2>Missing Timeline</h2><div id="chart-missing" class="chart-root"></div></section>
        <section class="panel"><h2>Relationship Diagnostics</h2><p class="note" id="relationship-note"></p><div id="chart-relationship" class="chart-root"></div></section>
        """
    else:
        chart_blocks = '<section class="panel"><h2>Charts</h2><p class="note">Charts are unavailable because this feature is missing or non-numeric.</p></section>'
    body = f"""
    <header class="feature-header">
      <a class="ghost-link" href="../index.html">Back to overview</a>
      <div>
        <p class="eyebrow">{html.escape(str(result.get('category', '')))}</p>
        <h1>{html.escape(result.get('feature_name', 'feature'))}</h1>
      </div>
      {_status_badge(status)}
    </header>
    <section class="panel">
      <h2>Definition</h2>
      <p class="description">{html.escape(result.get('description') or 'No documentation description was found for this output column.')}</p>
      <div class="mini-grid">{''.join(_metric_card(k, _fmt(v)) for k, v in rows)}</div>
      {_issues_block(issues)}
    </section>
    <section class="panel">
      <h2>Statistics</h2>
      {_stats_table(stats)}
    </section>
    <section class="panel">
      <h2>Outlier Time Points</h2>
      {_outlier_table(outliers.get('outlier_points', []))}
    </section>
    {chart_blocks}
    <script>window.FEATURE_DIAGNOSTIC_PAYLOAD = {payload_json}; window.FEATURE_RELATIONSHIP_NOTE = {json.dumps(relationship_note)};</script>
    """
    return _page(
        result.get("feature_name", "Feature Diagnostics"),
        body,
        "../assets/css/report.css",
        ["../assets/js/plotly.min.js", "../assets/js/feature_charts.js"],
    )


def _page(title: str, body: str, css_path: str, scripts: list[str]) -> str:
    script_tags = "\n".join(f'<script src="{html.escape(src)}"></script>' for src in scripts)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <link rel="stylesheet" href="{html.escape(css_path)}" />
</head>
<body>
  <main class="page">
    {body}
  </main>
  {script_tags}
</body>
</html>
"""


def _summary_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "feature_name",
        "category",
        "exists",
        "dtype",
        "valid_count",
        "missing_ratio",
        "inf_count",
        "mean",
        "std",
        "min",
        "max",
        "p01",
        "p50",
        "p99",
        "skew",
        "kurtosis",
        "outlier_count",
        "constant_flag",
        "status",
        "detail_link",
    ]
    body = []
    for row in rows:
        cells = []
        for key in headers:
            value = row.get(key)
            if key == "status":
                cells.append(f"<td>{_status_badge(str(value))}</td>")
            elif key == "detail_link" and value:
                cells.append(f'<td><a href="{html.escape(str(value))}">open</a></td>')
            elif key == "missing_ratio":
                cells.append(f"<td>{_ratio(value)}</td>")
            else:
                cells.append(f"<td>{html.escape(_fmt(value))}</td>")
        body.append(f"<tr data-status=\"{html.escape(str(row.get('status', '')))}\">{''.join(cells)}</tr>")
    head = "".join(f'<th data-sort="{html.escape(h)}">{html.escape(h)}</th>' for h in headers)
    return f'<div class="table-scroll"><table id="featureTable" class="data-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _stats_table(stats: dict[str, Any]) -> str:
    rows = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(_fmt(v))}</td></tr>" for k, v in stats.items())
    return f'<div class="table-scroll narrow"><table class="kv-table"><tbody>{rows}</tbody></table></div>'


def _outlier_table(points: list[dict[str, Any]]) -> str:
    if not points:
        return '<p class="note">No top outlier timestamps were selected.</p>'
    headers = ["time", "index", "value", "zscore"]
    head = "".join(f"<th>{h}</th>" for h in headers)
    rows = []
    for point in points:
        rows.append("<tr>" + "".join(f"<td>{html.escape(_fmt(point.get(h)))}</td>" for h in headers) + "</tr>")
    return f'<div class="table-scroll narrow"><table class="data-table"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def _high_corr_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<p class="note">No feature pairs exceeded the configured high-correlation threshold.</p>'
    head = "<th>feature_a</th><th>feature_b</th><th>correlation</th>"
    body = "".join(
        f"<tr><td>{html.escape(str(r.get('feature_a')))}</td><td>{html.escape(str(r.get('feature_b')))}</td><td>{_fmt(r.get('correlation'))}</td></tr>"
        for r in rows
    )
    return f'<div class="table-scroll narrow"><table class="data-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _issues_block(issues: list[str]) -> str:
    if not issues:
        return '<div class="issue-row"><span class="chip pass">No issues triggered by configured rules.</span></div>'
    return '<div class="issue-row">' + "".join(f'<span class="chip warn">{html.escape(issue)}</span>' for issue in issues) + "</div>"


def _leakage_block(checks: dict[str, Any]) -> str:
    if not checks:
        return '<p class="note">No leakage checks were recorded.</p>'
    rows = []
    for item in checks.get("items", []):
        rows.append(
            f"<tr><td>{html.escape(str(item.get('name')))}</td><td>{_status_badge(str(item.get('status')))}</td><td>{html.escape(str(item.get('detail')))}</td></tr>"
        )
    return f'<div class="table-scroll narrow"><table class="data-table"><thead><tr><th>check</th><th>status</th><th>detail</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def _plot_section(title: str, plot_html: str | None) -> str:
    if not plot_html:
        plot_html = '<p class="note">This plot is unavailable for the current feature.</p>'
    return f'<section class="panel"><h2>{html.escape(title)}</h2><div class="chart-wrap">{plot_html}</div></section>'


def _metric_card(label: str, value: Any) -> str:
    return f'<div class="metric-card"><span>{html.escape(str(label))}</span><strong>{html.escape(_fmt(value))}</strong></div>'


def _status_badge(status: str) -> str:
    cls = status.lower()
    return f'<span class="status {html.escape(cls)}">{html.escape(status)}</span>'


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value != value:
            return "n/a"
        if abs(value) >= 1000 or (0 < abs(value) < 0.001):
            return f"{value:.4e}"
        return f"{value:.6g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _ratio(value: Any) -> str:
    try:
        if value is None:
            return "n/a"
        return f"{float(value):.2%}"
    except Exception:
        return _fmt(value)


REPORT_CSS = """
:root {
  color-scheme: dark;
  --bg: #050B14;
  --panel: #08111F;
  --panel-2: #0D1828;
  --line: rgba(147, 190, 255, 0.18);
  --text: #E7F6FF;
  --muted: #8EA4B8;
  --cyan: #18E0FF;
  --green: #00FF88;
  --orange: #FF7A1A;
  --yellow: #F6C945;
  --red: #FF5A7A;
  --purple: #A46CFF;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
  letter-spacing: 0;
}
a { color: var(--cyan); text-decoration: none; }
a:hover { color: var(--green); }
.page {
  width: min(1680px, calc(100vw - 32px));
  margin: 0 auto;
  padding: 28px 0 60px;
}
.hero, .feature-header {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 20px;
  padding: 16px 0 22px;
  border-bottom: 1px solid var(--line);
}
.feature-header { align-items: center; }
.eyebrow {
  margin: 0 0 8px;
  color: var(--cyan);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}
h1, h2, h3 { margin: 0; line-height: 1.15; letter-spacing: 0; }
h1 { font-size: clamp(30px, 4vw, 52px); }
h2 { font-size: 20px; }
h3 { margin-top: 20px; font-size: 16px; color: var(--muted); }
.ghost-link {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 9px 12px;
  color: var(--text);
  background: rgba(24, 224, 255, 0.06);
}
.metric-grid, .mini-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px;
  margin: 18px 0;
}
.mini-grid { grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); }
.metric-card {
  min-height: 78px;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 13px 14px;
  background: linear-gradient(180deg, rgba(13, 24, 40, 0.96), rgba(8, 17, 31, 0.96));
}
.metric-card span {
  display: block;
  color: var(--muted);
  font-size: 12px;
  margin-bottom: 8px;
}
.metric-card strong {
  display: block;
  overflow-wrap: anywhere;
  font-size: 19px;
  color: var(--text);
}
.panel {
  margin-top: 18px;
  padding: 18px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
}
.section-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 14px;
}
.search {
  width: min(420px, 100%);
  height: 38px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #0C1726;
  color: var(--text);
  padding: 0 12px;
  outline: none;
}
.search:focus { border-color: var(--cyan); }
.table-scroll {
  width: 100%;
  overflow: auto;
  border: 1px solid rgba(147, 190, 255, 0.12);
  border-radius: 8px;
}
.table-scroll.narrow { max-width: 1100px; }
table {
  width: 100%;
  border-collapse: collapse;
  background: #07111F;
}
th, td {
  padding: 9px 10px;
  border-bottom: 1px solid rgba(147, 190, 255, 0.10);
  color: var(--text);
  font-size: 13px;
  white-space: nowrap;
}
th {
  position: sticky;
  top: 0;
  z-index: 1;
  background: #101C2C;
  color: #BFD5E8;
  text-align: left;
  cursor: pointer;
}
tbody tr:hover { background: rgba(24, 224, 255, 0.06); }
.kv-table th { width: 220px; color: var(--muted); }
.status {
  display: inline-flex;
  min-width: 54px;
  justify-content: center;
  border-radius: 999px;
  padding: 4px 9px;
  font-size: 12px;
  font-weight: 800;
}
.status.pass { color: #001B10; background: var(--green); }
.status.warn { color: #1F1400; background: var(--yellow); }
.status.fail { color: #22000A; background: var(--red); }
.status.review, .status.skip { color: var(--text); background: rgba(164, 108, 255, 0.35); }
.issue-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
.chip {
  display: inline-flex;
  border-radius: 999px;
  border: 1px solid var(--line);
  padding: 6px 10px;
  font-size: 12px;
}
.chip.pass { border-color: rgba(0, 255, 136, 0.40); color: var(--green); }
.chip.warn { border-color: rgba(246, 201, 69, 0.45); color: var(--yellow); }
.description, .note {
  color: var(--muted);
  line-height: 1.58;
  margin: 10px 0 0;
}
.chart-wrap {
  width: 100%;
  min-height: 420px;
}
.chart-root {
  width: 100%;
  min-height: 430px;
}
.chart-root.tall { min-height: 640px; }
.chart-root.compact { min-height: 360px; }
.chart-wrap .plotly-graph-div { width: 100% !important; }
.chart-root .plotly-graph-div { width: 100% !important; }
@media (max-width: 760px) {
  .hero, .feature-header, .section-head { align-items: flex-start; flex-direction: column; }
  .page { width: min(100vw - 20px, 1680px); padding-top: 16px; }
  .panel { padding: 12px; }
  th, td { font-size: 12px; padding: 8px; }
}
"""


TABLE_JS = """
(function () {
  function cellValue(row, idx) {
    return row.children[idx].innerText.trim();
  }
  function asNumber(value) {
    var cleaned = value.replace(/%$/, "");
    var n = Number(cleaned);
    return Number.isFinite(n) ? n : null;
  }
  document.addEventListener("DOMContentLoaded", function () {
    var table = document.getElementById("featureTable");
    var search = document.getElementById("featureSearch");
    if (!table) return;
    Array.prototype.forEach.call(table.querySelectorAll("th"), function (th, idx) {
      th.addEventListener("click", function () {
        var tbody = table.tBodies[0];
        var rows = Array.prototype.slice.call(tbody.rows);
        var dir = th.dataset.dir === "asc" ? "desc" : "asc";
        th.dataset.dir = dir;
        rows.sort(function (a, b) {
          var av = cellValue(a, idx);
          var bv = cellValue(b, idx);
          var an = asNumber(av);
          var bn = asNumber(bv);
          var cmp = an !== null && bn !== null ? an - bn : av.localeCompare(bv);
          return dir === "asc" ? cmp : -cmp;
        });
        rows.forEach(function (row) { tbody.appendChild(row); });
      });
    });
    if (search) {
      search.addEventListener("input", function () {
        var q = search.value.toLowerCase();
        Array.prototype.forEach.call(table.tBodies[0].rows, function (row) {
          row.style.display = row.innerText.toLowerCase().indexOf(q) >= 0 ? "" : "none";
        });
      });
    }
  });
})();
"""


FEATURE_CHARTS_JS = """
(function () {
  var payload = window.FEATURE_DIAGNOSTIC_PAYLOAD;
  if (!payload || !window.Plotly) return;

  var COLORS = {
    cyan: "#18E0FF",
    green: "#00FF88",
    orange: "#FF7A1A",
    yellow: "#F6C945",
    red: "#FF5A7A",
    purple: "#A46CFF",
    muted: "#7A8CA5",
    paper: "#07111F",
    plot: "#08111F",
    text: "#E7F6FF"
  };
  var CONFIG = { displaylogo: false, responsive: true, scrollZoom: true, modeBarButtonsToRemove: ["lasso2d", "select2d"] };

  function finite(v) {
    return typeof v === "number" && Number.isFinite(v);
  }
  function cleanValues(values) {
    return values.map(function (v) { return finite(v) ? v : null; });
  }
  function finiteValues(values) {
    return values.filter(finite);
  }
  function layout(title, extra) {
    var base = {
      title: { text: title, font: { color: COLORS.text, size: 17 } },
      template: "plotly_dark",
      paper_bgcolor: COLORS.paper,
      plot_bgcolor: COLORS.plot,
      font: { color: "#D8E7F5", family: "Inter, Segoe UI, Arial, sans-serif" },
      margin: { l: 58, r: 32, t: 76, b: 52 },
      hovermode: "closest",
      legend: {
        bgcolor: "rgba(7,17,31,0.76)",
        bordercolor: "rgba(130,180,255,0.22)",
        borderwidth: 1,
        orientation: "h",
        yanchor: "bottom",
        y: 1.02,
        xanchor: "right",
        x: 1
      },
      xaxis: axisStyle(),
      yaxis: axisStyle()
    };
    return Object.assign(base, extra || {});
  }
  function axisStyle() {
    return {
      gridcolor: "rgba(170, 210, 255, 0.10)",
      zerolinecolor: "rgba(170, 210, 255, 0.16)",
      showline: true,
      linecolor: "rgba(170, 210, 255, 0.18)",
      rangemode: "normal"
    };
  }
  function timeTraceProps() {
    if (payload.time_axis && payload.time_axis.mode === "linear") {
      return { x0: payload.time_axis.x0, dx: payload.time_axis.dx };
    }
    return { x: payload.time_axis ? payload.time_axis.values : [] };
  }
  function timeForIndex(i) {
    if (payload.time_axis && payload.time_axis.mode === "linear") {
      return new Date(Date.parse(payload.time_axis.x0) + payload.time_axis.dx * i).toISOString();
    }
    return payload.time_axis.values[i];
  }
  function timeForIndices(indices) {
    return indices.map(timeForIndex);
  }
  function quantile(sorted, q) {
    if (!sorted.length) return null;
    var pos = (sorted.length - 1) * q;
    var lo = Math.floor(pos);
    var hi = Math.ceil(pos);
    if (lo === hi) return sorted[lo];
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  }
  function sortedFinite(values) {
    return finiteValues(values).sort(function (a, b) { return a - b; });
  }
  function mean(values) {
    var vals = finiteValues(values);
    if (!vals.length) return null;
    return vals.reduce(function (a, b) { return a + b; }, 0) / vals.length;
  }
  function std(values) {
    var vals = finiteValues(values);
    if (vals.length < 2) return 0;
    var m = mean(vals);
    var s = vals.reduce(function (a, b) { return a + Math.pow(b - m, 2); }, 0);
    return Math.sqrt(s / (vals.length - 1));
  }
  function minValue(values) {
    var out = null;
    for (var i = 0; i < values.length; i++) {
      if (finite(values[i]) && (out === null || values[i] < out)) out = values[i];
    }
    return out;
  }
  function maxValue(values) {
    var out = null;
    for (var i = 0; i < values.length; i++) {
      if (finite(values[i]) && (out === null || values[i] > out)) out = values[i];
    }
    return out;
  }
  function bisectLeft(arr, x) {
    var lo = 0, hi = arr.length;
    while (lo < hi) {
      var mid = (lo + hi) >> 1;
      if (arr[mid] < x) lo = mid + 1; else hi = mid;
    }
    return lo;
  }
  function addSorted(arr, x) {
    arr.splice(bisectLeft(arr, x), 0, x);
  }
  function removeSorted(arr, x) {
    var idx = bisectLeft(arr, x);
    if (idx < arr.length) arr.splice(idx, 1);
  }
  function rollingStats(values, window) {
    var n = values.length;
    var sorted = [];
    var sum = 0;
    var sumSq = 0;
    var count = 0;
    var minPeriods = Math.min(window, 10);
    var out = { mean: new Array(n), std: new Array(n), q05: new Array(n), q95: new Array(n), miss: new Array(n) };
    var missWindow = [];
    var missSum = 0;
    for (var i = 0; i < n; i++) {
      var v = values[i];
      var miss = finite(v) ? 0 : 1;
      missWindow.push(miss);
      missSum += miss;
      if (finite(v)) {
        addSorted(sorted, v);
        sum += v;
        sumSq += v * v;
        count += 1;
      }
      if (i >= window) {
        var old = values[i - window];
        var oldMiss = missWindow.shift();
        missSum -= oldMiss;
        if (finite(old)) {
          removeSorted(sorted, old);
          sum -= old;
          sumSq -= old * old;
          count -= 1;
        }
      }
      if (count >= minPeriods) {
        var m = sum / count;
        var variance = Math.max(0, (sumSq - sum * sum / count) / Math.max(1, count - 1));
        out.mean[i] = m;
        out.std[i] = Math.sqrt(variance);
        out.q05[i] = quantile(sorted, 0.05);
        out.q95[i] = quantile(sorted, 0.95);
      } else {
        out.mean[i] = null; out.std[i] = null; out.q05[i] = null; out.q95[i] = null;
      }
      out.miss[i] = missSum / Math.min(window, i + 1);
    }
    return out;
  }
  function outlierIndices(values) {
    var vals = finiteValues(values);
    if (vals.length < 3) return [];
    var m = mean(vals);
    var s = std(vals);
    if (!s) return [];
    var idx = [];
    for (var i = 0; i < values.length; i++) {
      if (finite(values[i]) && Math.abs((values[i] - m) / s) > payload.zscore_threshold) idx.push(i);
    }
    idx.sort(function (a, b) { return Math.abs(values[b] - m) - Math.abs(values[a] - m); });
    return idx.slice(0, payload.max_plot_outlier_points || 500);
  }
  function missingIndices(values) {
    var idx = [];
    for (var i = 0; i < values.length; i++) if (!finite(values[i])) idx.push(i);
    var max = payload.max_missing_markers || 500;
    if (idx.length <= max) return idx;
    var sampled = [];
    for (var j = 0; j < max; j++) sampled.push(idx[Math.floor(j * (idx.length - 1) / Math.max(1, max - 1))]);
    return sampled;
  }
  function histogram(values, bins) {
    var vals = finiteValues(values);
    if (!vals.length) return { x: [], y: [], smooth: [] };
    var min = minValue(vals);
    var max = maxValue(vals);
    if (min === max) return { x: [min], y: [1], smooth: [1] };
    var width = (max - min) / bins;
    var counts = new Array(bins).fill(0);
    vals.forEach(function (v) {
      var idx = Math.min(bins - 1, Math.max(0, Math.floor((v - min) / width)));
      counts[idx] += 1;
    });
    var density = counts.map(function (c) { return c / vals.length / width; });
    var centers = counts.map(function (_, i) { return min + width * (i + 0.5); });
    var kernel = [1, 2, 3, 4, 3, 2, 1];
    var ksum = kernel.reduce(function (a, b) { return a + b; }, 0);
    var smooth = density.map(function (_, i) {
      var total = 0;
      for (var k = 0; k < kernel.length; k++) {
        var j = i + k - 3;
        if (j >= 0 && j < density.length) total += density[j] * kernel[k];
      }
      return total / ksum;
    });
    return { x: centers, y: density, smooth: smooth };
  }
  function renderTimeSeries(values) {
    var traces = [Object.assign({
      y: values,
      type: "scatter",
      mode: "lines",
      name: payload.feature_name,
      line: { color: COLORS.cyan, width: 1.2 },
      connectgaps: false,
      hovertemplate: "time=%{x}<br>value=%{y:.6g}<extra></extra>"
    }, timeTraceProps())];
    var outIdx = outlierIndices(values);
    if (outIdx.length) {
      traces.push({
        x: timeForIndices(outIdx),
        y: outIdx.map(function (i) { return values[i]; }),
        type: "scatter",
        mode: "markers",
        name: "marked outliers",
        marker: { color: COLORS.red, size: 6, symbol: "x" }
      });
    }
    var missIdx = missingIndices(values);
    if (missIdx.length) {
      var vals = finiteValues(values);
      var baseline = vals.length ? minValue(vals) : 0;
      traces.push({
        x: timeForIndices(missIdx),
        y: missIdx.map(function () { return baseline; }),
        type: "scatter",
        mode: "markers",
        name: "missing samples",
        marker: { color: COLORS.yellow, size: 4, symbol: "line-ns-open" }
      });
    }
    Plotly.newPlot("chart-time-series", traces, layout(payload.feature_name + " time series", {
      xaxis: Object.assign(axisStyle(), {
        rangeslider: { visible: true, bgcolor: COLORS.plot, bordercolor: "#1B2A3D" },
        rangeselector: {
          bgcolor: "#101C2C",
          activecolor: COLORS.cyan,
          font: { color: COLORS.text },
          buttons: [
            { count: 1, label: "1D", step: "day", stepmode: "backward" },
            { count: 7, label: "1W", step: "day", stepmode: "backward" },
            { count: 1, label: "1M", step: "month", stepmode: "backward" },
            { count: 3, label: "3M", step: "month", stepmode: "backward" },
            { count: 6, label: "6M", step: "month", stepmode: "backward" },
            { count: 1, label: "1Y", step: "year", stepmode: "backward" },
            { label: "ALL", step: "all" }
          ]
        }
      })
    }), CONFIG);
  }
  function renderDistribution(values) {
    var hist = histogram(values, 90);
    var vals = sortedFinite(values);
    var traces = [
      { x: hist.x, y: hist.y, type: "bar", name: "histogram", marker: { color: "rgba(24,224,255,0.62)", line: { color: COLORS.cyan, width: 0.5 } } },
      { x: hist.x, y: hist.smooth, type: "scatter", mode: "lines", name: "smoothed density", line: { color: COLORS.purple, width: 2 } }
    ];
    var shapes = [];
    var annotations = [];
    [["mean", mean(vals), COLORS.green], ["median", quantile(vals, 0.5), COLORS.yellow], ["p01", quantile(vals, 0.01), COLORS.orange], ["p99", quantile(vals, 0.99), COLORS.orange]].forEach(function (item) {
      if (!finite(item[1])) return;
      shapes.push({ type: "line", x0: item[1], x1: item[1], y0: 0, y1: 1, xref: "x", yref: "paper", line: { color: item[2], dash: "dash", width: 1.4 } });
      annotations.push({ x: item[1], y: 1.02, xref: "x", yref: "paper", text: item[0], showarrow: false, font: { color: item[2], size: 11 } });
    });
    Plotly.newPlot("chart-distribution", traces, layout(payload.feature_name + " distribution", { shapes: shapes, annotations: annotations }), CONFIG);
  }
  function renderBox(values) {
    var vals = sortedFinite(values);
    if (!vals.length) {
      Plotly.newPlot("chart-box", [], layout(payload.feature_name + " box plot"), CONFIG);
      return;
    }
    var trace = {
      type: "box",
      name: payload.feature_name,
      q1: [quantile(vals, 0.25)],
      median: [quantile(vals, 0.5)],
      q3: [quantile(vals, 0.75)],
      lowerfence: [vals[0]],
      upperfence: [vals[vals.length - 1]],
      boxpoints: false,
      fillcolor: "rgba(24,224,255,0.32)",
      line: { color: COLORS.cyan }
    };
    Plotly.newPlot("chart-box", [trace], layout(payload.feature_name + " box plot"), CONFIG);
  }
  function renderRolling(values, stats) {
    var traces = [
      Object.assign({ y: values, type: "scatter", mode: "lines", name: "raw", line: { color: "rgba(24,224,255,0.50)", width: 1 } }, timeTraceProps()),
      Object.assign({ y: stats.mean, type: "scatter", mode: "lines", name: "rolling mean", line: { color: COLORS.green, width: 1.6 } }, timeTraceProps()),
      Object.assign({ y: stats.q05, type: "scatter", mode: "lines", name: "rolling q05", line: { color: COLORS.muted, width: 1, dash: "dot" } }, timeTraceProps()),
      Object.assign({ y: stats.q95, type: "scatter", mode: "lines", name: "rolling q95", line: { color: COLORS.muted, width: 1, dash: "dot" } }, timeTraceProps()),
      Object.assign({ y: stats.std, type: "scatter", mode: "lines", name: "rolling std", xaxis: "x2", yaxis: "y2", line: { color: COLORS.orange, width: 1.4 } }, timeTraceProps())
    ];
    Plotly.newPlot("chart-rolling", traces, layout(payload.feature_name + " rolling diagnostics", {
      grid: { rows: 2, columns: 1, pattern: "independent", roworder: "top to bottom" },
      xaxis2: Object.assign(axisStyle(), { rangeslider: { visible: true, bgcolor: COLORS.plot, bordercolor: "#1B2A3D" } }),
      yaxis2: axisStyle(),
      height: 640
    }), CONFIG);
  }
  function renderMissing(values, stats) {
    var missing = values.map(function (v) { return finite(v) ? 0 : 1; });
    var traces = [
      Object.assign({ y: missing, type: "scatter", mode: "markers", name: "missing indicator", marker: { color: COLORS.yellow, size: 3 } }, timeTraceProps()),
      Object.assign({ y: stats.miss, type: "scatter", mode: "lines", name: "rolling missing ratio", line: { color: COLORS.orange, width: 1.5 } }, timeTraceProps())
    ];
    Plotly.newPlot("chart-missing", traces, layout(payload.feature_name + " missing value timeline", { yaxis: Object.assign(axisStyle(), { range: [-0.05, 1.05] }) }), CONFIG);
  }
  function renderRelationship() {
    var rel = payload.relationship || {};
    var note = document.getElementById("relationship-note");
    if (note) note.innerText = window.FEATURE_RELATIONSHIP_NOTE || "";
    if (!rel.available || !rel.panels || !rel.panels.length) {
      var el = document.getElementById("chart-relationship");
      if (el) el.innerHTML = '<p class="note">Relationship plots are unavailable for this feature.</p>';
      return;
    }
    var traces = [];
    var layoutExtra = { grid: { rows: 1, columns: rel.panels.length, pattern: "independent" }, height: 470 };
    rel.panels.forEach(function (panel, idx) {
      var axisSuffix = idx === 0 ? "" : String(idx + 1);
      traces.push({
        x: panel.x,
        y: panel.y,
        type: "scatter",
        mode: "markers",
        name: panel.name,
        xaxis: "x" + axisSuffix,
        yaxis: "y" + axisSuffix,
        marker: { color: panel.color, size: 4, opacity: 0.38 },
        hovertemplate: payload.feature_name + "=%{x:.6g}<br>" + panel.y_name + "=%{y:.6g}<extra></extra>"
      });
      layoutExtra["xaxis" + axisSuffix] = Object.assign(axisStyle(), { title: payload.feature_name });
      layoutExtra["yaxis" + axisSuffix] = Object.assign(axisStyle(), { title: panel.y_name });
    });
    Plotly.newPlot("chart-relationship", traces, layout(payload.feature_name + " relationship diagnostics", layoutExtra), CONFIG);
  }

  document.addEventListener("DOMContentLoaded", function () {
    var values = cleanValues(payload.values || []);
    renderTimeSeries(values);
    renderDistribution(values);
    renderBox(values);
    var stats = rollingStats(values, payload.rolling_window || 144);
    renderRolling(values, stats);
    renderMissing(values, stats);
    renderRelationship();
  });
})();
"""
