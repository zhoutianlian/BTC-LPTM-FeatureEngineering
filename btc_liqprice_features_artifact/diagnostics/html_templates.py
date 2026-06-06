from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, Iterable, List


THEME_CSS = """
:root {
  --bg: #050816;
  --panel: #0b1020;
  --panel-2: #0f172a;
  --line: #20304a;
  --grid: #24324a;
  --text: #dce7ff;
  --subtext: #8da2c0;
  --cyan: #00d1ff;
  --green: #39ff88;
  --orange: #ffb000;
  --purple: #b86bff;
  --pink: #ff4d8d;
  --red: #ff3b5c;
}
* { box-sizing: border-box; }
html, body { margin: 0; background: var(--bg); color: var(--text); font-family: Inter, Arial, Helvetica, sans-serif; }
body { min-width: 320px; }
a { color: var(--cyan); text-decoration: none; }
a:hover { color: var(--green); }
.page { max-width: 1840px; margin: 0 auto; padding: 24px; }
.topbar { display: flex; justify-content: space-between; align-items: flex-start; gap: 20px; margin-bottom: 18px; }
.title { font-size: 30px; font-weight: 760; line-height: 1.2; letter-spacing: 0; }
.subtitle { color: var(--subtext); margin-top: 8px; line-height: 1.6; max-width: 1100px; }
.nav-link { color: #8bd9ff; font-size: 14px; white-space: nowrap; }
.grid { display: grid; gap: 12px; }
.metric-grid { grid-template-columns: repeat(6, minmax(0, 1fr)); margin: 18px 0; }
.metric-card, .panel {
  background: linear-gradient(180deg, #0e1629 0%, #09101f 100%);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: 0 14px 34px rgba(0,0,0,0.28);
}
.metric-card { padding: 14px 16px; min-height: 84px; }
.metric-label { color: var(--subtext); font-size: 12px; margin-bottom: 8px; }
.metric-value { color: var(--text); font-size: 20px; font-weight: 740; overflow-wrap: anywhere; }
.panel { padding: 16px; margin: 16px 0; }
.panel-title { font-size: 18px; font-weight: 720; margin-bottom: 10px; }
.note { color: var(--subtext); line-height: 1.65; }
.status { display: inline-flex; align-items: center; min-width: 58px; justify-content: center; border-radius: 6px; padding: 4px 8px; font-size: 12px; font-weight: 760; border: 1px solid transparent; }
.status-pass { color: #06110b; background: var(--green); border-color: rgba(57,255,136,0.75); }
.status-warn { color: #161002; background: var(--orange); border-color: rgba(255,176,0,0.75); }
.status-fail { color: #ffffff; background: var(--red); border-color: rgba(255,59,92,0.75); }
.issue-list { display: grid; gap: 8px; margin: 0; padding: 0; list-style: none; }
.issue { border: 1px solid var(--line); background: rgba(15,23,42,0.78); border-radius: 8px; padding: 10px 12px; color: var(--subtext); line-height: 1.45; }
.issue strong { color: var(--text); margin-right: 8px; }
.table-toolbar { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 10px; }
.search { width: min(420px, 100%); background: #070d1a; color: var(--text); border: 1px solid var(--line); border-radius: 6px; padding: 10px 12px; outline: none; }
.table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; }
table { width: 100%; border-collapse: collapse; min-width: 1120px; }
th, td { padding: 10px 9px; border-bottom: 1px solid #18253a; white-space: nowrap; text-align: right; font-size: 13px; }
th:first-child, td:first-child { text-align: left; }
th { color: #bfd2f4; background: #0a1324; position: sticky; top: 0; cursor: pointer; user-select: none; z-index: 1; }
td { color: var(--text); }
tr:hover td { background: rgba(0,209,255,0.045); }
.row-fail td { background: rgba(255,59,92,0.075); }
.row-warn td { background: rgba(255,176,0,0.055); }
.plot { margin: 16px 0; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; background: var(--panel); }
.desc-block { white-space: pre-wrap; color: var(--subtext); line-height: 1.65; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
@media (max-width: 1280px) {
  .metric-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .two-col { grid-template-columns: 1fr; }
}
@media (max-width: 720px) {
  .page { padding: 16px; }
  .topbar { display: block; }
  .title { font-size: 24px; }
  .metric-grid { grid-template-columns: 1fr; }
  .table-toolbar { display: block; }
  .search { margin-top: 8px; }
}
"""


TABLE_JS = """
(function () {
  function parseValue(text) {
    var cleaned = text.trim().replace(/,/g, '');
    if (cleaned === 'true') return 1;
    if (cleaned === 'false') return 0;
    var num = Number(cleaned);
    return Number.isFinite(num) ? num : text.toLowerCase();
  }
  function sortTable(table, column, asc) {
    var tbody = table.tBodies[0];
    var rows = Array.prototype.slice.call(tbody.rows);
    rows.sort(function (a, b) {
      var av = parseValue(a.cells[column].innerText);
      var bv = parseValue(b.cells[column].innerText);
      if (av < bv) return asc ? -1 : 1;
      if (av > bv) return asc ? 1 : -1;
      return 0;
    });
    rows.forEach(function (row) { tbody.appendChild(row); });
  }
  document.querySelectorAll('table[data-sortable=\"true\"]').forEach(function (table) {
    table.querySelectorAll('th').forEach(function (th, idx) {
      th.addEventListener('click', function () {
        var asc = th.getAttribute('data-asc') !== 'true';
        table.querySelectorAll('th').forEach(function (h) { h.removeAttribute('data-asc'); });
        th.setAttribute('data-asc', asc ? 'true' : 'false');
        sortTable(table, idx, asc);
      });
    });
  });
  document.querySelectorAll('[data-table-search]').forEach(function (input) {
    var selector = input.getAttribute('data-table-search');
    var table = document.querySelector(selector);
    if (!table) return;
    input.addEventListener('input', function () {
      var needle = input.value.toLowerCase();
      Array.prototype.forEach.call(table.tBodies[0].rows, function (row) {
        row.style.display = row.innerText.toLowerCase().indexOf(needle) >= 0 ? '' : 'none';
      });
    });
  });
})();
"""


def write_assets(asset_dir: Path, plotly_js: str) -> Dict[str, Path]:
    css_dir = asset_dir / 'css'
    js_dir = asset_dir / 'js'
    css_dir.mkdir(parents=True, exist_ok=True)
    js_dir.mkdir(parents=True, exist_ok=True)
    css_path = css_dir / 'theme.css'
    table_js_path = js_dir / 'table.js'
    plotly_path = js_dir / 'plotly.min.js'
    css_path.write_text(THEME_CSS, encoding='utf-8')
    table_js_path.write_text(TABLE_JS, encoding='utf-8')
    if not plotly_path.exists():
        plotly_path.write_text(plotly_js, encoding='utf-8')
    return {'css': css_path, 'table_js': table_js_path, 'plotly_js': plotly_path}


def status_badge(status: str) -> str:
    cls = {'PASS': 'status-pass', 'WARN': 'status-warn', 'FAIL': 'status-fail'}.get(status, 'status-warn')
    return f'<span class="status {cls}">{html.escape(status)}</span>'


def metric_cards(items: Iterable[tuple[str, Any]]) -> str:
    cards = []
    for label, value in items:
        display = '' if value is None else str(value)
        cards.append(
            '<div class="metric-card">'
            f'<div class="metric-label">{html.escape(label)}</div>'
            f'<div class="metric-value">{html.escape(display)}</div>'
            '</div>'
        )
    return ''.join(cards)


def issues_html(issues: List[Dict[str, str]]) -> str:
    if not issues:
        return '<div class="note">No FAIL or WARN issues were detected by automated checks.</div>'
    rows = []
    for issue in issues:
        rows.append(
            '<li class="issue">'
            f'<strong>{html.escape(issue.get("severity", ""))}</strong>'
            f'{html.escape(issue.get("code", ""))}: {html.escape(issue.get("message", ""))}'
            '</li>'
        )
    return f'<ul class="issue-list">{"".join(rows)}</ul>'


def simple_table(headers: List[str], rows: List[List[Any]], *, table_id: str | None = None, sortable: bool = False) -> str:
    attrs = []
    if table_id:
        attrs.append(f'id="{html.escape(table_id)}"')
    if sortable:
        attrs.append('data-sortable="true"')
    head = ''.join(f'<th>{html.escape(str(h))}</th>' for h in headers)
    body_rows = []
    for row in rows:
        body_rows.append('<tr>' + ''.join(f'<td>{html.escape(format_value(v))}</td>' for v in row) + '</tr>')
    return f'<div class="table-wrap"><table {" ".join(attrs)}><thead><tr>{head}</tr></thead><tbody>{"".join(body_rows)}</tbody></table></div>'


def format_value(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, float):
        if value != value:
            return ''
        if abs(value) >= 1e5 or (0 < abs(value) < 1e-4):
            return f'{value:.6e}'
        return f'{value:.6f}'
    return str(value)

