from __future__ import annotations

import html
import shutil
from pathlib import Path

import plotly


PAGE_BG = "#0b0f14"
CARD_BG = "#12161c"
BORDER = "rgba(230,237,243,0.12)"
TEXT = "#e6edf3"
MUTED = "rgba(230,237,243,0.72)"
ACCENT = "#3d85c6"
GRID = "rgba(255,255,255,0.08)"
FONT = "Inter, IBM Plex Sans, Segoe UI, sans-serif"


def ensure_plotly_bundle(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    src = Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js"
    dst = output_dir / "plotly.min.js"
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dst)
    return dst


def html_template(
    *,
    title: str,
    body: str,
    plotly_src: str | None = None,
    extra_head: str = "",
    extra_scripts: str = "",
    page_class: str = "",
) -> str:
    script_tag = f"<script src='{html.escape(plotly_src)}'></script>" if plotly_src else ""
    return f"""
<!DOCTYPE html>
<html lang='zh-CN'>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>{html.escape(title)}</title>
  {script_tag}
  {extra_head}
  <style>
    :root {{ color-scheme: dark; }}
    body {{ margin: 0; background: {PAGE_BG}; color: {TEXT}; font-family: {FONT}; }}
    .page {{ width: min(2400px, calc(100vw - 24px)); max-width: none; margin: 0 auto; padding: 16px 12px 28px; box-sizing: border-box; }}
    .page.wide {{ width: calc(100vw - 12px); padding: 10px 6px 18px; }}
    .panel {{ background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 16px; padding: 16px 18px; margin-bottom: 18px; box-sizing: border-box; }}
    .chip {{ display: inline-block; border-radius: 999px; padding: 4px 10px; background: rgba(61,133,198,0.18); color: #9fc5f8; font-size: 12px; }}
    .small {{ color: {MUTED}; font-size: 13px; line-height: 1.65; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 18px; }}
    .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
    .stats-card {{ background: rgba(11,15,20,0.45); border: 1px solid {BORDER}; border-radius: 12px; padding: 12px 14px; }}
    .stats-card h3 {{ margin: 0 0 10px; font-size: 16px; }}
    .stats-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    .stats-table th, .stats-table td {{ padding: 6px 0; border-bottom: 1px solid rgba(230,237,243,0.08); text-align: left; vertical-align: top; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; }}
    .card {{ background: rgba(11,15,20,0.45); border: 1px solid {BORDER}; border-radius: 14px; padding: 14px; }}
    .card a, a.link {{ color: {TEXT}; text-decoration: none; font-weight: 700; }}
    .link-list a {{ display: inline-block; margin-right: 10px; margin-bottom: 10px; color: #9fc5f8; text-decoration: none; }}
    .topbar {{ display: flex; justify-content: space-between; align-items: center; gap: 16px; margin-bottom: 16px; flex-wrap: wrap; }}
    .back {{ color: {TEXT}; text-decoration: none; border: 1px solid {BORDER}; border-radius: 999px; padding: 8px 14px; background: {CARD_BG}; }}
    .figure-wrap {{ width: 100%; overflow-x: auto; overflow-y: visible; }}
    .plotly-graph-div {{ width: 100% !important; }}
    .toolbar {{ display:flex; align-items:center; gap:8px; flex-wrap: wrap; margin-bottom: 12px; }}
    .toolbar button, .toolbar a {{ cursor:pointer; border:1px solid {BORDER}; background:rgba(11,15,20,0.65); color:{TEXT}; border-radius:10px; padding:8px 12px; text-decoration:none; font-size:13px; }}
    .toolbar button:hover, .toolbar a:hover {{ border-color: rgba(159,197,248,0.65); }}
    .image-shell {{ width:100%; overflow:auto; background:#ffffff; border-radius:12px; border:1px solid rgba(230,237,243,0.08); min-height: 60vh; }}
    .image-shell img {{ display:block; width:100%; height:auto; max-width:none; }}
  </style>
</head>
<body>
  <div class='page {html.escape(page_class)}'>
    {body}
  </div>
  {extra_scripts}
</body>
</html>
"""
