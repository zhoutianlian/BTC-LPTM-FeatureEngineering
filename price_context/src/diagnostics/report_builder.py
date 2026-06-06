from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import resolve_path
from ..data_loader import load_ohlc_csv
from ..utils import window_to_bars
from .feature_checks import compact_summary_row, evaluate_feature
from .feature_plots import (
    figure_to_html,
    make_correlation_heatmap,
    write_plotly_asset,
)
from .feature_stats import infer_time_profile, rolling_window_from_config, to_numeric_series
from .html_templates import render_feature_html_client, render_index_html, write_static_assets
from .metadata import (
    NON_FEATURE_COLUMNS,
    build_feature_catalog,
    extract_feature_definitions_from_markdown,
    safe_feature_filename,
)


LOGGER = logging.getLogger(__name__)


def generate_feature_diagnostics_report(
    features: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    source_data: pd.DataFrame | None = None,
    output_path: Path | None = None,
    validation_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if features.empty:
        raise ValueError("Feature diagnostics cannot run on an empty feature DataFrame.")
    if "time" not in features.columns:
        raise ValueError("Feature diagnostics require a 'time' column in the feature output.")

    report_cfg = _report_config(cfg)
    report_dir = resolve_path(report_cfg["output_dir"], cfg.get("_project_root", "."))
    features_dir = report_dir / "features"
    assets_dir = report_dir / "assets"
    features_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    for stale_page in features_dir.glob("*.html"):
        stale_page.unlink()
    write_static_assets(report_dir)
    if report_cfg.get("generate_html", True):
        write_plotly_asset(report_dir)

    features = features.copy()
    features["time"] = pd.to_datetime(features["time"], errors="coerce")
    documented = extract_feature_definitions_from_markdown(_documentation_path(cfg, report_cfg))
    catalog = build_feature_catalog(
        features,
        documented,
        include_actual_output_features=bool(report_cfg.get("include_actual_output_features", True)),
    )
    time_profile = infer_time_profile(features, cfg)
    rolling_window = rolling_window_from_config(cfg, len(features))
    reference, relationship_note = _build_reference_frame(features, cfg, source_data)

    results: list[dict[str, Any]] = []
    for feature in catalog:
        results.append(evaluate_feature(features, feature, time_profile, cfg))

    relationship_corrs = _compute_relationship_correlations(features, results, reference)
    corr_matrix, high_corr_pairs = _compute_feature_correlation(features, results, cfg)
    leakage_checks = _compute_leakage_checks(features, cfg)

    summary_rows = [compact_summary_row(result) for result in results]
    generated_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    summary = {
        "project": "price_context Feature Diagnostics",
        "generated_at": generated_at,
        "feature_output_path": str(output_path) if output_path is not None else None,
        "report_dir": str(report_dir),
        "index_html": str(report_dir / "index.html"),
        "summary_json": str(report_dir / "summary.json"),
        "data_start": time_profile.get("start"),
        "data_end": time_profile.get("end"),
        "sample_count": time_profile.get("sample_count"),
        "feature_total": len(results),
        "actual_output_feature_count": len([c for c in features.columns if c not in NON_FEATURE_COLUMNS]),
        "documented_feature_count": len(documented),
        "documented_existing_count": sum(1 for r in results if r["documented"] and r["exists"]),
        "documented_missing_count": sum(1 for r in results if r["documented"] and not r["exists"]),
        "existing_feature_count": sum(1 for r in results if r["exists"]),
        "fail_count": sum(1 for r in results if r["status"] == "FAIL"),
        "warn_count": sum(1 for r in results if r["status"] == "WARN"),
        "pass_count": sum(1 for r in results if r["status"] == "PASS"),
        "time_profile": time_profile,
        "validation_report": validation_report or {},
        "rolling_window_bars": rolling_window,
        "relationship_diagnostics": {
            "available": reference is not None,
            "status_note": relationship_note,
            "columns": list(reference.columns) if reference is not None else [],
        },
        "leakage_checks": leakage_checks,
        "high_correlation_pairs": high_corr_pairs,
        "summary_rows": summary_rows,
        "features": results,
    }

    if report_cfg.get("generate_summary_json", True):
        (report_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if report_cfg.get("generate_html", True):
        for result in results:
            _write_feature_page(features, result, reference, relationship_corrs.get(result["feature_name"], {}), relationship_note, cfg, features_dir)
        heatmap_html = figure_to_html(make_correlation_heatmap(corr_matrix))
        index_html = render_index_html(summary, summary_rows, heatmap_html, high_corr_pairs)
        (report_dir / "index.html").write_text(index_html, encoding="utf-8")
    return summary


def _write_feature_page(
    features: pd.DataFrame,
    result: dict[str, Any],
    reference: pd.DataFrame | None,
    correlations: dict[str, float | None],
    relationship_note: str,
    cfg: dict[str, Any],
    features_dir: Path,
) -> None:
    feature_name = result["feature_name"]
    payload = _build_feature_payload(features, result, reference, correlations, cfg)
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    page = render_feature_html_client(result, payload_json, relationship_note)
    (features_dir / f"{safe_feature_filename(feature_name)}.html").write_text(page, encoding="utf-8")


def _build_feature_payload(
    features: pd.DataFrame,
    result: dict[str, Any],
    reference: pd.DataFrame | None,
    correlations: dict[str, float | None],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    name = result["feature_name"]
    if result["exists"] and name in features.columns and result.get("is_numeric", False):
        values = [_json_float(v) for v in to_numeric_series(features[name]).to_numpy()]
    else:
        values = []
    return {
        "feature_name": name,
        "values": values,
        "time_axis": _time_axis_payload(features["time"]),
        "rolling_window": int(result.get("temporal", {}).get("rolling_window") or rolling_window_from_config(cfg, len(features))),
        "zscore_threshold": float(cfg.get("report", {}).get("zscore_threshold", 5.0)),
        "max_plot_outlier_points": int(cfg.get("report", {}).get("max_plot_outlier_points", 500)),
        "max_missing_markers": int(cfg.get("report", {}).get("max_missing_markers", 500)),
        "relationship": _relationship_payload(features, name, reference, correlations, cfg) if values else {"available": False, "panels": []},
    }


def _time_axis_payload(time: pd.Series) -> dict[str, Any]:
    values = pd.to_datetime(time, errors="coerce")
    if len(values) >= 3 and not values.isna().any():
        diffs = values.diff().dropna()
        if not diffs.empty and (diffs == diffs.iloc[0]).all():
            dx_ms = int(diffs.iloc[0].total_seconds() * 1000)
            if dx_ms > 0:
                return {"mode": "linear", "x0": values.iloc[0].isoformat(), "dx": dx_ms}
    return {"mode": "array", "values": [None if pd.isna(v) else pd.Timestamp(v).isoformat() for v in values]}


def _relationship_payload(
    features: pd.DataFrame,
    feature_name: str,
    reference: pd.DataFrame | None,
    correlations: dict[str, float | None],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    if reference is None or reference.empty:
        return {"available": False, "panels": []}
    max_points = int(cfg.get("report", {}).get("relationship_max_points", 10000))
    colors = {"close": "#18E0FF", "return_1bar_bps": "#00FF88", "future_return_1h_bps": "#FF7A1A"}
    x_all = to_numeric_series(features[feature_name])
    panels: list[dict[str, Any]] = []
    for col in ["close", "return_1bar_bps", "future_return_1h_bps"]:
        if col not in reference.columns:
            continue
        y_all = to_numeric_series(reference[col])
        pair = pd.DataFrame({"x": x_all, "y": y_all}).replace([np.inf, -np.inf], np.nan).dropna()
        if pair.empty:
            continue
        if len(pair) > max_points:
            idx = np.linspace(0, len(pair) - 1, max_points).astype(int)
            pair = pair.iloc[idx]
        corr = correlations.get(col)
        corr_text = "n/a" if corr is None else f"{corr:.4f}"
        panels.append(
            {
                "name": f"{col} corr={corr_text}",
                "y_name": col,
                "color": colors.get(col, "#18E0FF"),
                "x": [_json_float(v) for v in pair["x"].to_numpy()],
                "y": [_json_float(v) for v in pair["y"].to_numpy()],
            }
        )
    return {"available": bool(panels), "panels": panels}


def _json_float(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return float(f"{v:.10g}")


def _report_config(cfg: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "enabled": True,
        "output_dir": "reports/feature_diagnostics",
        "generate_html": True,
        "generate_summary_json": True,
        "include_actual_output_features": True,
        "documentation_file": "docs/price_context_feature_engineering.md",
        "rolling_window_bars": None,
        "min_valid_count": 30,
        "warn_missing_ratio": 0.20,
        "fail_missing_ratio": 0.95,
        "fail_inf_ratio": 0.05,
        "zscore_threshold": 5.0,
        "iqr_multiplier": 1.5,
        "extreme_quantile_low": 0.001,
        "extreme_quantile_high": 0.999,
        "max_outlier_timestamps": 50,
        "max_plot_outlier_points": 500,
        "max_missing_markers": 500,
        "relationship_max_points": 10000,
        "future_return_window": "1h",
        "correlation_scope": "documented",
        "high_correlation_threshold": 0.95,
        "max_high_correlation_pairs": 100,
    }
    out = defaults.copy()
    out.update(cfg.get("report", {}) or {})
    return out


def _documentation_path(cfg: dict[str, Any], report_cfg: dict[str, Any]) -> Path:
    return resolve_path(report_cfg.get("documentation_file", "docs/price_context_feature_engineering.md"), cfg.get("_project_root", "."))


def _build_reference_frame(
    features: pd.DataFrame,
    cfg: dict[str, Any],
    source_data: pd.DataFrame | None,
) -> tuple[pd.DataFrame | None, str]:
    try:
        raw = source_data.copy() if source_data is not None else load_ohlc_csv(cfg)
    except Exception as exc:
        return None, f"Price/return relationship diagnostics skipped because raw OHLC data could not be loaded: {exc}"

    inp = cfg.get("input", {})
    time_col = inp.get("time_column", "time")
    close_col = inp.get("close_column", "close")
    if time_col not in raw.columns or close_col not in raw.columns:
        return None, f"Price/return relationship diagnostics skipped because raw OHLC lacks {time_col!r} or {close_col!r}."

    ref = raw[[time_col, close_col]].rename(columns={time_col: "time", close_col: "close"}).copy()
    ref["time"] = pd.to_datetime(ref["time"], format=inp.get("datetime_format"), errors="coerce")
    ref["close"] = pd.to_numeric(ref["close"], errors="coerce")
    ref = ref.dropna(subset=["time"]).sort_values("time").drop_duplicates(subset=["time"], keep="last")
    if ref.empty:
        return None, "Price/return relationship diagnostics skipped because no valid raw close prices were available."

    target = pd.DataFrame({"_row": np.arange(len(features)), "time": pd.to_datetime(features["time"], errors="coerce")})
    target = target.sort_values("time")
    tolerance = pd.to_timedelta(float(cfg.get("data", {}).get("bar_minutes", 10)) * 1.05, unit="min")
    merged = pd.merge_asof(target, ref, on="time", direction="backward", tolerance=tolerance)
    merged = merged.sort_values("_row").reset_index(drop=True)

    close = merged["close"].astype("float64")
    out = pd.DataFrame(index=features.index)
    out["close"] = close.to_numpy()
    out["return_1bar_bps"] = 10000.0 * np.log(out["close"] / out["close"].shift(1))
    future_window = cfg.get("report", {}).get("future_return_window", "1h")
    bars = window_to_bars(future_window, cfg.get("data", {}).get("bar_minutes", 10))
    out["future_return_1h_bps"] = 10000.0 * np.log(out["close"].shift(-bars) / out["close"])
    out = out.replace([np.inf, -np.inf], np.nan)
    note = (
        "Raw close prices were aligned by timestamp; price, short-return and future-return correlations are diagnostics only. "
        "Scatter plots may be sampled for browser responsiveness; feature time-series plots use the full history."
    )
    return out, note


def _compute_relationship_correlations(
    features: pd.DataFrame,
    results: list[dict[str, Any]],
    reference: pd.DataFrame | None,
) -> dict[str, dict[str, float | None]]:
    if reference is None:
        return {}
    ref_cols = [c for c in ["close", "return_1bar_bps", "future_return_1h_bps"] if c in reference.columns]
    out: dict[str, dict[str, float | None]] = {}
    for result in results:
        name = result["feature_name"]
        if not result.get("exists") or not result.get("is_numeric"):
            out[name] = {col: None for col in ref_cols}
            continue
        x = to_numeric_series(features[name])
        corr: dict[str, float | None] = {}
        for col in ref_cols:
            y = to_numeric_series(reference[col])
            pair = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
            corr[col] = float(pair["x"].corr(pair["y"])) if len(pair) >= 3 else None
        out[name] = corr
    return out


def _compute_feature_correlation(
    features: pd.DataFrame,
    results: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    report_cfg = _report_config(cfg)
    scope = report_cfg.get("correlation_scope", "documented")
    eligible: list[str] = []
    for result in results:
        if not result.get("exists") or not result.get("is_numeric"):
            continue
        if result.get("stats", {}).get("constant_flag"):
            continue
        if scope == "documented" and not result.get("documented"):
            continue
        eligible.append(result["feature_name"])

    if len(eligible) < 2:
        return pd.DataFrame(), []
    data = features[eligible].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    corr = data.corr(min_periods=int(report_cfg.get("min_valid_count", 30)))
    threshold = float(report_cfg.get("high_correlation_threshold", 0.95))
    max_pairs = int(report_cfg.get("max_high_correlation_pairs", 100))
    pairs: list[dict[str, Any]] = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            value = corr.loc[a, b]
            if pd.notna(value) and abs(float(value)) >= threshold:
                pairs.append({"feature_a": a, "feature_b": b, "correlation": float(value)})
    pairs.sort(key=lambda item: abs(item["correlation"]), reverse=True)
    return corr, pairs[:max_pairs]


def _compute_leakage_checks(features: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    items: list[dict[str, str]] = []
    if "price_feature_time" in features.columns and "time" in features.columns:
        feature_time = pd.to_datetime(features["price_feature_time"], errors="coerce")
        event_time = pd.to_datetime(features["time"], errors="coerce")
        bad = int((feature_time > event_time).sum())
        status = "PASS" if bad == 0 else "FAIL"
        items.append(
            {
                "name": "feature timestamp <= event timestamp",
                "status": status,
                "detail": f"{bad} rows have price_feature_time later than time.",
            }
        )
    else:
        items.append(
            {
                "name": "feature timestamp <= event timestamp",
                "status": "SKIP",
                "detail": "price_feature_time or time column is missing.",
            }
        )
    if "price_feature_age_min" in features.columns:
        age = pd.to_numeric(features["price_feature_age_min"], errors="coerce")
        bad_age = int((age < 0).sum())
        items.append(
            {
                "name": "feature age is non-negative",
                "status": "PASS" if bad_age == 0 else "FAIL",
                "detail": f"{bad_age} rows have negative price_feature_age_min.",
            }
        )
    suspicious = [c for c in features.columns if any(token in c.lower() for token in ["future", "fwd", "target", "label"])]
    items.append(
        {
            "name": "future label columns",
            "status": "PASS" if not suspicious else "REVIEW",
            "detail": "No future/target/label columns found." if not suspicious else f"Review potential label columns: {suspicious}",
        }
    )
    items.append(
        {
            "name": "rolling and shift direction",
            "status": "REVIEW",
            "detail": (
                "CSV-level diagnostics cannot prove rolling implementation direction. "
                "Review source-level no-future-leakage tests and past-only rolling/asof code paths."
            ),
        }
    )
    return {"items": items}
