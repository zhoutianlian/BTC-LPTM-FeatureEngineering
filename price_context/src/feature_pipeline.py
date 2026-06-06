from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from .config import resolve_path
from .data_loader import load_ohlc_csv
from .jump_features import compute_jump_features
from .preprocessing import add_base_price_columns
from .quality_features import compute_quality_features
from .range_features import compute_range_features
from .realized_vol import compute_realized_vol
from .returns import compute_past_returns
from .trend_features import compute_trend_features
from .utils import sorted_unique_windows, window_to_label
from .validation import assert_required_output_columns, validate_and_prepare_raw
from .vol_of_vol import compute_vol_of_vol


LOGGER = logging.getLogger(__name__)


REQUIRED_OUTPUT_COLUMNS = [
    "time",
    "price_feature_time",
    "price_feature_age_min",
    "past_return_1h_bps",
    "past_return_3h_bps",
    "past_return_6h_bps",
    "past_return_12h_bps",
    "past_return_24h_bps",
    "realized_vol_1h_bps",
    "realized_vol_6h_bps",
    "realized_vol_24h_bps",
    "realized_vol_1h_per_sqrt_hour_bps",
    "realized_vol_6h_per_sqrt_hour_bps",
    "realized_vol_24h_per_sqrt_hour_bps",
    "range_width_1h_bps",
    "range_width_6h_bps",
    "range_width_24h_bps",
    "range_compression_1h",
    "range_compression_6h",
    "range_compression_24h",
    "trend_strength_1h",
    "trend_strength_6h",
    "trend_strength_24h",
    "trend_consistency_1h",
    "trend_consistency_6h",
    "trend_consistency_24h",
    "trend_direction_1h",
    "trend_direction_6h",
    "trend_direction_24h",
    "vol_of_vol_6h",
    "vol_of_vol_24h",
    "vol_of_vol_48h",
    "jump_proxy_1h",
    "jump_proxy_6h",
    "jump_proxy_24h",
    "max_jump_z_1h",
    "max_jump_z_6h",
    "max_jump_z_24h",
    "jump_count_1h",
    "jump_count_6h",
    "jump_count_24h",
    "price_missing_ratio_1h",
    "price_missing_ratio_6h",
    "price_missing_ratio_24h",
    "price_gap_flag_1h",
    "price_gap_flag_6h",
    "price_gap_flag_24h",
    "price_outlier_flag_1h",
    "price_outlier_flag_6h",
    "price_outlier_flag_24h",
]


def _ordered_columns(features: pd.DataFrame, cfg: dict[str, Any]) -> list[str]:
    cols: list[str] = []
    for c in REQUIRED_OUTPUT_COLUMNS:
        if c in features.columns and c not in cols:
            cols.append(c)

    include_extended = cfg["output"].get("include_extended_features", True)
    if not include_extended:
        return cols

    # Mechanism-oriented extended columns, kept after required columns.
    extended_groups: list[str] = []
    core_labels = [window_to_label(w) for w in cfg["windows"]["core_windows"]]
    vov_labels = [window_to_label(w) for w in cfg["windows"].get("vol_of_vol_windows", [])]
    quality_labels = [window_to_label(w) for w in cfg["windows"].get("quality_windows", cfg["windows"]["core_windows"])]

    for label in core_labels:
        extended_groups.extend([
            f"realized_vol_{label}_z",
            f"range_to_vol_{label}",
            f"trend_efficiency_{label}",
            f"trend_snr_{label}",
            f"trend_slope_{label}",
            f"trend_slope_tstat_{label}",
            f"trend_r2_{label}",
            f"bar_direction_align_{label}",
            f"block_direction_align_{label}",
            f"jump_ratio_bv_{label}",
            f"signed_max_jump_return_{label}_bps",
        ])
    for label in vov_labels:
        extended_groups.extend([f"vol_of_vol_abs_{label}", f"vol_of_vol_{label}_z"])
    for label in quality_labels:
        extended_groups.extend([
            f"price_obs_count_{label}",
            f"price_expected_count_{label}",
            f"price_missing_ratio_{label}",
            f"price_gap_flag_{label}",
            f"price_outlier_flag_{label}",
        ])

    for c in extended_groups:
        if c in features.columns and c not in cols:
            cols.append(c)
    for c in features.columns:
        if c not in cols and not c.startswith("price_valid_window_"):
            cols.append(c)
    return cols


def build_features_from_dataframe(df_raw: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    prepared, report = validate_and_prepare_raw(df_raw, cfg)
    base = add_base_price_columns(prepared, cfg)

    return_windows = cfg["windows"]["return_windows"]
    core_windows = cfg["windows"]["core_windows"]
    vov_windows = cfg["windows"].get("vol_of_vol_windows", ["6h", "24h", "48h"])
    quality_windows = sorted_unique_windows(
        cfg["windows"].get("quality_windows", []),
        return_windows,
        core_windows,
        vov_windows,
    )

    quality = compute_quality_features(base, cfg, quality_windows)
    past_returns = compute_past_returns(base, quality, cfg, return_windows)
    realized_vol = compute_realized_vol(base, quality, cfg, core_windows)
    range_features = compute_range_features(base, quality, realized_vol, cfg, core_windows)
    trend_features = compute_trend_features(base, quality, past_returns, realized_vol, cfg, core_windows)
    vov_features = compute_vol_of_vol(realized_vol, quality, cfg, vov_windows)
    jump_features = compute_jump_features(base, quality, cfg, core_windows)

    features = pd.DataFrame(index=base.index)
    features["time"] = base["time"].values
    features["price_feature_time"] = base["time"].values
    features["price_feature_age_min"] = 0.0
    features = pd.concat(
        [
            features,
            past_returns,
            realized_vol,
            range_features,
            trend_features,
            vov_features,
            jump_features,
            quality,
        ],
        axis=1,
    )
    assert_required_output_columns(features, REQUIRED_OUTPUT_COLUMNS)
    ordered = _ordered_columns(features, cfg)
    features = features[ordered]
    return features.reset_index(drop=True), report.to_dict()



def write_features_csv_fast(features: pd.DataFrame, output_path: Path, float_precision: int = 10, chunk_size: int = 10000) -> None:
    """Write a wide feature frame to CSV faster than pandas.to_csv for this workload.

    The output keeps ISO-like datetime strings and uses empty fields for NaN,
    matching common CSV conventions. It avoids Python per-cell float_format calls.
    """
    import numpy as np

    columns = list(features.columns)
    datetime_cols = [c for c in columns if pd.api.types.is_datetime64_any_dtype(features[c])]
    numeric_cols = [c for c in columns if c not in datetime_cols]
    col_pos = {c: i for i, c in enumerate(columns)}
    fmt = f"%.{int(float_precision)}g"

    with output_path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(columns) + "\n")
        for start in range(0, len(features), chunk_size):
            chunk = features.iloc[start : start + chunk_size]
            # 48 chars is ample for configured datetimes and numeric precision.
            arr_out = np.empty((len(chunk), len(columns)), dtype="<U48")
            for col in datetime_cols:
                arr_out[:, col_pos[col]] = chunk[col].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(dtype="<U19")
            for col in numeric_cols:
                values = chunk[col].to_numpy(dtype="float64", copy=False)
                text = np.char.mod(fmt, values).astype("<U48")
                nan_mask = np.isnan(values)
                if nan_mask.any():
                    text[nan_mask] = ""
                arr_out[:, col_pos[col]] = text
            f.write("\n".join(",".join(row) for row in arr_out))
            f.write("\n")


def write_feature_zip(csv_path: Path, zip_path: Path, arcname: str | None = None) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname=arcname or csv_path.name)


def write_features_csv(features: pd.DataFrame, output_path: Path, cfg: dict[str, Any]) -> None:
    writer = cfg["output"].get("csv_writer", "pandas")
    if writer == "pandas":
        features.to_csv(output_path, index=False)
        return
    if writer == "fast":
        float_precision = int(cfg["output"].get("float_precision", 10))
        write_features_csv_fast(features, output_path, float_precision=float_precision)
        return
    raise ValueError("output.csv_writer must be either 'pandas' or 'fast'.")


def run_pipeline(cfg: dict[str, Any]) -> tuple[Path, pd.DataFrame, dict[str, Any]]:
    raw = load_ohlc_csv(cfg)
    features, report = build_features_from_dataframe(raw, cfg)
    output_dir = resolve_path(cfg["output"]["output_dir"], cfg.get("_project_root", "."))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / cfg["output"].get("feature_file", "price_context_features.csv")
    write_features_csv(features, output_path, cfg)
    report["output_file"] = str(output_path)
    if cfg["output"].get("write_zip", False):
        zip_path = output_dir / cfg["output"].get("zip_file", f"{output_path.name}.zip")
        write_feature_zip(output_path, zip_path, arcname=output_path.name)
        report["zip_file"] = str(zip_path)
    if cfg.get("report", {}).get("enabled", True):
        try:
            from .diagnostics import generate_feature_diagnostics_report

            diagnostics_summary = generate_feature_diagnostics_report(
                features,
                cfg,
                source_data=raw,
                output_path=output_path,
                validation_report=report,
            )
            report["feature_diagnostics"] = {
                "index_html": diagnostics_summary.get("index_html"),
                "summary_json": diagnostics_summary.get("summary_json"),
                "feature_total": diagnostics_summary.get("feature_total"),
                "fail_count": diagnostics_summary.get("fail_count"),
                "warn_count": diagnostics_summary.get("warn_count"),
            }
        except Exception as exc:
            LOGGER.exception("Feature diagnostics report generation failed.")
            if cfg.get("report", {}).get("fail_on_error", True):
                raise
            report["feature_diagnostics"] = {"error": str(exc)}
    return output_path, features, report
