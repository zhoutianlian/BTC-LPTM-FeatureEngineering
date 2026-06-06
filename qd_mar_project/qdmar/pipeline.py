"""End-to-end QD-MAR pipeline."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from .config import Config
from .io import load_plie_csv, save_csv, save_json
from .validation import run_core_validations, validation_results_to_frame, check_available_time, check_memory_asof
from .calibration import save_calibrators
from .absorption import prepare_base_context, fit_calibrators, compute_absorption_events, compute_absorption_curve
from .path_absorption import compute_path_absorption, build_path_absorption_multiscale
from .price_context import load_price_context, merge_price_context_asof
from .memory import build_memory_features
from .evaluation import (q65_coverage, context_distribution, label_proportions, directional_label_proportions, percentile_summary, directional_quality_summary, state_exit_rates, staleness_diagnostics, agent_feature_correlation, memory_quality, latest_summary, path_context_distribution, path_label_proportions, path_context_label_combo_counts, path_quality_summary, path_scenario_examples, state_proxy_distribution, state_evidence_summary, typical_state_evidence_examples)
from .visualization import build_all_html


BASE_CONTEXT_COLUMNS = [
    "time", "price", "split", "liq_feature_time_raw", "liq_feature_time", "liq_feature_age_min",
    "hmm_state", "hmm_conf", "liq_entropy",
    "age_in_state_source", "state_severity", "plie_direction",
    "plie_main_bps", "plie_reliability", "plie_intensity", "plie_phase",
    "mar_sigma_past_20m_bps", "mar_sigma_past_30m_bps", "mar_sigma_past_60m_bps",
    "state_severity_bucket", "plie_phase_group", "vol_regime",
]

BASE_CONTEXT_PREFIXES = [
    "realized_vol_", "range_compression_", "trend_strength_", "trend_consistency_",
    "trend_direction_", "jump_proxy_", "price_missing_ratio_", "price_gap_flag_",
    "price_outlier_flag_",
]

MEMORY_CONTEXT_COLUMNS = [
    "time", "price", "hmm_state", "plie_main_bps", "plie_reliability",
    "plie_direction", "plie_phase", "split",
]

PATH_ABSORPTION_COLUMNS = [
    "time", "available_time", "window_hours", "split", "price", "hmm_state",
    "path_pressure_name", "path_return_bps", "path_pressure_mass", "path_net_pressure",
    "path_directionality", "path_dominant_direction", "path_pressure_obs_count",
    "path_pressure_missing_ratio", "path_aligned_return_bps", "path_baseline_impact_bps",
    "path_signed_plie_effective_sum_bps",
    "path_raw_plie_total_bps", "path_net_braw_bps", "path_direction_consistency",
    "path_liq_neutrality_score", "path_snr", "path_active_z",
    "path_active_dominance_score", "path_active_dominance_price_score",
    "path_aligned_response_bps", "path_transmission_ratio",
    "path_absorption_score_0_100", "path_quality",
    "path_data_quality", "path_signal_clarity", "path_activity_level",
    "path_pressure_rejection_score", "path_cascade_score",
    "path_quiet_score", "path_chop_score",
    "realized_vol_used_bps", "range_compression_used", "trend_strength_used",
    "trend_consistency_used", "trend_direction_used", "jump_proxy_used",
    "price_missing_ratio_used", "price_gap_flag_used", "price_outlier_flag_used",
    "path_context", "path_label",
]

EVENT_MATURED_COLUMNS = [
    "event_time", "available_time", "horizon", "horizon_minutes", "split",
    "hmm_state", "state_severity", "plie_phase", "plie_direction",
    "plie_reliability", "signed_plie_effective_bps", "plie_reference_raw_bps",
    "actual_return_bps", "aligned_actual_response_bps", "snr_raw",
    "volatility_sigma_past_bps", "quality_weight", "response_context",
    "transmission_ratio_raw", "absorption_raw", "response_percentile",
    "absorption_score_q_0_100", "active_force_aligned_score",
    "active_force_price_score", "neutral_active_strength_score_0_100",
    "directional_absorption_label", "market_response_label",
]

REPORT_DEFAULTS = {
    "validation_results": "validation_results.csv",
    "q65_coverage": "q65_coverage.csv",
    "context_distribution": "context_distribution.csv",
    "label_proportions": "label_proportions.csv",
    "directional_label_proportions": "directional_label_proportions.csv",
    "percentile_summary": "percentile_summary.csv",
    "directional_quality_summary": "directional_quality_summary.csv",
    "state_exit_rates": "state_exit_rates.csv",
    "staleness_diagnostics": "staleness_diagnostics.csv",
    "agent_feature_correlation": "agent_feature_correlation.csv",
    "memory_quality": "memory_quality.csv",
    "path_context_distribution": "path_context_distribution.csv",
    "path_label_proportions": "path_label_proportions.csv",
    "path_context_label_combo_counts": "path_context_label_combo_counts.csv",
    "path_quality_summary": "path_quality_summary.csv",
    "path_scenario_examples": "path_scenario_examples.csv",
    "state_proxy_distribution": "state_proxy_distribution.csv",
    "state_evidence_summary": "state_evidence_summary.csv",
    "typical_state_evidence_examples": "typical_state_evidence_examples.csv",
    "latest_summary": "latest_summary.json",
}


def _save_csv(cfg: Config, df: pd.DataFrame, path: Path) -> None:
    save_csv(
        df,
        path,
        date_format=str(cfg.io_option("csv_date_format", "%Y-%m-%dT%H:%M:%SZ")),
        chunksize=int(cfg.io_option("csv_chunksize", 50000)),
    )


def _configured_windows(values) -> list[int]:
    return [int(str(v).replace("h", "")) for v in values]


def _auto_value(value, fallback):
    return fallback if value in (None, "auto") else value


def run_pipeline(config_path: str | Path, make_html: bool | None = None) -> dict[str, Path]:
    """Run the full offline training/evaluation/reporting pipeline.

    Steps are strictly time-causal:
    1. load source-clock PLIE dataframe;
    2. compute past-matured volatility context;
    3. fit empirical CDF calibrators on train split only;
    4. compute matured event-level absorption labels;
    5. build rolling memory using available_time <= current time;
    6. evaluate and render HTML reports.
    """
    cfg = Config.from_yaml(config_path)
    if make_html is None:
        make_html = bool(cfg.run_flag("make_html", True))
    cfg.ensure_dirs()
    input_csv = cfg.path("paths", "input_csv")
    df = load_plie_csv(
        input_csv,
        time_col=cfg.get("data", "time_col", default="time"),
        timestamp_cols=cfg.io_option("timestamp_parse_columns", ["liq_feature_time", "liq_feature_time_raw"]),
    )

    base_df = prepare_base_context(df, cfg)
    # Optional price-context features are past-only and joined backward-asof.
    # They are used as supplementary response/context evidence for path labels
    # and state-evidence diagnostics, never to redefine PLIE pressure context.
    price_context_df = load_price_context(cfg)
    base_df = merge_price_context_asof(base_df, price_context_df, cfg)
    calibrators = fit_calibrators(base_df, cfg)
    save_calibrators(calibrators, cfg.path("paths", "calibration_state"))

    event_df = compute_absorption_events(base_df, cfg, calibrators)
    # Offline source files can contain precomputed future-return labels at the
    # end of the sample. Production cannot use labels whose maturity time is
    # beyond the latest observed event time, so the persisted event table is
    # restricted to genuinely matured rows.
    max_observed_time = pd.to_datetime(base_df[cfg.get("data", "time_col", default="time")], utc=True).max()
    if bool(cfg.run_flag("drop_unmatured_tail", True)):
        event_df = event_df[pd.to_datetime(event_df["available_time"], utc=True) <= max_observed_time].copy()
    curve_df = compute_absorption_curve(event_df, cfg)
    if bool(cfg.run_flag("drop_unmatured_tail", True)):
        curve_df = curve_df[pd.to_datetime(curve_df["available_time"], utc=True) <= max_observed_time].copy()
    path_df = compute_path_absorption(base_df, cfg)
    if bool(cfg.run_flag("drop_unmatured_tail", True)):
        path_df = path_df[pd.to_datetime(path_df["available_time"], utc=True) <= max_observed_time].copy() if not path_df.empty else path_df
    path_windows = _configured_windows(cfg.get("path_context", "windows", default=cfg.get("path_absorption", "windows_hours", default=[6, 12, 24, 48])))
    path_multiscale_df = build_path_absorption_multiscale(path_df, windows=path_windows, cfg=cfg)
    time_col = cfg.get("data", "time_col", default="time")
    split_col = cfg.get("data", "split_col", default="split")
    if not path_multiscale_df.empty and "split" not in path_multiscale_df.columns and split_col in base_df.columns:
        split_frame = base_df[[time_col, split_col]].rename(columns={time_col: "time", split_col: "split"})
        path_multiscale_df = path_multiscale_df.merge(split_frame, on="time", how="left")
    memory_df = build_memory_features(base_df, event_df, curve_df, cfg, path_df)

    base_context_path = cfg.output_path("features", "base_context", "base_context.csv")
    event_matured_path = cfg.output_path("features", "absorption_event_matured", "absorption_event_matured.csv")
    curve_path = cfg.output_path("features", "absorption_curve", "absorption_curve.csv")
    memory_path = cfg.output_path("features", "absorption_memory", "absorption_memory.csv")
    path_absorption_path = cfg.output_path("features", "path_absorption", "path_absorption.csv")
    path_multiscale_path = cfg.output_path("features", "path_absorption_multiscale", "path_absorption_multiscale.csv")
    base_keep = [c for c in cfg.output_columns("base_context", BASE_CONTEXT_COLUMNS) if c in base_df.columns]
    # Preserve selected price-context columns for audit when available.
    base_prefixes = cfg.output_prefixes("base_context", BASE_CONTEXT_PREFIXES)
    for c in base_df.columns:
        if base_prefixes and c.startswith(base_prefixes) and c not in base_keep:
            base_keep.append(c)
    _save_csv(cfg, base_df[base_keep], base_context_path)

    _save_csv(cfg, curve_df, curve_path)

    memory_keep = []
    for c in cfg.output_columns("memory_context", MEMORY_CONTEXT_COLUMNS) + cfg.agent_inputs:
        if c in memory_df.columns and c not in memory_keep:
            memory_keep.append(c)
    # Keep latest maturity timestamps for audit even if they are not Agent inputs.
    audit_contains = cfg.get("outputs", "columns", "memory_audit_contains", default=["available_time", "last_directional_core"]) or []
    for c in memory_df.columns:
        if any(fragment in c for fragment in audit_contains) and c not in memory_keep:
            memory_keep.append(c)
    _save_csv(cfg, memory_df[memory_keep] if memory_keep else memory_df, memory_path)

    # Persist compact online/research outputs to keep the full pipeline fast and
    # usable in production.  In-memory dataframes still contain all columns for
    # evaluation and HTML rendering.
    path_keep = [c for c in cfg.output_columns("path_absorption", PATH_ABSORPTION_COLUMNS) if c in path_df.columns]
    # Persist the full 24h path table. Path absorption is a state audit layer,
    # and users need to verify every hourly context/label transition. We keep
    # all configured windows in memory for reports/HTML, while the CSV stores
    # the full primary 24h episode series rather than a recent tail slice.
    save_window = _auto_value(cfg.get("outputs", "filters", "path_window_hours", default=24), 24)
    path_to_save = path_df[path_df["window_hours"].eq(int(save_window))].copy() if "window_hours" in path_df.columns else path_df.copy()
    _save_csv(cfg, path_to_save[path_keep] if path_keep else path_to_save, path_absorption_path)
    _save_csv(cfg, path_multiscale_df, path_multiscale_path)

    main_horizon = cfg.get("memory", "main_horizon", default="30m")
    validation_results = run_core_validations(
        df,
        cfg.get("data", "time_col"),
        cfg.get("data", "split_col"),
        cfg.agent_inputs,
        expected_splits=cfg.get("data", "expected_splits", default=["train", "validation", "test"]),
        source_time_col=cfg.get("validation", "source_time_col", default=cfg.get("data", "source_time_col", default="liq_feature_time")),
        forbidden_agent_input_fragments=cfg.get("validation", "forbidden_agent_input_fragments", default=None),
    )
    validation_results.extend([check_available_time(event_df), check_memory_asof(memory_df, event_df, main_horizon=main_horizon)])
    validation_frame = validation_results_to_frame(validation_results)
    _save_csv(cfg, validation_frame, cfg.output_path("reports", "validation_results", REPORT_DEFAULTS["validation_results"]))

    q65 = q65_coverage(event_df)
    contexts = context_distribution(event_df)
    labels = label_proportions(event_df)
    directional_labels = directional_label_proportions(event_df)
    pct = percentile_summary(event_df)
    directional_quality = directional_quality_summary(event_df)
    state_exit_windows = tuple(int(v) for v in cfg.get("outputs", "report_params", "state_exit_windows", default=[12, 24]))
    exits = state_exit_rates(base_df, event_df, horizon=main_horizon, windows=state_exit_windows)
    staleness = staleness_diagnostics(memory_df, main_horizon)
    agent_corr = agent_feature_correlation(memory_df, cfg.agent_inputs)
    mem_quality = memory_quality(memory_df, cfg.agent_inputs)
    path_contexts = path_context_distribution(path_df)
    path_labels = path_label_proportions(path_df)
    path_combo_counts = path_context_label_combo_counts(path_df)
    path_quality = path_quality_summary(path_df)
    scenario_window = int(_auto_value(cfg.get("outputs", "report_params", "path_scenario_main_window", default=24), 24))
    path_scenarios = path_scenario_examples(path_df, main_window=scenario_window)
    state_proxy = state_proxy_distribution(path_multiscale_df)
    state_evidence = state_evidence_summary(path_multiscale_df)
    typical_examples = typical_state_evidence_examples(path_multiscale_df)
    latest = latest_summary(base_df, memory_df, event_df, cfg)

    reports = {
        "q65_coverage": q65,
        "context_distribution": contexts,
        "label_proportions": labels,
        "directional_label_proportions": directional_labels,
        "percentile_summary": pct,
        "directional_quality_summary": directional_quality,
        "state_exit_rates": exits,
        "staleness_diagnostics": staleness,
        "agent_feature_correlation": agent_corr,
        "memory_quality": mem_quality,
        "path_context_distribution": path_contexts,
        "path_label_proportions": path_labels,
        "path_context_label_combo_counts": path_combo_counts,
        "path_quality_summary": path_quality,
        "path_scenario_examples": path_scenarios,
        "state_proxy_distribution": state_proxy,
        "state_evidence_summary": state_evidence,
        "typical_state_evidence_examples": typical_examples,
        "latest_summary": latest,
    }
    for name, report in reports.items():
        if name == "latest_summary":
            save_json(report, cfg.output_path("reports", name, REPORT_DEFAULTS[name]))
        elif isinstance(report, pd.DataFrame):
            _save_csv(cfg, report, cfg.output_path("reports", name, REPORT_DEFAULTS[name]))
    if make_html:
        build_all_html(base_df, event_df, curve_df, memory_df, reports, cfg, path_df)

    # Save the compact main-horizon event table last.  This keeps heavy writes
    # from blocking subsequent report/HTML generation in constrained notebook
    # filesystems while preserving the full in-memory event dataframe for all
    # evaluations and visualizations.
    event_keep = [c for c in cfg.output_columns("absorption_event_matured", EVENT_MATURED_COLUMNS) if c in event_df.columns]
    save_horizon = _auto_value(cfg.get("outputs", "filters", "event_horizon", default=main_horizon), main_horizon)
    event_to_save = event_df[event_df["horizon"].eq(save_horizon)].copy()
    _save_csv(cfg, event_to_save[event_keep] if event_keep else event_to_save, event_matured_path)

    return {
        "base_context": base_context_path,
        "absorption_event_matured": event_matured_path,
        "absorption_curve": curve_path,
        "path_absorption": path_absorption_path,
        "path_absorption_multiscale": path_multiscale_path,
        "state_proxy_distribution": cfg.output_path("reports", "state_proxy_distribution", REPORT_DEFAULTS["state_proxy_distribution"]),
        "state_evidence_summary": cfg.output_path("reports", "state_evidence_summary", REPORT_DEFAULTS["state_evidence_summary"]),
        "typical_state_evidence_examples": cfg.output_path("reports", "typical_state_evidence_examples", REPORT_DEFAULTS["typical_state_evidence_examples"]),
        "absorption_memory": memory_path,
        "validation_results": cfg.output_path("reports", "validation_results", REPORT_DEFAULTS["validation_results"]),
        "html_index": cfg.output_path("html", "index", "index.html"),
        "calibration_state": cfg.path("paths", "calibration_state"),
    }
