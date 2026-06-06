from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import ProjectConfig
from .utils import robust_zscore_past_only, stable_softplus, to_utc_series


AGENT_INPUT_COLUMNS = [
    "time",
    "liq_feature_time",
    "liq_feature_age_min",
    "hmm_state",
    "hmm_conf",
    "liq_entropy",
    "age_in_state_source",
    "plie_direction",
    "plie_force_up",
    "plie_intensity",
    "plie_accel_pos",
    "plie_strong_entry",
    "plie_transition_type",
    "plie_transition_severity",
    "plie_reliability",
    "plie_phase",
    "plie_passive_20m_bps",
    "plie_passive_30m_bps",
    "plie_passive_60m_bps",
    "plie_main_bps",
]


@dataclass(frozen=True)
class FeatureMetadata:
    name: str
    financial_meaning: str
    application: str
    source: str
    time_alignment: str
    model_input: bool
    agent_input: bool
    visualization: bool


def feature_metadata() -> list[FeatureMetadata]:
    return [
        FeatureMetadata(
            "plie_direction",
            "Direction of passive liquidation pressure. +1 means short-liquidation forced buying/upward pressure; -1 means long-liquidation forced selling/downward pressure.",
            "Signed direction used to convert passive impact magnitude into signed PLIE and to align realized returns for absorption diagnostics.",
            "Fused raw liquidation imbalance and HMM posterior severity coordinate.",
            "Computed only on source-clock rows, then causally broadcast by liq_feature_time.",
            False,
            True,
            True,
        ),
        FeatureMetadata(
            "plie_intensity",
            "Magnitude of current passive liquidation pressure after robust total-liquidation normalization and directional dominance weighting.",
            "Core model input for passive impact curve. Higher values should imply larger passive impact baseline.",
            "fll_cwt_kf, fsl_cwt_kf, p_state_1..p_state_5.",
            "Past-only rolling robust z-score on source clock.",
            False,
            True,
            True,
        ),
        FeatureMetadata(
            "plie_accel_pos",
            "Positive acceleration of liquidation pressure in the current PLIE direction.",
            "Boosts PLIE when forced-flow pressure is strengthening rather than merely large.",
            "Source-clock difference of signed PLIE force.",
            "Uses only the previous source-clock snapshot via diff().",
            False,
            True,
            True,
        ),
        FeatureMetadata(
            "plie_strong_entry",
            "Indicator that market just entered state 1 or state 5, i.e. strong one-sided liquidation pressure regime.",
            "Regime-entry boost for PLIE-PIC and event diagnostics.",
            "hmm_state on source clock.",
            "Uses current and previous source-clock hard state only.",
            True,
            True,
            True,
        ),
        FeatureMetadata(
            "plie_transition_severity",
            "Mechanism-coded severity of HMM transition. Strengthening transitions get positive boost; abrupt 1<->5 reversals get negative severity.",
            "Model input and explanation layer for transition-sensitive passive impact.",
            "hmm_state transition_type.",
            "Uses previous and current source-clock state only.",
            True,
            True,
            True,
        ),
        FeatureMetadata(
            "plie_reliability",
            "Causal quality weight reflecting whether the PLIE baseline should be trusted under the current state/freshness/entropy context.",
            "Downweights neutral, stale, or high-entropy states without using future realized prices.",
            "direction coordinate, posterior entropy, liq_feature_age_min.",
            "All inputs are available at t; no future price is used.",
            False,
            True,
            True,
        ),
        FeatureMetadata(
            "plie_passive_30m_bps",
            "Signed 30m passive liquidation-implied impact baseline in bps.",
            "Main PLIE output for Agent. It is not a final price forecast; it is the forced-flow baseline used later for absorption residuals.",
            "Constrained quantile impact curve prediction times plie_direction and reliability.",
            "Generated at source-clock inference time and broadcast causally.",
            False,
            True,
            True,
        ),
    ]


def _state_series_to_transition(prev_state: pd.Series, state: pd.Series) -> pd.Series:
    prev = prev_state.astype("Int64").astype(str).replace("<NA>", "NA")
    cur = state.astype("Int64").astype(str).replace("<NA>", "NA")
    return prev + "->" + cur


def build_source_clock_frame(raw_df: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    """Build one row per liquidation source snapshot.

    This is the central anti-leakage step. The 10m raw frame contains repeated
    forward-filled liquidation snapshots. PLIE rolling, differencing, duration,
    transition, and labels are generated on this source-clock frame instead of
    on repeated 10m rows.
    """
    s = cfg.get("schema")
    age_zero = float(cfg.get("features", "source_age_zero_value", default=0.0))
    df = raw_df.copy()
    df[s["time_col"]] = to_utc_series(df[s["time_col"]])
    df[s["liq_time_col"]] = to_utc_series(df[s["liq_time_col"]])
    if s.get("liq_time_raw_col") in df.columns:
        df[s["liq_time_raw_col"]] = to_utc_series(df[s["liq_time_raw_col"]])
    df = df.sort_values(s["time_col"]).reset_index(drop=True)

    age = pd.to_numeric(df[s["liq_age_col"]], errors="coerce")
    hmm_state = pd.to_numeric(df[s["hmm_state_col"]], errors="coerce")
    valid_states = [int(x) for x in cfg.get("features", "source_valid_states", default=[1, 2, 3, 4, 5])]
    mask = age.eq(age_zero) & hmm_state.isin(valid_states)
    source = df.loc[mask].copy()
    if source.empty:
        raise ValueError("No valid source-clock rows found. Expected liq_feature_age_min == 0 and valid hmm_state.")

    source = source.sort_values(s["liq_time_col"])
    source = source.drop_duplicates(s["liq_time_col"], keep="last").reset_index(drop=True)
    return source


def build_plie_features(source: pd.DataFrame, raw_df: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    """Add PLIE-PIC engineered features and multi-horizon labels.

    Feature columns use only source-clock information up to t. Future price is
    used only to create training/evaluation labels; label columns are excluded
    from model and agent inputs.
    """
    s = cfg.get("schema")
    fcfg = cfg.get("features")
    eps = float(fcfg["eps"])
    lam = float(fcfg["hmm_posterior_weight_lambda"])
    deadzone = float(fcfg["direction_deadzone"])
    horizons = [int(h) for h in fcfg["horizons_min"]]
    w = int(fcfg["robust_window_source"])
    min_periods = int(fcfg["robust_min_periods"])
    strong_states = [int(x) for x in fcfg.get("strong_states", [1, 5])]

    df = source.copy().reset_index(drop=True)
    df[s["time_col"]] = to_utc_series(df[s["time_col"]])
    df[s["liq_time_col"]] = to_utc_series(df[s["liq_time_col"]])

    L = pd.to_numeric(df[s["long_liq_col"]], errors="coerce").clip(lower=0.0).fillna(0.0)
    S = pd.to_numeric(df[s["short_liq_col"]], errors="coerce").clip(lower=0.0).fillna(0.0)
    T = pd.to_numeric(df[s["total_liq_col"]], errors="coerce").fillna(L + S).clip(lower=0.0)
    T = np.maximum(T, L + S)

    df["plie_raw_signed_up"] = (S - L) / (T + eps)
    df["log_total_liq"] = np.log1p(T)
    df["z_log_total_liq"] = robust_zscore_past_only(df["log_total_liq"], w, min_periods, eps)
    df["plie_softplus_total_z"] = stable_softplus(df["z_log_total_liq"])

    posterior_cols = list(s["posterior_cols"])
    posterior_weights = np.asarray(fcfg.get("posterior_severity_weights", [2.0, 1.0, 0.0, -1.0, -2.0]), dtype=float)
    if len(posterior_cols) != len(posterior_weights):
        raise ValueError("features.posterior_severity_weights must have the same length as schema.posterior_cols.")
    posterior = df[posterior_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    severity_scale = float(fcfg.get("posterior_severity_scale", 2.0))
    df["plie_hmm_severity_coord"] = posterior @ posterior_weights / max(severity_scale, eps)
    df["plie_fused_pressure_coord"] = lam * df["plie_hmm_severity_coord"] + (1.0 - lam) * df["plie_raw_signed_up"]
    coord = df["plie_fused_pressure_coord"].to_numpy(dtype=float)
    direction = np.where(np.abs(coord) < deadzone, 0.0, np.sign(coord))
    df["plie_direction"] = direction.astype(float)

    df["plie_force_up"] = df["plie_fused_pressure_coord"] * df["plie_softplus_total_z"]
    df["plie_intensity"] = np.abs(df["plie_fused_pressure_coord"]) * df["plie_softplus_total_z"]
    df["plie_force_delta"] = df["plie_force_up"].diff().fillna(0.0)
    df["plie_accel_pos"] = np.maximum(0.0, df["plie_direction"] * df["plie_force_delta"])

    state = pd.to_numeric(df[s["hmm_state_col"]], errors="coerce")
    prev_state = state.shift(1)
    df["prev_hmm_state"] = prev_state
    df["plie_transition_type"] = _state_series_to_transition(prev_state, state)
    sev_map = fcfg.get("transition_severity_map", {}) or {}
    df["plie_transition_severity"] = df["plie_transition_type"].map(sev_map).fillna(0.0).astype(float)
    df["plie_strong_state"] = state.isin(strong_states).astype(int)
    df["plie_state_transition"] = (state != prev_state).fillna(False).astype(int)
    df["plie_strong_entry"] = ((df["plie_strong_state"] == 1) & (df["plie_state_transition"] == 1)).astype(int)

    entropy = pd.to_numeric(df[s["entropy_col"]], errors="coerce").fillna(float(fcfg["entropy_max"]))
    r_state = np.clip(np.abs(df["plie_fused_pressure_coord"].to_numpy(dtype=float)), 0.0, 1.0)
    r_entropy = np.clip(1.0 - entropy.to_numpy(dtype=float) / float(fcfg["entropy_max"]), 0.0, 1.0)
    age = pd.to_numeric(df[s["liq_age_col"]], errors="coerce").fillna(0.0)
    fresh_no_decay = float(fcfg["freshness_no_decay_min"])
    fresh_decay = float(fcfg["freshness_decay_min"])
    r_fresh = np.exp(-np.maximum(0.0, age.to_numpy(dtype=float) - fresh_no_decay) / max(fresh_decay, eps))
    df["plie_reliability_base"] = r_state * r_entropy
    df["plie_freshness_weight"] = r_fresh
    df["plie_reliability"] = df["plie_reliability_base"] * df["plie_freshness_weight"]

    age_source = pd.to_numeric(df[s["state_age_source_col"]], errors="coerce").fillna(0.0)
    phase_cfg = fcfg.get("phase", {}) or {}
    phase_labels = phase_cfg.get("labels", {}) or {}
    accel_window = int(phase_cfg.get("accel_window_source", 24))
    accel_min_periods = int(phase_cfg.get("accel_min_periods", 3))
    mature_age = float(phase_cfg.get("mature_state_age_source", 24))
    df["plie_phase"] = np.select(
        [
            df["plie_direction"].eq(0),
            df["plie_strong_entry"].eq(1),
            df["plie_accel_pos"].gt(df["plie_accel_pos"].rolling(accel_window, min_periods=accel_min_periods).median().fillna(0.0)),
            age_source.ge(mature_age),
        ],
        [
            phase_labels.get("neutral", "neutral"),
            phase_labels.get("early_strong_entry", "early_strong_entry"),
            phase_labels.get("accelerating", "accelerating"),
            phase_labels.get("mature", "mature"),
        ],
        default=phase_labels.get("normal", "normal"),
    )

    df["model_log1p_intensity"] = np.log1p(df["plie_intensity"].clip(lower=0.0))
    df["model_log1p_accel_pos"] = np.log1p(df["plie_accel_pos"].clip(lower=0.0))
    df["model_strong_entry"] = df["plie_strong_entry"].astype(float)
    df["model_transition_severity"] = df["plie_transition_severity"].astype(float)
    df["model_strong_state"] = df["plie_strong_state"].astype(float)

    # Labels: future price is only used here. Label columns must never enter model features.
    raw = raw_df.copy()
    raw[s["time_col"]] = to_utc_series(raw[s["time_col"]])
    raw = raw.dropna(subset=[s["time_col"]]).sort_values(s["time_col"])
    price_index = raw.drop_duplicates(s["time_col"]).set_index(s["time_col"])[s["price_col"]].astype(float).sort_index()
    p0 = price_index.reindex(df[s["time_col"]]).to_numpy(dtype=float)
    for h in horizons:
        future_time = df[s["time_col"]] + pd.to_timedelta(h, unit="m")
        ph = price_index.reindex(future_time).to_numpy(dtype=float)
        ret_bps = 10000.0 * np.log(ph / p0)
        df[f"ret_{h}m_bps"] = ret_bps
        df[f"plie_aligned_ret_{h}m_bps"] = df["plie_direction"].to_numpy(dtype=float) * ret_bps

    return df


def build_feature_frame(raw_df: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    source = build_source_clock_frame(raw_df, cfg)
    return build_plie_features(source, raw_df, cfg)


def assign_chronological_split(df: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    """Assign train/validation/test split chronologically."""
    from .utils import chronological_split_indices

    train_ratio = float(cfg.get("split", "train_ratio"))
    val_ratio = float(cfg.get("split", "validation_ratio"))
    out = df.sort_values(cfg.get("schema", "time_col")).reset_index(drop=True).copy()
    train_idx, val_idx, test_idx = chronological_split_indices(len(out), train_ratio, val_ratio)
    out["split"] = "test"
    out.loc[train_idx, "split"] = "train"
    out.loc[val_idx, "split"] = "validation"
    out.loc[test_idx, "split"] = "test"
    return out


def get_agent_input_columns(horizons: list[int] | None = None, cfg: ProjectConfig | None = None) -> list[str]:
    base_columns = AGENT_INPUT_COLUMNS.copy()
    if cfg is not None:
        configured = cfg.get("outputs", "agent_input_columns", default=None)
        if configured:
            base_columns = list(configured)
    if horizons is None:
        return base_columns.copy()
    cols = [c for c in base_columns if not c.startswith("plie_passive_")]
    cols += [f"plie_passive_{int(h)}m_bps" for h in horizons]
    if "plie_main_bps" not in cols:
        cols.append("plie_main_bps")
    return list(dict.fromkeys(cols))


def broadcast_source_predictions_to_10m(
    raw_df: pd.DataFrame,
    source_pred: pd.DataFrame,
    cfg: ProjectConfig,
) -> pd.DataFrame:
    """Broadcast source-clock PLIE outputs to 10m bars by liq_feature_time.

    The forecast magnitude is computed at source time. On 10m bars after that
    source snapshot, only freshness decay is recomputed from liq_feature_age_min.
    No new rolling or differencing liquidation feature is computed on repeated
    10m rows.
    """
    s = cfg.get("schema")
    horizons = [int(h) for h in cfg.get("features", "horizons_min")]
    eps = float(cfg.get("features", "eps"))
    no_decay = float(cfg.get("features", "freshness_no_decay_min"))
    decay = float(cfg.get("features", "freshness_decay_min"))

    key = s["liq_time_col"]
    raw = raw_df.copy()
    raw[s["time_col"]] = to_utc_series(raw[s["time_col"]])
    raw[key] = to_utc_series(raw[key])

    keep = [
        key,
        "plie_direction",
        "plie_force_up",
        "plie_intensity",
        "plie_accel_pos",
        "plie_strong_entry",
        "plie_transition_type",
        "plie_transition_severity",
        "plie_reliability_base",
        "plie_phase",
    ]
    for h in horizons:
        keep.append(f"plie_passive_{h}m_bps_mag_raw")
    keep = [c for c in keep if c in source_pred.columns]
    src = source_pred[keep].copy()
    src[key] = to_utc_series(src[key])
    src = src.drop_duplicates(key, keep="last")
    out = raw.merge(src, on=key, how="left", suffixes=("", "_source"))

    age = pd.to_numeric(out[s["liq_age_col"]], errors="coerce").fillna(np.inf)
    freshness = np.exp(-np.maximum(0.0, age.to_numpy(dtype=float) - no_decay) / max(decay, eps))
    out["plie_freshness_weight"] = freshness
    base = pd.to_numeric(out.get("plie_reliability_base", 0.0), errors="coerce").fillna(0.0)
    out["plie_reliability"] = base * out["plie_freshness_weight"]

    for h in horizons:
        raw_mag = pd.to_numeric(out.get(f"plie_passive_{h}m_bps_mag_raw", np.nan), errors="coerce")
        signed = pd.to_numeric(out.get("plie_direction", 0.0), errors="coerce").fillna(0.0) * raw_mag.fillna(0.0) * out["plie_reliability"]
        out[f"plie_passive_{h}m_bps"] = signed
    main_horizon = get_main_horizon(cfg, horizons)
    out["plie_main_bps"] = out[f"plie_passive_{main_horizon}m_bps"]
    return out


def select_bar_output_columns(df: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    """Return a compact 10m output frame for storage and Agent review."""
    horizons = [int(h) for h in cfg.get("features", "horizons_min")]
    cols = list(cfg.get("outputs", "bar_output_base_columns", default=[])) or [
        "time",
        "price",
        "liq_feature_time",
        "liq_feature_age_min",
        "hmm_state",
        "hmm_conf",
        "liq_entropy",
        "age_in_state",
        "age_in_state_source",
        "plie_direction",
        "plie_force_up",
        "plie_intensity",
        "plie_accel_pos",
        "plie_strong_entry",
        "plie_transition_type",
        "plie_transition_severity",
        "plie_freshness_weight",
        "plie_reliability",
        "plie_phase",
    ]
    cols += [f"plie_passive_{h}m_bps" for h in horizons]
    cols += ["plie_main_bps"]
    cols = [c for c in cols if c in df.columns]
    out = df.loc[:, cols].copy()
    max_rows = cfg.get("storage", "max_10m_output_rows", default=None)
    if max_rows is not None and int(max_rows) > 0 and len(out) > int(max_rows):
        out = out.tail(int(max_rows)).copy()
    return out


def get_main_horizon(cfg: ProjectConfig, horizons: list[int] | None = None) -> int:
    horizons = [int(h) for h in (horizons or cfg.get("features", "horizons_min"))]
    if not horizons:
        raise ValueError("features.horizons_min must contain at least one horizon.")
    configured = int(cfg.get("features", "main_horizon_min", default=horizons[0]))
    return configured if configured in horizons else horizons[0]
