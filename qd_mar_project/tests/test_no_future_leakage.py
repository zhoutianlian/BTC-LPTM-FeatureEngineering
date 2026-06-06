from __future__ import annotations

import numpy as np
import pandas as pd

from qdmar.statistics import infer_maturity_lag_minutes, past_matured_sigma
from qdmar.validation import (
    check_timestamp_monotonicity,
    check_source_clock_alignment,
    check_split_order,
    check_available_time,
    check_agent_inputs_no_future,
)


def test_maturity_lag_for_10m_grid():
    times = pd.date_range("2024-01-01", periods=20, freq="10min", tz="UTC")
    assert infer_maturity_lag_minutes(pd.Series(times), 20) == 2
    assert infer_maturity_lag_minutes(pd.Series(times), 30) == 3
    assert infer_maturity_lag_minutes(pd.Series(times), 60) == 6


def test_past_matured_sigma_does_not_use_current_future_label():
    times = pd.date_range("2024-01-01", periods=20, freq="10min", tz="UTC")
    df = pd.DataFrame({"time": times, "ret_20m_bps": np.ones(20)})
    df.loc[10, "ret_20m_bps"] = 10000.0
    sigma = past_matured_sigma(df, "time", "ret_20m_bps", horizon_minutes=20, window=5, min_periods=3)
    # At row 10, the row-10 label is future information and must not affect sigma.
    assert sigma.iloc[10] < 1e-6 or np.isnan(sigma.iloc[10])
    # The extreme label can only appear after the maturity lag.
    assert sigma.iloc[12] >= 0 or np.isnan(sigma.iloc[12])


def test_timestamp_and_source_clock_checks():
    df = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC"),
        "liq_feature_time": pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC"),
        "split": ["train", "validation", "test"],
    })
    assert check_timestamp_monotonicity(df).passed
    assert check_source_clock_alignment(df).passed
    assert check_split_order(df).passed


def test_available_time_is_after_event_time():
    df = pd.DataFrame({
        "event_time": pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC"),
        "available_time": pd.date_range("2024-01-01 00:30", periods=3, freq="1h", tz="UTC"),
    })
    assert check_available_time(df).passed


def test_agent_inputs_reject_future_like_columns():
    good = ["mar_abs_score_q_ewm_6_30m", "mar_takeover_count_12_30m"]
    bad = good + ["ret_30m_bps", "absorption_raw_30m"]
    assert check_agent_inputs_no_future(good).passed
    assert not check_agent_inputs_no_future(bad).passed


def test_staleness_aware_memory_decays_old_directional_signal():
    from qdmar.config import Config
    from qdmar.memory import build_memory_features
    import tempfile
    from pathlib import Path
    import yaml

    times = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
    base = pd.DataFrame({
        "time": times,
        "price": np.arange(6.0),
        "hmm_state": [1]*6,
        "plie_main_bps": [1.0]*6,
        "plie_reliability": [0.8]*6,
        "plie_direction": [1]*6,
        "plie_phase": ["normal"]*6,
        "split": ["train"]*6,
    })
    event = pd.DataFrame({
        "event_time": times[:5],
        "available_time": times[:5],
        "horizon": ["30m"]*5,
        "response_context": ["directional_core", "low_quality_plie", "low_quality_plie", "low_quality_plie", "low_quality_plie"],
        "absorption_score_q_0_100": [90.0, np.nan, np.nan, np.nan, np.nan],
        "active_force_aligned_score": [-0.5, np.nan, np.nan, np.nan, np.nan],
        "active_force_price_score": [-0.5, np.nan, np.nan, np.nan, np.nan],
        "neutral_active_strength_score_0_100": [np.nan, 20.0, 20.0, 20.0, 20.0],
        "market_response_label": ["reversal_takeover", "low_active_move", "low_active_move", "low_active_move", "low_active_move"],
        "quality_weight": [0.8, 0.0, 0.0, 0.0, 0.0],
    })
    curve = pd.DataFrame({
        "event_time": times[:5],
        "available_time": times[:5],
        "mar_curve_label": ["mixed_or_noise"]*5,
        "mar_response_conflict_score": [0.0]*5,
    })
    cfg_dict = {
        "paths": {"output_dir": "output", "features_dir": "output/features", "reports_dir": "output/reports", "html_dir": "output/html", "state_dir": "state"},
        "data": {"time_col": "time"},
        "memory": {"main_horizon": "30m", "ewm_spans": [6], "persistence_window": 6, "takeover_window": 12, "curve_mode_window": 6, "directional_decay_halflife_hours": 1},
        "agent_inputs": [],
    }
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cfg.yaml"
        p.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")
        cfg = Config.from_yaml(p)
        mem = build_memory_features(base, event, curve, cfg)
    raw_last = mem["mar_abs_score_q_ewm_6_30m"].iloc[-1]
    stale_last = mem["mar_abs_score_q_staleaware_ewm_6_30m"].iloc[-1]
    assert raw_last > 80
    assert stale_last < raw_last
    assert abs(stale_last - 50) < 10


def test_path_absorption_uses_only_current_and_past_prices():
    from qdmar.config import Config
    from qdmar.path_absorption import compute_path_absorption
    import tempfile
    from pathlib import Path
    import yaml

    times = pd.date_range("2024-01-01", periods=30, freq="1h", tz="UTC")
    base = pd.DataFrame({
        "time": times,
        "price": np.linspace(100.0, 130.0, 30),
        "split": ["train"] * 30,
        "hmm_state": [5] * 30,
        "state_severity": [2] * 30,
        "state_severity_bucket": ["strong"] * 30,
        "plie_phase": ["normal"] * 30,
        "vol_regime": ["mid"] * 30,
        "plie_direction": [-1] * 30,
        "plie_reliability": [0.8] * 30,
        "plie_passive_30m_bps": [-5.0] * 30,
        "plie_passive_30m_bps_mag_raw": [5.0] * 30,
    })
    cfg_dict = {
        "paths": {"output_dir": "output", "features_dir": "output/features", "reports_dir": "output/reports", "html_dir": "output/html", "state_dir": "state"},
        "data": {"time_col": "time", "price_col": "price"},
        "horizons": [{"name": "30m", "minutes": 30, "ret_col": "ret_30m_bps", "plie_col": "plie_passive_30m_bps", "raw_mag_col": "plie_passive_30m_bps_mag_raw", "eff_mag_col": "plie_passive_30m_bps_mag", "b_min_bps": 2.5}],
        "absorption": {"eps": 1e-9},
        "path_absorption": {"enabled": True, "main_horizon": "30m", "windows_hours": [6], "vol_window": 10, "vol_min_periods": 3},
    }
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cfg.yaml"
        p.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")
        cfg = Config.from_yaml(p)
        path_a = compute_path_absorption(base, cfg)
        modified = base.copy()
        modified.loc[20:, "price"] *= 10.0
        path_b = compute_path_absorption(modified, cfg)
    # Rows before the future mutation point must be identical.
    a = path_a[path_a["time"] < times[20]]["path_return_bps"].reset_index(drop=True)
    b = path_b[path_b["time"] < times[20]]["path_return_bps"].reset_index(drop=True)
    assert np.allclose(a.fillna(0), b.fillna(0))


def test_weak_directional_context_is_not_true_neutral():
    from qdmar.absorption import classify_response_context

    ctx = classify_response_context(
        direction=pd.Series([-1, 0, -1]),
        reliability=pd.Series([0.45, 0.9, 0.05]),
        snr_raw=pd.Series([0.12, 0.5, 0.01]),
        b_raw=pd.Series([5.0, 5.0, 1.0]),
        b_min=2.5,
        reliability_min=0.65,
        snr_min=0.20,
        weak_reliability_min=0.30,
        weak_snr_min=0.10,
    )
    assert ctx.iloc[0] == "weak_directional_context"
    assert ctx.iloc[1] == "true_neutral_plie"
    assert ctx.iloc[2] == "low_quality_plie"


def test_path_neutral_pressure_big_move_gets_active_dominance_label():
    from qdmar.config import Config
    from qdmar.path_absorption import compute_path_absorption
    import tempfile
    from pathlib import Path
    import yaml

    times = pd.date_range("2024-01-01", periods=40, freq="1h", tz="UTC")
    # Alternating directions cancel cumulative PLIE pressure, while price trends
    # strongly upward. This should not be labeled path_no_pressure.
    base = pd.DataFrame({
        "time": times,
        "price": np.r_[np.linspace(100.0, 102.0, 20), np.linspace(102.0, 140.0, 20)],
        "split": ["train"] * 40,
        "hmm_state": [3] * 40,
        "state_severity": [0] * 40,
        "state_severity_bucket": ["neutral"] * 40,
        "plie_phase": ["neutral"] * 40,
        "vol_regime": ["mid"] * 40,
        "plie_direction": [1, -1] * 20,
        "plie_reliability": [0.4] * 40,
        "plie_passive_30m_bps": [1.0, -1.0] * 20,
        "plie_passive_30m_bps_mag_raw": [2.0] * 40,
    })
    cfg_dict = {
        "paths": {"output_dir": "output", "features_dir": "output/features", "reports_dir": "output/reports", "html_dir": "output/html", "state_dir": "state"},
        "data": {"time_col": "time", "price_col": "price"},
        "horizons": [{"name": "30m", "minutes": 30, "ret_col": "ret_30m_bps", "plie_col": "plie_passive_30m_bps", "raw_mag_col": "plie_passive_30m_bps_mag_raw", "eff_mag_col": "plie_passive_30m_bps_mag", "b_min_bps": 2.5}],
        "absorption": {"eps": 1e-9},
        "path_absorption": {
            "enabled": True, "main_horizon": "30m", "windows_hours": [12],
            "vol_window": 10, "vol_min_periods": 3, "min_total_braw_bps": 4.0,
            "min_net_braw_bps": 8.0, "active_z_low": 0.5, "active_z_strong": 1.0,
            "active_z_extreme": 2.0
        },
    }
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cfg.yaml"
        p.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")
        cfg = Config.from_yaml(p)
        out = compute_path_absorption(base, cfg)
    last = out.dropna(subset=["path_return_bps"]).iloc[-1]
    assert last["path_context"] in {"path_mixed_pressure", "path_neutral_pressure"}
    assert "active" in last["path_label"]
    assert "up" in last["path_label"]


def test_path_absorption_multiscale_wide_output_schema():
    """The multiscale output exposes all requested windows in one causal row."""
    from qdmar.path_absorption import build_path_absorption_multiscale

    times = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    rows = []
    for t in times:
        for wh in [6, 12, 24, 48]:
            rows.append({
                "time": t,
                "available_time": t,
                "window_hours": wh,
                "path_context": "path_neutral_pressure",
                "path_label": "path_quiet_no_pressure",
                "path_absorption_score_0_100": 50.0 + wh,
                "path_pressure_rejection_score": 0.1,
                "path_active_dominance_score": 0.2,
                "path_transmission_ratio": 0.0,
                "path_direction_consistency": 0.0,
                "path_cascade_score": 0.0,
                "path_data_quality": 1.0,
                "path_signal_clarity": 0.8,
                "path_activity_level": 0.3,
            })
    long_df = pd.DataFrame(rows)
    wide = build_path_absorption_multiscale(long_df, windows=[6, 12, 24, 48])
    assert len(wide) == len(times)
    required_columns = ["time", "available_time"]
    for base in [
        "path_context", "path_label", "path_absorption_score",
        "path_pressure_rejection_score", "path_active_dominance_score",
        "path_transmission_ratio", "path_direction_consistency",
        "path_cascade_score", "path_data_quality",
        "path_signal_clarity", "path_activity_level",
    ]:
        for wh in [6, 12, 24, 48]:
            required_columns.append(f"{base}_{wh}h")
    missing = [c for c in required_columns if c not in wide.columns]
    assert not missing
    assert wide["available_time"].equals(wide["time"])
    assert wide["path_absorption_score_48h"].iloc[-1] == 98.0


def test_path_quality_decouples_low_activity_from_data_quality():
    from qdmar.path_absorption import build_path_absorption_multiscale

    t = pd.Timestamp("2026-01-01", tz="UTC")
    rows = []
    for wh in [6, 12, 24, 48]:
        rows.append({
            "time": t,
            "available_time": t,
            "window_hours": wh,
            "path_context": "path_neutral_pressure",
            "path_label": "path_quiet_no_pressure",
            "path_absorption_score_0_100": 50.0,
            "path_pressure_rejection_score": 0.0,
            "path_active_dominance_score": 0.0,
            "path_active_dominance_price_score": 0.0,
            "path_transmission_ratio": 0.0,
            "path_direction_consistency": 0.0,
            "path_cascade_score": 0.0,
            "path_data_quality": 1.0,
            "path_signal_clarity": 0.8,
            "path_activity_level": 0.05,
        })
    wide = build_path_absorption_multiscale(pd.DataFrame(rows), windows=[6, 12, 24, 48])
    assert wide["path_data_quality_24h"].iloc[0] == 1.0
    assert wide["path_activity_level_24h"].iloc[0] < 0.1
    assert wide["e_rc_quiet_pressure"].iloc[0] > 0
    assert wide["e_amb_data_quality_bad"].iloc[0] == 0.0
