from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from plie_pic.config import load_config
from plie_pic.features import assign_chronological_split, build_feature_frame, build_source_clock_frame, get_agent_input_columns
from plie_pic.utils import robust_zscore_past_only
from plie_pic.validation import NoFutureLeakageChecker


def _synthetic_raw() -> pd.DataFrame:
    times = pd.date_range("2024-01-01 00:50:00", periods=36, freq="10min", tz="UTC")
    liq_times = times.floor("h") + pd.Timedelta(minutes=50)
    liq_times = liq_times.where(liq_times <= times, liq_times - pd.Timedelta(hours=1))
    age = (times - liq_times).total_seconds() / 60.0
    # Keep only age values compatible with source clock and forward-filled source rows.
    total = np.linspace(10, 30, len(times))
    fll = total * 0.6
    fsl = total * 0.4
    state = np.where(np.arange(len(times)) < 18, 5, 1)
    rows = pd.DataFrame({
        "time": times,
        "price": 10000 + np.cumsum(np.ones(len(times))),
        "fll_cwt_kf": fll,
        "fsl_cwt_kf": fsl,
        "risk_priority_number": fll / total,
        "diff_ls_cwt_kf": fll - fsl,
        "total_ls_cwt_kf": total,
        "diff_dom_ls_cwt_kf": (fll - fsl) / total,
        "liq_feature_time_raw": liq_times - pd.Timedelta(minutes=50),
        "liq_feature_time": liq_times,
        "liq_feature_age_min": age,
        "hmm_state": state,
        "hmm_state_conf": 0.99,
        "p_state_1": np.where(state == 1, 0.98, 0.0),
        "p_state_2": 0.0,
        "p_state_3": 0.02,
        "p_state_4": 0.0,
        "p_state_5": np.where(state == 5, 0.98, 0.0),
        "p_short_liq": np.where(state == 1, 0.98, 0.0),
        "p_up_pressure": np.where(state == 1, 0.98, 0.0),
        "p_neutral": 0.02,
        "p_long_liq": np.where(state == 5, 0.98, 0.0),
        "p_down_pressure": np.where(state == 5, 0.98, 0.0),
        "dir_expect": np.where(state == 1, 0.98, -0.98),
        "p_bull": np.where(state == 1, 0.98, 0.0),
        "p_bear": np.where(state == 5, 0.98, 0.0),
        "liq_entropy": 0.05,
        "hmm_entropy": 0.05,
        "hmm_maxp": 0.98,
        "hmm_conf": 0.98,
        "age_in_state": np.arange(len(times)),
        "age_in_state_source": np.arange(len(times)) // 6,
        "state_name": "synthetic",
        "state_name_en": "synthetic",
        "state_name_cn": "synthetic",
        "state_liq_side": "synthetic",
        "state_liq_side_cn": "synthetic",
        "state_pressure": "synthetic",
        "state_pressure_cn": "synthetic",
        "state_pressure_direction": np.where(state == 1, 1, -1),
        "state_severity": 2,
    })
    return rows


def test_no_future_leakage_checker_passes_synthetic_alignment() -> None:
    cfg = load_config(ROOT / "config" / "config.yaml")
    raw = _synthetic_raw()
    checker = NoFutureLeakageChecker(cfg)
    checks = checker.run_all(raw)
    critical = [c for c in checks if c.severity == "critical"]
    assert all(c.passed for c in critical), [c.to_dict() for c in critical if not c.passed]


def test_source_clock_deduplicates_forward_filled_rows() -> None:
    cfg = load_config(ROOT / "config" / "config.yaml")
    raw = _synthetic_raw()
    source = build_source_clock_frame(raw, cfg)
    assert source["liq_feature_time"].duplicated().sum() == 0
    assert (source["liq_feature_age_min"] == 0).all()


def test_rolling_zscore_is_past_only() -> None:
    x = pd.Series(np.arange(30, dtype=float))
    z1 = robust_zscore_past_only(x, window=5, min_periods=3, eps=1e-12)
    x2 = x.copy()
    x2.iloc[-1] = 10_000.0
    z2 = robust_zscore_past_only(x2, window=5, min_periods=3, eps=1e-12)
    # Modifying a future value must not change prior rolling outputs.
    assert np.allclose(z1.iloc[:-1], z2.iloc[:-1], equal_nan=True)


def test_train_validation_test_split_is_chronological() -> None:
    cfg = load_config(ROOT / "config" / "config.yaml")
    raw = _synthetic_raw()
    features = assign_chronological_split(build_feature_frame(raw, cfg), cfg)
    checker = NoFutureLeakageChecker(cfg)
    result = checker.check_time_split(features)
    assert result.passed, result.to_dict()


def test_model_features_and_agent_inputs_exclude_future_labels() -> None:
    cfg = load_config(ROOT / "config" / "config.yaml")
    checker = NoFutureLeakageChecker(cfg)
    assert checker.check_model_features_no_labels(cfg.get("model", "feature_names")).passed
    assert checker.check_agent_inputs(get_agent_input_columns(cfg.get("features", "horizons_min"))).passed
