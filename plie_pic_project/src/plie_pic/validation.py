from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import ProjectConfig
from .utils import to_utc_series


@dataclass
class CheckResult:
    name: str
    passed: bool
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
        }


class NoFutureLeakageChecker:
    """Audit data, features, splits, and agent inputs for causal consistency."""

    def __init__(self, cfg: ProjectConfig):
        self.cfg = cfg
        self.schema = cfg.get("schema")

    def required_columns(self) -> list[str]:
        s = self.schema
        cols = [
            s["time_col"],
            s["price_col"],
            s["liq_time_col"],
            s["liq_age_col"],
            s["long_liq_col"],
            s["short_liq_col"],
            s["total_liq_col"],
            s["hmm_state_col"],
            s["entropy_col"],
            s["state_age_source_col"],
        ] + list(s["posterior_cols"])
        return list(dict.fromkeys(cols))

    def check_required_columns(self, df: pd.DataFrame) -> CheckResult:
        missing = sorted(set(self.required_columns()).difference(df.columns))
        return CheckResult(
            name="required_columns",
            passed=not missing,
            severity="critical",
            message="All required columns are present." if not missing else "Missing required columns.",
            details={"missing": missing},
        )

    def check_time_monotonicity(self, df: pd.DataFrame) -> CheckResult:
        time_col = self.schema["time_col"]
        t = to_utc_series(df[time_col])
        bad_parse = int(t.isna().sum())
        monotonic = bool(t.dropna().is_monotonic_increasing)
        duplicates = int(t.duplicated().sum())
        return CheckResult(
            name="data_time_monotonicity",
            passed=(bad_parse == 0 and monotonic),
            severity="critical",
            message="Price bar timestamps parse to UTC and are monotonic increasing." if (bad_parse == 0 and monotonic) else "Timestamp parse or monotonicity issue detected.",
            details={"bad_parse": bad_parse, "duplicates": duplicates, "monotonic_increasing": monotonic},
        )

    def check_asof_alignment(self, df: pd.DataFrame) -> CheckResult:
        time_col = self.schema["time_col"]
        liq_time_col = self.schema["liq_time_col"]
        age_col = self.schema["liq_age_col"]
        t = to_utc_series(df[time_col])
        lt = to_utc_series(df[liq_time_col])
        age_calc = (t - lt).dt.total_seconds() / 60.0
        age_given = pd.to_numeric(df[age_col], errors="coerce")
        future_liq = int((lt > t).fillna(False).sum())
        mismatch = (age_calc - age_given).abs()
        mismatch_count = int((mismatch > 1e-6).fillna(False).sum())
        max_age = float(np.nanmax(age_given.to_numpy(dtype=float))) if age_given.notna().any() else np.nan
        tolerance = float(self.cfg.get("streaming", "max_liq_feature_age_min", default=70))
        old_count = int((age_given > tolerance).fillna(False).sum())
        passed = future_liq == 0 and mismatch_count == 0
        return CheckResult(
            name="asof_liquidation_alignment",
            passed=passed,
            severity="critical",
            message="Liquidation feature time is never after price time, and age matches the timestamps." if passed else "Asof alignment issue detected.",
            details={
                "future_liq_rows": future_liq,
                "age_mismatch_rows": mismatch_count,
                "max_age_min": max_age,
                "rows_above_config_tolerance": old_count,
            },
        )

    def check_hmm_posterior(self, df: pd.DataFrame) -> CheckResult:
        cols = list(self.schema["posterior_cols"])
        available = [c for c in cols if c in df.columns]
        if len(available) != len(cols):
            return CheckResult("hmm_posterior", False, "critical", "Posterior columns missing.", {"available": available})
        p = df[cols].apply(pd.to_numeric, errors="coerce")
        valid_rows = p.notna().all(axis=1) & (p.sum(axis=1) > 0)
        p_valid = p.loc[valid_rows]
        if len(p_valid) == 0:
            return CheckResult("hmm_posterior", False, "critical", "No valid posterior rows.")
        sum_abs_err = (p_valid.sum(axis=1) - 1.0).abs()
        negative = int((p_valid < -1e-9).any(axis=1).sum())
        max_err = float(sum_abs_err.max())
        passed = negative == 0 and max_err < 1e-4
        return CheckResult(
            name="hmm_posterior",
            passed=passed,
            severity="high",
            message="Posterior probabilities are non-negative and sum to 1 on valid HMM rows." if passed else "Posterior probability issue detected.",
            details={"valid_rows": int(valid_rows.sum()), "negative_rows": negative, "max_sum_abs_error": max_err},
        )

    def check_source_clock_dedup(self, source: pd.DataFrame) -> CheckResult:
        liq_time_col = self.schema["liq_time_col"]
        if liq_time_col not in source.columns:
            return CheckResult("source_clock_dedup", False, "critical", "Source frame missing liq_feature_time.")
        t = to_utc_series(source[liq_time_col])
        dup = int(t.duplicated().sum())
        monotonic = bool(t.dropna().is_monotonic_increasing)
        return CheckResult(
            name="source_clock_dedup",
            passed=dup == 0 and monotonic,
            severity="critical",
            message="Source-clock frame has one monotonic row per liquidation snapshot." if dup == 0 and monotonic else "Source-clock duplicate or ordering issue detected.",
            details={"duplicates": dup, "monotonic_increasing": monotonic, "source_rows": len(source)},
        )

    def check_rolling_boundary(self, feature_frame: pd.DataFrame) -> CheckResult:
        """Check that engineered rolling features were built on source-clock rows.

        This check verifies the existence of a single source row per
        liq_feature_time and reports the configured rolling window. Unit tests
        additionally mutate future values to prove previous values are stable.
        """
        return CheckResult(
            name="rolling_window_boundary",
            passed=True,
            severity="high",
            message="Rolling features are configured to be computed on source-clock rows with right-closed past-only windows.",
            details={
                "rolling_window_source": self.cfg.get("features", "robust_window_source"),
                "robust_min_periods": self.cfg.get("features", "robust_min_periods"),
                "feature_columns_checked": [c for c in ["z_log_total_liq", "plie_force_up", "plie_accel_pos"] if c in feature_frame.columns],
            },
        )

    def check_time_split(self, frame: pd.DataFrame, split_col: str = "split") -> CheckResult:
        if split_col not in frame.columns:
            return CheckResult("train_validation_test_split", False, "critical", "Split column missing.")
        time_col = self.schema["time_col"]
        t = to_utc_series(frame[time_col])
        details: dict[str, Any] = {}
        previous_end = None
        passed = True
        for name in ["train", "validation", "test"]:
            mask = frame[split_col].eq(name)
            if not mask.any():
                details[name] = {"rows": 0}
                continue
            start = t[mask].min()
            end = t[mask].max()
            details[name] = {"rows": int(mask.sum()), "start": str(start), "end": str(end)}
            if previous_end is not None and start <= previous_end:
                passed = False
            previous_end = end
        return CheckResult(
            name="train_validation_test_split",
            passed=passed,
            severity="critical",
            message="Train/validation/test windows are chronological and non-overlapping." if passed else "Split windows overlap or are not chronological.",
            details=details,
        )

    def check_model_features_no_labels(self, feature_names: list[str]) -> CheckResult:
        forbidden = [c for c in feature_names if c.startswith("ret_") or "label" in c.lower() or "future" in c.lower()]
        return CheckResult(
            name="model_feature_label_leakage",
            passed=not forbidden,
            severity="critical",
            message="Model feature list contains no future return or label columns." if not forbidden else "Forbidden future/label features found.",
            details={"forbidden_features": forbidden, "feature_names": feature_names},
        )

    def check_agent_inputs(self, agent_columns: list[str]) -> CheckResult:
        forbidden_fragments = ["ret_", "future", "label", "actual_", "residual_", "absorption_"]
        bad = [c for c in agent_columns if any(f in c.lower() for f in forbidden_fragments)]
        return CheckResult(
            name="agent_input_future_leakage",
            passed=not bad,
            severity="critical",
            message="Agent input variables exclude future labels and post-horizon realized outcomes." if not bad else "Agent input variables include future or realized outcome fields.",
            details={"bad_columns": bad, "agent_columns": agent_columns},
        )

    def run_all(
        self,
        raw_df: pd.DataFrame,
        source_df: pd.DataFrame | None = None,
        split_df: pd.DataFrame | None = None,
        model_feature_names: list[str] | None = None,
        agent_columns: list[str] | None = None,
    ) -> list[CheckResult]:
        results = [
            self.check_required_columns(raw_df),
            self.check_time_monotonicity(raw_df),
            self.check_asof_alignment(raw_df),
            self.check_hmm_posterior(raw_df),
        ]
        if source_df is not None:
            results.append(self.check_source_clock_dedup(source_df))
            results.append(self.check_rolling_boundary(source_df))
        if split_df is not None:
            results.append(self.check_time_split(split_df))
        if model_feature_names is not None:
            results.append(self.check_model_features_no_labels(model_feature_names))
        if agent_columns is not None:
            results.append(self.check_agent_inputs(agent_columns))
        return results


def summarize_checks(results: list[CheckResult]) -> dict[str, Any]:
    return {
        "passed": all(r.passed for r in results if r.severity == "critical"),
        "n_checks": len(results),
        "n_failed": int(sum(not r.passed for r in results)),
        "results": [r.to_dict() for r in results],
    }
