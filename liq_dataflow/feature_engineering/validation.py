from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from liq_dataflow.feature_engineering.config import ProjectPaths, ValidationConfig
from liq_dataflow.feature_engineering.metadata import DELIVERED_FEATURES


@dataclass(frozen=True)
class ValidationCheck:
    dataset: str
    check: str
    column: str
    severity: str
    status: str
    invalid_count: int
    total_count: int
    invalid_ratio: float
    min_value: float | None
    max_value: float | None
    message: str


class OutputValidator:
    def __init__(self, cfg: ValidationConfig) -> None:
        self.cfg = cfg
        self._rows: list[ValidationCheck] = []

    @property
    def rows(self) -> list[ValidationCheck]:
        return list(self._rows)

    def _append(
        self,
        *,
        dataset: str,
        check: str,
        column: str,
        severity: str,
        status: str,
        invalid_count: int,
        total_count: int,
        message: str,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None:
        ratio = float(invalid_count) / float(total_count) if total_count else 0.0
        self._rows.append(
            ValidationCheck(
                dataset=dataset,
                check=check,
                column=column,
                severity=severity,
                status=status,
                invalid_count=int(invalid_count),
                total_count=int(total_count),
                invalid_ratio=ratio,
                min_value=min_value,
                max_value=max_value,
                message=message,
            )
        )

    def _series(self, df: pd.DataFrame, col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(dtype=float)
        return pd.to_numeric(df[col], errors="coerce")

    def _maybe_numeric(self, df: pd.DataFrame, col: str) -> pd.Series | None:
        if col not in df.columns:
            return None
        return pd.to_numeric(df[col], errors="coerce")

    def _pass_fail(self, invalid_count: int, *, severity: str) -> str:
        if invalid_count <= 0:
            return "PASS"
        return "FAIL" if severity == "error" else "WARN"

    def check_required_columns(self, dataset: str, df: pd.DataFrame, required: Sequence[str]) -> None:
        missing = [c for c in required if c not in df.columns]
        self._append(
            dataset=dataset,
            check="required_columns",
            column=",".join(required),
            severity="error",
            status=self._pass_fail(len(missing), severity="error"),
            invalid_count=len(missing),
            total_count=max(len(required), 1),
            message="missing columns: " + ", ".join(missing) if missing else "all required columns present",
        )

    def check_time_index(self, dataset: str, df: pd.DataFrame) -> None:
        if "time" not in df.columns:
            self.check_required_columns(dataset, df, ["time"])
            return
        time = pd.to_datetime(df["time"], errors="coerce")
        total = len(time)
        nat_count = int(time.isna().sum())
        self._append(
            dataset=dataset,
            check="time_not_null",
            column="time",
            severity="error",
            status=self._pass_fail(nat_count, severity="error"),
            invalid_count=nat_count,
            total_count=total,
            message="time contains NaT values" if nat_count else "time has no NaT",
        )
        valid = time.dropna()
        dup_count = int(valid.duplicated().sum())
        self._append(
            dataset=dataset,
            check="time_unique",
            column="time",
            severity="error",
            status=self._pass_fail(dup_count, severity="error"),
            invalid_count=dup_count,
            total_count=max(len(valid), 1),
            message="duplicate timestamps found" if dup_count else "time is unique",
        )
        decreasing = int((valid.diff().dropna() < pd.Timedelta(0)).sum())
        self._append(
            dataset=dataset,
            check="time_monotonic",
            column="time",
            severity="error",
            status=self._pass_fail(decreasing, severity="error"),
            invalid_count=decreasing,
            total_count=max(len(valid) - 1, 1),
            message="time is not sorted ascending" if decreasing else "time is monotonic ascending",
        )

    def check_nonnegative(self, dataset: str, df: pd.DataFrame, cols: Sequence[str], *, severity: str = "error") -> None:
        tol = float(self.cfg.numeric_tolerance)
        for col in cols:
            if col not in df.columns:
                continue
            s = self._series(df, col)
            bad = (s < -tol) & s.notna()
            self._append(
                dataset=dataset,
                check="nonnegative",
                column=col,
                severity=severity,
                status=self._pass_fail(int(bad.sum()), severity=severity),
                invalid_count=int(bad.sum()),
                total_count=len(s),
                min_value=float(s.min()) if s.notna().any() else None,
                max_value=float(s.max()) if s.notna().any() else None,
                message=f"{col} must be >= 0",
            )

    def check_positive(self, dataset: str, df: pd.DataFrame, cols: Sequence[str], *, severity: str = "error") -> None:
        tol = float(self.cfg.numeric_tolerance)
        for col in cols:
            if col not in df.columns:
                continue
            s = self._series(df, col)
            bad = (s <= tol) & s.notna()
            self._append(
                dataset=dataset,
                check="positive",
                column=col,
                severity=severity,
                status=self._pass_fail(int(bad.sum()), severity=severity),
                invalid_count=int(bad.sum()),
                total_count=len(s),
                min_value=float(s.min()) if s.notna().any() else None,
                max_value=float(s.max()) if s.notna().any() else None,
                message=f"{col} must be > 0",
            )

    def check_between(self, dataset: str, df: pd.DataFrame, col: str, *, lower: float, upper: float, severity: str = "error") -> None:
        if col not in df.columns:
            return
        tol = float(self.cfg.numeric_tolerance)
        s = self._series(df, col)
        bad = ((s < lower - tol) | (s > upper + tol)) & s.notna()
        self._append(
            dataset=dataset,
            check="between",
            column=col,
            severity=severity,
            status=self._pass_fail(int(bad.sum()), severity=severity),
            invalid_count=int(bad.sum()),
            total_count=len(s),
            min_value=float(s.min()) if s.notna().any() else None,
            max_value=float(s.max()) if s.notna().any() else None,
            message=f"{col} must be within [{lower}, {upper}]",
        )

    def check_binary(self, dataset: str, df: pd.DataFrame, col: str, *, severity: str = "error") -> None:
        self.check_allowed_values(dataset, df, col, {0, 1}, severity=severity, check_name="binary")

    def check_allowed_values(
        self,
        dataset: str,
        df: pd.DataFrame,
        col: str,
        allowed: Iterable,
        *,
        severity: str = "error",
        check_name: str = "allowed_values",
    ) -> None:
        if col not in df.columns:
            return
        allowed_set = set(allowed)
        s = df[col]
        bad = (~s.isin(list(allowed_set))) & s.notna()
        self._append(
            dataset=dataset,
            check=check_name,
            column=col,
            severity=severity,
            status=self._pass_fail(int(bad.sum()), severity=severity),
            invalid_count=int(bad.sum()),
            total_count=len(s),
            message=f"{col} must be within {sorted(allowed_set)}",
        )

    def check_integer_range(self, dataset: str, df: pd.DataFrame, col: str, *, lower: int, upper: int, severity: str = "error") -> None:
        if col not in df.columns:
            return
        tol = float(self.cfg.numeric_tolerance)
        s = self._series(df, col)
        integerish = (s - s.round()).abs() <= tol
        in_range = (s >= lower - tol) & (s <= upper + tol)
        bad = (~integerish | ~in_range) & s.notna()
        self._append(
            dataset=dataset,
            check="integer_range",
            column=col,
            severity=severity,
            status=self._pass_fail(int(bad.sum()), severity=severity),
            invalid_count=int(bad.sum()),
            total_count=len(s),
            min_value=float(s.min()) if s.notna().any() else None,
            max_value=float(s.max()) if s.notna().any() else None,
            message=f"{col} must be integer-like within [{lower}, {upper}]",
        )

    def check_identity(self, dataset: str, name: str, lhs: pd.Series | np.ndarray | None, rhs: pd.Series | np.ndarray | None, *, severity: str = "error") -> None:
        tol = float(self.cfg.numeric_tolerance)
        if lhs is None or rhs is None:
            self._append(
                dataset=dataset,
                check=name,
                column="*",
                severity=severity,
                status="FAIL" if severity == "error" else "WARN",
                invalid_count=1,
                total_count=1,
                message="identity inputs are missing",
            )
            return
        lhs_num = pd.Series(pd.to_numeric(lhs, errors="coerce"))
        rhs_num = pd.Series(pd.to_numeric(rhs, errors="coerce"))
        mask = lhs_num.notna() & rhs_num.notna()
        if not mask.any():
            self._append(
                dataset=dataset,
                check=name,
                column="*",
                severity=severity,
                status="WARN" if severity != "error" else "PASS",
                invalid_count=0,
                total_count=0,
                message="identity skipped because both sides are NaN",
            )
            return
        diff = (lhs_num[mask] - rhs_num[mask]).abs()
        bad = diff > tol
        self._append(
            dataset=dataset,
            check=name,
            column="*",
            severity=severity,
            status=self._pass_fail(int(bad.sum()), severity=severity),
            invalid_count=int(bad.sum()),
            total_count=int(mask.sum()),
            min_value=float(diff.min()) if len(diff) else None,
            max_value=float(diff.max()) if len(diff) else None,
            message=f"identity {name} must hold within tolerance {tol}",
        )

    def check_no_inf(self, dataset: str, df: pd.DataFrame, *, cols: Sequence[str] | None = None, severity: str = "error") -> None:
        target_cols = list(cols) if cols is not None else list(df.select_dtypes(include=[np.number]).columns)
        for col in target_cols:
            if col not in df.columns:
                continue
            s = self._series(df, col)
            bad = np.isinf(s.to_numpy(dtype=float, na_value=np.nan))
            self._append(
                dataset=dataset,
                check="finite",
                column=col,
                severity=severity,
                status=self._pass_fail(int(np.nansum(bad)), severity=severity),
                invalid_count=int(np.nansum(bad)),
                total_count=len(s),
                min_value=float(s.min()) if s.notna().any() else None,
                max_value=float(s.max()) if s.notna().any() else None,
                message=f"{col} must not contain ±inf",
            )

    def _validate_liquidation_family(self, dataset: str, df: pd.DataFrame, *, require_liq_active: bool = True) -> None:
        required = [
            "time",
            "price",
            "fll_cwt_kf",
            "fsl_cwt_kf",
            "risk_priority_number",
            "diff_ls_cwt_kf",
            "total_ls_cwt_kf",
            "diff_dom_ls_cwt_kf",
        ]
        if require_liq_active:
            required.insert(2, "liq_active_raw")
        self.check_required_columns(dataset, df, required)
        self.check_time_index(dataset, df)
        self.check_positive(dataset, df, ["price"])
        self.check_nonnegative(dataset, df, ["fll_cwt_kf", "fsl_cwt_kf", "total_ls_cwt_kf"])
        self.check_between(dataset, df, "risk_priority_number", lower=0.0, upper=1.0)
        self.check_between(dataset, df, "diff_dom_ls_cwt_kf", lower=-1.0, upper=1.0)
        fll = self._maybe_numeric(df, "fll_cwt_kf")
        fsl = self._maybe_numeric(df, "fsl_cwt_kf")
        total = self._maybe_numeric(df, "total_ls_cwt_kf")
        diff = self._maybe_numeric(df, "diff_ls_cwt_kf")
        sdom = self._maybe_numeric(df, "diff_dom_ls_cwt_kf")
        rpn = self._maybe_numeric(df, "risk_priority_number")
        self.check_identity(dataset, "total_eq_fll_plus_fsl", total, None if fll is None or fsl is None else fll + fsl)
        self.check_identity(dataset, "diff_eq_fll_minus_fsl", diff, None if fll is None or fsl is None else fll - fsl)
        self.check_identity(dataset, "sdom_eq_2rpn_minus_1", sdom, None if rpn is None else 2.0 * rpn - 1.0)
        self.check_no_inf(dataset, df)

    def check_nan_ratio(self, dataset: str, df: pd.DataFrame, cols: Sequence[str], *, max_ratio: float, severity: str = "warning") -> None:
        for col in cols:
            if col not in df.columns:
                continue
            s = df[col]
            nan_count = int(s.isna().sum())
            total = max(len(s), 1)
            ratio = nan_count / total
            status = "PASS" if ratio <= max_ratio else ("FAIL" if severity == "error" else "WARN")
            self._append(
                dataset=dataset,
                check="nan_ratio",
                column=col,
                severity=severity,
                status=status,
                invalid_count=nan_count if ratio > max_ratio else 0,
                total_count=total,
                message=f"{col} NaN ratio {ratio:.4%} should be <= {max_ratio:.4%}",
            )

    def validate_clean(self, df: pd.DataFrame) -> None:
        ds = "clean"
        required = [
            "time",
            "price",
            "fll_normal",
            "fsl_normal",
            "total_ls_normal",
            "diff_ls_normal",
            "lld_normal",
            "diff_dom_ls_normal",
            "liq_active_raw",
        ]
        self.check_required_columns(ds, df, required)
        self.check_time_index(ds, df)
        self.check_positive(ds, df, ["price"])
        self.check_nonnegative(ds, df, ["fll_normal", "fsl_normal", "total_ls_normal"])
        self.check_between(ds, df, "lld_normal", lower=0.0, upper=1.0)
        self.check_between(ds, df, "diff_dom_ls_normal", lower=-1.0, upper=1.0)
        self.check_binary(ds, df, "liq_active_raw")
        fll = self._maybe_numeric(df, "fll_normal")
        fsl = self._maybe_numeric(df, "fsl_normal")
        total = self._maybe_numeric(df, "total_ls_normal")
        diff = self._maybe_numeric(df, "diff_ls_normal")
        liq_active = self._maybe_numeric(df, "liq_active_raw")
        self.check_identity(ds, "total_eq_fll_plus_fsl", total, None if fll is None or fsl is None else fll + fsl)
        self.check_identity(ds, "diff_eq_fll_minus_fsl", diff, None if fll is None or fsl is None else fll - fsl)
        liq_flag = None if total is None else (total > self.cfg.numeric_tolerance).astype(float)
        self.check_identity(ds, "liq_active_matches_total", liq_active, liq_flag)
        self.check_no_inf(ds, df)

    def validate_canonical(self, df: pd.DataFrame) -> None:
        self._validate_liquidation_family("canonical", df, require_liq_active=True)

    def validate_dominance(self, df: pd.DataFrame) -> None:
        ds = "dominance"
        self._validate_liquidation_family(ds, df, require_liq_active=True)
        self.check_integer_range(ds, df, "bin_index", lower=0, upper=8)
        if "dominance" in df.columns:
            self.check_allowed_values(ds, df, "dominance", {-1, 0, 1})
        if "is_rpn_extreme" in df.columns:
            self.check_allowed_values(ds, df, "is_rpn_extreme", {-1, 0, 1}, check_name="ternary")
        for col in ["is_keep", "is_strengthen"]:
            if col in df.columns:
                self.check_binary(ds, df, col)
        for col in ["dominance", "dominance_last", "dominance_prev", "hit_ceiling_bottom", "reverse_ceiling_bottom"]:
            if col in df.columns:
                self.check_allowed_values(ds, df, col, {-1, 0, 1})
        self.check_no_inf(ds, df)

    def validate_contract(self, df: pd.DataFrame) -> None:
        ds = "contract"
        self._validate_liquidation_family(ds, df, require_liq_active=False)
        self.check_integer_range(ds, df, "bin_index", lower=0, upper=8)
        if "dominance" in df.columns:
            self.check_allowed_values(ds, df, "dominance", {-1, 0, 1})
        self.check_no_inf(ds, df)

    def validate_model(self, df: pd.DataFrame) -> None:
        ds = "model"
        required = ["time", "price", "risk_priority_number", "bin_index", "dominance", "z_logTotalP", "z_sdom", "z_fll_cwt_kf", "z_fsl_cwt_kf"]
        self.check_required_columns(ds, df, required)
        self.check_time_index(ds, df)
        self.check_positive(ds, df, ["price"])
        self.check_between(ds, df, "risk_priority_number", lower=0.0, upper=1.0)
        self.check_integer_range(ds, df, "bin_index", lower=0, upper=8)
        if "dominance" in df.columns:
            self.check_allowed_values(ds, df, "dominance", {-1, 0, 1})
        self.check_no_inf(ds, df)
        self.check_nan_ratio(ds, df, ["z_logTotalP", "z_sdom", "z_fll_cwt_kf", "z_fsl_cwt_kf"], max_ratio=self.cfg.max_model_nan_ratio)

    def validate_final_features(self, df: pd.DataFrame, *, final_feature_columns: Sequence[str]) -> None:
        ds = "liq_dataflow_features"
        required = ["time", "price", *final_feature_columns]
        self.check_required_columns(ds, df, required)
        self._validate_liquidation_family(ds, df, require_liq_active=False)
        self.check_integer_range(ds, df, "bin_index", lower=0, upper=8)
        if "dominance" in df.columns:
            self.check_allowed_values(ds, df, "dominance", {-1, 0, 1})
        z_cols = [c for c in final_feature_columns if c.startswith("z_")]
        self.check_nan_ratio(ds, df, z_cols, max_ratio=self.cfg.max_model_nan_ratio)

    def validate_feature_store(self, df: pd.DataFrame) -> None:
        ds = "feature_store"
        self.check_time_index(ds, df)
        if "price" in df.columns:
            self.check_positive(ds, df, ["price"])
        for col in ["fll_cwt_kf", "fsl_cwt_kf", "total_ls_cwt_kf"]:
            if col in df.columns:
                self.check_nonnegative(ds, df, [col])
        if {"fll_cwt_kf", "fsl_cwt_kf", "total_ls_cwt_kf"}.issubset(df.columns):
            self.check_identity(ds, "total_eq_fll_plus_fsl", self._maybe_numeric(df, "total_ls_cwt_kf"), self._maybe_numeric(df, "fll_cwt_kf") + self._maybe_numeric(df, "fsl_cwt_kf"))
        if "risk_priority_number" in df.columns:
            self.check_between(ds, df, "risk_priority_number", lower=0.0, upper=1.0)
        if "RPN" in df.columns:
            self.check_between(ds, df, "RPN", lower=0.0, upper=1.0)
        if "bin_index" in df.columns:
            self.check_integer_range(ds, df, "bin_index", lower=0, upper=8)
        if "dominance" in df.columns:
            self.check_allowed_values(ds, df, "dominance", {-1, 0, 1})
        self.check_no_inf(ds, df)

    def validate_cache_csv(self, dataset: str, path: Path) -> None:
        exists = path.exists() and path.stat().st_size > 0
        self._append(
            dataset=dataset,
            check="cache_exists",
            column=path.name,
            severity="error",
            status="PASS" if exists else "FAIL",
            invalid_count=0 if exists else 1,
            total_count=1,
            message=f"cache file {'present' if exists else 'missing'}: {path}",
        )
        if not exists:
            return
        df = pd.read_csv(path)
        self.check_time_index(dataset, df)
        numeric_cols = [c for c in df.columns if c != "time"]
        self.check_nonnegative(dataset, df, numeric_cols)
        self.check_no_inf(dataset, df, cols=numeric_cols)

    def validate_artifacts(self, paths: ProjectPaths, *, build_visualizations: bool, expected_feature_pages: int | None = None) -> None:
        required = [
            paths.clean_csv,
            paths.canonical_csv,
            paths.bin_stage_csv,
            paths.bin_stats_csv,
            paths.legacy_bin_stats_csv,
            paths.dominance_csv,
            paths.final_features_csv,
            paths.feature_store_csv,
        ]
        if build_visualizations:
            required.extend(
                [
                    paths.feature_overview_html,
                    paths.feature_catalog_csv,
                    paths.dominant_html,
                    paths.dominant_png,
                    paths.features_html,
                    paths.features_png,
                    paths.plotly_bundle,
                ]
            )
        for path in required:
            exists = path.exists() and path.stat().st_size > 0
            self._append(
                dataset="artifacts",
                check="file_exists",
                column=path.name,
                severity="error",
                status="PASS" if exists else "FAIL",
                invalid_count=0 if exists else 1,
                total_count=1,
                message=f"artifact {'present' if exists else 'missing'}: {path}",
            )
        if build_visualizations:
            page_count = len(list(paths.feature_pages_dir.glob("*.html")))
            target = int(expected_feature_pages if expected_feature_pages is not None else len(DELIVERED_FEATURES))
            self._append(
                dataset="artifacts",
                check="feature_pages_count",
                column="feature_pages/*.html",
                severity="error",
                status="PASS" if page_count >= target else "FAIL",
                invalid_count=0 if page_count >= target else target - page_count,
                total_count=target,
                message=f"feature pages present: {page_count}, expected at least {target}",
            )

    def to_frame(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame(columns=list(ValidationCheck.__dataclass_fields__.keys()))
        return pd.DataFrame([asdict(row) for row in self._rows])

    def write_reports(self, paths: ProjectPaths) -> pd.DataFrame:
        df = self.to_frame().sort_values(["status", "severity", "dataset", "check", "column"]).reset_index(drop=True)
        paths.validation_report_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(paths.validation_report_csv, index=False)
        with paths.validation_report_json.open("w", encoding="utf-8") as fh:
            json.dump(df.to_dict(orient="records"), fh, ensure_ascii=False, indent=2)

        pass_count = int((df["status"] == "PASS").sum()) if not df.empty else 0
        warn_count = int((df["status"] == "WARN").sum()) if not df.empty else 0
        fail_count = int((df["status"] == "FAIL").sum()) if not df.empty else 0
        lines = [
            "# 输出校验报告",
            "",
            f"- PASS: {pass_count}",
            f"- WARN: {warn_count}",
            f"- FAIL: {fail_count}",
            "",
            "## 失败或告警项",
            "",
        ]
        if df.empty:
            lines.append("当前没有校验记录。")
        else:
            focus = df[df["status"].isin(["FAIL", "WARN"])]
            if focus.empty:
                lines.append("所有阻塞性校验均通过。")
            else:
                for row in focus.itertuples(index=False):
                    lines.append(
                        f"- [{row.status}] {row.dataset} / {row.check} / {row.column}: {row.message} "
                        f"(invalid={row.invalid_count}, ratio={row.invalid_ratio:.4%})"
                    )
        paths.validation_report_md.write_text("\n".join(lines), encoding="utf-8")
        return df

    def assert_valid(self) -> None:
        if not self.cfg.raise_on_error:
            return
        failures = [row for row in self._rows if row.status == "FAIL" and row.severity == "error"]
        if failures:
            head = failures[:10]
            summary = "; ".join(f"{x.dataset}:{x.check}:{x.column}" for x in head)
            raise ValueError(f"Output validation failed with {len(failures)} blocking checks. First failures: {summary}")
