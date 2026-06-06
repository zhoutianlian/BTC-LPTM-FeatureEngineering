from __future__ import annotations

import logging
import shutil
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from liq_dataflow.feature_engineering.binning import cal_is_extreme_from_bins, get_bin_stats, point_in_time_quantile_binning
from liq_dataflow.feature_engineering.config import DATA_DIR, FeatureEngineeringConfig, ProjectPaths, load_feature_engineering_config
from liq_dataflow.feature_engineering.data_source import CsvInputSource
from liq_dataflow.feature_engineering.dominance import build_dominance_features
from liq_dataflow.feature_engineering.logging_utils import append_run_history, log_kv, log_message, log_section, setup_run_logger, utc_run_id, write_run_summary
from liq_dataflow.feature_engineering.model_features import LiquidationModelFeatureConfig, build_liquidation_model_features
from liq_dataflow.feature_engineering.preprocess import CLEAN_REQUIRED_COLUMNS, RAW_REQUIRED_COLUMNS, normalize_input_columns, preprocess_liquidation_data, safe_rpn, safe_sdom
from liq_dataflow.feature_engineering.transforms import CacheStore, TrailingWaveletKalmanDetrender, classify_delta_corr, ema
from liq_dataflow.feature_engineering.validation import OutputValidator
from liq_dataflow.visualizer.generic_feature_dashboard import generate_feature_portal
from liq_dataflow.visualizer.specialized_dashboard import generate_specialized_dashboards


@dataclass
class FeatureEngineeringResult:
    input_data: pd.DataFrame
    clean: pd.DataFrame
    dominance: pd.DataFrame
    liq_features: pd.DataFrame
    feature_store: pd.DataFrame
    validation_report: pd.DataFrame
    paths: ProjectPaths


class FeatureEngineeringProject:
    """Pure feature-engineering runtime for BTC liquidation features.

    Standard workflow:
    1) read a CSV produced by the sibling raw-data download project;
    2) preprocess + feature engineering;
    3) save artifacts;
    4) generate visualization pages.
    """

    def __init__(
        self,
        root_dir: Path | None = None,
        *,
        config: FeatureEngineeringConfig | None = None,
        df_input: pd.DataFrame | None = None,
        logger=None,
    ) -> None:
        self.config = config or load_feature_engineering_config()
        self.paths = ProjectPaths.from_config(root_dir=root_dir or DATA_DIR.parent, cfg=self.config)
        self.paths.ensure_dirs()
        self.run_id = utc_run_id()
        run_logger, self.run_context = setup_run_logger(logs_dir=self.paths.logs_dir, run_id=self.run_id)
        self.logger = run_logger
        self.external_logger = logger
        self.df_input = df_input
        self._stage_timings: dict[str, float] = {}
        self._summary: dict[str, Any] = {"run_id": self.run_id, "root_dir": str(self.paths.root_dir)}
        self._migrate_legacy_caches()
        self._log(f"Initialized feature engineering run_id={self.run_id}")
        self._log(f"Logs will be written to {self.run_context.run_dir}")

    def _log(self, message: str, level: int = logging.INFO) -> None:
        log_message(self.logger, message, level=level)
        if self.external_logger is not None and self.external_logger is not self.logger:
            log_message(self.external_logger, message, level=level)

    def _record_stage_timing(self, stage: str, start_ts: float) -> None:
        self._stage_timings[stage] = round(time.perf_counter() - start_ts, 4)
        self._summary[f"timing_{stage}_sec"] = self._stage_timings[stage]
        self._log(f"Stage completed | {stage} | elapsed_sec={self._stage_timings[stage]:.4f}")

    def _log_frame_overview(self, name: str, df: pd.DataFrame, *, extra_cols: list[str] | None = None) -> None:
        if df is None:
            self._log(f"{name}: dataframe is None", level=logging.WARNING)
            return
        payload: dict[str, Any] = {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "col_names": ", ".join(df.columns[:12]) + (" ..." if len(df.columns) > 12 else ""),
        }
        if "time" in df.columns and not df.empty:
            t = pd.to_datetime(df["time"], errors="coerce")
            payload["start_time"] = None if t.dropna().empty else str(t.min())
            payload["latest_time"] = None if t.dropna().empty else str(t.max())
            payload["missing_time"] = int(t.isna().sum())
        for col in extra_cols or []:
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")
                payload[f"{col}_min"] = None if s.dropna().empty else float(s.min())
                payload[f"{col}_max"] = None if s.dropna().empty else float(s.max())
                payload[f"{col}_na"] = int(s.isna().sum())
        log_kv(self.logger, f"Dataframe summary | {name}", payload)

    def _migrate_legacy_caches(self) -> None:
        legacy_root = self.paths.root_dir / "detrend_model" / "data_hub"
        if not legacy_root.exists():
            return
        for legacy_name, new_path in {"fll_cwt.csv": self.paths.fll_cache_csv, "fsl_cwt.csv": self.paths.fsl_cache_csv}.items():
            old_path = legacy_root / legacy_name
            if old_path.exists() and not new_path.exists():
                new_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(old_path, new_path)
                self._log(f"Migrated legacy cache {old_path} -> {new_path}")

    def _cleanup_legacy_visualization_outputs(self) -> None:
        if self.paths.report_dir == self.paths.features_dir or not self.paths.features_dir.exists():
            return

        generated_files = [
            self.paths.features_dir / self.config.visualization.overview_filename,
            self.paths.features_dir / self.config.visualization.catalog_filename,
            self.paths.features_dir / "plotly.min.js",
        ]
        for pattern in ["rpn_dominance-*.html", "rpn_dominance-*.png", "rpn_features-*.html", "rpn_features-*.png"]:
            generated_files.extend(self.paths.features_dir.glob(pattern))

        removed = 0
        for path in generated_files:
            if path.is_file():
                path.unlink()
                removed += 1

        legacy_pages_dir = self.paths.features_dir / self.config.data.feature_pages_subdir
        if legacy_pages_dir.exists():
            shutil.rmtree(legacy_pages_dir)
            removed += 1

        if removed:
            self._log(f"Removed {removed} legacy visualization artifact(s) from {self.paths.features_dir}")

    def _cleanup_legacy_feature_exports(self) -> None:
        removed = 0
        for filename in ["liq_dataflow_features.csv", "fhmv_liq_features.csv"]:
            path = self.paths.features_dir / filename
            if path.is_file():
                path.unlink()
                removed += 1
        old_nested = self.paths.features_dir / "features_" / "liq_dataflow.csv"
        if old_nested.is_file():
            old_nested.unlink()
            removed += 1
        old_nested_dir = old_nested.parent
        if old_nested_dir.exists() and not any(old_nested_dir.iterdir()):
            old_nested_dir.rmdir()
        if removed:
            self._log(f"Removed {removed} legacy split feature export(s) from {self.paths.features_dir}")

    def load_input_data(self, *, input_csv: Path | None = None) -> pd.DataFrame:
        log_section(self.logger, "INPUT")
        if self.df_input is not None:
            df = self.df_input.copy()
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"], errors="coerce")
            self._summary["input_source"] = "in_memory_dataframe"
            self._log("Using dataframe provided directly by caller.")
            self._log_frame_overview("input_data::in_memory", df)
            return df

        source = CsvInputSource(cfg=self.config, root_dir=self.paths.root_dir, logger=self.logger)
        df, path = source.load(override_path=input_csv)
        self._summary["input_source"] = str(path)
        self._log(f"Loaded input CSV from {path}")
        self._log_frame_overview("input_data::csv", df, extra_cols=["price", "futures_long_liquidations", "futures_short_liquidations", "fll_normal", "fsl_normal"])
        return df

    def preprocess(self, df_input: pd.DataFrame) -> pd.DataFrame:
        log_section(self.logger, "PREPROCESS")
        normalized = normalize_input_columns(df_input, columns=self.config.columns)
        columns = set(normalized.columns)
        input_type = "raw_hourly" if RAW_REQUIRED_COLUMNS.issubset(columns) else "clean_frame" if CLEAN_REQUIRED_COLUMNS.issubset(columns) else "unknown"
        self._log(f"Preprocessing liquidation input frame | detected_input_type={input_type}")
        clean = preprocess_liquidation_data(
            normalized,
            start_time=self.config.project.start_time,
            outlier_iqr_window_days=self.config.preprocess.outlier_iqr_window_days,
            columns=self.config.columns,
        )
        clean.to_csv(self.paths.clean_csv, index=False)
        self._log(f"Saved clean frame to {self.paths.clean_csv}")
        self._log_frame_overview("clean", clean, extra_cols=["price", "fll_normal", "fsl_normal", "total_ls_normal"])
        self._summary["clean_rows"] = int(len(clean))
        self._summary["clean_latest_time"] = None if clean.empty else str(pd.to_datetime(clean["time"]).max())
        return clean

    def _canonical_detrend(self, df_clean: pd.DataFrame, *, input_col: str, output_col: str, cache_path: Path) -> pd.DataFrame:
        latest_cache = CacheStore(cache_path).latest_time()
        self._log(f"Smoothing {input_col} -> {output_col} | cache={cache_path} | latest_cached_time={latest_cache}")
        detrender = TrailingWaveletKalmanDetrender(cfg=self.config.smoothing, cache=CacheStore(cache_path))
        out = detrender.transform(df_clean[["time", input_col]].copy(), input_col=input_col, output_col=output_col, clip_lower=0.0)
        self._log_frame_overview(output_col, out, extra_cols=[output_col])
        return out

    def build_canonical_liquidation_family(self, df_clean: pd.DataFrame) -> pd.DataFrame:
        log_section(self.logger, "CANONICAL FAMILY")
        df_fll = self._canonical_detrend(df_clean, input_col="fll_normal", output_col="fll_cwt_kf", cache_path=self.paths.fll_cache_csv)
        df_fsl = self._canonical_detrend(df_clean, input_col="fsl_normal", output_col="fsl_cwt_kf", cache_path=self.paths.fsl_cache_csv)

        out = df_clean[["time", "price", "liq_active_raw"]].copy()
        out = out.merge(df_fll, on="time", how="inner").merge(df_fsl, on="time", how="inner")
        out["fll_cwt_kf"] = pd.to_numeric(out["fll_cwt_kf"], errors="coerce").astype(float).clip(lower=0.0)
        out["fsl_cwt_kf"] = pd.to_numeric(out["fsl_cwt_kf"], errors="coerce").astype(float).clip(lower=0.0)
        out["total_ls_cwt_kf"] = out["fll_cwt_kf"] + out["fsl_cwt_kf"]
        out["diff_ls_cwt_kf"] = out["fll_cwt_kf"] - out["fsl_cwt_kf"]
        out["diff_dom_ls_cwt_kf"] = safe_sdom(out["diff_ls_cwt_kf"], out["total_ls_cwt_kf"])
        out["risk_priority_number"] = safe_rpn(out["fll_cwt_kf"], out["fsl_cwt_kf"])
        out["diff_ls_smooth"] = ema(out["diff_ls_cwt_kf"], span=16)
        out["lld_cwt_kf_smooth"] = ema(out["risk_priority_number"], span=12)
        out = classify_delta_corr(out, "fll_cwt_kf", "fsl_cwt_kf")

        keep_cols = [
            "time", "price", "liq_active_raw", "fll_cwt_kf", "fsl_cwt_kf", "risk_priority_number", "lld_cwt_kf_smooth",
            "diff_ls_cwt_kf", "diff_ls_smooth", "total_ls_cwt_kf", "diff_dom_ls_cwt_kf", "corr_case", "delta_fll", "delta_fsl",
        ]
        out = out[keep_cols].copy()
        out.to_csv(self.paths.canonical_csv, index=False)
        self._log(f"Saved canonical liquidation family to {self.paths.canonical_csv}")
        self._log_frame_overview("canonical", out, extra_cols=["fll_cwt_kf", "fsl_cwt_kf", "total_ls_cwt_kf", "risk_priority_number"])
        self._summary["canonical_rows"] = int(len(out))
        return out

    def build_bin_features(self, df_canonical: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        log_section(self.logger, "BINNING")
        cfg = self.config.binning
        df_bins = point_in_time_quantile_binning(df_canonical, n_bins=cfg.n_bins, col="risk_priority_number", min_history=cfg.min_history_bars, refit_every=cfg.refit_every_bars, neutral_bin=cfg.neutral_bin)
        df_bins = cal_is_extreme_from_bins(df_bins, bin_col="bin_index", col_new="is_rpn_extreme", window_size=cfg.extreme_window_bars)
        df_stats = get_bin_stats(df_bins, value_col="risk_priority_number", bin_col="bin_index")
        df_bins.to_csv(self.paths.bin_stage_csv, index=False)
        df_stats.to_csv(self.paths.bin_stats_csv, index=False)
        if self.config.data.legacy_bin_stats_filename:
            df_stats.to_csv(self.paths.legacy_bin_stats_csv, index=False)
        bin_counts = df_bins["bin_index"].value_counts(dropna=False).sort_index().to_dict() if "bin_index" in df_bins else {}
        log_kv(self.logger, "Binning summary", {"latest_bin": None if df_bins.empty else int(df_bins["bin_index"].iloc[-1]), "bin_counts": bin_counts})
        return df_bins, df_stats

    def _contract_feature_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        keep = ["time", "price", "fll_cwt_kf", "fsl_cwt_kf", "diff_ls_cwt_kf", "total_ls_cwt_kf", "risk_priority_number", "bin_index", "dominance", "diff_dom_ls_cwt_kf"]
        return df[keep].copy()

    def export_liq_dataflow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        log_section(self.logger, "FINAL FEATURES")
        self._cleanup_legacy_feature_exports()
        base = self._contract_feature_frame(df)
        cfg = LiquidationModelFeatureConfig(**asdict(self.config.model_features))
        model_df = build_liquidation_model_features(base, cfg=cfg)
        model_cols = [c for c in model_df.columns if c not in base.columns and c not in {"time", "price"}]
        merged = pd.merge(base, model_df[["time", "price", *model_cols]], on=["time", "price"], how="left")
        final_cols = ["time", "price", *self.config.output_features.final_feature_columns]
        missing = [c for c in final_cols if c not in merged.columns]
        if missing:
            raise ValueError(f"Missing final liq_dataflow feature columns: {missing}")
        out = merged[final_cols].copy()
        out.to_csv(self.paths.final_features_csv, index=False)
        self._log(f"Saved merged liq_dataflow feature export to {self.paths.final_features_csv}")
        delivered_cols = [c for c in self.config.output_features.final_feature_columns if c in out.columns]
        latest_values = {col: (None if out[col].dropna().empty else out[col].dropna().iloc[-1]) for col in delivered_cols}
        nan_ratio = {f"{col}_nan_pct": round(float(out[col].isna().mean() * 100.0), 4) for col in delivered_cols if col.startswith("z_")}
        log_kv(self.logger, "Final feature summary", {**latest_values, **nan_ratio})
        self._summary["final_feature_rows"] = int(len(out))
        return out

    def build_feature_store(self, df_clean: pd.DataFrame, df_dominance: pd.DataFrame, df_features: pd.DataFrame) -> pd.DataFrame:
        merged = pd.merge(df_clean, df_dominance, on=["time", "price", "liq_active_raw"], how="inner")
        extra_cols = [
            c
            for c in df_features.columns
            if c not in {"time", "price"} and c not in merged.columns
        ]
        if extra_cols:
            merged = pd.merge(merged, df_features[["time", "price", *extra_cols]], on=["time", "price"], how="left")
        merged = merged.sort_values("time").drop_duplicates(subset=["time"], keep="last")
        merged.to_csv(self.paths.feature_store_csv, index=False)
        self._log(f"Saved feature store to {self.paths.feature_store_csv}")
        self._log_frame_overview("feature_store", merged)
        self._summary["feature_store_rows"] = int(len(merged))
        self._summary["feature_store_latest_time"] = None if merged.empty else str(pd.to_datetime(merged["time"]).max())
        return merged

    def build_visualizations(self, *, dominance: pd.DataFrame, feature_store: pd.DataFrame) -> pd.DataFrame:
        log_section(self.logger, "VISUALIZATION")
        self._cleanup_legacy_visualization_outputs()
        self._log("Generating specialized dashboards.")
        generate_specialized_dashboards(
            dominance,
            output_dir=self.paths.report_dir,
            dominant_html_name=self.paths.dominant_html.name,
            dominant_png_name=self.paths.dominant_png.name,
            features_html_name=self.paths.features_html.name,
            features_png_name=self.paths.features_png.name,
            duration_months=self.config.visualization.specialized_duration_months,
            max_points=self.config.visualization.specialized_max_points,
        )
        self._log("Generating feature portal pages.")
        catalog = generate_feature_portal(
            feature_store,
            output_dir=self.paths.report_dir,
            overview_filename=self.paths.feature_overview_html.name,
            catalog_filename=self.paths.feature_catalog_csv.name,
            pages_dir_name=self.config.data.feature_pages_subdir,
            max_points=self.config.visualization.generic_max_points,
            features=self.config.output_features.final_feature_columns,
            specialized_links={
                "自定义可视化｜Liquidation Dominance": self.paths.dominant_html.name,
                "自定义可视化｜Liquidation Feature Stack": self.paths.features_html.name,
            },
        )
        self._log(f"Visualization artifacts written under {self.paths.report_dir}")
        return catalog

    def _finalize_run(self, *, status: str, error_message: str | None = None) -> None:
        self._summary["status"] = status
        if error_message:
            self._summary["error_message"] = error_message
        self._summary["log_file"] = str(self.run_context.log_file)
        self._summary["stage_timings"] = self._stage_timings
        write_run_summary(self.run_context.run_dir, self._summary)
        shutil.copy2(self.run_context.log_file, self.paths.latest_log_txt)
        append_run_history(self.paths.run_history_csv, {
            "run_id": self.run_id,
            "status": status,
            "input_source": self._summary.get("input_source"),
            "feature_store_latest_time": self._summary.get("feature_store_latest_time"),
            "feature_store_rows": self._summary.get("feature_store_rows"),
            "log_file": str(self.run_context.log_file),
            "run_dir": str(self.run_context.run_dir),
        })
        self._log(f"Run artifacts recorded at {self.run_context.run_dir}")

    def run(self, *, input_csv: Path | None = None, build_visualizations: bool | None = None) -> FeatureEngineeringResult:
        if build_visualizations is None:
            build_visualizations = bool(self.config.execution.build_visualizations and self.config.visualization.enabled)
        self._summary["visualizations_enabled"] = bool(build_visualizations)
        self._log("Starting BTC liquidation feature engineering pipeline.")
        validator = OutputValidator(self.config.validation)
        try:
            t = time.perf_counter()
            input_data = self.load_input_data(input_csv=input_csv)
            self._record_stage_timing("load_input_data", t)

            t = time.perf_counter()
            clean = self.preprocess(input_data)
            validator.validate_clean(clean)
            self._record_stage_timing("preprocess", t)

            t = time.perf_counter()
            canonical = self.build_canonical_liquidation_family(clean)
            validator.validate_canonical(canonical)
            validator.validate_cache_csv("cache_fll", self.paths.fll_cache_csv)
            validator.validate_cache_csv("cache_fsl", self.paths.fsl_cache_csv)
            self._record_stage_timing("canonical", t)

            t = time.perf_counter()
            binned, _ = self.build_bin_features(canonical)
            dominance = build_dominance_features(binned, cfg=self.config.dominance)
            dominance.to_csv(self.paths.dominance_csv, index=False)
            validator.validate_dominance(dominance)
            log_kv(self.logger, "Dominance summary", {
                "dominance_counts": dominance["dominance"].value_counts(dropna=False).sort_index().to_dict(),
                "hit_ceiling_bottom_events": int((dominance.get("hit_ceiling_bottom", 0) != 0).sum()) if "hit_ceiling_bottom" in dominance else 0,
                "reverse_ceiling_bottom_events": int((dominance.get("reverse_ceiling_bottom", 0) != 0).sum()) if "reverse_ceiling_bottom" in dominance else 0,
            })
            self._record_stage_timing("dominance", t)

            t = time.perf_counter()
            liq_features = self.export_liq_dataflow_features(dominance)
            validator.validate_final_features(liq_features, final_feature_columns=self.config.output_features.final_feature_columns)
            feature_store = self.build_feature_store(clean, dominance, liq_features)
            validator.validate_feature_store(feature_store)
            self._record_stage_timing("final_features_and_store", t)

            if build_visualizations:
                t = time.perf_counter()
                self.build_visualizations(dominance=dominance, feature_store=feature_store)
                self._record_stage_timing("visualization", t)
            else:
                self._log("Visualization generation skipped by caller.")

            t = time.perf_counter()
            validator.validate_artifacts(
                self.paths,
                build_visualizations=build_visualizations,
                expected_feature_pages=len(self.config.output_features.final_feature_columns),
            )
            validation_report = validator.write_reports(self.paths)
            validator.assert_valid()
            self._record_stage_timing("validation", t)

            self._finalize_run(status="SUCCESS")
            self._log("Feature engineering pipeline completed successfully.")
            return FeatureEngineeringResult(input_data=input_data, clean=clean, dominance=dominance, liq_features=liq_features, feature_store=feature_store, validation_report=validation_report, paths=self.paths)
        except Exception as exc:
            self._log(f"Pipeline failed: {exc}", level=logging.ERROR)
            self._log(traceback.format_exc(), level=logging.ERROR)
            self._finalize_run(status="FAILED", error_message=repr(exc))
            raise
