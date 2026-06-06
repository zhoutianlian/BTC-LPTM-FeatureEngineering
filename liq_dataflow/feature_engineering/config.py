from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "feature_engineering.json"
LEGACY_CONFIG_PATH = Path(__file__).with_name("config.yaml")


@dataclass(frozen=True)
class ProjectConfig:
    name: str = "BTC Liquidation Feature Engineering"
    start_time: str = "2021-02-01"


@dataclass(frozen=True)
class InputConfig:
    source_csv: str = "../liq_data_download/data/raw/hourly/BTC_price_lld.csv"
    fallback_csv: str = "data/clean/clean_data.csv"


@dataclass(frozen=True)
class ExecutionConfig:
    build_visualizations: bool = True


@dataclass(frozen=True)
class ColumnConfig:
    time_col: str = "time"
    price_col: str = "price"
    raw_long_liquidations_col: str = "futures_long_liquidations"
    raw_short_liquidations_col: str = "futures_short_liquidations"
    clean_long_col: str = "fll_normal"
    clean_short_col: str = "fsl_normal"


@dataclass(frozen=True)
class DataConfig:
    clean_subdir: str = "clean"
    cache_subdir: str = "cache"
    features_subdir: str = "features"
    final_features_subdir: str = ""
    report_subdir: str = "report"
    feature_pages_subdir: str = "feature_pages"
    clean_filename: str = "clean_data.csv"
    canonical_filename: str = "features_rpn.csv"
    bin_stage_filename: str = "features_rpn_bin_stage.csv"
    bin_stats_filename: str = "features_bin_stats.csv"
    legacy_bin_stats_filename: str = "features_kmeans_stats.csv"
    dominance_filename: str = "features_rpn_bin.csv"
    final_features_filename: str = "features_liq_dataflow.csv"
    feature_store_filename: str = "feature_store.csv"
    fll_cache_filename: str = "fll_cwt.csv"
    fsl_cache_filename: str = "fsl_cwt.csv"


@dataclass(frozen=True)
class PreprocessConfig:
    outlier_iqr_window_days: int = 180


@dataclass(frozen=True)
class SmoothingConfig:
    wavelet: str = "coif3"
    level: int = 6
    threshold_method: str = "soft"
    window_size_hours: int = 24 * 360 * 2
    kalman_initial_state_mean: float = 0.49
    kalman_initial_state_covariance: float = 50.0
    kalman_observation_covariance: float = 80.0
    kalman_transition_covariance: float = 0.05


@dataclass(frozen=True)
class BinningConfig:
    n_bins: int = 9
    min_history_bars: int = 24 * 3
    refit_every_bars: int = 24
    neutral_bin: int = 4
    extreme_window_bars: int = 30 * 24


@dataclass(frozen=True)
class DominanceConfig:
    rolling_window_bars: int = 365 * 24
    rolling_min_periods: int = 24 * 30
    reverse_window_bars: int = 8
    reverse_diff_threshold: float = 0.2


@dataclass(frozen=True)
class ModelFeatureConfig:
    z_window_bars: int = 24


@dataclass(frozen=True)
class OutputFeatureConfig:
    final_feature_columns: list[str] = field(
        default_factory=lambda: [
            "fll_cwt_kf",
            "fsl_cwt_kf",
            "diff_ls_cwt_kf",
            "total_ls_cwt_kf",
            "risk_priority_number",
            "bin_index",
            "dominance",
            "diff_dom_ls_cwt_kf",
            "z_logTotalP",
            "z_sdom",
            "z_fll_cwt_kf",
            "z_fsl_cwt_kf",
        ]
    )


@dataclass(frozen=True)
class VisualizationConfig:
    enabled: bool = True
    specialized_duration_months: int = 3
    specialized_max_points: int = 15000
    generic_max_points: int = 8000
    overview_filename: str = "feature_overview.html"
    catalog_filename: str = "feature_catalog.csv"
    dominant_html_filename: str = "rpn_dominance-latest.html"
    dominant_png_filename: str = "rpn_dominance-latest.png"
    features_html_filename: str = "rpn_features-latest.html"
    features_png_filename: str = "rpn_features-latest.png"


@dataclass(frozen=True)
class ValidationConfig:
    enabled: bool = True
    raise_on_error: bool = True
    numeric_tolerance: float = 1e-9
    max_model_nan_ratio: float = 0.01
    report_csv_filename: str = "output_validation_report.csv"
    report_md_filename: str = "output_validation_report.md"
    report_json_filename: str = "output_validation_report.json"


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    project: ProjectConfig = ProjectConfig()
    input: InputConfig = InputConfig()
    execution: ExecutionConfig = ExecutionConfig()
    columns: ColumnConfig = ColumnConfig()
    data: DataConfig = DataConfig()
    preprocess: PreprocessConfig = PreprocessConfig()
    smoothing: SmoothingConfig = SmoothingConfig()
    binning: BinningConfig = BinningConfig()
    dominance: DominanceConfig = DominanceConfig()
    model_features: ModelFeatureConfig = ModelFeatureConfig()
    output_features: OutputFeatureConfig = OutputFeatureConfig()
    visualization: VisualizationConfig = VisualizationConfig()
    validation: ValidationConfig = ValidationConfig()


@dataclass(frozen=True)
class ProjectPaths:
    root_dir: Path
    logs_dir: Path
    runs_dir: Path
    run_history_csv: Path
    latest_log_txt: Path
    data_dir: Path
    clean_dir: Path
    cache_dir: Path
    features_dir: Path
    final_features_dir: Path
    report_dir: Path
    feature_pages_dir: Path
    clean_csv: Path
    canonical_csv: Path
    bin_stage_csv: Path
    bin_stats_csv: Path
    legacy_bin_stats_csv: Path
    dominance_csv: Path
    final_features_csv: Path
    feature_store_csv: Path
    feature_overview_html: Path
    feature_catalog_csv: Path
    dominant_html: Path
    dominant_png: Path
    features_html: Path
    features_png: Path
    plotly_bundle: Path
    validation_report_csv: Path
    validation_report_md: Path
    validation_report_json: Path
    fll_cache_csv: Path
    fsl_cache_csv: Path

    @classmethod
    def from_config(cls, *, root_dir: Path | None = None, cfg: FeatureEngineeringConfig) -> "ProjectPaths":
        root = Path(root_dir) if root_dir is not None else PROJECT_ROOT
        logs_dir = root / "logs"
        runs_dir = logs_dir / "runs"
        data_dir = root / "data"
        clean_dir = data_dir / cfg.data.clean_subdir
        cache_dir = data_dir / cfg.data.cache_subdir
        features_dir = data_dir / cfg.data.features_subdir
        final_features_dir = features_dir / cfg.data.final_features_subdir
        report_dir = data_dir / cfg.data.report_subdir
        feature_pages_dir = report_dir / cfg.data.feature_pages_subdir
        return cls(
            root_dir=root,
            logs_dir=logs_dir,
            runs_dir=runs_dir,
            run_history_csv=logs_dir / "run_history.csv",
            latest_log_txt=logs_dir / "latest.log",
            data_dir=data_dir,
            clean_dir=clean_dir,
            cache_dir=cache_dir,
            features_dir=features_dir,
            final_features_dir=final_features_dir,
            report_dir=report_dir,
            feature_pages_dir=feature_pages_dir,
            clean_csv=clean_dir / cfg.data.clean_filename,
            canonical_csv=features_dir / cfg.data.canonical_filename,
            bin_stage_csv=features_dir / cfg.data.bin_stage_filename,
            bin_stats_csv=features_dir / cfg.data.bin_stats_filename,
            legacy_bin_stats_csv=features_dir / cfg.data.legacy_bin_stats_filename,
            dominance_csv=features_dir / cfg.data.dominance_filename,
            final_features_csv=final_features_dir / cfg.data.final_features_filename,
            feature_store_csv=features_dir / cfg.data.feature_store_filename,
            feature_overview_html=report_dir / cfg.visualization.overview_filename,
            feature_catalog_csv=report_dir / cfg.visualization.catalog_filename,
            dominant_html=report_dir / cfg.visualization.dominant_html_filename,
            dominant_png=report_dir / cfg.visualization.dominant_png_filename,
            features_html=report_dir / cfg.visualization.features_html_filename,
            features_png=report_dir / cfg.visualization.features_png_filename,
            plotly_bundle=report_dir / "plotly.min.js",
            validation_report_csv=features_dir / cfg.validation.report_csv_filename,
            validation_report_md=features_dir / cfg.validation.report_md_filename,
            validation_report_json=features_dir / cfg.validation.report_json_filename,
            fll_cache_csv=cache_dir / cfg.data.fll_cache_filename,
            fsl_cache_csv=cache_dir / cfg.data.fsl_cache_filename,
        )

    def ensure_dirs(self) -> None:
        for path in [
            self.logs_dir,
            self.runs_dir,
            self.data_dir,
            self.clean_dir,
            self.cache_dir,
            self.features_dir,
            self.final_features_dir,
            self.report_dir,
            self.feature_pages_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


def _construct_dataclass(dc_cls: type, payload: dict[str, Any] | None):
    allowed = {f.name for f in fields(dc_cls)}
    clean_payload = {k: v for k, v in (payload or {}).items() if k in allowed}
    return dc_cls(**clean_payload)


def load_config_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        if path.suffix.lower() == ".json":
            return json.load(fh) or {}
        return yaml.safe_load(fh) or {}


def load_feature_engineering_config(config_path: Path | None = None) -> FeatureEngineeringConfig:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if config_path is None and not path.exists():
        path = LEGACY_CONFIG_PATH
    payload = load_config_payload(path)
    return FeatureEngineeringConfig(
        project=_construct_dataclass(ProjectConfig, payload.get("project")),
        input=_construct_dataclass(InputConfig, payload.get("input")),
        execution=_construct_dataclass(ExecutionConfig, payload.get("execution")),
        columns=_construct_dataclass(ColumnConfig, payload.get("columns")),
        data=_construct_dataclass(DataConfig, payload.get("data")),
        preprocess=_construct_dataclass(PreprocessConfig, payload.get("preprocess")),
        smoothing=_construct_dataclass(SmoothingConfig, payload.get("smoothing")),
        binning=_construct_dataclass(BinningConfig, payload.get("binning")),
        dominance=_construct_dataclass(DominanceConfig, payload.get("dominance")),
        model_features=_construct_dataclass(ModelFeatureConfig, payload.get("model_features")),
        output_features=_construct_dataclass(OutputFeatureConfig, payload.get("output_features")),
        visualization=_construct_dataclass(VisualizationConfig, payload.get("visualization")),
        validation=_construct_dataclass(ValidationConfig, payload.get("validation")),
    )
