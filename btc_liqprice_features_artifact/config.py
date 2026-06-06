from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_CONFIG_FILENAME = 'resolved_feature_config.json'
DEFAULT_PIPELINE_CONFIG = Path(__file__).resolve().parent / 'configs' / 'feature_liqprice.json'


def _filter_dataclass_kwargs(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {f.name for f in fields(cls)}
    return {k: v for k, v in payload.items() if k in allowed}


@dataclass
class FeatureConfig:
    bar_minutes: Optional[int] = None
    decision_minutes: Optional[int] = None

    liq_spike_window_min: int = 24 * 60
    liq_kama_fast_ema: int = 2
    liq_kama_slow_ema: int = 30

    liq_roc_gaussian_window_min: int = 130

    trend_pressure_mom_window_min: int = 180
    trend_pressure_vol_window_min: int = 720

    price_kalman_process_noise: float = 1e-6
    price_kalman_slope_process_noise: float = 1e-7
    price_kalman_measurement_noise: float = 1e-4

    price_vol_adaptive_short_window_min: int = 6 * 60
    price_vol_adaptive_long_window_min: int = 24 * 60
    price_vol_adaptive_min_weight: float = 0.30
    price_vol_adaptive_sigmoid_gamma: float = 3.0

    def __post_init__(self) -> None:
        positive_ints = [
            'liq_spike_window_min',
            'liq_roc_gaussian_window_min',
            'trend_pressure_mom_window_min',
            'trend_pressure_vol_window_min',
            'price_vol_adaptive_short_window_min',
            'price_vol_adaptive_long_window_min',
        ]
        for name in positive_ints:
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f'{name} must be positive, got {value!r}.')

        if self.bar_minutes is not None and int(self.bar_minutes) <= 0:
            raise ValueError('bar_minutes must be positive when provided.')
        if self.decision_minutes is not None and int(self.decision_minutes) <= 0:
            raise ValueError('decision_minutes must be positive when provided.')
        if self.bar_minutes is not None and self.decision_minutes is not None:
            if int(self.decision_minutes) % int(self.bar_minutes) != 0:
                raise ValueError('decision_minutes must be a multiple of bar_minutes.')

        if int(self.liq_kama_fast_ema) <= 0 or int(self.liq_kama_slow_ema) <= 0:
            raise ValueError('liq_kama_fast_ema and liq_kama_slow_ema must be positive.')
        if int(self.liq_kama_fast_ema) >= int(self.liq_kama_slow_ema):
            raise ValueError('liq_kama_fast_ema must be strictly smaller than liq_kama_slow_ema.')

        positive_floats = [
            'price_kalman_process_noise',
            'price_kalman_slope_process_noise',
            'price_kalman_measurement_noise',
            'price_vol_adaptive_sigmoid_gamma',
        ]
        for name in positive_floats:
            value = float(getattr(self, name))
            if value <= 0.0:
                raise ValueError(f'{name} must be positive, got {value!r}.')

        min_weight = float(self.price_vol_adaptive_min_weight)
        if not (0.0 <= min_weight <= 1.0):
            raise ValueError('price_vol_adaptive_min_weight must be in [0, 1].')
        if int(self.price_vol_adaptive_short_window_min) >= int(self.price_vol_adaptive_long_window_min):
            raise ValueError('price_vol_adaptive_short_window_min must be smaller than long window.')

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PathsConfig:
    input: str = 'btc_liqprice_features_artifact/sample_input/sample_input.csv'
    output: str = 'btc_liqprice_features_artifact/output/liqprice_features.csv'
    resolved_config_output: str = 'btc_liqprice_features_artifact/output/resolved_feature_config.json'
    run_manifest: str = 'btc_liqprice_features_artifact/output/run_manifest.json'
    log_dir: str = 'btc_liqprice_features_artifact/logs'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ColumnsConfig:
    time_col: str = 'time'
    price_col: str = 'price'
    fll_col: str = 'fll_cwt_kf'
    fsl_col: str = 'fsl_cwt_kf'
    output_time_col: str = 'time'
    output_features: List[str] = field(default_factory=lambda: [
        'fll_spike_kama',
        'fsl_spike_kama',
        'fll_velocity_gaussian',
        'fll_acceleration_gaussian',
        'fsl_velocity_gaussian',
        'fsl_acceleration_gaussian',
        'trend_pressure',
        'kalman_slope',
        'vol_adaptive',
    ])
    aliases: Dict[str, str] = field(default_factory=lambda: {
        'close': 'price',
        'close_price': 'price',
        'mark_price': 'price',
        'last_price': 'price',
        'long_liq_cwt_kf': 'fll_cwt_kf',
        'short_liq_cwt_kf': 'fsl_cwt_kf',
    })

    def __post_init__(self) -> None:
        required_names = ['time_col', 'price_col', 'fll_col', 'fsl_col', 'output_time_col']
        for name in required_names:
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f'columns.{name} must be a non-empty string.')
            setattr(self, name, value)
        if not self.output_features:
            raise ValueError('columns.output_features must not be empty.')

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReportConfig:
    enabled: bool = True
    output_dir: str = 'btc_liqprice_features_artifact/reports/feature_diagnostics'
    feature_doc_path: str = 'btc_liqprice_features_artifact/docs/liqprice_feature_engineering.md'
    generate_html: bool = True
    generate_summary_json: bool = True
    rolling_window_minutes: int = 24 * 60
    rolling_quantiles: List[float] = field(default_factory=lambda: [0.05, 0.95])
    min_valid_count: int = 30
    missing_warn_ratio: float = 0.05
    missing_fail_ratio: float = 0.50
    inf_fail_count: int = 1
    zscore_threshold: float = 4.0
    iqr_multiplier: float = 3.0
    extreme_quantile_low: float = 0.01
    extreme_quantile_high: float = 0.99
    max_outlier_examples: int = 30
    constant_rtol: float = 1e-10
    constant_atol: float = 1e-12
    long_constant_min_bars: int = 24
    time_gap_multiplier: float = 3.0
    cliff_mad_multiplier: float = 20.0
    explosion_mad_multiplier: float = 1000.0
    high_corr_threshold: float = 0.95
    correlation_method: str = 'pearson'
    future_return_periods: List[int] = field(default_factory=lambda: [1, 3, 6])
    price_context_enabled: bool = True

    def __post_init__(self) -> None:
        positive_ints = ['rolling_window_minutes', 'min_valid_count', 'max_outlier_examples', 'long_constant_min_bars']
        for name in positive_ints:
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f'report.{name} must be positive.')
            setattr(self, name, value)

        ratios = ['missing_warn_ratio', 'missing_fail_ratio', 'extreme_quantile_low', 'extreme_quantile_high']
        for name in ratios:
            value = float(getattr(self, name))
            if not (0.0 <= value <= 1.0):
                raise ValueError(f'report.{name} must be in [0, 1].')
            setattr(self, name, value)
        if self.missing_warn_ratio > self.missing_fail_ratio:
            raise ValueError('report.missing_warn_ratio must be <= missing_fail_ratio.')
        if self.extreme_quantile_low >= self.extreme_quantile_high:
            raise ValueError('report.extreme_quantile_low must be < extreme_quantile_high.')

        positive_floats = [
            'zscore_threshold',
            'iqr_multiplier',
            'time_gap_multiplier',
            'cliff_mad_multiplier',
            'explosion_mad_multiplier',
            'high_corr_threshold',
        ]
        for name in positive_floats:
            value = float(getattr(self, name))
            if value <= 0.0:
                raise ValueError(f'report.{name} must be positive.')
            setattr(self, name, value)
        if self.correlation_method not in {'pearson', 'spearman'}:
            raise ValueError("report.correlation_method must be 'pearson' or 'spearman'.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    columns: ColumnsConfig = field(default_factory=ColumnsConfig)
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _coerce_dataclass(cls, payload: Any):
    return cls(**_filter_dataclass_kwargs(cls, dict(payload) if isinstance(payload, dict) else {}))


def feature_config_from_dict(payload: Dict[str, Any]) -> FeatureConfig:
    if isinstance(payload, dict) and any(k in payload for k in ('feature_config', 'features')):
        payload = payload.get('feature_config') or payload.get('features') or {}
    return FeatureConfig(**_filter_dataclass_kwargs(FeatureConfig, dict(payload) if isinstance(payload, dict) else {}))


def pipeline_config_from_dict(payload: Dict[str, Any]) -> PipelineConfig:
    data = dict(payload) if isinstance(payload, dict) else {}
    feature_payload = data.get('feature_config') or data.get('features')
    if feature_payload is None:
        feature_payload = _filter_dataclass_kwargs(FeatureConfig, data)
    return PipelineConfig(
        paths=_coerce_dataclass(PathsConfig, data.get('paths', {})),
        columns=_coerce_dataclass(ColumnsConfig, data.get('columns', {})),
        feature_config=feature_config_from_dict(feature_payload),
        report=_coerce_dataclass(ReportConfig, data.get('report', {})),
    )



def load_config(path: Optional[str | Path]) -> FeatureConfig:
    if not path:
        return FeatureConfig()
    p = Path(path)
    with p.open('r', encoding='utf-8') as f:
        data = json.load(f)
    return feature_config_from_dict(data)


def load_pipeline_config(path: Optional[str | Path]) -> PipelineConfig:
    if not path:
        p = DEFAULT_PIPELINE_CONFIG
    else:
        p = Path(path)
    with p.open('r', encoding='utf-8') as f:
        data = json.load(f)
    return pipeline_config_from_dict(data)



def save_config(cfg: FeatureConfig, output_path: str | Path) -> str:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)
    return str(p)


def save_pipeline_config(cfg: PipelineConfig, output_path: str | Path) -> str:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)
    return str(p)
