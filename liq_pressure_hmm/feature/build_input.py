from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from liq_pressure_hmm.alignment import AlignmentConfig, merge_features_asof


def merge_to_10min_with_ffill(
    df_10m: pd.DataFrame,
    df_other: pd.DataFrame,
    *,
    time_col: str = "time",
    price_col: str = "price",
    feature_timestamp_offset: str | None = "50min",
    merge_tolerance: str | None = "70min",
    source_time_col: str = "liq_feature_time",
    source_age_col: str = "liq_feature_age_min",
) -> pd.DataFrame:
    """Backward-compatible wrapper around the new explicit availability merge."""
    cfg = AlignmentConfig(
        time_col=time_col,
        price_col=price_col,
        feature_timestamp_offset=feature_timestamp_offset,
        merge_tolerance=merge_tolerance,
        source_time_col=source_time_col,
        source_age_col=source_age_col,
    )
    return merge_features_asof(df_10m, df_other, cfg=cfg)


def build_btc_price_lld_10m(
    *,
    features_rpn_path: Path,
    price_10m_path: Path,
    output_path: Path,
    alignment_cfg: AlignmentConfig | None = None,
) -> Path:
    df_features = pd.read_csv(features_rpn_path)
    df_price = pd.read_csv(price_10m_path)
    alignment_cfg = alignment_cfg or AlignmentConfig()

    required_feature_cols = [
        "time",
        "fll_cwt_kf",
        "fsl_cwt_kf",
        "risk_priority_number",
        "diff_ls_cwt_kf",
        "total_ls_cwt_kf",
        "diff_dom_ls_cwt_kf",
    ]
    missing = [c for c in required_feature_cols if c not in df_features.columns]
    if missing:
        raise ValueError(f"Missing required feature columns in {features_rpn_path}: {missing}")
    df_features = df_features[required_feature_cols].copy()

    required_price_cols = ["time", "price"]
    missing_p = [c for c in required_price_cols if c not in df_price.columns]
    if missing_p:
        raise ValueError(f"Missing required price columns in {price_10m_path}: {missing_p}")
    df_price = df_price[required_price_cols].copy()

    merged = merge_features_asof(df_price, df_features, cfg=alignment_cfg)
    merged = merged.dropna(subset=["fll_cwt_kf", "fsl_cwt_kf", alignment_cfg.source_time_col]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return output_path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    root = _repo_root()

    parser = argparse.ArgumentParser(description="Build liq_pressure_hmm merged input CSV (BTC_price_lld_10m.csv).")
    parser.add_argument("--features-rpn", type=Path, default=root / "liq_dataflow/data/features/features_rpn.csv")
    parser.add_argument("--price-10m", type=Path, default=root / "liq_dataflow/data/raw/BTC_price_10m.csv")
    parser.add_argument("--output", type=Path, default=root / "liq_pressure_hmm/input/BTC_price_lld_10m.csv")
    parser.add_argument("--feature-timestamp-offset", type=str, default="50min")
    parser.add_argument("--merge-tolerance", type=str, default="70min")
    args = parser.parse_args(argv)

    build_btc_price_lld_10m(
        features_rpn_path=args.features_rpn,
        price_10m_path=args.price_10m,
        output_path=args.output,
        alignment_cfg=AlignmentConfig(
            feature_timestamp_offset=args.feature_timestamp_offset,
            merge_tolerance=args.merge_tolerance,
        ),
    )
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
