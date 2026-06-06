"""Train-only empirical CDF calibration for QD-MAR."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable
import pickle
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class EmpiricalCDF:
    values: np.ndarray

    def __post_init__(self) -> None:
        self.values = np.sort(np.asarray(self.values, dtype=float))
        self.values = self.values[np.isfinite(self.values)]

    @property
    def n(self) -> int:
        return int(self.values.size)

    def cdf(self, x: float) -> float:
        if not np.isfinite(x) or self.n == 0:
            return np.nan
        # Right-continuous empirical CDF, bounded away from exact 0/1 slightly
        # for numerical stability in downstream transforms.
        rank = np.searchsorted(self.values, x, side="right")
        return float(rank / self.n)


@dataclass
class BucketCDFCalibrator:
    """Conditional empirical CDF with hierarchical fallback buckets.

    This is not a predictive model. It stores train-only response distributions
    and maps each matured event response into a comparable percentile.
    """

    value_col: str
    bucket_levels: list[list[str]]
    min_bucket_size: int = 500
    cdfs: dict[tuple[str, tuple[Any, ...]], EmpiricalCDF] = field(default_factory=dict)
    counts: dict[tuple[str, tuple[Any, ...]], int] = field(default_factory=dict)

    @staticmethod
    def _key(level: Iterable[str], row: pd.Series) -> tuple[str, tuple[Any, ...]]:
        cols = tuple(level)
        vals = tuple(row.get(c, "__missing__") for c in cols)
        return ("|".join(cols), vals)

    def fit(self, df: pd.DataFrame, train_mask: pd.Series) -> "BucketCDFCalibrator":
        train = df.loc[train_mask].copy()
        for level in self.bucket_levels:
            cols = list(level)
            if not cols:
                continue
            for vals, grp in train.groupby(cols, dropna=False):
                if not isinstance(vals, tuple):
                    vals = (vals,)
                values = pd.to_numeric(grp[self.value_col], errors="coerce").dropna().to_numpy()
                key = ("|".join(cols), tuple(vals))
                if len(values) > 0:
                    self.cdfs[key] = EmpiricalCDF(values)
                    self.counts[key] = len(values)
        # Always create a global fallback.
        values = pd.to_numeric(train[self.value_col], errors="coerce").dropna().to_numpy()
        self.cdfs[("__global__", tuple())] = EmpiricalCDF(values)
        self.counts[("__global__", tuple())] = len(values)
        return self

    def transform_row(self, row: pd.Series) -> tuple[float, str, int]:
        for level in self.bucket_levels:
            key = self._key(level, row)
            n = self.counts.get(key, 0)
            if n >= self.min_bucket_size:
                return self.cdfs[key].cdf(row.get(self.value_col, np.nan)), key[0] + ":" + str(key[1]), n
        key = ("__global__", tuple())
        return self.cdfs[key].cdf(row.get(self.value_col, np.nan)), "__global__", self.counts.get(key, 0)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        vals, keys, ns = [], [], []
        for _, row in df.iterrows():
            u, k, n = self.transform_row(row)
            vals.append(u)
            keys.append(k)
            ns.append(n)
        return pd.DataFrame({"percentile": vals, "bucket_id": keys, "bucket_n": ns}, index=df.index)


def save_calibrators(calibrators: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(calibrators, f)


def load_calibrators(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as f:
        return pickle.load(f)
