from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import ProjectConfig
from .utils import weighted_pinball_loss


@dataclass
class QuantileImpactCurveModel:
    """Constrained linear quantile passive impact curve.

    The model is intentionally low-degree and mechanism-constrained. Coefficients
    after the intercept are constrained to be non-negative. Because transition
    severity can be negative for abrupt direction reversal, a positive coefficient
    still lowers PLIE in those reversal cases.
    """

    feature_names: list[str]
    quantile: float = 0.65
    l2_alpha: float = 1e-4
    max_iter: int = 600
    intercept_: float | None = None
    coef_: np.ndarray | None = None
    success_: bool = False
    message_: str = "not fitted"

    def _validate_xy(self, X: pd.DataFrame | np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_arr = X[self.feature_names].to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        mask = np.isfinite(y_arr) & np.isfinite(X_arr).all(axis=1)
        X_arr = X_arr[mask]
        y_arr = y_arr[mask]
        if len(y_arr) == 0:
            raise ValueError("No finite rows available for QuantileImpactCurveModel.fit().")
        return X_arr, y_arr

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "QuantileImpactCurveModel":
        X_arr, y_arr = self._validate_xy(X, y)
        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=float)
            finite_mask = np.isfinite(y) & np.isfinite((X[self.feature_names].to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float))).all(axis=1)
            w = w[finite_mask]
            w = np.where(np.isfinite(w), w, 0.0)
        else:
            w = np.ones_like(y_arr, dtype=float)
        w = np.clip(w, 0.0, np.inf)
        if w.sum() <= 0:
            w = np.ones_like(y_arr, dtype=float)

        n_features = X_arr.shape[1]
        init_intercept = float(np.nanquantile(y_arr, self.quantile))
        x0 = np.r_[init_intercept, np.zeros(n_features)]
        bounds = [(None, None)] + [(0.0, None)] * n_features

        q = float(self.quantile)
        alpha = float(self.l2_alpha)

        # Smooth tilted absolute loss. This is an engineering approximation to
        # pinball loss that keeps the quantile objective differentiable and fast
        # on large source-clock datasets. The non-negative coefficient bounds
        # preserve the mechanism constraints from the algorithm design.
        smooth_eps = 1.0  # bps scale; small relative to liquidation event tails.
        w_sum = float(np.sum(w))

        def objective_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
            intercept = params[0]
            coef = params[1:]
            pred = intercept + X_arr @ coef
            err = y_arr - pred
            sqrt_term = np.sqrt(err * err + smooth_eps * smooth_eps)
            loss = (q - 0.5) * err + 0.5 * sqrt_term
            obj = float(np.sum(loss * w) / w_sum) + alpha * float(np.sum(coef * coef))

            dloss_derr = (q - 0.5) + 0.5 * err / sqrt_term
            weighted = w * dloss_derr / w_sum
            grad_intercept = -float(np.sum(weighted))
            grad_coef = -(X_arr.T @ weighted) + 2.0 * alpha * coef
            grad = np.r_[grad_intercept, grad_coef]
            return obj, grad

        res = minimize(
            lambda par: objective_and_grad(par),
            x0=x0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": int(self.max_iter), "ftol": 1e-8},
        )
        self.intercept_ = float(res.x[0])
        self.coef_ = np.asarray(res.x[1:], dtype=float)
        self.success_ = bool(res.success)
        self.message_ = str(res.message)
        return self

    def predict_raw(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.intercept_ is None or self.coef_ is None:
            raise RuntimeError("Model is not fitted.")
        X_arr = X[self.feature_names].to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        pred = self.intercept_ + X_arr @ self.coef_
        return pred

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict non-negative passive impact magnitude in bps."""
        return np.maximum(0.0, self.predict_raw(X))

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "quantile": self.quantile,
            "l2_alpha": self.l2_alpha,
            "max_iter": self.max_iter,
            "intercept": self.intercept_,
            "coef": None if self.coef_ is None else self.coef_.tolist(),
            "success": self.success_,
            "message": self.message_,
        }


@dataclass
class MultiHorizonPLIEPICModel:
    """One constrained quantile impact curve per horizon."""

    horizons_min: list[int]
    feature_names: list[str]
    quantile: float
    l2_alpha: float
    max_iter: int
    main_horizon_min: int | None = None
    multiply_by_reliability: bool = True
    use_sample_weight: bool = False
    sample_weight_floor: float = 0.10
    models: dict[int, QuantileImpactCurveModel] = field(default_factory=dict)
    fitted_: bool = False

    @classmethod
    def from_config(cls, cfg: ProjectConfig) -> "MultiHorizonPLIEPICModel":
        return cls(
            horizons_min=[int(h) for h in cfg.get("features", "horizons_min")],
            feature_names=list(cfg.get("model", "feature_names")),
            quantile=float(cfg.get("features", "quantile")),
            l2_alpha=float(cfg.get("model", "l2_alpha")),
            max_iter=int(cfg.get("model", "max_iter")),
            main_horizon_min=int(cfg.get("features", "main_horizon_min", default=cfg.get("features", "horizons_min")[0])),
            multiply_by_reliability=bool(cfg.get("model", "multiply_by_reliability", default=True)),
            use_sample_weight=bool(cfg.get("model", "use_sample_weight", default=False)),
            sample_weight_floor=float(cfg.get("model", "sample_weight_floor", default=0.10)),
        )

    def fit(self, train_df: pd.DataFrame) -> "MultiHorizonPLIEPICModel":
        if len(train_df) < 1:
            raise ValueError("Empty train_df.")
        sample_weight = None
        if self.use_sample_weight and "plie_reliability" in train_df.columns:
            sample_weight = train_df["plie_reliability"].to_numpy(dtype=float)
            sample_weight = np.clip(sample_weight, self.sample_weight_floor, 1.0)
        for h in self.horizons_min:
            label_col = f"plie_aligned_ret_{h}m_bps"
            if label_col not in train_df.columns:
                raise ValueError(f"Missing label column: {label_col}")
            model = QuantileImpactCurveModel(
                feature_names=self.feature_names,
                quantile=self.quantile,
                l2_alpha=self.l2_alpha,
                max_iter=self.max_iter,
            )
            y = train_df[label_col].to_numpy(dtype=float)
            model.fit(train_df, y, sample_weight=sample_weight)
            self.models[h] = model
        self.fitted_ = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_ or not self.models:
            raise RuntimeError("MultiHorizonPLIEPICModel is not fitted.")
        out = df.copy()
        direction = pd.to_numeric(out.get("plie_direction", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        reliability = pd.to_numeric(out.get("plie_reliability", 1.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        reliability = np.clip(reliability, 0.0, 1.0)
        for h, model in self.models.items():
            mag_raw = model.predict(out)
            mag = mag_raw * reliability if self.multiply_by_reliability else mag_raw
            out[f"plie_passive_{h}m_bps_mag_raw"] = mag_raw
            out[f"plie_passive_{h}m_bps_mag"] = mag
            out[f"plie_passive_{h}m_bps"] = direction * mag
            ret_col = f"ret_{h}m_bps"
            if ret_col in out.columns:
                out[f"plie_residual_{h}m_bps"] = out[ret_col].astype(float) - out[f"plie_passive_{h}m_bps"].astype(float)
                aligned_actual = direction * out[ret_col].astype(float).to_numpy(dtype=float)
                out[f"plie_absorption_{h}m"] = 1.0 - aligned_actual / (mag + 1e-9)
        main_horizon = self.main_horizon_min if self.main_horizon_min in self.horizons_min else self.horizons_min[0]
        out["plie_main_bps"] = out[f"plie_passive_{main_horizon}m_bps"]
        out["plie_abs_main_bps"] = out["plie_main_bps"].abs()
        return out

    def coefficients_frame(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for h, model in self.models.items():
            rows.append({"horizon_min": h, "term": "intercept", "value": model.intercept_})
            if model.coef_ is not None:
                for name, coef in zip(model.feature_names, model.coef_):
                    rows.append({"horizon_min": h, "term": name, "value": float(coef)})
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizons_min": self.horizons_min,
            "feature_names": self.feature_names,
            "quantile": self.quantile,
            "l2_alpha": self.l2_alpha,
            "max_iter": self.max_iter,
            "main_horizon_min": self.main_horizon_min,
            "multiply_by_reliability": self.multiply_by_reliability,
            "use_sample_weight": self.use_sample_weight,
            "sample_weight_floor": self.sample_weight_floor,
            "models": {str(h): m.to_dict() for h, m in self.models.items()},
        }


def model_feature_contribution(model: MultiHorizonPLIEPICModel) -> pd.DataFrame:
    """Return coefficients as feature contribution proxy.

    For this constrained low-degree model, coefficient magnitude is the relevant
    transparent contribution measure. It is not a generic black-box importance.
    """
    return model.coefficients_frame()
