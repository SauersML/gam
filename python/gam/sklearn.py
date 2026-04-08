from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.validation import check_is_fitted

from ._api import fit as fit_model
from ._tables import attach_target, response_column_name, table_columns

__all__ = ["GAMClassifier", "GAMRegressor"]


def _resolved_formula(formula: str, target_name: str) -> str:
    if "~" in formula:
        _lhs, rhs = formula.split("~", 1)
        return f"{target_name} ~ {rhs.strip()}"
    return f"{target_name} ~ {formula.strip()}"


def _prepare_fit_input(X: Any, y: Any, formula: str) -> tuple[Any, str, list[str]]:
    if isinstance(y, str):
        columns, _kind = table_columns(X)
        if y not in columns:
            raise ValueError(f"target column '{y}' is missing from the training table")
        feature_names = [name for name in columns if name != y]
        return X, _resolved_formula(formula, y), feature_names
    if y is None:
        target_name = response_column_name(formula)
        if target_name is None:
            raise ValueError("formula must include a response when y is not provided")
        columns, _kind = table_columns(X)
        if target_name not in columns:
            raise ValueError(
                f"response column '{target_name}' is missing from the training table"
            )
        feature_names = [name for name in columns if name != target_name]
        return X, formula, feature_names
    target_name = response_column_name(formula) or "y"
    bound_columns, _kind = attach_target(X, y, target_name=target_name)
    feature_names = [name for name in bound_columns if name != target_name]
    return bound_columns, _resolved_formula(formula, target_name), feature_names


@dataclass
class _BaseGAMEstimator(BaseEstimator):
    formula: str
    family: str = "auto"
    offset: str | None = None
    weights: str | None = None
    config: dict[str, Any] | None = None

    def _fit_model(self, X: Any, y: Any = None):
        training_data, fit_formula, feature_names = _prepare_fit_input(X, y, self.formula)
        self.model_ = fit_model(
            training_data,
            fit_formula,
            family=self.family,
            offset=self.offset,
            weights=self.weights,
            config=self.config,
        )
        self.formula_ = fit_formula
        self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        self.n_features_in_ = len(feature_names)
        return self

    def summary(self):
        check_is_fitted(self, "model_")
        return self.model_.summary()

    def report(self, path: str):
        check_is_fitted(self, "model_")
        return self.model_.report(path)

    def check(self, X: Any):
        check_is_fitted(self, "model_")
        return self.model_.check(X)


class GAMRegressor(_BaseGAMEstimator, RegressorMixin):
    def fit(self, X: Any, y: Any = None):
        return self._fit_model(X, y)

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "model_")
        predicted = self.model_.predict(X, return_type="dict")
        return np.asarray(predicted["mean"], dtype=float)

    def score(self, X: Any, y: Any) -> float:
        return float(r2_score(np.asarray(y, dtype=float), self.predict(X)))


class GAMClassifier(_BaseGAMEstimator, ClassifierMixin):
    def fit(self, X: Any, y: Any = None):
        fitted = self._fit_model(X, y)
        self.classes_ = np.asarray([0, 1])
        return fitted

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "model_")
        predicted = self.model_.predict(X, return_type="dict")
        positive = np.clip(np.asarray(predicted["mean"], dtype=float), 0.0, 1.0)
        negative = 1.0 - positive
        return np.column_stack([negative, positive])

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X: Any, y: Any) -> float:
        return float(accuracy_score(np.asarray(y, dtype=int), self.predict(X)))
