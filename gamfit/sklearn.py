from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.validation import check_is_fitted

from ._api import fit as fit_model
from ._model import Model
from ._tables import attach_target, response_column_name, table_columns

__all__ = ["GAMClassifier", "GAMRegressor"]

_BaseT = TypeVar("_BaseT", bound="_BaseGAMEstimator")

# No GAMSurvival wrapper: survival responses (e.g. Surv(time, event)) are a
# two-column construct that does not fit scikit-learn's (X, y) contract, and
# survival prediction is a per-time-grid hazard surface rather than a single
# response vector. Users who want a scikit-style API for survival should call
# gamfit.fit(...) directly with family="cox" (or equivalent) and operate on
# the SurvivalPrediction object returned by Model.predict.


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

    def _fit_model(self: _BaseT, X: Any, y: Any = None) -> _BaseT:
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

    def summary(self) -> Any:
        check_is_fitted(self, "model_")
        return self.model_.summary()

    def report(self, path: str) -> Any:
        check_is_fitted(self, "model_")
        if not isinstance(self.model_, Model):
            raise TypeError(
                "report() is only supported for scalar GAM models; "
                "response-geometry models do not expose a report() method"
            )
        return self.model_.report(path)

    def check(self, X: Any) -> Any:
        check_is_fitted(self, "model_")
        if not isinstance(self.model_, Model):
            raise TypeError(
                "check() is only supported for scalar GAM models; "
                "response-geometry models do not expose a check() method"
            )
        return self.model_.check(X)


class GAMRegressor(_BaseGAMEstimator, RegressorMixin):
    """scikit-learn-compatible regressor wrapping :func:`gamfit.fit`.

    Construct with a formula string and (optionally) pipeline kwargs such as
    ``family``, ``offset``, ``weights``, or a free-form ``config`` dict, then
    call :meth:`fit` with either a fully-formed table (``X``) or a feature
    table plus a target column / vector (``y``). After fitting, the estimator
    exposes the standard ``predict`` / ``score`` interface plus pass-through
    helpers :meth:`summary`, :meth:`report`, and :meth:`check` from the
    underlying :class:`Model`.

    Parameters
    ----------
    formula : str
        Wilkinson-style formula. May or may not include the response on the
        left-hand side; the response is resolved from ``y`` if missing.
    family : str, default ``"auto"``
        Likelihood family forwarded to :func:`gamfit.fit`.
    offset : str or None, optional
        Offset column name, forwarded to :func:`gamfit.fit`.
    weights : str or None, optional
        Observation-weight column name.
    config : dict or None, optional
        Escape-hatch dict of extra pipeline keys.

    Examples
    --------
    >>> from gamfit.sklearn import GAMRegressor
    >>> reg = GAMRegressor(formula="y ~ s(x1) + s(x2)").fit(X_train, y_train)
    >>> preds = reg.predict(X_test)
    >>> reg.score(X_test, y_test)
    0.87
    """

    def fit(self, X: Any, y: Any = None) -> "GAMRegressor":
        """Fit the underlying GAM and return ``self``.

        Parameters
        ----------
        X : Any
            Training table (pandas DataFrame, pyarrow Table, dict of columns,
            list of records, or anything :func:`gamfit.fit` accepts). May
            include the response column or not.
        y : str, array-like, or None, optional
            Target. ``str`` names a column already in ``X``; an array-like is
            bound to ``X`` under the response name implied by ``formula``;
            ``None`` means ``X`` already contains the response named by
            ``formula``.

        Returns
        -------
        GAMRegressor
            Fitted estimator (``self``) with ``model_``, ``formula_``,
            ``feature_names_in_``, and ``n_features_in_`` attributes set.

        Examples
        --------
        >>> GAMRegressor(formula="y ~ s(x)").fit(df, y="y")
        """
        return self._fit_model(X, y)

    def predict(self, X: Any) -> np.ndarray:
        """Predict the conditional mean for each row in ``X``.

        Parameters
        ----------
        X : Any
            Serving table with the feature columns seen at fit time.

        Returns
        -------
        numpy.ndarray
            One-dimensional float array of predicted means, one per row.

        Examples
        --------
        >>> reg.predict(X_test)[:3]
        array([1.02, 0.98, 1.41])
        """
        check_is_fitted(self, "model_")
        predicted = self.model_.predict(X, return_type="dict")
        return np.asarray(predicted["mean"], dtype=float)

    def score(self, X: Any, y: Any, sample_weight: Any = None) -> float:
        """Return the coefficient of determination :math:`R^2`.

        Parameters
        ----------
        X : Any
            Test feature table.
        y : array-like
            True response values.
        sample_weight : array-like or None, optional
            Per-row weights forwarded to :func:`sklearn.metrics.r2_score`.

        Returns
        -------
        float
            :math:`R^2` of the predictions.

        Examples
        --------
        >>> reg.score(X_test, y_test)
        0.87
        """
        return float(
            r2_score(
                np.asarray(y, dtype=float),
                self.predict(X),
                sample_weight=sample_weight,
            )
        )


class GAMClassifier(_BaseGAMEstimator, ClassifierMixin):
    """scikit-learn-compatible binary classifier wrapping :func:`gamfit.fit`.

    Same construction and ``fit`` semantics as :class:`GAMRegressor` (see that
    class for parameter documentation). Predictions interpret the model's
    mean as the probability of the positive class; classes are fixed to
    ``[0, 1]`` and a threshold of ``0.5`` is used by :meth:`predict`.

    Examples
    --------
    >>> from gamfit.sklearn import GAMClassifier
    >>> clf = GAMClassifier(formula="y ~ s(x1) + s(x2)", family="binomial")
    >>> clf.fit(X_train, y_train)
    >>> clf.predict_proba(X_test)[:1]
    array([[0.34, 0.66]])
    """

    def fit(self, X: Any, y: Any = None) -> "GAMClassifier":
        """Fit the binary GAM classifier and return ``self``.

        Parameters
        ----------
        X : Any
            Training table. See :meth:`GAMRegressor.fit` for accepted forms.
        y : str, array-like, or None, optional
            Binary target. See :meth:`GAMRegressor.fit` for accepted forms.

        Returns
        -------
        GAMClassifier
            Fitted estimator (``self``) with ``classes_`` set to ``[0, 1]``.

        Examples
        --------
        >>> GAMClassifier(formula="y ~ s(x)", family="binomial").fit(df, y="y")
        """
        fitted = self._fit_model(X, y)
        self.classes_ = np.asarray([0, 1])
        return fitted

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities for each row in ``X``.

        Parameters
        ----------
        X : Any
            Serving table with the feature columns seen at fit time.

        Returns
        -------
        numpy.ndarray
            Two-column float array ``[[P(y=0), P(y=1)], ...]``, clipped to
            ``[0, 1]``.

        Examples
        --------
        >>> clf.predict_proba(X_test).shape
        (100, 2)
        """
        check_is_fitted(self, "model_")
        predicted = self.model_.predict(X, return_type="dict")
        positive = np.clip(np.asarray(predicted["mean"], dtype=float), 0.0, 1.0)
        negative = 1.0 - positive
        return np.column_stack([negative, positive])

    def predict(self, X: Any) -> np.ndarray:
        """Predict the binary class label using a 0.5 threshold on the positive class.

        Parameters
        ----------
        X : Any
            Serving table with the feature columns seen at fit time.

        Returns
        -------
        numpy.ndarray
            One-dimensional integer array of class labels (``0`` or ``1``).

        Examples
        --------
        >>> clf.predict(X_test)[:5]
        array([1, 0, 1, 1, 0])
        """
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X: Any, y: Any, sample_weight: Any = None) -> float:
        """Return classification accuracy.

        Parameters
        ----------
        X : Any
            Test feature table.
        y : array-like
            True binary labels.
        sample_weight : array-like or None, optional
            Per-row weights forwarded to
            :func:`sklearn.metrics.accuracy_score`.

        Returns
        -------
        float
            Accuracy in ``[0, 1]``.

        Examples
        --------
        >>> clf.score(X_test, y_test)
        0.91
        """
        return float(
            accuracy_score(
                np.asarray(y, dtype=int),
                self.predict(X),
                sample_weight=sample_weight,
            )
        )
