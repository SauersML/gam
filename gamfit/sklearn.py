from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from ._binding import rust_module
from ._api import fit as fit_model
from ._model import Model
from ._tables import attach_target, table_columns

__all__ = ["GAMClassifier", "GAMRegressor"]

_BaseT = TypeVar("_BaseT", bound="_BaseGAMEstimator")

# No GAMSurvival wrapper: survival responses (e.g. Surv(time, event)) are a
# two-column construct that does not fit scikit-learn's (X, y) contract, and
# survival prediction is a per-time-grid hazard surface rather than a single
# response vector. Users who want a scikit-style API for survival should call
# gamfit.fit(...) directly with family="cox" (or equivalent) and operate on
# the SurvivalPrediction object returned by Model.predict.


def _prepare_fit_input(X: Any, y: Any, formula: str) -> tuple[Any, str, list[str]]:
    rust = rust_module()
    if isinstance(y, str):
        columns, _kind = table_columns(X)
        fit_formula, feature_names, _target_name = rust.sklearn_fit_metadata(
            list(columns),
            formula,
            y,
            False,
        )
        return X, fit_formula, list(feature_names)
    columns, _kind = table_columns(X)
    if y is None:
        fit_formula, feature_names, _target_name = rust.sklearn_fit_metadata(
            list(columns),
            formula,
            None,
            False,
        )
        return X, fit_formula, list(feature_names)
    fit_formula, feature_names, target_name = rust.sklearn_fit_metadata(
        list(columns),
        formula,
        None,
        True,
    )
    bound_columns, _kind = attach_target(X, y, target_name=target_name)
    return bound_columns, fit_formula, list(feature_names)


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


class GAMRegressor(RegressorMixin, _BaseGAMEstimator):
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


class GAMClassifier(ClassifierMixin, _BaseGAMEstimator):
    """scikit-learn-compatible binary classifier wrapping :func:`gamfit.fit`.

    Same construction and ``fit`` semantics as :class:`GAMRegressor` (see that
    class for parameter documentation). The supplied ``y`` may be any binary
    label vector (strings, ``{-1, +1}``, ``{1, 2}``, integer ``{0, 1}``, …);
    the wrapper records the observed classes in ``classes_`` (sorted, as
    sklearn requires) and label-encodes the positive class — i.e.
    ``classes_[1]`` — to ``1`` before fitting the binomial GAM. ``predict``
    returns labels drawn from ``classes_``.

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
            Fitted estimator (``self``) with ``classes_`` reflecting the
            observed labels (sorted ascending; the positive class is
            ``classes_[1]``).

        Examples
        --------
        >>> GAMClassifier(formula="y ~ s(x)", family="binomial").fit(df, y="y")
        """
        labels = self._resolve_target_labels(X, y)
        classes = np.unique(labels)
        if classes.size != 2:
            raise ValueError(
                "GAMClassifier requires exactly two observed classes; "
                f"got {classes.size}: {classes!r}"
            )
        # Positive class is the second sorted label, matching the convention
        # used by sklearn's LabelEncoder and LogisticRegression.
        positive = classes[1]
        encoded = np.where(labels == positive, 1, 0).astype(int)
        encoded_X, encoded_y = self._inject_encoded_target(X, y, encoded)
        fitted = self._fit_model(encoded_X, encoded_y)
        self.classes_ = classes
        return fitted

    def _resolve_target_labels(self, X: Any, y: Any) -> np.ndarray:
        """Return the observed target as a 1-D numpy array of original labels.

        Handles the three supported input shapes (array-like ``y``, string
        column name ``y``, and ``y=None`` with the formula carrying the target
        name) without coercing dtype so string and integer labels both survive.
        """
        if isinstance(y, str):
            columns, _ = table_columns(X)
            if y not in columns:
                raise KeyError(
                    f"GAMClassifier.fit: target column {y!r} not in input table"
                )
            return np.asarray(columns[y])
        if y is None:
            rust = rust_module()
            columns, _ = table_columns(X)
            _, _, target_name = rust.sklearn_fit_metadata(
                list(columns), self.formula, None, True,
            )
            if target_name not in columns:
                raise KeyError(
                    "GAMClassifier.fit: formula-derived target column "
                    f"{target_name!r} not in input table"
                )
            return np.asarray(columns[target_name])
        arr = np.asarray(y)
        if arr.ndim != 1:
            raise ValueError(
                f"GAMClassifier.fit: y must be 1-D; got shape {arr.shape}"
            )
        return arr

    def _inject_encoded_target(
        self,
        X: Any,
        y: Any,
        encoded: np.ndarray,
    ) -> tuple[Any, Any]:
        """Splice the encoded ``{0,1}`` target back into the data carrier.

        For array-style ``y`` we simply pass the encoded vector through;
        ``_prepare_fit_input``'s array branch then re-attaches it via
        :func:`attach_target` under the formula's target name. For the
        column-name and ``None`` cases the original target lives inside
        ``X``; we copy the table to a dict-of-lists carrier (mirroring the
        ``"dict"`` kind the rest of the FFI already accepts), overwrite
        the target column with the encoded values, and hand the rewritten
        carrier back with ``y=None`` so ``_prepare_fit_input`` will read
        the encoded column directly from the formula.
        """
        if not isinstance(y, str) and y is not None:
            return X, encoded
        columns, _ = table_columns(X)
        if isinstance(y, str):
            target_name = y
        else:
            rust = rust_module()
            _, _, target_name = rust.sklearn_fit_metadata(
                list(columns), self.formula, None, True,
            )
        new_columns: dict[str, list[Any]] = {
            name: list(values) for name, values in columns.items()
        }
        new_columns[target_name] = encoded.tolist()
        return new_columns, None

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
        """Predict the highest-probability class label drawn from ``classes_``.

        Parameters
        ----------
        X : Any
            Serving table with the feature columns seen at fit time.

        Returns
        -------
        numpy.ndarray
            One-dimensional array of class labels, one per input row. The
            dtype matches ``classes_`` — strings, ``{-1, +1}`` ints, or
            ``{0, 1}`` ints all round-trip.

        Examples
        --------
        >>> clf.predict(X_test)[:5]
        array([1, 0, 1, 1, 0])
        """
        return self.classes_.take(np.argmax(self.predict_proba(X), axis=1))
