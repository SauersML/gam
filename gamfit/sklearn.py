from __future__ import annotations

import warnings
from typing import Any, TypeVar

import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import (
    check_array,
    check_consistent_length,
    check_is_fitted,
    column_or_1d,
)

from ._binding import rust_module
from ._api import fit as fit_model
from ._api import is_multinomial_family
from ._model import Model, MultinomialModel
from ._tables import (
    _table_column_views,
    drop_columns,
    table_column_names,
    table_row_count,
    with_columns,
)

__all__ = ["GAMClassifier", "GAMRegressor"]

_BaseT = TypeVar("_BaseT", bound="_BaseGAMEstimator")

# No GAMSurvival wrapper: survival responses (e.g. Surv(time, event)) are a
# two-column construct that does not fit scikit-learn's (X, y) contract, and
# survival prediction is a per-time-grid hazard surface rather than a single
# response vector. Users who want a scikit-style API for survival should call
# gamfit.fit(...) directly with family="cox" (or equivalent) and operate on
# the SurvivalPrediction object returned by Model.predict.


def _positional_columns(X: Any) -> dict[str, Any]:
    """Validate an unnamed ``X`` and bind its columns as ``x0, x1, ...``.

    Unnamed inputs (numpy arrays, nested lists, ``__array__`` objects, frames
    without string column labels) carry no schema, so they must be numeric;
    scikit-learn's own validator rejects sparse, complex, empty, and
    non-finite input with the messages the estimator contract expects.
    """
    array = check_array(X, dtype="numeric", input_name="X")
    columns, _kind = _table_column_views(array)
    return columns


def _input_column_names(X: Any) -> list[str] | None:
    """Column names of a named ``X``, or ``None`` for positional input.

    Sparse matrices are positional whatever their container (``dok_matrix``
    is a ``dict``), so they reach :func:`_positional_columns` and its
    "dense data is required" rejection.
    """
    if issparse(X):
        return None
    return table_column_names(X)


class _BaseGAMEstimator(BaseEstimator):
    feature_names_in_: np.ndarray

    def __init__(
        self,
        formula: str | None = None,
        family: str = "auto",
        offset: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.formula = formula
        self.family = family
        self.offset = offset
        self.config = config

    def _prepare_target(self, y: Any) -> tuple[np.ndarray, str]:
        """Validate the target and return it with the family to fit it under."""
        raise NotImplementedError

    def _fit_model(self: _BaseT, X: Any, y: Any, sample_weight: Any) -> _BaseT:
        names = _input_column_names(X)
        if names is None:
            table: Any = _positional_columns(X)
            columns = list(table)
        else:
            table = X
            columns = names
        if y is None and names is None:
            raise ValueError(
                f"{type(self).__name__} requires y to be passed, but the target y is None."
            )
        target_column = y if isinstance(y, str) else None
        external_target = y is not None and target_column is None
        fit_formula, feature_names, target_name, weight_column = (
            rust_module().sklearn_fit_metadata(
                columns,
                self.formula,
                target_column,
                external_target,
                sample_weight is not None,
            )
        )
        if external_target:
            target = column_or_1d(y, warn=True)
            if names is None:
                check_consistent_length(table[columns[0]], target)
            self._response_column = None
        else:
            target = column_or_1d(_table_column_views(table)[0][target_name])
            self._response_column = target_name
        encoded, family = self._prepare_target(target)
        extra: dict[str, Any] = {target_name: encoded}
        if weight_column is not None:
            weights = np.asarray(sample_weight, dtype=np.float64)
            if weights.shape != (table_row_count(table),):
                raise ValueError(
                    f"sample_weight has shape {weights.shape}, but X has "
                    f"{table_row_count(table)} rows; pass one weight per row"
                )
            extra[weight_column] = weights
        self.model_ = fit_model(
            with_columns(table, extra),
            fit_formula,
            family=family,
            offset=self.offset,
            weights=weight_column,
            config=self.config,
        )
        # The fitted formula, with an automatic `.` expanded by the engine.
        self.formula_ = getattr(self.model_, "formula", fit_formula)
        self.n_features_in_ = len(feature_names)
        if names is None:
            if hasattr(self, "feature_names_in_"):
                del self.feature_names_in_
        else:
            self.feature_names_in_ = np.asarray(feature_names, dtype=object)
        return self

    def _serving_table(self, X: Any) -> Any:
        """Validate ``X`` against the fitted features and bind it for prediction.

        Named inputs must carry exactly the fitted feature names in the fitted
        order; the fit-time response column, when ``X`` carried it at fit, is
        dropped first because it is never needed to predict. Unnamed inputs
        must have ``n_features_in_`` columns and bind positionally.
        """
        check_is_fitted(self, "model_")
        estimator = type(self).__name__
        names = _input_column_names(X)
        if names is not None and self._response_column in names:
            X = drop_columns(X, [self._response_column])
            names = [name for name in names if name != self._response_column]
        fitted = getattr(self, "feature_names_in_", None)
        if names is not None and fitted is not None:
            expected = list(fitted)
            if names != expected:
                raise ValueError(_feature_name_mismatch(names, expected))
            return X
        if names is not None:
            warnings.warn(
                f"X has feature names, but {estimator} was fitted without feature names",
                UserWarning,
                stacklevel=3,
            )
        columns = _positional_columns(X)
        if len(columns) != self.n_features_in_:
            raise ValueError(
                f"X has {len(columns)} features, but {estimator} is expecting "
                f"{self.n_features_in_} features as input."
            )
        if fitted is None:
            return columns
        if names is None:
            warnings.warn(
                f"X does not have valid feature names, but {estimator} was fitted "
                "with feature names",
                UserWarning,
                stacklevel=3,
            )
        return dict(zip(fitted, columns.values()))

    def _posterior_mean(self, X: Any) -> np.ndarray:
        table = self._serving_table(X)
        predicted = self.model_.predict(table, return_type="dict")
        return np.asarray(predicted["posterior_mean"], dtype=float)

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


def _feature_name_mismatch(names: list[str], expected: list[str]) -> str:
    message = "The feature names should match those that were passed during fit.\n"
    unseen = [name for name in names if name not in expected]
    missing = [name for name in expected if name not in names]
    if unseen:
        message += f"Feature names unseen at fit time: {unseen}\n"
    if missing:
        message += f"Feature names seen at fit time, yet now missing: {missing}\n"
    if not unseen and not missing:
        message += "Feature names must be in the same order as they were in fit.\n"
    return message


class GAMRegressor(RegressorMixin, _BaseGAMEstimator):
    """scikit-learn-compatible regressor wrapping :func:`gamfit.fit`.

    Construct with an optional formula string and pipeline kwargs such as
    ``family``, ``offset``, or a free-form ``config`` dict, then call
    :meth:`fit` with either a fully-formed table (``X``) or a feature table
    plus a target column / vector (``y``). After fitting, the estimator
    exposes the standard ``predict`` / ``score`` interface plus pass-through
    helpers :meth:`summary`, :meth:`report`, and :meth:`check` from the
    underlying :class:`Model`.

    Parameters
    ----------
    formula : str or None, default ``None``
        Wilkinson-style formula. May or may not include the response on the
        left-hand side; the response is resolved from ``y`` if missing.
        Unnamed inputs (numpy arrays, nested lists) expose their columns to
        the formula as ``x0``, ``x1``, ... ``None`` fits the automatic formula
        ``y ~ .``: the engine builds one term per feature column from its
        schema (``s(x)`` for numeric columns with at least three distinct
        values, a linear term for two-valued numeric and boolean columns,
        ``factor(g)`` for categorical and string columns, constant columns
        dropped with a warning). Every such term is penalized and can shrink
        to zero; the formula actually fitted is ``formula_`` after
        :meth:`fit`.
    family : str, default ``"auto"``
        Likelihood family forwarded to :func:`gamfit.fit`.
    offset : str or None, optional
        Offset column name, forwarded to :func:`gamfit.fit`.
    config : dict or None, optional
        Escape-hatch dict of extra pipeline keys.

    Examples
    --------
    >>> from gamfit.sklearn import GAMRegressor
    >>> reg = GAMRegressor(formula="y ~ s(x1) + s(x2)").fit(X_train, y_train)
    >>> GAMRegressor().fit(X_train, y_train).formula_
    'y ~ s(x1) + s(x2)'
    >>> preds = reg.predict(X_test)
    >>> reg.score(X_test, y_test)
    0.87
    """

    def __sklearn_tags__(self) -> Any:
        # scikit-learn-stubs does not declare the tags protocol (scikit-learn
        # >= 1.6) on its mixins.
        tags = super().__sklearn_tags__()  # type: ignore[misc]
        # The formula fixes which columns the model reads, so an informative
        # column the formula does not name is invisible to it by construction;
        # scikit-learn's fixed-width scoring dataset (one informative column of
        # ten) cannot be scored against a formula written for other data.
        tags.regressor_tags.poor_score = True
        return tags

    def fit(self, X: Any, y: Any = None, sample_weight: Any = None) -> "GAMRegressor":
        """Fit the underlying GAM and return ``self``.

        Parameters
        ----------
        X : Any
            Training table (pandas / polars DataFrame, pyarrow Table, dict of
            columns, list of records) or a numeric array whose columns the
            formula names ``x0``, ``x1``, ... A table may include the response
            column or not.
        y : str, array-like, or None, optional
            Target. ``str`` names a column already in ``X``; an array-like is
            bound to ``X`` under the response name implied by ``formula``;
            ``None`` means the table ``X`` already contains the response named
            by ``formula`` (which then must be given).
        sample_weight : array-like of shape (n_samples,), optional
            Per-row prior weights of the likelihood (the precision weights of
            :func:`gamfit.fit`'s ``weights`` column).

        Returns
        -------
        GAMRegressor
            Fitted estimator (``self``) with ``model_``, ``formula_``, and
            ``n_features_in_`` set, plus ``feature_names_in_`` when ``X``
            names its columns.

        Examples
        --------
        >>> GAMRegressor(formula="y ~ s(x)").fit(df, y="y")
        """
        return self._fit_model(X, y, sample_weight)

    def _prepare_target(self, y: Any) -> tuple[np.ndarray, str]:
        target = check_array(y, ensure_2d=False, dtype="numeric", input_name="y")
        return target, self.family

    def predict(self, X: Any) -> np.ndarray:
        """Predict the conditional mean for each row in ``X``.

        Parameters
        ----------
        X : Any
            Serving input with the features seen at fit time: the same
            column names (in the same order) for a named table, or the same
            number of columns for an unnamed array.

        Returns
        -------
        numpy.ndarray
            One-dimensional float array of predicted means, one per row.

        Examples
        --------
        >>> reg.predict(X_test)[:3]
        array([1.02, 0.98, 1.41])
        """
        return self._posterior_mean(X)


class GAMClassifier(ClassifierMixin, _BaseGAMEstimator):
    """scikit-learn-compatible classifier wrapping :func:`gamfit.fit`.

    Same construction and ``fit`` semantics as :class:`GAMRegressor` (see that
    class for parameter documentation). The supplied ``y`` may be any label
    vector (strings, ``{-1, +1}``, ``{1, 2}``, integer codes, …); the wrapper
    records the observed classes in ``classes_`` (sorted, as sklearn requires),
    ``predict`` returns labels drawn from ``classes_``, and ``score`` is
    accuracy.

    * **Two classes** fit the binomial-logit GAM, with the positive class
      ``classes_[1]`` label-encoded to ``1``.
    * **Three or more classes** (or ``family="multinomial"`` at any class
      count) fit one joint multinomial-logit GAM: ``K − 1`` linear predictors,
      each with its own smooths and REML/LAML-selected smoothing parameters,
      estimated together by penalized likelihood. It is a single model, not
      ``K`` one-vs-rest binary fits, so its class probabilities are coherent
      by construction.

    ``predict_proba`` returns the ``(n, K)`` posterior-mean class
    probabilities with column ``j`` aligned to ``classes_[j]``; rows sum to 1.

    Examples
    --------
    >>> from gamfit.sklearn import GAMClassifier
    >>> clf = GAMClassifier(formula="y ~ s(x1) + s(x2)", family="binomial")
    >>> clf.fit(X_train, y_train)
    >>> clf.predict_proba(X_test)[:1]
    array([[0.34, 0.66]])
    >>> clf3 = GAMClassifier(formula="y ~ s(x1) + s(x2)").fit(X_train, species)
    >>> clf3.predict_proba(X_test).shape
    (100, 3)
    """

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()  # type: ignore[misc]
        # Any family other than "auto" and the multinomial names a binary
        # likelihood, which refuses a third class.
        tags.classifier_tags.multi_class = self.family == "auto" or is_multinomial_family(
            self.family
        )
        return tags

    def fit(self, X: Any, y: Any = None, sample_weight: Any = None) -> "GAMClassifier":
        """Fit the GAM classifier and return ``self``.

        Parameters
        ----------
        X : Any
            Training input. See :meth:`GAMRegressor.fit` for accepted forms.
        y : str, array-like, or None, optional
            Class labels. See :meth:`GAMRegressor.fit` for accepted forms.
        sample_weight : array-like of shape (n_samples,), optional
            Per-row prior weights of the classification likelihood.

        Returns
        -------
        GAMClassifier
            Fitted estimator (``self``) with ``classes_`` holding the observed
            labels sorted ascending (for two classes the positive class is
            ``classes_[1]``).

        Examples
        --------
        >>> GAMClassifier(formula="y ~ s(x)", family="binomial").fit(df, y="y")
        """
        self._fit_model(X, y, sample_weight)
        self._multinomial_columns_: np.ndarray | None = None
        if isinstance(self.model_, MultinomialModel):
            # The multinomial response is categorical and the engine orders its
            # levels by label text. Each row carries its class INDEX as the
            # label, and the engine's level order is mapped back to index
            # order, so column j of `predict_proba` is `classes_[j]` whatever
            # the label dtype.
            engine_levels = np.asarray([int(level) for level in self.model_.classes_])
            self._multinomial_columns_ = np.argsort(engine_levels)
        return self

    def _is_multinomial(self, n_classes: int) -> bool:
        return n_classes > 2 or is_multinomial_family(self.family)

    def _prepare_target(self, y: Any) -> tuple[np.ndarray, str]:
        check_classification_targets(y)
        classes, class_index = np.unique(y, return_inverse=True)
        if classes.size < 2:
            raise ValueError(
                "GAMClassifier requires at least two observed classes; "
                f"got {classes.size} class{'' if classes.size == 1 else 'es'}: {classes!r}"
            )
        self.classes_ = classes
        if not self._is_multinomial(classes.size):
            # Positive class is the second sorted label, matching the
            # convention used by sklearn's LabelEncoder and LogisticRegression.
            return (class_index == 1).astype(np.float64), self.family
        if self.family != "auto" and not is_multinomial_family(self.family):
            raise ValueError(
                "Only binary classification is supported by "
                f"GAMClassifier(family={self.family!r}); y has {classes.size} classes. "
                "Use family='auto' or family='multinomial' for a joint "
                "multinomial-logit GAM."
            )
        encoded = np.asarray([str(index) for index in class_index], dtype=object)
        return encoded, "multinomial"

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities for each row in ``X``.

        Parameters
        ----------
        X : Any
            Serving input with the features seen at fit time.

        Returns
        -------
        numpy.ndarray
            ``(n, K)`` float array of posterior-mean class probabilities;
            column ``j`` is ``P(y = classes_[j])`` and every row sums to 1.

        Examples
        --------
        >>> clf.predict_proba(X_test).shape
        (100, 2)
        """
        if getattr(self, "_multinomial_columns_", None) is not None:
            table = self._serving_table(X)
            probabilities = np.asarray(self.model_.predict(table), dtype=float)
            ordered: np.ndarray = probabilities[:, self._multinomial_columns_]
            return ordered
        positive = self._posterior_mean(X)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: Any) -> np.ndarray:
        """Predict the highest-probability class label drawn from ``classes_``.

        Parameters
        ----------
        X : Any
            Serving input with the features seen at fit time.

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
        probabilities = self.predict_proba(X)
        labels: np.ndarray = self.classes_.take(np.argmax(probabilities, axis=1))
        return labels

    def metrics(self, X: Any, y: Any) -> dict[str, float]:
        """Classification-metric panel for ``X`` against true labels ``y``.

        Surfaces the Rust ``classification_metrics`` routine on the model's
        positive-class probabilities: ``auc``, ``pr_auc``, ``brier``,
        ``logloss``, ``nagelkerke_r2`` (relative to the observed base rate),
        and ``ece``. ``y`` may carry the original label dtype (strings,
        ``{-1, +1}``, ``{0, 1}``, …); it is encoded against :attr:`classes_`
        exactly as at fit time so the positive class is ``classes_[1]``.

        Parameters
        ----------
        X : Any
            Serving input with the features seen at fit time.
        y : array-like
            True labels, drawn from :attr:`classes_`.

        Returns
        -------
        dict of str to float
            The classification-metric panel.

        Examples
        --------
        >>> clf.metrics(X_test, y_test)["auc"]
        0.91
        """
        check_is_fitted(self, "model_")
        if self._multinomial_columns_ is not None:
            raise ValueError(
                "GAMClassifier.metrics() is the binary classification panel; "
                f"this model has {self.classes_.size} classes"
            )
        observed = self._encode_labels(y)
        positive = self.predict_proba(X)[:, 1].astype(float)
        return dict(
            rust_module().classification_metrics(observed.tolist(), positive.tolist())
        )

    def _encode_labels(self, y: Any) -> np.ndarray:
        """Encode ``y`` to ``{0, 1}`` against :attr:`classes_`.

        Mirrors the fit-time convention (positive class is ``classes_[1]``)
        so metric inputs line up with the model's positive-class probability.
        Labels not present in :attr:`classes_` raise, rather than silently
        scoring against a phantom class.
        """
        check_is_fitted(self, "model_")
        arr = column_or_1d(y)
        positive = self.classes_[1]
        negative = self.classes_[0]
        is_positive: np.ndarray = arr == positive
        is_negative: np.ndarray = arr == negative
        unknown = ~(is_positive | is_negative)
        if np.any(unknown):
            raise ValueError(
                "GAMClassifier scoring received labels outside classes_="
                f"{self.classes_!r}: {np.unique(arr[unknown])!r}"
            )
        return is_positive.astype(int)
