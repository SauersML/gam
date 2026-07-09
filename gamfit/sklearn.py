from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from ._binding import rust_module
from ._api import fit as fit_model
from ._model import Model
from ._tables import attach_target, detect_table_kind, table_columns

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
            # `has_external_target=False`: the formula's LHS names a column
            # already present in `X`, and `sklearn_fit_metadata` returns that
            # name in the third slot. Passing `True` here would (correctly)
            # error out because the target column is in fact in `X`.
            rust = rust_module()
            columns, _ = table_columns(X)
            _, _, target_name = rust.sklearn_fit_metadata(
                list(columns), self.formula, None, False,
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
            # Array-style `y`: the target is external to the serving frame, so
            # inference never needs to strip a response column from `X`.
            self._encoded_target_name_ = None
            return X, encoded
        columns, _ = table_columns(X)
        if isinstance(y, str):
            target_name = y
        else:
            # y is None: target column already lives in X under the formula's
            # LHS name. `has_external_target=False` matches that state; True
            # would (correctly) refuse the call because the target is in X.
            rust = rust_module()
            _, _, target_name = rust.sklearn_fit_metadata(
                list(columns), self.formula, None, False,
            )
        # Cache the resolved response-column name so inference can drop it from
        # the caller's serving frame. Fitting label-encodes this column to
        # {0,1} and records that {0,1} schema in the Rust model; a serving frame
        # that still carries the ORIGINAL (string / {1,2} / {-1,+1}) labels
        # would be validated against that {0,1} schema and rejected. The column
        # is never needed to predict, so we strip it (see
        # `_strip_response_column`).
        self._encoded_target_name_ = target_name
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
        serving = self._strip_response_column(X)
        predicted = self.model_.predict(serving, return_type="dict")
        positive = np.clip(np.asarray(predicted["mean"], dtype=float), 0.0, 1.0)
        negative = 1.0 - positive
        return np.column_stack([negative, positive])

    def _strip_response_column(self, X: Any) -> Any:
        """Return ``X`` with the fit-time response column removed if present.

        The column-name (``y="col"``) and ``y=None`` fit paths label-encode the
        response to ``{0, 1}`` inside the fitted table, so the Rust model's
        schema records that column as binary ``{0, 1}``. The response is never
        needed to predict, yet callers naturally re-serve the SAME frame (which
        still holds the ORIGINAL string / ``{1, 2}`` / ``{-1, +1}`` labels);
        validated against the ``{0, 1}`` schema those labels are rejected with a
        ``GamError``. Dropping the column keeps inference consistent with
        fitting. The carrier kind is preserved so downstream dtype-based
        categorical inference is unchanged; only the response column goes.

        A no-op when no response column was cached (array-``y`` fits) or when
        the column is absent from the serving frame (an ``X`` that never carried
        the response), so the already-working ``{0, 1}`` and array-``y`` paths
        are untouched.
        """
        name = getattr(self, "_encoded_target_name_", None)
        if name is None:
            return X
        kind = detect_table_kind(X)
        if kind == "pandas":
            matches = [c for c in X.columns if str(c) == name]
            return X.drop(columns=matches) if matches else X
        if kind == "polars":
            matches = [c for c in X.columns if str(c) == name]
            return X.drop(matches) if matches else X
        if kind == "pyarrow":
            matches = [c for c in X.column_names if str(c) == name]
            return X.drop_columns(matches) if matches else X
        if isinstance(X, Mapping):
            if any(str(key) == name for key in X):
                return {
                    key: value for key, value in X.items() if str(key) != name
                }
            return X
        return X

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
            Serving table with the feature columns seen at fit time.
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
        observed = self._encode_labels(y)
        positive = self.predict_proba(X)[:, 1].astype(float)
        train_prev = float(np.mean(observed)) if observed.size else 0.0
        return dict(
            rust_module().classification_metrics(
                observed.tolist(),
                positive.tolist(),
                train_prev,
            )
        )

    def score(self, X: Any, y: Any, sample_weight: Any | None = None) -> float:
        """Area under the ROC curve (AUC) of the model on ``(X, y)``.

        Overrides scikit-learn's default accuracy score with AUC, the natural
        threshold-free discrimination metric for the probabilistic output of
        a GAM classifier. ``sample_weight`` is honoured as a genuine weighted
        Mann-Whitney statistic computed in the Rust core: each positive /
        negative pair ``(i, j)`` contributes with pair weight ``w_i * w_j``
        (ties counted half), exactly as :func:`sklearn.metrics.roc_auc_score`
        weights pairs. Weight magnitudes therefore matter — a row with weight
        100 dominates a row with weight 1 — rather than merely selecting
        which rows participate.

        Parameters
        ----------
        X : Any
            Serving table with the feature columns seen at fit time.
        y : array-like
            True labels, drawn from :attr:`classes_`.
        sample_weight : array-like or None, optional
            Per-row non-negative weights. ``None`` (default) weights every
            row equally.

        Returns
        -------
        float
            AUC in ``[0, 1]``; ``0.5`` is chance, ``1.0`` is perfect ranking.

        Examples
        --------
        >>> clf.score(X_test, y_test)
        0.91
        """
        check_is_fitted(self, "model_")
        observed = self._encode_labels(y)
        positive = self.predict_proba(X)[:, 1].astype(float)
        if sample_weight is None:
            return float(
                rust_module().auc_from_predictions(observed.tolist(), positive.tolist())
            )
        weights = np.asarray(sample_weight, dtype=float).reshape(-1)
        if weights.shape[0] != observed.shape[0]:
            raise ValueError(
                "GAMClassifier.score: sample_weight has length "
                f"{weights.shape[0]} but y has {observed.shape[0]} rows"
            )
        return float(
            rust_module().weighted_auc_from_predictions(
                observed.tolist(), positive.tolist(), weights.tolist()
            )
        )

    def _encode_labels(self, y: Any) -> np.ndarray:
        """Encode ``y`` to ``{0, 1}`` against :attr:`classes_`.

        Mirrors the fit-time convention (positive class is ``classes_[1]``)
        so metric inputs line up with the model's positive-class probability.
        Labels not present in :attr:`classes_` raise, rather than silently
        scoring against a phantom class.
        """
        arr = np.asarray(y).reshape(-1)
        positive = self.classes_[1]
        negative = self.classes_[0]
        is_positive = arr == positive
        is_negative = arr == negative
        unknown = ~(is_positive | is_negative)
        if np.any(unknown):
            raise ValueError(
                "GAMClassifier scoring received labels outside classes_="
                f"{self.classes_!r}: {np.unique(arr[unknown])!r}"
            )
        return is_positive.astype(int)
