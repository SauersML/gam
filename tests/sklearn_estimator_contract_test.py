"""scikit-learn estimator contract for ``gamfit.sklearn`` (pyGAM audit F7-F18).

``parametrize_with_checks`` runs scikit-learn's full estimator check suite on
both wrappers; every check must pass. The targeted tests below pin the
individual contract points the audit found broken, each against the behaviour
scikit-learn itself documents.
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pytest

pytest.importorskip("gamfit._rust")
pytest.importorskip("sklearn")

from sklearn.base import clone
from sklearn.exceptions import DataConversionWarning, NotFittedError
from sklearn.model_selection import cross_val_score
from sklearn.utils.estimator_checks import parametrize_with_checks

import gamfit
from gamfit.sklearn import GAMClassifier, GAMRegressor

CONTRACT_FORMULA = "x0 + x1"


@parametrize_with_checks(
    [
        GAMRegressor(formula=CONTRACT_FORMULA),
        # The default family fits two classes as binomial and more as one
        # joint multinomial GAM; an explicit binary family declares itself
        # binary-only through `classifier_tags.multi_class`.
        GAMClassifier(formula=CONTRACT_FORMULA),
        GAMClassifier(formula=CONTRACT_FORMULA, family="binomial"),
    ]
)
def test_sklearn_estimator_checks(estimator, check, monkeypatch):
    # check_array_api_input runs for every estimator (NumPy namespace with
    # array-API dispatch on) but raises SkipTest unless SciPy's array-API flag
    # is set; set it so that check runs instead of being skipped.
    monkeypatch.setenv("SCIPY_ARRAY_API", "1")
    check(estimator)


def _regression_data(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    y = 1.5 * X[:, 0] - 0.5 * X[:, 1] + rng.normal(0.0, 0.3, size=n)
    return X, y


def _classification_data(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    eta = 2.0 * X[:, 0] - 1.0 * X[:, 1]
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(int)
    return X, y


# F7: sample_weight reaches the likelihood.


def test_regressor_sample_weight_is_the_gaussian_prior_weight():
    # A Gaussian prior weight is a precision, y_i ~ N(mu_i, phi / w_i), exactly
    # gamfit.fit's `weights=` column. It is not a replication count: a weight-k
    # row carries one -1/2 log(phi) where k copies carry k, so the profiled
    # scale (and with it the REML smoothing parameter) depends on the row count.
    X, y = _regression_data()
    w = np.random.default_rng(1).uniform(0.25, 3.0, size=y.size)
    weighted = GAMRegressor(formula=CONTRACT_FORMULA).fit(X, y, sample_weight=w)
    direct = gamfit.fit(
        {"x0": X[:, 0], "x1": X[:, 1], "y": y, "w": w}, "y ~ x0 + x1", weights="w"
    )
    np.testing.assert_allclose(
        weighted.predict(X), direct.predict({"x0": X[:, 0], "x1": X[:, 1]}), rtol=1e-12
    )
    unweighted = GAMRegressor(formula=CONTRACT_FORMULA).fit(X, y)
    assert not np.allclose(weighted.predict(X), unweighted.predict(X))


def test_classifier_integer_sample_weight_equals_repeated_rows():
    # The Bernoulli likelihood has unit scale and no normalizer, so a weight-k
    # row is exactly k copies of it: same likelihood, Fisher information and
    # REML criterion, hence the same fit.
    X, y = _classification_data()
    w = np.random.default_rng(1).integers(0, 4, size=y.size)
    weighted = GAMClassifier(formula=CONTRACT_FORMULA, family="binomial").fit(
        X, y, sample_weight=w.astype(float)
    )
    repeated = GAMClassifier(formula=CONTRACT_FORMULA, family="binomial").fit(
        np.repeat(X, w, axis=0), np.repeat(y, w)
    )
    unweighted = GAMClassifier(formula=CONTRACT_FORMULA, family="binomial").fit(X, y)
    # Both fits solve the same REML problem in rho; they differ only in where
    # each outer search stops inside its certified-stationary band.
    np.testing.assert_allclose(weighted.predict_proba(X), repeated.predict_proba(X), atol=1e-5)
    assert not np.allclose(weighted.predict_proba(X), unweighted.predict_proba(X))


def test_multiclass_integer_sample_weight_equals_repeated_rows():
    # The multinomial likelihood is a sum over rows with no scale, so a
    # weight-k row is k copies of it, as for the binomial.
    rng = np.random.default_rng(4)
    X = rng.uniform(-1.0, 1.0, size=(300, 2))
    eta = np.column_stack([np.zeros(300), 2.0 * X[:, 0], -2.0 * X[:, 1]])
    probabilities = np.exp(eta) / np.exp(eta).sum(axis=1, keepdims=True)
    y = np.array(["a", "b", "c"])[(rng.uniform(size=(300, 1)) > probabilities.cumsum(1)).sum(1)]
    w = rng.integers(0, 3, size=y.size)
    weighted = GAMClassifier(formula=CONTRACT_FORMULA).fit(X, y, sample_weight=w.astype(float))
    repeated = GAMClassifier(formula=CONTRACT_FORMULA).fit(
        np.repeat(X, w, axis=0), np.repeat(y, w)
    )
    np.testing.assert_array_equal(weighted.classes_, ["a", "b", "c"])
    # Same REML problem; the fits differ only inside the outer search's
    # certified-stationary band.
    np.testing.assert_allclose(weighted.predict_proba(X), repeated.predict_proba(X), atol=1e-4)


def test_automatic_formula_does_not_read_the_sample_weight_column():
    X, y = _regression_data()
    w = np.random.default_rng(2).uniform(0.5, 2.0, size=y.size)
    reg = GAMRegressor().fit(X, y, sample_weight=w)
    assert reg.formula_ == "y ~ s(x0) + s(x1)"


def test_sample_weight_routes_through_cross_val_score_params():
    X, y = _classification_data()
    w = np.ones(y.size)
    scores = cross_val_score(
        GAMClassifier(formula=CONTRACT_FORMULA, family="binomial"),
        X,
        y,
        cv=3,
        params={"sample_weight": w},
    )
    assert scores.shape == (3,)
    assert np.all((scores > 0.6) & (scores <= 1.0))


def test_all_zero_sample_weight_is_a_value_error_naming_the_weights():
    X, y = _regression_data()
    with pytest.raises(ValueError, match="non-zero weight"):
        GAMRegressor(formula=CONTRACT_FORMULA).fit(X, y, sample_weight=np.zeros(y.size))


def test_sample_weight_length_mismatch_is_a_value_error():
    X, y = _regression_data()
    with pytest.raises(ValueError, match="one weight per row"):
        GAMRegressor(formula=CONTRACT_FORMULA).fit(X, y, sample_weight=np.ones(y.size - 1))


# F8: score is accuracy for the classifier and R^2 for the regressor.


def test_classifier_score_is_accuracy():
    X, y = _classification_data()
    clf = GAMClassifier(formula=CONTRACT_FORMULA, family="binomial").fit(X, y)
    assert clf.score(X, y) == pytest.approx(np.mean(clf.predict(X) == y))


def test_regressor_score_is_r_squared():
    X, y = _regression_data()
    reg = GAMRegressor(formula=CONTRACT_FORMULA).fit(X, y)
    residual = np.sum((y - reg.predict(X)) ** 2)
    total = np.sum((y - y.mean()) ** 2)
    assert reg.score(X, y) == pytest.approx(1.0 - residual / total)


# F14: error types and (n, 1) targets.


def test_column_vector_target_is_ravelled_with_a_data_conversion_warning():
    X, y = _regression_data()
    with pytest.warns(DataConversionWarning):
        column = GAMRegressor(formula=CONTRACT_FORMULA).fit(X, y.reshape(-1, 1))
    flat = GAMRegressor(formula=CONTRACT_FORMULA).fit(X, y)
    np.testing.assert_allclose(column.predict(X), flat.predict(X))


def test_non_finite_input_is_a_value_error_at_fit_and_predict():
    X, y = _regression_data()
    bad = X.copy()
    bad[3, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        GAMRegressor(formula=CONTRACT_FORMULA).fit(bad, y)
    reg = GAMRegressor(formula=CONTRACT_FORMULA).fit(X, y)
    with pytest.raises(ValueError, match="NaN"):
        reg.predict(bad)


# F16: the fitted feature schema is enforced at predict.


def test_unnamed_predict_with_the_wrong_width_is_rejected():
    X, y = _regression_data()
    reg = GAMRegressor(formula=CONTRACT_FORMULA).fit(X, y)
    wide = np.column_stack([X, X])
    with pytest.raises(ValueError, match="X has 4 features, but GAMRegressor is expecting 2"):
        reg.predict(wide)
    with pytest.raises(ValueError, match="X has 1 features"):
        reg.predict(X[:, :1])


def test_named_predict_rejects_unseen_missing_and_reordered_columns():
    pd = pytest.importorskip("pandas")
    X, y = _regression_data()
    frame = pd.DataFrame({"a": X[:, 0], "b": X[:, 1]})
    reg = GAMRegressor(formula="a + b").fit(frame, y)
    np.testing.assert_array_equal(reg.feature_names_in_, ["a", "b"])
    with pytest.raises(ValueError, match="same order"):
        reg.predict(frame[["b", "a"]])
    with pytest.raises(ValueError, match="unseen at fit time"):
        reg.predict(frame.rename(columns={"b": "c"}))
    with pytest.raises(ValueError, match="now missing"):
        reg.predict(frame[["a"]])


def test_named_fit_then_unnamed_predict_binds_positionally_with_a_warning():
    pd = pytest.importorskip("pandas")
    X, y = _regression_data()
    frame = pd.DataFrame({"a": X[:, 0], "b": X[:, 1]})
    reg = GAMRegressor(formula="a + b").fit(frame, y)
    with pytest.warns(UserWarning, match="does not have valid feature names"):
        positional = reg.predict(X)
    np.testing.assert_allclose(positional, reg.predict(frame))


def test_unnamed_fit_does_not_set_feature_names_in():
    X, y = _regression_data()
    reg = GAMRegressor(formula=CONTRACT_FORMULA).fit(X, y)
    assert reg.n_features_in_ == 2
    assert not hasattr(reg, "feature_names_in_")


# F17: the remaining contract gaps, each with scikit-learn's own error.


@pytest.mark.parametrize(
    "estimator",
    [GAMRegressor(formula=CONTRACT_FORMULA), GAMClassifier(formula=CONTRACT_FORMULA)],
)
def test_unfitted_predict_raises_not_fitted_error(estimator):
    X, _ = _regression_data()
    with pytest.raises(NotFittedError):
        estimator.predict(X)


def test_sparse_input_is_rejected_as_sparse():
    sparse = pytest.importorskip("scipy.sparse")
    X, y = _regression_data()
    with pytest.raises(TypeError, match="[Ss]parse"):
        GAMRegressor(formula=CONTRACT_FORMULA).fit(sparse.csr_matrix(X), y)


def test_array_like_objects_are_accepted():
    class ArrayLike:
        def __init__(self, data):
            self._data = data

        def __array__(self, dtype=None, copy=None):
            return np.asarray(self._data, dtype=dtype)

    X, y = _regression_data()
    wrapped = GAMRegressor(formula=CONTRACT_FORMULA).fit(ArrayLike(X), y)
    plain = GAMRegressor(formula=CONTRACT_FORMULA).fit(X, y)
    np.testing.assert_allclose(wrapped.predict(ArrayLike(X)), plain.predict(X))


def test_continuous_classifier_target_is_an_unknown_label_type():
    X, y = _regression_data()
    with pytest.raises(ValueError, match="Unknown label type"):
        GAMClassifier(formula=CONTRACT_FORMULA, family="binomial").fit(X, y)


def test_missing_y_for_unnamed_input_says_y_is_required():
    X, _ = _regression_data()
    with pytest.raises(ValueError, match="requires y to be passed"):
        GAMRegressor(formula=CONTRACT_FORMULA).fit(X)


def test_complex_input_is_a_value_error():
    X, y = _regression_data()
    with pytest.raises(ValueError, match="[Cc]omplex"):
        GAMRegressor(formula=CONTRACT_FORMULA).fit(X + 1j, y)


# F18: plain BaseEstimator semantics.


def test_estimators_are_hashable_with_identity_equality_and_sklearn_repr():
    reg = GAMRegressor(formula="s(x0)")
    other = GAMRegressor(formula="s(x0)")
    assert len({reg, other}) == 2
    assert reg != other
    assert repr(reg) == "GAMRegressor(formula='s(x0)')"
    assert repr(GAMClassifier(formula="s(x0)", family="binomial")) == (
        "GAMClassifier(family='binomial', formula='s(x0)')"
    )


def test_fitted_estimator_clones_and_pickles():
    X, y = _classification_data()
    clf = GAMClassifier(formula=CONTRACT_FORMULA, family="binomial").fit(X, y)
    fresh = clone(clf)
    assert not hasattr(fresh, "model_")
    assert fresh.get_params() == clf.get_params()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        restored = pickle.loads(pickle.dumps(clf))
    np.testing.assert_allclose(restored.predict_proba(X), clf.predict_proba(X))
