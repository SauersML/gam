"""GAMClassifier multiclass support (pyGAM audit, api.md F9).

pyGAM's ``LogisticGAM`` is binary-only, so multiclass users fake it with K
one-vs-rest fits whose probabilities do not sum to one. ``GAMClassifier`` used
to refuse ``K > 2`` outright although the engine carries a joint
multinomial-logit GAM (``K − 1`` linear predictors, a REML/LAML smoothing
parameter per (class, term), posterior-mean class probabilities). These tests
pin the classifier onto that model:

* known smooth class-probability surfaces are recovered on simulated 3- and
  5-class data, and every row of ``predict_proba`` sums to one;
* ``classes_`` and the ``predict_proba`` columns line up for any label dtype,
  including more than ten classes, where the engine's text order of the class
  index labels ("0", "1", "10", "2", ...) differs from numeric order;
* ``K = 2`` fit as ``family="multinomial"`` reproduces the binomial-logit fit;
* the sklearn contract (``predict`` labels, accuracy ``score``, column-name
  ``y``) holds for the multiclass model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gamfit.sklearn import GAMClassifier


def _softmax(eta: np.ndarray) -> np.ndarray:
    shifted = eta - eta.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


def _draw(rng: np.random.Generator, probabilities: np.ndarray) -> np.ndarray:
    cumulative = probabilities.cumsum(axis=1)
    u = rng.uniform(size=(probabilities.shape[0], 1))
    return (u > cumulative).sum(axis=1)


def _three_class_eta(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            np.zeros_like(x1),
            1.6 * np.sin(np.pi * x1) + 0.4 * x2,
            1.2 * x2**2 - 0.8 - 0.6 * x1,
        ]
    )


def _five_class_eta(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            np.zeros_like(x1),
            1.5 * np.sin(np.pi * x1),
            1.4 * x2,
            1.3 * np.cos(np.pi * x2) - 0.3,
            -1.2 * x1 + 0.8 * x1 * x1 - 0.2,
        ]
    )


def _frame(seed: int, n: int, eta_fn) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1.0, 1.0, n)
    x2 = rng.uniform(-1.0, 1.0, n)
    probabilities = _softmax(eta_fn(x1, x2))
    return pd.DataFrame({"x1": x1, "x2": x2}), _draw(rng, probabilities)


def _grid(eta_fn) -> tuple[pd.DataFrame, np.ndarray]:
    edge = np.linspace(-0.9, 0.9, 13)
    g1, g2 = np.meshgrid(edge, edge)
    x1, x2 = g1.ravel(), g2.ravel()
    return pd.DataFrame({"x1": x1, "x2": x2}), _softmax(eta_fn(x1, x2))


def _assert_simplex_rows(probabilities: np.ndarray, n_classes: int) -> None:
    assert probabilities.shape[1] == n_classes
    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0.0)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    "eta_fn, n_classes, n_rows, labels",
    [
        (_three_class_eta, 3, 1500, np.array(["setosa", "versicolor", "virginica"])),
        (_five_class_eta, 5, 2500, np.array([7, 30, 2, 100, 10])),
    ],
    ids=["three_class_strings", "five_class_ints"],
)
def test_multiclass_recovers_smooth_probability_surfaces(eta_fn, n_classes, n_rows, labels):
    X, index = _frame(seed=20260919 + n_classes, n=n_rows, eta_fn=eta_fn)
    y = labels[index]
    clf = GAMClassifier(formula="y ~ s(x1) + s(x2)").fit(X, y)

    assert list(clf.classes_) == sorted(labels.tolist())
    # Column j of predict_proba is classes_[j]: reorder the truth to match.
    truth_column = [labels.tolist().index(label) for label in clf.classes_]

    grid, truth = _grid(eta_fn)
    probabilities = clf.predict_proba(grid)
    _assert_simplex_rows(probabilities, n_classes)
    error = np.abs(probabilities - truth[:, truth_column])
    assert float(error.mean()) < 0.04, f"mean |p̂ − p| = {error.mean():.4f}"
    assert float(error.max()) < 0.2, f"max |p̂ − p| = {error.max():.4f}"

    # Training-set calibration: the average predicted probability of each class
    # matches its observed frequency.
    train_probabilities = clf.predict_proba(X)
    _assert_simplex_rows(train_probabilities, n_classes)
    for j, label in enumerate(clf.classes_):
        observed = float(np.mean(y == label))
        assert abs(float(train_probabilities[:, j].mean()) - observed) < 0.01

    predicted = clf.predict(grid)
    assert set(predicted.tolist()) <= set(labels.tolist())
    np.testing.assert_array_equal(predicted, clf.classes_[probabilities.argmax(axis=1)])


def test_classes_align_with_columns_beyond_ten_classes():
    """Eleven classes put the engine's text-ordered index labels ("0", "1",
    "10", "2", ...) out of numeric order, so the column map is exercised."""
    rng = np.random.default_rng(11)
    n_classes, n = 11, 2200
    x = rng.uniform(-1.0, 1.0, n)
    slopes = np.linspace(-3.0, 3.0, n_classes)
    probabilities = _softmax(np.outer(x, slopes))
    y = _draw(rng, probabilities) * 5  # labels 0, 5, ..., 50
    clf = GAMClassifier(formula="y ~ x").fit(pd.DataFrame({"x": x}), y)

    np.testing.assert_array_equal(clf.classes_, np.arange(n_classes) * 5)
    grid = np.array([-0.9, 0.0, 0.9])
    fitted = clf.predict_proba(pd.DataFrame({"x": grid}))
    _assert_simplex_rows(fitted, n_classes)
    truth = _softmax(np.outer(grid, slopes))
    # The most probable class at each end is the extreme slope's class; a
    # permuted column map would put the mass on the wrong labels.
    assert fitted[0].argmax() == 0 and fitted[2].argmax() == n_classes - 1
    assert np.abs(fitted - truth).max() < 0.1
    np.testing.assert_array_equal(clf.predict(pd.DataFrame({"x": [-0.9, 0.9]})), [0, 50])


def test_two_class_multinomial_matches_binomial_logit():
    rng = np.random.default_rng(2)
    n = 1200
    x1 = rng.uniform(-1.0, 1.0, n)
    x2 = rng.uniform(-1.0, 1.0, n)
    eta = 1.5 * np.sin(np.pi * x1) + x2
    y = np.where(rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta)), "b", "a")
    X = pd.DataFrame({"x1": x1, "x2": x2})

    binary = GAMClassifier(formula="y ~ s(x1) + s(x2)").fit(X, y)
    softmax = GAMClassifier(formula="y ~ s(x1) + s(x2)", family="multinomial").fit(X, y)

    np.testing.assert_array_equal(binary.classes_, softmax.classes_)
    grid, _ = _grid(lambda a, b: np.zeros((a.size, 1)))
    p_binary = binary.predict_proba(grid)
    p_softmax = softmax.predict_proba(grid)
    _assert_simplex_rows(p_softmax, 2)
    assert np.abs(p_binary - p_softmax).max() < 0.01


def test_multiclass_sklearn_contract_with_column_name_target():
    X, index = _frame(seed=5, n=900, eta_fn=_three_class_eta)
    labels = np.array(["lo", "mid", "hi"])
    frame = X.assign(label=labels[index])

    clf = GAMClassifier(formula="label ~ s(x1) + s(x2)").fit(frame, y="label")
    # The serving frame may still carry the original string labels.
    probabilities = clf.predict_proba(frame)
    _assert_simplex_rows(probabilities, 3)
    accuracy = clf.score(frame, frame["label"])
    assert accuracy == pytest.approx(float(np.mean(clf.predict(frame) == frame["label"])))
    assert accuracy > 0.5
    with pytest.raises(ValueError, match="binary classification panel"):
        clf.metrics(frame, frame["label"])


def test_binary_family_refuses_more_than_two_classes():
    X, index = _frame(seed=6, n=300, eta_fn=_three_class_eta)
    with pytest.raises(ValueError, match="family='multinomial'"):
        GAMClassifier(formula="y ~ s(x1)", family="binomial").fit(X, index)


def test_single_class_is_refused():
    X = pd.DataFrame({"x1": np.linspace(-1.0, 1.0, 50)})
    with pytest.raises(ValueError, match="at least two observed classes"):
        GAMClassifier(formula="y ~ s(x1)").fit(X, np.zeros(50, dtype=int))
