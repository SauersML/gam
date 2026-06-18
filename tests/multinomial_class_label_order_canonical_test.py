"""Regression guard (#1319, different angle): a multinomial response's class
order must be a deterministic function of the *label set*, not of training row
order.

The original #1319 symptom was that a numeric-string response (``"0","1","2"``)
was rejected outright. The deeper defect exposed once acceptance was restored is
that categorical levels were recorded in *first-appearance* order: the ``(n, K)``
prediction column layout (and the softmax reference class = last level) then
depended on which class happened to appear first in the training rows. A mere
row shuffle would permute the output columns even though the data is identical.

Canonical factor ordering — what R ``factor()`` (C-locale sort), pandas
``Categorical`` and sklearn ``LabelEncoder`` all do — sorts the level set. These
tests assert that contract from angles the calibration test does not cover:

* ``classes_`` is the *sorted* label set for numeric-string labels, so column
  ``k`` is class ``classes_[k]``;
* the predicted probability for a given *label* is invariant to the training row
  order (shuffling rows must not permute or perturb the per-class columns);
* alphabetic labels follow the same sorted contract, so the encoding does not
  depend on the spelling of the labels.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def _softmax_dataset(seed: int, n: int = 1500):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, n)
    eta = np.stack([np.zeros_like(x), 1.0 * x, -0.8 * x + 0.5], axis=1)
    probs = np.exp(eta)
    probs /= probs.sum(axis=1, keepdims=True)
    cls = np.array([rng.choice(3, p=row) for row in probs])
    return x, cls


def test_classes_are_sorted_for_numeric_string_labels() -> None:
    x, cls = _softmax_dataset(seed=20260618)
    df = pd.DataFrame({"x": x, "y": cls.astype(str)})
    model = gamfit.fit(df, "y ~ s(x)", family="multinomial")
    # Canonical (sorted) factor order, independent of which class appeared first.
    assert list(model.classes_) == ["0", "1", "2"], (
        f"multinomial classes_ must be the sorted label set, got {model.classes_!r}"
    )
    pred = np.asarray(model.predict(df), dtype=float)
    # Column k aligns with classes_[k] == class k, so mean column probability
    # tracks the empirical class frequency.
    for k in range(3):
        assert abs(float(pred[:, k].mean()) - float(np.mean(cls == k))) < 0.05


def test_prediction_is_invariant_to_training_row_order() -> None:
    """Shuffling the training rows must not permute or change the per-class
    output columns — the defining property first-appearance ordering violated."""
    x, cls = _softmax_dataset(seed=7)
    df = pd.DataFrame({"x": x, "y": cls.astype(str)})

    rng = np.random.default_rng(123)
    perm = rng.permutation(len(x))
    df_shuffled = df.iloc[perm].reset_index(drop=True)

    m0 = gamfit.fit(df, "y ~ s(x)", family="multinomial")
    m1 = gamfit.fit(df_shuffled, "y ~ s(x)", family="multinomial")

    assert list(m0.classes_) == list(m1.classes_) == ["0", "1", "2"]

    # Predict on the SAME ordered grid from both models; columns must align
    # class-for-class and agree to fit tolerance.
    grid = pd.DataFrame({"x": np.linspace(-3.0, 3.0, 200)})
    p0 = np.asarray(m0.predict(grid), dtype=float)
    p1 = np.asarray(m1.predict(grid), dtype=float)
    assert p0.shape == p1.shape == (200, 3)
    max_abs = float(np.max(np.abs(p0 - p1)))
    assert max_abs < 1e-3, (
        f"row-order shuffle changed the per-class probability surface by "
        f"{max_abs:.2e}; class columns are not row-order invariant"
    )


def test_alpha_labels_follow_same_sorted_contract() -> None:
    x, cls = _softmax_dataset(seed=20260618)
    labels = np.array(["alpha", "beta", "gamma"])[cls]
    df = pd.DataFrame({"x": x, "y": labels})
    model = gamfit.fit(df, "y ~ s(x)", family="multinomial")
    assert list(model.classes_) == ["alpha", "beta", "gamma"], (
        f"classes_ must be the sorted label set regardless of spelling, "
        f"got {model.classes_!r}"
    )
    pred = np.asarray(model.predict(df), dtype=float)
    # "alpha"->class0, "beta"->class1, "gamma"->class2 by construction; sorted
    # order preserves that mapping, so column k tracks empirical frequency of k.
    for k in range(3):
        assert abs(float(pred[:, k].mean()) - float(np.mean(cls == k))) < 0.05
