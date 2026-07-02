"""Regression for issue #2075: GAMClassifier column-name fit + predict crash.

`GAMClassifier.fit(df, y="col")` (and the `y=None` formula-LHS form) label-
encodes the response column in-place to {0, 1} FOR FITTING, so the fitted Rust
model records that column in its schema as binary {0, 1}. At inference,
`predict` / `predict_proba` / `score` forward the caller's ORIGINAL frame
straight through — the response column still holds its un-encoded labels
(strings, {1, 2}, {-1, +1}). Those labels are validated against the {0, 1}
schema and rejected by the Rust `predict_table`, so the round-trip crashes with
a `GamError` for any non-{0, 1} label space. The response column is never
needed to predict.

This test fits `y ~ s(x)` via the column-name form on several label spaces and
asserts the sklearn serving round-trip on the training frame succeeds:
`predict` returns labels drawn from `classes_`, `predict_proba` rows sum to 1,
and `score` returns a float in [0, 1]. Pre-fix the three non-{0,1} cases raise
`GamError`; the {0, 1} case is a control that must pass on both sides of the
fix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gamfit.sklearn import GAMClassifier


def _make_frame(label_a: object, label_b: object) -> pd.DataFrame:
    """Training frame with feature `x` and response `y` in {label_a, label_b}.

    `label_b` is the positive class (larger under numpy's sort for the numeric
    cases; lexicographically larger for the string case chosen below), i.e.
    `classes_[1]`, matching the fit-time convention.
    """
    rng = np.random.default_rng(0)
    n = 200
    x = rng.uniform(-2.0, 2.0, n)
    positive = rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-x))
    y = np.where(positive, label_b, label_a)
    return pd.DataFrame({"x": x, "y": y})


@pytest.mark.parametrize(
    "label_a, label_b",
    [
        ("neg", "pos"),   # string labels
        (1, 2),           # {1, 2} integer labels
        (-1, 1),          # {-1, +1} integer labels
        (0, 1),           # {0, 1} control that must also pass
    ],
    ids=["strings", "one_two", "pm1", "zero_one_control"],
)
def test_non01_labels_predict_roundtrip(label_a: object, label_b: object) -> None:
    df = _make_frame(label_a, label_b)

    clf = GAMClassifier(formula="y ~ s(x)", family="binomial").fit(df, y="y")

    classes = set(np.asarray(clf.classes_).tolist())
    assert classes == {label_a, label_b}, (
        f"classes_ must reflect observed labels; got {clf.classes_!r}"
    )

    # predict on the SAME frame (which still carries the un-encoded response
    # column) must succeed and return labels drawn from classes_.
    predicted = np.asarray(clf.predict(df))
    assert set(np.unique(predicted).tolist()).issubset(classes), (
        f"predict must return labels from classes_; got {np.unique(predicted)!r}"
    )

    # predict_proba rows are a proper distribution over the two classes.
    proba = clf.predict_proba(df)
    assert proba.shape == (len(df), 2)
    row_sums = proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-9), (
        f"predict_proba rows must sum to 1; got sums {row_sums!r}"
    )

    # score forwards the same frame; it must return a float AUC in [0, 1].
    auc = clf.score(df, df["y"])
    assert isinstance(auc, float)
    assert 0.0 <= auc <= 1.0, f"score must be in [0, 1]; got {auc!r}"
