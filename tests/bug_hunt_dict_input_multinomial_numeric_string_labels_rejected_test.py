"""Bug hunt: ``family="multinomial"`` HARD-REJECTS a response column of
numeric-string class labels (``"0"``, ``"1"``, ``"2"``) when the data is
supplied as a **dict of columns** (or list of records / numpy), even though the
identical labels fit fine as a pandas DataFrame, and even though *alphabetic*
string labels (``"a"``, ``"b"``, ``"c"``) fit fine through the very same dict
path.

Background
----------
Issue #1319 fixed this for the *typed* table path: a pandas/polars/pyarrow
string column whose class labels parse as numbers is now treated as categorical
and the multinomial fit accepts it. The fix is the categorical-dtype sentinel
in ``gamfit/_tables.py`` (``categorical_dtype_columns``, ``:110``), which only
inspects ``kind in {"pandas", "polars", "pyarrow"}`` and returns an EMPTY set
for every other container (``:152-157``: *"untyped inputs (mappings,
record/row sequences, numpy) return an empty set and keep the value-based
numeric inference"*).

So for a plain ``dict`` of columns — a documented, first-class input
(``README``: "dict of columns") — the response column ``y`` of string labels
``"0"``/``"1"``/``"2"`` is never marked categorical. Rust's column-kind
inference (``src/inference/data.rs`` ``all_numeric`` → numeric) then classifies
the response as numeric, and the multinomial family refuses it:

    InvalidInputError: multinomial fit: response 'y' is numeric, not
    categorical; use family='gaussian'/'binomial'/... or convert the column to
    a categorical

The suggested remedy ("convert the column to a categorical") is exactly what a
dict of *string* labels already expresses — there is no further conversion a
dict caller can apply, which is the #1319 complaint, still unfixed for dicts.

This is the multinomial-response sibling of #1467 (``by=`` factor smooth) and
#1468 (categorical main effect): the same dict/records/numpy gap in
``categorical_dtype_columns``, but a distinct code path (the multinomial
*response* classification) and a distinct failure mode — a HARD ERROR that
refuses the fit outright rather than a silent misfit.

The test fits a 3-class multinomial from a **dict** with numeric-string labels
and asserts the fit succeeds and returns valid probability rows. It fails today
(the fit raises ``InvalidInputError``). When ``categorical_dtype_columns`` also
marks all-string columns from dict / record / numpy inputs (the fix), the
response is categorical and the fit succeeds — no edits needed.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")

import gamfit


def _make_three_class(seed: int = 20260621):
    rng = np.random.default_rng(seed)
    n = 400
    x = rng.uniform(-2.0, 2.0, n)
    eta = np.stack([0.0 * x, 1.0 + 0.8 * x, -0.5 - 1.2 * x], axis=1)
    p = np.exp(eta)
    p /= p.sum(axis=1, keepdims=True)
    u = rng.uniform(size=n)
    yc = (u[:, None] < np.cumsum(p, axis=1)).argmax(axis=1)
    return x, yc


def test_dict_input_multinomial_accepts_numeric_string_labels() -> None:
    x, yc = _make_three_class()
    # Class labels as numeric STRINGS, supplied via a plain dict of columns.
    labels = [str(int(c)) for c in yc]
    data = {"x": list(map(float, x)), "y": labels}

    # Sanity anchor: alphabetic labels DO fit through the identical dict path,
    # so the dict container itself is fine — only the numeric-looking spelling
    # trips the value-based numeric inference.
    alpha = ["abc"[int(c)] for c in yc]
    control = gamfit.fit({"x": list(map(float, x)), "y": alpha}, "y ~ s(x)", family="multinomial")
    control_pred = np.asarray(control.predict({"x": np.array([-1.0, 0.0, 1.0])}), dtype=float)
    assert control_pred.shape == (3, 3), "control (alpha labels) should fit a 3-class model"

    # The bug: identical data with numeric-string labels is hard-rejected.
    model = gamfit.fit(data, "y ~ s(x)", family="multinomial")
    pred = np.asarray(model.predict({"x": np.array([-1.0, 0.0, 1.0])}), dtype=float)

    assert pred.shape == (3, 3), f"expected (3 rows, 3 classes), got {pred.shape}"
    # Valid probability rows: non-negative and summing to 1.
    assert np.all(pred >= -1e-9), "predicted probabilities must be non-negative"
    np.testing.assert_allclose(pred.sum(axis=1), 1.0, atol=1e-6)
