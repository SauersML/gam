"""Bug hunt: ``family="multinomial"`` rejects a categorical response whose class
labels are numeric strings (``"0"``, ``"1"``, ``"2"``, …) — the single most
common multiclass label encoding — with::

    InvalidInputError: multinomial fit: response 'y' is numeric, not
    categorical; use family='gaussian'/'binomial'/...

The identical model with the identical class structure but *alphabetic* labels
(``"a"``, ``"b"``, ``"c"``) fits and predicts a valid ``(n, K)`` probability
matrix. So the multinomial family is unusable on integer-coded classes purely
because of the *spelling* of the labels.

Root cause (same column-kind inference defect as #1317 / #1318): the response is
a *string* column, but ``src/inference/data.rs:649-657`` classifies a column as
``Categorical`` only when some row *fails* numeric parsing
(``kind = if all_numeric { Binary | Continuous } else { Categorical }``).
``"0","1","2"`` all parse as ``f64``, so the column is inferred ``Continuous`` /
``Binary``, and the multinomial fit path's "response must be categorical" guard
then rejects it. The categorical intent — that ``y`` is a *string* column — is
lost in stringification of the input table.

A response (or predictor) backed by a non-numeric (string/object) column should
be encoded as a factor regardless of whether its labels parse as numbers.

This test fits ``family="multinomial"`` on a 3-class softmax-in-``x`` dataset
with class labels ``"0","1","2"`` and asserts the fit succeeds and predicts a
well-posed probability matrix (shape ``(n, 3)``, rows summing to 1, calibrated
class frequencies). It currently fails at ``gamfit.fit(...)`` with the rejection
above. When a string-valued response is treated as categorical regardless of
label spelling, the fit succeeds and every assertion below holds without edits.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def test_multinomial_accepts_numeric_string_class_labels() -> None:
    rng = np.random.default_rng(20260618)
    n = 1500
    x = rng.uniform(-3.0, 3.0, n)
    # 3-class softmax in x.
    eta = np.stack([np.zeros_like(x), 1.0 * x, -0.8 * x + 0.5], axis=1)
    probs = np.exp(eta)
    probs /= probs.sum(axis=1, keepdims=True)
    cls = np.array([rng.choice(3, p=row) for row in probs])

    # Integer-coded class labels as a STRING column ("0"/"1"/"2"): the most
    # common multiclass encoding, and a genuinely categorical response.
    df = pd.DataFrame({"x": x, "y": cls.astype(str)})

    model = gamfit.fit(df, "y ~ s(x)", family="multinomial")
    pred = np.asarray(model.predict(df), dtype=float)

    assert pred.ndim == 2 and pred.shape == (n, 3), (
        f"multinomial prediction must be an (n, K) probability matrix, got {pred.shape}"
    )
    row_sums = pred.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), (
        f"predicted class probabilities must sum to 1, got range "
        f"[{row_sums.min():.6f}, {row_sums.max():.6f}]"
    )
    # Mean predicted class probability should match the empirical class frequency.
    for k in range(3):
        mean_pred = float(pred[:, k].mean())
        empirical = float(np.mean(cls == k))
        assert abs(mean_pred - empirical) < 0.05, (
            f"class {k}: mean predicted prob {mean_pred:.3f} vs empirical {empirical:.3f}"
        )
