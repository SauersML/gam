"""Bug hunt: a categorical main effect ``y ~ g`` is silently fit as a single
*linear numeric* term (one slope) instead of a treatment-coded factor when the
(string) grouping column's level labels are numeric strings (``"0"``, ``"1"``,
``"2"``, …).

A factor predictor must contribute one coefficient per level (treatment-coded
against a reference), so ``y ~ g`` can represent **arbitrary** per-group means.
With non-numeric labels (``"a"``, ``"b"``, …) gamfit does exactly that and
recovers any set of group means. But when the labels look numeric, the column is
inferred as a numeric column and ``y ~ g`` becomes a single linear term
``β·value(g)`` — the fitted per-group means are forced onto a straight line in
the integer label, so any non-monotone (or simply non-linear-in-label) set of
group means is unrecoverable.

Root cause (same column-kind inference defect as #1317): the grouping column is
a *string* column, but ``src/inference/data.rs:649-657`` classifies a column as
``Categorical`` only when some row *fails* numeric parsing
(``kind = if all_numeric { Binary | Continuous } else { Categorical }``).
``"0","1","2"`` all parse, so the column is ``Continuous`` / ``Binary`` and the
parametric-term builder wires it as a single numeric linear column rather than a
treatment-coded factor expansion. The user's categorical intent — encoded in the
fact that ``g`` is a *string* column — is lost in stringification.

The result is a silent, data-dependent corruption that depends only on the
*spelling* of the level labels: renaming ``"0"`` → ``"a"`` changes the fit,
violating the invariance that a model must not depend on the names of
categorical levels.

This test fits ``y ~ g`` on four groups with strongly non-monotone true means
``[2, -3, 8, 1]`` labeled ``"0".."3"`` and asserts the fitted per-group means
recover the truth. It currently fails: the integer-labeled fit collapses to a
linear ramp (``[0.88, 1.67, 2.45, 3.23]``, max error ≈ 5.5). When a ``by=``/main
factor over a string-valued column is treated as categorical regardless of
whether the labels look numeric, the per-group means are recovered and the
assertion holds without edits.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def test_categorical_main_effect_recovers_nonmonotone_group_means() -> None:
    rng = np.random.default_rng(20260618)
    n = 1200
    gi = rng.integers(0, 4, n)
    # Strongly non-monotone in the integer label: a single linear-in-label term
    # cannot fit these, a treatment-coded factor can.
    true_means = np.array([2.0, -3.0, 8.0, 1.0])
    y = true_means[gi] + rng.normal(0.0, 0.25, n)
    g = gi.astype(str)  # labels "0".."3" (a *string* column → categorical intent)
    df = pd.DataFrame({"g": g, "y": y})

    model = gamfit.fit(df, "y ~ g")
    pred = np.asarray(model.predict(df), dtype=float).reshape(-1)

    fitted_means = np.array([float(pred[gi == k].mean()) for k in range(4)])
    max_err = float(np.max(np.abs(fitted_means - true_means)))

    # A treatment-coded factor recovers each group mean to within sampling noise
    # (n/4 = 300 obs per group, σ = 0.25 → SE ≈ 0.014). A linear-in-label collapse
    # leaves a max error of several units.
    assert max_err < 0.5, (
        f"y ~ g fitted per-group means {fitted_means.tolist()} do not match the "
        f"true means {true_means.tolist()} (max error {max_err:.2f}); the factor "
        f"was collapsed to a single linear numeric term in the integer label"
    )
