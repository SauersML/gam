"""Bug hunt: a categorical ``by=`` factor smooth is silently treated as a
*numeric* by-variable when the factor's level labels happen to be numeric
strings (``"0"``, ``"1"``, ``"2"``, …), so the level ``"0"`` is annihilated and
the others are forced to be proportional to their integer label.

``docs/difference-smooths.md`` is explicit about the intended semantics:

    Unordered by-factor smooths: ``y ~ s(x, by=group)`` expands to one centred
    smooth per categorical level. The implementation also adds an unpenalized
    treatment-coded factor main effect …

so ``y ~ g + s(x, by=g)`` must fit an **independent** centred smooth for every
level of ``g`` — each level recovers its own shape. With non-numeric labels
(``"a"``, ``"b"``, …) it does exactly that.

But when the (string) grouping column carries numeric-looking labels, the
column-kind inference in ``src/inference/data.rs`` (``all_numeric`` → ``Binary``
/ ``Continuous``, lines 649-657 — a column is ``Categorical`` only if some row
*fails* numeric parsing) classifies it as a numeric column, and the by-smooth
builder (``src/terms/term_builder.rs:642``, ``Binary | Continuous =>
ByVariableSpec::Numeric``) then lowers ``by=g`` to a **numeric** by-variable.
The fitted term becomes ``value(g) · f(x)`` — a *single shared* shape scaled by
the integer value of the label — instead of one independent smooth per level.

The observable consequences on identical-per-group data (every level has the
SAME true shape ``sin(2x)``):

  * the level labeled ``"0"`` is multiplied by ``0`` and collapses to a flat
    line (zero nonlinear effect), and
  * the level labeled ``"k"`` is forced to amplitude ``∝ k`` rather than its own
    fitted amplitude.

This is a silent data-dependent corruption: integer-coded group IDs (``0/1/2``)
are extremely common, and the user gets per-group fits that are wrong in a way
that depends only on the *string spelling* of the level — renaming ``"0"`` to
``"a"`` changes the fit, violating the basic invariance that a model must not
depend on the names of categorical levels.

This test fits ``y ~ g + s(x, by=g)`` on two groups that share the **same** true
shape ``sin(2x)`` but are labeled ``"0"`` and ``"1"``, and asserts that BOTH
per-group fitted curves recover that shape (non-constant, well correlated with
the truth). It currently fails: the ``"0"`` group is flat (annihilated by the
``×0`` numeric scaling). When ``by=`` over a string-valued column is treated as
categorical regardless of whether the labels look numeric, every level gets its
own smooth and both assertions below hold without edits.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def _group_curve_stats(pred: "np.ndarray", mask: "np.ndarray", truth: "np.ndarray") -> tuple[float, float]:
    """Return ``(range, corr_with_truth)`` for the predicted curve on a group."""
    p = pred[mask]
    rng = float(np.max(p) - np.min(p))
    if rng <= 1e-9:
        return rng, 0.0  # constant curve has no (well-defined) correlation
    corr = float(np.corrcoef(p, truth[mask])[0, 1])
    return rng, corr


def test_by_factor_smooth_recovers_every_numeric_labeled_level() -> None:
    rng = np.random.default_rng(20260618)
    n = 1200
    gi = rng.integers(0, 2, n)
    x = rng.uniform(-3.0, 3.0, n)
    # IDENTICAL true shape for both groups, so a correct per-level by-smooth must
    # recover the SAME curve for each. A numeric-by misread forces group "k" to
    # amplitude ∝ k, annihilating "0".
    truth = np.sin(2.0 * x)
    y = truth + rng.normal(0.0, 0.10, n)
    g = gi.astype(str)  # labels "0" / "1" (a *string* column → categorical intent)
    df = pd.DataFrame({"x": x, "g": g, "y": y})

    model = gamfit.fit(df, "y ~ g + s(x, by=g)")
    pred = np.asarray(model.predict(df), dtype=float).reshape(-1)

    rng0, corr0 = _group_curve_stats(pred, gi == 0, truth)
    rng1, corr1 = _group_curve_stats(pred, gi == 1, truth)

    # Both levels share the same true shape sin(2x): each per-level smooth must
    # be a genuine (non-flat) curve that recovers it.
    assert rng0 > 1.0, (
        f"level '0' by-factor smooth collapsed to a flat line (range={rng0:.3f}); "
        f"it was treated as a numeric by-variable and multiplied by 0"
    )
    assert corr0 > 0.9, f"level '0' smooth does not recover its shape (corr={corr0:.3f})"
    assert rng1 > 1.0 and corr1 > 0.9, (
        f"level '1' smooth not recovered (range={rng1:.3f}, corr={corr1:.3f})"
    )
