"""Regression test for #1158.

A `numeric:factor` interaction-only model `y ~ x:g` (no `x` main effect) must
fit the "common intercept, separate slopes" design: every group gets its OWN
x-slope. The factor-aware `:` expansion used to UNCONDITIONALLY treatment-code
`g` (drop its reference level), which — with no `x` main effect to soak up the
reference group's slope — pinned the reference group's x-slope to exactly 0, a
rank-deficient fit. mgcv uses full dummy coding here, and the span-equivalent
form `y ~ x + x:g` already recovers every slope, so gam was internally
inconsistent.

With the marginality-aware fix, `g` is dummy-coded when the `x` parent is
absent, so:
  * every per-group slope is recovered (a=1.5, b=-2.0, c=0.7), and
  * the in-sample fitted values of `y ~ x:g` equal those of `y ~ x + x:g`
    (the two designs span the identical column space).
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

pytest: Any = importlib.import_module("pytest")

pytest.importorskip("gamfit._rust")

import gamfit

_TRUE_SLOPES = {"a": 1.5, "b": -2.0, "c": 0.7}


def _frame(seed: int = 7, n: int = 1500, noise: float = 0.1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, n)
    g = np.array(["a", "b", "c"])[rng.integers(0, 3, n)]
    slope = np.array([_TRUE_SLOPES[gi] for gi in g])
    # Common (zero) intercept, separate per-group slopes — exactly the design
    # `y ~ x:g` is meant to represent.
    y = slope * x + rng.normal(0.0, noise, n)
    return pd.DataFrame({"x": x, "g": g, "y": y})


def _recovered_slopes(model: Any) -> dict[str, float]:
    # Slope for group g = predict(x=1, g) - predict(x=0, g): a finite difference
    # that isolates the per-group x-coefficient regardless of intercept coding.
    slopes: dict[str, float] = {}
    for grp in _TRUE_SLOPES:
        at0 = pd.DataFrame({"x": [0.0], "g": [grp]})
        at1 = pd.DataFrame({"x": [1.0], "g": [grp]})
        p0 = float(np.asarray(model.predict(at0)).ravel()[0])
        p1 = float(np.asarray(model.predict(at1)).ravel()[0])
        slopes[grp] = p1 - p0
    return slopes


def test_interaction_only_recovers_every_group_slope() -> None:
    df = _frame()
    model = gamfit.fit(df, "y ~ x:g")

    slopes = _recovered_slopes(model)
    for grp, truth in _TRUE_SLOPES.items():
        assert abs(slopes[grp] - truth) < 0.05, (
            f"group {grp}: recovered slope {slopes[grp]:.4f} vs truth {truth}; "
            "the reference group's slope must NOT be clamped to 0"
        )


def test_interaction_only_matches_x_plus_x_by_g_in_sample() -> None:
    df = _frame()
    m_interaction = gamfit.fit(df, "y ~ x:g")
    m_spanning = gamfit.fit(df, "y ~ x + x:g")

    fitted_interaction = np.asarray(m_interaction.predict(df)).ravel()
    fitted_spanning = np.asarray(m_spanning.predict(df)).ravel()

    assert np.all(np.isfinite(fitted_interaction))
    # The two formulas span the identical column space, so their in-sample fits
    # must coincide to numerical precision.
    np.testing.assert_allclose(fitted_interaction, fitted_spanning, atol=1e-4)
