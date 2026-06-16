"""Regression test for #1159.

A `factor:factor` interaction-only model `y ~ g:h` (no `g` or `h` main effect)
must fit the SATURATED cell-means model: every (g, h) cell gets its own mean.
The factor-aware `:` expansion used to treatment-code BOTH factors (dropping a
reference level on each) and cross only the survivors, keeping just
(n_g-1)*(n_h-1) cells; the rest collapsed onto the intercept (a 2x3 design fell
to rank 3 instead of the saturated rank 6). The span-equivalent `y ~ g*h`
already fits perfectly, so gam was internally inconsistent.

With the marginality-aware fix, both factors are dummy-coded when neither main
effect is present and exactly one reference cell is absorbed by the intercept,
so:
  * all six cell means are recovered, and
  * the in-sample fitted values of `y ~ g:h` equal those of `y ~ g*h`.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

pytest: Any = importlib.import_module("pytest")

pytest.importorskip("gamfit._rust")

import gamfit

# truth per #1159: ap=0, aq=2, ar=-1, bp=1, bq=6, br=-2.
_TRUE_CELLS = {
    ("a", "p"): 0.0,
    ("a", "q"): 2.0,
    ("a", "r"): -1.0,
    ("b", "p"): 1.0,
    ("b", "q"): 6.0,
    ("b", "r"): -2.0,
}


def _frame(seed: int = 11, n: int = 2400, noise: float = 0.1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    g = np.array(["a", "b"])[rng.integers(0, 2, n)]
    h = np.array(["p", "q", "r"])[rng.integers(0, 3, n)]
    mean = np.array([_TRUE_CELLS[(gi, hi)] for gi, hi in zip(g, h)])
    y = mean + rng.normal(0.0, noise, n)
    return pd.DataFrame({"g": g, "h": h, "y": y})


def _recovered_cells(model: Any) -> dict[tuple[str, str], float]:
    cells: dict[tuple[str, str], float] = {}
    for (gi, hi) in _TRUE_CELLS:
        proto = pd.DataFrame({"g": [gi], "h": [hi]})
        cells[(gi, hi)] = float(np.asarray(model.predict(proto)).ravel()[0])
    return cells


def test_interaction_only_recovers_every_cell_mean() -> None:
    df = _frame()
    model = gamfit.fit(df, "y ~ g:h")

    cells = _recovered_cells(model)
    for key, truth in _TRUE_CELLS.items():
        assert abs(cells[key] - truth) < 0.05, (
            f"cell {key}: recovered mean {cells[key]:.4f} vs truth {truth}; "
            "cells must NOT collapse onto a common value"
        )


def test_interaction_only_matches_g_star_h_in_sample() -> None:
    df = _frame()
    m_interaction = gamfit.fit(df, "y ~ g:h")
    m_spanning = gamfit.fit(df, "y ~ g*h")

    fitted_interaction = np.asarray(m_interaction.predict(df)).ravel()
    fitted_spanning = np.asarray(m_spanning.predict(df)).ravel()

    assert np.all(np.isfinite(fitted_interaction))
    # `g:h` (saturated cell-means) and `g*h` (= g + h + g:h) span the identical
    # column space, so their in-sample fits must coincide to numerical precision.
    np.testing.assert_allclose(fitted_interaction, fitted_spanning, atol=1e-4)
