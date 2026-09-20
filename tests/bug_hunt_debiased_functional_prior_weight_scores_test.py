"""Regression test (#3542): ``debiased_functional`` must use the fit's prior
weights in its Gaussian row scores.

The penalized Hessian the Riesz engine inverts is ``H = XᵀWX + S(λ)`` with
``W = diag(prior weights)``. The per-row score of that objective is
``s_i = w_i · x_i · (η_i − y_i)``. The handler built ``x_i · (η_i − y_i)``, and it
also materialized the training rows with a default config, so a
``weights="w"`` fit was replayed with unit weights.

Multiplying every prior weight of a Gaussian fit by a constant ``c`` is a pure
dispersion change: β̂ does not move and neither may any uncertainty statement
about a functional of β̂. With unweighted scores ``H`` scaled by ``c`` and the
influence values ``ψ_i = −n sᵢᵀH⁻¹g`` by ``1/c``, so the reported SE was
``1/c`` of the truth — a 7× too-narrow CI at ``c = 7``.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

pytest: Any = importlib.import_module("pytest")

pytest.importorskip("gamfit._rust")

import gamfit

from gamfit.errors import GamfitError


def _frame(weight: float) -> pd.DataFrame:
    rng = np.random.default_rng(3542)
    n = 400
    x = rng.uniform(0.0, 1.0, n)
    y = np.sin(2.0 * np.pi * x) + rng.normal(0.0, 0.3, n)
    return pd.DataFrame({"x": x, "y": y, "w": np.full(n, weight)})


@pytest.mark.parametrize("target", ["average_value", "average_derivative"])
def test_constant_prior_weight_scale_leaves_debiased_se_unchanged(target: str) -> None:
    results = []
    for weight in (1.0, 7.0):
        frame = _frame(weight)
        model = gamfit.fit(frame, "y ~ s(x)", weights="w")
        results.append(model.debiased_functional(frame, target=target))
    unit, scaled = results
    for key in ("theta_plugin", "theta_debiased", "se", "ci_lower", "ci_upper"):
        assert np.isfinite(unit[key]) and np.isfinite(scaled[key]), (key, unit, scaled)
    # A factor-of-seven discrepancy is what the unweighted scores produced; the
    # two fits are the same model, so agreement is limited only by the outer
    # REML solve landing on the same λ̂ from ρ seeds that differ by ln 7.
    assert scaled["se"] == pytest.approx(unit["se"], rel=1e-4), (unit, scaled)
    assert scaled["theta_debiased"] == pytest.approx(unit["theta_debiased"], rel=1e-4, abs=1e-8)


def test_non_numeric_functional_weight_is_refused() -> None:
    frame = _frame(1.0)
    model = gamfit.fit(frame, "y ~ s(x)")
    weights: list[Any] = [1.0] * len(frame)
    weights[5] = None
    with pytest.raises(GamfitError, match="weights"):
        model.debiased_functional(frame, target="average_value", weights=weights)
