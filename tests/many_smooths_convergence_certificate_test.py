"""pyGAM audit speed F1: an additive model with many smooths must converge.

``y ~ s(x0) + ... + s(x19)`` carries forty smoothing parameters (each smooth
has a wiggliness and a null-space penalty). Most of those are driven toward
the over-smoothing asymptote, where the REML criterion is exponentially flat
in log-lambda. On the binomial family every outer gradient evaluation used to
re-trace each GLM curvature-correction operator ``-X^T diag(c * X v_k) X``
column by column, three times per evaluation, so the outer search spent
minutes per step and never reached a certified optimum.

Binomial and Poisson search those forty coordinates with ARC on the exact
LAML outer Hessian; before that, gradient-only BFGS rebuilt the 40x40
curvature from secant pairs and the fit timed out at every n (audit F1/F2).

Asserted here is the certificate, never the wall time: the fit returns, and
the optimizer's own stationarity verdict is certified.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamfit

_STATIONARITY_KINDS = {"analytic_gradient", "fixed_point", "asymptote_rail"}
_N_SMOOTHS = 20


def _many_smooth_data(family: str, n: int) -> tuple[dict[str, np.ndarray], str]:
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1.0, (n, _N_SMOOTHS))
    eta = np.zeros(n)
    for j in range(_N_SMOOTHS):
        eta += np.sin(2.0 * np.pi * x[:, j] + j) / np.sqrt(_N_SMOOTHS)
    if family == "gaussian":
        y = eta + rng.normal(0.0, 0.5, n)
    elif family == "poisson":
        y = rng.poisson(np.exp(0.5 + 0.7 * eta)).astype(float)
    else:
        y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-1.5 * eta))).astype(float)
    names = [f"x{j}" for j in range(_N_SMOOTHS)]
    data = {name: x[:, j] for j, name in enumerate(names)}
    data["y"] = y
    formula = "y ~ " + " + ".join(f"s({name})" for name in names)
    return data, formula


@pytest.mark.parametrize("family", ["gaussian", "binomial", "poisson"])
def test_twenty_smooth_additive_model_converges_with_a_certificate(family: str) -> None:
    data, formula = _many_smooth_data(family, 1000)
    model = gamfit.fit(data, formula, family=family)

    convergence = model.summary().convergence
    assert convergence is not None, "a penalized fit must carry its convergence certificate"
    assert convergence["certified"] is True, convergence

    outer = convergence["outer"]
    assert outer is not None, "forty smoothing parameters were optimized"
    assert outer["kind"] in _STATIONARITY_KINDS, outer["kind"]
    assert float(outer["projected_gradient_norm"]) <= float(outer["stationarity_bound"]), outer
