"""Regression for #2254: a Gaussian-identity fit on a perfectly constant
(zero-variance) response must fit cleanly and be fully usable — the same
continuous limit as the near-constant case that already works — for EVERY
model shape, not just the single-penalty `y ~ x`.

The defect surfaced through gamfit as a hard

    IntegrationError: constant Gaussian shortcut produced invalid fit:
    Invalid input: UnifiedFitResult EDF smoothing-parameter count mismatch:
    edf_by_block=1, lambdas=<k>

because the constant-response fast-path minted an inference bundle whose
``edf_by_block`` was hard-coded to length 1 regardless of the model's penalty
count ``k``. The constructor's ``edf_by_block.len() == lambdas.len()`` invariant
then rejected every fit with ``k != 1``: ``y ~ 1`` (k=0), ``y ~ s(x, k=10)``
(k=2), ``matern`` (k=3). Only ``y ~ x`` (k=1) slipped through. The fix computes
the full per-penalty EDF decomposition, so the bundle is complete and
self-consistent for any penalty count.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamfit

FORMULAS = [
    "y ~ 1",
    "y ~ x",
    "y ~ s(x, k=10)",
    "y ~ duchon(x, centers=8)",
    "y ~ matern(x, centers=8)",
]


def _pred(model, data) -> np.ndarray:
    out = model.predict(data)
    if isinstance(out, np.ndarray):
        return np.asarray(out, dtype=float).ravel()
    return np.asarray(out["mean"], dtype=float).ravel()


@pytest.mark.parametrize("formula", FORMULAS)
@pytest.mark.parametrize("yval", [3.7, 0.0, -2.5])
def test_constant_gaussian_fits_and_predicts_exactly(formula: str, yval: float) -> None:
    n = 200
    data = {"x": np.linspace(0.0, 1.0, n), "y": np.full(n, yval)}

    # Must NOT raise the "EDF smoothing-parameter count mismatch" IntegrationError.
    model = gamfit.fit(data, formula, family="gaussian")

    pred = _pred(model, data)
    assert pred.shape[0] == n
    assert np.all(np.isfinite(pred)), f"{formula!r}: non-finite prediction"
    # A constant response is reproduced exactly by the intercept (smooth ≡ 0).
    assert np.max(np.abs(pred - yval)) <= 1e-6, (
        f"{formula!r}: fit does not reproduce the constant {yval}: {pred[:3]}"
    )


@pytest.mark.parametrize("formula", FORMULAS)
def test_constant_gaussian_reports_finite_bounded_edf(formula: str) -> None:
    n = 160
    data = {"x": np.linspace(0.0, 1.0, n), "y": np.full(n, 1.25)}
    model = gamfit.fit(data, formula, family="gaussian")

    summary = model.summary()
    edf = float(summary.edf_total)
    p = int(len(summary.coefficients))

    assert np.isfinite(edf)
    # A constant fit spends only its unpenalized null space; EDF is bounded by
    # the design column count and, for a penalized model, strictly below it.
    assert 0.0 <= edf <= p + 1e-6, f"{formula!r}: EDF {edf} out of [0, {p}]"
    if len(summary.lambdas) > 0:
        assert edf < p - 1e-6, (
            f"{formula!r}: a penalized constant fit should collapse below the "
            f"full basis (EDF {edf}, p={p})"
        )


def test_near_constant_and_exact_constant_agree_in_the_limit() -> None:
    # #2254 was a cliff at EXACTLY zero variance: a response with variance ~1e-12
    # fit cleanly while the zero-variance limit hard-errored. The two must now be
    # continuous — same fit, same near-null EDF.
    n = 200
    x = np.linspace(0.0, 1.0, n)
    rng = np.random.default_rng(0)
    near = {"x": x, "y": 3.7 + 1e-6 * rng.standard_normal(n)}
    exact = {"x": x, "y": np.full(n, 3.7)}

    formula = "y ~ s(x, k=10)"
    m_near = gamfit.fit(near, formula, family="gaussian")
    m_exact = gamfit.fit(exact, formula, family="gaussian")

    edf_near = float(m_near.summary().edf_total)
    edf_exact = float(m_exact.summary().edf_total)
    # Both collapse toward the null space; the exact fit is the clean limit.
    assert edf_exact <= edf_near + 0.5, (
        f"exact-constant EDF {edf_exact} should not exceed the near-constant "
        f"limit {edf_near}"
    )
    assert np.max(np.abs(_pred(m_exact, exact) - 3.7)) <= 1e-6
