"""pyGAM audit F17 and F15: a certified outer minimum yields intervals.

F17. pyGAM's LogisticGAM example ``y ~ factor(student) + s(balance) + s(income)``
on the ``default`` data (n = 2000, about 3% positives, raw covariates) certified
an outer minimum and then refused the smoothing-corrected covariance because
the rho Hessian had negative curvature. That Hessian omitted the second
derivative of the #784 block-local correction; a latched correction now
declares no analytic outer Hessian and the search continues as BFGS. The
continuation then stalled, because the correction took its eigenpairs from an
``eigh`` of the assembled Hessian: with a smooth penalised onto its rail
(``lambda ~ 1e13``) that eigensolve resolves the soft modes the correction
lives on only to ``eps * ||H||``, and the correction's value changed with the
last bits of ``H`` between two solves at the same rho. The correction now reads
the criterion's own root-scale (#2644) eigensystem.

F15. The chicago Poisson model ``y ~ s(time) + s(tmpd) + te(pm10, o3)`` at
n = 1000-1500 raised "smoothing cubature has no positive-width proposal". It
now fits and the smoothing correction is integrated by the cubature itself.

Both fixtures are synthetic analogues with the covariate scales of the data.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamfit


def _default_like(seed: int, n: int = 2000) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    student = rng.random(n) < 0.3
    balance = np.clip(rng.normal(835.0, 480.0, n), 0.0, None)
    income = np.where(
        student, rng.normal(17_500.0, 4_500.0, n), rng.normal(40_000.0, 10_000.0, n)
    ).clip(700.0, None)
    eta = -10.8 + 0.0057 * balance - 0.65 * student
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    return {
        "student": np.where(student, "Yes", "No"),
        "balance": balance,
        "income": income,
        "y": y,
    }


def _chicago_like(seed: int, n: int = 1000) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    day = np.arange(n, dtype=float)
    season = np.cos(2.0 * np.pi * day / 365.25)
    tmpd = 50.0 - 20.0 * season + rng.normal(0.0, 7.0, n)
    o3 = -2.0 + 0.3 * (tmpd - 50.0) + rng.normal(0.0, 8.0, n)
    pm10 = 0.35 * (tmpd - 50.0) + rng.gamma(2.0, 9.0, n) - 18.0
    eta = (
        np.log(115.0)
        + 0.08 * season
        + 4e-4 * np.maximum(tmpd - 75.0, 0.0) ** 1.5
        + 0.002 * pm10
        + 0.001 * o3
    )
    y = rng.poisson(np.exp(eta)).astype(float)
    return {"time": day - day.mean(), "tmpd": tmpd, "pm10": pm10, "o3": o3, "y": y}


def _assert_intervals(prediction: dict, n: int) -> None:
    lower = np.asarray(prediction["posterior_mean_lower"], dtype=float)
    upper = np.asarray(prediction["posterior_mean_upper"], dtype=float)
    mean = np.asarray(prediction["posterior_mean"], dtype=float)
    assert lower.shape == upper.shape == mean.shape == (n,)
    assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
    assert np.all(lower <= mean) and np.all(mean <= upper)
    assert np.all(upper > lower)


# Seeds on which the binomial search stalled before the correction read the
# criterion's eigensystem.
@pytest.mark.parametrize("seed", [7, 8, 10])
@pytest.mark.parametrize(
    "formula",
    ["y ~ factor(student) + s(balance) + s(income)", "y ~ s(balance) + s(income)"],
)
def test_rare_event_binomial_on_raw_covariates_returns_intervals(seed, formula):
    data = _default_like(seed)
    assert 0.01 < data["y"].mean() < 0.06
    model = gamfit.fit(data, formula, family="binomial")
    assert model.summary().convergence.get("certified") is True
    prediction = model.predict(data, interval=0.95)
    _assert_intervals(prediction, len(data["y"]))


@pytest.mark.parametrize(("seed", "n"), [(1, 1000), (2, 1000), (1, 1500)])
def test_poisson_smooth_smooth_tensor_returns_smoothing_corrected_intervals(seed, n):
    data = _chicago_like(seed, n)
    model = gamfit.fit(data, "y ~ s(time) + s(tmpd) + te(pm10, o3)", family="poisson")
    assert model.summary().convergence.get("certified") is True
    prediction = model.predict(data, interval=0.95, covariance_mode="smoothing")
    assert prediction["covariance_source"] == "smoothing-corrected"
    _assert_intervals(prediction, n)
