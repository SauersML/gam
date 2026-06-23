"""Regression tests for issue #1509 — posterior draws of a shape-constrained
smooth must respect the shape cone.

Root cause: a ``shape='monotone_increasing'`` smooth imposes the monotone cone
as coefficient inequalities ``γ_j ≥ 0`` enforced by the constrained P-IRLS, so
the fitted curve is monotone — but ``model.sample()`` drew an unconstrained
Gaussian on the spline coefficients, leaving the cone, so ~30% of drawn curves
were non-monotone. The fix samples the Laplace posterior truncated to the
coefficient cone (exact reflective HMC), so every drawn coefficient vector — and
hence every drawn curve — is in the cone.

Angles:

* monotone-increasing: posterior-predictive curves are (essentially) monotone;
* monotone-decreasing: the mirror image;
* the fitted point estimate stays monotone (untouched by the fix); and
* the posterior is non-degenerate (the drawn curves are not all identical —
  truncation restricts the cone but does not collapse the spread).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import gamfit


def _grid(npts: int = 80) -> pd.DataFrame:
    return pd.DataFrame({"x": np.linspace(0, 1, npts)})


def test_monotone_increasing_posterior_curves_are_monotone() -> None:
    rng = np.random.default_rng(0)
    n = 300
    x = np.sort(rng.uniform(0, 1, n))
    y = np.log(x + 0.1) + rng.standard_normal(n) * 0.1
    df = pd.DataFrame({"x": x, "y": y})
    m = gamfit.fit(df, "y ~ s(x, shape='monotone_increasing')")

    grid = _grid()
    fitted = np.asarray(m.predict(grid)).ravel()
    assert np.diff(fitted).min() >= -1e-6  # point estimate monotone

    s = m.sample(df, samples=600, chains=2, seed=1)
    curves = np.asarray(s.predict_draws(grid).mean)  # (n_draws, n_grid)
    span = np.ptp(curves)
    worst = np.array([np.diff(c).min() for c in curves])
    # Was ~30% decreasing by >0.5% of range before the fix.
    frac_bad = float((worst < -0.005 * span).mean())
    assert frac_bad < 0.02, f"{frac_bad:.3f} of drawn curves are non-monotone"


def test_monotone_decreasing_posterior_curves_are_monotone() -> None:
    rng = np.random.default_rng(1)
    n = 300
    x = np.sort(rng.uniform(0, 1, n))
    y = -np.log(x + 0.1) + rng.standard_normal(n) * 0.1
    df = pd.DataFrame({"x": x, "y": y})
    m = gamfit.fit(df, "y ~ s(x, shape='monotone_decreasing')")

    grid = _grid()
    fitted = np.asarray(m.predict(grid)).ravel()
    assert np.diff(fitted).max() <= 1e-6  # point estimate monotone-decreasing

    s = m.sample(df, samples=600, chains=2, seed=2)
    curves = np.asarray(s.predict_draws(grid).mean)
    span = np.ptp(curves)
    worst = np.array([np.diff(c).max() for c in curves])  # largest *increase*
    frac_bad = float((worst > 0.005 * span).mean())
    assert frac_bad < 0.02, f"{frac_bad:.3f} of drawn curves are non-decreasing"


def test_monotone_posterior_is_non_degenerate() -> None:
    # The truncated posterior must still be dispersed: the drawn curves should
    # not all collapse onto the point estimate.
    rng = np.random.default_rng(0)
    n = 300
    x = np.sort(rng.uniform(0, 1, n))
    y = np.log(x + 0.1) + rng.standard_normal(n) * 0.1
    df = pd.DataFrame({"x": x, "y": y})
    m = gamfit.fit(df, "y ~ s(x, shape='monotone_increasing')")

    grid = _grid()
    s = m.sample(df, samples=600, chains=2, seed=1)
    curves = np.asarray(s.predict_draws(grid).mean)
    # Pointwise posterior SD at the grid points is non-trivial.
    sd = curves.std(axis=0)
    assert sd.max() > 1e-3, "monotone posterior collapsed to a spike"
