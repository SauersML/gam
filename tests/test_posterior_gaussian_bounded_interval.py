"""Regression tests for issue #1508 — a Gaussian ``bounded(x, min, max)``
posterior must stay inside ``[min, max]``.

Root cause: ``sample_standard`` returned the unconstrained
``laplace_gaussian_fallback`` at the very top whenever the family was
Gaussian-identity, *before* the ``has_bounded`` dispatch that routes to the
latent-logit sampler. So for a Gaussian model the bounded latent path was dead
code and the draws were a plain user-scale Gaussian that spilled outside the
interval. The binomial path already worked (it never hit the Gaussian
shortcut). The fix moves the bounded/constraint dispatch ahead of the
Gaussian-identity shortcut, and φ-scales the latent draw so the Gaussian
posterior also has the correct *width*, not merely the correct support.

Angles:

* the Gaussian ``bounded()`` posterior stays strictly inside the interval
  (the reported failure);
* the binomial ``bounded()`` control still stays inside (no regression); and
* the interior point estimate is unchanged by sampling.

(These also guard the constraint-aware dispatch gate added for #1507/#1509: a
`bounded()` term must keep routing through the latent sampler, not the Gaussian
closed-form shortcut.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import gamfit


def _bounded_draws(model, frame, *, samples=4000, chains=2, seed=1):
    return model.sample(frame, samples=samples, chains=chains, seed=seed).to_numpy()[:, 1]


def test_gaussian_bounded_posterior_stays_in_interval() -> None:
    np.random.seed(3)
    n = 300
    x = np.random.uniform(0, 1, n)
    y = 2.0 + 0.8 * x + np.random.randn(n) * 0.1  # interior true slope 0.8
    df = pd.DataFrame({"x": x, "y": y})
    m = gamfit.fit(df, "y ~ bounded(x, min=0, max=1)")

    d = _bounded_draws(m, df)
    # Was ~17% outside [0, 1] before the fix.
    assert d.min() > 0.0 and d.max() < 1.0, f"range [{d.min()}, {d.max()}] escaped (0, 1)"


def test_binomial_bounded_control_stays_in_interval() -> None:
    rng = np.random.default_rng(3)
    n = 600
    x = rng.uniform(0, 1, n)
    p = 1.0 / (1.0 + np.exp(-(0.2 + 0.8 * x)))
    y = rng.binomial(1, p, n)
    df = pd.DataFrame({"x": x, "y": y})
    m = gamfit.fit(df, "y ~ bounded(x, min=0, max=1)", family="binomial")

    d = _bounded_draws(m, df)
    assert d.min() > 0.0 and d.max() < 1.0, f"binomial range [{d.min()}, {d.max()}] escaped"


def test_bounded_point_estimate_unchanged_and_interior() -> None:
    np.random.seed(3)
    n = 300
    x = np.random.uniform(0, 1, n)
    y = 2.0 + 0.8 * x + np.random.randn(n) * 0.1
    df = pd.DataFrame({"x": x, "y": y})
    m = gamfit.fit(df, "y ~ bounded(x, min=0, max=1)")
    est = m.summary().coefficients[1]["estimate"]
    assert 0.0 < est < 1.0
    d = _bounded_draws(m, df)
    # The posterior brackets the interior point estimate.
    assert d.min() < est < d.max()
