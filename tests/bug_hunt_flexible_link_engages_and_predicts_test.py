"""Regression for #1596 (different angle): a binomial flexible link must *engage*
and produce a *usable, monotone, predictive* model — not merely fail loud.

The companion `bug_hunt_flexible_link_silent_noop_on_binomial_parametric_test`
pins the contract from the deviance side (engage-or-fail-loud) on cloglog-
misspecified data. This test attacks the same root cause (the link warp never
engaging) from three independent angles that the deviance check alone would not
catch:

  1. A *different* link misspecification (true link = probit, requested =
     flexible(logit)). The warp must still engage and recover a large fraction of
     the logit→probit gap — guarding against a fix that only happens to work for
     one data-generating link.
  2. `predict(...)` must succeed and reconstruct the *same* warped link the fit
     used (predicted in-sample deviance ≈ the fitted deviance). This guards the
     reduced-coordinate → full-width warp reconstruction wiring: a fit that
     engages internally but cannot reconstruct the warp at predict time (or
     reconstructs a different one) is still broken.
  3. The fitted learnable link must be *monotone* (invertible): predicting at a
     dense, increasing grid of η yields a non-decreasing mean. A non-monotone
     "link" is not a link.

Before the fix the flexible-link joint solve never converged (the dynamic warp
basis collapsed the trust region / the mean-aliased warp forced a rank-deficient
gauge drop), so the request was a permanent silent no-op / loud non-convergence.
"""

import os

os.environ.setdefault("GAM_LOG", "off")

import numpy as np
import pandas as pd
from scipy.stats import norm

import gamfit


def _deviance(fit):
    return float(fit.summary().deviance)


def _bernoulli_deviance(y, mu):
    mu = np.clip(np.asarray(mu, dtype=float), 1e-9, 1.0 - 1e-9)
    y = np.asarray(y, dtype=float)
    return float(-2.0 * np.sum(y * np.log(mu) + (1.0 - y) * np.log(1.0 - mu)))


def test_flexible_link_engages_predicts_and_is_monotone_on_probit_data():
    rng = np.random.default_rng(11)
    n = 2500
    x = rng.uniform(-2.5, 2.5, n)
    eta = -0.3 + 1.4 * x
    # TRUE link = probit; logit is misspecified (a different misspecification
    # than the cloglog companion test).
    p = np.clip(norm.cdf(eta), 1e-4, 1.0 - 1e-4)
    y = (rng.uniform(size=n) < p).astype(float)
    df = pd.DataFrame({"x": x, "y": y})

    plain = gamfit.fit(df, "y ~ x", family="binomial", link="logit")
    probit = gamfit.fit(df, "y ~ x", family="binomial", link="probit")
    dev_plain = _deviance(plain)
    dev_probit = _deviance(probit)

    # The flexible-link request must produce a successful fit here (the
    # parametric mean makes the warp identifiable). If the implementation
    # regresses to the loud-failure escape, this fails — which is the point:
    # the warp must genuinely engage on identifiable, misspecified data.
    flex = gamfit.fit(df, "y ~ x + link(type=flexible(logit))", family="binomial")
    dev_flex = _deviance(flex)

    # 1. Engage: recover a meaningful fraction of the logit→probit gap.
    gap = dev_plain - dev_probit
    assert gap > 5.0, f"sanity: expected a real logit/probit gap, got {gap}"
    assert dev_flex < dev_plain - 0.25 * gap, (
        f"flexible(logit) barely improved on plain logit: dev_flex={dev_flex}, "
        f"dev_plain={dev_plain}, dev_probit={dev_probit} (gap={gap}); the warp did "
        "not engage (#1596)."
    )

    # 2. Predict reconstructs the same warped link the fit used: the in-sample
    #    predicted deviance must match the fitted deviance closely.
    mu_hat = np.asarray(flex.predict(df), dtype=float)
    assert mu_hat.shape == (n,)
    assert np.all(np.isfinite(mu_hat))
    dev_predict = _bernoulli_deviance(y, mu_hat)
    assert abs(dev_predict - dev_flex) < 1.0 + 1e-3 * abs(dev_flex), (
        f"predict reconstructed a different link than the fit: predicted deviance "
        f"{dev_predict} vs fitted {dev_flex}."
    )

    # 3. The learnable link is monotone (invertible): predicting on a sorted x
    #    grid yields a non-decreasing mean (η = β0 + β1 x is increasing in x for
    #    β1 > 0, and a valid link is increasing in η).
    grid = pd.DataFrame({"x": np.linspace(x.min(), x.max(), 200)})
    mu_grid = np.asarray(flex.predict(grid), dtype=float)
    diffs = np.diff(mu_grid)
    assert np.all(diffs >= -1e-6), (
        "fitted learnable link is not monotone: predicted mean decreases along an "
        f"increasing predictor grid (min step {diffs.min():.3e})."
    )
