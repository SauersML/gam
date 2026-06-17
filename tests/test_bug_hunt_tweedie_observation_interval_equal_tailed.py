"""Regression for #1193/#1194 (Tweedie arm): the Tweedie observation/prediction
interval must be equal-tailed, not a symmetric ``mu ± z·sigma`` band.

Tweedie (1 < p < 2) is a compound Poisson-Gamma: a point mass at zero plus a
continuous right-skewed positive part. The symmetric band sits below the true
upper quantile (so the upper tail under-covers, ~0.05 at a nominal 2.5%-per-tail
interval) while the clamped lower edge over-covers and hides the defect in the
two-sided number — the Tweedie sibling of the closed #817 (Gamma). The fix builds
the edges from genuine equal-tailed Tweedie quantiles (a moment-matched
*continuous* Gamma surrogate is wrong here — it lacks the zero atom and would
over-cover the lower tail like the NB surrogate, #1193). See
``src/inference/predict/mod.rs`` (Tweedie arm) and
``tweedie_moment_matched_interval`` in ``src/inference/probability.rs``.
"""

import numpy as np
import pandas as pd
import pytest

import gamfit

POWER = 1.5
PHI = 1.0


def _tweedie_sample(rng, mu, phi, power):
    """Draw a compound Poisson-Gamma (Tweedie) variate per row: N ~ Poisson(lam)
    jumps, each Gamma(alpha, scale), summed (0 if N == 0)."""
    lam = mu ** (2.0 - power) / (phi * (2.0 - power))
    alpha = (2.0 - power) / (power - 1.0)
    scale = phi * (power - 1.0) * mu ** (power - 1.0)
    n = rng.poisson(lam)
    out = np.zeros_like(mu)
    for i in range(len(mu)):
        if n[i] > 0:
            out[i] = rng.gamma(alpha * n[i], scale[i])
    return out


def _skewed_tweedie(rng, n):
    x = rng.uniform(0, 1, n)
    mu = np.exp(0.4 + 0.9 * np.sin(2 * np.pi * x))  # mean ~0.6..3.7, right-skewed
    y = _tweedie_sample(rng, mu, PHI, POWER)
    return pd.DataFrame({"x": x, "y": y})


def test_tweedie_observation_interval_is_equal_tailed_not_symmetric():
    rng = np.random.default_rng(83)
    train, test = _skewed_tweedie(rng, 6000), _skewed_tweedie(rng, 20000)
    m = gamfit.fit(train, "y ~ s(x)", family="tweedie")
    p = m.predict(test, interval=0.95, observation_interval=True)

    y = test["y"].to_numpy()
    lo = p["observation_lower"].to_numpy()
    hi = p["observation_upper"].to_numpy()

    two_sided = np.mean((y >= lo) & (y <= hi))
    upper_tail = np.mean(y > hi)
    lower_tail = np.mean(y < lo)

    # The defect is a tail-SHAPE error, not a width error: two-sided coverage must
    # stay near nominal so this is not just a wider band.
    assert two_sided > 0.90, f"two-sided coverage collapsed: {two_sided}"
    # The bug: upper tail ~0.05 (2x nominal). Equal-tailed => ~0.025; allow slack.
    assert upper_tail <= 0.04, f"upper tail still under-covers: {upper_tail}"
    # The fat zero atom dominates the 2.5% lower quantile, so the lower tail must
    # stay small — the fix must not invent a spurious positive lower edge (the
    # continuous-surrogate failure mode).
    assert lower_tail <= 0.04, f"lower tail mis-covers: {lower_tail}"


def test_tweedie_observation_upper_edge_is_above_the_symmetric_band():
    """Independent cross-check: the skew-correct upper edge must lie ABOVE the
    old symmetric ``mu + z·sd`` edge for the right-skewed rows (the symmetric band
    undershot the true upper quantile, the cause of the upper-tail under-cover)."""
    rng = np.random.default_rng(83)
    train, test = _skewed_tweedie(rng, 6000), _skewed_tweedie(rng, 20000)
    m = gamfit.fit(train, "y ~ s(x)", family="tweedie")
    p = m.predict(test, interval=0.95, observation_interval=True)

    hi = p["observation_upper"].to_numpy()
    mu_hat = p["mean"].to_numpy()
    # Conditional Tweedie sd; the symmetric band's upper edge would be mu + 1.96*sd.
    sd = np.sqrt(PHI * mu_hat ** POWER)
    symmetric_upper = mu_hat + 1.959963984540054 * sd

    frac_above = np.mean(hi > symmetric_upper)
    assert frac_above > 0.6, (
        f"skew-correct upper edge not above symmetric band for enough rows: {frac_above:.2%}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
