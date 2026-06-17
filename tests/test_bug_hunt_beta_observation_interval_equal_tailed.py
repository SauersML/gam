"""Regression for #1194: the Beta observation/prediction interval must be
equal-tailed, not a symmetric ``mu ± z·sigma`` band.

Beta is continuous on (0, 1) and skewed toward whichever edge its mean is near,
so the symmetric band lands BOTH edges below the corresponding true Beta quantile
on small-mean proportions: the lower tail over-covers (~0.2% vs 2.5%) and the
upper tail under-covers (~4.9% vs 2.5%), while the two-sided number stays near
nominal and hides it. This is the Beta sibling of the closed #817 (Gamma): the
fix builds the edges from equal-tailed quantiles of a moment-matched Beta. See
``src/inference/predict/mod.rs`` (Beta arm) and ``beta_moment_matched_interval``
in ``src/inference/probability.rs``.
"""

import numpy as np
import pandas as pd
import pytest

import gamfit

PHI = 8.0


def _skewed_beta(rng, n):
    x = rng.uniform(0, 1, n)
    mu = 1.0 / (1.0 + np.exp(-(-1.6 + 1.3 * np.sin(2 * np.pi * x))))  # mean ~0.05..0.5
    return pd.DataFrame({"x": x, "y": rng.beta(mu * PHI, (1.0 - mu) * PHI)})


def test_beta_observation_interval_is_equal_tailed_both_tails():
    rng = np.random.default_rng(71)
    train, test = _skewed_beta(rng, 6000), _skewed_beta(rng, 20000)
    m = gamfit.fit(train, "y ~ s(x)", family="beta")
    p = m.predict(test, interval=0.95, observation_interval=True)

    y = test["y"].to_numpy()
    lo = p["observation_lower"].to_numpy()
    hi = p["observation_upper"].to_numpy()

    two_sided = np.mean((y >= lo) & (y <= hi))
    lower_tail = np.mean(y < lo)
    upper_tail = np.mean(y > hi)

    assert two_sided > 0.90, f"two-sided coverage collapsed: {two_sided}"
    # Pre-fix: lower ~0.002 (over-covers), upper ~0.049 (under-covers). Both tails
    # must converge toward the nominal 0.025.
    assert lower_tail >= 0.012, f"lower tail still over-covers: {lower_tail}"
    assert upper_tail <= 0.04, f"upper tail still under-covers: {upper_tail}"


def test_beta_observation_edges_track_true_quantiles():
    """Different angle: each edge must track the corresponding true Beta quantile
    rather than both sitting below it.
    """
    from scipy.stats import beta as B

    rng = np.random.default_rng(71)
    train, test = _skewed_beta(rng, 6000), _skewed_beta(rng, 20000)
    m = gamfit.fit(train, "y ~ s(x)", family="beta")
    p = m.predict(test, interval=0.95, observation_interval=True)

    lo = p["observation_lower"].to_numpy()
    hi = p["observation_upper"].to_numpy()
    mu = 1.0 / (1.0 + np.exp(-(-1.6 + 1.3 * np.sin(2 * np.pi * test["x"].to_numpy()))))
    q025 = B.ppf(0.025, mu * PHI, (1.0 - mu) * PHI)
    q975 = B.ppf(0.975, mu * PHI, (1.0 - mu) * PHI)

    # Pre-fix both edges sat below the true quantile for essentially every row
    # (upper: 0.935 below; lower: 1.0 below). After the fix the edges bracket the
    # true quantiles, so neither "below" fraction is near-universal.
    assert np.mean(hi < q975) < 0.25, "upper edge below true 0.975 quantile too often"
    assert np.mean(lo < q025) < 0.75, "lower edge below true 0.025 quantile too often"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
