"""Regression for #1193: the Negative-Binomial observation/prediction interval
must be equal-tailed, not a symmetric ``mu ± z·sigma`` band.

On right-skewed counts the symmetric band sits below the true NB upper quantile,
so the upper tail under-covers (``P(Y > upper) ≈ 0.05`` at a nominal 2.5%-per-tail
interval) while the clamped lower edge over-covers and hides the defect in the
two-sided number. This is the NB sibling of the closed #817 (Gamma): the fix is
to build the edges from genuine equal-tailed NB quantiles (a moment-matched
*continuous* surrogate is wrong here — it has no zero atom and over-covers the
lower tail). See ``src/inference/predict/mod.rs`` (NB arm) and
``negative_binomial_moment_matched_interval`` in ``src/inference/probability.rs``.
"""

import numpy as np
import pandas as pd
import pytest

import gamfit


def _skewed_nb(rng, n, theta=1.5):
    x = rng.uniform(0, 1, n)
    mu = np.exp(0.3 + 0.9 * np.sin(2 * np.pi * x))  # mean ~0.5..3, strong right skew
    lam = rng.gamma(theta, mu / theta)  # Gamma-Poisson mixture, Var = mu + mu^2/theta
    return pd.DataFrame({"x": x, "y": rng.poisson(lam)})


def test_nb_observation_interval_is_equal_tailed_not_symmetric():
    rng = np.random.default_rng(61)
    train, test = _skewed_nb(rng, 6000), _skewed_nb(rng, 20000)
    m = gamfit.fit(train, "y ~ s(x)", family="nb")
    p = m.predict(test, interval=0.95, observation_interval=True)

    y = test["y"].to_numpy()
    lo = p["observation_lower"].to_numpy()
    hi = p["observation_upper"].to_numpy()

    two_sided = np.mean((y >= lo) & (y <= hi))
    upper_tail = np.mean(y > hi)
    lower_tail = np.mean(y < lo)

    # The defect was a tail-SHAPE error, not a width error: two-sided coverage
    # must stay near nominal so this is not just a wider band.
    assert two_sided > 0.90, f"two-sided coverage collapsed: {two_sided}"
    # The bug: upper tail ~0.050 (2x nominal). Equal-tailed => <= ~0.025; allow
    # slack for discreteness (the NB upper quantile over-covers slightly).
    assert upper_tail <= 0.04, f"upper tail still under-covers: {upper_tail}"
    # Lower tail is genuinely ~0 here (NB zero-mass P(Y=0) ~ 0.34 dominates the
    # 2.5% lower quantile), so it must stay small — the fix must not invent a
    # spurious positive lower edge (the continuous-surrogate failure mode).
    assert lower_tail <= 0.04, f"lower tail mis-covers: {lower_tail}"


def test_nb_observation_upper_edge_reaches_true_quantile():
    """Independent cross-check (different angle than tail frequency): the upper
    edge must reach the true NB 0.975 quantile, not sit below it for ~every row.
    """
    from scipy.stats import nbinom

    rng = np.random.default_rng(61)
    train, test = _skewed_nb(rng, 6000), _skewed_nb(rng, 20000)
    m = gamfit.fit(train, "y ~ s(x)", family="nb")
    p = m.predict(test, interval=0.95, observation_interval=True)

    hi = p["observation_upper"].to_numpy()
    mu_te = np.exp(0.3 + 0.9 * np.sin(2 * np.pi * test["x"].to_numpy()))
    theta = 1.5
    q975 = nbinom.ppf(0.975, theta, theta / (theta + mu_te))

    # Pre-fix, ``hi < q975`` for ~100% of rows (the band undershot by ~1.2 counts
    # on average). After the fix the band tracks the true quantile, so the
    # fraction of rows below it must be small.
    frac_below = np.mean(hi < q975)
    assert frac_below < 0.25, f"upper edge below true 0.975 quantile for {frac_below:.2%} of rows"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
