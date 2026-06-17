"""Regression for the Poisson sibling of #1193/#817: the Poisson
observation/prediction interval must be equal-tailed, not a symmetric
``mu ± z·sigma`` band.

The #817 skew-aware fix was propagated to Gamma/NegBin/Tweedie/Beta but the
Poisson arm — the canonical right-skewed count family — was left on the
symmetric band. On low-rate counts the symmetric upper edge sits below the true
Poisson upper quantile, so the upper tail under-covers (``P(Y > upper)`` ~0.04+
at a nominal 2.5%-per-tail interval) while the clamped lower edge over-covers
and hides the defect in the two-sided number.

The fix builds the edges from genuine equal-tailed Poisson quantiles, widened
for estimation uncertainty by the conjugate Negative-Binomial (Gamma–Poisson)
posterior predictive. A moment-matched *continuous* surrogate (e.g. a Gamma) is
wrong here — it has no zero atom and over-covers the lower tail at low rates. See
``src/inference/predict/mod.rs`` (Poisson arm) and ``poisson_quantile`` /
``poisson_moment_matched_interval`` in ``src/inference/probability.rs``.
"""

import numpy as np
import pandas as pd
import pytest

import gamfit


def _skewed_poisson(rng, n):
    x = rng.uniform(0, 1, n)
    mu = np.exp(0.3 + 0.9 * np.sin(2 * np.pi * x))  # mean ~0.5..3, strong right skew
    return pd.DataFrame({"x": x, "y": rng.poisson(mu)})


def test_poisson_observation_interval_is_equal_tailed_not_symmetric():
    rng = np.random.default_rng(61)
    train, test = _skewed_poisson(rng, 6000), _skewed_poisson(rng, 20000)
    m = gamfit.fit(train, "y ~ s(x)", family="poisson")
    p = m.predict(test, interval=0.95, observation_interval=True)

    y = test["y"].to_numpy()
    lo = p["observation_lower"].to_numpy()
    hi = p["observation_upper"].to_numpy()

    two_sided = np.mean((y >= lo) & (y <= hi))
    upper_tail = np.mean(y > hi)
    lower_tail = np.mean(y < lo)

    # Tail-SHAPE error, not width: two-sided coverage must stay near nominal so
    # this is not just a wider band.
    assert two_sided > 0.90, f"two-sided coverage collapsed: {two_sided}"
    # The bug: upper tail ~0.04+ (under-covers). Equal-tailed => near 0.025; allow
    # slack for discreteness (the Poisson upper quantile over-covers slightly).
    assert upper_tail <= 0.035, f"upper tail still under-covers: {upper_tail}"
    # Lower tail is genuinely small here (Poisson zero-mass at low rates), so it
    # must stay small — the fix must not invent a spurious positive lower edge
    # (the continuous-surrogate failure mode).
    assert lower_tail <= 0.04, f"lower tail mis-covers: {lower_tail}"


def test_poisson_observation_upper_edge_reaches_true_quantile():
    """Independent cross-check (different angle than tail frequency): the upper
    edge must reach the true Poisson 0.975 quantile, not sit below it for ~every
    row as the symmetric band does.
    """
    from scipy.stats import poisson

    rng = np.random.default_rng(61)
    train, test = _skewed_poisson(rng, 6000), _skewed_poisson(rng, 20000)
    m = gamfit.fit(train, "y ~ s(x)", family="poisson")
    p = m.predict(test, interval=0.95, observation_interval=True)

    hi = p["observation_upper"].to_numpy()
    mu = np.exp(0.3 + 0.9 * np.sin(2 * np.pi * test["x"].to_numpy()))
    q975 = poisson.ppf(0.975, mu)

    # Pre-fix the symmetric upper edge sat below the true 0.975 quantile for the
    # large majority of rows (~0.77). The equal-tailed band reaches it: the edge
    # is at or above the true integer quantile for most rows.
    reaches = np.mean(hi >= q975)
    assert reaches >= 0.60, f"upper edge reaches true 0.975 quantile too rarely: {reaches}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
