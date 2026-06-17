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
    """Different angle: the band's *shape* must be the skew-correct equal-tailed
    Beta, not a symmetric band — i.e. the upper edge reaches the true upper Beta
    quantile (the symmetric band sat below it) and the band is right-skewed about
    the mean on these small-mean proportions.

    NB on what NOT to assert: a per-row "fraction of edges below the true
    quantile" is the *wrong* statistic. A correctly-calibrated equal-tailed band
    tracks the true conditional quantile, so it straddles it ~50/50 (plus a small
    upward shift from the estimation-uncertainty widening) — and the per-row
    fraction is dominated by the flexible-smooth mean wiggle, so it swings
    wildly with the seed (~0.02..0.75) and can never sit below 0.5, let alone
    0.25. The robust shape signatures below are seed-stable to ~1%.
    """
    from scipy.stats import beta as B

    rng = np.random.default_rng(71)
    train, test = _skewed_beta(rng, 6000), _skewed_beta(rng, 20000)
    m = gamfit.fit(train, "y ~ s(x)", family="beta")
    p = m.predict(test, interval=0.95, observation_interval=True)

    mu_hat = p["mean"].to_numpy()
    lo = p["observation_lower"].to_numpy()
    hi = p["observation_upper"].to_numpy()
    mu = 1.0 / (1.0 + np.exp(-(-1.6 + 1.3 * np.sin(2 * np.pi * test["x"].to_numpy()))))
    q975 = B.ppf(0.975, mu * PHI, (1.0 - mu) * PHI)

    # The upper edge must REACH the true upper quantile (median ratio ≈ 1). The
    # pre-fix symmetric band sat systematically below it (median ratio ≈ 0.90,
    # mean edge 0.443 vs true 0.494); the equal-tailed fix lifts it onto the
    # quantile. The median is robust to the per-row mean wiggle.
    median_ratio = np.median(hi / q975)
    assert 0.95 <= median_ratio <= 1.05, (
        f"upper edge does not track the true 0.975 quantile: median(hi/q975)={median_ratio}"
    )

    # The band must be right-skewed about the fitted mean — the defining
    # difference from a symmetric μ ± z·σ band, whose half-widths are equal. On
    # these small-mean (right-skewed) rows the upper half-width is ~1.9× the
    # lower; a symmetric band would give a ratio of 1.0.
    upper_half = np.median(hi - mu_hat)
    lower_half = np.median(mu_hat - lo)
    assert upper_half > 1.3 * lower_half, (
        f"band is not right-skewed (symmetric-band signature): "
        f"upper half-width {upper_half} vs lower {lower_half}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
