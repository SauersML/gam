"""Regression for #2026 (conditional-coverage angle): a Tweedie observation
interval must be calibrated *conditional on the mean* on data whose true power
differs from the old hardcoded fallback ``p = 1.5``, and a Tweedie fit must never
silently use a power the caller did not choose.

The mean-variance law is ``Var(Y|x) = phi * mu(x)**p``. The fitted mean (log-link
quasi-likelihood) is robust to a misspecified ``p``, but the observation-interval
WIDTH scales with ``mu**p`` and so is not: with ``p`` fixed too high on ``p<1.5``
data, the low-mean rows get intervals that are too wide relative to the truth at
one end and too narrow at the other, and the marginal (pooled) coverage hides it.
The #2026 report shows the low-mean quartile collapsing to ~0.73 at a nominal
0.90 on ``p_true = 1.1`` data while the hardcoded ``p = 1.5`` was used.

HISTORY: #2026 first profiled ``p`` for a bare ``family="tweedie"``, and this file
asserted that the profiled fit covered nominally. a893d85bc retired the profile
(a derivative-free hyperparameter search, forbidden by SPEC.md): a bare
``tweedie`` is now refused and the caller must name the power. This file keeps
the user-visible symptom and its falsifiable contrast under that contract:

  * bare ``family="tweedie"``            -> refused, never a silent default;
  * ``family="tweedie(1.1)"`` (the truth) -> low-mean coverage ~nominal;
  * ``family="tweedie(1.5)"``            -> ``p`` pinned at the old fallback on
    the same ``p_true = 1.1`` data -> low-mean coverage under-covers.

If the interval ignored the requested power, the true-power fit would inherit
the fallback's under-coverage AND the contrast would collapse (the two fits
would cover identically), so the tests cannot pass for the wrong reason.
"""

import numpy as np
import pandas as pd
import pytest

import gamfit

P_TRUE = 1.1  # worst case in the #2026 report (low-mean quartile fell to ~0.73)
PHI = 0.6
LEVEL = 0.90
NOMINAL_DEFICIT = 0.06  # an interval labelled 90% may miss by at most 6 points


def _tweedie_sample(rng, mu, phi, power):
    """Compound Poisson-Gamma (Jorgensen) Tweedie variate per row: ``N`` jumps,
    ``N ~ Poisson(lam)``, each ``Gamma(alpha, scale)``, summed (0 when ``N==0``)."""
    lam = mu ** (2.0 - power) / (phi * (2.0 - power))
    alpha = (2.0 - power) / (power - 1.0)
    scale = phi * (power - 1.0) * mu ** (power - 1.0)
    n = rng.poisson(lam)
    out = np.zeros_like(mu)
    for i in range(len(mu)):
        if n[i] > 0:
            out[i] = rng.gamma(alpha * n[i], scale[i])
    return out


def _gradient_tweedie(rng, n):
    """Strong log-linear mean gradient so the low- and high-mean strata are well
    separated (that separation is what exposes conditional miscoverage)."""
    x = rng.uniform(0, 1, n)
    mu = np.exp(-0.5 + 3.0 * x)
    y = _tweedie_sample(rng, mu, PHI, P_TRUE)
    return pd.DataFrame({"x": x, "y": y}), mu


def _low_mean_quartile_coverage(model, test):
    pr = model.predict(test, interval=LEVEL, observation_interval=True)
    y = test["y"].to_numpy()
    lo = pr["observation_lower"].to_numpy()
    hi = pr["observation_upper"].to_numpy()
    mean = pr["posterior_mean"].to_numpy()
    inside = (y >= lo) & (y <= hi)
    low_cut = np.quantile(mean, 0.25)
    low_stratum = mean <= low_cut
    return inside[low_stratum].mean(), inside.mean(), mean


def test_bare_tweedie_is_refused_instead_of_silently_pinning_1p5():
    rng = np.random.default_rng(20262)
    train, _ = _gradient_tweedie(rng, 600)
    with pytest.raises(Exception, match="explicit variance power"):
        gamfit.fit(train, "y ~ s(x)", family="tweedie")


def test_true_power_low_mean_coverage_is_near_nominal_on_p_neq_1p5_data():
    rng = np.random.default_rng(20260)
    train, _ = _gradient_tweedie(rng, 6000)
    test, mu_true = _gradient_tweedie(rng, 20000)

    m_true = gamfit.fit(train, "y ~ s(x)", family=f"tweedie({P_TRUE})")

    # Precondition: the mean must be recovered, so a coverage failure is an
    # INTERVAL defect (the reported bug), not a broken fit. The log-link mean is
    # robust to the power regardless, so this holds for both fits.
    pr = m_true.predict(test, interval=LEVEL, observation_interval=True)
    mean_hat = pr["posterior_mean"].to_numpy()
    corr = np.corrcoef(np.log(mean_hat), np.log(mu_true))[0, 1]
    assert corr > 0.95, f"mean not recovered (precondition failed): corr={corr:.3f}"

    low_true, _, _ = _low_mean_quartile_coverage(m_true, test)

    # The true-power fit must cover the low-mean stratum within 6 points of
    # nominal. With the fixed p=1.5 fallback on p_true=1.1 data this stratum fell
    # to ~0.73 -- a ~17-point deficit -- so this bound is a decisive gate.
    assert low_true >= LEVEL - NOMINAL_DEFICIT, (
        f"family='tweedie({P_TRUE})' under-covers the low-mean quartile: {low_true:.3f} "
        f"< {LEVEL - NOMINAL_DEFICIT:.3f} (nominal {LEVEL}); the observation interval "
        f"does not honour the requested power."
    )


def test_true_power_strictly_beats_pinned_1p5_in_low_mean_stratum():
    """Mechanism check: on the SAME p_true=1.1 data, pinning tweedie(1.5) (the old
    hardcoded fallback) reproduces the low-mean under-coverage, while naming the
    true power fixes it. This proves the cure is the power the interval uses, not
    an incidental change in interval shape or width."""
    rng = np.random.default_rng(20261)
    train, _ = _gradient_tweedie(rng, 6000)
    test, _ = _gradient_tweedie(rng, 20000)

    m_true = gamfit.fit(train, "y ~ s(x)", family=f"tweedie({P_TRUE})")
    m_pinned = gamfit.fit(train, "y ~ s(x)", family="tweedie(1.5)")

    low_true, _, _ = _low_mean_quartile_coverage(m_true, test)
    low_pinned, _, _ = _low_mean_quartile_coverage(m_pinned, test)

    # The pinned-1.5 fit is the pre-#2026 behaviour and must still under-cover.
    assert low_pinned < LEVEL - NOMINAL_DEFICIT, (
        f"precondition: pinned tweedie(1.5) should under-cover the low-mean "
        f"stratum on p_true={P_TRUE} data, but got {low_pinned:.3f}"
    )
    # Naming the true power must materially improve low-mean coverage over the
    # pinned fallback -- a strict, seed-robust gap.
    assert low_true >= low_pinned + 0.08, (
        f"the true power did not repair low-mean coverage: true={low_true:.3f} "
        f"vs pinned-1.5={low_pinned:.3f} (need >= +0.08)"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
