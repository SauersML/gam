"""Bug hunt: `summary()`'s Wald smooth-term p-value truncated the WRONG
eigen-subspace, reporting a dominant, overwhelmingly-significant smooth as
non-significant (issue #2142).

With ``y = sin(2*pi*x1) + 0.35*x2 + noise`` the term ``s(x1)`` is the entire
signal: the likelihood-ratio smooth test (``smooth_significance``) gives a
statistic of several hundred with ``p = 0``. Yet ``summary()`` reported
``chi_sq ~ 1.74``, ``p ~ 0.99`` for that same term.

Root cause: the summary Wald test (`gam::inference::smooth_test::wood_smooth_test`)
truncated the RAW fitted coefficient covariance block — it eigendecomposed
``Vp[block]`` directly, sorted eigenvalues descending, and summed
``proj^2 / lambda`` over only the ``round(edf)`` LARGEST. For a genuinely wiggly
smooth the fitted signal lives in the best-determined (small raw-variance)
coefficient directions, so keeping the largest-variance eigenvalues discards
exactly the signal and yields a tiny statistic for a hugely significant term.

The genuine Wood (2013) statistic is ``f' Vf^-_r f`` where ``Vf = R V R'`` is the
DESIGN-WHITENED covariance (``R'R = X'WX``, the term's weighted Gram) and
``f = R beta`` are the fitted values; the rank-r truncation there selects the
least-penalized directions that carry the fit. The fix threads the stored
weighted Gram ``X'WX = H - S(lambda)`` into the test as the whitening metric.

This is the `summary()` Wald path only; `smooth_significance()` (the LR refit)
was always correct and is used here as the ground-truth cross-check.
"""

import os

os.environ.setdefault("GAM_LOG", "off")

import numpy as np
import pandas as pd

import gamfit


def _fit(seed, n=300):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    y = np.sin(2 * np.pi * x1) + 0.35 * x2 + rng.normal(0, 0.3, n)
    d = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    m = gamfit.fit(d, "y ~ s(x1)+s(x2)", family="gaussian")
    return m, d


def _wald_rows(m):
    return {r["name"]: r for r in m.summary().smooth_terms}


def _lr_rows(m, d):
    return {r["name"]: r for r in m.smooth_significance(d)}


def test_dominant_smooth_wald_pvalue_is_significant():
    """The exact issue #2142 repro (seed 11): the dominant ``s(x1)`` term whose
    LR statistic is ~536 (p=0) must NOT be reported non-significant by the
    summary Wald p-value."""
    m, d = _fit(11)
    wald = _wald_rows(m)["s(x1)"]
    lr = _lr_rows(m, d)["s(x1)"]

    # Ground truth: the LR test finds the term overwhelmingly significant.
    assert lr["statistic_lr"] > 100.0, f"LR statistic unexpectedly small: {lr}"
    assert lr["p_value_corrected"] < 1e-6

    # The bug: summary Wald reported chi_sq ~ 1.74, p ~ 0.99 for this same term.
    assert wald["chi_sq"] is not None and wald["p_value"] is not None
    assert wald["p_value"] < 1e-3, (
        f"dominant smooth reported non-significant by summary Wald: "
        f"chi_sq={wald['chi_sq']:.3f}, p={wald['p_value']:.4g} (LR p=0)"
    )
    # The Wald chi_sq must be commensurate with the term's dominance, not the
    # ~1.74 the raw-covariance truncation produced.
    assert wald["chi_sq"] > 50.0, f"Wald statistic too small: {wald['chi_sq']}"


def test_dominant_smooth_significant_across_seeds():
    """The issue notes the summary p for this dominant term was erratic across
    seeds (0.79, 0.99, 0.00) — a symptom of the truncation catching the signal
    only by luck. With the whitening fix every seed must read as significant."""
    for seed in (1, 3, 11, 21, 42):
        m, d = _fit(seed)
        wald = _wald_rows(m)["s(x1)"]
        assert wald["p_value"] is not None and wald["p_value"] < 1e-3, (
            f"seed {seed}: dominant s(x1) not significant "
            f"(chi_sq={wald['chi_sq']}, p={wald['p_value']})"
        )


def test_wald_and_lr_agree_on_significance_direction():
    """Cross-check the whitened Wald against the independent LR refit on BOTH
    terms: whichever the LR test finds significant, the Wald must too, and the
    genuinely weak term must not be spuriously inflated (guards against the
    whitening over-correcting every term to significant)."""
    m, d = _fit(7)
    wald = _wald_rows(m)
    lr = _lr_rows(m, d)
    for name in ("s(x1)", "s(x2)"):
        w = wald[name]
        l = lr[name]
        if l["p_value_corrected"] < 1e-4:
            assert w["p_value"] < 1e-2, (
                f"{name}: LR significant (p={l['p_value_corrected']:.3g}) but "
                f"Wald not (p={w['p_value']:.3g})"
            )
        # A p-value is always a valid probability.
        assert 0.0 <= w["p_value"] <= 1.0


def test_pure_noise_smooth_not_spuriously_significant():
    """A smooth of a covariate that carries NO signal must not be reported
    significant. This is the false-positive guard: the whitening must recover
    real signal without manufacturing it where there is none."""
    rng = np.random.default_rng(0)
    n = 400
    x1 = rng.uniform(0, 1, n)
    z = rng.uniform(0, 1, n)  # pure noise covariate
    y = np.sin(2 * np.pi * x1) + rng.normal(0, 0.3, n)
    d = pd.DataFrame({"y": y, "x1": x1, "z": z})
    m = gamfit.fit(d, "y ~ s(x1)+s(z)", family="gaussian")
    wald = _wald_rows(m)
    # Signal term significant...
    assert wald["s(x1)"]["p_value"] < 1e-3
    # ...noise term not (comfortably above a strict alpha).
    assert wald["s(z)"]["p_value"] > 0.02, (
        f"pure-noise smooth spuriously significant: p={wald['s(z)']['p_value']}"
    )


if __name__ == "__main__":
    test_dominant_smooth_wald_pvalue_is_significant()
    test_dominant_smooth_significant_across_seeds()
    test_wald_and_lr_agree_on_significance_direction()
    test_pure_noise_smooth_not_spuriously_significant()
    print("all passed")
