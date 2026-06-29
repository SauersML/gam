"""Contract lock for Gaussian case/prior weights (rebuts #1617 / #1618).

Two bug-hunt issues (#1617, #1618) claimed a defect: a Gaussian fit with integer
weights does not reproduce the row-expanded fit (each row repeated ``w`` times),
even though Poisson / binomial / Gamma do. That asymmetry is **correct and
intended**, not a bug — this test pins it down so the false premise is not
re-filed and the behaviour cannot silently drift.

Why the asymmetry is correct
----------------------------
* Fixed-dispersion families (Poisson, binomial, Gamma) have **no scale
  parameter**. A prior weight ``w`` and ``w`` literal copies enter the LAML
  identically, so the weighted fit reproduces the row-expanded fit exactly.
  Frequency-weight and inverse-variance-weight interpretations coincide.

* The Gaussian identity-link fit has a **profiled scale** ``phi_hat``. Here the
  engine — like mgcv / Wood (2017) §6.2.7 — treats ``weights`` as **prior
  (inverse-variance) weights**: ``Var(y_i) = phi / w_i`` and the scale is
  ``phi_hat = sum(w_i r_i^2) / (n - edf)`` with ``n`` the number of *rows*. Only
  weight *ratios* carry information; a global rescale ``w -> c*w`` is absorbed by
  ``phi_hat -> c*phi_hat`` (``lambda_hat -> c*lambda_hat``) leaving the fit —
  predictions, EDF and **standard errors** — invariant (issue #877).

These two facts are mutually exclusive with row-expansion equivalence for the
Gaussian scale: row expansion by ``c`` would shrink every SE by ``sqrt(c)``,
whereas the prior-weight convention keeps SEs invariant under a uniform rescale.
The engine deliberately implements the prior-weight convention for Gaussian (to
match mgcv), so the weighted Gaussian fit is NOT row-expansion-equivalent — by
design. (To obtain genuine frequency-weight Gaussian behaviour, replicate the
rows explicitly.)
"""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pandas as pd

gamfit = import_module("gamfit")


def test_fixed_dispersion_weighted_matches_row_expansion() -> None:
    """Scale-free families: weighted fit == row-expanded fit (exact). This is
    the half of the bug-hunt observation that IS the intended contract."""
    rng = np.random.default_rng(11)
    n = 400
    x = rng.uniform(0.0, 1.0, n)
    w = rng.integers(1, 4, n).astype(float)
    y = rng.poisson(np.exp(0.5 + np.sin(2.0 * np.pi * x))).astype(float)
    df = pd.DataFrame({"x": x, "y": y, "w": w})
    idx = np.repeat(np.arange(n), w.astype(int))
    dfe = df.iloc[idx].reset_index(drop=True)

    formula = 'y ~ s(x, bs="ps", k=15)'
    mw = gamfit.fit(df, formula, family="poisson", weights="w")
    me = gamfit.fit(dfe, formula, family="poisson")

    grid = pd.DataFrame({"x": np.linspace(0.03, 0.97, 80)})
    pw = np.asarray(mw.predict(grid), dtype=float)
    pe = np.asarray(me.predict(grid), dtype=float)
    rel = float(np.max(np.abs(pw - pe)) / np.ptp(pe))
    assert rel < 1.5e-3, f"Poisson weighted vs row-expansion pred drift {rel:.3e}"
    assert abs(float(mw.summary().edf_total) - float(me.summary().edf_total)) < 0.05


def test_gaussian_weights_are_prior_weights_rescale_invariant() -> None:
    """Profiled-scale family: a global weight rescale ``w -> c*w`` is a no-op for
    the fit (predictions, EDF, SE). This is the prior-weight (inverse-variance)
    convention that makes the weighted Gaussian fit intentionally NOT
    row-expansion-equivalent. Mirrors issue #877 with a strong c=1000 rescale."""
    rng = np.random.default_rng(41)
    n = 500
    x = rng.uniform(-3.0, 3.0, n)
    y = np.sin(x) + rng.normal(0.0, 0.5, n)
    grid = pd.DataFrame({"x": np.linspace(-3.0, 3.0, 40)})

    def fit_at(c: float):
        df = pd.DataFrame({"y": y, "x": x, "w": np.full(n, float(c))})
        m = gamfit.fit(df, "y ~ s(x)", family="gaussian", weights="w")
        p = m.predict(grid, interval=0.95)
        return (
            np.asarray(p["mean"], dtype=float),
            np.asarray(p["std_error"], dtype=float),
            float(m.summary().edf_total),
        )

    mean1, se1, edf1 = fit_at(1.0)
    meanC, seC, edfC = fit_at(1000.0)

    assert float(np.max(np.abs(mean1 - meanC))) < 1e-3, (
        "Gaussian predictions must be invariant under a global weight rescale "
        "(prior-weight convention); a change here would be the #877 regression"
    )
    assert abs(edf1 - edfC) < 0.05, "EDF must be rescale-invariant"
    se_ratio = seC / se1
    assert np.max(np.abs(se_ratio - 1.0)) < 1e-2, (
        "Gaussian standard errors must be invariant under a global weight "
        "rescale — they do NOT shrink by sqrt(c) the way row-expansion would. "
        "This is the prior-weight semantics that makes weighted Gaussian "
        "intentionally not row-expansion-equivalent (#1617/#1618 working as "
        "intended)."
    )


def test_gaussian_weighted_se_is_not_row_expansion_se() -> None:
    """The explicit statement of the intended #1618 behaviour: a Gaussian fit
    with non-uniform integer weights does NOT reproduce the row-expanded fit's
    standard errors — it reproduces the *prior-weight* SEs (row-count scale
    denominator). The point estimates DO coincide (the estimating equations are
    weight-linear). This guards against a future 'fix' that mistakes the
    deliberate prior-weight convention for a bug and switches the Gaussian scale
    denominator to sum(w)."""
    rng = np.random.default_rng(5)
    n = 500
    x = rng.uniform(0.0, 1.0, n)
    w = rng.integers(1, 4, n).astype(float)
    y = 2.0 + 1.5 * x + rng.normal(0.0, 0.7, n)
    df = pd.DataFrame({"x": x, "y": y, "w": w})
    idx = np.repeat(np.arange(n), w.astype(int))
    dfe = df.iloc[idx].reset_index(drop=True)

    mw = gamfit.fit(df, "y ~ x", family="gaussian", weights="w")
    me = gamfit.fit(dfe, "y ~ x", family="gaussian")
    fw = mw.summary().coefficients_frame()
    fe = me.summary().coefficients_frame()

    # Point estimates coincide (weight-linear estimating equations).
    est_ratio = np.asarray(fw["estimate"]) / np.asarray(fe["estimate"])
    assert np.allclose(est_ratio, 1.0, atol=1e-4), est_ratio.tolist()

    # SEs intentionally differ by ~sqrt(sum(w)/n): prior-weight (row-count)
    # scale, NOT the frequency (sum-of-weights) scale that row expansion gives.
    se_ratio = np.asarray(fw["std_error"], dtype=float) / np.asarray(
        fe["std_error"], dtype=float
    )
    expected = float(np.sqrt(w.sum() / n))
    assert np.allclose(se_ratio, expected, rtol=5e-2), (
        f"weighted/expanded SE ratio {se_ratio.tolist()} should be ~sqrt(Σw/n) "
        f"= {expected:.4f} (prior-weight scale), not 1.0 (frequency scale)"
    )
