"""Bug hunt (#1618): a purely parametric Gaussian fit with integer case weights
must reproduce the row-expanded fit's standard errors, not just its point
estimates.

``weights`` are documented frequency / case weights — a row with weight ``w``
contributes ``w`` copies of the row to the likelihood
(``crates/gam-terms/src/inference/lawley.rs``, ``RowKappas::weighted``) — so a
weighted fit must reproduce the row-expanded fit (each row repeated ``w`` times)
in *both* coefficients and their standard errors.

Root cause (fixed): the profiled Gaussian-identity scale was
``phi_hat = sum(w_i * r_i^2) / (n_+ - p)`` with ``n_+`` the count of positive-
weight ROWS. The row-expanded fit divides the same numerator by ``sum(w) - p``.
Since the reported covariance is ``phi_hat * (X'WX)^-1`` and ``X'WX`` is
identical in both encodings,

    SE_weighted / SE_expanded = sqrt(phi_hat_w / phi_hat_e)
                              = sqrt((sum(w) - p) / (n_+ - p)) ~ sqrt(sum(w)/n_+),

so with weights in {1,2,3} every SE was inflated by ``~sqrt(2) ~ 1.41``. The fix
makes the denominator the effective sample size ``sum(w_i)``.
"""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pandas as pd

gamfit = import_module("gamfit")


def _weighted_and_expanded(seed: int, n: int, weight_hi: int):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    w = rng.integers(1, weight_hi, n).astype(float)  # frequency weights
    y = 2.0 + 1.5 * x + rng.normal(0.0, 0.7, n)

    df = pd.DataFrame({"x": x, "y": y, "w": w})
    idx = np.repeat(np.arange(n), w.astype(int))
    dfe = df.iloc[idx].reset_index(drop=True)  # row expansion

    mw = gamfit.fit(df, "y ~ x", family="gaussian", weights="w")
    me = gamfit.fit(dfe, "y ~ x", family="gaussian")
    fw = mw.summary().coefficients_frame()
    fe = me.summary().coefficients_frame()
    return fw, fe, float(w.sum() / n)


def test_parametric_gaussian_weighted_point_estimates_match_row_expansion() -> None:
    """Control half of the contract: the point estimates already matched before
    the fix (the Gram ``X'WX`` and cross-product ``X'Wy`` are encoding-
    invariant). Keeping this explicit guards against a fix that "corrects" the
    SEs by perturbing ``beta_hat``."""
    fw, fe, _ = _weighted_and_expanded(seed=5, n=500, weight_hi=4)
    est_ratio = (np.asarray(fw["estimate"]) / np.asarray(fe["estimate"]))
    assert np.allclose(est_ratio, 1.0, atol=1e-4), (
        f"estimate ratios drifted from 1.0: {est_ratio.tolist()}"
    )


def test_parametric_gaussian_weighted_standard_errors_match_row_expansion() -> None:
    """The headline #1618 regression: with weights in {1,2,3} (sum(w) ~ 2n)
    every reported SE was inflated by exactly ``sqrt(sum(w)/n) ~ 1.41``. The
    weighted and row-expanded SEs must now coincide."""
    fw, fe, sqrt_factor_sq = _weighted_and_expanded(seed=5, n=500, weight_hi=4)
    se_w = np.asarray(fw["std_error"], dtype=float)
    se_e = np.asarray(fe["std_error"], dtype=float)
    se_ratio = se_w / se_e
    assert np.allclose(se_ratio, 1.0, atol=1e-2), (
        f"std_error ratios {se_ratio.tolist()} != 1.0; the pre-fix bug inflated "
        f"them by ~sqrt(sum(w)/n) = {np.sqrt(sqrt_factor_sq):.4f}"
    )


def test_parametric_gaussian_weighted_se_large_weight_range() -> None:
    """Different angle: push the weight range to {1..7} so ``sum(w)/n ~ 4`` and
    the pre-fix inflation factor is ``~2x`` (not ``~1.41x``). A fix that merely
    nudged a tolerance rather than correcting the denominator would fail here.
    Verified across several seeds to rule out a lucky draw."""
    for seed in range(5):
        fw, fe, ratio_sq = _weighted_and_expanded(seed=seed, n=600, weight_hi=8)
        se_ratio = np.asarray(fw["std_error"], dtype=float) / np.asarray(
            fe["std_error"], dtype=float
        )
        assert np.allclose(se_ratio, 1.0, atol=1e-2), (
            f"seed {seed}: SE ratios {se_ratio.tolist()} != 1.0 "
            f"(pre-fix factor ~{np.sqrt(ratio_sq):.3f})"
        )
