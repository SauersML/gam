"""Regression test for issue #877: Gaussian REML weight-scale invariance.

A Gaussian REML fit with profiled dispersion + inverse-variance weights must be
INVARIANT to a global weight rescale ``w -> c*w``: only weight ratios carry
information, so the magnitude is absorbed by ``phi_hat -> c*phi_hat`` (hence
``lambda_hat -> c*lambda_hat``) and the fit (predictions, EDF, SE) is unchanged.

Before the fix (commit 81a34369b anchoring the outer rho-prior to the weight
geometric mean) the repro saw ``max|mean(1) - mean(1000)| = 0.014`` and the
smoothing parameters scaled by only 702x / 231x instead of the analytically
required 1000x.
"""

from __future__ import annotations

import pytest

pytest.importorskip("gamfit._rust")

import numpy as np
import pandas as pd

import gamfit


def _fit_at_weight(y, x, grid, C):
    df = pd.DataFrame(dict(y=y, x=x, w=np.full(len(y), float(C))))
    m = gamfit.fit(df, "y ~ s(x)", weights="w")
    p = m.predict(grid, interval=0.95)
    mean = np.asarray(p["mean"], dtype=float)
    std_error = np.asarray(p["std_error"], dtype=float)
    sps = list(m.smoothing_parameters().values())
    return mean, std_error, sps


def test_gaussian_reml_weight_rescaling_invariance_877():
    rng = np.random.default_rng(41)
    n = 500
    x = rng.uniform(-3, 3, n)
    y = np.sin(x) + rng.normal(0, 0.5, n)
    grid = pd.DataFrame(dict(x=np.linspace(-3, 3, 40)))

    c = 1000.0
    mean1, se1, sp1 = _fit_at_weight(y, x, grid, 1.0)
    meanC, seC, spC = _fit_at_weight(y, x, grid, c)

    # Predictions are invariant under a global weight rescale.
    max_pred_diff = float(np.max(np.abs(mean1 - meanC)))
    assert max_pred_diff < 1e-3, (
        f"predictions changed under w -> {c}*w: max|diff| = {max_pred_diff} "
        f"(issue #877 saw 0.014 when broken)"
    )

    # Each smoothing parameter must scale by exactly c (lambda_hat -> c*lambda_hat).
    assert len(sp1) == len(spC) and len(sp1) >= 1
    for i, (s1, sC) in enumerate(zip(sp1, spC)):
        assert s1 > 0.0 and sC > 0.0, f"smoothing parameter {i} non-positive"
        ratio = sC / s1
        rel = abs(ratio - c) / c
        assert rel < 0.01, (
            f"smoothing parameter {i} scaled by {ratio} under w -> {c}*w; "
            f"expected ~{c} (rel err {rel}; issue #877 saw 702x / 231x when broken)"
        )

    # Control: the SE was already correct and must stay invariant.
    se_ratio = seC / se1
    assert np.all(np.isfinite(se_ratio))
    assert np.max(np.abs(se_ratio - 1.0)) < 1e-2, (
        f"standard errors changed under w -> {c}*w: "
        f"max|se(C)/se(1) - 1| = {float(np.max(np.abs(se_ratio - 1.0)))}"
    )
