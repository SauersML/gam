"""Scaled Student-t response family (pyGAM audit accuracy ACC-5 / C2).

On the audit's ``outlier_n300`` data — a two-period sine with 5% of rows hit by
heavy-tailed ``t(1.5)`` noise — a Gaussian REML fit is pulled to a straight
line on some cross-validation folds. ``family="student-t"`` estimates the scale
σ and degrees of freedom ν by LAML jointly with the smoothing parameters, so
the outliers are downweighted and the curvature survives. The pyGAM
match-or-beat comparison on identical folds lives in the Rust quality suite
(``quality_vs_pygam_student_t_outliers``); these tests pin the Python surface:
the one family spelling, the reported (σ̂, ν̂), and the held-out gain over
the Gaussian fit.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import KFold

import gamfit

_N = 300


def _outlier_n300() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(_N + 3)
    x = rng.uniform(0.0, 1.0, _N)
    mu = np.sin(2.0 * np.pi * 2.0 * x)
    y = mu + rng.normal(0.0, 0.3, _N)
    idx = rng.choice(_N, _N // 20, replace=False)
    y[idx] += rng.standard_t(1.5, len(idx)) * 5.0
    return x, y, mu


def test_only_the_hyphenated_family_spelling_is_accepted() -> None:
    x, y, _ = _outlier_n300()
    for name in ("student_t", "t"):
        with pytest.raises(Exception, match=f"unknown family `{name}`; use `student-t`"):
            gamfit.fit({"x": x, "y": y}, "y ~ s(x)", family=name)
    model = gamfit.fit({"x": x, "y": y}, "y ~ s(x)", family="student-t")
    assert model.student_t_sigma is not None


def test_fitted_scale_and_degrees_of_freedom_are_reported() -> None:
    x, y, _ = _outlier_n300()
    robust = gamfit.fit({"x": x, "y": y}, "y ~ s(x)", family="student-t")
    sigma, nu = robust.student_t_sigma, robust.student_t_nu
    assert sigma is not None and nu is not None
    # The clean noise has sd 0.3; the contamination is t(1.5)-tailed, so the
    # fitted law must be heavy-tailed (far from the Gaussian limit) with a
    # scale on the order of the clean noise, not of the outliers.
    assert 0.1 < sigma < 0.6, sigma
    assert 1.0 < nu < 30.0, nu

    gaussian = gamfit.fit({"x": x, "y": y}, "y ~ s(x)")
    assert gaussian.student_t_sigma is None
    assert gaussian.student_t_nu is None


def test_outlier_n300_keeps_curvature_and_beats_gaussian_truth_mse() -> None:
    x, y, mu = _outlier_n300()
    robust_sq, gaussian_sq = [], []
    for train, test in KFold(5, shuffle=True, random_state=0).split(x):
        data = {"x": x[train], "y": y[train]}
        robust = gamfit.fit(data, "y ~ s(x)", family="student-t")
        gaussian = gamfit.fit(data, "y ~ s(x)")
        # A collapse to a line has the edf of the intercept plus the linear
        # null space of the second-derivative penalty (2).
        assert robust.summary().edf_total > 3.0
        for model, sink in ((robust, robust_sq), (gaussian, gaussian_sq)):
            pred = np.asarray(model.predict({"x": x[test]}), float).ravel()
            sink.extend((pred - mu[test]) ** 2)
    assert np.mean(robust_sq) < np.mean(gaussian_sq)
