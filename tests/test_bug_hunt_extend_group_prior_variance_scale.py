"""Regression tests for the unseen random-effect-level default prior variance.

``Model.extend_with_group`` adds a brand-new random-effect level at deployment
time (no refit) and assigns the new coefficient a *default* prior variance when
the caller does not supply one explicitly. The correct default is the fitted
random-effect variance component

    sigma_b^2 = phi_hat / lambda            (mgcv: lambda = phi_hat / sigma_b^2)

where ``phi_hat`` is the residual dispersion that scales every predict-time
covariance (Var(beta|lambda) = H^-1 * phi_hat). A previous implementation used
the scale-free ``1 / lambda``, silently dropping ``phi_hat``. For fixed-scale
families (Poisson/Binomial) ``phi_hat == 1`` and the two agree, but for Gaussian
(and Gamma/Tweedie/NB with an estimated scale) the default was wrong by a factor
``1 / phi_hat`` and was not response-scale equivariant.

These tests probe the corrected behaviour from three orthogonal directions, all
free of internal knobs (the fitted ``lambda`` is not exposed to Python):

1. ``..._is_response_scale_equivariant`` — scaling ``y`` by ``c`` must scale the
   prior variance by ``c**2`` (``phi`` scales by ``c**2``, ``lambda`` is
   invariant). The buggy ``1/lambda`` does not move at all.
2. ``..._matches_fitted_variance_component`` — with many observations per group
   the in-sample BLUPs are barely shrunk, so their empirical variance estimates
   ``sigma_b^2``; the unseen-level prior variance must match it. The buggy value
   is ``1/phi`` times too large.
3. ``..._is_insensitive_to_residual_noise_level`` — holding the *true* group
   variance fixed while raising the residual noise leaves ``sigma_b^2 = phi/lambda``
   ~constant, because ``lambda`` grows in proportion to ``phi``. The buggy
   ``1/lambda = sigma_b^2 / phi`` instead collapses like ``1/sigma_e^2``.

Regression target for #674.
"""
from __future__ import annotations

import contextlib
import importlib
import io
from typing import Any

pytest: Any = importlib.import_module("pytest")

pytest.importorskip("gamfit._rust")

import numpy as np
import pandas as pd

import gamfit


def _unseen_prior_var(
    scale_y: float = 1.0,
    *,
    sigma_e: float = 0.7,
    sigma_b: float = 1.5,
    seed: int = 0,
    n: int = 6000,
    groups: int = 80,
) -> tuple[float, "gamfit.Model"]:
    """Fit ``y ~ x + group(g)`` and return the unseen-level default prior
    variance together with the fitted model.

    ``y = scale_y * (1 + 0.5 x + b_g + sigma_e * eps)`` with the group effects
    ``b_g`` drawn at standard deviation ``sigma_b`` and centred, so the true
    random-effect variance component is ``(scale_y * sigma_b)**2``.
    """
    rng = np.random.default_rng(seed)
    g = rng.integers(0, groups, size=n)
    x = rng.normal(size=n)
    group_effect = rng.normal(size=groups) * sigma_b
    group_effect -= group_effect.mean()
    y = scale_y * (1.0 + 0.5 * x + group_effect[g] + rng.normal(size=n) * sigma_e)
    frame = pd.DataFrame({"y": y, "x": x, "g": g.astype(str)})
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        model = gamfit.fit(frame, "y ~ x + group(g)", family="gaussian")
        extended = model.extend_with_group({"term": "g", "level": "9999"})
    variance = float(extended.deployment_extensions[0]["coefficient_variance"])
    return variance, model


def _fitted_variance_component(model: "gamfit.Model") -> float:
    """Empirical variance of the in-sample group BLUPs.

    The first two coefficients are the intercept and the slope on ``x``; the
    remaining ones are the per-level random effects.
    """
    estimates = np.array([row["estimate"] for row in model.summary().coefficients[2:]])
    return float(estimates.var())


def test_unseen_group_prior_variance_is_response_scale_equivariant() -> None:
    """v(c*y) / v(y) == c**2 for the Gaussian unseen-level default prior."""
    base, _ = _unseen_prior_var(scale_y=1.0)
    scaled, _ = _unseen_prior_var(scale_y=4.0)
    assert base > 0.0
    # phi scales by 16, lambda is invariant => prior variance scales by 16.
    # The buggy 1/lambda gives a ratio of 1.0.
    assert scaled / base == pytest.approx(16.0, rel=1e-6)


def test_unseen_group_prior_variance_matches_fitted_variance_component() -> None:
    """The unseen-level default prior variance equals the in-sample variance
    component sigma_b^2 = phi/lambda (not 1/phi times too large)."""
    variance, model = _unseen_prior_var(scale_y=1.0)
    fitted = _fitted_variance_component(model)
    # With ~75 obs/group the BLUPs are barely shrunk, so their empirical
    # variance is a tight estimate of sigma_b^2; allow a modest tolerance for
    # the small remaining shrinkage and the centring degree of freedom.
    assert variance == pytest.approx(fitted, rel=0.12)
    # And it is comfortably below the buggy 1/lambda value, which is ~1/phi
    # (~1/0.49 ~ 2x) too large for this design.
    assert variance < 1.6 * fitted


def test_unseen_group_prior_variance_is_insensitive_to_residual_noise_level() -> None:
    """Holding the true group variance fixed, raising the residual noise leaves
    the corrected prior variance ~constant.

    sigma_b^2 = phi / lambda is a property of the random-effect distribution, so
    it should not track the residual noise sigma_e. lambda = phi / sigma_b^2
    grows like phi as the noise rises, cancelling it. The buggy 1/lambda =
    sigma_b^2 / phi instead shrinks like 1/sigma_e^2 — a >5x drop across the two
    noise levels below.
    """
    low_noise, _ = _unseen_prior_var(sigma_e=0.5, sigma_b=1.5)
    high_noise, _ = _unseen_prior_var(sigma_e=1.5, sigma_b=1.5)
    # phi differs by (1.5/0.5)**2 = 9x between the two fits; the corrected prior
    # variance must stay close to the shared true sigma_b^2 ~ 2.0**... ~ 2.25.
    assert low_noise == pytest.approx(high_noise, rel=0.25)
    # The buggy 1/lambda would have collapsed by ~9x; assert we are nowhere near
    # that failure mode.
    assert high_noise > 0.5 * low_noise


def test_unseen_group_prior_variance_respects_explicit_prior() -> None:
    """An explicitly supplied prior variance still overrides the default and is
    itself untouched by the scale correction."""
    rng = np.random.default_rng(3)
    n, groups = 2000, 40
    g = rng.integers(0, groups, size=n)
    x = rng.normal(size=n)
    group_effect = rng.normal(size=groups)
    group_effect -= group_effect.mean()
    y = 1.0 + 0.3 * x + group_effect[g] + rng.normal(size=n) * 0.5
    frame = pd.DataFrame({"y": y, "x": x, "g": g.astype(str)})
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        model = gamfit.fit(frame, "y ~ x + group(g)", family="gaussian")
        extended = model.extend_with_group(
            {"term": "g", "level": "9999"},
            prior={"variance": 3.14},
        )
    assert float(extended.deployment_extensions[0]["coefficient_variance"]) == pytest.approx(
        3.14, rel=1e-12
    )
