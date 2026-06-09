"""Regression test for issue #879: honest latent-fit diagnostics.

``gaussian_reml_optimize_latent`` now (a) reports the PROJECTED gradient as
``grad_t_norm`` and (b) carries payload keys ``response_r2``,
``response_residual_norm``, and ``latent_t_std`` so a caller can distinguish a
good decoder fit (high ``response_r2``) whose latent gradient simply did not
reach ``grad_tol`` (``converged`` may be False) from a genuine latent collapse
(``latent_t_std ~ 0``). Landed in commit 7a350692d.

The assertions check the CONTRACT (keys present, finite, sane ranges) on real
numerical data rather than brittle exact values.
"""

from __future__ import annotations

import pytest

pytest.importorskip("gamfit._rust")

import numpy as np

import gamfit


def test_latent_optimize_diagnostics_keys_879():
    rng = np.random.RandomState(0)
    n = 40
    th = np.sort(rng.uniform(0, 2 * np.pi, n))
    Y = np.c_[np.cos(th), np.sin(th)] + 0.02 * rng.randn(n, 2)
    Y = np.c_[Y, 0.02 * rng.randn(n, 3)]
    C = np.linspace(0, 1, 12).reshape(-1, 1)

    r = gamfit.gaussian_reml_optimize_latent(
        y=Y.astype(float),
        n_obs=n,
        latent_dim=1,
        centers=C,
        penalty=np.eye(12),
        m=2,
        manifold="euclidean",
        basis_kind="duchon",
        max_iter=200,
        seed=0,
    )

    # The honest-diagnostics keys (and the existing ones) must be present.
    for key in (
        "response_r2",
        "response_residual_norm",
        "latent_t_std",
        "grad_t_norm",
        "grad_t_norm_scaled",
        "converged",
    ):
        assert key in r, f"missing diagnostic key {key!r} in result payload"

    response_r2 = float(r["response_r2"])
    response_residual_norm = float(r["response_residual_norm"])
    latent_t_std = float(r["latent_t_std"])
    grad_t_norm = float(r["grad_t_norm"])
    grad_t_norm_scaled = float(r["grad_t_norm_scaled"])

    # A reasonable euclidean decoder fit reconstructs the response well.
    assert np.isfinite(response_r2)
    assert response_r2 > 0.3, f"decoder did not reconstruct: response_r2 = {response_r2}"

    # The latent was recovered, not collapsed onto a single point.
    assert np.isfinite(latent_t_std)
    assert latent_t_std > 0.0, f"latent collapsed: latent_t_std = {latent_t_std}"

    # Residual norm is a finite, non-negative magnitude.
    assert np.isfinite(response_residual_norm)
    assert response_residual_norm >= 0.0

    # The projected gradient norm is reported as a finite, non-negative scalar.
    assert np.isfinite(grad_t_norm)
    assert grad_t_norm >= 0.0

    # The scale-aware (relative) latent-gradient stationarity measure is a
    # finite, non-negative scalar, and `converged` is decided from IT (not the
    # raw `grad_t_norm`) -- this is the #879 fix: the profiled Gaussian REML
    # objective leaves the raw gradient O(n) near interpolation, so the absolute
    # test mis-flags near-perfect fits as non-converged.
    assert np.isfinite(grad_t_norm_scaled)
    assert grad_t_norm_scaled >= 0.0

    # `converged` is an honest boolean flag, equal to the scale-aware test.
    converged = bool(r["converged"])
    assert isinstance(converged, bool)
    grad_tol = 1.0e-8  # the default `grad_tol` of gaussian_reml_optimize_latent
    assert converged == (grad_t_norm_scaled <= grad_tol)


def test_latent_optimize_near_perfect_fit_converges_879():
    """A near-interpolating Euclidean Duchon fit (R²≈1) must be flagged
    ``converged=True`` by the scale-aware criterion even though the raw
    ``grad_t_norm`` of the profiled-scale objective stays large -- the core
    complaint of #879."""
    rng = np.random.RandomState(7)
    n = 30
    t_true = np.sort(rng.uniform(-1.0, 1.0, n))
    # A smooth 3-output decoder image of a 1-D latent, with negligible noise so
    # the Duchon basis can near-interpolate (R² -> 1).
    Y = np.c_[
        np.sin(2.0 * t_true),
        t_true**2,
        np.cos(1.5 * t_true),
    ] + 1e-4 * rng.randn(n, 3)
    C = np.linspace(-1.0, 1.0, 16).reshape(-1, 1)

    r = gamfit.gaussian_reml_optimize_latent(
        y=Y.astype(float),
        n_obs=n,
        latent_dim=1,
        centers=C,
        penalty=np.eye(16),
        m=2,
        manifold="euclidean",
        basis_kind="duchon",
        max_iter=300,
        init="caller",
        t=t_true.astype(float),
        seed=0,
    )

    response_r2 = float(r["response_r2"])
    grad_t_norm = float(r["grad_t_norm"])
    grad_t_norm_scaled = float(r["grad_t_norm_scaled"])
    converged = bool(r["converged"])

    # The decoder near-interpolates the response.
    assert response_r2 > 0.99, f"expected near-perfect fit, got response_r2={response_r2}"

    # The scale-aware measure must be far smaller than the raw absolute gradient
    # norm: the profiled scale that inflates the raw gradient is divided out.
    assert grad_t_norm_scaled <= grad_t_norm + 1e-12

    # And the fit -- being near-stationary at an excellent optimum -- is now
    # reported as converged, which the old absolute `grad_t_norm <= grad_tol`
    # test could not deliver for the profiled-scale objective.
    assert converged, (
        "near-perfect near-stationary fit reported non-converged: "
        f"grad_t_norm={grad_t_norm}, grad_t_norm_scaled={grad_t_norm_scaled}, "
        f"response_r2={response_r2}"
    )
