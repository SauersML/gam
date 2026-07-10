"""Regression tests for honest latent-REML convergence (issues #879/#954)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gamfit._rust")

import gamfit


def _problem():
    rng = np.random.RandomState(7)
    n = 30
    t = np.sort(rng.uniform(-1.0, 1.0, n))
    y = np.c_[np.sin(2.0 * t), t**2, np.cos(1.5 * t)]
    centers = np.linspace(-1.0, 1.0, 16).reshape(-1, 1)
    return n, t.astype(float), y.astype(float), centers, np.eye(16)


def test_nonconvergence_is_typed_evidence_with_resumable_checkpoint_879():
    n, t, y, centers, penalty = _problem()

    # Zero trust-region iterations deterministically leaves this non-stationary
    # caller start untouched. It must produce an error, never a fit dictionary.
    with pytest.raises(gamfit.RemlConvergenceError) as caught:
        gamfit.gaussian_reml_optimize_latent(
            y=y,
            n_obs=n,
            latent_dim=1,
            centers=centers,
            penalty=penalty,
            t=t,
            init="caller",
            max_iter=0,
        )

    error = caught.value
    for name in (
        "grad_t_norm",
        "grad_t_norm_init",
        "grad_t_norm_scaled",
        "grad_tol",
        "latent_t_std",
        "objective_value",
        "max_iter",
        "n_restarts",
        "restart_index",
        "checkpoint_t",
        "checkpoint_shape",
        "checkpoint_stationarity_reference",
    ):
        assert hasattr(error, name), f"missing convergence evidence {name!r}"

    grad = float(error.grad_t_norm)
    reference = float(error.grad_t_norm_init)
    scaled = float(error.grad_t_norm_scaled)
    assert np.isfinite(grad) and grad >= 0.0
    assert np.isfinite(reference) and reference >= 0.0
    assert scaled == pytest.approx(grad / max(reference, 1.0))
    assert scaled > float(error.grad_tol)
    assert np.asarray(error.checkpoint_t).shape == (n,)
    assert tuple(error.checkpoint_shape) == (n, 1)
    assert float(error.checkpoint_stationarity_reference) == reference

    # The checkpoint is in the exact 1-D shape accepted by the public API. Use
    # a tolerance derived from the recorded evidence to exercise the resume
    # plumbing without relying on a workload-specific iteration count.
    resume_tol = np.nextafter(scaled, np.inf)
    resumed = gamfit.gaussian_reml_optimize_latent(
        y=y,
        n_obs=n,
        latent_dim=1,
        centers=centers,
        penalty=penalty,
        t=np.asarray(error.checkpoint_t),
        init="caller",
        max_iter=0,
        grad_tol=resume_tol,
        stationarity_reference=float(error.checkpoint_stationarity_reference),
    )
    assert "converged" not in resumed
    assert float(resumed["grad_t_norm_scaled"]) <= resume_tol
    assert np.asarray(resumed["t_flat"]).shape == (n,)


@pytest.mark.parametrize("grad_tol", [np.inf, -np.inf, np.nan, 0.0, -1.0])
def test_invalid_stationarity_tolerance_cannot_certify_a_fit_954(grad_tol):
    n, t, y, centers, penalty = _problem()
    with pytest.raises(gamfit.GamError, match="grad_tol must be finite and positive"):
        gamfit.gaussian_reml_optimize_latent(
            y=y,
            n_obs=n,
            latent_dim=1,
            centers=centers,
            penalty=penalty,
            t=t,
            init="caller",
            max_iter=0,
            grad_tol=grad_tol,
        )
