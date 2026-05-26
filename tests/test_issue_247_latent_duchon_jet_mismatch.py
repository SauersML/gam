"""RED tests for issue #247: gaussian_reml_fit_latent Duchon design/jet mismatch.

The default `basis_kind="duchon"` path constructs a radial-only forward design
but a radial+polynomial-nullspace derivative jet, then rejects the call with
'latent design/jet column mismatch'. Every dimension is affected.

These tests pin the contract that the documented default succeeds end-to-end.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamfit


def _make_inputs(n_obs: int, latent_dim: int, n_centers: int):
    rng = np.random.default_rng(latent_dim * 17 + n_obs)
    t = rng.standard_normal((n_obs, latent_dim))
    centers = rng.standard_normal((n_centers, latent_dim))
    y = rng.standard_normal(n_obs)
    penalty = np.eye(n_centers)
    return t, y, centers, penalty


@pytest.mark.parametrize("latent_dim", [1, 2, 3])
def test_gaussian_reml_fit_latent_duchon_default_runs(latent_dim: int) -> None:
    """`basis_kind="duchon"` with default `m=2` should fit, not raise."""
    n_obs = 12
    n_centers = 6
    t, y, centers, penalty = _make_inputs(n_obs, latent_dim, n_centers)

    result = gamfit.gaussian_reml_fit_latent(
        t.ravel(),
        y,
        n_obs,
        latent_dim,
        centers,
        penalty,
        basis_kind="duchon",
        m=2,
    )

    assert isinstance(result, dict)
    assert "beta" in result or "coefficients" in result or "fitted" in result


def test_gaussian_reml_fit_latent_duchon_d1_default_repro() -> None:
    """Exact repro from issue #247."""
    n_obs, latent_dim = 10, 1
    t = np.linspace(0, 1, n_obs).reshape(n_obs, latent_dim)
    y = t.copy().ravel()
    centers = np.linspace(0, 1, 4).reshape(4, latent_dim)
    penalty = np.eye(4)

    result = gamfit.gaussian_reml_fit_latent(
        t.ravel(),
        y,
        n_obs,
        latent_dim,
        centers,
        penalty,
        basis_kind="duchon",
        m=2,
    )
    assert isinstance(result, dict)
