"""An isometry penalty is a function of the decoder Jacobian its caller supplies (#2627).

The isometry kernel reads the target only through the decoder Jacobian J. When J
was absent it returned a zero value and a zero gradient with a log warning, so a
fit carrying the penalty was the same fit without it, and value_grad reported a
penalty that was never evaluated. Each route now refuses by name, and an
independent J gives the value with an exactly zero target gradient and no warning.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import gamfit
from gamfit._binding import rust_module


def _isometry_json(n_obs: int, latent_dim: int, p_out: int) -> tuple[str, str]:
    latents = json.dumps({"t": {"name": "t", "n": n_obs, "d": latent_dim}})
    penalties = json.dumps(
        [{"kind": "isometry", "target": "t", "weight": 1.0, "p_out": p_out}]
    )
    return latents, penalties


def test_value_grad_refuses_an_isometry_penalty_without_a_decoder_jacobian() -> None:
    t = np.linspace(-1.0, 1.0, 12).reshape(6, 2)

    with pytest.raises(ValueError, match="decoder Jacobian J"):
        gamfit.penalties.IsometryPenalty(weight=1.0).value_grad(t)


def test_an_independent_decoder_jacobian_gives_the_value_and_a_zero_target_gradient(
    capfd: pytest.CaptureFixture[str],
) -> None:
    n_obs, latent_dim, p_out = 6, 2, 3
    rng = np.random.default_rng(7)
    target = rng.normal(size=n_obs * latent_dim)
    jacobian = rng.normal(size=(n_obs, p_out * latent_dim))
    latents, penalties = _isometry_json(n_obs, latent_dim, p_out)

    value, grad_target, grad_rho, grad_jacobian = rust_module().analytic_penalty_value_grad(
        latents, penalties, target, np.zeros(1), isometry_jacobian=jacobian
    )

    assert value > 0.0
    assert grad_rho[0] == pytest.approx(value, rel=1e-12)
    np.testing.assert_array_equal(np.asarray(grad_target), np.zeros_like(target))
    assert grad_jacobian is not None
    assert np.abs(np.asarray(grad_jacobian)).max() > 0.0
    assert "IsometryPenalty::" not in capfd.readouterr().err


def test_hvp_refuses_an_exact_isometry_hessian_without_the_third_decoder_jet() -> None:
    n_obs, latent_dim, p_out = 6, 2, 3
    rng = np.random.default_rng(11)
    target = rng.normal(size=n_obs * latent_dim)
    direction = rng.normal(size=n_obs * latent_dim)
    jacobian = rng.normal(size=(n_obs, p_out * latent_dim))
    motion = rng.normal(size=(n_obs, p_out * latent_dim * latent_dim))
    latents, penalties = _isometry_json(n_obs, latent_dim, p_out)

    with pytest.raises(ValueError, match="K = ∂H/∂t"):
        rust_module().analytic_penalty_hvp(
            latents,
            penalties,
            target,
            direction,
            np.zeros(1),
            isometry_jacobian=jacobian,
            isometry_jacobian_second=motion,
        )


def test_the_latent_coordinate_fit_refuses_an_isometry_penalty_by_name() -> None:
    rng = np.random.default_rng(3)
    n = 40
    t0 = np.sort(rng.uniform(-1.0, 1.0, size=(n, 1)), axis=0)
    y = np.sin(2.0 * t0[:, 0]) + 0.1 * rng.normal(size=n)

    with pytest.raises(
        gamfit.errors.GamfitError, match="supplies no decoder jets for an isometry penalty"
    ):
        gamfit.fit(
            pd.DataFrame({"y": y}),
            "y ~ s(t, type='duchon', centers=12)",
            family="gaussian",
            latents={
                "t": gamfit.smooth.LatentCoord(
                    n=n,
                    d=1,
                    init=t0,
                    aux_prior={"u": t0, "family": "ridge", "strength": "auto"},
                )
            },
            penalties=[gamfit.penalties.IsometryPenalty(weight=10.0)],
        )
