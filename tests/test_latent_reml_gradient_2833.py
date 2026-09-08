"""The latent adjoint differentiates the forward with its supplied penalty."""

import numpy as np
import pytest

import gamfit


def _problem(n, dim, k, outputs, weighted):
    rng = np.random.default_rng(1)
    t = rng.uniform(0.1, 0.9, (n, dim))
    centers = rng.uniform(0.05, 0.95, (k, dim))
    y = 0.3 * rng.normal(size=(n, outputs)) + np.sin(3 * t[:, :1])
    weights = rng.uniform(0.5, 1.5, n) if weighted else np.ones(n)
    args = (y, n, dim, centers, np.eye(k))
    return t, args, dict(m=2, weights=weights)


@pytest.mark.parametrize(
    "shape", [(12, 1, 5, 1, False), (20, 1, 8, 2, True), (24, 2, 9, 1, True)]
)
def test_latent_reml_score_gradient_and_descent(shape):
    t, args, kwargs = _problem(*shape)
    fit = gamfit.gaussian_reml_fit_latent(t.ravel(), *args, **kwargs)
    assert fit["cache_penalty_rank"] == shape[2]
    assert fit["cache_nullity"] == 0
    assert abs(fit["reml_grad_rho"]) < 1e-7
    initial_gradient = np.asarray(
        gamfit.gaussian_reml_fit_latent_backward(
            t.ravel(), *args, grad_reml_score=1.0, **kwargs
        )["grad_t"]
    )
    kwargs["init_lambda"] = fit["lambda"]
    gradient = np.asarray(
        gamfit.gaussian_reml_fit_latent_backward(
            t.ravel(), *args, grad_reml_score=1.0, **kwargs
        )["grad_t"]
    )
    np.testing.assert_allclose(gradient, initial_gradient, rtol=1e-7, atol=1e-8)

    def score(point):
        return gamfit.gaussian_reml_fit_latent(point.ravel(), *args, **kwargs)[
            "reml_score"
        ]

    for step in (1e-5, 1e-6):
        central = np.empty_like(t)
        for index in np.ndindex(t.shape):
            plus, minus = t.copy(), t.copy()
            plus[index] += step
            minus[index] -= step
            central[index] = (score(plus) - score(minus)) / (2 * step)
        np.testing.assert_allclose(gradient, central, rtol=1e-4, atol=1e-5)

    norm = np.linalg.norm(gradient)
    assert norm > 0
    direction = gradient / norm
    change = score(t - 1e-6 * direction) - fit["reml_score"]
    assert change < -0.5e-6 * norm


def test_latent_design_coefficient_frame_is_independent_of_other_rows():
    # Independent responses expose the design through fitted = design @ beta.
    # Moving one point must not change the function represented by beta at any
    # other point when centers and the supplied coefficient penalty are fixed.
    t, args, kwargs = _problem(20, 1, 5, 5, True)

    def design(point):
        fit = gamfit.gaussian_reml_fit_latent(point.ravel(), *args, **kwargs)
        return np.linalg.solve(
            np.asarray(fit["coefficients"]).T, np.asarray(fit["fitted"]).T
        ).T

    original = design(t)
    perturbed = t.copy()
    perturbed[0, 0] += 1e-3
    moved = design(perturbed)
    np.testing.assert_allclose(moved[1:], original[1:], rtol=1e-8, atol=1e-9)
    assert np.linalg.norm(moved[0] - original[0]) > 1e-6


def test_cached_adjoint_cannot_drop_an_identity_penalty_mode():
    t = np.linspace(0.1, 0.9, 12)
    powers = np.arange(5)
    x = t[:, None] ** powers * 10.0 ** -powers
    y = np.sin(9 * t)[:, None]
    penalty = np.eye(5)
    fit = gamfit.gaussian_reml_fit(x, y, penalty)
    assert fit["cache_penalty_rank"] == 5
    corrupted = dict(fit, cache_penalty_rank=4, cache_nullity=1)
    with pytest.raises(gamfit.GamError, match="null modes must be exactly zero"):
        gamfit.gaussian_reml_fit_backward(
            x, y, penalty, grad_reml_score=1.0, forward_state=corrupted
        )
