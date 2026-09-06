"""Bug hunt: ``gamfit.gaussian_reml_fit_latent_backward``'s ``grad_t`` is not the
gradient of the ``reml_score`` its own forward returns -- it is not even a
descent direction.

``gaussian_reml_fit_latent`` is the differentiable inner solve behind
``gamfit.LatentCoord`` and the manifold-SAE decoder: it builds ``Phi(t)`` at the
per-row latent coordinate ``t`` and solves ``beta`` / ``lambda`` by REML.
``gaussian_reml_fit_latent_backward`` documents ``grad_t`` as "the latent
gradient ... with shape ``(n_obs, latent_dim)``", and
``gaussian_reml_optimize_latent`` documents itself as minimising "the same
Gaussian-REML score over ``t`` with a Riemannian trust region driven by the
analytic ``d(reml_score)/d t``".

Measured on the 12-row fixture below (``aux_u=None``, ``aux_strength=None``,
``dim_selection_log_precision=None``, so none of the documented additive
identifiability contributions is active, and ``reml_grad_rho = 1.3e-15`` at the
returned optimum, so the envelope theorem makes the ``lambda`` re-optimisation
first-order irrelevant):

    analytic  [-0.1249  0.3382 -0.1336  0.2432 -0.0328 -0.2395 ...]
    central   [ 0.4435 -0.1318 -0.5043  0.1736 -0.9975  1.1257 ...]
    cosine(analytic, central) = -0.43

Central differences are stable to five decimals across ``h`` from ``1e-3`` to
``1e-6``, so this is not finite-difference noise.  The direction is wrong, not
just the scale: stepping along ``-grad_t / ||grad_t||`` *increases* the score
the same call reports, linearly in the step,

    eta = 1e-2 -> +8.019e-03      eta = 1e-5 -> +9.628e-06
    eta = 1e-3 -> +9.469e-04      eta = 1e-6 -> +9.629e-07

i.e. the directional derivative along the reported descent direction is
``+0.963`` where it must be ``-||grad||``.  Along the central-difference
gradient the same steps give ``-2.25e-03 / -2.24e-05``, as they should.

Root cause (a two-builder mismatch, the Rust-side sibling of gam#2097):

The design the latent forward actually fits is *not* ``gamfit.duchon_basis(t,
centers, m)``.  Recovering it from the forward's own outputs (fit ``dy = K``
independent responses, then ``Phi_int = fitted @ inv(coefficients)``) gives

    Phi_int = duchon_basis(t, centers, m) @ V        (residual 7.7e-14)

with ``V`` a dense ``K x K`` matrix -- the batch-global data-metric radial
reparameterization of gam#1355 -- and ``V`` depends on the *whole batch* of
latent coordinates.  Moving a single coordinate ``t[0]`` by ``1e-4`` leaves
every unmoved row of ``duchon_basis`` bit-identical (``max diff = 0.0``) but
moves every unmoved row of ``Phi_int`` by ``2.15e-05``, and moves ``V`` itself
by ``0.0229``.

So ``d Phi/d t`` has a cross-row term through ``V(t)`` on top of the row-local
radial term.  The analytic ``grad_t`` behaves like it carries only the row-local
one -- the same defect gam#2097 fixed in ``gamfit.torch._basis.duchon_basis``,
where the forward used ``build_duchon_basis`` (which applies ``V``) and the
backward contracted against jets from a builder that does not.  That fix landed
in the torch wrapper; this low-level Rust latent backward still disagrees with
its own forward.

Downstream: ``gaussian_reml_optimize_latent`` walks this gradient. Started
exactly at a planted latent on a smooth 3-D curve (n=40, K=8), its Riemannian
trust region does not converge in 100 iterations -- "stationarity residual
2.895320e4 exceeds tolerance 1.000000e-8" -- while the true central-difference
gradient norm at that point is 1.16e3.

Both assertions are fix-agnostic: they compare the analytic gradient to central
differences of the *same* forward, and require the reported gradient to be a
descent direction for the *same* score. Nothing about ``V``, jets, or a builder
is named.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

import gamfit

_M = 2


def _fixture(
    seed: int, n: int, latent_dim: int, n_centers: int, n_outputs: int
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.1, 0.9, (n, latent_dim))
    centers = rng.uniform(0.05, 0.95, (n_centers, latent_dim))
    signal = np.sin(3.0 * t[:, :1]) @ np.ones((1, n_outputs))
    y = 0.3 * rng.normal(size=(n, n_outputs)) + signal
    penalty = np.eye(n_centers)
    weights = np.ones(n)
    return t, y, centers, penalty, weights


def _score(
    t: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    centers: npt.NDArray[np.float64],
    penalty: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> float:
    n, latent_dim = t.shape
    fit = gamfit.gaussian_reml_fit_latent(
        np.asarray(t).ravel(),
        y,
        n,
        latent_dim,
        centers,
        penalty,
        m=_M,
        weights=weights,
    )
    return float(np.asarray(fit["reml_score"], dtype=np.float64))


def _analytic_grad_t(
    t: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    centers: npt.NDArray[np.float64],
    penalty: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    n, latent_dim = t.shape
    back = gamfit.gaussian_reml_fit_latent_backward(
        np.asarray(t).ravel(),
        y,
        n,
        latent_dim,
        centers,
        penalty,
        m=_M,
        weights=weights,
        grad_reml_score=1.0,
    )
    return np.asarray(back["grad_t"], dtype=np.float64).reshape(n, latent_dim)


_CASES = [
    # (seed, n, latent_dim, n_centers, n_outputs)
    (1, 12, 1, 5, 1),
    (2, 20, 1, 8, 2),
    (3, 24, 2, 9, 1),
]


@pytest.mark.parametrize(("seed", "n", "latent_dim", "n_centers", "n_outputs"), _CASES)
def test_grad_t_matches_central_differences_of_its_own_reml_score(
    seed: int, n: int, latent_dim: int, n_centers: int, n_outputs: int
) -> None:
    t, y, centers, penalty, weights = _fixture(
        seed, n, latent_dim, n_centers, n_outputs
    )

    fit = gamfit.gaussian_reml_fit_latent(
        t.ravel(), y, n, latent_dim, centers, penalty, m=_M, weights=weights
    )
    # The lambda re-optimisation is first-order irrelevant: the profile is
    # stationary in rho, so the total derivative in t equals the partial one.
    assert abs(float(np.asarray(fit["reml_grad_rho"], dtype=np.float64))) < 1e-8

    analytic = _analytic_grad_t(t, y, centers, penalty, weights)

    h = 1e-6
    central = np.zeros_like(t)
    for i in range(n):
        for j in range(latent_dim):
            plus = t.copy()
            plus[i, j] += h
            minus = t.copy()
            minus[i, j] -= h
            central[i, j] = (
                _score(plus, y, centers, penalty, weights)
                - _score(minus, y, centers, penalty, weights)
            ) / (2.0 * h)

    scale = max(1e-9, float(np.abs(central).max()))
    error = float(np.abs(analytic - central).max()) / scale
    cosine = float(
        (analytic * central).sum()
        / max(1e-30, np.linalg.norm(analytic) * np.linalg.norm(central))
    )
    assert error < 1e-4, (
        f"grad_t disagrees with central differences of its own forward by "
        f"{error:.3f} (relative); cosine = {cosine:+.3f}\n"
        f"analytic = {analytic.ravel()}\ncentral  = {central.ravel()}"
    )


@pytest.mark.parametrize(("seed", "n", "latent_dim", "n_centers", "n_outputs"), _CASES)
def test_negative_grad_t_is_a_descent_direction_for_the_reported_score(
    seed: int, n: int, latent_dim: int, n_centers: int, n_outputs: int
) -> None:
    """``-grad_t`` must not increase the score the same call reports.

    This is the weakest possible consequence of "``grad_t`` is the gradient of
    ``reml_score``", and the one the documented Riemannian trust region in
    ``gaussian_reml_optimize_latent`` actually relies on.
    """
    t, y, centers, penalty, weights = _fixture(
        seed, n, latent_dim, n_centers, n_outputs
    )
    base = _score(t, y, centers, penalty, weights)
    analytic = _analytic_grad_t(t, y, centers, penalty, weights)
    norm = float(np.linalg.norm(analytic))
    assert norm > 0.0
    direction = analytic / norm

    for eta in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        moved = _score(t - eta * direction, y, centers, penalty, weights)
        # A true gradient gives moved - base ~= -eta*norm < 0. Allow only
        # second-order slack, far below the observed +eta*0.96 ascent.
        assert moved - base <= 1e-3 * eta * norm, (
            f"stepping eta={eta:g} along -grad_t/|grad_t| raised the reported "
            f"REML score by {moved - base:+.6e}"
        )
