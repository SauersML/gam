"""Regression test for issue #875: N-D Duchon latent basis at latent_dim >= 4.

The latent Duchon design used by ``gaussian_reml_fit_latent`` worked for
``latent_dim`` 1-3 but raised for ``latent_dim >= 4``::

    GamError: failed to evaluate N-D Duchon basis for LatentCoord:
              Invalid input: Duchon pointwise kernel ...

Root cause: the latent Duchon design hard-coded the spectral power ``s = 0``
and used the null-space order straight from ``m``. The pure scale-free
polyharmonic kernel ``r^{2(p+s)-d}`` (its ``r^{2m-d} log r`` log case in even
``d``) only *exists* when ``2(p+s) > d``. With ``m = 2`` (so ``p = 2``) and
``s = 0`` that is ``4 > d`` — true for ``d <= 3`` but false at ``d = 4`` (the
exact-log / 2m=d case) and ``d = 5``. The fix routes the latent forward design
*and* its derivative jet through ``resolve_duchon_orders``, which lifts the
spectral power ``s`` (and, if pure-mode CPD requires it, the null-space order)
until ``2(p+s) > d`` holds for *any* ``d`` — exactly as every other Duchon
entry point already does.

The contract pinned here:

* construction succeeds for ``latent_dim`` 2, 3, 4, 5 (the repro grid), and
* the decoder recovers a smooth ground-truth function of the latent grid to
  high R² — i.e. the resolved high-d kernel is a usable basis, not just a
  non-erroring one.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import gamfit


def _grid_centers(k: int, n_target: int = 16) -> np.ndarray:
    """A k-D tensor grid of roughly ``n_target`` centers in the unit cube."""
    g = max(2, round(n_target ** (1.0 / k)))
    axes = [np.linspace(0.0, 1.0, g)] * k
    return np.array(list(itertools.product(*axes)), dtype=float)


def _r2(y: np.ndarray, fitted: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    fitted = np.asarray(fitted, dtype=float)
    return 1.0 - ((y - fitted) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_latent_duchon_constructs_for_high_dim(k: int) -> None:
    """The Duchon latent basis must build for every latent_dim, not just <= 3.

    Before the fix, k in {4, 5} raised
    ``Duchon pointwise kernel ... 2*(p+s) > dimension``.
    """
    rng = np.random.RandomState(0)
    n = 40
    centers = _grid_centers(k)
    penalty = np.eye(len(centers))

    # Latent coordinate: a random point cloud in the unit cube; the decoder is a
    # smooth (linear) function of it, so a well-posed kernel must fit it well.
    t_true = rng.rand(n, k)
    coeffs = rng.randn(k, 6)
    y = t_true @ coeffs + 0.01 * rng.randn(n, 6)

    # This is a basis-construction regression, so hold the latent coordinate
    # fixed instead of coupling it to the separate latent-optimizer convergence
    # contract.
    res = gamfit.gaussian_reml_fit_latent(
        t_true.reshape(-1),
        y,
        n,
        k,
        centers,
        penalty,
        m=2,
        basis_kind="duchon",
    )

    # Construction succeeded (no GamError) and produced a usable decoder.
    assert "fitted" in res
    fitted = np.asarray(res["fitted"], dtype=float)
    assert fitted.shape == y.shape
    assert np.all(np.isfinite(fitted))
    # A linear ground truth on the resolved high-d kernel is recovered well.
    assert _r2(y, fitted) >= 0.9, f"latent_dim={k}: poor recovery R²={_r2(y, fitted):.3f}"
