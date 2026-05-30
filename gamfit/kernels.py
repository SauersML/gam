"""Numerical kernels (cost matrices, Sinkhorn barycenter) for gamfit.

This module exposes the entropic Sinkhorn Wasserstein-barycenter
primitive together with three cost-matrix helpers. Everything is
backed by the gamfit Rust extension and runs in the log domain with
``logsumexp``, so it does not overflow for small regularization.

Public surface
--------------

* :func:`sinkhorn_barycenter` -- log-domain entropic Sinkhorn
  barycenter of ``K`` discrete distributions on a shared support.
* :func:`circular_cost` -- squared distance on a discrete cycle.
* :func:`euclidean_cost` -- squared Euclidean distance from a point
  cloud.
* :func:`geodesic_sphere_cost` -- squared great-circle distance on
  the unit 2-sphere.

References
----------
Benamou, Carlier, Cuturi, Nenna, Peyre (2015), "Iterative Bregman
Projections for Regularized Transportation Problems", SIAM J. Sci.
Comput., 37(2), A1111-A1138.

Cuturi, Peyre (2019), "Computational Optimal Transport", Foundations
and Trends in Machine Learning, 11(5-6), Chapter 9 (for the implicit
function theorem / adjoint-iteration approach used by the VJP).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._binding import rust_module

__all__ = [
    "sinkhorn_barycenter",
    "sinkhorn_barycenter_vjp",
    "circular_cost",
    "euclidean_cost",
    "geodesic_sphere_cost",
]


def _as_f64_2d(name: str, x: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {arr.shape}")
    return arr


def _as_f64_1d(name: str, x: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    return arr


def circular_cost(m: int) -> np.ndarray:
    """Return the ``(m, m)`` squared circular distance matrix.

    ``c[i, j] = min(|i - j|, m - |i - j|) ** 2``. Symmetric, zero
    diagonal, non-negative.
    """
    if not isinstance(m, (int, np.integer)) or int(m) <= 0:
        raise ValueError(f"m must be a positive integer, got {m}")
    return rust_module().sinkhorn_circular_cost(int(m))


def euclidean_cost(points: np.ndarray) -> np.ndarray:
    """Squared Euclidean cost from an ``(M, d)`` point array.

    Returns an ``(M, M)`` symmetric non-negative matrix with zero
    diagonal.
    """
    arr = _as_f64_2d("points", points)
    return rust_module().sinkhorn_euclidean_cost(arr)


def geodesic_sphere_cost(directions: np.ndarray) -> np.ndarray:
    """Squared great-circle cost on the 2-sphere from ``(M, 3)`` unit
    vectors.

    Each row must lie within ``1e-6`` of the unit sphere; accepted rows
    are renormalized to exact unit length before any cosine is formed,
    so the result is the true squared great-circle distance
    ``arccos(<x_i/|x_i|, x_j/|x_j|>) ** 2`` with a symmetric, exactly
    zero diagonal.
    """
    arr = _as_f64_2d("directions", directions)
    if arr.shape[1] != 3:
        raise ValueError(
            f"directions must have shape (M, 3), got {arr.shape}"
        )
    return rust_module().sinkhorn_geodesic_sphere_cost(arr)


def sinkhorn_barycenter(
    atoms: np.ndarray,
    weights: Optional[np.ndarray] = None,
    cost: Optional[np.ndarray] = None,
    eps: float = 0.01,
    n_iter: int = 20,
) -> np.ndarray:
    """Log-domain entropic Sinkhorn Wasserstein barycenter.

    Parameters
    ----------
    atoms : (K, M) array
        ``K`` non-negative input distributions on a shared support of
        size ``M``. Rows are normalized to sum to one internally;
        rows of zero total mass are rejected.
    weights : (K,) array, optional
        Non-negative mixing weights, summing to a positive total.
        Normalized to a probability vector internally. Defaults to
        the uniform ``1 / K``.
    cost : (M, M) array, optional
        Ground cost matrix. Must be finite, non-negative; symmetry
        and zero-diagonal are recommended for the canonical
        Wasserstein interpretation but not enforced here. Defaults
        to :func:`circular_cost`\ ``(M)``.
    eps : float, default 0.01
        Entropic regularization strength. Must be ``>= 1e-12``.
    n_iter : int, default 20
        Number of outer Sinkhorn iterations.

    Returns
    -------
    (M,) numpy.ndarray
        The entropic Wasserstein barycenter, a probability vector
        (sums to 1, non-negative).

    Notes
    -----
    All updates run in the log domain with stable ``logsumexp``; the
    kernel does not produce NaN for ``eps >= 1e-12`` even with input
    rows that have zero mass on some support points.

    Differentiability is provided by the companion VJP
    :func:`sinkhorn_barycenter_vjp` (used by the torch / JAX adapters
    in ``gamfit.kernels_torch`` / ``gamfit.kernels_jax``). The VJP is
    computed at the converged fixed point via adjoint iteration
    (Cuturi-Peyre, COT, Section 9.1.4) -- never by unrolling autograd
    through the Sinkhorn loop, so memory is ``O(K * M^2)`` and
    independent of ``n_iter``.

    References
    ----------
    Benamou, Carlier, Cuturi, Nenna, Peyre, "Iterative Bregman
    Projections for Regularized Transportation Problems",
    SIAM J. Sci. Comput. (2015).
    """
    atoms_arr = _as_f64_2d("atoms", atoms)
    k, m = atoms_arr.shape
    if weights is None:
        weights_arr = np.full(k, 1.0 / k, dtype=np.float64)
    else:
        weights_arr = _as_f64_1d("weights", weights)
    if cost is None:
        cost_arr = circular_cost(m)
    else:
        cost_arr = _as_f64_2d("cost", cost)
    return rust_module().sinkhorn_barycenter_forward(
        atoms_arr, weights_arr, cost_arr, float(eps), int(n_iter)
    )


def sinkhorn_barycenter_vjp(
    atoms: np.ndarray,
    weights: Optional[np.ndarray],
    cost: np.ndarray,
    eps: float,
    n_iter: int,
    cotangent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vector-Jacobian product for :func:`sinkhorn_barycenter`.

    Returns ``(d_atoms, d_weights)`` of shapes ``(K, M)`` and ``(K,)``
    respectively, computed via adjoint iteration at the converged
    fixed point (Cuturi-Peyre, COT, Section 9.1.4).
    """
    atoms_arr = _as_f64_2d("atoms", atoms)
    k, m = atoms_arr.shape
    if weights is None:
        weights_arr = np.full(k, 1.0 / k, dtype=np.float64)
    else:
        weights_arr = _as_f64_1d("weights", weights)
    cost_arr = _as_f64_2d("cost", cost)
    cot_arr = _as_f64_1d("cotangent", cotangent)
    return rust_module().sinkhorn_barycenter_vjp(
        atoms_arr,
        weights_arr,
        cost_arr,
        float(eps),
        int(n_iter),
        cot_arr,
    )
