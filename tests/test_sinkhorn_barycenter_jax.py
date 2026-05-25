"""JAX differentiability test for the Sinkhorn barycenter."""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from gamfit import kernels
from gamfit import kernels_jax


def test_jax_gradcheck_vs_finite_differences() -> None:
    rng = np.random.default_rng(11)
    m = 16
    k = 2
    n_iter = 200
    eps = 0.1
    atoms_np = rng.uniform(0.01, 1.0, size=(k, m))
    atoms_np = atoms_np / atoms_np.sum(axis=1, keepdims=True)
    weights_np = np.array([0.5, 0.5])
    cost_np = kernels.circular_cost(m)
    target_np = rng.normal(size=(m,))

    def loss_fn(atoms_in):
        bary = kernels_jax.sinkhorn_barycenter(
            atoms_in,
            jnp.asarray(weights_np),
            jnp.asarray(cost_np),
            eps,
            n_iter,
        )
        return (bary * jnp.asarray(target_np)).sum()

    grad = jax.grad(loss_fn)(jnp.asarray(atoms_np))
    analytic = np.asarray(grad)

    # FD at a few entries.
    h = 1e-5
    for (ki, j) in [(0, 3), (1, 7), (0, 10)]:
        plus = atoms_np.copy()
        minus = atoms_np.copy()
        plus[ki, j] += h
        minus[ki, j] -= h
        bp = kernels.sinkhorn_barycenter(plus, weights_np, cost_np, eps=eps, n_iter=n_iter)
        bm = kernels.sinkhorn_barycenter(minus, weights_np, cost_np, eps=eps, n_iter=n_iter)
        fd = float(((bp - bm) * target_np).sum() / (2 * h))
        denom = max(abs(analytic[ki, j]), abs(fd), 1e-6)
        rel = abs(analytic[ki, j] - fd) / denom
        assert rel < 0.05, f"mismatch at ({ki},{j}): analytic={analytic[ki,j]} fd={fd} rel={rel}"
