"""JAX adapter for the Sinkhorn-barycenter kernel.

Exposes :func:`sinkhorn_barycenter` as a :func:`jax.custom_vjp` whose
backward pass calls the Rust VJP for the same finite-iteration
Sinkhorn map used in the forward pass. Importing this module raises a
clear :class:`ImportError` if JAX is not installed.

All numerics live in Rust; this file is purely a JAX-autograd shim
and host/device marshalling.
"""

from __future__ import annotations

import numpy as np

from . import kernels as _kernels

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "gamfit.kernels_jax requires JAX. Install with 'pip install jax'."
    ) from exc


def _forward_host(atoms_np, weights_np, cost_np, eps, n_iter):
    return _kernels.sinkhorn_barycenter(
        atoms_np, weights_np, cost_np, eps=eps, n_iter=n_iter
    )


def _vjp_host(atoms_np, weights_np, cost_np, eps, n_iter, cot_np):
    return _kernels.sinkhorn_barycenter_vjp(
        atoms_np, weights_np, cost_np, eps, n_iter, cot_np
    )


@jax.custom_vjp
def sinkhorn_barycenter(atoms, weights, cost, eps=0.01, n_iter=20):
    """Differentiable Sinkhorn Wasserstein barycenter (JAX).

    Backward via Rust finite-iteration VJP at the same ``n_iter``.
    """
    atoms_np = np.asarray(atoms, dtype=np.float64)
    weights_np = np.asarray(weights, dtype=np.float64)
    cost_np = np.asarray(cost, dtype=np.float64)
    bary_np = _forward_host(atoms_np, weights_np, cost_np, float(eps), int(n_iter))
    return jnp.asarray(bary_np)


def _fwd(atoms, weights, cost, eps, n_iter):
    atoms_np = np.asarray(atoms, dtype=np.float64)
    weights_np = np.asarray(weights, dtype=np.float64)
    cost_np = np.asarray(cost, dtype=np.float64)
    bary_np = _forward_host(atoms_np, weights_np, cost_np, float(eps), int(n_iter))
    return jnp.asarray(bary_np), (atoms_np, weights_np, cost_np, float(eps), int(n_iter))


def _bwd(res, cotangent):
    atoms_np, weights_np, cost_np, eps, n_iter = res
    cot_np = np.asarray(cotangent, dtype=np.float64)
    d_atoms_np, d_weights_np = _vjp_host(
        atoms_np, weights_np, cost_np, eps, n_iter, cot_np
    )
    # custom_vjp returns one cotangent per primal arg. We treat cost
    # as non-differentiable here (zero gradient); eps and n_iter are
    # scalars also returned as zero.
    return (
        jnp.asarray(d_atoms_np),
        jnp.asarray(d_weights_np),
        jnp.zeros_like(jnp.asarray(cost_np)),
        jnp.float64(0.0),
        jnp.float64(0.0),
    )


sinkhorn_barycenter.defvjp(_fwd, _bwd)


__all__ = ["sinkhorn_barycenter"]
