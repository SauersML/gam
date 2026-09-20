"""JAX adapter for the Sinkhorn-barycenter kernel.

Exposes :func:`sinkhorn_barycenter` as a :func:`jax.custom_vjp` whose
backward pass calls the Rust implicit-function-theorem VJP of the
certified fixed point the forward pass returns. Importing this module raises a
clear :class:`ImportError` if JAX is not installed.

All numerics live in Rust; this file is purely a JAX-autograd shim
and host/device marshalling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import kernels as _kernels

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "gamfit.kernels_jax requires JAX. Install with 'pip install jax'."
    ) from exc

if TYPE_CHECKING:
    from jax.typing import ArrayLike

# Host-side residuals carried from the forward to the backward pass:
# ``(atoms, weights, cost, eps)``.
_Residuals = tuple[np.ndarray, np.ndarray, np.ndarray, float]


def _forward_host(
    atoms_np: np.ndarray,
    weights_np: np.ndarray,
    cost_np: np.ndarray,
    eps: float,
) -> np.ndarray:
    return _kernels.sinkhorn_barycenter(atoms_np, weights_np, cost_np, eps=eps)


def _vjp_host(
    atoms_np: np.ndarray,
    weights_np: np.ndarray,
    cost_np: np.ndarray,
    eps: float,
    cot_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return _kernels.sinkhorn_barycenter_vjp(atoms_np, weights_np, cost_np, eps, cot_np)


@jax.custom_vjp
def sinkhorn_barycenter(
    atoms: ArrayLike,
    weights: ArrayLike,
    cost: ArrayLike,
    eps: float = 0.01,
) -> jax.Array:
    """Differentiable Sinkhorn Wasserstein barycenter (JAX).

    Backward via the Rust implicit-function-theorem VJP of the certified
    fixed point.
    """
    atoms_np = np.asarray(atoms, dtype=np.float64)
    weights_np = np.asarray(weights, dtype=np.float64)
    cost_np = np.asarray(cost, dtype=np.float64)
    bary_np = _forward_host(atoms_np, weights_np, cost_np, float(eps))
    return jnp.asarray(bary_np)


def _fwd(
    atoms: ArrayLike,
    weights: ArrayLike,
    cost: ArrayLike,
    eps: float,
) -> tuple[jax.Array, _Residuals]:
    atoms_np = np.asarray(atoms, dtype=np.float64)
    weights_np = np.asarray(weights, dtype=np.float64)
    cost_np = np.asarray(cost, dtype=np.float64)
    bary_np = _forward_host(atoms_np, weights_np, cost_np, float(eps))
    return jnp.asarray(bary_np), (atoms_np, weights_np, cost_np, float(eps))


def _bwd(
    res: _Residuals, cotangent: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    atoms_np, weights_np, cost_np, eps = res
    cot_np = np.asarray(cotangent, dtype=np.float64)
    d_atoms_np, d_weights_np = _vjp_host(atoms_np, weights_np, cost_np, eps, cot_np)
    # custom_vjp returns one cotangent per primal arg. We treat cost
    # as non-differentiable here (zero gradient); the scalar eps is
    # also returned as zero.
    return (
        jnp.asarray(d_atoms_np),
        jnp.asarray(d_weights_np),
        jnp.zeros_like(jnp.asarray(cost_np)),
        jnp.float64(0.0),
    )


sinkhorn_barycenter.defvjp(_fwd, _bwd)


__all__ = ["sinkhorn_barycenter"]
