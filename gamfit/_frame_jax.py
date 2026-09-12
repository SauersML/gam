"""JAX frame adapter.

Wraps the same Rust kernels every other frame uses, but presents the
output as a :class:`jax.Array` that is safe under ``jit``, ``vmap``, and
``grad``. The forward pass is a :func:`jax.pure_callback`; the gradient
rule is a :class:`jax.custom_vjp` whose backward consults a numpy VJP from
the Rust core. No math is reimplemented in JAX.

JAX is an **optional** dependency. Importing :mod:`gamfit._frame_jax`
itself never raises; it imports jax lazily on first use of any helper.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._frame_numpy import to_numpy_f64 as _to_numpy_f64
from ._frame_shared import stack_coords_generic


def _jax() -> tuple[Any, Any]:
    from ._frame import import_jax

    return import_jax()


def to_numpy_f64(value: Any) -> np.ndarray:
    """Materialize a jax array as a contiguous float64 numpy array.

    The autograd graph is *not* preserved — callers that need a
    differentiable path must wrap the Rust call in a
    :class:`jax.custom_vjp`.
    """
    return _to_numpy_f64(value)


def from_numpy_like(array: Any, ref: Any | None) -> Any:
    """Wrap a numpy ndarray as a jax array matching ``ref``'s dtype.

    Device placement is left to JAX's default-device policy (we never
    force a device explicitly so users that have configured a non-CPU
    default see their tensor land where they expect).
    """
    _, jnp = _jax()
    np_arr = np.asarray(array, dtype=np.float64, order="C")
    out = jnp.asarray(np_arr)
    if ref is not None:
        ref_dtype = getattr(ref, "dtype", None)
        if ref_dtype is not None and out.dtype != ref_dtype:
            out = out.astype(ref_dtype)
    return out


def stack_coords(coords: list[Any] | tuple[Any, ...]) -> Any:
    """Stack 1D jax coordinates into a (B, d) jax array."""
    _, jnp = _jax()

    def _coerce(idx: int, c: Any, _ref_len: int | None) -> tuple[Any, int]:
        a = jnp.asarray(c)
        if a.ndim != 1:
            raise ValueError(
                f"coord {idx}: expected 1D array, got shape {tuple(a.shape)}"
            )
        if not jnp.issubdtype(a.dtype, jnp.floating):
            a = a.astype(jnp.float64)
        return a, int(a.shape[0])

    return stack_coords_generic(
        coords,
        coerce=_coerce,
        stack=lambda arrays: jnp.stack(arrays, axis=1),
    )


__all__ = [
    "to_numpy_f64",
    "from_numpy_like",
    "stack_coords",
]
