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

from typing import Any, Callable

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
    :class:`jax.custom_vjp` (see :func:`wrap_value_grad`).
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


def wrap_value_grad(
    fwd_numpy: Callable[..., np.ndarray],
    vjp_numpy: Callable[..., np.ndarray],
    *,
    out_shape: tuple[int, ...],
) -> Callable[..., Any]:
    """Build a ``jax.custom_vjp`` around a numpy forward + numpy VJP.

    ``fwd_numpy(x_np)`` must return the primal output as a NumPy array of
    shape ``out_shape`` and dtype float64. ``vjp_numpy(x_np, grad_out_np)``
    must return ``∂L/∂x`` shaped like ``x``.

    The forward dispatches through :func:`jax.pure_callback`, which is
    transparent to ``jit`` and ``vmap`` (the callback runs once per
    batched element when traced inside ``vmap``; for fused-batch behavior
    use the Rust pyfunction directly outside ``vmap``).
    """
    jax, jnp = _jax()

    out_dtype = jnp.float64

    @jax.custom_vjp
    def fwd(x: Any) -> Any:
        spec = jax.ShapeDtypeStruct(out_shape, out_dtype)

        def _host(x_arr: Any) -> np.ndarray:
            x_np = np.asarray(x_arr, dtype=np.float64)
            return np.ascontiguousarray(
                np.asarray(fwd_numpy(x_np), dtype=np.float64)
            )

        return jax.pure_callback(_host, spec, x)

    def fwd_fwd(x: Any) -> tuple[Any, Any]:
        return fwd(x), x

    def fwd_bwd(res: Any, grad_out: Any) -> tuple[Any]:
        x = res
        in_shape = x.shape
        spec_in = jax.ShapeDtypeStruct(in_shape, out_dtype)

        def _host_vjp(args: Any) -> np.ndarray:
            x_arr, g_arr = args
            x_np = np.asarray(x_arr, dtype=np.float64)
            g_np = np.asarray(g_arr, dtype=np.float64)
            return np.ascontiguousarray(
                np.asarray(vjp_numpy(x_np, g_np), dtype=np.float64)
            )

        gx = jax.pure_callback(_host_vjp, spec_in, (x, grad_out))
        return (gx,)

    fwd.defvjp(fwd_fwd, fwd_bwd)
    return fwd


__all__ = [
    "to_numpy_f64",
    "from_numpy_like",
    "stack_coords",
    "wrap_value_grad",
]
