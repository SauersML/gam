"""Shared JAX value/gradient custom-VJP marshalling for penalties.

Both the :class:`~gamfit._penalty_descriptors._RustPenaltyDescriptor` JAX
path and the dataclass-wrapper JAX path in :mod:`gamfit._penalty_frames`
need the same plumbing: run a Rust penalty kernel through
:func:`jax.pure_callback`, expose the scalar value as a
:class:`jax.custom_vjp` whose backward consults the kernel's analytic
gradient, and hand back the analytic gradient alongside.

That marshalling contract lives here once. The only thing that differs
between the two call sites is *which* Rust kernel runs and how its
``(value, grad)`` is produced — captured by the ``callback`` argument.
No penalty math is reimplemented in JAX; the callback is the single seam
to the Rust core (via ``analytic_penalty_value_grad``).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def jax_value_grad_from_rust(
    name: str,
    shape: tuple[int, ...],
    callback: Callable[[np.ndarray], tuple[float, np.ndarray]],
    *,
    ref: Any,
) -> tuple[Any, Any]:
    """JAX-frame ``(value, grad)`` for a Rust scalar penalty kernel.

    ``callback(x_np)`` runs the Rust kernel on a contiguous float64 array
    ``x_np`` (shaped like ``ref``) and returns ``(value, grad)`` where
    ``value`` is the scalar penalty and ``grad`` is ``∂value/∂x`` shaped
    like ``ref``. ``callback`` is the only place Rust is touched.

    The returned ``value`` is a :class:`jax.custom_vjp` scalar whose
    forward dispatches through :func:`jax.pure_callback` (safe under
    ``jit`` / ``vmap``) and whose backward multiplies the kernel's
    analytic gradient by the incoming cotangent — so
    ``jax.grad(lambda x: jax_value_grad_from_rust(...)[0])`` recovers the
    second return value to within float64 round-trip noise. ``grad`` is
    that same analytic gradient materialized as a :class:`jax.Array`
    matching ``ref``'s dtype.

    ``name`` labels the kernel for diagnostics; ``shape`` is the target
    shape used to size the gradient ``ShapeDtypeStruct``.
    """
    from ._frame import import_jax
    from ._frame_jax import from_numpy_like

    jax, jnp = import_jax()

    out_dtype = jnp.float64
    value_spec = jax.ShapeDtypeStruct((), out_dtype)
    grad_spec = jax.ShapeDtypeStruct(tuple(int(s) for s in shape), out_dtype)

    def _host_value(x: Any) -> np.ndarray:
        x_np = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
        value, _grad = callback(x_np)
        return np.asarray(value, dtype=np.float64)

    def _host_grad(args: Any) -> np.ndarray:
        x, gout = args
        x_np = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
        _value, grad = callback(x_np)
        return np.ascontiguousarray(
            (np.asarray(grad, dtype=np.float64) * float(np.asarray(gout))).reshape(
                grad_spec.shape
            )
        )

    @jax.custom_vjp
    def _value(x: Any) -> Any:
        return jax.pure_callback(_host_value, value_spec, x)

    def _value_fwd(x: Any) -> tuple[Any, Any]:
        return _value(x), x

    def _value_bwd(res: Any, g: Any) -> tuple[Any]:
        return (jax.pure_callback(_host_grad, grad_spec, (res, g)),)

    _value.defvjp(_value_fwd, _value_bwd)
    _value.__name__ = f"jax_penalty_value[{name}]"

    value_j = _value(ref)
    # Analytic gradient (no autograd needed) for the second return slot.
    _v0, grad0 = callback(np.ascontiguousarray(np.asarray(ref, dtype=np.float64)))
    grad_j = from_numpy_like(
        np.asarray(grad0, dtype=np.float64).reshape(grad_spec.shape), ref
    )
    return value_j, grad_j


__all__ = ["jax_value_grad_from_rust"]
