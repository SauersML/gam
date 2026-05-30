"""Per-frame penalty adapters.

The dataclass penalty wrappers in :mod:`gamfit._penalties` expose
``value_grad(t)`` and ``hvp(t, v)``. The Rust kernel returns NumPy
arrays; for the torch and jax frames we wrap the same kernel so the
output is a native tensor with an autograd graph connecting ``value``
back to ``t``.

The math lives in one place: the Rust ``analytic_penalty_value_grad`` /
``analytic_penalty_hvp`` PyFFI functions. Each frame adapter is a thin
detach-cast-wrap layer.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._penalties import _penalty_value_grad_via_rust, _penalty_hvp_via_rust


# ---------------------------------------------------------------------------
# Torch frame
# ---------------------------------------------------------------------------


def torch_penalty_value_grad(wrapper: Any, t: Any) -> tuple[Any, Any]:
    """Torch-frame ``(value, grad)`` with autograd connected to ``t``.

    The forward pass calls the Rust kernel once. ``grad`` is the analytic
    Rust gradient already shaped like ``t``. The returned ``value`` tensor
    is autograd-connected to ``t`` through a
    :class:`torch.autograd.Function` whose backward consults the Rust
    kernel again — so ``torch.autograd.grad(value, t)`` matches ``grad``
    exactly.
    """
    from ._frame import import_torch
    from ._frame_torch import from_numpy_like, to_numpy_f64

    torch = import_torch()
    t_np = to_numpy_f64(t)
    value_np, grad_np = _penalty_value_grad_via_rust(wrapper, t_np)

    class _Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, x: Any) -> Any:
            ctx.save_for_backward(x)
            return torch.as_tensor(value_np, dtype=x.dtype, device=x.device)

        @staticmethod
        def backward(ctx: Any, grad_out: Any) -> Any:
            (x,) = ctx.saved_tensors
            x_np = to_numpy_f64(x)
            _v, g_np = _penalty_value_grad_via_rust(wrapper, x_np)
            g_t = from_numpy_like(np.asarray(g_np, dtype=np.float64), x)
            return g_t * grad_out.to(dtype=g_t.dtype, device=g_t.device)

    value_t = _Fn.apply(t)
    grad_t = from_numpy_like(np.asarray(grad_np, dtype=np.float64), t)
    return value_t, grad_t


def torch_penalty_hvp(wrapper: Any, t: Any, v: Any) -> Any:
    """Torch-frame Hessian-vector product. Pure forward (no autograd hook)."""
    from ._frame import import_torch
    from ._frame_torch import from_numpy_like, to_numpy_f64

    _ = import_torch()
    hv_np = _penalty_hvp_via_rust(wrapper, to_numpy_f64(t), to_numpy_f64(v))
    return from_numpy_like(np.asarray(hv_np, dtype=np.float64), t)


# ---------------------------------------------------------------------------
# JAX frame
# ---------------------------------------------------------------------------


def jax_penalty_value_grad(wrapper: Any, t: Any) -> tuple[Any, Any]:
    """JAX-frame ``(value, grad)`` differentiable via ``jax.grad``.

    Thin adapter over :func:`jax_value_grad_from_rust`: the only
    wrapper-specific piece is the host callback that runs the Rust
    ``analytic_penalty_value_grad`` kernel for this dataclass wrapper.
    """
    from ._penalty_jax_vjp import jax_value_grad_from_rust

    shape = tuple(int(s) for s in t.shape)

    def _callback(x_np: np.ndarray) -> tuple[float, np.ndarray]:
        return _penalty_value_grad_via_rust(wrapper, x_np)

    return jax_value_grad_from_rust(
        type(wrapper).__name__, shape, _callback, ref=t
    )


def jax_penalty_hvp(wrapper: Any, t: Any, v: Any) -> Any:
    """JAX-frame Hessian-vector product through :func:`jax.pure_callback`."""
    from ._frame import import_jax
    from ._frame_jax import from_numpy_like, to_numpy_f64

    _ = import_jax()
    hv_np = _penalty_hvp_via_rust(wrapper, to_numpy_f64(t), to_numpy_f64(v))
    return from_numpy_like(np.asarray(hv_np, dtype=np.float64), t)


__all__ = [
    "torch_penalty_value_grad",
    "torch_penalty_hvp",
    "jax_penalty_value_grad",
    "jax_penalty_hvp",
]
