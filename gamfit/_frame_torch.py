"""Torch frame adapter.

Wraps the same Rust kernels every other frame uses, but presents the
output as a :class:`torch.Tensor` with a connected autograd graph. The
backward pass routes through a Rust-supplied VJP — no math is
reimplemented in torch.

Torch is an **optional** dependency. Importing :mod:`gamfit._frame_torch`
itself never raises; it imports torch lazily on first use of any helper.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def _torch() -> Any:
    from ._frame import import_torch

    return import_torch()


def to_numpy_f64(value: Any) -> np.ndarray:
    """Convert a torch tensor to a contiguous f64 NumPy array on CPU.

    The autograd graph is *not* preserved — callers that need a
    differentiable path must wrap the Rust call in a
    :class:`torch.autograd.Function` (see :func:`wrap_value_grad`).
    """
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(value).__name__}")
    tensor = value.detach()
    if tensor.device.type != "cpu":
        tensor = tensor.to(device="cpu", dtype=torch.float64)
    elif tensor.dtype != torch.float64:
        tensor = tensor.to(dtype=torch.float64)
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    arr = tensor.numpy()
    if arr.dtype == np.float64 and arr.flags.c_contiguous:
        return arr
    return np.ascontiguousarray(arr, dtype=np.float64)


def from_numpy_like(array: Any, ref: Any | None) -> Any:
    """Wrap a numpy ndarray as a torch tensor matching ``ref``.

    When ``ref`` is ``None`` we default to ``float64`` on CPU. When
    ``ref`` is a torch tensor we adopt its dtype and device.
    """
    torch = _torch()
    np_arr = np.asarray(array, dtype=np.float64, order="C")
    tensor = torch.as_tensor(np_arr, dtype=torch.float64, device="cpu")
    if isinstance(ref, torch.Tensor):
        if ref.device.type != "cpu" or ref.dtype != torch.float64:
            tensor = tensor.to(device=ref.device, dtype=ref.dtype)
    return tensor


def stack_coords(coords: list[Any] | tuple[Any, ...]) -> Any:
    """Stack 1D torch coordinates into a (B, d) float64 tensor."""
    torch = _torch()
    if len(coords) == 0:
        raise ValueError("stack_coords requires at least one coordinate")
    tensors: list[Any] = []
    ref_len: int | None = None
    for idx, c in enumerate(coords):
        if not isinstance(c, torch.Tensor):
            c = torch.as_tensor(c, dtype=torch.float64)
        if c.dim() != 1:
            raise ValueError(
                f"coord {idx}: expected 1D tensor, got shape {tuple(c.shape)}"
            )
        if ref_len is None:
            ref_len = int(c.numel())
        elif int(c.numel()) != ref_len:
            raise ValueError(
                f"coord {idx}: length {c.numel()} does not match reference "
                f"length {ref_len}"
            )
        if not torch.is_floating_point(c):
            c = c.to(dtype=torch.float64)
        tensors.append(c)
    return torch.stack(tensors, dim=1)


def wrap_value_grad(
    fwd_numpy: Callable[..., np.ndarray],
    vjp_numpy: Callable[..., np.ndarray],
) -> Callable[..., Any]:
    """Build a torch-autograd wrapper around a numpy forward + numpy VJP.

    ``fwd_numpy(x_np)`` must return the primal output as a NumPy array.
    ``vjp_numpy(x_np, grad_out_np)`` must return ``∂L/∂x`` shaped like
    ``x``. The returned callable takes a single ``torch.Tensor`` and
    returns a :class:`torch.Tensor` with the autograd graph connected.

    This helper does *not* assume any particular shape — both the primal
    and the cotangent are passed through numpy contiguously.
    """
    torch = _torch()

    class _Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, x: Any) -> Any:
            x_np = to_numpy_f64(x)
            y_np = np.asarray(fwd_numpy(x_np), dtype=np.float64)
            ctx.save_for_backward(x)
            ctx._gamfit_shape_y = tuple(y_np.shape)
            return from_numpy_like(y_np, x)

        @staticmethod
        def backward(ctx: Any, grad_out: Any) -> Any:
            (x,) = ctx.saved_tensors
            x_np = to_numpy_f64(x)
            g_np = to_numpy_f64(grad_out)
            gx_np = np.asarray(vjp_numpy(x_np, g_np), dtype=np.float64)
            return from_numpy_like(gx_np, x)

    def call(x: Any) -> Any:
        return _Fn.apply(x)

    return call


__all__ = [
    "to_numpy_f64",
    "from_numpy_like",
    "stack_coords",
    "wrap_value_grad",
]
