"""Torch frame adapter.

Wraps the same Rust kernels every other frame uses, but presents the
output as a :class:`torch.Tensor` with a connected autograd graph. The
backward pass routes through a Rust-supplied VJP — no math is
reimplemented in torch.

Torch is an **optional** dependency. Importing :mod:`gamfit._frame_torch`
itself never raises; it imports torch lazily on first use of any helper.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._frame_shared import stack_coords_generic


def _torch() -> Any:
    from ._frame import import_torch

    return import_torch()


def to_numpy_f64(value: Any) -> np.ndarray:
    """Convert a torch tensor to a contiguous f64 NumPy array on CPU.

    The autograd graph is *not* preserved — callers that need a
    differentiable path must wrap the Rust call in a
    :class:`torch.autograd.Function`.
    """
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(value).__name__}")
    tensor = value.detach()
    # Move host first, *then* cast dtype. A fused ``.to(device="cpu",
    # dtype=float64)`` makes PyTorch satisfy the float64 cast on the source
    # device, and MPS has no float64 — so it raises instead of just moving to
    # CPU and casting there. Splitting the ops keeps every accelerator working.
    if tensor.device.type != "cpu":
        tensor = tensor.to(device="cpu")
    if tensor.dtype != torch.float64:
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

    def _coerce(idx: int, c: Any, _ref_len: int | None) -> tuple[Any, int]:
        if not isinstance(c, torch.Tensor):
            c = torch.as_tensor(c, dtype=torch.float64)
        if c.dim() != 1:
            raise ValueError(
                f"coord {idx}: expected 1D tensor, got shape {tuple(c.shape)}"
            )
        length = int(c.numel())
        if not torch.is_floating_point(c):
            c = c.to(dtype=torch.float64)
        return c, length

    return stack_coords_generic(
        coords,
        coerce=_coerce,
        stack=lambda tensors: torch.stack(tensors, dim=1),
    )


__all__ = [
    "to_numpy_f64",
    "from_numpy_like",
    "stack_coords",
]
