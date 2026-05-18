"""Device, dtype, and array coercion for the gamfit torch bridge.

The gamfit Rust engine runs on f64 CPU and accepts contiguous NumPy arrays.
Inputs from torch arrive in arbitrary dtypes on arbitrary devices. This module
centralises the detach-cast-move dance so every wrapper performs the conversion
identically. Inputs must be ``torch.Tensor`` — NumPy-array callers go through
:mod:`gamfit._api` directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from . import _torch_compat as _tc


def to_numpy_f64(value: torch.Tensor) -> Any:
    """Convert a torch tensor to a contiguous f64 NumPy array on CPU.

    The autograd graph is *not* preserved — callers wanting differentiable
    paths must use the autograd ``Function`` wrappers, which call this helper
    inside their forward pass.
    """
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(value).__name__}")
    tensor = value.detach()
    if tensor.device.type != "cpu":
        tensor = tensor.to(device="cpu", dtype=_tc.float64)
    elif tensor.dtype != _tc.float64:
        tensor = tensor.to(dtype=_tc.float64)
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    arr = tensor.numpy()
    if arr.dtype == np.float64 and arr.flags.c_contiguous:
        return arr
    return np.ascontiguousarray(arr, dtype=np.float64)


def to_numpy_uintp(value: torch.Tensor) -> Any:
    """Convert a torch tensor to a contiguous ``uintp`` NumPy array on CPU."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(value).__name__}")
    tensor = value.detach()
    if tensor.device.type != "cpu":
        tensor = tensor.to(device="cpu")
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return np.ascontiguousarray(tensor.numpy(), dtype=np.uintp)


def from_numpy_like(array: Any, ref: torch.Tensor) -> torch.Tensor:
    """Wrap ``array`` as a torch tensor matching ``ref``'s device and dtype.

    The result is detached from any prior graph so callers can hand it to
    autograd without aliasing the originating buffer.
    """
    if not isinstance(ref, torch.Tensor):
        raise TypeError("from_numpy_like requires a torch tensor as reference")
    array = np.asarray(array, dtype=np.float64, order="C")
    tensor = _tc.as_tensor(array, dtype=_tc.float64, device="cpu")
    if ref.device.type == "cpu" and ref.dtype == _tc.float64:
        return tensor
    return tensor.to(device=ref.device, dtype=ref.dtype)
