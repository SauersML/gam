"""Device, dtype, and array coercion for the gamfit torch bridge.

The gamfit Rust engine runs on f64 CPU and accepts contiguous NumPy arrays.
Inputs from torch arrive in arbitrary dtypes on arbitrary devices. This module
centralises the detach-cast-move dance so every wrapper performs the conversion
identically. Non-tensor inputs pass through untouched so callers can mix
scalars, ints, lists, and tensors freely.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def to_numpy_f64(value: Any) -> Any:
    """Convert a torch tensor to a contiguous f64 NumPy array on CPU.

    Non-tensor values are returned untouched. The autograd graph is *not*
    preserved — callers wanting differentiable paths must use the autograd
    ``Function`` wrappers, which call this helper inside their forward pass.
    """
    import torch

    if isinstance(value, torch.Tensor):
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
    return value


def to_numpy_uintp(value: Any) -> Any:
    """Convert a torch tensor to a contiguous ``uintp`` NumPy array on CPU."""
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.device.type != "cpu":
            tensor = tensor.to(device="cpu")
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        arr = tensor.numpy()
        return np.ascontiguousarray(arr, dtype=np.uintp)
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value, dtype=np.uintp)
    return np.ascontiguousarray(np.asarray(value), dtype=np.uintp)


def from_numpy_like(array: Any, ref: Any) -> Any:
    """Wrap ``array`` as a torch tensor matching ``ref``'s device and dtype.

    ``ref`` must be a torch tensor; ``array`` may be a NumPy array or a scalar.
    The result is detached from any prior graph so callers can hand it to
    autograd without aliasing the originating buffer.
    """
    import torch

    if not isinstance(ref, torch.Tensor):
        raise TypeError("from_numpy_like requires a torch tensor as reference")
    array = np.asarray(array, dtype=np.float64, order="C")
    tensor = torch.as_tensor(array, dtype=torch.float64, device="cpu")
    if ref.device.type == "cpu" and ref.dtype == torch.float64:
        return tensor
    return tensor.to(device=ref.device, dtype=ref.dtype)
