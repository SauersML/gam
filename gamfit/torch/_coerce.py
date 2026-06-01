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

from .._frame_torch import from_numpy_like as _frame_from_numpy_like
from .._frame_torch import to_numpy_f64 as _frame_to_numpy_f64


def to_numpy_f64(value: torch.Tensor) -> Any:
    """Convert a torch tensor to a contiguous f64 NumPy array on CPU.

    The autograd graph is *not* preserved — callers wanting differentiable
    paths must use the autograd ``Function`` wrappers, which call this helper
    inside their forward pass.

    Delegates to :func:`gamfit._frame_torch.to_numpy_f64`, the single source
    of truth for the detach → CPU → float64 → contiguous → numpy dance.
    """
    return _frame_to_numpy_f64(value)


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

    The torch bridge always passes a concrete ``torch.Tensor`` reference, so
    this variant rejects a missing reference up front; the device/dtype-match
    body itself lives once in :func:`gamfit._frame_torch.from_numpy_like`.
    """
    if not isinstance(ref, torch.Tensor):
        raise TypeError("from_numpy_like requires a torch tensor as reference")
    return _frame_from_numpy_like(array, ref)
