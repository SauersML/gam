"""Device/dtype coercion helpers for the torch wrapper layer.

The Rust engine works exclusively on CPU f64 numpy arrays. The torch
wrappers accept tensors of any dtype/device, move them to CPU f64 numpy
for the FFI call, then cast outputs back to the dtype/device of the
input tensors. Pass-through types (ints, strings, Python sequences) are
left untouched.
"""

from __future__ import annotations

from typing import Any


def _torch() -> Any:
    import torch  # noqa: F401  - re-export only

    return torch


def to_numpy_f64(value: Any) -> Any:
    """Detach, move to CPU, cast to f64 and return a contiguous numpy array.

    Non-tensor inputs are converted via ``numpy.asarray``; non-array
    inputs (scalars, strings, ``None``) are returned untouched.
    """
    import numpy as np

    torch = _torch()
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return np.ascontiguousarray(value.detach().cpu().to(torch.float64).numpy())
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value, dtype=np.float64)
    return value


def from_numpy_like(array: Any, like: Any) -> Any:
    """Cast a numpy array back to the dtype/device of a reference tensor.

    If ``like`` is not a tensor the array is returned unchanged.
    """
    import numpy as np

    torch = _torch()
    if array is None:
        return None
    if not isinstance(like, torch.Tensor):
        return array
    arr = np.ascontiguousarray(array)
    return torch.as_tensor(arr, dtype=like.dtype, device=like.device)


def reference_tensor(*candidates: Any) -> Any:
    """Pick the first torch tensor from ``candidates`` to use as a reference.

    Returns ``None`` when none of the candidates are tensors. The result
    is used as the dtype/device anchor for output casting.
    """
    torch = _torch()
    for value in candidates:
        if isinstance(value, torch.Tensor):
            return value
    return None
