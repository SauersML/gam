"""NumPy frame adapter — the default, no-extra-deps path.

The NumPy frame is the canonical zero-overhead route into the Rust core:
inputs are already contiguous float64 NumPy arrays (or coercible to such),
the Rust pyfunction returns a float64 NumPy array, and that array is the
final result. There is no autograd, no jit, no vmap — those are the value
proposition of the other frames.

This module exists so per-frame adapters share a uniform import surface
(``gamfit._frame_<frame>``). Concrete primitives import the helpers they
need from here rather than open-coding ``np.ascontiguousarray`` everywhere.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def to_numpy_f64(value: Any) -> np.ndarray:
    """Coerce an arbitrary array-like to a contiguous float64 numpy array."""
    arr = np.asarray(value, dtype=np.float64)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr


def stack_coords_f64(coords: list[Any] | tuple[Any, ...]) -> np.ndarray:
    """Stack a sequence of 1D coordinate vectors into a (B, d) f64 array.

    Validates that every coordinate is 1D and shares a common length.
    """
    if len(coords) == 0:
        raise ValueError("stack_coords_f64 requires at least one coordinate")
    arrays: list[np.ndarray] = []
    ref_len: int | None = None
    for idx, c in enumerate(coords):
        a = to_numpy_f64(c)
        if a.ndim != 1:
            raise ValueError(
                f"coord {idx}: expected 1D array, got shape {tuple(a.shape)}"
            )
        if ref_len is None:
            ref_len = int(a.shape[0])
        elif int(a.shape[0]) != ref_len:
            raise ValueError(
                f"coord {idx}: length {a.shape[0]} does not match reference "
                f"length {ref_len}"
            )
        arrays.append(a)
    return np.stack(arrays, axis=1)


__all__ = ["to_numpy_f64", "stack_coords_f64"]
