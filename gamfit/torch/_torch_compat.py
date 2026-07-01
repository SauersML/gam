"""Pyright-safe re-exports of torch primitives used by the gamfit torch bridge.

PyTorch populates names like :func:`torch.as_tensor`, :func:`torch.where`,
:func:`torch.isfinite`, and the dtype constants (``torch.float64`` etc.) via a
runtime wildcard import from :mod:`torch._C` and a loop over
``torch._C._VariableFunctions``. Pyright analyses the public ``torch`` package
statically and flags every access to those names as
``reportPrivateImportUsage`` because they are not listed in ``torch.__all__``
at the source file level.

To keep the bridge type-clean without per-call suppressions, this module looks
the symbols up through a typed :data:`Any` view of the :mod:`torch` module and
re-exposes them as thin typed wrappers. Callers in :mod:`gamfit.torch` import
from this module instead of touching the flagged attributes directly.

The wrappers are intentionally minimal — they exist purely to satisfy static
analyzers and add no behavior. Runtime semantics match :mod:`torch` exactly.
"""

from __future__ import annotations

from typing import Any, cast

import torch as _torch

_torch_any: Any = cast(Any, _torch)


def as_tensor(data: Any, dtype: Any = None, device: Any = None) -> _torch.Tensor:
    """Forward to :func:`torch.as_tensor` preserving its runtime contract."""
    return cast(_torch.Tensor, _torch_any.as_tensor(data, dtype=dtype, device=device))


def is_floating_point(tensor: _torch.Tensor) -> bool:
    """Forward to :func:`torch.is_floating_point`."""
    return bool(_torch_any.is_floating_point(tensor))


def isfinite(tensor: _torch.Tensor) -> _torch.Tensor:
    """Forward to :func:`torch.isfinite`."""
    return cast(_torch.Tensor, _torch_any.isfinite(tensor))


def full(
    size: Any,
    fill_value: Any,
    *,
    dtype: Any = None,
    device: Any = None,
) -> _torch.Tensor:
    """Forward to :func:`torch.full`."""
    return cast(_torch.Tensor, _torch_any.full(size, fill_value, dtype=dtype, device=device))


def zeros(size: Any, *, dtype: Any = None, device: Any = None) -> _torch.Tensor:
    """Forward to :func:`torch.zeros`."""
    return cast(_torch.Tensor, _torch_any.zeros(size, dtype=dtype, device=device))


def where(condition: _torch.Tensor, x: _torch.Tensor, y: _torch.Tensor) -> _torch.Tensor:
    """Forward to :func:`torch.where` (3-argument form)."""
    return cast(_torch.Tensor, _torch_any.where(condition, x, y))


def ones_like(tensor: _torch.Tensor) -> _torch.Tensor:
    """Forward to :func:`torch.ones_like`."""
    return cast(_torch.Tensor, _torch_any.ones_like(tensor))


def zeros_like(tensor: _torch.Tensor) -> _torch.Tensor:
    """Forward to :func:`torch.zeros_like`."""
    return cast(_torch.Tensor, _torch_any.zeros_like(tensor))


# Dtype constants. Pyright treats ``torch.dtype`` itself as a private import,
# so the static type is intentionally ``Any``; the runtime value is the dtype
# instance and behaves identically to ``torch.float64``.
float64: Any = _torch_any.float64
