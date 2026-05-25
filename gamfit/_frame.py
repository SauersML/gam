"""Cross-frame interop dispatch.

Every numerical primitive in the gam ecosystem (basis descriptors,
penalties, kernels, manifold/SAE wrappers) is implemented exactly once
in the gam Rust core. The Python frontend can present that one
implementation to three different numerical frames — pure :mod:`numpy`,
:mod:`torch`, and :mod:`jax`. The *frame* is auto-detected from the type
of the inputs the user passes; the return is in the user's native frame;
gradients (when relevant) route through frame-native autograd hooked to
the same Rust value+VJP kernel.

Scope
-----

This module is the canonical cross-frame dispatcher for **any** Python
binding around the gam Rust core. Today the only such binding is
:mod:`gamfit`, so the module lives there; if another Python wrapper
(a CLI shim, a notebook helper, a downstream library) starts consuming
the gam Rust pyfunctions directly, it should import :class:`Frame`,
:func:`detect_frame`, :func:`to_numpy`, and :func:`from_numpy` from here
rather than re-implementing detection. The Rust-facing parts (FFI calls)
already live in Rust by construction — this module only handles the
Python-type boundary.

The default frame is :attr:`Frame.NUMPY`, which never imports torch or jax.
``import gamfit`` therefore costs nothing in a numpy-only environment.

Examples
--------
NumPy frame (no extra deps)::

    >>> import numpy as np
    >>> import gamfit
    >>> theta = np.linspace(0.0, 2 * np.pi, 8, endpoint=False)
    >>> ell = np.linspace(0.0, 1.0, 8)
    >>> phi = gamfit.Cylinder(n_knots=(7, 4)).evaluate(theta, ell)
    >>> phi.shape[0]
    8

Torch frame (torch arrays in, torch tensor out, grad connected)::

    >>> import torch  # doctest: +SKIP
    >>> theta_t = torch.linspace(0, 2*torch.pi, 8, requires_grad=True)
    >>> ell_t = torch.linspace(0, 1, 8, requires_grad=True)
    >>> phi_t = gamfit.Cylinder(n_knots=(7, 4)).evaluate(theta_t, ell_t)

JAX frame::

    >>> import jax.numpy as jnp  # doctest: +SKIP
    >>> phi_j = gamfit.Cylinder(n_knots=(7, 4)).evaluate(jnp.asarray(theta),
    ...                                                  jnp.asarray(ell))
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Iterable


class Frame(str, Enum):
    """Numerical frame for cross-frame interop.

    Membership and string values are stable across releases so frame
    detection results can be passed across module boundaries.
    """

    NUMPY = "numpy"
    TORCH = "torch"
    JAX = "jax"


_FRAME_VALUES = frozenset(f.value for f in Frame)


def _module_frame(obj: Any) -> Frame | None:
    """Inspect an object's class module string and map to a frame.

    Returns ``None`` when the object is not array-like. The inspection is
    deliberately non-importing: a user who has only numpy installed never
    accidentally triggers a torch or jax import here.
    """
    mod = type(obj).__module__
    if mod.startswith("torch"):
        return Frame.TORCH
    if mod.startswith("jax") or mod.startswith("jaxlib"):
        return Frame.JAX
    if mod.startswith("numpy") or mod.startswith("builtins"):
        return Frame.NUMPY
    return None


def detect_frame(*arrays: Any) -> Frame:
    """Detect the frame of an iterable of array-likes.

    The first non-``None`` array determines the frame. Empty input falls
    back to :attr:`Frame.NUMPY`. Subsequent arrays must agree; mixed frames
    raise :class:`TypeError` with a precise message naming both frames.

    Plain Python scalars / lists / tuples are treated as NumPy candidates
    (they cost nothing to coerce later) and do not constrain the frame.

    Examples
    --------
    >>> import numpy as np
    >>> detect_frame(np.zeros(3), np.ones(3))
    <Frame.NUMPY: 'numpy'>
    >>> detect_frame()
    <Frame.NUMPY: 'numpy'>
    """
    seen: Frame | None = None
    seen_obj: Any | None = None
    for arr in arrays:
        if arr is None:
            continue
        f = _module_frame(arr)
        if f is None:
            # plain python scalar/sequence -> defer to other inputs
            continue
        if f is Frame.NUMPY:
            # numpy never *forces* a frame; torch/jax win if present
            if seen is None:
                seen = Frame.NUMPY
                seen_obj = arr
            continue
        if seen is None or seen is Frame.NUMPY:
            seen = f
            seen_obj = arr
            continue
        if seen is not f:
            raise TypeError(
                "inputs must be in the same frame; got "
                f"{seen.value} + {f.value} "
                f"(first seen: {type(seen_obj).__name__}, "
                f"conflict: {type(arr).__name__})"
            )
    return seen if seen is not None else Frame.NUMPY


def normalize(frame: Frame | str | None, *arrays: Any) -> Frame:
    """Resolve an explicit ``frame`` argument or auto-detect.

    An explicit ``frame=`` string / enum overrides detection. Passing
    ``None`` triggers :func:`detect_frame`.
    """
    if frame is None:
        return detect_frame(*arrays)
    if isinstance(frame, Frame):
        return frame
    if not isinstance(frame, str) or frame not in _FRAME_VALUES:
        raise ValueError(
            f"unknown frame {frame!r}; expected one of {sorted(_FRAME_VALUES)}"
        )
    return Frame(frame)


def to_numpy(value: Any) -> Any:
    """Convert an array-like in any frame to a contiguous float64 numpy array.

    No autograd graph is preserved — callers that need a differentiable
    bridge should go through the per-frame adapters in
    :mod:`gamfit._frame_torch` / :mod:`gamfit._frame_jax`.
    """
    import numpy as np

    f = _module_frame(value)
    if f is Frame.TORCH:
        from . import _frame_torch as _t

        return _t.to_numpy_f64(value)
    if f is Frame.JAX:
        from . import _frame_jax as _j

        return _j.to_numpy_f64(value)
    arr = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return arr


def from_numpy(array: Any, frame: Frame, *, ref: Any | None = None) -> Any:
    """Wrap a numpy ndarray as the native array type of ``frame``.

    When ``ref`` is given and is from the same frame, the result inherits
    the dtype/device of ``ref`` so the round-trip preserves the user's
    placement. No graph is connected to ``ref``.
    """
    if frame is Frame.NUMPY:
        return array
    if frame is Frame.TORCH:
        from . import _frame_torch as _t

        return _t.from_numpy_like(array, ref)
    if frame is Frame.JAX:
        from . import _frame_jax as _j

        return _j.from_numpy_like(array, ref)
    raise ValueError(f"unsupported frame: {frame!r}")


def import_torch() -> Any:
    """Lazy-import torch with a stable error message when missing."""
    try:
        import torch  # noqa: F401

        return torch
    except ImportError as exc:  # pragma: no cover - environment specific
        raise ImportError(
            "this code path uses the torch frame, which requires torch. "
            "Install torch (`pip install torch`) or pass numpy / jax inputs."
        ) from exc


def import_jax() -> tuple[Any, Any]:
    """Lazy-import ``(jax, jax.numpy)`` with a stable error message."""
    try:
        import jax
        import jax.numpy as jnp

        return jax, jnp
    except ImportError as exc:  # pragma: no cover - environment specific
        raise ImportError(
            "this code path uses the jax frame, which requires jax. "
            "Install jax (`pip install jax`) or pass numpy / torch inputs."
        ) from exc


def assert_same_frame(*arrays: Any) -> Frame:
    """Detect the frame and raise :class:`TypeError` on mixed-frame inputs.

    Wrapper around :func:`detect_frame` with a single clear name for the
    public consumer-facing contract: "inputs must be in the same frame".
    """
    return detect_frame(*arrays)


def iter_array_likes(values: Iterable[Any]) -> list[Any]:
    """Collect plausible array-like inputs (filtering ``None``).

    Tiny helper used by primitives that accept a variadic ``*coords`` and
    want to feed it directly to :func:`detect_frame` without re-spelling
    the filter.
    """
    return [v for v in values if v is not None]


__all__ = [
    "Frame",
    "detect_frame",
    "normalize",
    "to_numpy",
    "from_numpy",
    "import_torch",
    "import_jax",
    "assert_same_frame",
    "iter_array_likes",
]
