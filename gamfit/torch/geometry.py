"""Forward-only torch passthroughs for the response-geometry transforms.

Each public function in this module is a thin wrapper around its
:mod:`gamfit._response_geometry` counterpart. Tensor inputs are detached and
moved to a CPU f64 NumPy buffer via :func:`gamfit.torch._coerce.to_numpy_f64`,
the numpy implementation runs, and — if at least one input was a torch
tensor — the result is wrapped back into a tensor on the first input tensor's
device and dtype via :func:`gamfit.torch._coerce.from_numpy_like`. If every
input was already a NumPy array (or list / scalar), the result is returned as
a NumPy array, exactly matching the numpy module's behaviour.

Forward-only. Gradients do not flow through these transforms; they cross the
NumPy boundary. For a differentiable closure, write
``x / x.sum(-1, keepdim=True)`` in your own code. For other transforms where
you need autograd, compose standard torch ops directly.
"""

from __future__ import annotations

from typing import Any

from .. import _response_geometry as _np_geom
from ._coerce import from_numpy_like, to_numpy_f64


def _is_tensor(value: Any) -> bool:
    import torch

    return isinstance(value, torch.Tensor)


def _passthrough(result: Any, ref: Any) -> Any:
    """Wrap ``result`` as a tensor matching ``ref`` iff ``ref`` is a tensor."""
    if _is_tensor(ref):
        return from_numpy_like(result, ref)
    return result


def closure(values: Any) -> Any:
    """Normalize rows onto the probability simplex.

    Forward-only torch passthrough; see module docstring.
    """
    out = _np_geom.closure(to_numpy_f64(values))
    return _passthrough(out, values)


def clr(values: Any) -> Any:
    """Centered log-ratio coordinates for positive compositions.

    Forward-only torch passthrough; see module docstring.
    """
    out = _np_geom.clr(to_numpy_f64(values))
    return _passthrough(out, values)


def alr(values: Any, *, reference: int = -1) -> Any:
    """Additive log-ratio coordinates for positive compositions.

    Forward-only torch passthrough; see module docstring.
    """
    out = _np_geom.alr(to_numpy_f64(values), reference=reference)
    return _passthrough(out, values)


def inverse_alr(coords: Any, *, reference: int = -1) -> Any:
    """Map ALR coordinates back to the simplex.

    Forward-only torch passthrough; see module docstring.
    """
    out = _np_geom.inverse_alr(to_numpy_f64(coords), reference=reference)
    return _passthrough(out, coords)


def simplex_frechet_mean(values: Any, weights: Any | None = None) -> Any:
    """Intrinsic Fréchet mean under Aitchison simplex geometry.

    Forward-only torch passthrough; see module docstring.
    """
    out = _np_geom.simplex_frechet_mean(
        to_numpy_f64(values),
        None if weights is None else to_numpy_f64(weights),
    )
    return _passthrough(out, values)


def simplex_log_map(
    values: Any,
    base: Any,
    *,
    coordinates: str = "clr",
    reference: int = -1,
) -> Any:
    """Log map at an intrinsic simplex base point in CLR or ALR coordinates.

    Forward-only torch passthrough; see module docstring.
    """
    out = _np_geom.simplex_log_map(
        to_numpy_f64(values),
        to_numpy_f64(base),
        coordinates=coordinates,
        reference=reference,
    )
    return _passthrough(out, values)


def simplex_exp_map(
    tangent: Any,
    base: Any,
    *,
    coordinates: str = "clr",
    reference: int = -1,
) -> Any:
    """Exponential map from simplex tangent coordinates back to compositions.

    Forward-only torch passthrough; see module docstring.
    """
    out = _np_geom.simplex_exp_map(
        to_numpy_f64(tangent),
        to_numpy_f64(base),
        coordinates=coordinates,
        reference=reference,
    )
    return _passthrough(out, tangent)


def sphere_frechet_mean(
    values: Any,
    weights: Any | None = None,
    *,
    tol: float = 1e-12,
    max_iter: int = 256,
) -> Any:
    """Intrinsic Fréchet/Karcher mean on the unit sphere.

    Forward-only torch passthrough; see module docstring.
    """
    out = _np_geom.sphere_frechet_mean(
        to_numpy_f64(values),
        None if weights is None else to_numpy_f64(weights),
        tol=tol,
        max_iter=max_iter,
    )
    return _passthrough(out, values)


def sphere_log_map(values: Any, base: Any) -> Any:
    """Log map from the unit sphere to the ambient tangent space at ``base``.

    Forward-only torch passthrough; see module docstring.
    """
    out = _np_geom.sphere_log_map(to_numpy_f64(values), to_numpy_f64(base))
    return _passthrough(out, values)


def sphere_exp_map(tangent: Any, base: Any) -> Any:
    """Exponential map from the ambient tangent space at ``base`` to the sphere.

    Forward-only torch passthrough; see module docstring.
    """
    out = _np_geom.sphere_exp_map(to_numpy_f64(tangent), to_numpy_f64(base))
    return _passthrough(out, tangent)
