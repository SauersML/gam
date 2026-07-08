"""Torch response-geometry transforms — thin marshalling around Rust kernels.

Every chart (closure, CLR/ALR/ILR log-ratio coordinates, the Aitchison Gram,
the simplex Fréchet/log/exp maps, and the sphere Fréchet/log/exp maps) is
computed by the ``gam::geometry`` Rust engine, the single source of truth shared
with the NumPy-facing :mod:`gamfit._response_geometry` surface. This module only
moves tensors across the FFI boundary.

Gradients survive that boundary through per-op :class:`torch.autograd.Function`
wrappers whose forward is an FFI round-trip returning the value **and** the
closed-form per-row Jacobian, and whose backward contracts ``grad_output`` with
that Jacobian (``einsum('nod,no->nd', jac, grad)``). The Jacobians are
hand-derived in Rust (``gam::geometry::manifolds::aitchison_ilr``), reproducing
exactly what torch autograd computed through the former elementwise form. Charts
that no caller differentiates (closure, ALR, ILR forward/inverse, the Fréchet
means, the sphere log map) route through value-only FFI, mirroring the
already-migrated ``sphere_frechet_mean`` seam.
"""

from __future__ import annotations

from typing import Any

import torch

from .._binding import rust_module
from . import _torch_compat as _tc
from ._coerce import from_numpy_like, to_numpy_f64


def _rust() -> Any:
    return rust_module()


def _as_matrix(value: torch.Tensor, *, label: str) -> torch.Tensor:
    """Coerce a tensor argument to a 2-D float tensor, keeping it on the tape.

    Domain validation (positivity, finiteness, minimum shape) is owned by the
    Rust kernels; this only enforces the tensor/rank contract and the float
    dtype the FFI expects, so a dtype cast stays differentiable.
    """
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if value.dim() != 2:
        raise ValueError(f"{label} must be a 2-D tensor")
    if not _tc.is_floating_point(value):
        value = value.to(dtype=_tc.float64)
    return value


def _as_tangent(value: torch.Tensor, *, label: str) -> torch.Tensor:
    """Coerce a tangent argument, promoting a 1-D vector to a single row."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if not _tc.is_floating_point(value):
        value = value.to(dtype=_tc.float64)
    if value.dim() == 1:
        value = value.reshape(1, -1)
    if value.dim() != 2:
        raise ValueError(f"{label} must be a 1-D or 2-D tensor")
    return value


def _base_numpy(base: torch.Tensor) -> Any:
    """Marshal a base point to a contiguous 1-D f64 NumPy array (detached)."""
    if not isinstance(base, torch.Tensor):
        raise TypeError("base must be a torch.Tensor")
    return to_numpy_f64(base).reshape(-1)


# ─────────────────────────── autograd Functions ─────────────────────────────
#
# Each forward calls a ``*_jet`` FFI returning (value, per-row Jacobian). The
# Jacobian ``jac[n, out, in]`` is saved and contracted with ``grad_output`` in
# backward. Non-tensor arguments (base point, coordinate label, reference) carry
# no gradient, so their backward slots are ``None``.


def _contract(jac: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    return torch.einsum("nod,no->nd", jac, grad_output)


class _ClrFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, values: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        value_np, jac_np = _rust().response_geometry_clr_jet(to_numpy_f64(values))
        ctx.save_for_backward(from_numpy_like(jac_np, values))
        return from_numpy_like(value_np, values)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):  # type: ignore[override]
        (jac,) = ctx.saved_tensors
        return _contract(jac, grad_output)


class _SimplexLogMapFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, values: torch.Tensor, base_np: Any, coordinates: str, reference: int
    ) -> torch.Tensor:
        value_np, jac_np = _rust().response_geometry_simplex_log_map_jet(
            to_numpy_f64(values), base_np, coordinates, int(reference)
        )
        ctx.save_for_backward(from_numpy_like(jac_np, values))
        return from_numpy_like(value_np, values)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):  # type: ignore[override]
        (jac,) = ctx.saved_tensors
        return _contract(jac, grad_output), None, None, None


class _SimplexExpMapFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, tangent: torch.Tensor, base_np: Any, coordinates: str, reference: int
    ) -> torch.Tensor:
        value_np, jac_np = _rust().response_geometry_simplex_exp_map_jet(
            to_numpy_f64(tangent), base_np, coordinates, int(reference)
        )
        ctx.save_for_backward(from_numpy_like(jac_np, tangent))
        return from_numpy_like(value_np, tangent)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):  # type: ignore[override]
        (jac,) = ctx.saved_tensors
        return _contract(jac, grad_output), None, None, None


class _SphereExpMapFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, tangent: torch.Tensor, base_np: Any) -> torch.Tensor:  # type: ignore[override]
        value_np, jac_np = _rust().response_geometry_sphere_exp_map_jet(
            to_numpy_f64(tangent), base_np
        )
        ctx.save_for_backward(from_numpy_like(jac_np, tangent))
        return from_numpy_like(value_np, tangent)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):  # type: ignore[override]
        (jac,) = ctx.saved_tensors
        return _contract(jac, grad_output), None


# ─────────────────────────── simplex charts ─────────────────────────────────


def closure(values: torch.Tensor) -> torch.Tensor:
    """Normalize rows onto the probability simplex."""
    v = _as_matrix(values, label="simplex values")
    return from_numpy_like(_rust().response_geometry_closure(to_numpy_f64(v)), v)


def clr(values: torch.Tensor) -> torch.Tensor:
    """Centered log-ratio coordinates for positive compositions."""
    v = _as_matrix(values, label="simplex values")
    return _ClrFn.apply(v)


def alr(values: torch.Tensor, *, reference: int = -1) -> torch.Tensor:
    """Additive log-ratio coordinates for positive compositions."""
    v = _as_matrix(values, label="simplex values")
    return from_numpy_like(
        _rust().response_geometry_alr(to_numpy_f64(v), int(reference)), v
    )


def inverse_alr(coords: torch.Tensor, *, reference: int = -1) -> torch.Tensor:
    """Map ALR coordinates back to the simplex."""
    z = _as_matrix(coords, label="ALR coordinates")
    return from_numpy_like(
        _rust().response_geometry_inverse_alr(to_numpy_f64(z), int(reference)), z
    )


def ilr(values: torch.Tensor, *, reference: int = -1) -> torch.Tensor:
    """Isometric log-ratio coordinates for positive compositions.

    ILR maps a ``d``-part composition to ``d-1`` Euclidean coordinates that are
    isometric to Aitchison geometry: Euclidean distance in ILR space equals
    Aitchison distance on the simplex. The ``reference`` argument is accepted for
    a uniform call signature with :func:`alr` but is unused — the Helmert basis
    is canonical and reference-free.
    """
    del reference  # Helmert ILR basis is reference-free; kept for signature parity.
    v = _as_matrix(values, label="simplex values")
    return from_numpy_like(_rust().response_geometry_ilr(to_numpy_f64(v)), v)


def inverse_ilr(coords: torch.Tensor, *, reference: int = -1) -> torch.Tensor:
    """Map ILR coordinates back to the simplex."""
    del reference  # Helmert ILR basis is reference-free; kept for signature parity.
    z = _as_matrix(coords, label="ILR coordinates")
    return from_numpy_like(_rust().response_geometry_inverse_ilr(to_numpy_f64(z)), z)


def aitchison_metric(
    d: int, *, dtype: Any = None, device: Any = None
) -> torch.Tensor:
    """Aitchison Gram matrix ``G = I_{d-1} − (1/d)·11ᵀ`` for ALR coordinates.

    ALR is a valid chart but is NOT isometric to Aitchison geometry: in ALR
    coordinates the Aitchison inner product is ``⟨u, v⟩ = uᵀ G v`` with this
    ``(d-1)×(d-1)`` Gram matrix (for ``d = 3`` it is ``[[2/3, -1/3], [-1/3,
    2/3]]`` ≠ I). Prefer ILR, which makes ``G = I``.
    """
    g = _rust().response_geometry_aitchison_metric(int(d))
    return torch.as_tensor(g, dtype=(_tc.float64 if dtype is None else dtype), device=device)


def simplex_frechet_mean(
    values: torch.Tensor, weights: torch.Tensor | None = None
) -> torch.Tensor:
    """Intrinsic Fréchet mean under Aitchison simplex geometry."""
    v = _as_matrix(values, label="simplex values")
    w = None
    if weights is not None:
        if not isinstance(weights, torch.Tensor):
            raise TypeError("weights must be a torch.Tensor or None")
        w = to_numpy_f64(weights).reshape(-1)
    out = _rust().response_geometry_simplex_frechet_mean(to_numpy_f64(v), w)
    return from_numpy_like(out, v)


def simplex_log_map(
    values: torch.Tensor,
    base: torch.Tensor,
    *,
    coordinates: str = "ilr",
    reference: int = -1,
) -> torch.Tensor:
    """Log map at an intrinsic simplex base point in ILR, CLR, or ALR coordinates.

    ``ilr`` (default) and ``clr`` are isometric to Aitchison geometry, so the
    Euclidean norm of the returned tangent equals the Aitchison geodesic
    distance from ``base`` to each row of ``values``. ``alr`` is a valid but
    NON-isometric chart (its Aitchison Gram is :func:`aitchison_metric`, not the
    identity).
    """
    v = _as_matrix(values, label="simplex values")
    return _SimplexLogMapFn.apply(v, _base_numpy(base), str(coordinates).lower(), int(reference))


def simplex_exp_map(
    tangent: torch.Tensor,
    base: torch.Tensor,
    *,
    coordinates: str = "ilr",
    reference: int = -1,
) -> torch.Tensor:
    """Exponential map from simplex tangent coordinates back to compositions.

    The default ``ilr`` chart (and ``clr``) is isometric to Aitchison geometry;
    ``alr`` is the non-isometric chart. Must match the ``coordinates`` used by
    :func:`simplex_log_map`. ILR/ALR tangents have ``D-1`` columns, CLR has ``D``.
    """
    z = _as_tangent(tangent, label="tangent")
    return _SimplexExpMapFn.apply(z, _base_numpy(base), str(coordinates).lower(), int(reference))


# ─────────────────────────── sphere charts ──────────────────────────────────


def sphere_frechet_mean(
    values: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    tol: float = 1e-12,
    max_iter: int = 256,
) -> torch.Tensor:
    """Intrinsic Fréchet/Karcher mean on the unit sphere.

    If the minimizer is not unique, as for an exactly antipodal pair, this
    returns one deterministic minimizer rather than an endpoint surrogate.
    """
    v = _as_matrix(values, label="spherical values")
    w = None
    if weights is not None:
        if not isinstance(weights, torch.Tensor):
            raise TypeError("weights must be a torch.Tensor or None")
        w = to_numpy_f64(weights).reshape(-1)
    out = _rust().sphere_frechet_mean(to_numpy_f64(v), w, float(tol), int(max_iter))
    return from_numpy_like(out, v)


def sphere_log_map(values: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """Log map from the unit sphere to the tangent space at ``base``.

    The log map is non-unique at antipodal points, so those inputs are
    rejected instead of being mapped to a false zero tangent.
    """
    v = _as_matrix(values, label="spherical values")
    return from_numpy_like(
        _rust().response_geometry_sphere_log_map(to_numpy_f64(v), _base_numpy(base)), v
    )


def sphere_exp_map(tangent: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """Exponential map from the ambient tangent space at ``base`` to the sphere."""
    z = _as_tangent(tangent, label="tangent")
    return _SphereExpMapFn.apply(z, _base_numpy(base))
