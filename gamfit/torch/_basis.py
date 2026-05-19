"""Differentiable basis, penalty, and closed-form ridge primitives for torch.

These wrappers mirror the NumPy entry points in :mod:`gamfit._api`. The
basis evaluators ``bspline_basis`` and ``duchon_basis_1d`` carry an analytic
backward with respect to ``t`` implemented through :class:`torch.autograd.Function`
subclasses; the derivative evaluators, penalty constructor, and closed-form
ridge solves are forward-only and produced via the detach-cast-call-numpy-wrap
path in :mod:`gamfit.torch._coerce`.
"""

from __future__ import annotations

from typing import Any, Callable, cast

import torch

from .. import _api
from ._coerce import from_numpy_like, to_numpy_f64, to_numpy_uintp


def _resolve_centers_tensor(t: torch.Tensor, centers: Any) -> torch.Tensor:
    """Accept None / int / Tensor for Duchon centers; auto-derive when needed."""
    if isinstance(centers, torch.Tensor):
        return centers
    resolved = _api._resolve_centers(centers, to_numpy_f64(t), label="centers")
    return from_numpy_like(resolved, t)


def _resolve_knots_tensor(
    t: torch.Tensor, knots: Any, *, degree: int
) -> torch.Tensor:
    """Accept None / int / Tensor for B-spline knots; auto-derive when needed."""
    if isinstance(knots, torch.Tensor):
        return knots
    resolved = _api._resolve_knots(knots, to_numpy_f64(t), label="knots", degree=degree)
    return from_numpy_like(resolved, t)


def _resolve_basis_locations_tensor(
    t: torch.Tensor,
    knots_or_centers: Any,
    *,
    basis_kind: str,
    degree: int,
) -> torch.Tensor:
    """Dispatch :func:`_resolve_centers_tensor`/:func:`_resolve_knots_tensor`."""
    if isinstance(knots_or_centers, torch.Tensor):
        return knots_or_centers
    kind = str(basis_kind).strip().lower().replace("_", "").replace("-", "")
    if kind in {"duchon", "duchonspline"}:
        return _resolve_centers_tensor(t, knots_or_centers)
    return _resolve_knots_tensor(t, knots_or_centers, degree=degree)


class _BsplineBasisFn(torch.autograd.Function):
    """Autograd Function evaluating the Rust B-spline basis with grad wrt ``t``."""

    @staticmethod
    def forward(
        ctx: Any, t: torch.Tensor, knots: torch.Tensor, degree: int, periodic: bool
    ) -> torch.Tensor:
        t_np = to_numpy_f64(t)
        knots_np = to_numpy_f64(knots)
        basis_np = _api.bspline_basis(t_np, knots_np, degree=degree, periodic=periodic)
        ctx.save_for_backward(t, knots)
        ctx.degree = degree
        ctx.periodic = periodic
        return from_numpy_like(basis_np, t)

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        (grad_basis,) = grad_outputs
        t, knots = ctx.saved_tensors
        deriv_np = _api.bspline_basis_derivative(
            to_numpy_f64(t),
            to_numpy_f64(knots),
            degree=ctx.degree,
            order=1,
            periodic=ctx.periodic,
        )
        deriv = from_numpy_like(deriv_np, t)
        grad_t = (grad_basis.to(dtype=deriv.dtype) * deriv).sum(dim=-1)
        return grad_t, None, None, None


class _DuchonBasis1dFn(torch.autograd.Function):
    """Autograd Function evaluating the Rust 1-D Duchon basis with grad wrt ``t``."""

    @staticmethod
    def forward(
        ctx: Any, t: torch.Tensor, centers: torch.Tensor, m: int, periodic: bool
    ) -> torch.Tensor:
        t_np = to_numpy_f64(t)
        centers_np = to_numpy_f64(centers)
        basis_np = _api.duchon_basis_1d(t_np, centers_np, m=m, periodic=periodic)
        ctx.save_for_backward(t, centers)
        ctx.m = m
        ctx.periodic = periodic
        return from_numpy_like(basis_np, t)

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        (grad_basis,) = grad_outputs
        t, centers = ctx.saved_tensors
        deriv_np = _api.duchon_basis_1d_derivative(
            to_numpy_f64(t),
            to_numpy_f64(centers),
            m=ctx.m,
            order=1,
            periodic=ctx.periodic,
        )
        deriv = from_numpy_like(deriv_np, t)
        grad_t = (grad_basis.to(dtype=deriv.dtype) * deriv).sum(dim=-1)
        return grad_t, None, None, None


def bspline_basis(
    t: torch.Tensor,
    knots: Any = None,
    *,
    degree: int = 3,
    periodic: bool = False,
) -> torch.Tensor:
    """Evaluate the B-spline basis at ``t`` and route gradients back to ``t``.

    Parameters
    ----------
    t : torch.Tensor
        Evaluation locations, shape ``(n_t,)``. Differentiable input.
    knots : torch.Tensor | int | None, optional
        Knot vector. ``None`` (default) auto-derives a clamped knot
        vector with quantile-spaced interior knots from ``t``; an ``int``
        ``K`` overrides the interior-knot count. Treated as structural
        either way — no gradient is propagated.
    degree : int, optional
        Spline degree. Default ``3``.
    periodic : bool, optional
        Whether to evaluate the periodic variant. Default ``False``.

    Returns
    -------
    torch.Tensor
        Basis matrix of shape ``(n_t, n_basis)``.
    """
    knots_t = _resolve_knots_tensor(t, knots, degree=int(degree))
    apply = cast(Callable[..., torch.Tensor], _BsplineBasisFn.apply)
    return apply(t, knots_t, int(degree), bool(periodic))


def bspline_basis_derivative(
    t: torch.Tensor,
    knots: Any = None,
    *,
    degree: int = 3,
    order: int = 1,
    periodic: bool = False,
) -> torch.Tensor:
    """Evaluate derivatives of the B-spline basis at ``t``.

    Forward-only: the returned tensor does not carry a backward through ``t``.
    Callers that need a differentiable basis should use ``bspline_basis`` and
    rely on autograd, since the derivative primitive has no analytic VJP in
    :mod:`gamfit._api`.

    Parameters
    ----------
    t : torch.Tensor
        Evaluation locations of shape ``(n_t,)``.
    knots : torch.Tensor
        Knot vector.
    degree : int, optional
        Spline degree. Default ``3``.
    order : int, optional
        Derivative order. Default ``1``.
    periodic : bool, optional
        Whether to evaluate the periodic variant. Default ``False``.

    Returns
    -------
    torch.Tensor
        Derivative basis matrix of shape ``(n_t, n_basis)``.
    """
    knots_t = _resolve_knots_tensor(t, knots, degree=int(degree))
    deriv = _api.bspline_basis_derivative(
        to_numpy_f64(t),
        to_numpy_f64(knots_t),
        degree=int(degree),
        order=int(order),
        periodic=bool(periodic),
    )
    return from_numpy_like(deriv, t)


def duchon_basis_1d(
    t: torch.Tensor,
    centers: Any = None,
    *,
    m: int = 2,
    periodic: bool = False,
) -> torch.Tensor:
    """Evaluate the one-dimensional Duchon basis at ``t`` with grad wrt ``t``.

    Parameters
    ----------
    t : torch.Tensor
        Evaluation locations, shape ``(n_t,)``. Differentiable input.
    centers : torch.Tensor
        Center locations. Treated as structural; no gradient is propagated.
    m : int, optional
        Duchon smoothness order. Default ``2``.
    periodic : bool, optional
        Whether to evaluate the periodic variant. Default ``False``.

    Returns
    -------
    torch.Tensor
        Basis matrix of shape ``(n_t, n_basis)``.
    """
    centers_t = _resolve_centers_tensor(t, centers)
    apply = cast(Callable[..., torch.Tensor], _DuchonBasis1dFn.apply)
    return apply(t, centers_t, int(m), bool(periodic))


def duchon_basis_1d_derivative(
    t: torch.Tensor,
    centers: Any = None,
    *,
    m: int = 2,
    order: int = 1,
    periodic: bool = False,
) -> torch.Tensor:
    """Evaluate derivatives of the 1-D Duchon basis at ``t``.

    Forward-only: the returned tensor does not carry a backward through ``t``.
    Callers needing a differentiable basis should use ``duchon_basis_1d`` and
    rely on autograd.

    Parameters
    ----------
    t : torch.Tensor
        Evaluation locations of shape ``(n_t,)``.
    centers : torch.Tensor
        Center locations.
    m : int, optional
        Duchon smoothness order. Default ``2``.
    order : int, optional
        Derivative order. Default ``1``.
    periodic : bool, optional
        Whether to evaluate the periodic variant. Default ``False``.

    Returns
    -------
    torch.Tensor
        Derivative basis matrix of shape ``(n_t, n_basis)``.
    """
    centers_t = _resolve_centers_tensor(t, centers)
    deriv = _api.duchon_basis_1d_derivative(
        to_numpy_f64(t),
        to_numpy_f64(centers_t),
        m=int(m),
        order=int(order),
        periodic=bool(periodic),
    )
    return from_numpy_like(deriv, t)


def smoothness_penalty(
    knots: torch.Tensor,
    *,
    degree: int = 3,
    order: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the B-spline difference penalty and its null-space basis.

    Forward-only: ``knots`` is structural and the result has no autograd path.

    Parameters
    ----------
    knots : torch.Tensor
        Knot vector.
    degree : int, optional
        Spline degree. Default ``3``.
    order : int, optional
        Difference penalty order. Default ``2``.

    Returns
    -------
    (torch.Tensor, torch.Tensor)
        ``S`` of shape ``(M, M)`` and ``null_basis`` of shape ``(M, p)``.
    """
    s_np, null_np = _api.smoothness_penalty(
        to_numpy_f64(knots), degree=int(degree), order=int(order)
    )
    return from_numpy_like(s_np, knots), from_numpy_like(null_np, knots)


def gaussian_weighted_ridge(
    X: torch.Tensor,
    Y: torch.Tensor,
    penalty: torch.Tensor,
    weights: torch.Tensor,
    *,
    ridge_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form Gaussian row-weighted ridge solve.

    Forward-only: :mod:`gamfit._api` exposes no analytic VJP for this primitive,
    so the returned tensors carry no autograd path. ``weights`` are likelihood
    row weights, not a multiplicative gate on the design row.

    Parameters
    ----------
    X : torch.Tensor
        Design matrix of shape ``(N, M)``.
    Y : torch.Tensor
        Response matrix of shape ``(N, D)``.
    penalty : torch.Tensor
        Penalty matrix of shape ``(M, M)``.
    weights : torch.Tensor
        Row weights of shape ``(N,)``.
    ridge_lambda : float
        Ridge multiplier on ``penalty``.

    Returns
    -------
    (torch.Tensor, torch.Tensor)
        ``coefficients`` of shape ``(M, D)`` and ``fitted`` of shape ``(N, D)``.
    """
    coef_np, fit_np = _api.gaussian_weighted_ridge(
        to_numpy_f64(X),
        to_numpy_f64(Y),
        to_numpy_f64(penalty),
        to_numpy_f64(weights),
        ridge_lambda=float(ridge_lambda),
    )
    return from_numpy_like(coef_np, X), from_numpy_like(fit_np, X)


def gaussian_weighted_ridge_batch(
    X: torch.Tensor,
    Y: torch.Tensor,
    penalty: torch.Tensor,
    weights: torch.Tensor,
    *,
    ridge_lambda: float,
    row_counts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched closed-form Gaussian row-weighted ridge solve.

    Forward-only: no analytic VJP is exposed in :mod:`gamfit._api`. ``X`` has
    shape ``(K, Nmax, M)``, ``Y`` has shape ``(K, Nmax, D)``, ``weights`` has
    shape ``(K, Nmax)``, and ``row_counts`` optionally marks the active row
    prefix per problem in a padded ragged batch.

    Returns
    -------
    (torch.Tensor, torch.Tensor)
        ``coefficients`` of shape ``(K, M, D)`` and ``fitted`` of shape
        ``(K, Nmax, D)``.
    """
    coef_np, fit_np = _api.gaussian_weighted_ridge_batch(
        to_numpy_f64(X),
        to_numpy_f64(Y),
        to_numpy_f64(penalty),
        to_numpy_f64(weights),
        ridge_lambda=float(ridge_lambda),
        row_counts=None if row_counts is None else to_numpy_uintp(row_counts),
    )
    return from_numpy_like(coef_np, X), from_numpy_like(fit_np, X)
