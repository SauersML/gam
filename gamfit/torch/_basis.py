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
from . import _torch_compat as _tc
from ._coerce import from_numpy_like, to_numpy_f64, to_numpy_uintp


def _as_tensor(value: Any) -> Any:
    """Ensure ``value`` is a torch tensor; promote NumPy/list/scalar to one."""
    if isinstance(value, torch.Tensor):
        return value
    return _tc.as_tensor(value, dtype=_tc.float64)


class _BsplineBasisFn(torch.autograd.Function):
    """Autograd Function evaluating the Rust B-spline basis with grad wrt ``t``."""

    @staticmethod
    def forward(ctx: Any, t: Any, knots: Any, degree: int, periodic: bool) -> Any:
        t_np = to_numpy_f64(t)
        knots_np = to_numpy_f64(knots)
        basis_np = _api.bspline_basis(t_np, knots_np, degree=degree, periodic=periodic)
        ctx.save_for_backward(t, knots)
        ctx.degree = degree
        ctx.periodic = periodic
        return from_numpy_like(basis_np, t)

    @staticmethod
    def backward(ctx: Any, grad_basis: Any) -> tuple[Any, None, None, None]:
        t, knots = ctx.saved_tensors
        t_np = to_numpy_f64(t)
        knots_np = to_numpy_f64(knots)
        deriv_np = _api.bspline_basis_derivative(
            t_np,
            knots_np,
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
    def forward(ctx: Any, t: Any, centers: Any, m: int, periodic: bool) -> Any:
        t_np = to_numpy_f64(t)
        centers_np = to_numpy_f64(centers)
        basis_np = _api.duchon_basis_1d(t_np, centers_np, m=m, periodic=periodic)
        ctx.save_for_backward(t, centers)
        ctx.m = m
        ctx.periodic = periodic
        return from_numpy_like(basis_np, t)

    @staticmethod
    def backward(ctx: Any, grad_basis: Any) -> tuple[Any, None, None, None]:
        t, centers = ctx.saved_tensors
        t_np = to_numpy_f64(t)
        centers_np = to_numpy_f64(centers)
        deriv_np = _api.duchon_basis_1d_derivative(
            t_np,
            centers_np,
            m=ctx.m,
            order=1,
            periodic=ctx.periodic,
        )
        deriv = from_numpy_like(deriv_np, t)
        grad_t = (grad_basis.to(dtype=deriv.dtype) * deriv).sum(dim=-1)
        return grad_t, None, None, None


def bspline_basis(
    t: Any,
    knots: Any,
    *,
    degree: int = 3,
    periodic: bool = False,
) -> Any:
    """Evaluate the B-spline basis at ``t`` and route gradients back to ``t``.

    Parameters
    ----------
    t : torch.Tensor
        Evaluation locations, shape ``(n_t,)``. Differentiable input.
    knots : torch.Tensor
        Knot vector. Treated as structural; no gradient is propagated.
    degree : int, optional
        Spline degree. Default ``3``.
    periodic : bool, optional
        Whether to evaluate the periodic variant. Default ``False``.

    Returns
    -------
    torch.Tensor
        Basis matrix of shape ``(n_t, n_basis)``.
    """
    apply = cast(Callable[..., Any], _BsplineBasisFn.apply)
    return apply(_as_tensor(t), _as_tensor(knots), int(degree), bool(periodic))


def bspline_basis_derivative(
    t: Any,
    knots: Any,
    *,
    degree: int = 3,
    order: int = 1,
    periodic: bool = False,
) -> Any:
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
    t_ref = _as_tensor(t)
    deriv = _api.bspline_basis_derivative(
        to_numpy_f64(t_ref),
        to_numpy_f64(_as_tensor(knots)),
        degree=int(degree),
        order=int(order),
        periodic=bool(periodic),
    )
    return from_numpy_like(deriv, t_ref)


def duchon_basis_1d(
    t: Any,
    centers: Any,
    *,
    m: int = 2,
    periodic: bool = False,
) -> Any:
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
    apply = cast(Callable[..., Any], _DuchonBasis1dFn.apply)
    return apply(_as_tensor(t), _as_tensor(centers), int(m), bool(periodic))


def duchon_basis_1d_derivative(
    t: Any,
    centers: Any,
    *,
    m: int = 2,
    order: int = 1,
    periodic: bool = False,
) -> Any:
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
    t_ref = _as_tensor(t)
    deriv = _api.duchon_basis_1d_derivative(
        to_numpy_f64(t_ref),
        to_numpy_f64(_as_tensor(centers)),
        m=int(m),
        order=int(order),
        periodic=bool(periodic),
    )
    return from_numpy_like(deriv, t_ref)


def smoothness_penalty(
    knots: Any,
    *,
    degree: int = 3,
    order: int = 2,
) -> tuple[Any, Any]:
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
    ref = _as_tensor(knots)
    s_np, null_np = _api.smoothness_penalty(
        to_numpy_f64(ref), degree=int(degree), order=int(order)
    )
    return from_numpy_like(s_np, ref), from_numpy_like(null_np, ref)


def gaussian_weighted_ridge(
    X: Any,
    Y: Any,
    penalty: Any,
    weights: Any,
    *,
    ridge_lambda: float,
) -> tuple[Any, Any]:
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
    X_ref = _as_tensor(X)
    coef_np, fit_np = _api.gaussian_weighted_ridge(
        to_numpy_f64(X_ref),
        to_numpy_f64(_as_tensor(Y)),
        to_numpy_f64(_as_tensor(penalty)),
        to_numpy_f64(_as_tensor(weights)),
        ridge_lambda=float(ridge_lambda),
    )
    return from_numpy_like(coef_np, X_ref), from_numpy_like(fit_np, X_ref)


def gaussian_weighted_ridge_batch(
    X: Any,
    Y: Any,
    penalty: Any,
    weights: Any,
    *,
    ridge_lambda: float,
    row_counts: Any | None = None,
) -> tuple[Any, Any]:
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
    X_ref = _as_tensor(X)
    coef_np, fit_np = _api.gaussian_weighted_ridge_batch(
        to_numpy_f64(X_ref),
        to_numpy_f64(_as_tensor(Y)),
        to_numpy_f64(_as_tensor(penalty)),
        to_numpy_f64(_as_tensor(weights)),
        ridge_lambda=float(ridge_lambda),
        row_counts=None if row_counts is None else to_numpy_uintp(row_counts),
    )
    return from_numpy_like(coef_np, X_ref), from_numpy_like(fit_np, X_ref)
