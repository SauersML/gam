"""Closed-form Gaussian REML autograd wrappers for the gamfit torch bridge.

The Rust engine ships analytic VJPs for every closed-form Gaussian REML
primitive in :mod:`gamfit._api`. This module is the canonical PyTorch face
of those primitives. Each public wrapper accepts torch tensors, runs the
forward in Rust on f64 CPU, and routes upstream gradients into the matching
Rust backward so the result composes inside larger torch graphs.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import torch

from gamfit import _api as _np_api

from ._coerce import from_numpy_like, to_numpy_f64, to_numpy_uintp


class GaussianRemlOutput(NamedTuple):
    """Forward outputs shared by every Gaussian REML wrapper."""

    coefficients: torch.Tensor
    fitted: torch.Tensor
    lam: torch.Tensor
    reml_score: torch.Tensor


def _scalar_grad(grad: torch.Tensor | None) -> float:
    if grad is None:
        return 0.0
    return float(grad.detach())


def _batch_grad(grad: torch.Tensor | None) -> Any:
    if grad is None:
        return None
    return to_numpy_f64(grad)


def _wrap_optional(arr: Any, ref: torch.Tensor) -> torch.Tensor | None:
    if arr is None:
        return None
    return from_numpy_like(np.asarray(arr, dtype=np.float64), ref)


def _copy_forward_state(out: dict[str, Any]) -> dict[str, Any]:
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in out.items()}


def _save_diff_tensors(ctx: Any, *tensors: Any) -> None:
    """Save tensor inputs so PyTorch version-tracks them and raises on in-place
    mutation between forward and backward. ``None`` values are skipped silently;
    callers retain their own presence flags."""
    ctx.save_for_backward(*(t for t in tensors if isinstance(t, torch.Tensor)))


def _check_saved_versions(ctx: Any) -> None:
    """Touch ``ctx.saved_tensors`` so PyTorch raises if any saved input was
    modified in-place between forward and backward."""
    _ = ctx.saved_tensors


class _GaussianRemlFitFn(torch.autograd.Function):
    """Autograd Function for the non-batched closed-form Gaussian REML fit."""

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        y: torch.Tensor,
        penalty: torch.Tensor,
        weights: torch.Tensor | None,
        by: torch.Tensor | None,
        init_lambda: float | None,
        by_start_col: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_np = to_numpy_f64(x)
        y_np = to_numpy_f64(y)
        penalty_np = to_numpy_f64(penalty)
        weights_np = None if weights is None else to_numpy_f64(weights)
        by_np = None if by is None else to_numpy_f64(by)

        out = _np_api.gaussian_reml_fit(
            x_np,
            y_np,
            penalty_np,
            weights=weights_np,
            init_lambda=init_lambda,
            by=by_np,
            by_start_col=by_start_col,
        )

        # Save the differentiable input tensors so PyTorch can version-check
        # them and raise on any in-place mutation between forward and backward.
        # Keep the numpy aliases on ctx for performance — they are only read in
        # backward, which is gated by the version check.
        _save_diff_tensors(ctx, x, y, weights, by)
        ctx.x_np = x_np
        ctx.y_np = y_np
        ctx.penalty_np = penalty_np
        ctx.weights_np = weights_np
        ctx.by_np = by_np
        ctx.init_lambda = init_lambda
        ctx.by_start_col = by_start_col
        ctx.forward_state = _copy_forward_state(out)
        ctx.ref = x

        coefficients = from_numpy_like(out["coefficients"], x)
        fitted = from_numpy_like(out["fitted"], x)
        lam = from_numpy_like(out["lambda"], x)
        reml_score = from_numpy_like(out["reml_score"], x)
        return coefficients, fitted, lam, reml_score

    @staticmethod
    def backward(
        ctx: Any,
        grad_coefficients: torch.Tensor | None,
        grad_fitted: torch.Tensor | None,
        grad_lam: torch.Tensor | None,
        grad_reml_score: torch.Tensor | None,
    ) -> tuple[Any, ...]:
        _check_saved_versions(ctx)
        grad_coef_np = None if grad_coefficients is None else to_numpy_f64(grad_coefficients)
        grad_fitted_np = None if grad_fitted is None else to_numpy_f64(grad_fitted)
        grad_lambda_scalar = _scalar_grad(grad_lam)
        grad_reml_scalar = _scalar_grad(grad_reml_score)

        result = _np_api.gaussian_reml_fit_backward(
            ctx.x_np,
            ctx.y_np,
            ctx.penalty_np,
            grad_lambda=grad_lambda_scalar,
            grad_coefficients=grad_coef_np,
            grad_fitted=grad_fitted_np,
            grad_reml_score=grad_reml_scalar,
            forward_state=ctx.forward_state,
            weights=ctx.weights_np,
            init_lambda=ctx.init_lambda,
            by=ctx.by_np,
            by_start_col=ctx.by_start_col,
        )

        ref = ctx.ref
        grad_x = _wrap_optional(result.get("grad_x"), ref)
        grad_y = _wrap_optional(result.get("grad_y"), ref)
        grad_weights = _wrap_optional(result.get("grad_weights"), ref) if ctx.has_weights else None
        grad_by = _wrap_optional(result.get("grad_by"), ref) if ctx.has_by else None
        # Order matches forward(...) positional args: x, y, penalty, weights, by,
        # init_lambda, by_start_col.
        return grad_x, grad_y, None, grad_weights, grad_by, None, None


class _GaussianRemlFitBatchedFn(torch.autograd.Function):
    """Autograd Function for the ragged batched closed-form Gaussian REML fit."""

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        y: torch.Tensor,
        row_offsets: torch.Tensor,
        penalty: torch.Tensor,
        weights: torch.Tensor | None,
        by: torch.Tensor | None,
        init_lambda: float | None,
        by_start_col: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_np = to_numpy_f64(x)
        y_np = to_numpy_f64(y)
        offsets_np = to_numpy_uintp(row_offsets)
        penalty_np = to_numpy_f64(penalty)
        weights_np = None if weights is None else to_numpy_f64(weights)
        by_np = None if by is None else to_numpy_f64(by)

        out = _np_api.gaussian_reml_fit_batched(
            x_np,
            y_np,
            offsets_np,
            penalty_np,
            weights=weights_np,
            init_lambda=init_lambda,
            by=by_np,
            by_start_col=by_start_col,
        )

        ctx.x_np = x_np
        ctx.y_np = y_np
        ctx.offsets_np = offsets_np
        ctx.penalty_np = penalty_np
        ctx.weights_np = weights_np
        ctx.by_np = by_np
        ctx.init_lambda = init_lambda
        ctx.by_start_col = by_start_col
        ctx.forward_state = _copy_forward_state(out)
        ctx.has_weights = weights is not None
        ctx.has_by = by is not None
        ctx.ref = x

        coefficients = from_numpy_like(out["coefficients"], x)
        fitted = from_numpy_like(out["fitted"], x)
        lam = from_numpy_like(out["lambda"], x)
        reml_score = from_numpy_like(out["reml_score"], x)
        return coefficients, fitted, lam, reml_score

    @staticmethod
    def backward(
        ctx: Any,
        grad_coefficients: torch.Tensor | None,
        grad_fitted: torch.Tensor | None,
        grad_lam: torch.Tensor | None,
        grad_reml_score: torch.Tensor | None,
    ) -> tuple[Any, ...]:
        grad_coef_np = None if grad_coefficients is None else to_numpy_f64(grad_coefficients)
        grad_fitted_np = None if grad_fitted is None else to_numpy_f64(grad_fitted)
        grad_lambda_vec = _batch_grad(grad_lam)
        grad_reml_vec = _batch_grad(grad_reml_score)

        result = _np_api.gaussian_reml_fit_batched_backward(
            ctx.x_np,
            ctx.y_np,
            ctx.offsets_np,
            ctx.penalty_np,
            grad_lambda=grad_lambda_vec,
            grad_coefficients=grad_coef_np,
            grad_fitted=grad_fitted_np,
            grad_reml_score=grad_reml_vec,
            forward_state=ctx.forward_state,
            weights=ctx.weights_np,
            init_lambda=ctx.init_lambda,
            by=ctx.by_np,
            by_start_col=ctx.by_start_col,
        )

        ref = ctx.ref
        grad_x = _wrap_optional(result.get("grad_x"), ref)
        grad_y = _wrap_optional(result.get("grad_y"), ref)
        grad_weights = _wrap_optional(result.get("grad_weights"), ref) if ctx.has_weights else None
        grad_by = _wrap_optional(result.get("grad_by"), ref) if ctx.has_by else None
        # Order matches forward(...) positional args: x, y, row_offsets, penalty,
        # weights, by, init_lambda, by_start_col.
        return grad_x, grad_y, None, None, grad_weights, grad_by, None, None


class _GaussianRemlFitPositionsFn(torch.autograd.Function):
    """Autograd Function for the position-based closed-form Gaussian REML fit."""

    @staticmethod
    def forward(
        ctx: Any,
        t: torch.Tensor,
        y: torch.Tensor,
        knots_or_centers: torch.Tensor,
        penalty: torch.Tensor,
        weights: torch.Tensor | None,
        by: torch.Tensor | None,
        basis_kind: str,
        basis_order: int | None,
        periodic: bool,
        period: float | None,
        init_lambda: float | None,
        by_start_col: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t_np = to_numpy_f64(t)
        y_np = to_numpy_f64(y)
        knots_np = to_numpy_f64(knots_or_centers)
        penalty_np = to_numpy_f64(penalty)
        weights_np = None if weights is None else to_numpy_f64(weights)
        by_np = None if by is None else to_numpy_f64(by)

        out = _np_api.gaussian_reml_fit_positions(
            t_np,
            y_np,
            basis_kind,
            knots_np,
            penalty_np,
            basis_order=basis_order,
            periodic=periodic,
            period=period,
            weights=weights_np,
            init_lambda=init_lambda,
            by=by_np,
            by_start_col=by_start_col,
        )

        ctx.t_np = t_np
        ctx.y_np = y_np
        ctx.knots_np = knots_np
        ctx.penalty_np = penalty_np
        ctx.weights_np = weights_np
        ctx.by_np = by_np
        ctx.basis_kind = basis_kind
        ctx.basis_order = basis_order
        ctx.periodic = periodic
        ctx.period = period
        ctx.init_lambda = init_lambda
        ctx.by_start_col = by_start_col
        ctx.forward_state = _copy_forward_state(out)
        ctx.has_weights = weights is not None
        ctx.has_by = by is not None
        ctx.ref = t

        coefficients = from_numpy_like(out["coefficients"], t)
        fitted = from_numpy_like(out["fitted"], t)
        lam = from_numpy_like(out["lambda"], t)
        reml_score = from_numpy_like(out["reml_score"], t)
        return coefficients, fitted, lam, reml_score

    @staticmethod
    def backward(
        ctx: Any,
        grad_coefficients: torch.Tensor | None,
        grad_fitted: torch.Tensor | None,
        grad_lam: torch.Tensor | None,
        grad_reml_score: torch.Tensor | None,
    ) -> tuple[Any, ...]:
        grad_coef_np = None if grad_coefficients is None else to_numpy_f64(grad_coefficients)
        grad_fitted_np = None if grad_fitted is None else to_numpy_f64(grad_fitted)
        grad_lambda_scalar = _scalar_grad(grad_lam)
        grad_reml_scalar = _scalar_grad(grad_reml_score)

        result = _np_api.gaussian_reml_fit_positions_backward(
            ctx.t_np,
            ctx.y_np,
            ctx.basis_kind,
            ctx.knots_np,
            ctx.penalty_np,
            grad_lambda=grad_lambda_scalar,
            grad_coefficients=grad_coef_np,
            grad_fitted=grad_fitted_np,
            grad_reml_score=grad_reml_scalar,
            forward_state=ctx.forward_state,
            basis_order=ctx.basis_order,
            periodic=ctx.periodic,
            period=ctx.period,
            weights=ctx.weights_np,
            init_lambda=ctx.init_lambda,
            by=ctx.by_np,
            by_start_col=ctx.by_start_col,
        )

        ref = ctx.ref
        grad_t = _wrap_optional(result.get("grad_t"), ref)
        grad_y = _wrap_optional(result.get("grad_y"), ref)
        grad_weights = _wrap_optional(result.get("grad_weights"), ref) if ctx.has_weights else None
        grad_by = _wrap_optional(result.get("grad_by"), ref) if ctx.has_by else None
        # Order matches forward(...): t, y, knots_or_centers, penalty, weights, by,
        # basis_kind, basis_order, periodic, period, init_lambda, by_start_col.
        return grad_t, grad_y, None, None, grad_weights, grad_by, None, None, None, None, None, None


class _GaussianRemlFitPositionsBatchedFn(torch.autograd.Function):
    """Autograd Function for the ragged batched position-based REML fit."""

    @staticmethod
    def forward(
        ctx: Any,
        t: torch.Tensor,
        y: torch.Tensor,
        row_offsets: torch.Tensor,
        knots_or_centers: torch.Tensor,
        penalty: torch.Tensor,
        weights: torch.Tensor | None,
        by: torch.Tensor | None,
        basis_kind: str,
        basis_order: int | None,
        periodic: bool,
        period: float | None,
        init_lambda: float | None,
        by_start_col: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t_np = to_numpy_f64(t)
        y_np = to_numpy_f64(y)
        offsets_np = to_numpy_uintp(row_offsets)
        knots_np = to_numpy_f64(knots_or_centers)
        penalty_np = to_numpy_f64(penalty)
        weights_np = None if weights is None else to_numpy_f64(weights)
        by_np = None if by is None else to_numpy_f64(by)

        out = _np_api.gaussian_reml_fit_positions_batched(
            t_np,
            y_np,
            offsets_np,
            basis_kind,
            knots_np,
            penalty_np,
            basis_order=basis_order,
            periodic=periodic,
            period=period,
            weights=weights_np,
            init_lambda=init_lambda,
            by=by_np,
            by_start_col=by_start_col,
        )

        ctx.t_np = t_np
        ctx.y_np = y_np
        ctx.offsets_np = offsets_np
        ctx.knots_np = knots_np
        ctx.penalty_np = penalty_np
        ctx.weights_np = weights_np
        ctx.by_np = by_np
        ctx.basis_kind = basis_kind
        ctx.basis_order = basis_order
        ctx.periodic = periodic
        ctx.period = period
        ctx.init_lambda = init_lambda
        ctx.by_start_col = by_start_col
        ctx.forward_state = _copy_forward_state(out)
        ctx.has_weights = weights is not None
        ctx.has_by = by is not None
        ctx.ref = t

        coefficients = from_numpy_like(out["coefficients"], t)
        fitted = from_numpy_like(out["fitted"], t)
        lam = from_numpy_like(out["lambda"], t)
        reml_score = from_numpy_like(out["reml_score"], t)
        return coefficients, fitted, lam, reml_score

    @staticmethod
    def backward(
        ctx: Any,
        grad_coefficients: torch.Tensor | None,
        grad_fitted: torch.Tensor | None,
        grad_lam: torch.Tensor | None,
        grad_reml_score: torch.Tensor | None,
    ) -> tuple[Any, ...]:
        grad_coef_np = None if grad_coefficients is None else to_numpy_f64(grad_coefficients)
        grad_fitted_np = None if grad_fitted is None else to_numpy_f64(grad_fitted)
        grad_lambda_vec = _batch_grad(grad_lam)
        grad_reml_vec = _batch_grad(grad_reml_score)

        result = _np_api.gaussian_reml_fit_positions_batched_backward(
            ctx.t_np,
            ctx.y_np,
            ctx.offsets_np,
            ctx.basis_kind,
            ctx.knots_np,
            ctx.penalty_np,
            grad_lambda=grad_lambda_vec,
            grad_coefficients=grad_coef_np,
            grad_fitted=grad_fitted_np,
            grad_reml_score=grad_reml_vec,
            forward_state=ctx.forward_state,
            basis_order=ctx.basis_order,
            periodic=ctx.periodic,
            period=ctx.period,
            weights=ctx.weights_np,
            init_lambda=ctx.init_lambda,
            by=ctx.by_np,
            by_start_col=ctx.by_start_col,
        )

        ref = ctx.ref
        grad_t = _wrap_optional(result.get("grad_t"), ref)
        grad_y = _wrap_optional(result.get("grad_y"), ref)
        grad_weights = _wrap_optional(result.get("grad_weights"), ref) if ctx.has_weights else None
        grad_by = _wrap_optional(result.get("grad_by"), ref) if ctx.has_by else None
        # Order matches forward(...): t, y, row_offsets, knots_or_centers, penalty,
        # weights, by, basis_kind, basis_order, periodic, period, init_lambda,
        # by_start_col.
        return (
            grad_t,
            grad_y,
            None,
            None,
            None,
            grad_weights,
            grad_by,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def gaussian_reml_fit(
    x: torch.Tensor,
    y: torch.Tensor,
    penalty: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    init_lambda: float | None = None,
    by: torch.Tensor | None = None,
    by_start_col: int = 0,
) -> GaussianRemlOutput:
    """Differentiable closed-form Gaussian REML fit.

    Parameters
    ----------
    x:
        Design matrix of shape ``(N, M)``. A 1D input is promoted to
        ``(N, 1)`` to match the numpy contract.
    y:
        Response matrix of shape ``(N, D)``.
    penalty:
        Penalty matrix of shape ``(M, M)``. Treated as structural;
        backward returns ``None`` for this slot.
    weights:
        Optional row weights of shape ``(N,)``.
    init_lambda:
        Optional initial smoothing parameter passed through to the engine.
    by:
        Optional ``by`` vector of shape ``(N,)``.
    by_start_col:
        First design column the ``by`` modulation applies to.

    Returns
    -------
    GaussianRemlOutput
        Tuple of ``(coefficients, fitted, lam, reml_score)`` as torch tensors
        sharing ``x``'s device and dtype.
    """
    coefficients, fitted, lam, reml_score = _GaussianRemlFitFn.apply(
        x, y, penalty, weights, by, init_lambda, by_start_col
    )
    return GaussianRemlOutput(coefficients, fitted, lam, reml_score)


def gaussian_reml_fit_batched(
    x: torch.Tensor,
    y: torch.Tensor,
    row_offsets: torch.Tensor,
    penalty: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    init_lambda: float | None = None,
    by: torch.Tensor | None = None,
    by_start_col: int = 0,
) -> GaussianRemlOutput:
    """Differentiable ragged batched closed-form Gaussian REML fit.

    Parameters
    ----------
    x:
        Stacked design matrix of shape ``(N_total, M)``.
    y:
        Stacked responses of shape ``(N_total, D)``.
    row_offsets:
        ``uintp`` 1D tensor of length ``K + 1`` marking the row boundaries
        of each problem in the packed batch.
    penalty:
        Penalty matrix of shape ``(M, M)``.
    weights, by:
        Optional 1D tensors of length ``N_total``.
    init_lambda, by_start_col:
        Forwarded to the engine.

    Returns
    -------
    GaussianRemlOutput
        ``lam`` and ``reml_score`` are length-``K`` tensors; ``coefficients``
        and ``fitted`` follow the engine's packed layout.
    """
    coefficients, fitted, lam, reml_score = _GaussianRemlFitBatchedFn.apply(
        x, y, row_offsets, penalty, weights, by, init_lambda, by_start_col
    )
    return GaussianRemlOutput(coefficients, fitted, lam, reml_score)


def gaussian_reml_fit_positions(
    t: torch.Tensor,
    y: torch.Tensor,
    basis_kind: str,
    knots_or_centers: torch.Tensor,
    penalty: torch.Tensor,
    *,
    basis_order: int | None = None,
    periodic: bool = False,
    period: float | None = None,
    weights: torch.Tensor | None = None,
    init_lambda: float | None = None,
    by: torch.Tensor | None = None,
    by_start_col: int = 0,
) -> GaussianRemlOutput:
    """Differentiable position-based closed-form Gaussian REML fit.

    Parameters
    ----------
    t:
        1D tensor of evaluation positions of length ``N``.
    y:
        Response matrix of shape ``(N, D)``.
    basis_kind:
        Internal basis identifier (e.g. ``"bspline"``, ``"duchon"``).
    knots_or_centers:
        1D tensor of basis knots (B-spline) or centers (Duchon).
    penalty:
        Penalty matrix of shape ``(M, M)``.
    basis_order:
        Optional override for the basis order; defaults from ``basis_kind``.
    periodic, period:
        Periodic-basis toggle and period length.
    weights, by, init_lambda, by_start_col:
        See :func:`gaussian_reml_fit`.
    """
    coefficients, fitted, lam, reml_score = _GaussianRemlFitPositionsFn.apply(
        t,
        y,
        knots_or_centers,
        penalty,
        weights,
        by,
        basis_kind,
        basis_order,
        periodic,
        period,
        init_lambda,
        by_start_col,
    )
    return GaussianRemlOutput(coefficients, fitted, lam, reml_score)


def gaussian_reml_fit_positions_batched(
    t: torch.Tensor,
    y: torch.Tensor,
    row_offsets: torch.Tensor,
    basis_kind: str,
    knots_or_centers: torch.Tensor,
    penalty: torch.Tensor,
    *,
    basis_order: int | None = None,
    periodic: bool = False,
    period: float | None = None,
    weights: torch.Tensor | None = None,
    init_lambda: float | None = None,
    by: torch.Tensor | None = None,
    by_start_col: int = 0,
) -> GaussianRemlOutput:
    """Differentiable ragged batched position-based Gaussian REML fit.

    Parameters
    ----------
    t:
        1D tensor of positions of length ``N_total``.
    y:
        Response matrix of shape ``(N_total, D)``.
    row_offsets:
        ``uintp`` 1D tensor of length ``K + 1``.
    basis_kind, knots_or_centers, penalty, basis_order, periodic, period:
        Internal basis configuration; see
        :func:`gaussian_reml_fit_positions`.
    weights, by, init_lambda, by_start_col:
        See :func:`gaussian_reml_fit`.
    """
    coefficients, fitted, lam, reml_score = _GaussianRemlFitPositionsBatchedFn.apply(
        t,
        y,
        row_offsets,
        knots_or_centers,
        penalty,
        weights,
        by,
        basis_kind,
        basis_order,
        periodic,
        period,
        init_lambda,
        by_start_col,
    )
    return GaussianRemlOutput(coefficients, fitted, lam, reml_score)
