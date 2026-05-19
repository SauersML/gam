"""Closed-form Gaussian REML autograd wrappers for the gamfit torch bridge.

The Rust engine ships analytic VJPs for every closed-form Gaussian REML
primitive in :mod:`gamfit._api`. This module is the canonical PyTorch face
of those primitives. Each public wrapper accepts torch tensors, runs the
forward in Rust on f64 CPU, and routes upstream gradients into the matching
Rust backward so the result composes inside larger torch graphs.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, cast

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
    edf: torch.Tensor


class GaussianRemlPositionOutput(NamedTuple):
    """Forward outputs for position-based REML wrappers.

    Adds the resolved basis state to :class:`GaussianRemlOutput` so the
    caller can replay the exact same basis at predict time without having
    to thread or recompute ``knots_or_centers``. Pass these fields back
    into :func:`gamfit.torch.duchon_basis_1d` /
    :func:`gamfit.torch.bspline_basis` and the basis is guaranteed to
    match the one used during the fit.
    """

    coefficients: torch.Tensor
    fitted: torch.Tensor
    lam: torch.Tensor
    reml_score: torch.Tensor
    edf: torch.Tensor
    knots_or_centers: torch.Tensor
    penalty: torch.Tensor
    basis_kind: str
    basis_order: int
    periodic: bool
    period: float | None

    def freeze(self) -> "FrozenPositionPredictor":
        """Return a cached predictor using this fit's resolved basis state."""
        return FrozenPositionPredictor(
            coefficients=self.coefficients.detach(),
            knots_or_centers=self.knots_or_centers.detach(),
            basis_kind=self.basis_kind,
            basis_order=self.basis_order,
            periodic=self.periodic,
            period=self.period,
        )


class FrozenPositionPredictor(NamedTuple):
    """Cached position-basis predictor for inference."""

    coefficients: torch.Tensor
    knots_or_centers: torch.Tensor
    basis_kind: str
    basis_order: int
    periodic: bool
    period: float | None

    def evaluate(self, t_new: torch.Tensor) -> torch.Tensor:
        from ._basis import bspline_basis, duchon_basis_1d

        kind = self.basis_kind.strip().lower().replace("_", "").replace("-", "")
        # 1D thin-plate and duchon_multipenalty share Duchon's basis evaluation;
        # only the penalty differs. So they evaluate through the same Duchon path.
        if kind in {"duchon", "duchonspline", "duchonmultipenalty",
                    "duchontripleoperator", "thinplate", "thinplatespline", "tps"}:
            m = 2 if kind in {"thinplate", "thinplatespline", "tps"} else int(self.basis_order)
            basis = duchon_basis_1d(
                t_new,
                self.knots_or_centers.to(device=t_new.device, dtype=t_new.dtype),
                m=m,
                periodic=self.periodic,
            )
        elif kind in {"bspline", "spline"}:
            basis = bspline_basis(
                t_new,
                self.knots_or_centers.to(device=t_new.device, dtype=t_new.dtype),
                degree=self.basis_order,
                periodic=self.periodic,
            )
        else:
            raise ValueError(f"unsupported frozen position basis {self.basis_kind!r}")
        coefficients = self.coefficients.to(device=t_new.device, dtype=t_new.dtype)
        if coefficients.ndim == 3:
            coefficients = coefficients[0]
        return basis @ coefficients


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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        _save_diff_tensors(ctx, x, y, penalty, weights, by)
        ctx.x_np = x_np
        ctx.y_np = y_np
        ctx.penalty_np = penalty_np
        ctx.weights_np = weights_np
        ctx.by_np = by_np
        ctx.init_lambda = init_lambda
        ctx.by_start_col = by_start_col
        ctx.forward_state = _copy_forward_state(out)
        ctx.has_weights = weights is not None
        ctx.has_by = by is not None
        ctx.ref = x
        ctx.penalty_ref = penalty

        coefficients = from_numpy_like(out["coefficients"], x)
        fitted = from_numpy_like(out["fitted"], x)
        lam = from_numpy_like(out["lambda"], x)
        reml_score = from_numpy_like(out["reml_score"], x)
        edf = from_numpy_like(out["edf"], x)
        return coefficients, fitted, lam, reml_score, edf

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[Any, ...]:
        grad_coefficients, grad_fitted, grad_lam, grad_reml_score, grad_edf = grad_outputs
        _check_saved_versions(ctx)
        grad_coef_np = None if grad_coefficients is None else to_numpy_f64(grad_coefficients)
        grad_fitted_np = None if grad_fitted is None else to_numpy_f64(grad_fitted)
        grad_lambda_scalar = _scalar_grad(grad_lam)
        grad_reml_scalar = _scalar_grad(grad_reml_score)
        grad_edf_scalar = _scalar_grad(grad_edf)

        result = _np_api.gaussian_reml_fit_backward(
            ctx.x_np,
            ctx.y_np,
            ctx.penalty_np,
            grad_lambda=grad_lambda_scalar,
            grad_coefficients=grad_coef_np,
            grad_fitted=grad_fitted_np,
            grad_reml_score=grad_reml_scalar,
            grad_edf=grad_edf_scalar,
            forward_state=ctx.forward_state,
            weights=ctx.weights_np,
            init_lambda=ctx.init_lambda,
            by=ctx.by_np,
            by_start_col=ctx.by_start_col,
        )

        ref = ctx.ref
        grad_x = _wrap_optional(result.get("grad_x"), ref)
        grad_y = _wrap_optional(result.get("grad_y"), ref)
        grad_penalty = _wrap_optional(result.get("grad_penalty"), ctx.penalty_ref)
        grad_weights = _wrap_optional(result.get("grad_weights"), ref) if ctx.has_weights else None
        grad_by = _wrap_optional(result.get("grad_by"), ref) if ctx.has_by else None
        # Order matches forward(...) positional args: x, y, penalty, weights, by,
        # init_lambda, by_start_col.
        return grad_x, grad_y, grad_penalty, grad_weights, grad_by, None, None


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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        _save_diff_tensors(ctx, x, y, penalty, weights, by)
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
        ctx.penalty_ref = penalty

        coefficients = from_numpy_like(out["coefficients"], x)
        fitted = from_numpy_like(out["fitted"], x)
        lam = from_numpy_like(out["lambda"], x)
        reml_score = from_numpy_like(out["reml_score"], x)
        edf = from_numpy_like(out["edf"], x)
        return coefficients, fitted, lam, reml_score, edf

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[Any, ...]:
        grad_coefficients, grad_fitted, grad_lam, grad_reml_score, grad_edf = grad_outputs
        _check_saved_versions(ctx)
        grad_coef_np = None if grad_coefficients is None else to_numpy_f64(grad_coefficients)
        grad_fitted_np = None if grad_fitted is None else to_numpy_f64(grad_fitted)
        grad_lambda_vec = _batch_grad(grad_lam)
        grad_reml_vec = _batch_grad(grad_reml_score)
        grad_edf_vec = _batch_grad(grad_edf)

        result = _np_api.gaussian_reml_fit_batched_backward(
            ctx.x_np,
            ctx.y_np,
            ctx.offsets_np,
            ctx.penalty_np,
            grad_lambda=grad_lambda_vec,
            grad_coefficients=grad_coef_np,
            grad_fitted=grad_fitted_np,
            grad_reml_score=grad_reml_vec,
            grad_edf=grad_edf_vec,
            forward_state=ctx.forward_state,
            weights=ctx.weights_np,
            init_lambda=ctx.init_lambda,
            by=ctx.by_np,
            by_start_col=ctx.by_start_col,
        )

        ref = ctx.ref
        grad_x = _wrap_optional(result.get("grad_x"), ref)
        grad_y = _wrap_optional(result.get("grad_y"), ref)
        grad_penalty = _wrap_optional(result.get("grad_penalty"), ctx.penalty_ref)
        grad_weights = _wrap_optional(result.get("grad_weights"), ref) if ctx.has_weights else None
        grad_by = _wrap_optional(result.get("grad_by"), ref) if ctx.has_by else None
        # Order matches forward(...) positional args: x, y, row_offsets, penalty,
        # weights, by, init_lambda, by_start_col.
        return grad_x, grad_y, None, grad_penalty, grad_weights, grad_by, None, None


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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        _save_diff_tensors(ctx, t, y, penalty, weights, by)
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
        ctx.penalty_ref = penalty

        coefficients = from_numpy_like(out["coefficients"], t)
        fitted = from_numpy_like(out["fitted"], t)
        lam = from_numpy_like(out["lambda"], t)
        reml_score = from_numpy_like(out["reml_score"], t)
        edf = from_numpy_like(out["edf"], t)
        return coefficients, fitted, lam, reml_score, edf

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[Any, ...]:
        grad_coefficients, grad_fitted, grad_lam, grad_reml_score, grad_edf = grad_outputs
        _check_saved_versions(ctx)
        grad_coef_np = None if grad_coefficients is None else to_numpy_f64(grad_coefficients)
        grad_fitted_np = None if grad_fitted is None else to_numpy_f64(grad_fitted)
        grad_lambda_scalar = _scalar_grad(grad_lam)
        grad_reml_scalar = _scalar_grad(grad_reml_score)
        grad_edf_scalar = _scalar_grad(grad_edf)

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
            grad_edf=grad_edf_scalar,
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
        grad_penalty = _wrap_optional(result.get("grad_penalty"), ctx.penalty_ref)
        grad_weights = _wrap_optional(result.get("grad_weights"), ref) if ctx.has_weights else None
        grad_by = _wrap_optional(result.get("grad_by"), ref) if ctx.has_by else None
        # Order matches forward(...): t, y, knots_or_centers, penalty, weights, by,
        # basis_kind, basis_order, periodic, period, init_lambda, by_start_col.
        return grad_t, grad_y, None, grad_penalty, grad_weights, grad_by, None, None, None, None, None, None


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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        _save_diff_tensors(ctx, t, y, penalty, weights, by)
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
        ctx.penalty_ref = penalty

        coefficients = from_numpy_like(out["coefficients"], t)
        fitted = from_numpy_like(out["fitted"], t)
        lam = from_numpy_like(out["lambda"], t)
        reml_score = from_numpy_like(out["reml_score"], t)
        edf = from_numpy_like(out["edf"], t)
        return coefficients, fitted, lam, reml_score, edf

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[Any, ...]:
        grad_coefficients, grad_fitted, grad_lam, grad_reml_score, grad_edf = grad_outputs
        _check_saved_versions(ctx)
        grad_coef_np = None if grad_coefficients is None else to_numpy_f64(grad_coefficients)
        grad_fitted_np = None if grad_fitted is None else to_numpy_f64(grad_fitted)
        grad_lambda_vec = _batch_grad(grad_lam)
        grad_reml_vec = _batch_grad(grad_reml_score)
        grad_edf_vec = _batch_grad(grad_edf)

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
            grad_edf=grad_edf_vec,
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
        grad_penalty = _wrap_optional(result.get("grad_penalty"), ctx.penalty_ref)
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
            grad_penalty,
            grad_weights,
            grad_by,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _GaussianRemlFreeBScoreBatchedFn(torch.autograd.Function):
    """Free-coefficient REML score with VJPs for B, log_lambda, and penalty."""

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        y: torch.Tensor,
        row_offsets: torch.Tensor,
        coefficients: torch.Tensor,
        log_lambda: torch.Tensor,
        penalty: torch.Tensor,
        weights: torch.Tensor | None,
        by: torch.Tensor | None,
        by_start_col: int,
    ) -> torch.Tensor:
        x_np = to_numpy_f64(x)
        y_np = to_numpy_f64(y)
        offsets_np = to_numpy_uintp(row_offsets)
        coef_np = to_numpy_f64(coefficients)
        log_lambda_np = to_numpy_f64(log_lambda)
        penalty_np = to_numpy_f64(penalty)
        weights_np = None if weights is None else to_numpy_f64(weights)
        by_np = None if by is None else to_numpy_f64(by)
        batch = int(offsets_np.size - 1)
        if coef_np.ndim == 2:
            coef_np = np.broadcast_to(coef_np, (batch, coef_np.shape[0], coef_np.shape[1]))
        scores = np.zeros(batch, dtype=np.float64)
        grad_coef = np.zeros_like(coef_np)
        grad_penalty = np.zeros((batch, penalty_np.shape[0], penalty_np.shape[1]), dtype=np.float64)
        grad_log_lambda = np.zeros(batch, dtype=np.float64)
        for b in range(batch):
            start = int(offsets_np[b])
            end = int(offsets_np[b + 1])
            if start == end:
                scores[b] = np.nan
                continue
            lam_value = float(log_lambda_np.reshape(-1)[0] if log_lambda_np.size == 1 else log_lambda_np.reshape(-1)[b])
            out = _np_api._gaussian_reml_score(
                x_np[start:end],
                y_np[start:end],
                coef_np[b],
                lam_value,
                penalty_np,
                weights=None if weights_np is None else weights_np[start:end],
                by=None if by_np is None else by_np[start:end],
                by_start_col=by_start_col,
            )
            scores[b] = float(out["reml_score"])
            grad_coef[b] = out["grad_coefficients"]
            grad_penalty[b] = out["grad_penalty"]
            grad_log_lambda[b] = float(out["grad_log_lambda"])
        _save_diff_tensors(ctx, coefficients, log_lambda, penalty)
        ctx.grad_coef = grad_coef
        ctx.grad_penalty = grad_penalty
        ctx.grad_log_lambda = grad_log_lambda
        ctx.batch = batch
        ctx.coefficients_was_2d = coefficients.ndim == 2
        ctx.log_lambda_shape = tuple(log_lambda.shape)
        ctx.ref = x
        ctx.coefficients_ref = coefficients
        ctx.log_lambda_ref = log_lambda
        ctx.penalty_ref = penalty
        return from_numpy_like(scores, x)

    @staticmethod
    def backward(ctx: Any, grad_score: torch.Tensor) -> tuple[Any, ...]:
        _check_saved_versions(ctx)
        upstream = to_numpy_f64(grad_score).reshape(-1)
        grad_coef_np = ctx.grad_coef * upstream[:, None, None]
        if ctx.coefficients_was_2d:
            grad_coef_np = grad_coef_np.sum(axis=0)
        grad_penalty_np = (ctx.grad_penalty * upstream[:, None, None]).sum(axis=0).astype(np.float64)
        grad_log_np = ctx.grad_log_lambda * upstream
        if ctx.log_lambda_shape == ():
            grad_log_np = np.asarray(grad_log_np.sum(), dtype=np.float64)
        elif int(np.prod(ctx.log_lambda_shape)) == 1:
            grad_log_np = np.asarray([grad_log_np.sum()], dtype=np.float64).reshape(ctx.log_lambda_shape)
        else:
            grad_log_np = grad_log_np.reshape(ctx.log_lambda_shape)
        return (
            None,
            None,
            None,
            from_numpy_like(grad_coef_np, ctx.coefficients_ref),
            from_numpy_like(grad_log_np, ctx.log_lambda_ref),
            from_numpy_like(grad_penalty_np, ctx.penalty_ref),
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
        Symmetric penalty matrix of shape ``(M, M)``. Symmetry is the
        function's domain; the Rust engine treats it as part of the input
        contract.
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
        Tuple of ``(coefficients, fitted, lam, reml_score, edf)`` as torch
        tensors sharing ``x``'s device and dtype.
    """
    apply = cast(Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], _GaussianRemlFitFn.apply)
    coefficients, fitted, lam, reml_score, edf = apply(
        x, y, penalty, weights, by, init_lambda, by_start_col
    )
    return GaussianRemlOutput(coefficients, fitted, lam, reml_score, edf)


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
        Symmetric penalty matrix of shape ``(M, M)``. See
        :func:`gaussian_reml_fit` for the input contract.
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
    apply = cast(Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], _GaussianRemlFitBatchedFn.apply)
    coefficients, fitted, lam, reml_score, edf = apply(
        x, y, row_offsets, penalty, weights, by, init_lambda, by_start_col
    )
    return GaussianRemlOutput(coefficients, fitted, lam, reml_score, edf)


def _canonical_basis_name(basis_kind: str | None, basis: str | None) -> str:
    return str(basis if basis is not None else basis_kind if basis_kind is not None else "bspline")


def _resolve_effective_basis(
    basis_kind: str | None,
    basis: str | None,
    basis_order: int | None,
) -> tuple[str, str, int]:
    """Return ``(display_kind, effective_kind, order)``.

    Aliases ``thinplate`` (1D ≡ Duchon ``m=2``) and ``duchon_multipenalty``
    onto the engine's Duchon basis, while keeping the user-supplied name
    available for output state and penalty resolution.
    """
    display_kind = _canonical_basis_name(basis_kind, basis)
    effective_kind, order, _ = _np_api._normalize_position_basis(display_kind, basis_order)
    return display_kind, effective_kind, order


def _canonical_penalty_tensor(
    locations: torch.Tensor,
    *,
    basis_kind: str,
    basis_order: int,
    periodic: bool,
    penalty: Any,
    smoothing: str,
    log_lambda: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    kind = basis_kind.strip().lower().replace("_", "").replace("-", "")
    penalty_kind = (
        None
        if penalty is None or isinstance(penalty, torch.Tensor)
        else str(penalty).strip().lower().replace("_", "-")
    )
    if isinstance(penalty, torch.Tensor):
        return penalty, log_lambda
    triple_operator_request = (
        (kind in {"duchon", "duchonspline"} and penalty_kind in {"triple-operator", "tripleoperator", "operator"})
        or kind in {"duchonmultipenalty", "duchontripleoperator"}
    )
    if triple_operator_request:
        if periodic:
            raise ValueError("triple-operator Duchon penalty is not defined for periodic bases")
        # ``duchon_multipenalty`` carries triple-operator by definition; allow
        # the explicit ``penalty="triple-operator"`` override on plain duchon
        # but error on incompatible override strings for the multipenalty kind.
        if kind in {"duchonmultipenalty", "duchontripleoperator"} and penalty_kind not in {
            None,
            "triple-operator",
            "tripleoperator",
            "operator",
        }:
            raise ValueError(
                f"basis='duchon_multipenalty' only supports the triple-operator penalty; got {penalty!r}"
            )
        m = 2 if kind in {"duchonmultipenalty", "duchontripleoperator"} else int(basis_order)
        mass, tension, stiffness = _np_api._duchon_operator_penalties(
            to_numpy_f64(locations), m=m
        )
        pieces = [
            from_numpy_like(mass, locations),
            from_numpy_like(tension, locations),
            from_numpy_like(stiffness, locations),
        ]
        if smoothing == "adam":
            if log_lambda is None:
                raise ValueError("smoothing='adam' with triple_operator requires log_lambda of length 3")
            if int(log_lambda.numel()) != 3:
                raise ValueError("triple_operator log_lambda must have length 3")
            weights = log_lambda.reshape(3).exp()
            combined = weights[0] * pieces[0] + weights[1] * pieces[1] + weights[2] * pieces[2]
            return combined, None
        if smoothing == "fixed":
            if log_lambda is None or int(log_lambda.numel()) != 3:
                raise ValueError("smoothing='fixed' with triple_operator requires log_lambda of length 3")
            weights = log_lambda.detach().reshape(3).exp()
            combined = weights[0] * pieces[0] + weights[1] * pieces[1] + weights[2] * pieces[2]
            return combined, None
        return pieces[0] + pieces[1] + pieces[2], None
    if penalty is None or penalty_kind is not None:
        penalty_np = _np_api._resolve_position_penalty(
            penalty,
            to_numpy_f64(locations),
            basis_kind=basis_kind,
            basis_order=basis_order,
            periodic=bool(periodic),
        )
        return from_numpy_like(penalty_np, locations), log_lambda
    return from_numpy_like(_np_api._numeric_matrix(penalty, "penalty"), locations), log_lambda


def _position_design(
    t: torch.Tensor,
    locations: torch.Tensor,
    *,
    basis_kind: str,
    basis_order: int,
    periodic: bool,
) -> torch.Tensor:
    from ._basis import bspline_basis, duchon_basis_1d

    kind = basis_kind.strip().lower().replace("_", "").replace("-", "")
    if kind in {"duchon", "duchonspline", "thinplate", "thinplatespline", "tps", "duchonmultipenalty", "duchontripleoperator"}:
        return duchon_basis_1d(t, locations, m=basis_order, periodic=periodic)
    if kind in {"bspline", "spline"}:
        return bspline_basis(t, locations, degree=basis_order, periodic=periodic)
    raise ValueError(f"position REML basis {basis_kind!r} is not supported by the torch path yet")


def gaussian_reml_fit_positions(
    t: torch.Tensor,
    y: torch.Tensor,
    basis_kind: str | None = None,
    knots_or_centers: Any = None,
    penalty: Any = None,
    *,
    basis: str | None = None,
    smoothing: str = "reml",
    log_lambda: torch.Tensor | None = None,
    coefficients: torch.Tensor | None = None,
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
        1D tensor of basis knots (B-spline) or centers (Duchon). May also
        be ``None`` to auto-derive from ``t`` quantiles, or an ``int`` to
        request a specific basis count with quantile placement.
    penalty:
        Symmetric penalty matrix of shape ``(M, M)``. ``None`` defaults to
        an identity ridge sized to the inferred basis dimension. See
        :func:`gaussian_reml_fit` for the input contract.
    basis_order:
        Optional override for the basis order; defaults from ``basis_kind``.
    periodic, period:
        Periodic-basis toggle and period length.
    weights, by, init_lambda, by_start_col:
        See :func:`gaussian_reml_fit`.
    """
    from ._basis import _resolve_basis_locations_tensor

    display_kind = _canonical_basis_name(basis_kind, basis)
    smoothing = str(smoothing).strip().lower()
    effective_kind, order, _ = _np_api._normalize_position_basis(display_kind, basis_order)
    knots_t = _resolve_basis_locations_tensor(
        t, knots_or_centers, basis_kind=effective_kind, degree=order
    )
    penalty_t, score_log_lambda = _canonical_penalty_tensor(
        knots_t,
        basis_kind=display_kind,
        basis_order=order,
        periodic=bool(periodic),
        penalty=penalty,
        smoothing=smoothing,
        log_lambda=log_lambda,
    )
    if coefficients is not None:
        if smoothing == "reml":
            raise ValueError("coefficients=... requires smoothing='adam' or smoothing='fixed' and log_lambda")
        if score_log_lambda is None:
            score_log_lambda = torch.zeros((), dtype=t.dtype, device=t.device)
        elif smoothing == "fixed":
            score_log_lambda = score_log_lambda.detach()
        x = _position_design(
            t, knots_t, basis_kind=effective_kind, basis_order=order, periodic=bool(periodic)
        )
        if coefficients.ndim != 2:
            raise ValueError("non-batched coefficients must have shape (n_basis, n_outputs)")
        offsets = torch.tensor([0, int(t.numel())], dtype=torch.long, device=t.device)
        apply_score = cast(Callable[..., torch.Tensor], _GaussianRemlFreeBScoreBatchedFn.apply)
        reml_score = apply_score(
            x,
            y,
            offsets,
            coefficients,
            score_log_lambda,
            penalty_t,
            weights,
            by,
            by_start_col,
        )[0]
        fitted = x @ coefficients
        lam = score_log_lambda.exp()
        edf = torch.full_like(reml_score, float("nan"))
        return GaussianRemlPositionOutput(
            coefficients,
            fitted,
            lam,
            reml_score,
            edf,
            knots_t,
            penalty_t,
            str(display_kind),
            int(order),
            bool(periodic),
            None if period is None else float(period),
        )
    if smoothing != "reml":
        raise ValueError("smoothing='adam' or 'fixed' requires coefficients=... in this API")
    apply = cast(Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], _GaussianRemlFitPositionsFn.apply)
    coefficients, fitted, lam, reml_score, edf = apply(
        t,
        y,
        knots_t,
        penalty_t,
        weights,
        by,
        effective_kind,
        order,
        periodic,
        period,
        init_lambda,
        by_start_col,
    )
    return GaussianRemlPositionOutput(
        coefficients,
        fitted,
        lam,
        reml_score,
        edf,
        knots_t,
        penalty_t,
        str(display_kind),
        int(order),
        bool(periodic),
        None if period is None else float(period),
    )


def gaussian_reml_fit_positions_batched(
    t: torch.Tensor,
    y: torch.Tensor,
    row_offsets: torch.Tensor,
    basis_kind: str | None = None,
    knots_or_centers: Any = None,
    penalty: Any = None,
    *,
    basis: str | None = None,
    smoothing: str = "reml",
    log_lambda: torch.Tensor | None = None,
    coefficients: torch.Tensor | None = None,
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
        :func:`gaussian_reml_fit_positions`. ``knots_or_centers`` and
        ``penalty`` accept the same auto-derived defaults — basis
        locations are inferred from the concatenated ``t``.
    weights, by, init_lambda, by_start_col:
        See :func:`gaussian_reml_fit`.
    """
    from ._basis import _resolve_basis_locations_tensor

    display_kind = _canonical_basis_name(basis_kind, basis)
    smoothing = str(smoothing).strip().lower()
    effective_kind, order, _ = _np_api._normalize_position_basis(display_kind, basis_order)
    knots_t = _resolve_basis_locations_tensor(
        t, knots_or_centers, basis_kind=effective_kind, degree=order
    )
    penalty_t, score_log_lambda = _canonical_penalty_tensor(
        knots_t,
        basis_kind=display_kind,
        basis_order=order,
        periodic=bool(periodic),
        penalty=penalty,
        smoothing=smoothing,
        log_lambda=log_lambda,
    )
    if coefficients is not None:
        if smoothing == "reml":
            raise ValueError("coefficients=... requires smoothing='adam' or smoothing='fixed' and log_lambda")
        if score_log_lambda is None:
            score_log_lambda = torch.zeros((), dtype=t.dtype, device=t.device)
        elif smoothing == "fixed":
            score_log_lambda = score_log_lambda.detach()
        x = _position_design(
            t, knots_t, basis_kind=effective_kind, basis_order=order, periodic=bool(periodic)
        )
        apply_score = cast(Callable[..., torch.Tensor], _GaussianRemlFreeBScoreBatchedFn.apply)
        reml_score = apply_score(
            x,
            y,
            row_offsets,
            coefficients,
            score_log_lambda,
            penalty_t,
            weights,
            by,
            by_start_col,
        )
        if coefficients.ndim == 2:
            fitted = x @ coefficients
        elif coefficients.ndim == 3:
            pieces = []
            offsets_np = to_numpy_uintp(row_offsets)
            for b in range(int(offsets_np.size - 1)):
                start = int(offsets_np[b])
                end = int(offsets_np[b + 1])
                pieces.append(x[start:end] @ coefficients[b])
            fitted = torch.cat(pieces, dim=0)
        else:
            raise ValueError("batched coefficients must have shape (n_basis, n_outputs) or (batch, n_basis, n_outputs)")
        lam = (
            score_log_lambda.exp()
            if score_log_lambda is not None
            else torch.ones_like(reml_score)
        )
        edf = torch.full_like(reml_score, float("nan"))
        return GaussianRemlPositionOutput(
            coefficients,
            fitted,
            lam,
            reml_score,
            edf,
            knots_t,
            penalty_t,
            str(display_kind),
            int(order),
            bool(periodic),
            None if period is None else float(period),
        )
    if smoothing != "reml":
        raise ValueError("smoothing='adam' or 'fixed' requires coefficients=... in this API")
    apply = cast(Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], _GaussianRemlFitPositionsBatchedFn.apply)
    coefficients, fitted, lam, reml_score, edf = apply(
        t,
        y,
        row_offsets,
        knots_t,
        penalty_t,
        weights,
        by,
        effective_kind,
        order,
        periodic,
        period,
        init_lambda,
        by_start_col,
    )
    return GaussianRemlPositionOutput(
        coefficients,
        fitted,
        lam,
        reml_score,
        edf,
        knots_t,
        penalty_t,
        str(display_kind),
        int(order),
        bool(periodic),
        None if period is None else float(period),
    )
