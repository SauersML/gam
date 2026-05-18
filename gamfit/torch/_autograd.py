"""Custom autograd Functions for gamfit's REML primitives.

Each Function pairs a forward numpy/Rust call with the matching
``*_backward`` partner from :mod:`gamfit._api`. Forward outputs are
returned as torch tensors on the input's device/dtype; the analytic
VJP is dispatched through the Rust backward when any upstream grad is
non-trivial. ``None`` upstream grads are honoured: when every upstream
grad is ``None`` the backward short-circuits to all-``None`` returns.

These Functions are private; users should call the public functional
wrappers in :mod:`gamfit.torch._functional`.
"""

from __future__ import annotations

from typing import Any

from .. import _api as _gam_api
from ._coerce import from_numpy_like, reference_tensor, to_numpy_f64


def _torch() -> Any:
    import torch

    return torch


def _grad_is_active(*grads: Any) -> bool:
    torch = _torch()
    for g in grads:
        if g is None:
            continue
        if isinstance(g, torch.Tensor):
            if g.numel() == 0:
                continue
            return True
        return True
    return False


def _maybe_grad_numpy(grad: Any) -> Any:
    if grad is None:
        return None
    return to_numpy_f64(grad)


def _grad_scalar(grad: Any) -> float:
    if grad is None:
        return 0.0
    torch = _torch()
    if isinstance(grad, torch.Tensor):
        return float(grad.detach().cpu().to(torch.float64).item())
    return float(grad)


def _grad_batch_vector(grad: Any, batch: int) -> Any:
    if grad is None:
        return None
    arr = to_numpy_f64(grad)
    import numpy as np

    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(batch, float(arr), dtype=np.float64)
    return np.ascontiguousarray(arr.reshape(-1), dtype=np.float64)


class _GaussianRemlFit(_torch().autograd.Function if False else object):
    pass


def _make_function_classes() -> dict[str, Any]:
    """Build the autograd Function classes lazily once torch is importable."""

    torch = _torch()

    class GaussianRemlFit(torch.autograd.Function):
        """Autograd Function for :func:`gamfit.gaussian_reml_fit`."""

        @staticmethod
        def forward(  # type: ignore[override]
            ctx: Any,
            x: Any,
            y: Any,
            penalty: Any,
            weights: Any,
            by: Any,
            init_lambda: Any,
            by_start_col: int,
        ) -> Any:
            x_np = to_numpy_f64(x)
            y_np = to_numpy_f64(y)
            penalty_np = to_numpy_f64(penalty)
            weights_np = None if weights is None else to_numpy_f64(weights)
            by_np = None if by is None else to_numpy_f64(by)
            init = None if init_lambda is None else float(init_lambda)
            out = _gam_api.gaussian_reml_fit(
                x_np,
                y_np,
                penalty_np,
                weights=weights_np,
                init_lambda=init,
                by=by_np,
                by_start_col=int(by_start_col),
            )
            ref = reference_tensor(x, y, weights, by)
            coef_t = from_numpy_like(out["coefficients"], ref)
            fit_t = from_numpy_like(out["fitted"], ref)
            lam_t = from_numpy_like(out["lambda"].reshape(()), ref)
            reml_t = from_numpy_like(out["reml_score"].reshape(()), ref)
            ctx.save_for_backward(
                _as_tensor(x_np), _as_tensor(y_np), _as_tensor(penalty_np),
                _as_tensor(weights_np), _as_tensor(by_np),
            )
            ctx.by_start_col = int(by_start_col)
            ctx.init_lambda = init
            ctx.ref_dtype = None if ref is None else ref.dtype
            ctx.ref_device = None if ref is None else ref.device
            ctx.has_weights = weights is not None
            ctx.has_by = by is not None
            return coef_t, fit_t, lam_t, reml_t

        @staticmethod
        def backward(  # type: ignore[override]
            ctx: Any,
            grad_coef: Any,
            grad_fit: Any,
            grad_lambda: Any,
            grad_reml: Any,
        ) -> Any:
            if not _grad_is_active(grad_coef, grad_fit, grad_lambda, grad_reml):
                return (None,) * 7
            x_t, y_t, penalty_t, weights_t, by_t = ctx.saved_tensors[:5]
            x_np = x_t.numpy()
            y_np = y_t.numpy()
            penalty_np = penalty_t.numpy()
            weights_np = weights_t.numpy() if ctx.has_weights else None
            by_np = by_t.numpy() if ctx.has_by else None
            out = _gam_api.gaussian_reml_fit_backward(
                x_np,
                y_np,
                penalty_np,
                grad_lambda=_grad_scalar(grad_lambda),
                grad_coefficients=_maybe_grad_numpy(grad_coef),
                grad_fitted=_maybe_grad_numpy(grad_fit),
                grad_reml_score=_grad_scalar(grad_reml),
                weights=weights_np,
                init_lambda=ctx.init_lambda,
                by=by_np,
                by_start_col=ctx.by_start_col,
            )
            ref = _ref_from_ctx(ctx)
            gx = _back_tensor(out.get("grad_x"), ref)
            gy = _back_tensor(out.get("grad_y"), ref)
            gw = _back_tensor(out.get("grad_weights"), ref) if ctx.has_weights else None
            gby = _back_tensor(out.get("grad_by"), ref) if ctx.has_by else None
            return gx, gy, None, gw, gby, None, None

    class GaussianRemlFitBatched(torch.autograd.Function):
        """Autograd Function for :func:`gamfit.gaussian_reml_fit_batched`."""

        @staticmethod
        def forward(  # type: ignore[override]
            ctx: Any,
            x: Any,
            y: Any,
            row_offsets: Any,
            penalty: Any,
            weights: Any,
            by: Any,
            init_lambda: Any,
            by_start_col: int,
        ) -> Any:
            x_np = to_numpy_f64(x)
            y_np = to_numpy_f64(y)
            penalty_np = to_numpy_f64(penalty)
            weights_np = None if weights is None else to_numpy_f64(weights)
            by_np = None if by is None else to_numpy_f64(by)
            offsets_np = _coerce_offsets(row_offsets)
            init = None if init_lambda is None else float(init_lambda)
            out = _gam_api.gaussian_reml_fit_batched(
                x_np,
                y_np,
                offsets_np,
                penalty_np,
                weights=weights_np,
                init_lambda=init,
            )
            ref = reference_tensor(x, y, weights, by)
            coef_t = from_numpy_like(out["coefficients"], ref)
            fit_t = from_numpy_like(out["fitted"], ref)
            lam_t = from_numpy_like(out["lambda"], ref)
            reml_t = from_numpy_like(out["reml_score"], ref)
            ctx.save_for_backward(
                _as_tensor(x_np), _as_tensor(y_np), _as_tensor(penalty_np),
                _as_tensor(weights_np), _as_tensor(by_np),
            )
            ctx.offsets_np = offsets_np
            ctx.batch = int(offsets_np.size - 1)
            ctx.by_start_col = int(by_start_col)
            ctx.init_lambda = init
            ctx.ref_dtype = None if ref is None else ref.dtype
            ctx.ref_device = None if ref is None else ref.device
            ctx.has_weights = weights is not None
            ctx.has_by = by is not None
            return coef_t, fit_t, lam_t, reml_t

        @staticmethod
        def backward(  # type: ignore[override]
            ctx: Any,
            grad_coef: Any,
            grad_fit: Any,
            grad_lambda: Any,
            grad_reml: Any,
        ) -> Any:
            if not _grad_is_active(grad_coef, grad_fit, grad_lambda, grad_reml):
                return (None,) * 8
            x_t, y_t, penalty_t, weights_t, by_t = ctx.saved_tensors[:5]
            x_np = x_t.numpy()
            y_np = y_t.numpy()
            penalty_np = penalty_t.numpy()
            weights_np = weights_t.numpy() if ctx.has_weights else None
            by_np = by_t.numpy() if ctx.has_by else None
            out = _gam_api.gaussian_reml_fit_batched_backward(
                x_np,
                y_np,
                ctx.offsets_np,
                penalty_np,
                grad_lambda=_grad_batch_vector(grad_lambda, ctx.batch),
                grad_coefficients=_maybe_grad_numpy(grad_coef),
                grad_fitted=_maybe_grad_numpy(grad_fit),
                grad_reml_score=_grad_batch_vector(grad_reml, ctx.batch),
                weights=weights_np,
                init_lambda=ctx.init_lambda,
                by=by_np,
                by_start_col=ctx.by_start_col,
            )
            ref = _ref_from_ctx(ctx)
            gx = _back_tensor(out.get("grad_x"), ref)
            gy = _back_tensor(out.get("grad_y"), ref)
            gw = _back_tensor(out.get("grad_weights"), ref) if ctx.has_weights else None
            gby = _back_tensor(out.get("grad_by"), ref) if ctx.has_by else None
            return gx, gy, None, None, gw, gby, None, None

    class GaussianRemlFitPositions(torch.autograd.Function):
        """Autograd Function for :func:`gamfit.gaussian_reml_fit_positions`."""

        @staticmethod
        def forward(  # type: ignore[override]
            ctx: Any,
            t: Any,
            y: Any,
            basis_kind: str,
            knots_or_centers: Any,
            penalty: Any,
            basis_order: Any,
            periodic: bool,
            period: Any,
            weights: Any,
            by: Any,
            init_lambda: Any,
            by_start_col: int,
        ) -> Any:
            t_np = to_numpy_f64(t)
            y_np = to_numpy_f64(y)
            knots_np = to_numpy_f64(knots_or_centers)
            penalty_np = to_numpy_f64(penalty)
            weights_np = None if weights is None else to_numpy_f64(weights)
            by_np = None if by is None else to_numpy_f64(by)
            init = None if init_lambda is None else float(init_lambda)
            order = None if basis_order is None else int(basis_order)
            per = None if period is None else float(period)
            out = _gam_api.gaussian_reml_fit_positions(
                t_np,
                y_np,
                str(basis_kind),
                knots_np,
                penalty_np,
                basis_order=order,
                periodic=bool(periodic),
                period=per,
                weights=weights_np,
                init_lambda=init,
                by=by_np,
                by_start_col=int(by_start_col),
            )
            ref = reference_tensor(t, y, weights, by)
            coef_t = from_numpy_like(out["coefficients"], ref)
            fit_t = from_numpy_like(out["fitted"], ref)
            lam_t = from_numpy_like(out["lambda"].reshape(()), ref)
            reml_t = from_numpy_like(out["reml_score"].reshape(()), ref)
            ctx.save_for_backward(
                _as_tensor(t_np), _as_tensor(y_np), _as_tensor(knots_np),
                _as_tensor(penalty_np), _as_tensor(weights_np), _as_tensor(by_np),
            )
            ctx.basis_kind = str(basis_kind)
            ctx.basis_order = order
            ctx.periodic = bool(periodic)
            ctx.period = per
            ctx.init_lambda = init
            ctx.by_start_col = int(by_start_col)
            ctx.ref_dtype = None if ref is None else ref.dtype
            ctx.ref_device = None if ref is None else ref.device
            ctx.has_weights = weights is not None
            ctx.has_by = by is not None
            return coef_t, fit_t, lam_t, reml_t

        @staticmethod
        def backward(  # type: ignore[override]
            ctx: Any,
            grad_coef: Any,
            grad_fit: Any,
            grad_lambda: Any,
            grad_reml: Any,
        ) -> Any:
            if not _grad_is_active(grad_coef, grad_fit, grad_lambda, grad_reml):
                return (None,) * 12
            t_t, y_t, knots_t, penalty_t, weights_t, by_t = ctx.saved_tensors[:6]
            t_np = t_t.numpy()
            y_np = y_t.numpy()
            knots_np = knots_t.numpy()
            penalty_np = penalty_t.numpy()
            weights_np = weights_t.numpy() if ctx.has_weights else None
            by_np = by_t.numpy() if ctx.has_by else None
            out = _gam_api.gaussian_reml_fit_positions_backward(
                t_np,
                y_np,
                ctx.basis_kind,
                knots_np,
                penalty_np,
                grad_lambda=_grad_scalar(grad_lambda),
                grad_coefficients=_maybe_grad_numpy(grad_coef),
                grad_fitted=_maybe_grad_numpy(grad_fit),
                grad_reml_score=_grad_scalar(grad_reml),
                basis_order=ctx.basis_order,
                periodic=ctx.periodic,
                period=ctx.period,
                weights=weights_np,
                init_lambda=ctx.init_lambda,
                by=by_np,
                by_start_col=ctx.by_start_col,
            )
            ref = _ref_from_ctx(ctx)
            gt = _back_tensor(out.get("grad_t"), ref)
            gy = _back_tensor(out.get("grad_y"), ref)
            gw = _back_tensor(out.get("grad_weights"), ref) if ctx.has_weights else None
            gby = _back_tensor(out.get("grad_by"), ref) if ctx.has_by else None
            return (
                gt, gy, None, None, None, None, None, None,
                gw, gby, None, None,
            )

    class GaussianRemlFitPositionsBatched(torch.autograd.Function):
        """Autograd Function for :func:`gamfit.gaussian_reml_fit_positions_batched`."""

        @staticmethod
        def forward(  # type: ignore[override]
            ctx: Any,
            t: Any,
            y: Any,
            row_offsets: Any,
            basis_kind: str,
            knots_or_centers: Any,
            penalty: Any,
            basis_order: Any,
            periodic: bool,
            period: Any,
            weights: Any,
            by: Any,
            init_lambda: Any,
            by_start_col: int,
        ) -> Any:
            t_np = to_numpy_f64(t)
            y_np = to_numpy_f64(y)
            knots_np = to_numpy_f64(knots_or_centers)
            penalty_np = to_numpy_f64(penalty)
            weights_np = None if weights is None else to_numpy_f64(weights)
            by_np = None if by is None else to_numpy_f64(by)
            offsets_np = _coerce_offsets(row_offsets)
            init = None if init_lambda is None else float(init_lambda)
            order = None if basis_order is None else int(basis_order)
            per = None if period is None else float(period)
            out = _gam_api.gaussian_reml_fit_positions_batched(
                t_np,
                y_np,
                offsets_np,
                str(basis_kind),
                knots_np,
                penalty_np,
                basis_order=order,
                periodic=bool(periodic),
                period=per,
                weights=weights_np,
                init_lambda=init,
                by=by_np,
                by_start_col=int(by_start_col),
            )
            ref = reference_tensor(t, y, weights, by)
            coef_t = from_numpy_like(out["coefficients"], ref)
            fit_t = from_numpy_like(out["fitted"], ref)
            lam_t = from_numpy_like(out["lambda"], ref)
            reml_t = from_numpy_like(out["reml_score"], ref)
            ctx.save_for_backward(
                _as_tensor(t_np), _as_tensor(y_np), _as_tensor(knots_np),
                _as_tensor(penalty_np), _as_tensor(weights_np), _as_tensor(by_np),
            )
            ctx.offsets_np = offsets_np
            ctx.batch = int(offsets_np.size - 1)
            ctx.basis_kind = str(basis_kind)
            ctx.basis_order = order
            ctx.periodic = bool(periodic)
            ctx.period = per
            ctx.init_lambda = init
            ctx.by_start_col = int(by_start_col)
            ctx.ref_dtype = None if ref is None else ref.dtype
            ctx.ref_device = None if ref is None else ref.device
            ctx.has_weights = weights is not None
            ctx.has_by = by is not None
            return coef_t, fit_t, lam_t, reml_t

        @staticmethod
        def backward(  # type: ignore[override]
            ctx: Any,
            grad_coef: Any,
            grad_fit: Any,
            grad_lambda: Any,
            grad_reml: Any,
        ) -> Any:
            if not _grad_is_active(grad_coef, grad_fit, grad_lambda, grad_reml):
                return (None,) * 13
            t_t, y_t, knots_t, penalty_t, weights_t, by_t = ctx.saved_tensors[:6]
            t_np = t_t.numpy()
            y_np = y_t.numpy()
            knots_np = knots_t.numpy()
            penalty_np = penalty_t.numpy()
            weights_np = weights_t.numpy() if ctx.has_weights else None
            by_np = by_t.numpy() if ctx.has_by else None
            out = _gam_api.gaussian_reml_fit_positions_batched_backward(
                t_np,
                y_np,
                ctx.offsets_np,
                ctx.basis_kind,
                knots_np,
                penalty_np,
                grad_lambda=_grad_batch_vector(grad_lambda, ctx.batch),
                grad_coefficients=_maybe_grad_numpy(grad_coef),
                grad_fitted=_maybe_grad_numpy(grad_fit),
                grad_reml_score=_grad_batch_vector(grad_reml, ctx.batch),
                basis_order=ctx.basis_order,
                periodic=ctx.periodic,
                period=ctx.period,
                weights=weights_np,
                init_lambda=ctx.init_lambda,
                by=by_np,
                by_start_col=ctx.by_start_col,
            )
            ref = _ref_from_ctx(ctx)
            gt = _back_tensor(out.get("grad_t"), ref)
            gy = _back_tensor(out.get("grad_y"), ref)
            gw = _back_tensor(out.get("grad_weights"), ref) if ctx.has_weights else None
            gby = _back_tensor(out.get("grad_by"), ref) if ctx.has_by else None
            return (
                gt, gy, None, None, None, None, None, None, None,
                gw, gby, None, None,
            )

    return {
        "GaussianRemlFit": GaussianRemlFit,
        "GaussianRemlFitBatched": GaussianRemlFitBatched,
        "GaussianRemlFitPositions": GaussianRemlFitPositions,
        "GaussianRemlFitPositionsBatched": GaussianRemlFitPositionsBatched,
    }


def _as_tensor(value: Any) -> Any:
    """Wrap an ndarray (or None) as a CPU f64 tensor for ``save_for_backward``."""
    torch = _torch()
    if value is None:
        return torch.zeros(0, dtype=torch.float64)
    return torch.from_numpy(value)


def _coerce_offsets(value: Any) -> Any:
    import numpy as np

    torch = _torch()
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    return np.ascontiguousarray(arr, dtype=np.uintp)


def _ref_from_ctx(ctx: Any) -> Any:
    torch = _torch()
    if getattr(ctx, "ref_dtype", None) is None:
        return None
    return torch.empty(0, dtype=ctx.ref_dtype, device=ctx.ref_device)


def _back_tensor(value: Any, ref: Any) -> Any:
    if value is None:
        return None
    return from_numpy_like(value, ref)


_CACHE: dict[str, Any] = {}


def get_classes() -> dict[str, Any]:
    """Return (memoised) torch autograd Function classes for the REML wrappers."""
    if not _CACHE:
        _CACHE.update(_make_function_classes())
    return _CACHE
