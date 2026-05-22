"""Closed-form Gaussian REML autograd wrappers for the gamfit torch bridge.

The Rust engine ships analytic VJPs for every closed-form Gaussian REML
primitive in :mod:`gamfit._api`. This module is the canonical PyTorch face
of those primitives. Each public wrapper accepts torch tensors, runs the
forward in Rust on f64 CPU, and routes upstream gradients into the matching
Rust backward so the result composes inside larger torch graphs.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class AdditiveRemlOutput:
    """Forward outputs for the multi-smooth additive REML fit.

    ``coefficients`` is a list of per-smooth coefficient blocks (each of
    shape ``(K_i, D)``); ``fitted`` is the joint fit ``(N, D)``; ``lam`` is
    the shared scalar smoothing parameter; ``reml_score`` and ``edf`` are
    scalars from the underlying single-λ fit.
    """

    coefficients: list[torch.Tensor]
    fitted: torch.Tensor
    lam: torch.Tensor
    reml_score: torch.Tensor
    edf: torch.Tensor


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
        Optional ``by`` vector of shape ``(N,)``. Acts as a per-row
        multiplier on the design's contribution: the model becomes
        ``y_i ≈ by_i * (phi_i @ B)`` for rows ``i`` where the ``by``
        modulation is active (see ``by_start_col``). Equivalently,
        ``fitted = by * (Phi @ B)``, NOT ``Phi @ B``. When ``by_i = 0``
        the corresponding ``fitted_i`` is exactly zero; the REML fit
        is also weighted by ``by`` so zero-``by`` rows do not influence
        ``B``. Common SAE use: ``by`` = per-token feature amplitude;
        ``fitted`` is already amplitude-weighted so callers must NOT
        multiply the prediction by ``by`` a second time downstream.
    by_start_col:
        First design column the ``by`` modulation applies to. Columns
        before this index are NOT multiplied by ``by``.

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


def gaussian_reml_fit_additive(
    designs: list[torch.Tensor],
    response: torch.Tensor,
    penalties: list[torch.Tensor],
    *,
    bys: list[torch.Tensor | None] | None = None,
    weights: torch.Tensor | None = None,
    init_lambda: float | None = None,
) -> AdditiveRemlOutput:
    """Joint multi-smooth additive Gaussian REML fit (single shared λ).

    .. warning::
       **This path fits a single shared smoothing parameter λ across every
       smooth block.** It is *not* the mgcv-style per-smooth λ that most
       users expect from an additive REML fit. Concretely: the per-smooth
       design blocks are concatenated into a wide design
       ``[D_1 | D_2 | ...]`` and the per-smooth ``S_i`` are assembled into
       a block-diagonal joint penalty, but the combined problem is then
       handed to the single-λ :func:`gaussian_reml_fit` — so one scalar λ
       multiplies the entire block-diagonal penalty rather than one λ_k
       per block.

    For per-smooth λ — i.e. the standard ``ŷ ~ s(x1) + s(x2) + ...`` fit
    with independent smoothing parameters for each term — use the formula
    API instead::

        model = gamfit.fit(data, 'y ~ s(x1) + s(x2) + ...')

    The formula API drives the PIRLS workflow, which performs the full
    multi-block REML/LAML outer optimisation (one λ_k per smooth) and
    returns a serialised :class:`gamfit.Model`. If you only need the
    final coefficients / λ vector / EDF you can post-process that
    ``Model``; if you specifically need a differentiable ``torch.Tensor``
    pipeline you are constrained to the single-λ closed-form here until
    the Rust kernel grows multi-block support.

    Why this is single-λ (math note)
    --------------------------------
    The closed-form Gaussian REML kernel (see
    ``crates/gam-core/src/solver/gaussian_reml.rs``) exploits the joint
    generalised eigendecomposition of the pair ``(S, X'WX)`` so that the
    REML score and its gradient become rational functions of the single
    scalar λ, evaluated at O(1) per probe after one eigendecomposition.
    With multiple λ_k the relevant pair is ``(Σ_k λ_k S_k, X'WX)`` and a
    *single* eigendecomposition no longer diagonalises every λ_k jointly,
    so the closed-form collapses: the multi-block path needs a fresh
    per-iteration solve, an outer multi-dimensional optimiser over
    ``log λ_k``, and an analytic VJP through the F×F Hessian of the
    REML criterion. That refactor is tracked as a deliberate API
    limitation; see :mod:`gamfit.torch._multi_lambda_status` and the
    "True multi-λ in additive REML" feature request. Until then the
    canonical recommendation for multi-smooth fits is the formula API.

    Parameters
    ----------
    designs:
        ``F`` per-smooth design matrices, each of shape ``(N, K_i)``.
    response:
        Response matrix of shape ``(N, D)``.
    penalties:
        ``F`` per-smooth penalty matrices, each of shape ``(K_i, K_i)``.
    bys:
        Optional list of ``F`` per-smooth ``by`` vectors of shape ``(N,)``;
        ``None`` entries skip the modulation for that smooth. When the full
        list is ``None`` no modulation is applied anywhere.
    weights:
        Optional row weights of shape ``(N,)``.
    init_lambda:
        Optional initial smoothing parameter, forwarded to the single-λ fit.

    Returns
    -------
    AdditiveRemlOutput
        ``coefficients`` is a list of length ``F`` with the per-smooth
        blocks; ``fitted`` is the joint ``(N, D)`` reconstruction;
        ``lam``/``reml_score``/``edf`` are the shared scalar outputs of the
        underlying single-λ REML fit. Note ``lam`` is a single scalar
        applied across all smooths, *not* a length-``F`` vector.
    """
    if len(designs) != len(penalties):
        raise ValueError(
            f"designs and penalties must have equal length; got {len(designs)} vs {len(penalties)}"
        )
    if len(designs) == 0:
        raise ValueError("gaussian_reml_fit_additive requires at least one smooth")
    if bys is not None and len(bys) != len(designs):
        raise ValueError(
            f"bys must have one entry per smooth; got {len(bys)} for {len(designs)} smooths"
        )

    # Apply per-smooth ``by`` modulation row-wise to each design block.
    modulated: list[torch.Tensor] = []
    for i, design in enumerate(designs):
        if design.dim() != 2:
            raise ValueError(
                f"designs[{i}] must be 2D (N, K_i); got shape {tuple(design.shape)}"
            )
        if bys is not None and bys[i] is not None:
            by_i = bys[i]
            assert by_i is not None  # narrow for type-checkers
            if by_i.dim() != 1 or by_i.shape[0] != design.shape[0]:
                raise ValueError(
                    f"bys[{i}] must be 1D of length N={design.shape[0]}; got {tuple(by_i.shape)}"
                )
            modulated.append(design * by_i.unsqueeze(1))
        else:
            modulated.append(design)

    joint_design = torch.cat(modulated, dim=1)
    joint_penalty = torch.block_diag(*penalties)

    fit = gaussian_reml_fit(
        joint_design,
        response,
        joint_penalty,
        weights=weights,
        init_lambda=init_lambda,
    )

    # Split coefficients back per-smooth at the column offsets of ``joint_design``.
    offsets: list[int] = [0]
    for design in designs:
        offsets.append(offsets[-1] + int(design.shape[1]))
    per_smooth_coefs: list[torch.Tensor] = [
        fit.coefficients[offsets[i]:offsets[i + 1]] for i in range(len(designs))
    ]

    return AdditiveRemlOutput(
        coefficients=per_smooth_coefs,
        fitted=fit.fitted,
        lam=fit.lam,
        reml_score=fit.reml_score,
        edf=fit.edf,
    )
