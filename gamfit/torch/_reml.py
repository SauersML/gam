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
    shape ``(K_i, D)``); ``fitted`` is the joint fit ``(N, D)``; ``lam``
    carries the smoothing parameter(s): a length-``F`` 1D tensor when the
    fit went through the multi-block per-smooth-λ Rust path, or a scalar
    tensor for the legacy single-λ block-diagonal path; ``reml_score`` is
    the converged REML criterion; ``edf`` is length-``F`` per-smooth on
    the multi-block path and a scalar on the legacy path.

    .. note::
       The multi-block path is forward-only: the underlying Rust
       ``optimize_external_design`` driver does not yet expose an
       analytic IFT/envelope VJP through its outer F×F Hessian to
       Python. See :func:`gaussian_reml_fit_blocks` for the canonical
       multi-λ entrypoint and its NotImplementedError contract on
       ``.backward()``.
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


def gaussian_reml_fit_blocks(
    design_blocks: list[torch.Tensor],
    penalty_blocks: list[torch.Tensor],
    y: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    init_log_lambdas: torch.Tensor | None = None,
) -> AdditiveRemlOutput:
    """Multi-block Gaussian REML forward fit with per-smooth λ.

    Forward-only torch face of the Rust ``optimize_external_design``
    multi-block outer-loop (Gaussian identity link). Each
    ``design_blocks[k]`` of shape ``(N, K_k)`` carries its own penalty
    ``penalty_blocks[k]`` of shape ``(K_k, K_k)`` with its own scalar
    ``λ_k`` driven by the outer Newton/EFS loop. Returns the per-block
    smoothing parameters in ``AdditiveRemlOutput.lam`` (1D tensor of
    length ``F``) together with per-block EDFs in ``.edf``.

    Parameters
    ----------
    design_blocks:
        List of ``F`` per-smooth design matrices, each of shape ``(N, K_k)``.
    penalty_blocks:
        List of ``F`` per-smooth penalty matrices, each of shape ``(K_k, K_k)``.
    y:
        Response tensor of shape ``(N,)`` or ``(N, 1)``. Multi-output
        ``(N, D>1)`` is not supported on this path; call this function once
        per output column or use the legacy single-λ
        :func:`gaussian_reml_fit_additive` path.
    weights:
        Optional row weights of shape ``(N,)``.
    init_log_lambdas:
        Optional warm-start log-λ vector of shape ``(F,)`` (one entry per
        block). When omitted the driver chooses its own seeds.

    Returns
    -------
    AdditiveRemlOutput
        ``coefficients`` is a list of ``F`` per-block ``(K_k, 1)`` tensors;
        ``fitted`` is ``(N, 1)``; ``lam`` is a length-``F`` 1D tensor of
        per-smooth λ; ``reml_score`` is the converged REML criterion;
        ``edf`` is a length-``F`` tensor of per-smooth EDFs.

    Notes
    -----
    **Forward only.** ``.backward()`` through this path is currently
    unsupported and will raise ``NotImplementedError`` if attempted. A
    differentiable backward needs an analytic IFT through the F×F outer
    Hessian of the REML criterion plus envelope-theorem VJPs of the
    per-block penalty derivatives — that work is intentionally deferred
    to a separate workstream. Tensors returned here are detached.
    """
    if len(design_blocks) != len(penalty_blocks):
        raise ValueError(
            f"design_blocks and penalty_blocks must have equal length; got "
            f"{len(design_blocks)} vs {len(penalty_blocks)}"
        )
    if len(design_blocks) == 0:
        raise ValueError("gaussian_reml_fit_blocks requires at least one block")

    designs_np = [to_numpy_f64(d) for d in design_blocks]
    penalties_np = [to_numpy_f64(p) for p in penalty_blocks]

    if y.dim() == 1:
        y_mat = y.unsqueeze(1)
    elif y.dim() == 2:
        if y.shape[1] != 1:
            raise NotImplementedError(
                "gaussian_reml_fit_blocks: multi-output y (D>1) is not supported on "
                "the multi-block REML path. Call once per output column, or use "
                "gaussian_reml_fit_additive (single shared λ) for D>1."
            )
        y_mat = y
    else:
        raise ValueError(f"y must be 1D or 2D; got shape {tuple(y.shape)}")

    y_np = to_numpy_f64(y_mat)
    weights_np = None if weights is None else to_numpy_f64(weights)
    rhos_np = None if init_log_lambdas is None else to_numpy_f64(init_log_lambdas)

    out = _np_api.gaussian_reml_fit_blocks_forward(
        designs_np,
        penalties_np,
        y_np,
        weights=weights_np,
        init_rhos=rhos_np,
    )

    ref = design_blocks[0]
    coefs_full = np.asarray(out["coefficients"], dtype=np.float64)  # (p_total, 1)
    fitted = np.asarray(out["fitted"], dtype=np.float64)
    lambdas = np.asarray(out["lambdas"], dtype=np.float64)
    reml_score = float(out["reml_score"])
    edf = np.asarray(out["edf"], dtype=np.float64)

    offsets: list[int] = [0]
    for d in design_blocks:
        offsets.append(offsets[-1] + int(d.shape[1]))
    per_block_coefs: list[torch.Tensor] = [
        from_numpy_like(coefs_full[offsets[i]:offsets[i + 1], :], ref)
        for i in range(len(design_blocks))
    ]
    fitted_t = from_numpy_like(fitted, ref)
    lam_t = from_numpy_like(lambdas, ref)
    reml_t = from_numpy_like(np.asarray(reml_score, dtype=np.float64), ref)
    edf_t = from_numpy_like(edf, ref)

    return AdditiveRemlOutput(
        coefficients=per_block_coefs,
        fitted=fitted_t,
        lam=lam_t,
        reml_score=reml_t,
        edf=edf_t,
    )


def gaussian_reml_fit_additive(
    designs: list[torch.Tensor],
    response: torch.Tensor,
    penalties: list[torch.Tensor],
    *,
    bys: list[torch.Tensor | None] | None = None,
    weights: torch.Tensor | None = None,
    init_lambda: float | None = None,
) -> AdditiveRemlOutput:
    """Joint multi-smooth additive Gaussian REML fit.

    For ``F > 1`` blocks with a single-column response, this routes to the
    multi-block Rust REML path (see :func:`gaussian_reml_fit_blocks`) so
    each smooth recovers its own ``λ_k`` from the outer optimisation; the
    returned ``AdditiveRemlOutput.lam`` is a length-``F`` 1D tensor on
    that path.

    For ``F == 1`` (single block), or for multi-output ``response`` of
    shape ``(N, D>1)`` where the multi-block driver is not yet
    multi-response, the call falls back to the legacy single-λ
    closed-form path: the per-smooth ``S_i`` are assembled into a
    block-diagonal joint penalty and a single scalar λ multiplies the
    entire block-diagonal, evaluated through the closed-form Gaussian
    REML kernel. In that fallback ``AdditiveRemlOutput.lam`` is a scalar.

    .. note::
       The multi-block path is forward-only. ``.backward()`` through any
       of its outputs raises ``NotImplementedError`` because the analytic
       VJP through the F×F outer Hessian is a separate workstream. The
       legacy single-λ fallback (``F == 1`` or ``D > 1``) remains fully
       differentiable.

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

    # Route to the multi-block per-smooth-λ Rust path when we have
    # multiple distinct penalties and a single-column response. The
    # multi-block driver is single-response today, so D > 1 still falls
    # back to the legacy single-λ closed-form path.
    response_is_2d = response.dim() == 2 and response.shape[1] > 1
    if len(designs) > 1 and not response_is_2d:
        if init_lambda is None:
            init_log = None
        else:
            init_log = torch.full(
                (len(designs),),
                float(np.log(max(float(init_lambda), 1e-300))),
                dtype=torch.float64,
                device=modulated[0].device,
            )
        return gaussian_reml_fit_blocks(
            modulated,
            list(penalties),
            response,
            weights=weights,
            init_log_lambdas=init_log,
        )

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
