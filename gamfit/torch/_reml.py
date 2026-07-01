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
from gamfit._reml_common import (
    RemlCallSpec,
    run_point_design_backward,
    run_point_design_forward,
)

from ._coerce import from_numpy_like, to_numpy_f64, to_numpy_uintp


class GaussianRemlOutput(NamedTuple):
    """Forward outputs shared by every Gaussian REML wrapper."""

    coefficients: torch.Tensor
    fitted: torch.Tensor
    lam: torch.Tensor
    reml_score: torch.Tensor
    edf: torch.Tensor


@dataclass(slots=True)
class AdditiveRemlOutput:
    """Forward outputs for the multi-smooth additive REML fit.

    On the multi-block per-smooth-λ Rust path every field is an
    autograd-tracked tensor with a working backward through
    :class:`_GaussianRemlFitBlocksFn`:

    * ``coefficients``: list of ``F`` per-smooth blocks, each of shape
      ``(K_k, 1)``.
    * ``fitted``: joint fit ``(N, 1)``.
    * ``lambdas`` / ``log_lambdas``: length-``F`` 1D tensors of converged
      per-smooth smoothing parameters and their logs.
    * ``reml_score``: converged REML criterion (scalar tensor).
    * ``edf``: length-``F`` per-smooth EDFs.

    Multi-output block fits use the shared-scale block-orthogonal estimator
    and still report one ``λ`` / EDF per smooth.
    """

    coefficients: list[torch.Tensor]
    fitted: torch.Tensor
    lambdas: torch.Tensor
    log_lambdas: torch.Tensor
    reml_score: torch.Tensor
    edf: torch.Tensor

def _gaussian_reml_fit_blocks_orthogonal(
    design_blocks: list[torch.Tensor],
    penalty_blocks: list[torch.Tensor],
    y: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    init_log_lambdas: torch.Tensor | None = None,
) -> AdditiveRemlOutput:
    designs_np = [to_numpy_f64(d) for d in design_blocks]
    penalties_np = [to_numpy_f64(p) for p in penalty_blocks]
    y_np = to_numpy_f64(y.unsqueeze(1) if y.dim() == 1 else y)
    weights_np = None if weights is None else to_numpy_f64(weights)
    rhos_np = None if init_log_lambdas is None else to_numpy_f64(init_log_lambdas)

    out = _np_api.gaussian_reml_fit_blocks_orthogonal_forward(
        designs_np,
        penalties_np,
        y_np,
        weights=weights_np,
        init_rhos=rhos_np,
    )

    ref = design_blocks[0]
    coefficients = [
        from_numpy_like(np.asarray(coef, dtype=np.float64), ref)
        for coef in out["coefficients"]
    ]
    fitted = from_numpy_like(np.asarray(out["fitted"], dtype=np.float64), ref)
    lambdas = from_numpy_like(np.asarray(out["lambdas"], dtype=np.float64), ref)
    log_lambdas = from_numpy_like(
        np.asarray(out["log_lambdas"], dtype=np.float64), ref
    )
    reml_score = from_numpy_like(np.asarray(out["reml_score"], dtype=np.float64), ref)
    edf = from_numpy_like(np.asarray(out["edf"], dtype=np.float64), ref)
    return AdditiveRemlOutput(
        coefficients=coefficients,
        fitted=fitted,
        lambdas=lambdas,
        log_lambdas=log_lambdas,
        reml_score=reml_score,
        edf=edf,
    )


def _scalar_grad(grad: torch.Tensor | None) -> float:
    if grad is None:
        return 0.0
    return float(grad.detach())


def _batch_grad(grad: torch.Tensor | None) -> Any | None:
    if grad is None:
        return None
    return to_numpy_f64(grad)


def _wrap_optional(arr: Any, ref: torch.Tensor) -> torch.Tensor | None:
    if arr is None:
        return None
    return from_numpy_like(np.asarray(arr, dtype=np.float64), ref)


class _TorchRemlOps:
    """PyTorch binding of the framework-agnostic REML bridge contract.

    Implements :class:`gamfit._reml_common.RemlBridgeOps` so the shared
    point-design forward/backward marshalling can move torch tensors through
    the Rust engine without the bridge ever importing torch.
    """

    @staticmethod
    def to_numpy_f64(value: Any) -> Any:
        return to_numpy_f64(value)

    @staticmethod
    def to_numpy_uintp(value: Any) -> Any:
        return to_numpy_uintp(value)

    @staticmethod
    def from_numpy_like(array: Any, ref: Any) -> Any:
        return from_numpy_like(array, ref)

    @staticmethod
    def scalar_grad(grad: Any | None) -> float:
        return _scalar_grad(grad)

    @staticmethod
    def vector_grad(grad: Any | None) -> Any | None:
        return _batch_grad(grad)


_TORCH_REML_OPS = _TorchRemlOps()

# The single and ragged-batched point-design fits share a byte-for-byte
# identical marshalling contract; only the ``row_offsets`` positional arg and
# the scalar-vs-vector cotangent reduction differ. Both are expressed as
# :class:`RemlCallSpec`s and driven through one shared engine.
_SINGLE_SPEC = RemlCallSpec(
    name="gaussian_reml_fit",
    forward=_np_api.gaussian_reml_fit,
    backward=_np_api.gaussian_reml_fit_backward,
    batched=False,
)
_BATCHED_SPEC = RemlCallSpec(
    name="gaussian_reml_fit_batched",
    forward=_np_api.gaussian_reml_fit_batched,
    backward=_np_api.gaussian_reml_fit_batched_backward,
    batched=True,
)


def _save_diff_tensors(ctx: Any, *tensors: Any) -> None:
    """Save tensor inputs so PyTorch version-tracks them and raises on in-place
    mutation between forward and backward. ``None`` values are skipped silently;
    callers retain their own presence flags."""
    ctx.save_for_backward(*(t for t in tensors if isinstance(t, torch.Tensor)))


def _check_saved_versions(ctx: Any) -> None:
    """Touch ``ctx.saved_tensors`` so PyTorch raises if any saved input was
    modified in-place between forward and backward."""
    ctx.saved_tensors  # noqa: B018 — version-check side effect, value discarded


class _PointDesignRemlFn(torch.autograd.Function):
    """Autograd Function for a point-design closed-form Gaussian REML fit.

    A single class drives both the non-batched and ragged-batched variants.
    The variant is carried as the leading non-differentiable ``spec``
    argument, so the forward and backward marshalling — coercion, forward-state
    persistence, cotangent reduction, gradient routing — lives once in
    :mod:`gamfit._reml_common`. ``row_offsets`` is ``None`` for the single
    fit and the packed boundary vector for the batched fit; the positional
    layout is shared so both variants route through one Function.
    """

    @staticmethod
    def forward(
        ctx: Any,
        spec: RemlCallSpec,
        x: torch.Tensor,
        y: torch.Tensor,
        row_offsets: torch.Tensor | None,
        penalty: torch.Tensor,
        weights: torch.Tensor | None,
        by: torch.Tensor | None,
        init_lambda: float | None,
        by_start_col: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        run = run_point_design_forward(
            spec,
            _TORCH_REML_OPS,
            x=x,
            y=y,
            penalty=penalty,
            row_offsets=row_offsets,
            weights=weights,
            by=by,
            init_lambda=init_lambda,
            by_start_col=by_start_col,
            ref=x,
        )

        # Save the differentiable input tensors so PyTorch can version-check
        # them and raise on any in-place mutation between forward and backward.
        # Keep the numpy aliases (in ``run.arrays``) on ctx for performance —
        # they are only read in backward, which is gated by the version check.
        _save_diff_tensors(ctx, x, y, penalty, weights, by)
        ctx.spec = spec
        ctx.arrays = run.arrays
        ctx.forward_state = run.forward_state
        ctx.init_lambda = init_lambda
        ctx.by_start_col = by_start_col
        ctx.has_weights = weights is not None
        ctx.has_by = by is not None
        ctx.ref = x
        ctx.penalty_ref = penalty

        coefficients, fitted, lam, reml_score, edf = run.outputs
        return coefficients, fitted, lam, reml_score, edf

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[Any, ...]:
        _check_saved_versions(ctx)
        grads = run_point_design_backward(
            ctx.spec,
            _TORCH_REML_OPS,
            arrays=ctx.arrays,
            forward_state=ctx.forward_state,
            grad_outputs=cast(
                tuple[Any, Any, Any, Any, Any], grad_outputs
            ),
            init_lambda=ctx.init_lambda,
            by_start_col=ctx.by_start_col,
            has_weights=ctx.has_weights,
            has_by=ctx.has_by,
            ref=ctx.ref,
            penalty_ref=ctx.penalty_ref,
        )
        # Order matches forward(...) positional args:
        #   spec, x, y, row_offsets, penalty, weights, by, init_lambda,
        #   by_start_col. ``spec``, ``row_offsets``, ``init_lambda`` and
        #   ``by_start_col`` are non-differentiable.
        return (
            None,
            grads["grad_x"],
            grads["grad_y"],
            None,
            grads["grad_penalty"],
            grads["grad_weights"],
            grads["grad_by"],
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
    apply = cast(Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], _PointDesignRemlFn.apply)
    coefficients, fitted, lam, reml_score, edf = apply(
        _SINGLE_SPEC, x, y, None, penalty, weights, by, init_lambda, by_start_col
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
    apply = cast(Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], _PointDesignRemlFn.apply)
    coefficients, fitted, lam, reml_score, edf = apply(
        _BATCHED_SPEC, x, y, row_offsets, penalty, weights, by, init_lambda, by_start_col
    )
    return GaussianRemlOutput(coefficients, fitted, lam, reml_score, edf)


class _GaussianRemlFitBlocksFn(torch.autograd.Function):
    """Autograd Function for the multi-block per-smooth-λ Gaussian REML fit.

    Forward calls the Rust :func:`_np_api.gaussian_reml_fit_blocks_forward`
    multi-block joint REML driver. Backward calls the Rust analytic
    :func:`_np_api.gaussian_reml_fit_blocks_backward`, which composes the
    full VJP through the converged outer-loop optimum (envelope theorem +
    IFT through the F×F outer Hessian) so every returned tensor —
    coefficients, fitted values, λ, log-λ, REML score, EDF — is
    differentiable back to the design blocks, penalty blocks, y, and
    optional row weights.
    """

    @staticmethod
    def forward(
        ctx: Any,
        y: torch.Tensor,
        weights: torch.Tensor | None,
        init_log_lambdas: torch.Tensor | None,
        n_blocks: int,
        *block_tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        # ``block_tensors`` packs designs first (F entries), then penalties
        # (F entries). This positional layout is what lets PyTorch route
        # backward gradients elementwise onto each block tensor.
        if len(block_tensors) != 2 * n_blocks:
            raise RuntimeError(
                f"_GaussianRemlFitBlocksFn expected {2 * n_blocks} block tensors, "
                f"got {len(block_tensors)}"
            )
        designs = list(block_tensors[:n_blocks])
        penalties = list(block_tensors[n_blocks:])
        designs_np = [to_numpy_f64(d) for d in designs]
        penalties_np = [to_numpy_f64(p) for p in penalties]
        y_np = to_numpy_f64(y)
        weights_np = None if weights is None else to_numpy_f64(weights)
        rhos_np = None if init_log_lambdas is None else to_numpy_f64(init_log_lambdas)

        out = _np_api.gaussian_reml_fit_blocks_forward(
            designs_np,
            penalties_np,
            y_np,
            weights=weights_np,
            init_rhos=rhos_np,
        )

        ref = designs[0]
        coefs_full = np.asarray(out["coefficients"], dtype=np.float64)  # (P_total, 1)
        fitted_arr = np.asarray(out["fitted"], dtype=np.float64)
        lambdas_arr = np.asarray(out["lambdas"], dtype=np.float64)
        log_lambdas_arr = np.asarray(out["log_lambdas"], dtype=np.float64)
        reml_score_val = float(out["reml_score"])
        edf_arr = np.asarray(out["edf"], dtype=np.float64)

        # Save numpy aliases on ctx for the backward pass. Saving torch
        # tensors via save_for_backward gives us version-tracking and
        # prevents silent in-place mutation between forward and backward.
        _save_diff_tensors(ctx, y, weights, *designs, *penalties)
        ctx.designs_np = designs_np
        ctx.penalties_np = penalties_np
        ctx.y_np = y_np
        ctx.weights_np = weights_np
        ctx.log_lambdas_np = log_lambdas_arr
        ctx.n_blocks = n_blocks
        ctx.has_weights = weights is not None
        ctx.has_init = init_log_lambdas is not None
        ctx.ref = ref

        offsets: list[int] = [0]
        for d in designs:
            offsets.append(offsets[-1] + int(d.shape[1]))
        ctx.offsets = offsets

        fitted_t = from_numpy_like(fitted_arr, ref)
        lambdas_t = from_numpy_like(lambdas_arr, ref)
        log_lambdas_t = from_numpy_like(log_lambdas_arr, ref)
        reml_t = from_numpy_like(np.asarray(reml_score_val, dtype=np.float64), ref)
        edf_t = from_numpy_like(edf_arr, ref)

        # The full coefficient vector is returned as a single tensor so
        # gradient routing stays simple: a single ``grad_coefficients``
        # tensor goes back through the Rust analytic backward. We split
        # it back per-block downstream of this Function.
        coefs_t = from_numpy_like(coefs_full, ref)

        return coefs_t, fitted_t, lambdas_t, log_lambdas_t, reml_t, edf_t

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[Any, ...]:
        (
            grad_coefficients,
            grad_fitted,
            grad_lambdas,
            grad_log_lambdas,
            grad_reml_score,
            grad_edf,
        ) = grad_outputs
        _check_saved_versions(ctx)
        gc = None if grad_coefficients is None else to_numpy_f64(grad_coefficients)
        gf = None if grad_fitted is None else to_numpy_f64(grad_fitted)
        gl = None if grad_lambdas is None else to_numpy_f64(grad_lambdas)
        glog = None if grad_log_lambdas is None else to_numpy_f64(grad_log_lambdas)
        gr_scalar = _scalar_grad(grad_reml_score)
        ge = None if grad_edf is None else to_numpy_f64(grad_edf)

        result = _np_api.gaussian_reml_fit_blocks_backward(
            ctx.designs_np,
            ctx.penalties_np,
            ctx.y_np,
            ctx.log_lambdas_np,
            weights=ctx.weights_np,
            grad_coefficients=gc,
            grad_fitted=gf,
            grad_lambdas=gl,
            grad_log_lambdas=glog,
            grad_reml_score=gr_scalar,
            grad_edf=ge,
        )

        ref = ctx.ref
        grad_designs: list[torch.Tensor | None] = [
            from_numpy_like(np.asarray(g, dtype=np.float64), ref)
            for g in result["grad_designs"]
        ]
        grad_penalties: list[torch.Tensor | None] = [
            from_numpy_like(np.asarray(g, dtype=np.float64), ref)
            for g in result["grad_penalties"]
        ]
        grad_y = from_numpy_like(np.asarray(result["grad_y"], dtype=np.float64), ref)
        grad_weights = (
            from_numpy_like(np.asarray(result["grad_weights"], dtype=np.float64), ref)
            if ctx.has_weights
            else None
        )

        # Match positional order of forward(...):
        #   y, weights, init_log_lambdas, n_blocks, *designs, *penalties
        # init_log_lambdas / n_blocks are not differentiable through this Function.
        return (
            grad_y,
            grad_weights,
            None,
            None,
            *grad_designs,
            *grad_penalties,
        )


def gaussian_reml_fit_blocks(
    design_blocks: list[torch.Tensor],
    penalty_blocks: list[torch.Tensor],
    y: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    init_log_lambdas: torch.Tensor | None = None,
) -> AdditiveRemlOutput:
    """Multi-block Gaussian REML forward fit with per-smooth λ.

    Single-output fits use the Rust ``optimize_external_design`` multi-block
    outer-loop (Gaussian identity link). Multi-output fits use the internal
    shared-scale block-orthogonal estimator: one λ per block, one profiled
    residual scale per output column, and no single-λ block-diagonal
    collapse.

    Parameters
    ----------
    design_blocks:
        List of ``F`` per-smooth design matrices, each of shape ``(N, K_k)``.
    penalty_blocks:
        List of ``F`` per-smooth penalty matrices, each of shape ``(K_k, K_k)``.
    y:
        Response tensor of shape ``(N,)`` or ``(N, D)``.
    weights:
        Optional row weights of shape ``(N,)``.
    init_log_lambdas:
        Optional warm-start log-λ vector of shape ``(F,)`` (one entry per
        block). When omitted the driver chooses its own seeds.

    Returns
    -------
    AdditiveRemlOutput
        ``coefficients`` is a list of ``F`` per-block ``(K_k, D)`` tensors;
        ``fitted`` is ``(N, D)``; ``lambdas``/``log_lambdas`` are length-``F``
        1D tensors of per-smooth λ and log-λ; ``reml_score`` is the
        converged REML criterion (scalar); ``edf`` is length-``F``.
    """
    if len(design_blocks) != len(penalty_blocks):
        raise ValueError(
            f"design_blocks and penalty_blocks must have equal length; got "
            f"{len(design_blocks)} vs {len(penalty_blocks)}"
        )
    if len(design_blocks) == 0:
        raise ValueError("gaussian_reml_fit_blocks requires at least one block")
    if y.dim() == 1:
        y_mat = y.unsqueeze(1)
    elif y.dim() == 2:
        y_mat = y
    else:
        raise ValueError(f"y must be 1D or 2D; got shape {tuple(y.shape)}")
    if y_mat.shape[1] != 1:
        return _gaussian_reml_fit_blocks_orthogonal(
            design_blocks,
            penalty_blocks,
            y_mat,
            weights=weights,
            init_log_lambdas=init_log_lambdas,
        )

    n_blocks = len(design_blocks)
    apply = cast(
        Callable[..., tuple[torch.Tensor, ...]],
        _GaussianRemlFitBlocksFn.apply,
    )
    coefs_full, fitted_t, lambdas_t, log_lambdas_t, reml_t, edf_t = apply(
        y_mat,
        weights,
        init_log_lambdas,
        n_blocks,
        *design_blocks,
        *penalty_blocks,
    )

    # Split the flat coefficient tensor back per-block at the column
    # offsets of the joint design. Slicing keeps autograd connected, so
    # downstream losses on individual blocks still backprop through the
    # multi-block Function.
    offsets: list[int] = [0]
    for d in design_blocks:
        offsets.append(offsets[-1] + int(d.shape[1]))
    per_block_coefs: list[torch.Tensor] = [
        coefs_full[offsets[i]:offsets[i + 1], :] for i in range(n_blocks)
    ]

    return AdditiveRemlOutput(
        coefficients=per_block_coefs,
        fitted=fitted_t,
        lambdas=lambdas_t,
        log_lambdas=log_lambdas_t,
        reml_score=reml_t,
        edf=edf_t,
    )


class _GaussianRemlFitWithConstraintsFn(torch.autograd.Function):
    """Autograd Function for the constrained Gaussian REML fit.

    Backward uses the envelope-theorem-based analytic VJP exposed by Rust:

    * **Interior cert (empty active set):** the constrained fit coincides
      with the unconstrained one and the VJP is the closed-form Gaussian
      REML backward in full p-space.
    * **Active cert (non-empty active set):** the math identity is
      ``H⁻¹ → Z(ZᵀHZ)⁻¹Zᵀ``, ``S⁺ → Z(ZᵀSZ)⁺Zᵀ`` with ``Z = null(A_act)``.
      The constrained problem is reduced to the ``Z`` subspace, the same
      closed-form backward runs on the reduced operators, and the gradients
      are lifted back to p-space. Both certs return exact gradients;
      ``.backward()`` no longer raises ``NotImplementedError``.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        y: torch.Tensor,
        penalty: torch.Tensor,
        weights: torch.Tensor | None,
        a_inequality: torch.Tensor | None,
        b_inequality: torch.Tensor | None,
        init_log_lambda: float | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        x_np = to_numpy_f64(x)
        y_np = to_numpy_f64(y)
        penalty_np = to_numpy_f64(penalty)
        weights_np = None if weights is None else to_numpy_f64(weights)
        a_np = None if a_inequality is None else to_numpy_f64(a_inequality)
        b_np = None if b_inequality is None else to_numpy_f64(b_inequality)

        out = _np_api.gaussian_reml_fit_with_constraints_forward(
            x_np,
            y_np,
            penalty_np,
            weights=weights_np,
            init_log_lambda=init_log_lambda,
            a_inequality=a_np,
            b_inequality=b_np,
        )

        _save_diff_tensors(ctx, x, y, penalty, weights, a_inequality, b_inequality)
        # Cache numpy aliases and converged-state for the analytic VJP.
        ctx.x_np = x_np
        ctx.y_np = y_np
        ctx.penalty_np = penalty_np
        ctx.weights_np = weights_np
        ctx.a_np = a_np
        ctx.b_np = b_np
        ctx.has_weights = weights is not None
        ctx.coefficients_np = np.asarray(out["coefficients"], dtype=np.float64)
        ctx.fitted_np = np.asarray(out["fitted"], dtype=np.float64)
        ctx.log_lambda_at_optimum = float(out["log_lambda"])
        # Recompute the true active set honestly from |A·β̂ - b|: the
        # `active_indices` field returned by the Rust forward uses a
        # feasibility-style threshold and overcounts inactive feasible
        # rows. The analytic VJP branches on the *true* active set
        # (empty ⇒ envelope/interior cert path; non-empty ⇒ deferred
        # tangent-projection path), so doing the test ourselves keeps
        # the contract robust to that quirk.
        if a_np is None or a_np.shape[0] == 0:
            ctx.active_indices_np = np.zeros(0, dtype=np.uint64)
        else:
            beta_flat = ctx.coefficients_np.reshape(-1)
            slack = a_np @ beta_flat - (b_np if b_np is not None else np.zeros(a_np.shape[0]))
            scale = max(1.0, float(np.abs(beta_flat).max()))
            tol = 1e-7 * scale
            true_active = np.where(np.abs(slack) <= tol)[0]
            ctx.active_indices_np = np.asarray(true_active, dtype=np.uint64)
        ctx.ref = x
        ctx.penalty_ref = penalty

        coefficients = from_numpy_like(out["coefficients"], x)
        fitted = from_numpy_like(out["fitted"], x)
        lam = from_numpy_like(np.asarray(out["lambda"], dtype=np.float64), x)
        log_lambda = from_numpy_like(
            np.asarray(out["log_lambda"], dtype=np.float64), x
        )
        reml_score = from_numpy_like(
            np.asarray(out["reml_score"], dtype=np.float64), x
        )
        edf = from_numpy_like(np.asarray(out["edf"], dtype=np.float64), x)
        active_indices_np = np.asarray(out["active_indices"], dtype=np.int64)
        active_indices = torch.as_tensor(
            active_indices_np, dtype=torch.int64, device=x.device
        )
        return coefficients, fitted, lam, log_lambda, reml_score, edf, active_indices

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[Any, ...]:
        (
            grad_coefficients,
            grad_fitted,
            grad_lam,
            grad_log_lambda,
            grad_reml_score,
            grad_edf,
            _grad_active_indices,  # int64 indices — not differentiable
        ) = grad_outputs
        _check_saved_versions(ctx)

        grad_coef_np = (
            None if grad_coefficients is None else to_numpy_f64(grad_coefficients)
        )
        grad_fitted_np = None if grad_fitted is None else to_numpy_f64(grad_fitted)
        grad_lambda_scalar = _scalar_grad(grad_lam)
        grad_log_lambda_scalar = _scalar_grad(grad_log_lambda)
        grad_reml_scalar = _scalar_grad(grad_reml_score)
        grad_edf_scalar = _scalar_grad(grad_edf)

        result = _np_api.gaussian_reml_fit_with_constraints_backward(
            ctx.x_np,
            ctx.y_np,
            ctx.penalty_np,
            weights=ctx.weights_np,
            a_inequality=ctx.a_np,
            b_inequality=ctx.b_np,
            log_lambda_at_optimum=ctx.log_lambda_at_optimum,
            coefficients_at_optimum=ctx.coefficients_np,
            fitted_at_optimum=ctx.fitted_np,
            active_indices=ctx.active_indices_np,
            grad_coefficients=grad_coef_np,
            grad_fitted=grad_fitted_np,
            grad_lambda=grad_lambda_scalar,
            grad_log_lambda=grad_log_lambda_scalar,
            grad_reml_score=grad_reml_scalar,
            grad_edf=grad_edf_scalar,
        )

        ref = ctx.ref
        grad_x = _wrap_optional(result.get("grad_x"), ref)
        grad_y = _wrap_optional(result.get("grad_y"), ref)
        grad_penalty = _wrap_optional(result.get("grad_penalty"), ctx.penalty_ref)
        grad_weights = (
            _wrap_optional(result.get("grad_weights"), ref) if ctx.has_weights else None
        )
        # Order matches forward(...) positional args: x, y, penalty, weights,
        # a_inequality, b_inequality, init_log_lambda. The constraint
        # geometry (A, b) and the init are non-differentiable.
        return grad_x, grad_y, grad_penalty, grad_weights, None, None, None


class ConstrainedRemlOutput(NamedTuple):
    """Forward outputs for the constrained Gaussian REML fit."""

    coefficients: torch.Tensor
    fitted: torch.Tensor
    lam: torch.Tensor
    log_lambda: torch.Tensor
    reml_score: torch.Tensor
    edf: torch.Tensor
    active_indices: torch.Tensor


def gaussian_reml_fit_with_constraints(
    x: torch.Tensor,
    y: torch.Tensor,
    penalty: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    init_log_lambda: float | None = None,
    a_inequality: torch.Tensor | None = None,
    b_inequality: torch.Tensor | None = None,
) -> ConstrainedRemlOutput:
    """Constrained Gaussian REML fit, fully differentiable.

    Routes a single-block design ``x``/penalty ``penalty`` with an optional
    linear inequality system ``A·β ≥ b`` through the same Rust active-set
    + REML driver used by the formula-API shape constraints. ``.backward()``
    returns exact analytic gradients w.r.t. ``x``, ``y``, ``penalty`` and
    ``weights`` in both the interior cert (envelope theorem in full p-space)
    and the active cert (tangent-projected ``Z = null(A_act)`` reduction).
    """
    apply = cast(
        Callable[
            ...,
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
        ],
        _GaussianRemlFitWithConstraintsFn.apply,
    )
    coefficients, fitted, lam, log_lambda, reml_score, edf, active_indices = apply(
        x, y, penalty, weights, a_inequality, b_inequality, init_log_lambda,
    )
    return ConstrainedRemlOutput(
        coefficients=coefficients,
        fitted=fitted,
        lam=lam,
        log_lambda=log_lambda,
        reml_score=reml_score,
        edf=edf,
        active_indices=active_indices,
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
    each smooth recovers its own ``λ_k`` from the outer optimisation. The
    multi-block path is now fully differentiable: backward composes the
    closed-form Gaussian VJPs and the IFT through the F×F smoothing
    adjoint system.

    Multi-output response (``D > 1``) uses the internal shared-scale
    block-orthogonal estimator, preserving one λ per smooth instead of
    falling back to one scalar λ for the whole block-diagonal penalty.

    Both paths return fully differentiable tensors.

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
        Optional initial smoothing parameter, broadcast to every smooth.

    Returns
    -------
    AdditiveRemlOutput
        ``coefficients`` is a list of length ``F`` with the per-smooth
        blocks; ``fitted`` is the joint ``(N, D)`` reconstruction;
        ``lambdas``/``log_lambdas``/``edf`` are length-``F`` vectors.
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
            if by_i is None:
                raise RuntimeError("by_i is None")  # narrow for type-checkers
            if by_i.dim() != 1 or by_i.shape[0] != design.shape[0]:
                raise ValueError(
                    f"bys[{i}] must be 1D of length N={design.shape[0]}; got {tuple(by_i.shape)}"
                )
            modulated.append(design * by_i.unsqueeze(1))
        else:
            modulated.append(design)

    # Route single-response additive fits to the exact dense multi-block
    # Rust path. Multi-output uses the scalable shared-scale orthogonal
    # additive estimator so it keeps one λ per smooth instead of collapsing
    # to a single block-diagonal λ.
    response_is_2d = response.dim() == 2 and response.shape[1] > 1
    if not response_is_2d:
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

    init_log = None
    if init_lambda is not None:
        init_log = torch.full(
            (len(designs),),
            float(np.log(max(float(init_lambda), 1e-300))),
            dtype=torch.float64,
            device=modulated[0].device,
        )
    return _gaussian_reml_fit_blocks_orthogonal(
        modulated,
        list(penalties),
        response,
        weights=weights,
        init_log_lambdas=init_log,
    )
