"""Unified GAM fit for torch — one entry point for any smooth, any dimensionality.

The user describes smooth-term specs (:class:`gamfit.Smooth` subclasses) and
calls :func:`fit`. The library constructs the right basis matrices and
penalty matrices internally per spec, dispatches to Gaussian REML
(single-smooth or joint additive depending on input shape), and returns a
:class:`FitResult` carrying coefficients, fitted values, per-smooth λ, and
REML score. The user never constructs a penalty matrix.

Response is multi-output by default — pass ``(N,)`` for scalar response,
``(N, D)`` for matrix-valued. Coefficients come back with the matching last
dim. Per-smooth ``by`` gating is applied internally.

Autograd flows back to ``points``, ``by``, and ``response`` through the
engine's analytic VJP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ..smooth import (
    BSpline,
    Categorical,
    Duchon,
    Matern,
    Sphere,
    Smooth,
    TensorBSpline,
)
from ._basis import bspline_basis, duchon_basis
from ._reml import (
    AdditiveRemlOutput,
    GaussianRemlOutput,
    gaussian_reml_fit,
    gaussian_reml_fit_additive,
)


# ---------------------------------------------------------------------------
# FitResult — the user-facing output
# ---------------------------------------------------------------------------


@dataclass
class FitResult:
    """Output of :func:`gamfit.torch.fit`.

    Single smooth (``smooths`` was a :class:`Smooth`):

    * ``coefficients`` — ``Tensor`` of shape ``(M, D)``
    * ``edf`` — scalar

    Multi-smooth additive (``smooths`` was a list):

    * ``coefficients`` — ``list[Tensor]``, one per smooth, shape ``(M_k, D)``
    * ``edf`` — ``Tensor`` shape ``(F,)``

    Always:

    * ``fitted`` — ``Tensor`` shape ``(N, D)``
    * ``lambdas`` — scalar (single) or ``Tensor`` shape ``(F,)`` (additive)
    * ``reml_score`` — scalar
    * ``smooths`` — echo of the input specs (for downstream indexing)
    """

    coefficients: torch.Tensor | list[torch.Tensor]
    fitted: torch.Tensor
    lambdas: torch.Tensor
    reml_score: torch.Tensor
    edf: torch.Tensor
    smooths: list[Smooth]


# ---------------------------------------------------------------------------
# Internal: per-smooth-kind dispatch to (design, penalty)
# ---------------------------------------------------------------------------


def _to_tensor(value, like: torch.Tensor) -> torch.Tensor:
    """Coerce an array-like (numpy ndarray / torch tensor / list) to a torch
    tensor matching ``like``'s device. dtype stays float64 for REML."""
    if isinstance(value, torch.Tensor):
        return value.to(device=like.device)
    return torch.as_tensor(value, dtype=torch.float64, device=like.device)


def _coerce_2d(t: torch.Tensor, name: str) -> torch.Tensor:
    """Promote ``(N,)`` to ``(N, 1)`` and validate 2D shape."""
    if t.dim() == 1:
        return t.unsqueeze(1)
    if t.dim() != 2:
        raise ValueError(f"{name} must be 1D or 2D, got {t.dim()}D shape {tuple(t.shape)}")
    return t


def _build_design_penalty(
    smooth: Smooth, points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the (design, penalty) pair for one smooth at given input points.

    The penalty matrix lives in the same M-dim coefficient space as the
    design's column space. Multi-d penalties are sourced from gamfit's
    primitive functions per smooth kind.

    Returns (design (N, M), penalty (M, M)) as float64 torch tensors.
    """
    from .. import duchon_function_norm_penalty

    points = _coerce_2d(points, "points")
    N = points.shape[0]

    if isinstance(smooth, Duchon):
        centers = _coerce_2d(_to_tensor(smooth.centers, points), "Duchon.centers")
        if centers.shape[1] != points.shape[1]:
            raise ValueError(
                f"Duchon: points d={points.shape[1]} but centers d={centers.shape[1]}"
            )
        per = (
            tuple(bool(p) for p in smooth.periodic_per_axis)
            if smooth.periodic_per_axis is not None
            else None
        )
        design = duchon_basis(points, centers, m=smooth.m, periodic_per_axis=per)
        try:
            penalty_np = duchon_function_norm_penalty(
                centers.detach().cpu().numpy(),
                m=smooth.m,
                periodic_per_axis=per,
            )
        except NotImplementedError as exc:
            raise NotImplementedError(
                f"Duchon penalty for d={centers.shape[1]} not yet exposed by gam-pyffi. "
                f"d=1 works; multi-d penalty binding pending. Underlying error: {exc}"
            ) from exc
        penalty = torch.as_tensor(penalty_np, dtype=torch.float64, device=points.device)
        return design.to(torch.float64), penalty

    if isinstance(smooth, BSpline):
        if points.shape[1] != 1:
            raise ValueError(
                f"BSpline is 1D-only; got points with d={points.shape[1]}. "
                "Use TensorBSpline for multi-d with different units, or Duchon for radial."
            )
        knots = (
            _to_tensor(smooth.knots, points).reshape(-1)
            if smooth.knots is not None
            else None
        )
        design = bspline_basis(
            points.squeeze(1), knots, degree=smooth.degree, periodic=smooth.periodic,
        )
        from .._api import smoothness_penalty as _smoothness_penalty
        knots_np = (
            knots.detach().cpu().numpy()
            if knots is not None
            else None
        )
        if knots_np is None:
            from .._api import _resolve_knots
            knots_np = _resolve_knots(None, points.squeeze(1).detach().cpu().numpy(),
                                      label="knots", degree=smooth.degree)
        penalty_np, _null_basis = _smoothness_penalty(
            knots_np, degree=smooth.degree, order=smooth.penalty_order,
        )
        penalty = torch.as_tensor(penalty_np, dtype=torch.float64, device=points.device)
        return design.to(torch.float64), penalty

    if isinstance(smooth, (TensorBSpline, Matern, Sphere, Categorical)):
        raise NotImplementedError(
            f"{type(smooth).__name__} not yet wired to gamfit.torch.fit; "
            "needs Rust PyO3 binding for the underlying basis + penalty. "
            "Currently supported: Duchon (any d for basis; d=1 for penalty), BSpline (d=1)."
        )

    raise TypeError(f"unknown Smooth subclass: {type(smooth).__name__}")


# ---------------------------------------------------------------------------
# Public: fit()
# ---------------------------------------------------------------------------


def fit(
    points: torch.Tensor | Sequence[torch.Tensor],
    response: torch.Tensor,
    smooths: Smooth | Sequence[Smooth],
    *,
    weights: torch.Tensor | None = None,
    init_lambdas: torch.Tensor | None = None,
) -> FitResult:
    """Fit one or more smooths against a multi-dimensional response.

    Parameters
    ----------
    points : ``Tensor`` of shape ``(N, d)`` (or ``(N,)`` for d=1), OR a
        list of per-smooth tensors when ``smooths`` is a list. Each
        per-smooth tensor's dimensionality must match its smooth's
        ``centers`` / ``knots`` shape.
    response : ``Tensor`` of shape ``(N,)`` or ``(N, D)``. Multi-output
        supported; the smooth structure is shared across all D outputs.
        Coefficients come back with matching last dim.
    smooths : a single :class:`Smooth` (single-smooth fit) or a list
        (joint multi-smooth additive fit).
    weights : optional ``Tensor`` of shape ``(N,)``. Per-row weights for
        weighted REML.
    init_lambdas : optional initial λ values. Scalar for single smooth,
        ``(F,)`` for additive. Defaults to gamfit's automatic init.

    Returns
    -------
    :class:`FitResult` with coefficients, fitted values, per-smooth λ,
    REML score, and effective DoF.

    Examples
    --------
    Single Duchon smooth, 1D positions, scalar response::

        >>> result = fit(t, y, Duchon(centers=c, m=2))

    Single Duchon, 3D positions (RGB), scalar response::

        >>> result = fit(rgb_points, y, Duchon(centers=c_3d, m=2))

    Multi-output response (residual stream)::

        >>> result = fit(positions, residual_stream_NxD, Duchon(centers=c, m=2))
        >>> # result.coefficients.shape == (K, D)

    Additive fit — F atoms with per-atom amplitude gating::

        >>> result = fit(
        ...     points=[positions[:, k:k+1] for k in range(F)],
        ...     response=x_centered,  # (N, D)
        ...     smooths=[
        ...         Duchon(centers=c, m=2, by=amp[:, k])
        ...         for k in range(F)
        ...     ],
        ... )
    """
    # Normalize: response always (N, D)
    if response.dim() == 1:
        response = response.unsqueeze(1)
    if response.dim() != 2:
        raise ValueError(f"response must be 1D or 2D, got shape {tuple(response.shape)}")
    N = response.shape[0]
    response_f64 = response.to(torch.float64)

    weights_f64 = (
        weights.to(torch.float64).reshape(-1) if weights is not None else None
    )

    # Branch: single smooth vs list of smooths
    if isinstance(smooths, Smooth):
        if isinstance(points, (list, tuple)):
            raise ValueError(
                "got a list of points but a single smooth; pass one points tensor."
            )
        design, penalty = _build_design_penalty(smooths, points)
        by_t = (
            _to_tensor(smooths.by, design).reshape(-1)
            if smooths.by is not None else None
        )
        out: GaussianRemlOutput = gaussian_reml_fit(
            design, response_f64, penalty,
            weights=weights_f64, by=by_t,
            init_lambda=float(init_lambdas) if init_lambdas is not None else None,
        )
        return FitResult(
            coefficients=out.coefficients,
            fitted=out.fitted,
            lambdas=out.lam,
            reml_score=out.reml_score,
            edf=out.edf,
            smooths=[smooths],
        )

    smooths_list = list(smooths)
    if len(smooths_list) == 0:
        raise ValueError("smooths must contain at least one Smooth")
    if not all(isinstance(s, Smooth) for s in smooths_list):
        bad = [type(s).__name__ for s in smooths_list if not isinstance(s, Smooth)]
        raise TypeError(f"all entries must be Smooth, got: {bad}")

    # Per-smooth points
    if isinstance(points, (list, tuple)):
        points_list = list(points)
        if len(points_list) != len(smooths_list):
            raise ValueError(
                f"got {len(points_list)} points tensors but {len(smooths_list)} smooths"
            )
    else:
        # Same points for every smooth.
        points_list = [points] * len(smooths_list)

    designs: list[torch.Tensor] = []
    penalties: list[torch.Tensor] = []
    bys: list[torch.Tensor | None] = []
    for s, pts in zip(smooths_list, points_list):
        design, penalty = _build_design_penalty(s, pts)
        designs.append(design)
        penalties.append(penalty)
        bys.append(_to_tensor(s.by, design).reshape(-1) if s.by is not None else None)

    init_lam = float(init_lambdas[0]) if init_lambdas is not None else None
    add_out: AdditiveRemlOutput = gaussian_reml_fit_additive(
        designs=designs,
        response=response_f64,
        penalties=penalties,
        bys=bys,
        weights=weights_f64,
        init_lambda=init_lam,
    )
    return FitResult(
        coefficients=list(add_out.coefficients),
        fitted=add_out.fitted,
        lambdas=add_out.lam,
        reml_score=add_out.reml_score,
        edf=add_out.edf,
        smooths=smooths_list,
    )


__all__ = ["fit", "FitResult"]
