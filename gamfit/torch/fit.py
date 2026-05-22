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
    PeriodicSplineCurve,
    Sphere,
    Smooth,
    TensorBSpline,
)
from ._basis import bspline_basis, duchon_basis, periodic_spline_curve_basis, sphere_basis
from ._reml import (
    AdditiveRemlOutput,
    GaussianRemlOutput,
    gaussian_reml_fit,
    gaussian_reml_fit_additive,
    gaussian_reml_fit_with_constraints,
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
        if knots is not None:
            knots_np = knots.detach().cpu().to(torch.float64).numpy()
        else:
            from .._api import _resolve_knots
            knots_np = _resolve_knots(
                None,
                points.squeeze(1).detach().cpu().to(torch.float64).numpy(),
                label="knots", degree=smooth.degree,
            )
        penalty_np, _null_basis = _smoothness_penalty(
            knots_np, degree=smooth.degree, order=smooth.penalty_order,
        )
        penalty = torch.as_tensor(penalty_np, dtype=torch.float64, device=points.device)
        return design.to(torch.float64), penalty

    if isinstance(smooth, Sphere):
        if points.shape[1] != 2:
            raise ValueError(
                f"Sphere expects points of shape (N, 2) [lat, lon]; got d={points.shape[1]}"
            )
        if not torch.isfinite(points).all():
            raise ValueError("Sphere: points contains NaN/Inf")
        lat = points[:, 0]
        if smooth.radians:
            import math
            bound = math.pi / 2.0
            if (lat.min().item() < -bound - 1e-9) or (lat.max().item() > bound + 1e-9):
                raise ValueError(
                    "Sphere(radians=True): latitude must lie in [-π/2, π/2]"
                )
        else:
            if (lat.min().item() < -90.0 - 1e-9) or (lat.max().item() > 90.0 + 1e-9):
                raise ValueError(
                    "Sphere(radians=False): latitude must lie in [-90, 90]"
                )
        design, penalty = sphere_basis(
            points,
            n_centers=smooth.n_centers,
            penalty_order=smooth.penalty_order,
            kernel=smooth.kernel,
            radians=smooth.radians,
        )
        return design.to(torch.float64), penalty.to(torch.float64)

    if isinstance(smooth, PeriodicSplineCurve):
        if points.shape[1] != 1:
            raise ValueError(
                f"PeriodicSplineCurve expects 1D parameter t with shape (N,) or "
                f"(N, 1); got d={points.shape[1]}"
            )
        t1d = points.squeeze(1)
        design, penalty = periodic_spline_curve_basis(
            t1d,
            n_knots=smooth.n_knots,
            degree=smooth.degree,
            penalty_order=smooth.penalty_order,
        )
        return design.to(torch.float64), penalty.to(torch.float64)

    if isinstance(smooth, (TensorBSpline, Matern, Categorical)):
        raise NotImplementedError(
            f"{type(smooth).__name__} not yet wired to gamfit.torch.fit; "
            "needs Rust PyO3 binding for the underlying basis + penalty. "
            "Currently supported: Duchon (any d for basis; d=1 for penalty), "
            "BSpline (d=1), Sphere (S²)."
        )

    raise TypeError(f"unknown Smooth subclass: {type(smooth).__name__}")


# ---------------------------------------------------------------------------
# Shape constraints — build A from the smooth's design on a 1D grid
# ---------------------------------------------------------------------------


def _shape_constraint_grid_1d(x: torch.Tensor) -> torch.Tensor:
    """Replicate the Rust grid for 1D shape constraints.

    Build a uniform grid of ``clamp(unique_count, 96, 320)`` points spanning
    ``[min(x), max(x)]``. Constraint feasibility on this grid implies the
    shape constraint on the smooth's image under the usual B-spline / RBF
    density argument.
    """
    if x.dim() != 1:
        raise ValueError(
            f"shape constraint grid requires a 1D location tensor; got shape "
            f"{tuple(x.shape)}"
        )
    if not torch.isfinite(x).all():
        raise ValueError("shape constraint requires finite covariate values")
    x_min = float(x.min().item())
    x_max = float(x.max().item())
    if x_max - x_min <= 1e-12 * max(abs(x_min), abs(x_max), 1.0):
        raise ValueError(
            "shape-constrained smooth requires a non-degenerate covariate range"
        )
    # Match Rust: clamp(unique_count, 96, 320). Approximate unique-count with
    # nunique via torch.unique for the common case; cheap on the small N
    # we see in 1D smooth fits.
    unique_count = int(torch.unique(x).numel())
    target = max(96, min(320, unique_count))
    grid = torch.linspace(x_min, x_max, target, dtype=torch.float64, device=x.device)
    return grid


def _build_shape_constraint_inequality(
    smooth: Smooth, points: torch.Tensor, shape_kind: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(A, b)`` for the inequality system ``A·β ≥ b`` enforcing
    ``shape_kind`` on ``smooth``'s 1D basis at a dense grid over the data
    range. Returns finite-difference rows on the grid:

    * ``monotone_increasing``  → ``(B_grid[i+1] - B_grid[i]) · β ≥ 0``
    * ``monotone_decreasing``  → negation of the above
    * ``convex``               → ``(B_grid[i+2] - 2 B_grid[i+1] + B_grid[i]) · β ≥ 0``
    * ``concave``              → negation of the above
    """
    points = _coerce_2d(points, "points")
    if points.shape[1] != 1:
        raise NotImplementedError(
            "shape_constraint on the torch path requires a 1D covariate (d==1); "
            f"got d={points.shape[1]}. Multidimensional shape constraints are "
            "not supported on the torch path."
        )
    grid_1d = _shape_constraint_grid_1d(points.squeeze(1))

    if isinstance(smooth, BSpline):
        knots = (
            _to_tensor(smooth.knots, points).reshape(-1)
            if smooth.knots is not None
            else None
        )
        b_grid = bspline_basis(
            grid_1d, knots, degree=smooth.degree, periodic=smooth.periodic,
        ).to(torch.float64).detach()
    elif isinstance(smooth, Duchon):
        if isinstance(smooth, Duchon) and (
            (hasattr(smooth, "centers") and _to_tensor(smooth.centers, points).reshape(
                _to_tensor(smooth.centers, points).shape[0], -1).shape[1] != 1)
        ):
            raise NotImplementedError(
                "shape_constraint on torch Duchon path requires 1D centers."
            )
        centers = _coerce_2d(_to_tensor(smooth.centers, points), "Duchon.centers")
        per = (
            tuple(bool(p) for p in smooth.periodic_per_axis)
            if smooth.periodic_per_axis is not None
            else None
        )
        b_grid = duchon_basis(
            grid_1d.unsqueeze(1), centers, m=smooth.m, periodic_per_axis=per,
        ).to(torch.float64).detach()
    else:
        raise NotImplementedError(
            f"shape_constraint not supported on the torch path for "
            f"{type(smooth).__name__}; supported: BSpline, Duchon (d=1)."
        )

    sk = shape_kind.lower()
    if sk in ("monotone_increasing", "monotone_decreasing"):
        diff = b_grid[1:] - b_grid[:-1]
        if sk == "monotone_decreasing":
            diff = -diff
        a = diff
    elif sk in ("convex", "concave"):
        d2 = b_grid[2:] - 2.0 * b_grid[1:-1] + b_grid[:-2]
        if sk == "concave":
            d2 = -d2
        a = d2
    else:
        raise ValueError(
            f"unknown shape_constraint kind {shape_kind!r}; expected one of "
            "monotone_increasing, monotone_decreasing, convex, concave"
        )

    # Drop near-zero rows (matches Rust's norm>1e-12 cull).
    row_norms = a.norm(dim=1)
    keep = row_norms > 1e-12
    a = a[keep].contiguous()
    b = torch.zeros(a.shape[0], dtype=torch.float64, device=a.device)
    return a, b


def _fit_single_constrained(
    smooth: Smooth,
    points: torch.Tensor,
    response: torch.Tensor,
    *,
    weights: torch.Tensor | None,
    shape_kind: str,
    init_lambdas: torch.Tensor | None,
) -> "FitResult":
    design, penalty = _build_design_penalty(smooth, points)
    a_ineq, b_ineq = _build_shape_constraint_inequality(
        smooth, points, shape_kind,
    )
    if smooth.by is not None:
        raise NotImplementedError(
            "shape_constraint combined with `by` modulation is not supported "
            "on the torch path."
        )
    weights_f64 = (
        weights.to(torch.float64).reshape(-1) if weights is not None else None
    )
    init_log = None
    if init_lambdas is not None:
        import math as _math
        init_log = _math.log(max(float(init_lambdas), 1e-300))
    out = gaussian_reml_fit_with_constraints(
        design.to(torch.float64),
        response.to(torch.float64),
        penalty.to(torch.float64),
        weights=weights_f64,
        init_log_lambda=init_log,
        a_inequality=a_ineq,
        b_inequality=b_ineq,
    )
    return FitResult(
        coefficients=out.coefficients,
        fitted=out.fitted,
        lambdas=out.lam,
        reml_score=out.reml_score,
        edf=out.edf,
        smooths=[smooth],
    )


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
    # Shape constraints route through the constrained Gaussian REML driver
    # in gam-pyffi (active-set + tangent-projected outer REML). Supported on
    # the torch path only for a single 1D smooth (BSpline / Duchon with
    # d==1); a multi-smooth list with constraints, or constrained
    # multivariate smooths, are rejected with a clear error.
    def _shape_kind(s: Smooth) -> str | None:
        sc = getattr(s, "shape_constraint", None)
        if sc is None:
            return None
        sc_str = str(sc).lower()
        return None if sc_str == "none" else sc_str

    if isinstance(smooths, Smooth):
        _shape = _shape_kind(smooths)
    else:
        kinds = [_shape_kind(s) for s in smooths if isinstance(s, Smooth)]
        if any(k is not None for k in kinds):
            raise NotImplementedError(
                "shape_constraint on the torch fit path is currently only "
                "supported for a single Smooth (not a list). For joint "
                "multi-smooth additive fits with shape constraints use "
                "gamfit.fit(df, formula, constraints={...})."
            )
        _shape = None

    if isinstance(smooths, Smooth) and _shape is not None:
        if isinstance(points, (list, tuple)):
            raise ValueError(
                "got a list of points but a single smooth; pass one points tensor."
            )
        if response.dim() == 1:
            response_in = response.unsqueeze(1)
        else:
            response_in = response
        if response_in.dim() != 2 or response_in.shape[1] != 1:
            raise NotImplementedError(
                "shape_constraint on the torch fit path requires a single-"
                "column response of shape (N,) or (N, 1); got shape "
                f"{tuple(response_in.shape)}. Multi-output responses with "
                "constraints are not yet supported."
            )
        return _fit_single_constrained(
            smooths, points, response_in, weights=weights, shape_kind=_shape,
            init_lambdas=init_lambdas,
        )

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
    # `gaussian_reml_fit_additive` routes single-response (D == 1, F > 1)
    # to the multi-block per-smooth-λ Rust path automatically; multi-output
    # responses (D > 1) use the closed-form block-diagonal kernel because
    # the multi-block driver is single-response.
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
