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
from typing import Literal, Sequence

import torch

from ..smooth import (
    BSpline,
    Categorical,
    Duchon,
    Matern,
    Pca,
    PeriodicSplineCurve,
    Sphere,
    Smooth,
    TensorBSpline,
)
from ._basis import bspline_basis, duchon_basis, periodic_spline_curve_basis, sphere_basis
from ._dispatch import (
    resolve_fit_mode,
    shape_kind_for_smooths_arg,
    validate_2d_shape,
    validate_points_list_length,
    validate_smooths_arg,
)
from ._reml import (
    AdditiveRemlOutput,
    GaussianRemlOutput,
    _gaussian_reml_fit_blocks_orthogonal,
    gaussian_reml_fit,
    gaussian_reml_fit_blocks,
    gaussian_reml_fit_with_constraints,
)

FitMode = Literal["joint", "independent", "auto"]
ShapeConstrainedSmooth = BSpline


# ---------------------------------------------------------------------------
# FitResult — the user-facing output
# ---------------------------------------------------------------------------


@dataclass(slots=True)
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


def _smooth_by_tensor(smooth: Smooth, design: torch.Tensor) -> torch.Tensor | None:
    by = smooth.by
    if by is None:
        return None
    return _to_tensor(by, design).reshape(-1)


def _coerce_2d(t: torch.Tensor, name: str) -> torch.Tensor:
    """Promote ``(N,)`` to ``(N, 1)`` and validate 2D shape."""
    ndim = t.dim()
    if ndim == 1:
        return t.unsqueeze(1)
    validate_2d_shape(name, ndim, tuple(t.shape))
    return t


def _torch_smooth_dispatch_key(class_name: str) -> str:
    from .._binding import rust_module

    dispatch_key = getattr(rust_module(), "torch_smooth_dispatch_key", None)
    if not callable(dispatch_key):
        raise RuntimeError(
            "gamfit._rust is missing torch_smooth_dispatch_key; rebuild gamfit"
        )
    return str(dispatch_key(class_name))


def _bspline_penalty_np(knots_np: Any, degree: int, order: int, periodic: bool) -> Any:
    """Exact derivative roughness matching the design ``bspline_basis`` builds.

    The open basis spans ``len(knots) - degree - 1`` columns and takes the
    open-spline derivative Gram. The periodic basis is cyclic on the
    knot-interval lattice — ``len(knots) - 1`` columns for any degree — so
    its penalty integrates wrapped basis derivatives over that SAME period.
    """
    if periodic:
        import numpy as np

        from .._binding import rust_module

        knots_array = np.asarray(knots_np, dtype=float)
        num_basis = int(knots_array.size - 1)
        period = float(knots_array[-1] - knots_array[0])
        return np.asarray(
            rust_module().cyclic_bspline_roughness_penalty(
                num_basis,
                int(degree),
                period,
                int(order),
            ),
            dtype=float,
        )
    from .._api import smoothness_penalty as _smoothness_penalty

    penalty_np, _null_basis = _smoothness_penalty(
        knots_np, degree=int(degree), order=int(order),
    )
    return penalty_np


def _resolve_bspline_knots_for_fit(
    smooth: BSpline, points_1d: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Resolve the knot vector shared by design, penalty, and shape cone."""
    knots_spec = smooth.knots
    if knots_spec is None or isinstance(knots_spec, int):
        # Auto-knot placement and small-sample degree reduction belong to Rust.
        # Resolve through that authority so every downstream object uses the
        # same realized spline chart.
        from .._api import _resolve_knots

        resolved = _resolve_knots(
            knots_spec,
            points_1d.detach().cpu().to(torch.float64).numpy(),
            label="knots",
            degree=smooth.degree,
        )
        return (
            torch.as_tensor(
                resolved.locations, dtype=torch.float64, device=points_1d.device,
            ),
            int(resolved.order),
        )
    return (
        _to_tensor(knots_spec, points_1d)
        .detach()
        .reshape(-1)
        .to(torch.float64),
        int(smooth.degree),
    )


def _marginal_bspline_design_penalty(
    marginal: BSpline, x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one marginal's ``(design, penalty)`` for a tensor-product smooth.

    Mirrors the scalar :class:`BSpline` branch exactly: the design carries the
    autograd VJP back to ``x`` through :func:`bspline_basis`, and the penalty
    shares the SAME resolved knot vector, effective degree, and (cyclic vs
    open) topology as the design so the derivative roughness regularizes the
    function the design actually spans (auto-knot derivation may downgrade the
    degree for small n — #340).

    ``x`` is the 1D marginal coordinate ``(N,)``. Returns ``(B_x, S_x)`` where
    ``B_x`` is ``(N, k)`` (differentiable) and ``S_x`` is ``(k, k)``.
    """
    marg_knots = marginal.knots
    if marg_knots is None or isinstance(marg_knots, int):
        from .._api import _resolve_knots
        resolved = _resolve_knots(
            marg_knots,
            x.detach().cpu().to(torch.float64).numpy(),
            label="knots", degree=int(marginal.degree),
        )
        knots_np = resolved.locations
        eff_degree = int(resolved.order)
        knots = torch.as_tensor(knots_np, dtype=torch.float64, device=x.device)
    else:
        knots = _to_tensor(marg_knots, x).reshape(-1)
        knots_np = knots.detach().cpu().to(torch.float64).numpy()
        eff_degree = int(marginal.degree)
    design = bspline_basis(
        x, knots, degree=eff_degree, periodic=bool(marginal.periodic),
    )
    penalty_np = _bspline_penalty_np(
        knots_np, eff_degree, int(marginal.penalty_order), bool(marginal.periodic),
    )
    penalty = torch.as_tensor(penalty_np, dtype=torch.float64, device=x.device)
    return design.to(torch.float64), penalty


def _kron_eye(left: int, mat: torch.Tensor, right: int) -> torch.Tensor:
    """Form ``I_left ⊗ mat ⊗ I_right`` for the Kronecker-sum tensor penalty.

    ``left`` / ``right`` are the products of the basis sizes of the marginals
    before / after the axis ``mat`` penalizes. The result lives in the same
    column space as the row-major Khatri-Rao tensor design (earliest marginal
    varies slowest), so it composes term-by-term into ``S = Σ_a I ⊗ S_a ⊗ I``.
    """
    eye_l = torch.eye(left, dtype=mat.dtype, device=mat.device)
    eye_r = torch.eye(right, dtype=mat.dtype, device=mat.device)
    return torch.kron(torch.kron(eye_l, mat), eye_r)


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

    # Dispatch decision (which Rust entry to call) lives in Rust as the single
    # source of truth for supported torch-fit specs — the Rust call validates
    # the spec is recognised and supported. The tensor construction under each
    # branch stays here because torch autograd VJP must flow back through
    # `points`, `centers`, and `by`; ``isinstance`` is used so pyright narrows
    # ``smooth`` to the matching subclass on each branch.
    try:
        entry = _torch_smooth_dispatch_key(type(smooth).__name__)
    except ValueError as exc:
        # The Rust dispatch only errors for truly unknown subclass names;
        # surface those as TypeError to match the previous Python cascade.
        message = str(exc)
        if message.startswith("unknown Smooth subclass"):
            raise TypeError(message) from exc
        raise NotImplementedError(message) from exc

    if entry == "duchon" and isinstance(smooth, Duchon):
        duchon_centers = smooth.centers
        duchon_m = smooth.m
        periodic_per_axis = smooth.periodic_per_axis

        centers = _coerce_2d(_to_tensor(duchon_centers, points), "Duchon.centers")
        if centers.shape[1] != points.shape[1]:
            raise ValueError(
                f"Duchon: points d={points.shape[1]} but centers d={centers.shape[1]}"
            )
        per = (
            tuple(bool(p) for p in periodic_per_axis)
            if periodic_per_axis is not None
            else None
        )
        design = duchon_basis(points, centers, m=duchon_m, periodic_per_axis=per)
        try:
            penalty_np = duchon_function_norm_penalty(
                centers.detach().cpu().numpy(),
                m=duchon_m,
                periodic_per_axis=per,
            )
        except NotImplementedError as exc:
            raise NotImplementedError(
                f"Duchon penalty for d={centers.shape[1]} not yet exposed by gam-pyffi. "
                f"d=1 works; multi-d penalty binding pending. Underlying error: {exc}"
            ) from exc
        penalty = torch.as_tensor(penalty_np, dtype=torch.float64, device=points.device)
        return design.to(torch.float64), penalty

    if entry == "bspline" and isinstance(smooth, BSpline):
        if points.shape[1] != 1:
            raise ValueError(
                f"BSpline is 1D-only; got points with d={points.shape[1]}. "
                "Use TensorBSpline for multi-d with different units, or Duchon for radial."
            )
        bspline_periodic = smooth.periodic
        bspline_penalty_order = smooth.penalty_order
        knots, eff_degree = _resolve_bspline_knots_for_fit(
            smooth, points.squeeze(1),
        )
        knots_np = knots.detach().cpu().numpy()
        design = bspline_basis(
            points.squeeze(1), knots, degree=eff_degree, periodic=bspline_periodic,
        )
        penalty_np = _bspline_penalty_np(
            knots_np, eff_degree, bspline_penalty_order, bool(bspline_periodic),
        )
        penalty = torch.as_tensor(penalty_np, dtype=torch.float64, device=points.device)
        return design.to(torch.float64), penalty

    if entry == "sphere" and isinstance(smooth, Sphere):
        radians = smooth.radians
        n_centers = smooth.n_centers
        penalty_order = smooth.penalty_order
        kernel = smooth.kernel
        if points.shape[1] != 2:
            raise ValueError(
                f"Sphere expects points of shape (N, 2) [lat, lon]; got d={points.shape[1]}"
            )
        if not torch.isfinite(points).all():
            raise ValueError("Sphere: points contains NaN/Inf")
        lat = points[:, 0]
        if radians:
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
            n_centers=n_centers,
            penalty_order=penalty_order,
            kernel=kernel,
            radians=radians,
        )
        return design.to(torch.float64), penalty.to(torch.float64)

    if entry == "periodic_spline_curve" and isinstance(smooth, PeriodicSplineCurve):
        if points.shape[1] != 1:
            raise ValueError(
                f"PeriodicSplineCurve expects 1D parameter t with shape (N,) or "
                f"(N, 1); got d={points.shape[1]}"
            )
        t1d = points.squeeze(1)
        n_knots = smooth.n_knots
        degree = smooth.degree
        penalty_order = smooth.penalty_order
        design, penalty = periodic_spline_curve_basis(
            t1d,
            n_knots=n_knots,
            degree=degree,
            penalty_order=penalty_order,
        )
        return design.to(torch.float64), penalty.to(torch.float64)

    if entry == "pca" and isinstance(smooth, Pca):
        from .._basis_eval import pca_basis_matrix, pca_training_mean

        pca = smooth
        if pca.lazy_path is not None:
            raise NotImplementedError("Pca lazy_path is available on the Rust formula path")
        design_points = points.to(torch.float64)
        if pca.centered:
            # Fitting is the fit/transform boundary: resolve the training mean
            # here and persist it on the spec so predict-time evaluations
            # subtract the SAME mean (a fixed affine map), not their own
            # batch mean.
            mean_np = pca_training_mean(pca, design_points.detach().cpu().numpy())
            design_points = design_points - torch.as_tensor(
                mean_np, dtype=torch.float64, device=points.device
            ).reshape(1, -1)
        if pca.basis is None:
            if pca.K is None:
                raise ValueError("Pca requires K when basis is None")
            _u, _s, vh = torch.linalg.svd(design_points, full_matrices=False)
            basis = vh[: int(pca.K)].T.contiguous()
            # Persist the fitted projection so later descriptor evaluations
            # reuse the map this fit selected.
            pca.basis = basis.detach().cpu().numpy()
        else:
            basis = torch.as_tensor(
                pca_basis_matrix(pca), dtype=torch.float64, device=points.device
            )
        if basis.shape[0] != points.shape[1]:
            raise ValueError(
                f"Pca: points d={points.shape[1]} but basis has {basis.shape[0]} rows"
            )
        design = design_points @ basis
        penalty = torch.eye(
            basis.shape[1], dtype=torch.float64, device=points.device
        ) * float(pca.smooth_penalty)
        return design, penalty

    if entry == "tensor_bspline" and isinstance(smooth, TensorBSpline):
        marginals = list(smooth.marginals)
        if not marginals:
            raise ValueError("TensorBSpline: no marginals")
        if points.shape[1] != len(marginals):
            raise ValueError(
                f"TensorBSpline has {len(marginals)} marginals but points have "
                f"d={points.shape[1]}"
            )
        # Per-marginal 1D B-spline design + exact derivative Gram (shared knots).
        marg_designs: list[torch.Tensor] = []
        marg_penalties: list[torch.Tensor] = []
        for j, marg in enumerate(marginals):
            b_j, s_j = _marginal_bspline_design_penalty(marg, points[:, j])
            marg_designs.append(b_j)
            marg_penalties.append(s_j)
        sizes = [b.shape[1] for b in marg_designs]
        # Design: row-wise Khatri-Rao (tensor) product of the marginal bases,
        # earliest marginal varying slowest — the autograd VJP flows back to
        # `points` through each `bspline_basis` factor exactly as the scalar
        # BSpline path does, since the Hadamard-outer product is plain torch
        # algebra over the differentiable marginal designs.
        design = marg_designs[0]
        for b_j in marg_designs[1:]:
            n = design.shape[0]
            design = (design.unsqueeze(2) * b_j.unsqueeze(1)).reshape(n, -1)
        # Penalty: single-λ Kronecker-sum  S = Σ_a I ⊗ S_a ⊗ I  (mgcv te()-style
        # isotropic tensor penalty), matching the Rust TensorBSpline penalty
        # `S = Σ_i I ⊗ … ⊗ S_i ⊗ … ⊗ I` (src/terms/smooth/part_001.rs:486).
        total = 1
        for k in sizes:
            total *= k
        penalty = torch.zeros(
            total, total, dtype=torch.float64, device=points.device
        )
        for a, s_a in enumerate(marg_penalties):
            left = 1
            for k in sizes[:a]:
                left *= k
            right = 1
            for k in sizes[a + 1:]:
                right *= k
            penalty = penalty + _kron_eye(left, s_a, right)
        return design.to(torch.float64), penalty

    if entry == "matern" and isinstance(smooth, Matern):
        from .._api import matern_basis as _matern_basis
        if smooth.centers is None:
            raise ValueError("Matern requires centers on the torch path")
        from .._basis_eval import matern_evaluate, _matern_nu_string
        centers_t = _coerce_2d(_to_tensor(smooth.centers, points), "Matern.centers")
        if centers_t.shape[1] != points.shape[1]:
            raise ValueError(
                f"Matern: points d={points.shape[1]} but centers "
                f"d={centers_t.shape[1]}"
            )
        # Design: Matérn kernel evaluated points-vs-centers, autograd VJP back to
        # `points` via the analytic Rust input-location jet (matern_evaluate).
        design = matern_evaluate(smooth, points)
        # Penalty: the Matérn covariance Gram K_cc among centers — the
        # REML-compatible RKHS penalty by Duchon's kernel-Gram identity (the
        # RKHS norm of f = Σ αᵢ k(·, cᵢ) is αᵀ K_cc α). centers/length_scale/ν
        # are structural, so this carries no autograd path.
        centers_np = centers_t.detach().cpu().to(torch.float64).numpy()
        gram_np = _matern_basis(
            centers_np,
            centers_np,
            length_scale=float(smooth.length_scale),
            nu=_matern_nu_string(float(smooth.nu)),
            aniso_log_scales=smooth.aniso_log_scales,
        )
        penalty = torch.as_tensor(
            gram_np, dtype=torch.float64, device=points.device
        )
        # Symmetrize against round-off so REML sees an exactly symmetric penalty.
        penalty = 0.5 * (penalty + penalty.transpose(0, 1))
        return design.to(torch.float64), penalty

    if entry == "categorical" and isinstance(smooth, Categorical):
        # Sum-to-zero coded categorical contrast = i.i.d. Gaussian random
        # effect with an identity ridge penalty on the level contrasts. This
        # mirrors the Rust `RandomEffectTermSpec` (one-hot dummy block with an
        # identity penalty on group coefficients) and the `Pca` torch branch
        # (linear projection design + identity ridge penalty). The level codes
        # are structural (integer category labels), so the design carries no
        # autograd path back to `points`.
        if smooth.levels is None:
            raise ValueError("Categorical requires `levels` on the torch path")
        n_levels = int(smooth.n_levels)
        if n_levels < 2:
            raise ValueError(
                f"Categorical requires n_levels >= 2; got {n_levels}"
            )
        levels = _to_tensor(smooth.levels, points).reshape(-1).round().to(torch.int64)
        if levels.shape[0] != N:
            raise ValueError(
                f"Categorical: levels has {levels.shape[0]} rows but points "
                f"have N={N}"
            )
        lo = int(levels.min().item())
        hi = int(levels.max().item())
        if lo < 0 or hi >= n_levels:
            raise ValueError(
                f"Categorical: level codes must lie in [0, {n_levels - 1}]; "
                f"observed range [{lo}, {hi}]"
            )
        # Sum-to-zero contrast: one column per non-reference level; the
        # reference level (the last code) gets -1 across every column so the
        # fitted level effects sum to zero (drop-last sum-to-zero coding).
        contrast = n_levels - 1
        onehot = torch.zeros(
            N, n_levels, dtype=torch.float64, device=points.device
        )
        onehot[torch.arange(N, device=points.device), levels] = 1.0
        design = onehot[:, :contrast] - onehot[:, contrast:contrast + 1]
        penalty = torch.eye(
            contrast, dtype=torch.float64, device=points.device
        )
        return design, penalty

    expected_smooth_type = {
        "duchon": "Duchon",
        "bspline": "BSpline",
        "sphere": "Sphere",
        "periodic_spline_curve": "PeriodicSplineCurve",
        "pca": "Pca",
    }.get(entry)
    if expected_smooth_type is not None:
        raise NotImplementedError(
            f"torch fit dispatch returned {entry!r} for {type(smooth).__name__}, "
            f"but that branch requires {expected_smooth_type}; the dispatch key "
            "and Smooth subclass are inconsistent"
        )

    # Recognised-but-not-yet-wired entries: the Rust dispatch registers every
    # `gamfit.torch`-exported Smooth subclass (single source of truth). Every
    # currently-exported kind now has a tensor design/penalty backend wired on
    # the torch path; this guard remains so that a future Rust enum variant
    # added without a matching torch branch raises a consistent
    # NotImplementedError-shape rather than falling through silently.
    unwired_entries: dict[str, str] = {}
    if entry in unwired_entries:
        kind_name = unwired_entries[entry]
        raise NotImplementedError(
            f"{kind_name} is recognised by the torch dispatch but its "
            "design/penalty tensor backend is not yet wired into "
            "gamfit.torch.fit; needs a Rust PyO3 binding for the underlying "
            "basis + penalty. Currently supported on the torch path: "
            "Duchon (any d for basis; d=1 for penalty), BSpline (d=1), "
            "TensorBSpline (te), Matern (kernel-Gram penalty), "
            "Sphere (S²), PeriodicSplineCurve, Pca, Categorical."
        )

    # Defensive raise: the Rust dispatch already rejected unknown specs
    # above. Reaching here means the Rust enumeration grew a new variant
    # without a matching torch branch, so raise to make the gap visible.
    raise NotImplementedError(
        f"torch fit dispatch returned {entry!r} but no matching branch is "
        f"wired for {type(smooth).__name__}"
    )


# ---------------------------------------------------------------------------
# Shape constraints — exact B-spline derivative-control cones
# ---------------------------------------------------------------------------


def _build_shape_constraint_inequality(
    smooth: ShapeConstrainedSmooth,
    points: torch.Tensor,
    shape_kind: str,
    coefficient_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``A·β ≥ 0`` certifying shape on every open knot span.

    For ``f = Σ β_i N_{i,d}``, the first derivative is a degree-``d-1``
    B-spline whose control coefficients have the signs of
    ``β[i+1] - β[i]``. The second derivative has the signs of consecutive
    differences of the Greville-scaled first-derivative controls. Requiring
    those derivative control coefficients to have the requested sign is a
    finite spanwise certificate over the continuum; its size depends only on
    the realized spline basis, never on a sampling resolution.
    """
    points = _coerce_2d(points, "points")
    if points.shape[1] != 1:
        raise NotImplementedError(
            "shape_constraint on the torch path requires a 1D covariate (d==1); "
            f"got d={points.shape[1]}. Multidimensional shape constraints are "
            "not supported on the torch path."
        )
    if not torch.isfinite(points).all():
        raise ValueError("shape constraint requires finite covariate values")
    if smooth.periodic:
        raise NotImplementedError(
            "shape_constraint requires an open BSpline; a globally monotone "
            "periodic spline is necessarily constant and needs a distinct "
            "cyclic coefficient chart."
        )

    knots, degree = _resolve_bspline_knots_for_fit(smooth, points.squeeze(1))
    if degree < 1:
        raise ValueError("shape_constraint requires BSpline degree >= 1")
    basis_width = int(knots.numel()) - degree - 1
    if basis_width != coefficient_count:
        raise ValueError(
            "shape-constraint spline chart mismatch: knot vector and degree "
            f"imply {basis_width} coefficients, design has {coefficient_count}"
        )

    sk = shape_kind.lower()
    if sk in ("monotone_increasing", "monotone_decreasing"):
        if coefficient_count < 2:
            raise ValueError("monotonicity requires at least two B-spline coefficients")
        sign = 1.0 if sk == "monotone_increasing" else -1.0
        a = torch.zeros(
            (coefficient_count - 1, coefficient_count),
            dtype=torch.float64,
            device=points.device,
        )
        row = torch.arange(coefficient_count - 1, device=points.device)
        a[row, row] = -sign
        a[row, row + 1] = sign
    elif sk in ("convex", "concave"):
        if coefficient_count < 3:
            raise ValueError(
                "curvature constraints require at least three B-spline coefficients"
            )
        sign = 1.0 if sk == "convex" else -1.0
        # ξ_i = mean(t[i+1 : i+d+1]); derivative controls are
        # (β[i+1]-β[i]) / (ξ[i+1]-ξ[i]).
        greville = knots[1:-1].unfold(0, degree, 1).mean(dim=1)
        spans = greville[1:] - greville[:-1]
        if spans.numel() != coefficient_count - 1 or not bool((spans > 0.0).all()):
            raise ValueError(
                "shape_constraint requires strictly increasing B-spline "
                "Greville abscissae"
            )
        inverse_spans = spans.reciprocal()
        a = torch.zeros(
            (coefficient_count - 2, coefficient_count),
            dtype=torch.float64,
            device=points.device,
        )
        row = torch.arange(coefficient_count - 2, device=points.device)
        a[row, row] = sign * inverse_spans[:-1]
        a[row, row + 1] = -sign * (inverse_spans[:-1] + inverse_spans[1:])
        a[row, row + 2] = sign * inverse_spans[1:]
    else:
        raise ValueError(
            f"unknown shape_constraint kind {shape_kind!r}; expected one of "
            "monotone_increasing, monotone_decreasing, convex, concave"
        )

    # Positive row scaling preserves the cone and keeps the constrained solve
    # insensitive to the physical units of an irregular knot sequence.
    a = (a / torch.linalg.vector_norm(a, dim=1, keepdim=True)).contiguous()
    b = torch.zeros(a.shape[0], dtype=torch.float64, device=a.device)
    return a, b


def _fit_single_constrained(
    smooth: ShapeConstrainedSmooth,
    points: torch.Tensor,
    response: torch.Tensor,
    *,
    weights: torch.Tensor | None,
    shape_kind: str,
    init_lambdas: torch.Tensor | None,
) -> "FitResult":
    design, penalty = _build_design_penalty(smooth, points)
    a_ineq, b_ineq = _build_shape_constraint_inequality(
        smooth, points, shape_kind, design.shape[1],
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


def _fit_independent(
    points_list: list[torch.Tensor],
    response_f64: torch.Tensor,
    smooths_list: list[Smooth],
    *,
    weights_f64: torch.Tensor | None,
    init_lambdas: torch.Tensor | None,
) -> FitResult:
    """Scalable block-orthogonal additive REML with shared residual scale.

    The coefficient solves remain per-block, but λ_k are selected against the
    additive residual quadratic shared by all blocks and all output columns.
    This is exact when the by-modulated block designs are W-orthogonal and is
    the large-F replacement for the old loop of private single-smooth REML
    fits.

    Complexity: ``O(F · M_k³)`` for the inner Cholesky vs ``O((F · M_k)³)``
    for the joint additive path. This is the production path for
    SAE-scale work where F ≫ 64.

    Multi-output ``response`` shape ``(N, D)`` is supported with one λ per
    block and one profiled residual scale per output column.
    """
    F = len(smooths_list)
    designs: list[torch.Tensor] = []
    penalties: list[torch.Tensor] = []
    if init_lambdas is not None:
        init_lam_arr = init_lambdas.detach().reshape(-1)
        if init_lam_arr.numel() != F:
            raise ValueError(
                f"init_lambdas must have length F={F}; got {init_lam_arr.numel()}"
            )
        init_log_lambdas = torch.log(
            init_lambdas.to(torch.float64).reshape(-1).clamp_min(1.0e-300)
        )
    else:
        init_log_lambdas = None

    for smooth, pts in zip(smooths_list, points_list):
        design, penalty = _build_design_penalty(smooth, pts)
        design = design.to(torch.float64)
        penalty = penalty.to(torch.float64)
        by_t = _smooth_by_tensor(smooth, design)
        if by_t is not None:
            design = design * by_t.to(torch.float64).unsqueeze(1)
        designs.append(design)
        penalties.append(penalty)

    out = _gaussian_reml_fit_blocks_orthogonal(
        designs,
        penalties,
        response_f64,
        weights=weights_f64,
        init_log_lambdas=init_log_lambdas,
    )
    return FitResult(
        coefficients=list(out.coefficients),
        fitted=out.fitted,
        lambdas=out.lambdas,
        reml_score=out.reml_score,
        edf=out.edf,
        smooths=smooths_list,
    )


def fit(
    points: torch.Tensor | Sequence[torch.Tensor],
    response: torch.Tensor,
    smooths: Smooth | Sequence[Smooth],
    *,
    weights: torch.Tensor | None = None,
    init_lambdas: torch.Tensor | None = None,
    mode: FitMode = "auto",
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
    mode : ``"joint" | "independent" | "auto"`` (default ``"auto"``).
        Selects the additive-fit algorithm:

        * ``"joint"`` — joint additive REML via the multi-block Rust
          driver. Per-smooth λ jointly selected against a single joint
          design ``Z = [Z_1 | ... | Z_F]``. Inner Cholesky cost is
          ``O((F · M_k)³)`` — feasible for F ≲ 64, infeasible at
          F ≳ 1000. Currently single-output only (D = 1).
        * ``"independent"`` — scalable block-orthogonal additive REML.
          Coefficient solves are per smooth, but all λ_k share the additive
          profiled residual scale. Cost is ``O(F · M_k³)`` and it supports
          multi-output D > 1 with one λ per smooth.
        * ``"auto"`` (default) — routes to ``"joint"`` if
          ``F ≤ 64`` and ``D == 1``, else to ``"independent"``.

        Mathematical caveat for ``"independent"``: this is exact when the
        by-modulated block designs are W-orthogonal. For genuinely
        overlapping smooths on the same predictor, joint additive REML is
        statistically tighter but only computable at moderate F.

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
    # the torch path only for a single open 1D B-spline, whose derivative
    # control coefficients give an exact spanwise certificate. A multi-smooth
    # list, radial basis, periodic basis, or multivariate smooth is rejected.
    _shape = shape_kind_for_smooths_arg(smooths)

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
        if not isinstance(smooths, BSpline):
            raise NotImplementedError(
                "shape_constraint not supported on the torch path for "
                f"{type(smooths).__name__}; supported: open BSpline."
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
        by_t = _smooth_by_tensor(smooths, design)
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

    smooths_list = validate_smooths_arg(smooths)

    # Per-smooth points
    if isinstance(points, (list, tuple)):
        points_list = list(points)
        validate_points_list_length(len(points_list), len(smooths_list))
    else:
        # Same points for every smooth.
        points_list = [points] * len(smooths_list)

    # Mode dispatch — large-F additive fits route through `independent`;
    # small-F diagnostics route through `joint`. ``auto`` thresholding and
    # joint multi-output rejection are pure dispatch decisions.
    F = len(smooths_list)
    D = response_f64.shape[1]
    effective_mode = resolve_fit_mode(mode, F, D)

    if effective_mode == "independent":
        return _fit_independent(
            points_list, response_f64, smooths_list,
            weights_f64=weights_f64, init_lambdas=init_lambdas,
        )

    # mode == "joint" — proceed with the block-joint additive REML below.
    designs: list[torch.Tensor] = []
    penalties: list[torch.Tensor] = []
    bys: list[torch.Tensor | None] = []
    for s, pts in zip(smooths_list, points_list):
        design, penalty = _build_design_penalty(s, pts)
        designs.append(design)
        penalties.append(penalty)
        bys.append(_smooth_by_tensor(s, design))

    modulated: list[torch.Tensor] = []
    for design, by_t in zip(designs, bys):
        modulated.append(design * by_t.unsqueeze(1) if by_t is not None else design)
    init_log_lambdas = None
    if init_lambdas is not None:
        init_lam_arr = init_lambdas.to(torch.float64).reshape(-1)
        if init_lam_arr.numel() != F:
            raise ValueError(
                f"init_lambdas must have length F={F}; got {init_lam_arr.numel()}"
            )
        init_log_lambdas = torch.log(init_lam_arr.clamp_min(1.0e-300))
    # ``mode='joint'`` reaches here only for D == 1. Preserve the full
    # vector warm start instead of collapsing to init_lambdas[0].
    add_out: AdditiveRemlOutput = gaussian_reml_fit_blocks(
        modulated,
        penalties,
        response_f64,
        weights=weights_f64,
        init_log_lambdas=init_log_lambdas,
    )
    return FitResult(
        coefficients=list(add_out.coefficients),
        fitted=add_out.fitted,
        lambdas=add_out.lambdas,
        reml_score=add_out.reml_score,
        edf=add_out.edf,
        smooths=smooths_list,
    )


__all__ = ["fit", "FitResult"]
