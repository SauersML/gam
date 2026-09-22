"""Unified GAM fit for torch — one entry point for any smooth, any dimensionality.

The user describes smooth-term specs (:class:`gamfit.basis.Smooth` subclasses) and
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

import collections.abc
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import torch

from .._binding import rust_module
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
from ._coerce import from_numpy_like, to_numpy_f64
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


def _to_tensor(value: object, like: torch.Tensor) -> torch.Tensor:
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


def _weighted_sum_to_zero_chart(
    design: torch.Tensor,
    penalty: torch.Tensor,
    weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Restrict one smooth to the engine's weighted sum-to-zero chart.

    The chart itself is structural and comes from the terms-layer canonical
    RRQR policy.  The congruence products stay in torch, so gradients through
    the supplied design and penalty are preserved.  The returned transform
    lifts fitted chart coefficients back to the smooth's raw coefficient
    coordinates.
    """
    transform_np = rust_module().weighted_sum_to_zero_transform(
        to_numpy_f64(design),
        None if weights is None else to_numpy_f64(weights),
    )
    transform = from_numpy_like(transform_np, design).to(torch.float64)
    constrained_design = design @ transform
    constrained_penalty = transform.mT @ penalty @ transform
    return constrained_design, constrained_penalty, transform


def _per_smooth_points(
    points: torch.Tensor | Sequence[torch.Tensor], n_smooths: int,
) -> list[torch.Tensor]:
    """Split ``points`` into one tensor per smooth.

    One tensor is shared by every smooth; any other sequence holds exactly one
    tensor per smooth. ``fit`` and the frozen forward both read ``points``
    through this, so a sequence the one accepts the other accepts too.
    """
    if isinstance(points, torch.Tensor):
        return [points] * n_smooths
    if not isinstance(points, collections.abc.Sequence):
        raise TypeError(
            "points must be a torch.Tensor or a sequence of torch.Tensor, "
            f"got {type(points).__name__}"
        )
    points_list = list(points)
    validate_points_list_length(len(points_list), n_smooths)
    bad = [type(p).__name__ for p in points_list if not isinstance(p, torch.Tensor)]
    if bad:
        raise TypeError(
            f"every per-smooth points entry must be a torch.Tensor, got: {bad}"
        )
    return points_list


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
            periodic=bool(smooth.periodic),
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


def _engine_realized_block(
    smooth: Smooth, points: torch.Tensor, term: str,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """The design block and per-λ penalties the Rust term builder realizes.

    ``term`` is the formula term over the axis names the engine entry assigns
    to ``points``' columns (``x0``, ``x1``, ...), e.g. ``te(x0, x1)``. The
    smooth's own descriptor -- the same payload ``gamfit.fit(..., smooths={...})``
    sends -- carries every tunable, so this returns exactly what ``gamfit.fit``
    would realize for the same spec (gam#4492).

    The DESIGN comes back too, and is used rather than a torch-built one. A
    smooth's penalty is only meaningful in the chart its design is expressed
    in, and that chart is not a right-multiplication this side could apply: the
    collection composes the joint-null rotation ``Q``, the term's own
    identifiability transform, any unabsorbed global orthogonality, and the
    span-preserving parametric residualization ``X·T − C·R``, which is affine in
    the parametric block rather than a factor of ``X``. Rebuilding that
    composition here would be a second implementation of the thing this entry
    exists to stop duplicating. The consequence is explicit: for a term routed
    through this helper the fit carries no autograd path back to ``points``,
    because the design is not a torch expression of them.
    """
    import json

    from .._api import _jsonable_array
    from .._binding import rust_module

    points_np = (
        points.detach().cpu().to(torch.float64).contiguous().numpy()
    )
    descriptor = _jsonable_array(dict(smooth.to_rust_descriptor()))
    design_np, penalties_np = rust_module().smooth_term_realized_penalties(
        points_np, term, json.dumps(descriptor),
    )
    design = torch.as_tensor(
        design_np, dtype=torch.float64, device=points.device,
    ).contiguous()
    penalties = [
        torch.as_tensor(
            penalty, dtype=torch.float64, device=points.device,
        ).contiguous()
        for penalty in penalties_np
    ]
    width = design.shape[1]
    for index, penalty in enumerate(penalties):
        if penalty.shape != (width, width):
            raise RuntimeError(
                f"engine penalty {index} for {term} is {tuple(penalty.shape)}, "
                f"not ({width}, {width}) as its realized design block"
            )
    return design, penalties


def _refuse_multi_lambda(term: str, kind: str, count: int) -> None:
    """Refuse a term whose realized penalty count the backend cannot carry.

    ``gaussian_reml_fit_blocks_exact`` prices ``P = blockdiag(λ_k S_k)``: one
    smoothing parameter per COEFFICIENT BLOCK. A term the builder realizes with
    several penalties -- one per tensor margin, one per active Matérn operator
    dial -- puts several of them on ONE block, which that criterion has no
    coordinate for. Summing them under a single λ is the divergence gam#4492
    is about, so the fit refuses instead (gam#4492 step 2: the block API takes a
    penalty LIST per block).
    """
    if count > 1:
        raise NotImplementedError(
            f"{kind} realizes {count} penalties for {term} -- one smoothing "
            f"parameter each, as gamfit.fit carries them -- and the torch "
            f"Gaussian REML backend prices one λ per coefficient block. "
            f"Fitting them under a single λ would be a different model than "
            f"gamfit.fit's (gam#4492). Use gamfit.fit for this term until the "
            f"block backend takes a penalty list per block."
        )


def _build_design_penalty(
    smooth: Smooth,
    points: torch.Tensor,
    *,
    resolved_bspline: tuple[torch.Tensor, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the (design, penalty) pair for one smooth at given input points.

    The penalty matrix lives in the same M-dim coefficient space as the
    design's column space. Multi-d penalties are sourced from gamfit's
    primitive functions per smooth kind.

    Returns (design (N, M), penalty (M, M)) as float64 torch tensors.

    Two kinds take BOTH halves from the term builder instead
    (:func:`_engine_realized_block`, gam#4492): ``TensorBSpline`` and
    ``Matern``, whose penalties this side used to build itself and build
    differently from the fit. Their design carries no autograd path back to
    ``points``, because the chart their penalty is expressed in is not a factor
    of the raw basis; every other kind is unchanged and still differentiable.
    """
    from .._api import duchon_function_norm_penalty

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
        knots, eff_degree = (
            resolved_bspline
            if resolved_bspline is not None
            else _resolve_bspline_knots_for_fit(smooth, points.squeeze(1))
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
        # Pca is a precomputed projection on every entry point (the Rust
        # formula builder, the `smooths=` override, and every descriptor
        # evaluator all require the supplied basis); `pca_basis_matrix` is the
        # one definition of it.
        basis = torch.as_tensor(
            pca_basis_matrix(pca), dtype=torch.float64, device=points.device
        )
        if basis.shape[0] != points.shape[1]:
            raise ValueError(
                f"Pca: points d={points.shape[1]} but basis has {basis.shape[0]} rows"
            )
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
        design = design_points @ basis
        # Penalty: the empirical function mass `βᵀSβ = mean_i((Zβ)_i²)`, i.e.
        # `S = ZᵀZ / N` — the Rust `pca_function_mass_penalty` functional. It
        # prices the fitted function, not whichever coefficient chart encodes
        # the score columns (an identity ridge would agree only when the
        # scores are empirically orthonormal). REML learns the strength, so it
        # carries no scale of its own. A null direction of `Z` is a shared
        # design/penalty null direction, which the REML backend refuses rather
        # than stabilizing with a ridge.
        penalty = design.transpose(0, 1) @ design / float(design.shape[0])
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
        # The term builder realizes this term, design and penalties together
        # (gam#4492). What stood here summed `I ⊗ S_a ⊗ I` over the margins
        # under ONE λ, and diverged from `gamfit.fit` in three ways at once:
        # the builder emits one candidate per margin with its own λ; it
        # measures the other margins by their FUNCTION Grams
        # `S_dim = G_0/m_0 ⊗ … ⊗ S̃_dim ⊗ … ⊗ G_{d-1}/m_{d-1}` with
        # `m_j = 1ᵀG_j1`, where `I ⊗ S_a ⊗ I` measures them by their
        # COEFFICIENTS (SPEC: penalties are on the function, never on the
        # coefficients; #1561); and it normalizes the marginal roughness first,
        # so no λ carries a basis-size or length unit (#2315).
        axes = ", ".join(f"x{axis}" for axis in range(len(marginals)))
        term = f"te({axes})"
        design, penalties = _engine_realized_block(smooth, points, term)
        _refuse_multi_lambda(term, "TensorBSpline", len(penalties))
        return design, penalties[0]

    if entry == "matern" and isinstance(smooth, Matern):
        if smooth.centers is None:
            raise ValueError("Matern requires centers on the torch path")
        centers_t = _coerce_2d(_to_tensor(smooth.centers, points), "Matern.centers")
        if centers_t.shape[1] != points.shape[1]:
            raise ValueError(
                f"Matern: points d={points.shape[1]} but centers "
                f"d={centers_t.shape[1]}"
            )
        # The term builder realizes this term (gam#4492). What stood here took
        # the raw symmetrised covariance Gram `K_cc` among the centres, on the
        # raw kernel columns, under one λ. The builder never fits a Matérn term
        # that way: its default path uses the ν-gated collocation operator
        # candidates (D0/D1/D2 dials gated by
        # `DuchonOperatorPenaltySpec::matern_for_smoothness`, #707) and its
        # double-penalty path the chart-restricted `Zᵀ K Z` factor read against
        # K's own roundoff envelope beside the centre function Gram, both
        # through the kernel identifiability chart and both with more than one
        # λ. Nothing checked `K_cc` against a roundoff envelope here either.
        axes = ", ".join(f"x{axis}" for axis in range(points.shape[1]))
        term = f"matern({axes})"
        design, penalties = _engine_realized_block(smooth, points, term)
        _refuse_multi_lambda(term, "Matern", len(penalties))
        return design, penalties[0]

    if entry == "categorical" and isinstance(smooth, Categorical):
        # Sum-to-zero coded categorical contrast: an i.i.d. Gaussian random
        # effect on the level effects, restricted to effects that sum to zero.
        # The ridge prices the level effects, as the Rust `RandomEffectTermSpec`
        # prices its one-hot group coefficients. The level codes are structural
        # (integer category labels), so the design carries no autograd path
        # back to `points`.
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
        # The design's level effects are e = C·c with C = [I; -1ᵀ], so the
        # ridge on them is ‖e‖² = cᵀ(I + 11ᵀ)c. It treats every level alike: an
        # identity ridge on c would give the last-coded level K - 1 times the
        # prior variance of the others, and the fit would move with the coding.
        identity = torch.eye(contrast, dtype=torch.float64, device=points.device)
        penalty = identity + torch.ones_like(identity)
        return design, penalty

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
    *,
    resolved_bspline: tuple[torch.Tensor, int] | None = None,
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

    knots, degree = (
        resolved_bspline
        if resolved_bspline is not None
        else _resolve_bspline_knots_for_fit(smooth, points.squeeze(1))
    )
    if degree < 1:
        raise ValueError("shape_constraint requires BSpline degree >= 1")
    basis_width = int(knots.numel()) - degree - 1
    if basis_width != coefficient_count:
        raise ValueError(
            "shape-constraint spline chart mismatch: knot vector and degree "
            f"imply {basis_width} coefficients, design has {coefficient_count}"
        )

    # Knot validation, derivative-control geometry, row scaling, and shape-kind
    # parsing live in Rust. Torch only transfers the canonical linear cone to
    # the target device; it does not maintain a second implementation of the
    # spline mathematics.
    from .._binding import rust_module

    a_np, b_np = rust_module().bspline_shape_constraints(
        knots.detach().cpu().numpy(), int(degree), str(shape_kind),
    )
    a = torch.as_tensor(a_np, dtype=torch.float64, device=points.device).contiguous()
    b = torch.as_tensor(b_np, dtype=torch.float64, device=points.device).contiguous()
    if a.dim() != 2 or a.shape[1] != coefficient_count or b.shape != (a.shape[0],):
        raise RuntimeError(
            "Rust shape-constraint payload does not match the realized "
            f"B-spline chart: A={tuple(a.shape)}, b={tuple(b.shape)}, "
            f"coefficients={coefficient_count}"
        )
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
    points_2d = _coerce_2d(points, "points")
    if not isinstance(smooth, BSpline):
        raise NotImplementedError(
            "shape_constraint on the torch fit path requires an open BSpline"
        )
    realized_bspline = _resolve_bspline_knots_for_fit(
        smooth, points_2d.squeeze(1),
    )
    design, penalty = _build_design_penalty(
        smooth, points_2d, resolved_bspline=realized_bspline,
    )
    a_ineq, b_ineq = _build_shape_constraint_inequality(
        smooth,
        points_2d,
        shape_kind,
        design.shape[1],
        resolved_bspline=realized_bspline,
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
        init_value = float(init_lambdas)
        if not _math.isfinite(init_value) or init_value <= 0.0:
            raise ValueError("init_lambdas must be finite and strictly positive")
        init_log = _math.log(init_value)
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
    coefficient_transforms: list[torch.Tensor | None] = []
    if init_lambdas is not None:
        init_lam_arr = init_lambdas.detach().to(torch.float64).reshape(-1)
        if init_lam_arr.numel() != F:
            raise ValueError(
                f"init_lambdas must have length F={F}; got {init_lam_arr.numel()}"
            )
        if not bool(torch.isfinite(init_lam_arr).all()) or not bool(
            (init_lam_arr > 0.0).all()
        ):
            raise ValueError("init_lambdas must be finite and strictly positive")
        init_log_lambdas = torch.log(init_lam_arr)
    else:
        init_log_lambdas = None

    for smooth, pts in zip(smooths_list, points_list):
        design, penalty = _build_design_penalty(smooth, pts)
        design = design.to(torch.float64)
        penalty = penalty.to(torch.float64)
        by_t = _smooth_by_tensor(smooth, design)
        if by_t is not None:
            design = design * by_t.to(torch.float64).unsqueeze(1)
        if isinstance(smooth, PeriodicSplineCurve):
            design, penalty, transform = _weighted_sum_to_zero_chart(
                design, penalty, weights_f64,
            )
            coefficient_transforms.append(transform)
        else:
            coefficient_transforms.append(None)
        designs.append(design)
        penalties.append(penalty)

    out = _gaussian_reml_fit_blocks_orthogonal(
        designs,
        penalties,
        response_f64,
        weights=weights_f64,
        init_log_lambdas=init_log_lambdas,
    )
    coefficients = [
        coefficient if transform is None else transform @ coefficient
        for coefficient, transform in zip(out.coefficients, coefficient_transforms)
    ]
    return FitResult(
        coefficients=coefficients,
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
        if not isinstance(points, torch.Tensor):
            raise TypeError(
                "a single smooth takes one points torch.Tensor, "
                f"got {type(points).__name__}"
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
        if not isinstance(points, torch.Tensor):
            raise TypeError(
                "a single smooth takes one points torch.Tensor, "
                f"got {type(points).__name__}"
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

    points_list = _per_smooth_points(points, len(smooths_list))

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
    coefficient_transforms: list[torch.Tensor | None] = []
    identified_penalties: list[torch.Tensor] = []
    for smooth, design, penalty, by_t in zip(smooths_list, designs, penalties, bys):
        actual_design = design * by_t.unsqueeze(1) if by_t is not None else design
        if isinstance(smooth, PeriodicSplineCurve):
            actual_design, penalty, transform = _weighted_sum_to_zero_chart(
                actual_design.to(torch.float64),
                penalty.to(torch.float64),
                weights_f64,
            )
            coefficient_transforms.append(transform)
        else:
            coefficient_transforms.append(None)
        modulated.append(actual_design)
        identified_penalties.append(penalty)
    init_log_lambdas = None
    if init_lambdas is not None:
        init_lam_arr = init_lambdas.to(torch.float64).reshape(-1)
        if init_lam_arr.numel() != F:
            raise ValueError(
                f"init_lambdas must have length F={F}; got {init_lam_arr.numel()}"
            )
        if not bool(torch.isfinite(init_lam_arr).all()) or not bool(
            (init_lam_arr > 0.0).all()
        ):
            raise ValueError("init_lambdas must be finite and strictly positive")
        init_log_lambdas = torch.log(init_lam_arr)
    # ``mode='joint'`` reaches here only for D == 1. Preserve the full
    # vector warm start instead of collapsing to init_lambdas[0].
    add_out: AdditiveRemlOutput = gaussian_reml_fit_blocks(
        modulated,
        identified_penalties,
        response_f64,
        weights=weights_f64,
        init_log_lambdas=init_log_lambdas,
    )
    coefficients = [
        coefficient if transform is None else transform @ coefficient
        for coefficient, transform in zip(add_out.coefficients, coefficient_transforms)
    ]
    return FitResult(
        coefficients=coefficients,
        fitted=add_out.fitted,
        lambdas=add_out.lambdas,
        reml_score=add_out.reml_score,
        edf=add_out.edf,
        smooths=smooths_list,
    )


__all__ = ["fit", "FitResult"]
