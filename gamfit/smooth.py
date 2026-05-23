"""Smooth-term specs — universal across the formula path and the torch path.

A ``Smooth`` describes the *kind* of smooth a user wants (Duchon m-spline,
B-spline, Matérn covariance, spherical-harmonic, etc.) plus its
hyperparameters (centers / knots / order / periodicity / gating). Both the
tabular ``gamfit.fit(data, smooths=...)`` path and the torch
``gamfit.torch.fit(points, response, smooths=...)`` path consume the same
specs, route them through the same Rust engine, and return the same
mathematical fit. Penalty matrices and REML mechanics are an implementation
detail the user never sees.

These dataclasses have no torch dependency — ``centers``, ``knots`` etc.
are typed as ``Any`` and accept array-likes (numpy ndarray or torch tensor).
Dispatch layers do the appropriate conversion.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence


ShapeConstraintLiteral = Literal[
    "none",
    "monotone_increasing",
    "monotone_decreasing",
    "convex",
    "concave",
]


@dataclass
class Smooth:
    """Base class for all smooth-term specs. Don't instantiate directly.

    Common fields shared by all smooth kinds:

    Parameters
    ----------
    name : optional, for diagnostics + result indexing.
    by : optional array-like of shape ``(N,)``. A per-row multiplier on
        the smooth's contribution (e.g. TopK amplitude in a sparse-coding
        layer). Mathematically equivalent to ``by · s(x)`` in the additive
        model. When the underlying basis is a column-block of the joint
        design, ``by`` multiplies every column in the block per row.
    double_penalty : if True, add a ridge-shrinkage penalty on the null
        space of the main penalty (mgcv ``bs="..."`` ``select=TRUE`` style).
        Useful when the smooth might be entirely redundant and you want
        REML to shrink it out.
    shape_constraint : optional shape constraint on the fitted function.
        One of ``None`` / ``"none"`` (unconstrained, the default),
        ``"monotone_increasing"`` (f'(x) ≥ 0 everywhere on the data range),
        ``"monotone_decreasing"`` (f'(x) ≤ 0), ``"convex"`` (f''(x) ≥ 0),
        or ``"concave"`` (f''(x) ≤ 0). Shape constraints are enforced by
        the inner solver as joint linear inequalities ``A·β ≤ b`` on the
        coefficient vector (the constraint matrix ``A`` is generated from
        the basis on a dense 1D grid spanning the data range, so the
        inequality at the grid points implies the constraint on the
        smooth function under standard B-spline / radial-basis density
        arguments). The solver is an active-set / interior-point method;
        when the constraint is active at the cert exit the outer REML
        score uses the tangent-projected LAML formulation so the smoothing
        parameter is selected over the working subspace. This mirrors
        mgcv's ``scop=...`` argument and the ``scam`` R library's shape-
        constrained smooths. Currently restricted to univariate 1D smooths
        (B-splines and thin-plate / Duchon with a single feature axis);
        a multivariate spec on a constrained smooth will be rejected with
        a clear error from the Rust core. Spherical-harmonic and tensor
        smooths reject all non-``None`` shape constraints.
    """

    name: str | None = None
    by: Any | None = None
    double_penalty: bool = False
    shape_constraint: ShapeConstraintLiteral | None = None


@dataclass
class Duchon(Smooth):
    """Duchon m-spline. Isotropic radial kernel. Works at any d ≥ 1.

    Special cases:

    * m=2, d=1 — natural cubic smoothing spline.
    * m=2, d=2 — thin-plate spline (the TPS *is* the d=2 special case of
      Duchon m=2 with the function-norm penalty).
    * m=2, d ≥ 3 — Duchon's generalized thin-plate spline.

    For a Duchon radial basis with kernel ϕ and centers c_1, ..., c_K, the
    (K × K) Gram matrix P with P_{ij} = ϕ(‖c_i − c_j‖) IS exactly the
    function-norm penalty matrix. That is, P = R^T R is the Cholesky factor
    of the smoothness penalty matrix used internally; users who manually
    build a radial-basis design X with rows X_{n,k} = ϕ(‖x_n − c_k‖) and
    want the corresponding REML-compatible penalty should pass P directly.
    The same identity holds for Matérn (M = K K^T where K is the covariance)
    — this is why kernel ridge regression and penalized smoothing are the
    same problem.

    Parameters
    ----------
    centers : array-like of shape ``(K, d)`` — control points in ℝ^d.
        ``d`` is inferred from ``centers.shape[1]``. For 1D smooths,
        a shape ``(K,)`` 1D array is auto-promoted to ``(K, 1)``.
    m : int, default 2. Spline order.
    length_scale : optional positive float. ``None`` selects the
        scale-free pure Duchon spectrum ``‖w‖^(2(p+s))``. A positive
        value enables the hybrid spectrum
        ``‖w‖^(2p) · (κ² + ‖w‖²)^s`` with ``κ = 1/length_scale``,
        which is closer to a Matérn for finite kernels at high d.
        Honored on both the formula API and the primitive numpy API
        (``gamfit.duchon_basis`` / ``gamfit.duchon_function_norm_penalty``);
        the hybrid kernel keeps the polynomial nullspace order **linear in
        d**, letting the same smooth scale cleanly to d=8, 16, 32, 64
        without ratcheting the nullspace to absorb the Wendland CPD
        constraint ``2s < d``.
    periodic_per_axis : optional sequence of bool of length d. Each
        axis can be periodic independently (cylinder = d=2 with
        ``(True, False)``, torus = d=2 with ``(True, True)``). The d=1
        case uses the Bernoulli-Green builder; d ≥ 2 routes through the
        mixed-periodicity radial polyharmonic kernel on cylinder/torus
        chord distance. Per-axis periods are auto-derived from the
        ``centers``' span along each periodic axis.
    """

    centers: Any = None
    m: int = 2
    length_scale: float | None = None
    periodic_per_axis: Sequence[bool] | None = None


@dataclass
class BSpline(Smooth):
    """1D B-spline (Eilers-Marx P-spline) with a difference penalty.

    For multi-dimensional inputs with DIFFERENT axis units (space × time
    being the canonical example), use :class:`TensorBSpline` instead.
    For multi-dimensional inputs with the SAME axis units (a continuous
    coordinate field), use :class:`Duchon` (isotropic radial kernel).
    """

    knots: Any = None                    # (K,) — auto-derived if None
    degree: int = 3
    penalty_order: int = 2
    periodic: bool = False


@dataclass
class TensorBSpline(Smooth):
    """Tensor-product B-spline (mgcv ``te()`` style).

    Use when axes have different units / scales (e.g. space × time, age ×
    treatment) and isotropic radial kernels are inappropriate. Each
    marginal is a 1D :class:`BSpline`; the joint smooth is their tensor
    product with one λ per marginal selected jointly by REML.
    """

    marginals: list[BSpline] = field(default_factory=list)


@dataclass
class Matern(Smooth):
    """Matérn covariance kernel basis. Multi-d. Optional axis anisotropy.

    As with :class:`Duchon`, the Matérn covariance Gram matrix is the
    REML-compatible penalty by Duchon's kernel-Gram identity.

    Parameters
    ----------
    centers : array-like ``(K, d)``.
    nu : smoothness order. Standard values are 0.5 (= exponential),
        1.5 (continuously differentiable once), 2.5 (twice).
    length_scale : positive float controlling the global kernel range.
    aniso_log_scales : optional length-d sequence. Per-axis log-scale
        contrasts with sum constraint Σ η_a = 0, implementing geometric
        anisotropy ``Λ = κ·diag(exp η)``. ``None`` uses isotropic.
    """

    centers: Any = None
    nu: float = 1.5
    length_scale: float = 1.0
    aniso_log_scales: Sequence[float] | None = None


@dataclass
class Sphere(Smooth):
    """Spherical-harmonic basis on the unit sphere S².

    Input is expected to be ``(N, 2)`` (latitude, longitude) — degrees by
    default, radians when ``radians=True``.

    Parameters
    ----------
    n_centers : number of basis centers when using ``kernel="sobolev"`` or
        ``"pseudo"`` (Wahba-style). For ``kernel="harmonic"`` (eigen
        basis) this is the truncation degree.
    penalty_order : roughness penalty order m ∈ {1, 2, 3, 4}.
        ``m=2`` is the canonical TPS-on-sphere analogue (curvature).
    kernel : one of ``"sobolev"`` (default), ``"pseudo"``, ``"harmonic"``.
    radians : default ``False`` (degrees, Earth/data-frame convention).
    """

    n_centers: int = 50
    penalty_order: int = 2
    kernel: str = "sobolev"
    radians: bool = False


@dataclass
class PeriodicSplineCurve(Smooth):
    """Closed parametric curve in R^d, periodic in the parameter t ∈ [0, 1].

    Use case: directly fit closed loops in a high-D space (e.g., a color
    loop in a language model's residual stream). The fit returns
    coefficients of shape (K, d) describing the curve's basis representation.

    Parameters
    ----------
    n_knots : number of basis knots along the periodic parameter.
    degree : B-spline degree (default 3 = cubic).
    output_dim : d, the dimension of the ambient space the curve lives in.
    """

    n_knots: int = 20
    degree: int = 3
    output_dim: int = 1
    penalty_order: int = 2


@dataclass
class LatentCoord:
    """Per-row latent coordinates ``t ∈ ℝ^{N × d}`` as a first-class parameter.

    See ``proposals/latent_coord.md`` in the gam repository for the design
    rationale and the math; see ``src/terms/latent_coord.rs`` for the Rust
    side.

    The familiar GAM picture is

    .. code-block::

        y = Φ(x) · β + ε,   penalty(β, ψ),   x observed.

    ``LatentCoord`` promotes one of the covariates ``x`` from *observed* to
    *latent*: ``t`` is estimated per row alongside the spline coefficients
    ``β`` and the smoothing parameters ``ρ``. Mechanically this is the same
    problem the ``ψ`` (log-anisotropy) machinery already solves — both move
    the design matrix Φ and reuse the IFT-warm-started outer optimizer —
    except differentiating with respect to the *first* argument of the
    radial kernel rather than its scale.

    Use cases: GP-LVM, principal manifolds, identifiable nonlinear ICA via
    iVAE-style auxiliary priors, intrinsic-dimension discovery via ARD.

    Gauge fixing — required to keep the inner Hessian full-rank
    ----------------------------------------------------------------

    The bare data-fit ``½ ‖y − Φ(t) β‖²`` is invariant under any
    diffeomorphism ``t ↦ φ(t)`` (any reparameterization can be absorbed
    into a re-fit of β). This makes the inner Hessian *rank-deficient*
    along the gauge orbit and IFT breaks. Supply ``aux_prior`` or pair this
    block with an :class:`gamfit.IsometryPenalty` before fitting. ARD via
    ``dim_selection=True`` is useful for pruning axes after the gauge is
    pinned, but it is rotation-symmetric and is not itself a gauge fix.

    Parameters
    ----------
    n : int
        Number of rows ``N``. Must match the response array's row count.
    d : int
        Latent dimensionality. For ``dim_selection=True`` this is the
        ``d_max`` budget — REML will drive unused axes' precision to ∞.
    init : ``"pca"`` | ``"random"`` | array-like ``(N, d)``
        Initial values for ``t``. ``"pca"`` (default) PCA-projects the
        response down to ``d`` and centers; ``"random"`` draws uniform
        values in ``[0, 1)``; an explicit array overrides both.
    aux_prior : dict, optional
        iVAE-style identifiability fix. Keys:

        * ``u``: array-like ``(N, p)``, auxiliary covariate
          (e.g. environment label, observed RGB, …).
        * ``family``: ``"ridge"`` (default) or ``"linear"`` — choice of
          internal conditional-mean estimator ``ĥ(u)``.
        * ``strength``: positive float, the penalty weight ``μ`` in
          ``R_id = ½ μ ‖t − ĥ(u)‖²``. Pass ``"auto"`` to let REML choose
          it as one extra outer ``ρ``-axis.

        This is the principled identifiability fix (Khemakhem et al. 2020)
        when the marginal likelihood includes the log-``μ`` normalizer,
        ``ĥ`` is at least C¹, and the conditional precision is positive
        definite on the subspace anchored by ``u``.
    dim_selection : bool, default False
        Enable ARD on each latent axis. One ridge penalty per axis with
        its own log-precision; REML drives unused axes' precision to ∞.
        Combines cleanly with ``aux_prior`` or ``IsometryPenalty``: the
        gauge fix pins which axes are interpretable, ARD prunes which axes
        are needed. ARD alone does not identify a coordinate system.
    manifold : str | dict, default ``"auto"``
        Per-row update manifold. ``"auto"`` infers from the consuming basis:
        periodic/cyclic smooths use ``"circle"``, spherical smooths use
        ``"sphere"``, tensor periodic margins form a torus/cylinder product,
        and ordinary Duchon/Matérn/TPS smooths stay Euclidean. Explicit
        values include ``"euclidean"``, ``"circle"``, ``"sphere"``,
        ``"torus"``, ``"cylinder"``, or ``{"type": "interval", "lo": ...,
        "hi": ...}``.

    Examples
    --------
    >>> import gamfit
    >>> t = gamfit.LatentCoord(
    ...     n=N, d=4, init="pca",
    ...     aux_prior={"u": rgb, "family": "ridge", "strength": 1.0},
    ... )
    """

    n: int
    d: int
    init: Any = "pca"
    aux_prior: Mapping[str, Any] | None = None
    dim_selection: bool = False
    manifold: Any = "auto"
    name: str | None = None


@dataclass
class Categorical(Smooth):
    """Sum-to-zero contrast for a categorical predictor.

    A "smooth" only in the gamfit-as-engine sense — really a random
    effect with a ridge penalty on the level contrasts. Included here so
    additive fits can mix continuous smooths and categorical factors in
    one call.
    """

    levels: Any = None                   # (N,) integer level codes
    n_levels: int = 0


__all__ = [
    "Smooth",
    "Duchon",
    "BSpline",
    "TensorBSpline",
    "Matern",
    "Sphere",
    "Categorical",
    "LatentCoord",
    "PeriodicSplineCurve",
]
