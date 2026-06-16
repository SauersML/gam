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
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Sequence

if TYPE_CHECKING:
    import torch


ShapeConstraintLiteral = Literal[
    "none",
    "monotone_increasing",
    "monotone_decreasing",
    "convex",
    "concave",
]


from ._basis_protocol import BasisDescriptor as _BasisDescriptor


def _smooth_kind_name(cls: type) -> str:
    """Map a ``Smooth`` subclass to its canonical Rust ``kind`` discriminator.

    The discriminator is the same string the formula DSL uses as the
    smooth-term head (``duchon``, ``matern``, ``measurejet``, ``sphere``,
    ``bspline``, ``tensor_bspline``, ``pca``, ``periodic_spline_curve``,
    ``categorical``).
    Used by :meth:`Smooth.to_rust_descriptor` so the Rust bridge can match a
    descriptor against the formula-DSL smooth that names the same symbol.
    """
    name = cls.__name__
    if name == "TensorBSpline":
        return "tensor_bspline"
    if name == "PeriodicSplineCurve":
        return "periodic_spline_curve"
    return name.lower()


def _array_to_list(value: Any) -> Any:
    """Coerce an array-like to a plain nested list of floats.

    The Rust bridge accepts JSON-able ``Vec<Vec<f64>>`` for 2-D center
    matrices and ``Vec<f64>`` for 1-D knot vectors. Anything that does not
    look numeric round-trips as-is so downstream serialization can reject it
    with a precise error.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_array_to_list(v) for v in value]
    import numpy as np

    if isinstance(value, np.ndarray):
        if value.ndim == 1:
            return [float(v) for v in value.tolist()]
        if value.ndim == 2:
            return [[float(v) for v in row] for row in value.tolist()]
        raise ValueError(
            f"smooth descriptor array must be 1-D or 2-D; got {value.ndim}-D"
        )
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return value
    return _array_to_list(arr)


@dataclass(slots=True)
class Smooth(_BasisDescriptor):
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
    _gamfit_topology_dim: int | None = field(default=None, init=False, repr=False)
    _gamfit_tensor_k: tuple[int, int] | None = field(default=None, init=False, repr=False)
    _gamfit_tensor_periods: tuple[str | None, str | None] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _gamfit_tensor_identifiability: str = field(
        default="sum_tozero",
        init=False,
        repr=False,
    )

    # ------------------------------------------------------------------
    # Callable-basis protocol is inherited from
    # :class:`gamfit._basis_protocol.BasisDescriptor`. Concrete subclasses
    # override at least :meth:`_evaluate_torch` plus :attr:`intrinsic_dim` /
    # :attr:`basis_size`.
    # ------------------------------------------------------------------

    def to_rust_descriptor(self) -> dict[str, Any]:
        """Serialize this smooth descriptor for the Rust ``smooths=`` bridge.

        Returns a plain JSON-able dict consumed by ``FitConfig.smooths`` on
        the Rust side. The same canonical lowering produces a
        ``SmoothBasisSpec`` bit-identical to what the formula DSL would
        produce for the equivalent ``smooth(...)`` invocation — only the
        ``CenterStrategy::UserProvided`` array is threaded in addition.

        The base implementation handles the universal ``Smooth`` fields
        (``name``, ``shape_constraint``, ``double_penalty``). Each concrete
        subclass extends this with its own ``kind`` discriminator and
        kind-specific tunables.
        """
        out: dict[str, Any] = {"kind": _smooth_kind_name(type(self))}
        if self.name is not None:
            out["name"] = str(self.name)
        if self.shape_constraint is not None:
            out["shape_constraint"] = str(self.shape_constraint)
        if self.double_penalty:
            out["double_penalty"] = True
        if self.by is not None:
            # `by` is the per-row multiplier `by · s(x)`. On the formula
            # `smooths={}` descriptor path the data frame is available to the
            # Rust merge, so `by` travels as the *column name* of the gating
            # variable — exactly what the formula `s(x, by=g)` syntax resolves
            # to a `by_col`. The merge in `smooth_overrides.rs` resolves the
            # name against the data headers and wraps the inner basis in the
            # `SmoothBasisSpec::ByVariable` envelope. (Raw per-row `by` arrays
            # are the contract of the primitive numpy API, not this path; they
            # are rejected with a pointer in `_normalize_smooths`.)
            out["by"] = str(self.by)
        return out


def _as_torch_tensor(x: Any, *, name: str) -> "torch.Tensor":
    """Coerce ``x`` to a torch tensor without copying when possible.

    Used by ``Smooth.evaluate`` overrides. We intentionally do not strip a
    requires_grad flag: routing through the existing
    :mod:`gamfit.torch._basis` autograd Functions preserves the analytic
    VJP back to ``x`` for kernels that expose one.
    """
    import torch as _torch

    if isinstance(x, _torch.Tensor):
        return x
    return _torch.as_tensor(x, dtype=_torch.float64)


@dataclass(slots=True)
class Duchon(Smooth):
    """Structural amplitude/slope/curvature smoother on a cubic (r³)
    polyharmonic radial basis. Works at any d ≥ 1.

    This is **not** the classical Duchon native seminorm. The design is the
    cubic (r³) polyharmonic radial basis on the supplied centers, and the
    penalty is *structural*: three separately REML-tuned penalties act on the
    fitted function — its amplitude (deviation from the mean), its slope, and
    its curvature — each with its own smoothing parameter ``λ``. Only the
    **global mean** is left free (unpenalized); the polynomial nullspace order
    selected by ``m`` (below) is what stays exempt from the slope/curvature
    operators. Penalizing amplitude, slope, and curvature independently lets
    REML decide, from the data, how much of each structure to keep, rather
    than collapsing them into a single native-seminorm ``λ``.

    Special cases (intuition only — the penalty here is structural, not the
    native seminorm):

    * m=2, d=1 — behaves like a natural cubic smoothing spline.
    * m=2, d=2 — thin-plate-like cubic surface.
    * m=2, d ≥ 3 — generalized thin-plate-like cubic smoother.

    Parameters
    ----------
    centers : array-like of shape ``(K, d)`` — control points in ℝ^d.
        ``d`` is inferred from ``centers.shape[1]``. For 1D smooths,
        a shape ``(K,)`` 1D array is auto-promoted to ``(K, 1)``.
    m : int, default 2. Spline ORDER — selects the polynomial nullspace the
        slope/curvature operators leave unpenalized (1 → mean only, 2 → mean +
        linear, k → degree k-1). It is the nullspace order, **not** the spectral
        power ``s``.
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

    @property
    def intrinsic_dim(self) -> int:
        """``d`` inferred from :attr:`centers`. Falls back to 1 when None."""
        if self.centers is None or isinstance(self.centers, int):
            return 1
        import numpy as np

        arr = np.asarray(self.centers)
        if arr.ndim == 1:
            return 1
        if arr.ndim == 2:
            return int(arr.shape[1])
        raise ValueError(
            f"Duchon.centers must be 1D or 2D; got {arr.ndim}D"
        )

    @property
    def basis_size(self) -> int:
        """``K`` (the number of centers)."""
        if self.centers is None:
            from ._api import _DEFAULT_BASIS_K
            return int(_DEFAULT_BASIS_K)
        if isinstance(self.centers, int):
            return int(self.centers)
        import numpy as np

        return int(np.asarray(self.centers).shape[0])

    def _evaluate_torch(self, coords: Any) -> Any:
        from ._basis_eval import duchon_evaluate
        return duchon_evaluate(self, coords)

    def _evaluate_numpy(self, coords: Any) -> Any:
        from ._basis_eval import duchon_evaluate_numpy
        return duchon_evaluate_numpy(self, coords)

    def to_rust_descriptor(self) -> dict[str, Any]:
        out = super().to_rust_descriptor()
        if self.centers is not None:
            if isinstance(self.centers, int):
                out["n_centers"] = int(self.centers)
            else:
                out["centers"] = _array_to_list(self.centers)
        out["m"] = int(self.m)
        if self.length_scale is not None:
            out["length_scale"] = float(self.length_scale)
        if self.periodic_per_axis is not None:
            out["periodic_per_axis"] = [bool(b) for b in self.periodic_per_axis]
        return out

    SUPPORTED_BACKENDS: ClassVar[frozenset[str]] = frozenset({"torch", "numpy", "jax"})


@dataclass(slots=True)
class BSpline(Smooth):
    """1D B-spline (Eilers-Marx P-spline) with a difference penalty.

    For multi-dimensional inputs with DIFFERENT axis units (space × time
    being the canonical example), use :class:`TensorBSpline` instead.
    For multi-dimensional inputs with the SAME axis units (a continuous
    coordinate field), use :class:`Duchon` (isotropic radial kernel).
    Streaming row-chunked evaluation activates automatically when the
    would-be dense basis buffer exceeds ~1 GiB; no opt-in is required.
    """

    knots: Any = None                    # (K,) — auto-derived if None
    degree: int = 3
    penalty_order: int = 2
    periodic: bool = False

    @property
    def intrinsic_dim(self) -> int:
        return 1

    @property
    def basis_size(self) -> int:
        from ._basis_eval import bspline_basis_size
        return bspline_basis_size(self)

    def _evaluate_torch(self, coords: Any) -> Any:
        from ._basis_eval import bspline_evaluate
        return bspline_evaluate(self, coords)

    def _evaluate_numpy(self, coords: Any) -> Any:
        from ._basis_eval import bspline_evaluate_numpy
        return bspline_evaluate_numpy(self, coords)

    def to_rust_descriptor(self) -> dict[str, Any]:
        out = super().to_rust_descriptor()
        if self.knots is not None:
            if isinstance(self.knots, int):
                out["n_knots"] = int(self.knots)
            else:
                out["knots"] = _array_to_list(self.knots)
        out["degree"] = int(self.degree)
        out["penalty_order"] = int(self.penalty_order)
        if self.periodic:
            out["periodic"] = True
        return out

    SUPPORTED_BACKENDS: ClassVar[frozenset[str]] = frozenset({"torch", "numpy", "jax"})


@dataclass(slots=True)
class TensorBSpline(Smooth):
    """Tensor-product B-spline (mgcv ``te()`` style).

    Use when axes have different units / scales (e.g. space × time, age ×
    treatment) and isotropic radial kernels are inappropriate. Each
    marginal is a 1D :class:`BSpline`; the joint smooth is their tensor
    product with one λ per marginal selected jointly by REML.
    """

    marginals: list[BSpline] = field(default_factory=list)

    @property
    def intrinsic_dim(self) -> int:
        return len(self.marginals)

    @property
    def basis_size(self) -> int:
        from ._basis_eval import bspline_basis_size
        total = 1
        for marg in self.marginals:
            total *= bspline_basis_size(marg)
        return int(total)

    def _evaluate_torch(self, coords: Any) -> Any:
        from ._basis_eval import tensor_bspline_evaluate
        return tensor_bspline_evaluate(self, coords)

    def _evaluate_numpy(self, coords: Any) -> Any:
        from ._basis_eval import tensor_bspline_evaluate_numpy
        return tensor_bspline_evaluate_numpy(self, coords)

    def to_rust_descriptor(self) -> dict[str, Any]:
        out = super().to_rust_descriptor()
        out["marginals"] = [m.to_rust_descriptor() for m in self.marginals]
        return out

    SUPPORTED_BACKENDS: ClassVar[frozenset[str]] = frozenset({"torch", "numpy", "jax"})


@dataclass(slots=True)
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

    Notes
    -----
    Streaming row-chunked evaluation activates automatically when the
    would-be dense basis buffer exceeds ~1 GiB; no opt-in is required.
    """

    centers: Any = None
    nu: float = 1.5
    length_scale: float = 1.0
    aniso_log_scales: Sequence[float] | None = None

    @property
    def intrinsic_dim(self) -> int:
        if self.centers is None:
            raise ValueError("Matern.intrinsic_dim: centers must be provided")
        import numpy as np

        arr = np.asarray(self.centers)
        if arr.ndim == 1:
            return 1
        if arr.ndim == 2:
            return int(arr.shape[1])
        raise ValueError(f"Matern.centers must be 1D or 2D; got {arr.ndim}D")

    @property
    def basis_size(self) -> int:
        if self.centers is None:
            raise ValueError("Matern.basis_size: centers must be provided")
        import numpy as np

        return int(np.asarray(self.centers).shape[0])

    def _evaluate_torch(self, coords: Any) -> Any:
        from ._basis_eval import matern_evaluate
        return matern_evaluate(self, coords)

    def _evaluate_numpy(self, coords: Any) -> Any:
        from ._basis_eval import matern_evaluate_numpy
        return matern_evaluate_numpy(self, coords)

    def to_rust_descriptor(self) -> dict[str, Any]:
        out = super().to_rust_descriptor()
        if self.centers is not None:
            if isinstance(self.centers, int):
                out["n_centers"] = int(self.centers)
            else:
                out["centers"] = _array_to_list(self.centers)
        out["nu"] = float(self.nu)
        out["length_scale"] = float(self.length_scale)
        if self.aniso_log_scales is not None:
            out["aniso_log_scales"] = [float(v) for v in self.aniso_log_scales]
        return out

    SUPPORTED_BACKENDS: ClassVar[frozenset[str]] = frozenset({"torch", "numpy", "jax"})


@dataclass(slots=True)
class MeasureJet(Smooth):
    """Measure-jet spline (formula head ``mjs(x0, …, xd)``). Multi-d.

    Penalizes the multiscale local-jet-residual energy of the empirical
    measure: at every scale the fitted function is compared against its best
    local affine jet under the data's own sampling measure, so the smoother
    learns the geometry from data concentrated near an unknown
    low-dimensional (possibly stratified) set — centers as quadrature nodes,
    masses as μ-weights, local jet residuals as the roughness carrier, with
    no graph, mesh, or neighbor-set inside the statistical object. Every
    option is auto-derived from the data when omitted; set a field only to
    pin it. The per-scale spectral penalty split and the smoothness/density
    dials auto-enable only at large center counts (a spectrum needs many
    centers to identify); small fits use one fused penalty at Duchon-class
    cost.

    Parameters
    ----------
    centers : optional array-like ``(K, d)`` with ``K ≥ 3`` — explicit
        center coordinates (quadrature nodes of the empirical measure).
        Routes through ``CenterStrategy::UserProvided``.
    n_centers : optional int ``≥ 3`` — number of farthest-point-sampled
        centers when ``centers`` is omitted.
    s : optional float in ``(0, 2)`` — continuous smoothness order of the
        affine-jet energy.
    alpha : optional finite float — density-normalization exponent (outer
        weight ``q^(1−2α)``).
    tau : optional float ``≥ 0`` — dimensionless jet-ridge floor on the
        local slope Gram; ``0`` selects the exact pseudo-inverse.
    scales : optional int — number of scale nodes in the multiscale band.
    length_scale : optional positive float — representer (Gaussian RBF)
        range.
    double_penalty : optional bool — add the ridge-like shrinkage penalty
        alongside the jet-energy penalty. ``None`` defers to the engine
        default (on); pass ``False`` to disable it explicitly.
    """

    centers: Any = None
    n_centers: int | None = None
    s: float | None = None
    alpha: float | None = None
    tau: float | None = None
    scales: int | None = None
    length_scale: float | None = None
    # Tri-state override of the base ``bool = False`` field: the engine
    # default for measure-jet is double-penalty ON, so ``None`` must mean
    # "defer to the engine" and an explicit ``False`` must be emitted.
    double_penalty: bool | None = None

    def to_rust_descriptor(self) -> dict[str, Any]:
        out = super().to_rust_descriptor()
        if self.centers is not None:
            out["centers"] = _array_to_list(self.centers)
        if self.n_centers is not None:
            out["n_centers"] = int(self.n_centers)
        if self.s is not None:
            out["s"] = float(self.s)
        if self.alpha is not None:
            out["alpha"] = float(self.alpha)
        if self.tau is not None:
            out["tau"] = float(self.tau)
        if self.scales is not None:
            out["scales"] = int(self.scales)
        if self.length_scale is not None:
            out["length_scale"] = float(self.length_scale)
        if self.double_penalty is not None:
            out["double_penalty"] = bool(self.double_penalty)
        return out

    # MeasureJet is a formula-path smooth consumed by the Rust core (the
    # ``mjs(...)`` head plus the ``smooths=`` override bridge); it has no
    # Python-side basis evaluator. The empty set is the honest contract:
    # there are no `_evaluate_<backend>` paths to advertise.
    SUPPORTED_BACKENDS: ClassVar[frozenset[str]] = frozenset()


@dataclass(init=False, slots=True)
class Pca(Smooth):
    """Precomputed-PCA linear feature projection as a first-class smooth.

    A common pattern is to manually do
    ``from _pca_basis import load_pc_basis; basis = load_pc_basis(K=64);
    X_proj = X @ basis`` and then fit gamfit on the projected ``(N, K)``
    data. Pca makes that recurring projection a GAM basis. It also captures
    the Schur elimination benefit: when ``D >> K_pca`` (for example
    ``D=7168, K=64``), gamfit projects through the cached ``(D, K_pca)``
    basis once and fits the smaller ``(N, K_pca)`` design with a ridge
    penalty on PCA coefficients, rather than materializing a full ``(N, D)``
    smooth and ``(D, D)`` penalty.

    Parameters
    ----------
    K : latent rank. Required by formula use; defaults to the provided
        basis width when ``basis`` is supplied.
    basis : optional array-like of shape ``(D, K)``. A fixed precomputed
        projection matrix, e.g. ``_pca_basis.load_pc_basis(K=64)``.
    lazy_path : optional path to a memmap-able ``.npy`` scores matrix ``(N, K)``.
    centered : if True, subtract the training feature mean before projection.
    smooth_penalty : ridge multiplier for PCA coefficients.
    """

    K: int | None = None
    basis: Any | None = None
    lazy_path: Path | None = None
    chunk_size: int = 4096
    centered: bool = True
    smooth_penalty: float = 1.0

    def __init__(
        self,
        K: int | None = None,
        basis: Any | None = None,
        lazy_path: Path | None = None,
        chunk_size: int = 4096,
        centered: bool = True,
        name: str | None = None,
        smooth_penalty: float = 1.0,
        by: Any | None = None,
        double_penalty: bool = False,
        shape_constraint: ShapeConstraintLiteral | None = None,
    ) -> None:
        self.name = name
        self.by = by
        self.double_penalty = double_penalty
        self.shape_constraint = shape_constraint
        self.K = K
        self.basis = basis
        self.lazy_path = None if lazy_path is None else Path(lazy_path)
        self.chunk_size = int(chunk_size)
        self.centered = centered
        self.smooth_penalty = smooth_penalty

    @property
    def intrinsic_dim(self) -> int:
        if self.basis is None:
            raise ValueError("Pca.intrinsic_dim: basis must be provided")
        import numpy as np

        return int(np.asarray(self.basis).shape[0])

    @property
    def basis_size(self) -> int:
        if self.basis is not None:
            import numpy as np

            return int(np.asarray(self.basis).shape[1])
        if self.K is None:
            raise ValueError("Pca.basis_size: K or basis must be provided")
        return int(self.K)

    def _evaluate_torch(self, coords: Any) -> Any:
        from ._basis_eval import pca_evaluate
        return pca_evaluate(self, coords)

    def _evaluate_numpy(self, coords: Any) -> Any:
        from ._basis_eval import pca_evaluate_numpy
        return pca_evaluate_numpy(self, coords)

    def to_rust_descriptor(self) -> dict[str, Any]:
        out = super().to_rust_descriptor()
        if self.K is not None:
            out["K"] = int(self.K)
        if self.basis is not None:
            out["basis"] = _array_to_list(self.basis)
        if self.lazy_path is not None:
            out["lazy_path"] = str(self.lazy_path)
        out["chunk_size"] = int(self.chunk_size)
        out["centered"] = bool(self.centered)
        out["smooth_penalty"] = float(self.smooth_penalty)
        return out

    SUPPORTED_BACKENDS: ClassVar[frozenset[str]] = frozenset({"torch", "numpy", "jax"})


@dataclass(slots=True)
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
    centers : optional explicit ``(K, 2)`` center array (lat, lon) in the
        same angular convention as ``radians``. When provided, ``n_centers``
        is ignored and ``K = centers.shape[0]`` drives the basis size.

    Notes
    -----
    Centers are a property of the basis, not of the evaluation set. If the
    user does not supply explicit ``centers``, the descriptor resolves
    centers on first evaluation: farthest-point sampling from the eval
    rows when there are at least ``n_centers`` of them, otherwise a
    deterministic Fibonacci-lattice fallback so that small/test eval sets
    still work. The resolved centers are then cached and reused for every
    later evaluation.

    Streaming row-chunked evaluation activates automatically when the
    would-be dense basis buffer exceeds ~1 GiB; no opt-in is required.
    """

    n_centers: int = 50
    penalty_order: int = 2
    kernel: str = "sobolev"
    radians: bool = False
    centers: Any = None

    @property
    def intrinsic_dim(self) -> int:
        return 2

    @property
    def basis_size(self) -> int:
        """Analytic basis dimension — no Rust call required.

        - ``kernel='sobolev' | 'pseudo'``: ``K = n_centers - 1`` after the
          area-weighted sum-to-zero identifiability transform applied by
          the Rust builder.
        - ``kernel='harmonic'``: ``K = L * (L + 2)`` where ``L = n_centers``
          is the truncation degree.
        """
        if self.centers is not None:
            import numpy as np

            k = int(np.asarray(self.centers, dtype=np.float64).shape[0])
        else:
            k = int(self.n_centers)
        if str(self.kernel).lower() == "harmonic":
            return k * (k + 2)
        return k - 1

    def _resolve_centers(self, coords: Any) -> Any:
        """Resolve and cache the basis center matrix.

        For ``kernel='harmonic'`` no centers are needed and this returns
        ``None``. For Wahba kernels the resolution order is:

        1. User-supplied ``centers``.
        2. Farthest-point sampling from ``coords`` when it has at least
           ``n_centers`` rows.
        3. Deterministic Fibonacci-lattice fallback otherwise.
        """
        if str(self.kernel).lower() == "harmonic":
            return None

        cached = getattr(self, "_cached_centers", None)
        if cached is not None:
            return cached

        import numpy as np

        if self.centers is not None:
            ctrs = np.ascontiguousarray(
                np.asarray(self.centers, dtype=np.float64)
            )
            if ctrs.ndim != 2 or ctrs.shape[1] != 2:
                raise ValueError(
                    f"Sphere.centers must have shape (K, 2); got {ctrs.shape}"
                )
        else:
            n_centers_i = int(self.n_centers)
            pts = np.ascontiguousarray(np.asarray(coords, dtype=np.float64))
            if pts.shape[0] >= n_centers_i:
                from . import _api

                ctrs = np.asarray(
                    _api.rust_module().sphere_select_farthest_point_centers(
                        pts, n_centers_i, bool(self.radians),
                    ),
                    dtype=np.float64,
                )
            else:
                ctrs = _fibonacci_sphere_lat_lon(n_centers_i, bool(self.radians))

        object.__setattr__(self, "_cached_centers", ctrs)
        return ctrs

    def _evaluate_numpy(self, coords: Any) -> Any:
        from ._basis_eval import sphere_evaluate_numpy
        return sphere_evaluate_numpy(self, coords)

    def _evaluate_torch(self, coords: Any) -> Any:
        from ._basis_eval import sphere_evaluate
        return sphere_evaluate(self, coords)

    def to_rust_descriptor(self) -> dict[str, Any]:
        out = super().to_rust_descriptor()
        out["n_centers"] = int(self.n_centers)
        out["penalty_order"] = int(self.penalty_order)
        out["kernel"] = str(self.kernel)
        out["radians"] = bool(self.radians)
        if self.centers is not None:
            out["centers"] = _array_to_list(self.centers)
        return out

    SUPPORTED_BACKENDS: ClassVar[frozenset[str]] = frozenset({"torch", "numpy", "jax"})


def _fibonacci_sphere_lat_lon(n: int, radians: bool) -> Any:
    """Deterministic quasi-uniform sphere lattice — (n, 2) [lat, lon].

    Golden-angle Fibonacci spiral on S². Used as the default center set
    when ``Sphere.evaluate`` is called without pre-fit context (eval rows
    < ``n_centers`` and no user-supplied ``centers``).
    """
    import numpy as np

    if n < 2:
        raise ValueError(f"Sphere needs at least 2 centers; got n_centers={n}")
    k = np.arange(n, dtype=np.float64)
    z = 1.0 - (2.0 * k + 1.0) / float(n)
    z = np.clip(z, -1.0, 1.0)
    lat_rad = np.arcsin(z)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    lon_rad = (k * golden_angle) % (2.0 * np.pi)
    lon_rad = np.where(lon_rad > np.pi, lon_rad - 2.0 * np.pi, lon_rad)
    if radians:
        return np.column_stack([lat_rad, lon_rad]).astype(np.float64)
    return np.column_stack([np.degrees(lat_rad), np.degrees(lon_rad)]).astype(
        np.float64
    )


@dataclass(slots=True)
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

    @property
    def intrinsic_dim(self) -> int:
        return 1

    @property
    def basis_size(self) -> int:
        return int(self.n_knots)

    def _evaluate_torch(self, coords: Any) -> Any:
        from ._basis_eval import _periodic_curve_basis
        t = coords[:, 0]
        # Reduce modulo 1 so the cyclic forward sees t ∈ [0, 1).
        from ._basis_protocol import _torch
        torch = _torch()
        t_mod = t - torch.floor(t)
        return _periodic_curve_basis(t_mod, int(self.n_knots), int(self.degree))

    def _evaluate_numpy(self, coords: Any) -> Any:
        import numpy as np

        from ._basis_eval import periodic_curve_evaluate_numpy

        coords_np = np.asarray(coords, dtype=np.float64)
        # Match the torch path: reduce t modulo 1 before evaluating.
        coords_np = coords_np.copy()
        coords_np[:, 0] = coords_np[:, 0] - np.floor(coords_np[:, 0])
        return periodic_curve_evaluate_numpy(self, coords_np)

    def to_rust_descriptor(self) -> dict[str, Any]:
        out = super().to_rust_descriptor()
        out["n_knots"] = int(self.n_knots)
        out["degree"] = int(self.degree)
        out["output_dim"] = int(self.output_dim)
        out["penalty_order"] = int(self.penalty_order)
        return out

    SUPPORTED_BACKENDS: ClassVar[frozenset[str]] = frozenset({"torch", "numpy", "jax"})


@dataclass(slots=True)
class LatentCoord:
    """Per-row latent coordinates ``t ∈ ℝ^{N × d}`` as a first-class parameter.

    See ``src/terms/latent_coord.rs`` for the Rust side.

    The familiar GAM picture is

    .. code-block::

        y = Φ(x) · β + ε,   penalty(β; ρ),   x observed.

    ``LatentCoord`` promotes one of the covariates ``x`` from *observed* to
    *latent*: ``t`` is estimated per row alongside the spline coefficients
    ``β`` and the smoothing parameters ``ρ``. Mechanically this is the same
    problem the kernel-shape ``ψ`` (log-anisotropy) machinery already solves:
    both move the design matrix Φ and reuse the IFT-warm-started outer optimizer,
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
        * ``strength``: ``"auto"`` (default) or a positive float, the
          penalty weight ``μ`` in ``R_id = ½ μ ‖t − ĥ(u)‖²``. ``"auto"``
          lets REML choose it as one extra outer ``ρ``-axis.

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
    retraction : str | dict | list, default ``"euclidean"``
        Per-row latent update retraction. Values are passed to the Rust
        workflow as strings such as ``"euclidean"``, ``"circle"``, or
        ``"sphere"``, or as products like ``{"type": "product", "parts": [...]}``.

    Examples
    --------
    >>> import gamfit
    >>> t = gamfit.LatentCoord(
    ...     n=N, d=4, init="pca",
    ...     aux_prior={"u": rgb, "family": "ridge", "strength": "auto"},
    ... )
    """

    n: int
    d: int
    init: Any = "pca"
    aux_prior: Mapping[str, Any] | None = None
    dim_selection: bool = False
    manifold: Any = "auto"
    retraction: Any = "euclidean"
    name: str | None = None


@dataclass(slots=True)
class Categorical(Smooth):
    """Sum-to-zero contrast for a categorical predictor.

    A "smooth" only in the gamfit-as-engine sense — really a random
    effect with a ridge penalty on the level contrasts. Included here so
    additive fits can mix continuous smooths and categorical factors in
    one call.
    """

    levels: Any = None                   # (N,) integer level codes
    n_levels: int = 0

    # Categorical is a level-code carrier consumed by formula compilation
    # (it materializes into sum-to-zero contrasts inside the Rust core), not
    # a descriptor with its own basis evaluator. The empty set is the honest
    # contract: there are no `_evaluate_<backend>` paths to advertise.
    SUPPORTED_BACKENDS: ClassVar[frozenset[str]] = frozenset()

    def to_rust_descriptor(self) -> dict[str, Any]:
        out = super().to_rust_descriptor()
        if self.levels is not None:
            out["levels"] = _array_to_list(self.levels)
        out["n_levels"] = int(self.n_levels)
        return out


__all__ = [
    "Smooth",
    "Duchon",
    "BSpline",
    "TensorBSpline",
    "Matern",
    "MeasureJet",
    "Pca",
    "Sphere",
    "Categorical",
    "LatentCoord",
    "PeriodicSplineCurve",
]
