"""Four-tuple composition of smooth terms: ``Smooth + Penalty + Manifold + Basis``.

The Python-boundary composition layer for ``gamfit.Smooth(latent=..., basis=...,
penalty=...)``. Each component is independently swappable: a ``Manifold``
declares its coordinate domain and intrinsic dimension; a ``Basis`` declares
which manifolds it can ride on; a ``Penalty`` (or sum of penalties) declares
its regularization shape. The ``Smooth`` constructor enforces the
``Manifold`` x ``Basis`` compatibility contract and lowers the whole four-tuple
to a Rust descriptor for the engine.

Operator overloading rule: ``Penalty + Penalty`` returns a ``PenaltySum``;
the components stay addressable for diagnostics. ``PenaltySum.to_rust_descriptor()``
round-trips both children verbatim under the ``"children"`` key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Sequence

from . import topology as _topology


__all__ = [
    "Manifold",
    "EuclideanLatent",
    "CircleLatent",
    "SphereLatent",
    "TorusLatent",
    "CylinderLatent",
    "Circle",
    "Sphere",
    "Torus",
    "Cylinder",
    "Euclidean",
    "Basis",
    "Fourier",
    "BSplineBasis",
    "SphericalHarmonics",
    "RBF",
    "TensorProduct",
    "PenaltyDescriptor",
    "PenaltySum",
    "ARD",
    "IBP",
    "Sparsity",
    "Isometry",
    "TotalVariation",
    "NuclearNorm",
    "Orthogonality",
    "ScadMcp",
    "BlockSparsity",
    "BlockOrthogonality",
    "ComposedSmooth",
    "compose_smooth",
]


# --- Manifolds ---------------------------------------------------------------


class Manifold:
    """Coordinate-domain descriptor; declares a compatibility contract with bases."""

    kind: ClassVar[str] = "manifold"
    dim: ClassVar[int] = 0
    periodic_axes: ClassVar[tuple[bool, ...]] = ()
    compatible_bases: ClassVar[frozenset[str]] = frozenset()

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "dim": self.dim, "periodic_axes": list(self.periodic_axes)}

    def default_basis(self) -> "Basis":
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class EuclideanLatent(Manifold):
    kind = "euclidean"
    dim = 1
    periodic_axes = (False,)
    compatible_bases = frozenset({"bspline", "rbf", "tensor"})

    def __init__(self, d: int = 1) -> None:
        if int(d) < 1:
            raise ValueError("EuclideanLatent.d must be >= 1")
        self.dim = int(d)
        self.periodic_axes = tuple(False for _ in range(self.dim))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "dim": self.dim, "periodic_axes": list(self.periodic_axes)}

    def default_basis(self) -> "Basis":
        return BSplineBasis()


class CircleLatent(Manifold):
    kind = "circle"
    dim = 1
    periodic_axes = (True,)
    compatible_bases = frozenset({"fourier", "bspline_periodic"})

    def default_basis(self) -> "Basis":
        return Fourier(harmonics=4)


class SphereLatent(Manifold):
    kind = "sphere"
    dim = 2
    periodic_axes = (False, False)  # intrinsic 2-sphere, not lat/lon periodic
    compatible_bases = frozenset({"spherical_harmonics", "rbf"})

    def default_basis(self) -> "Basis":
        return SphericalHarmonics(degree=4)


class TorusLatent(Manifold):
    kind = "torus"
    dim = 2
    periodic_axes = (True, True)
    compatible_bases = frozenset({"tensor", "fourier"})

    def default_basis(self) -> "Basis":
        return TensorProduct(marginals=(Fourier(harmonics=3), Fourier(harmonics=3)))


class CylinderLatent(Manifold):
    kind = "cylinder"
    dim = 2
    periodic_axes = (True, False)
    compatible_bases = frozenset({"tensor"})

    def default_basis(self) -> "Basis":
        return TensorProduct(marginals=(Fourier(harmonics=3), BSplineBasis()))


# Singleton-style manifold tokens — both an instance (usable as default) and a class.
Circle = CircleLatent
Sphere = SphereLatent
Torus = TorusLatent
Cylinder = CylinderLatent
Euclidean = EuclideanLatent


def _coerce_manifold(latent: Any) -> Manifold:
    if isinstance(latent, Manifold):
        return latent
    if isinstance(latent, type) and issubclass(latent, Manifold):
        return latent()
    if isinstance(latent, str):
        lookup = {
            "circle": CircleLatent,
            "sphere": SphereLatent,
            "torus": TorusLatent,
            "cylinder": CylinderLatent,
            "euclidean": EuclideanLatent,
        }
        key = latent.lower()
        if key not in lookup:
            raise ValueError(f"unknown manifold name {latent!r}; expected one of {sorted(lookup)}")
        return lookup[key]()
    raise TypeError(
        f"latent must be a Manifold instance/class or a manifold name string, got {type(latent).__name__}"
    )


# --- Bases -------------------------------------------------------------------


class Basis:
    """Function-space basis; declares which manifolds it can ride on."""

    kind: ClassVar[str] = "basis"
    compatible_manifolds: ClassVar[frozenset[str]] = frozenset()

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind}

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


@dataclass(slots=True)
class Fourier(Basis):
    """Real Fourier basis on a periodic 1-D coordinate (circle / torus marginal)."""

    harmonics: int = 4

    kind: ClassVar[str] = "fourier"
    compatible_manifolds: ClassVar[frozenset[str]] = frozenset({"circle", "torus", "cylinder"})

    def __post_init__(self) -> None:
        if int(self.harmonics) < 1:
            raise ValueError("Fourier.harmonics must be >= 1")
        self.harmonics = int(self.harmonics)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "harmonics": self.harmonics}


@dataclass(slots=True)
class BSplineBasis(Basis):
    """B-spline basis on an open Euclidean 1-D coordinate."""

    n_knots: int = 20
    degree: int = 3
    penalty_order: int = 2
    periodic: bool = False

    kind: ClassVar[str] = "bspline"
    compatible_manifolds: ClassVar[frozenset[str]] = frozenset({"euclidean", "cylinder"})

    def __post_init__(self) -> None:
        if int(self.n_knots) < self.degree + 1:
            raise ValueError("BSplineBasis.n_knots must exceed degree")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "n_knots": int(self.n_knots),
            "degree": int(self.degree),
            "penalty_order": int(self.penalty_order),
            "periodic": bool(self.periodic),
        }


@dataclass(slots=True)
class SphericalHarmonics(Basis):
    """Spherical-harmonic basis on the intrinsic 2-sphere."""

    degree: int = 4
    penalty_order: int = 2

    kind: ClassVar[str] = "spherical_harmonics"
    compatible_manifolds: ClassVar[frozenset[str]] = frozenset({"sphere"})

    def __post_init__(self) -> None:
        if int(self.degree) < 0:
            raise ValueError("SphericalHarmonics.degree must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "degree": int(self.degree),
            "penalty_order": int(self.penalty_order),
        }


@dataclass(slots=True)
class RBF(Basis):
    """Isotropic radial-basis kernel (Duchon / Matern-style); rides on Euclidean or sphere."""

    n_centers: int = 64
    m: int = 2
    length_scale: float | None = None

    kind: ClassVar[str] = "rbf"
    compatible_manifolds: ClassVar[frozenset[str]] = frozenset({"euclidean", "sphere"})

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "n_centers": int(self.n_centers),
            "m": int(self.m),
            "length_scale": None if self.length_scale is None else float(self.length_scale),
        }


@dataclass(slots=True)
class TensorProduct(Basis):
    """Tensor product of 1-D marginals; rides on product manifolds (torus, cylinder)."""

    marginals: Sequence[Basis] = ()

    kind: ClassVar[str] = "tensor"
    compatible_manifolds: ClassVar[frozenset[str]] = frozenset({"torus", "cylinder", "euclidean"})

    def __post_init__(self) -> None:
        marginals = tuple(self.marginals)
        if len(marginals) < 2:
            raise ValueError("TensorProduct requires >= 2 marginal bases")
        for m in marginals:
            if not isinstance(m, Basis):
                raise TypeError(f"TensorProduct marginals must be Basis instances, got {type(m).__name__}")
        object.__setattr__(self, "marginals", marginals)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "marginals": [m.to_dict() for m in self.marginals],
        }


def _coerce_basis(basis: Any, manifold: Manifold) -> Basis:
    if basis is None:
        return manifold.default_basis()
    if isinstance(basis, Basis):
        return basis
    if isinstance(basis, type) and issubclass(basis, Basis):
        return basis()
    raise TypeError(f"basis must be a Basis instance/class or None, got {type(basis).__name__}")


def _check_compatibility(manifold: Manifold, basis: Basis) -> None:
    # Tensor product must have one marginal per manifold axis, and each marginal
    # must be compatible with its axis-local topology (periodic→Fourier/bspline_periodic, open→BSpline).
    if isinstance(basis, TensorProduct):
        if len(basis.marginals) != manifold.dim:
            raise ValueError(
                f"TensorProduct has {len(basis.marginals)} marginals but {type(manifold).__name__} has dim={manifold.dim}"
            )
        for axis, (periodic, m) in enumerate(zip(manifold.periodic_axes, basis.marginals)):
            if periodic and isinstance(m, BSplineBasis) and not m.periodic:
                raise ValueError(
                    f"axis {axis} of {type(manifold).__name__} is periodic; marginal BSplineBasis must set periodic=True or use Fourier"
                )
            if not periodic and isinstance(m, Fourier):
                raise ValueError(
                    f"axis {axis} of {type(manifold).__name__} is open; cannot use Fourier marginal"
                )
        return
    if manifold.kind not in basis.compatible_manifolds:
        raise ValueError(
            f"{type(basis).__name__} (kind={basis.kind}) is not compatible with "
            f"{type(manifold).__name__} (kind={manifold.kind}); compatible manifolds for this basis: "
            f"{sorted(basis.compatible_manifolds)}"
        )


# --- Penalties ---------------------------------------------------------------


class PenaltyDescriptor:
    """Python-boundary penalty descriptor with operator overloading.

    Each concrete subclass declares ``KIND`` (the Rust descriptor's "kind"
    string) and ``_payload()`` returning the rest of the rust dict. The base
    class enforces ``+`` returns a ``PenaltySum`` flattening any nested sums.
    """

    KIND: ClassVar[str] = "penalty"

    def _payload(self) -> dict[str, Any]:
        return {}

    def to_rust_descriptor(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": self.KIND}
        payload.update(self._payload())
        return payload

    def __add__(self, other: Any) -> "PenaltySum":
        if not isinstance(other, PenaltyDescriptor):
            return NotImplemented
        children: list[PenaltyDescriptor] = []
        for term in (self, other):
            if isinstance(term, PenaltySum):
                children.extend(term.children)
            else:
                children.append(term)
        return PenaltySum(children=tuple(children))

    def __radd__(self, other: Any) -> "PenaltySum":
        if not isinstance(other, PenaltyDescriptor):
            return NotImplemented
        return other.__add__(self)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


@dataclass(slots=True)
class PenaltySum(PenaltyDescriptor):
    """Sum-composite of individual penalties; children remain addressable."""

    children: tuple[PenaltyDescriptor, ...] = ()

    KIND: ClassVar[str] = "sum"

    def __post_init__(self) -> None:
        if len(self.children) < 1:
            raise ValueError("PenaltySum requires >= 1 child")
        for c in self.children:
            if not isinstance(c, PenaltyDescriptor):
                raise TypeError(
                    f"PenaltySum child must be PenaltyDescriptor, got {type(c).__name__}"
                )
            if isinstance(c, PenaltySum):
                raise TypeError("PenaltySum children must already be flattened")

    def to_rust_descriptor(self) -> dict[str, Any]:
        return {
            "kind": self.KIND,
            "children": [c.to_rust_descriptor() for c in self.children],
        }

    def __iter__(self):
        return iter(self.children)

    def __len__(self) -> int:
        return len(self.children)

    def __getitem__(self, idx: int) -> PenaltyDescriptor:
        return self.children[idx]


# Concrete penalty descriptors. Each carries a small kwargs payload and lowers
# to a Rust descriptor dict matching the keys gam-pyffi already understands.
@dataclass(slots=True)
class ARD(PenaltyDescriptor):
    """Automatic Relevance Determination: one weight per latent axis."""

    n_axes: int | None = None
    weight: Any = "auto"
    target: Any = None
    KIND: ClassVar[str] = "ard"

    def _payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {"weight": self.weight}
        if self.n_axes is not None:
            out["n_axes"] = int(self.n_axes)
        if self.target is not None:
            out["target"] = self.target
        return out


@dataclass(slots=True)
class IBP(PenaltyDescriptor):
    """Indian-Buffet-Process assignment penalty (latent-feature usage prior)."""

    alpha: float = 1.0
    weight: Any = "auto"
    target: Any = None
    KIND: ClassVar[str] = "ibp_assignment"

    def _payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {"weight": self.weight, "alpha": float(self.alpha)}
        if self.target is not None:
            out["target"] = self.target
        return out


@dataclass(slots=True)
class Sparsity(PenaltyDescriptor):
    """Smoothed L1 element-wise sparsity."""

    eps: float = 1e-3
    weight: Any = "auto"
    target: Any = None
    KIND: ClassVar[str] = "sparsity"

    def _payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {"weight": self.weight, "eps": float(self.eps)}
        if self.target is not None:
            out["target"] = self.target
        return out


@dataclass(slots=True)
class Isometry(PenaltyDescriptor):
    """Isometry penalty pulling the decoder pullback toward a reference metric."""

    weight: Any = "auto"
    target: Any = None
    KIND: ClassVar[str] = "isometry"

    def _payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {"weight": self.weight}
        if self.target is not None:
            out["target"] = self.target
        return out


@dataclass(slots=True)
class TotalVariation(PenaltyDescriptor):
    """Smoothed L1 on first differences (piecewise-constant prior)."""

    eps: float = 1e-3
    weight: Any = "auto"
    target: Any = None
    KIND: ClassVar[str] = "total_variation"

    def _payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {"weight": self.weight, "eps": float(self.eps)}
        if self.target is not None:
            out["target"] = self.target
        return out


@dataclass(slots=True)
class NuclearNorm(PenaltyDescriptor):
    """Smoothed L1 on singular values (low-rank prior)."""

    eps: float = 1e-3
    weight: Any = "auto"
    target: Any = None
    KIND: ClassVar[str] = "nuclear_norm"

    def _payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {"weight": self.weight, "eps": float(self.eps)}
        if self.target is not None:
            out["target"] = self.target
        return out


@dataclass(slots=True)
class Orthogonality(PenaltyDescriptor):
    """Rotation-gauge fix: penalize latent-axis correlations."""

    weight: Any = "auto"
    target: Any = None
    KIND: ClassVar[str] = "orthogonality"

    def _payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {"weight": self.weight}
        if self.target is not None:
            out["target"] = self.target
        return out


@dataclass(slots=True)
class ScadMcp(PenaltyDescriptor):
    """Concave SCAD/MCP element-wise sparsity."""

    a: float = 3.7
    lam: float = 1.0
    weight: Any = "auto"
    target: Any = None
    KIND: ClassVar[str] = "scad_mcp"

    def _payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {"weight": self.weight, "a": float(self.a), "lam": float(self.lam)}
        if self.target is not None:
            out["target"] = self.target
        return out


@dataclass(slots=True)
class BlockSparsity(PenaltyDescriptor):
    """Group-lasso block sparsity over predefined latent-axis groups."""

    groups: Sequence[Sequence[int]] = ()
    eps: float = 1e-3
    weight: Any = "auto"
    target: Any = None
    KIND: ClassVar[str] = "block_sparsity"

    def _payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "weight": self.weight,
            "eps": float(self.eps),
            "groups": [list(g) for g in self.groups],
        }
        if self.target is not None:
            out["target"] = self.target
        return out


@dataclass(slots=True)
class BlockOrthogonality(PenaltyDescriptor):
    """Between-block-only orthogonality; blocks internally free."""

    blocks: Sequence[Sequence[int]] = ()
    weight: Any = "auto"
    target: Any = None
    KIND: ClassVar[str] = "block_orthogonality"

    def _payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "weight": self.weight,
            "blocks": [list(b) for b in self.blocks],
        }
        if self.target is not None:
            out["target"] = self.target
        return out


def _coerce_penalty(penalty: Any, manifold: Manifold) -> PenaltyDescriptor | None:
    if penalty is None:
        # Magic-by-default: pick a sensible penalty for the manifold's intrinsic dim.
        # ARD on the latent axes is the safe default for d>=2; for d=1 use Sparsity.
        if manifold.dim >= 2:
            return ARD(n_axes=manifold.dim)
        return Sparsity()
    if isinstance(penalty, PenaltyDescriptor):
        return penalty
    if hasattr(penalty, "to_rust_descriptor") and not isinstance(penalty, PenaltyDescriptor):
        # Wrap a foreign descriptor (e.g. the rust-side ARDPenalty) by capturing its dict.
        captured = dict(penalty.to_rust_descriptor())

        class _WrappedForeign(PenaltyDescriptor):
            KIND = captured.get("kind", "foreign")

            def to_rust_descriptor(self_inner) -> dict[str, Any]:
                return dict(captured)

        return _WrappedForeign()
    raise TypeError(
        f"penalty must be a PenaltyDescriptor (or sum) or None, got {type(penalty).__name__}"
    )


# --- ComposedSmooth ----------------------------------------------------------


def _topology_smooth(manifold: Manifold, basis: Basis) -> Any:
    """Lower (manifold, basis) to one of the existing Smooth-spec factories.

    This keeps the formula path + Rust dispatcher unchanged: composed smooths
    reuse the proven topology.Circle/Sphere/Torus/Cylinder/EuclideanPatch
    constructors and just attach extra composition metadata.
    """
    if isinstance(manifold, CircleLatent):
        # Fourier ↔ closed-loop with n_knots ≈ 2*harmonics+1; BSpline periodic uses n_knots directly.
        if isinstance(basis, Fourier):
            return _topology.Circle(n_knots=max(2 * basis.harmonics + 1, 4))
        if isinstance(basis, BSplineBasis):
            return _topology.Circle(n_knots=basis.n_knots, degree=basis.degree, penalty_order=basis.penalty_order)
    if isinstance(manifold, SphereLatent):
        if isinstance(basis, SphericalHarmonics):
            return _topology.Sphere(n_knots=(basis.degree + 1) ** 2, penalty_order=basis.penalty_order, kernel="harmonic")
        if isinstance(basis, RBF):
            return _topology.Sphere(n_knots=basis.n_centers, kernel="sobolev")
    if isinstance(manifold, TorusLatent):
        if isinstance(basis, TensorProduct):
            knots = []
            for m in basis.marginals:
                if isinstance(m, Fourier):
                    knots.append(max(2 * m.harmonics + 1, 4))
                elif isinstance(m, BSplineBasis):
                    knots.append(m.n_knots)
                else:
                    knots.append(8)
            return _topology.Torus(n_knots=tuple(knots[:2]))
    if isinstance(manifold, CylinderLatent):
        if isinstance(basis, TensorProduct):
            knots = []
            for m in basis.marginals:
                if isinstance(m, Fourier):
                    knots.append(max(2 * m.harmonics + 1, 4))
                elif isinstance(m, BSplineBasis):
                    knots.append(m.n_knots)
                else:
                    knots.append(8)
            return _topology.Cylinder(n_knots=tuple(knots[:2]))
    if isinstance(manifold, EuclideanLatent):
        if isinstance(basis, BSplineBasis):
            return _topology.EuclideanPatch(d=manifold.dim, n_centers=basis.n_knots)
        if isinstance(basis, RBF):
            return _topology.EuclideanPatch(
                d=manifold.dim, n_centers=basis.n_centers, m=basis.m, length_scale=basis.length_scale
            )
    # Fallback: still return an EuclideanPatch so downstream paths receive a real Smooth.
    return _topology.EuclideanPatch(d=max(manifold.dim, 1))


def compose_smooth(
    latent: Any,
    basis: Any = None,
    penalty: Any = None,
    *,
    name: str | None = None,
    by: Any = None,
    double_penalty: bool = False,
    shape_constraint: Any = None,
) -> "ComposedSmooth":
    """Compose a four-tuple (Smooth, Manifold, Basis, Penalty) into a Smooth instance.

    Magic defaults: missing ``basis`` uses ``manifold.default_basis()``; missing
    ``penalty`` uses an ARD on the latent axes (or Sparsity for d=1).
    """
    manifold = _coerce_manifold(latent)
    coerced_basis = _coerce_basis(basis, manifold)
    _check_compatibility(manifold, coerced_basis)
    coerced_penalty = _coerce_penalty(penalty, manifold)
    inner = _topology_smooth(manifold, coerced_basis)
    composed = ComposedSmooth(
        manifold=manifold,
        basis=coerced_basis,
        penalty=coerced_penalty,
        inner=inner,
        name=name if name is not None else getattr(inner, "name", None),
        by=by if by is not None else getattr(inner, "by", None),
        double_penalty=bool(double_penalty or getattr(inner, "double_penalty", False)),
        shape_constraint=shape_constraint
        if shape_constraint is not None
        else getattr(inner, "shape_constraint", None),
    )
    return composed


# Defer the Smooth-subclass definition to a closure that imports lazily, so the
# composition module is import-safe in any order relative to gamfit.smooth.
from .smooth import Smooth as _SmoothBase  # noqa: E402


class ComposedSmooth(_SmoothBase):
    """A ``Smooth`` carrying its (Manifold, Basis, Penalty) components verbatim.

    The instance is a real ``gamfit.Smooth`` (so ``isinstance(x, Smooth)`` and
    every downstream dispatcher keep working), while the four components remain
    addressable for diagnostics through ``manifold`` / ``basis`` / ``penalty`` /
    ``inner``.
    """

    __slots__ = ("manifold", "basis", "penalty", "inner")

    def __init__(
        self,
        *,
        manifold: Manifold,
        basis: Basis,
        penalty: PenaltyDescriptor | None,
        inner: Any,
        name: str | None = None,
        by: Any = None,
        double_penalty: bool = False,
        shape_constraint: Any = None,
    ) -> None:
        # Use object.__setattr__ to bypass slotted dataclass init for the base.
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "by", by)
        object.__setattr__(self, "double_penalty", bool(double_penalty))
        object.__setattr__(self, "shape_constraint", shape_constraint)
        object.__setattr__(self, "_gamfit_topology_dim", getattr(inner, "_gamfit_topology_dim", None))
        object.__setattr__(self, "_gamfit_tensor_k", getattr(inner, "_gamfit_tensor_k", None))
        object.__setattr__(self, "_gamfit_tensor_periods", getattr(inner, "_gamfit_tensor_periods", None))
        object.__setattr__(
            self, "_gamfit_tensor_identifiability", getattr(inner, "_gamfit_tensor_identifiability", "sum_tozero")
        )
        object.__setattr__(self, "manifold", manifold)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "penalty", penalty)
        object.__setattr__(self, "inner", inner)

    def to_rust_descriptor(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": "composed_smooth",
            "manifold": self.manifold.to_dict(),
            "basis": self.basis.to_dict(),
            "penalty": None if self.penalty is None else self.penalty.to_rust_descriptor(),
            "inner_kind": type(self.inner).__name__,
        }
        if self.name is not None:
            payload["name"] = self.name
        if self.double_penalty:
            payload["double_penalty"] = True
        if self.shape_constraint is not None:
            payload["shape_constraint"] = str(self.shape_constraint)
        return payload

    def __repr__(self) -> str:
        return (
            f"ComposedSmooth(manifold={self.manifold!r}, basis={self.basis!r}, "
            f"penalty={self.penalty!r})"
        )
