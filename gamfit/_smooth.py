"""Composable smooth term: a :class:`Smooth` ties together a latent manifold,
a basis descriptor, and one (composite) penalty.

``Smooth(latent=..., basis=..., penalty=...)`` is itself a
:class:`BasisDescriptor` so it composes uniformly with the rest of the
protocol: ``.evaluate``, ``.jacobian``, ``.hessian`` all route through the
underlying basis.

Eager compatibility checks
--------------------------
* ``latent.dimension`` must match ``basis.input_dim`` (when statically
  available). Mismatched specs error at construction with a clear message.
* Penalties are accepted as-is and forwarded to consumers (formula
  builder, REML loop, torch trainers). Composite penalties via
  :class:`gamfit._composite_penalty.CompositePenalty` work transparently.
"""

from __future__ import annotations

from typing import Any

from ._protocol import BasisDescriptor, ManifoldDescriptor, PenaltyDescriptor


def _basis_input_dim(basis: BasisDescriptor) -> int | None:
    dim = getattr(basis, "input_dim", None)
    if dim is None:
        return None
    return int(dim)


class Smooth(BasisDescriptor):
    """A composable smooth term tying latent / basis / penalty together.

    Parameters
    ----------
    latent : :class:`ManifoldDescriptor`
        The intrinsic geometry of the input. Determines the expected input
        dimensionality and the natural retraction for inner-loop updates.
    basis : :class:`BasisDescriptor`
        The evaluator producing ``Phi(t)``. Must accept inputs of
        ``latent.dimension`` columns.
    penalty : :class:`PenaltyDescriptor`, optional
        One or a composite of analytic penalties. May be ``None`` for an
        unpenalized smooth.
    name : str, optional
        Diagnostic / formula name.

    Examples
    --------
    >>> import gamfit
    >>> sm = gamfit.Smooth(
    ...     latent=gamfit.Circle(),
    ...     basis=gamfit.Fourier(harmonics=3),
    ...     penalty=gamfit.ARDPenalty(0.1),
    ... )
    >>> phi = sm.evaluate(torch.linspace(0.0, 6.28, 64))   # (64, 7)
    """

    def __init__(
        self,
        *,
        latent: ManifoldDescriptor | type,
        basis: BasisDescriptor | type | None = None,
        penalty: PenaltyDescriptor | None = None,
        name: str | None = None,
    ) -> None:
        # Magic-by-default: accept the class form ``latent=Circle`` (auto-
        # instantiate), and auto-pick a sensible basis + penalty when omitted.
        if isinstance(latent, type) and issubclass(latent, ManifoldDescriptor):
            latent = latent()
        if not isinstance(latent, ManifoldDescriptor):
            raise TypeError(
                f"Smooth.latent must be a ManifoldDescriptor (class or instance), "
                f"got {type(latent).__name__}"
            )

        if basis is None:
            basis = _default_basis_for(latent)
        elif isinstance(basis, type) and issubclass(basis, BasisDescriptor):
            basis = basis()
        if not isinstance(basis, BasisDescriptor):
            raise TypeError(
                f"Smooth.basis must be a BasisDescriptor (class or instance), "
                f"got {type(basis).__name__}"
            )

        _check_manifold_basis_compatibility(latent, basis)

        if penalty is not None and not isinstance(penalty, PenaltyDescriptor):
            raise TypeError(
                f"Smooth.penalty must be a PenaltyDescriptor or None, "
                f"got {type(penalty).__name__}"
            )

        # Compatibility check already done above via
        # :func:`_check_manifold_basis_compatibility`, which delegates to the
        # Rust validator with a soft ``input_dim`` fallback. Touching
        # ``latent.dimension`` here would force the Rust manifold-dim oracle
        # to be present in every wheel.

        self.latent = latent
        self.basis = basis
        self.penalty = penalty
        self.name = name

    @property
    def dimension(self) -> int:
        return self.latent.dimension

    @property
    def output_dim(self) -> int | None:
        return getattr(self.basis, "output_dim", None)

    def evaluate(self, t: Any) -> Any:
        return self.basis.evaluate(t)

    def jacobian(self, t: Any) -> Any:
        return self.basis.jacobian(t)

    def hessian(self, t: Any) -> Any:
        return self.basis.hessian(t)

    def penalty_value(self, t: Any) -> Any:
        """Convenience: evaluate the smooth's penalty at ``t`` (or ``0`` if
        none)."""
        if self.penalty is None:
            torch = _require_torch_local()
            if hasattr(t, "dtype"):
                return torch.zeros((), dtype=t.dtype, device=t.device)
            return torch.zeros(())
        return self.penalty.value(t)

    def __add__(self, other: "Smooth") -> "SmoothSum":
        if not isinstance(other, (Smooth, SmoothSum)):
            return NotImplemented
        return SmoothSum(self, other)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the smooth's structural fields into a plain dict."""
        latent_payload = (
            self.latent.to_json() if hasattr(self.latent, "to_json") else {"kind": type(self.latent).__name__.lower()}
        )
        basis_payload = (
            self.basis.to_dict() if hasattr(self.basis, "to_dict") else {"kind": type(self.basis).__name__}
        )
        return {
            "kind": "composed_smooth",
            "name": self.name,
            "latent": latent_payload,
            "basis": basis_payload,
            "penalty": None if self.penalty is None else _penalty_to_dict(self.penalty),
        }

    def to_rust_descriptor(self) -> dict[str, Any]:
        """Full Rust-engine descriptor for the four-tuple composition.

        Equivalent to :meth:`to_dict`, but routed through each component's own
        ``to_rust_descriptor()`` first so composite penalties round-trip every
        child's payload verbatim. The exact key names match what the formula
        builder consumes on the Rust side.
        """
        latent_payload = (
            self.latent.to_json() if hasattr(self.latent, "to_json")
            else {"kind": type(self.latent).__name__.lower()}
        )
        basis_payload: dict[str, Any]
        if hasattr(self.basis, "to_dict"):
            basis_payload = self.basis.to_dict()
        else:
            basis_payload = {"kind": type(self.basis).__name__}
        if self.penalty is None:
            penalty_payload: dict[str, Any] | None = None
        elif hasattr(self.penalty, "to_rust_descriptor"):
            penalty_payload = self.penalty.to_rust_descriptor()
        else:
            penalty_payload = _penalty_to_dict(self.penalty)
        return {
            "kind": "composed_smooth",
            "name": self.name,
            "latent": latent_payload,
            "basis": basis_payload,
            "penalty": penalty_payload,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Smooth":
        from ._manifold import Circle, Sphere, Torus, Euclidean, CylinderManifold
        from ._basis_descriptors import PeriodicHarmonic

        latent_p = payload["latent"]
        basis_p = payload["basis"]
        lkind = str(latent_p.get("kind", "")).lower()
        if lkind == "circle":
            latent: ManifoldDescriptor = Circle()
        elif lkind == "sphere":
            latent = Sphere(intrinsic_dim=int(latent_p.get("intrinsic_dim", 2)))
        elif lkind == "torus":
            latent = Torus(dim=int(latent_p.get("dim", 2)))
        elif lkind == "euclidean":
            latent = Euclidean(dim=int(latent_p.get("dim", 1)))
        elif lkind == "cylinder":
            latent = CylinderManifold(open_dim=int(latent_p.get("open_dim", 1)))
        else:
            raise ValueError(f"Smooth.from_dict: unknown latent kind {lkind!r}")

        bkind = str(basis_p.get("kind", "")).lower()
        if bkind == "periodic_harmonic":
            basis: BasisDescriptor = PeriodicHarmonic(harmonics=int(basis_p.get("harmonics", 3)))
        else:
            raise ValueError(f"Smooth.from_dict: unknown basis kind {bkind!r}")

        # Penalty deserialization is not yet implemented for the full
        # descriptor zoo; rather than silently discard a penalty that the
        # caller serialized, refuse to round-trip and tell them explicitly.
        if payload.get("penalty") is not None:
            raise NotImplementedError(
                "Smooth.from_dict cannot reconstruct penalty descriptors yet; "
                "round-tripping a penalised Smooth would silently drop it. "
                "Reconstruct the penalty manually and pass it to Smooth(...)."
            )

        return cls(latent=latent, basis=basis, penalty=None, name=payload.get("name"))

    def __repr__(self) -> str:
        parts = [f"latent={self.latent!r}", f"basis={self.basis!r}"]
        if self.penalty is not None:
            parts.append(f"penalty={self.penalty!r}")
        if self.name is not None:
            parts.append(f"name={self.name!r}")
        return f"Smooth({', '.join(parts)})"


def _default_basis_for(latent: ManifoldDescriptor) -> BasisDescriptor:
    """Magic-by-default basis selection from a manifold.

    The rule is: pick the canonical basis for the latent's topology. Today
    that's a periodic harmonic basis for circle/1-D-torus. For other latent
    topologies the user must pass ``basis=...`` explicitly.
    """
    from ._manifold import Circle, Torus
    from ._basis_descriptors import PeriodicHarmonic
    if isinstance(latent, Circle):
        return PeriodicHarmonic(harmonics=3)
    if isinstance(latent, Torus) and latent.dimension == 1:
        return PeriodicHarmonic(harmonics=3)
    raise ValueError(
        f"Smooth: no default basis is registered for latent="
        f"{type(latent).__name__}(dim={latent.dimension}); pass `basis=...` explicitly."
    )


def _check_manifold_basis_compatibility(
    latent: ManifoldDescriptor, basis: BasisDescriptor
) -> None:
    """Eager compatibility check between a manifold and a basis.

    Marshals manifold + basis kind tags to the Rust validator when the local
    extension exposes one; falls back to a soft ``input_dim`` match otherwise.
    All policy lives in Rust (see ``crates/gam-pyffi`` ``validate_smooth_composition``).
    """
    latent_kind = None
    if hasattr(latent, "to_json"):
        latent_kind = str(latent.to_json().get("kind", ""))
    basis_kind = None
    if hasattr(basis, "to_dict"):
        basis_kind = str(basis.to_dict().get("kind", ""))
    if latent_kind and basis_kind:
        from ._binding import rust_module
        validator = getattr(rust_module(), "validate_smooth_composition", None)
        if validator is not None:
            validator(latent_kind, basis_kind)
            return
    expected = _basis_input_dim(basis)
    if expected is None:
        return
    try:
        latent_dim = int(latent.dimension)
    except AttributeError:
        # Rust dimension oracle not yet exposed by the local extension; the
        # Rust core will reject incompatible specs again at fit time.
        return
    if expected != latent_dim:
        raise ValueError(
            f"Smooth: latent={type(latent).__name__}(dim={latent_dim}) is "
            f"incompatible with basis={type(basis).__name__}(input_dim={expected}). "
            "Choose a basis whose input dimension matches the latent manifold."
        )


def _penalty_to_dict(penalty: PenaltyDescriptor) -> dict[str, Any]:
    from ._composite_penalty import CompositePenalty
    if isinstance(penalty, CompositePenalty):
        return {
            "kind": "composite",
            "parts": [_penalty_to_dict(p) for p in penalty.parts],
        }
    if hasattr(penalty, "to_dict"):
        return penalty.to_dict()  # type: ignore[no-any-return]
    return {"kind": type(penalty).__name__, "repr": repr(penalty)}


def _require_torch_local() -> Any:
    from ._protocol import _require_torch
    return _require_torch()


class SmoothSum:
    """Sum of two or more :class:`Smooth` terms.

    Lightweight placeholder; the formula builder consumes these by flattening
    into its term list. Exposes ``__add__`` for chaining.
    """

    def __init__(self, *parts: Any) -> None:
        flat: list[Any] = []
        for p in parts:
            if isinstance(p, SmoothSum):
                flat.extend(p.parts)
            else:
                flat.append(p)
        self.parts: list[Any] = flat

    def __add__(self, other: Any) -> "SmoothSum":
        return SmoothSum(self, other)

    def __iter__(self) -> Any:
        return iter(self.parts)

    def __repr__(self) -> str:
        return " + ".join(repr(p) for p in self.parts)


__all__ = ["Smooth", "SmoothSum"]
