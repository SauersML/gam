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
        latent: ManifoldDescriptor,
        basis: BasisDescriptor,
        penalty: PenaltyDescriptor | None = None,
        name: str | None = None,
    ) -> None:
        if not isinstance(latent, ManifoldDescriptor):
            raise TypeError(
                f"Smooth.latent must be a ManifoldDescriptor, got {type(latent).__name__}"
            )
        if not isinstance(basis, BasisDescriptor):
            raise TypeError(
                f"Smooth.basis must be a BasisDescriptor, got {type(basis).__name__}"
            )
        if penalty is not None and not isinstance(penalty, PenaltyDescriptor):
            raise TypeError(
                f"Smooth.penalty must be a PenaltyDescriptor or None, "
                f"got {type(penalty).__name__}"
            )

        expected = _basis_input_dim(basis)
        if expected is not None and expected != latent.dimension:
            raise ValueError(
                f"Smooth: latent.dimension ({latent.dimension}) does not match "
                f"basis input_dim ({expected}). For {type(latent).__name__} use "
                f"a basis that accepts {latent.dimension}-D inputs."
            )

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

        return cls(latent=latent, basis=basis, penalty=None, name=payload.get("name"))

    def __repr__(self) -> str:
        parts = [f"latent={self.latent!r}", f"basis={self.basis!r}"]
        if self.penalty is not None:
            parts.append(f"penalty={self.penalty!r}")
        if self.name is not None:
            parts.append(f"name={self.name!r}")
        return f"Smooth({', '.join(parts)})"


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
