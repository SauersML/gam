"""Composite penalty: sum of two or more :class:`PenaltyDescriptor` instances.

Composition rule (per the descriptor protocol):

* ``value``: sum of children's ``value``.
* ``value_grad``: sums of value and gradient.
* ``hvp``: sums of hvps.
* ``hessian_diag``: sums of diagonals.
* No extra weight on the composite — each child keeps its own weight.
"""

from __future__ import annotations

from typing import Any, Iterable, Protocol, runtime_checkable

from ._protocol import PenaltyDescriptor


@runtime_checkable
class _RustSerializablePenalty(Protocol):
    """A penalty that can marshal itself into the Rust registry schema."""

    def to_rust_descriptor(self) -> dict[str, Any]: ...


class CompositePenalty(PenaltyDescriptor):
    """Sum-composite of :class:`PenaltyDescriptor` instances.

    Equivalent to building a single multi-penalty registry, but composable
    on the Python side so users can write ``ARDPenalty(...) + OrderedBetaBernoulliPenalty(...)``
    and pass the result anywhere a penalty descriptor is expected.
    """

    def __init__(self, *parts: PenaltyDescriptor) -> None:
        flat: list[PenaltyDescriptor] = []
        for part in parts:
            if isinstance(part, CompositePenalty):
                flat.extend(part.parts)
            elif isinstance(part, PenaltyDescriptor):
                flat.append(part)
            else:
                raise TypeError(
                    f"CompositePenalty parts must be PenaltyDescriptor, got {type(part).__name__}"
                )
        if not flat:
            raise ValueError("CompositePenalty requires at least one part")
        self.parts: list[PenaltyDescriptor] = flat

    def __iter__(self) -> Iterable[PenaltyDescriptor]:
        return iter(self.parts)

    def __len__(self) -> int:
        return len(self.parts)

    def __repr__(self) -> str:
        inner = " + ".join(repr(p) for p in self.parts)
        return f"CompositePenalty({inner})"

    def value(self, t: Any) -> Any:
        total = None
        for part in self.parts:
            v = part.value(t)
            total = v if total is None else total + v
        return total

    def value_grad(self, t: Any) -> tuple[Any, Any]:
        total_v = None
        total_g = None
        for part in self.parts:
            v, g = part.value_grad(t)
            total_v = v if total_v is None else total_v + v
            total_g = g if total_g is None else total_g + g
        return total_v, total_g

    def hvp(self, t: Any, v: Any) -> Any:
        total = None
        for part in self.parts:
            contrib = part.hvp(t, v)
            total = contrib if total is None else total + contrib
        return total

    def hessian_diag(self, t: Any) -> Any:
        total = None
        for part in self.parts:
            contrib = part.hessian_diag(t)
            total = contrib if total is None else total + contrib
        return total

    def to_rust_descriptor(self) -> dict[str, Any]:
        """Sum-composite descriptor that round-trips each child verbatim.

        Children stay addressable for diagnostics: ``len(composite)`` reports
        the count, iteration yields the originals, and the dict carries each
        child's own ``to_rust_descriptor()`` under ``"children"``.
        """
        children: list[dict[str, Any]] = []
        for part in self.parts:
            if not isinstance(part, _RustSerializablePenalty):
                raise TypeError(
                    f"CompositePenalty part {type(part).__name__} does not "
                    "expose to_rust_descriptor()"
                )
            children.append(part.to_rust_descriptor())
        return {
            "kind": "sum",
            "children": children,
        }


__all__ = ["CompositePenalty"]
