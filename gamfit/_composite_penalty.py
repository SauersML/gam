"""Composite penalty: sum of two or more :class:`PenaltyDescriptor` instances.

Composition rule (per the descriptor protocol):

* ``value``: sum of children's ``value``.
* ``value_grad``: sums of value and gradient.
* ``hvp``: sums of hvps.
* ``hessian_diag``: sums of diagonals.
* No extra weight on the composite — each child keeps its own weight.
"""

from __future__ import annotations

from typing import Any, Iterable

from ._protocol import PenaltyDescriptor


class CompositePenalty(PenaltyDescriptor):
    """Sum-composite of :class:`PenaltyDescriptor` instances.

    Equivalent to building a single multi-penalty registry, but composable
    on the Python side so users can write ``ARDPenalty(...) + IBPPenalty(...)``
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


__all__ = ["CompositePenalty"]
