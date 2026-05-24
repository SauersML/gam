from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class EuclideanManifold:
    dim: int

    def to_json(self) -> dict[str, Any]:
        return {"kind": "euclidean", "dim": int(self.dim)}


@dataclass(frozen=True, slots=True)
class CircleManifold:
    def to_json(self) -> dict[str, Any]:
        return {"kind": "circle"}


@dataclass(frozen=True, slots=True)
class SphereManifold:
    intrinsic_dim: int

    def to_json(self) -> dict[str, Any]:
        return {"kind": "sphere", "intrinsic_dim": int(self.intrinsic_dim)}


@dataclass(frozen=True, slots=True)
class TorusManifold:
    dim: int

    def to_json(self) -> dict[str, Any]:
        return {"kind": "torus", "dim": int(self.dim)}


@dataclass(frozen=True, slots=True)
class GrassmannManifold:
    k: int
    n: int

    def to_json(self) -> dict[str, Any]:
        return {"kind": "grassmann", "k": int(self.k), "n": int(self.n)}


@dataclass(frozen=True, slots=True)
class StiefelManifold:
    k: int
    n: int

    def to_json(self) -> dict[str, Any]:
        return {"kind": "stiefel", "k": int(self.k), "n": int(self.n)}


@dataclass(frozen=True, slots=True)
class SpdManifold:
    n: int

    def to_json(self) -> dict[str, Any]:
        return {"kind": "spd", "n": int(self.n)}


@dataclass(frozen=True, slots=True)
class ProductManifold:
    parts: tuple[Any, ...]

    def __init__(self, *parts: Any) -> None:
        object.__setattr__(self, "parts", tuple(parts))

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": "product",
            "parts": [
                part.to_json() if hasattr(part, "to_json") else part
                for part in self.parts
            ],
        }


__all__ = [
    "CircleManifold",
    "EuclideanManifold",
    "GrassmannManifold",
    "ProductManifold",
    "SpdManifold",
    "SphereManifold",
    "StiefelManifold",
    "TorusManifold",
]
