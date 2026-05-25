"""Riemannian manifold descriptors with callable ``exp / log / metric / geodesic``.

These are the intrinsic-geometry primitives consumed by the new compositional
``Smooth(latent=..., basis=..., penalty=...)`` API; see :mod:`gamfit._manifold`
for the implementations and :mod:`gamfit._protocol.ManifoldDescriptor` for
the protocol.

They live in this submodule (rather than the top-level ``gamfit`` namespace)
to avoid colliding with the existing basis-spec / topology-factory names
``gamfit.Circle / Cylinder / Sphere / Torus``, which are basis descriptors
in the new callable-descriptor world.

>>> from gamfit.manifolds import Circle, Sphere
>>> Circle().dimension
1
>>> Sphere(intrinsic_dim=2).dimension
2
"""

from __future__ import annotations

from ._manifold import (
    Circle,
    CylinderManifold,
    Euclidean,
    Sphere,
    Torus,
)

__all__ = [
    "Circle",
    "CylinderManifold",
    "Euclidean",
    "Sphere",
    "Torus",
]
