"""Riemannian manifold descriptors with callable ``exp / log / metric / geodesic``.

The flat / spherical / toroidal descriptors here (Euclidean, Circle, Sphere,
Torus, CylinderManifold) are the intrinsic-geometry primitives consumed by the
new compositional ``Smooth(latent=..., basis=..., penalty=...)`` API; see
:mod:`gamfit._manifold` for the implementations and
:mod:`gamfit._protocol.ManifoldDescriptor` for the protocol.

This module also surfaces the *curved matrix / hyperbolic* manifolds —
Grassmann, Stiefel, SPD, and Poincaré (hyperbolic) — that already exist in the
Rust geometry library, so they sit in the documented ``gamfit.manifolds``
namespace next to the others instead of only in the torch-only / ``gamfit.geometry``
corners.

GEOMETRY PRIMITIVES, NOT (YET) FITTABLE SMOOTHS
-----------------------------------------------
:class:`Grassmann`, :class:`Stiefel`, :class:`Spd`, and :class:`Poincare`
expose *only* the intrinsic-geometry primitives (``exp / log / metric /
distance / dimension``). They are deliberately **not** yet fittable latent
smooths: wiring these manifolds into the penalized
``Smooth(latent=..., basis=..., penalty=...)`` solver needs new Rust solver
work and is out of scope. Use them for geodesic computations, tangent-space
maps, and distances — not as drop-in latent smooths.

They live in this submodule (rather than the top-level ``gamfit`` namespace)
to avoid colliding with the existing basis-spec / topology-factory names
``gamfit.Circle / Cylinder / Sphere / Torus``, which are basis descriptors
in the new callable-descriptor world.

>>> from gamfit.manifolds import Circle, Sphere, Poincare, Spd
>>> Circle().dimension
1
>>> Sphere(intrinsic_dim=2).dimension
2
>>> Spd(n=3).dimension
6
>>> Poincare(dim=2).dimension
2
"""

from __future__ import annotations

from ._manifold import (
    Circle,
    CylinderManifold,
    Euclidean,
    Grassmann,
    Poincare,
    Spd,
    Sphere,
    Stiefel,
    Torus,
)

__all__ = [
    "Circle",
    "CylinderManifold",
    "Euclidean",
    "Grassmann",
    "Poincare",
    "Spd",
    "Sphere",
    "Stiefel",
    "Torus",
]
