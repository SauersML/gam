"""Exact analytic ground-truth objects for the manifold-superposition zoo.

Every sampler returns points in its native one-, two-, or three-dimensional
embedding together with the parameters that generated them.  The defining
equations are checked on every sampled block before any centering, scaling,
random isometry, or additive superposition is allowed to happen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


Sampler = Callable[[np.random.Generator, int], tuple[np.ndarray, np.ndarray]]
GEOMETRY_REVISION = "analytic-zoo-v2-standard-swiss-height"
SWISS_HALF_HEIGHT = 10.5


@dataclass(frozen=True)
class ZooType:
    name: str
    intrinsic_dim: int
    span_dim: int
    atom_basis: str
    sampler: Sampler


def analytic_points(kind: str, parameters: np.ndarray) -> np.ndarray:
    """Evaluate one zoo object's canonical analytic parameterization."""
    parameters = np.asarray(parameters, dtype=float)
    if parameters.ndim != 2:
        raise ValueError(f"{kind} parameters must be a matrix; got {parameters.shape}")
    if kind == "segment":
        return parameters[:, :1].copy()
    if kind == "circle":
        angle = parameters[:, 0]
        return np.column_stack((np.cos(angle), np.sin(angle)))
    if kind == "disk":
        radius, angle = parameters.T
        return np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
    if kind == "sphere":
        polar, azimuth = parameters.T
        return np.column_stack(
            (
                np.sin(polar) * np.cos(azimuth),
                np.sin(polar) * np.sin(azimuth),
                np.cos(polar),
            )
        )
    if kind == "torus":
        azimuth, tube_angle = parameters.T
        ring_radius = 1.0 + 0.4 * np.cos(tube_angle)
        return np.column_stack(
            (
                ring_radius * np.cos(azimuth),
                ring_radius * np.sin(azimuth),
                0.4 * np.sin(tube_angle),
            )
        )
    if kind == "mobius":
        angle, width = parameters.T
        radial = 1.0 + width * np.cos(0.5 * angle)
        return np.column_stack(
            (
                radial * np.cos(angle),
                radial * np.sin(angle),
                width * np.sin(0.5 * angle),
            )
        )
    if kind == "swiss":
        angle, height = parameters.T
        return np.column_stack((angle * np.cos(angle), height, angle * np.sin(angle)))
    if kind == "helix":
        angle = parameters[:, 0]
        return np.column_stack((np.cos(angle), np.sin(angle), 0.25 * angle))
    raise ValueError(f"unknown analytic zoo kind {kind!r}")


def _segment(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    t = rng.uniform(-1.0, 1.0, size=n)
    parameters = t[:, None]
    return analytic_points("segment", parameters), parameters


def _circle(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    angle = rng.uniform(0.0, 2.0 * np.pi, size=n)
    parameters = angle[:, None]
    return analytic_points("circle", parameters), parameters


def _disk(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    angle = rng.uniform(0.0, 2.0 * np.pi, size=n)
    radius = np.sqrt(rng.uniform(0.0, 1.0, size=n))
    parameters = np.column_stack((radius, angle))
    return analytic_points("disk", parameters), parameters


def _sphere(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    azimuth = rng.uniform(0.0, 2.0 * np.pi, size=n)
    polar = np.arccos(rng.uniform(-1.0, 1.0, size=n))
    parameters = np.column_stack((polar, azimuth))
    return analytic_points("sphere", parameters), parameters


def _torus(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    azimuth = rng.uniform(0.0, 2.0 * np.pi, size=n)
    tube_angle = rng.uniform(0.0, 2.0 * np.pi, size=n)
    parameters = np.column_stack((azimuth, tube_angle))
    return analytic_points("torus", parameters), parameters


def _mobius(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    angle = rng.uniform(0.0, 2.0 * np.pi, size=n)
    width = rng.uniform(-0.5, 0.5, size=n)
    parameters = np.column_stack((angle, width))
    return analytic_points("mobius", parameters), parameters


def _swiss_roll(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    angle = rng.uniform(1.5 * np.pi, 4.5 * np.pi, size=n)
    height = rng.uniform(-SWISS_HALF_HEIGHT, SWISS_HALF_HEIGHT, size=n)
    parameters = np.column_stack((angle, height))
    return analytic_points("swiss", parameters), parameters


def _helix(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    angle = rng.uniform(0.0, 4.0 * np.pi, size=n)
    parameters = angle[:, None]
    return analytic_points("helix", parameters), parameters


ZOO_ORDER = ("segment", "circle", "disk", "sphere", "torus", "mobius", "swiss", "helix")
ZOO: dict[str, ZooType] = {
    spec.name: spec
    for spec in (
        ZooType("segment", 1, 1, "euclidean", _segment),
        ZooType("circle", 1, 2, "periodic", _circle),
        ZooType("disk", 2, 2, "duchon", _disk),
        ZooType("sphere", 2, 3, "sphere", _sphere),
        ZooType("torus", 2, 3, "torus", _torus),
        ZooType("mobius", 2, 3, "mobius", _mobius),
        ZooType("swiss", 2, 3, "duchon", _swiss_roll),
        ZooType("helix", 1, 3, "duchon", _helix),
    )
}
CURVED_CYCLE = list(ZOO_ORDER[1:])


def declared_atom_spec(kinds: list[str], atoms: int) -> tuple[list[str], list[int]]:
    """Return the exact one-atom-per-factor Rust chart declaration."""
    if atoms != len(kinds):
        raise ValueError(
            "declared topology requires exactly one fitted atom per planted factor; "
            f"got atoms={atoms} and factors={len(kinds)}"
        )
    specs = [ZOO[kind] for kind in kinds]
    return [spec.atom_basis for spec in specs], [spec.intrinsic_dim for spec in specs]


def first_coordinate_hue(kind: str, parameters: np.ndarray) -> np.ndarray:
    """Normalize the first planted coordinate on its exact analytic domain."""
    if parameters.ndim != 2 or parameters.shape[1] != ZOO[kind].intrinsic_dim:
        raise ValueError(f"invalid {kind} parameter block {parameters.shape}")
    value = parameters[:, 0]
    if kind == "segment":
        hue = 0.5 * (value + 1.0)
    elif kind in {"circle", "torus", "mobius"}:
        hue = value / (2.0 * np.pi)
    elif kind == "disk":
        hue = value
    elif kind == "sphere":
        hue = value / np.pi
    elif kind == "swiss":
        hue = (value - 1.5 * np.pi) / (3.0 * np.pi)
    elif kind == "helix":
        hue = value / (4.0 * np.pi)
    else:
        raise ValueError(f"unknown analytic zoo kind {kind!r}")
    return np.clip(hue, 0.0, 1.0)


def validate_analytic_sample(
    kind: str,
    points: np.ndarray,
    parameters: np.ndarray,
    *,
    atol: float = 2.0e-12,
) -> None:
    """Fail loudly unless a sampled cloud satisfies its defining equations."""
    spec = ZOO[kind]
    n = points.shape[0]
    if points.shape != (n, spec.span_dim):
        raise ValueError(f"{kind} points have shape {points.shape}, expected {(n, spec.span_dim)}")
    if parameters.shape != (n, spec.intrinsic_dim):
        raise ValueError(
            f"{kind} parameters have shape {parameters.shape}, "
            f"expected {(n, spec.intrinsic_dim)}"
        )
    if not np.isfinite(points).all() or not np.isfinite(parameters).all():
        raise ValueError(f"{kind} sampler returned non-finite values")

    tau = 2.0 * np.pi
    if kind == "segment":
        domains = ((-1.0, 1.0),)
    elif kind == "circle":
        domains = ((0.0, tau),)
    elif kind == "disk":
        domains = ((0.0, 1.0), (0.0, tau))
    elif kind == "sphere":
        domains = ((0.0, np.pi), (0.0, tau))
    elif kind == "torus":
        domains = ((0.0, tau), (0.0, tau))
    elif kind == "mobius":
        domains = ((0.0, tau), (-0.5, 0.5))
    elif kind == "swiss":
        domains = (
            (1.5 * np.pi, 4.5 * np.pi),
            (-SWISS_HALF_HEIGHT, SWISS_HALF_HEIGHT),
        )
    elif kind == "helix":
        domains = ((0.0, 4.0 * np.pi),)
    else:
        raise ValueError(f"unknown analytic zoo kind {kind!r}")
    for axis, (lower, upper) in enumerate(domains):
        values = parameters[:, axis]
        if np.any((values < lower - atol) | (values > upper + atol)):
            raise ValueError(f"{kind} parameter axis {axis} escaped [{lower}, {upper}]")

    expected = analytic_points(kind, parameters)

    error = float(np.max(np.abs(points - expected), initial=0.0))
    if error > atol:
        raise ValueError(f"{kind} defining-equation error {error:.3e} exceeds {atol:.3e}")
