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


@dataclass(frozen=True)
class ZooType:
    name: str
    intrinsic_dim: int
    span_dim: int
    sampler: Sampler


def _segment(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    t = rng.uniform(-1.0, 1.0, size=n)
    return t[:, None], t[:, None]


def _circle(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    angle = rng.uniform(0.0, 2.0 * np.pi, size=n)
    points = np.column_stack((np.cos(angle), np.sin(angle)))
    return points, angle[:, None]


def _disk(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    angle = rng.uniform(0.0, 2.0 * np.pi, size=n)
    radius = np.sqrt(rng.uniform(0.0, 1.0, size=n))
    points = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
    return points, np.column_stack((radius, angle))


def _sphere(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    azimuth = rng.uniform(0.0, 2.0 * np.pi, size=n)
    polar = np.arccos(rng.uniform(-1.0, 1.0, size=n))
    points = np.column_stack(
        (
            np.sin(polar) * np.cos(azimuth),
            np.sin(polar) * np.sin(azimuth),
            np.cos(polar),
        )
    )
    return points, np.column_stack((polar, azimuth))


def _torus(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    major_radius = 1.0
    minor_radius = 0.4
    azimuth = rng.uniform(0.0, 2.0 * np.pi, size=n)
    tube_angle = rng.uniform(0.0, 2.0 * np.pi, size=n)
    ring_radius = major_radius + minor_radius * np.cos(tube_angle)
    points = np.column_stack(
        (
            ring_radius * np.cos(azimuth),
            ring_radius * np.sin(azimuth),
            minor_radius * np.sin(tube_angle),
        )
    )
    return points, np.column_stack((azimuth, tube_angle))


def _mobius(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    angle = rng.uniform(0.0, 2.0 * np.pi, size=n)
    width = rng.uniform(-0.5, 0.5, size=n)
    radial = 1.0 + width * np.cos(0.5 * angle)
    points = np.column_stack(
        (
            radial * np.cos(angle),
            radial * np.sin(angle),
            width * np.sin(0.5 * angle),
        )
    )
    return points, np.column_stack((angle, width))


def _swiss_roll(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    angle = rng.uniform(1.5 * np.pi, 4.5 * np.pi, size=n)
    height = rng.uniform(-1.0, 1.0, size=n)
    points = np.column_stack((angle * np.cos(angle), height, angle * np.sin(angle)))
    return points, np.column_stack((angle, height))


def _helix(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    angle = rng.uniform(0.0, 4.0 * np.pi, size=n)
    points = np.column_stack((np.cos(angle), np.sin(angle), 0.25 * angle))
    return points, angle[:, None]


ZOO_ORDER = ("segment", "circle", "disk", "sphere", "torus", "mobius", "swiss", "helix")
ZOO: dict[str, ZooType] = {
    spec.name: spec
    for spec in (
        ZooType("segment", 1, 1, _segment),
        ZooType("circle", 1, 2, _circle),
        ZooType("disk", 2, 2, _disk),
        ZooType("sphere", 2, 3, _sphere),
        ZooType("torus", 2, 3, _torus),
        ZooType("mobius", 2, 3, _mobius),
        ZooType("swiss", 2, 3, _swiss_roll),
        ZooType("helix", 1, 3, _helix),
    )
}
CURVED_CYCLE = list(ZOO_ORDER[1:])


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

    if kind == "segment":
        expected = parameters[:, :1]
    elif kind == "circle":
        angle = parameters[:, 0]
        expected = np.column_stack((np.cos(angle), np.sin(angle)))
    elif kind == "disk":
        radius, angle = parameters.T
        if np.any((radius < 0.0) | (radius > 1.0)):
            raise ValueError("disk radius escaped [0, 1]")
        expected = np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
    elif kind == "sphere":
        polar, azimuth = parameters.T
        expected = np.column_stack(
            (
                np.sin(polar) * np.cos(azimuth),
                np.sin(polar) * np.sin(azimuth),
                np.cos(polar),
            )
        )
    elif kind == "torus":
        azimuth, tube_angle = parameters.T
        ring_radius = 1.0 + 0.4 * np.cos(tube_angle)
        expected = np.column_stack(
            (
                ring_radius * np.cos(azimuth),
                ring_radius * np.sin(azimuth),
                0.4 * np.sin(tube_angle),
            )
        )
    elif kind == "mobius":
        angle, width = parameters.T
        radial = 1.0 + width * np.cos(0.5 * angle)
        expected = np.column_stack(
            (
                radial * np.cos(angle),
                radial * np.sin(angle),
                width * np.sin(0.5 * angle),
            )
        )
    elif kind == "swiss":
        angle, height = parameters.T
        expected = np.column_stack((angle * np.cos(angle), height, angle * np.sin(angle)))
    elif kind == "helix":
        angle = parameters[:, 0]
        expected = np.column_stack((np.cos(angle), np.sin(angle), 0.25 * angle))
    else:
        raise ValueError(f"unknown analytic zoo kind {kind!r}")

    error = float(np.max(np.abs(points - expected), initial=0.0))
    if error > atol:
        raise ValueError(f"{kind} defining-equation error {error:.3e} exceeds {atol:.3e}")

