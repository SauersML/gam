"""``Sphere.basis_size`` is the Rust builder's width rule, not a Python copy.

The descriptor used to restate the rule in Python, and the copy disagreed with
the builder in two places: a harmonic Sphere past the degree-32 cap (among them
``Sphere(kernel="harmonic")`` at its default ``n_centers=50``) reported
``L * (L + 2)`` columns for a basis the builder refuses, and a harmonic Sphere
with explicit ``centers`` read the center count as its degree while evaluation
ignored the centers and built degree ``n_centers``. The width is now read from
``spherical_spline_basis_width``, the rule the builder itself uses.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamfit
from gamfit.errors import BasisError


def _lat_lon(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lat = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=n)))
    lon = rng.uniform(-180.0, 180.0, size=n)
    return lat, lon


@pytest.mark.parametrize(
    ("kernel", "n_centers"),
    [("sobolev", 20), ("pseudo", 20), ("pseudo", 150), ("harmonic", 6)],
)
def test_basis_size_is_the_evaluated_width(kernel: str, n_centers: int) -> None:
    spec = gamfit.smooth.Sphere(n_centers=n_centers, kernel=kernel)
    size = spec.basis_size
    lat, lon = _lat_lon(240, 7)
    design = np.asarray(spec.evaluate(lat, lon, backend="numpy"))
    assert design.shape == (240, size)


def test_basis_size_refuses_a_harmonic_degree_the_builder_refuses() -> None:
    spec = gamfit.smooth.Sphere(kernel="harmonic")  # degree n_centers = 50 > 32
    with pytest.raises(BasisError, match="cap is 32"):
        _ = spec.basis_size
    lat, lon = _lat_lon(60, 8)
    with pytest.raises(BasisError, match="cap is 32"):
        spec.evaluate(lat, lon, backend="numpy")


def test_harmonic_sphere_refuses_explicit_centers() -> None:
    lat, lon = _lat_lon(10, 9)
    spec = gamfit.smooth.Sphere(
        n_centers=4, kernel="harmonic", centers=np.column_stack([lat, lon])
    )
    with pytest.raises(ValueError, match="has no centers"):
        _ = spec.basis_size
    with pytest.raises(ValueError, match="has no centers"):
        spec.evaluate(lat, lon, backend="numpy")
