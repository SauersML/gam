"""Regression tests for invalid values crossing the public Python boundary."""

from __future__ import annotations

import pytest

import gamfit


@pytest.mark.parametrize(
    ("constructor", "keyword", "value", "message"),
    [
        (gamfit.EuclideanManifold, "dim", 0, "EuclideanManifold.dim must be a positive integer"),
        (gamfit.SphereManifold, "intrinsic_dim", -1, "SphereManifold.intrinsic_dim must be a positive integer"),
        (gamfit.TorusManifold, "dim", 0, "TorusManifold.dim must be a positive integer"),
    ],
)
def test_manifold_constructors_reject_nonpositive_dimensions_with_gam_error(
    constructor: object, keyword: str, value: int, message: str
) -> None:
    """Bad dimensions must raise, rather than reaching indexing-heavy geometry code."""
    with pytest.raises(gamfit.GamError, match=message):
        constructor(**{keyword: value})


@pytest.mark.parametrize(
    ("manifold", "attribute"),
    [
        (gamfit.EuclideanManifold(2), "dim"),
        (gamfit.SphereManifold(2), "intrinsic_dim"),
        (gamfit.TorusManifold(2), "dim"),
    ],
)
def test_manifold_dimension_setters_preserve_valid_boundary(
    manifold: object, attribute: str
) -> None:
    """Mutation cannot bypass the same constructor validation."""
    with pytest.raises(gamfit.GamError, match="must be a positive integer .*got 0"):
        setattr(manifold, attribute, 0)
    assert getattr(manifold, attribute) == 2
