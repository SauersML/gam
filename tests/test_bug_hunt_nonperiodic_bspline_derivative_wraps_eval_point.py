"""Regression for gam#1348 — non-periodic B-spline derivative must not wrap.

`gamfit.bspline_basis_derivative(..., periodic=False)` on a UNIFORM OPEN knot
vector (e.g. ``np.linspace(a, b, m)`` with no repeated boundary knots) used to
run the evaluation point through a *periodic* wrap in the two boundary spans, so
the analytic derivative was supported on the wrong basis columns and disagreed
with a finite difference of the value basis by ~1.7. The wrap was geometry-
inferred (uniform open knots looked "periodic") with no signal from the caller;
only a genuinely cyclic basis should wrap, and the cyclic evaluator pre-wraps
its own input. These tests pin the non-periodic derivative to the non-periodic
value basis everywhere, including the boundary spans.
"""

from __future__ import annotations

import importlib
import typing

import numpy as np

pytest = typing.cast(typing.Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import gamfit


def test_boundary_span_derivative_supported_on_value_columns() -> None:
    """At a point in the first knot span the order-1 derivative must be
    supported on the same columns as the value (no forward periodic wrap)."""
    knots = np.linspace(-2.0, 3.0, 14)  # uniform, OPEN
    degree = 3
    # A point inside the first knot span [knots[0], knots[1]).
    t = np.array([knots[1] - 0.25 * (knots[1] - knots[0])])

    value = gamfit.bspline_basis(t, knots, degree=degree, periodic=False)
    deriv = gamfit.bspline_basis_derivative(
        t, knots, degree=degree, order=1, periodic=False
    )

    value_cols = np.nonzero(np.abs(value[0]) > 1e-9)[0]
    deriv_cols = np.nonzero(np.abs(deriv[0]) > 1e-9)[0]

    # The derivative's support must be a subset of the value's support: a basis
    # function's derivative lives exactly where the basis function does.
    assert set(deriv_cols.tolist()).issubset(set(value_cols.tolist())), (
        f"derivative wrapped: value cols {value_cols.tolist()} but "
        f"derivative cols {deriv_cols.tolist()}"
    )


def test_full_range_first_derivative_matches_central_difference() -> None:
    """Across the full open domain (including both boundary spans) the analytic
    order-1 derivative equals the central difference of the value basis."""
    knots = np.linspace(-2.0, 3.0, 14)
    degree = 3
    tt = np.linspace(knots[0], knots[-1], 121)
    h = 1e-6
    fd = (
        gamfit.bspline_basis(tt + h, knots, degree=degree, periodic=False)
        - gamfit.bspline_basis(tt - h, knots, degree=degree, periodic=False)
    ) / (2 * h)
    d1 = gamfit.bspline_basis_derivative(
        tt, knots, degree=degree, order=1, periodic=False
    )
    assert np.max(np.abs(d1 - fd)) < 1e-5


def test_full_range_second_derivative_matches_central_difference() -> None:
    """Order-2 wraps the same way as order-1 if the bug is present; it must
    match a central difference of the order-1 derivative across the full range."""
    knots = np.linspace(-2.0, 3.0, 14)
    degree = 3
    tt = np.linspace(knots[0], knots[-1], 121)
    h = 1e-5
    fd = (
        gamfit.bspline_basis_derivative(
            tt + h, knots, degree=degree, order=1, periodic=False
        )
        - gamfit.bspline_basis_derivative(
            tt - h, knots, degree=degree, order=1, periodic=False
        )
    ) / (2 * h)
    d2 = gamfit.bspline_basis_derivative(
        tt, knots, degree=degree, order=2, periodic=False
    )
    assert np.max(np.abs(d2 - fd)) < 1e-4


def test_derivative_is_not_degenerately_zero() -> None:
    """Guard against a degenerate 'fix' that just zeros the boundary spans:
    the derivative must actually be non-trivial in the boundary spans."""
    knots = np.linspace(-2.0, 3.0, 14)
    degree = 3
    # Points inside the two boundary spans.
    t = np.array(
        [
            knots[1] - 0.25 * (knots[1] - knots[0]),  # first span
            knots[-2] + 0.25 * (knots[-1] - knots[-2]),  # last span
        ]
    )
    deriv = gamfit.bspline_basis_derivative(
        t, knots, degree=degree, order=1, periodic=False
    )
    assert np.max(np.abs(deriv)) > 1e-6
