"""Public-API regression for issue #1348 — non-periodic open-knot B-spline derivative.

``gamfit.bspline_basis_derivative(..., periodic=False)`` on a *uniform, open*
knot vector (``np.linspace(a, b, m)`` with no repeated boundary knots) used to
evaluate the analytic derivative at a *wrapped* point in the two boundary spans,
so the derivative was supported on disjoint columns from the value and disagreed
with the central difference of the value basis by O(1). The differentiable
``gamfit.torch`` B-spline basis backprops into input coordinates through exactly
this kernel, so the bug silently corrupted input-coordinate gradients near the
data boundary.

The open-knot value basis is held constant outside the active interval
``[t[degree], t[num_basis]]``, so the correct derivative is ``B'(x)`` inside and
identically zero outside. These tests pin that through the public Python API
from several angles (support, full-range finite difference, exterior zero).
"""

from __future__ import annotations

import importlib
import typing

import numpy as np

pytest = typing.cast(typing.Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import gamfit


def _open_knots(a: float, b: float, m: int) -> np.ndarray:
    return np.linspace(a, b, m)


def test_boundary_span_derivative_support_is_subset_of_value() -> None:
    """Original RED repro: at a first-span point the derivative support must be a
    subset of the value support, not a wrapped set on the far side of the domain."""
    knots = _open_knots(-2.0, 3.0, 14)
    degree = 3
    t = np.array([knots[1] - 0.25 * (knots[1] - knots[0])])
    assert t[0] < knots[degree]

    value = gamfit.bspline_basis(t, knots, degree=degree, periodic=False)
    value_cols = set(np.nonzero(np.abs(value[0]) > 1e-9)[0].tolist())
    assert value_cols

    for order in (1, 2):
        deriv = gamfit.bspline_basis_derivative(
            t, knots, degree=degree, order=order, periodic=False
        )
        deriv_cols = set(np.nonzero(np.abs(deriv[0]) > 1e-9)[0].tolist())
        assert deriv_cols <= value_cols, (
            f"order-{order} derivative columns {deriv_cols} escaped value support "
            f"{value_cols} (wrapped-boundary regression)"
        )


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_full_range_first_derivative_matches_value_central_difference(degree: int) -> None:
    m = degree + 10
    knots = _open_knots(-2.0, 3.0, m)
    num_basis = m - degree - 1
    left, right = knots[degree], knots[num_basis]

    tt = np.linspace(knots[0], knots[-1], 241)
    h = 1e-6
    fd = (
        gamfit.bspline_basis(tt + h, knots, degree=degree, periodic=False)
        - gamfit.bspline_basis(tt - h, knots, degree=degree, periodic=False)
    ) / (2 * h)
    d1 = gamfit.bspline_basis_derivative(tt, knots, degree=degree, order=1, periodic=False)

    # Skip a small neighborhood of every knot (where a low-order derivative is
    # discontinuous) and of the active-interval corners (clamp kink) — a
    # straddling central stencil is not a valid oracle there.
    width = 4 * h
    keep = np.ones(tt.shape, dtype=bool)
    for k in np.r_[knots, left, right]:
        keep &= np.abs(tt - k) > width
    err = np.max(np.abs(d1[keep] - fd[keep]))
    assert err < 1e-6, f"degree={degree}: max|d1 - fd| = {err}"

    strictly_interior = (tt > left + width) & (tt < right - width)
    assert np.max(np.abs(d1[strictly_interior])) > 1e-3


def test_derivative_is_zero_outside_active_interval() -> None:
    knots = _open_knots(0.0, 1.0, 12)
    degree = 3
    num_basis = len(knots) - degree - 1
    left, right = knots[degree], knots[num_basis]

    candidates = np.array([knots[0], knots[1], 0.5 * (knots[0] + left), right + 0.01, knots[-1]])
    outside = candidates[(candidates < left - 1e-9) | (candidates > right + 1e-9)]
    assert outside.size > 0

    for order in (1, 2):
        d = gamfit.bspline_basis_derivative(
            outside, knots, degree=degree, order=order, periodic=False
        )
        assert np.allclose(d, 0.0, atol=1e-12), f"order-{order} nonzero outside interval"
