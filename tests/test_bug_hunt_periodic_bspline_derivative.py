"""Regression tests for issue #520 — periodic B-spline dense derivative.

`gamfit.bspline_basis_derivative(..., periodic=True)` used to raise
``GamError: periodic B-spline first-derivative as a dense (N, K) matrix is no
longer exposed`` for *every* call, even though the exact periodic first
derivative is already computed by the Rust core
(``periodic_bspline_first_derivative_nd``). These tests pin the dense periodic
derivative path to that closed form from several angles so the dead-parameter
regression cannot return.
"""

from __future__ import annotations

import importlib
import typing

import numpy as np

pytest = typing.cast(typing.Any, importlib.import_module("pytest"))
pytest.importorskip("gamfit._rust")

import gamfit


def test_periodic_derivative_matches_central_differences() -> None:
    """The advertised periodic=True derivative must succeed and be correct.

    This is the original RED repro from issue #520.
    """
    knots = np.linspace(0.0, 1.0, 9)  # periodic dense path -> domain [0,1], K=8
    t = np.linspace(0.02, 0.98, 40)

    value = gamfit.bspline_basis(t, knots, degree=3, periodic=True)
    deriv = gamfit.bspline_basis_derivative(t, knots, degree=3, order=1, periodic=True)

    # Same shape as the periodic value basis, and finite.
    assert deriv.shape == value.shape == (40, 8)
    assert np.all(np.isfinite(deriv))

    # Matches central finite differences of the periodic value basis.
    h = 1e-6
    fd = (
        gamfit.bspline_basis(t + h, knots, degree=3, periodic=True)
        - gamfit.bspline_basis(t - h, knots, degree=3, periodic=True)
    ) / (2 * h)
    np.testing.assert_allclose(deriv, fd, atol=1e-7)

    # Derivative of a partition of unity: each row sums to ~0.
    np.testing.assert_allclose(deriv.sum(axis=1), 0.0, atol=1e-10)


def test_periodic_dense_derivative_equals_squeezed_core_jet() -> None:
    """The dense matrix must be exactly the (N, K, 1) core jet, squeezed.

    Different angle from finite differences: ties the public dense path to the
    same closed-form jet the modelling code (`basis_with_jet`,
    `PeriodicSplineCurve`) consumes, to machine precision rather than ~1e-7.
    """
    from gamfit._binding import rust_module

    knots = np.linspace(0.0, 1.0, 9)
    t = np.linspace(0.0, 1.0, 37)
    dense = gamfit.bspline_basis_derivative(t, knots, degree=3, order=1, periodic=True)
    jet = np.asarray(
        rust_module().periodic_bspline_input_location_first_derivative(
            t.reshape(-1, 1), 0.0, 1.0, 3, 8
        )
    )[:, :, 0]
    np.testing.assert_allclose(dense, jet, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_periodic_derivative_finite_difference_sweep(degree: int) -> None:
    """Correctness holds across spline degrees, not just the cubic repro."""
    num_basis = degree + 5
    knots = np.linspace(0.0, 1.0, num_basis + 1)
    t = np.linspace(0.03, 0.97, 50)

    deriv = gamfit.bspline_basis_derivative(t, knots, degree=degree, order=1, periodic=True)
    assert deriv.shape == (t.size, num_basis)

    h = 1e-6
    fd = (
        gamfit.bspline_basis(t + h, knots, degree=degree, periodic=True)
        - gamfit.bspline_basis(t - h, knots, degree=degree, periodic=True)
    ) / (2 * h)
    np.testing.assert_allclose(deriv, fd, atol=1e-6)
    np.testing.assert_allclose(deriv.sum(axis=1), 0.0, atol=1e-10)


def test_periodic_derivative_order_zero_returns_value_basis() -> None:
    """order=0 is the 0th derivative, i.e. the periodic value basis itself."""
    knots = np.linspace(0.0, 1.0, 11)  # K = 10
    t = np.linspace(0.0, 1.0, 33)
    value = gamfit.bspline_basis(t, knots, degree=3, periodic=True)
    deriv0 = gamfit.bspline_basis_derivative(t, knots, degree=3, order=0, periodic=True)
    np.testing.assert_allclose(deriv0, value, rtol=0.0, atol=0.0)
    # Partition of unity carried through the derivative helper.
    np.testing.assert_allclose(deriv0.sum(axis=1), 1.0, atol=1e-12)


def test_periodic_derivative_is_genuinely_periodic_at_seam() -> None:
    """The derivative wraps continuously across the 0<->1 seam.

    A non-cyclic (open) derivative would disagree at the seam; the periodic
    one must give the same row for t and t+period.
    """
    knots = np.linspace(0.0, 1.0, 13)  # K = 12
    t = np.array([0.0, 0.13, 0.5, 0.87])
    d_at_t = gamfit.bspline_basis_derivative(t, knots, degree=3, order=1, periodic=True)
    d_at_t_plus_period = gamfit.bspline_basis_derivative(
        t + 1.0, knots, degree=3, order=1, periodic=True
    )
    np.testing.assert_allclose(d_at_t, d_at_t_plus_period, atol=1e-10)


def test_periodic_second_derivative_rejected_with_precise_message() -> None:
    """Orders >= 2 have no exposed periodic jet: error must say so precisely.

    The old code rejected order=1 too (the bug). The fix must keep rejecting
    order>=2 but with a message that names the real limitation, not the stale
    "no longer exposed" blanket.
    """
    knots = np.linspace(0.0, 1.0, 9)
    t = np.linspace(0.05, 0.95, 10)
    with pytest.raises(Exception) as excinfo:
        gamfit.bspline_basis_derivative(t, knots, degree=3, order=2, periodic=True)
    msg = str(excinfo.value).lower()
    assert "no longer exposed" not in msg
    assert "order" in msg


def test_nonperiodic_derivative_unaffected() -> None:
    """The non-periodic path keeps working (orders 0, 1, 2)."""
    knots = np.linspace(0.0, 1.0, 12)
    t = np.linspace(0.05, 0.95, 30)
    for order in (0, 1, 2):
        d = gamfit.bspline_basis_derivative(t, knots, degree=3, order=order, periodic=False)
        assert np.all(np.isfinite(d))
    # First derivative matches finite differences of the open value basis.
    d1 = gamfit.bspline_basis_derivative(t, knots, degree=3, order=1, periodic=False)
    h = 1e-6
    fd = (
        gamfit.bspline_basis(t + h, knots, degree=3, periodic=False)
        - gamfit.bspline_basis(t - h, knots, degree=3, periodic=False)
    ) / (2 * h)
    np.testing.assert_allclose(d1, fd, atol=1e-6)
