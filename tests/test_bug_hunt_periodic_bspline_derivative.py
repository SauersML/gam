"""Regression for #520: dense periodic B-spline derivatives are public."""

from __future__ import annotations

import numpy as np

import gamfit
from gamfit._binding import rust_module


def test_periodic_bspline_basis_derivative_matches_existing_jet_and_fd() -> None:
    knots = np.linspace(0.0, 2.0 * np.pi, 13)
    t = np.linspace(-1.25, 2.0 * np.pi + 1.25, 97)

    basis = gamfit.bspline_basis(t, knots, degree=3, periodic=True)
    deriv = gamfit.bspline_basis_derivative(t, knots, degree=3, order=1, periodic=True)

    assert deriv.shape == basis.shape
    assert np.all(np.isfinite(deriv))
    np.testing.assert_allclose(deriv.sum(axis=1), 0.0, atol=1e-12)

    h = 1e-6
    plus = gamfit.bspline_basis(t + h, knots, degree=3, periodic=True)
    minus = gamfit.bspline_basis(t - h, knots, degree=3, periodic=True)
    finite_difference = (plus - minus) / (2.0 * h)
    np.testing.assert_allclose(deriv, finite_difference, rtol=0.0, atol=2e-9)

    jet = rust_module().periodic_bspline_input_location_first_derivative(
        t.reshape(-1, 1), float(knots[0]), float(knots[-1]), 3, knots.size - 1
    )
    np.testing.assert_allclose(deriv, jet[:, :, 0], rtol=0.0, atol=0.0)
