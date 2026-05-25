"""Callable-basis contract for :class:`gamfit.PeriodicSplineCurve`."""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import gamfit


def test_evaluate_shape_and_periodicity() -> None:
    spec = gamfit.PeriodicSplineCurve(n_knots=12, degree=3)
    B = 20
    t = torch.linspace(0.0, 1.0 - 1e-6, B, dtype=torch.float64)
    phi = spec.evaluate(t)
    assert phi.shape == (B, 12)

    # Periodicity: evaluate(t + 2π) ≡ evaluate(t) because PSC reduces modulo 1
    # internally; supplying t shifted by 1 (the period in the canonical
    # parametrization) yields the same basis.
    phi_shift = spec.evaluate(t + 1.0)
    assert torch.allclose(phi, phi_shift, atol=1e-9)


def test_jacobian_shape() -> None:
    spec = gamfit.PeriodicSplineCurve(n_knots=8, degree=3)
    t = torch.linspace(0.05, 0.95, 9, dtype=torch.float64, requires_grad=True)
    jac = spec.jacobian(t)
    assert jac.shape == (9, 8, 1)
