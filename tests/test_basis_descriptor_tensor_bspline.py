"""Callable-basis contract for :class:`gamfit.TensorBSpline` (also covers
:func:`gamfit.Cylinder` / :func:`gamfit.Torus`, which are factory functions
returning a :class:`TensorBSpline`).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import gamfit


def _tensor_spec() -> "gamfit.TensorBSpline":
    knots_a = np.linspace(0.0, 1.0, 7 + 2 * 3)
    knots_b = np.linspace(0.0, 1.0, 5 + 2 * 3)
    return gamfit.TensorBSpline(
        marginals=[
            gamfit.BSpline(knots=knots_a, degree=3, periodic=False),
            gamfit.BSpline(knots=knots_b, degree=3, periodic=False),
        ]
    )


def test_evaluate_shape() -> None:
    spec = _tensor_spec()
    B = 11
    x = torch.linspace(0.05, 0.95, B, dtype=torch.float64)
    y = torch.linspace(0.1, 0.9, B, dtype=torch.float64)
    phi = spec.evaluate(x, y)
    assert phi.shape == (B, spec.basis_size)


def test_jacobian_matches_autograd() -> None:
    spec = _tensor_spec()
    B = 5
    x = torch.linspace(0.05, 0.95, B, dtype=torch.float64, requires_grad=True)
    y = torch.linspace(0.1, 0.9, B, dtype=torch.float64, requires_grad=True)
    jac = spec.jacobian(x, y)
    M = spec.basis_size
    assert jac.shape == (B, M, 2)


def test_partition_of_unity() -> None:
    spec = _tensor_spec()
    B = 17
    x = torch.linspace(0.1, 0.9, B, dtype=torch.float64)
    y = torch.linspace(0.1, 0.9, B, dtype=torch.float64)
    phi = spec.evaluate(x, y)
    row_sums = phi.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-9)
