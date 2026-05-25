"""Callable-basis contract for :class:`gamfit.Matern`."""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import gamfit


def test_shapes_and_autograd_nu_2_5() -> None:
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((8, 2))
    spec = gamfit.Matern(centers=centers, nu=2.5, length_scale=1.0)
    B = 6
    x = torch.as_tensor(rng.standard_normal(B), dtype=torch.float64)
    y = torch.as_tensor(rng.standard_normal(B), dtype=torch.float64)
    phi = spec.evaluate(x, y)
    assert phi.shape == (B, 8)
    jac = spec.jacobian(x, y)
    assert jac.shape == (B, 8, 2)

    # Compare a single point: analytic d/dx φ_i(x, y) for nu=2.5.
    cx = torch.as_tensor(centers[:, 0], dtype=torch.float64)
    cy = torch.as_tensor(centers[:, 1], dtype=torch.float64)
    dx = x[:, None] - cx[None, :]
    dy = y[:, None] - cy[None, :]
    r = torch.sqrt(dx * dx + dy * dy + 1e-300)
    a = math.sqrt(5.0) * r
    # φ = (1 + a + a²/3) exp(-a) = expr; dφ/dr = -5/3 · r · exp(-a) · (1 + a)
    dphi_dr = -5.0 / 3.0 * r * torch.exp(-a) * (1.0 + a)
    expected_dx = dphi_dr * (dx / r)
    expected_dy = dphi_dr * (dy / r)
    assert torch.allclose(jac[..., 0], expected_dx, atol=1e-7)
    assert torch.allclose(jac[..., 1], expected_dy, atol=1e-7)


def test_hessian_shape() -> None:
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((4, 2))
    spec = gamfit.Matern(centers=centers, nu=1.5, length_scale=0.5)
    x = torch.as_tensor(rng.standard_normal(5), dtype=torch.float64)
    y = torch.as_tensor(rng.standard_normal(5), dtype=torch.float64)
    H = spec.hessian(x, y)
    assert H.shape == (5, 4, 2, 2)
