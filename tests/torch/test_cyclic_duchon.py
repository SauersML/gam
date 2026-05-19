"""Tests for the cyclic (periodic) Duchon spline primitives in gamfit.torch.

Verifies seam continuity, Bernoulli polynomial spot values, PSD penalty
construction, free intercept, agreement with numerical integration and with
the truncated Fourier reference for ``G_m``, REML gradient flow, mini-training
convergence, and constant-signal absorption into the unpenalized intercept.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

try:
    import torch
    from gamfit.torch import (
        CyclicDuchonTripleSmoother,
        cyclic_duchon_bernoulli_basis,
        cyclic_duchon_quadratic_fit,
        cyclic_duchon_triple_penalty,
    )
    from gamfit.torch._cyclic_duchon import _periodic_bernoulli, _coeff_tensor
except ImportError:
    pytest.skip("torch dependency unavailable", allow_module_level=True)


torch.manual_seed(0)


# 1. Seam continuity ----------------------------------------------------------


@pytest.mark.parametrize("m", [1, 2, 3])
@pytest.mark.parametrize("k", [4, 8, 16])
def test_seam_continuity(m: int, k: int) -> None:
    period = 2.7
    centers = torch.linspace(0.0, period, k + 1, dtype=torch.float64)[:-1]
    t_zero = torch.tensor([0.0], dtype=torch.float64)
    t_period = torch.tensor([period], dtype=torch.float64)
    b0 = cyclic_duchon_bernoulli_basis(t_zero, centers, period, m)
    bP = cyclic_duchon_bernoulli_basis(t_period, centers, period, m)
    assert torch.allclose(b0, bP, atol=1e-12, rtol=0.0)


# 2. Bernoulli polynomial spot checks ----------------------------------------


def _bernoulli_eval(n: int, u: float) -> float:
    coeffs = _coeff_tensor(n, dtype=torch.float64, device=torch.device("cpu"))
    return float(_periodic_bernoulli(torch.tensor([u], dtype=torch.float64), coeffs).item())


def test_bernoulli_polynomial_spot_values() -> None:
    assert math.isclose(_bernoulli_eval(2, 0.0), 1.0 / 6.0, abs_tol=1e-14)
    assert math.isclose(_bernoulli_eval(4, 0.0), -1.0 / 30.0, abs_tol=1e-14)
    assert math.isclose(_bernoulli_eval(6, 0.0), 1.0 / 42.0, abs_tol=1e-14)
    assert math.isclose(_bernoulli_eval(6, 0.5), -31.0 / 1344.0, abs_tol=1e-13)
    expected_b2_0p7 = 0.7 * 0.7 - 0.7 + 1.0 / 6.0
    assert math.isclose(_bernoulli_eval(2, 0.7), expected_b2_0p7, abs_tol=1e-14)


def test_bernoulli_periodic_extension() -> None:
    # B_tilde_n(u + 1) == B_tilde_n(u)
    coeffs = _coeff_tensor(4, dtype=torch.float64, device=torch.device("cpu"))
    u = torch.tensor([0.13, 0.42, 0.91], dtype=torch.float64)
    base = _periodic_bernoulli(u, coeffs)
    shifted = _periodic_bernoulli(u + 1.0, coeffs)
    shifted_neg = _periodic_bernoulli(u - 3.0, coeffs)
    assert torch.allclose(base, shifted, atol=1e-14)
    assert torch.allclose(base, shifted_neg, atol=1e-14)


# 3. Penalty PSD --------------------------------------------------------------


@pytest.mark.parametrize("m", [2, 3])
def test_triple_penalty_psd_and_symmetric(m: int) -> None:
    period = 1.7
    k = 10
    centers = torch.linspace(0.0, period, k + 1, dtype=torch.float64)[:-1]
    s0, s1, s2 = cyclic_duchon_triple_penalty(centers, period, m)
    for s in (s0, s1, s2):
        assert torch.allclose(s, s.t(), atol=1e-12)
        eigvals = torch.linalg.eigvalsh(s)
        max_eig = eigvals.abs().max().item()
        assert eigvals.min().item() >= -1e-10 * max(max_eig, 1.0)


# 4. Free intercept -----------------------------------------------------------


def test_free_intercept_block_zero() -> None:
    period = 2.0
    centers = torch.linspace(0.0, period, 7, dtype=torch.float64)[:-1]
    for m in (2, 3):
        s0, s1, s2 = cyclic_duchon_triple_penalty(centers, period, m)
        for s in (s0, s1, s2):
            assert torch.all(s[-1, :] == 0.0)
            assert torch.all(s[:, -1] == 0.0)


# 5. Numerical integration cross-check for stiffness penalty -----------------


def test_stiffness_penalty_matches_numerical_integration() -> None:
    period = 1.0
    m = 2
    k = 8
    centers = torch.linspace(0.0, period, k + 1, dtype=torch.float64)[:-1]
    _, _, s2 = cyclic_duchon_triple_penalty(centers, period, m)

    grid = torch.linspace(0.0, period, 4096 + 1, dtype=torch.float64)[:-1]
    h = period / 4096.0
    # Second derivative of B_i(x) = G_m(x - c_i) via finite differences along
    # the grid (periodic). f'' approx (f(x+h) - 2 f(x) + f(x-h)) / h^2.
    Xg = cyclic_duchon_bernoulli_basis(grid, centers, period, m)[:, :k]
    Xp = cyclic_duchon_bernoulli_basis((grid + h) % period, centers, period, m)[:, :k]
    Xm_ = cyclic_duchon_bernoulli_basis((grid - h) % period, centers, period, m)[:, :k]
    d2 = (Xp - 2.0 * Xg + Xm_) / (h * h)
    numerical = (d2.t() @ d2) * h  # trapezoidal-ish on uniform grid
    analytic = s2[:k, :k]
    rel = (numerical - analytic).abs().max() / analytic.abs().max()
    assert rel.item() < 5e-4


# 6. REML gradient flow -------------------------------------------------------


def test_reml_gradient_flow() -> None:
    period = 2.0 * math.pi
    k = 16
    centers = torch.linspace(0.0, period, k + 1, dtype=torch.float64)[:-1]
    module = CyclicDuchonTripleSmoother(centers, period, m=2).double()

    torch.manual_seed(7)
    n = 200
    t = torch.linspace(0.0, period, n + 1, dtype=torch.float64)[:-1]
    truth = torch.sin(2.0 * math.pi * t / period) + 0.1 * torch.sin(
        4.0 * math.pi * t / period
    )
    y = truth + 0.1 * torch.randn(n, dtype=torch.float64)

    out = module(t, y)
    (-out.reml_score).backward()
    grad = module.log_lambdas.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().max().item() > 0.0


# 7. Mini training loop converges --------------------------------------------


def test_mini_training_loop_converges() -> None:
    period = 2.0 * math.pi
    k = 20
    centers = torch.linspace(0.0, period, k + 1, dtype=torch.float64)[:-1]
    module = CyclicDuchonTripleSmoother(centers, period, m=2).double()
    init_log_lambdas = module.log_lambdas.detach().clone()

    torch.manual_seed(11)
    n = 400
    t = torch.linspace(0.0, period, n + 1, dtype=torch.float64)[:-1]
    truth = torch.sin(2.0 * math.pi * t / period) + 0.1 * torch.sin(
        4.0 * math.pi * t / period
    )
    y = truth + 0.1 * torch.randn(n, dtype=torch.float64)

    opt = torch.optim.Adam(module.parameters(), lr=0.1)
    for _ in range(100):
        opt.zero_grad()
        out = module(t, y)
        loss = -out.reml_score
        loss.backward()
        opt.step()

    with torch.no_grad():
        final = module(t, y)
        mse = ((final.fitted - truth) ** 2).mean().item()
    assert mse < 5e-3
    delta = (module.log_lambdas.detach() - init_log_lambdas).abs()
    assert (delta > 1e-3).all().item()


# 8. Constant signal absorbs into free intercept -----------------------------


def test_constant_signal_absorbs_into_intercept() -> None:
    period = 3.0
    k = 12
    centers = torch.linspace(0.0, period, k + 1, dtype=torch.float64)[:-1]
    module = CyclicDuchonTripleSmoother(
        centers, period, m=2, init_log_lambdas=(2.0, 2.0, 2.0)
    ).double()

    n = 150
    t = torch.linspace(0.0, period, n + 1, dtype=torch.float64)[:-1]
    y = 3.5 * torch.ones(n, dtype=torch.float64)

    with torch.no_grad():
        out = module(t, y)
    kernel_coefs = out.coefficients[:-1]
    intercept = out.coefficients[-1].item()
    assert kernel_coefs.abs().max().item() < 1e-6
    assert abs(intercept - 3.5) < 1e-6


# 9. Bernoulli kernel matches truncated Fourier reference --------------------


@pytest.mark.parametrize("m", [2, 3])
def test_kernel_matches_fourier_reference(m: int) -> None:
    period = 1.0
    tau = torch.linspace(0.0, period, 17, dtype=torch.float64)[:-1]
    center = torch.zeros(1, dtype=torch.float64)
    basis = cyclic_duchon_bernoulli_basis(tau, center, period, m)[:, 0]

    # Truncated Fourier reference:
    # G_m(tau) = 2 (P / (2 pi))^{2m} sum_{k>=1} cos(2 pi k tau / P) / k^{2m}.
    ks = torch.arange(1, 401, dtype=torch.float64)
    angles = (2.0 * math.pi * ks.unsqueeze(0) * tau.unsqueeze(1)) / period
    fourier = (
        2.0
        * (period / (2.0 * math.pi)) ** (2 * m)
        * (torch.cos(angles) / ks ** (2 * m)).sum(dim=1)
    )
    assert torch.allclose(basis, fourier, atol=1e-10, rtol=0.0)


# Closed-form quadratic fit smoke test ---------------------------------------


def test_cyclic_duchon_quadratic_fit_passes_intercept_through() -> None:
    period = 2.0
    k = 6
    centers = torch.linspace(0.0, period, k + 1, dtype=torch.float64)[:-1]
    t = torch.linspace(0.0, period, 80 + 1, dtype=torch.float64)[:-1]
    X = cyclic_duchon_bernoulli_basis(t, centers, period, 2)
    _, _, s2 = cyclic_duchon_triple_penalty(centers, period, 2)
    S = 0.5 * s2
    y = 2.1 + 0.0 * t
    beta, fitted, reml = cyclic_duchon_quadratic_fit(X, y, S)
    assert torch.allclose(fitted, y, atol=1e-8)
    assert beta[:-1].abs().max().item() < 1e-8
    assert torch.isfinite(reml)
