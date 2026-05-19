"""Tests for the periodic smoother in gamfit.torch.

Math tests reach into private symbols to verify the underlying primitives;
user-facing tests go through the public ``PeriodicSmoother`` API.
"""

from __future__ import annotations

import math

import pytest

try:
    import torch
    from gamfit.torch import PeriodicSmoother
    from gamfit.torch._cyclic_duchon import (
        _coeff_tensor,
        _cyclic_duchon_basis,
        _cyclic_triple_penalty,
        _periodic_bernoulli,
        _quadratic_fit,
    )
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
    b0 = _cyclic_duchon_basis(t_zero, centers, period, m)
    bP = _cyclic_duchon_basis(t_period, centers, period, m)
    assert torch.allclose(b0, bP, atol=1e-12, rtol=0.0)


# 2. Bernoulli polynomial spot checks ----------------------------------------


def _bernoulli_eval(n: int, u: float) -> float:
    coeffs = _coeff_tensor(n, dtype=torch.float64, device=torch.device("cpu"))
    return float(
        _periodic_bernoulli(torch.tensor([u], dtype=torch.float64), coeffs).item()
    )


def test_bernoulli_polynomial_spot_values() -> None:
    assert math.isclose(_bernoulli_eval(2, 0.0), 1.0 / 6.0, abs_tol=1e-14)
    assert math.isclose(_bernoulli_eval(4, 0.0), -1.0 / 30.0, abs_tol=1e-14)
    assert math.isclose(_bernoulli_eval(6, 0.0), 1.0 / 42.0, abs_tol=1e-14)
    assert math.isclose(_bernoulli_eval(6, 0.5), -31.0 / 1344.0, abs_tol=1e-13)
    expected_b2_0p7 = 0.7 * 0.7 - 0.7 + 1.0 / 6.0
    assert math.isclose(_bernoulli_eval(2, 0.7), expected_b2_0p7, abs_tol=1e-14)


def test_bernoulli_periodic_extension() -> None:
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
    s0, s1, s2 = _cyclic_triple_penalty(centers, period, m)
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
        s0, s1, s2 = _cyclic_triple_penalty(centers, period, m)
        for s in (s0, s1, s2):
            assert torch.all(s[-1, :] == 0.0)
            assert torch.all(s[:, -1] == 0.0)


# 5. Numerical integration cross-check for stiffness penalty -----------------


def test_stiffness_penalty_matches_numerical_integration() -> None:
    period = 1.0
    m = 2
    k = 8
    centers = torch.linspace(0.0, period, k + 1, dtype=torch.float64)[:-1]
    _, _, s2 = _cyclic_triple_penalty(centers, period, m)

    grid = torch.linspace(0.0, period, 4096 + 1, dtype=torch.float64)[:-1]
    h = period / 4096.0
    Xg = _cyclic_duchon_basis(grid, centers, period, m)[:, :k]
    Xp = _cyclic_duchon_basis((grid + h) % period, centers, period, m)[:, :k]
    Xm_ = _cyclic_duchon_basis((grid - h) % period, centers, period, m)[:, :k]
    d2 = (Xp - 2.0 * Xg + Xm_) / (h * h)
    numerical = (d2.t() @ d2) * h
    analytic = s2[:k, :k]
    rel = (numerical - analytic).abs().max() / analytic.abs().max()
    assert rel.item() < 5e-4


# 6. Bernoulli kernel matches truncated Fourier reference --------------------


@pytest.mark.parametrize("m", [2, 3])
def test_kernel_matches_fourier_reference(m: int) -> None:
    period = 1.0
    tau = torch.linspace(0.0, period, 17, dtype=torch.float64)[:-1]
    center = torch.zeros(1, dtype=torch.float64)
    basis = _cyclic_duchon_basis(tau, center, period, m)[:, 0]

    ks = torch.arange(1, 401, dtype=torch.float64)
    angles = (2.0 * math.pi * ks.unsqueeze(0) * tau.unsqueeze(1)) / period
    fourier = (
        2.0
        * (period / (2.0 * math.pi)) ** (2 * m)
        * (torch.cos(angles) / ks ** (2 * m)).sum(dim=1)
    )
    assert torch.allclose(basis, fourier, atol=1e-10, rtol=0.0)


# 7. _quadratic_fit smoke test -----------------------------------------------


def test_quadratic_fit_constant_passes_through_unpenalized_intercept() -> None:
    period = 2.0
    k = 6
    centers = torch.linspace(0.0, period, k + 1, dtype=torch.float64)[:-1]
    t = torch.linspace(0.0, period, 80 + 1, dtype=torch.float64)[:-1]
    X = _cyclic_duchon_basis(t, centers, period, 2)
    _, _, s2 = _cyclic_triple_penalty(centers, period, 2)
    S = 0.5 * s2
    y = 2.1 + 0.0 * t
    beta, fitted, score = _quadratic_fit(X, y, S)
    assert torch.allclose(fitted, y, atol=1e-8)
    assert beta[:-1].abs().max().item() < 1e-8
    assert torch.isfinite(score)


# --- Public API tests --------------------------------------------------------


def _synthetic_periodic_data(period: float, n: int, noise: float = 0.1):
    t = torch.linspace(0.0, period, n + 1, dtype=torch.float64)[:-1]
    truth = torch.sin(2.0 * math.pi * t / period) + 0.1 * torch.sin(
        4.0 * math.pi * t / period
    )
    y = truth + noise * torch.randn(n, dtype=torch.float64)
    return t, y, truth


def test_auto_mode_returns_sensible_fit_without_parameter_exposure() -> None:
    period = 2.0 * math.pi
    smoother = PeriodicSmoother(period=period).double()

    # No trainable parameters surface in automatic mode.
    assert list(smoother.parameters()) == []

    torch.manual_seed(3)
    t, y, truth = _synthetic_periodic_data(period, 250, noise=0.1)
    out = smoother(t, y)

    assert out.coefficients.shape[0] == int(smoother.centers.shape[0]) + 1
    mse = ((out.fitted - truth) ** 2).mean().item()
    assert mse < 5e-2
    assert torch.isfinite(out.smoothing_score)


def test_learned_mode_exposes_one_parameter_and_grad_flows() -> None:
    period = 2.0 * math.pi
    smoother = PeriodicSmoother(period=period, mode="learned").double()

    params = list(smoother.parameters())
    assert len(params) == 1
    (log_smoothing,) = params
    assert log_smoothing.requires_grad

    torch.manual_seed(5)
    t, y, _ = _synthetic_periodic_data(period, 200, noise=0.1)
    out = smoother(t, y)
    (-out.smoothing_score).backward()

    assert log_smoothing.grad is not None
    assert torch.isfinite(log_smoothing.grad).all()
    assert log_smoothing.grad.abs().max().item() > 0.0


def test_learned_mode_mini_training_loop_converges() -> None:
    period = 2.0 * math.pi
    smoother = PeriodicSmoother(
        period=period, n_centers=20, mode="learned"
    ).double()
    init_log_smoothing = next(smoother.parameters()).detach().clone()

    torch.manual_seed(11)
    t, y, truth = _synthetic_periodic_data(period, 400, noise=0.1)

    opt = torch.optim.Adam(smoother.parameters(), lr=0.1)
    for _ in range(100):
        opt.zero_grad()
        out = smoother(t, y)
        loss = -out.smoothing_score
        loss.backward()
        opt.step()

    with torch.no_grad():
        final = smoother(t, y)
        mse = ((final.fitted - truth) ** 2).mean().item()
    assert mse < 5e-3
    delta = (next(smoother.parameters()).detach() - init_log_smoothing).abs()
    assert (delta > 1e-3).all().item()


@pytest.mark.parametrize("mode", ["auto", "learned"])
def test_constant_signal_absorbs_into_free_intercept(mode: str) -> None:
    period = 3.0
    smoother = PeriodicSmoother(period=period, n_centers=12, mode=mode).double()
    n = 150
    t = torch.linspace(0.0, period, n + 1, dtype=torch.float64)[:-1]
    y = 3.5 * torch.ones(n, dtype=torch.float64)

    with torch.no_grad():
        out = smoother(t, y)

    kernel_coefs = out.coefficients[:-1]
    intercept = out.coefficients[-1].item()
    assert kernel_coefs.abs().max().item() < 1e-6
    assert abs(intercept - 3.5) < 1e-6


def test_predict_round_trip_in_auto_mode() -> None:
    period = 2.0 * math.pi
    smoother = PeriodicSmoother(period=period).double()
    torch.manual_seed(2)
    t, y, _ = _synthetic_periodic_data(period, 200, noise=0.05)
    out = smoother(t, y)
    fitted_via_predict = smoother.predict(t, out.coefficients)
    assert torch.allclose(fitted_via_predict, out.fitted, atol=1e-10)
