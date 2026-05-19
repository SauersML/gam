"""Tests for :class:`gamfit.torch.DuchonSmoother` and :func:`penalized_ridge_solve`.

Covers both modes of the user-facing 1-D Duchon smoother (automatic + learned,
with the length-3 ``log_smoothing`` triple-operator parameter) plus a
``gradcheck`` of the fixed-``λ`` ridge primitive that backs learned mode.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

try:
    import torch
    import gamfit.torch as gt
except ImportError:
    pytest.skip("torch dependency unavailable", allow_module_level=True)


# ---------------------------- penalized_ridge_solve --------------------------- #


def test_penalized_ridge_solve_matches_normal_equations() -> None:
    """``penalized_ridge_solve`` should reproduce the closed-form normal-equation solve."""
    rng = np.random.default_rng(0)
    n, m, d = 30, 5, 2
    x = torch.tensor(rng.standard_normal((n, m)), dtype=torch.float64)
    y = torch.tensor(rng.standard_normal((n, d)), dtype=torch.float64)
    s0 = torch.tensor(rng.standard_normal((m, m)), dtype=torch.float64)
    penalty = s0 @ s0.t() + 0.1 * torch.eye(m, dtype=torch.float64)
    lam = torch.tensor(0.7, dtype=torch.float64)

    beta, fitted = gt.penalized_ridge_solve(x, y, penalty, lam)

    hessian_np = x.numpy().T @ x.numpy() + 0.7 * penalty.numpy()
    expected_beta = np.linalg.solve(hessian_np, x.numpy().T @ y.numpy())
    np.testing.assert_allclose(beta.numpy(), expected_beta, rtol=0, atol=1e-12)
    np.testing.assert_allclose(fitted.numpy(), x.numpy() @ expected_beta, rtol=0, atol=1e-12)


def test_penalized_ridge_solve_weights_match_weighted_normal_equations() -> None:
    rng = np.random.default_rng(1)
    n, m, d = 24, 4, 1
    x = torch.tensor(rng.standard_normal((n, m)), dtype=torch.float64)
    y = torch.tensor(rng.standard_normal((n, d)), dtype=torch.float64)
    penalty = torch.eye(m, dtype=torch.float64)
    weights = torch.tensor(rng.uniform(0.5, 1.5, size=n), dtype=torch.float64)
    lam = torch.tensor(0.3, dtype=torch.float64)

    beta, fitted = gt.penalized_ridge_solve(x, y, penalty, lam, weights=weights)

    wx_np = weights.numpy()[:, None] * x.numpy()
    hessian_np = wx_np.T @ x.numpy() + 0.3 * penalty.numpy()
    expected_beta = np.linalg.solve(hessian_np, wx_np.T @ y.numpy())
    np.testing.assert_allclose(beta.numpy(), expected_beta, rtol=0, atol=1e-12)
    np.testing.assert_allclose(fitted.numpy(), x.numpy() @ expected_beta, rtol=0, atol=1e-12)


def test_penalized_ridge_solve_gradcheck() -> None:
    """All four inputs (x, y, S, λ) must carry analytic gradients."""
    rng = np.random.default_rng(2)
    n, m, d = 12, 4, 1
    x = torch.tensor(rng.standard_normal((n, m)), dtype=torch.float64, requires_grad=True)
    y = torch.tensor(rng.standard_normal((n, d)), dtype=torch.float64, requires_grad=True)
    s0 = rng.standard_normal((m, m))
    penalty = torch.tensor(
        s0 @ s0.T + 0.5 * np.eye(m), dtype=torch.float64, requires_grad=True
    )
    lam = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)

    def f(
        x_: torch.Tensor, y_: torch.Tensor, s_: torch.Tensor, lam_: torch.Tensor
    ) -> torch.Tensor:
        beta, fitted = gt.penalized_ridge_solve(x_, y_, s_, lam_)
        return beta.sum() + fitted.sum()

    assert torch.autograd.gradcheck(
        f, (x, y, penalty, lam), eps=1e-6, atol=1e-5, rtol=1e-3
    )


# ------------------------------- DuchonSmoother ------------------------------- #


def _smooth_problem(n: int = 64, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    x = torch.tensor(np.linspace(0.0, 1.0, n), dtype=torch.float64)
    truth = torch.sin(2 * torch.pi * x)
    noise = torch.tensor(0.05 * rng.standard_normal(n), dtype=torch.float64)
    return x, truth + noise


def test_duchon_smoother_auto_mode_exposes_no_parameters() -> None:
    sm = gt.DuchonSmoother(domain=(0.0, 1.0))
    assert sm.mode == "auto"
    assert list(sm.parameters()) == []
    x, y = _smooth_problem(seed=10)
    out = sm(x, y)
    rss = float(((out.fitted - y) ** 2).sum())
    naive_rss = float((y ** 2).sum())
    assert rss < 0.5 * naive_rss, (rss, naive_rss)


def test_duchon_smoother_learned_mode_exposes_three_parameters() -> None:
    """Learned-mode ``log_smoothing`` is a single length-3 parameter (triple operator)."""
    sm = gt.DuchonSmoother(
        domain=(0.0, 1.0), mode="learned", init_log_smoothing=(1.0, 0.5, -0.5)
    )
    assert sm.mode == "learned"
    params = list(sm.parameters())
    assert len(params) == 1
    assert params[0].numel() == 3
    assert params[0].requires_grad
    assert params[0].detach().tolist() == [1.0, 0.5, -0.5]


def test_duchon_smoother_learned_mode_carries_per_component_autograd() -> None:
    """Every entry of ``log_smoothing`` must receive a finite gradient."""
    torch.manual_seed(0)
    sm = gt.DuchonSmoother(domain=(0.0, 1.0), mode="learned")
    x, y = _smooth_problem(seed=11)
    out = sm(x, y)
    torch.autograd.backward(out.smoothing_score)
    assert sm.log_smoothing is not None
    grad = sm.log_smoothing.grad
    assert grad is not None
    assert grad.shape == (3,)
    assert torch.isfinite(grad).all()
    # Each operator should influence the score, not just one.
    assert (grad.abs() > 0).all(), grad.tolist()


def test_duchon_smoother_learned_mode_drives_log_smoothing_toward_optimum() -> None:
    """Adam descent on ``-smoothing_score`` should drive each weight toward a better region."""
    torch.manual_seed(0)
    sm = gt.DuchonSmoother(
        domain=(0.0, 1.0), mode="learned", init_log_smoothing=(8.0, 8.0, 8.0)
    )
    x, y = _smooth_problem(seed=11)
    score_before = float(sm(x, y).smoothing_score.detach())

    opt = torch.optim.Adam(sm.parameters(), lr=0.05)
    for _ in range(150):
        opt.zero_grad()
        out = sm(x, y)
        torch.autograd.backward(-out.smoothing_score)
        opt.step()

    score_after = float(sm(x, y).smoothing_score.detach())
    assert sm.log_smoothing is not None
    assert score_after > score_before, (score_before, score_after)
    # Heavy oversmoothing should have been backed off in at least one operator.
    final = sm.log_smoothing.detach().tolist()
    assert min(final) < 7.0, final


def test_duchon_smoother_predict_at_new_locations() -> None:
    sm = gt.DuchonSmoother(domain=(0.0, 1.0))
    x_train, y_train = _smooth_problem(n=80, seed=13)
    out = sm(x_train, y_train)
    x_test = torch.linspace(0.05, 0.95, 50, dtype=torch.float64)
    y_test_truth = torch.sin(2 * torch.pi * x_test)
    y_test_pred = sm.predict(x_test, out.coefficients)
    err = float(((y_test_pred - y_test_truth) ** 2).mean())
    assert err < 0.05, err


def test_duchon_smoother_modes_produce_consistent_shapes() -> None:
    x, y = _smooth_problem(seed=12)
    sm_auto = gt.DuchonSmoother(domain=(0.0, 1.0))
    sm_learn = gt.DuchonSmoother(domain=(0.0, 1.0), mode="learned")
    out_auto = sm_auto(x, y)
    out_learn = sm_learn(x, y)
    assert out_auto.coefficients.shape == out_learn.coefficients.shape
    assert out_auto.fitted.shape == out_learn.fitted.shape == y.shape


def test_duchon_smoother_explicit_centers() -> None:
    centers = torch.tensor([0.0, 0.25, 0.4, 0.55, 0.75, 1.0], dtype=torch.float64)
    sm = gt.DuchonSmoother(domain=(0.0, 1.0), centers=centers)
    assert sm.get_buffer("centers").shape == (6,)
    x, y = _smooth_problem(seed=14)
    out = sm(x, y)
    # Basis width = n_centers + 2 polynomial nulls (1, t).
    assert out.coefficients.shape[-1] == 8 if out.coefficients.dim() > 1 else True
    assert out.coefficients.shape[0] == 8


def test_duchon_smoother_invalid_domain_rejected() -> None:
    with pytest.raises(ValueError, match="lo < hi"):
        gt.DuchonSmoother(domain=(1.0, 0.0))


def test_duchon_smoother_invalid_init_length_rejected() -> None:
    with pytest.raises(ValueError, match="3 entries"):
        gt.DuchonSmoother(
            domain=(0.0, 1.0), mode="learned", init_log_smoothing=(0.0, 0.0)
        )


def test_duchon_smoother_invalid_mode_rejected() -> None:
    with pytest.raises(ValueError, match="mode"):
        gt.DuchonSmoother(domain=(0.0, 1.0), mode="bogus")


def test_duchon_smoother_per_operator_effect_on_fit() -> None:
    """Increasing the curvature weight alone should produce a stiffer fit."""
    x, y = _smooth_problem(n=80, seed=15)
    sm_soft = gt.DuchonSmoother(
        domain=(0.0, 1.0), mode="learned",
        init_log_smoothing=(-4.0, -4.0, -4.0),
    )
    sm_stiff = gt.DuchonSmoother(
        domain=(0.0, 1.0), mode="learned",
        init_log_smoothing=(6.0, -4.0, -4.0),  # curvature dialled way up only
    )
    out_soft = sm_soft(x, y)
    out_stiff = sm_stiff(x, y)
    # Stiffer fit should be smoother — second-difference of the fitted values
    # is smaller in absolute sum.
    def total_variation(v: torch.Tensor) -> float:
        return float(torch.abs(v[2:] - 2 * v[1:-1] + v[:-2]).sum())

    tv_soft = total_variation(out_soft.fitted.detach())
    tv_stiff = total_variation(out_stiff.fitted.detach())
    assert tv_stiff < tv_soft, (tv_soft, tv_stiff)
