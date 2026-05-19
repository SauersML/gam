"""Tests for :class:`gamfit.torch.BSplineSmoother` and :func:`penalized_ridge_solve`.

Covers both modes of the user-facing smoother (automatic + learned) plus a
``gradcheck`` of the fixed-``λ`` ridge primitive that backs the learned mode.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
    import gamfit.torch as gt
except ImportError:
    pytest.skip("torch dependency unavailable", allow_module_level=True)


def _require_ffi(name: str) -> None:
    from gamfit._binding import rust_module

    if not hasattr(rust_module(), name):
        pytest.skip(f"engine missing FFI export `{name}`")


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


# ------------------------------ BSplineSmoother ------------------------------- #


def _smooth_problem(n: int = 64, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    x = torch.tensor(np.linspace(0.0, 1.0, n), dtype=torch.float64)
    truth = torch.sin(2 * torch.pi * x)
    noise = torch.tensor(0.05 * rng.standard_normal(n), dtype=torch.float64)
    return x, truth + noise


def test_bspline_smoother_auto_mode_exposes_no_parameters() -> None:
    _require_ffi("bspline_basis")
    knots = torch.linspace(0.0, 1.0, 10, dtype=torch.float64)
    sm = gt.BSplineSmoother(knots)
    assert sm.mode == "auto"
    assert list(sm.parameters()) == []
    x, y = _smooth_problem(seed=10)
    out = sm(x, y)
    rss = float(((out.fitted - y) ** 2).sum())
    naive_rss = float((y ** 2).sum())
    assert rss < 0.5 * naive_rss, (rss, naive_rss)


def test_bspline_smoother_learned_mode_has_one_parameter() -> None:
    _require_ffi("bspline_basis")
    knots = torch.linspace(0.0, 1.0, 10, dtype=torch.float64)
    sm = gt.BSplineSmoother(knots, mode="learned", init_log_smoothing=1.5)
    assert sm.mode == "learned"
    params = list(sm.parameters())
    assert len(params) == 1
    assert params[0].requires_grad
    assert params[0].numel() == 1


def test_bspline_smoother_learned_mode_carries_autograd() -> None:
    """In learned mode, ``log_smoothing`` should receive a finite gradient."""
    _require_ffi("bspline_basis")
    torch.manual_seed(0)
    knots = torch.linspace(0.0, 1.0, 10, dtype=torch.float64)
    sm = gt.BSplineSmoother(knots, mode="learned", init_log_smoothing=0.0)
    x, y = _smooth_problem(seed=11)
    out = sm(x, y)
    torch.autograd.backward(out.smoothing_score)
    assert sm.log_smoothing is not None
    grad = sm.log_smoothing.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0


def test_bspline_smoother_learned_mode_drives_log_smoothing_toward_optimum() -> None:
    """Adam descent on ``-smoothing_score`` should move ``log_smoothing`` toward higher score."""
    _require_ffi("bspline_basis")
    torch.manual_seed(0)
    knots = torch.linspace(0.0, 1.0, 10, dtype=torch.float64)
    # Start far above the optimum (heavy oversmoothing) so the local gradient
    # is large and pushes us into a clearly better region.
    sm = gt.BSplineSmoother(knots, mode="learned", init_log_smoothing=8.0)
    x, y = _smooth_problem(seed=11)
    score_before = float(sm(x, y).smoothing_score.detach())
    assert sm.log_smoothing is not None
    log_lambda_before = float(sm.log_smoothing.detach())

    opt = torch.optim.Adam(sm.parameters(), lr=0.05)
    for _ in range(120):
        opt.zero_grad()
        out = sm(x, y)
        torch.autograd.backward(-out.smoothing_score)
        opt.step()

    score_after = float(sm(x, y).smoothing_score.detach())
    log_lambda_after = float(sm.log_smoothing.detach())
    assert log_lambda_after < log_lambda_before, (log_lambda_before, log_lambda_after)
    assert score_after > score_before, (score_before, score_after)


def test_bspline_smoother_modes_produce_consistent_shapes() -> None:
    _require_ffi("bspline_basis")
    knots = torch.linspace(0.0, 1.0, 10, dtype=torch.float64)
    x, y = _smooth_problem(seed=12)
    sm_auto = gt.BSplineSmoother(knots)
    sm_learn = gt.BSplineSmoother(knots, mode="learned")
    out_auto = sm_auto(x, y)
    out_learn = sm_learn(x, y)
    assert out_auto.coefficients.shape == out_learn.coefficients.shape
    assert out_auto.fitted.shape == out_learn.fitted.shape == y.shape


def test_bspline_smoother_invalid_mode_rejected() -> None:
    knots = torch.linspace(0.0, 1.0, 10, dtype=torch.float64)
    with pytest.raises(ValueError, match="mode"):
        gt.BSplineSmoother(knots, mode="bogus")


def test_bspline_smoother_predict_at_new_locations() -> None:
    _require_ffi("bspline_basis")
    knots = torch.linspace(0.0, 1.0, 12, dtype=torch.float64)
    sm = gt.BSplineSmoother(knots)
    x_train, y_train = _smooth_problem(n=80, seed=13)
    out = sm(x_train, y_train)

    x_test = torch.linspace(0.05, 0.95, 50, dtype=torch.float64)
    y_test_truth = torch.sin(2 * torch.pi * x_test)
    y_test_pred = sm.predict(x_test, out.coefficients)
    err = float(((y_test_pred - y_test_truth) ** 2).mean())
    assert err < 0.05, err
