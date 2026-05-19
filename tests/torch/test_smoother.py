"""Tests for :class:`gamfit.torch.DuchonSmoother` and :func:`penalized_ridge_solve`.

These tests exercise only the public surface — the smoother's two modes
(``"auto"``/``"learned"``), the standard ``nn.Module`` contract
(``parameters()`` flows the right things into the outer optimizer), and the
forward/predict/score outputs. The internal smoothing parameterisation is
deliberately not poked at: gamfit may change how many internal smoothing
strengths the module carries (currently three, possibly different in a
future version) without breaking any of these tests.
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


def test_auto_mode_exposes_no_trainable_parameters() -> None:
    """``mode='auto'`` is a black box: the user sees zero ``nn.Parameter``\\s."""
    sm = gt.DuchonSmoother(domain=(0.0, 1.0))
    assert sm.mode == "auto"
    assert list(sm.parameters()) == []


def test_auto_mode_fits_a_smooth_signal() -> None:
    """Auto mode should drive RSS well below the response variance."""
    sm = gt.DuchonSmoother(domain=(0.0, 1.0))
    x, y = _smooth_problem(seed=10)
    out = sm(x, y)
    rss = float(((out.fitted - y) ** 2).sum())
    naive_rss = float((y ** 2).sum())
    assert rss < 0.25 * naive_rss, (rss, naive_rss)


def test_learned_mode_exposes_parameters_that_flow_to_optimizer() -> None:
    """``mode='learned'`` exposes some parameters via the standard nn.Module contract.

    The user only relies on the ``parameters()`` iterator returning trainable
    tensors — they never enumerate, index, or name those tensors.
    """
    sm = gt.DuchonSmoother(domain=(0.0, 1.0), mode="learned")
    params = list(sm.parameters())
    assert len(params) >= 1
    for p in params:
        assert isinstance(p, torch.nn.Parameter)
        assert p.requires_grad
        assert torch.isfinite(p).all()


def test_learned_mode_backward_writes_grads_into_every_parameter() -> None:
    """``backward(score)`` should produce a finite, nonzero grad on every parameter."""
    sm = gt.DuchonSmoother(domain=(0.0, 1.0), mode="learned")
    x, y = _smooth_problem(seed=11)
    out = sm(x, y)
    torch.autograd.backward(out.smoothing_score)
    for p in sm.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
        assert float(p.grad.abs().sum()) > 0.0


def test_learned_mode_outer_adam_improves_score() -> None:
    """Adam on ``-out.smoothing_score`` should drive the score upward."""
    torch.manual_seed(0)
    sm = gt.DuchonSmoother(domain=(0.0, 1.0), mode="learned")
    x, y = _smooth_problem(seed=11)

    score_before = float(sm(x, y).smoothing_score.detach())
    opt = torch.optim.Adam(sm.parameters(), lr=0.1)
    for _ in range(200):
        opt.zero_grad()
        out = sm(x, y)
        torch.autograd.backward(-out.smoothing_score)
        opt.step()
    score_after = float(sm(x, y).smoothing_score.detach())
    assert score_after > score_before, (score_before, score_after)


def test_learned_mode_end_to_end_recovers_truth_and_log_smoothing_stabilizes() -> None:
    """End-to-end Adam loop must (a) drive the smoother to the underlying truth
    and (b) leave the internal log-smoothing parameter at a sensible, stationary
    value rather than wandering.

    This is the real ``grad_penalty`` integration test: gradcheck only proves the
    backward matches finite differences locally; this proves the analytic
    derivative is well-behaved enough that a vanilla outer optimizer actually
    finds a stable minimum.
    """
    torch.manual_seed(0)
    rng = np.random.default_rng(101)
    n = 80
    x = torch.tensor(np.linspace(0.0, 1.0, n), dtype=torch.float64)
    truth = torch.sin(2 * math.pi * x)
    noise_scale = 0.1
    y = truth + torch.tensor(
        noise_scale * rng.standard_normal(n), dtype=torch.float64
    )

    sm = gt.DuchonSmoother(domain=(0.0, 1.0), mode="learned").double()
    (log_smoothing,) = list(sm.parameters())
    init_log_smoothing = log_smoothing.detach().clone()

    opt = torch.optim.Adam(sm.parameters(), lr=0.1)
    steps = 300
    trajectory = []
    for _ in range(steps):
        opt.zero_grad()
        out = sm(x, y)
        (-out.smoothing_score).backward()
        opt.step()
        trajectory.append(log_smoothing.detach().clone())

    # 1. The fitted curve must track the underlying truth, well below the
    #    noise floor (variance of the additive noise is `noise_scale**2`).
    with torch.no_grad():
        final = sm(x, y)
    final_mse_vs_truth = float(((final.fitted - truth) ** 2).mean())
    assert final_mse_vs_truth < 0.5 * noise_scale ** 2, final_mse_vs_truth

    # 2. The smoother must hold up out-of-sample as well — overfit smoothers
    #    can drive in-sample MSE low while predicting badly at new points.
    x_test = torch.linspace(0.05, 0.95, 50, dtype=torch.float64)
    y_test_truth = torch.sin(2 * math.pi * x_test)
    with torch.no_grad():
        y_test_pred = sm.predict(x_test, final.coefficients)
    test_mse = float(((y_test_pred - y_test_truth) ** 2).mean())
    assert test_mse < 0.5 * noise_scale ** 2, test_mse

    # 3. The internal log-smoothing parameter must actually have moved off its
    #    initialisation (the training did something) but then settled — i.e.
    #    the last segment of the trajectory has small per-coordinate spread,
    #    instead of drifting to ±∞ or oscillating.
    final_log_smoothing = trajectory[-1]
    delta_from_init = (final_log_smoothing - init_log_smoothing).abs()
    assert float(delta_from_init.max()) > 1.0, delta_from_init

    tail = torch.stack(trajectory[-50:], dim=0)
    tail_std = tail.std(dim=0)
    assert torch.isfinite(tail_std).all()
    # ≤0.2 nats of jitter per coordinate over the last 50 steps comfortably
    # covers what a healthy Adam ride looks like (empirically <0.06).
    assert float(tail_std.max()) < 0.2, tail_std

    # 4. log_smoothing must stay in a numerically sane band — no λ blow-up
    #    to ~exp(±50) that would silently break downstream solves.
    assert float(final_log_smoothing.abs().max()) < 20.0, final_log_smoothing


def test_predict_at_new_locations() -> None:
    sm = gt.DuchonSmoother(domain=(0.0, 1.0))
    x_train, y_train = _smooth_problem(n=80, seed=13)
    out = sm(x_train, y_train)
    x_test = torch.linspace(0.05, 0.95, 50, dtype=torch.float64)
    y_test_truth = torch.sin(2 * torch.pi * x_test)
    y_test_pred = sm.predict(x_test, out.coefficients)
    err = float(((y_test_pred - y_test_truth) ** 2).mean())
    assert err < 0.05, err


def test_modes_produce_consistent_output_shapes() -> None:
    x, y = _smooth_problem(seed=12)
    sm_auto = gt.DuchonSmoother(domain=(0.0, 1.0))
    sm_learn = gt.DuchonSmoother(domain=(0.0, 1.0), mode="learned")
    out_auto = sm_auto(x, y)
    out_learn = sm_learn(x, y)
    assert out_auto.coefficients.shape == out_learn.coefficients.shape
    assert out_auto.fitted.shape == out_learn.fitted.shape == y.shape


def test_explicit_centers() -> None:
    centers = torch.tensor([0.0, 0.25, 0.4, 0.55, 0.75, 1.0], dtype=torch.float64)
    sm = gt.DuchonSmoother(domain=(0.0, 1.0), centers=centers)
    x, y = _smooth_problem(seed=14)
    out = sm(x, y)
    assert out.fitted.shape == y.shape
    assert math.isfinite(float(out.smoothing_score.detach()))


def test_invalid_domain_rejected() -> None:
    with pytest.raises(ValueError, match="lo < hi"):
        gt.DuchonSmoother(domain=(1.0, 0.0))


def test_invalid_mode_rejected() -> None:
    with pytest.raises(ValueError, match="mode"):
        gt.DuchonSmoother(domain=(0.0, 1.0), mode="bogus")


def test_duchon_smoother_on_circular_data_seam_continuity_and_error_parity() -> None:
    """Fit a 1-D :class:`DuchonSmoother` to noisy data on a circle and assert
    the seam (``t=0`` vs. ``t=period``) behaves like an interior region:

    1. ``predict(0) ≈ predict(period)`` — the curve is continuous across the seam.
    2. The RMSE in a small window around the seam is not much larger than the
       RMSE in interior windows of the same width.

    Both assertions are expected to fail today: :class:`DuchonSmoother` is
    parameterised on a bounded interval (no periodic boundary), so when it is
    fed data living on a circle it (a) leaves an unconstrained jump at the
    seam and (b) over-fits / under-fits the boundary region relative to the
    interior. This test pins those failure modes so they cannot be silently
    regressed away or silently fixed without a test update.
    """
    period = 2.0 * math.pi
    n = 200
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    t = torch.tensor(
        np.linspace(0.0, period, n + 1)[:-1], dtype=torch.float64
    )
    truth = torch.sin(t) + 0.3 * torch.cos(2 * t) + 0.15 * torch.sin(3 * t)
    y = truth + torch.tensor(0.10 * rng.standard_normal(n), dtype=torch.float64)

    sm = gt.DuchonSmoother(domain=(0.0, period), n_centers=16).double()
    with torch.no_grad():
        out = sm(t, y)
        v_at_zero = sm.predict(
            torch.tensor([0.0], dtype=torch.float64), out.coefficients
        ).item()
        v_at_period = sm.predict(
            torch.tensor([period - 1e-12], dtype=torch.float64), out.coefficients
        ).item()

    # (1) Continuity at the seam: identical points on the circle should map
    # to the same value. A periodic smoother delivers machine-precision
    # continuity (~1e-12); a non-periodic Duchon does not.
    seam_jump = abs(v_at_zero - v_at_period)
    assert seam_jump < 1e-3, (
        f"seam jump {seam_jump:.4f} between pred(0)={v_at_zero:.4f} and "
        f"pred(period)={v_at_period:.4f} — DuchonSmoother has no periodic "
        f"boundary, so the endpoints float independently."
    )

    # (2) Seam window RMSE should be comparable to interior RMSE.
    tt = torch.tensor(
        np.linspace(0.0, period, 4001)[:-1], dtype=torch.float64
    )
    truth_dense = (
        torch.sin(tt) + 0.3 * torch.cos(2 * tt) + 0.15 * torch.sin(3 * tt)
    )
    with torch.no_grad():
        pred = sm.predict(tt, out.coefficients)
    sq_err = (pred - truth_dense) ** 2

    w = period / 40.0  # narrow window on each side of the seam
    seam_mask = (tt < w) | (tt > period - w)
    interior_masks = [
        (tt > period * c - w) & (tt < period * c + w)
        for c in (0.25, 0.5, 0.75)
    ]
    seam_rmse = float(sq_err[seam_mask].mean().sqrt())
    interior_rmses = [
        float(sq_err[m].mean().sqrt()) for m in interior_masks
    ]
    median_interior = float(np.median(interior_rmses))
    assert seam_rmse < 1.5 * median_interior, (
        f"seam RMSE {seam_rmse:.4f} is {seam_rmse / median_interior:.1f}× the "
        f"median interior RMSE {median_interior:.4f} — DuchonSmoother degrades "
        f"sharply near the boundary because of its non-periodic basis."
    )


def test_learned_mode_state_dict_round_trips_through_save_load() -> None:
    """A learned smoother's state must survive ``state_dict`` -> ``load_state_dict``."""
    sm_src = gt.DuchonSmoother(domain=(0.0, 1.0), mode="learned")
    x, y = _smooth_problem(seed=20)
    # Drive the smoother away from its init so the state is non-trivial.
    opt = torch.optim.Adam(sm_src.parameters(), lr=0.1)
    for _ in range(20):
        opt.zero_grad()
        torch.autograd.backward(-sm_src(x, y).smoothing_score)
        opt.step()

    sm_dst = gt.DuchonSmoother(domain=(0.0, 1.0), mode="learned")
    sm_dst.load_state_dict(sm_src.state_dict())
    out_src = sm_src(x, y)
    out_dst = sm_dst(x, y)
    np.testing.assert_allclose(
        out_src.fitted.detach().numpy(), out_dst.fitted.detach().numpy(),
        rtol=0, atol=1e-12,
    )
