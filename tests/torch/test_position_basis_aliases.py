"""Tests for the user-facing position-REML API surface.

Exercises the basis-aliasing layer (`basis="duchon" | "duchon_multipenalty" |
"thinplate" | "bspline"`), the canonical default penalty per basis, the
`smoothing` / `coefficients` flags, the `freeze()` inference helper, and the
periodic Duchon path.

Tests aim for behavioral correctness, not just smoke — for example, the
``test_smoothing_fixed_matches_explicit_solve`` check pins the auto-solve
path against a hand-built closed-form ridge, and the periodic test asserts
seam continuity along with the basic finiteness.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

try:
    import torch
    import gamfit.torch as gt
    from gamfit.torch._reml import FrozenPositionPredictor
except ImportError:
    pytest.skip("torch dependency unavailable", allow_module_level=True)


def _require_ffi(name: str) -> None:
    from gamfit._binding import rust_module

    if not hasattr(rust_module(), name):
        pytest.skip(f"engine missing FFI export `{name}`")


def _sample(n: int = 64, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, 1.0, size=n))
    y = np.sin(2.0 * np.pi * t) + 0.1 * rng.standard_normal(n)
    return (
        torch.tensor(t, dtype=torch.float64),
        torch.tensor(y.reshape(-1, 1), dtype=torch.float64),
    )


def _circular_sample(
    n: int = 96,
    period: float = 2 * math.pi,
    seed: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Positions covering the full period and a single-cycle smooth signal."""
    rng = np.random.default_rng(seed)
    # Spread samples uniformly across [0, period) so the centers we choose can
    # match the period exactly without trimming the support.
    t = np.sort(rng.uniform(0.0, period, size=n))
    y = np.sin(2.0 * np.pi * t / period) + 0.05 * rng.standard_normal(n)
    return (
        torch.tensor(t, dtype=torch.float64),
        torch.tensor(y.reshape(-1, 1), dtype=torch.float64),
    )


# ---------------------------------------------------------------------------
# Default penalty per basis
# ---------------------------------------------------------------------------


def test_default_basis_is_bspline() -> None:
    t, y = _sample()
    fit = gt.gaussian_reml_fit_positions(t, y)
    assert fit.basis_kind == "bspline"
    assert torch.isfinite(fit.reml_score)


def test_duchon_default_function_norm() -> None:
    _require_ffi("duchon_function_norm_penalty")
    t, y = _sample()
    fit = gt.gaussian_reml_fit_positions(t, y, basis="duchon")
    assert fit.basis_kind == "duchon"
    pen = fit.penalty.detach().numpy()
    # Function-norm penalty is symmetric PSD.
    assert pen.shape[0] == pen.shape[1]
    assert np.allclose(pen, pen.T, atol=1e-10)
    eigs = np.linalg.eigvalsh(0.5 * (pen + pen.T))
    assert eigs.min() >= -1e-8
    # Duchon m=2 has the constant+linear polynomial nullspace, so at least
    # two near-zero eigenvalues are expected.
    near_zero = int((np.abs(eigs) < 1e-6 * max(1.0, eigs.max())).sum())
    assert near_zero >= 2


def test_thinplate_alias_to_duchon_m2() -> None:
    _require_ffi("duchon_function_norm_penalty")
    t, y = _sample()
    fit_tps = gt.gaussian_reml_fit_positions(t, y, basis="thinplate")
    fit_duchon = gt.gaussian_reml_fit_positions(t, y, basis="duchon", basis_order=2)
    pen_a = fit_tps.penalty.detach().numpy()
    pen_b = fit_duchon.penalty.detach().numpy()
    assert pen_a.shape == pen_b.shape
    np.testing.assert_allclose(pen_a, pen_b, atol=1e-10)
    # Fitted curves should agree pointwise (same basis + penalty + REML).
    np.testing.assert_allclose(
        fit_tps.fitted.detach().numpy(),
        fit_duchon.fitted.detach().numpy(),
        atol=1e-10,
    )
    assert fit_tps.basis_kind == "thinplate"
    assert fit_duchon.basis_kind == "duchon"


def test_bspline_default_difference_penalty() -> None:
    t, y = _sample()
    fit = gt.gaussian_reml_fit_positions(t, y, basis="bspline")
    pen = fit.penalty.detach().numpy()
    pen_sym = 0.5 * (pen + pen.T)
    eigs = np.linalg.eigvalsh(pen_sym)
    near_zero = int((np.abs(eigs) < 1e-8 * max(1.0, eigs.max())).sum())
    # 2nd-difference penalty has nullspace dim 2 (constants + linears).
    assert near_zero >= 2


def test_duchon_multipenalty_combines_three_operators() -> None:
    _require_ffi("duchon_operator_penalties")
    t, y = _sample()
    fit = gt.gaussian_reml_fit_positions(t, y, basis="duchon_multipenalty")
    assert fit.basis_kind == "duchon_multipenalty"
    pen = fit.penalty.detach().numpy()
    # Cross-check against the explicit triple-operator override on plain duchon.
    fit_op = gt.gaussian_reml_fit_positions(
        t,
        y,
        basis="duchon",
        knots_or_centers=fit.knots_or_centers,
        penalty="triple_operator",
    )
    np.testing.assert_allclose(pen, fit_op.penalty.detach().numpy(), atol=1e-10)


# ---------------------------------------------------------------------------
# Freeze inference helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("basis", ["duchon", "thinplate", "bspline"])
def test_freeze_evaluate_roundtrip(basis: str) -> None:
    _require_ffi("duchon_function_norm_penalty")
    t, y = _sample()
    fit = gt.gaussian_reml_fit_positions(t, y, basis=basis)
    frozen = fit.freeze()
    assert isinstance(frozen, FrozenPositionPredictor)
    pred = frozen.evaluate(t)
    np.testing.assert_allclose(
        pred.detach().numpy(), fit.fitted.detach().numpy(), atol=1e-8
    )


# ---------------------------------------------------------------------------
# Penalty override (matrix and string)
# ---------------------------------------------------------------------------


def test_explicit_penalty_matrix_override() -> None:
    t, y = _sample()
    # Identity ridge of matching size — bypasses the canonical default and
    # uses the user-supplied matrix verbatim.
    knots = torch.linspace(0.0, 1.0, 10, dtype=torch.float64)
    custom_pen = torch.eye(10, dtype=torch.float64)
    fit = gt.gaussian_reml_fit_positions(
        t, y, basis="duchon", knots_or_centers=knots, penalty=custom_pen
    )
    np.testing.assert_allclose(
        fit.penalty.detach().numpy(), custom_pen.numpy(), atol=1e-12
    )
    assert torch.isfinite(fit.reml_score)


def test_string_penalty_overrides_distinct_from_default() -> None:
    _require_ffi("duchon_function_norm_penalty")
    _require_ffi("duchon_operator_penalties")
    t, y = _sample()
    fit_fn = gt.gaussian_reml_fit_positions(
        t, y, basis="duchon", penalty="function_norm"
    )
    fit_op = gt.gaussian_reml_fit_positions(
        t, y, basis="duchon", penalty="triple_operator"
    )
    pen_fn = fit_fn.penalty.detach().numpy()
    pen_op = fit_op.penalty.detach().numpy()
    assert not np.allclose(pen_fn, pen_op, atol=1e-6)
    # Both should reproduce when re-passed through the resolver.
    fit_fn_default = gt.gaussian_reml_fit_positions(t, y, basis="duchon")
    np.testing.assert_allclose(pen_fn, fit_fn_default.penalty.detach().numpy(), atol=1e-10)


def test_unsupported_penalty_string_rejects() -> None:
    t, y = _sample()
    with pytest.raises(Exception):
        gt.gaussian_reml_fit_positions(
            t, y, basis="bspline", penalty="triple_operator"
        )


# ---------------------------------------------------------------------------
# Smoothing modes
# ---------------------------------------------------------------------------


def test_smoothing_reml_default_matches_explicit() -> None:
    _require_ffi("duchon_function_norm_penalty")
    t, y = _sample()
    fit_default = gt.gaussian_reml_fit_positions(t, y, basis="duchon")
    fit_explicit = gt.gaussian_reml_fit_positions(
        t, y, basis="duchon", smoothing="reml"
    )
    np.testing.assert_allclose(
        fit_default.fitted.detach().numpy(),
        fit_explicit.fitted.detach().numpy(),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        float(fit_default.lam), float(fit_explicit.lam), atol=1e-12
    )


def test_smoothing_fixed_returns_user_lambda() -> None:
    _require_ffi("gaussian_reml_score")
    t, y = _sample()
    log_lambda = torch.tensor(-0.5, dtype=torch.float64)
    fit = gt.gaussian_reml_fit_positions(
        t, y, basis="duchon", smoothing="fixed", log_lambda=log_lambda
    )
    assert math.isclose(float(fit.lam), math.exp(-0.5), rel_tol=1e-12)
    assert torch.isfinite(fit.reml_score)


def test_smoothing_fixed_matches_explicit_solve() -> None:
    """The auto-solve β at fixed λ must equal a hand-built ridge solve."""
    _require_ffi("gaussian_reml_score")
    _require_ffi("duchon_function_norm_penalty")
    t, y = _sample()
    log_lambda = torch.tensor(0.25, dtype=torch.float64)
    fit = gt.gaussian_reml_fit_positions(
        t, y, basis="duchon", smoothing="fixed", log_lambda=log_lambda
    )
    # Build the design + penalty the API would have built, solve directly,
    # and compare to the API's coefficients.
    X = gt.duchon_basis_1d(t, fit.knots_or_centers, m=2)
    S = fit.penalty
    lam = float(log_lambda.exp())
    beta_ref = torch.linalg.solve(X.T @ X + lam * S, X.T @ y)
    np.testing.assert_allclose(
        fit.coefficients.detach().numpy(),
        beta_ref.detach().numpy(),
        rtol=1e-9,
        atol=1e-9,
    )


def test_smoothing_adam_propagates_gradient_to_log_lambda() -> None:
    _require_ffi("gaussian_reml_score")
    _require_ffi("duchon_function_norm_penalty")
    t, y = _sample()
    log_lambda = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    fit = gt.gaussian_reml_fit_positions(
        t, y, basis="duchon", smoothing="adam", log_lambda=log_lambda
    )
    fit.reml_score.backward()
    assert log_lambda.grad is not None
    assert torch.isfinite(log_lambda.grad)
    # Score should depend on λ — gradient must be nonzero away from a stationary
    # point of the REML curve. Use a small absolute floor to avoid spurious
    # nonzero-gradient assertions on near-degenerate sample draws.
    assert abs(float(log_lambda.grad)) > 1e-6


def test_triple_operator_3_vec_gradient_distinguishes_components() -> None:
    """Three-vector λ on duchon_multipenalty must yield per-component gradients."""
    _require_ffi("gaussian_reml_score")
    _require_ffi("duchon_operator_penalties")
    t, y = _sample()
    log_lambdas = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
    fit = gt.gaussian_reml_fit_positions(
        t,
        y,
        basis="duchon_multipenalty",
        smoothing="adam",
        log_lambda=log_lambdas,
    )
    fit.reml_score.backward()
    grad = log_lambdas.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    # The three operator pieces are distinct objects (mass, tension,
    # stiffness) — their REML contributions must differ at the symmetric
    # rho=0 starting point, otherwise the multipenalty path collapses to
    # a single λ in disguise.
    g = grad.detach().numpy()
    assert not np.allclose(g[0], g[1], atol=1e-6) or not np.allclose(g[1], g[2], atol=1e-6)


def test_coefficients_free_b_mode_uses_user_beta() -> None:
    """`coefficients=…` evaluates the score at the user's β without solving."""
    _require_ffi("gaussian_reml_score")
    _require_ffi("duchon_function_norm_penalty")
    t, y = _sample()
    # Fit once to discover the canonical knots/penalty.
    fit = gt.gaussian_reml_fit_positions(t, y, basis="duchon")
    # Now pass an explicit β that is NOT the ridge solution.
    fake_beta = torch.zeros_like(fit.coefficients)
    log_lambda = torch.tensor(0.0, dtype=torch.float64)
    free_fit = gt.gaussian_reml_fit_positions(
        t,
        y,
        basis="duchon",
        knots_or_centers=fit.knots_or_centers,
        penalty=fit.penalty,
        smoothing="fixed",
        log_lambda=log_lambda,
        coefficients=fake_beta,
    )
    # The free-B path must report back exactly the β we passed in.
    np.testing.assert_allclose(
        free_fit.coefficients.detach().numpy(),
        fake_beta.detach().numpy(),
        atol=1e-12,
    )
    # And the fitted value at zero β is exactly zero.
    np.testing.assert_allclose(
        free_fit.fitted.detach().numpy(),
        np.zeros_like(fit.fitted.detach().numpy()),
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Periodic Duchon
# ---------------------------------------------------------------------------


def test_periodic_duchon_end_to_end_and_seam_continuity() -> None:
    _require_ffi("duchon_function_norm_penalty")
    period = 2 * math.pi
    t, y = _circular_sample(n=128, period=period, seed=1)
    # The position-API validator requires ``max(centers) − min(centers) ==
    # period`` — the natural way to describe a closed periodic lattice. We
    # supply ``K + 1`` linspace points across ``[0, period]``; the basis
    # builder collapses the wrap duplicate at the right endpoint, leaving
    # K distinct circle points.
    #
    # ``K`` is chosen *odd* on purpose. The periodic Duchon ``m=2`` kernel
    # the engine currently uses is ``phi(r) = r`` (a triangle wave on the
    # circle); its Fourier series has only odd harmonics. On a uniform
    # K-point lattice with EVEN K, the discrete DFT lands exactly on the
    # zero (even-harmonic) modes — ``K/2 − 1`` singular values of the
    # design collapse to machine zero and ``X'X`` becomes singular. The
    # right long-term fix is to swap the triangle-wave kernel for the
    # periodic Green's function of ``(d²/dx²)²`` (Bernoulli ``B_4``),
    # which has every harmonic with ``1/n^4`` decay; until then,
    # odd-K uniform lattices avoid the degeneracy.
    centers = torch.linspace(0.0, period, 16, dtype=torch.float64)  # → 15 effective
    fit = gt.gaussian_reml_fit_positions(
        t,
        y,
        basis="duchon",
        knots_or_centers=centers,
        periodic=True,
        period=period,
    )
    assert torch.isfinite(fit.reml_score)
    # Seam continuity: the fitted curve must agree at points symmetric about
    # the wrap. Evaluate near 0 and near period and compare.
    frozen = fit.freeze()
    eps = 1.0e-3
    seam = frozen.evaluate(torch.tensor([eps, period - eps], dtype=torch.float64))
    seam_gap = float(abs(seam[0] - seam[1]))
    assert seam_gap < 0.1, f"periodic Duchon seam gap too large: {seam_gap:.3e}"

    # Signal recovery: the fitted curve at the training positions should track
    # the noiseless ``sin(2π t / period)`` reasonably well after REML smoothing.
    fitted = fit.fitted.detach().numpy().reshape(-1)
    truth = np.sin(2.0 * np.pi * t.numpy() / period)
    # 1-σ noise was 0.05; fitted should be much closer than that on average.
    rmse = float(np.sqrt(np.mean((fitted - truth) ** 2)))
    assert rmse < 0.05


def test_periodic_duchon_rejects_centers_not_matching_period() -> None:
    _require_ffi("duchon_function_norm_penalty")
    period = 2 * math.pi
    t, y = _circular_sample(period=period)
    # Centers that do NOT span the full period — the validator must reject.
    bad_centers = torch.linspace(0.0, period * 0.8, 12, dtype=torch.float64)
    with pytest.raises(Exception):
        gt.gaussian_reml_fit_positions(
            t,
            y,
            basis="duchon",
            knots_or_centers=bad_centers,
            periodic=True,
            period=period,
        )


# ---------------------------------------------------------------------------
# Determinism + smoke
# ---------------------------------------------------------------------------


def test_fit_is_deterministic() -> None:
    """Two identical calls produce bit-identical outputs."""
    t, y = _sample()
    fit_a = gt.gaussian_reml_fit_positions(t, y, basis="duchon")
    fit_b = gt.gaussian_reml_fit_positions(t, y, basis="duchon")
    np.testing.assert_array_equal(
        fit_a.coefficients.detach().numpy(),
        fit_b.coefficients.detach().numpy(),
    )
    np.testing.assert_array_equal(
        fit_a.penalty.detach().numpy(),
        fit_b.penalty.detach().numpy(),
    )
    assert float(fit_a.reml_score) == float(fit_b.reml_score)
