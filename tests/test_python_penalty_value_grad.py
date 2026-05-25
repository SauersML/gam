"""Smoke tests for `.value_grad(t)` on analytic penalty descriptor classes.

Each descriptor under test exposes a `value_grad(t)` method that returns
`(value, grad)` where `grad` has the same shape as `t`. The tests evaluate
each penalty at a random point and check the analytic gradient against a
central finite-difference baseline at two coordinates.
"""

from __future__ import annotations

import numpy as np

from gamfit._penalties import (
    AuxConditionalPriorPenalty,
    IvaeRidgeMeanGauge,
    MechanismSparsityPenalty,
    ParametricAuxConditionalPriorPenalty,
)


def _finite_diff_grad_at(
    fn, t: np.ndarray, indices: list[tuple[int, ...]], h: float = 1.0e-5
) -> np.ndarray:
    out = np.zeros(len(indices))
    for idx_i, idx in enumerate(indices):
        t_plus = t.copy()
        t_minus = t.copy()
        t_plus[idx] += h
        t_minus[idx] -= h
        v_plus, _ = fn(t_plus)
        v_minus, _ = fn(t_minus)
        out[idx_i] = (v_plus - v_minus) / (2.0 * h)
    return out


def _check_grad(fn, t: np.ndarray, sample_indices: list[tuple[int, ...]]) -> None:
    value, grad = fn(t)
    assert np.isfinite(value), f"value is not finite: {value}"
    assert grad.shape == t.shape, f"grad shape {grad.shape} != t shape {t.shape}"
    assert np.all(np.isfinite(grad)), "grad has non-finite entries"
    fd = _finite_diff_grad_at(fn, t, sample_indices)
    analytic = np.array([grad[idx] for idx in sample_indices])
    rel = np.abs(analytic - fd) / np.maximum(np.abs(fd), 1.0e-6)
    assert np.all(rel < 1.0e-4), f"gradient mismatch: analytic={analytic}, fd={fd}, rel={rel}"


def test_mechanism_sparsity_value_grad() -> None:
    rng = np.random.default_rng(0)
    d_latent, p_features = 3, 4
    feature_groups = [[0, 1], [2, 3]]
    pen = MechanismSparsityPenalty(
        feature_groups=feature_groups,
        weight=0.7,
        n_eff=50.0,
        smoothing_eps=1.0e-6,
    )
    w = rng.standard_normal((d_latent, p_features))
    _check_grad(pen.value_grad, w, [(0, 0), (1, 2), (2, 3)])


def test_aux_conditional_prior_value_grad() -> None:
    rng = np.random.default_rng(1)
    n_eff = 6
    d = 3
    # Build symmetric positive-definite per-row precisions.
    lambdas = np.zeros((n_eff, d, d))
    for i in range(n_eff):
        a = rng.standard_normal((d, d))
        lambdas[i] = a @ a.T + np.eye(d)
    pen = AuxConditionalPriorPenalty(
        lambda_per_row=lambdas,
        weight=0.5,
        n_eff=n_eff,
    )
    t = rng.standard_normal((n_eff, d))
    _check_grad(pen.value_grad, t, [(0, 0), (3, 1), (5, 2)])


def test_ivae_ridge_mean_gauge_value_grad() -> None:
    rng = np.random.default_rng(2)
    n_eff = 8
    d = 3
    q = 2
    aux = rng.standard_normal((n_eff, q))
    pen = IvaeRidgeMeanGauge(
        aux=aux,
        weight=0.4,
        n_eff=n_eff,
        ridge_eps=1.0e-6,
    )
    t = rng.standard_normal((n_eff, d))
    _check_grad(pen.value_grad, t, [(0, 0), (4, 1), (7, 2)])


def test_parametric_aux_conditional_prior_value_grad() -> None:
    rng = np.random.default_rng(3)
    n_eff = 7
    d = 3
    q = 2
    aux = rng.standard_normal((n_eff, q))
    alpha = np.abs(rng.standard_normal(d)) + 0.1
    beta = np.abs(rng.standard_normal(d)) + 0.1
    mu = rng.standard_normal((d, q))
    pen = ParametricAuxConditionalPriorPenalty(
        aux=aux,
        alpha_init=alpha,
        beta_init=beta,
        mu_init=mu,
        weight=0.6,
        n_eff=n_eff,
    )
    t = rng.standard_normal((n_eff, d))
    _check_grad(pen.value_grad, t, [(0, 0), (3, 1), (6, 2)])
