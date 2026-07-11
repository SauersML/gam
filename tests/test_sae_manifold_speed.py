"""Deterministic bound for the periodic atom fit path.

This uses a moderately large synthetic one-harmonic circle dataset. Per #2055
the SPEC forbids wall-clock time budgets/deadlines, so this test does NOT assert
"returns under N seconds" (that is non-deterministic and machine-dependent).
Instead the fit is bounded by DETERMINISTIC WORK -- a fixed ``n_iter`` iteration
cap -- and the test asserts an algorithmic property: the bounded fit reconstructs
the low-noise one-harmonic structure. The synthetic signal has per-feature signal
variance ~0.5 against noise variance 0.05**2 = 0.0025 (SNR ~200), so a correct
rank-2 periodic-atom recovery must clear the R^2 floor by a wide margin; a
regression that breaks the vectorized fitting path fails the floor instead of a
timer.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")

N_SAMPLES = 5000
N_FEATURES = 256
MAX_ITER = 25
# Conservative reconstruction floor. The one-harmonic signal has SNR ~200, so a
# correct rank-2 recovery reaches R^2 ~0.99; 0.5 robustly separates a working fit
# from a broken/degenerate one without depending on exact convergence in MAX_ITER
# iterations.
MIN_RECONSTRUCTION_R2 = 0.5


def _synthetic_one_harmonic(
    n: int = N_SAMPLES,
    p: int = N_FEATURES,
    noise: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * math.pi, n)
    harm = np.column_stack([np.cos(theta), np.sin(theta)])
    mixing = rng.normal(size=(harm.shape[1], p))
    mixing /= np.maximum(np.linalg.norm(mixing, axis=0, keepdims=True), 1e-8)
    z = harm @ mixing + noise * rng.normal(size=(n, p))
    z -= z.mean(axis=0, keepdims=True)
    return z


def _r2(x: np.ndarray, fitted: np.ndarray) -> float:
    ss_res = float(np.sum((x - fitted) ** 2))
    ss_tot = float(np.sum((x - x.mean(0, keepdims=True)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def test_periodic_atom_fit_recovers_one_harmonic_within_iteration_cap():
    z = _synthetic_one_harmonic()

    # Work bound: the fit runs at most MAX_ITER outer iterations (deterministic),
    # no wall-clock budget (#2055).
    fit = gamfit.sae_manifold_fit(
        X=z,
        K=4,
        atom_basis="periodic",
        d_atom=2,
        assignment="ordered_beta_bernoulli",
        n_iter=MAX_ITER,
        learning_rate=0.04,
        random_state=0,
    )

    assert fit.fitted.shape == (N_SAMPLES, N_FEATURES)
    assert np.all(np.isfinite(fit.fitted)), "fitted reconstruction contains non-finite values"
    score = _r2(z, fit.fitted)
    assert score >= MIN_RECONSTRUCTION_R2, (
        f"periodic atom fit reconstruction R^2={score:.4f} for "
        f"{N_SAMPLES}x{N_FEATURES}, K=4, n_iter={MAX_ITER}; "
        f"floor is {MIN_RECONSTRUCTION_R2:.2f} (one-harmonic SNR ~200 should reach ~0.99)"
    )
