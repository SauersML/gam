"""Hard wall-clock test for the periodic atom fit path.

This uses a moderately large synthetic one-harmonic circle dataset and
keeps the threshold fixed at 120 seconds. The value is intentionally loose:
it should tolerate ordinary 2-3x CPU variance in shared CI while still
catching an order-of-magnitude regression in the vectorized fitting path.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")

N_SAMPLES = 5000
N_FEATURES = 256
MAX_ITER = 25
TIME_LIMIT_SECONDS = 120.0


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


def test_periodic_atom_fit_completes_within_wall_clock_limit():
    z = _synthetic_one_harmonic()

    start = time.perf_counter()
    fit = gamfit.sae_manifold_fit(
        Z=z,
        n_atoms=4,
        atom_basis="periodic",
        atom_dim=2,
        assignment_prior="ibp_map",
        max_iter=MAX_ITER,
        learning_rate=0.04,
        random_state=0,
    )
    elapsed = time.perf_counter() - start

    assert fit.fitted.shape == (N_SAMPLES, N_FEATURES)
    score = _r2(z, fit.fitted)
    assert elapsed < TIME_LIMIT_SECONDS, (
        f"periodic atom fit took {elapsed:.2f}s for "
        f"{N_SAMPLES}x{N_FEATURES}, K=4, max_iter={MAX_ITER}; "
        f"limit is {TIME_LIMIT_SECONDS:.2f}s, reconstruction R^2={score:.4f}"
    )
