"""Hard failing capacity / K-selection test for the SAE-manifold engine.

For data with a single circular harmonic mixed into many output dims,
a 1-atom periodic fit is the truth: any K > 1 is wasted capacity and
should be penalised by the fitted criterion. The FFI exposes the certified
``penalized_laml_criterion`` (lower is better), so the K=1 model must have a
strictly lower value than K in {2, 4, 8} on this 1-harmonic dataset.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")


def _circle_data(n: int, p: int, noise: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * math.pi, n)
    harm = np.column_stack([np.cos(theta), np.sin(theta)])
    mixing = rng.normal(size=(harm.shape[1], p))
    mixing /= np.maximum(np.linalg.norm(mixing, axis=0, keepdims=True), 1e-8)
    z = harm @ mixing + noise * rng.normal(size=(n, p))
    z -= z.mean(axis=0, keepdims=True)
    return z


def _criterion(fit) -> float:
    value = float(fit.penalized_laml_criterion)
    assert np.isfinite(value)
    return value


@pytest.mark.slow
def test_penalized_laml_picks_k1_on_one_harmonic_data():
    z = _circle_data(n=400, p=64, noise=0.04, seed=0)
    candidates = [1, 2, 4, 8]
    scores: dict[int, float] = {}
    for k in candidates:
        fit = gamfit.sae_manifold_fit(
            X=z,
            K=k,
            atom_basis="periodic",
            d_atom=2,
            assignment="ordered_beta_bernoulli",
            n_iter=50,
            learning_rate=0.04,
            random_state=0,
        )
        scores[k] = _criterion(fit)

    best_k = min(scores, key=scores.get)
    assert best_k == 1, (
        f"penalized LAML (lower is better) failed to pick K=1 on "
        f"1-harmonic circle data; per-K criterion = "
        f"{ {k: round(v, 6) for k, v in scores.items()} }, winner K={best_k}."
    )
    for k in candidates:
        if k == 1:
            continue
        assert scores[1] < scores[k], (
            f"penalized LAML at K=1 ({scores[1]:.6f}) did not remain strictly "
            f"below K={k} ({scores[k]:.6f}); full sweep = "
            f"{ {kk: round(vv, 6) for kk, vv in scores.items()} }."
        )
