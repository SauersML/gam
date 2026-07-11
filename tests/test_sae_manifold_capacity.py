"""Hard failing capacity / K-selection test for the SAE-manifold engine.

For data with a single circular harmonic mixed into many output dims,
a 1-atom periodic fit is the truth: any K > 1 is wasted capacity and
should be penalised by the model evidence. The FFI exposes
``reml_score = -total_loss`` (larger is better — see
``evidence_proxy`` in the underlying term), so the K=1 model must
strictly outscore K in {2, 4, 8} on this 1-harmonic dataset.
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


def _evidence(fit) -> float:
    # Each atom carries a copy of the payload-wide REML evidence.
    if hasattr(fit, "atoms") and fit.atoms:
        atom = fit.atoms[0]
        if hasattr(atom, "evidence"):
            return float(atom.evidence)
    for attr in ("reml_score", "evidence", "score"):
        if hasattr(fit, attr):
            value = getattr(fit, attr)
            if isinstance(value, (int, float)):
                return float(value)
    raise AssertionError(
        "SAE-manifold fit result exposes no REML evidence attribute "
        "(checked atoms[0].evidence, reml_score, evidence, score); "
        "cannot run capacity / K-selection comparison."
    )


@pytest.mark.slow
def test_reml_evidence_picks_k1_on_one_harmonic_data():
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
        scores[k] = _evidence(fit)

    best_k = max(scores, key=scores.get)
    assert best_k == 1, (
        f"REML evidence (larger is better) failed to pick K=1 on "
        f"1-harmonic circle data; per-K evidence = "
        f"{ {k: round(v, 6) for k, v in scores.items()} }, winner K={best_k}."
    )
    for k in candidates:
        if k == 1:
            continue
        assert scores[1] > scores[k], (
            f"REML evidence at K=1 ({scores[1]:.6f}) did not strictly "
            f"exceed K={k} ({scores[k]:.6f}); full sweep = "
            f"{ {kk: round(vv, 6) for kk, vv in scores.items()} }."
        )
