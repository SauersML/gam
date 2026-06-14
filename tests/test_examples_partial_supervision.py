"""Smoke test for gamfit.examples.partial_supervision.

Constructs a synthetic 200x6 latent that contains the auxiliary signal
in the first three columns plus noise, plus three free noise columns
already partially correlated with the supervised block. After running
the procrustes + orthogonal_to_sup example we expect:

* ``T_supervised`` has high column-wise Pearson correlation with the
  aux signal (> 0.6 by design — the noise floor is set conservatively).
* ``T_free @ T_supervised.T`` (per-sample inner products) has small
  Frobenius norm relative to the unconstrained baseline, because the
  orthogonal_to_sup constraint should remove the leakage.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamfit


def _column_correlations(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Per-column Pearson correlations between A[:, k] and B[:, k]."""
    Ac = A - A.mean(axis=0, keepdims=True)
    Bc = B - B.mean(axis=0, keepdims=True)
    num = (Ac * Bc).sum(axis=0)
    den = np.sqrt((Ac * Ac).sum(axis=0) * (Bc * Bc).sum(axis=0))
    return num / np.clip(den, 1e-300, None)


def _build_synthetic(rng: np.random.Generator, n: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    aux = rng.standard_normal((n, 3))
    # Supervised slice: aux rotated by a known orthogonal matrix plus mild noise.
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    sup = aux @ Q.T + 0.10 * rng.standard_normal((n, 3))
    # Free slice: noise plus deliberate leakage from the supervised block.
    free = 0.4 * sup + rng.standard_normal((n, 3))
    T_init = np.concatenate([sup, free], axis=1)
    # X is a wider predictor we never actually use beyond shape checks.
    X = T_init + 0.05 * rng.standard_normal((n, 6))
    return X, aux, T_init


def test_partial_supervision_procrustes_smoke() -> None:
    rng = np.random.default_rng(20260525)
    X, aux, T_init = _build_synthetic(rng)

    example = gamfit.examples.partial_supervision(
        T_dim=6,
        aux=aux,
        d_supervised=3,
        d_free=3,
        sup_method="procrustes",
        free_constraint="orthogonal_to_sup",
    )
    fit = example.fit(X, T_init=T_init)

    assert fit.T_supervised.shape == (200, 3)
    assert fit.T_free.shape == (200, 3)

    # Column-wise correlation with the aux signal (one per sup column).
    corr = _column_correlations(fit.T_supervised, aux)
    assert (corr > 0.6).all(), (
        f"expected all sup columns to correlate > 0.6 with aux; got {corr}"
    )

    # Decorrelation: after the QR projection, the column inner-products
    # between T_free and T_supervised should be (numerically) zero.
    cross = fit.T_free.T @ fit.T_supervised
    cross_frob = float(np.linalg.norm(cross))
    sup_frob = float(np.linalg.norm(fit.T_supervised))
    free_frob = float(np.linalg.norm(fit.T_free))
    rel = cross_frob / max(sup_frob * free_frob, 1e-300)
    assert rel < 1e-8, (
        f"orthogonal_to_sup should null the cross Gram; rel={rel}"
    )

    # Alignment score: with ~10% noise + Procrustes the score lands well > 0.5.
    assert fit.alignment_score > 0.5
    assert fit.sup_method == "procrustes"
    assert fit.map_R is not None and fit.map_R.shape == (3, 3)


def test_partial_supervision_anchor_and_softl2_smoke() -> None:
    rng = np.random.default_rng(7)
    X, aux, T_init = _build_synthetic(rng)

    anchor = gamfit.examples.partial_supervision(
        T_dim=6, aux=aux, d_supervised=3, d_free=3,
        sup_method="anchor", free_constraint="orthogonal_to_sup",
        anchor_idx=list(range(20)),
    ).fit(X, T_init=T_init)
    assert anchor.T_supervised.shape == (200, 3)
    assert anchor.map_A is not None and anchor.map_A.shape == (3, 3)
    assert anchor.map_b is not None and anchor.map_b.shape == (3,)

    soft = gamfit.examples.partial_supervision(
        T_dim=6, aux=aux, d_supervised=3, d_free=3,
        sup_method="soft_l2", free_constraint=None,
    ).fit(X, T_init=T_init)
    assert soft.selected_weight is not None and soft.selected_weight > 0.0
    assert soft.map_A is not None and soft.map_A.shape == (3, 3)


def test_partial_supervision_shape_guards() -> None:
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="d_supervised \\+ d_free"):
        gamfit.examples.partial_supervision(
            T_dim=6, aux=rng.standard_normal((10, 3)),
            d_supervised=3, d_free=2,
        )
    with pytest.raises(ValueError, match="aux must have d_supervised"):
        gamfit.examples.partial_supervision(
            T_dim=6, aux=rng.standard_normal((10, 2)),
            d_supervised=3, d_free=3,
        )
