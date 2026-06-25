"""#1026 collapsed linear lane: held-out reconstruction EV of the sparse,
minibatched dictionary trainer must match-or-beat a linear (PCA) baseline at a
modest K, and the routing must stay fixed-width sparse (never dense N x K)."""

from __future__ import annotations

import numpy as np
import pytest

from gamfit import sparse_dictionary_fit


def _planted(rng, k, p, n, second_share=0.2):
    """n rows, each a scaled single planted atom plus a small second atom."""
    # Orthonormal planted atoms (rows) via QR of a random matrix.
    a = rng.standard_normal((p, p)).astype(np.float64)
    q, _ = np.linalg.qr(a)
    atoms = q[:k].astype(np.float32)  # k x p, orthonormal rows
    x = np.zeros((n, p), dtype=np.float32)
    for row in range(n):
        primary = row % k
        secondary = (primary + 1) % k
        scale = np.float32(0.7 + 0.01 * (row // k))
        x[row] = scale * atoms[primary] + second_share * scale * atoms[secondary]
    return x, atoms


def _ev(x, recon):
    rss = float(np.sum((x - recon) ** 2))
    tss = float(np.sum((x - x.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - rss / tss if tss > 0 else (1.0 if rss == 0 else 0.0)


def _pca_held_out_ev(train, test, rank):
    """Rank-`rank` PCA fit on train, evaluated (held-out reconstruction) on test."""
    mean = train.mean(axis=0, keepdims=True)
    tc = train - mean
    _, _, vt = np.linalg.svd(tc, full_matrices=False)
    basis = vt[:rank]  # rank x p
    proj = (test - mean) @ basis.T @ basis + mean
    return _ev(test, proj)


def test_sparse_trainer_held_out_ev_beats_pca_baseline_modest_k():
    rng = np.random.default_rng(0)
    k, p = 16, 24
    train, _ = _planted(rng, k, p, n=640)
    test, _ = _planted(rng, k, p, n=200)

    fit = sparse_dictionary_fit(
        train,
        K=k,
        active=2,
        max_epochs=50,
        score_tile=8,
        tolerance=1e-9,
    )

    # Fixed-width sparse routing, never dense N x K.
    assert fit.indices.shape == (train.shape[0], 2)
    assert fit.codes.shape == (train.shape[0], 2)
    assert fit.decoder.shape == (k, p)

    # Held-out: route the test rows through the frozen decoder, reconstruct.
    idx, cod = fit.transform(test, active=2)
    recon = fit.reconstruct(idx, cod)
    held_out_ev = _ev(test, recon)

    baseline = _pca_held_out_ev(train, test, rank=k)

    assert held_out_ev > 0.9, f"held-out EV too low: {held_out_ev}"
    assert held_out_ev + 1e-4 >= baseline, (
        f"sparse trainer held-out EV {held_out_ev} must match-or-beat "
        f"rank-{k} PCA baseline {baseline}"
    )


def test_sparse_trainer_storage_is_fixed_width_not_dense():
    rng = np.random.default_rng(1)
    k, p = 1000, 12  # K >> rank: dense N x K would be wasteful
    train, _ = _planted(rng, 8, p, n=200, second_share=0.1)
    fit = sparse_dictionary_fit(train, K=k, active=1, max_epochs=6, score_tile=128)
    # Routing width is `active`, not K.
    assert fit.indices.shape == (200, 1)
    assert fit.codes.shape == (200, 1)
    assert fit.explained_variance > 0.85


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
