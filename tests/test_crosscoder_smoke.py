"""Smoke test for :class:`gamfit.Crosscoder`.

Fits the Anthropic-2024 shared-encoder / per-layer-decoder SAE on three random
Gaussian "layer" matrices and asserts that the diagnostics methods return
finite, well-shaped outputs.
"""
from __future__ import annotations

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")
pytest.importorskip("torch")


def test_crosscoder_smoke_three_layer_gaussian() -> None:
    rng = np.random.default_rng(0)
    layer_dims = [16, 32, 16]
    n_rows = 200
    X_stack = [rng.normal(size=(n_rows, d)).astype(np.float64) for d in layer_dims]

    cc = gamfit.Crosscoder(
        layer_dims=layer_dims,
        n_atoms=32,
        decoder_weighted_l1=True,
        shared_encoder="mlp[1024]",
        l1_weight=1e-3,
    )
    cc.fit(X_stack, epochs=100, lr=1e-3, seed=0)

    r2 = cc.per_layer_r2()
    assert r2.shape == (3,)
    assert np.all(np.isfinite(r2))

    affinity = cc.atom_layer_affinity()
    assert affinity.shape == (32, 3)
    assert np.all(np.isfinite(affinity))

    harmonic = cc.harmonic_atoms(tol=0.05)
    assert harmonic is not None
    assert harmonic.dtype == np.int64
    # The index set must be a valid subset of atom ids.
    if harmonic.size > 0:
        assert int(harmonic.min()) >= 0
        assert int(harmonic.max()) < 32
