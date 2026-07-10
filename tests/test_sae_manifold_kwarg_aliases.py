"""Regression tests for the SAE atom-count kwargs (issue #160).

``K=`` and ``n_atoms=`` are aliases for the number of atoms; supplying
both with DIFFERENT values raises an eager ``ValueError``; equal values pass
through.
"""
from __future__ import annotations

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")

def _tiny_x() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((8, 6))


def test_k_and_n_atoms_conflict_raises():
    with pytest.raises(ValueError) as exc:
        gamfit.sae_manifold_fit(X=_tiny_x(), K=2, n_atoms=3)
    msg = str(exc.value)
    assert "K and n_atoms both supplied with different values" in msg
    assert "2 vs 3" in msg
