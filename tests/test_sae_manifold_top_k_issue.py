"""Tests pinning the top_k / target_k contract for SAE manifold fits.

The Rust ``sae_manifold_fit_minimal`` ``#[pyfunction]`` takes ``top_k`` and
owns the hard per-row top-k projection end to end; ``gamfit/_sae_manifold.py``
forwards it unconditionally and ``gamfit.torch.manifold_sae.ManifoldSAE.fit``
honours ``cfg.sparsity.target_k``.

Each path must produce assignments where at most ``top_k`` atoms are active
per row. Silently dropping the parameter is the regression these tests guard.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamfit


def _random_inputs() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((24, 4))


def test_sae_manifold_fit_top_k_one_yields_at_most_one_active_atom() -> None:
    """Public ``gamfit.sae_manifold_fit`` must honour ``top_k=1``."""
    X = _random_inputs()
    fit = gamfit.sae_manifold_fit(
        Z=X,
        n_atoms=3,
        atom_basis="periodic",
        atom_dim=1,
        assignment="softmax",
        top_k=1,
        max_iter=5,
        random_state=0,
    )
    A = np.asarray(fit.assignments)
    active_per_row = (A > 1e-12).sum(axis=1)
    assert int(active_per_row.max()) <= 1, (
        f"top_k=1 should leave at most 1 atom active per row, got max={int(active_per_row.max())}"
    )


@pytest.mark.parametrize("k", [1, 2])
def test_sae_manifold_fit_top_k_general(k: int) -> None:
    X = _random_inputs()
    fit = gamfit.sae_manifold_fit(
        Z=X,
        n_atoms=4,
        atom_basis="periodic",
        atom_dim=1,
        assignment="softmax",
        top_k=k,
        max_iter=5,
        random_state=0,
    )
    A = np.asarray(fit.assignments)
    active_per_row = (A > 1e-12).sum(axis=1)
    assert int(active_per_row.max()) <= k


def test_manifold_sae_module_fit_respects_target_k() -> None:
    """``ManifoldSAE.fit`` must honour ``cfg.sparsity.target_k``."""
    torch = pytest.importorskip("torch")
    gt = pytest.importorskip("gamfit.torch")

    cfg = gt.ManifoldSAEConfig(
        input_dim=6,
        n_atoms=4,
        intrinsic_rank=1,
        atom_manifold="circle",
        atom_basis="fourier",
        sparsity={"kind": "softmax_topk", "target_k": 1},
    )
    sae = gt.ManifoldSAE(cfg).double()
    rng = np.random.default_rng(1)
    X = rng.standard_normal((16, 6))
    fit = sae.fit(torch.tensor(X, dtype=torch.float64), max_iter=3, random_state=0)

    A = np.asarray(fit.assignments)
    active_per_row = (A > 1e-12).sum(axis=1)
    assert int(active_per_row.max()) <= 1, (
        f"sparsity.target_k=1 should leave at most 1 atom active in closed-form fit; got "
        f"max={int(active_per_row.max())}"
    )
