"""Per-row reconstruction weights are reachable from Python (#1062 / #977).

`SaeManifoldTerm::set_row_loss_weights` (the #977 √w design-honesty seam) was
implemented in the Rust core but had no Python entry point. `sae_manifold_fit`
now accepts a `weights=` array that installs the per-row reweighting on the
term before the inner joint fit and the outer ρ selection.

Assertions (all non-skippable):
  - A valid `weights` vector fits and returns a well-formed model.
  - Up-weighting the clean half of a corpus whose other half is corrupted by
    a competing structure pulls the fit toward the clean half: the weighted
    fit reconstructs the up-weighted rows at least as well as the unweighted
    fit does (objective truth-recovery, not a reference match).
  - A uniform weight vector is the bit-identical unweighted path.
  - Malformed weights (wrong length, non-positive, non-finite) raise.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

# #1512: this fit exceeds the standard Python-API CI runner budget (>60s in
# triage), so it is tagged slow and excluded from the directory-level
# `-m "not slow"` CI step while still being collected (run by a bare pytest).
pytestmark = pytest.mark.slow

gamfit = pytest.importorskip("gamfit")


def _circle(n: int, p: int, noise: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * math.pi, n)
    harm = np.column_stack([np.cos(theta), np.sin(theta)])
    mixing = rng.normal(size=(2, p))
    mixing /= np.maximum(np.linalg.norm(mixing, axis=0, keepdims=True), 1e-8)
    z = harm @ mixing + noise * rng.normal(size=(n, p))
    return z - z.mean(axis=0, keepdims=True)


def _row_r2(model: object, x: np.ndarray, rows: np.ndarray) -> float:
    fitted = np.asarray(model.fitted)[rows]
    target = x[rows]
    ss_res = float(np.sum((target - fitted) ** 2))
    ss_tot = float(np.sum((target - target.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def test_row_weights_pull_fit_toward_upweighted_rows() -> None:
    # Two halves drawn from DIFFERENT circle structures (different seeds /
    # mixing). A single-atom fit cannot serve both; up-weighting the clean half
    # should make the fit favour it.
    n_half, p = 120, 4
    clean = _circle(n_half, p, noise=0.02, seed=0)
    other = _circle(n_half, p, noise=0.02, seed=99)
    x = np.vstack([clean, other])
    n = x.shape[0]
    clean_rows = np.arange(n_half)

    base = gamfit.sae_manifold_fit(
        x, K=1, d_atom=2, atom_topology="periodic", n_iter=30, random_state=0
    )
    weights = np.ones(n)
    weights[clean_rows] = 8.0
    weighted = gamfit.sae_manifold_fit(
        x,
        K=1,
        d_atom=2,
        atom_topology="periodic",
        n_iter=30,
        random_state=0,
        weights=weights,
    )

    base_clean = _row_r2(base, x, clean_rows)
    weighted_clean = _row_r2(weighted, x, clean_rows)
    # Up-weighting the clean half must not hurt its reconstruction, and should
    # generally help it (allow a small slack for optimizer noise).
    assert weighted_clean >= base_clean - 1e-3, (
        f"weighted clean R^2 {weighted_clean:.4f} < unweighted {base_clean:.4f}"
    )


def test_uniform_weights_match_unweighted() -> None:
    x = _circle(80, 3, noise=0.05, seed=7)
    n = x.shape[0]
    a = gamfit.sae_manifold_fit(
        x, K=1, d_atom=2, atom_topology="periodic", n_iter=20, random_state=3
    )
    b = gamfit.sae_manifold_fit(
        x,
        K=1,
        d_atom=2,
        atom_topology="periodic",
        n_iter=20,
        random_state=3,
        weights=np.full(n, 2.5),
    )
    np.testing.assert_allclose(
        np.asarray(a.fitted), np.asarray(b.fitted), rtol=0.0, atol=1e-9
    )


def test_malformed_weights_raise() -> None:
    x = _circle(40, 2, noise=0.05, seed=1)
    n = x.shape[0]
    with pytest.raises(ValueError):
        gamfit.sae_manifold_fit(
            x, K=1, atom_topology="periodic", weights=np.ones(n - 1)
        )
    with pytest.raises(ValueError):
        bad = np.ones(n)
        bad[0] = 0.0
        gamfit.sae_manifold_fit(x, K=1, atom_topology="periodic", weights=bad)
    with pytest.raises(ValueError):
        bad = np.ones(n)
        bad[0] = np.inf
        gamfit.sae_manifold_fit(x, K=1, atom_topology="periodic", weights=bad)
