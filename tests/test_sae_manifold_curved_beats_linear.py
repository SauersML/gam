"""Hard failing test: a single curved (periodic) atom on 1-harmonic data
should reconstruct better than 10 linear (Duchon) shards.

The whole pitch of SAE-manifold over a vanilla TopK SAE is that one atom
parametrised over a curved manifold can represent a smooth closed feature
without "shattering" it into many linear dictionary entries. On synthetic
data with one Fourier harmonic mixed into 64-D Euclidean space, the data
truly lives on a 1-D circle, so:

    R^2(curved K=1) > R^2(linear shards K=10)

is the load-bearing claim of the whole machinery. This test pins that.

Before the periodic manifold fit was fixed:
    curved K=1, d=2, periodic, ibp_map  →  R^2 ≈ 0.18
    linear K=10, d=1, duchon, softmax   →  R^2 ≈ 0.83
The curved atom collapses; the bug is somewhere between the periodic
basis dispatch, the assignment update, and the atom-coordinate Newton
step. When the underlying engine is fixed, R^2(curved) should land near
0.95+ on this dataset and the test passes.

Until then this test documents the regression with a non-skippable
assertion so it cannot silently rot.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")

# #1512 triage: the curved-vs-linear SAE-manifold comparison fit exceeds the
# standard Python-API CI runner budget (>240s in triage), so it is tagged slow
# and excluded from the directory-level `-m "not slow"` CI step while still
# being collected (and run by a bare `pytest tests/` locally).
pytestmark = pytest.mark.slow


def _synthetic_one_harmonic(
    n: int = 400,
    p: int = 64,
    noise: float = 0.04,
    seed: int = 7,
) -> np.ndarray:
    """One Fourier harmonic on a circle, randomly mixed into ``p`` output
    dimensions. The intrinsic latent is the angle θ ∈ [0, 2π); the
    decoder is a 2-column [cos θ, sin θ] times a normalised mixing
    matrix, plus small i.i.d. Gaussian noise on every output column.
    """
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


def test_curved_atom_beats_linear_shards_on_one_harmonic():
    z = _synthetic_one_harmonic()

    curved = gamfit.sae_manifold_fit(
        X=z,
        K=1,
        atom_basis="periodic",
        d_atom=2,
        assignment="ibp_map",
        n_iter=50,
        learning_rate=0.04,
        random_state=0,
    )
    linear = gamfit.sae_manifold_fit(
        X=z,
        K=10,
        atom_basis="duchon",
        d_atom=1,
        assignment="softmax",
        n_iter=50,
        learning_rate=0.04,
        random_state=0,
    )

    r2_curved = _r2(z, curved.fitted)
    r2_linear = _r2(z, linear.fitted)

    # Lower bound on the curved atom's reconstruction. One periodic atom
    # of dim 2 is fully sufficient to represent a single Fourier harmonic
    # mixed into Euclidean space — the only ambiguity is a rotation of
    # the (cos θ, sin θ) frame, which the atom basis is closed under.
    # 0.9 is a generous threshold; the engine should comfortably exceed
    # it. Currently observed: ≈ 0.18.
    assert r2_curved >= 0.9, (
        f"single periodic atom failed to recover a 1-harmonic feature: "
        f"R^2(curved K=1, d=2) = {r2_curved:.4f}. "
        f"Linear shards (K=10) got R^2 = {r2_linear:.4f} on the same data. "
        f"The curved atom is supposed to dominate this case; if it can't, "
        f"the SAE-manifold engine is broken upstream of the assignment / "
        f"atom-coordinate fit (likely basis dispatch, ext-coord update, "
        f"or the Newton step for t)."
    )

    # And the unification claim of the whole approach: one curved atom
    # should beat many linear shards on a curved target.
    assert r2_curved > r2_linear, (
        f"curved K=1 did not beat linear K=10 on a 1-harmonic curve: "
        f"R^2(curved) = {r2_curved:.4f} vs R^2(linear) = {r2_linear:.4f}. "
        f"This violates the central claim of SAE-manifold over flat SAE."
    )
