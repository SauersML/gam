"""#1026 — RED e2e: production manifold-SAE on REAL OLMo-3-32B activations.

This is the real end-to-end task the manifold-SAE machinery exists for:
fit the production SAE (`gamfit.sae_manifold_fit` / `ManifoldSAE`) on a
train split of genuine LLM residual-stream activations, reconstruct a
*held-out* split via the OOS path (`m.reconstruct(z_test)`), and require a
principled, defensible held-out reconstruction-quality bar.

The fixture `tests/data/olmo_l25_pca64_768.npy` is a real-activation slice:
OLMo-3-32B layer-25 residual stream, 768 token positions, mean-centered and
PCA-reduced to 64 dims (the SAE PCA-reduces its input anyway). Spectrum:
PC1=0.253, cum@K8=0.459, cum@K32=0.631, cum@K64=0.737 — a structured
minority sitting on an unstructured bulk, which is exactly the regime the
manifold dictionary is supposed to win in.

WHY THIS IS RED (a real e2e gap, not a flaky/bogus assertion):
  The defensible bar here is *linear parity*: a TopK / linear dictionary of
  K shards trivially attains held-out EV equal to the cumulative PCA
  spectrum at its rank. The whole wager of curved atoms (#1026 recon-parity
  roadmap) is that the manifold SAE matches-or-beats that linear ceiling at
  equal or lower K on real activations. We assert the production SAE clears
  a *modest fraction* of the K8 linear ceiling (cum@K8 = 0.459) on held-out
  real data. On real OLMo activations the curved-atom reconstruct path does
  not currently reach this bar — it under-recovers on the long-tailed real
  spectrum (degenerate-basin under-recovery, the #976/#1117 collapse class
  manifesting on real-vs-synthetic data). This pins the genuine e2e gap:
  the SAE reconstructs clean planted synthetic circles/spheres but does not
  yet earn its EV on real LLM activations.

Closing this requires the manifold dictionary to actually earn held-out EV
on real activations at parity with (or beating) the linear PCA ceiling at
equal K — i.e. the #1026 recon-parity frontier landing in the green on real
data, not just on planted synthetic harmonics.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")

_FIXTURE = Path(__file__).resolve().parent / "data" / "olmo_l25_pca64_768.npy"

# PCA spectrum of the fixture (from README_olmo_fixture.md): the cumulative
# explained-variance fraction at rank K is the held-out EV a rank-K *linear*
# reconstruction attains. This is the linear ceiling the manifold SAE must
# match-or-beat on real data. cum@K8 = 0.459.
_LINEAR_CEILING_K8 = 0.459
# Modest, defensible fraction of the K8 linear ceiling: we do not even demand
# parity, only that the curved SAE earns a clear majority of the linear EV at
# matched rank on held-out real activations.
_TARGET_FRACTION = 0.75
_EV_TARGET = _TARGET_FRACTION * _LINEAR_CEILING_K8  # ~0.344


def _ev(x: np.ndarray, fitted: np.ndarray) -> float:
    """Explained-variance fraction = 1 - SS_res / SS_tot, centered on x's
    own mean (the held-out baseline a constant predictor would attain)."""
    ss_res = float(np.sum((x - fitted) ** 2))
    ss_tot = float(np.sum((x - x.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def _load_fixture() -> np.ndarray:
    assert _FIXTURE.exists(), f"real OLMo fixture missing: {_FIXTURE}"
    z = np.load(_FIXTURE).astype(np.float64)
    assert z.shape == (768, 64), f"unexpected fixture shape {z.shape}"
    assert np.all(np.isfinite(z)), "fixture contains non-finite values"
    return z


def test_olmo_real_heldout_reconstruction_ev_meets_linear_parity():
    """Production manifold-SAE held-out reconstruction EV on real OLMo-3-32B
    activations must clear 75% of the rank-8 linear PCA ceiling (>= ~0.344).

    RED: on real activations the curved-atom reconstruct path under-recovers
    relative to this bar (it clears the synthetic planted-circle bars but not
    the real long-tailed spectrum). This pins the #1026 recon-parity gap on
    REAL data.
    """
    z = _load_fixture()
    # 512 train / 256 held-out test, deterministic split (no shuffle: the
    # fixture rows were already randomly sampled at extraction time).
    z_train = z[:512]
    z_test = z[512:]
    assert z_test.shape[0] == 256

    fit = gamfit.sae_manifold_fit(
        X=z_train,
        K=8,
        atom_basis="periodic",
        d_atom=2,
        assignment="ibp_map",
        n_iter=60,
        learning_rate=0.04,
        random_state=0,
    )

    in_sample_ev = _ev(z_train, fit.fitted)

    assert hasattr(fit, "reconstruct"), (
        "ManifoldSAE fit must expose `reconstruct` for OOS scoring; "
        f"in-sample EV was {in_sample_ev:.4f}."
    )
    oos = fit.reconstruct(z_test)
    assert oos.shape == z_test.shape
    assert np.all(np.isfinite(oos)), "OOS reconstruction produced NaN/Inf"

    oos_ev = _ev(z_test, oos)

    assert oos_ev >= _EV_TARGET, (
        f"Held-out reconstruction EV on real OLMo activations = {oos_ev:.4f}, "
        f"below the bar {_EV_TARGET:.4f} (= {_TARGET_FRACTION:.0%} of the "
        f"rank-8 linear PCA ceiling {_LINEAR_CEILING_K8:.3f}). "
        f"In-sample EV was {in_sample_ev:.4f}. "
        f"The manifold SAE does not yet earn its EV on real LLM activations "
        f"at parity with the linear dictionary — the #1026 recon-parity gap "
        f"on REAL (not synthetic) data."
    )
