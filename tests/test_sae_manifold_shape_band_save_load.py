"""Posterior shape bands must SURVIVE save/load (Objective 2).

The closed-form joint-Laplace shape bands are the engine's flagship honesty
feature. This pins the contract that they round-trip through
:meth:`ManifoldSAE.save` / :meth:`ManifoldSAE.load`:

* the band (``coords`` / ``mean`` / ``sd``) evaluated after a save/load matches
  the band before it, to tight numerical tolerance;
* the per-atom decoder covariance restored on load reproduces the analytic
  ``Σ_{b1,b2} Φ[b1]Φ[b2] Cov[(b1,c),(b2,c)]`` band variance the fresh fit
  produced, to tight tolerance;
* the ON-DISK format stores a COMPACT per-atom, per-channel covariance factor
  (``(p, M_k, M_k)`` same-channel Schur blocks) — NOT the dense ``(M_k·p)²``
  joint covariance the band never reads cross-channel entries of.

These tests are authored under the campaign rule "write the test, do NOT run it";
they are HQ-verified centrally.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")


def _fit_circle(n: int = 400, noise: float = 0.18, seed: int = 0, n_iter: int = 40):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, 1.0, n)
    clean = np.column_stack([np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)])
    x = clean + noise * rng.standard_normal((n, 2))
    fit = gamfit.sae_manifold_fit(
        X=x, K=1, d_atom=1, atom_topology="circle", assignment="softmax",
        isometry_weight=0.0, ard_per_atom=False, sparsity_weight=0.01,
        smoothness_weight=0.01, n_iter=n_iter, learning_rate=1.0, random_state=seed,
    )
    return fit, x


def test_shape_band_survives_save_load_to_tight_tolerance(tmp_path: Path):
    fit, _x = _fit_circle(seed=7)
    atom = fit.atoms[0]
    assert atom.decoder_covariance is not None, "fresh fit must expose decoder_covariance"

    before = {
        "coords": np.asarray(atom.shape_band_coords),
        "mean": np.asarray(atom.shape_band_mean),
        "sd": np.asarray(atom.shape_band_sd),
    }
    decoder_before = np.asarray(atom.decoder_coefficients, dtype=float).copy()
    coords_before = np.asarray(atom.coords, dtype=float).copy()
    rendered_before = np.asarray(fit.reconstruct_training(), dtype=float).copy()
    topology_before = (fit.atom_topology, tuple(fit.atom_topologies))

    path = tmp_path / "sae_band.json"
    fit.save(path)
    restored = gamfit.ManifoldSAE.load(path)
    restored_atom = restored.atoms[0]
    after = {
        "coords": np.asarray(restored_atom.shape_band_coords),
        "mean": np.asarray(restored_atom.shape_band_mean),
        "sd": np.asarray(restored_atom.shape_band_sd),
    }

    np.testing.assert_array_equal(
        np.asarray(restored.atoms[0].decoder_coefficients), decoder_before
    )
    np.testing.assert_array_equal(np.asarray(restored.atoms[0].coords), coords_before)
    np.testing.assert_array_equal(
        np.asarray(restored.reconstruct_training()), rendered_before
    )
    assert (restored.atom_topology, tuple(restored.atom_topologies)) == topology_before

    for key in ("coords", "mean", "sd"):
        np.testing.assert_allclose(
            after[key], before[key], rtol=1e-10, atol=1e-12,
            err_msg=f"shape band '{key}' did not survive save/load",
        )


def test_restored_covariance_reproduces_analytic_band(tmp_path: Path):
    """The reconstructed per-atom covariance must reproduce the same-channel
    ``Φ Cov Φ`` band variance the fresh fit's covariance produced."""
    from gamfit._binding import rust_module

    fit, _x = _fit_circle(n=300, seed=1)
    atom = fit.atoms[0]
    p = atom.decoder_coefficients.shape[1]
    m = atom.decoder_coefficients.shape[0]
    cov_before = np.asarray(atom.decoder_covariance, dtype=float)

    path = tmp_path / "sae_band.json"
    fit.save(path)
    restored = gamfit.ManifoldSAE.load(path)
    cov_after = np.asarray(restored.atoms[0].decoder_covariance, dtype=float)
    assert cov_after.shape == (m * p, m * p)

    restored_atom = restored.atoms[0]
    coords = np.asarray(restored_atom.shape_band_coords)
    band_sd = np.asarray(restored_atom.shape_band_sd)
    phi, _jet, _pen = rust_module().basis_with_jet(
        "periodic", np.ascontiguousarray(coords.reshape(-1, 1)), {"n_harmonics": 1}
    )
    phi = np.asarray(phi, dtype=float)
    assert phi.shape[1] == m

    g = coords.shape[0]
    for gi in range(0, g, max(1, g // 12)):
        for c in range(p):
            idx = [b * p + c for b in range(m)]
            # Same-channel block survives exactly in the reconstructed covariance.
            sub_before = cov_before[np.ix_(idx, idx)]
            sub_after = cov_after[np.ix_(idx, idx)]
            np.testing.assert_allclose(sub_after, sub_before, rtol=1e-10, atol=1e-12)
            var = float(phi[gi] @ sub_after @ phi[gi])
            np.testing.assert_allclose(
                band_sd[gi, c], np.sqrt(max(var, 0.0)), rtol=1e-6, atol=1e-9,
                err_msg=f"restored covariance band mismatch at coord {gi}, channel {c}",
            )


def test_on_disk_format_is_compact_per_channel_not_dense_joint(tmp_path: Path):
    """Storage must be per-atom compact factors, not the dense joint covariance."""
    fit, _x = _fit_circle(seed=3)
    atom = fit.atoms[0]
    p = atom.decoder_coefficients.shape[1]
    m = atom.decoder_coefficients.shape[0]

    path = tmp_path / "sae_band.json"
    fit.save(path)
    payload = json.loads(path.read_text())
    atom_payload = payload["atoms"][0]

    # The dense joint covariance key is GONE; the compact per-channel factor is
    # what is serialized.
    assert "decoder_covariance" not in atom_payload, (
        "dense (M_k·p)² joint covariance must not be serialized"
    )
    factor = atom_payload["decoder_covariance_channel_factors"]
    assert factor is not None, "compact covariance factor must be present"
    blocks = np.asarray(factor, dtype=float)
    # Per-atom compact factor: p blocks of M_k × M_k (the band-consumed Schur
    # blocks), NOT a dense (M_k·p, M_k·p) matrix.
    assert blocks.shape == (p, m, m)
    assert blocks.size == p * m * m
    assert blocks.size < (m * p) * (m * p), "factor must be smaller than the dense joint"


def test_reconstruction_and_band_are_physical_under_heterogeneous_column_scale(
    tmp_path: Path,
):
    """#2271 persistence-surface regression.

    Tier-0 input standardization (#2015) fits on ``(Z - mu) / sigma`` with
    ``sigma`` the per-column centered RMS, so the per-atom decoder / decoder
    covariance / shape bands the fit reports live in that standardized frame.
    If the sigma lift is dropped anywhere on the path from the fitted term to
    the persisted/exposed artifact, a column with a large RMS (real activation
    data measures ratios of ~1e4) comes back mis-scaled by that same ratio --
    silently, since a same-order-of-magnitude column would hide the bug. This
    pins that the reconstruction and the shape band are in the ORIGINAL
    (physical) units, both on the fresh fit and after a save/load round trip.
    """
    rng = np.random.default_rng(11)
    n = 400
    t = rng.uniform(0.0, 1.0, n)
    col_scale = np.array([1.0, 1.0e4])
    clean = np.column_stack([np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)]) * col_scale
    x = clean + 0.02 * col_scale * rng.standard_normal((n, 2))
    fit = gamfit.sae_manifold_fit(
        X=x, K=1, d_atom=1, atom_topology="circle", assignment="softmax",
        isometry_weight=0.0, ard_per_atom=False, sparsity_weight=0.01,
        smoothness_weight=0.01, n_iter=40, learning_rate=1.0, random_state=11,
    )
    recon = np.asarray(fit.reconstruct_training())
    resid_rms = np.sqrt(((recon - x) ** 2).mean(axis=0))
    data_rms = np.sqrt((x ** 2).mean(axis=0))
    rel_resid = resid_rms / data_rms
    assert np.all(rel_resid < 0.5), (
        f"reconstruction is not in physical units (relative residual {rel_resid}); "
        "the high-scale column looks mis-scaled by the Tier-0 sigma ratio"
    )

    atom = fit.atoms[0]
    band_sd = np.asarray(atom.shape_band_sd)
    # The high-scale column's band sd must itself land on the ~1e4 scale, not
    # the ~1 standardized-frame scale a dropped sigma lift would leave behind.
    assert band_sd[:, 1].mean() > 10.0 * band_sd[:, 0].mean(), (
        "shape_band_sd ratio between columns does not track the data's column-scale "
        "ratio -- the band looks like it is still in the standardized frame"
    )

    path = tmp_path / "sae_hetero.json"
    fit.save(path)
    restored = gamfit.ManifoldSAE.load(path)
    np.testing.assert_array_equal(
        np.asarray(restored.atoms[0].decoder_coefficients),
        np.asarray(atom.decoder_coefficients),
    )
    np.testing.assert_allclose(
        np.asarray(restored.reconstruct_training()), recon, rtol=1e-10, atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(restored.atoms[0].shape_band_sd), band_sd, rtol=1e-10, atol=1e-12,
    )
