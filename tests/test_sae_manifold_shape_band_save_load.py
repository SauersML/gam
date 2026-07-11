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

    before = fit.shape_uncertainty(0)
    decoder_before = np.asarray(atom.decoder_coefficients, dtype=float).copy()
    coords_before = np.asarray(atom.coords, dtype=float).copy()
    rendered_before = np.asarray(fit.reconstruct_training(), dtype=float).copy()
    topology_before = (fit.atom_topology, tuple(fit.atom_topologies))

    path = tmp_path / "sae_band.json"
    fit.save(path)
    restored = gamfit.ManifoldSAE.load(path)
    after = restored.shape_uncertainty(0)

    np.testing.assert_array_equal(
        np.asarray(restored.atoms[0].decoder_coefficients), decoder_before
    )
    np.testing.assert_array_equal(np.asarray(restored.atoms[0].coords), coords_before)
    np.testing.assert_array_equal(
        np.asarray(restored.reconstruct_training()), rendered_before
    )
    assert (restored.atom_topology, tuple(restored.atom_topologies)) == topology_before

    for key in ("coords", "mean", "sd", "lower", "upper"):
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

    band = restored.shape_uncertainty(0)
    coords = band["coords"]
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
                band["sd"][gi, c], np.sqrt(max(var, 0.0)), rtol=1e-6, atol=1e-9,
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
