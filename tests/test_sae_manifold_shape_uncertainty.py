"""Posterior shape-uncertainty contract for the manifold SAE.

The fit exposes, per atom, the phi-scaled posterior covariance of the decoder
coefficients ``Cov(beta_k) = phi * S_beta^-1[block]`` and its closed-form
push-forward to an ambient band (``shape_band``). These tests pin the contract:
the covariance is a real SPD-ish posterior, the band sd matches the analytic
``Phi Cov Phi^T`` propagation, it scales with the dispersion, and it is a
*different, tighter* quantity than the per-observation data deviation.
"""
from __future__ import annotations

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")


def test_shape_uncertainty_periodic_payload_decodes_ring_not_chord():
    from gamfit._sae_manifold import ManifoldSAE

    n = 250
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    x = np.column_stack([np.cos(2.0 * np.pi * t), np.sin(2.0 * np.pi * t)])
    decoder = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=float,
    )
    order = np.r_[np.arange(n // 2, n), np.arange(0, n // 2)]
    band_coords = t[order, None]
    chord = np.column_stack([
        np.linspace(-0.1, 0.1, n),
        np.linspace(1.0, -1.0, n),
    ])
    payload = {
        "atom_plans": [
            {
                "kind": "periodic",
                "latent_dim": 1,
                "basis_size": 3,
                "n_harmonics": 1,
                "duchon_centers": None,
            }
        ],
        "atoms": [
            {
                "basis_kind": "periodic",
                "decoder_B": decoder,
                "assignments_z": np.ones(n),
                "on_atom_coords_t": t[:, None],
                "active_dim": 1,
                "decoder_covariance": np.eye(decoder.size) * 1.0e-4,
                "shape_band_coords": band_coords,
                "shape_band_mean": chord,
                "shape_band_sd": np.full_like(chord, 0.01),
            }
        ],
        "assignments_z": np.ones((n, 1)),
        "logits": np.ones((n, 1)),
        "fitted": x,
        "penalized_quasi_laplace_criterion": 0.0,
        # #1512: the SAE facade now reads a penalized-loss score from the fit
        # payload (penalized_loss_score / oos_penalized_loss); the orphaned fake
        # predates it and tripped KeyError. Mirror the fitted criterion here.
        "penalized_loss_score": 0.0,
        "chosen_k": 1,
        "dispersion": 1.0e-4,
        "oos_projection_top1": False,
        "diagnostics": {
            "atom_trust": np.ones(1),
            "atoms": [
                {
                    "trust_score": 1.0,
                    "sigma_min_tangent": 1.0,
                    "sigma_max_tangent": 1.0,
                    "tangent_condition_score": 1.0,
                    "coverage": 1.0,
                    "activation_frequency": 1.0,
                    "untyped": False,
                    "active_token_count": n,
                }
            ],
        },
    }
    fit = ManifoldSAE.from_payload(
        x,
        payload,
        "circle",
        "softmax",
        [],
        tau=0.5,
    )

    band = fit.shape_uncertainty(atom=0)
    mean = np.asarray(band["mean"], dtype=float)
    radii = np.linalg.norm(mean, axis=1)
    assert np.all((radii >= 0.7) & (radii <= 1.3))
    singular_values = np.linalg.svd(mean - mean.mean(axis=0), compute_uv=False)
    assert singular_values[1] / singular_values[0] > 0.25


def _fit_circle(n=400, noise=0.18, seed=0, n_iter=40):
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


def test_shape_uncertainty_fields_present_and_well_shaped():
    fit, x = _fit_circle()
    p = x.shape[1]
    atom = fit.atoms[0]
    cov = atom.decoder_covariance
    assert cov is not None, "fresh fit must expose decoder_covariance"
    m = atom.decoder_coefficients.shape[0]  # basis size M_k
    assert cov.shape == (m * p, m * p)
    # phi-scaled posterior covariance: symmetric with non-negative diagonal.
    np.testing.assert_allclose(cov, cov.T, atol=1e-8)
    assert np.all(np.diag(cov) >= -1e-12)

    band = fit.shape_uncertainty(0)
    g = band["mean"].shape[0]
    assert band["coords"].shape == (g, 1)
    assert band["mean"].shape == (g, p)
    assert band["sd"].shape == (g, p)
    assert np.all(band["sd"] >= 0.0)
    np.testing.assert_allclose(band["lower"], band["mean"] - 1.96 * band["sd"])
    np.testing.assert_allclose(band["upper"], band["mean"] + 1.96 * band["sd"])
    assert float(fit.dispersion) > 0.0


def test_band_sd_matches_analytic_phi_cov_phi_propagation():
    """band_sd[g, c]^2 must equal Sum_{b1,b2} Phi[b1] Phi[b2] Cov[(b1,c),(b2,c)]."""
    from gamfit._binding import rust_module

    fit, x = _fit_circle(n=300, seed=1)
    p = x.shape[1]
    atom = fit.atoms[0]
    cov = atom.decoder_covariance
    m = atom.decoder_coefficients.shape[0]
    band = fit.shape_uncertainty(0)
    coords = band["coords"]  # (G, 1)

    # Recompute Phi at the band coordinates via the engine's own periodic basis.
    phi, _jet, _pen = rust_module().basis_with_jet(
        "periodic", np.ascontiguousarray(coords.reshape(-1, 1)), {"n_harmonics": 1}
    )
    phi = np.asarray(phi, dtype=float)  # (G, m)
    assert phi.shape[1] == m

    g = coords.shape[0]
    for gi in range(0, g, max(1, g // 12)):
        for c in range(p):
            idx = [b * p + c for b in range(m)]
            sub = cov[np.ix_(idx, idx)]
            var = float(phi[gi] @ sub @ phi[gi])
            assert var >= -1e-10
            np.testing.assert_allclose(
                band["sd"][gi, c], np.sqrt(max(var, 0.0)), rtol=1e-6, atol=1e-9,
                err_msg=f"band sd mismatch at coord {gi}, channel {c}",
            )


def test_posterior_shape_band_is_tighter_than_data_deviation():
    """Epistemic shape uncertainty must be far smaller than the per-point data
    scatter: with N points the manifold is pinned tightly even though each
    observation is noisy. They are distinct quantities (issue: uncertainty vs
    typical data deviation)."""
    fit, x = _fit_circle(n=400, noise=0.18, seed=2)
    band = fit.shape_uncertainty(0)
    data_sd = np.sqrt(float(fit.dispersion))
    median_post_sd = float(np.median(band["sd"]))
    assert median_post_sd > 0.0
    # the data deviation should dwarf the shape posterior sd (well-pinned shape).
    assert data_sd > 5.0 * median_post_sd, (
        f"data_sd={data_sd:.4f} not >> posterior shape sd median={median_post_sd:.4f}"
    )


def test_more_data_tightens_the_posterior_band():
    """Increasing the sample size shrinks the posterior shape band (the band is
    an honest 1/sqrt(N)-style posterior, not a fixed cosmetic ribbon)."""
    fit_small, _ = _fit_circle(n=120, noise=0.18, seed=3)
    fit_big, _ = _fit_circle(n=400, noise=0.18, seed=3)
    sd_small = float(np.median(fit_small.shape_uncertainty(0)["sd"]))
    sd_big = float(np.median(fit_big.shape_uncertainty(0)["sd"]))
    assert sd_big < sd_small, (
        f"posterior band did not shrink with more data: "
        f"median sd {sd_small:.4f} (n=120) -> {sd_big:.4f} (n=400)"
    )
