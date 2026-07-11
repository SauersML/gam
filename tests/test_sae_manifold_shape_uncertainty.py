"""Native per-atom posterior shape-band contract."""

from __future__ import annotations

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")


def _fit_circle(n: int = 400, noise: float = 0.18, seed: int = 0, n_iter: int = 40):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, 1.0, n)
    clean = np.column_stack([np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)])
    x = clean + noise * rng.standard_normal((n, 2))
    fit = gamfit.sae_manifold_fit(
        X=x,
        K=1,
        d_atom=1,
        atom_topology="circle",
        assignment="softmax",
        isometry_weight=0.0,
        ard_per_atom=False,
        sparsity_weight=0.01,
        smoothness_weight=0.01,
        n_iter=n_iter,
        learning_rate=1.0,
        random_state=seed,
    )
    return fit, x


def _shape_band(atom):
    coords = np.asarray(atom.shape_band_coords, dtype=float)
    mean = np.asarray(atom.shape_band_mean, dtype=float)
    sd = np.asarray(atom.shape_band_sd, dtype=float)
    return coords, mean, sd


def test_shape_uncertainty_fields_present_and_well_shaped():
    fit, x = _fit_circle()
    p = x.shape[1]
    atom = fit.atoms[0]
    cov = np.asarray(atom.decoder_covariance, dtype=float)
    m = atom.decoder_coefficients.shape[0]
    assert cov.shape == (m * p, m * p)
    np.testing.assert_allclose(cov, cov.T, atol=1e-8)
    assert np.all(np.diag(cov) >= -1e-12)

    coords, mean, sd = _shape_band(atom)
    assert coords.ndim == 2 and coords.shape[1] == 1
    assert mean.shape == sd.shape == (coords.shape[0], p)
    assert np.all(sd >= 0.0)
    assert float(fit.dispersion) > 0.0


def test_band_sd_matches_analytic_phi_cov_phi_propagation():
    from gamfit._binding import rust_module

    fit, x = _fit_circle(n=300, seed=1)
    p = x.shape[1]
    atom = fit.atoms[0]
    cov = np.asarray(atom.decoder_covariance, dtype=float)
    m = atom.decoder_coefficients.shape[0]
    coords, _mean, sd = _shape_band(atom)
    phi, _jet, _pen = rust_module().basis_with_jet(
        "periodic", np.ascontiguousarray(coords), {"n_harmonics": 1}
    )
    phi = np.asarray(phi, dtype=float)
    assert phi.shape[1] == m

    for gi in range(0, coords.shape[0], max(1, coords.shape[0] // 12)):
        for channel in range(p):
            indices = [basis * p + channel for basis in range(m)]
            covariance = cov[np.ix_(indices, indices)]
            variance = float(phi[gi] @ covariance @ phi[gi])
            assert variance >= -1e-10
            np.testing.assert_allclose(
                sd[gi, channel],
                np.sqrt(max(variance, 0.0)),
                rtol=1e-6,
                atol=1e-9,
            )


def test_posterior_shape_band_is_tighter_than_data_deviation():
    fit, _x = _fit_circle(n=400, noise=0.18, seed=2)
    _coords, _mean, sd = _shape_band(fit.atoms[0])
    assert np.sqrt(float(fit.dispersion)) > 5.0 * float(np.median(sd))


def test_more_data_tightens_the_posterior_band():
    fit_small, _ = _fit_circle(n=120, noise=0.18, seed=3)
    fit_big, _ = _fit_circle(n=400, noise=0.18, seed=3)
    sd_small = float(np.median(np.asarray(fit_small.atoms[0].shape_band_sd)))
    sd_big = float(np.median(np.asarray(fit_big.atoms[0].shape_band_sd)))
    assert sd_big < sd_small
