from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")


def _two_atom_curve_data(n: int = 120, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    first = np.column_stack(
        [
            np.cos(2.0 * math.pi * t),
            np.sin(2.0 * math.pi * t),
            0.35 * np.cos(4.0 * math.pi * t),
        ]
    )
    second = np.column_stack(
        [
            0.40 * np.sin(4.0 * math.pi * t + 0.3),
            np.cos(2.0 * math.pi * t + 0.6),
            np.sin(2.0 * math.pi * t + 0.6),
        ]
    )
    z = first + second + 0.02 * rng.standard_normal(first.shape)
    return z - z.mean(axis=0, keepdims=True)


def _fresh_two_atom_fit_or_fail(z: np.ndarray):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fit = gamfit.sae_manifold_fit(
                Z=z,
                K=2,
                d_atom=1,
                atom_topology="circle",
                assignment="softmax",
                isometry_weight=0.0,
                ard_per_atom=False,
                sparsity_weight=0.01,
                smoothness_weight=0.01,
                decoder_incoherence_weight=0.0,
                max_iter=25,
                learning_rate=1.0,
                random_state=21,
            )
    except Exception as exc:
        pytest.fail(
            "per-atom uncertainty and coordinate-range public API test is "
            "guarded until the current d=1 multi-atom solver gap converges: "
            f"{type(exc).__name__}: {exc}"
        )
    if (
        not np.all(np.isfinite(fit.fitted))
        or not np.isfinite(fit.reconstruction_r2)
        or fit.reconstruction_r2 < 0.05
    ):
        pytest.fail(
            "per-atom uncertainty and coordinate-range public API test did "
            "not pass the convergence guard: "
            f"reconstruction_r2={fit.reconstruction_r2!r}"
        )
    return fit


def test_per_atom_uncertainty_shape_band_and_coordinate_range_shapes_are_sane():
    z = _two_atom_curve_data()
    fit = _fresh_two_atom_fit_or_fail(z)
    p = z.shape[1]

    assert len(fit.atoms) == 2
    for atom_k, atom in enumerate(fit.atoms):
        d_k = fit.coords[atom_k].shape[1]
        m_k = atom.decoder_coefficients.shape[0]

        if (
            atom.decoder_covariance is None
            or atom.shape_band_coords is None
            or atom.shape_band_mean is None
            or atom.shape_band_sd is None
        ):
            pytest.fail(
                "fresh SAE fit did not populate per-atom posterior "
                f"uncertainty arrays for atom={atom_k}"
            )

        assert atom.decoder_covariance.shape == (m_k * p, m_k * p)
        np.testing.assert_allclose(atom.decoder_covariance, atom.decoder_covariance.T, atol=1e-8)
        assert np.all(np.diag(atom.decoder_covariance) >= -1e-10)

        band = fit.shape_uncertainty(atom=atom_k, n_sd=1.5)
        alias = fit.shape_band(atom_k, n_sd=1.5)
        assert set(band) == {"coords", "mean", "sd", "lower", "upper"}
        assert band["coords"].shape == atom.shape_band_coords.shape
        assert band["coords"].ndim == 2
        assert band["coords"].shape[1] == d_k
        assert band["mean"].shape == band["sd"].shape == band["lower"].shape
        assert band["upper"].shape == band["mean"].shape
        assert band["mean"].shape == atom.shape_band_mean.shape
        assert band["sd"].shape == atom.shape_band_sd.shape
        assert band["mean"].shape[1] == p
        for key in ("coords", "mean", "sd", "lower", "upper"):
            assert np.all(np.isfinite(band[key]))
            np.testing.assert_allclose(alias[key], band[key])
        assert np.all(band["sd"] >= 0.0)
        np.testing.assert_allclose(band["lower"], band["mean"] - 1.5 * band["sd"])
        np.testing.assert_allclose(band["upper"], band["mean"] + 1.5 * band["sd"])

        coord_range = fit.coordinate_range(atom=atom_k)
        assert coord_range["n"] == z.shape[0]
        assert coord_range["quantile_levels"].shape == (3,)
        assert coord_range["quantiles"].shape == (3, d_k)
        np.testing.assert_allclose(coord_range["quantile_levels"], [0.05, 0.50, 0.95])
        for key in ("min", "max", "p05", "p50", "median", "p95"):
            assert coord_range[key].shape == (d_k,)
            assert np.all(np.isfinite(coord_range[key]))
        np.testing.assert_allclose(coord_range["median"], coord_range["p50"])
        assert np.all(coord_range["min"] <= coord_range["p05"])
        assert np.all(coord_range["p05"] <= coord_range["p50"])
        assert np.all(coord_range["p50"] <= coord_range["p95"])
        assert np.all(coord_range["p95"] <= coord_range["max"])
