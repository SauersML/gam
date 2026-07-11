"""Integration coverage for sphere-atom coordinate optimization at the poles.

The existing sphere tests deliberately avoid the poles: the Rust
integration test samples ``lat in [-0.45*pi, +0.45*pi]`` ("avoiding the
poles where the chart degenerates") and the Python multi-topology test
draws latitudes uniformly (pole probability zero). The
``SphereChartEvaluator`` implements correct pole handling — it clamps the
latitude and zeroes the lat-derivatives (``chain_lat``) at the poles so the
basis Jacobian has vanishing latitude components there — but no integration
test feeds pole observations through the Arrow-Schur Newton loop.

This test synthesizes data with observations *exactly* at both poles and
asserts the fit stays finite and recovers the signal, exercising
``SphereChartEvaluator::evaluate`` and its second jet ``chain_lat`` logic
under the degenerate-Jacobian condition the solver must tolerate.
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


def _sphere_data_with_poles(
    n_pole: int, n_interior: int, p: int, noise: float, seed: int
) -> np.ndarray:
    """S^2 ground truth (first-order spherical harmonics) with explicit
    pole observations: ``n_pole`` rows at the north pole (lat=+pi/2),
    ``n_pole`` rows at the south pole (lat=-pi/2), and ``n_interior`` rows
    spread over the interior. Longitudes cover the full circle (degenerate
    at the poles, which is precisely the case the chart must absorb)."""
    rng = np.random.default_rng(seed)

    lat_north = np.full(n_pole, math.pi / 2.0)
    lat_south = np.full(n_pole, -math.pi / 2.0)
    lat_interior = rng.uniform(-math.pi / 2.0, math.pi / 2.0, n_interior)
    lat = np.concatenate([lat_north, lat_south, lat_interior])

    n = lat.shape[0]
    lon = rng.uniform(0.0, 2.0 * math.pi, n)

    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z_axis = np.sin(lat)
    harm = np.column_stack([x, y, z_axis])
    mixing = rng.normal(size=(harm.shape[1], p))
    mixing /= np.maximum(np.linalg.norm(mixing, axis=0, keepdims=True), 1e-8)
    out = harm @ mixing + noise * rng.normal(size=(n, p))
    out -= out.mean(axis=0, keepdims=True)
    return out


def _r2(x: np.ndarray, fitted: np.ndarray) -> float:
    ss_res = float(np.sum((x - fitted) ** 2))
    ss_tot = float(np.sum((x - x.mean(0, keepdims=True)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def test_sphere_atom_stable_at_poles():
    """A single sphere atom fit on data containing exact pole observations
    must (1) stay finite throughout and (2) recover the signal at
    R^2 >= 0.65 (pole data is harder than the interior-only 0.85 test)."""
    z = _sphere_data_with_poles(
        n_pole=50, n_interior=150, p=48, noise=0.03, seed=0
    )

    fit = gamfit.sae_manifold_fit(
        X=z,
        K=1,
        atom_basis="sphere",
        d_atom=2,
        assignment="ordered_beta_bernoulli",
        n_iter=30,
        learning_rate=0.04,
        random_state=0,
    )

    # Reconstruction must be finite everywhere — the zeroed lat-derivatives
    # at the poles must not produce a NaN/Inf Newton step.
    assert np.all(np.isfinite(fit.fitted)), (
        "sphere fit with pole observations produced non-finite reconstruction; "
        "the degenerate-Jacobian (chain_lat=0) pole case is destabilizing the "
        "Arrow-Schur step."
    )

    # Decoder coefficients must be finite.
    for k, block in enumerate(fit.decoder_blocks):
        block = np.asarray(block)
        assert np.all(np.isfinite(block)), (
            f"sphere atom {k} decoder coefficients contain non-finite entries"
        )

    # Coordinates must be finite (the on-atom lat/lon estimates).
    for k, coords in enumerate(fit.coords):
        coords = np.asarray(coords)
        assert np.all(np.isfinite(coords)), (
            f"sphere atom {k} coordinates contain non-finite entries"
        )

    r2 = _r2(z, fit.fitted)
    assert r2 >= 0.65, (
        f"sphere atom on pole-inclusive S^2 data: R^2 = {r2:.4f}; expected "
        f">= 0.65. Below threshold indicates the boundary-clamped basis and "
        f"the Arrow-Schur step direction are interacting badly at the poles."
    )
