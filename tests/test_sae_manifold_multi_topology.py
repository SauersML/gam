from __future__ import annotations

import math

import numpy as np
import pytest

# #1512: this fit exceeds the standard Python-API CI runner budget (>60s in
# triage), so it is tagged slow and excluded from the directory-level
# `-m "not slow"` CI step while still being collected (run by a bare pytest).
pytestmark = pytest.mark.slow

gamfit = pytest.importorskip("gamfit")

R2_THRESHOLD = 0.85


def _circle_data(n: int, p: int, noise: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * math.pi, n)
    harm = np.column_stack([np.cos(theta), np.sin(theta)])
    mixing = rng.normal(size=(harm.shape[1], p))
    mixing /= np.maximum(np.linalg.norm(mixing, axis=0, keepdims=True), 1e-8)
    z = harm @ mixing + noise * rng.normal(size=(n, p))
    z -= z.mean(axis=0, keepdims=True)
    return z


def _sphere_data(n: int, p: int, noise: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lat = rng.uniform(-math.pi / 2.0, math.pi / 2.0, n)
    lon = rng.uniform(0.0, 2.0 * math.pi, n)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z_axis = np.sin(lat)
    harm = np.column_stack([x, y, z_axis])
    mixing = rng.normal(size=(harm.shape[1], p))
    mixing /= np.maximum(np.linalg.norm(mixing, axis=0, keepdims=True), 1e-8)
    z = harm @ mixing + noise * rng.normal(size=(n, p))
    z -= z.mean(axis=0, keepdims=True)
    return z


def _square_data(n: int, p: int, noise: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, 1.0, size=(n, 2))
    latent = np.column_stack(
        [
            np.sin(2.0 * math.pi * t[:, 0]),
            np.cos(2.0 * math.pi * t[:, 1]),
        ]
    )
    mixing = rng.normal(size=(latent.shape[1], p))
    mixing /= np.maximum(np.linalg.norm(mixing, axis=0, keepdims=True), 1e-8)
    z = latent @ mixing + noise * rng.normal(size=(n, p))
    z -= z.mean(axis=0, keepdims=True)
    return z


def _data_for_basis(basis: str) -> np.ndarray:
    if basis == "periodic":
        return _circle_data(n=400, p=64, noise=0.04, seed=0)
    if basis == "sphere":
        return _sphere_data(n=500, p=64, noise=0.03, seed=0)
    return _square_data(n=400, p=64, noise=0.03, seed=0)


def _r2(x: np.ndarray, fitted: np.ndarray) -> float:
    ss_res = float(np.sum((x - fitted) ** 2))
    ss_tot = float(np.sum((x - x.mean(0, keepdims=True)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


@pytest.mark.parametrize("basis", ["periodic", "sphere", "euclidean", "duchon"])
def test_single_atom_recovers_each_supported_topology(basis: str):
    z = _data_for_basis(basis)

    fit = gamfit.sae_manifold_fit(
        X=z,
        K=1,
        atom_basis=basis,
        d_atom=2,
        assignment="ibp_map",
        n_iter=50,
        learning_rate=0.04,
        random_state=0,
    )

    r2 = _r2(z, fit.fitted)
    assert r2 >= R2_THRESHOLD, (
        f"{basis} basis reconstruction R^2 = {r2:.4f}; "
        f"expected >= {R2_THRESHOLD:.2f}"
    )
