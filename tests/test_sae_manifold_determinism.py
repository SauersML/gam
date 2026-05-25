"""Hard failing determinism test for repeated SAE fitting."""
from __future__ import annotations

import math

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")


def _synthetic_one_harmonic(
    n: int = 400,
    p: int = 64,
    noise: float = 0.04,
    seed: int = 0,
) -> np.ndarray:
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


def _diff_under(a: np.ndarray, b: np.ndarray, atol: float) -> bool:
    return bool(np.allclose(a, b, rtol=0.0, atol=atol))


def test_sae_fit_is_deterministic_for_fixed_seed():
    z = _synthetic_one_harmonic()
    kwargs = dict(
        Z=z,
        n_atoms=2,
        atom_basis="periodic",
        atom_dim=2,
        assignment_prior="ibp_map",
        max_iter=50,
        learning_rate=0.04,
        random_state=123,
    )

    fit_a = gamfit.sae_manifold_fit(**kwargs)
    fit_b = gamfit.sae_manifold_fit(**kwargs)

    r2_a = _r2(z, fit_a.fitted)
    r2_b = _r2(z, fit_b.fitted)
    assert np.isfinite(r2_a) and np.isfinite(r2_b)

    maxdiff = float(np.max(np.abs(fit_a.fitted - fit_b.fitted)))
    assert np.array_equal(fit_a.fitted, fit_b.fitted) or _diff_under(
        fit_a.fitted, fit_b.fitted, 1e-10
    ), f"determinism violated: max abs diff = {maxdiff:.2e}"

    np.testing.assert_allclose(
        fit_a.assignments,
        fit_b.assignments,
        rtol=0.0,
        atol=1e-10,
    )
    for atom_a, atom_b in zip(fit_a.atoms, fit_b.atoms, strict=True):
        np.testing.assert_allclose(
            atom_a.decoder_coefficients,
            atom_b.decoder_coefficients,
            rtol=0.0,
            atol=1e-10,
        )
