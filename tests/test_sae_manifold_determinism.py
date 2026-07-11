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
        X=z,
        K=2,
        atom_basis="periodic",
        d_atom=2,
        assignment="ordered_beta_bernoulli",
        n_iter=50,
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


def test_sae_fit_random_state_changes_output():
    """Distinct `random_state` values must produce observably different fits
    (issue #178). With the seed wired into the assignment-logit init, any two
    seeds should perturb the Newton trajectory enough to change the final
    decoder and assignment posterior past trivial precision."""
    z = _synthetic_one_harmonic(noise=0.08)
    common = dict(
        X=z,
        K=2,
        atom_basis="periodic",
        d_atom=2,
        assignment="ordered_beta_bernoulli",
        n_iter=30,
        learning_rate=0.2,
    )
    fit_0 = gamfit.sae_manifold_fit(**common, random_state=0)
    fit_1 = gamfit.sae_manifold_fit(**common, random_state=1)
    fit_2 = gamfit.sae_manifold_fit(**common, random_state=42)

    fitted_diff_01 = float(np.max(np.abs(fit_0.fitted - fit_1.fitted)))
    fitted_diff_02 = float(np.max(np.abs(fit_0.fitted - fit_2.fitted)))
    assign_diff_01 = float(np.max(np.abs(fit_0.assignments - fit_1.assignments)))
    assert fitted_diff_01 > 1e-6, (
        f"random_state ignored: rs=0 vs rs=1 fitted max-abs diff = {fitted_diff_01:.2e}"
    )
    assert fitted_diff_02 > 1e-6, (
        f"random_state ignored: rs=0 vs rs=42 fitted max-abs diff = {fitted_diff_02:.2e}"
    )
    assert assign_diff_01 > 1e-6, (
        f"random_state ignored on assignments: rs=0 vs rs=1 max-abs diff = {assign_diff_01:.2e}"
    )


def _synthetic_disjoint_top1(n: int = 300, p_block: int = 16, seed: int = 0) -> np.ndarray:
    """Two well-separated periodic atoms, each loading a disjoint output block,
    with hard top-1 row ownership — the disjoint top-1 softmax configuration."""
    rng = np.random.default_rng(seed)
    half = n // 2
    t0 = rng.uniform(0.0, 2.0 * math.pi, n)
    t1 = rng.uniform(0.0, 2.0 * math.pi, n)
    block0 = np.zeros((n, p_block))
    block1 = np.zeros((n, p_block))
    block0[:half] = np.column_stack([np.cos(t0[:half]), np.sin(t0[:half])]) @ rng.normal(
        size=(2, p_block)
    )
    block1[half:] = np.column_stack([np.cos(t1[half:]), np.sin(t1[half:])]) @ rng.normal(
        size=(2, p_block)
    )
    x = np.concatenate([block0, block1], axis=1)
    x -= x.mean(axis=0, keepdims=True)
    return x


def test_sae_fit_random_state_changes_output_topk_path():
    """The disjoint TopK path must also honour
    ``random_state`` (issue #178): it built its init from an SVD and so was
    seed-independent before the fix."""
    x = _synthetic_disjoint_top1()
    common = dict(
        X=x,
        K=2,
        atom_basis="periodic",
        d_atom=1,
        assignment="topk",
        top_k=1,
        n_iter=10,
        learning_rate=0.2,
    )
    fit_0 = gamfit.sae_manifold_fit(**common, random_state=0)
    fit_1 = gamfit.sae_manifold_fit(**common, random_state=1)
    fit_0b = gamfit.sae_manifold_fit(**common, random_state=0)

    # Distinct seeds diverge.
    diff_01 = float(np.max(np.abs(fit_0.fitted - fit_1.fitted)))
    assert diff_01 > 1e-6, (
        f"random_state ignored on top-1 softmax path: 0 vs 1 diff = {diff_01:.2e}"
    )
    # Equal seeds remain bit-identical.
    np.testing.assert_array_equal(fit_0.fitted, fit_0b.fitted)
