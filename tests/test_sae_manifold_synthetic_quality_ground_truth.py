"""Synthetic ground-truth quality checks for SAE-manifold.

These tests intentionally score against known latent structure, not only
reconstruction R^2. A bad SAE can reconstruct an additive signal with entangled
atoms; the quality contract here asks for the planted atom routing, inactive
leakage, and OOS reconstruction to be correct on data generated from the exact
periodic basis primitive the Rust OOS solver consumes.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")
from gamfit._binding import rust_module  # noqa: E402


def _periodic_phi(t: np.ndarray, n_harmonics: int = 1) -> np.ndarray:
    phi, _jet, _penalty = rust_module().basis_with_jet(
        "periodic",
        np.ascontiguousarray(np.asarray(t, dtype=float).reshape(-1, 1)),
        {"n_harmonics": int(n_harmonics)},
    )
    return np.asarray(phi, dtype=float)


def _r2(x: np.ndarray, fitted: np.ndarray) -> float:
    ss_res = float(np.sum((x - fitted) ** 2))
    ss_tot = float(np.sum((x - x.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def _best_binary_assignment_accuracy(assignments: np.ndarray, truth: np.ndarray) -> float:
    pred = np.argmax(assignments, axis=1)
    direct = float(np.mean(pred == truth))
    swapped = float(np.mean((1 - pred) == truth))
    return max(direct, swapped)


def _oracle_periodic_decoder() -> list[np.ndarray]:
    """Two periodic atoms in disjoint output subspaces.

    With ``Phi(t) = [1, sin(2 pi t), cos(2 pi t)]``, atom 0 writes to columns
    0:3 and atom 1 writes to columns 3:6. The zero intercept row keeps the
    generated data centered around the harmonic terms.
    """
    block0 = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.35, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.80, 0.45, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    block1 = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.90, -0.30, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.65, 1.05],
        ],
        dtype=float,
    )
    return [block0, block1]


def _planted_one_hot_periodic(
    n: int,
    *,
    seed: int,
    noise: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    truth = np.arange(n, dtype=int) % 2
    rng.shuffle(truth)
    t = rng.uniform(0.0, 1.0, size=(2, n, 1))
    decoder = _oracle_periodic_decoder()
    x = np.zeros((n, 6), dtype=float)
    for atom in (0, 1):
        rows = truth == atom
        x[rows] = _periodic_phi(t[atom, rows, 0]) @ decoder[atom]
    if noise:
        x += noise * rng.standard_normal(size=x.shape)
    return x, truth, t


def test_periodic_basis_public_entrypoints_agree_after_wrapping() -> None:
    """Both public periodic basis entrypoints must represent S1 coordinates.

    ``basis_with_jet(kind="periodic")`` wraps coordinates with ``rem_euclid``.
    The standalone ``periodic_basis_with_jet`` is also a public periodic basis
    primitive, so it must agree for equivalent coordinates outside [0, 1).
    """
    t = np.array([-1.25, -0.25, 0.25, 0.75, 1.25, 2.25], dtype=float)
    generic = _periodic_phi(t, n_harmonics=2)
    direct, _jet, _penalty = rust_module().periodic_basis_with_jet(t, 2)
    np.testing.assert_allclose(
        np.asarray(direct, dtype=float),
        generic,
        rtol=0.0,
        atol=1e-12,
        err_msg=(
            "periodic_basis_with_jet must wrap S1 coordinates the same way as "
            "basis_with_jet(kind='periodic')"
        ),
    )


def test_oos_fixed_decoder_recovers_one_hot_oracle_assignments() -> None:
    """Known decoder, known one-hot atoms: OOS should recover routing."""
    x, truth, _t = _planted_one_hot_periodic(n=16, seed=0, noise=0.0)
    decoder = _oracle_periodic_decoder()
    payload = rust_module().sae_manifold_predict_oos(
        np.ascontiguousarray(x),
        ["periodic", "periodic"],
        [1, 1],
        [np.ascontiguousarray(block) for block in decoder],
        [None, None],
        [1, 1],
        [3, 3],
        alpha=1.0,
        tau=0.25,
        assignment_kind="softmax",
        max_iter=4,
        learning_rate=1.0,
        log_lambda_sparse=float(np.log(0.01)),
        log_lambda_smooth=[float(np.log(0.01)), float(np.log(0.01))],
        log_ard=[[], []],
    )
    assignments = np.asarray(payload["assignments_z"], dtype=float)
    fitted = np.asarray(payload["fitted"], dtype=float)

    assert _r2(x, fitted) >= 0.98
    assert _best_binary_assignment_accuracy(assignments, truth) >= 0.95
    assert float(np.mean(np.min(assignments, axis=1))) <= 0.05
    for atom_idx, expected_block in enumerate(decoder):
        np.testing.assert_allclose(
            np.asarray(payload["atoms"][atom_idx]["decoder_B"], dtype=float),
            expected_block,
            rtol=0.0,
            atol=0.0,
            err_msg="OOS prediction must not mutate caller-provided decoder blocks",
        )


def test_fit_learns_disjoint_periodic_atoms_without_inactive_leakage() -> None:
    """Training-time SAE quality on planted one-hot atoms.

    Reconstruction alone is insufficient: assignment leakage into the inactive
    atom must stay low and the winning atom must match the planted atom up to
    permutation.
    """
    x, truth, _t = _planted_one_hot_periodic(n=48, seed=4, noise=0.01)
    fit = gamfit.sae_manifold_fit(
        X=x,
        K=2,
        atom_basis="periodic",
        d_atom=1,
        assignment="topk",
        top_k=1,
        isometry_weight=0.0,
        ard_per_atom=False,
        sparsity_weight=0.01,
        smoothness_weight=0.01,
        n_iter=6,
        learning_rate=1.0,
        random_state=1,
    )
    assignments = np.asarray(fit.assignments, dtype=float)

    assert _r2(x, np.asarray(fit.fitted, dtype=float)) >= 0.95
    assert _best_binary_assignment_accuracy(assignments, truth) >= 0.90
    assert float(np.mean(np.min(assignments, axis=1))) <= 0.02


def test_fit_oos_quality_matches_training_on_planted_oracle_distribution() -> None:
    """Fit on one draw, score OOS on another draw from the same oracle."""
    x_train, _truth_train, _ = _planted_one_hot_periodic(n=48, seed=10, noise=0.01)
    x_test, truth_test, _ = _planted_one_hot_periodic(n=16, seed=11, noise=0.01)
    fit = gamfit.sae_manifold_fit(
        X=x_train,
        K=2,
        atom_basis="periodic",
        d_atom=1,
        assignment="topk",
        top_k=1,
        isometry_weight=0.0,
        ard_per_atom=False,
        sparsity_weight=0.01,
        smoothness_weight=0.01,
        n_iter=6,
        learning_rate=1.0,
        random_state=2,
    )
    oos = fit.reconstruct(x_test)
    assignments = fit.encode(x_test)

    assert _r2(x_test, oos) >= 0.90


def _planted_circle(
    n: int = 200, d_ambient: int = 12, noise: float = 0.02, seed: int = 0
) -> np.ndarray:
    """A 1-D circle (period 2*pi) linearly embedded in ``d_ambient`` dims.

    Matches the #795 / #681 ``circ`` geometry: ``J^T J = (2*pi)^2`` along the
    intrinsic coordinate, so a *fixed* identity isometry reference ``G_ref = I``
    would pull the decoder radius toward ``1/(2*pi)`` and fight reconstruction.
    """
    rng = np.random.default_rng(seed)
    basis = rng.standard_normal((2, d_ambient))
    basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    t = rng.uniform(0.0, 2.0 * np.pi, n)
    clean = np.column_stack([np.cos(t), np.sin(t)]) @ basis
    return clean + noise * rng.standard_normal((n, d_ambient))


def test_isometry_on_circle_recovers_planted_geometry_normalized_reference() -> None:
    """#737 item 1: with the isometry penalty ON, the normalized metric
    reference must recover the planted circle.

    A *fixed* ``G_ref = I`` reference fixes the metric scale; for a period-1
    circle (``J^T J = (2*pi)^2``) that pulls the radius toward ``1/(2*pi)`` and
    fought reconstruction, collapsing recovery to R^2 ~= 0.47. The SAE isometry
    penalty now compares ``J^T J / gbar`` to identity, where ``gbar`` is the mean
    pullback trace per latent dimension. Recovery must therefore stay high with
    ``isometry_weight=0.1`` (the #737 repro value), NOT collapse toward 0.47.
    """
    z = _planted_circle(noise=0.02, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = gamfit.sae_manifold_fit(
            X=z,
            K=1,
            d_atom=1,
            atom_topology="circle",
            isometry_weight=0.1,
            n_iter=20,
            random_state=0,
        )
    assert np.all(np.isfinite(fit.fitted)), "isometry-on circle fit is non-finite"
    r2 = _r2(z, np.asarray(fit.fitted, dtype=float))
    assert r2 >= 0.90, (
        "normalized isometry reference must recover the planted "
        f"circle with isometry_weight=0.1 (#737); got R^2={r2:.4f} (the fixed "
        "identity reference regressed this to ~0.47)"
    )
