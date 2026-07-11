"""End-to-end coverage for K>1 atoms with heterogeneous dimensions and
mixed topologies within a single SAE-manifold fit.

The production code fully supports per-atom topologies: the Python API's
``_bases`` / ``_dims`` accept per-atom lists, the FFI threads a
``Vec<String> atom_basis`` and ``Vec<usize> atom_dim`` through
``sae_build_atom_plans``, and ``build_sae_basis_evaluators`` dispatches a
topology-specific evaluator per atom (matching on ``basis_kinds[k]``).
``SaeManifoldTerm::apply_newton_step`` refreshes each atom's basis
independently. None of the existing tests exercises a heterogeneous
``atom_basis`` list end to end:

  - ``test_sae_manifold_multi_topology`` only tests K=1 per topology.
  - the K=2 torus recovery test uses homogeneous (both periodic) atoms.

This test passes a genuinely mixed list — a 1-D circle, a 2-D sphere, and
a 2-D torus — so the per-atom basis-refresh loop and cross-manifold
coordinate optimization are actually run together.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")


def _mixing(latent_dim: int, p: int, rng: np.random.Generator) -> np.ndarray:
    mixing = rng.normal(size=(latent_dim, p))
    mixing /= np.maximum(np.linalg.norm(mixing, axis=0, keepdims=True), 1e-8)
    return mixing


def _heterogeneous_manifold_data(
    n: int, p: int, noise: float, seed: int
) -> np.ndarray:
    """Three independent latent components, each living on a distinct
    manifold, additively mixed into a shared ``p``-dimensional output:

      - a 1-D circle (one Fourier harmonic),
      - a 2-D sphere (first-order spherical harmonics, lat/lon),
      - a 2-D torus (two independent circular harmonics).

    Each component is mixed through its own block of output features so the
    signals are not collinear and every atom carries identifiable signal.
    """
    rng = np.random.default_rng(seed)

    # 1-D circle component.
    theta = rng.uniform(0.0, 2.0 * math.pi, n)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])

    # 2-D sphere component (interior latitudes; pole stability is covered
    # by a dedicated test).
    lat = rng.uniform(-0.45 * math.pi, 0.45 * math.pi, n)
    lon = rng.uniform(0.0, 2.0 * math.pi, n)
    sphere = np.column_stack(
        [
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat),
        ]
    )

    # 2-D torus component.
    a = rng.uniform(0.0, 2.0 * math.pi, n)
    b = rng.uniform(0.0, 2.0 * math.pi, n)
    torus = np.column_stack([np.cos(a), np.sin(a), np.cos(b), np.sin(b)])

    z = (
        circle @ _mixing(circle.shape[1], p, rng)
        + sphere @ _mixing(sphere.shape[1], p, rng)
        + torus @ _mixing(torus.shape[1], p, rng)
        + noise * rng.normal(size=(n, p))
    )
    z -= z.mean(axis=0, keepdims=True)
    return z


def _r2(x: np.ndarray, fitted: np.ndarray) -> float:
    ss_res = float(np.sum((x - fitted) ** 2))
    ss_tot = float(np.sum((x - x.mean(0, keepdims=True)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def test_heterogeneous_mixed_topology_atoms_reconstruct():
    """A single fit with atom_basis=['periodic','sphere','torus'] and
    per-atom d_atom=[1,2,2] must reconstruct mixed-manifold data and
    expose per-atom coordinates whose width matches each declared dim."""
    z = _heterogeneous_manifold_data(n=600, p=72, noise=0.04, seed=0)

    atom_basis = ["periodic", "sphere", "torus"]
    atom_dim = [1, 2, 2]

    fit = gamfit.sae_manifold_fit(
        X=z,
        K=3,
        atom_basis=atom_basis,
        d_atom=atom_dim,
        assignment="ordered_beta_bernoulli",
        n_iter=60,
        learning_rate=0.04,
        random_state=0,
    )

    r2 = _r2(z, fit.fitted)
    assert r2 >= 0.75, (
        f"heterogeneous mixed-topology fit reconstruction R^2 = {r2:.4f}; "
        f"expected >= 0.75. Per-atom bases were {atom_basis} with dims "
        f"{atom_dim}; a low R^2 here means cross-manifold coordinate "
        f"optimization or the per-atom basis-refresh loop is broken."
    )

    # Each atom's coordinate block must have exactly its declared width.
    assert len(fit.coords) == 3, (
        f"expected 3 per-atom coordinate blocks, got {len(fit.coords)}"
    )
    for k, (basis, dim) in enumerate(zip(atom_basis, atom_dim)):
        coords_k = np.asarray(fit.coords[k])
        assert coords_k.ndim == 2 and coords_k.shape[0] == z.shape[0], (
            f"atom {k} ({basis}) coords shape {coords_k.shape} should be "
            f"(n={z.shape[0]}, d={dim})"
        )
        assert coords_k.shape[1] == dim, (
            f"atom {k} ({basis}) coords width {coords_k.shape[1]} != "
            f"declared atom_dim {dim}"
        )

    # Assignments are a finite (n, K) distribution.
    A = np.asarray(fit.assignments)
    assert A.shape == (z.shape[0], 3), (
        f"assignments shape {A.shape} should be (n={z.shape[0]}, K=3)"
    )
    assert np.all(np.isfinite(A)), "assignments contain non-finite entries"
    assert np.all(A >= -1e-9) and np.all(A <= 1.0 + 1e-9), (
        "assignments must lie in [0, 1] for bounded gates; "
        f"observed range [{A.min():.6g}, {A.max():.6g}]"
    )

    # Softmax invariant: each row's assignment mass sums to ~1.
    row_sums = A.sum(axis=1)
    np.testing.assert_allclose(
        row_sums, np.ones_like(row_sums), rtol=0.0, atol=1e-6
    )

    # The reconstruction itself must be finite.
    assert np.all(np.isfinite(fit.fitted)), "fitted reconstruction has NaN/Inf"
