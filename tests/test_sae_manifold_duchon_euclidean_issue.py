"""RED tests for issue #246 — sae_manifold_fit with atom_basis="duchon"/"euclidean" fails.

Repros from issue #246:

- atom_basis="duchon", d_atom=2 → "Duchon D2 collocation requires 2*(p+s) > dimension+2"
- atom_basis="duchon", d_atom=1 → "sae_build_duchon_atom: primary penalty was not built"
- atom_basis="euclidean", d_atom={1,2,3} → same routing into broken Duchon builder

These tests should be RED today (the user-facing API rejects documented atom_basis
values for documented atom_dim choices) and GREEN once the Rust path is reworked
to (a) pick valid (m, s, power) per atom_dim, (b) give EuclideanPatch its own builder.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamfit

# #1512 triage: these Duchon/Euclidean SAE-manifold fits exceed the standard
# Python-API CI runner budget (>240s in triage), so they are tagged slow and
# excluded from the directory-level `-m "not slow"` CI step (still collected,
# and run by a bare `pytest tests/` locally).
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def random_data() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((40, 6))


@pytest.mark.parametrize("atom_dim", [1, 2, 3])
def test_sae_manifold_fit_duchon_atom_dim_succeeds(
    random_data: np.ndarray, atom_dim: int
) -> None:
    """sae_manifold_fit(atom_basis="duchon", d_atom=d) should succeed for d in {1,2,3}."""
    fit = gamfit.sae_manifold_fit(
        X=random_data,
        K=1,
        atom_basis="duchon",
        d_atom=atom_dim,
        assignment="softmax",
        n_iter=1,
        random_state=0,
    )
    assert fit is not None
    assert hasattr(fit, "assignments")
    assert hasattr(fit, "fitted")
    assert fit.assignments.shape[0] == random_data.shape[0]


@pytest.mark.parametrize("atom_dim", [1, 2, 3])
def test_sae_manifold_fit_euclidean_atom_dim_succeeds(
    random_data: np.ndarray, atom_dim: int
) -> None:
    """sae_manifold_fit(atom_basis="euclidean", d_atom=d) should succeed for d in {1,2,3}.

    Today this routes through sae_build_duchon_atom (lib.rs:9754, 9925), so it fails
    with the same Duchon errors. Euclidean atoms are mathematically distinct from
    thin-plate splines and should get their own builder.
    """
    fit = gamfit.sae_manifold_fit(
        X=random_data,
        K=1,
        atom_basis="euclidean",
        d_atom=atom_dim,
        assignment="softmax",
        n_iter=1,
        random_state=0,
    )
    assert fit is not None
    assert hasattr(fit, "assignments")
    assert fit.assignments.shape[0] == random_data.shape[0]


def test_sae_manifold_fit_duchon_2d_does_not_violate_collocation(
    random_data: np.ndarray,
) -> None:
    """The exact failure mode from the issue: d=2 trips 2*(p+s) > d+2."""
    fit = gamfit.sae_manifold_fit(
        X=random_data,
        K=1,
        atom_basis="duchon",
        d_atom=2,
        assignment="softmax",
        n_iter=1,
        random_state=0,
    )
    assert fit is not None


def test_sae_manifold_fit_duchon_1d_builds_primary_penalty(
    random_data: np.ndarray,
) -> None:
    """The exact 1D failure: 'sae_build_duchon_atom: primary penalty was not built'."""
    fit = gamfit.sae_manifold_fit(
        X=random_data,
        K=1,
        atom_basis="duchon",
        d_atom=1,
        assignment="softmax",
        n_iter=1,
        random_state=0,
    )
    assert fit is not None


def test_sae_manifold_fit_multi_atom_duchon_mix(random_data: np.ndarray) -> None:
    """Per-atom mixed Duchon dims should all work, not just one."""
    fit = gamfit.sae_manifold_fit(
        X=random_data,
        K=3,
        atom_basis="duchon",
        d_atom=[1, 2, 3],
        assignment="softmax",
        n_iter=1,
        random_state=0,
    )
    assert fit is not None
    assert fit.assignments.shape[1] == 3


def test_euclidean_atom_is_not_thin_plate(random_data: np.ndarray) -> None:
    """Euclidean atoms should not produce a thin-plate kernel design.

    A clear contract: when atom_basis="euclidean", the basis_specs metadata should
    distinguish it from "duchon". Today they share the underlying builder.
    """
    fit = gamfit.sae_manifold_fit(
        X=random_data,
        K=1,
        atom_basis="euclidean",
        d_atom=2,
        assignment="softmax",
        n_iter=1,
        random_state=0,
    )
    # basis_specs should report "euclidean" or "euclidean_patch", not "duchon"
    specs = list(fit.basis_specs)
    assert "duchon" not in specs[0].lower(), (
        f"euclidean atom_basis must not be reported as duchon; got {specs}"
    )
