"""RED tests for issue #243 — sae_manifold_fit drops user-supplied atom_basis
from the topology metadata, leaving `summary()["atom_topology"]` stuck at the
default `"circle"` even when atom geometry was resolved correctly.

These tests must fail today and pass once the topology field is reconciled
with `atom_basis` (or callers are forced to supply both consistently).
"""
from __future__ import annotations

import numpy as np
import pytest

import gamfit

# #1512: this fit exceeds the standard Python-API CI runner budget (>60s in
# triage), so it is tagged slow and excluded from the directory-level
# `-m "not slow"` CI step while still being collected (run by a bare pytest).
pytestmark = pytest.mark.slow


def _toy_inputs(n: int = 24, p: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, p)).astype(np.float64)


@pytest.mark.parametrize(
    "atom_basis,expected_topology,atom_dim",
    [
        ("sphere", "sphere", 2),
        ("torus", "torus", 2),
        ("periodic", "circle", 1),
        ("linear", "linear", 1),
    ],
)
def test_summary_topology_matches_atom_basis(
    atom_basis: str, expected_topology: str, atom_dim: int
) -> None:
    X = _toy_inputs()
    fit = gamfit.sae_manifold_fit(
        X=X,
        K=1,
        atom_basis=atom_basis,
        d_atom=atom_dim,
        assignment="softmax",
        n_iter=1,
        random_state=0,
    )
    summary = fit.summary()
    assert summary["atom_topology"] == expected_topology, (
        f"atom_basis={atom_basis!r} → expected topology={expected_topology!r}, "
        f"got {summary['atom_topology']!r} (geometry_plans={fit.geometry_plans!r})"
    )


@pytest.mark.parametrize(
    "atom_basis,expected_topology,atom_dim",
    [
        ("sphere", "sphere", 2),
        ("torus", "torus", 2),
    ],
)
def test_payload_round_trip_preserves_topology(
    atom_basis: str, expected_topology: str, atom_dim: int
) -> None:
    """The stored `atom_topology` field must also be reachable via the raw
    attribute (not just summary), to guard against fix-only-the-formatter
    regressions."""
    X = _toy_inputs()
    fit = gamfit.sae_manifold_fit(
        X=X,
        K=1,
        atom_basis=atom_basis,
        d_atom=atom_dim,
        assignment="softmax",
        n_iter=1,
        random_state=0,
    )
    assert fit.atom_topology == expected_topology, (
        f"atom_basis={atom_basis!r} → expected fit.atom_topology="
        f"{expected_topology!r}, got {fit.atom_topology!r}"
    )


def test_geometry_plans_and_topology_are_internally_consistent() -> None:
    """The most user-visible invariant: geometry plans and summary topology must
    not contradict each other. Pin the consistency rule directly so future
    drift in either field fails this test."""
    X = _toy_inputs()
    fit = gamfit.sae_manifold_fit(
        X=X,
        K=2,
        atom_basis="sphere",
        d_atom=2,
        assignment="softmax",
        n_iter=1,
        random_state=0,
    )
    topology = fit.summary()["atom_topology"]
    assert all(plan["kind"] == "sphere" for plan in fit.geometry_plans), fit.geometry_plans
    assert topology == "sphere", (
        f"geometry plans say sphere but summary['atom_topology']={topology!r}"
    )


def test_linear_topology_is_distinct_from_euclidean_quadratic_patch() -> None:
    """The #1026 EV-vs-K driver must compare curved atoms against the genuine
    linear atom, not the degree-2 Euclidean patch."""
    X = _toy_inputs()
    linear = gamfit.sae_manifold_fit(
        X=X,
        K=1,
        atom_topology="linear",
        d_atom=1,
        assignment="softmax",
        n_iter=1,
        random_state=0,
    )
    euclidean = gamfit.sae_manifold_fit(
        X=X,
        K=1,
        atom_topology="euclidean",
        d_atom=1,
        assignment="softmax",
        n_iter=1,
        random_state=0,
    )

    assert linear.atom_topology == "linear"
    assert [plan["kind"] for plan in linear.geometry_plans] == ["linear"]
    assert linear.atoms[0].basis == "linear"
    assert linear.atoms[0].decoder_coefficients.shape[0] == 2

    assert euclidean.atom_topology == "euclidean"
    assert [plan["kind"] for plan in euclidean.geometry_plans] == ["euclidean_patch"]
    assert euclidean.atoms[0].basis == "euclidean"
    assert euclidean.atoms[0].decoder_coefficients.shape[0] > linear.atoms[0].decoder_coefficients.shape[0]


def test_explicit_conflicting_topology_and_basis_raises() -> None:
    """If the caller explicitly passes both and they disagree, the fitter must
    refuse rather than silently keeping the stale topology default. Mirrors
    the existing `assignment` vs `assignment_prior` conflict pattern in
    `sae_manifold_fit` (gamfit/_sae_manifold.py:401-409)."""
    X = _toy_inputs()
    with pytest.raises((ValueError, TypeError)):
        gamfit.sae_manifold_fit(
            X=X,
            K=1,
            atom_basis="sphere",
            atom_topology="circle",
            d_atom=2,
            assignment="softmax",
            n_iter=1,
            random_state=0,
        )
