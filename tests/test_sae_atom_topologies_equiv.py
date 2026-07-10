"""The Python topology helpers consume the single Rust-owned SAE schema."""
from __future__ import annotations

import pytest

gamfit = pytest.importorskip("gamfit")
from gamfit._sae_manifold import (  # noqa: E402
    _basis_to_topology,
    _canonical_basis_kind,
    _canonical_topology,
    _topologies_for_bases,
    _topology_for_bases,
    rust_module,
)


@pytest.mark.parametrize(
    "bases,expected_scalar,expected_per_atom",
    [
        (["periodic"], "circle", ["circle"]),
        (["periodic", "circle", "periodic_spline"], "circle", ["circle"] * 3),
        (["euclidean", "duchon", "euclidean_patch"], "euclidean", ["euclidean"] * 3),
        (["periodic", "euclidean"], "mixed", ["circle", "euclidean"]),
        (["linear", "linear_rank1", "affine"], "linear", ["linear"] * 3),
        (["linear_block", "flat_block"], "linear_block", ["linear_block"] * 2),
        (["poincare", "hyperbolic", "poincare_patch"], "poincare", ["poincare"] * 3),
        (["sphere", "torus", "cylinder"], "mixed", ["sphere", "torus", "cylinder"]),
        ([" AUTO "], "auto", ["auto"]),
        (["mobius-band"], "mobius", ["mobius"]),
        (["totally_unknown_kind"], "totally_unknown_kind", ["totally_unknown_kind"]),
    ],
)
def test_python_topology_helpers_consume_rust_schema(bases, expected_scalar, expected_per_atom):
    assert _topologies_for_bases(bases) == expected_per_atom
    assert _topology_for_bases(bases) == expected_scalar


def test_scalar_topology_and_basis_canonicalizers_share_rust_schema():
    assert _canonical_basis_kind(" Periodic-Spline ") == "periodic"
    assert _canonical_basis_kind("mobius-band") == "mobius"
    assert _canonical_topology(" Periodic-Spline ") == "circle"
    assert _canonical_topology(" AUTO ") == "auto"
    assert _basis_to_topology("Weird-Kind") == "Weird-Kind"


def test_sae_atom_topologies_empty_returns_none_scalar():
    scalar, per_atom = rust_module().sae_atom_topologies([])
    assert scalar is None
    assert per_atom == []
    with pytest.raises(ValueError, match="at least one basis"):
        _topology_for_bases([])
