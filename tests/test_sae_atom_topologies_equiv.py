"""#2091 — the Rust `sae_atom_topologies` owner must reproduce the Python
`_topology_for_bases` / `_topologies_for_bases` derivation bit-for-bit.

This is the first `from_payload` coercion slice moved behind a Rust owner (the
same pattern as `sae_canonical_n_harmonics`): a later increment builds the full
`ManifoldSaePayload` directly from the raw fit payload and derives
`atom_topology` / `atom_topologies` through this function, so it must match the
dataclass derivation exactly (alias map, casing, `"mixed"` collapse). The empty
dictionary returns `(None, [])` so the caller supplies its own seed fallback
(`_topology_for_bases(kinds) if kinds else str(topology)`), which the Python
helper cannot express (it index-errors on an empty list)."""
from __future__ import annotations

import pytest

gamfit = pytest.importorskip("gamfit")
from gamfit._sae_manifold import (  # noqa: E402
    _topologies_for_bases,
    _topology_for_bases,
    rust_module,
)


def _rust_atom_topologies(bases):
    return rust_module().sae_atom_topologies(list(bases))


@pytest.mark.parametrize(
    "bases",
    [
        ["periodic"],
        ["periodic", "circle", "periodic_spline"],
        ["euclidean", "duchon", "euclidean_patch"],
        ["periodic", "euclidean"],  # -> mixed
        ["linear", "linear_rank1", "affine"],
        ["linear_block", "flat_block"],
        ["poincare", "hyperbolic", "poincare_patch"],
        ["Circle", "PERIODIC", "flat-block"],  # casing / dash aliases
        ["sphere", "torus", "cylinder"],
        ["totally_unknown_kind"],  # unknown -> passthrough label
        ["periodic", "totally_unknown_kind"],  # -> mixed
    ],
)
def test_sae_atom_topologies_matches_python(bases):
    scalar, per_atom = _rust_atom_topologies(bases)
    assert per_atom == _topologies_for_bases(bases)
    assert scalar == _topology_for_bases(bases)


def test_sae_atom_topologies_empty_returns_none_scalar():
    scalar, per_atom = _rust_atom_topologies([])
    assert scalar is None
    assert per_atom == []
