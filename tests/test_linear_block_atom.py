"""``atom_topology="linear_block"`` — BSF block as a manifold-SAE atom.

This pins the executable "BSF ⊂ ManifoldSAE" claim: a Block-Sparse-Featurizer
block is exactly a manifold-SAE ``Linear`` atom with an orthonormal decoder frame
``D_g`` and block-level gating, i.e. ``γ_g(t) = t·D_g``. Rather than a first-class
``LinearBlock`` enum variant (deliberately DEFERRED — it would force exhaustive-
match edits across ~10 ``manifold/`` files, a large collision-prone change), the
topology is a CONFIG on the linear atom: the Rust ``sae_atom_basis_kind_from_str``
maps ``"linear_block"`` -> ``SaeAtomBasisKind::Linear``, while the gamfit facade
preserves the ``"linear_block"`` label so an artifact round-trips as linear_block
(not linear).

These tests exercise the layers that do NOT need a live REML fit (the
extension's ``sae_manifold_fit`` path is separately known-broken in the dev venv);
the engine-level fit is exercised by the AMM zoo ``linear_block`` arm once the
extension is rebuilt.
"""

import numpy as np
import pytest

from gamfit._sae_manifold import (
    _basis_to_topology,
    _bases,
    _canonical_topology,
    flat_block_assignment,
    rust_module,
)


def test_linear_block_is_its_own_topology_label():
    # linear_block resolves to its OWN label (not collapsed to "linear"), so the
    # per-atom topology list round-trips through save/load (which stores
    # atom_topologies / basis_kinds verbatim).
    assert _bases(3, None, "linear_block") == ["linear_block"] * 3
    _scalar, topologies = rust_module().sae_atom_topologies(["linear_block"] * 3)
    assert topologies == ["linear_block"] * 3
    assert _basis_to_topology("linear_block") == "linear_block"
    # flat_block is an accepted alias; case / dash normalise.
    assert _basis_to_topology("flat_block") == "linear_block"
    assert _canonical_topology("Linear-Block") == "linear_block"


def test_both_block_gating_modes_are_constructible():
    # norm-selection mirrors the BSF paper (group-l2 block-TopK) -> ordered_beta_bernoulli;
    # separate-gate is presence separate from amplitude -> threshold_gate.
    assert flat_block_assignment("norm_selection") == "ordered_beta_bernoulli"
    assert flat_block_assignment("separate_gate") == "threshold_gate"
    # aliases + rejection of unknown modes.
    assert flat_block_assignment("norm") == "ordered_beta_bernoulli"
    assert flat_block_assignment("separate") == "threshold_gate"
    with pytest.raises(ValueError):
        flat_block_assignment("softmax")


def test_linear_block_decode_equals_t_times_orthonormal_frame():
    # A linear_block atom is the degree-1 (Linear) monomial patch [1, t_1..t_b]
    # with a ZERO intercept row and an ORTHONORMAL decoder frame D_g, so its
    # reconstruction is exactly γ(t) = [1, t] @ B = t @ D_g — the BSF block decode.
    rng = np.random.default_rng(0)
    p, b, n = 8, 3, 64
    # Orthonormal block frame D_g (b x p): b orthonormal rows (a Stiefel point).
    q, _ = np.linalg.qr(rng.standard_normal((p, b)))  # (p, b) orthonormal columns
    dframe = q.T  # (b, p), orthonormal rows
    assert np.allclose(dframe @ dframe.T, np.eye(b), atol=1e-10)

    # Degree-1 patch decoder B = [[intercept=0], [D_g]] (linear_block: no intercept).
    decoder_b = np.vstack([np.zeros((1, p)), dframe])  # (b+1, p)
    t = rng.standard_normal((n, b))
    phi = np.hstack([np.ones((n, 1)), t])  # monomial basis [1, t]
    gamma = phi @ decoder_b  # manifold-SAE reconstruction of the flat block

    # Exactly the BSF block decode t·D_g.
    assert np.allclose(gamma, t @ dframe, atol=1e-12)
    # And it is a genuine linear map through an orthonormal frame (norm-preserving
    # on the block subspace): ‖γ(t)‖ = ‖t‖ since D_g has orthonormal rows.
    assert np.allclose(np.linalg.norm(gamma, axis=1), np.linalg.norm(t, axis=1), atol=1e-10)


# The declared-linear_block relabel-after-fit behavior (formerly the Python
# facade helper `_preserve_linear_block_labels`, removed in the #2091 cutover)
# now lives in the Rust builder as `declared_bases=` and is unit-tested at its
# new home: crates/gam-pyffi/src/manifold/manifold_sae_coercion.rs
# (`preserve_linear_block_labels` tests cover relabel, plain-linear no-op, and
# structure-search retype non-clobber).
