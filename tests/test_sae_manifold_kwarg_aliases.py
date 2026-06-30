"""Regression tests for the ``sae_manifold_fit`` kwarg-alias contract.

Issues #159 (``assignment`` / ``assignment_prior``) and #160 (``K`` /
``n_atoms``): the two members of each alias pair must resolve through one code
path so they behave identically, and supplying both with conflicting values
must raise an eager ``ValueError`` naming both — never a silent winner (#160)
or a cryptic downstream Rust failure (#159).
"""
from __future__ import annotations

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")


def _data(n: int = 200, p: int = 4, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, p))


# ---- #159: assignment / assignment_prior ---------------------------------


@pytest.mark.parametrize(
    ("spelling", "canonical"),
    [
        ("ibp", "ibp_map"),
        ("ibp_map", "ibp_map"),
        ("ibp-map", "ibp_map"),
        ("softmax", "softmax"),
        ("jumprelu", "jumprelu"),
        ("jump_relu", "jumprelu"),
        ("gated", "jumprelu"),
    ],
)
def test_assignment_aliases_normalize(spelling: str, canonical: str) -> None:
    from gamfit._sae_manifold import _canonical_public_assignment

    assert _canonical_public_assignment(spelling) == canonical


def test_assignment_prior_matches_assignment() -> None:
    """``assignment_prior="ibp"`` must behave identically to
    ``assignment="ibp"`` (issue #159 — the two validators used to disagree)."""
    z = _data()
    via_assignment = gamfit.sae_manifold_fit(X=z, K=2, assignment="ibp", n_iter=2)
    via_prior = gamfit.sae_manifold_fit(X=z, K=2, assignment_prior="ibp", n_iter=2)
    assert via_assignment.assignment == via_prior.assignment == "ibp_map"


def test_assignment_conflict_raises_eagerly() -> None:
    """Conflicting ``assignment`` / ``assignment_prior`` must raise a clean
    ValueError naming both — not fall into Rust (issue #159 reported a Schur
    Cholesky crash here)."""
    z = _data()
    with pytest.raises(ValueError, match="assignment_prior"):
        gamfit.sae_manifold_fit(
            X=z, K=2, assignment="ibp", assignment_prior="softmax", n_iter=2
        )


def test_assignment_prior_equal_value_passes() -> None:
    z = _data()
    model = gamfit.sae_manifold_fit(
        X=z, K=2, assignment="softmax", assignment_prior="softmax", n_iter=2
    )
    assert model.assignment == "softmax"


def test_unknown_assignment_prior_lists_aliases() -> None:
    z = _data()
    with pytest.raises(ValueError, match="not a recognized assignment kind"):
        gamfit.sae_manifold_fit(X=z, K=2, assignment_prior="bogus", n_iter=2)


# ---- #160: K / n_atoms ----------------------------------------------------


def test_n_atoms_alias_resolves_k() -> None:
    z = _data()
    model = gamfit.sae_manifold_fit(X=z, n_atoms=2, n_iter=2)
    assert model.summary()["K"] == 2
    assert len(model.atoms) == 2


def test_k_and_n_atoms_conflict_raises() -> None:
    """Issue #160: conflicting ``K`` / ``n_atoms`` was silently accepted; now a
    clean ValueError naming both values."""
    z = _data()
    with pytest.raises(ValueError, match=r"K and n_atoms both supplied"):
        gamfit.sae_manifold_fit(X=z, K=2, n_atoms=3, n_iter=2)


def test_k_and_n_atoms_equal_value_passes() -> None:
    z = _data()
    model = gamfit.sae_manifold_fit(X=z, K=2, n_atoms=2, n_iter=2)
    assert model.summary()["K"] == 2
