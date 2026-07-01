"""Regression tests for SAE kwarg aliases (issues #159 and #160).

#159: ``assignment=`` and ``assignment_prior=`` must normalize through ONE
validator with a shared public alias table; conflicting resolved values raise
an eager ``ValueError`` naming both, and unknown values list the accepted alias
set.

#160: ``K=`` and ``n_atoms=`` are aliases for the number of atoms; supplying
both with DIFFERENT values raises an eager ``ValueError``; equal values pass
through.

These checks are PURE PYTHON and run BEFORE any Rust call, so they exercise the
normalizer/validators directly (no compiled extension required for the helper
tests) and the public ``sae_manifold_fit`` eager validators (which raise before
reaching the Rust solver).
"""
from __future__ import annotations

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")

from gamfit._sae_manifold import (  # noqa: E402
    _ASSIGNMENT_PRIOR_UNSET,
    _canonical_public_assignment,
    _resolve_public_assignment,
)


# --------------------------------------------------------------------------- #
# #159: assignment / assignment_prior alias normalization
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value,expected",
    [
        ("ibp", "ibp_map"),
        ("ibp-map", "ibp_map"),
        ("ibp_map", "ibp_map"),
        ("IBP", "ibp_map"),
        ("softmax", "softmax"),
        ("SoftMax", "softmax"),
        ("gated", "jumprelu"),
        ("jump_relu", "jumprelu"),
        ("jumprelu", "jumprelu"),
    ],
)
def test_assignment_alias_normalizes(value, expected):
    assert _canonical_public_assignment(value) == expected


def test_assignment_prior_alias_accepts_ibp():
    # #159 core: assignment_prior="ibp" must NOT raise (it used to).
    assert (
        _resolve_public_assignment("ibp_map", "ibp") == "ibp_map"
    )


def test_assignment_alias_accepts_ibp():
    assert _resolve_public_assignment("ibp", _ASSIGNMENT_PRIOR_UNSET) == "ibp_map"


def test_assignment_unset_prior_passes_through():
    assert (
        _resolve_public_assignment("softmax", _ASSIGNMENT_PRIOR_UNSET) == "softmax"
    )


def test_assignment_equal_values_pass_through():
    # softmax via both kwargs (one aliased) resolves cleanly.
    assert _resolve_public_assignment("softmax", "softmax") == "softmax"
    # #1777 — "gated"/"jump_relu"/"jumprelu" are deprecated aliases that all
    # canonicalize to the primary "threshold_gate" token.
    assert _resolve_public_assignment("gated", "jump_relu") == "threshold_gate"
    assert _resolve_public_assignment("threshold_gate", "jumprelu") == "threshold_gate"


def test_assignment_conflict_raises_naming_both():
    with pytest.raises(ValueError) as exc:
        _resolve_public_assignment("ibp", "softmax")
    msg = str(exc.value)
    assert "assignment" in msg and "assignment_prior" in msg
    # Both raw values and both resolved kinds are surfaced.
    assert "'ibp'" in msg and "'softmax'" in msg
    assert "ibp_map" in msg


def test_assignment_unknown_value_lists_accepted_aliases():
    with pytest.raises(ValueError) as exc:
        _canonical_public_assignment("not_a_kind")
    msg = str(exc.value)
    # The accepted alias set is listed.
    for alias in ("ibp", "ibp_map", "softmax", "jumprelu", "gated"):
        assert alias in msg


def test_assignment_prior_unknown_value_raises():
    with pytest.raises(ValueError):
        _resolve_public_assignment("ibp_map", "definitely_bad")


# --------------------------------------------------------------------------- #
# #160: K / n_atoms aliases (eager validation in the public entry point)
# --------------------------------------------------------------------------- #
def _tiny_x() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((8, 6))


def test_k_and_n_atoms_conflict_raises():
    with pytest.raises(ValueError) as exc:
        gamfit.sae_manifold_fit(X=_tiny_x(), K=2, n_atoms=3)
    msg = str(exc.value)
    assert "K and n_atoms both supplied with different values" in msg
    assert "2 vs 3" in msg


def test_assignment_conflict_raises_in_public_fit():
    # The eager #159 conflict check fires before any Rust call.
    with pytest.raises(ValueError) as exc:
        gamfit.sae_manifold_fit(
            X=_tiny_x(), K=2, assignment="ibp", assignment_prior="softmax"
        )
    msg = str(exc.value)
    assert "assignment" in msg and "assignment_prior" in msg
