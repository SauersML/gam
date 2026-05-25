"""Tests for ``gamfit.diagnostics.identifiability_report`` composite dispatch."""

from __future__ import annotations

import numpy as np
import pytest

from dataclasses import dataclass

gamfit = pytest.importorskip("gamfit")
diagnostics = gamfit.diagnostics


@dataclass
class _IdentifiableModel:
    """Stand-in for an iVAE + manifold-SAE fit that satisfies every theorem.

    * Linear decoder is sparse (mostly zero) with full column rank.
    * Aux is rich (2D, varies, finite).
    * Assignment matrix has clean anchors covering all atoms.
    """
    _aux_diagnostic: np.ndarray
    _latents_diagnostic: np.ndarray
    decoder: np.ndarray
    assignments: np.ndarray


@dataclass
class _NonIdentifiableModel:
    """K=2 atoms fitted on rotationally-symmetric data: all rows weight
    both atoms equally, so neither atom is individually anchored."""
    assignments: np.ndarray


def _build_identifiable_model() -> _IdentifiableModel:
    rng = np.random.default_rng(0)
    n = 60
    # Aux: 2D, varies, integer-valued with three levels.
    aux = rng.integers(0, 3, size=(n, 2)).astype(float)
    latents = aux + 0.01 * rng.normal(size=(n, 2))

    # Sparse decoder: 6 features, 2 latents, only one nonzero per column.
    decoder = np.zeros((6, 2))
    decoder[0, 0] = 1.0
    decoder[1, 1] = 1.0

    # Three blocks of clean anchors over K=2 atoms.
    assignments = np.full((n, 2), 0.01)
    assignments[: n // 2, 0] = 1.0
    assignments[n // 2 :, 1] = 1.0
    return _IdentifiableModel(
        _aux_diagnostic=aux, _latents_diagnostic=latents,
        decoder=decoder, assignments=assignments,
    )


def test_identifiability_report_passes_on_well_specified_model():
    model = _build_identifiable_model()
    report = diagnostics.identifiability_report(model)
    assert report.passes, report.detail()
    assert len(report.reports) == 3  # aux_richness, jacobian_sparsity, anchor_consistency
    names = {r.name for r in report.reports}
    assert names == {"aux_richness", "jacobian_sparsity", "anchor_consistency"}
    assert "Identified" in report.summary()


def test_identifiability_report_flags_anchor_violation_with_recommendation():
    # K=2 atoms, every row weights both equally -> zero anchors, both atoms uncovered.
    assignments = np.full((40, 2), 0.5)
    model = _NonIdentifiableModel(assignments=assignments)
    report = diagnostics.identifiability_report(model)
    assert not report.passes
    # Anchor-consistency must appear and must fail.
    anchor_reports = [r for r in report.reports if r.name == "anchor_consistency"]
    assert len(anchor_reports) == 1
    ar = anchor_reports[0]
    assert not ar.passes
    # Summary surfaces the "linear transform in atom space" wording.
    assert "atom" in report.summary().lower()
    # The recommendation must be concrete (not empty, not vague).
    assert len(ar.recommendations) == len(ar.violations)
    assert all(len(rec) > 10 for rec in ar.recommendations)
