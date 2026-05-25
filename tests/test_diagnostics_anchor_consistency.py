"""Tests for ``gamfit.diagnostics.check_anchor_consistency``.

These tests build a small stub object that mimics the
``ManifoldSAE.assignments`` interface — checking the diagnostic against
hand-constructed assignment matrices rather than running an end-to-end
SAE fit (which is expensive and not the unit under test).
"""

from __future__ import annotations

import numpy as np
import pytest

from dataclasses import dataclass

gamfit = pytest.importorskip("gamfit")
diagnostics = gamfit.diagnostics


@dataclass
class _StubSAE:
    """Minimal interface that ``check_anchor_consistency`` consumes."""
    assignments: np.ndarray


def _three_cluster_assignments(n_per: int = 30) -> np.ndarray:
    """Build assignments for a 3-cluster, K=3 model with clear anchors.

    Each of the three blocks of rows is dominated by one atom.
    """
    n = 3 * n_per
    A = np.full((n, 3), 0.01)
    for k in range(3):
        A[k * n_per : (k + 1) * n_per, k] = 1.0
    return A


def test_anchor_consistency_passes_on_three_cluster_K3():
    A = _three_cluster_assignments(n_per=30)
    report = diagnostics.check_anchor_consistency(_StubSAE(A))
    assert report.passes, repr(report)
    assert report.preconditions["enough_anchors_total"]
    assert report.preconditions["anchors_cover_all_atoms"]
    # 30 anchors per atom, 90 total.
    assert report.details["n_anchors"] == 90
    assert report.details["anchors_per_atom"] == [30, 30, 30]


def test_anchor_consistency_fails_on_over_specified_K10_with_uniform_assignments():
    # 50 rows, K=10 atoms, all assignments equal -> no row dominates any
    # atom -> zero anchors total -> identifiability fails.
    n, K = 50, 10
    A = np.full((n, K), 1.0 / K)
    report = diagnostics.check_anchor_consistency(_StubSAE(A))
    assert not report.passes
    assert not report.preconditions["enough_anchors_total"]
    # Concrete recommendation must mention a smaller K or sharper prior.
    rec_joined = " ".join(report.recommendations)
    assert "K" in rec_joined or "atom" in rec_joined
    assert len(report.recommendations) == len(report.violations)
