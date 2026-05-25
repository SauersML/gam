"""Tests for ``gamfit.diagnostics.check_jacobian_sparsity``."""

from __future__ import annotations

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")
diagnostics = gamfit.diagnostics


def test_jacobian_sparsity_passes_on_known_sparse_decoder():
    # P = 20, latent_dim = 3. Diagonal-ish sparse decoder: only 3 entries
    # nonzero out of 60, so sparsity = 57/60 = 0.95.
    P, K = 20, 3
    J = np.zeros((P, K))
    J[0, 0] = 1.0
    J[1, 1] = 1.0
    J[2, 2] = 1.0

    report = diagnostics.check_jacobian_sparsity(jacobians=J, sparsity_threshold=0.5)
    assert report.passes, repr(report)
    assert report.preconditions["sparsity_above_threshold"]
    assert report.preconditions["jacobian_full_column_rank"]
    assert report.details["mean_sparsity"] >= 0.5


def test_jacobian_sparsity_fails_on_dense_jacobian():
    rng = np.random.default_rng(0)
    # Fully dense decoder: every entry well above the zero threshold.
    J = rng.normal(loc=2.0, scale=0.1, size=(20, 3))

    report = diagnostics.check_jacobian_sparsity(jacobians=J, sparsity_threshold=0.5)
    assert not report.passes
    assert not report.preconditions["sparsity_above_threshold"]
    # Concrete: violation mentions the measured sparsity.
    assert any("sparsity" in v for v in report.violations)
    assert len(report.recommendations) == len(report.violations)
