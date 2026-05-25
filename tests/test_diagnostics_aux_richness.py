"""Tests for ``gamfit.diagnostics.check_aux_richness`` (iVAE precondition)."""

from __future__ import annotations

import numpy as np
import pytest

gamfit = pytest.importorskip("gamfit")
diagnostics = gamfit.diagnostics


def test_aux_richness_passes_on_rich_2d_aux():
    rng = np.random.default_rng(0)
    n = 200
    # Aux is 2D, three distinct categorical levels, 200 rows.
    aux = rng.integers(0, 3, size=(n, 2)).astype(float)
    # Latents are 2D and depend linearly on aux (rich Jacobian).
    latents = aux @ np.array([[1.0, 0.0], [0.0, 1.0]]) + 0.01 * rng.normal(size=(n, 2))

    report = diagnostics.check_aux_richness(aux, latents)
    assert report.passes, repr(report)
    assert report.preconditions["aux_observed"]
    assert report.preconditions["aux_dim_at_least_latent_dim"]
    assert report.preconditions["aux_varies_across_rows"]
    assert report.preconditions["jacobian_rank_full"]
    assert report.violations == []
    assert report.recommendations == []


def test_aux_richness_fails_on_constant_1d_aux_against_2d_latents():
    n = 100
    aux = np.zeros(n)  # 1D constant
    rng = np.random.default_rng(1)
    latents = rng.normal(size=(n, 2))

    report = diagnostics.check_aux_richness(aux, latents)
    assert not report.passes
    # Concrete violation strings, not vague.
    joined = " ".join(report.violations)
    assert "aux dimension 1 is less than latent dimension 2" in joined
    assert any("constant" in v for v in report.violations)
    # Each violation must have a matching recommendation.
    assert len(report.recommendations) == len(report.violations)
    # Recommendations are concrete (mention specific numbers).
    rec_joined = " ".join(report.recommendations)
    assert "2 dimensions" in rec_joined or "factor_dim" in rec_joined
