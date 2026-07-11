"""Regression tests for the exact hard-TopK assignment contract."""

from __future__ import annotations

import numpy as np
import pytest

import gamfit


def _random_inputs() -> np.ndarray:
    return np.random.default_rng(0).standard_normal((24, 4))


@pytest.mark.parametrize("assignment", ["softmax", "ordered_beta_bernoulli", "threshold_gate"])
def test_smooth_assignments_reject_top_k(assignment: str) -> None:
    with pytest.raises(ValueError, match="valid only with assignment='topk'"):
        gamfit.sae_manifold_fit(
            X=_random_inputs(),
            K=3,
            atom_basis="periodic",
            d_atom=1,
            assignment=assignment,
            top_k=1,
            n_iter=2,
        )


@pytest.mark.parametrize("support", [1, 2])
def test_topk_fit_uses_exact_fixed_support(support: int) -> None:
    fit = gamfit.sae_manifold_fit(
        X=_random_inputs(),
        K=4,
        atom_basis="periodic",
        d_atom=1,
        assignment="topk",
        top_k=support,
        n_iter=5,
        random_state=0,
    )
    assignments = np.asarray(fit.assignments)
    np.testing.assert_array_equal((assignments != 0.0).sum(axis=1), support)
    np.testing.assert_array_equal(assignments[assignments != 0.0], 1.0)


def test_topk_payload_is_one_unprojected_model() -> None:
    fit = gamfit.sae_manifold_fit(
        X=_random_inputs(),
        K=3,
        atom_basis="periodic",
        d_atom=1,
        assignment="topk",
        top_k=1,
        n_iter=5,
        random_state=0,
    )
    payload = fit.to_dict()
    assert "top_k_projection" not in payload
    assert "pre_topk" not in payload
    assignments = np.asarray(payload["assignments"])
    np.testing.assert_array_equal((assignments != 0.0).sum(axis=1), 1)
    assert np.isfinite(float(payload["penalized_quasi_laplace_criterion"]))
