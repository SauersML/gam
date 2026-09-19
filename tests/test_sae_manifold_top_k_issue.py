"""Regression tests for the exact hard-TopK assignment contract."""

from __future__ import annotations

import numpy as np
import pytest

import gamfit


def _random_inputs() -> np.ndarray:
    return np.random.default_rng(0).standard_normal((24, 4))


@pytest.mark.parametrize("assignment", ["softmax", "ordered_beta_bernoulli", "threshold_gate"])
def test_smooth_assignments_reject_top_k(assignment: str) -> None:
    with pytest.raises(
        ValueError, match=r"valid only with assignment_kind 'topk'"
    ):
        gamfit.sae.sae_manifold_fit(
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
    fit = gamfit.sae.sae_manifold_fit(
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
    fit = gamfit.sae.sae_manifold_fit(
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


def test_overcomplete_topk_defaults_reach_the_support_lane_2627() -> None:
    """#2627: the documented overcomplete TopK call refused its own defaults.

    At K > P a TopK dictionary is fitted by the support-sparse lane, which accepts no
    coordinate-shrinkage penalty. The public default sparsity strength built a SCAD
    descriptor that the lane then refused ("does not accept dense-coordinate or coefficient
    penalties"). With the default resolved to what the lane accepts, the default call fits.
    """
    fit = gamfit.sae.sae_manifold_fit(
        X=_random_inputs(),
        K=6,
        d_atom=1,
        assignment="topk",
        top_k=1,
        n_iter=5,
        random_state=0,
    )
    assert fit.requested_k == 6
    assert fit.top_k == 1
    assert fit.certificates["representation"] == "support_sparse"
    assert fit.termination["verdict"] == "converged"


def test_overcomplete_topk_refuses_an_explicit_sparsity_weight_2627() -> None:
    with pytest.raises(ValueError, match=r"sparsity_weight=0.5 has no coordinate to scale"):
        gamfit.sae.sae_manifold_fit(
            X=_random_inputs(),
            K=6,
            d_atom=1,
            assignment="topk",
            top_k=1,
            n_iter=5,
            random_state=0,
            sparsity_weight=0.5,
        )
