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
    target = _random_inputs()
    rust = gamfit._sae_manifold.rust_module()
    payload = dict(
        rust.sae_manifold_fit_minimal(
            np.ascontiguousarray(target),
            ["periodic"] * 3,
            [1] * 3,
            1.0,
            0.5,
            False,
            "topk",
            sparsity_strength=1.0,
            smoothness=1.0,
            max_iter=5,
            learning_rate=0.04,
            gumbel_schedule=None,
            analytic_penalties=None,
            random_state=0,
            top_k=1,
            initial_logits=None,
            initial_coords=None,
            threshold_gate_threshold=0.0,
            fisher_factors=None,
            fisher_mass_residual=None,
            fisher_provenance=None,
            row_loss_weights=None,
        )
    )
    assert "top_k_projection" not in payload
    assert "pre_topk" not in payload
    assignments = np.asarray(payload["assignments_z"])
    np.testing.assert_array_equal((assignments != 0.0).sum(axis=1), 1)
    fitted = np.asarray(payload["fitted"])
    expected_data_fit = 0.5 * float(np.square(target - fitted).sum())
    assert float(payload["penalized_loss_breakdown"]["data_fit"]) == pytest.approx(
        expected_data_fit
    )
    assert np.isfinite(float(payload["penalized_laml_criterion"]))
