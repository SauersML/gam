from __future__ import annotations

import numpy as np
import pytest

from gamfit.distill import _activation_from_logits


def test_distilled_ibp_activation_is_the_posterior_mean_gate() -> None:
    assignments = _activation_from_logits(
        np.zeros((1, 3)),
        assignment="ordered_beta_bernoulli",
        tau=1.0,
        threshold_gate_threshold=0.0,
    )

    np.testing.assert_allclose(assignments, [[0.5, 0.5, 0.5]])


def test_distilled_softmax_and_threshold_gate_use_rust_assignment_kernel() -> None:
    softmax = _activation_from_logits(
        np.array([[0.0, 1.0, -1.0]]),
        assignment="SoftMax",
        tau=0.5,
        threshold_gate_threshold=0.0,
    )
    assert softmax.shape == (1, 3)
    np.testing.assert_allclose(np.sum(softmax, axis=1), [1.0])

    threshold = _activation_from_logits(
        np.array([[-1000.0, 0.0, 1.0]]),
        assignment="jump-relu",
        tau=1.0,
        threshold_gate_threshold=0.0,
    )
    np.testing.assert_allclose(threshold, [[0.0, 0.0, 1.0 / (1.0 + np.exp(-1.0))]])


def test_distilled_topk_uses_the_core_hard_support() -> None:
    assignments = _activation_from_logits(
        np.array([[1.0, 3.0, 2.0], [5.0, -1.0, 0.0]]),
        assignment="topk",
        tau=1.0,
        threshold_gate_threshold=0.0,
        top_k=2,
    )
    np.testing.assert_array_equal(assignments, [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])


def test_distilled_topk_requires_fitted_support_size() -> None:
    with pytest.raises(ValueError, match="requires the fitted top_k"):
        _activation_from_logits(
            np.zeros((1, 3), dtype=float),
            assignment="topk",
            tau=1.0,
            threshold_gate_threshold=0.0,
        )


def test_distilled_activation_rejects_invalid_temperature() -> None:
    with pytest.raises(ValueError, match="temperature must be finite and positive"):
        _activation_from_logits(
            np.zeros((1, 2)),
            assignment="softmax",
            tau=0.0,
            threshold_gate_threshold=0.0,
        )
