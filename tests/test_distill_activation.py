from __future__ import annotations

import numpy as np
import pytest

from gamfit.distill import _activation_from_logits


def test_distilled_ibp_activation_applies_the_first_stick_to_atom_zero() -> None:
    assignments = _activation_from_logits(
        np.zeros((1, 3)),
        assignment="ibp_map",
        tau=1.0,
        alpha=1.0,
        jumprelu_threshold=0.0,
    )

    np.testing.assert_allclose(assignments, [[0.25, 0.125, 0.0625]])


def test_distilled_softmax_and_threshold_gate_use_rust_assignment_kernel() -> None:
    softmax = _activation_from_logits(
        np.array([[0.0, 1.0, -1.0]]),
        assignment="SoftMax",
        tau=0.5,
        alpha=1.0,
        jumprelu_threshold=0.0,
    )
    assert softmax.shape == (1, 3)
    np.testing.assert_allclose(np.sum(softmax, axis=1), [1.0])

    threshold = _activation_from_logits(
        np.array([[-1000.0, 0.0, 1.0]]),
        assignment="jump-relu",
        tau=1.0,
        alpha=1.0,
        jumprelu_threshold=0.0,
    )
    np.testing.assert_allclose(threshold, [[0.0, 0.0, 1.0 / (1.0 + np.exp(-1.0))]])


@pytest.mark.parametrize("alpha", [0.0, -1.0, np.nan, np.inf])
def test_distilled_ibp_activation_rejects_invalid_concentration(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha must be finite and positive"):
        _activation_from_logits(
            np.zeros((1, 2)),
            assignment="ibp_map",
            tau=1.0,
            alpha=alpha,
            jumprelu_threshold=0.0,
        )


def test_distilled_activation_rejects_invalid_temperature() -> None:
    with pytest.raises(ValueError, match="temperature must be finite and positive"):
        _activation_from_logits(
            np.zeros((1, 2)),
            assignment="softmax",
            tau=0.0,
            alpha=1.0,
            jumprelu_threshold=0.0,
        )
