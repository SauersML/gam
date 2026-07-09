from __future__ import annotations

import math

import numpy as np
import pytest

from gamfit._description_length import FittedFeaturizer, description_length


def _featurizer(*, rows: int = 8, dimensions: int = 2) -> FittedFeaturizer:
    x = np.arange(rows * dimensions, dtype=float).reshape(rows, dimensions)
    gate = np.ones((rows, 1))
    return FittedFeaturizer(
        name="identity",
        gate=gate,
        atom_contribution=lambda _atom: x,
        code_dims=np.ones(1, dtype=int),
        dictionary_params=0,
        recon=x,
        fit_seconds=0.0,
    )


def test_description_length_scores_one_dimensional_residual_covariance() -> None:
    fitted = _featurizer(dimensions=1)
    test_x = fitted.recon + np.linspace(-0.5, 0.5, fitted.recon.shape[0])[:, None]

    result = description_length(fitted, test_x, r2_targets=(0.9,))

    assert set(result) == {
        "support_bits",
        "achieved_block_l0",
        "bits_at_r2_0.9",
        "code_bits_at_r2_0.9",
        "resid_bits_at_r2_0.9",
    }
    assert all(math.isfinite(float(value)) for value in result.values())


def test_description_length_rejects_shape_mismatch() -> None:
    fitted = _featurizer()

    with pytest.raises(ValueError, match="same shape"):
        description_length(fitted, np.ones((fitted.recon.shape[0] + 1, 2)))


def test_fitted_featurizer_requires_one_code_dimension_per_atom() -> None:
    with pytest.raises(ValueError, match="one entry per atom"):
        FittedFeaturizer(
            name="invalid",
            gate=np.ones((4, 2)),
            atom_contribution=lambda _atom: np.ones((4, 1)),
            code_dims=np.ones(1, dtype=int),
            dictionary_params=0,
            recon=np.ones((4, 1)),
            fit_seconds=0.0,
        )
