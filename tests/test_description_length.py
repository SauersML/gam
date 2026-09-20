from __future__ import annotations

import math

import numpy as np
import pytest

from gamfit._description_length import (
    FittedFeaturizer,
    description_length,
)


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

    result = description_length(
        fitted, test_x, amortization_horizon=1000, r2_targets=(0.9,)
    )

    assert set(result) == {
        "support_bits",
        "independent_support_bits",
        "achieved_block_l0",
        "dictionary_bits",
        "estimation_rows",
        "amortization_horizon",
        "bits_at_r2_0.9",
        "code_bits_at_r2_0.9",
        "resid_bits_at_r2_0.9",
        "truncation_bits_at_r2_0.9",
        "intrinsic_atoms",
        "score_kind",
    }
    assert result["intrinsic_atoms"] == 0
    assert result["score_kind"] == "gaussian_surrogate"
    assert result["dictionary_bits"] == 0.0
    assert result["estimation_rows"] == test_x.shape[0]
    assert result["amortization_horizon"] == 1000
    assert all(
        math.isfinite(float(value))
        for key, value in result.items()
        if key != "score_kind"
    )


def test_description_length_rejects_shape_mismatch() -> None:
    fitted = _featurizer()

    with pytest.raises(ValueError, match="same shape"):
        description_length(
            fitted,
            np.ones((fitted.recon.shape[0] + 1, 2)),
            amortization_horizon=1000,
        )


def test_fitted_featurizer_requires_one_code_dimension_per_atom() -> None:
    fitted = FittedFeaturizer(
        name="invalid",
        gate=np.ones((4, 2)),
        atom_contribution=lambda _atom: np.ones((4, 1)),
        code_dims=np.ones(1, dtype=np.int64),
        dictionary_params=0,
        recon=np.ones((4, 1)),
        fit_seconds=0.0,
    )
    with pytest.raises(ValueError, match="one entry per atom"):
        description_length(fitted, np.ones((4, 1)), amortization_horizon=1000)


def test_flat_atom_chart_costs_exactly_its_ambient_price() -> None:
    # A flat atom c = a * w coded through its chart u = a, J = w on an amplitude
    # axis has pullback metric |w|^2 and chart moment E a^2, so its intrinsic
    # spectrum {|w|^2 E a^2} is the ambient rank-one spectrum (#3437).
    rows = 12
    amplitude = 2.0 + np.cos(np.linspace(0.0, 2.0 * np.pi, rows, endpoint=False))
    direction = np.array([1.5, -0.5, 2.0])
    contribution = amplitude[:, None] * direction[None, :]
    base = dict(
        name="flat",
        gate=np.ones((rows, 1)),
        atom_contribution=lambda _atom: contribution,
        code_dims=np.ones(1, dtype=int),
        dictionary_params=0,
        recon=contribution,
        fit_seconds=0.0,
    )
    noise = np.sin(np.arange(rows * 3, dtype=float)).reshape(rows, 3) * 0.3
    test_x = contribution + noise
    ambient = description_length(
        FittedFeaturizer(**base), test_x, amortization_horizon=1000, r2_targets=(0.9,)
    )
    chart = {
        "code": amplitude[:, None],
        "jacobian": np.broadcast_to(direction[None, :, None], (rows, 3, 1)).copy(),
        "axes": ["amplitude"],
    }
    intrinsic = description_length(
        FittedFeaturizer(**base, atom_chart=lambda _atom: chart),
        test_x,
        amortization_horizon=1000,
        r2_targets=(0.9,),
    )
    assert ambient["intrinsic_atoms"] == 0
    assert intrinsic["intrinsic_atoms"] == 1
    assert intrinsic["bits_at_r2_0.9"] == pytest.approx(
        ambient["bits_at_r2_0.9"], rel=1e-12
    )


def test_atom_chart_contract_is_refused_typed() -> None:
    rows = 6
    x = np.arange(rows * 2, dtype=float).reshape(rows, 2)
    fitted = FittedFeaturizer(
        name="bad-chart",
        gate=np.ones((rows, 1)),
        atom_contribution=lambda _atom: x,
        code_dims=np.ones(1, dtype=int),
        dictionary_params=0,
        recon=x,
        fit_seconds=0.0,
        atom_chart=lambda _atom: {
            "code": np.ones((rows, 1)),
            "jacobian": np.ones((rows, 2, 1)),
            "axes": ["polar"],
        },
    )
    with pytest.raises(ValueError, match="polar"):
        description_length(
            fitted,
            x + np.linspace(-0.5, 0.5, rows)[:, None],
            amortization_horizon=1000,
        )
