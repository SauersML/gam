from __future__ import annotations

import pytest

import gamfit._sae_spectral as sae


class _ChartInterpRust:
    def __init__(self) -> None:
        self.call: tuple[object, ...] | None = None

    def chart_interp_score(self, *args: object) -> dict[str, object]:
        self.call = args
        return {
            "statistic": "orientation_quotiented_weighted_phase_lock_v1",
            "observed": {
                "circular_correlation": 0.8,
                "signed_circular_correlation": -0.8,
                "effective_weight": 12.0,
            },
            "calibration": {
                "statistic": "orientation_quotiented_weighted_phase_lock_v1",
                "protocol": "matched_spectrum_gaussian_chart_refit_v1",
                "null_kind": "matched_spectrum_gaussian",
                "draw_policy": (
                    "regenerate_surrogate_refit_chart_and_readout_each_draw"
                ),
                "seed": 2250,
                "tail": "larger",
                "draws": 3,
                "observed_statistic": 0.8,
                "mean": 0.5,
                "sd": 0.3,
                "min": 0.1,
                "q25": 0.1,
                "median": 0.5,
                "q75": 0.9,
                "max": 0.9,
                "z": 1.0,
                "p_value": 0.5,
                "monte_carlo_standard_error": 0.25,
                "extreme_draws": 1,
                "null_statistics": [0.9, 0.1, 0.5],
            },
            "significance_level": 0.05,
            "verdict": "null_compatible",
        }


def test_chart_interp_wrapper_preserves_null_provenance_and_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rust = _ChartInterpRust()
    monkeypatch.setattr(sae, "rust_module", lambda: rust)
    observations = [(0.0, 0.0, 1.0), (0.5, 0.5, 1.0)]
    null_draws = [observations, observations, observations]
    calibration = sae.ChartInterpNullCalibration(
        protocol=sae.ChartInterpNullProtocol.MATCHED_SPECTRUM_GAUSSIAN_CHART_REFIT_V1,
        seed=2250,
        expected_draws=3,
        observation_draws=null_draws,
    )

    report = sae.chart_interp_score(observations, calibration, 0.05)

    assert rust.call is not None
    assert rust.call[2] == "matched_spectrum_gaussian_chart_refit_v1"
    assert rust.call[3:5] == (2250, 3)
    assert report.statistic == report.calibration.statistic
    assert report.calibration.null_kind == "matched_spectrum_gaussian"
    assert report.calibration.seed == 2250
    assert report.calibration.null_statistics == (0.9, 0.1, 0.5)
    assert report.verdict == "null_compatible"


def test_chart_interp_wrapper_has_no_scalar_only_call_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rust = _ChartInterpRust()
    monkeypatch.setattr(sae, "rust_module", lambda: rust)

    with pytest.raises(AttributeError, match="observation_draws"):
        sae.chart_interp_score([(0.0, 0.0, 1.0)], 0.973, 0.05)  # type: ignore[arg-type]

    assert rust.call is None
