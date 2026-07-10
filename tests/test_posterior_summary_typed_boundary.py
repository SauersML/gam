from __future__ import annotations

from typing import Any

import pytest

np = pytest.importorskip("numpy")

from gamfit._sampling import PosteriorSamples, SamplingConfig


def _draws() -> PosteriorSamples:
    return PosteriorSamples(
        samples=np.asarray(
            [
                [0.0, 10.0, 100.0],
                [2.0, 12.0, 102.0],
                [4.0, 14.0, 104.0],
            ],
            dtype=np.float64,
        ),
        coefficient_names=("intercept", "slope", "curve"),
        mean=np.asarray([2.0, 12.0, 102.0], dtype=np.float64),
        std=np.asarray([1.0, 1.5, 2.0], dtype=np.float64),
        rhat=1.001,
        ess=240.0,
        converged=True,
        method="nuts",
        model_class="standard",
        family_kind="identity",
        config=SamplingConfig(
            n_samples=3,
            n_warmup=2,
            n_chains=1,
            target_accept=0.8,
            seed=7,
        ),
    )


def test_posterior_summary_keeps_numeric_columns_typed_and_records_lazy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intervals = np.asarray(
        [[0.25, 3.75], [9.5, 14.5], [98.0, 106.0]], dtype=np.float64
    )
    calls: list[tuple[str, tuple[Any, ...]]] = []

    def typed_call(name: str, *args: Any) -> Any:
        calls.append((name, args))
        assert name == "posterior_credible_interval"
        assert isinstance(args[0], np.ndarray)
        assert args[0].dtype == np.float64
        return intervals

    monkeypatch.setattr("gamfit._sampling._call", typed_call)

    summary = _draws().summary(level=0.9)
    records = summary.coefficients

    assert [name for name, _args in calls] == ["posterior_credible_interval"]
    assert not isinstance(records, list)
    assert summary["coefficients"] is records
    assert len(records) == 3
    assert records[1] == {
        "index": 1,
        "name": "slope",
        "estimate": 12.0,
        "std_error": 1.5,
        "ci_lower": 9.5,
        "ci_upper": 14.5,
    }
    assert records[-1]["name"] == "curve"
    assert [record["name"] for record in records] == ["intercept", "slope", "curve"]

    materialized = summary.to_dict()["coefficients"]
    assert isinstance(materialized, list)
    assert materialized[0]["ci_lower"] == pytest.approx(0.25)
    assert summary.coefficients is records


def test_posterior_summary_dataframe_is_built_from_typed_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pandas")
    intervals = np.asarray(
        [[0.25, 3.75], [9.5, 14.5], [98.0, 106.0]], dtype=np.float64
    )

    def typed_call(name: str, *_args: Any) -> Any:
        assert name == "posterior_credible_interval"
        return intervals

    monkeypatch.setattr("gamfit._sampling._call", typed_call)
    frame = _draws().summary(level=0.9).coefficients_frame()

    assert list(frame.columns) == [
        "index",
        "name",
        "estimate",
        "std_error",
        "ci_lower",
        "ci_upper",
    ]
    np.testing.assert_array_equal(frame["index"].to_numpy(), np.arange(3))
    np.testing.assert_allclose(frame["estimate"].to_numpy(), [2.0, 12.0, 102.0])
    np.testing.assert_allclose(frame["ci_upper"].to_numpy(), intervals[:, 1])
