"""Unit tests for the predict-shape dispatcher.

These exercise :func:`gamfit._predict_shape.shape_predict_response` against
synthetic Rust-FFI payloads. The dispatcher's contract is:

* the shape policy is fully determined by caller intent
  ``(return_type, id_column, interval, with_uncertainty)`` via
  :func:`wants_table` — no sniffing of payload columns;
* Bernoulli marginal-slope, transformation-normal, and standard GAMs all
  default to a 1-D ``ndarray`` of point predictions when no tabular knob
  was set, and to a column payload otherwise.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from gamfit import _predict_shape
from gamfit._tables import restore_output_table


class _FakeRust:
    @staticmethod
    def ordered_prediction_columns(columns_json: str) -> str:
        return columns_json

    @staticmethod
    def marginal_slope_clip_probabilities(values: list[float]) -> list[float]:
        return values

    @staticmethod
    def vec_to_array1_f64(values: list[float]) -> Any:
        return np.asarray(values, dtype=float)


def _dispatch(
    monkeypatch: Any,
    raw: str,
    *,
    fallback_model_class: str,
    fallback_family: str,
    interval: float | None = None,
    return_type: str | None = None,
    id_column: str | None = None,
    row_ids: list[str] | None = None,
) -> Any:
    rust = _FakeRust()
    monkeypatch.setattr(_predict_shape, "rust_module", lambda: rust)
    return _predict_shape.shape_predict_response(
        raw,
        headers=[],
        rows=[],
        table_kind="pandas",
        training_table_kind="pandas",
        fallback_model_class=fallback_model_class,
        fallback_family=fallback_family,
        interval=interval,
        return_type=return_type,
        id_column=id_column,
        row_ids=row_ids,
        restore=restore_output_table,
    )


def test_bernoulli_marginal_slope_saved_kind_returns_1d_probabilities(
    monkeypatch: Any,
) -> None:
    raw = json.dumps(
        {
            "columns": {
                "eta": [-1.0, 0.0, 1.0],
                "mean": [0.2, 0.5, 0.8],
            }
        }
    )

    out = _dispatch(
        monkeypatch,
        raw,
        fallback_model_class="marginal-slope",
        fallback_family="bernoulli-marginal-slope",
    )

    arr = np.asarray(out, dtype=float)
    assert arr.shape == (3,)
    np.testing.assert_allclose(arr, [0.2, 0.5, 0.8])


def test_standard_gam_default_returns_1d_mean(monkeypatch: Any) -> None:
    """The default predict shape for a plain GAM is a 1-D ``ndarray``.

    This is the contract issue #329 codified: no ``interval`` /
    ``id_column`` / ``return_type`` means no DataFrame, regardless of
    what auxiliary columns the backend put in the payload. The presence
    of ``effective_se`` here is a *data* artefact and must not flip the
    return shape — that decision is a caller-intent decision only.
    """
    raw = json.dumps(
        {
            "columns": {
                "eta": [-1.0, 0.0, 1.0],
                "mean": [0.2, 0.5, 0.8],
                "effective_se": [0.01, 0.02, 0.03],
            }
        }
    )

    out = _dispatch(
        monkeypatch,
        raw,
        fallback_model_class="standard",
        fallback_family="gaussian",
    )

    arr = np.asarray(out, dtype=float)
    assert arr.shape == (3,)
    np.testing.assert_allclose(arr, [0.2, 0.5, 0.8])


def test_standard_gam_with_return_type_returns_table(monkeypatch: Any) -> None:
    """``return_type="dict"`` alone is enough to opt into the table even
    when ``interval`` / ``id_column`` are ``None``. This pins the shape
    policy as a pure caller-intent decision: every public knob promotes
    the return shape independently."""
    raw = json.dumps(
        {
            "columns": {
                "eta": [-1.0, 0.0, 1.0],
                "mean": [0.2, 0.5, 0.8],
            }
        }
    )

    out = _dispatch(
        monkeypatch,
        raw,
        fallback_model_class="standard",
        fallback_family="gaussian",
        return_type="dict",
    )

    assert isinstance(out, dict)
    assert list(out) == ["eta", "mean"]


def test_standard_gam_with_id_column_returns_table(monkeypatch: Any) -> None:
    """``id_column`` only makes sense alongside a tabular shape, so its
    presence must flip the return shape on its own."""
    raw = json.dumps(
        {
            "columns": {
                "eta": [-1.0, 0.0, 1.0],
                "mean": [0.2, 0.5, 0.8],
            }
        }
    )

    out = _dispatch(
        monkeypatch,
        raw,
        fallback_model_class="standard",
        fallback_family="gaussian",
        id_column="person_id",
    )

    # ``row_ids=None`` is normalised to an empty list by the shaper; the
    # important contract here is that we got a tabular shape with the id
    # column on the left, not a bare 1-D ndarray.
    import pandas as pd

    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["person_id", "eta", "mean"]


def test_standard_gam_with_interval_returns_table(monkeypatch: Any) -> None:
    """``interval=`` alone is also enough to opt into the table — matches
    the documented ``Model.predict`` contract."""
    raw = json.dumps(
        {
            "columns": {
                "eta": [-1.0, 0.0, 1.0],
                "mean": [0.2, 0.5, 0.8],
                "effective_se": [0.01, 0.02, 0.03],
                "mean_lower": [0.19, 0.48, 0.77],
                "mean_upper": [0.21, 0.52, 0.83],
            }
        }
    )

    out = _dispatch(
        monkeypatch,
        raw,
        fallback_model_class="standard",
        fallback_family="gaussian",
        interval=0.95,
    )

    import pandas as pd

    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == [
        "eta",
        "mean",
        "effective_se",
        "mean_lower",
        "mean_upper",
    ]


def test_marginal_slope_non_bernoulli_falls_through_to_standard_shaper(
    monkeypatch: Any,
) -> None:
    """A model with ``model_class='marginal-slope'`` that is *not*
    Bernoulli (here: royston-parmar family) is not a known survival
    class and is not a transformation-normal model, so it lands on the
    standard shaper. Under the principled shape policy that means a 1-D
    mean array by default — the same as any other plain GAM."""
    raw = json.dumps(
        {
            "columns": {
                "eta": [-1.0, 0.0, 1.0],
                "mean": [0.2, 0.5, 0.8],
            }
        }
    )

    out = _dispatch(
        monkeypatch,
        raw,
        fallback_model_class="marginal-slope",
        fallback_family="royston-parmar",
    )

    arr = np.asarray(out, dtype=float)
    assert arr.shape == (3,)
    np.testing.assert_allclose(arr, [0.2, 0.5, 0.8])


def test_wants_table_predicate_is_purely_caller_driven() -> None:
    """:func:`wants_table` is the single source of truth for the shape
    decision. Every opt-in flag flips it to ``True`` independently;
    omitting them all yields ``False``. This test pins the predicate
    directly so future shape-policy work has an unambiguous spec."""
    assert _predict_shape.wants_table(
        return_type=None,
        id_column=None,
        interval=None,
        with_uncertainty=False,
    ) is False
    assert _predict_shape.wants_table(
        return_type="dict",
        id_column=None,
        interval=None,
        with_uncertainty=False,
    ) is True
    assert _predict_shape.wants_table(
        return_type=None,
        id_column="person_id",
        interval=None,
        with_uncertainty=False,
    ) is True
    assert _predict_shape.wants_table(
        return_type=None,
        id_column=None,
        interval=0.95,
        with_uncertainty=False,
    ) is True
    assert _predict_shape.wants_table(
        return_type=None,
        id_column=None,
        interval=None,
        with_uncertainty=True,
    ) is True
