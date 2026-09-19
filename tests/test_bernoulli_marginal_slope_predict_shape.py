"""Unit tests for the predict-shape dispatcher.

These exercise :func:`gamfit._predict_shape.shape_predict_response` against
synthetic Rust-FFI payloads. The dispatcher's contract is:

* the shape policy is fully determined by caller intent
  ``(return_type, id_column, interval)`` via :func:`wants_table` — no
  sniffing of payload columns (issue #342 collapsed an earlier
  ``with_uncertainty`` flag into ``interval``);
* the per-class shaper is chosen by the ``point_shape`` and ``point_column``
  the Rust ``PredictModelClass`` publishes on the wire, never by the
  ``model_class`` / ``family`` labels (#2899);
* Bernoulli marginal-slope, transformation-normal, and standard GAMs all
  default to a 1-D ``ndarray`` of point predictions when no tabular knob
  was set, and to a column payload otherwise.

The synthetic payloads mirror what ``predict_table`` hands back: a dict whose
``columns`` map each column name to a float64 ``ndarray``, already in the Rust
preferred order.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gamfit import _predict_shape
from gamfit._tables import PredictionResult, restore_output_table


class _FakeRust:
    @staticmethod
    def marginal_slope_clip_probabilities(values: np.ndarray) -> np.ndarray:
        return values


def _dispatch(
    monkeypatch: Any,
    raw: dict[str, Any],
    *,
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
        interval=interval,
        return_type=return_type,
        id_column=id_column,
        row_ids=row_ids,
        restore=restore_output_table,
    )


def _payload(
    model_class: str,
    family: str,
    point_shape: str,
    point_column: str,
    columns: dict[str, list[float]],
    *,
    covariance_source: str | None = None,
) -> dict[str, Any]:
    """A point payload carrying the fields the FFI ``PredictionPayload`` serializes.

    ``point_shape`` / ``point_column`` are what ``PredictModelClass::point_shape``
    and ``point_column`` publish: ``estimand_explicit`` / ``posterior_mean`` for
    the standard and location-scale classes, ``marginal_slope_probability`` /
    ``mean`` for Bernoulli marginal-slope, ``transformation_normal_mean`` /
    ``mean`` for transformation-normal.
    """
    payload: dict[str, Any] = {
        "model_class": model_class,
        "family": family,
        "point_shape": point_shape,
        "point_column": point_column,
        "columns": {
            name: np.asarray(values, dtype=np.float64) for name, values in columns.items()
        },
    }
    if covariance_source is not None:
        payload["covariance_source"] = covariance_source
    return payload


def test_interval_dict_exposes_exact_covariance_provenance(monkeypatch: Any) -> None:
    raw = _payload(
        "standard",
        "identity",
        "estimand_explicit",
        "posterior_mean",
        {
            "linear_predictor_plugin": [1.0, 2.0],
            "posterior_mean": [1.0, 2.0],
            "posterior_mean_standard_error": [0.1, 0.2],
            "posterior_mean_lower": [0.8, 1.6],
            "posterior_mean_upper": [1.2, 2.4],
        },
        covariance_source="smoothing-corrected",
    )

    out = _dispatch(monkeypatch, raw, interval=0.95, return_type="dict")

    assert isinstance(out, PredictionResult)
    assert out["covariance_source"] == "smoothing-corrected"
    assert out.covariance_source == "smoothing-corrected"


def test_bernoulli_marginal_slope_saved_kind_returns_1d_probabilities(
    monkeypatch: Any,
) -> None:
    raw = _payload(
        "marginal-slope",
        "bernoulli-marginal-slope",
        "marginal_slope_probability",
        "mean",
        {
            "linear_predictor": [-1.0, 0.0, 1.0],
            "mean": [0.2, 0.5, 0.8],
        },
    )

    out = _dispatch(
        monkeypatch,
        raw,
    )

    arr = np.asarray(out, dtype=float)
    assert arr.shape == (3,)
    np.testing.assert_allclose(arr, [0.2, 0.5, 0.8])


def test_transformation_normal_predict_shapes_response_mean_not_latent_score(
    monkeypatch: Any,
) -> None:
    """A CTM predict payload is response-scale ``E[Y|x]``.  Its linear
    predictor slot is not an observed-response PIT score; labelled-data scores
    are available only from ``Model.transformation_score``."""
    raw = _payload(
        "transformation-normal",
        "transformation-normal",
        "transformation_normal_mean",
        "mean",
        {
            "linear_predictor": [-3.0, 4.0],
            "mean": [12.5, 18.25],
        },
    )

    point = _dispatch(monkeypatch, raw)
    np.testing.assert_allclose(np.asarray(point), [12.5, 18.25])

    table = _dispatch(monkeypatch, raw, return_type="dict")
    assert list(table) == ["mean"]
    np.testing.assert_allclose(table["mean"], [12.5, 18.25])


def test_bernoulli_marginal_slope_interval_carries_clipped_bounds(
    monkeypatch: Any,
) -> None:
    """Issue #1049: when an ``interval=`` request makes the Rust core emit
    ``std_error`` + response-scale credible bounds, the marginal-slope shaper
    must carry them into the table (it previously dropped everything but
    ``mean``). The probability-scale bounds are clipped to ``[0, 1]`` with the
    same map as the point ``mean``; ``std_error`` (the probability-scale posterior SE) is passed
    through untouched."""

    class _ClippingRust(_FakeRust):
        @staticmethod
        def marginal_slope_clip_probabilities(values: np.ndarray) -> np.ndarray:
            return np.clip(values, 0.0, 1.0)

    rust = _ClippingRust()
    monkeypatch.setattr(_predict_shape, "rust_module", lambda: rust)

    raw = _payload(
        "marginal-slope",
        "bernoulli-marginal-slope",
        "marginal_slope_probability",
        "mean",
        {
            "linear_predictor": [-1.0, 0.0, 1.0],
            # `mean` and the bounds are response-scale (probability) values; the
            # bounds straddle [0, 1] to exercise the clip.
            "mean": [0.16, 0.50, 0.84],
            "std_error": [0.30, 0.20, 0.30],
            "mean_lower": [-0.05, 0.42, 0.70],
            "mean_upper": [0.40, 0.58, 1.10],
        },
    )

    out = _predict_shape.shape_predict_response(
        raw,
        headers=[],
        rows=[],
        table_kind="pandas",
        training_table_kind="pandas",
        interval=0.95,
        return_type="dict",
        id_column=None,
        row_ids=None,
        restore=restore_output_table,
    )

    assert isinstance(out, dict)
    assert isinstance(out, PredictionResult)
    assert set(out) == {
        "linear_predictor",
        "mean",
        "std_error",
        "mean_lower",
        "mean_upper",
    }
    # linear_predictor is the η-scale point — passed through untouched so the
    # band reconstructs as link^{-1}(η ± z·std_error).
    np.testing.assert_allclose(out["linear_predictor"], [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(out["mean"], [0.16, 0.50, 0.84])
    np.testing.assert_allclose(out.mean, [0.16, 0.50, 0.84])
    # std_error is the probability-scale posterior SE — untouched, not clipped.
    np.testing.assert_allclose(out["std_error"], [0.30, 0.20, 0.30])
    np.testing.assert_allclose(out.std_error, [0.30, 0.20, 0.30])
    # Probability-scale bounds clipped into [0, 1].
    np.testing.assert_allclose(out["mean_lower"], [0.0, 0.42, 0.70])
    np.testing.assert_allclose(out["mean_upper"], [0.40, 0.58, 1.0])
    np.testing.assert_allclose(out.mean_lower, [0.0, 0.42, 0.70])
    np.testing.assert_allclose(out.mean_upper, [0.40, 0.58, 1.0])


def test_withheld_fit_point_note_reaches_the_prediction_dict_2985(monkeypatch: Any) -> None:
    """gam#2985: a fit that withheld its covariance predicts its posterior mean
    conditional on the fitted latent law, and the FFI payload says so under
    ``point_covariance_note``; the dict result carries that note beside
    ``point_covariance_source``, and a payload without it adds no key."""
    columns = {"linear_predictor": [-0.2, 0.3], "mean": [0.43, 0.61]}
    note = (
        "posterior mean conditional on the fitted latent law; the generated-regressor "
        "correction was declined: no coefficient covariance was published"
    )
    withheld = _payload(
        "marginal-slope",
        "bernoulli-marginal-slope",
        "marginal_slope_probability",
        "mean",
        columns,
    )
    withheld["point_covariance_source"] = "conditional"
    withheld["point_covariance_note"] = note

    out = _dispatch(monkeypatch, withheld, return_type="dict")

    assert isinstance(out, PredictionResult)
    assert out["point_covariance_source"] == "conditional"
    assert out["point_covariance_note"] == note

    published = dict(withheld)
    del published["point_covariance_note"]
    plain = _dispatch(monkeypatch, published, return_type="dict")
    assert "point_covariance_note" not in plain


def test_bernoulli_marginal_slope_no_interval_stays_1d(monkeypatch: Any) -> None:
    """Without an interval request the marginal-slope point payload is still a
    bare 1-D probability vector even if the backend volunteered extra columns —
    the #1049 fix only surfaces bounds, it does not flip the no-interval
    shape."""
    raw = _payload(
        "marginal-slope",
        "bernoulli-marginal-slope",
        "marginal_slope_probability",
        "mean",
        {
            "linear_predictor": [-1.0, 0.0, 1.0],
            "mean": [0.2, 0.5, 0.8],
        },
    )

    out = _dispatch(monkeypatch, raw)

    arr = np.asarray(out, dtype=float)
    assert arr.shape == (3,)
    np.testing.assert_allclose(arr, [0.2, 0.5, 0.8])


def test_standard_gam_default_returns_1d_mean(monkeypatch: Any) -> None:
    """The default predict shape for a plain GAM is a 1-D ``ndarray``.

    This is the contract issue #329 codified: no ``interval`` /
    ``id_column`` / ``return_type`` means no DataFrame, regardless of
    what auxiliary columns the backend put in the payload. The presence
    of ``std_error`` here is a *data* artefact and must not flip the
    return shape — that decision is a caller-intent decision only.
    """
    raw = _payload(
        "standard",
        "gaussian",
        "estimand_explicit",
        "posterior_mean",
        {
            "linear_predictor_plugin": [-1.0, 0.0, 1.0],
            "posterior_mean": [0.2, 0.5, 0.8],
            "posterior_mean_standard_error": [0.01, 0.02, 0.03],
        },
    )

    out = _dispatch(monkeypatch, raw)

    arr = np.asarray(out, dtype=float)
    assert arr.shape == (3,)
    np.testing.assert_allclose(arr, [0.2, 0.5, 0.8])


def test_standard_gam_with_return_type_returns_table(monkeypatch: Any) -> None:
    """``return_type="dict"`` alone is enough to opt into the table even
    when ``interval`` / ``id_column`` are ``None``. This pins the shape
    policy as a pure caller-intent decision: every public knob promotes
    the return shape independently."""
    raw = _payload(
        "standard",
        "gaussian",
        "estimand_explicit",
        "posterior_mean",
        {
            "linear_predictor_plugin": [-1.0, 0.0, 1.0],
            "posterior_mean": [0.2, 0.5, 0.8],
        },
    )

    out = _dispatch(
        monkeypatch,
        raw,
        return_type="dict",
    )

    assert isinstance(out, dict)
    assert isinstance(out, PredictionResult)
    assert list(out) == ["linear_predictor_plugin", "posterior_mean"]
    np.testing.assert_allclose(out.posterior_mean, [0.2, 0.5, 0.8])


def test_standard_gam_with_id_column_returns_table(monkeypatch: Any) -> None:
    """``id_column`` only makes sense alongside a tabular shape, so its
    presence must flip the return shape on its own."""
    raw = _payload(
        "standard",
        "gaussian",
        "estimand_explicit",
        "posterior_mean",
        {
            "linear_predictor_plugin": [-1.0, 0.0, 1.0],
            "posterior_mean": [0.2, 0.5, 0.8],
        },
    )

    out = _dispatch(
        monkeypatch,
        raw,
        id_column="person_id",
        row_ids=["a", "b", "c"],
    )

    # The id column lands on the left so callers can join predictions
    # back to the input by row; the data columns follow in their
    # preferred order.
    import pandas as pd

    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["person_id", "linear_predictor_plugin", "posterior_mean"]
    assert out["person_id"].tolist() == ["a", "b", "c"]


def test_standard_gam_with_interval_returns_table(monkeypatch: Any) -> None:
    """``interval=`` alone is also enough to opt into the table — matches
    the documented ``Model.predict`` contract."""
    raw = _payload(
        "standard",
        "gaussian",
        "estimand_explicit",
        "posterior_mean",
        {
            "linear_predictor_plugin": [-1.0, 0.0, 1.0],
            "posterior_mean": [0.2, 0.5, 0.8],
            "posterior_mean_standard_error": [0.01, 0.02, 0.03],
            "posterior_mean_lower": [0.19, 0.48, 0.77],
            "posterior_mean_upper": [0.21, 0.52, 0.83],
        },
    )

    out = _dispatch(
        monkeypatch,
        raw,
        interval=0.95,
    )

    import pandas as pd

    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == [
        "linear_predictor_plugin",
        "posterior_mean",
        "posterior_mean_standard_error",
        "posterior_mean_lower",
        "posterior_mean_upper",
    ]


def test_marginal_slope_non_bernoulli_falls_through_to_standard_shaper(
    monkeypatch: Any,
) -> None:
    """The shaper follows the Rust ``point_shape``, not the labels (#2899).

    Here the labels read ``model_class='marginal-slope'`` with a
    royston-parmar family, but the payload publishes the estimand-explicit
    schema, so it lands on the standard shaper: a 1-D ``posterior_mean`` array
    by default, the same as any other plain GAM, and the full estimand-explicit
    payload as its table. Both shapers read the point from the published
    ``point_column``, so the point alone cannot tell them apart; the table
    can, because the marginal-slope shaper keeps only its probability column."""
    raw = _payload(
        "marginal-slope",
        "royston-parmar",
        "estimand_explicit",
        "posterior_mean",
        {
            "linear_predictor_plugin": [-1.0, 0.0, 1.0],
            "posterior_mean": [0.2, 0.5, 0.8],
        },
    )

    out = _dispatch(monkeypatch, raw)

    arr = np.asarray(out, dtype=float)
    assert arr.shape == (3,)
    np.testing.assert_allclose(arr, [0.2, 0.5, 0.8])

    table = _dispatch(monkeypatch, raw, return_type="dict")
    assert list(table) == ["linear_predictor_plugin", "posterior_mean"]


def test_wants_table_predicate_is_purely_caller_driven() -> None:
    """:func:`wants_table` is the single source of truth for the shape
    decision. Each of the three public knobs flips it to ``True``
    independently; omitting them all yields ``False``. This test pins
    the predicate so future shape-policy work has an unambiguous spec.

    Issue #342 collapsed the prior ``with_uncertainty`` flag into the
    single ``interval`` knob; the predicate's three signals are exactly
    the public ``Model.predict`` keywords that can promote the return
    shape, and nothing else.
    """
    assert _predict_shape.wants_table(
        return_type=None,
        id_column=None,
        interval=None,
    ) is False
    assert _predict_shape.wants_table(
        return_type="dict",
        id_column=None,
        interval=None,
    ) is True
    assert _predict_shape.wants_table(
        return_type=None,
        id_column="person_id",
        interval=None,
    ) is True
    assert _predict_shape.wants_table(
        return_type=None,
        id_column=None,
        interval=0.95,
    ) is True
