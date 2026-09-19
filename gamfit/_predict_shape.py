"""Shape policy and dispatcher for ``Model.predict`` return values.

The Rust core hands back a column payload for every prediction -- a dict whose
``columns`` entry maps each prediction column, in the Rust preferred order, to
a float64 ``ndarray`` (plus a handful of structured payloads for survival /
competing-risks). The Python side has to translate that into one of several
public return shapes:

* a 1-D ``ndarray`` of point predictions (the default for standard GAMs,
  Bernoulli marginal-slope, and transformation-normal models);
* a tabular payload (``dict`` / ``DataFrame`` / ``polars`` / ``pyarrow`` /
  ``numpy``) when the caller explicitly opts in via ``return_type=``,
  ``id_column=``, or ``interval=``;
* a structured :class:`SurvivalPrediction` /
  :class:`CompetingRisksPrediction` for survival families (those have their
  own dedicated containers and ignore the shape policy by construction).

Two design rules this module enforces:

1. **Single shape predicate.** :func:`wants_table` is the *only* place that
   maps user intent ``(return_type, id_column, interval)`` to "1-D array vs
   table". Every per-class shaper consults it. Adding a new keyword to
   ``Model.predict`` is a one-line change here.
2. **Policy is driven by caller intent, never by what columns the Rust core
   happened to return.** If the backend emits ``std_error`` without
   being asked, that is a backend bug to fix upstream, not something to
   paper over with column sniffing.

Historically the dispatcher lived in ``_survival.py`` under the name
``shape_prediction_response`` even though it also handled standard GAMs,
transformation-normal models, and Bernoulli marginal-slope. The name lied;
this module owns the shape policy for *every* model class and re-exports
the survival containers from ``_survival.py``.
"""

from __future__ import annotations

from typing import Any

from ._binding import rust_module
from ._survival import (
    competing_risks_prediction_from_ffi_payload,
    survival_prediction_from_ffi_payload,
)


def wants_table(
    *,
    return_type: str | None,
    id_column: str | None,
    interval: float | None,
) -> bool:
    """Return ``True`` when the caller has opted into a tabular return shape.

    These three signals are the *complete* set of public knobs that promote
    ``Model.predict`` from "1-D point estimate" to "full column payload":

    * ``return_type`` — explicit output kind (``"dict"`` / ``"pandas"`` /
      ``"polars"`` / ``"pyarrow"`` / ``"numpy"``).
    * ``id_column`` — propagate an identifier column alongside predictions,
      which only makes sense in a tabular shape.
    * ``interval`` — a credible-interval coverage produces
      ``std_error`` / ``mean_lower`` / ``mean_upper`` columns; those are
      meaningful only as a table.

    (Issue #342: an earlier ``with_uncertainty`` boolean was redundant with
    ``interval`` — coverage and the request to quantify uncertainty are the
    same decision — and was removed.)

    Any other backend-visible state (e.g. presence of a ``std_error``
    column in the payload) is intentionally ignored: shape policy belongs
    to the caller, not the wire format.
    """
    return return_type is not None or id_column is not None or interval is not None


def shape_predict_response(
    payload: dict[str, Any],
    *,
    table_kind: str | None,
    training_table_kind: str,
    interval: float | None,
    return_type: str | None,
    id_column: str | None,
    row_ids: list[str] | None,
    restore: Any,
) -> Any:
    """Dispatch a ``predict_table`` payload to the right per-class shaper.

    Survival and competing-risks payloads are recognised by their class
    discriminator and routed to their structured containers. Every Survival
    predict class takes that route in Rust, so every remaining payload is a
    point payload dispatched on the Rust ``point_shape``. The dispatcher never
    decides shape itself — it picks a shaper and the shaper consults
    :func:`wants_table`.
    """
    payload_class = payload.get("class")
    if payload_class == "survival_prediction":
        return survival_prediction_from_ffi_payload(
            payload, id_column=id_column, row_ids=row_ids
        )
    if payload_class == "competing_risks_prediction":
        return competing_risks_prediction_from_ffi_payload(payload)

    # Rust already ordered the columns and handed each one over as a float64
    # array, so nothing here re-encodes or re-orders the payload.
    columns = payload["columns"]
    point_shape = str(payload["point_shape"])
    point_column = str(payload["point_column"])

    table_requested = wants_table(
        return_type=return_type,
        id_column=id_column,
        interval=interval,
    )

    # Every remaining class is a POINT payload: one scalar per row. They differ
    # only in (a) which column carries the point and how it is transformed, and
    # (b) the column key used when a table is requested. `_point_payload_spec`
    # encodes exactly those two per-class differences; the shared shaper
    # (`_shape_point_payload`) owns the identical "return the vector, or restore
    # a one-column table" tail that the three forked shapers used to duplicate.
    point, table_columns = _point_payload_spec(
        point_shape, point_column, columns, parsed.get("point_columns")
    )
    shaped = _shape_point_payload(
        point,
        table_columns,
        table_requested=table_requested,
        return_type=return_type,
        id_column=id_column,
        row_ids=row_ids,
        table_kind=table_kind,
        training_table_kind=training_table_kind,
        restore=restore,
    )
    shaped = _attach_covariance_provenance(
        shaped, "covariance_source", payload.get("covariance_source")
    )
    # #2296: a curved-link posterior-mean POINT integrates the conditional
    # posterior even when the band is smoothing-corrected — a separate,
    # result-owned fact carried under its own key.
    shaped = _attach_covariance_provenance(
        shaped, "point_covariance_source", payload.get("point_covariance_source")
    )
    # gam#2985: when the fit withheld its covariance, the posterior-mean point
    # says what it is conditional on, under its own key.
    return _attach_covariance_provenance(
        shaped, "point_covariance_note", payload.get("point_covariance_note")
    )


def _attach_covariance_provenance(result: Any, key: str, source: Any) -> Any:
    """Expose a Rust covariance-provenance tag on public Python results.

    The source is prediction metadata, not a row-valued numeric column. Dict
    results therefore receive a scalar key, while pandas stores it in the
    container's metadata mapping so numeric prediction columns keep their
    dtype. Other table implementations still retain the source in the raw FFI
    payload instead of fabricating an in-band numeric encoding.
    """
    if source is None:
        return result
    covariance_source = str(source)
    if isinstance(result, dict):
        result[key] = covariance_source
        return result
    attrs = getattr(result, "attrs", None)
    if isinstance(attrs, dict):
        attrs[key] = covariance_source
    return result


def _point_payload_spec(
    point_shape: str,
    point_column: str,
    columns: dict[str, Any],
    point_columns: list[str] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Resolve a point-payload class to its ``(point_vector, table_columns)``.

    Every non-survival predictive class returns one scalar per row. The only
    per-class differences are which column carries that scalar, how it is
    transformed to the response scale, and which columns make up the tabular
    form when the caller opts into a table:

    * **transformation-normal** — the per-row response-scale conditional mean
      ``E[Y|x]`` (issue #1612), read from the ``mean`` column; table form is the
      single ``mean`` column.  The labelled-data latent score is exposed only
      by ``Model.transformation_score``.
    * **Bernoulli marginal-slope** — ``mean`` clipped back to ``(0, 1)`` as a
      probability; table form is the ``mean`` (probability) column, plus
      ``linear_predictor`` (the η-scale point) and the probability-scale
      ``std_error`` / ``mean_lower`` / ``mean_upper`` (and
      ``observation_lower`` / ``observation_upper``) when the Rust core emitted
      them for an ``interval=`` request (#1049). The credible bounds
      are response-scale (probability) quantiles from the marginal-slope
      coefficient covariance, so they are clipped to ``(0, 1)`` exactly like
      the point ``mean``; ``std_error`` is the probability-scale posterior SE
      (the documented response-scale column, not the η-scale SE) and is left
      untouched.
    * **joint expectile fit** — one curve per expectile level: an ``(n, K)``
      array whose columns are the Rust ``point_columns`` (``expectile_{tau}``,
      increasing level order); table form is the full payload, which carries
      those curves beside the location-scale ``posterior_mean`` and
      ``noise_scale``.
    * **standard GAM / GLM, including the location-scale classes** —
      ``posterior_mean`` as emitted; table form is the *full* Rust
      estimand-explicit payload (``linear_predictor_plugin``, ``mean_plugin``,
      and ``posterior_mean`` always, plus ``posterior_mean_standard_error`` /
      ``posterior_mean_lower`` / ``posterior_mean_upper`` when an interval was
      set, and ``noise_scale`` when the family fits a response-side scale).
      The Rust ``PredictModelClass::point_shape`` / ``point_column`` decide
      which shape and point column a class publishes.

    The shared "return the vector, else restore a table" tail lives in
    :func:`_shape_point_payload`; this function owns only the differences.
    """
    if point_shape == "transformation_normal_mean":
        mean = columns[point_column]
        return mean, {point_column: mean}

    if point_shape == "marginal_slope_probability":
        # The Rust core may emit linear-predictor-scale values that need
        # clipping back to (0, 1) before exposure — the only transformation.
        probs = rust_module().marginal_slope_clip_probabilities(columns[point_column])
        # #1049: when an interval was requested the Rust posterior-mean path
        # emits std_error + response-scale (probability) credible bounds from
        # the marginal-slope coefficient covariance. They were silently dropped
        # here, making predict(interval=) a no-op for this family. Carry them
        # into the table when present, clipping the probability-scale bounds to
        # (0, 1) with the same map as the point mean (std_error is the
        # probability-scale posterior SE — left as emitted), and surface linear_predictor (the η-scale
        # point) alongside so the band can be reconstructed downstream as the
        # TransformEta construction link^{-1}(η ± z·std_error). Without an
        # interval the table form stays the single `mean` (probability) column.
        # The 1-D point vector is always the clipped probability.
        has_interval = "std_error" in columns
        table_columns: dict[str, Any] = {}
        if has_interval and "linear_predictor" in columns:
            table_columns["linear_predictor"] = columns["linear_predictor"]
        table_columns[point_column] = probs
        if has_interval:
            table_columns["std_error"] = columns["std_error"]
            for bound_key in (
                "mean_lower",
                "mean_upper",
                "observation_lower",
                "observation_upper",
            ):
                if bound_key in columns:
                    table_columns[bound_key] = rust_module().marginal_slope_clip_probabilities(
                        columns[bound_key]
                    )
        return probs, table_columns

    if point_shape == "expectile_curves":
        curves = rust_module().column_stack_f64([columns[name] for name in point_columns])
        return curves, columns

    # Standard models keep the full multi-column payload in tabular form.
    return columns[point_column], columns


def _shape_point_payload(
    point: Any,
    table_columns: dict[str, list[Any]],
    *,
    table_requested: bool,
    return_type: str | None,
    id_column: str | None,
    row_ids: list[str] | None,
    table_kind: str | None,
    training_table_kind: str,
    restore: Any,
) -> Any:
    """Shape any point payload: the 1-D ``point`` vector, or its table form.

    Default contract: return the 1-D ``ndarray`` ``point``. When the caller
    opted into a tabular shape, restore ``table_columns`` (with the optional id
    column) through ``restore_output_table``. This is the single shaper for the
    standard, transformation-normal, and Bernoulli marginal-slope classes,
    which differ only in ``point`` / ``table_columns`` (see
    :func:`_point_payload_spec`).
    """
    if not table_requested:
        return point
    return _restore_with_optional_id(
        table_columns,
        id_column=id_column,
        row_ids=row_ids,
        return_type=return_type,
        table_kind=table_kind,
        training_table_kind=training_table_kind,
        restore=restore,
    )


def _restore_with_optional_id(
    columns: dict[str, list[Any]],
    *,
    id_column: str | None,
    row_ids: list[str] | None,
    return_type: str | None,
    table_kind: str | None,
    training_table_kind: str,
    restore: Any,
) -> Any:
    """Tack on the id column (if any) and hand off to ``restore_output_table``."""
    out_columns: dict[str, list[Any]] = dict(columns)
    if id_column is not None:
        out_columns = {id_column: list(row_ids or []), **out_columns}
    return restore(
        out_columns,
        requested=return_type,
        input_kind=table_kind,
        training_kind=training_table_kind,
    )


__all__ = [
    "shape_predict_response",
    "wants_table",
]
