"""Shape policy and dispatcher for ``Model.predict`` return values.

The Rust core hands back a column-payload JSON for every prediction (plus a
handful of structured payloads for survival / competing-risks). The Python
side has to translate that into one of several public return shapes:

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

import json
from typing import Any

from ._binding import rust_module
from ._survival import (
    _BERNOULLI_FAMILY_PREFIXES,
    _SURVIVAL_MODEL_CLASSES,
    _TRANSFORMATION_NORMAL_MODEL_CLASSES,
    competing_risks_prediction_from_ffi_payload,
    survival_prediction_from_columns,
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
    raw: str,
    *,
    headers: list[str],
    rows: list[list[str]],
    table_kind: str | None,
    training_table_kind: str | None,
    interval: float | None,
    return_type: str | None,
    id_column: str | None,
    row_ids: list[str] | None,
    restore: Any,
) -> Any:
    """Dispatch a ``predict_table`` payload to the right per-class shaper.

    Survival and competing-risks payloads are recognised by their class
    discriminator and routed to their structured containers; everything
    else is dispatched on ``model_class`` (with ``family`` consulted for
    Bernoulli marginal-slope, whose model_class overlaps with the survival
    marginal-slope variant). The dispatcher never decides shape itself —
    it picks a shaper and the shaper consults :func:`wants_table`.
    """
    parsed = json.loads(raw)
    payload_class = parsed.get("class")
    if payload_class == "survival_prediction":
        return survival_prediction_from_ffi_payload(
            raw, id_column=id_column, row_ids=row_ids
        )
    if payload_class == "competing_risks_prediction":
        return competing_risks_prediction_from_ffi_payload(raw)

    columns_json = json.dumps(parsed["columns"], separators=(",", ":"))
    columns = json.loads(rust_module().ordered_prediction_columns(columns_json))
    model_class = str(parsed["model_class"])
    family = str(parsed["family"])

    table_requested = wants_table(
        return_type=return_type,
        id_column=id_column,
        interval=interval,
    )

    # Survival column payloads (no structured FFI container) keep their own
    # tabular shaper: they expand into a per-time grid, not a single point
    # vector, so they are a genuinely distinct shape — not a point payload.
    if (
        model_class in _SURVIVAL_MODEL_CLASSES
        and not _is_bernoulli_marginal_slope(model_class, family)
    ):
        return survival_prediction_from_columns(
            model_class, columns, id_column=id_column, row_ids=row_ids
        )

    # Every remaining class is a POINT payload: one scalar per row. They differ
    # only in (a) which column carries the point and how it is transformed, and
    # (b) the column key used when a table is requested. `_point_payload_spec`
    # encodes exactly those two per-class differences; the shared shaper
    # (`_shape_point_payload`) owns the identical "return the vector, or restore
    # a one-column table" tail that the three forked shapers used to duplicate.
    point, table_columns = _point_payload_spec(model_class, family, columns)
    return _shape_point_payload(
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


def _point_payload_spec(
    model_class: str,
    family: str,
    columns: dict[str, list[Any]],
) -> tuple[Any, dict[str, list[Any]]]:
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
      the point ``mean``; ``std_error`` is the η-scale SE and is left untouched.
    * **standard GAM / GLM** — ``mean`` as emitted; table form is the *full*
      Rust column payload (``linear_predictor`` + ``mean`` always, plus
      ``std_error`` / ``mean_lower`` / ``mean_upper`` when an interval was set).

    The shared "return the vector, else restore a table" tail lives in
    :func:`_shape_point_payload`; this function owns only the differences.
    """
    if model_class in _TRANSFORMATION_NORMAL_MODEL_CLASSES:
        mean = rust_module().vec_to_array1_f64(
            [float(value) for value in columns["mean"]]
        )
        return mean, {"mean": mean.tolist()}

    if _is_bernoulli_marginal_slope(model_class, family):
        # The Rust core may emit linear-predictor-scale values that need
        # clipping back to (0, 1) before exposure — the only transformation.
        prob_values = rust_module().marginal_slope_clip_probabilities(
            [float(value) for value in columns.get("mean", [])]
        )
        probs = rust_module().vec_to_array1_f64(prob_values)
        # #1049: when an interval was requested the Rust posterior-mean path
        # emits std_error + response-scale (probability) credible bounds from
        # the marginal-slope coefficient covariance. They were silently dropped
        # here, making predict(interval=) a no-op for this family. Carry them
        # into the table when present, clipping the probability-scale bounds to
        # (0, 1) with the same map as the point mean (std_error is the η-scale
        # SE — left as emitted), and surface linear_predictor (the η-scale
        # point) alongside so the band can be reconstructed downstream as the
        # TransformEta construction link^{-1}(η ± z·std_error). Without an
        # interval the table form stays the single `mean` (probability) column.
        # The 1-D point vector is always the clipped probability.
        has_interval = "std_error" in columns
        table_columns: dict[str, list[Any]] = {}
        if has_interval and "linear_predictor" in columns:
            table_columns["linear_predictor"] = [
                float(value) for value in columns["linear_predictor"]
            ]
        table_columns["mean"] = probs.tolist()
        if has_interval:
            table_columns["std_error"] = [
                float(value) for value in columns["std_error"]
            ]
            for bound_key in (
                "mean_lower",
                "mean_upper",
                "observation_lower",
                "observation_upper",
            ):
                if bound_key in columns:
                    table_columns[bound_key] = rust_module().marginal_slope_clip_probabilities(
                        [float(value) for value in columns[bound_key]]
                    )
        return probs, table_columns

    mean = rust_module().vec_to_array1_f64(
        [float(value) for value in columns["mean"]]
    )
    # Standard models keep the full multi-column payload in tabular form.
    return mean, columns


def _shape_point_payload(
    point: Any,
    table_columns: dict[str, list[Any]],
    *,
    table_requested: bool,
    return_type: str | None,
    id_column: str | None,
    row_ids: list[str] | None,
    table_kind: str | None,
    training_table_kind: str | None,
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
    training_table_kind: str | None,
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


def _is_bernoulli_marginal_slope(model_class: str, family: str) -> bool:
    """Distinguish Bernoulli marginal-slope from survival marginal-slope.

    The two share ``"marginal-slope"`` as a model_class label in some
    code paths; the family discriminator (``"bernoulli"`` /
    ``"binomial"`` prefix) settles which shaper to use.
    """
    normalized_family = family.strip().lower().replace("_", "-")
    return (
        model_class == "bernoulli marginal-slope"
        or (
            model_class == "marginal-slope"
            and normalized_family.startswith(_BERNOULLI_FAMILY_PREFIXES)
        )
    )


__all__ = [
    "shape_predict_response",
    "wants_table",
]
