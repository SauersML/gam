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
   happened to return.** If the backend emits ``effective_se`` without
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
    transformation_normal_z,
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
      ``effective_se`` / ``mean_lower`` / ``mean_upper`` columns; those are
      meaningful only as a table.

    (Issue #342: an earlier ``with_uncertainty`` boolean was redundant with
    ``interval`` — coverage and the request to quantify uncertainty are the
    same decision — and was removed.)

    Any other backend-visible state (e.g. presence of an ``effective_se``
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
    fallback_model_class: str,
    fallback_family: str,
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
    if raw.startswith('{"class":"survival_prediction"'):
        return survival_prediction_from_ffi_payload(
            raw, id_column=id_column, row_ids=row_ids
        )
    if raw.startswith('{"class":"competing_risks_prediction"'):
        return competing_risks_prediction_from_ffi_payload(raw)
    parsed = json.loads(raw)
    if parsed.get("class") == "survival_prediction":
        return survival_prediction_from_ffi_payload(
            raw, id_column=id_column, row_ids=row_ids
        )
    if parsed.get("class") == "competing_risks_prediction":
        return competing_risks_prediction_from_ffi_payload(raw)

    columns_json = json.dumps(parsed["columns"], separators=(",", ":"))
    columns = json.loads(rust_module().ordered_prediction_columns(columns_json))
    model_class = str(parsed.get("model_class") or fallback_model_class)
    family = str(parsed.get("family") or parsed.get("family_kind") or fallback_family)

    table_requested = wants_table(
        return_type=return_type,
        id_column=id_column,
        interval=interval,
    )

    if model_class in _TRANSFORMATION_NORMAL_MODEL_CLASSES:
        return _shape_transformation_normal(
            columns,
            table_requested=table_requested,
            return_type=return_type,
            id_column=id_column,
            row_ids=row_ids,
            table_kind=table_kind,
            training_table_kind=training_table_kind,
            restore=restore,
        )

    if _is_bernoulli_marginal_slope(model_class, family):
        return _shape_bernoulli_marginal_slope(
            columns,
            table_requested=table_requested,
            return_type=return_type,
            id_column=id_column,
            row_ids=row_ids,
            table_kind=table_kind,
            training_table_kind=training_table_kind,
            restore=restore,
        )

    if model_class in _SURVIVAL_MODEL_CLASSES:
        return survival_prediction_from_columns(
            model_class, columns, id_column=id_column, row_ids=row_ids
        )

    return _shape_standard(
        columns,
        table_requested=table_requested,
        return_type=return_type,
        id_column=id_column,
        row_ids=row_ids,
        table_kind=table_kind,
        training_table_kind=training_table_kind,
        restore=restore,
    )


def _shape_standard(
    columns: dict[str, list[Any]],
    *,
    table_requested: bool,
    return_type: str | None,
    id_column: str | None,
    row_ids: list[str] | None,
    table_kind: str | None,
    training_table_kind: str | None,
    restore: Any,
) -> Any:
    """Shape a standard-GAM / GLM column payload.

    Default contract: a 1-D ``ndarray`` of fitted means. When the caller
    opted into a tabular shape, return whatever the Rust core emitted
    (``eta`` + ``mean`` always; plus ``effective_se`` / ``mean_lower`` /
    ``mean_upper`` when uncertainty was requested via the FFI options).
    """
    if not table_requested:
        mean_values = [float(value) for value in columns["mean"]]
        return rust_module().vec_to_array1_f64(mean_values)
    return _restore_with_optional_id(
        columns,
        id_column=id_column,
        row_ids=row_ids,
        return_type=return_type,
        table_kind=table_kind,
        training_table_kind=training_table_kind,
        restore=restore,
    )


def _shape_transformation_normal(
    columns: dict[str, list[Any]],
    *,
    table_requested: bool,
    return_type: str | None,
    id_column: str | None,
    row_ids: list[str] | None,
    table_kind: str | None,
    training_table_kind: str | None,
    restore: Any,
) -> Any:
    """Shape a transformation-normal payload (z-score per row)."""
    z = rust_module().vec_to_array1_f64(
        [float(value) for value in transformation_normal_z(columns)]
    )
    if not table_requested:
        return z
    return _restore_with_optional_id(
        {"z": z.tolist()},
        id_column=id_column,
        row_ids=row_ids,
        return_type=return_type,
        table_kind=table_kind,
        training_table_kind=training_table_kind,
        restore=restore,
    )


def _shape_bernoulli_marginal_slope(
    columns: dict[str, list[Any]],
    *,
    table_requested: bool,
    return_type: str | None,
    id_column: str | None,
    row_ids: list[str] | None,
    table_kind: str | None,
    training_table_kind: str | None,
    restore: Any,
) -> Any:
    """Shape a Bernoulli marginal-slope payload (probability per row).

    The Rust core may emit linear-predictor-scale values that need clipping
    back to ``(0, 1)`` before exposure — that conversion is the only
    transformation done here.
    """
    prob_values = rust_module().marginal_slope_clip_probabilities(
        [float(value) for value in columns.get("mean", [])]
    )
    probs = rust_module().vec_to_array1_f64(prob_values)
    if not table_requested:
        return probs
    return _restore_with_optional_id(
        {"mean": probs.tolist()},
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
