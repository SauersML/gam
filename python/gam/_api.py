from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._binding import RustExtensionUnavailableError, extension_status, rust_module
from ._exceptions import map_exception
from ._model import Model
from ._tables import normalize_table
from ._validation import FormulaValidation


def build_info() -> dict[str, Any]:
    return extension_status()


def fit(
    data: Any,
    formula: str,
    *,
    family: str = "auto",
    offset: str | None = None,
    weights: str | None = None,
    config: dict[str, Any] | None = None,
) -> Model:
    headers, rows, table_kind = normalize_table(data)
    payload = {
        "family": family,
        "offset": offset,
        "weights": weights,
    }
    if config:
        payload.update(config)
    try:
        model_bytes = bytes(
            rust_module().fit_table(headers, rows, formula, json.dumps(payload))
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    return Model(_model_bytes=model_bytes, _training_table_kind=table_kind)


def load(path: str | Path) -> Model:
    model_bytes = Path(path).read_bytes()
    return loads(model_bytes)


def loads(model_bytes: bytes) -> Model:
    try:
        rust_module().load_model(model_bytes)
    except Exception as exc:
        raise map_exception(exc) from exc
    return Model(_model_bytes=model_bytes)


def validate_formula(
    data: Any,
    formula: str,
    *,
    family: str = "auto",
    offset: str | None = None,
    weights: str | None = None,
    config: dict[str, Any] | None = None,
) -> FormulaValidation:
    headers, rows, _table_kind = normalize_table(data)
    payload = {
        "family": family,
        "offset": offset,
        "weights": weights,
    }
    if config:
        payload.update(config)
    try:
        raw = rust_module().validate_formula_json(
            headers,
            rows,
            formula,
            json.dumps(payload),
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    return FormulaValidation.from_dict(json.loads(raw))


def explain_error(exc: BaseException) -> str:
    if isinstance(exc, RustExtensionUnavailableError):
        return "Build the extension with maturin before calling Rust-backed APIs."
    from ._exceptions import FormulaError, GamError, PredictionError, SchemaMismatchError

    if isinstance(exc, FormulaError):
        return "Check the formula syntax and confirm every referenced column exists."
    if isinstance(exc, SchemaMismatchError):
        return "Compare the serving data with the training schema using model.check(...)."
    if isinstance(exc, PredictionError):
        return "Prediction failed. Validate the new data and confirm the fitted model is supported by the Python binding."
    if isinstance(exc, GamError):
        return "The Rust engine returned an error. Inspect the exception message for the underlying failure detail."
    return "Unexpected error. Inspect the full traceback and the original exception message."
