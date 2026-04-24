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


def _build_fit_payload(
    *,
    family: str,
    offset: str | None,
    weights: str | None,
    transformation_normal: bool | None,
    survival_likelihood: str | None,
    baseline_target: str | None,
    baseline_scale: float | None,
    baseline_shape: float | None,
    baseline_rate: float | None,
    baseline_makeham: float | None,
    z_column: str | None,
    link: str | None,
    logslope_formula: str | None,
    scale_dimensions: str | None,
    disable_link_dev: bool | None,
    disable_score_warp: bool | None,
    firth: bool | None,
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "family": family,
        "offset": offset,
        "weights": weights,
    }
    kwarg_items: dict[str, Any] = {
        "transformation_normal": transformation_normal,
        "survival_likelihood": survival_likelihood,
        "baseline_target": baseline_target,
        "baseline_scale": baseline_scale,
        "baseline_shape": baseline_shape,
        "baseline_rate": baseline_rate,
        "baseline_makeham": baseline_makeham,
        "z_column": z_column,
        "link": link,
        "logslope_formula": logslope_formula,
        "scale_dimensions": scale_dimensions,
        "disable_link_dev": disable_link_dev,
        "disable_score_warp": disable_score_warp,
        "firth": firth,
    }
    for key, value in kwarg_items.items():
        if value is not None:
            payload[key] = value
    if config:
        for key, value in config.items():
            payload.setdefault(key, value)
    return payload


def fit(
    data: Any,
    formula: str,
    *,
    family: str = "auto",
    offset: str | None = None,
    weights: str | None = None,
    transformation_normal: bool | None = None,
    survival_likelihood: str | None = None,
    baseline_target: str | None = None,
    baseline_scale: float | None = None,
    baseline_shape: float | None = None,
    baseline_rate: float | None = None,
    baseline_makeham: float | None = None,
    z_column: str | None = None,
    link: str | None = None,
    logslope_formula: str | None = None,
    scale_dimensions: str | None = None,
    disable_link_dev: bool | None = None,
    disable_score_warp: bool | None = None,
    firth: bool | None = None,
    config: dict[str, Any] | None = None,
) -> Model:
    """Fit a GAM model from a formula and a tabular dataset.

    Parameters
    ----------
    data:
        Input table. Accepts a pandas DataFrame, pyarrow Table, dict of columns,
        list of records, or any object normalize_table understands.
    formula:
        Wilkinson-style formula string (e.g. ``"y ~ s(x1) + te(x2, x3)"``).
    family:
        Likelihood family, or ``"auto"`` to infer from the response. Corresponds
        to the ``--family`` CLI flag.
    offset:
        Name of the offset column. Corresponds to ``--offset-column``.
    weights:
        Name of the observation-weight column. Corresponds to ``--weights-column``.
    transformation_normal:
        Fit a conditional transformation-normal model (``h(Y|x) ~ N(0,1))``).
        Corresponds to ``--transformation-normal``.
    survival_likelihood:
        Survival likelihood formulation. One of ``"transformation"``,
        ``"marginal-slope"``, ``"location-scale"``, ``"weibull"``. Corresponds to
        ``--survival-likelihood``.
    baseline_target:
        Parametric baseline target for survival models. One of ``"linear"``,
        ``"weibull"``, ``"gompertz"``, ``"gompertz-makeham"``. Corresponds to
        ``--baseline-target``.
    baseline_scale:
        Weibull baseline scale (>0) when ``baseline_target="weibull"``.
        Corresponds to ``--baseline-scale``.
    baseline_shape:
        Weibull baseline shape (>0). Corresponds to ``--baseline-shape``.
    baseline_rate:
        Gompertz hazard rate (>0) when ``baseline_target`` is ``"gompertz"``
        or ``"gompertz-makeham"``. Corresponds to ``--baseline-rate``.
    baseline_makeham:
        Makeham additive hazard (>0) when ``baseline_target="gompertz-makeham"``.
        Corresponds to ``--baseline-makeham``.
    z_column:
        Name of the latent/observed z-score column used by score-warp families
        and latent transformation models. Corresponds to ``--z-column``.
    link:
        Override the default link function. Corresponds to ``--link``.
    logslope_formula:
        Secondary formula for the logslope / score-warp submodel. Corresponds to
        ``--logslope-formula``.
    scale_dimensions:
        Per-axis anisotropic length-scale selector. Either ``"auto"`` or a
        comma-separated list. Corresponds to ``--scale-dimensions``.
    disable_link_dev:
        Disable the link-deviation correction. Corresponds to
        ``--disable-link-dev``.
    disable_score_warp:
        Disable routing of ``linkwiggle(...)`` in ``--logslope-formula`` into
        the anchored internal score-warp block. Corresponds to
        ``--disable-score-warp``.
    firth:
        Enable Firth bias-reduced estimation. Corresponds to ``--firth``.
    config:
        Escape-hatch dict of extra pipeline keys. Any key already set via a
        dedicated kwarg wins over the same key in ``config``.

    Returns
    -------
    Model
        A fitted model object with ``predict``, ``summary``, and save/load
        helpers.
    """
    headers, rows, table_kind = normalize_table(data)
    payload = _build_fit_payload(
        family=family,
        offset=offset,
        weights=weights,
        transformation_normal=transformation_normal,
        survival_likelihood=survival_likelihood,
        baseline_target=baseline_target,
        baseline_scale=baseline_scale,
        baseline_shape=baseline_shape,
        baseline_rate=baseline_rate,
        baseline_makeham=baseline_makeham,
        z_column=z_column,
        link=link,
        logslope_formula=logslope_formula,
        scale_dimensions=scale_dimensions,
        disable_link_dev=disable_link_dev,
        disable_score_warp=disable_score_warp,
        firth=firth,
        config=config,
    )
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
    transformation_normal: bool | None = None,
    survival_likelihood: str | None = None,
    baseline_target: str | None = None,
    baseline_scale: float | None = None,
    baseline_shape: float | None = None,
    baseline_rate: float | None = None,
    baseline_makeham: float | None = None,
    z_column: str | None = None,
    link: str | None = None,
    logslope_formula: str | None = None,
    scale_dimensions: str | None = None,
    disable_link_dev: bool | None = None,
    disable_score_warp: bool | None = None,
    firth: bool | None = None,
    config: dict[str, Any] | None = None,
) -> FormulaValidation:
    """Validate a formula against a dataset without fitting.

    Accepts every pipeline kwarg that :func:`fit` accepts, with identical
    semantics. See :func:`fit` for parameter documentation.
    """
    headers, rows, _table_kind = normalize_table(data)
    payload = _build_fit_payload(
        family=family,
        offset=offset,
        weights=weights,
        transformation_normal=transformation_normal,
        survival_likelihood=survival_likelihood,
        baseline_target=baseline_target,
        baseline_scale=baseline_scale,
        baseline_shape=baseline_shape,
        baseline_rate=baseline_rate,
        baseline_makeham=baseline_makeham,
        z_column=z_column,
        link=link,
        logslope_formula=logslope_formula,
        scale_dimensions=scale_dimensions,
        disable_link_dev=disable_link_dev,
        disable_score_warp=disable_score_warp,
        firth=firth,
        config=config,
    )
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
