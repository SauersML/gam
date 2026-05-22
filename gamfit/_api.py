from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, overload

from ._binding import RustExtensionUnavailableError, extension_status, rust_module
from ._cuda import cuda_diagnostics as _cuda_diagnostics
from ._cuda import format_cuda_diagnostics as _format_cuda_diagnostics
from ._exceptions import map_exception
from ._model import Model
from ._response_geometry import ResponseGeometryModel, fit_response_geometry
from ._tables import normalize_table
from ._validation import FormulaValidation


@dataclass(frozen=True)
class SharedPrecisionGroup:
    """Cross-fit coefficient precision group.

    ``name`` is the shared precision coordinate. By default it selects the
    same named coefficient term/column/label in every model. ``labels`` can
    override that with either one label for all models or a mapping keyed by
    the model name/index supplied to :func:`cross_fit_shared_precision_groups`.
    """

    name: str
    shape: float = 1.0
    rate: float = 0.0
    labels: str | Mapping[str | int, str] | None = None


def _normalize_shared_precision_group(
    value: Any,
    default_name: str | None = None,
) -> SharedPrecisionGroup:
    if isinstance(value, SharedPrecisionGroup):
        return value
    if isinstance(value, Mapping):
        name = value.get("name", value.get("label", default_name))
        if name is None:
            raise ValueError("shared precision group mapping needs a name/label")
        shape = value.get("shape", value.get("a", value.get("a_p", 1.0)))
        rate = value.get("rate", value.get("b", value.get("b_p", 0.0)))
        labels = value.get("labels", value.get("terms", value.get("selectors")))
        return SharedPrecisionGroup(
            name=str(name),
            shape=float(shape),
            rate=float(rate),
            labels=labels,
        )
    if default_name is not None:
        shape, rate = value
        return SharedPrecisionGroup(default_name, float(shape), float(rate))
    raise TypeError("shared precision groups must be SharedPrecisionGroup or mapping entries")


def _normalize_shared_precision_groups(groups: Any) -> list[SharedPrecisionGroup]:
    if isinstance(groups, Mapping):
        normalized = [
            _normalize_shared_precision_group(value, str(name))
            for name, value in groups.items()
        ]
    else:
        normalized = [_normalize_shared_precision_group(value) for value in groups]
    seen: set[str] = set()
    duplicates: list[str] = []
    for group in normalized:
        if group.name in seen:
            duplicates.append(group.name)
        seen.add(group.name)
    if duplicates:
        raise ValueError(
            "duplicate shared precision group name(s): " + ", ".join(sorted(set(duplicates)))
        )
    return normalized


def _normalize_model_mapping(models: Any) -> list[tuple[str | int, Model]]:
    if isinstance(models, Mapping):
        items = list(models.items())
    else:
        items = list(enumerate(models))
    if not items:
        raise ValueError("at least one model is required")
    out: list[tuple[str | int, Model]] = []
    for key, model in items:
        if not isinstance(model, Model):
            raise TypeError("cross-fit shared precision groups require Model instances")
        out.append((key, model))
    return out


def _shared_group_label(group: SharedPrecisionGroup, model_key: str | int) -> str:
    labels = group.labels
    if labels is None:
        return group.name
    if isinstance(labels, str):
        return labels
    if model_key in labels:
        return str(labels[model_key])
    key_text = str(model_key)
    if key_text in labels:
        return str(labels[key_text])
    raise ValueError(
        f"shared precision group {group.name!r} has no label for model {model_key!r}"
    )


def _coefficient_indices_for_precision_label(state: Mapping[str, Any], label: str) -> list[int]:
    provenance = state.get("coefficient_provenance")
    if not isinstance(provenance, list):
        raise ValueError("model coefficient state does not include coefficient provenance")
    matches: list[int] = []
    for fallback_index, item in enumerate(provenance):
        if not isinstance(item, Mapping):
            continue
        index = int(item.get("index", fallback_index))
        candidates = (
            item.get("term"),
            item.get("column"),
            item.get("label"),
        )
        if any(str(candidate) == label for candidate in candidates if candidate is not None):
            matches.append(index)
    return matches


def cross_fit_shared_precision_groups(
    models: Sequence[Model] | Mapping[str, Model],
    groups: Sequence[SharedPrecisionGroup | Mapping[str, Any]] | Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Compute EB precision updates shared across separately fitted models.

    For each declared group ``p``, the update is

    ``lambda_p = (N_fits(p) * d_p + 2 * (a_p - 1)) / (sum_q_p + 2 * b_p)``,

    where ``sum_q_p`` pools ``||beta_p||² + tr(Sigma_pp)`` over models where
    the selected term/column/label appears. If a model does not contain the
    selected block, it is skipped for that group.
    """

    import numpy as np

    model_items = _normalize_model_mapping(models)
    group_specs = _normalize_shared_precision_groups(groups)
    if not group_specs:
        raise ValueError("at least one shared precision group is required")

    states: dict[str | int, Mapping[str, Any]] = {}
    for key, model in model_items:
        try:
            states[key] = json.loads(rust_module().coefficient_state_json(model._model_bytes))
        except Exception as exc:
            raise map_exception(exc) from exc

    result: dict[str, dict[str, Any]] = {}
    for group in group_specs:
        shape = float(group.shape)
        rate = float(group.rate)
        if not math.isfinite(shape) or shape <= 0.0:
            raise ValueError(
                f"shared precision group {group.name!r} requires finite shape > 0"
            )
        if not math.isfinite(rate) or rate < 0.0:
            raise ValueError(
                f"shared precision group {group.name!r} requires finite rate >= 0"
            )

        fit_entries: list[dict[str, Any]] = []
        dims: set[int] = set()
        quadratic_sum = 0.0
        for key, _model in model_items:
            state = states[key]
            label = _shared_group_label(group, key)
            indices = _coefficient_indices_for_precision_label(state, label)
            if not indices:
                continue
            beta = np.asarray(state.get("beta", []), dtype=float)
            # EB must use the fixed-lambda conditional Laplace covariance. The
            # smoothing-parameter-corrected covariance feeds lambda uncertainty
            # back into the lambda update, inflates the trace term, and biases
            # lambda downward.
            cov_n = int(state.get("covariance_n", 0))
            cov_flat = np.asarray(state.get("covariance_flat", []), dtype=float)
            if beta.size != cov_n or cov_flat.size != cov_n * cov_n:
                raise ValueError(
                    f"model {key!r} has inconsistent beta/covariance dimensions"
                )
            index_array = np.asarray(indices, dtype=int)
            if np.any(index_array < 0) or np.any(index_array >= cov_n):
                raise ValueError(
                    f"model {key!r} has coefficient provenance outside covariance bounds"
                )
            cov = cov_flat.reshape(cov_n, cov_n)
            beta_block = beta[index_array]
            trace = float(np.trace(cov[np.ix_(index_array, index_array)]))
            beta_norm_sq = float(beta_block.dot(beta_block))
            contribution = beta_norm_sq + trace
            if not math.isfinite(contribution):
                raise ValueError(
                    f"shared precision group {group.name!r} has non-finite contribution in model {key!r}"
                )
            quadratic_sum += contribution
            dims.add(int(index_array.size))
            fit_entries.append(
                {
                    "model": key,
                    "label": label,
                    "coefficient_indices": [int(i) for i in indices],
                    "dimension": int(index_array.size),
                    "beta_norm_sq": beta_norm_sq,
                    "trace_covariance": trace,
                    "quadratic_contribution": contribution,
                }
            )

        if not fit_entries:
            raise ValueError(
                f"shared precision group {group.name!r} did not match any model coefficients"
            )
        if len(dims) != 1:
            raise ValueError(
                f"shared precision group {group.name!r} matched inconsistent dimensions: {sorted(dims)}"
            )
        dimension = dims.pop()
        numerator = len(fit_entries) * dimension + 2.0 * (shape - 1.0)
        denominator = quadratic_sum + 2.0 * rate
        if numerator <= 0.0:
            raise ValueError(
                f"shared precision group {group.name!r} has non-positive MAP numerator"
            )
        if denominator <= 0.0 or not math.isfinite(denominator):
            raise ValueError(
                f"shared precision group {group.name!r} has non-positive/non-finite denominator"
            )
        lam = numerator / denominator
        result[group.name] = {
            "lambda": lam,
            "log_lambda": math.log(lam),
            "shape": shape,
            "rate": rate,
            "n_fits": len(fit_entries),
            "dimension": dimension,
            "quadratic_sum": quadratic_sum,
            "numerator": numerator,
            "denominator": denominator,
            "fits": fit_entries,
        }
    return result


def build_info() -> dict[str, Any]:
    """Return build/runtime metadata for the Rust extension.

    Reports whether ``gamfit._rust`` was importable and, when available, the
    build-time information exposed by the extension (version, commit, feature
    flags). Useful for bug reports and for confirming a development build is
    being used.

    Returns
    -------
    dict
        Always contains ``available`` (bool) and ``module`` (str). When the
        extension loaded, additional engine-specific keys are merged in;
        otherwise ``reason`` describes why import failed.

    Examples
    --------
    >>> info = gamfit.build_info()
    >>> info["available"]
    True
    """
    return extension_status()


def cuda_diagnostics() -> dict[str, object]:
    """Return CUDA loader diagnostics without forcing Rust GPU dispatch."""

    return _cuda_diagnostics()


def format_cuda_diagnostics() -> str:
    """Return CUDA loader diagnostics as stable, grep-friendly text."""

    return _format_cuda_diagnostics()


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
    frailty_kind: str | None,
    frailty_sd: float | None,
    hazard_loading: str | None,
    scale_dimensions: bool | None,
    adaptive_regularization: bool | None,
    firth: bool | None,
    precision_hyperpriors: Any | None,
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
        "frailty_kind": frailty_kind,
        "frailty_sd": frailty_sd,
        "hazard_loading": hazard_loading,
        "scale_dimensions": scale_dimensions,
        "adaptive_regularization": adaptive_regularization,
        "firth": firth,
        "precision_hyperpriors": precision_hyperpriors,
    }
    for key, value in kwarg_items.items():
        if value is not None:
            payload[key] = value
    if config:
        for key, value in config.items():
            payload.setdefault(key, value)
    return payload


def _normalize_precision_pair(value: Any, label: str) -> list[float]:
    if isinstance(value, dict):
        shape = value.get("shape", value.get("a", value.get("a_p")))
        rate = value.get("rate", value.get("b", value.get("b_p")))
    else:
        try:
            shape, rate = value
        except Exception as exc:  # pragma: no cover - defensive shape guard
            raise ValueError(
                f"precision_hyperpriors[{label!r}] must be (shape, rate)"
            ) from exc
    shape_f = float(shape)
    rate_f = float(rate)
    if (
        not math.isfinite(shape_f)
        or not math.isfinite(rate_f)
        or shape_f <= 0.0
        or rate_f < 0.0
    ):
        raise ValueError(
            f"precision_hyperpriors[{label!r}] needs finite shape > 0 and finite rate >= 0"
        )
    return [shape_f, rate_f]


def _group_terms_from_formula(formula: str) -> list[str]:
    return [m.group(1).strip() for m in re.finditer(r"\bgroup\s*\(\s*([^)]+?)\s*\)", formula)]


def _resolve_precision_hyperpriors(
    value: Any | None,
    formula: str,
    headers: list[str],
    rows: list[list[str]],
    group_metadata: Any | None = None,
) -> Any | None:
    if value is None:
        return None
    if callable(value):
        out: dict[str, list[float]] = {}
        metadata_by_label = group_metadata if isinstance(group_metadata, dict) else {}
        labels: list[str] = []
        for label in _group_terms_from_formula(formula):
            if label not in labels:
                labels.append(label)
        for label in metadata_by_label:
            if str(label) not in labels:
                labels.append(str(label))
        for label in labels:
            levels: list[str] = []
            if label in headers:
                col = headers.index(label)
                levels = sorted({row[col] for row in rows})
            pair = value(
                {
                    "label": label,
                    "term": label,
                    "column": label,
                    "levels": levels,
                    "n_coefficients": len(levels),
                    "metadata": metadata_by_label.get(label),
                    "group_metadata": metadata_by_label.get(label),
                }
            )
            out[label] = _normalize_precision_pair(pair, label)
        return out
    if isinstance(value, dict):
        return {str(k): _normalize_precision_pair(v, str(k)) for k, v in value.items()}
    return value


@overload
def fit(
    data: Any,
    formula: str,
    *,
    family: str = ...,
    offset: str | None = ...,
    weights: str | None = ...,
    transformation_normal: bool | None = ...,
    survival_likelihood: str | None = ...,
    baseline_target: str | None = ...,
    baseline_scale: float | None = ...,
    baseline_shape: float | None = ...,
    baseline_rate: float | None = ...,
    baseline_makeham: float | None = ...,
    z_column: str | None = ...,
    link: str | None = ...,
    logslope_formula: str | None = ...,
    frailty_kind: str | None = ...,
    frailty_sd: float | None = ...,
    hazard_loading: str | None = ...,
    scale_dimensions: bool | None = ...,
    adaptive_regularization: bool | None = ...,
    firth: bool | None = ...,
    precision_hyperpriors: Any | None = ...,
    response_geometry: None = ...,
    response_columns: list[str] | tuple[str, ...] | None = ...,
    response_coordinates: str | None = ...,
    response_reference: int | None = ...,
    config: dict[str, Any] | None = ...,
) -> Model: ...


@overload
def fit(
    data: Any,
    formula: str,
    *,
    family: str = ...,
    offset: str | None = ...,
    weights: str | None = ...,
    transformation_normal: bool | None = ...,
    survival_likelihood: str | None = ...,
    baseline_target: str | None = ...,
    baseline_scale: float | None = ...,
    baseline_shape: float | None = ...,
    baseline_rate: float | None = ...,
    baseline_makeham: float | None = ...,
    z_column: str | None = ...,
    link: str | None = ...,
    logslope_formula: str | None = ...,
    frailty_kind: str | None = ...,
    frailty_sd: float | None = ...,
    hazard_loading: str | None = ...,
    scale_dimensions: bool | None = ...,
    adaptive_regularization: bool | None = ...,
    firth: bool | None = ...,
    precision_hyperpriors: Any | None = ...,
    response_geometry: str,
    response_columns: list[str] | tuple[str, ...] | None = ...,
    response_coordinates: str | None = ...,
    response_reference: int | None = ...,
    config: dict[str, Any] | None = ...,
) -> ResponseGeometryModel: ...


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
    frailty_kind: str | None = None,
    frailty_sd: float | None = None,
    hazard_loading: str | None = None,
    scale_dimensions: bool | None = None,
    adaptive_regularization: bool | None = None,
    firth: bool | None = None,
    precision_hyperpriors: Any | None = None,
    response_geometry: str | None = None,
    response_columns: list[str] | tuple[str, ...] | None = None,
    response_coordinates: str | None = None,
    response_reference: int | None = None,
    config: dict[str, Any] | None = None,
) -> Model | ResponseGeometryModel:
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
        ``"weibull"``, ``"location-scale"``, ``"marginal-slope"``,
        ``"latent"``, or ``"latent-binary"``. Corresponds to
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
    frailty_kind:
        Frailty family for frailty-aware survival models. One of
        ``"gaussian-shift"`` or ``"hazard-multiplier"``. Corresponds to
        ``--frailty-kind``.
    response_geometry:
        Optional manifold-valued response geometry. Use ``"spherical"`` for
        unit-sphere responses, or ``"simplex"`` / ``"clr"`` / ``"alr"`` for
        strictly positive compositional responses. The base point is the
        intrinsic Fréchet mean of the training responses, not an extrinsic
        arithmetic mean.
    response_columns:
        Sequence of response component columns used when ``response_geometry``
        is set. One scalar Gaussian GAM is fitted for each tangent coordinate.
    response_coordinates:
        Coordinate chart for simplex responses: ``"clr"`` (default) or
        ``"alr"``. Spherical responses always use ambient tangent coordinates.
    response_reference:
        Reference component for ``"alr"`` coordinates (default: last column).
    frailty_sd:
        Fixed frailty standard deviation. Omit to let latent hazard-multiplier
        models learn it. Corresponds to ``--frailty-sd``.
    hazard_loading:
        Hazard loading for ``frailty_kind="hazard-multiplier"``. One of
        ``"full"`` or ``"loaded-vs-unloaded"``. Corresponds to
        ``--hazard-loading``.
    scale_dimensions:
        When ``True``, enables learned per-axis anisotropic length scales on
        spatial smooths (e.g. multi-dim Duchon / Matern / TPS). Per-axis
        scales are learned, not specified. Corresponds to ``--scale-dimensions``.
    adaptive_regularization:
        Enable exact local adaptive regularization for compatible spatial
        smooths. Omit to use the quality-first automatic policy, which leaves
        it off unless explicitly requested.
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
    if config:
        if response_geometry is None and config.get("response_geometry") is not None:
            response_geometry = str(config["response_geometry"])
        if response_columns is None and config.get("response_columns") is not None:
            raw_columns = config["response_columns"]
            if isinstance(raw_columns, (str, bytes)):
                raise ValueError(
                    "response_columns must be a sequence of column names, not a string"
                )
            response_columns = tuple(str(name) for name in raw_columns)
        if response_coordinates is None and config.get("response_coordinates") is not None:
            response_coordinates = str(config["response_coordinates"])
        if response_reference is None and config.get("response_reference") is not None:
            response_reference = int(config["response_reference"])

    if response_geometry is not None:
        if response_columns is None:
            raise ValueError("response_columns is required when response_geometry is set")
        nested_config = dict(config or {})
        # Geometry is handled by the Python wrapper; scalar coordinate fits keep
        # using the ordinary Rust standard-GAM path.
        for key in (
            "response_geometry",
            "response_columns",
            "response_coordinates",
            "response_reference",
        ):
            nested_config.pop(key, None)
        return fit_response_geometry(
            fit,
            data,
            formula,
            response_geometry=response_geometry,
            response_columns=tuple(response_columns),
            coordinates=response_coordinates,
            reference=-1 if response_reference is None else int(response_reference),
            weights=weights,
            fit_kwargs={
                "offset": offset,
                "weights": weights,
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
                "frailty_kind": frailty_kind,
                "frailty_sd": frailty_sd,
                "hazard_loading": hazard_loading,
                "scale_dimensions": scale_dimensions,
                "adaptive_regularization": adaptive_regularization,
                "firth": firth,
                "precision_hyperpriors": precision_hyperpriors,
                "config": nested_config or None,
            },
        )

    headers, rows, table_kind = normalize_table(data)
    rust_config = dict(config or {})
    for key in (
        "response_geometry",
        "response_columns",
        "response_coordinates",
        "response_reference",
    ):
        rust_config.pop(key, None)
    resolved_precision_hyperpriors = _resolve_precision_hyperpriors(
        precision_hyperpriors, formula, headers, rows, rust_config.get("group_metadata")
    )
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
        frailty_kind=frailty_kind,
        frailty_sd=frailty_sd,
        hazard_loading=hazard_loading,
        scale_dimensions=scale_dimensions,
        adaptive_regularization=adaptive_regularization,
        firth=firth,
        precision_hyperpriors=resolved_precision_hyperpriors,
        config=rust_config or None,
    )
    try:
        model_bytes = bytes(
            rust_module().fit_table(headers, rows, formula, json.dumps(payload))
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    return Model(_model_bytes=model_bytes, _training_table_kind=table_kind)


def fit_array(
    X: Any,
    Y: Any,
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
    frailty_kind: str | None = None,
    frailty_sd: float | None = None,
    hazard_loading: str | None = None,
    scale_dimensions: bool | None = None,
    adaptive_regularization: bool | None = None,
    firth: bool | None = None,
    precision_hyperpriors: Any | None = None,
    config: dict[str, Any] | None = None,
) -> Model:
    """Fit directly from numeric NumPy-compatible arrays.

    ``X`` is named ``x0``, ``x1``, ... at the formula boundary. A one-column
    ``Y`` is named from the formula response; multi-column ``Y`` is named
    ``y0``, ``y1``, ...
    """
    X_arr = _numeric_matrix(X, "X")
    Y_arr = _numeric_matrix(Y, "Y")
    rust_config = dict(config or {})
    resolved_precision_hyperpriors = _resolve_precision_hyperpriors(
        precision_hyperpriors, formula, [], [], rust_config.get("group_metadata")
    )
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
        frailty_kind=frailty_kind,
        frailty_sd=frailty_sd,
        hazard_loading=hazard_loading,
        scale_dimensions=scale_dimensions,
        adaptive_regularization=adaptive_regularization,
        firth=firth,
        precision_hyperpriors=resolved_precision_hyperpriors,
        config=rust_config or None,
    )
    try:
        model_bytes = bytes(
            rust_module().fit_array(X_arr, Y_arr, formula, json.dumps(payload))
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    return Model(_model_bytes=model_bytes, _training_table_kind="numpy")


def load(path: str | Path) -> Model:
    """Load a fitted :class:`Model` previously written with :meth:`Model.save`.

    Reads the raw bytes from ``path`` and dispatches to :func:`loads`.

    Parameters
    ----------
    path : str or pathlib.Path
        Filesystem path to the serialized model file.

    Returns
    -------
    Model
        Fitted model ready for prediction.

    Raises
    ------
    GamError
        If the file cannot be decoded by the Rust engine.

    Examples
    --------
    >>> model = gamfit.load("model.gam")
    >>> model.predict(test_df)
    """
    model_bytes = Path(path).read_bytes()
    return loads(model_bytes)


def loads(model_bytes: bytes) -> Model:
    """Load a fitted :class:`Model` from an in-memory bytes payload.

    Parameters
    ----------
    model_bytes : bytes
        Raw serialized model produced by :meth:`Model.save` or
        :meth:`Model.saves`.

    Returns
    -------
    Model
        Fitted model ready for prediction.

    Raises
    ------
    GamError
        If the payload is malformed or incompatible with the current engine.

    Examples
    --------
    >>> with open("model.gam", "rb") as fh:
    ...     model = gamfit.loads(fh.read())
    """
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
    frailty_kind: str | None = None,
    frailty_sd: float | None = None,
    hazard_loading: str | None = None,
    scale_dimensions: bool | None = None,
    adaptive_regularization: bool | None = None,
    firth: bool | None = None,
    config: dict[str, Any] | None = None,
) -> FormulaValidation:
    """Validate a formula against a dataset without fitting.

    Accepts every pipeline kwarg that :func:`fit` accepts, with identical
    semantics. See :func:`fit` for parameter documentation.
    """
    headers, rows, _table_kind = normalize_table(data)
    rust_config = dict(config or {})
    for key in (
        "response_geometry",
        "response_columns",
        "response_coordinates",
        "response_reference",
    ):
        rust_config.pop(key, None)
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
        frailty_kind=frailty_kind,
        frailty_sd=frailty_sd,
        hazard_loading=hazard_loading,
        scale_dimensions=scale_dimensions,
        adaptive_regularization=adaptive_regularization,
        firth=firth,
        precision_hyperpriors=None,
        config=rust_config or None,
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
    """Return a short, actionable hint describing how to recover from ``exc``.

    Inspects the exception type and returns a one-line suggestion tailored to
    the gamfit error hierarchy (:class:`FormulaError`,
    :class:`SchemaMismatchError`, :class:`PredictionError`, :class:`GamError`,
    :class:`RustExtensionUnavailableError`). Unrecognized exceptions fall back
    to a generic message.

    Parameters
    ----------
    exc : BaseException
        The exception caught from a gamfit call.

    Returns
    -------
    str
        Human-readable remediation hint.

    Examples
    --------
    >>> try:
    ...     gamfit.fit(df, "y ~ s(nope)")
    ... except gamfit.GamError as exc:
    ...     print(gamfit.explain_error(exc))
    Check the formula syntax and confirm every referenced column exists.
    """
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


def bspline_basis(
    t: Any,
    knots: Any = None,
    *,
    degree: int = 3,
    periodic: bool = False,
) -> Any:
    """Evaluate the Rust B-spline basis as a NumPy array.

    ``knots`` may be:

    * ``None`` — auto-derive a clamped knot vector with quantile-spaced
      interior knots inferred from ``t``.
    * an ``int`` ``K`` — auto-derive with ``K`` interior knots.
    * an array-like — used verbatim (must be a valid clamped knot vector).
    """
    import numpy as np

    t_np = _numeric_vector(t, "t")
    knots_np = _resolve_knots(knots, t_np, label="knots", degree=int(degree))
    try:
        return np.asarray(
            rust_module().bspline_basis(
                t_np,
                knots_np,
                int(degree),
                bool(periodic),
            ),
            dtype=float,
        )
    except Exception as exc:
        raise map_exception(exc) from exc


def bspline_basis_derivative(
    t: Any,
    knots: Any = None,
    *,
    degree: int = 3,
    order: int = 1,
    periodic: bool = False,
) -> Any:
    """Evaluate derivatives of the Rust B-spline basis as a NumPy array.

    ``knots`` accepts ``None`` / ``int`` / array — see :func:`bspline_basis`.
    """
    import numpy as np

    t_np = _numeric_vector(t, "t")
    knots_np = _resolve_knots(knots, t_np, label="knots", degree=int(degree))
    try:
        return np.asarray(
            rust_module().bspline_basis_derivative(
                t_np,
                knots_np,
                int(degree),
                int(order),
                bool(periodic),
            ),
            dtype=float,
        )
    except Exception as exc:
        raise map_exception(exc) from exc


def duchon_basis(
    points: Any,
    centers: Any = None,
    *,
    m: int = 2,
    periodic_per_axis: Any = None,
) -> Any:
    """Evaluate the Duchon m-spline basis at ``points`` against ``centers``.

    Works for any input dimensionality ``d`` ≥ 1.

    Parameters
    ----------
    points : array-like of shape (N, d) — N evaluation points in d-dim.
        For 1D, pass shape (N, 1) or a 1D array (auto-promoted).
    centers : array-like of shape (K, d), or ``None`` (auto: K=10 quantile
        centers for d=1), or an ``int`` K (auto-quantile centers, d=1 only).
    m : int, default 2 — spline order.
    periodic_per_axis : sequence of bool of length d, optional. Currently
        only ``d=1`` supports periodicity.

    Returns
    -------
    ndarray of shape (N, K)
    """
    import numpy as np

    pts_np = np.asarray(points, dtype=float)
    if pts_np.ndim == 1:
        pts_np = pts_np.reshape(-1, 1)
    if pts_np.ndim != 2:
        raise ValueError(f"points must be 1D or 2D, got {pts_np.ndim}D")
    d = pts_np.shape[1]
    if centers is None or isinstance(centers, int):
        if d != 1:
            raise ValueError(f"auto centers only supported for d=1, got d={d}")
        ctrs_np = _resolve_centers(centers, pts_np[:, 0], label="centers").reshape(-1, 1)
    else:
        ctrs_np = np.asarray(centers, dtype=float)
        if ctrs_np.ndim == 1:
            ctrs_np = ctrs_np.reshape(-1, 1)
        if ctrs_np.ndim != 2:
            raise ValueError(f"centers must be 1D or 2D, got {ctrs_np.ndim}D")
        if ctrs_np.shape[1] != d:
            raise ValueError(
                f"points has d={d} but centers has d={ctrs_np.shape[1]}"
            )

    periodic_arg = (
        None if periodic_per_axis is None
        else [bool(p) for p in periodic_per_axis]
    )
    try:
        return np.asarray(
            rust_module().duchon_basis(pts_np, ctrs_np, int(m), periodic_arg),
            dtype=float,
        )
    except Exception as exc:
        raise map_exception(exc) from exc


def smoothness_penalty(
    knots: Any,
    *,
    degree: int = 3,
    order: int = 2,
) -> tuple[Any, Any]:
    """Return ``(S, null_basis)`` for the Rust B-spline difference penalty.

    ``knots`` must be a knot vector here — auto-derivation requires
    sample positions, which this penalty constructor does not take. Build
    one with :func:`bspline_basis`'s defaults (or pass any 1D array).
    """
    import numpy as np

    try:
        penalty, null_basis = rust_module().smoothness_penalty(
            _numeric_vector(knots, "knots"),
            int(degree),
            int(order),
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    return np.asarray(penalty, dtype=float), np.asarray(null_basis, dtype=float)


def _duchon_operator_penalties(
    centers: Any,
    *,
    m: int = 2,
    periodic: bool = False,
    period: float | None = None,
) -> tuple[Any, Any, Any]:
    """Internal Duchon operator penalty constructor."""
    import numpy as np

    try:
        mass, tension, stiffness = rust_module().duchon_operator_penalties(
            _numeric_vector(centers, "centers"),
            int(m),
            bool(periodic),
            None if period is None else float(period),
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    return (
        np.asarray(mass, dtype=float),
        np.asarray(tension, dtype=float),
        np.asarray(stiffness, dtype=float),
    )


def duchon_function_norm_penalty(
    centers: Any,
    *,
    m: int = 2,
    periodic_per_axis: Any = None,
) -> Any:
    """Duchon m-spline RKHS / function-norm penalty matrix.

    Parameters
    ----------
    centers : array-like
        Control points. Shape ``(K,)`` or ``(K, 1)`` for 1D (auto-promoted).
        Shape ``(K, d)`` for higher dimensions (currently only the 1D path
        is exposed by the Rust binding — d > 1 will raise an error until
        the multi-d penalty binding lands).
    m : int, default 2
        Spline order.
    periodic_per_axis : sequence of bool of length d, optional
        Per-axis periodicity. Currently only d=1 supports periodicity.

    Returns
    -------
    ndarray of shape (K, K) — SPD penalty matrix.
    """
    import numpy as np

    ctrs = np.asarray(centers, dtype=float)
    if ctrs.ndim == 2:
        if ctrs.shape[1] != 1:
            raise NotImplementedError(
                f"d={ctrs.shape[1]} Duchon penalty not yet exposed; "
                "Rust binding accepts only 1D centers currently."
            )
        ctrs_1d = ctrs[:, 0]
    elif ctrs.ndim == 1:
        ctrs_1d = ctrs
    else:
        raise ValueError(f"centers must be 1D or 2D, got {ctrs.ndim}D")

    per = False
    if periodic_per_axis is not None:
        per_list = [bool(p) for p in periodic_per_axis]
        if len(per_list) != 1:
            raise ValueError(
                f"periodic_per_axis must have length matching centers dim (1), got {len(per_list)}"
            )
        per = per_list[0]

    try:
        penalty = rust_module().duchon_function_norm_penalty(
            ctrs_1d,
            int(m),
            per,
            None,
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    return np.asarray(penalty, dtype=float)


# Backward-compat private alias (callers in this module still reference it)
_duchon_function_norm_penalty = duchon_function_norm_penalty


def _thin_plate_penalty(
    centers: Any,
    *,
    m: int = 2,
    length_scale: float = 1.0,
) -> Any:
    """Internal thin-plate bending-energy penalty constructor."""
    import numpy as np

    try:
        penalty = rust_module().thin_plate_penalty(
            _numeric_matrix(centers, "centers"),
            int(m),
            float(length_scale),
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    return np.asarray(penalty, dtype=float)


def _gaussian_reml_score(
    X: Any,
    Y: Any,
    coefficients: Any,
    log_lambda: float,
    penalty: Any,
    *,
    weights: Any | None = None,
    by: Any | None = None,
    by_start_col: int = 0,
) -> dict[str, Any]:
    """Internal free-coefficient REML score evaluator."""
    import numpy as np

    try:
        out = rust_module().gaussian_reml_score(
            _numeric_matrix(X, "X"),
            _numeric_matrix(Y, "Y"),
            _numeric_matrix(coefficients, "coefficients"),
            float(log_lambda),
            _numeric_matrix(penalty, "penalty"),
            None if weights is None else _numeric_vector(weights, "weights"),
            None if by is None else _numeric_vector(by, "by"),
            int(by_start_col),
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    result = dict(out)
    for key in ("grad_coefficients", "grad_penalty", "fitted", "sigma2"):
        # ``grad_penalty`` is only populated when the Rust kernel has built
        # the VJP through ``S``. The closed-form path always returns the
        # others; leaving ``grad_penalty`` absent keeps the Python surface
        # forward-compatible with older builds.
        if key in result:
            result[key] = np.asarray(result[key], dtype=float)
    return result


def gaussian_weighted_ridge(
    X: Any,
    Y: Any,
    penalty: Any,
    weights: Any,
    *,
    ridge_lambda: float,
) -> tuple[Any, Any]:
    """Closed-form Gaussian row-weighted ridge on NumPy-compatible arrays.

    ``weights`` are likelihood row weights. They are not a multiplicative
    gate on the mean/design row.
    """
    import numpy as np

    try:
        coefficients, fitted = rust_module().gaussian_weighted_ridge_array(
            _numeric_matrix(X, "X"),
            _numeric_matrix(Y, "Y"),
            _numeric_matrix(penalty, "penalty"),
            _numeric_vector(weights, "weights"),
            float(ridge_lambda),
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    return np.asarray(coefficients, dtype=float), np.asarray(fitted, dtype=float)


def gaussian_weighted_ridge_batch(
    X: Any,
    Y: Any,
    penalty: Any,
    weights: Any,
    *,
    ridge_lambda: float,
    row_counts: Any | None = None,
) -> tuple[Any, Any]:
    """Batched closed-form Gaussian row-weighted ridge.

    ``X`` has shape ``(K, Nmax, M)``, ``Y`` has shape ``(K, Nmax, D)``, and
    ``weights`` has shape ``(K, Nmax)``. ``row_counts`` optionally marks the
    active row prefix for each problem in a padded ragged batch.
    """
    import numpy as np

    try:
        coefficients, fitted = rust_module().gaussian_weighted_ridge_batch(
            _numeric_tensor3(X, "X"),
            _numeric_tensor3(Y, "Y"),
            _numeric_matrix(penalty, "penalty"),
            _numeric_matrix(weights, "weights"),
            float(ridge_lambda),
            None if row_counts is None else _index_vector(row_counts, "row_counts"),
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    return np.asarray(coefficients, dtype=float), np.asarray(fitted, dtype=float)


def gaussian_reml_fit(
    x: Any,
    y: Any,
    penalty: Any,
    *,
    weights: Any | None = None,
    init_lambda: float | None = None,
    by: Any | None = None,
    by_start_col: int = 0,
) -> dict[str, Any]:
    """Fit a closed-form Gaussian REML problem from NumPy-compatible arrays."""
    import numpy as np

    try:
        out = rust_module().gaussian_reml_fit(
            _numeric_matrix(x, "x"),
            _numeric_matrix(y, "y"),
            _numeric_matrix(penalty, "penalty"),
            None if weights is None else _numeric_vector(weights, "weights"),
            None if init_lambda is None else float(init_lambda),
            None if by is None else _numeric_vector(by, "by"),
            int(by_start_col),
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    return _coerce_gaussian_reml_payload(out, np)


def gaussian_reml_fit_backward(
    x: Any,
    y: Any,
    penalty: Any,
    *,
    grad_lambda: float = 0.0,
    grad_coefficients: Any | None = None,
    grad_fitted: Any | None = None,
    grad_reml_score: float = 0.0,
    grad_edf: float = 0.0,
    forward_state: dict[str, Any] | None = None,
    weights: Any | None = None,
    init_lambda: float | None = None,
    by: Any | None = None,
    by_start_col: int = 0,
) -> dict[str, Any]:
    """Run the analytic VJP for ``gaussian_reml_fit`` outputs."""
    import numpy as np

    try:
        out = rust_module().gaussian_reml_fit_backward(
            _numeric_matrix(x, "x"),
            _numeric_matrix(y, "y"),
            _numeric_matrix(penalty, "penalty"),
            float(grad_lambda),
            None
            if grad_coefficients is None
            else _numeric_matrix(grad_coefficients, "grad_coefficients"),
            None if grad_fitted is None else _numeric_matrix(grad_fitted, "grad_fitted"),
            float(grad_reml_score),
            float(grad_edf),
            forward_state,
            None if weights is None else _numeric_vector(weights, "weights"),
            None if init_lambda is None else float(init_lambda),
            None if by is None else _numeric_vector(by, "by"),
            int(by_start_col),
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    result = dict(out)
    for key in ("grad_x", "grad_y", "grad_penalty", "grad_weights", "grad_by"):
        if result.get(key) is not None:
            result[key] = np.asarray(result[key], dtype=float)
    return result


def gaussian_reml_fit_batched(
    x: Any,
    y: Any,
    row_offsets: Any,
    penalty: Any,
    *,
    weights: Any | None = None,
    init_lambda: float | None = None,
    by: Any | None = None,
    by_start_col: int = 0,
) -> dict[str, Any]:
    """Fit K closed-form Gaussian REML problems packed by row offsets."""
    import numpy as np

    try:
        out = rust_module().gaussian_reml_fit_batched(
            _numeric_matrix(x, "x"),
            _numeric_matrix(y, "y"),
            _index_vector(row_offsets, "row_offsets"),
            _numeric_matrix(penalty, "penalty"),
            None if weights is None else _numeric_vector(weights, "weights"),
            None if init_lambda is None else float(init_lambda),
            None if by is None else _numeric_vector(by, "by"),
            int(by_start_col),
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    return _coerce_gaussian_reml_payload(out, np)


def gaussian_reml_fit_batched_backward(
    x: Any,
    y: Any,
    row_offsets: Any,
    penalty: Any,
    *,
    grad_lambda: Any | None = None,
    grad_coefficients: Any | None = None,
    grad_fitted: Any | None = None,
    grad_reml_score: Any | None = None,
    grad_edf: Any | None = None,
    forward_state: dict[str, Any] | None = None,
    weights: Any | None = None,
    init_lambda: float | None = None,
    by: Any | None = None,
    by_start_col: int = 0,
) -> dict[str, Any]:
    """Run packed ragged analytic VJPs for ``gaussian_reml_fit_batched``."""
    import numpy as np

    offsets = _index_vector(row_offsets, "row_offsets")
    batch = int(offsets.size - 1)
    try:
        out = rust_module().gaussian_reml_fit_batched_backward(
            _numeric_matrix(x, "x"),
            _numeric_matrix(y, "y"),
            offsets,
            _numeric_matrix(penalty, "penalty"),
            _optional_batch_vector(grad_lambda, batch, "grad_lambda"),
            None
            if grad_coefficients is None
            else _numeric_tensor3(grad_coefficients, "grad_coefficients"),
            None if grad_fitted is None else _numeric_matrix(grad_fitted, "grad_fitted"),
            _optional_batch_vector(grad_reml_score, batch, "grad_reml_score"),
            _optional_batch_vector(grad_edf, batch, "grad_edf"),
            forward_state,
            None if weights is None else _numeric_vector(weights, "weights"),
            None if init_lambda is None else float(init_lambda),
            None if by is None else _numeric_vector(by, "by"),
            int(by_start_col),
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    result = dict(out)
    for key in ("grad_x", "grad_y", "grad_penalty", "grad_weights", "grad_by"):
        if result.get(key) is not None:
            result[key] = np.asarray(result[key], dtype=float)
    return result


def gaussian_reml_fit_positions(
    t: Any,
    y: Any,
    basis_kind: str | None = None,
    knots_or_centers: Any = None,
    penalty: Any | None = None,
    *,
    basis: str | None = None,
    basis_order: int | None = None,
    periodic: bool = False,
    period: float | None = None,
    weights: Any | None = None,
    init_lambda: float | None = None,
    by: Any | None = None,
    by_start_col: int = 0,
) -> dict[str, Any]:
    """Fit closed-form Gaussian REML from 1D positions and an internal basis.

    ``knots_or_centers`` may be ``None``, an ``int`` (basis count), or an
    array; the basis-location vector is auto-derived from ``t`` when not
    supplied. ``penalty`` may be ``None`` for a neutral identity ridge of
    matching size.
    """
    import numpy as np

    display_kind = str(
        basis if basis is not None else basis_kind if basis_kind is not None else "bspline"
    )
    effective_kind, order, _ = _normalize_position_basis(display_kind, basis_order)
    t_np = _numeric_vector(t, "t")
    knots_np = _resolve_basis_locations(
        knots_or_centers,
        t_np,
        basis_kind=effective_kind,
        label="knots_or_centers",
        degree=order,
    )
    penalty_np = _resolve_position_penalty(
        penalty,
        knots_np,
        basis_kind=display_kind,
        basis_order=order,
        periodic=periodic,
    )
    try:
        out = rust_module().gaussian_reml_fit_positions(
            t_np,
            _numeric_matrix(y, "y"),
            str(effective_kind),
            knots_np,
            penalty_np,
            order,
            bool(periodic),
            None if period is None else float(period),
            None if weights is None else _numeric_vector(weights, "weights"),
            None if init_lambda is None else float(init_lambda),
            None if by is None else _numeric_vector(by, "by"),
            int(by_start_col),
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    return _attach_basis_state(
        _coerce_gaussian_reml_payload(out, np),
        knots_or_centers=knots_np,
        penalty=penalty_np,
        basis_kind=display_kind,
        basis_order=order,
        periodic=periodic,
        period=period,
    )


def gaussian_reml_fit_positions_backward(
    t: Any,
    y: Any,
    basis_kind: str | None = None,
    knots_or_centers: Any = None,
    penalty: Any | None = None,
    *,
    basis: str | None = None,
    grad_lambda: float = 0.0,
    grad_coefficients: Any | None = None,
    grad_fitted: Any | None = None,
    grad_reml_score: float = 0.0,
    grad_edf: float = 0.0,
    forward_state: dict[str, Any] | None = None,
    basis_order: int | None = None,
    periodic: bool = False,
    period: float | None = None,
    weights: Any | None = None,
    init_lambda: float | None = None,
    by: Any | None = None,
    by_start_col: int = 0,
) -> dict[str, Any]:
    """Run the analytic VJP for ``gaussian_reml_fit_positions`` outputs.

    ``knots_or_centers`` and ``penalty`` accept the same auto-derived
    defaults as :func:`gaussian_reml_fit_positions`.
    """
    import numpy as np

    display_kind = str(
        basis if basis is not None else basis_kind if basis_kind is not None else "bspline"
    )
    effective_kind, order, _ = _normalize_position_basis(display_kind, basis_order)
    t_np = _numeric_vector(t, "t")
    knots_np = _resolve_basis_locations(
        knots_or_centers,
        t_np,
        basis_kind=effective_kind,
        label="knots_or_centers",
        degree=order,
    )
    penalty_np = _resolve_position_penalty(
        penalty,
        knots_np,
        basis_kind=display_kind,
        basis_order=order,
        periodic=periodic,
    )
    try:
        out = rust_module().gaussian_reml_fit_positions_backward(
            t_np,
            _numeric_matrix(y, "y"),
            str(effective_kind),
            knots_np,
            penalty_np,
            float(grad_lambda),
            None
            if grad_coefficients is None
            else _numeric_matrix(grad_coefficients, "grad_coefficients"),
            None if grad_fitted is None else _numeric_matrix(grad_fitted, "grad_fitted"),
            float(grad_reml_score),
            float(grad_edf),
            forward_state,
            order,
            bool(periodic),
            None if period is None else float(period),
            None if weights is None else _numeric_vector(weights, "weights"),
            None if init_lambda is None else float(init_lambda),
            None if by is None else _numeric_vector(by, "by"),
            int(by_start_col),
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    result = dict(out)
    for key in ("grad_t", "grad_y", "grad_penalty", "grad_weights", "grad_by"):
        if result.get(key) is not None:
            result[key] = np.asarray(result[key], dtype=float)
    return result


def gaussian_reml_fit_positions_batched(
    t: Any,
    y: Any,
    row_offsets: Any,
    basis_kind: str | None = None,
    knots_or_centers: Any = None,
    penalty: Any | None = None,
    *,
    basis: str | None = None,
    basis_order: int | None = None,
    periodic: bool = False,
    period: float | None = None,
    weights: Any | None = None,
    init_lambda: float | None = None,
    by: Any | None = None,
    by_start_col: int = 0,
) -> dict[str, Any]:
    """Fit packed ragged closed-form Gaussian REML problems from positions.

    ``knots_or_centers`` and ``penalty`` accept the same auto-derived
    defaults as :func:`gaussian_reml_fit_positions`. The basis locations
    are inferred from the concatenated positions across all groups.
    """
    import numpy as np

    display_kind = str(
        basis if basis is not None else basis_kind if basis_kind is not None else "bspline"
    )
    effective_kind, order, _ = _normalize_position_basis(display_kind, basis_order)
    t_np = _numeric_vector(t, "t")
    knots_np = _resolve_basis_locations(
        knots_or_centers,
        t_np,
        basis_kind=effective_kind,
        label="knots_or_centers",
        degree=order,
    )
    penalty_np = _resolve_position_penalty(
        penalty,
        knots_np,
        basis_kind=display_kind,
        basis_order=order,
        periodic=periodic,
    )
    try:
        out = rust_module().gaussian_reml_fit_positions_batched(
            t_np,
            _numeric_matrix(y, "y"),
            _index_vector(row_offsets, "row_offsets"),
            str(effective_kind),
            knots_np,
            penalty_np,
            order,
            bool(periodic),
            None if period is None else float(period),
            None if weights is None else _numeric_vector(weights, "weights"),
            None if init_lambda is None else float(init_lambda),
            None if by is None else _numeric_vector(by, "by"),
            int(by_start_col),
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    return _attach_basis_state(
        _coerce_gaussian_reml_payload(out, np),
        knots_or_centers=knots_np,
        penalty=penalty_np,
        basis_kind=display_kind,
        basis_order=order,
        periodic=periodic,
        period=period,
    )


def gaussian_reml_fit_positions_batched_backward(
    t: Any,
    y: Any,
    row_offsets: Any,
    basis_kind: str | None = None,
    knots_or_centers: Any = None,
    penalty: Any | None = None,
    *,
    basis: str | None = None,
    grad_lambda: Any | None = None,
    grad_coefficients: Any | None = None,
    grad_fitted: Any | None = None,
    grad_reml_score: Any | None = None,
    grad_edf: Any | None = None,
    forward_state: dict[str, Any] | None = None,
    basis_order: int | None = None,
    periodic: bool = False,
    period: float | None = None,
    weights: Any | None = None,
    init_lambda: float | None = None,
    by: Any | None = None,
    by_start_col: int = 0,
) -> dict[str, Any]:
    """Run the analytic VJP for packed position-based Gaussian REML fits.

    ``knots_or_centers`` and ``penalty`` accept the same auto-derived
    defaults as :func:`gaussian_reml_fit_positions_batched`.
    """
    import numpy as np

    offsets = _index_vector(row_offsets, "row_offsets")
    batch = int(offsets.size - 1)
    display_kind = str(
        basis if basis is not None else basis_kind if basis_kind is not None else "bspline"
    )
    effective_kind, order, _ = _normalize_position_basis(display_kind, basis_order)
    t_np = _numeric_vector(t, "t")
    knots_np = _resolve_basis_locations(
        knots_or_centers,
        t_np,
        basis_kind=effective_kind,
        label="knots_or_centers",
        degree=order,
    )
    penalty_np = _resolve_position_penalty(
        penalty,
        knots_np,
        basis_kind=display_kind,
        basis_order=order,
        periodic=periodic,
    )
    try:
        out = rust_module().gaussian_reml_fit_positions_batched_backward(
            t_np,
            _numeric_matrix(y, "y"),
            offsets,
            str(effective_kind),
            knots_np,
            penalty_np,
            _optional_batch_vector(grad_lambda, batch, "grad_lambda"),
            None
            if grad_coefficients is None
            else _numeric_tensor3(grad_coefficients, "grad_coefficients"),
            None if grad_fitted is None else _numeric_matrix(grad_fitted, "grad_fitted"),
            _optional_batch_vector(grad_reml_score, batch, "grad_reml_score"),
            _optional_batch_vector(grad_edf, batch, "grad_edf"),
            forward_state,
            order,
            bool(periodic),
            None if period is None else float(period),
            None if weights is None else _numeric_vector(weights, "weights"),
            None if init_lambda is None else float(init_lambda),
            None if by is None else _numeric_vector(by, "by"),
            int(by_start_col),
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    result = dict(out)
    for key in ("grad_t", "grad_y", "grad_penalty", "grad_weights", "grad_by"):
        if result.get(key) is not None:
            result[key] = np.asarray(result[key], dtype=float)
    return result


def gaussian_reml_fit_formula(
    data: Any,
    formula: str,
    y: Any,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit closed-form Gaussian REML after materialising a formula design."""
    import numpy as np

    headers, rows, _kind = normalize_table(data)
    y_arr = np.asarray(y, dtype=float)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)
    if y_arr.ndim != 2:
        raise ValueError("y must be a 1D or 2D numeric array")
    if not np.all(np.isfinite(y_arr)):
        raise ValueError("y must contain only finite values")
    try:
        out = rust_module().gaussian_reml_fit_formula_table(
            headers,
            rows,
            formula,
            np.ascontiguousarray(y_arr, dtype=np.float64),
            None if config is None else json.dumps(config),
        )
    except Exception as exc:
        raise map_exception(exc) from exc
    return _coerce_gaussian_reml_payload(out, np)


def _coerce_gaussian_reml_payload(payload: Any, np: Any) -> dict[str, Any]:
    out = dict(payload)
    for key in (
        "coefficients",
        "fitted",
        "sigma2",
        "lambda",
        "rho",
        "reml_score",
        "reml_grad_lambda",
        "reml_hess_lambda",
        "reml_grad_rho",
        "reml_hess_rho",
        "edf",
        "cache_penalty_eigenvalues",
        "cache_eigenvectors",
        "cache_coefficient_basis",
    ):
        if key in out:
            out[key] = np.asarray(out[key], dtype=float)
    return out


def _position_basis_order(basis_kind: str, basis_order: int | None) -> int:
    if basis_order is not None:
        order = int(basis_order)
    else:
        normalized = basis_kind.strip().lower().replace("_", "").replace("-", "")
        order = 2 if normalized in {"duchon", "duchonspline"} else 3
    if order < 1:
        raise ValueError("basis_order must be at least 1")
    return order


def _optional_batch_vector(values: Any | None, batch: int, label: str) -> Any | None:
    import numpy as np

    if values is None:
        return None
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = np.full(batch, float(arr), dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{label} must be a scalar or 1D numeric array")
    if arr.size != batch:
        raise ValueError(f"{label} length mismatch: expected {batch}, got {arr.size}")
    if arr.dtype != np.float64:
        raise TypeError(f"{label} must be a float64 numpy array for zero-copy FFI")
    return arr


def _index_vector(values: Any, label: str) -> Any:
    import numpy as np

    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{label} must be a 1D integer array")
    if arr.size == 0:
        raise ValueError(f"{label} cannot be empty")
    if arr.dtype != np.dtype(np.uintp):
        raise TypeError(f"{label} must be a numpy uintp array for zero-copy FFI")
    return arr


def _numeric_vector(values: Any, label: str) -> Any:
    import numpy as np

    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{label} must be a 1D numeric array")
    if arr.size == 0:
        raise ValueError(f"{label} cannot be empty")
    if arr.dtype != np.float64:
        raise TypeError(f"{label} must be a float64 numpy array for zero-copy FFI")
    return arr


def _numeric_matrix(values: Any, label: str) -> Any:
    import numpy as np

    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{label} must be a 1D or 2D numeric array")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{label} cannot be empty")
    if arr.dtype != np.float64:
        raise TypeError(f"{label} must be a float64 numpy array for zero-copy FFI")
    return arr


# Default number of basis functions when the caller does not pin centers /
# knots themselves. Matches mgcv's ``k = 10`` convention. The actual
# placement (quantile knots, equal-mass centers, boundary clamping, ...)
# is performed by the Rust engine; Python only forwards the request.
_DEFAULT_BASIS_K = 10


def _resolve_centers(centers: Any, t_arr: Any, *, label: str = "centers") -> Any:
    """Coerce ``centers`` (None / int / array) into a 1D float64 array.

    Auto-derivation delegates to the Rust ``auto_centers_1d`` FFI export so
    Python never reimplements basis-placement logic.
    """
    if centers is None:
        return _numpy_module().asarray(
            rust_module().auto_centers_1d(t_arr, int(_DEFAULT_BASIS_K)),
            dtype=float,
        )
    if isinstance(centers, int) and not isinstance(centers, bool):
        if centers < 2:
            raise ValueError(f"{label}: integer count must be >= 2, got {centers}")
        return _numpy_module().asarray(
            rust_module().auto_centers_1d(t_arr, int(centers)),
            dtype=float,
        )
    return _numeric_vector(centers, label)


def _resolve_knots(
    knots: Any,
    t_arr: Any,
    *,
    label: str = "knots",
    degree: int = 3,
) -> Any:
    """Coerce ``knots`` (None / int / array) into a 1D float64 array.

    Auto-derivation delegates to the Rust ``auto_knots_1d`` FFI export.
    """
    if knots is None:
        return _numpy_module().asarray(
            rust_module().auto_knots_1d(t_arr, int(_DEFAULT_BASIS_K), int(degree)),
            dtype=float,
        )
    if isinstance(knots, int) and not isinstance(knots, bool):
        if knots < 0:
            raise ValueError(f"{label}: integer interior-knot count must be >= 0, got {knots}")
        return _numpy_module().asarray(
            rust_module().auto_knots_1d(t_arr, int(knots), int(degree)),
            dtype=float,
        )
    return _numeric_vector(knots, label)


def _numpy_module() -> Any:
    """Return the cached ``numpy`` module without polluting global imports."""
    import numpy as np

    return np


def _resolve_basis_locations(
    arg: Any,
    t_arr: Any,
    *,
    basis_kind: str,
    label: str = "knots_or_centers",
    degree: int = 3,
) -> Any:
    """Resolve the basis-location argument for kind-dispatched primitives.

    Mirrors :func:`_resolve_centers` for ``basis_kind == "duchon"`` and
    :func:`_resolve_knots` for B-spline-like kinds.
    """
    kind = str(basis_kind).strip().lower().replace("_", "").replace("-", "")
    if kind in {"duchon", "duchonspline"}:
        return _resolve_centers(arg, t_arr, label=label)
    return _resolve_knots(arg, t_arr, label=label, degree=degree)


def _position_basis_dim(
    knots_or_centers: Any,
    basis_kind: str,
    basis_order: int,
    periodic: bool,
) -> int:
    """Return the basis dimension (= design ncols) for a position basis.

    Used to size a default identity penalty when the caller omits one.
    """
    kind = str(basis_kind).strip().lower().replace("_", "").replace("-", "")
    n = int(knots_or_centers.shape[0])
    if kind in {"duchon", "duchonspline"}:
        # The REML position path uses only the radial-basis block; the
        # polynomial nullspace is carried as an unpenalised side channel
        # and is not part of the penalty matrix. So the penalty must be
        # sized to the number of centers.
        return n
    # Clamped B-spline: ncols = len(knots) - degree - 1. The Rust impl
    # uses the same convention for the periodic variant; rely on the
    # engine to reject mismatches if a periodic caller supplies a custom
    # penalty themselves.
    return max(n - int(basis_order) - 1, 0)


def _attach_basis_state(
    payload: dict[str, Any],
    *,
    knots_or_centers: Any,
    penalty: Any,
    basis_kind: str,
    basis_order: int,
    periodic: bool,
    period: float | None,
) -> dict[str, Any]:
    """Embed the resolved basis state in a REML position-fit payload.

    The keys ``knots_or_centers``, ``penalty``, ``basis_kind``,
    ``basis_order``, ``periodic``, and ``period`` are added so a caller
    can replay the exact same basis at predict time without recomputing
    anything — pass them straight back into ``duchon_basis`` or
    ``bspline_basis``.
    """
    payload["knots_or_centers"] = knots_or_centers
    payload["penalty"] = penalty
    payload["basis_kind"] = str(basis_kind)
    payload["basis_order"] = int(basis_order)
    payload["periodic"] = bool(periodic)
    payload["period"] = None if period is None else float(period)
    return payload


def _resolve_position_penalty(
    penalty: Any | None,
    knots_or_centers: Any,
    *,
    basis_kind: str,
    basis_order: int,
    periodic: bool,
) -> Any:
    """Resolve the canonical penalty for position-based REML helpers.

    Picking a basis chooses the penalty by default — pairing a basis with
    the wrong penalty is not a recognised statistical object. Power users
    can override with an explicit matrix in ``penalty``; the string-form
    ``penalty="function-norm" | "triple-operator" | "difference"`` selects
    a non-default canonical penalty for the same basis.

    Basis → default penalty:

    * ``"duchon"``               → function-norm (RKHS semi-norm ``∫(f^{(m)})² dx``)
    * ``"duchon_multipenalty"``  → triple-operator (mass + tension + stiffness)
    * ``"thinplate"`` (1D ``t``) → 1D thin-plate ≡ cubic smoothing spline ≡ Duchon ``m=2`` function-norm
    * ``"bspline"``              → P-spline 2nd-difference coefficient penalty
    """
    if isinstance(penalty, str):
        penalty_kind = str(penalty).strip().lower().replace("_", "-")
    else:
        penalty_kind = None

    if isinstance(penalty, str) or penalty is None:
        kind = str(basis_kind).strip().lower().replace("_", "").replace("-", "")
        if kind in {"duchon", "duchonspline"}:
            if penalty_kind in {None, "function-norm", "functionnorm", "rkhs"}:
                return _duchon_function_norm_penalty(
                    knots_or_centers,
                    m=int(basis_order),
                    periodic=bool(periodic),
                    period=None,
                )
            if penalty_kind in {"triple-operator", "tripleoperator", "operator"}:
                mass, tension, stiffness = _duchon_operator_penalties(
                    knots_or_centers,
                    m=int(basis_order),
                    periodic=False,
                    period=None,
                )
                return mass + tension + stiffness
            raise ValueError(f"unsupported Duchon penalty {penalty!r}")
        if kind in {"duchonmultipenalty", "duchontripleoperator"}:
            # The triple-operator basis carries the additive sum of mass +
            # tension + stiffness as its default single-λ penalty. The proper
            # multi-λ entry point is the ``smoothing="adam"`` route, which
            # accepts a length-3 ``log_lambda`` tensor and routes gradients
            # through each component individually.
            if penalty_kind not in {None, "triple-operator", "tripleoperator", "operator"}:
                raise ValueError(f"unsupported duchon_multipenalty penalty {penalty!r}")
            mass, tension, stiffness = _duchon_operator_penalties(
                knots_or_centers,
                m=int(basis_order),
                periodic=False,
                period=None,
            )
            return mass + tension + stiffness
        if kind in {"bspline", "spline"}:
            if penalty_kind not in {None, "coefficient-difference", "coefficientdifference", "difference"}:
                raise ValueError(f"unsupported B-spline penalty {penalty!r}")
            s, _ = smoothness_penalty(knots_or_centers, degree=int(basis_order), order=2)
            return s
        if kind in {"thinplate", "thinplatespline", "tps"}:
            # 1D thin-plate spline = cubic smoothing spline = Duchon m=2 with
            # bending-energy ``∫(f'')² dx``. Position-API positions are 1D,
            # so route the bending-energy penalty through the Duchon m=2
            # function-norm helper rather than the multi-dimensional thin-plate
            # path (which expects 2D centers and a different kernel).
            if penalty_kind not in {None, "function-norm", "functionnorm", "bending-energy", "bendingenergy"}:
                raise ValueError(f"unsupported thin-plate penalty {penalty!r}")
            return _duchon_function_norm_penalty(
                knots_or_centers,
                m=2,
                periodic=bool(periodic),
                period=None,
            )
    return _numeric_matrix(penalty, "penalty")


def _normalize_position_basis(
    basis_kind: str,
    basis_order: int | None,
) -> tuple[str, int, str]:
    """Resolve user-facing basis names to the engine's basis kind + order.

    Returns ``(effective_kind, effective_order, display_kind)``.

    * ``thinplate`` (1D positions): an alias for Duchon ``m=2`` — the cubic
      smoothing spline is the canonical 1D thin-plate spline.
    * ``duchon_multipenalty``: same engine basis as ``duchon`` (``m=2``);
      the difference is the penalty (triple-operator vs function-norm).
    """
    raw = str(basis_kind)
    kind = raw.strip().lower().replace("_", "").replace("-", "")
    if kind in {"thinplate", "thinplatespline", "tps"}:
        order = 2 if basis_order is None else int(basis_order)
        return ("duchon", order, raw)
    if kind in {"duchonmultipenalty", "duchontripleoperator"}:
        # Engine sees a plain Duchon basis; the penalty (triple-operator
        # combined) is constructed by `_resolve_position_penalty` from the
        # display_kind.
        order = 2 if basis_order is None else int(basis_order)
        return ("duchon", order, raw)
    return (raw, _position_basis_order(raw, basis_order), raw)


def _numeric_tensor3(values: Any, label: str) -> Any:
    import numpy as np

    arr = np.asarray(values)
    if arr.ndim != 3:
        raise ValueError(f"{label} must be a 3D numeric array")
    if 0 in arr.shape:
        raise ValueError(f"{label} cannot be empty")
    if arr.dtype != np.float64:
        raise TypeError(f"{label} must be a float64 numpy array for zero-copy FFI")
    return arr
