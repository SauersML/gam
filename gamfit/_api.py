from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._binding import RustExtensionUnavailableError, extension_status, rust_module
from ._exceptions import map_exception
from ._model import Model
from ._response_geometry import ResponseGeometryModel, fit_response_geometry
from ._tables import normalize_table
from ._validation import FormulaValidation


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
    frailty_kind: str | None = None,
    frailty_sd: float | None = None,
    hazard_loading: str | None = None,
    scale_dimensions: bool | None = None,
    adaptive_regularization: bool | None = None,
    firth: bool | None = None,
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
    config: dict[str, Any] | None = None,
) -> Model:
    """Fit directly from numeric NumPy-compatible arrays.

    ``X`` is named ``x0``, ``x1``, ... at the formula boundary. A one-column
    ``Y`` is named from the formula response; multi-column ``Y`` is named
    ``y0``, ``y1``, ...
    """
    X_arr = _numeric_matrix(X, "X")
    Y_arr = _numeric_matrix(Y, "Y")
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
        config=dict(config or {}) or None,
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
    knots: Any,
    *,
    degree: int = 3,
    periodic: bool = False,
) -> Any:
    """Evaluate the Rust B-spline basis as a NumPy array."""
    import numpy as np

    try:
        return np.asarray(
            rust_module().bspline_basis(
                _numeric_vector(t, "t"),
                _numeric_vector(knots, "knots"),
                int(degree),
                bool(periodic),
            ),
            dtype=float,
        )
    except Exception as exc:
        raise map_exception(exc) from exc


def bspline_basis_derivative(
    t: Any,
    knots: Any,
    *,
    degree: int = 3,
    order: int = 1,
    periodic: bool = False,
) -> Any:
    """Evaluate derivatives of the Rust B-spline basis as a NumPy array."""
    import numpy as np

    try:
        return np.asarray(
            rust_module().bspline_basis_derivative(
                _numeric_vector(t, "t"),
                _numeric_vector(knots, "knots"),
                int(degree),
                int(order),
                bool(periodic),
            ),
            dtype=float,
        )
    except Exception as exc:
        raise map_exception(exc) from exc


def duchon_basis_1d(
    t: Any,
    centers: Any,
    *,
    m: int = 2,
    periodic: bool = False,
) -> Any:
    """Evaluate the Rust one-dimensional Duchon basis as a NumPy array."""
    import numpy as np

    try:
        return np.asarray(
            rust_module().duchon_basis_1d(
                _numeric_vector(t, "t"),
                _numeric_vector(centers, "centers"),
                int(m),
                bool(periodic),
            ),
            dtype=float,
        )
    except Exception as exc:
        raise map_exception(exc) from exc


def duchon_basis_1d_derivative(
    t: Any,
    centers: Any,
    *,
    m: int = 2,
    order: int = 1,
    periodic: bool = False,
) -> Any:
    """Evaluate derivatives of the Rust one-dimensional Duchon basis."""
    import numpy as np

    try:
        return np.asarray(
            rust_module().duchon_basis_1d_derivative(
                _numeric_vector(t, "t"),
                _numeric_vector(centers, "centers"),
                int(m),
                int(order),
                bool(periodic),
            ),
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
    """Return ``(S, null_basis)`` for the Rust B-spline difference penalty."""
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


def _numeric_vector(values: Any, label: str) -> Any:
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{label} must be a 1D numeric array")
    if arr.size == 0:
        raise ValueError(f"{label} cannot be empty")
    return np.ascontiguousarray(arr)


def _numeric_matrix(values: Any, label: str) -> Any:
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{label} must be a 1D or 2D numeric array")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{label} cannot be empty")
    return np.ascontiguousarray(arr)
