from __future__ import annotations

import json
from pathlib import Path
from typing import Any, overload

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


def duchon_basis_1d(
    t: Any,
    centers: Any = None,
    *,
    m: int = 2,
    periodic: bool = False,
) -> Any:
    """Evaluate the Rust one-dimensional Duchon basis as a NumPy array.

    ``centers`` may be:

    * ``None`` — auto-derive ``K = 10`` centers at empirical quantiles of ``t``.
    * an ``int`` ``K`` — auto-derive ``K`` quantile centers.
    * an array-like — used verbatim.
    """
    import numpy as np

    t_np = _numeric_vector(t, "t")
    centers_np = _resolve_centers(centers, t_np, label="centers")
    try:
        return np.asarray(
            rust_module().duchon_basis_1d(
                t_np,
                centers_np,
                int(m),
                bool(periodic),
            ),
            dtype=float,
        )
    except Exception as exc:
        raise map_exception(exc) from exc


def duchon_basis_1d_derivative(
    t: Any,
    centers: Any = None,
    *,
    m: int = 2,
    order: int = 1,
    periodic: bool = False,
) -> Any:
    """Evaluate derivatives of the Rust one-dimensional Duchon basis.

    ``centers`` accepts ``None`` / ``int`` / array — see :func:`duchon_basis_1d`.
    """
    import numpy as np

    t_np = _numeric_vector(t, "t")
    centers_np = _resolve_centers(centers, t_np, label="centers")
    try:
        return np.asarray(
            rust_module().duchon_basis_1d_derivative(
                t_np,
                centers_np,
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
    basis_kind: str,
    knots_or_centers: Any = None,
    penalty: Any | None = None,
    *,
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

    order = _position_basis_order(basis_kind, basis_order)
    t_np = _numeric_vector(t, "t")
    knots_np = _resolve_basis_locations(
        knots_or_centers,
        t_np,
        basis_kind=basis_kind,
        label="knots_or_centers",
        degree=order,
    )
    penalty_np = _resolve_position_penalty(
        penalty,
        knots_np,
        basis_kind=basis_kind,
        basis_order=order,
        periodic=periodic,
    )
    try:
        out = rust_module().gaussian_reml_fit_positions(
            t_np,
            _numeric_matrix(y, "y"),
            str(basis_kind),
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
    return _coerce_gaussian_reml_payload(out, np)


def gaussian_reml_fit_positions_backward(
    t: Any,
    y: Any,
    basis_kind: str,
    knots_or_centers: Any = None,
    penalty: Any | None = None,
    *,
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

    order = _position_basis_order(basis_kind, basis_order)
    t_np = _numeric_vector(t, "t")
    knots_np = _resolve_basis_locations(
        knots_or_centers,
        t_np,
        basis_kind=basis_kind,
        label="knots_or_centers",
        degree=order,
    )
    penalty_np = _resolve_position_penalty(
        penalty,
        knots_np,
        basis_kind=basis_kind,
        basis_order=order,
        periodic=periodic,
    )
    try:
        out = rust_module().gaussian_reml_fit_positions_backward(
            t_np,
            _numeric_matrix(y, "y"),
            str(basis_kind),
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
    basis_kind: str,
    knots_or_centers: Any = None,
    penalty: Any | None = None,
    *,
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

    order = _position_basis_order(basis_kind, basis_order)
    t_np = _numeric_vector(t, "t")
    knots_np = _resolve_basis_locations(
        knots_or_centers,
        t_np,
        basis_kind=basis_kind,
        label="knots_or_centers",
        degree=order,
    )
    penalty_np = _resolve_position_penalty(
        penalty,
        knots_np,
        basis_kind=basis_kind,
        basis_order=order,
        periodic=periodic,
    )
    try:
        out = rust_module().gaussian_reml_fit_positions_batched(
            t_np,
            _numeric_matrix(y, "y"),
            _index_vector(row_offsets, "row_offsets"),
            str(basis_kind),
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
    return _coerce_gaussian_reml_payload(out, np)


def gaussian_reml_fit_positions_batched_backward(
    t: Any,
    y: Any,
    row_offsets: Any,
    basis_kind: str,
    knots_or_centers: Any = None,
    penalty: Any | None = None,
    *,
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
    order = _position_basis_order(basis_kind, basis_order)
    t_np = _numeric_vector(t, "t")
    knots_np = _resolve_basis_locations(
        knots_or_centers,
        t_np,
        basis_kind=basis_kind,
        label="knots_or_centers",
        degree=order,
    )
    penalty_np = _resolve_position_penalty(
        penalty,
        knots_np,
        basis_kind=basis_kind,
        basis_order=order,
        periodic=periodic,
    )
    try:
        out = rust_module().gaussian_reml_fit_positions_batched_backward(
            t_np,
            _numeric_matrix(y, "y"),
            offsets,
            str(basis_kind),
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
        # build_duchon_basis appends the polynomial nullspace columns
        # to the radial-basis block; for 1-D Duchon the nullspace
        # dimension equals ``m`` (the basis order).
        return n + max(int(basis_order), 0)
    # Clamped B-spline: ncols = len(knots) - degree - 1. The Rust impl
    # uses the same convention for the periodic variant; rely on the
    # engine to reject mismatches if a periodic caller supplies a custom
    # penalty themselves.
    return max(n - int(basis_order) - 1, 0)


def _resolve_position_penalty(
    penalty: Any | None,
    knots_or_centers: Any,
    *,
    basis_kind: str,
    basis_order: int,
    periodic: bool,
) -> Any:
    """Resolve the penalty argument for ``gaussian_reml_fit_positions*``.

    ``None`` → identity matrix of basis-dimension size (a neutral ridge
    that lets REML pick ``lambda`` from data). Otherwise forwarded to
    :func:`_numeric_matrix` unchanged.
    """
    import numpy as np

    if penalty is None:
        dim = _position_basis_dim(knots_or_centers, basis_kind, basis_order, periodic)
        if dim <= 0:
            raise ValueError(
                "cannot auto-derive identity penalty: inferred basis dim is non-positive "
                f"(basis_kind={basis_kind!r}, locations.size={knots_or_centers.shape[0]}, "
                f"basis_order={basis_order})"
            )
        return np.eye(dim, dtype=np.float64)
    return _numeric_matrix(penalty, "penalty")


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
