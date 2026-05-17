use csv::StringRecord;
use faer::Side;
use gam::basis::create_duchon_basis_1d_derivative_dense;
use gam::bernoulli_marginal_slope::{
    BernoulliMarginalSlopeFitResult, DeviationRuntime, LatentMeasureKind,
};
use gam::estimate::{
    BlockRole, EstimationError, saved_latent_cloglog_state_from_fit, saved_mixture_state_from_fit,
    saved_sas_state_from_fit,
};
use gam::faer_ndarray::{array2_to_matmut, factorize_symmetricwith_fallback};
use gam::families::family_meta::{
    family_to_link, inverse_link_to_binomial_family, pretty_familyname,
};
use gam::families::scale_design::{build_scale_deviation_transform, infer_non_intercept_start};
use gam::families::survival_construction::survival_likelihood_modename;
use gam::families::survival_predict::{
    apply_inverse_link_state_to_fit_result, fit_result_from_saved_model_for_prediction,
};
use gam::gamlss::{BinomialLocationScaleFitResult, GaussianLocationScaleFitResult};
use gam::gaussian_reml::{
    gaussian_reml_multi_closed_form_backward, gaussian_reml_multi_closed_form_backward_from_fit,
    gaussian_reml_multi_closed_form_with_cache,
};
use gam::hmc::{NutsConfig, NutsResult};
use gam::inference::data::{
    EncodedDataset, UnseenCategoryPolicy, encode_recordswith_inferred_schema,
    encode_recordswith_schema,
};
use gam::inference::formula_dsl::{ParsedTerm, parse_formula, parse_surv_response};
use gam::inference::model::{
    ColumnKindTag, DataSchema, FittedFamily, FittedModel, FittedModelPayload,
    MODEL_PAYLOAD_VERSION, ModelKind, PredictModelClass, SavedAnchoredDeviationRuntime,
    SavedLatentZNormalization, SchemaColumn,
};
use gam::inference::predict_input::build_predict_input_for_model;
use gam::report::{CoefficientRow, EdfBlockRow, ReportInput, render_html};
use gam::smooth::{TermCollectionSpec, freeze_term_collection_from_design};
use gam::survival_marginal_slope::SurvivalMarginalSlopeFitResult;
use gam::terms::basis::{
    BasisOptions, CenterStrategy, Dense, DuchonBasisSpec, DuchonNullspaceOrder,
    SpatialIdentifiability, build_duchon_basis, create_basis, create_difference_penalty_matrix,
    create_periodic_bspline_basis_dense, create_periodic_bspline_derivative_dense,
};
use gam::transformation_normal::TransformationNormalFitResult;
use gam::types::{InverseLink, LikelihoodFamily};
use gam::{FitConfig, FitRequest, FitResult, fit_model, materialize, resolve_offset_column};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis, s};
use numpy::{
    IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};

const MODEL_VERSION: u32 = MODEL_PAYLOAD_VERSION;

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct PyFitConfig {
    family: Option<String>,
    offset: Option<String>,
    weights: Option<String>,
    ridge_lambda: Option<f64>,

    // Transformation-normal routing.
    transformation_normal: Option<bool>,

    // Survival-specific.
    survival_likelihood: Option<String>,
    baseline_target: Option<String>,
    baseline_scale: Option<f64>,
    baseline_shape: Option<f64>,
    baseline_rate: Option<f64>,
    baseline_makeham: Option<f64>,

    // Marginal-slope.
    z_column: Option<String>,
    logslope_formula: Option<String>,

    // Link / flexibility.
    link: Option<String>,
    flexible_link: Option<bool>,
    scale_dimensions: Option<bool>,
    adaptive_regularization: Option<bool>,

    // Location-scale (GAMLSS).
    noise_formula: Option<String>,
    noise_offset: Option<String>,

    firth: Option<bool>,

    // Frailty (only consumed by survival families today). Mirrors the CLI
    // names: --frailty-kind, --frailty-sd, --hazard-loading.
    frailty_kind: Option<String>,
    frailty_sd: Option<f64>,
    hazard_loading: Option<String>,
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct PySampleOptions {
    /// Posterior draws per chain (after warmup). When omitted, falls back to
    /// `NutsConfig::for_dimension`.
    samples: Option<usize>,
    /// Warmup iterations per chain. When omitted, matches `samples` via the
    /// dimension-adaptive default.
    warmup: Option<usize>,
    /// Number of parallel chains.
    chains: Option<usize>,
    /// Target HMC acceptance rate (0, 1).
    target_accept: Option<f64>,
    /// RNG seed for deterministic chain initialisation.
    seed: Option<u64>,
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct PyPredictOptions {
    interval: Option<f64>,
    time_grid: Option<Vec<f64>>,
    /// Request delta-method standard errors on the survival surface
    /// (and the linear predictor at each row's exit time).  Honored
    /// only by the survival prediction path.  When set, the resulting
    /// payload carries `survival_se` (per-cell SE) and `eta_se` (per-row
    /// SE at the exit time).
    #[serde(default)]
    with_uncertainty: bool,
}

#[derive(Serialize)]
struct SummaryCoefficientRow {
    index: usize,
    estimate: f64,
    std_error: Option<f64>,
}

#[derive(Serialize)]
struct SummaryPayload {
    formula: String,
    family_name: String,
    model_class: String,
    deviance: f64,
    reml_score: f64,
    iterations: usize,
    edf_total: Option<f64>,
    coefficients: Vec<SummaryCoefficientRow>,
}

#[derive(Serialize)]
struct SchemaIssue {
    kind: String,
    message: String,
    column: Option<String>,
}

#[derive(Serialize)]
struct SchemaCheckPayload {
    ok: bool,
    issues: Vec<SchemaIssue>,
}

#[derive(Serialize)]
struct PredictionPayload {
    columns: BTreeMap<String, Vec<f64>>,
}

/// JSON wire format for NUTS posterior draws.
///
/// `samples_flat` is a row-major flattening of an `(n_draws, n_coeffs)`
/// matrix.  The Python side rebuilds the numpy array via
/// `np.asarray(samples_flat).reshape(n_draws, n_coeffs)` — flat lists round
/// trip through `serde_json` faster than nested ones at the sizes typical
/// for biobank-scale work (millions of doubles), and they sidestep the
/// per-row Python list construction cost.
#[derive(Serialize)]
struct SamplePayload {
    samples_flat: Vec<f64>,
    n_draws: usize,
    n_coeffs: usize,
    coefficient_names: Vec<String>,
    posterior_mean: Vec<f64>,
    posterior_std: Vec<f64>,
    rhat: f64,
    ess: f64,
    converged: bool,
    config: SampleConfigPayload,
    /// Short identifier for the saved model's predictive class (e.g.
    /// "standard", "gaussian location-scale"). Lets the Python wrapper
    /// pick the right posterior-predict path without re-parsing the
    /// model.
    model_class: String,
    /// Inverse-link kind tag used by the Python wrapper to apply the
    /// correct response-scale transform (`identity`, `logit`, `probit`,
    /// `cloglog`, `log`, ...).
    family_kind: String,
    /// Whether the draws came from exact NUTS or the Laplace-Gaussian
    /// fallback. Currently only "nuts" or "laplace"; callers can use
    /// this to badge the posterior or to warn when a class has fallen
    /// back to the approximate path.
    method: String,
}

#[derive(Serialize)]
struct SampleConfigPayload {
    n_samples: usize,
    n_warmup: usize,
    n_chains: usize,
    target_accept: f64,
    seed: u64,
}

#[derive(Serialize)]
struct SurvivalPredictionPayload {
    class: &'static str,
    model_class: String,
    likelihood_mode: String,
    times: Vec<f64>,
    hazard: Vec<Vec<f64>>,
    survival: Vec<Vec<f64>>,
    cumulative_hazard: Vec<Vec<f64>>,
    linear_predictor: Vec<f64>,
    columns: BTreeMap<String, Vec<f64>>,
    /// Delta-method standard errors on the survival surface, when the
    /// caller requested `with_uncertainty=true`.  Same shape as
    /// `survival`.  `None` otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    survival_se: Option<Vec<Vec<f64>>>,
    /// Delta-method SE on the linear predictor at each row's own exit
    /// time, when uncertainty was requested.  Length equals
    /// `linear_predictor.len()`.
    #[serde(skip_serializing_if = "Option::is_none")]
    eta_se: Option<Vec<f64>>,
}

#[derive(Serialize)]
struct ValidationPayload {
    formula: String,
    family_name: String,
    model_class: String,
    response_column: Option<String>,
    columns: Vec<String>,
    n_rows: usize,
    n_columns: usize,
    supported_by_python: bool,
}

#[pyfunction]
fn build_info(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let info = PyDict::new(py);
    info.set_item("crate", "gam-pyffi")?;
    info.set_item("engine_crate", "gam")?;
    info.set_item("python_module", "gam._rust")?;
    info.set_item("abi3", "cp310+")?;
    info.set_item("version", env!("CARGO_PKG_VERSION"))?;
    info.set_item(
        "capabilities",
        vec![
            "fit",
            "fit_array",
            "load",
            "predict",
            "predict_array",
            "sample",
            "summary",
            "check",
            "report",
            "save",
            "validate_formula",
            "design_matrix_array",
            "basis",
            "gaussian_weighted_ridge_array",
            "gaussian_weighted_ridge_batch",
            "gaussian_reml_fit",
            "gaussian_reml_fit_backward",
            "gaussian_reml_fit_batched",
            "gaussian_reml_fit_batched_backward",
            "gaussian_reml_fit_positions",
            "gaussian_reml_fit_positions_backward",
            "gaussian_reml_fit_positions_batched",
            "gaussian_reml_fit_positions_batched_backward",
            "gaussian_reml_fit_formula_table",
        ],
    )?;
    info.set_item(
        "supported_model_classes",
        vec![
            "standard",
            "transformation-normal",
            "survival",
            "bernoulli-marginal-slope",
            "survival-marginal-slope",
            "survival-location-scale",
            "latent-survival",
            "latent-binary",
            "gaussian-location-scale",
            "binomial-location-scale",
        ],
    )?;
    Ok(info.unbind())
}

#[pyfunction]
fn fit_table(
    py: Python<'_>,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    formula: String,
    config_json: Option<String>,
) -> PyResult<Py<PyBytes>> {
    // PyO3 0.28 names the old `allow_threads` API `detach`: the closure
    // runs without the GIL, so Python signal handling (KeyboardInterrupt,
    // SIGALRM handlers, etc.) can run while the Rust solver is in progress.
    let model_bytes = py
        .detach(move || fit_table_impl(headers, rows, formula, config_json.as_deref()))
        .map_err(py_value_error)?;
    Ok(PyBytes::new(py, &model_bytes).unbind())
}

#[pyfunction]
fn fit_array(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray2<'_, f64>,
    formula: String,
    config_json: Option<String>,
) -> PyResult<Py<PyBytes>> {
    let model_bytes = fit_array_impl(x.as_array(), y.as_array(), formula, config_json.as_deref())
        .map_err(py_value_error)?;
    Ok(PyBytes::new(py, &model_bytes).unbind())
}

#[pyfunction]
fn load_model(py: Python<'_>, model_bytes: Vec<u8>) -> PyResult<()> {
    py.detach(move || load_model_impl(&model_bytes))
        .map_err(py_value_error)?;
    Ok(())
}

#[pyfunction]
fn validate_formula_json(
    py: Python<'_>,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    formula: String,
    config_json: Option<String>,
) -> PyResult<String> {
    py.detach(move || validate_formula_json_impl(headers, rows, formula, config_json.as_deref()))
        .map_err(py_value_error)
}

#[pyfunction]
fn predict_table(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    options_json: Option<String>,
) -> PyResult<String> {
    py.detach(move || predict_table_impl(&model_bytes, headers, rows, options_json.as_deref()))
        .map_err(py_value_error)
}

#[pyfunction]
fn predict_array<'py>(
    py: Python<'py>,
    model_bytes: Vec<u8>,
    x: PyReadonlyArray2<'py, f64>,
    options_json: Option<String>,
) -> PyResult<Py<PyArray2<f64>>> {
    let out = predict_array_impl(&model_bytes, x.as_array(), options_json.as_deref())
        .map_err(py_value_error)?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction]
fn sample_table(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    options_json: Option<String>,
) -> PyResult<String> {
    py.detach(move || sample_table_impl(&model_bytes, headers, rows, options_json.as_deref()))
        .map_err(py_value_error)
}

#[pyfunction]
fn design_matrix_table(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
) -> PyResult<String> {
    py.detach(move || design_matrix_table_impl(&model_bytes, headers, rows))
        .map_err(py_value_error)
}

#[pyfunction]
fn design_matrix_array<'py>(
    py: Python<'py>,
    model_bytes: Vec<u8>,
    x: PyReadonlyArray2<'py, f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let out = design_matrix_array_impl(&model_bytes, x.as_array()).map_err(py_value_error)?;
    Ok(out.into_pyarray(py).unbind())
}

#[pyfunction(signature = (t, knots, degree = 3, periodic = false))]
fn bspline_basis<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    knots: PyReadonlyArray1<'py, f64>,
    degree: usize,
    periodic: bool,
) -> PyResult<Py<PyArray2<f64>>> {
    let basis = bspline_basis_impl(t.as_array(), knots.as_array(), degree, periodic)
        .map_err(py_value_error)?;
    Ok(basis.into_pyarray(py).unbind())
}

#[pyfunction(signature = (t, knots, degree = 3, order = 1, periodic = false))]
fn bspline_basis_derivative<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    knots: PyReadonlyArray1<'py, f64>,
    degree: usize,
    order: usize,
    periodic: bool,
) -> PyResult<Py<PyArray2<f64>>> {
    let basis =
        bspline_basis_derivative_impl(t.as_array(), knots.as_array(), degree, order, periodic)
            .map_err(py_value_error)?;
    Ok(basis.into_pyarray(py).unbind())
}

#[pyfunction(signature = (t, centers, m = 2, periodic = false))]
fn duchon_basis_1d<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    centers: PyReadonlyArray1<'py, f64>,
    m: usize,
    periodic: bool,
) -> PyResult<Py<PyArray2<f64>>> {
    let basis = duchon_basis_1d_impl(t.as_array(), centers.as_array(), m, periodic)
        .map_err(py_value_error)?;
    Ok(basis.into_pyarray(py).unbind())
}

#[pyfunction(signature = (t, centers, m = 2, order = 1, periodic = false))]
fn duchon_basis_1d_derivative<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    centers: PyReadonlyArray1<'py, f64>,
    m: usize,
    order: usize,
    periodic: bool,
) -> PyResult<Py<PyArray2<f64>>> {
    let basis =
        duchon_basis_1d_derivative_impl(t.as_array(), centers.as_array(), m, order, periodic)
            .map_err(py_value_error)?;
    Ok(basis.into_pyarray(py).unbind())
}

#[pyfunction(signature = (knots, degree = 3, order = 2))]
fn smoothness_penalty<'py>(
    py: Python<'py>,
    knots: PyReadonlyArray1<'py, f64>,
    degree: usize,
    order: usize,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    let (penalty, null_basis) =
        smoothness_penalty_impl(knots.as_array(), degree, order).map_err(py_value_error)?;
    Ok((
        penalty.into_pyarray(py).unbind(),
        null_basis.into_pyarray(py).unbind(),
    ))
}

#[pyfunction(signature = (x, y, penalty, weights, ridge_lambda))]
fn gaussian_weighted_ridge_array<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    penalty: PyReadonlyArray2<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    ridge_lambda: f64,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    let (coefficients, fitted) = gaussian_weighted_ridge_array_impl(
        x.as_array(),
        y.as_array(),
        penalty.as_array(),
        weights.as_array(),
        ridge_lambda,
    )
    .map_err(|err| py_value_error(err.to_string()))?;
    Ok((
        coefficients.into_pyarray(py).unbind(),
        fitted.into_pyarray(py).unbind(),
    ))
}

#[pyfunction(signature = (x, y, penalty, weights, ridge_lambda, row_counts = None))]
fn gaussian_weighted_ridge_batch<'py>(
    py: Python<'py>,
    x: PyReadonlyArray3<'py, f64>,
    y: PyReadonlyArray3<'py, f64>,
    penalty: PyReadonlyArray2<'py, f64>,
    weights: PyReadonlyArray2<'py, f64>,
    ridge_lambda: f64,
    row_counts: Option<PyReadonlyArray1<'py, usize>>,
) -> PyResult<(Py<PyArray3<f64>>, Py<PyArray3<f64>>)> {
    let row_count_view = row_counts.as_ref().map(|counts| counts.as_array());
    let (coefficients, fitted) = gaussian_weighted_ridge_batch_impl(
        x.as_array(),
        y.as_array(),
        penalty.as_array(),
        weights.as_array(),
        ridge_lambda,
        row_count_view,
    )
    .map_err(py_value_error)?;
    Ok((
        coefficients.into_pyarray(py).unbind(),
        fitted.into_pyarray(py).unbind(),
    ))
}

#[pyfunction(signature = (x, y, penalty, weights = None, init_lambda = None, by = None, by_start_col = 0))]
fn gaussian_reml_fit<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    penalty: PyReadonlyArray2<'py, f64>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    init_lambda: Option<f64>,
    by: Option<PyReadonlyArray1<'py, f64>>,
    by_start_col: usize,
) -> PyResult<Py<PyDict>> {
    let x_view = x.as_array();
    let y_view = y.as_array();
    let penalty_view = penalty.as_array();
    let n_rows = x_view.nrows();
    let n_outputs = y_view.ncols();
    let n_coefficients = penalty_view.nrows();
    let weight_view = weights.as_ref().map(|w| w.as_array());
    let by_view = by.as_ref().map(|b| b.as_array());
    let gated_x;
    let fit_x = if let Some(by_values) = by_view {
        gated_x = apply_by_gate(x_view, by_values, by_start_col).map_err(py_value_error)?;
        gated_x.view()
    } else {
        x_view
    };
    let result = gaussian_reml_multi_closed_form_with_cache(
        fit_x,
        y_view,
        penalty_view,
        weight_view,
        init_lambda,
        None,
    );
    let out = PyDict::new(py);
    match result {
        Ok(fit) => {
            set_ok_gaussian_reml_items(py, &out, fit)?;
        }
        Err(EstimationError::ModelIsIllConditioned { .. }) => {
            set_degenerate_gaussian_reml_items(py, &out, n_rows, n_outputs, n_coefficients)?;
        }
        Err(err) => return Err(py_value_error(err.to_string())),
    }
    Ok(out.unbind())
}

#[pyfunction(signature = (
    x,
    y,
    penalty,
    grad_lambda = 0.0,
    grad_coefficients = None,
    grad_fitted = None,
    grad_reml_score = 0.0,
    forward_state = None,
    weights = None,
    init_lambda = None,
    by = None,
    by_start_col = 0
))]
fn gaussian_reml_fit_backward<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    penalty: PyReadonlyArray2<'py, f64>,
    grad_lambda: f64,
    grad_coefficients: Option<PyReadonlyArray2<'py, f64>>,
    grad_fitted: Option<PyReadonlyArray2<'py, f64>>,
    grad_reml_score: f64,
    forward_state: Option<&Bound<'py, PyDict>>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    init_lambda: Option<f64>,
    by: Option<PyReadonlyArray1<'py, f64>>,
    by_start_col: usize,
) -> PyResult<Py<PyDict>> {
    let x_view = x.as_array();
    let y_view = y.as_array();
    let penalty_view = penalty.as_array();
    let weight_view = weights.as_ref().map(|w| w.as_array());
    let by_view = by.as_ref().map(|b| b.as_array());
    let gated_x;
    let fit_x = if let Some(by_values) = by_view {
        gated_x = apply_by_gate(x_view, by_values, by_start_col).map_err(py_value_error)?;
        gated_x.view()
    } else {
        x_view
    };
    let backward = if let Some(state) = forward_state {
        let fit = gaussian_reml_fit_state_from_pydict(state).map_err(py_value_error)?;
        gaussian_reml_multi_closed_form_backward_from_fit(
            fit_x,
            y_view,
            penalty_view,
            weight_view,
            &fit,
            grad_lambda,
            grad_coefficients.as_ref().map(|g| g.as_array()),
            grad_fitted.as_ref().map(|g| g.as_array()),
            grad_reml_score,
        )
    } else {
        gaussian_reml_multi_closed_form_backward(
            fit_x,
            y_view,
            penalty_view,
            weight_view,
            init_lambda,
            grad_lambda,
            grad_coefficients.as_ref().map(|g| g.as_array()),
            grad_fitted.as_ref().map(|g| g.as_array()),
            grad_reml_score,
        )
    }
    .map_err(|err| py_value_error(err.to_string()))?;

    let out = PyDict::new(py);
    let (grad_x, grad_by) = if let Some(by_values) = by_view {
        let (grad_x, grad_by) =
            apply_by_gate_backward(x_view, by_values, by_start_col, backward.grad_x.view())
                .map_err(py_value_error)?;
        (grad_x, Some(grad_by))
    } else {
        (backward.grad_x, None)
    };
    out.set_item("grad_x", grad_x.into_pyarray(py))?;
    out.set_item("grad_y", backward.grad_y.into_pyarray(py))?;
    out.set_item("grad_weights", backward.grad_weights.into_pyarray(py))?;
    if let Some(grad_by) = grad_by {
        out.set_item("grad_by", grad_by.into_pyarray(py))?;
    } else {
        out.set_item("grad_by", py.None())?;
    }
    Ok(out.unbind())
}

#[pyfunction(signature = (headers, rows, formula, y, config_json = None))]
fn gaussian_reml_fit_formula_table<'py>(
    py: Python<'py>,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    formula: String,
    y: PyReadonlyArray2<'py, f64>,
    config_json: Option<String>,
) -> PyResult<Py<PyDict>> {
    let y_values = y.as_array().to_owned();
    let result = py
        .detach(move || {
            gaussian_reml_fit_formula_table_impl(
                headers,
                rows,
                formula,
                y_values.view(),
                config_json.as_deref(),
            )
        })
        .map_err(py_value_error)?;
    gaussian_reml_result_to_pydict(py, result)
}

#[pyfunction(signature = (x, y, row_offsets, penalty, weights = None, init_lambda = None))]
fn gaussian_reml_fit_batched<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    row_offsets: PyReadonlyArray1<'py, usize>,
    penalty: PyReadonlyArray2<'py, f64>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    init_lambda: Option<f64>,
) -> PyResult<Py<PyDict>> {
    let weight_view = weights.as_ref().map(|w| w.as_array());
    let result = gaussian_reml_fit_batched_impl(
        x.as_array(),
        y.as_array(),
        row_offsets.as_array(),
        penalty.as_array(),
        weight_view,
        init_lambda,
    )
    .map_err(py_value_error)?;
    let out = PyDict::new(py);
    out.set_item("status", result.statuses)?;
    out.set_item("lambda", result.lambdas.into_pyarray(py))?;
    out.set_item("rho", result.rhos.into_pyarray(py))?;
    out.set_item("reml_score", result.reml_scores.into_pyarray(py))?;
    out.set_item(
        "reml_grad_lambda",
        result.reml_grad_lambdas.into_pyarray(py),
    )?;
    out.set_item(
        "reml_hess_lambda",
        result.reml_hess_lambdas.into_pyarray(py),
    )?;
    out.set_item("reml_grad_rho", result.reml_grad_rhos.into_pyarray(py))?;
    out.set_item("reml_hess_rho", result.reml_hess_rhos.into_pyarray(py))?;
    out.set_item("edf", result.edf.into_pyarray(py))?;
    out.set_item("coefficients", result.coefficients.into_pyarray(py))?;
    out.set_item("fitted", result.fitted.into_pyarray(py))?;
    out.set_item("sigma2", result.sigma2.into_pyarray(py))?;
    Ok(out.unbind())
}

#[pyfunction(signature = (
    x,
    y,
    row_offsets,
    penalty,
    grad_lambda = None,
    grad_coefficients = None,
    grad_fitted = None,
    grad_reml_score = None,
    weights = None,
    init_lambda = None,
    by = None,
    by_start_col = 0
))]
fn gaussian_reml_fit_batched_backward<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    row_offsets: PyReadonlyArray1<'py, usize>,
    penalty: PyReadonlyArray2<'py, f64>,
    grad_lambda: Option<PyReadonlyArray1<'py, f64>>,
    grad_coefficients: Option<PyReadonlyArray3<'py, f64>>,
    grad_fitted: Option<PyReadonlyArray2<'py, f64>>,
    grad_reml_score: Option<PyReadonlyArray1<'py, f64>>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    init_lambda: Option<f64>,
    by: Option<PyReadonlyArray1<'py, f64>>,
    by_start_col: usize,
) -> PyResult<Py<PyDict>> {
    let x_view = x.as_array();
    let y_view = y.as_array();
    let weight_view = weights.as_ref().map(|w| w.as_array());
    let by_view = by.as_ref().map(|b| b.as_array());
    let gated_x;
    let fit_x = if let Some(by_values) = by_view {
        gated_x = apply_by_gate(x_view, by_values, by_start_col).map_err(py_value_error)?;
        gated_x.view()
    } else {
        x_view
    };
    let backward = gaussian_reml_fit_batched_backward_impl(
        fit_x,
        y_view,
        row_offsets.as_array(),
        penalty.as_array(),
        weight_view,
        init_lambda,
        grad_lambda.as_ref().map(|g| g.as_array()),
        grad_coefficients.as_ref().map(|g| g.as_array()),
        grad_fitted.as_ref().map(|g| g.as_array()),
        grad_reml_score.as_ref().map(|g| g.as_array()),
    )
    .map_err(py_value_error)?;

    let out = PyDict::new(py);
    let (grad_x, grad_by) = if let Some(by_values) = by_view {
        let (grad_x, grad_by) =
            apply_by_gate_backward(x_view, by_values, by_start_col, backward.grad_x.view())
                .map_err(py_value_error)?;
        (grad_x, Some(grad_by))
    } else {
        (backward.grad_x, None)
    };
    out.set_item("status", backward.statuses)?;
    out.set_item("grad_x", grad_x.into_pyarray(py))?;
    out.set_item("grad_y", backward.grad_y.into_pyarray(py))?;
    out.set_item("grad_weights", backward.grad_weights.into_pyarray(py))?;
    if let Some(grad_by) = grad_by {
        out.set_item("grad_by", grad_by.into_pyarray(py))?;
    } else {
        out.set_item("grad_by", py.None())?;
    }
    Ok(out.unbind())
}

#[pyfunction(signature = (
    t,
    y,
    basis_kind,
    knots_or_centers,
    penalty,
    basis_order = 3,
    periodic = false,
    period = None,
    weights = None,
    init_lambda = None,
    by = None,
    by_start_col = 0
))]
fn gaussian_reml_fit_positions<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    basis_kind: String,
    knots_or_centers: PyReadonlyArray1<'py, f64>,
    penalty: PyReadonlyArray2<'py, f64>,
    basis_order: usize,
    periodic: bool,
    period: Option<f64>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    init_lambda: Option<f64>,
    by: Option<PyReadonlyArray1<'py, f64>>,
    by_start_col: usize,
) -> PyResult<Py<PyDict>> {
    let x = position_basis_design(
        t.as_array(),
        knots_or_centers.as_array(),
        &basis_kind,
        basis_order,
        periodic,
        period,
    )
    .map_err(py_value_error)?;
    let y_view = y.as_array();
    let penalty_view = penalty.as_array();
    let weight_view = weights.as_ref().map(|w| w.as_array());
    let by_view = by.as_ref().map(|b| b.as_array());
    let gated_x;
    let fit_x = if let Some(by_values) = by_view {
        gated_x = apply_by_gate(x.view(), by_values, by_start_col).map_err(py_value_error)?;
        gated_x.view()
    } else {
        x.view()
    };
    let result = gaussian_reml_multi_closed_form_with_cache(
        fit_x,
        y_view,
        penalty_view,
        weight_view,
        init_lambda,
        None,
    );
    let out = PyDict::new(py);
    match result {
        Ok(fit) => set_ok_gaussian_reml_items(py, &out, fit)?,
        Err(EstimationError::ModelIsIllConditioned { .. }) => {
            set_degenerate_gaussian_reml_items(
                py,
                &out,
                x.nrows(),
                y_view.ncols(),
                penalty_view.nrows(),
            )?;
        }
        Err(err) => return Err(py_value_error(err.to_string())),
    }
    Ok(out.unbind())
}

#[pyfunction(signature = (
    t,
    y,
    basis_kind,
    knots_or_centers,
    penalty,
    grad_lambda = 0.0,
    grad_coefficients = None,
    grad_fitted = None,
    grad_reml_score = 0.0,
    basis_order = 3,
    periodic = false,
    period = None,
    weights = None,
    init_lambda = None,
    by = None,
    by_start_col = 0
))]
fn gaussian_reml_fit_positions_backward<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    basis_kind: String,
    knots_or_centers: PyReadonlyArray1<'py, f64>,
    penalty: PyReadonlyArray2<'py, f64>,
    grad_lambda: f64,
    grad_coefficients: Option<PyReadonlyArray2<'py, f64>>,
    grad_fitted: Option<PyReadonlyArray2<'py, f64>>,
    grad_reml_score: f64,
    basis_order: usize,
    periodic: bool,
    period: Option<f64>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    init_lambda: Option<f64>,
    by: Option<PyReadonlyArray1<'py, f64>>,
    by_start_col: usize,
) -> PyResult<Py<PyDict>> {
    let backward = gaussian_reml_fit_positions_backward_impl(
        t.as_array(),
        y.as_array(),
        knots_or_centers.as_array(),
        &basis_kind,
        basis_order,
        periodic,
        period,
        penalty.as_array(),
        weights.as_ref().map(|w| w.as_array()),
        init_lambda,
        grad_lambda,
        grad_coefficients.as_ref().map(|g| g.as_array()),
        grad_fitted.as_ref().map(|g| g.as_array()),
        grad_reml_score,
        by.as_ref().map(|b| b.as_array()),
        by_start_col,
    )
    .map_err(py_value_error)?;

    let out = PyDict::new(py);
    out.set_item("grad_t", backward.grad_t.into_pyarray(py))?;
    out.set_item("grad_y", backward.grad_y.into_pyarray(py))?;
    out.set_item("grad_weights", backward.grad_weights.into_pyarray(py))?;
    if let Some(grad_by) = backward.grad_by {
        out.set_item("grad_by", grad_by.into_pyarray(py))?;
    } else {
        out.set_item("grad_by", py.None())?;
    }
    Ok(out.unbind())
}

#[pyfunction(signature = (
    t,
    y,
    row_offsets,
    basis_kind,
    knots_or_centers,
    penalty,
    basis_order = 3,
    periodic = false,
    period = None,
    weights = None,
    init_lambda = None,
    by = None,
    by_start_col = 0
))]
fn gaussian_reml_fit_positions_batched<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    row_offsets: PyReadonlyArray1<'py, usize>,
    basis_kind: String,
    knots_or_centers: PyReadonlyArray1<'py, f64>,
    penalty: PyReadonlyArray2<'py, f64>,
    basis_order: usize,
    periodic: bool,
    period: Option<f64>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    init_lambda: Option<f64>,
    by: Option<PyReadonlyArray1<'py, f64>>,
    by_start_col: usize,
) -> PyResult<Py<PyDict>> {
    let result = gaussian_reml_fit_positions_batched_impl(
        t.as_array(),
        y.as_array(),
        row_offsets.as_array(),
        knots_or_centers.as_array(),
        &basis_kind,
        basis_order,
        periodic,
        period,
        penalty.as_array(),
        weights.as_ref().map(|w| w.as_array()),
        init_lambda,
        by.as_ref().map(|b| b.as_array()),
        by_start_col,
    )
    .map_err(py_value_error)?;
    let out = PyDict::new(py);
    out.set_item("status", result.statuses)?;
    out.set_item("lambda", result.lambdas.into_pyarray(py))?;
    out.set_item("rho", result.rhos.into_pyarray(py))?;
    out.set_item("reml_score", result.reml_scores.into_pyarray(py))?;
    out.set_item(
        "reml_grad_lambda",
        result.reml_grad_lambdas.into_pyarray(py),
    )?;
    out.set_item(
        "reml_hess_lambda",
        result.reml_hess_lambdas.into_pyarray(py),
    )?;
    out.set_item("reml_grad_rho", result.reml_grad_rhos.into_pyarray(py))?;
    out.set_item("reml_hess_rho", result.reml_hess_rhos.into_pyarray(py))?;
    out.set_item("edf", result.edf.into_pyarray(py))?;
    out.set_item("coefficients", result.coefficients.into_pyarray(py))?;
    out.set_item("fitted", result.fitted.into_pyarray(py))?;
    out.set_item("sigma2", result.sigma2.into_pyarray(py))?;
    Ok(out.unbind())
}

#[pyfunction(signature = (
    t,
    y,
    row_offsets,
    basis_kind,
    knots_or_centers,
    penalty,
    grad_lambda = None,
    grad_coefficients = None,
    grad_fitted = None,
    grad_reml_score = None,
    basis_order = 3,
    periodic = false,
    period = None,
    weights = None,
    init_lambda = None,
    by = None,
    by_start_col = 0
))]
fn gaussian_reml_fit_positions_batched_backward<'py>(
    py: Python<'py>,
    t: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray2<'py, f64>,
    row_offsets: PyReadonlyArray1<'py, usize>,
    basis_kind: String,
    knots_or_centers: PyReadonlyArray1<'py, f64>,
    penalty: PyReadonlyArray2<'py, f64>,
    grad_lambda: Option<PyReadonlyArray1<'py, f64>>,
    grad_coefficients: Option<PyReadonlyArray3<'py, f64>>,
    grad_fitted: Option<PyReadonlyArray2<'py, f64>>,
    grad_reml_score: Option<PyReadonlyArray1<'py, f64>>,
    basis_order: usize,
    periodic: bool,
    period: Option<f64>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    init_lambda: Option<f64>,
    by: Option<PyReadonlyArray1<'py, f64>>,
    by_start_col: usize,
) -> PyResult<Py<PyDict>> {
    let backward = gaussian_reml_fit_positions_batched_backward_impl(
        t.as_array(),
        y.as_array(),
        row_offsets.as_array(),
        knots_or_centers.as_array(),
        &basis_kind,
        basis_order,
        periodic,
        period,
        penalty.as_array(),
        weights.as_ref().map(|w| w.as_array()),
        init_lambda,
        grad_lambda.as_ref().map(|g| g.as_array()),
        grad_coefficients.as_ref().map(|g| g.as_array()),
        grad_fitted.as_ref().map(|g| g.as_array()),
        grad_reml_score.as_ref().map(|g| g.as_array()),
        by.as_ref().map(|b| b.as_array()),
        by_start_col,
    )
    .map_err(py_value_error)?;

    let out = PyDict::new(py);
    out.set_item("status", backward.statuses)?;
    out.set_item("grad_t", backward.grad_t.into_pyarray(py))?;
    out.set_item("grad_y", backward.grad_y.into_pyarray(py))?;
    out.set_item("grad_weights", backward.grad_weights.into_pyarray(py))?;
    if let Some(grad_by) = backward.grad_by {
        out.set_item("grad_by", grad_by.into_pyarray(py))?;
    } else {
        out.set_item("grad_by", py.None())?;
    }
    Ok(out.unbind())
}

fn gaussian_reml_result_to_pydict<'py>(
    py: Python<'py>,
    fit: gam::gaussian_reml::GaussianRemlMultiResult,
) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(py);
    set_ok_gaussian_reml_items(py, &out, fit)?;
    Ok(out.unbind())
}

fn set_ok_gaussian_reml_items<'py>(
    py: Python<'py>,
    out: &Bound<'py, PyDict>,
    fit: gam::gaussian_reml::GaussianRemlMultiResult,
) -> PyResult<()> {
    out.set_item("status", "ok")?;
    out.set_item("lambda", fit.lambda)?;
    out.set_item("rho", fit.rho)?;
    out.set_item("reml_score", fit.reml_score)?;
    out.set_item("reml_grad_lambda", fit.reml_grad_lambda)?;
    out.set_item("reml_hess_lambda", fit.reml_hess_lambda)?;
    out.set_item("reml_grad_rho", fit.reml_grad_rho)?;
    out.set_item("reml_hess_rho", fit.reml_hess_rho)?;
    out.set_item("edf", fit.edf)?;
    out.set_item("coefficients", fit.coefficients.into_pyarray(py))?;
    out.set_item("fitted", fit.fitted.into_pyarray(py))?;
    out.set_item("sigma2", fit.sigma2.into_pyarray(py))?;
    out.set_item(
        "cache_penalty_eigenvalues",
        fit.cache.penalty_eigenvalues.into_pyarray(py),
    )?;
    out.set_item(
        "cache_eigenvectors",
        fit.cache.eigenvectors.into_pyarray(py),
    )?;
    out.set_item(
        "cache_coefficient_basis",
        fit.cache.coefficient_basis.into_pyarray(py),
    )?;
    out.set_item("cache_xtwx_fingerprint", fit.cache.xtwx_fingerprint)?;
    out.set_item("cache_penalty_fingerprint", fit.cache.penalty_fingerprint)?;
    out.set_item("cache_logdet_xtwx", fit.cache.logdet_xtwx)?;
    out.set_item(
        "cache_logdet_penalty_positive",
        fit.cache.logdet_penalty_positive,
    )?;
    out.set_item("cache_penalty_rank", fit.cache.penalty_rank)?;
    out.set_item("cache_nullity", fit.cache.nullity)?;
    Ok(())
}

fn gaussian_reml_fit_state_from_pydict(
    state: &Bound<'_, PyDict>,
) -> Result<gam::gaussian_reml::GaussianRemlMultiResult, String> {
    fn get<'py>(state: &'py Bound<'py, PyDict>, key: &str) -> Result<Bound<'py, PyAny>, String> {
        state
            .get_item(key)
            .map_err(|err| err.to_string())?
            .ok_or_else(|| format!("forward_state is missing key {key:?}"))
    }

    let penalty_eigenvalues = get(state, "cache_penalty_eigenvalues")?
        .extract::<PyReadonlyArray1<'_, f64>>()
        .map_err(|err| err.to_string())?
        .as_array()
        .to_owned();
    let eigenvectors = get(state, "cache_eigenvectors")?
        .extract::<PyReadonlyArray2<'_, f64>>()
        .map_err(|err| err.to_string())?
        .as_array()
        .to_owned();
    let coefficient_basis = get(state, "cache_coefficient_basis")?
        .extract::<PyReadonlyArray2<'_, f64>>()
        .map_err(|err| err.to_string())?
        .as_array()
        .to_owned();
    let coefficients = get(state, "coefficients")?
        .extract::<PyReadonlyArray2<'_, f64>>()
        .map_err(|err| err.to_string())?
        .as_array()
        .to_owned();
    let fitted = get(state, "fitted")?
        .extract::<PyReadonlyArray2<'_, f64>>()
        .map_err(|err| err.to_string())?
        .as_array()
        .to_owned();
    let sigma2 = get(state, "sigma2")?
        .extract::<PyReadonlyArray1<'_, f64>>()
        .map_err(|err| err.to_string())?
        .as_array()
        .to_owned();

    Ok(gam::gaussian_reml::GaussianRemlMultiResult {
        lambda: get(state, "lambda")?
            .extract::<f64>()
            .map_err(|err| err.to_string())?,
        rho: get(state, "rho")?
            .extract::<f64>()
            .map_err(|err| err.to_string())?,
        coefficients,
        fitted,
        reml_score: get(state, "reml_score")?
            .extract::<f64>()
            .map_err(|err| err.to_string())?,
        reml_grad_lambda: get(state, "reml_grad_lambda")?
            .extract::<f64>()
            .map_err(|err| err.to_string())?,
        reml_hess_lambda: get(state, "reml_hess_lambda")?
            .extract::<f64>()
            .map_err(|err| err.to_string())?,
        reml_grad_rho: get(state, "reml_grad_rho")?
            .extract::<f64>()
            .map_err(|err| err.to_string())?,
        reml_hess_rho: get(state, "reml_hess_rho")?
            .extract::<f64>()
            .map_err(|err| err.to_string())?,
        edf: get(state, "edf")?
            .extract::<f64>()
            .map_err(|err| err.to_string())?,
        sigma2,
        cache: gam::gaussian_reml::GaussianRemlEigenCache {
            penalty_eigenvalues,
            eigenvectors,
            coefficient_basis,
            xtwx_fingerprint: get(state, "cache_xtwx_fingerprint")?
                .extract::<u64>()
                .map_err(|err| err.to_string())?,
            penalty_fingerprint: get(state, "cache_penalty_fingerprint")?
                .extract::<u64>()
                .map_err(|err| err.to_string())?,
            logdet_xtwx: get(state, "cache_logdet_xtwx")?
                .extract::<f64>()
                .map_err(|err| err.to_string())?,
            logdet_penalty_positive: get(state, "cache_logdet_penalty_positive")?
                .extract::<f64>()
                .map_err(|err| err.to_string())?,
            penalty_rank: get(state, "cache_penalty_rank")?
                .extract::<usize>()
                .map_err(|err| err.to_string())?,
            nullity: get(state, "cache_nullity")?
                .extract::<usize>()
                .map_err(|err| err.to_string())?,
        },
    })
}

fn set_degenerate_gaussian_reml_items<'py>(
    py: Python<'py>,
    out: &Bound<'py, PyDict>,
    n_rows: usize,
    n_outputs: usize,
    n_coefficients: usize,
) -> PyResult<()> {
    out.set_item("status", "degenerate")?;
    out.set_item("lambda", f64::NAN)?;
    out.set_item("rho", f64::NAN)?;
    out.set_item("reml_score", f64::NAN)?;
    out.set_item("reml_grad_lambda", f64::NAN)?;
    out.set_item("reml_hess_lambda", f64::NAN)?;
    out.set_item("reml_grad_rho", f64::NAN)?;
    out.set_item("reml_hess_rho", f64::NAN)?;
    out.set_item("edf", 0.0)?;
    out.set_item(
        "coefficients",
        Array2::<f64>::zeros((n_coefficients, n_outputs)).into_pyarray(py),
    )?;
    out.set_item(
        "fitted",
        Array2::<f64>::zeros((n_rows, n_outputs)).into_pyarray(py),
    )?;
    out.set_item(
        "sigma2",
        Array1::<f64>::from_elem(n_outputs, f64::NAN).into_pyarray(py),
    )?;
    Ok(())
}

struct BatchedGaussianRemlResult {
    statuses: Vec<String>,
    lambdas: Array1<f64>,
    rhos: Array1<f64>,
    reml_scores: Array1<f64>,
    reml_grad_lambdas: Array1<f64>,
    reml_hess_lambdas: Array1<f64>,
    reml_grad_rhos: Array1<f64>,
    reml_hess_rhos: Array1<f64>,
    edf: Array1<f64>,
    coefficients: Array3<f64>,
    fitted: Array2<f64>,
    sigma2: Array2<f64>,
}

struct BatchedGaussianRemlBackwardResult {
    statuses: Vec<String>,
    grad_x: Array2<f64>,
    grad_y: Array2<f64>,
    grad_weights: Array1<f64>,
}

struct PositionGaussianRemlBackwardResult {
    grad_t: Array1<f64>,
    grad_y: Array2<f64>,
    grad_weights: Array1<f64>,
    grad_by: Option<Array1<f64>>,
}

struct BatchedPositionGaussianRemlBackwardResult {
    statuses: Vec<String>,
    grad_t: Array1<f64>,
    grad_y: Array2<f64>,
    grad_weights: Array1<f64>,
    grad_by: Option<Array1<f64>>,
}

fn gaussian_reml_fit_formula_table_impl(
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    formula: String,
    y: ArrayView2<'_, f64>,
    config_json: Option<&str>,
) -> Result<gam::gaussian_reml::GaussianRemlMultiResult, String> {
    let dataset = dataset_with_inferred_schema(headers, rows)?;
    let mut fit_config = parse_fit_config(config_json)?;
    fit_config.family = Some("gaussian".to_string());
    fit_config.link = Some("identity".to_string());
    let materialized = materialize(&formula, &dataset, &fit_config)?;
    let standard = match materialized.request {
        FitRequest::Standard(request) => request,
        _ => {
            return Err(
                "closed-form Gaussian REML formula fitting requires a standard Gaussian formula"
                    .to_string(),
            );
        }
    };
    if standard.family != LikelihoodFamily::GaussianIdentity {
        return Err(
            "closed-form Gaussian REML formula fitting requires Gaussian identity".to_string(),
        );
    }
    if standard.wiggle.is_some() {
        return Err(
            "closed-form Gaussian REML formula fitting does not support link wiggle".to_string(),
        );
    }
    if standard.offset.iter().any(|value| value.abs() > 0.0) {
        return Err(
            "closed-form Gaussian REML formula fitting does not support offsets".to_string(),
        );
    }
    let design = gam::smooth::build_term_collection_design(standard.data, &standard.spec)
        .map_err(|err| format!("failed to build formula design matrix: {err}"))?;
    let x = design.design.to_dense();
    if y.nrows() != x.nrows() {
        return Err(format!(
            "closed-form Gaussian REML response row mismatch: formula design has {} rows but Y has {}",
            x.nrows(),
            y.nrows()
        ));
    }
    if design.penalties.len() != 1 {
        return Err(format!(
            "closed-form Gaussian REML formula fitting requires exactly one smoothing penalty; got {}",
            design.penalties.len()
        ));
    }
    let penalty = global_penalty_from_block(x.ncols(), &design.penalties[0])?;
    gam::gaussian_reml::gaussian_reml_multi_closed_form(
        x.view(),
        y,
        penalty.view(),
        Some(standard.weights.view()),
        None,
    )
    .map_err(|err| err.to_string())
}

fn global_penalty_from_block(
    p: usize,
    penalty: &gam::smooth::BlockwisePenalty,
) -> Result<Array2<f64>, String> {
    if penalty.col_range.end > p || penalty.col_range.len() != penalty.local.nrows() {
        return Err(format!(
            "formula penalty range {:?} is incompatible with design width {p}",
            penalty.col_range
        ));
    }
    let mut out = Array2::<f64>::zeros((p, p));
    out.slice_mut(s![penalty.col_range.clone(), penalty.col_range.clone()])
        .assign(&penalty.local);
    Ok(out)
}

fn gaussian_reml_fit_batched_impl(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    row_offsets: ArrayView1<'_, usize>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    init_lambda: Option<f64>,
) -> Result<BatchedGaussianRemlResult, String> {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    if row_offsets.len() < 2 {
        return Err("row_offsets must contain at least [0, n]".to_string());
    }
    if row_offsets[0] != 0 || row_offsets[row_offsets.len() - 1] != x.nrows() {
        return Err(format!(
            "row_offsets must start at 0 and end at X.nrows(); got start={}, end={}, n={}",
            row_offsets[0],
            row_offsets[row_offsets.len() - 1],
            x.nrows()
        ));
    }
    for idx in 0..row_offsets.len() - 1 {
        if row_offsets[idx] > row_offsets[idx + 1] {
            return Err("row_offsets must be non-decreasing".to_string());
        }
    }
    if y.nrows() != x.nrows() {
        return Err(format!(
            "batched Gaussian REML row mismatch: X has {} rows but Y has {}",
            x.nrows(),
            y.nrows()
        ));
    }
    if x.ncols() == 0 || y.ncols() == 0 {
        return Err("batched Gaussian REML requires non-empty X and Y columns".to_string());
    }
    if penalty.nrows() != x.ncols() || penalty.ncols() != x.ncols() {
        return Err(format!(
            "penalty shape mismatch: expected {}x{}, got {}x{}",
            x.ncols(),
            x.ncols(),
            penalty.nrows(),
            penalty.ncols()
        ));
    }
    if let Some(weights) = weights {
        if weights.len() != x.nrows() {
            return Err(format!(
                "weights length mismatch: expected {}, got {}",
                x.nrows(),
                weights.len()
            ));
        }
        if weights
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(
                "batched Gaussian REML weights must be finite non-negative values".to_string(),
            );
        }
    }
    if x.iter()
        .chain(y.iter())
        .chain(penalty.iter())
        .any(|value| !value.is_finite())
    {
        return Err("batched Gaussian REML inputs must be finite".to_string());
    }
    if let Some(lambda) = init_lambda {
        if !lambda.is_finite() || lambda <= 0.0 {
            return Err(format!(
                "init_lambda must be finite and positive when provided; got {lambda}"
            ));
        }
    }

    let batch = row_offsets.len() - 1;
    let p = x.ncols();
    let d = y.ncols();
    let fit_results: Vec<
        Result<(usize, Option<gam::gaussian_reml::GaussianRemlMultiResult>), String>,
    > = (0..batch)
        .into_par_iter()
        .map(|b| {
            let start = row_offsets[b];
            let end = row_offsets[b + 1];
            if start == end {
                return Ok((b, None));
            }
            let weight_slice = weights.as_ref().map(|w| w.slice(s![start..end]));
            match gaussian_reml_multi_closed_form_with_cache(
                x.slice(s![start..end, ..]),
                y.slice(s![start..end, ..]),
                penalty,
                weight_slice,
                init_lambda,
                None,
            ) {
                Ok(result) => Ok((b, Some(result))),
                Err(EstimationError::ModelIsIllConditioned { .. }) => Ok((b, None)),
                Err(err) => Err(format!("batched Gaussian REML fit {b} failed: {err}")),
            }
        })
        .collect();

    let mut lambdas = Array1::<f64>::from_elem(batch, f64::NAN);
    let mut rhos = Array1::<f64>::from_elem(batch, f64::NAN);
    let mut reml_scores = Array1::<f64>::from_elem(batch, f64::NAN);
    let mut reml_grad_lambdas = Array1::<f64>::from_elem(batch, f64::NAN);
    let mut reml_hess_lambdas = Array1::<f64>::from_elem(batch, f64::NAN);
    let mut reml_grad_rhos = Array1::<f64>::from_elem(batch, f64::NAN);
    let mut reml_hess_rhos = Array1::<f64>::from_elem(batch, f64::NAN);
    let mut edf = Array1::<f64>::zeros(batch);
    let mut coefficients = Array3::<f64>::zeros((batch, p, d));
    let mut fitted = Array2::<f64>::zeros((x.nrows(), d));
    let mut sigma2 = Array2::<f64>::from_elem((batch, d), f64::NAN);
    let mut statuses = vec!["degenerate".to_string(); batch];

    for result in fit_results {
        let (b, fit) = result?;
        if let Some(fit) = fit {
            let start = row_offsets[b];
            let end = row_offsets[b + 1];
            statuses[b] = "ok".to_string();
            lambdas[b] = fit.lambda;
            rhos[b] = fit.rho;
            reml_scores[b] = fit.reml_score;
            reml_grad_lambdas[b] = fit.reml_grad_lambda;
            reml_hess_lambdas[b] = fit.reml_hess_lambda;
            reml_grad_rhos[b] = fit.reml_grad_rho;
            reml_hess_rhos[b] = fit.reml_hess_rho;
            edf[b] = fit.edf;
            coefficients
                .slice_mut(s![b, .., ..])
                .assign(&fit.coefficients);
            fitted.slice_mut(s![start..end, ..]).assign(&fit.fitted);
            sigma2.slice_mut(s![b, ..]).assign(&fit.sigma2);
        }
    }

    Ok(BatchedGaussianRemlResult {
        statuses,
        lambdas,
        rhos,
        reml_scores,
        reml_grad_lambdas,
        reml_hess_lambdas,
        reml_grad_rhos,
        reml_hess_rhos,
        edf,
        coefficients,
        fitted,
        sigma2,
    })
}

#[allow(clippy::too_many_arguments)]
fn gaussian_reml_fit_batched_backward_impl(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    row_offsets: ArrayView1<'_, usize>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    init_lambda: Option<f64>,
    grad_lambda: Option<ArrayView1<'_, f64>>,
    grad_coefficients: Option<ArrayView3<'_, f64>>,
    grad_fitted: Option<ArrayView2<'_, f64>>,
    grad_reml_score: Option<ArrayView1<'_, f64>>,
) -> Result<BatchedGaussianRemlBackwardResult, String> {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    validate_batched_reml_common(x, y, row_offsets, penalty, weights, init_lambda)?;
    let batch = row_offsets.len() - 1;
    let p = x.ncols();
    let d = y.ncols();
    validate_batched_reml_upstreams(
        batch,
        p,
        d,
        y.nrows(),
        grad_lambda,
        grad_coefficients,
        grad_fitted,
        grad_reml_score,
    )?;

    let results: Vec<
        Result<
            (
                usize,
                Option<gam::gaussian_reml::GaussianRemlBackwardResult>,
            ),
            String,
        >,
    > = (0..batch)
        .into_par_iter()
        .map(|b| {
            let start = row_offsets[b];
            let end = row_offsets[b + 1];
            if start == end {
                return Ok((b, None));
            }
            let weight_slice = weights.as_ref().map(|w| w.slice(s![start..end]));
            let upstream_lambda = grad_lambda.as_ref().map_or(0.0, |g| g[b]);
            let upstream_reml_score = grad_reml_score.as_ref().map_or(0.0, |g| g[b]);
            let upstream_coefficients = grad_coefficients.as_ref().map(|g| g.slice(s![b, .., ..]));
            let upstream_fitted = grad_fitted.as_ref().map(|g| g.slice(s![start..end, ..]));
            match gaussian_reml_multi_closed_form_backward(
                x.slice(s![start..end, ..]),
                y.slice(s![start..end, ..]),
                penalty,
                weight_slice,
                init_lambda,
                upstream_lambda,
                upstream_coefficients,
                upstream_fitted,
                upstream_reml_score,
            ) {
                Ok(backward) => Ok((b, Some(backward))),
                Err(EstimationError::ModelIsIllConditioned { .. }) => Ok((b, None)),
                Err(err) => Err(format!("batched Gaussian REML backward {b} failed: {err}")),
            }
        })
        .collect();

    let mut statuses = vec!["degenerate".to_string(); batch];
    let mut grad_x = Array2::<f64>::zeros(x.dim());
    let mut grad_y = Array2::<f64>::zeros(y.dim());
    let mut grad_weights = Array1::<f64>::zeros(x.nrows());
    for result in results {
        let (b, backward) = result?;
        if let Some(backward) = backward {
            let start = row_offsets[b];
            let end = row_offsets[b + 1];
            statuses[b] = "ok".to_string();
            grad_x
                .slice_mut(s![start..end, ..])
                .assign(&backward.grad_x);
            grad_y
                .slice_mut(s![start..end, ..])
                .assign(&backward.grad_y);
            grad_weights
                .slice_mut(s![start..end])
                .assign(&backward.grad_weights);
        }
    }

    Ok(BatchedGaussianRemlBackwardResult {
        statuses,
        grad_x,
        grad_y,
        grad_weights,
    })
}

fn validate_batched_reml_common(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    row_offsets: ArrayView1<'_, usize>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    init_lambda: Option<f64>,
) -> Result<(), String> {
    if row_offsets.len() < 2 {
        return Err("row_offsets must contain at least [0, n]".to_string());
    }
    if row_offsets[0] != 0 || row_offsets[row_offsets.len() - 1] != x.nrows() {
        return Err(format!(
            "row_offsets must start at 0 and end at X.nrows(); got start={}, end={}, n={}",
            row_offsets[0],
            row_offsets[row_offsets.len() - 1],
            x.nrows()
        ));
    }
    for idx in 0..row_offsets.len() - 1 {
        if row_offsets[idx] > row_offsets[idx + 1] {
            return Err("row_offsets must be non-decreasing".to_string());
        }
    }
    if y.nrows() != x.nrows() {
        return Err(format!(
            "batched Gaussian REML row mismatch: X has {} rows but Y has {}",
            x.nrows(),
            y.nrows()
        ));
    }
    if x.ncols() == 0 || y.ncols() == 0 {
        return Err("batched Gaussian REML requires non-empty X and Y columns".to_string());
    }
    if penalty.nrows() != x.ncols() || penalty.ncols() != x.ncols() {
        return Err(format!(
            "penalty shape mismatch: expected {}x{}, got {}x{}",
            x.ncols(),
            x.ncols(),
            penalty.nrows(),
            penalty.ncols()
        ));
    }
    if let Some(weights) = weights {
        if weights.len() != x.nrows() {
            return Err(format!(
                "weights length mismatch: expected {}, got {}",
                x.nrows(),
                weights.len()
            ));
        }
        if weights
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(
                "batched Gaussian REML weights must be finite non-negative values".to_string(),
            );
        }
    }
    if x.iter()
        .chain(y.iter())
        .chain(penalty.iter())
        .any(|value| !value.is_finite())
    {
        return Err("batched Gaussian REML inputs must be finite".to_string());
    }
    if let Some(lambda) = init_lambda {
        if !lambda.is_finite() || lambda <= 0.0 {
            return Err(format!(
                "init_lambda must be finite and positive when provided; got {lambda}"
            ));
        }
    }
    Ok(())
}

fn validate_batched_reml_upstreams(
    batch: usize,
    p: usize,
    d: usize,
    n: usize,
    grad_lambda: Option<ArrayView1<'_, f64>>,
    grad_coefficients: Option<ArrayView3<'_, f64>>,
    grad_fitted: Option<ArrayView2<'_, f64>>,
    grad_reml_score: Option<ArrayView1<'_, f64>>,
) -> Result<(), String> {
    if let Some(grad_lambda) = grad_lambda {
        if grad_lambda.len() != batch {
            return Err(format!(
                "grad_lambda length mismatch: expected {batch}, got {}",
                grad_lambda.len()
            ));
        }
        if grad_lambda.iter().any(|value| !value.is_finite()) {
            return Err("grad_lambda must contain only finite values".to_string());
        }
    }
    if let Some(grad_reml_score) = grad_reml_score {
        if grad_reml_score.len() != batch {
            return Err(format!(
                "grad_reml_score length mismatch: expected {batch}, got {}",
                grad_reml_score.len()
            ));
        }
        if grad_reml_score.iter().any(|value| !value.is_finite()) {
            return Err("grad_reml_score must contain only finite values".to_string());
        }
    }
    if let Some(grad_coefficients) = grad_coefficients {
        if grad_coefficients.dim() != (batch, p, d) {
            let (got_b, got_p, got_d) = grad_coefficients.dim();
            return Err(format!(
                "grad_coefficients shape mismatch: expected {batch}x{p}x{d}, got {got_b}x{got_p}x{got_d}"
            ));
        }
        if grad_coefficients.iter().any(|value| !value.is_finite()) {
            return Err("grad_coefficients must contain only finite values".to_string());
        }
    }
    if let Some(grad_fitted) = grad_fitted {
        if grad_fitted.dim() != (n, d) {
            return Err(format!(
                "grad_fitted shape mismatch: expected {n}x{d}, got {}x{}",
                grad_fitted.nrows(),
                grad_fitted.ncols()
            ));
        }
        if grad_fitted.iter().any(|value| !value.is_finite()) {
            return Err("grad_fitted must contain only finite values".to_string());
        }
    }
    Ok(())
}

fn gaussian_reml_fit_positions_backward_impl(
    t: ArrayView1<'_, f64>,
    y: ArrayView2<'_, f64>,
    knots_or_centers: ArrayView1<'_, f64>,
    basis_kind: &str,
    basis_order: usize,
    periodic: bool,
    period: Option<f64>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    init_lambda: Option<f64>,
    grad_lambda: f64,
    grad_coefficients: Option<ArrayView2<'_, f64>>,
    grad_fitted: Option<ArrayView2<'_, f64>>,
    grad_reml_score: f64,
    by: Option<ArrayView1<'_, f64>>,
    by_start_col: usize,
) -> Result<PositionGaussianRemlBackwardResult, String> {
    let x = position_basis_design(
        t,
        knots_or_centers,
        basis_kind,
        basis_order,
        periodic,
        period,
    )?;
    let basis_derivative = position_basis_derivative(
        t,
        knots_or_centers,
        basis_kind,
        basis_order,
        periodic,
        period,
    )?;
    let gated_x;
    let fit_x = if let Some(by_values) = by {
        gated_x = apply_by_gate(x.view(), by_values, by_start_col)?;
        gated_x.view()
    } else {
        x.view()
    };
    let backward = gaussian_reml_multi_closed_form_backward(
        fit_x,
        y,
        penalty,
        weights,
        init_lambda,
        grad_lambda,
        grad_coefficients,
        grad_fitted,
        grad_reml_score,
    )
    .map_err(|err| err.to_string())?;
    let (grad_x, grad_by) = if let Some(by_values) = by {
        let (grad_x, grad_by) =
            apply_by_gate_backward(x.view(), by_values, by_start_col, backward.grad_x.view())?;
        (grad_x, Some(grad_by))
    } else {
        (backward.grad_x, None)
    };
    let grad_t = contract_position_gradient(grad_x.view(), basis_derivative.view())?;
    Ok(PositionGaussianRemlBackwardResult {
        grad_t,
        grad_y: backward.grad_y,
        grad_weights: backward.grad_weights,
        grad_by,
    })
}

fn gaussian_reml_fit_positions_batched_impl(
    t: ArrayView1<'_, f64>,
    y: ArrayView2<'_, f64>,
    row_offsets: ArrayView1<'_, usize>,
    knots_or_centers: ArrayView1<'_, f64>,
    basis_kind: &str,
    basis_order: usize,
    periodic: bool,
    period: Option<f64>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    init_lambda: Option<f64>,
    by: Option<ArrayView1<'_, f64>>,
    by_start_col: usize,
) -> Result<BatchedGaussianRemlResult, String> {
    let x = position_basis_design(
        t,
        knots_or_centers,
        basis_kind,
        basis_order,
        periodic,
        period,
    )?;
    let gated_x;
    let fit_x = if let Some(by_values) = by {
        gated_x = apply_by_gate(x.view(), by_values, by_start_col)?;
        gated_x.view()
    } else {
        x.view()
    };
    gaussian_reml_fit_batched_impl(fit_x, y, row_offsets, penalty, weights, init_lambda)
}

fn gaussian_reml_fit_positions_batched_backward_impl(
    t: ArrayView1<'_, f64>,
    y: ArrayView2<'_, f64>,
    row_offsets: ArrayView1<'_, usize>,
    knots_or_centers: ArrayView1<'_, f64>,
    basis_kind: &str,
    basis_order: usize,
    periodic: bool,
    period: Option<f64>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    init_lambda: Option<f64>,
    grad_lambda: Option<ArrayView1<'_, f64>>,
    grad_coefficients: Option<ArrayView3<'_, f64>>,
    grad_fitted: Option<ArrayView2<'_, f64>>,
    grad_reml_score: Option<ArrayView1<'_, f64>>,
    by: Option<ArrayView1<'_, f64>>,
    by_start_col: usize,
) -> Result<BatchedPositionGaussianRemlBackwardResult, String> {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    if row_offsets.len() < 2 {
        return Err("row_offsets must contain at least [0, n]".to_string());
    }
    if row_offsets[0] != 0 || row_offsets[row_offsets.len() - 1] != t.len() {
        return Err(format!(
            "row_offsets must start at 0 and end at t.len(); got start={}, end={}, n={}",
            row_offsets[0],
            row_offsets[row_offsets.len() - 1],
            t.len()
        ));
    }
    for idx in 0..row_offsets.len() - 1 {
        if row_offsets[idx] > row_offsets[idx + 1] {
            return Err("row_offsets must be non-decreasing".to_string());
        }
    }
    if y.nrows() != t.len() {
        return Err(format!(
            "position batched Gaussian REML row mismatch: t has {} rows but Y has {}",
            t.len(),
            y.nrows()
        ));
    }
    if let Some(weights) = weights {
        if weights.len() != t.len() {
            return Err(format!(
                "weights length mismatch: expected {}, got {}",
                t.len(),
                weights.len()
            ));
        }
    }

    let x = position_basis_design(
        t,
        knots_or_centers,
        basis_kind,
        basis_order,
        periodic,
        period,
    )?;
    let basis_derivative = position_basis_derivative(
        t,
        knots_or_centers,
        basis_kind,
        basis_order,
        periodic,
        period,
    )?;
    let gated_x;
    let fit_x = if let Some(by_values) = by {
        gated_x = apply_by_gate(x.view(), by_values, by_start_col)?;
        gated_x.view()
    } else {
        x.view()
    };
    if penalty.dim() != (x.ncols(), x.ncols()) {
        return Err(format!(
            "penalty shape mismatch: expected {}x{}, got {}x{}",
            x.ncols(),
            x.ncols(),
            penalty.nrows(),
            penalty.ncols()
        ));
    }

    let batch = row_offsets.len() - 1;
    let p = x.ncols();
    let d = y.ncols();
    if let Some(grad_lambda) = grad_lambda {
        if grad_lambda.len() != batch {
            return Err(format!(
                "grad_lambda length mismatch: expected {batch}, got {}",
                grad_lambda.len()
            ));
        }
        if grad_lambda.iter().any(|value| !value.is_finite()) {
            return Err("grad_lambda must contain only finite values".to_string());
        }
    }
    if let Some(grad_reml_score) = grad_reml_score {
        if grad_reml_score.len() != batch {
            return Err(format!(
                "grad_reml_score length mismatch: expected {batch}, got {}",
                grad_reml_score.len()
            ));
        }
        if grad_reml_score.iter().any(|value| !value.is_finite()) {
            return Err("grad_reml_score must contain only finite values".to_string());
        }
    }
    if let Some(grad_coefficients) = grad_coefficients {
        if grad_coefficients.dim() != (batch, p, d) {
            let (got_b, got_p, got_d) = grad_coefficients.dim();
            return Err(format!(
                "grad_coefficients shape mismatch: expected {batch}x{p}x{d}, got {got_b}x{got_p}x{got_d}"
            ));
        }
        if grad_coefficients.iter().any(|value| !value.is_finite()) {
            return Err("grad_coefficients must contain only finite values".to_string());
        }
    }
    if let Some(grad_fitted) = grad_fitted {
        if grad_fitted.dim() != y.dim() {
            return Err(format!(
                "grad_fitted shape mismatch: expected {}x{}, got {}x{}",
                y.nrows(),
                y.ncols(),
                grad_fitted.nrows(),
                grad_fitted.ncols()
            ));
        }
        if grad_fitted.iter().any(|value| !value.is_finite()) {
            return Err("grad_fitted must contain only finite values".to_string());
        }
    }

    let results: Vec<Result<(usize, Option<(Array2<f64>, Array2<f64>, Array1<f64>)>), String>> = (0
        ..batch)
        .into_par_iter()
        .map(|b| {
            let start = row_offsets[b];
            let end = row_offsets[b + 1];
            if start == end {
                return Ok((b, None));
            }
            let weight_slice = weights.as_ref().map(|w| w.slice(s![start..end]));
            let upstream_lambda = grad_lambda.as_ref().map_or(0.0, |g| g[b]);
            let upstream_reml_score = grad_reml_score.as_ref().map_or(0.0, |g| g[b]);
            let upstream_coefficients = grad_coefficients.as_ref().map(|g| g.slice(s![b, .., ..]));
            let upstream_fitted = grad_fitted.as_ref().map(|g| g.slice(s![start..end, ..]));
            match gaussian_reml_multi_closed_form_backward(
                fit_x.slice(s![start..end, ..]),
                y.slice(s![start..end, ..]),
                penalty,
                weight_slice,
                init_lambda,
                upstream_lambda,
                upstream_coefficients,
                upstream_fitted,
                upstream_reml_score,
            ) {
                Ok(backward) => Ok((
                    b,
                    Some((backward.grad_x, backward.grad_y, backward.grad_weights)),
                )),
                Err(EstimationError::ModelIsIllConditioned { .. }) => Ok((b, None)),
                Err(err) => Err(format!(
                    "batched position Gaussian REML backward {b} failed: {err}"
                )),
            }
        })
        .collect();

    let mut statuses = vec!["degenerate".to_string(); batch];
    let mut grad_fit_x = Array2::<f64>::zeros(x.dim());
    let mut grad_y = Array2::<f64>::zeros(y.dim());
    let mut grad_weights = Array1::<f64>::zeros(t.len());
    for result in results {
        let (b, backward) = result?;
        if let Some((batch_grad_x, batch_grad_y, batch_grad_weights)) = backward {
            let start = row_offsets[b];
            let end = row_offsets[b + 1];
            statuses[b] = "ok".to_string();
            grad_fit_x
                .slice_mut(s![start..end, ..])
                .assign(&batch_grad_x);
            grad_y.slice_mut(s![start..end, ..]).assign(&batch_grad_y);
            grad_weights
                .slice_mut(s![start..end])
                .assign(&batch_grad_weights);
        }
    }
    let (grad_x, grad_by) = if let Some(by_values) = by {
        let (grad_x, grad_by) =
            apply_by_gate_backward(x.view(), by_values, by_start_col, grad_fit_x.view())?;
        (grad_x, Some(grad_by))
    } else {
        (grad_fit_x, None)
    };
    let grad_t = contract_position_gradient(grad_x.view(), basis_derivative.view())?;

    Ok(BatchedPositionGaussianRemlBackwardResult {
        statuses,
        grad_t,
        grad_y,
        grad_weights,
        grad_by,
    })
}

fn position_basis_design(
    t: ArrayView1<'_, f64>,
    knots_or_centers: ArrayView1<'_, f64>,
    basis_kind: &str,
    basis_order: usize,
    periodic: bool,
    period: Option<f64>,
) -> Result<Array2<f64>, String> {
    match normalized_position_basis_kind(basis_kind)?.as_str() {
        "bspline" => {
            bspline_position_basis_impl(t, knots_or_centers, basis_order, periodic, period)
        }
        "duchon" => {
            validate_position_period("duchon", knots_or_centers, periodic, period)?;
            duchon_basis_1d_impl(t, knots_or_centers, basis_order, periodic)
        }
        _ => unreachable!("normalized_position_basis_kind only returns supported basis names"),
    }
}

fn position_basis_derivative(
    t: ArrayView1<'_, f64>,
    knots_or_centers: ArrayView1<'_, f64>,
    basis_kind: &str,
    basis_order: usize,
    periodic: bool,
    period: Option<f64>,
) -> Result<Array2<f64>, String> {
    match normalized_position_basis_kind(basis_kind)?.as_str() {
        "bspline" => {
            bspline_position_derivative_impl(t, knots_or_centers, basis_order, 1, periodic, period)
        }
        "duchon" => {
            validate_position_period("duchon", knots_or_centers, periodic, period)?;
            duchon_basis_1d_derivative_impl(t, knots_or_centers, basis_order, 1, periodic)
        }
        _ => unreachable!("normalized_position_basis_kind only returns supported basis names"),
    }
}

fn bspline_position_basis_impl(
    t: ArrayView1<'_, f64>,
    knots: ArrayView1<'_, f64>,
    degree: usize,
    periodic: bool,
    period: Option<f64>,
) -> Result<Array2<f64>, String> {
    if !periodic {
        validate_position_period("B-spline", knots, periodic, period)?;
        return bspline_basis_impl(t, knots, degree, false);
    }
    let (left, right, num_basis) = periodic_position_domain(knots, period)?;
    validate_vector("t", t)?;
    create_periodic_bspline_basis_dense(t, (left, right), degree, num_basis)
        .map_err(|err| format!("failed to evaluate periodic B-spline basis: {err}"))
}

fn bspline_position_derivative_impl(
    t: ArrayView1<'_, f64>,
    knots: ArrayView1<'_, f64>,
    degree: usize,
    order: usize,
    periodic: bool,
    period: Option<f64>,
) -> Result<Array2<f64>, String> {
    if !periodic {
        validate_position_period("B-spline", knots, periodic, period)?;
        return bspline_basis_derivative_impl(t, knots, degree, order, false);
    }
    if order != 1 {
        return Err(format!(
            "periodic B-spline derivative supports order=1; got order={order}"
        ));
    }
    let (left, right, num_basis) = periodic_position_domain(knots, period)?;
    validate_vector("t", t)?;
    create_periodic_bspline_derivative_dense(t, (left, right), degree, num_basis)
        .map_err(|err| format!("failed to evaluate periodic B-spline derivative: {err}"))
}

fn periodic_position_domain(
    knots_or_centers: ArrayView1<'_, f64>,
    period: Option<f64>,
) -> Result<(f64, f64, usize), String> {
    validate_vector("knots_or_centers", knots_or_centers)?;
    if knots_or_centers.len() < 2 {
        return Err("periodic position basis requires at least two knots or centers".to_string());
    }
    let Some(period) = period else {
        return Err(
            "periodic position basis requires an explicit finite positive period".to_string(),
        );
    };
    if !period.is_finite() || period <= 0.0 {
        return Err(format!(
            "periodic position basis period must be finite and positive; got {period}"
        ));
    }
    let left = knots_or_centers[0];
    let right = left + period;
    Ok((left, right, knots_or_centers.len() - 1))
}

fn validate_position_period(
    label: &str,
    knots_or_centers: ArrayView1<'_, f64>,
    periodic: bool,
    period: Option<f64>,
) -> Result<(), String> {
    if periodic {
        let Some(period) = period else {
            return Err(format!(
                "{label} periodic position basis requires an explicit period"
            ));
        };
        if !period.is_finite() || period <= 0.0 {
            return Err(format!(
                "{label} period must be finite and positive; got {period}"
            ));
        }
        let left = knots_or_centers
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));
        let right = knots_or_centers
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        if !left.is_finite() || !right.is_finite() || left >= right {
            return Err(format!(
                "{label} periodic support must have increasing finite endpoints"
            ));
        }
        let implied = right - left;
        if (implied - period).abs() > 1.0e-10 * period.max(1.0) {
            return Err(format!(
                "{label} periodic support range ({implied}) must match explicit period ({period})"
            ));
        }
    } else if period.is_some() {
        return Err(format!("{label} period is only valid when periodic=true"));
    }
    Ok(())
}

fn normalized_position_basis_kind(basis_kind: &str) -> Result<String, String> {
    let normalized = basis_kind
        .trim()
        .to_ascii_lowercase()
        .replace(['_', '-'], "");
    match normalized.as_str() {
        "bspline" | "spline" => Ok("bspline".to_string()),
        "duchon" | "duchonspline" => Ok("duchon".to_string()),
        _ => Err(format!(
            "basis_kind must be 'bspline' or 'duchon'; got {basis_kind:?}"
        )),
    }
}

fn contract_position_gradient(
    grad_x: ArrayView2<'_, f64>,
    basis_derivative: ArrayView2<'_, f64>,
) -> Result<Array1<f64>, String> {
    if grad_x.dim() != basis_derivative.dim() {
        return Err(format!(
            "basis derivative shape mismatch: grad_x is {}x{} but dX/dt is {}x{}",
            grad_x.nrows(),
            grad_x.ncols(),
            basis_derivative.nrows(),
            basis_derivative.ncols()
        ));
    }
    let mut grad_t = Array1::<f64>::zeros(grad_x.nrows());
    for row in 0..grad_x.nrows() {
        grad_t[row] = grad_x.row(row).dot(&basis_derivative.row(row));
    }
    Ok(grad_t)
}

fn apply_by_gate(
    x: ArrayView2<'_, f64>,
    by: ArrayView1<'_, f64>,
    start_col: usize,
) -> Result<Array2<f64>, String> {
    if by.len() != x.nrows() {
        return Err(format!(
            "by gate length mismatch: expected {}, got {}",
            x.nrows(),
            by.len()
        ));
    }
    if start_col > x.ncols() {
        return Err(format!(
            "by_start_col must be <= number of columns; got {start_col} for {} columns",
            x.ncols()
        ));
    }
    if by.iter().any(|value| !value.is_finite()) {
        return Err("by gate must contain only finite values".to_string());
    }
    let mut out = x.to_owned();
    for row in 0..out.nrows() {
        let gate = by[row];
        for col in start_col..out.ncols() {
            out[[row, col]] *= gate;
        }
    }
    Ok(out)
}

fn apply_by_gate_backward(
    x: ArrayView2<'_, f64>,
    by: ArrayView1<'_, f64>,
    start_col: usize,
    grad_gated_x: ArrayView2<'_, f64>,
) -> Result<(Array2<f64>, Array1<f64>), String> {
    if grad_gated_x.dim() != x.dim() {
        return Err(format!(
            "by gate backward gradient shape mismatch: expected {}x{}, got {}x{}",
            x.nrows(),
            x.ncols(),
            grad_gated_x.nrows(),
            grad_gated_x.ncols()
        ));
    }
    if by.len() != x.nrows() {
        return Err(format!(
            "by gate length mismatch: expected {}, got {}",
            x.nrows(),
            by.len()
        ));
    }
    if start_col > x.ncols() {
        return Err(format!(
            "by_start_col must be <= number of columns; got {start_col} for {} columns",
            x.ncols()
        ));
    }
    let mut grad_x = grad_gated_x.to_owned();
    let mut grad_by = Array1::<f64>::zeros(x.nrows());
    for row in 0..x.nrows() {
        let gate = by[row];
        for col in start_col..x.ncols() {
            grad_x[[row, col]] = grad_gated_x[[row, col]] * gate;
            grad_by[row] += grad_gated_x[[row, col]] * x[[row, col]];
        }
    }
    Ok((grad_x, grad_by))
}

#[pyfunction]
fn posterior_predict_table(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    samples_flat: Vec<f64>,
    n_draws: usize,
    n_coeffs: usize,
) -> PyResult<String> {
    py.detach(move || {
        posterior_predict_table_impl(&model_bytes, headers, rows, samples_flat, n_draws, n_coeffs)
    })
    .map_err(py_value_error)
}

#[pyfunction]
fn summary_json(py: Python<'_>, model_bytes: Vec<u8>) -> PyResult<String> {
    py.detach(move || summary_json_impl(&model_bytes))
        .map_err(py_value_error)
}

#[pyfunction]
fn check_json(
    py: Python<'_>,
    model_bytes: Vec<u8>,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
) -> PyResult<String> {
    py.detach(move || check_json_impl(&model_bytes, headers, rows))
        .map_err(py_value_error)
}

#[pyfunction]
fn report_html(py: Python<'_>, model_bytes: Vec<u8>) -> PyResult<String> {
    py.detach(move || report_html_impl(&model_bytes))
        .map_err(py_value_error)
}

#[pymodule(name = "_rust", gil_used = false)]
fn rust_extension(module: &Bound<'_, PyModule>) -> PyResult<()> {
    gam::init_parallelism();
    // Install the same stderr logger used by the CLI so long-running Rust
    // solver phases (including survival marginal-slope joint-Newton cycles)
    // are visible from Python without requiring a separate shell.
    gam::visualizer::init_logging();
    module.add("__doc__", "PyO3 boundary for the gam Rust engine.")?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_function(wrap_pyfunction!(build_info, module)?)?;
    module.add_function(wrap_pyfunction!(fit_table, module)?)?;
    module.add_function(wrap_pyfunction!(fit_array, module)?)?;
    module.add_function(wrap_pyfunction!(load_model, module)?)?;
    module.add_function(wrap_pyfunction!(validate_formula_json, module)?)?;
    module.add_function(wrap_pyfunction!(predict_table, module)?)?;
    module.add_function(wrap_pyfunction!(predict_array, module)?)?;
    module.add_function(wrap_pyfunction!(sample_table, module)?)?;
    module.add_function(wrap_pyfunction!(design_matrix_table, module)?)?;
    module.add_function(wrap_pyfunction!(design_matrix_array, module)?)?;
    module.add_function(wrap_pyfunction!(bspline_basis, module)?)?;
    module.add_function(wrap_pyfunction!(bspline_basis_derivative, module)?)?;
    module.add_function(wrap_pyfunction!(duchon_basis_1d, module)?)?;
    module.add_function(wrap_pyfunction!(duchon_basis_1d_derivative, module)?)?;
    module.add_function(wrap_pyfunction!(smoothness_penalty, module)?)?;
    module.add_function(wrap_pyfunction!(gaussian_weighted_ridge_array, module)?)?;
    module.add_function(wrap_pyfunction!(gaussian_weighted_ridge_batch, module)?)?;
    module.add_function(wrap_pyfunction!(gaussian_reml_fit, module)?)?;
    module.add_function(wrap_pyfunction!(gaussian_reml_fit_backward, module)?)?;
    module.add_function(wrap_pyfunction!(gaussian_reml_fit_formula_table, module)?)?;
    module.add_function(wrap_pyfunction!(gaussian_reml_fit_batched, module)?)?;
    module.add_function(wrap_pyfunction!(
        gaussian_reml_fit_batched_backward,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(gaussian_reml_fit_positions, module)?)?;
    module.add_function(wrap_pyfunction!(
        gaussian_reml_fit_positions_backward,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        gaussian_reml_fit_positions_batched,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        gaussian_reml_fit_positions_batched_backward,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(posterior_predict_table, module)?)?;
    module.add_function(wrap_pyfunction!(summary_json, module)?)?;
    module.add_function(wrap_pyfunction!(check_json, module)?)?;
    module.add_function(wrap_pyfunction!(report_html, module)?)?;
    Ok(())
}

fn py_value_error(message: String) -> PyErr {
    // Final defensive translation at the Python boundary: convert the
    // cryptic Rust assertion "SurvivalLocationScaleFamily expects N blocks,
    // got 0" (and its blockwise_fit_from_parts cousin) into a clear
    // user-actionable message before it surfaces as a Python ValueError.
    //
    // The Rust call stack that produces this string has many entry points
    // (validate_joint_states from build_dynamic_geometry callers,
    // blockwise_fit_from_parts when its inner refit produced empty states,
    // etc.). We catch ALL of them here at the Python boundary so the
    // gamfit caller never sees the implementation detail.
    let user_facing = if message.contains("expects 3 blocks, got 0")
        || message.contains("expects 4 blocks, got 0")
        || (message.contains("block_states") && message.contains("got 0"))
        || message.contains("blockwise fit requires at least one block state")
    {
        format!(
            "gam fit failed at a degenerate inner-solver iterate. The most likely \
             cause is an under-identified smooth (one of your covariates has no \
             signal in the noise/scale dimension at this sample size) being \
             driven to a numerically pathological smoothing parameter (λ > 10⁸). \
             Try one of: (1) reduce the covariate count in your formula or \
             noise_formula, (2) increase the training-set size, \
             (3) use `baseline_target=\"linear\"` to drop the parametric baseline, \
             or (4) use `noise_formula=\"1\"` to make the noise model an intercept. \
             Underlying error: {message}"
        )
    } else {
        message
    };
    PyValueError::new_err(user_facing)
}

fn fit_table_impl(
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    formula: String,
    config_json: Option<&str>,
) -> Result<Vec<u8>, String> {
    let dataset = dataset_with_inferred_schema(headers, rows)?;
    fit_dataset_impl(dataset, formula, config_json)
}

fn fit_array_impl(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    formula: String,
    config_json: Option<&str>,
) -> Result<Vec<u8>, String> {
    let dataset = dataset_from_xy_arrays(x, y, &formula)?;
    fit_dataset_impl(dataset, formula, config_json)
}

fn fit_dataset_impl(
    dataset: EncodedDataset,
    formula: String,
    config_json: Option<&str>,
) -> Result<Vec<u8>, String> {
    let fit_config = parse_fit_config(config_json)?;
    let materialized = materialize(&formula, &dataset, &fit_config)?;
    let request = materialized.request;

    let payload = match request {
        FitRequest::Standard(standard_request) => {
            let family = standard_request.family;
            let fit_result = fit_model(FitRequest::Standard(standard_request))?;
            let standard_result = match fit_result {
                FitResult::Standard(standard_result) => standard_result,
                _ => {
                    return Err(
                        "python binding expected the standard workflow to return a standard fit result"
                            .to_string(),
                    );
                }
            };
            let saved_fit = standard_result.fit.clone();
            build_standard_payload(
                formula,
                &dataset,
                &fit_config,
                family,
                &saved_fit,
                standard_result.resolvedspec,
                standard_result.adaptive_diagnostics,
                standard_result.wiggle_knots.map(|knots| knots.to_vec()),
                standard_result.wiggle_degree,
            )
        }
        FitRequest::TransformationNormal(tn_request) => {
            let fit_result = fit_model(FitRequest::TransformationNormal(tn_request))?;
            let tn_result = match fit_result {
                FitResult::TransformationNormal(result) => result,
                _ => {
                    return Err(
                        "python binding expected the transformation-normal workflow to return a transformation-normal fit result"
                            .to_string(),
                    );
                }
            };
            build_transformation_normal_ffi_payload(formula, &dataset, &fit_config, tn_result)?
        }
        FitRequest::BernoulliMarginalSlope(ms_request) => {
            let base_link = ms_request.spec.base_link.clone();
            let frailty = ms_request.spec.frailty.clone();
            let fit_result = fit_model(FitRequest::BernoulliMarginalSlope(ms_request))?;
            let ms_result = match fit_result {
                FitResult::BernoulliMarginalSlope(result) => result,
                _ => {
                    return Err(
                        "python binding expected the bernoulli marginal-slope workflow to return a marginal-slope fit result"
                            .to_string(),
                    );
                }
            };
            build_bernoulli_marginal_slope_ffi_payload(
                formula,
                &dataset,
                &fit_config,
                base_link,
                frailty,
                ms_result,
            )?
        }
        FitRequest::SurvivalMarginalSlope(ms_request) => {
            let frailty = ms_request.spec.frailty.clone();
            let fit_result = fit_model(FitRequest::SurvivalMarginalSlope(ms_request))?;
            let ms_result = match fit_result {
                FitResult::SurvivalMarginalSlope(result) => result,
                _ => {
                    return Err(
                        "python binding expected the survival marginal-slope workflow to return a survival marginal-slope fit result"
                            .to_string(),
                    );
                }
            };
            build_survival_marginal_slope_ffi_payload(
                formula,
                &dataset,
                &fit_config,
                frailty,
                ms_result,
            )?
        }
        FitRequest::GaussianLocationScale(ls_request) => {
            let fit_result = fit_model(FitRequest::GaussianLocationScale(ls_request))?;
            let ls_result = match fit_result {
                FitResult::GaussianLocationScale(result) => result,
                _ => {
                    return Err(
                        "python binding expected the gaussian location-scale workflow to return a gaussian location-scale fit result"
                            .to_string(),
                    );
                }
            };
            build_gaussian_location_scale_ffi_payload(
                formula,
                &dataset,
                &fit_config,
                ls_result,
                1.0,
            )?
        }
        FitRequest::BinomialLocationScale(ls_request) => {
            let weights = ls_request.spec.weights.clone();
            let link_kind = ls_request.spec.link_kind.clone();
            let fit_result = fit_model(FitRequest::BinomialLocationScale(ls_request))?;
            let ls_result = match fit_result {
                FitResult::BinomialLocationScale(result) => result,
                _ => {
                    return Err(
                        "python binding expected the binomial location-scale workflow to return a binomial location-scale fit result"
                            .to_string(),
                    );
                }
            };
            build_binomial_location_scale_ffi_payload(
                formula,
                &dataset,
                &fit_config,
                link_kind,
                &weights,
                ls_result,
            )?
        }
        FitRequest::SurvivalLocationScale(ls_request) => {
            let weights = ls_request.spec.weights.clone();
            let fit_result = fit_model(FitRequest::SurvivalLocationScale(ls_request))?;
            let ls_result = match fit_result {
                FitResult::SurvivalLocationScale(result) => result,
                _ => {
                    return Err(
                        "python binding expected the survival location-scale workflow to return a survival location-scale fit result"
                            .to_string(),
                    );
                }
            };
            build_survival_location_scale_ffi_payload(
                formula,
                &dataset,
                &fit_config,
                &weights,
                ls_result,
            )?
        }
        FitRequest::SurvivalTransformation(rp_request) => {
            let fit_result = fit_model(FitRequest::SurvivalTransformation(rp_request))?;
            let rp_result = match fit_result {
                FitResult::SurvivalTransformation(result) => result,
                _ => {
                    return Err(
                        "python binding expected the survival transformation workflow to return a survival transformation fit result"
                            .to_string(),
                    );
                }
            };
            build_survival_transformation_ffi_payload(formula, &dataset, &fit_config, rp_result)?
        }
        FitRequest::LatentSurvival(lat_request) => {
            let frailty = lat_request.frailty.clone();
            let fit_result = fit_model(FitRequest::LatentSurvival(lat_request))?;
            let lat_result = match fit_result {
                FitResult::LatentSurvival(result) => result,
                _ => {
                    return Err(
                        "python binding expected the latent survival workflow to return a latent survival fit result"
                            .to_string(),
                    );
                }
            };
            build_latent_survival_ffi_payload(formula, &dataset, &fit_config, frailty, lat_result)?
        }
        FitRequest::LatentBinary(lat_request) => {
            let frailty = lat_request.frailty.clone();
            let fit_result = fit_model(FitRequest::LatentBinary(lat_request))?;
            let lat_result = match fit_result {
                FitResult::LatentBinary(result) => result,
                _ => {
                    return Err(
                        "python binding expected the latent binary workflow to return a latent binary fit result"
                            .to_string(),
                    );
                }
            };
            build_latent_binary_ffi_payload(formula, &dataset, &fit_config, frailty, lat_result)?
        }
    };
    let model = FittedModel::from_payload(payload);
    serde_json::to_vec_pretty(&model).map_err(|err| format!("failed to serialize model: {err}"))
}

fn load_model_impl(model_bytes: &[u8]) -> Result<FittedModel, String> {
    let model: FittedModel = serde_json::from_slice(model_bytes)
        .map_err(|err| format!("failed to parse model json: {err}"))?;
    model.validate_for_persistence()?;
    model.validate_numeric_finiteness()?;
    Ok(model)
}

fn validate_formula_json_impl(
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    formula: String,
    config_json: Option<&str>,
) -> Result<String, String> {
    let dataset = dataset_with_inferred_schema(headers, rows)?;
    let fit_config = parse_fit_config(config_json)?;
    let materialized = materialize(&formula, &dataset, &fit_config)?;
    let (family_name, model_class, supported_by_python) = request_metadata(&materialized.request);
    let response_column = response_column_name(&formula);
    let payload = ValidationPayload {
        formula,
        family_name: family_name.to_string(),
        model_class: model_class.to_string(),
        response_column,
        columns: dataset.headers.clone(),
        n_rows: dataset.values.nrows(),
        n_columns: dataset.values.ncols(),
        supported_by_python,
    };
    serde_json::to_string(&payload)
        .map_err(|err| format!("failed to serialize validation payload: {err}"))
}

fn predict_table_impl(
    model_bytes: &[u8],
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    options_json: Option<&str>,
) -> Result<String, String> {
    let model = load_model_impl(model_bytes)?;
    let model_class = model.predict_model_class();
    let dataset = dataset_with_model_schema(&model, headers, rows)?;
    predict_dataset_impl(&model, model_class, dataset, options_json)
}

fn predict_array_impl(
    model_bytes: &[u8],
    x: ArrayView2<'_, f64>,
    options_json: Option<&str>,
) -> Result<Array2<f64>, String> {
    let model = load_model_impl(model_bytes)?;
    let model_class = model.predict_model_class();
    let dataset = dataset_from_x_array_with_model_schema(&model, x)?;
    if matches!(model_class, PredictModelClass::Survival) {
        return Err("predict_array does not support survival prediction payloads".to_string());
    }
    let options = parse_predict_options(options_json)?;
    let columns = predict_columns(&model, dataset, &options)?;
    columns_to_array(columns)
}

fn predict_dataset_impl(
    model: &FittedModel,
    model_class: PredictModelClass,
    dataset: EncodedDataset,
    options_json: Option<&str>,
) -> Result<String, String> {
    let options = parse_predict_options(options_json)?;
    if matches!(model_class, PredictModelClass::Survival) {
        return predict_table_survival(model, &dataset, &options);
    }
    let columns = predict_columns(model, dataset, &options)?;
    serde_json::to_string(&PredictionPayload { columns })
        .map_err(|err| format!("failed to serialize prediction payload: {err}"))
}

fn predict_columns(
    model: &FittedModel,
    dataset: EncodedDataset,
    options: &PyPredictOptions,
) -> Result<BTreeMap<String, Vec<f64>>, String> {
    let col_map = dataset.column_map();
    let offset = resolve_offset_column(&dataset, &col_map, model.offset_column.as_deref())?;
    let offset_noise =
        resolve_offset_column(&dataset, &col_map, model.noise_offset_column.as_deref())?;
    let predict_input = build_predict_input_for_model(
        &model,
        dataset.values.view(),
        &col_map,
        model.training_headers.as_ref(),
        &offset,
        &offset_noise,
        false,
    )?;
    let predictor = model
        .predictor()
        .ok_or_else(|| "saved model could not construct a predictor".to_string())?;
    let fit = fit_result_from_saved_model_for_prediction(model)?;

    let mut columns = BTreeMap::<String, Vec<f64>>::new();
    if let Some(interval) = options.interval {
        let uncertainty_options = gam::predict::PredictUncertaintyOptions {
            confidence_level: interval,
            covariance_mode:
                gam::predict::InferenceCovarianceMode::ConditionalPlusSmoothingPreferred,
            mean_interval_method: gam::predict::MeanIntervalMethod::TransformEta,
            includeobservation_interval: false,
            apply_bias_correction: true,
            ..gam::predict::PredictUncertaintyOptions::default()
        };
        let prediction = predictor
            .predict_full_uncertainty(&predict_input, &fit, &uncertainty_options)
            .map_err(|err| format!("prediction with uncertainty failed: {err}"))?;
        columns.insert("eta".to_string(), prediction.eta.to_vec());
        columns.insert("mean".to_string(), prediction.mean.to_vec());
        columns.insert(
            "effective_se".to_string(),
            prediction.eta_standard_error.to_vec(),
        );
        columns.insert("mean_lower".to_string(), prediction.mean_lower.to_vec());
        columns.insert("mean_upper".to_string(), prediction.mean_upper.to_vec());
    } else {
        let prediction = predictor
            .predict_plugin_response(&predict_input)
            .map_err(|err| format!("prediction failed: {err}"))?;
        columns.insert("eta".to_string(), prediction.eta.to_vec());
        columns.insert("mean".to_string(), prediction.mean.to_vec());
    }

    Ok(columns)
}

fn columns_to_array(columns: BTreeMap<String, Vec<f64>>) -> Result<Array2<f64>, String> {
    let ordered = ordered_prediction_column_values(&columns);
    let n_cols = ordered.len();
    let n_rows = ordered.first().map(|values| values.len()).unwrap_or(0);
    let mut out = Array2::<f64>::zeros((n_rows, n_cols));
    for (j, values) in ordered.into_iter().enumerate() {
        if values.len() != n_rows {
            return Err("prediction columns have inconsistent lengths".to_string());
        }
        for (i, value) in values.into_iter().enumerate() {
            out[[i, j]] = value;
        }
    }
    Ok(out)
}

fn ordered_prediction_column_values(columns: &BTreeMap<String, Vec<f64>>) -> Vec<Vec<f64>> {
    let preferred = ["eta", "mean", "effective_se", "mean_lower", "mean_upper"];
    let mut out = Vec::<Vec<f64>>::new();
    let mut seen = BTreeSet::<&str>::new();
    for key in preferred {
        if let Some(values) = columns.get(key) {
            out.push(values.clone());
            seen.insert(key);
        }
    }
    for (key, values) in columns {
        if !seen.contains(key.as_str()) {
            out.push(values.clone());
        }
    }
    out
}

fn parse_sample_options(options_json: Option<&str>) -> Result<PySampleOptions, String> {
    match options_json {
        Some(raw) if !raw.trim().is_empty() => serde_json::from_str(raw)
            .map_err(|err| format!("failed to parse sample options json: {err}")),
        _ => Ok(PySampleOptions::default()),
    }
}

fn resolve_nuts_config(model: &FittedModel, options: PySampleOptions) -> NutsConfig {
    // Mirror the CLI's adaptive sizing so Python users get sensible defaults
    // without having to think about chain/warmup counts: NUTS samples needed
    // grow with the coefficient count, so we anchor on the saved beta length.
    let n_base_params = model
        .fit_result
        .as_ref()
        .map(|fr| fr.beta.len())
        .unwrap_or(0);
    let adaptive = NutsConfig::for_dimension(n_base_params);
    NutsConfig {
        n_samples: options.samples.unwrap_or(adaptive.n_samples),
        nwarmup: options.warmup.unwrap_or(adaptive.nwarmup),
        n_chains: options.chains.unwrap_or(adaptive.n_chains),
        target_accept: options.target_accept.unwrap_or(adaptive.target_accept),
        seed: options.seed.unwrap_or(adaptive.seed),
    }
}

#[derive(Serialize)]
struct DesignMatrixPayload {
    x_flat: Vec<f64>,
    n_rows: usize,
    n_cols: usize,
    family_kind: String,
    model_class: String,
}

fn design_matrix_table_impl(
    model_bytes: &[u8],
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
) -> Result<String, String> {
    let model = load_model_impl(model_bytes)?;
    let dataset = dataset_with_model_schema(&model, headers, rows)?;
    design_matrix_dataset_impl(&model, dataset)
}

fn design_matrix_array_impl(
    model_bytes: &[u8],
    x: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, String> {
    let model = load_model_impl(model_bytes)?;
    let dataset = dataset_from_x_array_with_model_schema(&model, x)?;
    design_matrix_dense(&model, dataset)
}

fn design_matrix_dataset_impl(
    model: &FittedModel,
    dataset: EncodedDataset,
) -> Result<String, String> {
    let dense = design_matrix_dense(model, dataset)?;
    let n_rows = dense.nrows();
    let n_cols = dense.ncols();
    // Row-major flatten matches numpy's default reshape order.
    let x_flat: Vec<f64> = dense.iter().copied().collect();
    let payload = DesignMatrixPayload {
        x_flat,
        n_rows,
        n_cols,
        family_kind: family_link_kind(model.likelihood()).to_string(),
        model_class: prediction_model_class_label(model),
    };
    serde_json::to_string(&payload)
        .map_err(|err| format!("failed to serialize design matrix payload: {err}"))
}

fn design_matrix_dense(
    model: &FittedModel,
    dataset: EncodedDataset,
) -> Result<Array2<f64>, String> {
    if !matches!(model.predict_model_class(), PredictModelClass::Standard) {
        return Err(format!(
            "design_matrix currently supports only standard GAM models; got '{}'. \
             For other classes use Model.predict / posterior.predict, which dispatch \
             through the saved-model predictor.",
            prediction_model_class_label(model)
        ));
    }
    if model.has_link_wiggle() {
        return Err(
            "design_matrix does not yet support link-wiggle models because the \
             linear predictor is q0 + B(q0)·theta, not a single X·beta product. \
             Use Model.predict for these models."
                .to_string(),
        );
    }
    let col_map = dataset.column_map();
    let training_headers = model.training_headers.as_ref();
    let spec = gam::survival_predict::resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        &col_map,
        "resolved_termspec",
    )?;
    let design = gam::smooth::build_term_collection_design(dataset.values.view(), &spec)
        .map_err(|err| format!("failed to build design matrix: {err}"))?;
    Ok(design.design.to_dense())
}

#[derive(Serialize)]
struct PosteriorPredictPayload {
    eta_flat: Vec<f64>,
    n_draws: usize,
    n_rows: usize,
    model_class: String,
    family_kind: String,
}

fn posterior_predict_table_impl(
    model_bytes: &[u8],
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    samples_flat: Vec<f64>,
    n_draws: usize,
    n_coeffs: usize,
) -> Result<String, String> {
    // Validation up front. The Python wrapper is expected to ship samples that
    // already match the saved coefficient count; we catch shape mismatches
    // here so the error surface stays consistent with the rest of the FFI
    // surface instead of failing inside a matmul.
    if samples_flat.len() != n_draws * n_coeffs {
        return Err(format!(
            "posterior_predict samples shape mismatch: got {} floats, expected {} * {}",
            samples_flat.len(),
            n_draws,
            n_coeffs
        ));
    }
    let model = load_model_impl(model_bytes)?;
    if !matches!(model.predict_model_class(), PredictModelClass::Standard) {
        return Err(format!(
            "posterior_predict currently supports only standard GAM models; got '{}'. \
             Per-class posterior-predict paths for location-scale / marginal-slope / \
             transformation-normal will be wired in a follow-up.",
            prediction_model_class_label(&model)
        ));
    }
    if model.has_link_wiggle() {
        return Err(
            "posterior_predict does not yet support link-wiggle models. The basis \
             chain rule q0 + B(q0)·theta means eta is not X·beta and per-draw \
             evaluation needs the link-wiggle runtime, which is a follow-up."
                .to_string(),
        );
    }
    let dataset = dataset_with_model_schema(&model, headers, rows)?;
    let col_map = dataset.column_map();
    let training_headers = model.training_headers.as_ref();
    let spec = gam::survival_predict::resolve_termspec_for_prediction(
        &model.resolved_termspec,
        training_headers,
        &col_map,
        "resolved_termspec",
    )?;
    let design = gam::smooth::build_term_collection_design(dataset.values.view(), &spec)
        .map_err(|err| format!("failed to build design matrix: {err}"))?;
    let dense = design.design.to_dense();
    let n_rows = dense.nrows();
    if dense.ncols() != n_coeffs {
        return Err(format!(
            "posterior_predict coefficient count mismatch: samples have {} coefficients \
             but rebuilt design has {} columns. The posterior was likely produced from a \
             different fit than this model; rerun model.sample(...) on the same model.",
            n_coeffs,
            dense.ncols()
        ));
    }
    let samples = Array2::<f64>::from_shape_vec((n_draws, n_coeffs), samples_flat)
        .map_err(|err| format!("failed to reshape samples: {err}"))?;
    // eta[k, i] = sum_j samples[k, j] * X[i, j] = (samples · X^T)[k, i].
    let eta = samples.dot(&dense.t());
    let eta_flat: Vec<f64> = eta.iter().copied().collect();
    let payload = PosteriorPredictPayload {
        eta_flat,
        n_draws,
        n_rows,
        model_class: prediction_model_class_label(&model),
        family_kind: family_link_kind(model.likelihood()).to_string(),
    };
    serde_json::to_string(&payload)
        .map_err(|err| format!("failed to serialize posterior_predict payload: {err}"))
}

fn sample_table_impl(
    model_bytes: &[u8],
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    options_json: Option<&str>,
) -> Result<String, String> {
    let model = load_model_impl(model_bytes)?;
    let dataset = dataset_with_model_schema(&model, headers, rows)?;
    let options = parse_sample_options(options_json)?;
    let cfg = resolve_nuts_config(&model, options);
    let col_map = dataset.column_map();
    let training_headers = model.training_headers.as_ref();
    let nuts = gam::sample::sample_saved_model(
        &model,
        dataset.values.view(),
        &col_map,
        training_headers,
        &cfg,
    )?;
    serialize_sample_payload(&model, &nuts, &cfg)
}

/// Returns the inverse-link kind tag emitted to the Python wrapper.
///
/// The tag is intentionally lower-kebab-case so it can be matched as a
/// plain string on the Python side (the Rust `LikelihoodFamily` enum
/// itself is not part of the FFI surface). Families that don't have a
/// closed-form scalar inverse link (`royston-parmar`, `binomial-sas`,
/// `binomial-beta-logistic`) get their own tags so the Python side can
/// refuse to compute a response-scale prediction by string-compare
/// instead of by guessing.
fn family_link_kind(family: LikelihoodFamily) -> &'static str {
    match family {
        LikelihoodFamily::GaussianIdentity => "identity",
        LikelihoodFamily::BinomialLogit => "logit",
        LikelihoodFamily::BinomialProbit => "probit",
        LikelihoodFamily::BinomialCLogLog => "cloglog",
        LikelihoodFamily::BinomialLatentCLogLog => "cloglog",
        LikelihoodFamily::PoissonLog => "log",
        LikelihoodFamily::GammaLog => "log",
        LikelihoodFamily::BinomialMixture => "logit",
        LikelihoodFamily::BinomialSas => "sas",
        LikelihoodFamily::BinomialBetaLogistic => "beta-logistic",
        LikelihoodFamily::RoystonParmar => "royston-parmar",
    }
}

/// Did this `NutsResult` come from exact NUTS or the Laplace fallback?
///
/// We badge the payload so the Python wrapper can surface the method
/// to users without re-deriving it from the model class. The fallback
/// produces iid draws and reports `rhat = 1.0` exactly with
/// `ess == n_draws`, which is a stable signature.
fn nuts_method_label(model: &FittedModel) -> &'static str {
    match model.predict_model_class() {
        PredictModelClass::Standard => "nuts",
        PredictModelClass::Survival => {
            // Survival latent / latent-binary / location-scale fall
            // back; everything else uses exact NUTS. Mirror the
            // dispatch in `gam::sample::sample_saved_model`.
            match model.survival_likelihood.as_deref() {
                Some("latent") | Some("latent-binary") | Some("location-scale") => "laplace",
                _ => "nuts",
            }
        }
        _ => "laplace",
    }
}

fn serialize_sample_payload(
    model: &FittedModel,
    nuts: &NutsResult,
    cfg: &NutsConfig,
) -> Result<String, String> {
    let n_draws = nuts.samples.nrows();
    let n_coeffs = nuts.samples.ncols();
    // Row-major flatten — ndarray's iter visits row-major already.
    let samples_flat: Vec<f64> = nuts.samples.iter().copied().collect();
    let coefficient_names: Vec<String> = (0..n_coeffs).map(|j| format!("beta_{j}")).collect();
    let payload = SamplePayload {
        samples_flat,
        n_draws,
        n_coeffs,
        coefficient_names,
        posterior_mean: nuts.posterior_mean.to_vec(),
        posterior_std: nuts.posterior_std.to_vec(),
        rhat: nuts.rhat,
        ess: nuts.ess,
        converged: nuts.converged,
        config: SampleConfigPayload {
            n_samples: cfg.n_samples,
            n_warmup: cfg.nwarmup,
            n_chains: cfg.n_chains,
            target_accept: cfg.target_accept,
            seed: cfg.seed,
        },
        model_class: prediction_model_class_label(model),
        family_kind: family_link_kind(model.likelihood()).to_string(),
        method: nuts_method_label(model).to_string(),
    };
    serde_json::to_string(&payload)
        .map_err(|err| format!("failed to serialize sample payload: {err}"))
}

fn summary_json_impl(model_bytes: &[u8]) -> Result<String, String> {
    let model = load_model_impl(model_bytes)?;
    let fit = fit_result_from_saved_model_for_prediction(&model)?;
    let standard_errors = fit
        .beta_standard_errors_corrected()
        .or_else(|| fit.beta_standard_errors());
    let coefficients = fit
        .beta
        .iter()
        .enumerate()
        .map(|(index, estimate)| SummaryCoefficientRow {
            index,
            estimate: *estimate,
            std_error: standard_errors.and_then(|values| values.get(index).copied()),
        })
        .collect();
    let payload = SummaryPayload {
        formula: model.payload().formula.clone(),
        family_name: pretty_familyname(model.likelihood()).to_string(),
        model_class: prediction_model_class_label(&model),
        deviance: fit.deviance,
        reml_score: fit.reml_score,
        iterations: fit.outer_iterations,
        edf_total: fit.edf_total(),
        coefficients,
    };
    serde_json::to_string(&payload).map_err(|err| format!("failed to serialize summary: {err}"))
}

fn check_json_impl(
    model_bytes: &[u8],
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
) -> Result<String, String> {
    let model = load_model_impl(model_bytes)?;
    let check = schema_check(&model, &headers, &rows)?;
    serde_json::to_string(&check).map_err(|err| format!("failed to serialize schema check: {err}"))
}

fn report_html_impl(model_bytes: &[u8]) -> Result<String, String> {
    let model = load_model_impl(model_bytes)?;
    let fit = fit_result_from_saved_model_for_prediction(&model)?;
    let standard_errors = fit
        .beta_standard_errors_corrected()
        .or_else(|| fit.beta_standard_errors());
    let coefficients = fit
        .beta
        .iter()
        .enumerate()
        .map(|(index, estimate)| CoefficientRow {
            index,
            estimate: *estimate,
            std_error: standard_errors.and_then(|values| values.get(index).copied()),
        })
        .collect::<Vec<_>>();
    let edf_blocks = fit
        .edf_by_block()
        .iter()
        .enumerate()
        .map(|(index, edf)| EdfBlockRow {
            index,
            edf: *edf,
            role: fit
                .blocks
                .get(index)
                .map(|block| block.role.name().to_string()),
        })
        .collect::<Vec<_>>();
    let report_input = ReportInput {
        model_path: "<in-memory>".to_string(),
        family_name: pretty_familyname(model.likelihood()).to_string(),
        model_class: prediction_model_class_label(&model),
        formula: model.payload().formula.clone(),
        n_obs: None,
        deviance: fit.deviance,
        reml_score: fit.reml_score,
        iterations: fit.outer_iterations,
        edf_total: fit.edf_total().unwrap_or(0.0),
        r_squared: None,
        coefficients,
        edf_blocks,
        continuous_order: Vec::new(),
        anisotropic_scales: Vec::new(),
        diagnostics: None,
        smooth_plots: Vec::new(),
        alo: None,
        notes: vec![
            "Python report currently omits data-dependent diagnostics and smooth plots."
                .to_string(),
        ],
    };
    render_html(&report_input)
}

fn parse_fit_config(config_json: Option<&str>) -> Result<FitConfig, String> {
    let py_config = match config_json {
        Some(raw) if !raw.trim().is_empty() => serde_json::from_str::<PyFitConfig>(raw)
            .map_err(|err| format!("invalid fit config json: {err}"))?,
        _ => PyFitConfig::default(),
    };
    let mut fit_config = FitConfig::default();
    fit_config.family = normalize_optional_family(py_config.family);
    fit_config.offset_column = py_config.offset;
    fit_config.weight_column = py_config.weights;
    if let Some(ridge_lambda) = py_config.ridge_lambda {
        fit_config.ridge_lambda = ridge_lambda;
    }
    if let Some(flag) = py_config.transformation_normal {
        fit_config.transformation_normal = flag;
    }
    if let Some(mode) = py_config.survival_likelihood {
        let trimmed = mode.trim();
        if trimmed.is_empty() {
            return Err("survival_likelihood must be a non-empty string".to_string());
        }
        fit_config.survival_likelihood = trimmed.to_string();
    }
    if let Some(target) = py_config.baseline_target {
        let trimmed = target.trim();
        if trimmed.is_empty() {
            return Err("baseline_target must be a non-empty string".to_string());
        }
        fit_config.baseline_target = trimmed.to_string();
    }
    if let Some(value) = py_config.baseline_scale {
        fit_config.baseline_scale = Some(value);
    }
    if let Some(value) = py_config.baseline_shape {
        fit_config.baseline_shape = Some(value);
    }
    if let Some(value) = py_config.baseline_rate {
        fit_config.baseline_rate = Some(value);
    }
    if let Some(value) = py_config.baseline_makeham {
        fit_config.baseline_makeham = Some(value);
    }
    if let Some(z) = py_config.z_column {
        let trimmed = z.trim();
        if trimmed.is_empty() {
            return Err("z_column must be a non-empty string".to_string());
        }
        fit_config.z_column = Some(trimmed.to_string());
    }
    if let Some(formula) = py_config.logslope_formula {
        fit_config.logslope_formula = Some(formula);
    }
    if let Some(link) = py_config.link {
        let trimmed = link.trim();
        fit_config.link = if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        };
    }
    if let Some(flag) = py_config.flexible_link {
        fit_config.flexible_link = flag;
    }
    if let Some(flag) = py_config.scale_dimensions {
        fit_config.scale_dimensions = flag;
    }
    if let Some(flag) = py_config.adaptive_regularization {
        fit_config.adaptive_regularization = Some(flag);
    }
    if let Some(formula) = py_config.noise_formula {
        fit_config.noise_formula = Some(formula);
    }
    if let Some(column) = py_config.noise_offset {
        fit_config.noise_offset_column = Some(column);
    }
    if let Some(flag) = py_config.firth {
        fit_config.firth = flag;
    }
    if let Some(kind) = py_config.frailty_kind {
        let trimmed = kind.trim().to_ascii_lowercase();
        let sigma = py_config.frailty_sd;
        if let Some(value) = sigma
            && (!value.is_finite() || value < 0.0)
        {
            return Err(format!("frailty_sd must be finite and >= 0, got {value}"));
        }
        let hazard_loading = py_config
            .hazard_loading
            .as_ref()
            .map(|raw| raw.trim().to_ascii_lowercase());
        let frailty = match trimmed.as_str() {
            "none" | "" => {
                if sigma.is_some() || hazard_loading.is_some() {
                    return Err(
                        "frailty_kind='none' does not accept frailty_sd or hazard_loading"
                            .to_string(),
                    );
                }
                gam::families::lognormal_kernel::FrailtySpec::None
            }
            "hazard-multiplier" => {
                let loading = match hazard_loading.as_deref() {
                    Some("full") | None => gam::families::lognormal_kernel::HazardLoading::Full,
                    Some("loaded-vs-unloaded") => {
                        gam::families::lognormal_kernel::HazardLoading::LoadedVsUnloaded
                    }
                    Some(other) => {
                        return Err(format!(
                            "unknown hazard_loading '{other}'; supported: 'full', 'loaded-vs-unloaded'"
                        ));
                    }
                };
                gam::families::lognormal_kernel::FrailtySpec::HazardMultiplier {
                    sigma_fixed: sigma,
                    loading,
                }
            }
            "gaussian-shift" => {
                if hazard_loading.is_some() {
                    return Err(
                        "hazard_loading is valid only with frailty_kind='hazard-multiplier'"
                            .to_string(),
                    );
                }
                gam::families::lognormal_kernel::FrailtySpec::GaussianShift { sigma_fixed: sigma }
            }
            other => {
                return Err(format!(
                    "unknown frailty_kind '{other}'; supported: 'none', 'hazard-multiplier', 'gaussian-shift'"
                ));
            }
        };
        fit_config.frailty = Some(frailty);
    } else if py_config.frailty_sd.is_some() || py_config.hazard_loading.is_some() {
        return Err(
            "frailty_kind is required when frailty_sd or hazard_loading is provided".to_string(),
        );
    }
    Ok(fit_config)
}

fn request_metadata(request: &FitRequest<'_>) -> (&'static str, &'static str, bool) {
    match request {
        FitRequest::Standard(standard_request) => {
            (pretty_familyname(standard_request.family), "standard", true)
        }
        FitRequest::GaussianLocationScale(_) => {
            ("Gaussian location-scale", "gaussian location-scale", true)
        }
        FitRequest::BinomialLocationScale(_) => {
            ("Binomial location-scale", "binomial location-scale", true)
        }
        FitRequest::SurvivalLocationScale(_) => {
            ("Survival location-scale", "survival location-scale", true)
        }
        FitRequest::SurvivalTransformation(request) => match request.spec.likelihood_mode {
            gam::families::survival_construction::SurvivalLikelihoodMode::Weibull => {
                ("Survival Weibull", "survival", true)
            }
            _ => ("Survival", "survival", true),
        },
        FitRequest::BernoulliMarginalSlope(_) => {
            ("Bernoulli marginal-slope", "bernoulli marginal-slope", true)
        }
        FitRequest::SurvivalMarginalSlope(_) => {
            ("Survival marginal-slope", "survival marginal-slope", true)
        }
        FitRequest::LatentSurvival(_) => ("Latent survival", "latent survival", true),
        FitRequest::LatentBinary(_) => ("Latent binary", "latent binary", true),
        FitRequest::TransformationNormal(_) => {
            ("Transformation-normal", "transformation-normal", true)
        }
    }
}

fn parse_predict_options(options_json: Option<&str>) -> Result<PyPredictOptions, String> {
    let options = match options_json {
        Some(raw) if !raw.trim().is_empty() => serde_json::from_str::<PyPredictOptions>(raw)
            .map_err(|err| format!("invalid predict options json: {err}"))?,
        _ => PyPredictOptions::default(),
    };
    if let Some(interval) = options.interval {
        if !(0.0 < interval && interval < 1.0) {
            return Err(format!(
                "prediction interval must be in (0, 1); got {interval}"
            ));
        }
    }
    Ok(options)
}

fn normalize_optional_family(family: Option<String>) -> Option<String> {
    match family {
        Some(value) if value.eq_ignore_ascii_case("auto") => None,
        other => other,
    }
}

fn dataset_with_inferred_schema(
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
) -> Result<EncodedDataset, String> {
    let n_rows = rows.len();
    let n_cols = headers.len();
    let t_records = std::time::Instant::now();
    let records = string_records_from_rows(&headers, rows)?;
    let records_ms = t_records.elapsed().as_secs_f64() * 1000.0;
    if records_ms > 100.0 {
        log::info!(
            "[DATA-LOAD] ffi_string_records | n_rows={} | n_cols={} | {:.1}ms",
            n_rows,
            n_cols,
            records_ms
        );
    }
    let t_encode = std::time::Instant::now();
    let result = encode_recordswith_inferred_schema(headers, records)?;
    let encode_ms = t_encode.elapsed().as_secs_f64() * 1000.0;
    if encode_ms > 100.0 {
        log::info!(
            "[DATA-LOAD] ffi_encode_inferred | n_rows={} | n_cols={} | {:.1}ms",
            n_rows,
            n_cols,
            encode_ms
        );
    }
    Ok(result)
}

fn dataset_with_model_schema(
    model: &FittedModel,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
) -> Result<EncodedDataset, String> {
    let check = schema_check(model, &headers, &rows)?;
    if !check.ok {
        let messages = check
            .issues
            .iter()
            .map(|issue| issue.message.clone())
            .collect::<Vec<_>>();
        return Err(messages.join(" "));
    }
    let schema = model.require_data_schema()?;
    let records = string_records_from_rows(&headers, rows)?;
    encode_recordswith_schema(headers, records, schema, UnseenCategoryPolicy::Error)
}

fn dataset_from_xy_arrays(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    formula: &str,
) -> Result<EncodedDataset, String> {
    if x.nrows() == 0 || y.nrows() == 0 {
        return Err("array data cannot be empty".to_string());
    }
    if x.nrows() != y.nrows() {
        return Err(format!(
            "X/Y row mismatch: X has {} rows but Y has {} rows",
            x.nrows(),
            y.nrows()
        ));
    }
    if x.ncols() == 0 {
        return Err("X must have at least one column".to_string());
    }
    if y.ncols() == 0 {
        return Err("Y must have at least one column".to_string());
    }
    let response_name = response_column_name(formula).unwrap_or_else(|| "y".to_string());
    let mut headers = Vec::<String>::with_capacity(y.ncols() + x.ncols());
    if y.ncols() == 1 {
        headers.push(response_name);
    } else {
        headers.extend((0..y.ncols()).map(|index| format!("y{index}")));
    }
    headers.extend((0..x.ncols()).map(|index| format!("x{index}")));
    ensure_unique_headers(&headers)?;

    let mut values = Array2::<f64>::zeros((x.nrows(), headers.len()));
    values.slice_mut(ndarray::s![.., 0..y.ncols()]).assign(&y);
    values.slice_mut(ndarray::s![.., y.ncols()..]).assign(&x);
    dataset_from_numeric_array(headers, values)
}

fn dataset_from_x_array_with_model_schema(
    model: &FittedModel,
    x: ArrayView2<'_, f64>,
) -> Result<EncodedDataset, String> {
    if x.nrows() == 0 {
        return Err("array data cannot be empty".to_string());
    }
    if x.ncols() == 0 {
        return Err("X must have at least one column".to_string());
    }
    let headers = (0..x.ncols())
        .map(|index| format!("x{index}"))
        .collect::<Vec<_>>();
    ensure_required_numeric_array_columns(model, &headers)?;
    let schema = model.require_data_schema()?;
    dataset_from_numeric_array_with_schema(headers, x.to_owned(), schema)
}

fn dataset_from_numeric_array(
    headers: Vec<String>,
    values: Array2<f64>,
) -> Result<EncodedDataset, String> {
    ensure_unique_headers(&headers)?;
    validate_numeric_array_values(&headers, values.view())?;
    let mut schema_cols = Vec::<SchemaColumn>::with_capacity(headers.len());
    let mut column_kinds = Vec::<ColumnKindTag>::with_capacity(headers.len());
    for (j, name) in headers.iter().enumerate() {
        let kind = infer_numeric_array_column_kind(values.column(j));
        column_kinds.push(kind);
        schema_cols.push(SchemaColumn {
            name: name.clone(),
            kind,
            levels: Vec::new(),
        });
    }
    Ok(EncodedDataset {
        headers,
        values,
        schema: DataSchema {
            columns: schema_cols,
        },
        column_kinds,
    })
}

fn dataset_from_numeric_array_with_schema(
    headers: Vec<String>,
    values: Array2<f64>,
    schema: &DataSchema,
) -> Result<EncodedDataset, String> {
    ensure_unique_headers(&headers)?;
    validate_numeric_array_values(&headers, values.view())?;
    let schema_byname = schema
        .columns
        .iter()
        .map(|column| (column.name.as_str(), column))
        .collect::<HashMap<_, _>>();
    let mut schema_cols = Vec::<SchemaColumn>::with_capacity(headers.len());
    let mut column_kinds = Vec::<ColumnKindTag>::with_capacity(headers.len());
    for (j, name) in headers.iter().enumerate() {
        let column = schema_byname.get(name.as_str()).ok_or_else(|| {
            format!("array column '{name}' was not present in the training schema")
        })?;
        match column.kind {
            ColumnKindTag::Categorical => {
                return Err(format!(
                    "array FFI only supports numeric continuous/binary columns; column '{name}' is categorical in the training schema"
                ));
            }
            ColumnKindTag::Binary => {
                for (row, value) in values.column(j).iter().enumerate() {
                    if (*value - 0.0).abs() >= 1e-12 && (*value - 1.0).abs() >= 1e-12 {
                        return Err(format!(
                            "column '{name}' is binary in schema but row {} has value {}; expected 0 or 1",
                            row + 1,
                            value
                        ));
                    }
                }
            }
            ColumnKindTag::Continuous => {}
        }
        column_kinds.push(column.kind);
        schema_cols.push((*column).clone());
    }
    Ok(EncodedDataset {
        headers,
        values,
        schema: DataSchema {
            columns: schema_cols,
        },
        column_kinds,
    })
}

fn ensure_required_numeric_array_columns(
    model: &FittedModel,
    headers: &[String],
) -> Result<(), String> {
    let expected_names = required_prediction_columns(model)?;
    let present_names = headers.iter().cloned().collect::<BTreeSet<_>>();
    let missing = expected_names
        .difference(&present_names)
        .map(|name| format!("missing required column '{name}'"))
        .collect::<Vec<_>>();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(missing.join(" "))
    }
}

fn ensure_unique_headers(headers: &[String]) -> Result<(), String> {
    if headers.is_empty() {
        return Err("table must have at least one column".to_string());
    }
    if headers.iter().any(|header| header.trim().is_empty()) {
        return Err("table headers must be non-empty strings".to_string());
    }
    let mut unique_headers = BTreeSet::<String>::new();
    for header in headers {
        if !unique_headers.insert(header.clone()) {
            return Err(format!("duplicate column name '{header}'"));
        }
    }
    Ok(())
}

fn validate_numeric_array_values(
    headers: &[String],
    values: ArrayView2<'_, f64>,
) -> Result<(), String> {
    if values.nrows() == 0 {
        return Err("array data cannot be empty".to_string());
    }
    if values.ncols() != headers.len() {
        return Err(format!(
            "array column count {} does not match header count {}",
            values.ncols(),
            headers.len()
        ));
    }
    for ((row, col), value) in values.indexed_iter() {
        if !value.is_finite() {
            return Err(format!(
                "non-finite value at row {}, column '{}'",
                row + 1,
                headers[col]
            ));
        }
    }
    Ok(())
}

fn infer_numeric_array_column_kind(column: ArrayView1<'_, f64>) -> ColumnKindTag {
    if column
        .iter()
        .all(|value| (*value - 0.0).abs() < 1e-12 || (*value - 1.0).abs() < 1e-12)
    {
        ColumnKindTag::Binary
    } else {
        ColumnKindTag::Continuous
    }
}

fn schema_check(
    model: &FittedModel,
    headers: &[String],
    rows: &[Vec<String>],
) -> Result<SchemaCheckPayload, String> {
    let schema = model.require_data_schema()?;
    let expected_names = required_prediction_columns(model)?;
    let present_names = headers.iter().cloned().collect::<BTreeSet<_>>();
    let mut issues = Vec::<SchemaIssue>::new();

    for missing in expected_names.difference(&present_names) {
        issues.push(SchemaIssue {
            kind: "missing_column".to_string(),
            message: format!("missing required column '{missing}'"),
            column: Some(missing.clone()),
        });
    }

    if issues.is_empty() {
        let records = string_records_from_rows(headers, rows.to_vec())?;
        if let Err(message) = encode_recordswith_schema(
            headers.to_vec(),
            records,
            schema,
            UnseenCategoryPolicy::Error,
        ) {
            issues.push(SchemaIssue {
                kind: "schema_error".to_string(),
                message,
                column: None,
            });
        }
    }

    Ok(SchemaCheckPayload {
        ok: issues.is_empty(),
        issues,
    })
}

fn response_column_name(formula: &str) -> Option<String> {
    let candidate = formula.split('~').next()?.trim();
    if candidate.is_empty() || candidate.starts_with("Surv(") {
        None
    } else {
        Some(candidate.to_string())
    }
}

fn prediction_model_class_label(model: &FittedModel) -> String {
    let payload = model.payload();
    match &payload.family_state {
        FittedFamily::Survival {
            survival_likelihood,
            ..
        } => match survival_likelihood
            .as_deref()
            .or(payload.survival_likelihood.as_deref())
        {
            Some("marginal-slope") => "survival marginal-slope".to_string(),
            Some("location-scale") => "survival location-scale".to_string(),
            _ => model.predict_model_class().name().to_string(),
        },
        FittedFamily::LatentSurvival { .. } => "latent survival".to_string(),
        FittedFamily::LatentBinary { .. } => "latent binary".to_string(),
        _ => model.predict_model_class().name().to_string(),
    }
}

fn required_prediction_columns(model: &FittedModel) -> Result<BTreeSet<String>, String> {
    let payload = model.payload();
    let parsed = parse_formula(payload.formula.as_str())?;
    let mut required = BTreeSet::<String>::new();
    add_formula_term_columns(&mut required, &parsed.terms);

    if let Some((entry, exit, _event)) = parse_surv_response(parsed.response.as_str())? {
        required.insert(entry);
        required.insert(exit);
    } else if matches!(
        model.predict_model_class(),
        PredictModelClass::TransformationNormal
    ) {
        if let Some(response) = response_column_name(payload.formula.as_str()) {
            required.insert(response);
        }
    }

    if let Some(offset) = payload.offset_column.as_ref() {
        required.insert(offset.clone());
    }
    if let Some(noise_offset) = payload.noise_offset_column.as_ref() {
        required.insert(noise_offset.clone());
    }
    if matches!(
        model.predict_model_class(),
        PredictModelClass::BernoulliMarginalSlope | PredictModelClass::Survival
    ) {
        if let Some(z_column) = payload.z_column.as_ref() {
            required.remove("z");
            required.insert(z_column.clone());
        }
    }
    if let Some(noise_formula) = payload.formula_noise.as_ref() {
        add_auxiliary_formula_columns(&mut required, noise_formula, parsed.response.as_str())?;
    }
    if let Some(logslope_formula) = payload.formula_logslope.as_ref() {
        if logslope_formula != "same-as-main" {
            add_auxiliary_formula_columns(
                &mut required,
                logslope_formula,
                parsed.response.as_str(),
            )?;
        }
    }
    Ok(required)
}

fn add_auxiliary_formula_columns(
    required: &mut BTreeSet<String>,
    formula_or_rhs: &str,
    response: &str,
) -> Result<(), String> {
    let trimmed = formula_or_rhs.trim();
    if trimmed.is_empty() || trimmed == "1" {
        return Ok(());
    }
    let formula = if trimmed.contains('~') {
        trimmed.to_string()
    } else {
        format!("{response} ~ {trimmed}")
    };
    let parsed = parse_formula(formula.as_str())?;
    add_formula_term_columns(required, &parsed.terms);
    Ok(())
}

fn add_formula_term_columns(required: &mut BTreeSet<String>, terms: &[ParsedTerm]) {
    for term in terms {
        match term {
            ParsedTerm::Linear { name, .. }
            | ParsedTerm::BoundedLinear { name, .. }
            | ParsedTerm::RandomEffect { name } => {
                required.insert(name.clone());
            }
            ParsedTerm::Smooth { vars, .. } => {
                required.extend(vars.iter().cloned());
            }
            ParsedTerm::LinkWiggle { .. }
            | ParsedTerm::TimeWiggle { .. }
            | ParsedTerm::LinkConfig { .. }
            | ParsedTerm::SurvivalConfig { .. } => {}
        }
    }
}

fn string_records_from_rows(
    headers: &[String],
    rows: Vec<Vec<String>>,
) -> Result<Vec<StringRecord>, String> {
    if headers.is_empty() {
        return Err("table must have at least one column".to_string());
    }
    if headers.iter().any(|header| header.trim().is_empty()) {
        return Err("table headers must be non-empty strings".to_string());
    }
    let mut unique_headers = BTreeSet::<String>::new();
    for header in headers {
        if !unique_headers.insert(header.clone()) {
            return Err(format!("duplicate column name '{header}'"));
        }
    }
    rows.into_iter()
        .enumerate()
        .map(|(index, row)| {
            if row.len() != headers.len() {
                return Err(format!(
                    "row {} has width {} but expected {}",
                    index + 1,
                    row.len(),
                    headers.len()
                ));
            }
            Ok(StringRecord::from(row))
        })
        .collect()
}

fn bspline_basis_impl(
    t: ArrayView1<'_, f64>,
    knots: ArrayView1<'_, f64>,
    degree: usize,
    periodic: bool,
) -> Result<Array2<f64>, String> {
    validate_vector("t", t)?;
    validate_vector("knots", knots)?;
    if periodic {
        let (left, right, num_basis) = periodic_knot_domain(knots)?;
        create_periodic_bspline_basis_dense(t, (left, right), degree, num_basis)
            .map_err(|err| format!("failed to evaluate periodic B-spline basis: {err}"))
    } else {
        let (basis, _) = create_basis::<Dense>(
            t,
            gam::basis::KnotSource::Provided(knots),
            degree,
            BasisOptions::value(),
        )
        .map_err(|err| format!("failed to evaluate B-spline basis: {err}"))?;
        Ok((*basis).clone())
    }
}

fn bspline_basis_derivative_impl(
    t: ArrayView1<'_, f64>,
    knots: ArrayView1<'_, f64>,
    degree: usize,
    order: usize,
    periodic: bool,
) -> Result<Array2<f64>, String> {
    validate_vector("t", t)?;
    validate_vector("knots", knots)?;
    if periodic {
        if order != 1 {
            return Err(format!(
                "periodic B-spline derivative supports order=1; got order={order}"
            ));
        }
        let (left, right, num_basis) = periodic_knot_domain(knots)?;
        create_periodic_bspline_derivative_dense(t, (left, right), degree, num_basis)
            .map_err(|err| format!("failed to evaluate periodic B-spline derivative: {err}"))
    } else {
        let options = match order {
            0 => BasisOptions::value(),
            1 => BasisOptions::first_derivative(),
            2 => BasisOptions::second_derivative(),
            _ => {
                return Err(format!(
                    "B-spline derivative supports orders 0, 1, and 2; got order={order}"
                ));
            }
        };
        let (basis, _) =
            create_basis::<Dense>(t, gam::basis::KnotSource::Provided(knots), degree, options)
                .map_err(|err| format!("failed to evaluate B-spline derivative: {err}"))?;
        Ok((*basis).clone())
    }
}

fn duchon_basis_1d_impl(
    t: ArrayView1<'_, f64>,
    centers: ArrayView1<'_, f64>,
    m: usize,
    periodic: bool,
) -> Result<Array2<f64>, String> {
    validate_vector("t", t)?;
    validate_vector("centers", centers)?;
    if m == 0 {
        return Err("Duchon m must be at least 1".to_string());
    }
    let data = column_array(t);
    let center_matrix = column_array(centers);
    let spec = DuchonBasisSpec {
        center_strategy: CenterStrategy::UserProvided(center_matrix),
        length_scale: None,
        power: 0,
        nullspace_order: duchon_nullspace_from_m(m),
        identifiability: SpatialIdentifiability::None,
        aniso_log_scales: None,
        operator_penalties: Default::default(),
        periodic,
    };
    let built = build_duchon_basis(data.view(), &spec)
        .map_err(|err| format!("failed to evaluate Duchon basis: {err}"))?;
    Ok(built.design.to_dense())
}

fn duchon_basis_1d_derivative_impl(
    t: ArrayView1<'_, f64>,
    centers: ArrayView1<'_, f64>,
    m: usize,
    order: usize,
    periodic: bool,
) -> Result<Array2<f64>, String> {
    validate_vector("t", t)?;
    validate_vector("centers", centers)?;
    if m == 0 {
        return Err("Duchon m must be at least 1".to_string());
    }
    create_duchon_basis_1d_derivative_dense(
        t,
        centers,
        0,
        duchon_nullspace_from_m(m),
        periodic,
        order,
    )
    .map_err(|err| format!("failed to evaluate Duchon basis derivative: {err}"))
}

fn smoothness_penalty_impl(
    knots: ArrayView1<'_, f64>,
    degree: usize,
    order: usize,
) -> Result<(Array2<f64>, Array2<f64>), String> {
    validate_vector("knots", knots)?;
    if knots.len() <= degree + 1 {
        return Err(format!(
            "knot vector is too short for degree={degree}: got {} knots",
            knots.len()
        ));
    }
    let num_basis = knots.len() - degree - 1;
    let greville = greville_abscissae(knots, degree, num_basis)?;
    let penalty = create_difference_penalty_matrix(num_basis, order, Some(greville.view()))
        .map_err(|err| format!("failed to build smoothness penalty: {err}"))?;
    let (null_basis, _) = gam::faer_ndarray::rrqr_nullspace_basis(
        &penalty,
        gam::faer_ndarray::default_rrqr_rank_alpha(),
    )
    .map_err(|err| format!("failed to build penalty null basis: {err}"))?;
    Ok((penalty, null_basis))
}

fn gaussian_weighted_ridge_array_impl(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: ArrayView1<'_, f64>,
    ridge_lambda: f64,
) -> Result<(Array2<f64>, Array2<f64>), String> {
    let n = x.nrows();
    let p = x.ncols();
    if n == 0 || p == 0 {
        return Err("X cannot be empty".to_string());
    }
    if y.nrows() != n {
        return Err(format!(
            "X/Y row mismatch: X has {n} rows but Y has {} rows",
            y.nrows()
        ));
    }
    if y.ncols() == 0 {
        return Err("Y must have at least one column".to_string());
    }
    if weights.len() != n {
        return Err(format!(
            "weights length mismatch: expected {n}, got {}",
            weights.len()
        ));
    }
    if penalty.nrows() != p || penalty.ncols() != p {
        return Err(format!(
            "penalty shape mismatch: expected {p}x{p}, got {}x{}",
            penalty.nrows(),
            penalty.ncols()
        ));
    }
    if !ridge_lambda.is_finite() || ridge_lambda < 0.0 {
        return Err(format!(
            "ridge_lambda must be finite and non-negative; got {ridge_lambda}"
        ));
    }
    if x.iter()
        .chain(y.iter())
        .chain(penalty.iter())
        .chain(weights.iter())
        .any(|value| !value.is_finite())
    {
        return Err("weighted ridge inputs must be finite".to_string());
    }
    if weights.iter().any(|value| *value < 0.0) {
        return Err("weights must be non-negative likelihood row weights".to_string());
    }

    let mut wx = x.to_owned();
    let mut wy = y.to_owned();
    for i in 0..n {
        let wi = weights[i];
        wx.row_mut(i).iter_mut().for_each(|value| *value *= wi);
        wy.row_mut(i).iter_mut().for_each(|value| *value *= wi);
    }
    let mut system = x.t().dot(&wx);
    if ridge_lambda > 0.0 {
        system += &(penalty.to_owned() * ridge_lambda);
    }
    let rhs = x.t().dot(&wy);
    let factor = factorize_symmetricwith_fallback(
        gam::faer_ndarray::FaerArrayView::new(&system).as_ref(),
        Side::Lower,
    )
    .map_err(|err| format!("weighted ridge factorization failed: {err}"))?;
    let mut coefficients = rhs;
    let mut coefficients_view = array2_to_matmut(&mut coefficients);
    factor.solve_in_place(coefficients_view.as_mut());
    if coefficients.iter().any(|value| !value.is_finite()) {
        return Err("weighted ridge solve produced non-finite coefficients".to_string());
    }
    let fitted = x.dot(&coefficients);
    Ok((coefficients, fitted))
}

fn gaussian_weighted_ridge_batch_impl(
    x: ArrayView3<'_, f64>,
    y: ArrayView3<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: ArrayView2<'_, f64>,
    ridge_lambda: f64,
    row_counts: Option<ArrayView1<'_, usize>>,
) -> Result<(Array3<f64>, Array3<f64>), String> {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    let (batch, n_max, p) = x.dim();
    let (y_batch, y_n_max, d) = y.dim();
    if batch == 0 || n_max == 0 || p == 0 {
        return Err("batched X must have non-empty K, N, and coefficient dimensions".to_string());
    }
    if y_batch != batch || y_n_max != n_max {
        return Err(format!(
            "batched X/Y shape mismatch: X is ({batch}, {n_max}, {p}) but Y is ({y_batch}, {y_n_max}, {d})"
        ));
    }
    if d == 0 {
        return Err("batched Y must have at least one output column".to_string());
    }
    if weights.nrows() != batch || weights.ncols() != n_max {
        return Err(format!(
            "batched weights shape mismatch: expected ({batch}, {n_max}), got ({}, {})",
            weights.nrows(),
            weights.ncols()
        ));
    }
    if penalty.nrows() != p || penalty.ncols() != p {
        return Err(format!(
            "penalty shape mismatch: expected {p}x{p}, got {}x{}",
            penalty.nrows(),
            penalty.ncols()
        ));
    }
    if !ridge_lambda.is_finite() || ridge_lambda < 0.0 {
        return Err(format!(
            "ridge_lambda must be finite and non-negative; got {ridge_lambda}"
        ));
    }
    if x.iter()
        .chain(y.iter())
        .chain(penalty.iter())
        .chain(weights.iter())
        .any(|value| !value.is_finite())
    {
        return Err("batched weighted ridge inputs must be finite".to_string());
    }
    if weights.iter().any(|value| *value < 0.0) {
        return Err("batched weights must be non-negative likelihood row weights".to_string());
    }

    let active_rows: Vec<usize> = match row_counts {
        Some(counts) => {
            if counts.len() != batch {
                return Err(format!(
                    "row_counts length mismatch: expected {batch}, got {}",
                    counts.len()
                ));
            }
            counts.to_vec()
        }
        None => vec![n_max; batch],
    };
    for (b, &n_rows) in active_rows.iter().enumerate() {
        if n_rows > n_max {
            return Err(format!(
                "row_counts[{b}]={n_rows} exceeds padded row count {n_max}"
            ));
        }
    }

    let results: Vec<Result<(usize, Array2<f64>, Array2<f64>), String>> = (0..batch)
        .into_par_iter()
        .map(|b| {
            let n_rows = active_rows[b];
            if n_rows == 0 {
                return Ok((
                    b,
                    Array2::<f64>::zeros((p, d)),
                    Array2::<f64>::zeros((0, d)),
                ));
            }
            gaussian_weighted_ridge_array_impl(
                x.slice(s![b, 0..n_rows, ..]),
                y.slice(s![b, 0..n_rows, ..]),
                penalty,
                weights.slice(s![b, 0..n_rows]),
                ridge_lambda,
            )
            .map(|(coefficients, fitted)| (b, coefficients, fitted))
            .map_err(|err| format!("batched weighted ridge fit {b} failed: {err}"))
        })
        .collect();

    let mut coefficients = Array3::<f64>::zeros((batch, p, d));
    let mut fitted = Array3::<f64>::zeros((batch, n_max, d));
    for result in results {
        let (b, fit_coefficients, fit_fitted) = result?;
        coefficients
            .slice_mut(s![b, .., ..])
            .assign(&fit_coefficients);
        let n_rows = fit_fitted.nrows();
        if n_rows > 0 {
            fitted.slice_mut(s![b, 0..n_rows, ..]).assign(&fit_fitted);
        }
    }
    Ok((coefficients, fitted))
}

#[cfg(test)]
mod batch_tests {
    use super::*;
    use ndarray::{array, s};

    fn assert_close(lhs: ArrayView2<'_, f64>, rhs: ArrayView2<'_, f64>, tol: f64) {
        assert_eq!(lhs.dim(), rhs.dim());
        for ((i, j), value) in lhs.indexed_iter() {
            let diff = (*value - rhs[[i, j]]).abs();
            assert!(
                diff <= tol,
                "matrix mismatch at ({i}, {j}): lhs={}, rhs={}, diff={diff}",
                value,
                rhs[[i, j]]
            );
        }
    }

    #[test]
    fn weighted_ridge_batch_matches_single_fit_on_active_rows() {
        let x = Array3::from_shape_vec(
            (2, 3, 2),
            vec![1.0, 0.0, 1.0, 1.0, 0.5, 1.0, 2.0, 1.0, 0.0, 1.0, 9.0, 9.0],
        )
        .unwrap();
        let y = Array3::from_shape_vec((2, 3, 1), vec![1.0, 2.0, 1.5, 2.5, -0.5, 99.0]).unwrap();
        let weights = array![[1.0, 0.5, 2.0], [1.0, 3.0, 0.0]];
        let penalty = Array2::eye(2);
        let row_counts = array![3_usize, 2_usize];

        let (coefficients, fitted) = gaussian_weighted_ridge_batch_impl(
            x.view(),
            y.view(),
            penalty.view(),
            weights.view(),
            0.25,
            Some(row_counts.view()),
        )
        .unwrap();

        for b in 0..2 {
            let n = row_counts[b];
            let (expected_coefficients, expected_fitted) = gaussian_weighted_ridge_array_impl(
                x.slice(s![b, 0..n, ..]),
                y.slice(s![b, 0..n, ..]),
                penalty.view(),
                weights.slice(s![b, 0..n]),
                0.25,
            )
            .unwrap();
            assert_close(
                coefficients.slice(s![b, .., ..]),
                expected_coefficients.view(),
                1.0e-10,
            );
            assert_close(
                fitted.slice(s![b, 0..n, ..]),
                expected_fitted.view(),
                1.0e-10,
            );
        }
        assert_eq!(fitted[[1, 2, 0]], 0.0);
    }

    #[test]
    fn gaussian_reml_batched_matches_single_fit_on_ragged_offsets() {
        let x = array![
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, -1.0],
            [1.0, 0.5],
            [1.0, 1.5],
        ];
        let y = array![[0.0], [1.0], [1.8], [-1.0], [0.2], [1.1]];
        let weights = array![1.0, 0.7, 1.3, 1.0, 1.1, 0.9];
        let penalty = array![[0.0, 0.0], [0.0, 1.0]];
        let offsets = array![0_usize, 3_usize, 6_usize];

        let batched = gaussian_reml_fit_batched_impl(
            x.view(),
            y.view(),
            offsets.view(),
            penalty.view(),
            Some(weights.view()),
            Some(0.5),
        )
        .unwrap();

        for b in 0..2 {
            let start = offsets[b];
            let end = offsets[b + 1];
            let single = gaussian_reml_multi_closed_form_with_cache(
                x.slice(s![start..end, ..]),
                y.slice(s![start..end, ..]),
                penalty.view(),
                Some(weights.slice(s![start..end])),
                Some(0.5),
                None,
            )
            .unwrap();
            assert_eq!(batched.statuses[b], "ok");
            assert!((batched.lambdas[b] - single.lambda).abs() < 1.0e-10);
            assert!((batched.reml_grad_lambdas[b] - single.reml_grad_lambda).abs() < 1.0e-10);
            assert!((batched.reml_hess_lambdas[b] - single.reml_hess_lambda).abs() < 1.0e-10);
            assert!((batched.reml_grad_rhos[b] - single.reml_grad_rho).abs() < 1.0e-10);
            assert!((batched.reml_hess_rhos[b] - single.reml_hess_rho).abs() < 1.0e-10);
            assert_close(
                batched.coefficients.slice(s![b, .., ..]),
                single.coefficients.view(),
                1.0e-10,
            );
            assert_close(
                batched.fitted.slice(s![start..end, ..]),
                single.fitted.view(),
                1.0e-10,
            );
        }
    }

    #[test]
    fn gaussian_reml_batched_backward_matches_single_backward_on_ragged_offsets() {
        let x = array![
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, -1.0],
            [1.0, 0.5],
            [1.0, 1.5],
        ];
        let y = array![[0.0], [1.0], [1.8], [-1.0], [0.2], [1.1]];
        let weights = array![1.0, 0.7, 1.3, 1.0, 1.1, 0.9];
        let penalty = array![[0.0, 0.0], [0.0, 1.0]];
        let offsets = array![0_usize, 3_usize, 6_usize];
        let grad_lambda = array![0.17, -0.03];
        let grad_coefficients =
            Array3::from_shape_vec((2, 2, 1), vec![0.2, -0.1, 0.05, 0.07]).unwrap();
        let grad_fitted = array![[0.01], [0.03], [-0.02], [0.04], [-0.01], [0.02]];
        let grad_reml_score = array![-0.11, 0.09];

        let batched = gaussian_reml_fit_batched_backward_impl(
            x.view(),
            y.view(),
            offsets.view(),
            penalty.view(),
            Some(weights.view()),
            Some(0.5),
            Some(grad_lambda.view()),
            Some(grad_coefficients.view()),
            Some(grad_fitted.view()),
            Some(grad_reml_score.view()),
        )
        .unwrap();

        for b in 0..2 {
            let start = offsets[b];
            let end = offsets[b + 1];
            let single = gaussian_reml_multi_closed_form_backward(
                x.slice(s![start..end, ..]),
                y.slice(s![start..end, ..]),
                penalty.view(),
                Some(weights.slice(s![start..end])),
                Some(0.5),
                grad_lambda[b],
                Some(grad_coefficients.slice(s![b, .., ..])),
                Some(grad_fitted.slice(s![start..end, ..])),
                grad_reml_score[b],
            )
            .unwrap();
            assert_eq!(batched.statuses[b], "ok");
            assert_close(
                batched.grad_x.slice(s![start..end, ..]),
                single.grad_x.view(),
                1.0e-10,
            );
            assert_close(
                batched.grad_y.slice(s![start..end, ..]),
                single.grad_y.view(),
                1.0e-10,
            );
            for row in start..end {
                assert!(
                    (batched.grad_weights[row] - single.grad_weights[row - start]).abs() <= 1.0e-10
                );
            }
        }
    }

    #[test]
    fn position_batched_backward_grad_t_matches_direct_t_finite_difference() {
        let t = array![
            0.08, 0.16, 0.27, 0.39, 0.51, 0.64, 0.76, 0.14, 0.24, 0.36, 0.52, 0.68, 0.84
        ];
        let y = Array2::from_shape_fn((t.len(), 2), |(row, output)| {
            let u = t[row];
            let scale = output as f64 + 1.0;
            0.3 + 0.4 * scale * u + 0.15 * (2.0 * u + 0.2 * scale).sin()
        });
        let offsets = array![0_usize, 7_usize, 13_usize];
        let knots = array![0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0];
        let penalty = Array2::from_diag(&array![0.0, 0.35, 0.8, 1.4, 2.2, 3.1, 4.0]);
        let weights = array![
            1.0, 0.9, 1.1, 1.2, 0.85, 1.05, 0.95, 1.0, 1.15, 0.88, 1.04, 0.93, 1.08
        ];
        let grad_lambda = array![0.07, -0.04];
        let grad_reml_score = array![0.11, -0.06];
        let grad_coefficients =
            Array3::from_shape_fn((2, penalty.nrows(), y.ncols()), |(b, i, j)| {
                0.015 * (b as f64 + 1.0) * (i as f64 - 2.0) + 0.01 * (j as f64 + 1.0)
            });
        let grad_fitted = Array2::from_shape_fn(y.dim(), |(row, output)| {
            0.02 * ((row as f64 + 1.0) * (output as f64 + 1.5)).sin()
        });

        let analytic = gaussian_reml_fit_positions_batched_backward_impl(
            t.view(),
            y.view(),
            offsets.view(),
            knots.view(),
            "bspline",
            3,
            false,
            None,
            penalty.view(),
            Some(weights.view()),
            Some(0.7),
            Some(grad_lambda.view()),
            Some(grad_coefficients.view()),
            Some(grad_fitted.view()),
            Some(grad_reml_score.view()),
            None,
            0,
        )
        .expect("position batched backward");

        let loss = |candidate_t: ArrayView1<'_, f64>| -> f64 {
            let fit = gaussian_reml_fit_positions_batched_impl(
                candidate_t,
                y.view(),
                offsets.view(),
                knots.view(),
                "bspline",
                3,
                false,
                None,
                penalty.view(),
                Some(weights.view()),
                Some(0.7),
                None,
                0,
            )
            .expect("position batched forward");
            let mut value = 0.0;
            for b in 0..offsets.len() - 1 {
                value += grad_lambda[b] * fit.lambdas[b];
                value += grad_reml_score[b] * fit.reml_scores[b];
                for i in 0..penalty.nrows() {
                    for j in 0..y.ncols() {
                        value += grad_coefficients[[b, i, j]] * fit.coefficients[[b, i, j]];
                    }
                }
            }
            for row in 0..y.nrows() {
                for j in 0..y.ncols() {
                    value += grad_fitted[[row, j]] * fit.fitted[[row, j]];
                }
            }
            value
        };

        let eps = 1.0e-6;
        for row in 0..t.len() {
            let mut plus = t.clone();
            let mut minus = t.clone();
            plus[row] += eps;
            minus[row] -= eps;
            let fd = (loss(plus.view()) - loss(minus.view())) / (2.0 * eps);
            let diff = (analytic.grad_t[row] - fd).abs();
            let tol = 5.0e-5_f64.max(5.0e-5 * analytic.grad_t[row].abs().max(fd.abs()));
            assert!(
                diff <= tol,
                "grad_t[{row}] mismatch: analytic={:.12e}, finite_difference={:.12e}, diff={diff:.3e}, tol={tol:.3e}",
                analytic.grad_t[row],
                fd
            );
        }
    }
}

fn validate_vector(name: &str, values: ArrayView1<'_, f64>) -> Result<(), String> {
    if values.is_empty() {
        return Err(format!("{name} cannot be empty"));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{name} must contain only finite values"));
    }
    Ok(())
}

fn periodic_knot_domain(knots: ArrayView1<'_, f64>) -> Result<(f64, f64, usize), String> {
    if knots.len() < 2 {
        return Err("periodic knots must contain at least start and end".to_string());
    }
    let left = knots[0];
    let right = knots[knots.len() - 1];
    if left >= right {
        return Err(format!(
            "periodic knot domain must be increasing; got [{left}, {right}]"
        ));
    }
    Ok((left, right, knots.len() - 1))
}

fn duchon_nullspace_from_m(m: usize) -> DuchonNullspaceOrder {
    match m {
        1 => DuchonNullspaceOrder::Zero,
        2 => DuchonNullspaceOrder::Linear,
        other => DuchonNullspaceOrder::Degree(other - 1),
    }
}

fn column_array(values: ArrayView1<'_, f64>) -> Array2<f64> {
    values.to_owned().insert_axis(Axis(1))
}

fn greville_abscissae(
    knots: ArrayView1<'_, f64>,
    degree: usize,
    num_basis: usize,
) -> Result<Array1<f64>, String> {
    if degree == 0 {
        return Err("smoothness_penalty requires degree >= 1".to_string());
    }
    let mut out = Array1::<f64>::zeros(num_basis);
    for i in 0..num_basis {
        let mut acc = 0.0;
        for j in 1..=degree {
            acc += knots[i + j];
        }
        out[i] = acc / degree as f64;
    }
    Ok(out)
}

fn build_standard_payload(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    family: LikelihoodFamily,
    saved_fit: &gam::estimate::UnifiedFitResult,
    resolved_termspec: TermCollectionSpec,
    adaptive_regularization_diagnostics: Option<gam::smooth::AdaptiveRegularizationDiagnostics>,
    wiggle_knots: Option<Vec<f64>>,
    wiggle_degree: Option<usize>,
) -> FittedModelPayload {
    let latent_cloglog_state =
        if matches!(family, LikelihoodFamily::BinomialLatentCLogLog) {
            Some(saved_latent_cloglog_state_from_fit(saved_fit).expect(
                "latent-cloglog-binomial fit must produce an explicit latent-cloglog state",
            ))
        } else {
            saved_latent_cloglog_state_from_fit(saved_fit)
        };
    let mut payload = FittedModelPayload::new(
        MODEL_VERSION,
        formula,
        ModelKind::Standard,
        FittedFamily::Standard {
            likelihood: family,
            link: Some(family_to_link(family)),
            latent_cloglog_state,
            mixture_state: saved_mixture_state_from_fit(saved_fit),
            sas_state: saved_sas_state_from_fit(saved_fit),
        },
        family.name().to_string(),
    );
    payload.unified = Some(saved_fit.clone());
    payload.fit_result = Some(saved_fit.clone());
    payload.data_schema = Some(dataset.schema.clone());
    payload.link = Some(family_to_link(family).name().to_string());
    payload.linkwiggle_knots = wiggle_knots;
    payload.linkwiggle_degree = wiggle_degree;
    payload.training_headers = Some(dataset.headers.clone());
    payload.resolved_termspec = Some(resolved_termspec);
    payload.adaptive_regularization_diagnostics = adaptive_regularization_diagnostics;
    payload.offset_column = fit_config.offset_column.clone();
    payload.noise_offset_column = fit_config.noise_offset_column.clone();
    payload
}

fn build_transformation_normal_ffi_payload(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    tn_result: TransformationNormalFitResult,
) -> Result<FittedModelPayload, String> {
    let frozen_covariate = freeze_term_collection_from_design(
        &tn_result.covariate_spec_resolved,
        &tn_result.covariate_design,
    )
    .map_err(|err| format!("failed to freeze transformation-normal covariate spec: {err}"))?;

    let family = &tn_result.family;
    let response_transform_rows: Vec<Vec<f64>> = family
        .response_transform()
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect();

    let mut payload = FittedModelPayload::new(
        MODEL_VERSION,
        formula,
        ModelKind::TransformationNormal,
        FittedFamily::TransformationNormal {
            likelihood: LikelihoodFamily::GaussianIdentity,
        },
        "transformation-normal".to_string(),
    );
    payload.unified = Some(tn_result.fit.clone());
    payload.fit_result = Some(tn_result.fit);
    payload.data_schema = Some(dataset.schema.clone());
    payload.set_training_feature_metadata(dataset.headers.clone(), dataset.feature_ranges());
    payload.resolved_termspec = Some(frozen_covariate);
    payload.transformation_response_knots = Some(family.response_knots().to_vec());
    payload.transformation_response_transform = Some(response_transform_rows);
    payload.transformation_response_degree = Some(family.response_degree());
    payload.transformation_response_median = Some(family.response_median());
    payload.transformation_score_calibration = Some(tn_result.score_calibration);
    payload.offset_column = fit_config.offset_column.clone();
    Ok(payload)
}

fn build_bernoulli_marginal_slope_ffi_payload(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    base_link: InverseLink,
    frailty: gam::families::lognormal_kernel::FrailtySpec,
    ms_result: BernoulliMarginalSlopeFitResult,
) -> Result<FittedModelPayload, String> {
    let frozen_marginal = freeze_term_collection_from_design(
        &ms_result.marginalspec_resolved,
        &ms_result.marginal_design,
    )
    .map_err(|err| format!("failed to freeze marginal spec: {err}"))?;
    let frozen_logslope = freeze_term_collection_from_design(
        &ms_result.logslopespec_resolved,
        &ms_result.logslope_design,
    )
    .map_err(|err| format!("failed to freeze logslope spec: {err}"))?;

    let logslope_formula = fit_config
        .logslope_formula
        .clone()
        .ok_or_else(|| "bernoulli marginal-slope requires logslope_formula".to_string())?;
    let z_column = fit_config
        .z_column
        .clone()
        .ok_or_else(|| "bernoulli marginal-slope requires z_column".to_string())?;

    let likelihood = inverse_link_to_binomial_family(&base_link);

    let mut payload = FittedModelPayload::new(
        MODEL_VERSION,
        formula,
        ModelKind::MarginalSlope,
        FittedFamily::MarginalSlope {
            likelihood,
            base_link: Some(base_link.clone()),
            frailty,
        },
        "bernoulli-marginal-slope".to_string(),
    );
    payload.unified = Some(ms_result.fit.clone());
    payload.fit_result = Some(ms_result.fit);
    payload.data_schema = Some(dataset.schema.clone());
    payload.formula_logslope = Some(logslope_formula);
    payload.z_column = Some(z_column);
    payload.latent_z_normalization = Some(SavedLatentZNormalization {
        mean: ms_result.z_normalization.mean,
        sd: ms_result.z_normalization.sd,
    });
    payload.latent_measure = Some(ms_result.latent_measure.clone());
    payload.latent_z_rank_int_calibration = ms_result.latent_z_rank_int_calibration.clone();
    payload.marginal_baseline = Some(ms_result.baseline_marginal);
    payload.logslope_baseline = Some(ms_result.baseline_logslope);
    payload.link = Some(base_link.saved_string());
    payload.training_headers = Some(dataset.headers.clone());
    payload.resolved_termspec = Some(frozen_marginal);
    payload.resolved_termspec_logslope = Some(frozen_logslope);
    payload.score_warp_runtime = ms_result
        .score_warp_runtime
        .as_ref()
        .map(saved_anchored_deviation_runtime);
    payload.link_deviation_runtime = ms_result
        .link_dev_runtime
        .as_ref()
        .map(saved_anchored_deviation_runtime);
    payload.offset_column = fit_config.offset_column.clone();
    payload.noise_offset_column = fit_config.noise_offset_column.clone();
    Ok(payload)
}

fn build_survival_marginal_slope_ffi_payload(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    frailty: gam::families::lognormal_kernel::FrailtySpec,
    ms_result: SurvivalMarginalSlopeFitResult,
) -> Result<FittedModelPayload, String> {
    let frozen_marginal = freeze_term_collection_from_design(
        &ms_result.marginalspec_resolved,
        &ms_result.marginal_design,
    )
    .map_err(|err| format!("failed to freeze survival marginal spec: {err}"))?;
    let frozen_logslope = freeze_term_collection_from_design(
        &ms_result.logslopespec_resolved,
        &ms_result.logslope_design,
    )
    .map_err(|err| format!("failed to freeze survival logslope spec: {err}"))?;

    let logslope_formula = fit_config
        .logslope_formula
        .clone()
        .unwrap_or_else(|| "same-as-main".to_string());
    let z_column = fit_config
        .z_column
        .clone()
        .ok_or_else(|| "survival marginal-slope requires z_column".to_string())?;
    let parsed = parse_formula(&formula)
        .map_err(|err| format!("failed to re-parse survival marginal formula: {err}"))?;
    let (entryname, exitname, eventname) = parse_surv_response(&parsed.response)?
        .ok_or_else(|| "survival marginal-slope FFI requires Surv(...) response".to_string())?;

    let mut payload = FittedModelPayload::new(
        MODEL_VERSION,
        formula,
        ModelKind::Survival,
        FittedFamily::Survival {
            likelihood: LikelihoodFamily::RoystonParmar,
            survival_likelihood: Some("marginal-slope".to_string()),
            survival_distribution: Some("probit".to_string()),
            frailty,
        },
        "royston-parmar".to_string(),
    );
    payload.unified = Some(ms_result.fit.clone());
    payload.fit_result = Some(ms_result.fit);
    payload.data_schema = Some(dataset.schema.clone());
    payload.survival_entry = Some(entryname);
    payload.survival_exit = Some(exitname);
    payload.survival_event = Some(eventname);
    payload.survivalridge_lambda = Some(fit_config.ridge_lambda);
    payload.survival_likelihood = Some("marginal-slope".to_string());
    payload.training_headers = Some(dataset.headers.clone());
    payload.resolved_termspec = Some(frozen_marginal);
    payload.resolved_termspec_logslope = Some(frozen_logslope);
    payload.formula_logslope = Some(logslope_formula);
    payload.z_column = Some(z_column);
    payload.latent_z_normalization = Some(SavedLatentZNormalization {
        mean: ms_result.z_normalization.mean,
        sd: ms_result.z_normalization.sd,
    });
    payload.latent_measure = Some(LatentMeasureKind::StandardNormal);
    payload.logslope_baseline = Some(ms_result.baseline_slope);
    payload.score_warp_runtime = ms_result
        .score_warp_runtime
        .as_ref()
        .map(saved_anchored_deviation_runtime);
    payload.link_deviation_runtime = ms_result
        .link_dev_runtime
        .as_ref()
        .map(saved_anchored_deviation_runtime);
    payload.offset_column = fit_config.offset_column.clone();
    payload.noise_offset_column = fit_config.noise_offset_column.clone();
    Ok(payload)
}

fn build_survival_transformation_ffi_payload(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    rp_result: gam::SurvivalTransformationFitResult,
) -> Result<FittedModelPayload, String> {
    use gam::families::survival_construction::{
        survival_baseline_targetname, survival_likelihood_modename,
    };
    use ndarray::s;

    let parsed = parse_formula(&formula)
        .map_err(|err| format!("failed to re-parse survival transformation formula: {err}"))?;
    let (entryname, exitname, eventname) = parse_surv_response(&parsed.response)?
        .ok_or_else(|| "survival transformation FFI requires Surv(...) response".to_string())?;
    let likelihood_label = survival_likelihood_modename(rp_result.likelihood_mode).to_string();

    let mut payload = FittedModelPayload::new(
        MODEL_VERSION,
        formula,
        ModelKind::Survival,
        FittedFamily::Survival {
            likelihood: LikelihoodFamily::RoystonParmar,
            survival_likelihood: Some(likelihood_label.clone()),
            survival_distribution: None,
            frailty: gam::families::lognormal_kernel::FrailtySpec::None,
        },
        "royston-parmar".to_string(),
    );
    payload.unified = Some(rp_result.fit.clone());
    payload.fit_result = Some(rp_result.fit.clone());
    payload.data_schema = Some(dataset.schema.clone());
    payload.survival_entry = Some(entryname);
    payload.survival_exit = Some(exitname);
    payload.survival_event = Some(eventname);
    payload.survivalspec = Some("net".to_string());
    payload.survival_baseline_target =
        Some(survival_baseline_targetname(rp_result.baseline_cfg.target).to_string());
    payload.survival_baseline_scale = rp_result.baseline_cfg.scale;
    payload.survival_baseline_shape = rp_result.baseline_cfg.shape;
    payload.survival_baseline_rate = rp_result.baseline_cfg.rate;
    payload.survival_baseline_makeham = rp_result.baseline_cfg.makeham;
    payload.survival_time_basis = Some(rp_result.time_basisname);
    payload.survival_time_degree = rp_result.time_degree;
    payload.survival_time_knots = rp_result.time_knots;
    payload.survival_time_keep_cols = rp_result.time_keep_cols;
    payload.survival_time_smooth_lambda = rp_result.time_smooth_lambda;
    payload.survival_time_anchor = Some(rp_result.time_anchor);
    payload.survivalridge_lambda = Some(fit_config.ridge_lambda);
    payload.survival_likelihood = Some(likelihood_label);
    if let Some(timewiggle) = rp_result.baseline_timewiggle.as_ref() {
        payload.baseline_timewiggle_degree = Some(timewiggle.degree);
        payload.baseline_timewiggle_knots = Some(timewiggle.knots.to_vec());
        payload.baseline_timewiggle_penalty_orders = parsed
            .timewiggle
            .as_ref()
            .map(|cfg| cfg.penalty_orders.clone());
        payload.baseline_timewiggle_double_penalty =
            parsed.timewiggle.as_ref().map(|cfg| cfg.double_penalty);
        let beta = &rp_result.fit.beta;
        let start = rp_result.time_base_ncols;
        let end = start + timewiggle.ncols;
        if beta.len() < end {
            return Err(format!(
                "survival transformation timewiggle beta mismatch: beta has {}, needs {end}",
                beta.len()
            ));
        }
        payload.beta_baseline_timewiggle = Some(beta.slice(s![start..end]).to_vec());
    }
    payload.training_headers = Some(dataset.headers.clone());
    payload.resolved_termspec = Some(rp_result.resolvedspec);
    payload.offset_column = fit_config.offset_column.clone();
    Ok(payload)
}

fn build_gaussian_location_scale_ffi_payload(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    ls_result: GaussianLocationScaleFitResult,
    response_scale: f64,
) -> Result<FittedModelPayload, String> {
    let frozen_meanspec = freeze_term_collection_from_design(
        &ls_result.fit.meanspec_resolved,
        &ls_result.fit.mean_design,
    )
    .map_err(|err| format!("failed to freeze gaussian location-scale mean spec: {err}"))?;
    let frozen_noisespec = freeze_term_collection_from_design(
        &ls_result.fit.noisespec_resolved,
        &ls_result.fit.noise_design,
    )
    .map_err(|err| format!("failed to freeze gaussian location-scale noise spec: {err}"))?;

    let noise_formula = fit_config
        .noise_formula
        .clone()
        .ok_or_else(|| "gaussian location-scale requires noise_formula".to_string())?;

    let fit = ls_result.fit.fit;
    let scale_beta = fit
        .block_by_role(BlockRole::Scale)
        .map(|block| block.beta.to_vec());
    let wiggle_meta = match (
        ls_result.wiggle_knots,
        ls_result.wiggle_degree,
        ls_result.beta_link_wiggle,
    ) {
        (Some(knots), Some(degree), Some(beta)) => Some((knots, degree, beta)),
        _ => None,
    };

    let mut payload = FittedModelPayload::new(
        MODEL_VERSION,
        formula,
        ModelKind::LocationScale,
        FittedFamily::LocationScale {
            likelihood: LikelihoodFamily::GaussianIdentity,
            base_link: None,
        },
        "gaussian-location-scale".to_string(),
    );
    payload.unified = Some(fit.clone());
    payload.fit_result = Some(fit);
    payload.data_schema = Some(dataset.schema.clone());
    payload.link = Some("identity".to_string());
    payload.formula_noise = Some(noise_formula);
    payload.beta_noise = scale_beta;
    payload.gaussian_response_scale = Some(response_scale);
    payload.training_headers = Some(dataset.headers.clone());
    payload.resolved_termspec = Some(frozen_meanspec);
    payload.resolved_termspec_noise = Some(frozen_noisespec);
    payload.offset_column = fit_config.offset_column.clone();
    payload.noise_offset_column = fit_config.noise_offset_column.clone();
    if let Some((knots, degree, beta_link_wiggle)) = wiggle_meta {
        payload.linkwiggle_knots = Some(knots.to_vec());
        payload.linkwiggle_degree = Some(degree);
        payload.beta_link_wiggle = Some(beta_link_wiggle);
    }
    Ok(payload)
}

fn build_binomial_location_scale_ffi_payload(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    link_kind: InverseLink,
    weights: &Array1<f64>,
    ls_result: BinomialLocationScaleFitResult,
) -> Result<FittedModelPayload, String> {
    let frozen_meanspec = freeze_term_collection_from_design(
        &ls_result.fit.meanspec_resolved,
        &ls_result.fit.mean_design,
    )
    .map_err(|err| format!("failed to freeze binomial location-scale threshold spec: {err}"))?;
    let frozen_noisespec = freeze_term_collection_from_design(
        &ls_result.fit.noisespec_resolved,
        &ls_result.fit.noise_design,
    )
    .map_err(|err| format!("failed to freeze binomial location-scale noise spec: {err}"))?;

    let noise_formula = fit_config
        .noise_formula
        .clone()
        .ok_or_else(|| "binomial location-scale requires noise_formula".to_string())?;

    let dense_mean = ls_result.fit.mean_design.design.to_dense();
    let dense_noise = ls_result.fit.noise_design.design.to_dense();
    let non_intercept_start = ls_result
        .fit
        .noise_design
        .intercept_range
        .end
        .min(ls_result.fit.noise_design.design.ncols());
    let binomial_noise_transform =
        build_scale_deviation_transform(&dense_mean, &dense_noise, weights, non_intercept_start)
            .map_err(|err| format!("failed to encode binomial noise transform: {err}"))?;

    let fit = ls_result.fit.fit;
    let scale_beta = fit
        .block_by_role(BlockRole::Scale)
        .map(|block| block.beta.to_vec());
    let wiggle_meta = match (
        ls_result.wiggle_knots,
        ls_result.wiggle_degree,
        ls_result.beta_link_wiggle,
    ) {
        (Some(knots), Some(degree), Some(beta)) => Some((knots, degree, beta)),
        _ => None,
    };

    let mut payload = FittedModelPayload::new(
        MODEL_VERSION,
        formula,
        ModelKind::LocationScale,
        FittedFamily::LocationScale {
            likelihood: inverse_link_to_binomial_family(&link_kind),
            base_link: Some(link_kind.clone()),
        },
        "binomial-location-scale".to_string(),
    );
    payload.unified = Some(fit.clone());
    payload.fit_result = Some(fit);
    payload.data_schema = Some(dataset.schema.clone());
    payload.link = Some(link_kind.saved_string());
    payload.formula_noise = Some(noise_formula);
    payload.beta_noise = scale_beta;
    payload.noise_projection = Some(
        binomial_noise_transform
            .projection_coef
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect(),
    );
    payload.noise_center = Some(binomial_noise_transform.weighted_column_mean.to_vec());
    payload.noise_scale = Some(binomial_noise_transform.rescale.to_vec());
    payload.noise_non_intercept_start = Some(binomial_noise_transform.non_intercept_start);
    payload.training_headers = Some(dataset.headers.clone());
    payload.resolved_termspec = Some(frozen_meanspec);
    payload.resolved_termspec_noise = Some(frozen_noisespec);
    payload.offset_column = fit_config.offset_column.clone();
    payload.noise_offset_column = fit_config.noise_offset_column.clone();
    if let Some((knots, degree, beta_link_wiggle)) = wiggle_meta {
        payload.linkwiggle_knots = Some(knots.to_vec());
        payload.linkwiggle_degree = Some(degree);
        payload.beta_link_wiggle = Some(beta_link_wiggle);
    }
    Ok(payload)
}

fn build_survival_location_scale_ffi_payload(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    weights: &Array1<f64>,
    ls_result: gam::SurvivalLocationScaleFitResult,
) -> Result<FittedModelPayload, String> {
    use gam::families::survival_construction::{
        build_survival_time_basis, parse_survival_baseline_config, parse_survival_likelihood_mode,
        parse_survival_time_basis_config, resolve_survival_time_anchor_value,
        survival_baseline_targetname,
    };
    use ndarray::{Array2, s};

    // Re-derive survival metadata from the formula and FitConfig so we can
    // reproduce the saved model layout that the CLI persists.
    let parsed = gam::inference::formula_dsl::parse_formula(&formula)
        .map_err(|err| format!("failed to re-parse survival formula for FFI payload: {err}"))?;
    let (entryname, exitname, eventname) = parse_surv_response(&parsed.response)?
        .ok_or_else(|| "survival location-scale FFI requires Surv(...) response".to_string())?;
    let col_map: HashMap<String, usize> = dataset
        .headers
        .iter()
        .enumerate()
        .map(|(i, h)| (h.clone(), i))
        .collect();
    let entry_idx = *col_map
        .get(&entryname)
        .ok_or_else(|| format!("entry column '{entryname}' not found"))?;
    let exit_idx = *col_map
        .get(&exitname)
        .ok_or_else(|| format!("exit column '{exitname}' not found"))?;
    let n = dataset.values.nrows();
    let mut age_entry = Array1::<f64>::zeros(n);
    let mut age_exit = Array1::<f64>::zeros(n);
    for i in 0..n {
        let (t0, t1) = gam::families::survival_construction::normalize_survival_time_pair(
            dataset.values[[i, entry_idx]],
            dataset.values[[i, exit_idx]],
            i,
        )?;
        age_entry[i] = t0;
        age_exit[i] = t1;
    }
    let baseline_cfg = parse_survival_baseline_config(
        &fit_config.baseline_target,
        fit_config.baseline_scale,
        fit_config.baseline_shape,
        fit_config.baseline_rate,
        fit_config.baseline_makeham,
    )?;
    let likelihood_mode = parse_survival_likelihood_mode(&fit_config.survival_likelihood)?;
    let time_cfg = if parsed.timewiggle.is_some() {
        gam::families::survival_construction::SurvivalTimeBasisConfig::None
    } else {
        parse_survival_time_basis_config(
            &fit_config.time_basis,
            fit_config.time_degree,
            fit_config.time_num_internal_knots,
            fit_config.time_smooth_lambda,
        )?
    };
    let time_anchor = resolve_survival_time_anchor_value(&age_entry, None)?;
    let mut time_build = build_survival_time_basis(
        &age_entry,
        &age_exit,
        time_cfg,
        Some((
            fit_config.time_num_internal_knots,
            fit_config.time_smooth_lambda,
        )),
    )?;
    let resolved_time_cfg =
        gam::families::survival_construction::resolved_survival_time_basis_config_from_build(
            &time_build.basisname,
            time_build.degree,
            time_build.knots.as_ref(),
            time_build.keep_cols.as_ref(),
            time_build.smooth_lambda,
        )?;
    let time_anchor_row = gam::families::survival_construction::evaluate_survival_time_basis_row(
        time_anchor,
        &resolved_time_cfg,
    )?;
    gam::families::survival_construction::center_survival_time_designs_at_anchor(
        &mut time_build.x_entry_time,
        &mut time_build.x_exit_time,
        &time_anchor_row,
    )?;

    let fitted_inverse_link = ls_result.inverse_link.clone();
    // Compact the inner UnifiedFitResult and apply the fitted link state so
    // downstream prediction can recover the inverse-link parameters from the
    // saved fit_result. Mirrors the CLI's
    // compact_saved_survival_location_scale_fit_result helper.
    let mut fit_result = ls_result.fit.fit.clone();
    apply_inverse_link_state_to_fit_result(&mut fit_result, &fitted_inverse_link);
    fit_result.artifacts.survival_link_wiggle_knots = ls_result.wiggle_knots.clone();
    fit_result.artifacts.survival_link_wiggle_degree = ls_result.wiggle_degree;

    // Reconstruct survival_noise_projection / center / scale by replaying the
    // CLI's build_scale_deviation_transform on the time + threshold + log_sigma
    // designs. Required for predict_survival_location_scale to recover the
    // saved scale-deviation transform.
    let dense_threshold = ls_result.fit.threshold_design.design.to_dense();
    let dense_log_sigma = ls_result.fit.log_sigma_design.design.to_dense();
    let dense_time_exit = time_build.x_exit_time.to_dense();
    let mut survival_primary_design =
        Array2::<f64>::zeros((n, dense_time_exit.ncols() + dense_threshold.ncols()));
    survival_primary_design
        .slice_mut(s![.., 0..dense_time_exit.ncols()])
        .assign(&dense_time_exit);
    survival_primary_design
        .slice_mut(s![.., dense_time_exit.ncols()..])
        .assign(&dense_threshold);
    let survival_noise_transform = build_scale_deviation_transform(
        &survival_primary_design,
        &dense_log_sigma,
        weights,
        infer_non_intercept_start(&dense_log_sigma, weights),
    )
    .map_err(|err| format!("failed to encode survival noise transform: {err}"))?;

    let mut payload = FittedModelPayload::new(
        MODEL_VERSION,
        formula,
        ModelKind::Survival,
        FittedFamily::Survival {
            likelihood: LikelihoodFamily::RoystonParmar,
            survival_likelihood: Some(survival_likelihood_modename(likelihood_mode).to_string()),
            survival_distribution: Some(fitted_inverse_link.saved_string()),
            frailty: gam::families::lognormal_kernel::FrailtySpec::None,
        },
        "royston-parmar".to_string(),
    );
    payload.unified = Some(fit_result.clone());
    payload.fit_result = Some(fit_result);
    payload.data_schema = Some(dataset.schema.clone());
    payload.link = Some(fitted_inverse_link.saved_string());
    payload.linkwiggle_degree = ls_result.wiggle_degree;
    payload.beta_link_wiggle = ls_result
        .fit
        .fit
        .beta_link_wiggle()
        .as_ref()
        .map(|b| b.to_vec());
    payload.linkwiggle_knots = ls_result.wiggle_knots.as_ref().map(|k| k.to_vec());
    payload.survival_entry = Some(entryname);
    payload.survival_exit = Some(exitname);
    payload.survival_event = Some(eventname);
    payload.survivalspec = Some("net".to_string());
    payload.survival_baseline_target =
        Some(survival_baseline_targetname(baseline_cfg.target).to_string());
    payload.survival_baseline_scale = baseline_cfg.scale;
    payload.survival_baseline_shape = baseline_cfg.shape;
    payload.survival_baseline_rate = baseline_cfg.rate;
    payload.survival_baseline_makeham = baseline_cfg.makeham;
    payload.survival_time_basis = Some(time_build.basisname.clone());
    payload.survival_time_degree = time_build.degree;
    payload.survival_time_knots = time_build.knots.clone();
    payload.survival_time_keep_cols = time_build.keep_cols.clone();
    payload.survival_time_smooth_lambda = time_build.smooth_lambda;
    payload.survival_time_anchor = Some(time_anchor);
    payload.survivalridge_lambda = Some(fit_config.ridge_lambda);
    payload.survival_likelihood = Some(survival_likelihood_modename(likelihood_mode).to_string());
    payload.survival_beta_time = Some(ls_result.fit.fit.beta_time().to_vec());
    payload.survival_beta_threshold = Some(ls_result.fit.fit.beta_threshold().to_vec());
    payload.survival_beta_log_sigma = Some(ls_result.fit.fit.beta_log_sigma().to_vec());
    payload.survival_noise_projection = Some(
        survival_noise_transform
            .projection_coef
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect(),
    );
    payload.survival_noise_center = Some(survival_noise_transform.weighted_column_mean.to_vec());
    payload.survival_noise_scale = Some(survival_noise_transform.rescale.to_vec());
    payload.survival_noise_non_intercept_start = Some(survival_noise_transform.non_intercept_start);
    payload.survival_distribution = Some(fitted_inverse_link.saved_string());
    payload.training_headers = Some(dataset.headers.clone());
    payload.resolved_termspec = Some(
        freeze_term_collection_from_design(
            &ls_result.fit.resolved_thresholdspec,
            &ls_result.fit.threshold_design,
        )
        .map_err(|err| err.to_string())?,
    );
    payload.resolved_termspec_noise = Some(
        freeze_term_collection_from_design(
            &ls_result.fit.resolved_log_sigmaspec,
            &ls_result.fit.log_sigma_design,
        )
        .map_err(|err| err.to_string())?,
    );
    payload.offset_column = fit_config.offset_column.clone();
    payload.noise_offset_column = fit_config.noise_offset_column.clone();
    Ok(payload)
}

fn build_latent_survival_ffi_payload(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    request_frailty: gam::families::lognormal_kernel::FrailtySpec,
    lat_result: gam::families::latent_survival::LatentSurvivalTermFitResult,
) -> Result<FittedModelPayload, String> {
    build_latent_window_ffi_payload(
        formula,
        dataset,
        fit_config,
        request_frailty,
        lat_result.fit,
        lat_result.resolvedspec,
        lat_result.design,
        Some(lat_result.latent_sd),
        true,
    )
}

fn build_latent_binary_ffi_payload(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    request_frailty: gam::families::lognormal_kernel::FrailtySpec,
    lat_result: gam::families::latent_survival::LatentBinaryTermFitResult,
) -> Result<FittedModelPayload, String> {
    build_latent_window_ffi_payload(
        formula,
        dataset,
        fit_config,
        request_frailty,
        lat_result.fit,
        lat_result.resolvedspec,
        lat_result.design,
        None,
        false,
    )
}

fn build_latent_window_ffi_payload(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: &FitConfig,
    request_frailty: gam::families::lognormal_kernel::FrailtySpec,
    fit: gam::estimate::UnifiedFitResult,
    resolvedspec: gam::smooth::TermCollectionSpec,
    cov_design: gam::smooth::TermCollectionDesign,
    learned_latent_sd: Option<f64>,
    is_survival: bool,
) -> Result<FittedModelPayload, String> {
    use gam::families::survival_construction::{
        build_survival_time_basis, parse_survival_baseline_config,
        parse_survival_time_basis_config, resolve_survival_time_anchor_value,
        survival_baseline_targetname,
    };

    let parsed = gam::inference::formula_dsl::parse_formula(&formula).map_err(|err| {
        format!("failed to re-parse latent survival formula for FFI payload: {err}")
    })?;
    let (entryname, exitname, eventname) = parse_surv_response(&parsed.response)?
        .ok_or_else(|| "latent survival/binary FFI requires Surv(...) response".to_string())?;
    let col_map: HashMap<String, usize> = dataset
        .headers
        .iter()
        .enumerate()
        .map(|(i, h)| (h.clone(), i))
        .collect();
    let entry_idx = *col_map
        .get(&entryname)
        .ok_or_else(|| format!("entry column '{entryname}' not found"))?;
    let exit_idx = *col_map
        .get(&exitname)
        .ok_or_else(|| format!("exit column '{exitname}' not found"))?;
    let n = dataset.values.nrows();
    let mut age_entry = Array1::<f64>::zeros(n);
    let mut age_exit = Array1::<f64>::zeros(n);
    for i in 0..n {
        let (t0, t1) = gam::families::survival_construction::normalize_survival_time_pair(
            dataset.values[[i, entry_idx]],
            dataset.values[[i, exit_idx]],
            i,
        )?;
        age_entry[i] = t0;
        age_exit[i] = t1;
    }
    let baseline_cfg = parse_survival_baseline_config(
        &fit_config.baseline_target,
        fit_config.baseline_scale,
        fit_config.baseline_shape,
        fit_config.baseline_rate,
        fit_config.baseline_makeham,
    )?;
    let time_cfg = parse_survival_time_basis_config(
        &fit_config.time_basis,
        fit_config.time_degree,
        fit_config.time_num_internal_knots,
        fit_config.time_smooth_lambda,
    )?;
    let time_anchor = resolve_survival_time_anchor_value(&age_entry, None)?;
    let time_build = build_survival_time_basis(
        &age_entry,
        &age_exit,
        time_cfg,
        Some((
            fit_config.time_num_internal_knots,
            fit_config.time_smooth_lambda,
        )),
    )?;

    // For latent survival, splice the fitted latent_sd into the persisted
    // HazardMultiplier frailty (mirrors CLI behaviour at main.rs:5541).
    let saved_family = if is_survival {
        let frailty = match (&request_frailty, learned_latent_sd) {
            (
                gam::families::lognormal_kernel::FrailtySpec::HazardMultiplier {
                    sigma_fixed: None,
                    loading,
                },
                Some(sigma),
            ) => gam::families::lognormal_kernel::FrailtySpec::HazardMultiplier {
                sigma_fixed: Some(sigma),
                loading: *loading,
            },
            _ => request_frailty.clone(),
        };
        FittedFamily::LatentSurvival { frailty }
    } else {
        FittedFamily::LatentBinary {
            frailty: request_frailty.clone(),
        }
    };
    let model_class_label = if is_survival {
        "latent-survival".to_string()
    } else {
        "latent-binary".to_string()
    };
    let likelihood_label = if is_survival {
        "latent".to_string()
    } else {
        "latent-binary".to_string()
    };

    let mut payload = FittedModelPayload::new(
        MODEL_VERSION,
        formula,
        ModelKind::Survival,
        saved_family,
        model_class_label,
    );
    payload.unified = Some(fit.clone());
    payload.fit_result = Some(fit.clone());
    payload.data_schema = Some(dataset.schema.clone());
    payload.survival_entry = Some(entryname);
    payload.survival_exit = Some(exitname);
    payload.survival_event = Some(eventname);
    payload.survivalspec = Some("net".to_string());
    payload.survival_baseline_target =
        Some(survival_baseline_targetname(baseline_cfg.target).to_string());
    payload.survival_baseline_scale = baseline_cfg.scale;
    payload.survival_baseline_shape = baseline_cfg.shape;
    payload.survival_baseline_rate = baseline_cfg.rate;
    payload.survival_baseline_makeham = baseline_cfg.makeham;
    payload.survival_time_basis = Some(time_build.basisname.clone());
    payload.survival_time_degree = time_build.degree;
    payload.survival_time_knots = time_build.knots.clone();
    payload.survival_time_keep_cols = time_build.keep_cols.clone();
    payload.survival_time_smooth_lambda = time_build.smooth_lambda;
    payload.survival_likelihood = Some(likelihood_label);
    payload.survival_time_anchor = Some(time_anchor);
    payload.survival_beta_time = Some(fit.beta_time().to_vec());
    payload.survivalridge_lambda = Some(fit_config.ridge_lambda);
    payload.training_headers = Some(dataset.headers.clone());
    payload.resolved_termspec = Some(
        freeze_term_collection_from_design(&resolvedspec, &cov_design)
            .map_err(|err| err.to_string())?,
    );
    payload.offset_column = fit_config.offset_column.clone();
    payload.noise_offset_column = fit_config.noise_offset_column.clone();
    Ok(payload)
}

fn saved_anchored_deviation_runtime(runtime: &DeviationRuntime) -> SavedAnchoredDeviationRuntime {
    SavedAnchoredDeviationRuntime {
        kernel: gam::families::cubic_cell_kernel::ANCHORED_DEVIATION_KERNEL.to_string(),
        breakpoints: runtime.breakpoints().to_vec(),
        basis_dim: runtime.basis_dim(),
        span_c0: runtime
            .span_c0()
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect(),
        span_c1: runtime
            .span_c1()
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect(),
        span_c2: runtime
            .span_c2()
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect(),
        span_c3: runtime
            .span_c3()
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect(),
        anchor_residual_coefficients: None,
        anchor_residual_components: Vec::new(),
        anchor_residual_rotation: None,
    }
}

fn predict_table_survival(
    model: &FittedModel,
    dataset: &EncodedDataset,
    options: &PyPredictOptions,
) -> Result<String, String> {
    use gam::survival_predict::{SurvivalPredictRequest, predict_survival};

    let col_map = dataset.column_map();
    let payload = model.payload();
    let training_headers = payload.training_headers.as_ref();
    let primary_offset =
        resolve_offset_column(dataset, &col_map, payload.offset_column.as_deref())?;
    let supports_noise_offset = matches!(
        gam::survival_predict::require_saved_survival_likelihood_mode(model)?,
        gam::survival_construction::SurvivalLikelihoodMode::LocationScale
            | gam::survival_construction::SurvivalLikelihoodMode::MarginalSlope
    );
    let noise_offset = if supports_noise_offset {
        resolve_offset_column(dataset, &col_map, payload.noise_offset_column.as_deref())?
    } else {
        ndarray::Array1::<f64>::zeros(dataset.values.nrows())
    };
    let time_grid_slice: Option<&[f64]> = options.time_grid.as_deref();
    let request = SurvivalPredictRequest {
        model,
        data: dataset.values.view(),
        col_map: &col_map,
        training_headers,
        primary_offset: &primary_offset,
        noise_offset: &noise_offset,
        time_grid: time_grid_slice,
        with_uncertainty: options.with_uncertainty,
    };
    let result = predict_survival(request)?;

    // Rowwise flatten for JSON transport.
    let n = result.hazard.nrows();
    let t = result.hazard.ncols();
    let mut hazard_rows = Vec::with_capacity(n);
    let mut survival_rows = Vec::with_capacity(n);
    let mut cumulative_rows = Vec::with_capacity(n);
    for i in 0..n {
        let mut hrow = Vec::with_capacity(t);
        let mut srow = Vec::with_capacity(t);
        let mut crow = Vec::with_capacity(t);
        for j in 0..t {
            hrow.push(result.hazard[[i, j]]);
            srow.push(result.survival[[i, j]]);
            crow.push(result.cumulative_hazard[[i, j]]);
        }
        hazard_rows.push(hrow);
        survival_rows.push(srow);
        cumulative_rows.push(crow);
    }
    let survival_se_rows = result.survival_se.as_ref().map(|se| {
        let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = Vec::with_capacity(t);
            for j in 0..t {
                row.push(se[[i, j]]);
            }
            rows.push(row);
        }
        rows
    });
    let eta_se_vec = result.eta_se.as_ref().map(|v| v.to_vec());

    // Populate explicit probability columns. For survival models the event
    // probability is `1 - survival_prob`; an ambiguous `mean` column makes
    // fixed-time Brier/calibration scoring easy to invert.
    let mut columns = BTreeMap::<String, Vec<f64>>::new();
    columns.insert("eta".to_string(), result.linear_predictor.to_vec());
    let survival_col: Vec<f64> = (0..n)
        .map(|i| result.survival[[i, t.saturating_sub(1)]])
        .collect();
    let failure_col: Vec<f64> = survival_col
        .iter()
        .map(|s| (1.0 - *s).clamp(0.0, 1.0))
        .collect();
    columns.insert("survival_prob".to_string(), survival_col);
    columns.insert("failure_prob".to_string(), failure_col);

    let likelihood_mode_str = match result.likelihood_mode {
        gam::survival_construction::SurvivalLikelihoodMode::MarginalSlope => "marginal-slope",
        gam::survival_construction::SurvivalLikelihoodMode::LocationScale => "location-scale",
        gam::survival_construction::SurvivalLikelihoodMode::Transformation => "transformation",
        gam::survival_construction::SurvivalLikelihoodMode::Weibull => "weibull",
        gam::survival_construction::SurvivalLikelihoodMode::Latent => "latent",
        gam::survival_construction::SurvivalLikelihoodMode::LatentBinary => "latent-binary",
    };
    let model_class_label = match result.likelihood_mode {
        gam::survival_construction::SurvivalLikelihoodMode::MarginalSlope => {
            "survival marginal-slope".to_string()
        }
        gam::survival_construction::SurvivalLikelihoodMode::LocationScale => {
            "survival location-scale".to_string()
        }
        gam::survival_construction::SurvivalLikelihoodMode::Latent => "latent survival".to_string(),
        _ => model.predict_model_class().name().to_string(),
    };
    let survival_payload = SurvivalPredictionPayload {
        class: "survival_prediction",
        model_class: model_class_label,
        likelihood_mode: likelihood_mode_str.to_string(),
        times: result.times.clone(),
        hazard: hazard_rows,
        survival: survival_rows,
        cumulative_hazard: cumulative_rows,
        linear_predictor: result.linear_predictor.to_vec(),
        columns,
        survival_se: survival_se_rows,
        eta_se: eta_se_vec,
    };
    serde_json::to_string(&survival_payload)
        .map_err(|err| format!("failed to serialize survival prediction payload: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn model_version_matches_canonical_payload_version() {
        assert_eq!(MODEL_VERSION, MODEL_PAYLOAD_VERSION);
    }

    #[test]
    fn load_model_rejects_payload_version_mismatch() {
        let model = FittedModel::from_payload(FittedModelPayload::new(
            MODEL_VERSION + 1,
            "y ~ x".to_string(),
            ModelKind::Standard,
            FittedFamily::Standard {
                likelihood: LikelihoodFamily::GaussianIdentity,
                link: Some(family_to_link(LikelihoodFamily::GaussianIdentity)),
                latent_cloglog_state: None,
                mixture_state: None,
                sas_state: None,
            },
            LikelihoodFamily::GaussianIdentity.name().to_string(),
        ));
        let mismatched_bytes =
            serde_json::to_vec(&model).expect("mismatched model should serialize");

        let err = match load_model_impl(&mismatched_bytes) {
            Ok(_) => panic!("load_model_impl should reject mismatched payload versions"),
            Err(e) => e,
        };
        assert!(
            err.contains("saved model payload schema mismatch"),
            "unexpected error: {err}"
        );
    }

    #[derive(Clone, Copy, Debug)]
    enum RemlForwardScalar {
        Lambda,
        RemlScore,
        Coefficient(usize, usize),
        Fitted(usize, usize),
    }

    fn by_gate_fd_design() -> Array2<f64> {
        Array2::from_shape_fn((20, 5), |(row, col)| {
            let t = (row as f64 - 9.5) / 8.0;
            match col {
                0 => 1.0,
                1 => t,
                2 => t * t - 0.45,
                3 => (0.9 * t).sin() + 0.05 * t,
                4 => (1.2 * t).cos() - 0.15 * t * t,
                _ => unreachable!(),
            }
        })
    }

    fn by_gate_fd_response() -> Array2<f64> {
        Array2::from_shape_fn((20, 3), |(row, output)| {
            let t = (row as f64 - 9.5) / 8.0;
            let phase = output as f64 + 1.0;
            0.2 + 0.35 * phase * t
                + 0.18 * t * t
                + (0.4 + 0.1 * phase) * (0.8 * t + 0.25 * phase).cos()
        })
    }

    fn by_gate_fd_values() -> Array1<f64> {
        Array1::from_shape_fn(20, |row| {
            let t = (row as f64 - 9.5) / 8.0;
            0.85 + 0.12 * (0.7 * t).sin() + 0.03 * t
        })
    }

    fn by_gate_fd_weights() -> Array1<f64> {
        Array1::from_shape_fn(20, |row| {
            let t = (row as f64 - 9.5) / 8.0;
            0.9 + 0.15 * (1.3 * t).cos() + 0.04 * t
        })
    }

    fn by_gate_fd_penalty() -> Array2<f64> {
        Array2::from_diag(&array![0.0, 0.4, 1.1, 1.9, 3.0])
    }

    fn by_gate_objective(
        x: ArrayView2<'_, f64>,
        y: ArrayView2<'_, f64>,
        by: ArrayView1<'_, f64>,
        weights: ArrayView1<'_, f64>,
        penalty: ArrayView2<'_, f64>,
        target: RemlForwardScalar,
    ) -> f64 {
        let gated_x = apply_by_gate(x, by, 1).expect("by-gated design");
        let fit = gaussian_reml_multi_closed_form_with_cache(
            gated_x.view(),
            y,
            penalty,
            Some(weights),
            Some(0.85),
            None,
        )
        .expect("by-gated finite-difference forward fit");
        match target {
            RemlForwardScalar::Lambda => fit.lambda,
            RemlForwardScalar::RemlScore => fit.reml_score,
            RemlForwardScalar::Coefficient(row, col) => fit.coefficients[[row, col]],
            RemlForwardScalar::Fitted(row, col) => fit.fitted[[row, col]],
        }
    }

    fn by_gate_backward(
        x: ArrayView2<'_, f64>,
        y: ArrayView2<'_, f64>,
        by: ArrayView1<'_, f64>,
        weights: ArrayView1<'_, f64>,
        penalty: ArrayView2<'_, f64>,
        target: RemlForwardScalar,
    ) -> (Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>) {
        let mut grad_coefficients = Array2::<f64>::zeros((x.ncols(), y.ncols()));
        let mut grad_fitted = Array2::<f64>::zeros(y.dim());
        let (grad_lambda, grad_score, coefficient_upstream, fitted_upstream) = match target {
            RemlForwardScalar::Lambda => (1.0, 0.0, None, None),
            RemlForwardScalar::RemlScore => (0.0, 1.0, None, None),
            RemlForwardScalar::Coefficient(row, col) => {
                grad_coefficients[[row, col]] = 1.0;
                (0.0, 0.0, Some(grad_coefficients.view()), None)
            }
            RemlForwardScalar::Fitted(row, col) => {
                grad_fitted[[row, col]] = 1.0;
                (0.0, 0.0, None, Some(grad_fitted.view()))
            }
        };
        let gated_x = apply_by_gate(x, by, 1).expect("by-gated design");
        let backward = gaussian_reml_multi_closed_form_backward(
            gated_x.view(),
            y,
            penalty,
            Some(weights),
            Some(0.85),
            grad_lambda,
            coefficient_upstream,
            fitted_upstream,
            grad_score,
        )
        .expect("by-gated analytic backward");
        let (grad_x, grad_by) =
            apply_by_gate_backward(x, by, 1, backward.grad_x.view()).expect("by-gate backward");
        (grad_x, backward.grad_y, grad_by, backward.grad_weights)
    }

    fn assert_fd_close(label: &str, analytic: f64, finite_difference: f64) {
        let tol = 1.0e-6_f64.max(1.0e-6 * analytic.abs().max(finite_difference.abs()));
        let diff = (analytic - finite_difference).abs();
        assert!(
            diff <= tol,
            "{label}: analytic={analytic:.12e}, finite_difference={finite_difference:.12e}, diff={diff:.3e}, tol={tol:.3e}"
        );
    }

    fn position_fd_inputs() -> (
        Array1<f64>,
        Array2<f64>,
        Array1<f64>,
        Array2<f64>,
        Array1<f64>,
        Array1<f64>,
    ) {
        let t = Array1::linspace(0.07, 0.93, 18);
        let y = Array2::from_shape_fn((18, 2), |(row, col)| {
            let x = t[row];
            let phase = col as f64 + 1.0;
            0.1 + 0.4 * phase * x + (1.2 * x + 0.3 * phase).sin() * 0.25
        });
        let knots = Array1::linspace(0.0, 1.0, 7);
        let penalty = Array2::from_diag(&array![0.0, 0.8, 1.1, 1.5, 2.0, 2.8]);
        let by = Array1::from_shape_fn(18, |row| 0.9 + 0.1 * (2.0 * t[row]).cos());
        let weights = Array1::from_shape_fn(18, |row| 0.88 + 0.14 * (1.4 * t[row]).sin());
        (t, y, knots, penalty, by, weights)
    }

    fn position_objective(
        t: ArrayView1<'_, f64>,
        y: ArrayView2<'_, f64>,
        by: ArrayView1<'_, f64>,
        weights: ArrayView1<'_, f64>,
        knots: ArrayView1<'_, f64>,
        penalty: ArrayView2<'_, f64>,
        grad_lambda: f64,
        grad_coefficients: ArrayView2<'_, f64>,
        grad_fitted: ArrayView2<'_, f64>,
        grad_reml_score: f64,
    ) -> f64 {
        let x =
            position_basis_design(t, knots, "bspline", 3, true, Some(1.0)).expect("position basis");
        let gated_x = apply_by_gate(x.view(), by, 0).expect("by gate");
        let fit = gaussian_reml_multi_closed_form_with_cache(
            gated_x.view(),
            y,
            penalty,
            Some(weights),
            Some(0.7),
            None,
        )
        .expect("position finite-difference fit");
        grad_lambda * fit.lambda
            + grad_reml_score * fit.reml_score
            + (&fit.coefficients * &grad_coefficients).sum()
            + (&fit.fitted * &grad_fitted).sum()
    }

    #[test]
    fn position_batched_forward_matches_prebuilt_by_gated_design() {
        let (t, y, knots, penalty, by, _weights) = position_fd_inputs();
        let offsets = array![0_usize, 9_usize, 18_usize];
        let x = position_basis_design(t.view(), knots.view(), "bspline", 3, true, Some(1.0))
            .expect("position basis");
        let gated_x = apply_by_gate(x.view(), by.view(), 0).expect("by gate");

        let expected = gaussian_reml_fit_batched_impl(
            gated_x.view(),
            y.view(),
            offsets.view(),
            penalty.view(),
            None,
            Some(0.7),
        )
        .expect("prebuilt batched fit");
        let actual = gaussian_reml_fit_positions_batched_impl(
            t.view(),
            y.view(),
            offsets.view(),
            knots.view(),
            "bspline",
            3,
            true,
            Some(1.0),
            penalty.view(),
            None,
            Some(0.7),
            Some(by.view()),
            0,
        )
        .expect("position batched fit");

        assert_eq!(actual.statuses, expected.statuses);
        for b in 0..2 {
            assert!((actual.lambdas[b] - expected.lambdas[b]).abs() < 1.0e-12);
            assert!((actual.reml_scores[b] - expected.reml_scores[b]).abs() < 1.0e-12);
        }
        for (actual, expected) in actual.coefficients.iter().zip(expected.coefficients.iter()) {
            assert!((*actual - *expected).abs() < 1.0e-12);
        }
        for (actual, expected) in actual.fitted.iter().zip(expected.fitted.iter()) {
            assert!((*actual - *expected).abs() < 1.0e-12);
        }
    }

    #[test]
    fn position_backward_grad_t_y_by_and_weight_match_finite_difference() {
        let (t, y, knots, penalty, by, weights) = position_fd_inputs();
        let x = position_basis_design(t.view(), knots.view(), "bspline", 3, true, Some(1.0))
            .expect("position basis");
        let mut grad_coefficients = Array2::<f64>::zeros((x.ncols(), y.ncols()));
        grad_coefficients[[3, 1]] = -0.25;
        let grad_fitted = Array2::from_shape_fn(y.dim(), |(row, col)| {
            0.02 * (row as f64 + 1.0) - 0.03 * (col as f64 + 1.0)
        });
        let grad_lambda = 0.17;
        let grad_reml_score = -0.11;
        let backward = gaussian_reml_fit_positions_backward_impl(
            t.view(),
            y.view(),
            knots.view(),
            "bspline",
            3,
            true,
            Some(1.0),
            penalty.view(),
            Some(weights.view()),
            Some(0.7),
            grad_lambda,
            Some(grad_coefficients.view()),
            Some(grad_fitted.view()),
            grad_reml_score,
            Some(by.view()),
            0,
        )
        .expect("position analytic backward");
        let grad_by = backward.grad_by.expect("by gradient");
        let eps = 1.0e-5;

        for row in [2_usize, 8, 15] {
            let mut plus = t.clone();
            let mut minus = t.clone();
            plus[row] += eps;
            minus[row] -= eps;
            let fd = (position_objective(
                plus.view(),
                y.view(),
                by.view(),
                weights.view(),
                knots.view(),
                penalty.view(),
                grad_lambda,
                grad_coefficients.view(),
                grad_fitted.view(),
                grad_reml_score,
            ) - position_objective(
                minus.view(),
                y.view(),
                by.view(),
                weights.view(),
                knots.view(),
                penalty.view(),
                grad_lambda,
                grad_coefficients.view(),
                grad_fitted.view(),
                grad_reml_score,
            )) / (2.0 * eps);
            assert_fd_close(&format!("position t[{row}]"), backward.grad_t[row], fd);
        }

        for (row, col) in [(1_usize, 0_usize), (10, 1)] {
            let mut plus = y.clone();
            let mut minus = y.clone();
            plus[[row, col]] += eps;
            minus[[row, col]] -= eps;
            let fd = (position_objective(
                t.view(),
                plus.view(),
                by.view(),
                weights.view(),
                knots.view(),
                penalty.view(),
                grad_lambda,
                grad_coefficients.view(),
                grad_fitted.view(),
                grad_reml_score,
            ) - position_objective(
                t.view(),
                minus.view(),
                by.view(),
                weights.view(),
                knots.view(),
                penalty.view(),
                grad_lambda,
                grad_coefficients.view(),
                grad_fitted.view(),
                grad_reml_score,
            )) / (2.0 * eps);
            assert_fd_close(
                &format!("position y[{row},{col}]"),
                backward.grad_y[[row, col]],
                fd,
            );
        }

        for row in [0_usize, 9, 17] {
            let mut plus = by.clone();
            let mut minus = by.clone();
            plus[row] += eps;
            minus[row] -= eps;
            let fd = (position_objective(
                t.view(),
                y.view(),
                plus.view(),
                weights.view(),
                knots.view(),
                penalty.view(),
                grad_lambda,
                grad_coefficients.view(),
                grad_fitted.view(),
                grad_reml_score,
            ) - position_objective(
                t.view(),
                y.view(),
                minus.view(),
                weights.view(),
                knots.view(),
                penalty.view(),
                grad_lambda,
                grad_coefficients.view(),
                grad_fitted.view(),
                grad_reml_score,
            )) / (2.0 * eps);
            assert_fd_close(&format!("position by[{row}]"), grad_by[row], fd);
        }

        for row in [1_usize, 7, 13] {
            let mut plus = weights.clone();
            let mut minus = weights.clone();
            plus[row] += eps;
            minus[row] -= eps;
            let fd = (position_objective(
                t.view(),
                y.view(),
                by.view(),
                plus.view(),
                knots.view(),
                penalty.view(),
                grad_lambda,
                grad_coefficients.view(),
                grad_fitted.view(),
                grad_reml_score,
            ) - position_objective(
                t.view(),
                y.view(),
                by.view(),
                minus.view(),
                knots.view(),
                penalty.view(),
                grad_lambda,
                grad_coefficients.view(),
                grad_fitted.view(),
                grad_reml_score,
            )) / (2.0 * eps);
            assert_fd_close(
                &format!("position weights[{row}]"),
                backward.grad_weights[row],
                fd,
            );
        }
    }

    #[test]
    fn by_gate_backward_matches_forward_finite_difference_for_all_x_y_gate_and_weight_entries() {
        let x = by_gate_fd_design();
        let y = by_gate_fd_response();
        let by = by_gate_fd_values();
        let weights = by_gate_fd_weights();
        let penalty = by_gate_fd_penalty();
        let targets = [
            RemlForwardScalar::Lambda,
            RemlForwardScalar::RemlScore,
            RemlForwardScalar::Coefficient(4, 2),
            RemlForwardScalar::Fitted(11, 1),
        ];
        let eps = 1.0e-5;

        for target in targets {
            let (grad_x, grad_y, grad_by, grad_weights) = by_gate_backward(
                x.view(),
                y.view(),
                by.view(),
                weights.view(),
                penalty.view(),
                target,
            );

            for row in 0..x.nrows() {
                for col in 0..x.ncols() {
                    let mut plus = x.clone();
                    let mut minus = x.clone();
                    plus[[row, col]] += eps;
                    minus[[row, col]] -= eps;
                    let fd = (by_gate_objective(
                        plus.view(),
                        y.view(),
                        by.view(),
                        weights.view(),
                        penalty.view(),
                        target,
                    ) - by_gate_objective(
                        minus.view(),
                        y.view(),
                        by.view(),
                        weights.view(),
                        penalty.view(),
                        target,
                    )) / (2.0 * eps);
                    assert_fd_close(
                        &format!("target={target:?} x[{row},{col}]"),
                        grad_x[[row, col]],
                        fd,
                    );
                }
            }

            for row in 0..y.nrows() {
                for col in 0..y.ncols() {
                    let mut plus = y.clone();
                    let mut minus = y.clone();
                    plus[[row, col]] += eps;
                    minus[[row, col]] -= eps;
                    let fd = (by_gate_objective(
                        x.view(),
                        plus.view(),
                        by.view(),
                        weights.view(),
                        penalty.view(),
                        target,
                    ) - by_gate_objective(
                        x.view(),
                        minus.view(),
                        by.view(),
                        weights.view(),
                        penalty.view(),
                        target,
                    )) / (2.0 * eps);
                    assert_fd_close(
                        &format!("target={target:?} y[{row},{col}]"),
                        grad_y[[row, col]],
                        fd,
                    );
                }
            }

            for row in 0..by.len() {
                let mut plus = by.clone();
                let mut minus = by.clone();
                plus[row] += eps;
                minus[row] -= eps;
                let fd = (by_gate_objective(
                    x.view(),
                    y.view(),
                    plus.view(),
                    weights.view(),
                    penalty.view(),
                    target,
                ) - by_gate_objective(
                    x.view(),
                    y.view(),
                    minus.view(),
                    weights.view(),
                    penalty.view(),
                    target,
                )) / (2.0 * eps);
                assert_fd_close(&format!("target={target:?} by[{row}]"), grad_by[row], fd);
            }

            for row in 0..weights.len() {
                let mut plus = weights.clone();
                let mut minus = weights.clone();
                plus[row] += eps;
                minus[row] -= eps;
                let fd = (by_gate_objective(
                    x.view(),
                    y.view(),
                    by.view(),
                    plus.view(),
                    penalty.view(),
                    target,
                ) - by_gate_objective(
                    x.view(),
                    y.view(),
                    by.view(),
                    minus.view(),
                    penalty.view(),
                    target,
                )) / (2.0 * eps);
                assert_fd_close(
                    &format!("target={target:?} weights[{row}]"),
                    grad_weights[row],
                    fd,
                );
            }
        }
    }
}
