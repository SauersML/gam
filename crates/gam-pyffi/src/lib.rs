use csv::StringRecord;
use gam::bernoulli_marginal_slope::{BernoulliMarginalSlopeFitResult, DeviationRuntime};
use gam::estimate::{
    BlockRole, saved_latent_cloglog_state_from_fit, saved_mixture_state_from_fit,
    saved_sas_state_from_fit,
};
use gam::families::family_meta::{
    family_to_link, inverse_link_to_binomial_family, pretty_familyname,
};
use gam::families::scale_design::{build_scale_deviation_transform, infer_non_intercept_start};
use gam::families::survival_construction::survival_likelihood_modename;
use gam::families::survival_predict::{
    apply_inverse_link_state_to_fit_result, fit_result_from_saved_model_for_prediction,
};
use gam::gamlss::{BinomialLocationScaleFitResult, GaussianLocationScaleFitResult};
use gam::inference::data::{
    EncodedDataset, UnseenCategoryPolicy, encode_recordswith_inferred_schema,
    encode_recordswith_schema,
};
use gam::inference::formula_dsl::{ParsedTerm, parse_formula, parse_surv_response};
use gam::inference::model::{
    FittedFamily, FittedModel, FittedModelPayload, MODEL_PAYLOAD_VERSION, ModelKind,
    PredictModelClass, SavedAnchoredDeviationRuntime, SavedLatentZNormalization,
};
use gam::inference::predict_input::build_predict_input_for_model;
use gam::report::{CoefficientRow, EdfBlockRow, ReportInput, render_html};
use gam::smooth::{TermCollectionSpec, freeze_term_collection_from_design};
use gam::survival_marginal_slope::SurvivalMarginalSlopeFitResult;
use gam::transformation_normal::TransformationNormalFitResult;
use gam::types::{InverseLink, LikelihoodFamily};
use gam::{FitConfig, FitRequest, FitResult, fit_model, materialize, resolve_offset_column};
use ndarray::Array1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
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
            "load",
            "predict",
            "summary",
            "check",
            "report",
            "save",
            "validate_formula",
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
    module.add_function(wrap_pyfunction!(load_model, module)?)?;
    module.add_function(wrap_pyfunction!(validate_formula_json, module)?)?;
    module.add_function(wrap_pyfunction!(predict_table, module)?)?;
    module.add_function(wrap_pyfunction!(summary_json, module)?)?;
    module.add_function(wrap_pyfunction!(check_json, module)?)?;
    module.add_function(wrap_pyfunction!(report_html, module)?)?;
    Ok(())
}

fn py_value_error(message: String) -> PyErr {
    PyValueError::new_err(message)
}

fn fit_table_impl(
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    formula: String,
    config_json: Option<&str>,
) -> Result<Vec<u8>, String> {
    let dataset = dataset_with_inferred_schema(headers, rows)?;
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
    let options = parse_predict_options(options_json)?;
    if matches!(model_class, PredictModelClass::Survival) {
        return predict_table_survival(&model, &dataset, &options);
    }
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
    let fit = fit_result_from_saved_model_for_prediction(&model)?;

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

    serde_json::to_string(&PredictionPayload { columns })
        .map_err(|err| format!("failed to serialize prediction payload: {err}"))
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
}
