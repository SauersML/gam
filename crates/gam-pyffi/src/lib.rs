use csv::StringRecord;
use gam::estimate::{BlockRole, PredictInput};
use gam::families::family_meta::{family_to_link, pretty_familyname};
use gam::inference::data::{
    EncodedDataset, UnseenCategoryPolicy, encode_recordswith_inferred_schema,
    encode_recordswith_schema,
};
use gam::inference::model::{
    FittedFamily, FittedModel, FittedModelPayload, ModelKind, PredictModelClass,
};
use gam::report::{CoefficientRow, EdfBlockRow, ReportInput, render_html};
use gam::smooth::{SmoothBasisSpec, TermCollectionSpec, build_term_collection_design};
use gam::types::{LatentCLogLogState, LikelihoodFamily, MixtureLinkState, SasLinkState};
use gam::{FitConfig, FitRequest, FitResult, fit_model, materialize, resolve_offset_column};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};

const MODEL_VERSION: u32 = 4;

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct PyFitConfig {
    family: Option<String>,
    offset: Option<String>,
    weights: Option<String>,
    ridge_lambda: Option<f64>,
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct PyPredictOptions {
    interval: Option<f64>,
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
    info.set_item("supported_model_classes", vec!["standard"])?;
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

    let family = match &request {
        FitRequest::Standard(standard_request) => standard_request.family,
        _ => {
            return Err(
                "python binding currently supports standard models only; remove noise/survival/marginal-slope features from the formula"
                    .to_string(),
            );
        }
    };

    let fit_result = fit_model(request)?;
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

    let payload = build_standard_payload(
        formula,
        &dataset,
        &fit_config,
        family,
        &saved_fit,
        standard_result.resolvedspec,
        standard_result.adaptive_diagnostics,
        standard_result.wiggle_knots.map(|knots| knots.to_vec()),
        standard_result.wiggle_degree,
    );
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
    if model.predict_model_class() != PredictModelClass::Standard {
        return Err(
            "python binding currently supports prediction for standard models only".to_string(),
        );
    }
    let dataset = dataset_with_model_schema(&model, headers, rows)?;
    let predict_input = build_standard_predict_input(&model, &dataset)?;
    let predictor = model
        .predictor()
        .ok_or_else(|| "saved model could not construct a predictor".to_string())?;
    let fit = fit_result_from_saved_model_for_prediction(&model)?;
    let options = parse_predict_options(options_json)?;

    let mut columns = BTreeMap::<String, Vec<f64>>::new();
    if let Some(interval) = options.interval {
        let uncertainty_options = gam::predict::PredictUncertaintyOptions {
            confidence_level: interval,
            covariance_mode:
                gam::predict::InferenceCovarianceMode::ConditionalPlusSmoothingPreferred,
            mean_interval_method: gam::predict::MeanIntervalMethod::TransformEta,
            includeobservation_interval: false,
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
        model_class: predict_model_class_name(model.predict_model_class()).to_string(),
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
                .map(|block| block_role_name(block.role.clone()).to_string()),
        })
        .collect::<Vec<_>>();
    let report_input = ReportInput {
        model_path: "<in-memory>".to_string(),
        family_name: pretty_familyname(model.likelihood()).to_string(),
        model_class: predict_model_class_name(model.predict_model_class()).to_string(),
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
    Ok(fit_config)
}

fn request_metadata(request: &FitRequest<'_>) -> (&'static str, &'static str, bool) {
    match request {
        FitRequest::Standard(standard_request) => {
            (pretty_familyname(standard_request.family), "standard", true)
        }
        FitRequest::GaussianLocationScale(_) => {
            ("Gaussian location-scale", "gaussian location-scale", false)
        }
        FitRequest::BinomialLocationScale(_) => {
            ("Binomial location-scale", "binomial location-scale", false)
        }
        FitRequest::SurvivalLocationScale(_) => {
            ("Survival location-scale", "survival location-scale", false)
        }
        FitRequest::BernoulliMarginalSlope(_) => (
            "Bernoulli marginal-slope",
            "bernoulli marginal-slope",
            false,
        ),
        FitRequest::SurvivalMarginalSlope(_) => {
            ("Survival marginal-slope", "survival marginal-slope", false)
        }
        FitRequest::LatentSurvival(_) => ("Latent survival", "latent survival", false),
        FitRequest::LatentBinary(_) => ("Latent binary", "latent binary", false),
        FitRequest::TransformationNormal(_) => {
            ("Transformation-normal", "transformation-normal", false)
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
    let records = string_records_from_rows(&headers, rows)?;
    encode_recordswith_inferred_schema(headers, records)
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
    let response_column = response_column_name(model.payload().formula.as_str());
    let expected_names = schema
        .columns
        .iter()
        .filter(|column| response_column.as_deref() != Some(column.name.as_str()))
        .map(|column| column.name.clone())
        .collect::<BTreeSet<_>>();
    let present_names = headers.iter().cloned().collect::<BTreeSet<_>>();
    let mut issues = Vec::<SchemaIssue>::new();

    for missing in expected_names.difference(&present_names) {
        issues.push(SchemaIssue {
            kind: "missing_column".to_string(),
            message: format!("missing required column '{missing}'"),
            column: Some(missing.clone()),
        });
    }
    for unknown in present_names.difference(&expected_names) {
        issues.push(SchemaIssue {
            kind: "unknown_column".to_string(),
            message: format!("unknown column '{unknown}' is not part of the training schema"),
            column: Some(unknown.clone()),
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
    payload.link = Some(link_name(family_to_link(family)).to_string());
    payload.linkwiggle_knots = wiggle_knots;
    payload.linkwiggle_degree = wiggle_degree;
    payload.training_headers = Some(dataset.headers.clone());
    payload.resolved_termspec = Some(resolved_termspec);
    payload.adaptive_regularization_diagnostics = adaptive_regularization_diagnostics;
    payload.offset_column = fit_config.offset_column.clone();
    payload.noise_offset_column = fit_config.noise_offset_column.clone();
    payload
}

fn saved_mixture_state_from_fit(fit: &gam::estimate::UnifiedFitResult) -> Option<MixtureLinkState> {
    match &fit.fitted_link {
        gam::estimate::FittedLinkState::Mixture { state, .. } => Some(state.clone()),
        _ => None,
    }
}

fn saved_latent_cloglog_state_from_fit(
    fit: &gam::estimate::UnifiedFitResult,
) -> Option<LatentCLogLogState> {
    match &fit.fitted_link {
        gam::estimate::FittedLinkState::LatentCLogLog { state } => Some(*state),
        _ => None,
    }
}

fn saved_sas_state_from_fit(fit: &gam::estimate::UnifiedFitResult) -> Option<SasLinkState> {
    match &fit.fitted_link {
        gam::estimate::FittedLinkState::Sas { state, .. }
        | gam::estimate::FittedLinkState::BetaLogistic { state, .. } => Some(*state),
        _ => None,
    }
}

fn build_standard_predict_input(
    model: &FittedModel,
    dataset: &EncodedDataset,
) -> Result<PredictInput, String> {
    let col_map = build_col_map(dataset);
    let training_headers = model.payload().training_headers.as_ref();
    let spec = resolve_termspec_for_prediction(
        &model.payload().resolved_termspec,
        training_headers,
        &col_map,
        "resolved_termspec",
    )?;
    let design = build_term_collection_design(dataset.values.view(), &spec)
        .map_err(|err| format!("failed to build prediction design: {err}"))?;
    let offset =
        resolve_offset_column(dataset, &col_map, model.payload().offset_column.as_deref())?;
    let fit = fit_result_from_saved_model_for_prediction(model)?;
    let beta = if model.has_link_wiggle() {
        fit.block_by_role(BlockRole::Mean)
            .ok_or_else(|| {
                "standard link-wiggle model is missing Mean coefficient block".to_string()
            })?
            .beta
            .clone()
    } else {
        fit.beta.clone()
    };
    if beta.len() != design.design.ncols() {
        return Err(format!(
            "model/design mismatch: model beta has {} coefficients but new-data design has {} columns",
            beta.len(),
            design.design.ncols()
        ));
    }
    Ok(PredictInput {
        design: design.design.clone(),
        offset,
        design_noise: None,
        offset_noise: None,
        auxiliary_scalar: None,
    })
}

fn build_col_map(dataset: &EncodedDataset) -> HashMap<String, usize> {
    dataset
        .headers
        .iter()
        .enumerate()
        .map(|(index, header)| (header.clone(), index))
        .collect()
}

fn resolve_termspec_for_prediction(
    model_spec: &Option<TermCollectionSpec>,
    training_headers: Option<&Vec<String>>,
    col_map: &HashMap<String, usize>,
    spec_label: &str,
) -> Result<TermCollectionSpec, String> {
    let saved = model_spec.as_ref().ok_or_else(|| {
        format!(
            "model is missing {spec_label}; refit with the current engine to guarantee train/predict design consistency"
        )
    })?;
    saved.validate_frozen(spec_label)?;
    let headers = training_headers.ok_or_else(|| {
        "model is missing training_headers; refit with the current engine to guarantee stable feature mapping at prediction time"
            .to_string()
    })?;
    let remapped = remap_term_collectionspec_columns(saved, headers, col_map)?;
    remapped.validate_frozen(spec_label)?;
    Ok(remapped)
}

fn remap_term_collectionspec_columns(
    spec: &TermCollectionSpec,
    training_headers: &[String],
    prediction_column_map: &HashMap<String, usize>,
) -> Result<TermCollectionSpec, String> {
    let mut remapped = spec.clone();
    let resolve_training_index = |index: usize| -> Result<usize, String> {
        let name = training_headers
            .get(index)
            .ok_or_else(|| format!("saved training column index {index} is out of bounds"))?;
        prediction_column_map
            .get(name)
            .copied()
            .ok_or_else(|| format!("prediction data is missing required column '{name}'"))
    };

    for linear_term in &mut remapped.linear_terms {
        linear_term.feature_col = resolve_training_index(linear_term.feature_col)?;
    }
    for random_effect_term in &mut remapped.random_effect_terms {
        random_effect_term.feature_col = resolve_training_index(random_effect_term.feature_col)?;
    }
    for smooth_term in &mut remapped.smooth_terms {
        match &mut smooth_term.basis {
            SmoothBasisSpec::BSpline1D { feature_col, .. } => {
                *feature_col = resolve_training_index(*feature_col)?;
            }
            SmoothBasisSpec::ThinPlate { feature_cols, .. }
            | SmoothBasisSpec::Matern { feature_cols, .. }
            | SmoothBasisSpec::Duchon { feature_cols, .. }
            | SmoothBasisSpec::TensorBSpline { feature_cols, .. } => {
                for feature_col in feature_cols.iter_mut() {
                    *feature_col = resolve_training_index(*feature_col)?;
                }
            }
        }
    }

    Ok(remapped)
}

fn fit_result_from_saved_model_for_prediction(
    model: &FittedModel,
) -> Result<gam::estimate::UnifiedFitResult, String> {
    model.payload().fit_result.clone().ok_or_else(|| {
        "model is missing canonical fit_result payload; refit with the current engine".to_string()
    })
}

fn link_name(link: gam::types::LinkFunction) -> &'static str {
    match link {
        gam::types::LinkFunction::Logit => "logit",
        gam::types::LinkFunction::Probit => "probit",
        gam::types::LinkFunction::CLogLog => "cloglog",
        gam::types::LinkFunction::Sas => "sas",
        gam::types::LinkFunction::BetaLogistic => "beta-logistic",
        gam::types::LinkFunction::Identity => "identity",
        gam::types::LinkFunction::Log => "log",
    }
}

fn predict_model_class_name(class: PredictModelClass) -> &'static str {
    match class {
        PredictModelClass::Standard => "standard",
        PredictModelClass::GaussianLocationScale => "gaussian location-scale",
        PredictModelClass::BinomialLocationScale => "binomial location-scale",
        PredictModelClass::BernoulliMarginalSlope => "bernoulli marginal-slope",
        PredictModelClass::Survival => "survival",
        PredictModelClass::TransformationNormal => "transformation-normal",
    }
}

fn block_role_name(role: BlockRole) -> &'static str {
    match role {
        BlockRole::Mean => "mean",
        BlockRole::Location => "location",
        BlockRole::Scale => "scale",
        BlockRole::Time => "time",
        BlockRole::Threshold => "threshold",
        BlockRole::LinkWiggle => "link-wiggle",
    }
}
